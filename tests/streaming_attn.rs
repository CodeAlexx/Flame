use cudarc::driver::CudaDevice;
use flame_core::{
    config,
    ops::attn::{
        streaming_attn_bf16_fp32, streaming_attn_last_launch_info, StreamingAttnCfg,
        StreamingFallbackReason,
    },
    DType, Result, Shape, Tensor,
};
use once_cell::sync::Lazy;
use std::env;
use std::sync::{Arc, Mutex, MutexGuard};

fn test_device() -> Option<Arc<CudaDevice>> {
    CudaDevice::new(0).ok()
}

static PLANNER_ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

struct PlannerBudgetGuard {
    key: &'static str,
    old: Option<String>,
    _lock: MutexGuard<'static, ()>,
}

impl PlannerBudgetGuard {
    fn set(bytes: u64) -> Self {
        let lock = PLANNER_ENV_LOCK.lock().expect("planner env mutex poisoned");
        let key = "FLAME_SDPA_PLANNER_BUDGET_BYTES";
        let old = env::var(key).ok();
        env::set_var(key, bytes.to_string());
        Self {
            key,
            old,
            _lock: lock,
        }
    }
}

impl Drop for PlannerBudgetGuard {
    fn drop(&mut self) {
        match &self.old {
            Some(value) => env::set_var(self.key, value),
            None => env::remove_var(self.key),
        }
    }
}

fn ensure_dtype(t: Tensor, target: DType) -> Result<Tensor> {
    if t.dtype() == target {
        Ok(t)
    } else {
        t.to_dtype(target)
    }
}

#[test]
fn streaming_attention_matches_dense_small_seq() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let b = 1usize;
    let h = 2usize;
    let s = 256usize;
    let dh = 64usize;
    let dv = 80usize;
    let shape_q = Shape::from_dims(&[b, h, s, dh]);
    let shape_v = Shape::from_dims(&[b, h, s, dv]);

    let q = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(shape_v.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;

    let scale = 1.0f32 / (dh as f32).sqrt();
    let cfg = StreamingAttnCfg {
        scale,
        chunk_size: 512,
        causal: false,
        mask: None,
    };
    let streaming = streaming_attn_bf16_fp32(&q, &k, &v, cfg)?;

    let bh = b * h;
    let q_flat = q.reshape(&[bh, s, dh])?;
    let k_flat = k.reshape(&[bh, s, dh])?;
    let v_flat = v.reshape(&[bh, s, dv])?;

    assert_eq!(q_flat.dtype(), DType::BF16, "expected BF16 q_flat");
    assert_eq!(k_flat.dtype(), DType::BF16, "expected BF16 k_flat");
    assert_eq!(v_flat.dtype(), DType::BF16, "expected BF16 v_flat");

    let k_t = ensure_dtype(k_flat.transpose_dims(1, 2)?, DType::BF16)?;
    assert_eq!(k_t.dtype(), DType::BF16, "transpose should preserve BF16");

    let scores = ensure_dtype(q_flat.bmm(&k_t)?, DType::BF16)?.mul_scalar(scale)?;
    let attn = ensure_dtype(scores.softmax(-1)?, DType::BF16)?;
    let dense = attn
        .bmm(&v_flat)?
        .reshape(&[b, h, s, dv])?
        .to_dtype(DType::F32)?;

    let streaming_f32 = streaming.to_dtype(DType::F32)?;
    let diff = streaming_f32.sub(&dense)?;
    let diff_host = diff.to_vec_f32()?;

    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    for val in diff_host.iter() {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
        sum_abs += abs;
    }
    let mean_abs = sum_abs / diff_host.len() as f32;

    assert!(max_abs <= 1e-3, "max abs diff too high: {max_abs}");
    assert!(mean_abs <= 1e-4, "mean abs diff too high: {mean_abs}");

    Ok(())
}

#[test]
fn streaming_attention_clamps_chunk_size() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let shape_dims = [1usize, 1, 64, 32];
    let shape = Shape::from_dims(&shape_dims);
    let q = Tensor::randn(shape.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(shape.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(shape.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;

    let dh = shape_dims[3] as f32;
    let scale = 1.0 / dh.sqrt();
    let cfg = StreamingAttnCfg {
        scale,
        chunk_size: 10_000, // deliberately larger than S and kernel cap
        causal: true,
        mask: None,
    };

    let out = streaming_attn_bf16_fp32(&q, &k, &v, cfg)?;
    assert_eq!(out.shape().dims(), &[1, 1, 64, 32]);

    Ok(())
}

#[test]
fn streaming_attention_enforces_chunk_floor() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let b = 1usize;
    let h = 1usize;
    let s = 512usize;
    let dh = 64usize;
    let dv = 64usize;

    let shape_q = Shape::from_dims(&[b, h, s, dh]);
    let shape_v = Shape::from_dims(&[b, h, s, dv]);

    let q = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(shape_v.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;

    let scale = 1.0f32 / (dh as f32).sqrt();

    let cfg_small = StreamingAttnCfg {
        scale,
        chunk_size: 32,
        causal: false,
        mask: None,
    };
    let cfg_floor = StreamingAttnCfg {
        scale,
        chunk_size: 64,
        causal: false,
        mask: None,
    };

    let out_small = streaming_attn_bf16_fp32(&q, &k, &v, cfg_small)?;
    let out_floor = streaming_attn_bf16_fp32(&q, &k, &v, cfg_floor)?;

    let diff = out_small
        .to_dtype(DType::F32)?
        .sub(&out_floor.to_dtype(DType::F32)?)?;
    let diff_host = diff.to_vec_f32()?;

    let mut max_abs = 0f32;
    for val in diff_host {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    assert!(
        max_abs <= 1e-5,
        "outputs diverged despite chunk floor clamp: max_abs={max_abs}"
    );

    Ok(())
}

#[test]
fn streaming_attention_respects_causal_flag() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let b = 1usize;
    let h = 2usize;
    let s = 128usize;
    let dh = 64usize;
    let dv = 80usize;

    let shape_q = Shape::from_dims(&[b, h, s, dh]);
    let shape_v = Shape::from_dims(&[b, h, s, dv]);

    let q = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(shape_v.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;

    let scale = 1.0f32 / (dh as f32).sqrt();
    let cfg = StreamingAttnCfg {
        scale,
        chunk_size: 64,
        causal: true,
        mask: None,
    };
    let streaming = streaming_attn_bf16_fp32(&q, &k, &v, cfg)?;

    let bh = b * h;
    let q_flat = q.reshape(&[bh, s, dh])?;
    let k_flat = k.reshape(&[bh, s, dh])?;
    let v_flat = v.reshape(&[bh, s, dv])?;

    let k_t = ensure_dtype(k_flat.transpose_dims(1, 2)?, DType::BF16)?;
    let scores = ensure_dtype(q_flat.bmm(&k_t)?, DType::F32)?.mul_scalar(scale)?;

    // Build causal mask (1 where valid, 0 otherwise) and broadcast to [bh, s, s]
    let mut mask_vals = Vec::with_capacity(s * s);
    for q_idx in 0..s {
        for k_idx in 0..s {
            mask_vals.push(if k_idx <= q_idx { 1.0 } else { 0.0 });
        }
    }
    let mask_shape = Shape::from_dims(&[1, 1, s, s]);
    let mask_f32 = Tensor::from_vec_dtype(mask_vals, mask_shape, device.clone(), DType::F32)?
        .broadcast_to(&Shape::from_dims(&[b, h, s, s]))?
        .reshape(&[bh, s, s])?;

    let mask_adjust = mask_f32.add_scalar(-1.0)?.mul_scalar(1e9f32)?;
    let masked_scores = scores.add(&mask_adjust)?;
    let attn = ensure_dtype(masked_scores.softmax(-1)?, DType::F32)?;
    let dense = attn
        .bmm(&ensure_dtype(v_flat, DType::F32)?)?
        .reshape(&[b, h, s, dv])?
        .to_dtype(DType::F32)?;

    let streaming_f32 = streaming.to_dtype(DType::F32)?;
    let diff = streaming_f32.sub(&dense)?;
    let diff_host = diff.to_vec_f32()?;

    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    for val in diff_host.iter() {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
        sum_abs += abs;
    }
    let mean_abs = sum_abs / diff_host.len() as f32;

    assert!(max_abs <= 1.5e-3, "max abs diff too high: {max_abs}");
    assert!(mean_abs <= 5e-4, "mean abs diff too high: {mean_abs}");

    Ok(())
}

#[test]
fn streaming_attention_tracks_budget_trim() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);
    let _guard = PlannerBudgetGuard::set(64 * 1024 * 1024);

    let b = 1usize;
    let h = 2usize;
    let s = 8192usize;
    let dh = 64usize;
    let dv = 160usize;
    let shape_q = Shape::from_dims(&[b, h, s, dh]);
    let shape_v = Shape::from_dims(&[b, h, s, dv]);

    let q = Tensor::zeros_dtype(shape_q.clone(), DType::BF16, device.clone())?;
    let k = Tensor::zeros_dtype(shape_q, DType::BF16, device.clone())?;
    let v = Tensor::zeros_dtype(shape_v, DType::BF16, device.clone())?;

    let scale = 1.0f32 / (dh as f32).sqrt();
    let cfg = StreamingAttnCfg {
        scale,
        chunk_size: 2048,
        causal: false,
        mask: None,
    };

    let out = streaming_attn_bf16_fp32(&q, &k, &v, cfg)?;
    assert_eq!(out.shape().dims(), &[b, h, s, dv]);

    let info = streaming_attn_last_launch_info();
    assert!(!info.fallbacks.is_empty(), "expected at least one fallback");
    let Some(event) = info
        .fallbacks
        .iter()
        .find(|ev| matches!(ev.reason, StreamingFallbackReason::WorkspaceBudget { .. }))
    else {
        panic!(
            "workspace budget fallback not recorded: {:?}",
            info.fallbacks
        );
    };

    assert_eq!(
        event.from_chunk, 2048,
        "planner should report trimming from the requested chunk"
    );
    assert_eq!(
        info.chosen_chunk,
        Some(event.to_chunk),
        "chosen chunk should match the trimmed target: {:?}",
        info
    );
    if let StreamingFallbackReason::WorkspaceBudget {
        required_bytes,
        budget_bytes,
    } = &event.reason
    {
        assert!(
            required_bytes > budget_bytes,
            "workspace requirement should exceed budget when trimming occurs"
        );
    } else {
        unreachable!("expected WorkspaceBudget reason");
    }

    Ok(())
}

#[test]
fn streaming_attention_honors_generous_budget() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);
    let _guard = PlannerBudgetGuard::set(512 * 1024 * 1024);

    let b = 1usize;
    let h = 2usize;
    let s = 8192usize;
    let dh = 64usize;
    let dv = 160usize;
    let shape_q = Shape::from_dims(&[b, h, s, dh]);
    let shape_v = Shape::from_dims(&[b, h, s, dv]);

    let q = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(shape_v.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;

    let scale = 1.0f32 / (dh as f32).sqrt();
    let cfg = StreamingAttnCfg {
        scale,
        chunk_size: 2048,
        causal: false,
        mask: None,
    };

    let _ = streaming_attn_bf16_fp32(&q, &k, &v, cfg)?;
    let info = streaming_attn_last_launch_info();

    assert_eq!(
        info.chosen_chunk,
        Some(2048),
        "expected planner to keep the requested chunk when budget is ample"
    );
    assert!(
        info.fallbacks
            .iter()
            .all(|ev| !matches!(ev.reason, StreamingFallbackReason::WorkspaceBudget { .. })),
        "unexpected WorkspaceBudget fallback under generous budget: {:?}",
        info.fallbacks
    );

    Ok(())
}

#[test]
fn streaming_attention_applies_explicit_mask() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let b = 1usize;
    let h = 2usize;
    let s = 96usize;
    let dh = 64usize;
    let dv = 48usize;

    let shape_q = Shape::from_dims(&[b, h, s, dh]);
    let shape_v = Shape::from_dims(&[b, h, s, dv]);

    let q = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(shape_q.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(shape_v.clone(), 0.0, 0.02, device.clone())?.to_dtype(DType::BF16)?;

    // Construct deterministic mask that drops roughly a third of keys per head.
    let mut mask_vals = Vec::with_capacity(b * h * s * s);
    for _batch in 0..b {
        for head in 0..h {
            for q_idx in 0..s {
                for k_idx in 0..s {
                    let keep = if (k_idx + head + q_idx) % 3 == 0 {
                        0.0
                    } else {
                        1.0
                    };
                    mask_vals.push(keep);
                }
            }
        }
    }
    let mask_shape = Shape::from_dims(&[b, h, s, s]);
    let mask_f32 =
        Tensor::from_vec_dtype(mask_vals, mask_shape.clone(), device.clone(), DType::F32)?;
    let mask_bf16 = mask_f32.to_dtype(DType::BF16)?;

    let scale = 1.0f32 / (dh as f32).sqrt();
    let cfg = StreamingAttnCfg {
        scale,
        chunk_size: 64,
        causal: false,
        mask: Some(&mask_bf16),
    };
    let streaming = streaming_attn_bf16_fp32(&q, &k, &v, cfg)?;

    let bh = b * h;
    let q_flat = q.reshape(&[bh, s, dh])?;
    let k_flat = k.reshape(&[bh, s, dh])?;
    let v_flat = v.reshape(&[bh, s, dv])?;

    let k_t = ensure_dtype(k_flat.transpose_dims(1, 2)?, DType::BF16)?;
    let scores = ensure_dtype(q_flat.bmm(&k_t)?, DType::F32)?.mul_scalar(scale)?;

    let mask_dense = mask_f32.reshape(&[bh, s, s])?;
    let mask_adjust = mask_dense.add_scalar(-1.0)?.mul_scalar(1e9f32)?;
    let masked_scores = scores.add(&mask_adjust)?;
    let attn = ensure_dtype(masked_scores.softmax(-1)?, DType::F32)?;
    let dense = attn
        .bmm(&ensure_dtype(v_flat, DType::F32)?)?
        .reshape(&[b, h, s, dv])?
        .to_dtype(DType::F32)?;

    let streaming_f32 = streaming.to_dtype(DType::F32)?;
    let diff = streaming_f32.sub(&dense)?;
    let diff_host = diff.to_vec_f32()?;

    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    for val in diff_host.iter() {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
        sum_abs += abs;
    }
    let mean_abs = sum_abs / diff_host.len() as f32;

    assert!(max_abs <= 1.5e-3, "max abs diff too high: {max_abs}");
    assert!(mean_abs <= 5e-4, "mean abs diff too high: {mean_abs}");

    Ok(())
}
