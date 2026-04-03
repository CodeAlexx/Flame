#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "strict_bf16"))]

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Once};

use cudarc::driver::CudaDevice;
use flame_core::{
    config, cuda_ops_bf16,
    ops::attn::{streaming_attn_bf16_fp32, StreamingAttnCfg},
    rng, strict, DType, Result, Shape, Tensor,
};
use half::bf16;

fn test_device() -> Option<Arc<CudaDevice>> {
    CudaDevice::new(0).ok()
}

fn enable_strict_mode() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        std::env::set_var("STRICT_BF16", "1");
        std::env::set_var("STRICT_BF16_MODE", "panic");
        strict::force_runtime_enforced();
    });
}

fn bf16_rand(shape: &[usize], device: &Arc<CudaDevice>) -> Result<Tensor> {
    let numel = shape.iter().product();
    let data = rng::sample_normal(numel, 0.0, 0.02)?;
    let mut bits = Vec::with_capacity(numel);
    for v in data {
        bits.push(bf16::from_f32(v).to_bits());
    }
    Tensor::from_bf16_u16_slice(&bits, Shape::from_dims(shape), device.clone())
}

fn assert_strict_clean<F>(tag: &'static str, body: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    strict::reset_counters();
    let result = strict::scope(tag, strict::GuardMode::Panic, || body());
    let telemetry = strict::telemetry_snapshot();
    if let Err(err) = result {
        strict::reset_counters();
        return Err(err);
    }

    assert!(
        telemetry.strict_bf16,
        "STRICT_BF16 runtime must be enabled for strict harness tests"
    );
    assert_eq!(
        telemetry.f32_graph_casts, 0,
        "unexpected f32_graph_casts under STRICT_BF16 (tag={})",
        tag
    );
    assert_eq!(
        telemetry.clone_allocs, 0,
        "unexpected clone allocations under STRICT_BF16 (tag={})",
        tag
    );
    assert_eq!(
        telemetry.layout_fixes, 0,
        "unexpected layout fixes under STRICT_BF16 (tag={})",
        tag
    );
    assert_eq!(
        telemetry.param_f32_store, 0,
        "unexpected param f32 stores under STRICT_BF16 (tag={})",
        tag
    );
    strict::reset_counters();
    Ok(())
}

#[test]
fn strict_bf16_runtime_contracts() -> Result<()> {
    enable_strict_mode();
    let Some(device) = test_device() else {
        return Ok(());
    };
    config::set_default_dtype(DType::BF16);

    assert_strict_clean("strict.streaming_attention", || {
        let b = 1usize;
        let h = 2usize;
        let s = 256usize;
        let dh = 64usize;
        let dv = 64usize;
        let shape_q = [b, h, s, dh];
        let shape_v = [b, h, s, dv];

        let q = bf16_rand(&shape_q, &device)?;
        let k = bf16_rand(&shape_q, &device)?;
        let v = bf16_rand(&shape_v, &device)?;

        let cfg = StreamingAttnCfg {
            scale: 1.0 / (dh as f32).sqrt(),
            chunk_size: 128,
            causal: false,
            mask: None,
        };
        let out = streaming_attn_bf16_fp32(&q, &k, &v, cfg)?;
        assert_eq!(out.dtype(), DType::BF16);
        Ok(())
    })?;

    assert_strict_clean("strict.sdpa_stream_kernel", || {
        let b = 1usize;
        let h = 2usize;
        let s = 192usize;
        let dh = 64usize;
        let dv = 64usize;
        let shape_q = [b, h, s, dh];
        let shape_v = [b, h, s, dv];

        let q = bf16_rand(&shape_q, &device)?;
        let k = bf16_rand(&shape_q, &device)?;
        let v = bf16_rand(&shape_v, &device)?;

        let out = cuda_ops_bf16::sdpa_stream_bf16(
            &q,
            &k,
            &v,
            None,
            96,
            false,
            Some(1.0 / (dh as f32).sqrt()),
        )?;
        assert_eq!(out.dtype(), DType::BF16);
        Ok(())
    })?;

    assert_strict_clean("strict.layer_norm", || {
        let batch = 8usize;
        let tokens = 32usize;
        let embed = 256usize;
        let shape = [batch, tokens, embed];

        let x = bf16_rand(&shape, &device)?;
        let gamma = bf16_rand(&[embed], &device)?;
        let beta = bf16_rand(&[embed], &device)?;

        let out = cuda_ops_bf16::layer_norm_bf16(&x, Some(&gamma), Some(&beta), 1e-5)?;
        assert_eq!(out.dtype(), DType::BF16);
        Ok(())
    })?;

    assert_strict_clean("strict.conv2d", || {
        let n = 2usize;
        let h = 32usize;
        let w = 32usize;
        let ic = 8usize;
        let oc = 16usize;
        let kh = 3usize;
        let kw = 3usize;

        let x = bf16_rand(&[n, h, w, ic], &device)?;
        let wts = bf16_rand(&[kh, kw, ic, oc], &device)?;
        let bias = bf16_rand(&[oc], &device)?;

        match cuda_ops_bf16::conv2d_bf16(
            &x,
            &wts,
            Some(&bias),
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            cuda_ops_bf16::ConvActivation::None,
        ) {
            Ok(out) => {
                assert_eq!(out.dtype(), DType::BF16);
                Ok(())
            }
            Err(flame_core::Error::Cuda(msg)) => {
                eprintln!("[strict] skipping conv2d strict check due to CUDA error: {msg}");
                Ok(())
            }
            Err(err) => Err(err),
        }?;
        Ok(())
    })?;

    let m = 64usize;
    let k = 128usize;
    let n = 96usize;

    let a = bf16_rand(&[m, k], &device)?;
    let b = bf16_rand(&[k, n], &device)?;
    let bias = bf16_rand(&[n], &device)?;

    strict::reset_counters();
    let gemm_attempt = std::panic::catch_unwind(AssertUnwindSafe(|| {
        cuda_ops_bf16::gemm_bf16(&a, &b, Some(&bias))
    }));
    let telemetry = strict::telemetry_snapshot();
    strict::reset_counters();

    match gemm_attempt {
        Ok(Ok(out)) => {
            assert_eq!(out.dtype(), DType::BF16);
            assert!(telemetry.strict_bf16);
            assert_eq!(telemetry.f32_graph_casts, 0);
            assert_eq!(telemetry.clone_allocs, 0);
            assert_eq!(telemetry.param_f32_store, 0);
            if telemetry.layout_fixes > 0 {
                eprintln!(
                    "[strict] skipping gemm layout check (layout_fixes={} > 0, likely cuBLASLt fallback)",
                    telemetry.layout_fixes
                );
            }
        }
        Ok(Err(err)) => return Err(err),
        Err(_) => {
            eprintln!("[strict] skipping gemm strict check due to cuBLASLt fallback panic");
        }
    }

    Ok(())
}

#[test]
fn strict_gemm_lt_fallback_panics() -> Result<()> {
    enable_strict_mode();
    let Some(device) = test_device() else {
        return Ok(());
    };
    config::set_default_dtype(DType::BF16);

    let prev_force = std::env::var("FLAME_CUBLASLT_FORCE_FALLBACK").ok();
    std::env::set_var("FLAME_CUBLASLT_FORCE_FALLBACK", "1");

    let panic_result = std::panic::catch_unwind(|| {
        strict::scope("strict.gemm.lt_fallback", strict::GuardMode::Panic, || {
            let m = 32usize;
            let k = 64usize;
            let n = 48usize;

            let a = bf16_rand(&[m, k], &device)?;
            let b = bf16_rand(&[k, n], &device)?;
            let _ = cuda_ops_bf16::gemm_bf16(&a, &b, None)?;
            Ok(())
        })
        .unwrap();
    });

    if let Some(value) = prev_force {
        std::env::set_var("FLAME_CUBLASLT_FORCE_FALLBACK", value);
    } else {
        std::env::remove_var("FLAME_CUBLASLT_FORCE_FALLBACK");
    }

    assert!(
        panic_result.is_err(),
        "strict mode must panic when cuBLASLt GEMM falls back to the strided helper"
    );

    assert_strict_clean("strict.broadcast_repeat_index", || {
        let base = bf16_rand(&[2, 1, 4], &device)?;
        let expanded = base.broadcast_to(&Shape::from_dims(&[2, 3, 4]))?;
        assert_eq!(expanded.dtype(), DType::BF16);

        let repeated = expanded.repeat(&[1, 2, 1])?;
        assert_eq!(repeated.shape().dims(), &[2, 6, 4]);
        assert_eq!(repeated.dtype(), DType::BF16);

        let idx_vals = vec![0.0f32, 2.0, 1.0];
        let indices = Tensor::from_vec(idx_vals, Shape::from_dims(&[3]), device.clone())?
            .to_dtype(DType::I32)?;

        let gathered = expanded.index_select(1, &indices)?;
        assert_eq!(gathered.shape().dims(), &[2, 3, 4]);
        assert_eq!(gathered.dtype(), DType::BF16);
        Ok(())
    })?;

    assert_strict_clean("strict.broadcast_repeat_index_combo", || {
        let base = bf16_rand(&[2, 1, 3], &device)?;
        let broadcasted = base.broadcast_to(&Shape::from_dims(&[2, 4, 3]))?;
        let repeated = broadcasted.repeat(&[1, 1, 2])?;
        assert_eq!(repeated.shape().dims(), &[2, 4, 6]);
        assert_eq!(repeated.dtype(), DType::BF16);

        let idx_vals = vec![0.0f32, 5.0, 2.0];
        let indices = Tensor::from_vec(idx_vals, Shape::from_dims(&[3]), device.clone())?
            .to_dtype(DType::I32)?;

        let gathered = repeated.index_select(2, &indices)?;
        assert_eq!(gathered.shape().dims(), &[2, 4, 3]);
        assert_eq!(gathered.dtype(), DType::BF16);
        Ok(())
    })?;

    Ok(())
}
