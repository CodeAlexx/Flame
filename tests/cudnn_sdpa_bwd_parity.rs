//! Phase 2c parity test — cuDNN v9 SDPA backward vs decomposed recompute.
//!
//! Ran `flame_cudnn_sdpa_bf16_train_fwd` to get (O, Stats), then
//! `flame_cudnn_sdpa_bwd_bf16` with a synthetic dO. Compared the resulting
//! dQ / dK / dV against the decomposed-recompute backward that trainers
//! were hitting pre-Phase-2c (7 GEMMs + 2 softmaxes, computed here inline
//! via public tensor ops).
//!
//! Gate: cos_sim ≥ 0.9999, mean_rel ≤ 5e-3 on each gradient. max_rel is
//! logged but not gated — BF16 reduction-order differences naturally push
//! single-element relatives up to ~1% at any non-trivial N.
//!
//! Shapes kept small so the O(N²) reference math stays fast; the cuDNN
//! kernel is not sensitive to shape for correctness (only for perf), so a
//! small shape is sufficient to verify the wiring is right.

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn ptr_bf16(t: &Tensor, tag: &str) -> *const core::ffi::c_void {
    t.as_device_ptr_bf16(tag).unwrap() as *const core::ffi::c_void
}
fn ptr_bf16_mut(t: &Tensor, tag: &str) -> *mut core::ffi::c_void {
    t.as_device_ptr_bf16(tag).unwrap() as *mut core::ffi::c_void
}

/// Decomposed SDPA forward + backward reference, all in FP32 via public ops.
/// Returns (O, dQ, dK, dV) as 4D BF16 tensors. Exposing the forward output
/// lets callers isolate forward-vs-backward parity failures.
fn reference_sdpa_forward_and_backward(
    q: &Tensor, k: &Tensor, v: &Tensor, d_o: &Tensor, scale: f32,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let dims = q.shape().dims();
    let (b, h, n_q, d) = (dims[0], dims[1], dims[2], dims[3]);
    let n_kv = k.shape().dims()[2];
    let bh = b * h;

    // to_dtype(F32) on a permuted/reshaped BF16 view goes through a reshape
    // that may not be contiguous. Normalize via clone_result() first (which
    // materializes) then cast.
    let q32 = q.to_dtype(DType::F32).unwrap().reshape(&[bh, n_q, d]).unwrap();
    let k32 = k.to_dtype(DType::F32).unwrap().reshape(&[bh, n_kv, d]).unwrap();
    let v32 = v.to_dtype(DType::F32).unwrap().reshape(&[bh, n_kv, d]).unwrap();
    let do32 = d_o.to_dtype(DType::F32).unwrap().reshape(&[bh, n_q, d]).unwrap();

    // Forward recompute: S = Q@K^T * scale; P = softmax(S); O = P @ V.
    //
    // Manual softmax (max-shift + exp + sum + div) for full control over
    // numerical stability. Avoids relying on flame_core's softmax dispatch
    // which picks among several BF16/F32 paths depending on dtype and
    // requires_grad — for a deterministic reference we want predictable
    // math, not a moving target.
    let k_t = k32.transpose_dims(1, 2).unwrap().contiguous().unwrap();
    let scores = q32.bmm(&k_t).unwrap().mul_scalar(scale).unwrap();
    // [bh, nq, nkv] softmax along last axis.
    let max_vals = flame_core::cuda_ops::GpuOps::max_dim(&scores, 2, true).unwrap();
    let shifted = scores.sub(&max_vals).unwrap();
    let exp_vals = shifted.exp().unwrap();
    let sum_vals = flame_core::cuda_ops::GpuOps::sum_dim_keepdim(&exp_vals, 2).unwrap();
    let attn = exp_vals.div(&sum_vals).unwrap();
    let o_ref = attn.bmm(&v32).unwrap();

    // Backward:
    //   dV = attn^T @ dO
    //   dP = dO @ V^T
    //   D  = rowsum(dO * O_ref)   per (bh, q_row)
    //   dS = attn * (dP - D)
    //   dQ = dS @ K  * scale
    //   dK = dS^T @ Q * scale
    let attn_t = attn.transpose_dims(1, 2).unwrap().contiguous().unwrap();
    let dv3 = attn_t.bmm(&do32).unwrap();

    let v_t = v32.transpose_dims(1, 2).unwrap().contiguous().unwrap();
    let dp = do32.bmm(&v_t).unwrap();

    // D[bh, i] = sum_d dO[bh, i, d] * O[bh, i, d]; keepdim to broadcast.
    let d_row = do32
        .mul(&o_ref).unwrap()
        .sum_dim_keepdim(2).unwrap(); // [bh, n_q, 1]

    let dp_minus_d = dp.sub(&d_row).unwrap(); // broadcasts [bh,nq,nkv] - [bh,nq,1]
    let ds = attn.mul(&dp_minus_d).unwrap();

    let dq3 = ds.bmm(&k32).unwrap().mul_scalar(scale).unwrap();
    let ds_t = ds.transpose_dims(1, 2).unwrap().contiguous().unwrap();
    let dk3 = ds_t.bmm(&q32).unwrap().mul_scalar(scale).unwrap();

    let o_bf16 = o_ref.reshape(&[b, h, n_q, d]).unwrap()
        .to_dtype(DType::BF16).unwrap();
    let dq = dq3.reshape(&[b, h, n_q, d]).unwrap()
        .to_dtype(DType::BF16).unwrap();
    let dk = dk3.reshape(&[b, h, n_kv, d]).unwrap()
        .to_dtype(DType::BF16).unwrap();
    let dv = dv3.reshape(&[b, h, n_kv, d]).unwrap()
        .to_dtype(DType::BF16).unwrap();
    (o_bf16, dq, dk, dv)
}

fn compare_grads(name: &str, got: &Tensor, reference: &Tensor) {
    let got_f32  = got.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let ref_f32  = reference.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_eq!(got_f32.len(), ref_f32.len(),
        "{name}: length mismatch ({} vs {})", got_f32.len(), ref_f32.len());

    let mut dot = 0.0f64;
    let mut got_n2 = 0.0f64;
    let mut ref_n2 = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut sum_abs_ref = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut ref_abs_max = 0.0f64;
    let mut nans = 0usize;
    for (a, b) in got_f32.iter().zip(ref_f32.iter()) {
        let (a, b) = (*a as f64, *b as f64);
        if a.is_nan() || b.is_nan() { nans += 1; continue; }
        dot    += a * b;
        got_n2 += a * a;
        ref_n2 += b * b;
        let ab_ref = b.abs();
        if ab_ref > ref_abs_max { ref_abs_max = ab_ref; }
        let diff = (a - b).abs();
        sum_abs_diff += diff;
        sum_abs_ref  += ab_ref;
        let rel = diff / b.abs().max(ref_abs_max.max(1e-6) / 128.0);
        if rel > max_rel { max_rel = rel; }
    }
    let cos_sim = dot / (got_n2.sqrt() * ref_n2.sqrt() + 1e-20);
    let mean_rel = sum_abs_diff / sum_abs_ref.max(1e-20);

    println!(
        "[sdpa_bwd_parity] {name}: cos_sim={:.6} mean_rel={:.4e} max_rel={:.4e} ref_max={:.3} nans={nans}",
        cos_sim, mean_rel, max_rel, ref_abs_max
    );

    assert_eq!(nans, 0, "{name}: NaN in gradient");
    assert!(cos_sim >= 0.9999, "{name}: cos_sim {cos_sim:.6} < 0.9999");
    assert!(mean_rel <= 5e-3, "{name}: mean_rel {mean_rel:.4e} > 5e-3");
}

fn run_bwd_parity(b: usize, h: usize, n: usize, d: usize) {
    let device = global_cuda_device();
    let stream = flame_core::cuda::device_lt::stream_ptr(&device).expect("stream_ptr");

    let scale = 1.0f32 / (d as f32).sqrt();
    let shape4 = Shape::from_dims(&[b, h, n, d]);

    // Random BF16 Q, K, V, dO. Distinct seeds per tensor so we aren't
    // testing a degenerate case where gradients cancel. std=1.0 for Q/K/V
    // produces scores ~ sqrt(D) * scale ≈ 1.0, which after softmax yields
    // an attention distribution sharp enough that gradients have real
    // magnitude (not buried in BF16 rounding noise).
    let q = Tensor::randn_seeded(shape4.clone(), 0.0, 1.0, 11, device.clone())
        .unwrap().to_dtype(DType::BF16).unwrap();
    let k = Tensor::randn_seeded(shape4.clone(), 0.0, 1.0, 22, device.clone())
        .unwrap().to_dtype(DType::BF16).unwrap();
    let v = Tensor::randn_seeded(shape4.clone(), 0.0, 1.0, 33, device.clone())
        .unwrap().to_dtype(DType::BF16).unwrap();
    let d_o = Tensor::randn_seeded(shape4.clone(), 0.0, 1.0, 44, device.clone())
        .unwrap().to_dtype(DType::BF16).unwrap();

    // ---- cuDNN forward-train → (O, Stats) ----
    let o = Tensor::zeros_dtype(shape4.clone(), DType::BF16, device.clone()).unwrap();
    let stats = Tensor::zeros_dtype(
        Shape::from_dims(&[b * h, n]),
        DType::F32,
        device.clone(),
    ).unwrap();

    // Contiguous [B, H, N, D] strides: [H*N*D, N*D, D, 1].
    let strides4: [i64; 4] = [(h * n * d) as i64, (n * d) as i64, d as i64, 1];

    let stats_ptr = {
        use cudarc::driver::DevicePtr;
        let s = stats.as_slice_f32("stats").unwrap();
        *s.device_ptr() as *mut core::ffi::c_void
    };

    let ret = unsafe {
        flame_core::cuda::ffi::flame_cudnn_sdpa_bf16_train_fwd(
            ptr_bf16(&q, "q"),
            ptr_bf16(&k, "k"),
            ptr_bf16(&v, "v"),
            ptr_bf16_mut(&o, "o"),
            stats_ptr,
            b as i32, h as i32, n as i32, n as i32, d as i32,
            scale,
            strides4.as_ptr(), strides4.as_ptr(),
            strides4.as_ptr(), strides4.as_ptr(),
            0, 0, 0, 0, 0,
            stream,
        )
    };
    assert_eq!(ret, 0, "cuDNN train-fwd returned {ret}");

    // ---- cuDNN backward → (dQ, dK, dV) ----
    let dq = Tensor::zeros_dtype(shape4.clone(), DType::BF16, device.clone()).unwrap();
    let dk = Tensor::zeros_dtype(shape4.clone(), DType::BF16, device.clone()).unwrap();
    let dv = Tensor::zeros_dtype(shape4.clone(), DType::BF16, device.clone()).unwrap();

    let stats_ptr_const = {
        use cudarc::driver::DevicePtr;
        let s = stats.as_slice_f32("stats").unwrap();
        *s.device_ptr() as *const core::ffi::c_void
    };

    let ret = unsafe {
        flame_core::cuda::ffi::flame_cudnn_sdpa_bwd_bf16(
            ptr_bf16(&q, "q"),
            ptr_bf16(&k, "k"),
            ptr_bf16(&v, "v"),
            ptr_bf16(&o, "o"),
            ptr_bf16(&d_o, "do"),
            stats_ptr_const,
            ptr_bf16_mut(&dq, "dq"),
            ptr_bf16_mut(&dk, "dk"),
            ptr_bf16_mut(&dv, "dv"),
            b as i32, h as i32, n as i32, n as i32, d as i32,
            scale,
            strides4.as_ptr(), strides4.as_ptr(),
            strides4.as_ptr(), strides4.as_ptr(),
            strides4.as_ptr(),
            strides4.as_ptr(), strides4.as_ptr(), strides4.as_ptr(),
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            stream,
        )
    };
    assert_eq!(ret, 0, "cuDNN bwd returned {ret}");

    device.synchronize().unwrap();

    // ---- Reference: decomposed recompute (and forward O for isolation) ----
    let (o_ref, dq_ref, dk_ref, dv_ref) =
        reference_sdpa_forward_and_backward(&q, &k, &v, &d_o, scale);

    println!(
        "[sdpa_bwd_parity] shape B={b} H={h} N={n} D={d} (seeds q=11 k=22 v=33 dO=44)"
    );
    // Forward parity first — if this fails the backward comparison is noise.
    compare_grads("O", &o, &o_ref);
    compare_grads("dQ", &dq, &dq_ref);
    compare_grads("dK", &dk, &dk_ref);
    compare_grads("dV", &dv, &dv_ref);
}

/// Sanity: cuDNN train-fwd must produce the same O as cuDNN inference-fwd
/// on the same Q/K/V (they share a graph topology except for the Stats
/// output). If this fails, the train-fwd shim itself is wrong; no point
/// trying to debug the backward.
#[test]
fn cudnn_sdpa_train_fwd_matches_inference_fwd() {
    let device = global_cuda_device();
    let stream = flame_core::cuda::device_lt::stream_ptr(&device).expect("stream_ptr");
    let (b, h, n, d) = (1usize, 4usize, 128usize, 64usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let shape4 = Shape::from_dims(&[b, h, n, d]);

    let q = Tensor::randn_seeded(shape4.clone(), 0.0, 1.0, 11, device.clone())
        .unwrap().to_dtype(DType::BF16).unwrap();
    let k = Tensor::randn_seeded(shape4.clone(), 0.0, 1.0, 22, device.clone())
        .unwrap().to_dtype(DType::BF16).unwrap();
    let v = Tensor::randn_seeded(shape4.clone(), 0.0, 1.0, 33, device.clone())
        .unwrap().to_dtype(DType::BF16).unwrap();

    let o_inf = Tensor::zeros_dtype(shape4.clone(), DType::BF16, device.clone()).unwrap();
    let o_tr  = Tensor::zeros_dtype(shape4.clone(), DType::BF16, device.clone()).unwrap();
    let stats = Tensor::zeros_dtype(Shape::from_dims(&[b * h, n]), DType::F32, device.clone()).unwrap();
    let strides4: [i64; 4] = [(h * n * d) as i64, (n * d) as i64, d as i64, 1];

    let stats_ptr = {
        use cudarc::driver::DevicePtr;
        *stats.as_slice_f32("stats").unwrap().device_ptr() as *mut core::ffi::c_void
    };

    // Inference fwd
    let ret = unsafe {
        flame_core::cuda::ffi::flame_cudnn_sdpa_bf16(
            ptr_bf16(&q, "q"), ptr_bf16(&k, "k"), ptr_bf16(&v, "v"),
            ptr_bf16_mut(&o_inf, "o_inf"),
            b as i32, h as i32, n as i32, n as i32, d as i32,
            scale,
            strides4.as_ptr(), strides4.as_ptr(),
            strides4.as_ptr(), strides4.as_ptr(),
            0, 0, 0, 0,
            stream,
        )
    };
    assert_eq!(ret, 0);

    // Train fwd
    let ret = unsafe {
        flame_core::cuda::ffi::flame_cudnn_sdpa_bf16_train_fwd(
            ptr_bf16(&q, "q"), ptr_bf16(&k, "k"), ptr_bf16(&v, "v"),
            ptr_bf16_mut(&o_tr, "o_tr"),
            stats_ptr,
            b as i32, h as i32, n as i32, n as i32, d as i32,
            scale,
            strides4.as_ptr(), strides4.as_ptr(),
            strides4.as_ptr(), strides4.as_ptr(),
            0, 0, 0, 0, 0,
            stream,
        )
    };
    assert_eq!(ret, 0);
    device.synchronize().unwrap();

    compare_grads("train-fwd O vs inference-fwd O", &o_tr, &o_inf);
}

#[test]
fn cudnn_sdpa_bwd_matches_recompute_small() {
    // Small exploratory shape. Compiles fast, runs fast. HD=64 covers one
    // of the three cuDNN-supported head_dims; HD=128 is covered below.
    run_bwd_parity(1, 4, 128, 64);
}

#[test]
fn cudnn_sdpa_bwd_matches_recompute_hd128() {
    run_bwd_parity(1, 4, 256, 128);
}
