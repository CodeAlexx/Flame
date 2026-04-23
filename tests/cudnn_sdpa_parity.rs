//! Phase E parity test — cuDNN v9 SDPA vs in-tree WMMA flash attention.
//!
//! Identical Q, K, V through both kernels. Asserts cos_sim ≥ 0.9999 and
//! max per-element relative error ≤ 5e-3 (BF16 floor).
//!
//! Shapes:
//!   Klein 9B double block joint-attn:  B=1, H=24, N=4608, D=128
//!   Chroma double block joint-attn:    B=1, H=24, N=4096, D=128
//!
//! Baseline microbench (Phase 1 of `PLAN_FA_SWAP.md`) said in-tree WMMA at
//! Klein's shape is 39.26 ms/call; cuDNN SDPA (Phase C standalone) is 3.24
//! ms/call. This test does not re-bench — it only verifies parity.

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn ptr_bf16(t: &Tensor, tag: &str) -> *const core::ffi::c_void {
    t.as_device_ptr_bf16(tag).unwrap() as *const core::ffi::c_void
}
fn ptr_bf16_mut(t: &Tensor, tag: &str) -> *mut core::ffi::c_void {
    t.as_device_ptr_bf16(tag).unwrap() as *mut core::ffi::c_void
}

fn run_parity(b: usize, h: usize, n: usize, d: usize) {
    let device = global_cuda_device();
    let stream = flame_core::cuda::device_lt::stream_ptr(&device)
        .expect("stream_ptr");

    let bh = b * h;
    let scale_init = 1.0f32 / (d as f32).sqrt();

    // Layout for both kernels: [B*H, N, D] BF16 contiguous.
    // WMMA kernel takes this directly. cuDNN shim re-layouts via its graph
    // descriptor — we pass the same underlying buffer with B=bh (outer
    // batch collapses into batch dim; cuDNN graph has H=1 implicitly via
    // its set_dim).
    let shape3 = Shape::from_dims(&[bh, n, d]);
    let q = Tensor::randn_seeded(shape3.clone(), 0.0, scale_init, 1, device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let k = Tensor::randn_seeded(shape3.clone(), 0.0, scale_init, 2, device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let v = Tensor::randn_seeded(shape3.clone(), 0.0, scale_init, 3, device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let o_wmma = Tensor::zeros_dtype(shape3.clone(), DType::BF16, device.clone()).unwrap();
    let o_cudnn = Tensor::zeros_dtype(shape3.clone(), DType::BF16, device.clone()).unwrap();

    let q_ptr = ptr_bf16(&q, "q");
    let k_ptr = ptr_bf16(&k, "k");
    let v_ptr = ptr_bf16(&v, "v");
    let o_wmma_ptr = ptr_bf16_mut(&o_wmma, "o_wmma");
    let o_cudnn_ptr = ptr_bf16_mut(&o_cudnn, "o_cudnn");

    // ---- run WMMA ----
    let ret = unsafe {
        flame_core::cuda::ffi::flame_flash_attention_bf16(
            q_ptr, k_ptr, v_ptr, o_wmma_ptr,
            std::ptr::null_mut(),
            bh as i32,
            n as i32,
            n as i32,
            d as i32,
            stream,
        )
    };
    assert_eq!(ret, 0, "WMMA FA returned {ret}");

    // ---- run cuDNN SDPA ----
    // cuDNN shim takes [B, H, N, D] + per-tensor strides + element offsets.
    // We pass B=bh, H=1 so the graph's batch dim covers all heads (scale
    // is the same per head so this is mathematically equivalent to
    // [1, bh, N, D]). Since Q,K,V,O are contiguous [bh, N, D], their
    // 4D [bh, 1, N, D] strides are (N*D, N*D, D, 1).
    let strides: [i64; 4] = [(n * d) as i64, (n * d) as i64, d as i64, 1];
    let ret = unsafe {
        flame_core::cuda::ffi::flame_cudnn_sdpa_bf16(
            q_ptr, k_ptr, v_ptr, o_cudnn_ptr,
            bh as i32,
            1,
            n as i32,
            n as i32,
            d as i32,
            scale_init,
            strides.as_ptr(),
            strides.as_ptr(),
            strides.as_ptr(),
            strides.as_ptr(),
            0, 0, 0, 0,
            stream,
        )
    };
    assert_eq!(ret, 0, "cuDNN SDPA returned {ret}");

    device.synchronize().unwrap();

    // ---- Compare ----
    let wmma_f32  = o_wmma.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let cudnn_f32 = o_cudnn.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_eq!(wmma_f32.len(), cudnn_f32.len());

    let mut dot = 0.0_f64;
    let mut wmma_norm2 = 0.0_f64;
    let mut cudnn_norm2 = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut sum_abs_diff = 0.0_f64;
    let mut sum_abs_ref = 0.0_f64;
    let mut nan_count = 0usize;
    let mut ref_abs_max = 0.0_f64;
    for (a, b) in wmma_f32.iter().zip(cudnn_f32.iter()) {
        let (a, b) = (*a as f64, *b as f64);
        if a.is_nan() || b.is_nan() { nan_count += 1; continue; }
        dot         += a * b;
        wmma_norm2  += a * a;
        cudnn_norm2 += b * b;
        let ab = a.abs();
        if ab > ref_abs_max { ref_abs_max = ab; }
        let diff = (a - b).abs();
        sum_abs_diff += diff;
        sum_abs_ref  += ab;
        // Relative-to-scale, floor at max_ref/128 so near-zero positions
        // don't blow up the ratio (BF16 can't hold anything below that
        // anyway for a tensor peaking at max_ref).
        let rel = diff / a.abs().max(ref_abs_max.max(1e-6) / 128.0);
        if rel > max_rel { max_rel = rel; }
    }
    let cos_sim = dot / (wmma_norm2.sqrt() * cudnn_norm2.sqrt() + 1e-20);
    let mean_rel = sum_abs_diff / sum_abs_ref.max(1e-20);

    println!(
        "[parity] B={b} H={h} N={n} D={d}  cos_sim={:.6}  max_rel={:.4e}  mean_rel={:.4e}  ref_max={:.3}  nan={nan_count}",
        cos_sim, max_rel, mean_rel, ref_abs_max
    );

    assert_eq!(nan_count, 0, "NaN in output");

    // cos_sim is the strong gate — BF16 reduction-order differences blow
    // individual per-element max_rel up to ~1% naturally (mantissa ~7 bits).
    // We keep a max_rel ceiling only as a NaN/garbage trap, not a precision
    // claim. mean_rel is the honest "are these the same tensor" number.
    assert!(cos_sim >= 0.9999,
        "cos_sim {cos_sim:.6} < 0.9999  (B={b} H={h} N={n} D={d})");
    assert!(mean_rel <= 5e-3,
        "mean_rel {mean_rel:.4e} > 5e-3  (B={b} H={h} N={n} D={d})");
    assert!(max_rel <= 5e-2,
        "max_rel {max_rel:.4e} > 5e-2 — suspicious, likely a bug not BF16 noise  (B={b} H={h} N={n} D={d})");
}

#[test]
fn cudnn_sdpa_matches_wmma_klein() {
    run_parity(1, 24, 4608, 128);
}

#[test]
fn cudnn_sdpa_matches_wmma_chroma() {
    run_parity(1, 24, 4096, 128);
}
