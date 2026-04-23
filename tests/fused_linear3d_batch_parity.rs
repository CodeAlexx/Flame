//! Parity test — `fused_linear3d` / `fused_linear3d_native` at B>1 must match
//! the B=1-looped result bit-per-bit (up to BF16 noise).
//!
//! Why this test exists: prior to the batch-fold fix, `flame_linear3d_bf16*`
//! set `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT=1` on the weight layout and
//! `=batch_size` on the input/output layouts. cuBLASLt's heuristic for that
//! mixed-batch configuration has a gap at certain (M, N, K) shapes with B>1,
//! returning `CUBLAS_STATUS_INVALID_VALUE` (err 7). The fix collapses
//! `[B, N, C] → [1, B*N, C]` inside the C++ function (Linear is a per-position
//! op and the row-major buffer is contiguous, so the reshape is a no-op on
//! memory). This test verifies:
//!
//!   1. The B>1 call actually succeeds (no cublasLt error)
//!   2. The B>1 result equals the result of calling the same op B=1 at a time
//!      and stitching the outputs together.
//!
//! Gate: cos_sim ≥ 0.9999 and mean_rel ≤ 5e-3. `max_rel` is not gated (see
//! `linear_transpose_free.rs` for precedent): at K=3072 the B=1-looped vs
//! folded GEMM can pick different cuBLASLt algos, and the resulting
//! reduction-order difference spikes per-element relative error on
//! near-zero outputs even though the tensor as a whole matches. `max_abs`
//! gives a more honest BF16 bound and is checked alongside.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn pseudo_stream(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 32) as u32) as f32 / (u32::MAX as f32);
        out.push(u - 0.5);
    }
    out
}

fn bf16_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    let dev = global_cuda_device();
    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(shape),
        dev.clone(),
        DType::F32,
    )
    .expect("from_vec_dtype")
    .to_dtype(DType::BF16)
    .expect("to bf16")
}

fn compare(tag: &str, a: &Tensor, b: &Tensor) {
    let av = a.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let bv = b.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_eq!(av.len(), bv.len(), "{tag}: length mismatch");

    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut sum_abs_diff = 0.0_f64;
    let mut sum_abs_ref = 0.0_f64;
    let mut max_abs_diff = 0.0_f64;
    let mut ref_abs_max = 0.0_f64;
    let mut nan_count = 0usize;

    for (&x, &y) in av.iter().zip(bv.iter()) {
        let (x, y) = (x as f64, y as f64);
        if x.is_nan() || y.is_nan() {
            nan_count += 1;
            continue;
        }
        dot += x * y;
        na += x * x;
        nb += y * y;
        let ax = x.abs();
        if ax > ref_abs_max {
            ref_abs_max = ax;
        }
        let diff = (x - y).abs();
        sum_abs_diff += diff;
        sum_abs_ref += ax;
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    let cos_sim = dot / (na.sqrt() * nb.sqrt() + 1e-20);
    let mean_rel = sum_abs_diff / sum_abs_ref.max(1e-20);
    // Scale-aware abs ceiling: BF16 ≈ 7 bits of mantissa, so an absolute
    // diff of ref_max/128 is the noise floor from a single rounding.
    // We allow 8× that as a garbage trap while staying well below anything
    // that would indicate a real bug.
    let max_abs_ceiling = ref_abs_max / 16.0;

    println!(
        "[{tag}] cos_sim={:.6} mean_rel={:.4e} max_abs={:.4e} ref_max={:.3} nan={nan_count}",
        cos_sim, mean_rel, max_abs_diff, ref_abs_max
    );

    assert_eq!(nan_count, 0, "{tag}: NaN in output");
    assert!(
        cos_sim >= 0.9999,
        "{tag}: cos_sim {cos_sim:.6} < 0.9999"
    );
    assert!(
        mean_rel <= 5e-3,
        "{tag}: mean_rel {mean_rel:.4e} > 5e-3"
    );
    assert!(
        max_abs_diff <= max_abs_ceiling,
        "{tag}: max_abs {max_abs_diff:.4e} > ref_max/16 ({max_abs_ceiling:.4e})"
    );
}

/// Run fused_linear3d_native at [B, N, Cin] and compare against the same op
/// run B=1 at a time on each batch slice, concatenated.
fn run_native_parity(b: usize, n: usize, cin: usize, cout: usize, with_bias: bool) {
    println!("== native  B={b} N={n} Cin={cin} Cout={cout} bias={with_bias}");

    let w_data = pseudo_stream(0xC0FFEE_u64.wrapping_add(cin as u64 * 131 + cout as u64), cout * cin);
    let weight = bf16_tensor(&[cout, cin], w_data); // PyTorch layout [Cout, Cin]

    let bias = if with_bias {
        let b_data = pseudo_stream(0xBADF00D, cout);
        Some(bf16_tensor(&[cout], b_data))
    } else {
        None
    };

    let x_data = pseudo_stream(
        0x1234_5678_u64
            .wrapping_add(b as u64 * 1009 + n as u64 * 17 + cin as u64),
        b * n * cin,
    );
    let input_b = bf16_tensor(&[b, n, cin], x_data.clone());

    // --- single B>1 call (this used to return err 7) ---
    let out_b = flame_core::ops::fused_inference::fused_linear3d_native(
        &input_b,
        &weight,
        bias.as_ref(),
    )
    .expect("fused_linear3d_native B>1 must succeed");
    assert_eq!(out_b.shape().dims(), &[b, n, cout]);

    // --- reference: split into B slices of [1, N, Cin], stitch results ---
    let dev = global_cuda_device();
    let mut ref_parts: Vec<f32> = Vec::with_capacity(b * n * cout);
    let slice_len = n * cin;
    for i in 0..b {
        let slice = x_data[i * slice_len..(i + 1) * slice_len].to_vec();
        let x_i = bf16_tensor(&[1, n, cin], slice);
        let out_i = flame_core::ops::fused_inference::fused_linear3d_native(
            &x_i,
            &weight,
            bias.as_ref(),
        )
        .expect("fused_linear3d_native B=1 slice");
        let part = out_i.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        ref_parts.extend_from_slice(&part);
    }
    // Turn the stitched reference back into a tensor for the compare helper.
    let ref_tensor = Tensor::from_vec_dtype(
        ref_parts,
        Shape::from_dims(&[b, n, cout]),
        dev.clone(),
        DType::F32,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();

    compare(
        &format!("native B={b} N={n} Cin={cin} Cout={cout} bias={with_bias}"),
        &out_b,
        &ref_tensor,
    );
}

/// Same parity check for the pre-transposed `fused_linear3d` variant.
fn run_pretrans_parity(b: usize, n: usize, cin: usize, cout: usize, with_bias: bool) {
    println!("== pretrans B={b} N={n} Cin={cin} Cout={cout} bias={with_bias}");

    // Pre-transposed weight: [Cin, Cout]
    let w_data = pseudo_stream(0xFEEDBEEF_u64.wrapping_add(cin as u64 * 131 + cout as u64), cin * cout);
    let weight = bf16_tensor(&[cin, cout], w_data);

    let bias = if with_bias {
        let b_data = pseudo_stream(0xDEADCAFE, cout);
        Some(bf16_tensor(&[cout], b_data))
    } else {
        None
    };

    let x_data = pseudo_stream(
        0xABCDEF01_u64.wrapping_add(b as u64 * 7919 + n as u64 * 13 + cin as u64),
        b * n * cin,
    );
    let input_b = bf16_tensor(&[b, n, cin], x_data.clone());

    let out_b = flame_core::ops::fused_inference::fused_linear3d(
        &input_b,
        &weight,
        bias.as_ref(),
    )
    .expect("fused_linear3d B>1 must succeed");
    assert_eq!(out_b.shape().dims(), &[b, n, cout]);

    let dev = global_cuda_device();
    let mut ref_parts: Vec<f32> = Vec::with_capacity(b * n * cout);
    let slice_len = n * cin;
    for i in 0..b {
        let slice = x_data[i * slice_len..(i + 1) * slice_len].to_vec();
        let x_i = bf16_tensor(&[1, n, cin], slice);
        let out_i = flame_core::ops::fused_inference::fused_linear3d(
            &x_i,
            &weight,
            bias.as_ref(),
        )
        .expect("fused_linear3d B=1 slice");
        let part = out_i.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        ref_parts.extend_from_slice(&part);
    }
    let ref_tensor = Tensor::from_vec_dtype(
        ref_parts,
        Shape::from_dims(&[b, n, cout]),
        dev.clone(),
        DType::F32,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();

    compare(
        &format!("pretrans B={b} N={n} Cin={cin} Cout={cout} bias={with_bias}"),
        &out_b,
        &ref_tensor,
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

/// The specific failure that prompted the fix: B=2 CFG through a FLUX-sized
/// linear (3072 → 3072). With the pre-fix code this returned cublasLt err 7.
#[test]
fn flux_qkv_3072_b2_native() {
    run_native_parity(2, 256, 3072, 3072, true);
}

/// Chroma-style double-block QKV shape at B=2, no bias.
#[test]
fn chroma_qkv_3072_b2_nobias() {
    run_native_parity(2, 4096, 3072, 3072, false);
}

/// Small fast sanity — B=1 must still work (regression guard for the
/// trivial `n_eff = 1 * seq_len` path).
#[test]
fn small_b1_native() {
    run_native_parity(1, 17, 64, 48, true);
    run_native_parity(1, 17, 64, 48, false);
}

/// Small B=4 across several aspect ratios including a non-square GEMM.
#[test]
fn small_b4_native() {
    run_native_parity(4, 32, 64, 48, true);
    run_native_parity(4, 32, 64, 48, false);
    run_native_parity(4, 7, 128, 32, true);
}

/// B=3 is an odd batch count — makes sure we aren't relying on power-of-two
/// batches.
#[test]
fn odd_b3_native() {
    run_native_parity(3, 11, 96, 72, true);
}

/// Pre-transposed variant — smaller coverage since both funcs share the
/// batch-fold logic, but we still want at least one B>1 smoke test.
#[test]
fn small_b2_pretrans() {
    run_pretrans_parity(2, 33, 64, 48, true);
    run_pretrans_parity(2, 33, 64, 48, false);
}

/// Direct FFI-level error-code check: call the native cublasLt entry point
/// at B=2 and confirm the raw return code is 0 (was 7 before the fix).
#[test]
fn ffi_b2_returns_zero() {
    use flame_core::cuda::device_lt;

    let dev = global_cuda_device();
    let stream = device_lt::stream_ptr(&dev).expect("stream_ptr");
    let lt = device_lt::cublaslt_handle_ptr(&dev).expect("cublaslt handle");

    let (b, n, cin, cout) = (2usize, 128, 512, 768);
    let x = bf16_tensor(&[b, n, cin], pseudo_stream(0x11, b * n * cin));
    let w = bf16_tensor(&[cout, cin], pseudo_stream(0x22, cout * cin));
    let bias = bf16_tensor(&[cout], pseudo_stream(0x33, cout));
    let out = Tensor::empty_dtype(
        Shape::from_dims(&[b, n, cout]),
        DType::BF16,
        dev.clone(),
    )
    .unwrap();

    let workspace_size: usize = 4 * 1024 * 1024;
    let workspace: cudarc::driver::CudaSlice<u8> =
        unsafe { dev.alloc(workspace_size).unwrap() };

    let ret = unsafe {
        flame_core::cuda::ffi::flame_linear3d_bf16_native(
            lt,
            x.as_device_ptr_bf16("x").unwrap() as *const _,
            w.as_device_ptr_bf16("w").unwrap() as *const _,
            bias.as_device_ptr_bf16("bias").unwrap() as *const _,
            out.as_device_ptr_bf16("out").unwrap() as *mut _,
            b as i32,
            n as i32,
            cin as i32,
            cout as i32,
            {
                use cudarc::driver::DevicePtr;
                *workspace.device_ptr() as *mut _
            },
            workspace_size,
            stream,
        )
    };
    assert_eq!(
        ret, 0,
        "flame_linear3d_bf16_native returned cublasLt error {ret} at B=2 — fix regressed"
    );
    dev.synchronize().unwrap();
}
