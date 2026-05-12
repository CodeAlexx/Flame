//! Regression test for `cuda_kernel_sources.rs::SUM_KERNEL` (F32 path).
//!
//! Before Stage 0a of the backward-overhaul plan, this kernel had no
//! grid-stride loop. Combined with the launch config in
//! `cuda_kernels.rs::sum()` that caps `gridDim.x` at 1024 with
//! `blockDim.x = 256`, any element past index 262144 (= 1024*256) was
//! silently dropped. A 10M-ones tensor would sum to 262144 instead of
//! 10_000_000.
//!
//! The fix adds a grid-stride loop mirroring PyTorch's
//! `aten/src/ATen/native/cuda/Reduce.cuh::reduce_kernel`.
//!
//! This test exercises the F32 path (BF16 dispatches to a different
//! kernel in `bf16_reduce.rs` which already had grid-stride).

#![cfg(feature = "cuda")]

use flame_core::{device::Device, DType, Result, Shape, Tensor};

fn cuda_device() -> Device {
    Device::cuda(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

/// The headline regression case: 10M F32 ones.
///
/// Pre-fix: this returned ~262144.0 (kernel silently dropped the tail).
/// Post-fix: must return ~10_000_000.0 (modulo F32 atomicAdd slop).
#[test]
fn f32_sum_10m_ones_covers_all_elements() -> Result<()> {
    let device = cuda_device().cuda_device().clone();
    let n: usize = 10_000_000;

    let x = Tensor::ones(Shape::from_dims(&[n]), device.clone())?;
    assert_eq!(x.dtype(), DType::F32);

    let s = x.sum()?;
    assert_eq!(s.dtype(), DType::F32);
    assert_eq!(s.shape().dims(), &[] as &[usize]);

    let v = s.to_vec_f32()?[0];

    // Expected exact value: 10_000_000.0. The kernel does shared-mem
    // tree reduce per block then atomicAdd into a single F32 output.
    // F32 atomicAdd ordering is non-deterministic; summing 10M ones
    // into a single F32 accumulator introduces ULP-level rounding once
    // the running sum exceeds 2^24 ≈ 16.7M (not hit here, n < 2^24),
    // but partial-block atomicAdd ordering can still produce a small
    // drift. Allow ±10.0 (well below pre-fix gap of 9_737_856).
    let expected = n as f32;
    let diff = (v - expected).abs();
    assert!(
        diff <= 10.0,
        "f32 sum(10M ones) = {v}, expected ~{expected} (diff {diff}). \
         Pre-fix this returned ~262144 because SUM_KERNEL lacked grid-stride."
    );

    Ok(())
}

/// Smaller cases that the pre-fix kernel handled correctly. The fix
/// must not regress these.
#[test]
fn f32_sum_small_cases_still_correct() -> Result<()> {
    let device = cuda_device().cuda_device().clone();

    for &n in &[1usize, 256, 1024, 100_000, 262144] {
        let x = Tensor::ones(Shape::from_dims(&[n]), device.clone())?;
        let s = x.sum()?;
        let v = s.to_vec_f32()?[0];
        let expected = n as f32;
        let diff = (v - expected).abs();
        // Tighter tolerance for small N: no atomicAdd ordering noise
        // can accumulate beyond ULP scale on integer counts < 2^23.
        let tol = (expected * 1e-6).max(1.0);
        assert!(
            diff <= tol,
            "f32 sum({n} ones) = {v}, expected {expected} (diff {diff}, tol {tol})"
        );
    }

    Ok(())
}

/// Boundary case: exactly one element past the pre-fix grid coverage
/// (262144 + 1). Pre-fix this would have returned 262144.0; post-fix
/// must return 262145.0.
#[test]
fn f32_sum_just_past_grid_cap() -> Result<()> {
    let device = cuda_device().cuda_device().clone();
    let n: usize = 262_145;

    let x = Tensor::ones(Shape::from_dims(&[n]), device.clone())?;
    let s = x.sum()?;
    let v = s.to_vec_f32()?[0];
    let expected = n as f32;
    let diff = (v - expected).abs();

    assert!(
        diff <= 1.0,
        "f32 sum({n} ones) = {v}, expected {expected} (diff {diff}). \
         This is the n = grid_cap + 1 case that proves grid-stride works."
    );

    Ok(())
}
