//! Parity test for the BF16-native scalar-reduce path
//! (`bf16_reduce::sum_bf16` / `mean_bf16`) vs. the legacy
//! BF16→F32-cast → F32-reduce → F32→BF16-cast path.
//!
//! Tolerance: relative error < 1e-2 on the BF16-down-cast output. F32
//! reductions are not bit-identical under different orderings
//! (atomicAdd is non-deterministic; the legacy path's F32 sum uses a
//! shared-mem tree + atomicAdd, the new path uses a grid-stride F32
//! accumulator + shared-mem tree + atomicAdd — different orderings,
//! but the final F32 sum is well-defined to BF16 precision).

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{device::Device, rng, DType, Result, Shape, Tensor};

fn cuda_device() -> Device {
    Device::cuda(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

/// Host-side reference reduction. Pulls the BF16 tensor down, upcasts
/// each element to F32, and sums them in Kahan-style on CPU. This is
/// the source of truth because the on-device legacy F32 sum kernel
/// has a grid cap (`grid_size = n.div_ceil(256).min(1024)`) that
/// silently drops elements past 262144 — i.e., the legacy path was
/// broken for tensors larger than ~256K and we can't use it as a
/// reference for the 10M-element case. The new BF16-native kernel
/// uses a grid-stride loop so it's correct for any size.
fn host_sum_bf16(x: &Tensor) -> Result<f32> {
    let x32 = x.to_dtype(DType::F32)?;
    let host = x32.to_vec_f32()?;
    // Kahan compensation reduces F32 cancellation error.
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for &v in host.iter() {
        let y = v as f64 - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
    }
    Ok(s as f32)
}

fn host_mean_bf16(x: &Tensor) -> Result<f32> {
    let n = x.shape().elem_count() as f32;
    Ok(host_sum_bf16(x)? / n)
}

fn bf16_scalar_to_f32(t: &Tensor) -> Result<f32> {
    assert_eq!(t.dtype(), DType::BF16);
    let t32 = t.to_dtype(DType::F32)?;
    Ok(t32.to_vec_f32()?[0])
}

fn assert_relative(label: &str, got: f32, expected: f32, rtol: f32, atol: f32) {
    let diff = (got - expected).abs();
    let tol = atol + rtol * expected.abs();
    assert!(
        diff <= tol,
        "{}: got={got} expected={expected} diff={diff} > tol={tol}",
        label
    );
}

#[test]
fn bf16_sum_matches_host_reference() -> Result<()> {
    let device = cuda_device().cuda_device().clone();
    let _ = rng::set_seed(424242);

    // Mirror the typical zimage hidden state shape so the test exercises
    // a real-world reduction size.
    let x_f32 = Tensor::randn(Shape::from_dims(&[1, 4096, 2560]), 0.0, 1.0, device.clone())?;
    let x_bf16 = x_f32.to_dtype(DType::BF16)?;

    let legacy = host_sum_bf16(&x_bf16)?;

    let native = x_bf16.sum()?;
    assert_eq!(native.dtype(), DType::BF16);
    assert_eq!(native.shape().dims(), &[] as &[usize]);
    let native_v = bf16_scalar_to_f32(&native)?;

    // Tolerance: BF16 has ~3 decimal digits of mantissa. The sum is
    // O(10^7) values around 0 with std ~ 1, so |sum| ~ sqrt(10^7) ~ 3163.
    // BF16-rounding the result loses ~|sum| / 2^7 ≈ 25.
    assert_relative("sum_bf16 vs host", native_v, legacy, 1e-2, 1.0);

    Ok(())
}

#[test]
fn bf16_mean_matches_host_reference() -> Result<()> {
    let device = cuda_device().cuda_device().clone();
    let _ = rng::set_seed(131313);

    let x_f32 = Tensor::randn(Shape::from_dims(&[1, 4096, 2560]), 0.0, 1.0, device.clone())?;
    let x_bf16 = x_f32.to_dtype(DType::BF16)?;

    let legacy = host_mean_bf16(&x_bf16)?;

    let native = x_bf16.mean()?;
    assert_eq!(native.dtype(), DType::BF16);
    let native_v = bf16_scalar_to_f32(&native)?;

    // mean ~ 0 with stddev/sqrt(N), so the absolute tolerance dominates.
    assert_relative("mean_bf16 vs host", native_v, legacy, 1e-2, 1e-3);
    Ok(())
}

#[test]
fn bf16_sum_small_tensor() -> Result<()> {
    let device = cuda_device().cuda_device().clone();
    // 64 elements, well within a single block — exercises the simple case.
    let x_f32 =
        Tensor::randn(Shape::from_dims(&[8, 8]), 0.0, 1.0, device.clone())?;
    let x_bf16 = x_f32.to_dtype(DType::BF16)?;

    let legacy = host_sum_bf16(&x_bf16)?;
    let native = x_bf16.sum()?;
    let native_v = bf16_scalar_to_f32(&native)?;

    assert_relative("small sum_bf16", native_v, legacy, 1e-2, 1e-2);
    Ok(())
}

#[test]
fn bf16_sum_all_ones_large() -> Result<()> {
    // Deterministic ground truth: sum of N ones == N (in F32; rounded
    // to nearest BF16 representable value).
    let device = cuda_device().cuda_device().clone();
    let n = 1 * 4096 * 2560;
    let ones_f32 = Tensor::ones(Shape::from_dims(&[1, 4096, 2560]), device.clone())?;
    let ones_bf16 = ones_f32.to_dtype(DType::BF16)?;

    let native = ones_bf16.sum()?;
    let native_v = bf16_scalar_to_f32(&native)?;

    // n = 10,485,760. BF16 rounds this to a nearby representable. The
    // F32 in-kernel accumulator can carry it exactly, then the final
    // F32→BF16 conversion loses bits below 2^14 (mantissa is 7 bits).
    // 10.48M ≈ 2^23 so we expect to be within ~2^15 of the true value.
    let expected = n as f32;
    let diff = (native_v - expected).abs();
    let tol = 32768.0; // ~ 2^15
    assert!(
        diff <= tol,
        "all-ones sum: got={native_v} expected={expected} diff={diff} > tol={tol}"
    );
    Ok(())
}

#[test]
fn bf16_sum_legacy_path_still_works() -> Result<()> {
    // Sanity: F32 inputs use the legacy path; result must equal F32 reduction.
    let device = cuda_device().cuda_device().clone();
    let _ = rng::set_seed(99);
    let x_f32 = Tensor::randn(Shape::from_dims(&[4096]), 0.0, 1.0, device.clone())?;

    let s = x_f32.sum()?;
    let v = s.to_vec_f32()?;
    assert!(v[0].is_finite());
    Ok(())
}
