#![cfg(feature = "cuda")]

//! Regression tests for two issues surfaced by the LanPaint Rust port:
//!
//! 1. `Tensor::clamp` used to build its min/max constant tensors via
//!    `full_like`, which applies the workspace default dtype. With the
//!    default set to BF16 (as it is in production), clamping an F32
//!    tensor would build BF16 constants and trip a dtype-mismatch inside
//!    `maximum` / `minimum`.
//! 2. `Tensor::randn_seeded` is a new deterministic sibling of `randn`
//!    driven by `StdRng::seed_from_u64` + CPU Box-Muller. It exists so
//!    LanPaint-style parity tests against Python references can inject
//!    reproducible noise.

use flame_core::{default_dtype, global_cuda_device, DType, Result, Shape, Tensor};

#[test]
fn clamp_preserves_f32_dtype() -> Result<()> {
    // Workspace default is BF16 in production. This is the failure mode
    // the old `clamp` impl produced when called on F32 — full_like would
    // build BF16 constants and maximum/minimum would reject the mismatch.
    assert_eq!(
        default_dtype(),
        DType::BF16,
        "test assumes production default (BF16); adjust if this changes"
    );

    let device = global_cuda_device();
    let t = Tensor::from_vec(
        vec![-2.0_f32, 0.5, 3.0],
        Shape::from_dims(&[3]),
        device.clone(),
    )?
    .to_dtype(DType::F32)?;
    assert_eq!(t.dtype(), DType::F32);

    let clamped = t.clamp(-1.0, 1.0)?;
    assert_eq!(clamped.dtype(), DType::F32, "clamp must preserve dtype");

    let v = clamped.to_vec_f32()?;
    assert!((v[0] + 1.0).abs() < 1e-6, "v[0]={}", v[0]);
    assert!((v[1] - 0.5).abs() < 1e-6, "v[1]={}", v[1]);
    assert!((v[2] - 1.0).abs() < 1e-6, "v[2]={}", v[2]);
    Ok(())
}

#[test]
fn clamp_works_on_bf16() -> Result<()> {
    // Sanity: clamp on BF16 input still produces BF16 output (no regression
    // for the common case).
    let device = global_cuda_device();
    let t = Tensor::from_vec(
        vec![-2.0_f32, 0.5, 3.0],
        Shape::from_dims(&[3]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let clamped = t.clamp(-1.0, 1.0)?;
    assert_eq!(clamped.dtype(), DType::BF16);

    let v = clamped.to_dtype(DType::F32)?.to_vec_f32()?;
    // BF16 has ~3 decimal digits of precision; generous tolerance.
    assert!((v[0] + 1.0).abs() < 1e-2, "v[0]={}", v[0]);
    assert!((v[1] - 0.5).abs() < 1e-2, "v[1]={}", v[1]);
    assert!((v[2] - 1.0).abs() < 1e-2, "v[2]={}", v[2]);
    Ok(())
}

#[test]
fn randn_seeded_is_deterministic() -> Result<()> {
    let device = global_cuda_device();
    let a = Tensor::randn_seeded(
        Shape::from_dims(&[1024]),
        0.0,
        1.0,
        42,
        device.clone(),
    )?;
    let b = Tensor::randn_seeded(
        Shape::from_dims(&[1024]),
        0.0,
        1.0,
        42,
        device.clone(),
    )?;

    // Compare in F32 to avoid BF16 rounding if the workspace default is BF16.
    let va = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let vb = b.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(va, vb, "randn_seeded must be bit-deterministic for a seed");
    Ok(())
}

#[test]
fn randn_seeded_stats() -> Result<()> {
    let device = global_cuda_device();
    let a = Tensor::randn_seeded(
        Shape::from_dims(&[16384]),
        0.0,
        1.0,
        42,
        device.clone(),
    )?;
    // Stats in F32 so BF16 quantization (if any) doesn't skew variance.
    let v = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let n = v.len() as f32;
    let mean: f32 = v.iter().sum::<f32>() / n;
    let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    // BF16 round-trip can widen variance slightly; keep tolerances loose
    // enough to survive that but tight enough to catch a broken Box-Muller.
    assert!(mean.abs() < 0.05, "mean={mean}");
    assert!((var - 1.0).abs() < 0.1, "var={var}");
    Ok(())
}

#[test]
fn randn_seeded_different_seeds_differ() -> Result<()> {
    let device = global_cuda_device();
    let a = Tensor::randn_seeded(Shape::from_dims(&[256]), 0.0, 1.0, 1, device.clone())?;
    let b = Tensor::randn_seeded(Shape::from_dims(&[256]), 0.0, 1.0, 2, device.clone())?;
    let va = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let vb = b.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_ne!(va, vb, "distinct seeds must produce distinct streams");
    Ok(())
}
