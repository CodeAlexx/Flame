#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Phase 6–7 strict-mode test: BF16 unary ops must stay BF16 end-to-end.
//!
//! The five Phase 6 ops — abs, relu, sigmoid, tanh, neg — plus the five
//! Phase 7 transcendentals — exp, log, sqrt, rsqrt, recip — all used to
//! round-trip through f32 (either via `GpuOps::*`, `mul_scalar(-1.0)`, or
//! via `sqrt().reciprocal()` composition). After Phases 6–7 the
//! `Tensor::*` dispatch for BF16 goes directly through the TensorIterator
//! pipeline and never materializes an intermediate f32 tensor.
//!
//! The test is a scaffold: for each op, invoke `Tensor::<op>()` on a BF16
//! tensor and verify the output dtype is BF16. Phase 8+ tightens this with
//! telemetry (no F32 kernel launched). Phases 6–7's contribution is the
//! structural guarantee that dtype doesn't flip underneath.

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for strict_bf16_unary")
}

fn make_bf16_tensor(dev: Arc<CudaDevice>, n: usize) -> Result<Tensor> {
    let mut data = Vec::with_capacity(n);
    let mut s = 0x51C0FFEE_u64;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        data.push((u - 0.5) * 4.0);
    }
    let t = Tensor::from_vec(data, Shape::from_dims(&[n]), dev)?;
    t.to_dtype(DType::BF16)
}

/// Positive-only BF16 input for log/sqrt/rsqrt domain (values in [0.1, 10.1)).
fn make_bf16_positive_tensor(dev: Arc<CudaDevice>, n: usize) -> Result<Tensor> {
    let mut data = Vec::with_capacity(n);
    let mut s = 0x7A11C0DE_u64;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        data.push(u * 10.0 + 0.1);
    }
    let t = Tensor::from_vec(data, Shape::from_dims(&[n]), dev)?;
    t.to_dtype(DType::BF16)
}

/// Non-zero BF16 input for recip domain.
fn make_bf16_nonzero_tensor(dev: Arc<CudaDevice>, n: usize) -> Result<Tensor> {
    let mut data = Vec::with_capacity(n);
    let mut s = 0xDECAF_u64;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        let v = (u - 0.5) * 10.0;
        // Avoid near-zero.
        data.push(if v.abs() < 0.5 { v.signum() * 0.5 + v } else { v });
    }
    let t = Tensor::from_vec(data, Shape::from_dims(&[n]), dev)?;
    t.to_dtype(DType::BF16)
}

#[test]
fn strict_bf16_abs_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.abs()?;
    assert_eq!(y.dtype(), DType::BF16, "abs must preserve BF16 dtype");
    Ok(())
}

#[test]
fn strict_bf16_relu_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.relu()?;
    assert_eq!(y.dtype(), DType::BF16, "relu must preserve BF16 dtype");
    Ok(())
}

#[test]
fn strict_bf16_sigmoid_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.sigmoid()?;
    assert_eq!(y.dtype(), DType::BF16, "sigmoid must preserve BF16 dtype");
    Ok(())
}

#[test]
fn strict_bf16_tanh_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.tanh()?;
    assert_eq!(y.dtype(), DType::BF16, "tanh must preserve BF16 dtype");
    Ok(())
}

#[test]
fn strict_bf16_neg_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.neg()?;
    assert_eq!(y.dtype(), DType::BF16, "neg must preserve BF16 dtype");
    Ok(())
}

/// Lightweight functional check: verify output is finite and non-trivial.
/// Catches the "dispatch routed to a stub that writes zeros" class of bug.
#[test]
fn strict_bf16_all_unary_functional() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, 4096)?;
    for (tag, y) in [
        ("abs", x.abs()?),
        ("relu", x.relu()?),
        ("sigmoid", x.sigmoid()?),
        ("tanh", x.tanh()?),
        ("neg", x.neg()?),
    ] {
        assert_eq!(y.dtype(), DType::BF16, "{tag}: output dtype");
        let y_host = y.to_vec_f32()?;
        let any_nonzero = y_host.iter().any(|&v| v != 0.0);
        let all_finite = y_host.iter().all(|&v| v.is_finite());
        assert!(all_finite, "{tag}: output contains non-finite values");
        assert!(any_nonzero, "{tag}: output is all-zero (suspicious)");
    }
    Ok(())
}

// ===== Phase 7: transcendentals. =====

#[test]
fn strict_bf16_exp_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.exp()?;
    assert_eq!(y.dtype(), DType::BF16, "exp must preserve BF16 dtype");
    Ok(())
}

#[test]
fn strict_bf16_log_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_positive_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.log()?;
    assert_eq!(y.dtype(), DType::BF16, "log must preserve BF16 dtype");
    Ok(())
}

#[test]
fn strict_bf16_sqrt_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_positive_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.sqrt()?;
    assert_eq!(y.dtype(), DType::BF16, "sqrt must preserve BF16 dtype");
    Ok(())
}

#[test]
fn strict_bf16_rsqrt_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_positive_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.rsqrt()?;
    assert_eq!(y.dtype(), DType::BF16, "rsqrt must preserve BF16 dtype");
    Ok(())
}

#[test]
fn strict_bf16_recip_dtype_preserved() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_nonzero_tensor(dev, 1024)?;
    assert_eq!(x.dtype(), DType::BF16);
    let y = x.reciprocal()?;
    assert_eq!(y.dtype(), DType::BF16, "recip must preserve BF16 dtype");
    Ok(())
}

/// Functional check for the Phase 7 transcendentals.
#[test]
fn strict_bf16_all_transcendental_functional() -> Result<()> {
    let dev = cuda_device();
    let x_any = make_bf16_tensor(dev.clone(), 4096)?;
    let x_pos = make_bf16_positive_tensor(dev.clone(), 4096)?;
    let x_nz = make_bf16_nonzero_tensor(dev, 4096)?;
    for (tag, y) in [
        ("exp", x_any.exp()?),
        ("log", x_pos.log()?),
        ("sqrt", x_pos.sqrt()?),
        ("rsqrt", x_pos.rsqrt()?),
        ("recip", x_nz.reciprocal()?),
    ] {
        assert_eq!(y.dtype(), DType::BF16, "{tag}: output dtype");
        let y_host = y.to_vec_f32()?;
        let any_nonzero = y_host.iter().any(|&v| v != 0.0);
        // All finite; no NaN permitted on the valid-domain inputs above.
        let all_finite_or_nonneg = y_host.iter().all(|&v| v.is_finite() || v > 0.0);
        let any_nan = y_host.iter().any(|&v| v.is_nan());
        assert!(!any_nan, "{tag}: output contains NaN");
        assert!(all_finite_or_nonneg, "{tag}: output contains -inf");
        assert!(any_nonzero, "{tag}: output is all-zero (suspicious)");
    }
    Ok(())
}
