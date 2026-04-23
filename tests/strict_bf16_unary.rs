#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Phase 6 strict-mode test: BF16 unary activations must stay BF16 end-to-end.
//!
//! The five Phase 6 ops — abs, relu, sigmoid, tanh, neg — used to round-trip
//! through f32 via `GpuOps` or via `mul_scalar(-1.0)`. After Phase 6 the
//! `Tensor::*` dispatch for BF16 goes directly through the TensorIterator
//! pipeline and never materializes an intermediate f32 tensor.
//!
//! The test is a scaffold: for each op, invoke `Tensor::<op>()` on a BF16
//! tensor and verify the output dtype is BF16. Phase 7 tightens this with
//! telemetry (no F32 kernel launched). Phase 6's contribution is the
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
