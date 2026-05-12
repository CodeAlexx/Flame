#![cfg(all(feature = "cuda", feature = "bf16_u16"))]
//! Tests for `Tensor::eye` and `Tensor::detach`.
//!
//! These primitives back the LyCORIS family (OFT-Neumann needs `I + 2Q + ...`
//! and DoRA detaches the norm denominator from the autograd tape).

use flame_core::{device::Device, DType, Result, Shape, Tensor};

fn cuda_device() -> Device {
    Device::cuda(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

#[test]
fn eye_4x4_is_identity_f32() -> Result<()> {
    let dev = cuda_device().cuda_device().clone();
    let n = 4usize;

    let i = Tensor::eye(n, dev.clone())?;
    assert_eq!(i.shape().dims(), &[n, n]);
    assert_eq!(i.dtype(), DType::F32);

    let host = i.to_vec()?; // F32 readback
    assert_eq!(host.len(), n * n);

    for r in 0..n {
        for c in 0..n {
            let v = host[r * n + c];
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!(
                (v - expected).abs() < 1e-6,
                "eye({n})[{r},{c}] = {v}, expected {expected}",
            );
        }
    }
    Ok(())
}

#[test]
fn eye_dtype_bf16_matches() -> Result<()> {
    let dev = cuda_device().cuda_device().clone();
    let n = 4usize;

    let i = Tensor::eye_dtype(n, DType::BF16, dev.clone())?;
    assert_eq!(i.shape().dims(), &[n, n]);
    assert_eq!(i.dtype(), DType::BF16);

    // Cast back to F32 to inspect values portably.
    let host = i.to_dtype(DType::F32)?.to_vec()?;
    for r in 0..n {
        for c in 0..n {
            let v = host[r * n + c];
            let expected = if r == c { 1.0 } else { 0.0 };
            // BF16 represents 0.0 and 1.0 exactly.
            assert!(
                (v - expected).abs() < 1e-6,
                "eye_dtype(BF16)[{r},{c}] = {v}, expected {expected}",
            );
        }
    }
    Ok(())
}

#[test]
fn detach_breaks_tape() -> Result<()> {
    let dev = cuda_device().cuda_device().clone();

    // Leaf with grad enabled.
    let x = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        dev.clone(),
    )?
    .requires_grad_(true);
    assert!(x.requires_grad);

    // Detach: requires_grad must be false on the result.
    let xd = x.detach()?;
    assert!(
        !xd.requires_grad,
        "detach() must produce requires_grad=false"
    );

    // Fresh TensorId — not the same node in the graph.
    assert_ne!(xd.id(), x.id(), "detach() must produce a fresh TensorId");

    // Storage is a deep copy (clone_result), values match.
    let xv = x.to_vec()?;
    let xdv = xd.to_vec()?;
    assert_eq!(xv, xdv);

    Ok(())
}
