#![cfg(feature = "cuda")]

use approx::assert_relative_eq;
use flame_core::autograd_ops_complete::{gelu_backward, silu_backward, softmax_backward};
use flame_core::{global_cuda_device, set_default_dtype, CudaDevice, DType, Result, Shape, Tensor};
use half::bf16;
use libm::erff;
use std::f32::consts::SQRT_2;
use std::sync::Arc;

#[inline]
fn tolerances_for(dtype: DType) -> (f32, f32) {
    match dtype {
        DType::BF16 => (2e-2, 1e-2),
        DType::F32 => (1e-5, 1e-7),
        _ => (1e-5, 1e-7),
    }
}

#[inline]
fn bf16_round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

#[inline]
fn phi_pdf(x: f32) -> f32 {
    const INV_SQRT_2PI: f32 = 0.398_942_3_f32;
    (-0.5 * x * x).exp() * INV_SQRT_2PI
}

#[inline]
fn phi_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erff(x / SQRT_2))
}

fn cuda_device() -> Arc<CudaDevice> {
    global_cuda_device()
}

fn linspace(device: &Arc<CudaDevice>, start: f32, end: f32, steps: usize) -> Result<Tensor> {
    let mut data = Vec::with_capacity(steps);
    if steps == 0 {
        return Tensor::from_vec(Vec::new(), Shape::from_dims(&[0]), device.clone());
    }
    if steps == 1 {
        data.push(start);
    } else {
        let delta = (end - start) / ((steps - 1) as f32);
        for i in 0..steps {
            data.push(start + delta * i as f32);
        }
    }
    Tensor::from_vec(data, Shape::from_dims(&[steps]), device.clone())
}

fn randn(device: &Arc<CudaDevice>, shape: &[usize]) -> Result<Tensor> {
    Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, device.clone())
}

#[test]
fn gelu_backward_matches_analytic_dx() -> Result<()> {
    let device = cuda_device();
    set_default_dtype(DType::BF16);

    let x = linspace(&device, -5.0, 5.0, 513)?.to_dtype(DType::BF16)?;
    let y = x.gelu()?;
    let dy = y.ones_like()?.to_dtype(DType::BF16)?;
    assert_eq!(x.dtype(), DType::BF16);
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(dy.dtype(), DType::BF16);

    let dx = gelu_backward(&dy, &x)?;
    assert_eq!(dx.dtype(), DType::BF16);

    let xs = x.to_vec_f32()?;
    let got_dx = dx.to_vec_f32()?;
    let (rtol, atol) = tolerances_for(dx.dtype());
    for (xv, dxv) in xs.iter().zip(got_dx.iter()) {
        let expected = bf16_round(phi_cdf(*xv) + *xv * phi_pdf(*xv));
        assert_relative_eq!(*dxv, expected, max_relative = rtol, epsilon = atol);
    }
    Ok(())
}

#[test]
fn silu_backward_matches_analytic_dx() -> Result<()> {
    let device = cuda_device();
    set_default_dtype(DType::BF16);

    let x = linspace(&device, -12.0, 12.0, 601)?.to_dtype(DType::BF16)?;
    let y = x.silu()?;
    let dy = y.ones_like()?.to_dtype(DType::BF16)?;
    assert_eq!(x.dtype(), DType::BF16);
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(dy.dtype(), DType::BF16);

    let dx = silu_backward(&dy, &x)?;
    assert_eq!(dx.dtype(), DType::BF16);

    let xs = x.to_vec_f32()?;
    let got_dx = dx.to_vec_f32()?;
    let (rtol, atol) = tolerances_for(dx.dtype());
    for (xv, dxv) in xs.iter().zip(got_dx.iter()) {
        let s = 1.0 / (1.0 + (-xv).exp());
        let expected = bf16_round(s * (1.0 + xv * (1.0 - s)));
        assert_relative_eq!(*dxv, expected, max_relative = rtol, epsilon = atol);
    }
    Ok(())
}

#[test]
fn gelu_backward_random_no_nans() -> Result<()> {
    let device = cuda_device();
    set_default_dtype(DType::BF16);

    let x = Tensor::randn(Shape::from_dims(&[8, 256]), 0.0, 3.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let dy = Tensor::randn(Shape::from_dims(&[8, 256]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    let dx = gelu_backward(&dy, &x)?;
    assert_eq!(dx.dtype(), DType::BF16);

    let dx_host = dx.to_vec_f32()?;
    assert!(
        dx_host.iter().all(|v| v.is_finite()),
        "gelu_backward produced non-finite gradients"
    );
    Ok(())
}

#[test]
fn gelu_backward_random_allclose() -> Result<()> {
    let device = cuda_device();
    set_default_dtype(DType::BF16);

    let (rtol, atol) = tolerances_for(DType::BF16);
    for _ in 0..5 {
        let x = Tensor::randn(Shape::from_dims(&[4, 128]), 0.0, 3.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let dy = Tensor::randn(Shape::from_dims(&[4, 128]), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let dx = gelu_backward(&dy, &x)?;

        let xs = x.to_vec_f32()?;
        let dys = dy.to_vec_f32()?;
        let dxs = dx.to_vec_f32()?;
        for ((xv, dyv), dxv) in xs.iter().zip(dys.iter()).zip(dxs.iter()) {
            let expected = bf16_round(*dyv * (phi_cdf(*xv) + *xv * phi_pdf(*xv)));
            assert_relative_eq!(*dxv, expected, max_relative = rtol, epsilon = atol);
        }
    }
    Ok(())
}

#[test]
fn silu_backward_random_no_nans() -> Result<()> {
    let device = cuda_device();
    set_default_dtype(DType::BF16);

    let x = Tensor::randn(Shape::from_dims(&[8, 256]), 0.0, 4.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let dy = Tensor::randn(Shape::from_dims(&[8, 256]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    let dx = silu_backward(&dy, &x)?;
    assert_eq!(dx.dtype(), DType::BF16);

    let dx_host = dx.to_vec_f32()?;
    assert!(
        dx_host.iter().all(|v| v.is_finite()),
        "silu_backward produced non-finite gradients"
    );
    Ok(())
}

#[test]
fn silu_backward_random_allclose() -> Result<()> {
    let device = cuda_device();
    set_default_dtype(DType::BF16);

    let (rtol, atol) = tolerances_for(DType::BF16);
    for _ in 0..5 {
        let x = Tensor::randn(Shape::from_dims(&[4, 128]), 0.0, 4.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let dy = Tensor::randn(Shape::from_dims(&[4, 128]), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let dx = silu_backward(&dy, &x)?;

        let xs = x.to_vec_f32()?;
        let dys = dy.to_vec_f32()?;
        let dxs = dx.to_vec_f32()?;
        for ((xv, dyv), dxv) in xs.iter().zip(dys.iter()).zip(dxs.iter()) {
            let s = 1.0 / (1.0 + (-xv).exp());
            let expected = bf16_round(*dyv * (s * (1.0 + xv * (1.0 - s))));
            assert_relative_eq!(*dxv, expected, max_relative = rtol, epsilon = atol);
        }
    }
    Ok(())
}

#[test]
fn softmax_backward_matches_projection_rule_last_axis() -> Result<()> {
    let device = cuda_device();
    set_default_dtype(DType::BF16);

    let shape = [2usize, 7];
    let x = randn(&device, &shape)?.to_dtype(DType::BF16)?;
    let y = x.softmax(-1)?;
    let dy = randn(&device, &shape)?.to_dtype(DType::BF16)?;
    let dx = softmax_backward(&y, &dy, -1)?;

    assert_eq!(x.dtype(), DType::BF16);
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(dy.dtype(), DType::BF16);
    assert_eq!(dx.dtype(), DType::BF16);

    let y_host = y.to_vec_f32()?;
    let dy_host = dy.to_vec_f32()?;
    let dx_host = dx.to_vec_f32()?;
    let cols = shape[1];

    for b in 0..shape[0] {
        let offset = b * cols;
        let dot: f32 = (0..cols)
            .map(|c| dy_host[offset + c] * y_host[offset + c])
            .sum();
        for c in 0..cols {
            let expected = y_host[offset + c] * (dy_host[offset + c] - dot);
            assert_relative_eq!(
                dx_host[offset + c],
                expected,
                max_relative = 1e-3,
                epsilon = 3e-3
            );
        }
    }
    Ok(())
}
