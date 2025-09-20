#![cfg(test)]

use approx::assert_relative_eq;
use flame_core::autograd_ops_complete::{gelu_backward, silu_backward, softmax_backward};
use flame_core::{device::global_cuda_device, CudaDevice, Result, Shape, Tensor};
use libm::erff;
use std::f32::consts::SQRT_2;
use std::sync::Arc;

#[inline]
fn phi_pdf(x: f32) -> f32 {
    const INV_SQRT_2PI: f32 = 0.398_942_3_f32;
    (-0.5 * x * x).exp() * INV_SQRT_2PI
}

#[inline]
fn phi_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erff(x / SQRT_2))
}

fn cuda_device() -> Option<Arc<CudaDevice>> {
    Some(global_cuda_device().clone())
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
    let device = match cuda_device() {
        Some(dev) => dev,
        None => return Ok(()),
    };

    let x = linspace(&device, -5.0, 5.0, 513)?;
    let y = x.gelu()?;
    let dy = y.ones_like()?;
    let dx = gelu_backward(&dy, &x)?;

    let xs = x.to_vec_f32()?;
    let got_dx = dx.to_vec_f32()?;
    for (xv, dxv) in xs.iter().zip(got_dx.iter()) {
        let expected = phi_cdf(*xv) + *xv * phi_pdf(*xv);
        assert_relative_eq!(dxv, &expected, max_relative = 1e-4, epsilon = 1e-5);
    }
    Ok(())
}

#[test]
fn silu_backward_matches_analytic_dx() -> Result<()> {
    let device = match cuda_device() {
        Some(dev) => dev,
        None => return Ok(()),
    };

    let x = linspace(&device, -12.0, 12.0, 601)?;
    let y = x.silu()?;
    let dy = y.ones_like()?;
    let dx = silu_backward(&dy, &x)?;

    let xs = x.to_vec_f32()?;
    let got_dx = dx.to_vec_f32()?;
    for (xv, dxv) in xs.iter().zip(got_dx.iter()) {
        let s = 1.0 / (1.0 + (-xv).exp());
        let expected = s * (1.0 + xv * (1.0 - s));
        assert_relative_eq!(dxv, &expected, max_relative = 1e-4, epsilon = 1e-5);
    }
    Ok(())
}

#[test]
fn softmax_backward_matches_projection_rule_last_axis() -> Result<()> {
    let device = match cuda_device() {
        Some(dev) => dev,
        None => return Ok(()),
    };

    let shape = [2usize, 7];
    let x = randn(&device, &shape)?;
    let y = x.softmax(-1)?;
    let dy = randn(&device, &shape)?;
    let dx = softmax_backward(&y, &dy, -1)?;

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
                max_relative = 2e-4,
                epsilon = 1e-5
            );
        }
    }
    Ok(())
}
