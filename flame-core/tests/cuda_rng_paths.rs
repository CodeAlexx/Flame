#![cfg(test)]

use flame_core::{device::global_cuda_device, CudaDevice, Result, Shape, Tensor};
use std::sync::Arc;

fn cuda_enabled() -> bool {
    std::env::var("FLAME_CUDA_SMOKE").ok().as_deref() == Some("1")
}

fn cuda_device() -> Option<Arc<CudaDevice>> {
    if !cuda_enabled() {
        return None;
    }
    Some(global_cuda_device().clone())
}

#[test]
fn randn_cuda_respects_global_seed() -> Result<()> {
    let device = match cuda_device() {
        Some(dev) => dev,
        None => return Ok(()),
    };

    flame_core::rng::set_seed(4242)?;
    let a = Tensor::randn(Shape::from_dims(&[64]), 0.0, 1.0, device.clone())?;

    flame_core::rng::set_seed(4242)?;
    let b = Tensor::randn(Shape::from_dims(&[64]), 0.0, 1.0, device.clone())?;

    assert_eq!(a.to_vec_f32()?, b.to_vec_f32()?);
    Ok(())
}
