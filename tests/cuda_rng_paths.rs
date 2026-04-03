#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, global_rng, Result, Shape, Tensor};

#[test]
fn randn_cuda_respects_global_seed() -> Result<()> {
    let device = global_cuda_device();
    // Touch the RNG so the test links the production singleton.
    let _ = global_rng();

    flame_core::rng::set_seed(4242)?;
    let a = Tensor::randn(Shape::from_dims(&[64]), 0.0, 1.0, device.clone())?;

    flame_core::rng::set_seed(4242)?;
    let b = Tensor::randn(Shape::from_dims(&[64]), 0.0, 1.0, device.clone())?;

    assert_eq!(a.to_vec_f32()?, b.to_vec_f32()?);
    Ok(())
}
