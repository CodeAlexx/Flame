use flame_core::*;

#[test]
fn vae_encode_decode_shapes() -> Result<()> {
    let dev = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::BF16);
    let vae = vae::AutoencoderKL::new_random(dev.clone(), DType::BF16)?;

    let x = Tensor::randn(Shape::from_dims(&[2,256,256,3]), 0.0, 1.0, dev.clone())?;
    let z = vae.encode(&x)?;
    assert_eq!(z.shape().dims(), &[2,32,32,4]);
    let y = vae.decode(&z)?;
    assert_eq!(y.shape().dims(), &[2,256,256,3]);
    let s = y.to_vec()?;
    assert!(s.iter().all(|v| v.is_finite()));
    Ok(())
}

