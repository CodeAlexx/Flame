use flame_core::*;

#[test]
fn image_ops_nhwc_smoke() -> Result<()> {
    let dev = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::F32);

    // Random NHWC image batch
    let x = Tensor::randn(Shape::from_dims(&[2, 23, 31, 3]), 0.0, 1.0, dev.clone())?;

    // Resize to 32x32
    let r = image_ops_nhwc::resize_bilinear_nhwc(&x, 32, 32, false)?;
    assert_eq!(r.shape().dims(), &[2, 32, 32, 3]);

    // Center crop to 28x28
    let c = image_ops_nhwc::center_crop_nhwc(&r, 28, 28)?;
    assert_eq!(c.shape().dims(), &[2, 28, 28, 3]);

    // Normalize with mean=0.5, std=0.5 per channel
    let mean = [0.5f32, 0.5f32, 0.5f32];
    let std = [0.5f32, 0.5f32, 0.5f32];
    let y = image_ops_nhwc::normalize_nhwc(&c, &mean, &std)?;
    let vals = y.to_vec()?;
    assert!(vals.iter().all(|v| v.is_finite()));

    // Stats sanity: mean(|y|) should be roughly centered near 0 for random inputs
    let mean_abs = vals.iter().map(|v| v.abs()).sum::<f32>() / vals.len() as f32;
    assert!(mean_abs < 0.2, "mean_abs too large: {}", mean_abs);
    Ok(())
}

