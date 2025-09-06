use anyhow::Result;
use flame_core::tensor::{Device, DType, Shape, Tensor};

#[test]
fn image_ops_edge_cases() -> Result<()> {
    let dev = Device::cuda(0);

    // Build ramp NHWC image [1,H,W,C]
    let (n, h, w, c) = (1usize, 23usize, 31usize, 3usize);
    let mut v = vec![0f32; n * h * w * c];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                v[((y * w + x) * c + ch)] = (y as f32) * 0.1 + (x as f32) * 0.01 + ch as f32 * 0.001;
            }
        }
    }
    let x = Tensor::from_vec_dtype(v, Shape::from_dims(&[n, h, w, c]), dev.cuda_device_arc(), DType::F32)?;

    // 1) Odd -> even resize (align_corners=false)
    let y = flame_core::image_ops_nhwc::resize_bilinear_nhwc(&x, 32, 32, false)?;
    assert_eq!(y.shape().dims(), &[1, 32, 32, 3]);
    assert!(y.is_finite()?);

    // Last pixel equals clamped bottom-right of source
    let br = x
        .narrow(1, h - 1, 1)?.narrow(2, w - 1, 1)?
        .narrow(3, 0, 1)?.squeeze(None)?.to_scalar::<f32>()?;
    let last = y
        .narrow(1, 31, 1)?.narrow(2, 31, 1)?
        .narrow(3, 0, 1)?.squeeze(None)?.to_scalar::<f32>()?;
    assert!((last - br).abs() < 1e-3);

    // 2) Center crop odd -> odd
    let z = flame_core::image_ops_nhwc::center_crop_nhwc(&x, 15, 21)?;
    assert_eq!(z.shape().dims(), &[1, 15, 21, 3]);

    // 3) Per-channel normalize
    let nrm = flame_core::image_ops_nhwc::normalize_nhwc(&z, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])?;
    assert!(nrm.is_finite()?);

    Ok(())
}

