use flame_core::*;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let diff = a.sub(b)?.abs()?;
    let vals = diff.to_vec()?;
    Ok(vals.into_iter().fold(0f32, |m, v| if v > m { v } else { m }))
}

#[test]
fn conv2d_nhwc_forward_backward_parity() -> Result<()> {
    let device = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::BF16);

    let (n, h, w, ic, oc) = (2usize, 10, 11, 3, 5);
    let (kh, kw) = (3usize, 3usize);
    let stride = (2usize, 1usize);
    let padding = (1usize, 1usize);

    // NHWC tensors with grads
    let x_nhwc = Tensor::rand(Shape::from_dims(&[n, h, w, ic]), device.clone())?.requires_grad_(true);
    let w_khwkicoc = Tensor::rand(Shape::from_dims(&[kh, kw, ic, oc]), device.clone())?.requires_grad_(true);
    let b = Tensor::rand(Shape::from_dims(&[oc]), device.clone())?.requires_grad_(true);

    // NHWC adapter forward
    let y_nhwc = cuda_conv2d::CudaConv2d::conv2d_forward_nhwc(&x_nhwc, &w_khwkicoc, Some(&b), stride, padding)?;

    // NCHW reference forward (manual permutations)
    let x_nchw = cuda_ops::GpuOps::permute_nhwc_to_nchw(&x_nhwc)?;
    let w_ocic = cuda_ops::GpuOps::weight_khwkicoc_to_ocickhkw(&w_khwkicoc)?;
    let y_ref_nchw = cuda_conv2d::CudaConv2d::conv2d_forward(&x_nchw, &w_ocic, Some(&b), stride, padding, 1)?;
    let y_ref_nhwc = cuda_ops::GpuOps::permute_nchw_to_nhwc(&y_ref_nchw)?;

    // Forward parity
    let fwd_diff = max_abs_diff(&y_nhwc, &y_ref_nhwc)?;
    assert!(fwd_diff < 1e-4, "forward max abs diff {}", fwd_diff);

    // Backward: NHWC path
    let loss = y_nhwc.mean()?;
    let grads = AutogradContext::backward(&loss)?;
    let gx_nhwc = grads.get(x_nhwc.id()).unwrap().clone()?;
    let gw_khwkicoc = grads.get(w_khwkicoc.id()).unwrap().clone()?;
    let gb = grads.get(b.id()).unwrap().clone()?;

    // Backward: NCHW reference
    let x2_nchw = x_nchw.clone()?.requires_grad_(true);
    let w2_ocic = w_ocic.clone()?.requires_grad_(true);
    let b2 = b.clone()?.requires_grad_(true);
    let y2_nchw = cuda_conv2d::CudaConv2d::conv2d_forward(&x2_nchw, &w2_ocic, Some(&b2), stride, padding, 1)?;
    let loss2 = y2_nchw.mean()?;
    let grads2 = AutogradContext::backward(&loss2)?;
    let gx2_nchw = grads2.get(x2_nchw.id()).unwrap().clone()?;
    let gw2_ocic = grads2.get(w2_ocic.id()).unwrap().clone()?;
    let gb2 = grads2.get(b2.id()).unwrap().clone()?;

    // Convert reference grads back to NHWC / [KH,KW,IC,OC]
    let gx2_nhwc = cuda_ops::GpuOps::permute_nchw_to_nhwc(&gx2_nchw)?;
    let gw2_khwkicoc = cuda_ops::GpuOps::weight_ocickhkw_to_khwkicoc(&gw2_ocic)?;

    let dx_diff = max_abs_diff(&gx_nhwc, &gx2_nhwc)?;
    let dw_diff = max_abs_diff(&gw_khwkicoc, &gw2_khwkicoc)?;
    let db_diff = max_abs_diff(&gb, &gb2)?;
    assert!(dx_diff < 1e-3, "dx max abs diff {}", dx_diff);
    assert!(dw_diff < 1e-3, "dw max abs diff {}", dw_diff);
    assert!(db_diff < 1e-3, "db max abs diff {}", db_diff);

    Ok(())
}

