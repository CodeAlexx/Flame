use flame_core::*;

#[test]
fn conv2d_nhwc_forward_backward_smoke() -> Result<()> {
    // Set default dtype for params/activations; grads are FP32 internally
    set_default_dtype(DType::BF16);
    let device = CudaDevice::new(0).unwrap();

    // NHWC input [N,H,W,C]
    let (n, h, w, ic) = (2usize, 8usize, 8usize, 3usize);
    let x = Tensor::rand(Shape::from_dims(&[n, h, w, ic]), device.clone())?
        .requires_grad_(true);

    // Weights [KH,KW,IC,OC]
    let (kh, kw, oc) = (3usize, 3usize, 4usize);
    let w = Tensor::rand(Shape::from_dims(&[kh, kw, ic, oc]), device.clone())?
        .requires_grad_(true);

    // Bias [OC]
    let b = Tensor::zeros(Shape::from_dims(&[oc]), device.clone())?
        .requires_grad_(true);

    // Stride/padding
    let stride = (1usize, 1usize);
    let padding = (1usize, 1usize);

    // Forward via NHWC adapter
    let y = cuda_conv2d::CudaConv2d::conv2d_forward_nhwc(&x, &w, Some(&b), stride, padding)?;
    let yd = y.shape().dims().to_vec();
    assert_eq!(yd, vec![n, h, w, oc]);

    // Simple scalar loss
    let loss = y.sum()?;
    let grads = AutogradContext::backward(&loss)?;

    // Check grads exist and shapes match
    let gx = grads.get(x.id()).unwrap().clone()?;
    let gw = grads.get(w.id()).unwrap().clone()?;
    let gb = grads.get(b.id()).unwrap().clone()?;
    assert_eq!(gx.shape().dims(), &[n, h, w, ic]);
    assert_eq!(gw.shape().dims(), &[kh, kw, ic, oc]);
    assert_eq!(gb.shape().dims(), &[oc]);

    // Parity vs. direct NCHW reference path
    let x_nchw = cuda_ops::GpuOps::permute_nhwc_to_nchw(&x)?;
    let w_ocic = cuda_ops::GpuOps::weight_khwkicoc_to_ocickhkw(&w)?;
    let y_ref_nchw = cuda_conv2d::CudaConv2d::conv2d_forward(&x_nchw, &w_ocic, Some(&b), stride, padding, 1)?;
    let y_ref = cuda_ops::GpuOps::permute_nchw_to_nhwc(&y_ref_nchw)?;
    // Compare elementwise closely
    let a = y.to_vec()?;
    let bvec = y_ref.to_vec()?;
    assert_eq!(a.len(), bvec.len());
    let mut max_abs = 0f32;
    for i in 0..a.len() { max_abs = max_abs.max((a[i] - bvec[i]).abs()); }
    // Allow tiny numerical jitter
    assert!(max_abs < 1e-4, "parity max_abs={}", max_abs);

    Ok(())
}
