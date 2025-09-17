use flame_core::*;

fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() <= tol }

#[test]
fn softmax_backward_gpu() -> Result<()> {
    let device = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::BF16);
    let x = Tensor::randn(Shape::from_dims(&[2, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = x.softmax(-1)?; // shifted-exp path
    // simple loss: sum
    let loss = y.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    // Check gradient exists and is finite
    let gx = grads.get(x.id()).unwrap().clone_result()?;
    let nans = gx.to_vec()?.iter().any(|v| !v.is_finite());
    assert!(!nans, "softmax backward produced NaNs");
    Ok(())
}

#[test]
fn layernorm_backward_gpu() -> Result<()> {
    let device = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::BF16);
    let x = Tensor::randn(Shape::from_dims(&[3, 8]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let ln = layer_norm::LayerNorm::new(8, device.clone())?;
    let y = ln.forward(&x)?;
    let loss = y.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    let gx = grads.get(x.id()).unwrap().clone_result()?;
    assert!(gx.shape().dims() == &[3,8]);
    Ok(())
}

#[test]
fn groupnorm_backward_gpu() -> Result<()> {
    let device = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::BF16);
    // Shape [N,C,H,W] used by existing GN; this is a sanity presence test
    let x = Tensor::randn(Shape::from_dims(&[2, 4, 2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = group_norm(&x, 2, None, None)?;
    let loss = y.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    let gx = grads.get(x.id()).unwrap().clone_result()?;
    assert!(gx.shape().dims() == &[2,4,2,2]);
    Ok(())
}

#[test]
fn matmul_backward_gpu() -> Result<()> {
    let device = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::BF16);
    let a = Tensor::randn(Shape::from_dims(&[5, 3]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let b = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = a.matmul(&b)?;
    let loss = y.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    assert!(grads.get(a.id()).is_some() && grads.get(b.id()).is_some());
    Ok(())
}

#[test]
fn conv2d_layout_asserts() {
    // Verify we fail closed when NHWC/[KH,KW,IC,OC] not provided
    let device = CudaDevice::new(0).unwrap();
    let x = Tensor::randn(Shape::from_dims(&[1, 3, 8, 8]), 0.0, 1.0, device.clone()).unwrap(); // NCHW
    let w = Tensor::randn(Shape::from_dims(&[16, 3, 3, 3]), 0.0, 1.0, device.clone()).unwrap(); // [OC,IC,KH,KW]
    let err = cuda_conv2d::CudaConv2d::conv2d_forward(&x, &w, None, (1,1), (1,1), 1).unwrap_err();
    if let FlameError::InvalidOperation(msg) = err { assert!(msg.contains("NHWC")); } else { panic!("unexpected error"); }
}
