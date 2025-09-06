use flame_core::*;

fn mean_scalar(t: &Tensor) -> Result<f32> {
    let m = t.mean()?;
    Ok(m.to_vec()?[0])
}

#[test]
fn exp_log_max_gpu_smoke() -> Result<()> {
    set_default_dtype(DType::BF16);
    let device = CudaDevice::new(0).unwrap();

    // exp/log round-trip
    let x = Tensor::rand(Shape::from_dims(&[1024]), device.clone())?.requires_grad_(true);
    let y_exp = x.exp()?;
    let y_log = y_exp.log()?;
    let diff = y_log.sub(&x)?.abs()?;
    let m = mean_scalar(&diff)?;
    assert!(m < 1e-3, "exp/log round-trip mean abs {}", m);

    // elementwise maximum + backward
    let a = Tensor::rand(Shape::from_dims(&[1024]), device.clone())?.requires_grad_(true);
    let b = Tensor::rand(Shape::from_dims(&[1024]), device.clone())?.requires_grad_(true);
    let mmax = a.maximum(&b)?;
    let loss = mmax.mean()?;
    let grads = AutogradContext::backward(&loss)?;
    let ga = grads.get(a.id()).unwrap().clone()?;
    let gb = grads.get(b.id()).unwrap().clone()?;
    assert!(ga.to_vec()?.iter().all(|v| v.is_finite()));
    assert!(gb.to_vec()?.iter().all(|v| v.is_finite()));

    Ok(())
}

