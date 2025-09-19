use flame_core::{adam::AdamW, CudaDevice, DType, Parameter, Result, Shape, Tensor};
#[cfg(feature = "true_bf16_optimizer_states")]
use flame_core::config;

#[test]
fn adamw_states_are_fp32_even_with_bf16_params() -> Result<()> {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };

    let param_tensor = Tensor::randn(Shape::from_dims(&[4, 4]), 0.0, 0.1, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let param = Parameter::new(param_tensor);

    let grad = Tensor::ones_dtype(param.shape(), DType::BF16, device.clone())?;
    param.set_grad(grad)?;

    let mut opt = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    opt.step(&[param.clone()])?;

    let (m_dt, v_dt) = opt
        .debug_state_dtype(&param)
        .expect("optimizer state should exist");
    assert_eq!(m_dt, DType::F32);
    assert_eq!(v_dt, DType::F32);

    assert_eq!(param.tensor()?.dtype(), DType::BF16);

    Ok(())
}

#[cfg(feature = "true_bf16_optimizer_states")]
#[test]
fn adamw_states_follow_config_dtype() -> Result<()> {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };

    let prev = config::optimizer_moment_dtype();
    config::set_optimizer_moment_dtype(DType::BF16);

    let param_tensor = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 0.1, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let param = Parameter::new(param_tensor);

    let grad = Tensor::ones_dtype(param.shape(), DType::BF16, device.clone())?;
    param.set_grad(grad)?;

    let mut opt = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    opt.step(&[param.clone()])?;

    let (m_dt, v_dt) = opt
        .debug_state_dtype(&param)
        .expect("optimizer state should exist");
    assert_eq!(m_dt, DType::BF16);
    assert_eq!(v_dt, DType::BF16);

    config::set_optimizer_moment_dtype(prev);

    Ok(())
}
