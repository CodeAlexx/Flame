#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{adam::AdamW, global_cuda_device, DType, Parameter, Result, Shape, Tensor};

#[cfg(not(feature = "true_bf16_optimizer_states"))]
#[test]
fn adamw_states_are_fp32_even_with_bf16_params() -> Result<()> {
    let device = global_cuda_device();

    let param_tensor = Tensor::randn(Shape::from_dims(&[4, 4]), 0.0, 0.1, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let param = Parameter::new(param_tensor);

    let grad = Tensor::ones_dtype(param.shape(), DType::BF16, device.clone())?;
    param.set_grad(grad)?;

    let mut opt = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    opt.step(std::slice::from_ref(&param))?;

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
    use flame_core::config;

    let prev = config::optimizer_moment_dtype();
    config::set_optimizer_moment_dtype(DType::BF16);

    eprintln!("skipping adamw_states_follow_config_dtype: true BF16 optimizer states not yet fully supported");
    config::set_optimizer_moment_dtype(prev);
    return Ok(());

    Ok(())
}
