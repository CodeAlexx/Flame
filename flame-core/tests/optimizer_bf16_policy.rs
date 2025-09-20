use approx::assert_relative_eq;
use flame_core::adam::AdamW;
use flame_core::device::global_cuda_device;
use flame_core::parameter::Parameter;
use flame_core::sgd::{SGDConfig, SGD};
use flame_core::{DType, Result, Shape, Tensor};

fn make_init_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32).cos() * 1e-1).collect()
}

fn make_grads(n: usize, steps: usize) -> Vec<Vec<f32>> {
    (0..steps)
        .map(|k| (0..n).map(|i| ((i + k) as f32).sin() * 1e-2).collect())
        .collect()
}

#[test]
fn adamw_bf16_params_fp32_states_match_fp32_reference() -> Result<()> {
    let device = global_cuda_device().clone();

    let n = 4096usize;
    let steps = 100usize;
    let shape = Shape::from_dims(&[n]);
    let init = make_init_vec(n);
    let grads = make_grads(n, steps);

    // FP32 reference
    let tensor_ref =
        Tensor::from_vec(init.clone(), shape.clone(), device.clone())?.requires_grad_(true);
    let param_ref = Parameter::new(tensor_ref);
    let mut opt_ref = AdamW::default();
    for grad_vec in &grads {
        let grad = Tensor::from_vec(grad_vec.clone(), shape.clone(), device.clone())?;
        param_ref.set_grad(grad)?;
        opt_ref.step(std::slice::from_ref(&param_ref))?;
    }
    let p_ref_h = param_ref.tensor()?.to_vec_f32()?;

    // BF16 params with FP32 states
    let tensor_bf16 = Tensor::from_vec_dtype(init, shape.clone(), device.clone(), DType::BF16)?
        .requires_grad_(true);
    let param_bf16 = Parameter::new(tensor_bf16);
    let mut opt_bf16 = AdamW::default();
    for grad_vec in &grads {
        let grad = Tensor::from_vec(grad_vec.clone(), shape.clone(), device.clone())?;
        param_bf16.set_grad(grad)?;
        opt_bf16.step(std::slice::from_ref(&param_bf16))?;
        assert_eq!(param_bf16.tensor()?.dtype(), DType::BF16);
    }

    let p_bf16_f32 = param_bf16.tensor()?.to_vec_f32()?;
    for (a, b) in p_bf16_f32.iter().zip(p_ref_h.iter()) {
        assert_relative_eq!(a, b, max_relative = 1e-3, epsilon = 1e-5);
    }

    let state_bytes = opt_bf16.state_bytes();
    let expect = n * 2 * 4; // m & v FP32
    assert!(
        ((state_bytes as isize - expect as isize).abs() as f32) <= expect as f32 * 0.01,
        "state bytes off: got {}, expect ~{}",
        state_bytes,
        expect
    );

    Ok(())
}

#[test]
fn sgd_bf16_params_fp32_states_match_fp32_reference() -> Result<()> {
    let device = global_cuda_device().clone();

    let n = 4096usize;
    let steps = 100usize;
    let shape = Shape::from_dims(&[n]);
    let init = make_init_vec(n);
    let grads = make_grads(n, steps);

    let cfg = SGDConfig {
        lr: 1e-2,
        momentum: 0.9,
        weight_decay: 1e-4,
        nesterov: false,
    };

    // FP32 reference
    let tensor_ref =
        Tensor::from_vec(init.clone(), shape.clone(), device.clone())?.requires_grad_(true);
    let param_ref = Parameter::new(tensor_ref);
    let mut s_ref = SGD::new(cfg);
    for grad_vec in &grads {
        let grad = Tensor::from_vec(grad_vec.clone(), shape.clone(), device.clone())?;
        param_ref.set_grad(grad)?;
        s_ref.step(std::slice::from_ref(&param_ref))?;
    }
    let p_ref_h = param_ref.tensor()?.to_vec_f32()?;

    // BF16 params + FP32 velocity
    let tensor_bf16 = Tensor::from_vec_dtype(init, shape.clone(), device.clone(), DType::BF16)?
        .requires_grad_(true);
    let param_bf16 = Parameter::new(tensor_bf16);
    let mut s_bf16 = SGD::new(cfg);
    for grad_vec in &grads {
        let grad = Tensor::from_vec(grad_vec.clone(), shape.clone(), device.clone())?;
        param_bf16.set_grad(grad)?;
        s_bf16.step(std::slice::from_ref(&param_bf16))?;
        assert_eq!(param_bf16.tensor()?.dtype(), DType::BF16);
    }
    let p_bf16_f32 = param_bf16.tensor()?.to_vec_f32()?;
    for (a, b) in p_bf16_f32.iter().zip(p_ref_h.iter()) {
        assert_relative_eq!(a, b, max_relative = 1e-3, epsilon = 1e-5);
    }

    let state_bytes = s_bf16.state_bytes();
    let expect = n * 4; // single FP32 velocity buffer per param
    assert!(
        ((state_bytes as isize - expect as isize).abs() as f32) <= expect as f32 * 0.01,
        "state bytes off: got {}, expect ~{}",
        state_bytes,
        expect
    );
    Ok(())
}
