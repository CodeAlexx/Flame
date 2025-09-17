use flame_core::{Tensor, Shape, CudaDevice, AutogradContext};
use std::sync::Arc;

fn mlp_forward(x: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor, b2: &Tensor) -> flame_core::Result<Tensor> {
    let h = x.matmul(w1)?.add(b1)?.relu()?;
    let y = h.matmul(w2)?.add(b2)?;
    Ok(y)
}

#[test]
fn gradient_checkpoint_smoke() -> flame_core::Result<()> {
    // Skip test if CUDA not available
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let bs = 32usize; let in_dim = 64usize; let hid = 48usize; let out = 16usize;
    let x = Tensor::randn(Shape::from_dims(&[bs, in_dim]), 0.0, 1.0, device.clone())?.requires_grad_(false);
    let w1 = Tensor::randn(Shape::from_dims(&[in_dim, hid]), 0.0, 0.02, device.clone())?.requires_grad_(true);
    let b1 = Tensor::zeros(Shape::from_dims(&[1, hid]), device.clone())?.requires_grad_(true);
    let w2 = Tensor::randn(Shape::from_dims(&[hid, out]), 0.0, 0.02, device.clone())?.requires_grad_(true);
    let b2 = Tensor::zeros(Shape::from_dims(&[1, out]), device.clone())?.requires_grad_(true);

    // Baseline
    let y = mlp_forward(&x, &w1, &b1, &w2, &b2)?;
    let loss = y.square()?.mean()?;
    let grads0 = AutogradContext::backward(&loss)?;
    let gw1_0 = grads0.get(w1.id()).unwrap().clone_result()?;
    let gw2_0 = grads0.get(w2.id()).unwrap().clone_result()?;

    // Enable CPU offload policy
    {
        use flame_core::gradient_checkpointing::{CHECKPOINT_MANAGER, CheckpointPolicy};
        CHECKPOINT_MANAGER.lock().unwrap().set_policy(CheckpointPolicy::CPUOffload);
    }

    // Re-run
    let y1 = mlp_forward(&x, &w1, &b1, &w2, &b2)?;
    let loss1 = y1.square()?.mean()?;
    let grads1 = AutogradContext::backward(&loss1)?;
    let gw1_1 = grads1.get(w1.id()).unwrap().clone_result()?;
    let gw2_1 = grads1.get(w2.id()).unwrap().clone_result()?;

    // Compare gradients
    let a = gw1_0.to_vec()?; let b = gw1_1.to_vec()?;
    let max_diff_w1 = a.iter().zip(b.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
    assert!(max_diff_w1 < 1e-3, "w1 grad mismatch: {}", max_diff_w1);
    let a2 = gw2_0.to_vec()?; let b2v = gw2_1.to_vec()?;
    let max_diff_w2 = a2.iter().zip(b2v.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
    assert!(max_diff_w2 < 1e-3, "w2 grad mismatch: {}", max_diff_w2);

    Ok(())
}
