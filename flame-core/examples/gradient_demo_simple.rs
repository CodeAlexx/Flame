#![cfg(feature = "legacy_examples")]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]
#![cfg_attr(
    clippy,
    allow(
        clippy::unused_imports,
        clippy::useless_vec,
        clippy::needless_borrow,
        clippy::needless_clone
    )
)]

//! Simple demonstration of FLAME gradient modifications

use flame_core::{AutogradContext, CudaDevice, Result, Shape, Tensor};

fn main() -> Result<()> {
    println!("🔥 FLAME Gradient Modification Demo\n");

    let device = CudaDevice::new(0)?;

    // Simple linear layer: y = x @ w
    let x = Tensor::randn(Shape::from_dims(&[4, 8]), 1.0, 0.0, device.clone())?;
    let w =
        Tensor::randn(Shape::from_dims(&[8, 4]), 0.1, 0.0, device.clone())?.requires_grad_(true);

    // Forward
    let y = x.matmul(&w)?;
    let loss = y.sum()?;
    println!("Loss: {:.4}", loss.to_vec()?[0]);

    // Backward
    let mut gradients = AutogradContext::backward(&loss)?;

    // Get original gradient
    if let Some(w_grad) = gradients.get(w.id()) {
        let grad_vec = w_grad.to_vec()?;
        let orig_norm = grad_vec.iter().map(|g| g * g).sum::<f32>().sqrt();
        println!("\nOriginal gradient norm: {:.4}", orig_norm);

        // 1. Gradient Clipping by Value
        println!("\n1. Gradient Clipping by Value [-0.5, 0.5]");
        let clipped: Vec<f32> = grad_vec.iter().map(|&g| g.clamp(-0.5, 0.5)).collect();
        let clipped_tensor = Tensor::from_vec(clipped, w_grad.shape().clone(), device.clone())?;
        let clipped_norm = clipped_tensor
            .to_vec()?
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();
        println!("   Clipped gradient norm: {:.4}", clipped_norm);

        // 2. L2 Normalization
        println!("\n2. L2 Gradient Normalization (max_norm=1.0)");
        let normalized = if orig_norm > 1.0 {
            w_grad.mul_scalar(1.0 / orig_norm)?
        } else {
            w_grad.clone()?
        };
        let normalized_norm = normalized
            .to_vec()?
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();
        println!("   Normalized gradient norm: {:.4}", normalized_norm);

        // 3. Add Gradient Noise
        println!("\n3. Gradient Noise (std=0.01)");
        let noise = Tensor::randn(w_grad.shape().clone(), 0.01, 0.0, device.clone())?;
        let noisy = w_grad.add(&noise)?;
        let noisy_norm = noisy.to_vec()?.iter().map(|g| g * g).sum::<f32>().sqrt();
        println!("   Noisy gradient norm: {:.4}", noisy_norm);

        // 4. Gradient Scaling
        println!("\n4. Gradient Scaling (×0.1)");
        let scaled = w_grad.mul_scalar(0.1)?;
        let scaled_norm = scaled.to_vec()?.iter().map(|g| g * g).sum::<f32>().sqrt();
        println!("   Scaled gradient norm: {:.4}", scaled_norm);
    }

    println!("\n✅ All gradient modifications successful!");
    println!("These operations are IMPOSSIBLE in Candle! 🔥");

    Ok(())
}
