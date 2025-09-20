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

//! Comprehensive demonstration of FLAME's gradient modification capabilities
//! This shows all the gradient modifications that Candle cannot do

use flame_core::{
    gradient_clip::{GradientClipStrategy, GradientClipper},
    AutogradContext, CudaDevice, Result, Shape, Tensor,
};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🔥 FLAME Gradient Modification Demo\n");
    println!("This demonstrates gradient modifications that are IMPOSSIBLE in Candle!\n");

    let device = CudaDevice::new(0)?;

    // Create a simple neural network simulation
    let batch_size = 8;
    let input_dim = 16;
    let hidden_dim = 32;
    let output_dim = 10;

    // Layer 1: input -> hidden
    let w1 = Tensor::randn(
        Shape::from_dims(&[input_dim, hidden_dim]),
        0.02,
        0.0,
        device.clone(),
    )?
    .requires_grad_(true);

    let b1 = Tensor::zeros(Shape::from_dims(&[hidden_dim]), device.clone())?.requires_grad_(true);

    // Layer 2: hidden -> output
    let w2 = Tensor::randn(
        Shape::from_dims(&[hidden_dim, output_dim]),
        0.02,
        0.0,
        device.clone(),
    )?
    .requires_grad_(true);

    let b2 = Tensor::zeros(Shape::from_dims(&[output_dim]), device.clone())?.requires_grad_(true);

    // Create some dummy data
    let x = Tensor::randn(
        Shape::from_dims(&[batch_size, input_dim]),
        1.0,
        0.0,
        device.clone(),
    )?;

    let target = Tensor::randn(
        Shape::from_dims(&[batch_size, output_dim]),
        1.0,
        0.0,
        device.clone(),
    )?;

    // Forward pass
    println!("=== Forward Pass ===");
    let h1 = x.matmul(&w1)?;
    let h1_bias = h1.add(&b1)?;
    let h1_relu = h1_bias.relu()?;

    let out = h1_relu.matmul(&w2)?;
    let out_bias = out.add(&b2)?;

    // Loss (MSE)
    let diff = out_bias.sub(&target)?;
    let loss = diff.mul(&diff)?.mean()?;
    let loss_val = loss.to_vec()?[0];
    println!("Loss: {:.4}", loss_val);

    // Backward pass
    println!("\n=== Backward Pass ===");
    let mut gradients = AutogradContext::backward(&loss)?;
    println!("✅ Gradients computed");

    // Collect original gradients
    let mut original_grads = HashMap::new();
    for (name, tensor_id) in [
        ("w1", w1.id()),
        ("b1", b1.id()),
        ("w2", w2.id()),
        ("b2", b2.id()),
    ] {
        if let Some(grad) = gradients.get(tensor_id) {
            let grad_norm = grad.to_vec()?.iter().map(|g| g * g).sum::<f32>().sqrt();
            original_grads.insert(name, grad_norm);
            println!("{} gradient norm: {:.4}", name, grad_norm);
        }
    }

    println!("\n=== Gradient Modifications ===");

    // 1. Gradient Clipping by Value
    println!("\n1. Gradient Clipping by Value (clip to [-0.5, 0.5])");
    if let Some(w1_grad) = gradients.get(w1.id()) {
        let grad_vec = w1_grad.to_vec()?;
        let clipped: Vec<f32> = grad_vec.iter().map(|&g| g.clamp(-0.5, 0.5)).collect();

        let clipped_grad = Tensor::from_vec(clipped, w1_grad.shape().clone(), device.clone())?;
        let new_norm = clipped_grad
            .to_vec()?
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();

        println!(
            "  w1 gradient norm: {:.4} -> {:.4}",
            original_grads["w1"], new_norm
        );
        gradients.insert(w1.id(), clipped_grad);
    }

    // 2. L2 Gradient Normalization (max norm = 1.0)
    println!("\n2. L2 Gradient Normalization (max norm = 1.0)");
    if let Some(w2_grad) = gradients.get(w2.id()) {
        let grad_vec = w2_grad.to_vec()?;
        let grad_norm = grad_vec.iter().map(|g| g * g).sum::<f32>().sqrt();

        if grad_norm > 1.0 {
            let normalized_grad = w2_grad.mul_scalar(1.0 / grad_norm)?;
            let new_norm = normalized_grad
                .to_vec()?
                .iter()
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt();

            println!(
                "  w2 gradient norm: {:.4} -> {:.4}",
                original_grads["w2"], new_norm
            );
            gradients.insert(w2.id(), normalized_grad);
        }
    }

    // 3. Gradient Noise for Regularization
    println!("\n3. Adding Gradient Noise (std=0.01)");
    if let Some(b1_grad) = gradients.get(b1.id()) {
        let noise = Tensor::randn(b1_grad.shape().clone(), 0.01, 0.0, device.clone())?;
        let noisy_grad = b1_grad.add(&noise)?;

        let orig_norm = original_grads["b1"];
        let new_norm = noisy_grad
            .to_vec()?
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();

        println!(
            "  b1 gradient norm: {:.4} -> {:.4} (with noise)",
            orig_norm, new_norm
        );
        gradients.insert(b1.id(), noisy_grad);
    }

    // 4. Gradient Scaling
    println!("\n4. Gradient Scaling (scale by 0.1)");
    if let Some(b2_grad) = gradients.get(b2.id()) {
        let scaled_grad = b2_grad.mul_scalar(0.1)?;
        let new_norm = scaled_grad
            .to_vec()?
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();

        println!(
            "  b2 gradient norm: {:.4} -> {:.4}",
            original_grads["b2"], new_norm
        );
        gradients.insert(b2.id(), scaled_grad);
    }

    // 5. Using Built-in Gradient Clipping by Norm
    println!("\n5. Using FLAME's Built-in Gradient Clipping");
    let norm_clipper = GradientClipper::clip_by_norm(5.0);

    // Collect gradients as mutable references
    let mut w1_grad_clone = gradients.get(w1.id()).unwrap().clone()?;
    let mut w2_grad_clone = gradients.get(w2.id()).unwrap().clone()?;
    let mut grad_refs = vec![&mut w1_grad_clone, &mut w2_grad_clone];

    let total_norm = norm_clipper.clip_grads(&mut grad_refs)?;

    println!("  Applied gradient clipping by norm:");
    println!("    - Max norm: 5.0");
    println!("    - Original total norm: {:.4}", total_norm);

    // 6. Adaptive Gradient Clipping
    println!("\n6. Adaptive Gradient Clipping");
    let adaptive_clipper = GradientClipper::adaptive(0.1);

    // Apply adaptive clipping
    let mut b1_grad_clone = gradients.get(b1.id()).unwrap().clone()?;
    let mut b2_grad_clone = gradients.get(b2.id()).unwrap().clone()?;
    let mut bias_grads = vec![&mut b1_grad_clone, &mut b2_grad_clone];

    adaptive_clipper.clip_grads(&mut bias_grads)?;

    println!("  Applied adaptive gradient clipping");
    println!("  This adjusts clipping based on gradient statistics");

    println!("\n=== Summary ===");
    println!("FLAME successfully demonstrated:");
    println!("  1. ✅ Gradient clipping by value");
    println!("  2. ✅ L2 gradient normalization");
    println!("  3. ✅ Gradient noise injection");
    println!("  4. ✅ Gradient scaling");
    println!("  5. ✅ Global gradient clipping");
    println!("  6. ✅ Adaptive gradient clipping");
    println!("\nALL of these are IMPOSSIBLE in Candle due to immutable gradients!");
    println!("This is why FLAME exists! 🔥");

    Ok(())
}
