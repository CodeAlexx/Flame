//! Test LoRA gradient modifications in FLAME

use flame_core::{
    Tensor, Shape, Result, CudaDevice, AutogradContext,
};
use std::sync::Arc;

fn main() -> Result<()> {
    let device = CudaDevice::new(0)?;
    println!("🔥 FLAME LoRA Gradient Test\n");
    
    // LoRA parameters
    let rank = 4;
    let in_features = 16;
    let out_features = 16;
    let scale = 0.1;
    
    // Initialize LoRA matrices
    let lora_down = Tensor::randn(
        Shape::from_dims(&[in_features, rank]),
        0.02,
        0.0,
        device.clone(),
    )?.requires_grad_(true);
    
    let lora_up = Tensor::zeros(
        Shape::from_dims(&[rank, out_features]),
        device.clone(),
    )?.requires_grad_(true);
    
    println!("LoRA shapes:");
    println!("  down: {:?}", lora_down.shape().dims());
    println!("  up: {:?}", lora_up.shape().dims());
    
    // Create input and base weight
    let input = Tensor::randn(
        Shape::from_dims(&[8, in_features]),
        1.0,
        0.0,
        device.clone(),
    )?.requires_grad_(true);
    
    let base_weight = Tensor::randn(
        Shape::from_dims(&[in_features, out_features]),
        0.1,
        0.0,
        device.clone(),
    )?;
    
    // Forward pass with LoRA
    let base_output = input.matmul(&base_weight)?;
    let lora_hidden = input.matmul(&lora_down)?;
    let lora_output = lora_hidden.matmul(&lora_up)?;
    let scaled_lora = lora_output.mul_scalar(scale)?;
    let output = base_output.add(&scaled_lora)?;
    
    // Create target and compute loss
    let target = Tensor::randn(
        Shape::from_dims(&[8, out_features]),
        1.0,
        0.0,
        device.clone(),
    )?;
    
    let diff = output.sub(&target)?;
    let loss = diff.mul(&diff)?.mean()?;
    let loss_value = loss.to_vec()?[0];
    println!("\nLoss: {:.4}", loss_value);
    
    // Backward pass
    let mut gradients = AutogradContext::backward(&loss)?;
    println!("✅ Gradients computed!");
    
    // MODIFY GRADIENTS
    println!("\n=== Gradient Modifications ===");
    
    // Modify lora_down gradient
    if let Some(down_grad) = gradients.get(lora_down.id()) {
        let original = down_grad.to_vec()?;
        println!("Original down gradient norm: {:.4}", 
                 original.iter().map(|g| g * g).sum::<f32>().sqrt());
        
        // Clip gradients
        let clipped: Vec<f32> = original.iter()
            .map(|&g| g.clamp(-1.0, 1.0))
            .collect();
        let clipped_grad = Tensor::from_vec(
            clipped,
            down_grad.shape().clone(),
            device.clone(),
        )?;
        
        // Add noise
        let noise = Tensor::randn(
            down_grad.shape().clone(),
            0.001,
            0.0,
            device.clone(),
        )?;
        let noisy_grad = clipped_grad.add(&noise)?;
        
        println!("Modified down gradient norm: {:.4}", 
                 noisy_grad.to_vec()?.iter().map(|g| g * g).sum::<f32>().sqrt());
        
        // Update gradient map
        gradients.insert(lora_down.id(), noisy_grad);
    }
    
    // Modify lora_up gradient
    if let Some(up_grad) = gradients.get(lora_up.id()) {
        let original = up_grad.to_vec()?;
        let grad_norm = original.iter().map(|g| g * g).sum::<f32>().sqrt();
        println!("\nOriginal up gradient norm: {:.4}", grad_norm);
        
        // L2 normalize if norm > 1
        let normalized_grad = if grad_norm > 1.0 {
            up_grad.mul_scalar(1.0 / grad_norm)?
        } else {
            up_grad.clone()?
        };
        
        println!("Normalized up gradient norm: {:.4}", 
                 normalized_grad.to_vec()?.iter().map(|g| g * g).sum::<f32>().sqrt());
        
        // Update gradient map
        gradients.insert(lora_up.id(), normalized_grad);
    }
    
    println!("\n✅ SUCCESS! Gradients modified in FLAME!");
    println!("   - Gradient clipping applied");
    println!("   - Gradient noise added");
    println!("   - L2 normalization applied");
    println!("\nThis is impossible in Candle! 🔥");
    
    Ok(())
}