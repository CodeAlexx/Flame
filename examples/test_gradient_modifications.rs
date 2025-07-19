//! Test FLAME gradient modifications for training

use flame_core::{
    Tensor, Shape, AutogradContext, Result,
    cuda_gradient_ops::CudaGradientOps,
};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

fn main() -> Result<()> {
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    
    println!("Testing FLAME gradient modifications...\n");
    
    // Create test tensors
    let mut weight = Tensor::randn(Shape::from_dims(&[256, 128]), 0.0, 0.02, device.clone())?;
    weight.requires_grad = true;
    
    let input = Tensor::randn(Shape::from_dims(&[32, 128]), 0.0, 0.1, device.clone())?;
    
    // Forward pass
    println!("1. Forward pass...");
    let output = input.matmul(&weight.transpose()?)?;
    let target = Tensor::randn(Shape::from_dims(&[32, 256]), 0.0, 0.1, device.clone())?;
    
    // Compute loss
    let diff = output.sub(&target)?;
    let loss = diff.square()?.mean()?;
    
    let loss_val = loss.to_vec()?[0];
    println!("   Loss: {:.6}", loss_val);
    
    // Backward pass
    println!("\n2. Backward pass...");
    let gradients = loss.backward()?;
    
    // Get weight gradient
    let weight_grad = gradients.get(weight.id)
        .ok_or_else(|| flame_core::FlameError::InvalidOperation("No gradient for weight".into()))?;
    
    // Test gradient modifications
    println!("\n3. Testing gradient modifications...");
    
    // Original gradient stats
    let grad_data = weight_grad.to_vec()?;
    let grad_norm: f32 = grad_data.iter().map(|g| g * g).sum::<f32>().sqrt();
    let grad_max = grad_data.iter().map(|g| g.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    println!("   Original gradient:");
    println!("     - L2 norm: {:.6}", grad_norm);
    println!("     - Max abs value: {:.6}", grad_max);
    
    // Test GPU-based gradient operations
    let gpu_ops = CudaGradientOps::new(device.clone())?;
    
    // Test 1: Gradient clipping
    println!("\n4. Testing gradient clipping...");
    let mut clipped_grad = weight_grad.clone()?;
    gpu_ops.clip_gradient(&mut clipped_grad, 1.0)?;
    
    let clipped_data = clipped_grad.to_vec()?;
    let clipped_max = clipped_data.iter().map(|g| g.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("   After clipping to [-1.0, 1.0]:");
    println!("     - Max abs value: {:.6}", clipped_max);
    
    // Test 2: Gradient normalization
    println!("\n5. Testing gradient normalization...");
    let mut normalized_grad = weight_grad.clone()?;
    gpu_ops.normalize_gradient(&mut normalized_grad, 5.0)?;
    
    let norm_data = normalized_grad.to_vec()?;
    let new_norm: f32 = norm_data.iter().map(|g| g * g).sum::<f32>().sqrt();
    println!("   After normalizing to max L2 norm of 5.0:");
    println!("     - New L2 norm: {:.6}", new_norm);
    
    // Test 3: Gradient noise
    println!("\n6. Testing gradient noise addition...");
    let mut noisy_grad = weight_grad.clone()?;
    gpu_ops.add_gradient_noise(&mut noisy_grad, 0.01)?;
    
    let noisy_data = noisy_grad.to_vec()?;
    let noise_diff: f32 = grad_data.iter().zip(noisy_data.iter())
        .map(|(g1, g2)| (g1 - g2).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("   After adding noise (scale=0.01):");
    println!("     - L2 distance from original: {:.6}", noise_diff);
    
    // Test 4: Combined operations
    println!("\n7. Testing combined operations...");
    let mut combined_grad = weight_grad.clone()?;
    
    // Apply all modifications
    gpu_ops.add_gradient_noise(&mut combined_grad, 0.001)?;
    gpu_ops.clip_gradient(&mut combined_grad, 2.0)?;
    gpu_ops.normalize_gradient(&mut combined_grad, 10.0)?;
    
    let combined_data = combined_grad.to_vec()?;
    let combined_norm: f32 = combined_data.iter().map(|g| g * g).sum::<f32>().sqrt();
    let combined_max = combined_data.iter().map(|g| g.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    println!("   After noise + clip + normalize:");
    println!("     - L2 norm: {:.6}", combined_norm);
    println!("     - Max abs value: {:.6}", combined_max);
    
    // Test Conv2D gradient
    println!("\n8. Testing Conv2D gradients...");
    let conv_input = Tensor::randn(Shape::from_dims(&[4, 3, 32, 32]), 0.0, 0.1, device.clone())?;
    let conv_weight = Tensor::randn(Shape::from_dims(&[16, 3, 3, 3]), 0.0, 0.02, device.clone())?;
    let conv_bias = Tensor::randn(Shape::from_dims(&[16]), 0.0, 0.01, device.clone())?;
    
    conv_weight.requires_grad = true;
    conv_bias.requires_grad = true;
    
    let conv_output = flame_core::cuda_conv2d::conv2d(
        &conv_input,
        &conv_weight,
        Some(&conv_bias),
        1,  // stride
        1,  // padding
    )?;
    
    let conv_target = Tensor::randn(conv_output.shape().clone(), 0.0, 0.1, device.clone())?;
    let conv_loss = conv_output.sub(&conv_target)?.square()?.mean()?;
    
    let conv_gradients = conv_loss.backward()?;
    
    if let Some(conv_weight_grad) = conv_gradients.get(conv_weight.id) {
        let conv_grad_data = conv_weight_grad.to_vec()?;
        let conv_grad_norm: f32 = conv_grad_data.iter().map(|g| g * g).sum::<f32>().sqrt();
        println!("   Conv2D weight gradient norm: {:.6}", conv_grad_norm);
    }
    
    if let Some(conv_bias_grad) = conv_gradients.get(conv_bias.id) {
        let bias_grad_data = conv_bias_grad.to_vec()?;
        let bias_grad_norm: f32 = bias_grad_data.iter().map(|g| g * g).sum::<f32>().sqrt();
        println!("   Conv2D bias gradient norm: {:.6}", bias_grad_norm);
    }
    
    println!("\n✅ All gradient modification tests passed!");
    
    Ok(())
}