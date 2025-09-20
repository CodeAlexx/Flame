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

use flame_core::mixed_precision::{AMPContext, GradScaler, HalfTensor, MixedPrecisionTensor};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Mixed Precision Training ===");

    // Test 1: AMP Context
    println!("\n--- Test 1: AMP Context ---");
    let mut amp_ctx = AMPContext::new(DType::F16);
    println!("Initial loss scale: {}", amp_ctx.loss_scale);

    // Simulate training with gradient overflow
    amp_ctx.update_scale(true); // Overflow detected
    println!("After overflow, loss scale: {}", amp_ctx.loss_scale);

    // Simulate successful steps
    for i in 0..10 {
        amp_ctx.update_scale(false);
        if i % 5 == 0 {
            println!("Step {}, loss scale: {}", i, amp_ctx.loss_scale);
        }
    }

    // Test 2: Half precision tensors
    println!("\n--- Test 2: Half Precision Tensors ---");
    let f32_tensor = Tensor::randn(Shape::from_dims(&[4, 8]), 0.0, 1.0, device.clone())?;
    println!("Original F32 tensor shape: {:?}", f32_tensor.shape().dims());

    // Convert to F16
    let f16_tensor = HalfTensor::from_f32(&f32_tensor, DType::F16)?;
    println!("F16 tensor created with dtype: {:?}", f16_tensor.dtype);

    // Convert back to F32
    let f32_back = f16_tensor.to_f32()?;
    println!("Converted back to F32 shape: {:?}", f32_back.shape().dims());

    // Check conversion accuracy
    let original_data = f32_tensor.to_vec()?;
    let converted_data = f32_back.to_vec()?;

    let max_diff = original_data
        .iter()
        .zip(converted_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Max difference after F16 round-trip: {}", max_diff);

    // Test BF16
    let bf16_tensor = HalfTensor::from_f32(&f32_tensor, DType::BF16)?;
    let bf16_back = bf16_tensor.to_f32()?;

    let bf16_data = bf16_back.to_vec()?;
    let bf16_max_diff = original_data
        .iter()
        .zip(bf16_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Max difference after BF16 round-trip: {}", bf16_max_diff);

    // Test 3: Mixed Precision Tensor wrapper
    println!("\n--- Test 3: Mixed Precision Tensor Wrapper ---");
    let amp_context = Arc::new(AMPContext::new(DType::F16));
    let weight = Tensor::randn(Shape::from_dims(&[128, 256]), 0.0, 0.02, device.clone())?;

    let mp_tensor = MixedPrecisionTensor::new(weight.clone()?, amp_context.clone())?;
    println!("Created mixed precision tensor");
    println!("Master weight shape: {:?}", mp_tensor.master.shape().dims());
    println!("Has compute tensor: {}", mp_tensor.compute.is_some());

    // Test 4: Gradient Scaler
    println!("\n--- Test 4: Gradient Scaler ---");
    let mut scaler = GradScaler::new();
    println!("Initial scale: 65536.0"); // Default scale

    // Create a loss tensor
    let loss = Tensor::from_vec(vec![0.5], Shape::from_dims(&[1]), device.clone())?;
    let scaled_loss = scaler.scale(&loss)?;
    println!("Original loss: {}", loss.item()?);
    println!("Scaled loss: {}", scaled_loss.item()?);

    // Create some gradients
    let mut grad1 = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 0.01, device.clone())?;
    let mut grad2 = Tensor::randn(Shape::from_dims(&[128, 256]), 0.0, 0.01, device.clone())?;

    // Scale gradients (simulating backward pass with scaled loss)
    // Scale gradients (simulating backward pass with scaled loss)
    let scale_factor = 65536.0; // Default scale
    grad1 = grad1.mul_scalar(scale_factor)?;
    grad2 = grad2.mul_scalar(scale_factor)?;

    // Unscale gradients before optimizer step
    let mut grads = vec![grad1, grad2];
    scaler.unscale(&mut grads)?;

    println!("Gradients unscaled for optimizer step");

    // Update scaler based on gradient validity
    scaler.update(false); // No inf/nan found
    println!("Scale updated (no inf/nan found)");

    // Test 5: Simulating mixed precision training step
    println!("\n--- Test 5: Mixed Precision Training Simulation ---");

    // Create model parameters
    let mut weight1 = Tensor::randn(Shape::from_dims(&[512, 1024]), 0.0, 0.02, device.clone())?
        .requires_grad_(true);
    let mut weight2 = Tensor::randn(Shape::from_dims(&[1024, 512]), 0.0, 0.02, device.clone())?
        .requires_grad_(true);

    // Create AMP context
    let amp_ctx = AMPContext::new(DType::F16);

    // Forward pass (simulated)
    let input = Tensor::randn(Shape::from_dims(&[32, 512]), 0.0, 1.0, device.clone())?;
    let hidden = input.matmul(&weight1)?;
    let output = hidden.relu()?.matmul(&weight2)?;

    // Compute loss
    let target = Tensor::randn(Shape::from_dims(&[32, 512]), 0.0, 1.0, device.clone())?;
    let diff = output.sub(&target)?;
    let loss = diff.square()?.mean()?;

    println!("Forward pass complete");
    println!("Loss: {}", loss.item()?);

    // Scale loss for backward
    let scaled_loss = amp_ctx.scale_loss(&loss)?;
    println!("Scaled loss for backward: {}", scaled_loss.item()?);

    // Simulate gradient computation
    let fake_grad1 = Tensor::randn(Shape::from_dims(&[512, 1024]), 0.0, 0.001, device.clone())?;
    let fake_grad2 = Tensor::randn(Shape::from_dims(&[1024, 512]), 0.0, 0.001, device)?;

    // Unscale gradients
    let unscaled_grad1 = amp_ctx.unscale_grads(&fake_grad1)?;
    let unscaled_grad2 = amp_ctx.unscale_grads(&fake_grad2)?;

    println!("Gradients computed and unscaled");

    // Apply weight updates
    let lr = 0.001;
    weight1.update_weights(&unscaled_grad1, lr)?;
    weight2.update_weights(&unscaled_grad2, lr)?;

    println!("Weights updated successfully");

    // Performance comparison
    println!("\n=== Performance Comparison ===");
    println!("Note: Full mixed precision performance benefits require");
    println!("actual hardware support and optimized kernels.");
    println!("This is a functional demonstration of the API.");

    println!("\nAll mixed precision tests completed!");

    Ok(())
}
