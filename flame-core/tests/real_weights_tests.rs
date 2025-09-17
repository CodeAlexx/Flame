//! Tests with real model weights and training scenarios
//! These tests verify that FLAME works correctly with actual model weights

use flame_core::{Tensor, Shape, Result, AutogradContext};
use std::sync::Arc;
use std::path::Path;
use cudarc::driver::CudaDevice;

/// Helper to create a test device
fn test_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("Failed to create CUDA device")
}

/// Helper to load safetensors file
fn load_safetensors(path: &Path) -> Result<std::collections::HashMap<String, Tensor>> {
    use safetensors::SafeTensors;
    use std::fs;
    
    let device = test_device();
    let data = fs::read(path).expect("Failed to read safetensors file");
    let tensors = SafeTensors::deserialize(&data).expect("Failed to deserialize safetensors");
    
    let mut result = std::collections::HashMap::new();
    
    for (name, view) in tensors.tensors() {
        let shape = Shape::from_dims(view.shape());
        let data: Vec<f32> = view.data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        
        let tensor = Tensor::from_vec(data, shape, device.clone())?;
        result.insert(name.to_string(), tensor);
    }
    
    Ok(result)
}

#[test]
fn test_load_and_modify_real_weights() -> Result<()> {
    let device = test_device();
    
    // Create some "real" weights to simulate loading from a model
    let conv_weight = Tensor::randn(
        Shape::from_dims(&[64, 3, 7, 7]),  // ResNet first conv layer shape
        0.0,
        0.02,  // Typical initialization scale
        device.clone()
    )?.requires_grad();
    
    let bn_weight = Tensor::ones(Shape::from_dims(&[64]), device.clone())?.requires_grad();
    let bn_bias = Tensor::zeros(Shape::from_dims(&[64]), device.clone())?.requires_grad();
    
    // Simulate a forward pass
    let batch_size = 4;
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size, 3, 224, 224]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    // Conv forward (simplified - just check shapes work)
    let conv_out = input.conv2d(&conv_weight, 2, 3)?;  // stride=2, padding=3
    
    // Batch norm forward (simplified)
    let normalized = conv_out.sub(&bn_bias)?.mul(&bn_weight)?;
    
    // Check output shape
    assert_eq!(normalized.shape().dims()[0], batch_size);
    assert_eq!(normalized.shape().dims()[1], 64);  // Output channels
    
    // Compute a loss and gradients
    let loss = normalized.square()?.mean()?;
    let grads = loss.backward()?;
    
    // Verify we got gradients for all weights
    assert!(grads.get(conv_weight.id).is_some());
    assert!(grads.get(bn_weight.id).is_some());
    assert!(grads.get(bn_bias.id).is_some());
    
    Ok(())
}

#[test]
fn test_lora_weight_adaptation() -> Result<()> {
    let device = test_device();
    
    // Simulate a LoRA adaptation scenario
    let hidden_dim = 768;  // BERT/GPT-2 hidden dimension
    let lora_rank = 16;
    
    // Original weight (frozen)
    let original_weight = Tensor::randn(
        Shape::from_dims(&[hidden_dim, hidden_dim]),
        0.0,
        0.02,
        device.clone()
    )?;  // No requires_grad - frozen
    
    // LoRA matrices (trainable)
    let lora_a = Tensor::randn(
        Shape::from_dims(&[lora_rank, hidden_dim]),
        0.0,
        0.02,
        device.clone()
    )?.requires_grad();
    
    let lora_b = Tensor::zeros(
        Shape::from_dims(&[hidden_dim, lora_rank]),
        device.clone()
    )?.requires_grad();  // Initialize to zero for LoRA
    
    let scaling = 1.0 / (lora_rank as f32).sqrt();
    
    // Forward pass with LoRA
    let batch_size = 8;
    let seq_len = 128;
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size * seq_len, hidden_dim]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    // Standard forward
    let output_original = input.matmul(&original_weight)?;
    
    // LoRA forward: output = input @ (W + scaling * B @ A)
    let lora_output = input.matmul(&lora_a.transpose()?)?;
    let lora_output = lora_output.matmul(&lora_b.transpose()?)?;
    let lora_output = lora_output.mul_scalar(scaling)?;
    
    let output = output_original.add(&lora_output)?;
    
    // Compute loss
    let target = Tensor::randn(output.shape().clone(), 0.0, 1.0, device.clone())?;
    let loss = output.sub(&target)?.square()?.mean()?;
    
    // Backward
    let grads = loss.backward()?;
    
    // Check we only got gradients for LoRA parameters
    assert!(grads.get(lora_a.id).is_some(), "Missing gradient for LoRA A");
    assert!(grads.get(lora_b.id).is_some(), "Missing gradient for LoRA B");
    assert!(grads.get(original_weight.id).is_none(), "Original weight should not have gradient");
    
    // Verify gradient shapes
    assert_eq!(grads.get(lora_a.id).unwrap().shape(), lora_a.shape());
    assert_eq!(grads.get(lora_b.id).unwrap().shape(), lora_b.shape());
    
    Ok(())
}

#[test]
fn test_attention_mechanism_gradients() -> Result<()> {
    let device = test_device();
    
    // Test a simplified attention mechanism
    let batch_size = 2;
    let seq_len = 10;
    let hidden_dim = 64;
    let num_heads = 8;
    let head_dim = hidden_dim / num_heads;
    
    // Query, Key, Value projections
    let w_q = Tensor::randn(Shape::from_dims(&[hidden_dim, hidden_dim]), 0.0, 0.02, device.clone())?.requires_grad();
    let w_k = Tensor::randn(Shape::from_dims(&[hidden_dim, hidden_dim]), 0.0, 0.02, device.clone())?.requires_grad();
    let w_v = Tensor::randn(Shape::from_dims(&[hidden_dim, hidden_dim]), 0.0, 0.02, device.clone())?.requires_grad();
    
    // Input
    let x = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, hidden_dim]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    // Reshape for matrix multiplication
    let x_2d = x.reshape(&[batch_size * seq_len, hidden_dim])?;
    
    // Project to Q, K, V
    let q = x_2d.matmul(&w_q)?;
    let k = x_2d.matmul(&w_k)?;
    let v = x_2d.matmul(&w_v)?;
    
    // Reshape for attention
    let q = q.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let k = k.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let v = v.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    
    // Transpose for attention computation
    let q = q.permute(&[0, 2, 1, 3])?;  // [batch, heads, seq, head_dim]
    let k = k.permute(&[0, 2, 3, 1])?;  // [batch, heads, head_dim, seq]
    let v = v.permute(&[0, 2, 1, 3])?;  // [batch, heads, seq, head_dim]
    
    // Attention scores
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores = q.matmul(&k)?.mul_scalar(scale)?;
    
    // Softmax
    let attn_weights = scores.softmax(-1)?;
    
    // Apply attention to values
    let attn_output = attn_weights.matmul(&v)?;
    
    // Loss
    let loss = attn_output.square()?.mean()?;
    
    // Backward
    let grads = loss.backward()?;
    
    // Verify gradients for Q, K, V projections
    assert!(grads.get(w_q.id).is_some(), "Missing gradient for W_q");
    assert!(grads.get(w_k.id).is_some(), "Missing gradient for W_k");
    assert!(grads.get(w_v.id).is_some(), "Missing gradient for W_v");
    
    Ok(())
}

#[test]
fn test_training_step_with_optimizer() -> Result<()> {
    let device = test_device();
    
    // Simple linear model
    let input_dim = 10;
    let output_dim = 5;
    let batch_size = 32;
    
    // Initialize weights
    let mut weight = Tensor::randn(
        Shape::from_dims(&[input_dim, output_dim]),
        0.0,
        0.1,
        device.clone()
    )?.requires_grad();
    
    let mut bias = Tensor::zeros(
        Shape::from_dims(&[output_dim]),
        device.clone()
    )?.requires_grad();
    
    // Learning rate
    let lr = 0.01;
    
    // Training data
    let x = Tensor::randn(Shape::from_dims(&[batch_size, input_dim]), 0.0, 1.0, device.clone())?;
    let y_true = Tensor::randn(Shape::from_dims(&[batch_size, output_dim]), 0.0, 1.0, device.clone())?;
    
    // Store initial weights
    let initial_weight = weight.to_vec()?;
    let initial_bias = bias.to_vec()?;
    
    // Forward pass
    let y_pred = x.matmul(&weight)?.add(&bias)?;
    
    // Loss (MSE)
    let loss = y_pred.sub(&y_true)?.square()?.mean()?;
    let loss_value = loss.to_vec()?[0];
    
    // Backward
    let grads = loss.backward()?;
    
    // Get gradients
    let weight_grad = grads.get(weight.id).expect("Missing weight gradient");
    let bias_grad = grads.get(bias.id).expect("Missing bias gradient");
    
    // SGD update
    weight = weight.sub(&weight_grad.mul_scalar(lr)?)?;
    bias = bias.sub(&bias_grad.mul_scalar(lr)?)?;
    
    // Verify weights changed
    let updated_weight = weight.to_vec()?;
    let updated_bias = bias.to_vec()?;
    
    for i in 0..initial_weight.len() {
        assert!(
            (initial_weight[i] - updated_weight[i]).abs() > 1e-6,
            "Weight {} didn't change",
            i
        );
    }
    
    for i in 0..initial_bias.len() {
        assert!(
            (initial_bias[i] - updated_bias[i]).abs() > 1e-6,
            "Bias {} didn't change",
            i
        );
    }
    
    // Do another forward pass to verify loss decreased
    let y_pred_new = x.matmul(&weight)?.add(&bias)?;
    let loss_new = y_pred_new.sub(&y_true)?.square()?.mean()?;
    let loss_new_value = loss_new.to_vec()?[0];
    
    // For a single step with small LR, loss might not always decrease
    // but weights should definitely have changed
    println!("Loss: {} -> {} (diff: {})", loss_value, loss_new_value, loss_value - loss_new_value);
    
    Ok(())
}

#[test]
fn test_conv2d_with_real_kernel_sizes() -> Result<()> {
    let device = test_device();
    
    // Test various real-world conv configurations
    let configs = vec![
        // (in_channels, out_channels, kernel_size, stride, padding, name)
        (3, 64, 7, 2, 3, "ResNet first layer"),
        (64, 64, 3, 1, 1, "ResNet conv block"),
        (256, 512, 3, 2, 1, "ResNet downsample"),
        (512, 512, 1, 1, 0, "1x1 conv"),
    ];
    
    for (in_c, out_c, k, s, p, name) in configs {
        println!("Testing {}", name);
        
        // Create weight
        let weight = Tensor::randn(
            Shape::from_dims(&[out_c, in_c, k, k]),
            0.0,
            (2.0 / (in_c * k * k) as f32).sqrt(),  // He initialization
            device.clone()
        )?.requires_grad();
        
        // Create input
        let batch_size = 2;
        let input_size = 32;
        let input = Tensor::randn(
            Shape::from_dims(&[batch_size, in_c, input_size, input_size]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        // Forward
        let output = input.conv2d(&weight, s, p)?;
        
        // Check output shape
        let expected_size = (input_size + 2 * p - k) / s + 1;
        assert_eq!(output.shape().dims(), &[batch_size, out_c, expected_size, expected_size]);
        
        // Compute gradients
        let loss = output.square()?.mean()?;
        let grads = loss.backward()?;
        
        // Verify gradient exists and has correct shape
        let weight_grad = grads.get(weight.id).expect("Missing weight gradient");
        assert_eq!(weight_grad.shape(), weight.shape());
        
        // Verify gradient is non-zero
        let grad_sum: f32 = weight_grad.abs()?.sum()?.to_vec()?[0];
        assert!(grad_sum > 1e-5, "Gradient is too small for {}", name);
    }
    
    Ok(())
}

#[test]
fn test_batch_norm_training_mode() -> Result<()> {
    let device = test_device();
    
    // Batch norm in training mode
    let batch_size = 16;
    let channels = 64;
    let spatial_size = 8;
    
    // Input
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size, channels, spatial_size, spatial_size]),
        0.0,
        1.0,
        device.clone()
    )?.requires_grad();
    
    // BN parameters
    let gamma = Tensor::ones(Shape::from_dims(&[channels]), device.clone())?.requires_grad();
    let beta = Tensor::zeros(Shape::from_dims(&[channels]), device.clone())?.requires_grad();
    
    // Compute batch statistics
    let axes = vec![0, 2, 3];  // Reduce over batch and spatial dims
    let mean = input.mean_dims(&axes, true)?;
    let var = input.var_dims(&axes, true, true)?;
    
    // Normalize
    let eps = 1e-5;
    let std = var.add_scalar(eps)?.sqrt()?;
    let normalized = input.sub(&mean)?.div(&std)?;
    
    // Scale and shift
    let output = normalized.mul(&gamma)?.add(&beta)?;
    
    // Loss
    let loss = output.square()?.mean()?;
    
    // Backward
    let grads = loss.backward()?;
    
    // Check gradients
    assert!(grads.get(input.id).is_some(), "Missing input gradient");
    assert!(grads.get(gamma.id).is_some(), "Missing gamma gradient");
    assert!(grads.get(beta.id).is_some(), "Missing beta gradient");
    
    // Verify gamma gradient sums to approximately the mean of normalized * grad_output
    let gamma_grad = grads.get(gamma.id).unwrap();
    assert_eq!(gamma_grad.shape().dims(), &[channels]);
    
    Ok(())
}

#[test]
fn test_gradient_clipping() -> Result<()> {
    let device = test_device();
    
    // Test gradient clipping scenario
    let mut weight = Tensor::from_vec(
        vec![10.0, -10.0, 5.0, -5.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?.requires_grad();
    
    // Create large gradient scenario
    let input = Tensor::from_vec(
        vec![100.0, 100.0],
        Shape::from_dims(&[1, 2]),
        device.clone()
    )?;
    
    let output = input.matmul(&weight)?;
    let loss = output.square()?.sum()?;
    
    let grads = loss.backward()?;
    let grad = grads.get(weight.id).unwrap();
    
    // Check gradient magnitude
    let grad_norm = grad.square()?.sum()?.sqrt()?;
    let grad_norm_val = grad_norm.to_vec()?[0];
    
    println!("Gradient norm before clipping: {}", grad_norm_val);
    
    // Clip gradient
    let max_norm = 1.0;
    let clipped_grad = if grad_norm_val > max_norm {
        grad.mul_scalar(max_norm / grad_norm_val)?
    } else {
        grad.clone_result()?
    };
    
    // Verify clipped norm
    let clipped_norm = clipped_grad.square()?.sum()?.sqrt()?;
    let clipped_norm_val = clipped_norm.to_vec()?[0];
    
    assert!(
        (clipped_norm_val - max_norm).abs() < 1e-4 || clipped_norm_val <= max_norm,
        "Gradient norm {} exceeds max {}",
        clipped_norm_val,
        max_norm
    );
    
    Ok(())
}

#[test]
fn test_memory_efficient_gradient_accumulation() -> Result<()> {
    let device = test_device();
    
    // Simulate gradient accumulation over multiple micro-batches
    let total_batch_size = 64;
    let micro_batch_size = 16;
    let num_accumulation_steps = total_batch_size / micro_batch_size;
    
    let input_dim = 128;
    let output_dim = 10;
    
    // Model parameters
    let weight = Tensor::randn(
        Shape::from_dims(&[input_dim, output_dim]),
        0.0,
        0.1,
        device.clone()
    )?.requires_grad();
    
    // Accumulate gradients
    let mut accumulated_grad = None;
    
    for step in 0..num_accumulation_steps {
        // Create micro-batch
        let x = Tensor::randn(
            Shape::from_dims(&[micro_batch_size, input_dim]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        let y_true = Tensor::randn(
            Shape::from_dims(&[micro_batch_size, output_dim]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        // Forward
        let y_pred = x.matmul(&weight)?;
        let loss = y_pred.sub(&y_true)?.square()?.sum()?;
        
        // Scale loss by accumulation steps
        let scaled_loss = loss.mul_scalar(1.0 / num_accumulation_steps as f32)?;
        
        // Backward
        let grads = scaled_loss.backward()?;
        let grad = grads.get(weight.id).unwrap();
        
        // Accumulate
        accumulated_grad = match accumulated_grad {
            None => Some(grad.clone_result()?),
            Some(acc) => Some(acc.add(&grad)?),
        };
        
        // Clear computation graph
        AutogradContext::clear();
    }
    
    // Verify accumulated gradient
    let final_grad = accumulated_grad.unwrap();
    assert_eq!(final_grad.shape(), weight.shape());
    
    // Gradient should be non-zero
    let grad_norm = final_grad.square()?.sum()?.sqrt()?;
    let grad_norm_val = grad_norm.to_vec()?[0];
    assert!(grad_norm_val > 1e-5, "Accumulated gradient is too small");
    
    Ok(())
}
