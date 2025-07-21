use flame_core::{
    Tensor, Shape, Result, 
    linear::Linear,
    conv::Conv2d,
    optimizers::{Adam, SGD},
    loss::{mse_loss, cross_entropy_loss},
    GradientMap,
};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// Test a realistic CNN training scenario
#[test]
fn test_realistic_cnn_training() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Build a small CNN architecture
    let conv1 = Conv2d::new_with_bias(3, 32, 3, 1, 1, device.clone(), true)?;
    let conv2 = Conv2d::new_with_bias(32, 64, 3, 1, 1, device.clone(), true)?;
    let fc1 = Linear::new(64 * 8 * 8, 128, true, device.clone())?;
    let fc2 = Linear::new(128, 10, true, device.clone())?;
    
    // Create optimizer
    let mut adam = Adam::new(AdamConfig {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    });
    
    // Training loop
    let batch_size = 4;
    let num_iterations = 10;
    let mut losses = Vec::new();
    
    for iter in 0..num_iterations {
        // Create dummy input (batch_size, channels, height, width)
        let input = Tensor::randn(
            Shape::from_dims(&[batch_size, 3, 32, 32]),
            0.0,
            0.1,
            device.clone()
        )?.requires_grad();
        
        // Create dummy labels
        let labels = vec![
            (iter % 10) as i64,
            ((iter + 1) % 10) as i64,
            ((iter + 2) % 10) as i64,
            ((iter + 3) % 10) as i64,
        ];
        let target = Tensor::from_vec(
            labels.iter().map(|&x| x as f32).collect(),
            Shape::from_dims(&[batch_size]),
            device.clone()
        )?;
        
        // Forward pass
        let h1 = conv1.forward(&input)?;
        let h1_relu = h1.relu()?;
        let h1_pool = pool2d_max(&h1_relu, 2, 2)?; // 32x32 -> 16x16
        
        let h2 = conv2.forward(&h1_pool)?;
        let h2_relu = h2.relu()?;
        let h2_pool = pool2d_max(&h2_relu, 2, 2)?; // 16x16 -> 8x8
        
        // Flatten for FC layers
        let h2_flat = h2_pool.view(&[batch_size, -1])?;
        
        let h3 = fc1.forward(&h2_flat)?;
        let h3_relu = h3.relu()?;
        
        let logits = fc2.forward(&h3_relu)?;
        
        // Compute loss
        let loss = cross_entropy_loss(&logits, &target)?;
        let loss_val = loss.item()?;
        losses.push(loss_val);
        
        println!("Iteration {}: loss = {:.4}", iter, loss_val);
        
        // Backward pass
        let grad_map = loss.backward()?;
        
        // Collect all parameters and their gradients
        let mut params_and_grads = vec![];
        
        // Conv1 parameters
        if let Some(grad) = grad_map.get(&conv1.weight.id) {
            params_and_grads.push((&conv1.weight, grad));
        }
        if let Some(bias) = &conv1.bias {
            if let Some(grad) = grad_map.get(&bias.id) {
                params_and_grads.push((bias, grad));
            }
        }
        
        // Conv2 parameters
        if let Some(grad) = grad_map.get(&conv2.weight.id) {
            params_and_grads.push((&conv2.weight, grad));
        }
        if let Some(bias) = &conv2.bias {
            if let Some(grad) = grad_map.get(&bias.id) {
                params_and_grads.push((bias, grad));
            }
        }
        
        // FC1 parameters
        if let Some(grad) = grad_map.get(&fc1.weight.id) {
            params_and_grads.push((&fc1.weight, grad));
        }
        if let Some(bias) = &fc1.bias {
            if let Some(grad) = grad_map.get(&bias.id) {
                params_and_grads.push((bias, grad));
            }
        }
        
        // FC2 parameters
        if let Some(grad) = grad_map.get(&fc2.weight.id) {
            params_and_grads.push((&fc2.weight, grad));
        }
        if let Some(bias) = &fc2.bias {
            if let Some(grad) = grad_map.get(&bias.id) {
                params_and_grads.push((bias, grad));
            }
        }
        
        // Update parameters
        adam.step(&params_and_grads)?;
    }
    
    // Verify training is working (loss should decrease on average)
    let initial_loss = losses[0];
    let final_loss = losses[losses.len() - 1];
    
    println!("\nTraining summary:");
    println!("Initial loss: {:.4}", initial_loss);
    println!("Final loss: {:.4}", final_loss);
    println!("Loss decreased: {}", final_loss < initial_loss);
    
    // Loss should generally decrease (allowing for some fluctuation)
    let avg_first_half: f32 = losses[..5].iter().sum::<f32>() / 5.0;
    let avg_second_half: f32 = losses[5..].iter().sum::<f32>() / 5.0;
    
    assert!(avg_second_half < avg_first_half, 
        "Average loss should decrease: {:.4} -> {:.4}", avg_first_half, avg_second_half);
    
    println!("✅ Realistic CNN training test passed!");
    Ok(())
}

/// Test gradient flow through complex computation graph
#[test]
fn test_complex_gradient_flow() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Create a complex computation graph
    let x = Tensor::randn(Shape::from_dims(&[2, 4]), 0.0, 1.0, device.clone())?.requires_grad();
    let w1 = Tensor::randn(Shape::from_dims(&[4, 8]), 0.0, 0.5, device.clone())?.requires_grad();
    let w2 = Tensor::randn(Shape::from_dims(&[8, 4]), 0.0, 0.5, device.clone())?.requires_grad();
    let w3 = Tensor::randn(Shape::from_dims(&[4, 2]), 0.0, 0.5, device.clone())?.requires_grad();
    
    // Forward pass with multiple operations
    let h1 = x.matmul(&w1)?;
    let h1_relu = h1.relu()?;
    let h1_norm = h1_relu.div(&h1_relu.sum()?.add_scalar(1e-8)?)?; // Normalization
    
    let h2 = h1_norm.matmul(&w2)?;
    let h2_tanh = h2.tanh()?;
    
    // Skip connection
    let h2_skip = h2_tanh.add(&x)?;
    
    let output = h2_skip.matmul(&w3)?;
    let loss = output.sum()?;
    
    // Backward pass
    let grad_map = loss.backward()?;
    
    // Verify all gradients exist
    assert!(grad_map.contains_key(&x.id), "x gradient missing");
    assert!(grad_map.contains_key(&w1.id), "w1 gradient missing");
    assert!(grad_map.contains_key(&w2.id), "w2 gradient missing");
    assert!(grad_map.contains_key(&w3.id), "w3 gradient missing");
    
    // Verify gradients are non-zero and finite
    for (name, tensor_id) in [("x", x.id), ("w1", w1.id), ("w2", w2.id), ("w3", w3.id)] {
        let grad = grad_map.get(&tensor_id).unwrap();
        let grad_data = grad.to_vec()?;
        
        let has_nonzero = grad_data.iter().any(|&v| v != 0.0);
        let all_finite = grad_data.iter().all(|&v| v.is_finite());
        
        assert!(has_nonzero, "{} gradient is all zeros", name);
        assert!(all_finite, "{} gradient has non-finite values", name);
    }
    
    println!("✅ Complex gradient flow test passed!");
    Ok(())
}

/// Test training with different batch sizes
#[test]
fn test_variable_batch_sizes() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Create a simple model
    let linear = Linear::new(10, 5, true, device.clone())?;
    
    // Test with different batch sizes
    for batch_size in [1, 4, 16, 32] {
        let input = Tensor::randn(
            Shape::from_dims(&[batch_size, 10]),
            0.0,
            1.0,
            device.clone()
        )?.requires_grad();
        
        let target = Tensor::zeros(Shape::from_dims(&[batch_size, 5]), device.clone())?;
        
        let output = linear.forward(&input)?;
        let loss = mse_loss(&output, &target)?;
        
        let grad_map = loss.backward()?;
        
        // Verify gradient exists and has correct shape
        let weight_grad = grad_map.get(&linear.weight.id)
            .expect("Weight gradient should exist");
        assert_eq!(weight_grad.shape(), linear.weight.shape());
        
        println!("✅ Batch size {} test passed", batch_size);
    }
    
    Ok(())
}

/// Test gradient accumulation (important for large models)
#[test]
fn test_gradient_accumulation() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    let linear = Linear::new(5, 3, true, device.clone())?;
    let accumulation_steps = 4;
    
    // Track accumulated gradients manually
    let mut accumulated_weight_grad: Option<Tensor> = None;
    let mut accumulated_bias_grad: Option<Tensor> = None;
    
    for step in 0..accumulation_steps {
        let input = Tensor::randn(Shape::from_dims(&[2, 5]), 0.0, 1.0, device.clone())?.requires_grad();
        let target = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?;
        
        let output = linear.forward(&input)?;
        let loss = mse_loss(&output, &target)?;
        
        // Scale loss by accumulation steps
        let scaled_loss = loss.div_scalar(accumulation_steps as f32)?;
        let grad_map = scaled_loss.backward()?;
        
        // Accumulate gradients
        if let Some(grad) = grad_map.get(&linear.weight.id) {
            match &mut accumulated_weight_grad {
                None => accumulated_weight_grad = Some(grad.clone()?),
                Some(acc) => *acc = acc.add(grad)?,
            }
        }
        
        if let Some(bias) = &linear.bias {
            if let Some(grad) = grad_map.get(&bias.id) {
                match &mut accumulated_bias_grad {
                    None => accumulated_bias_grad = Some(grad.clone()?),
                    Some(acc) => *acc = acc.add(grad)?,
                }
            }
        }
    }
    
    // Verify accumulated gradients exist and are reasonable
    assert!(accumulated_weight_grad.is_some(), "Weight gradients should be accumulated");
    assert!(accumulated_bias_grad.is_some(), "Bias gradients should be accumulated");
    
    let weight_grad = accumulated_weight_grad.unwrap();
    let weight_grad_norm = weight_grad.pow_scalar(2.0)?.sum()?.sqrt()?.item()?;
    
    assert!(weight_grad_norm > 0.0, "Gradient norm should be non-zero");
    assert!(weight_grad_norm.is_finite(), "Gradient norm should be finite");
    
    println!("✅ Gradient accumulation test passed!");
    Ok(())
}

// Helper function for max pooling
fn pool2d_max(input: &Tensor, kernel_size: usize, stride: usize) -> Result<Tensor> {
    use flame_core::pooling::MaxPool2d;
    let pool = MaxPool2d::new(kernel_size, stride, 0);
    pool.forward(input)
}