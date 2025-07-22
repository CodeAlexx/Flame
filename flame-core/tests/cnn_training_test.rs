//! CNN Training Integration Test for FLAME
//! Tests complete training loop with Conv2D, pooling, and backpropagation

use flame_core::{
    Tensor, Shape, Result,
    conv::Conv2d, pooling::{MaxPool2d, MaxPool2dConfig},
    linear::Linear,
};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Simple CNN for MNIST-like data
struct SimpleCNN {
    device: Arc<CudaDevice>,
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    pool: MaxPool2d,
}

impl SimpleCNN {
    fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Architecture:
        // Conv2d(1, 16, 3) -> ReLU -> MaxPool2d(2) 
        // Conv2d(16, 32, 3) -> ReLU -> MaxPool2d(2)
        // Flatten -> Linear(32*5*5, 128) -> ReLU
        // Linear(128, 10)
        
        let conv1 = Conv2d::new(1, 16, 3, 1, 1, device.clone())?;
        let conv2 = Conv2d::new(16, 32, 3, 1, 1, device.clone())?;
        
        // After two conv layers with same padding and two 2x2 pools:
        // Conv1: 28x28 -> 28x28 (padding preserves size)
        // Pool1: 28x28 -> 14x14
        // Conv2: 14x14 -> 14x14 (padding preserves size)
        // Pool2: 14x14 -> 7x7
        // Final: 7x7x32 = 1568
        let fc1 = Linear::new(32 * 7 * 7, 128, true, &device)?;
        let fc2 = Linear::new(128, 10, true, &device)?;
        
        let pool = MaxPool2d::new(MaxPool2dConfig::new((2, 2)));
        
        Ok(Self {
            device,
            conv1,
            conv2,
            fc1,
            fc2,
            pool,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // First conv block
        let x = self.conv1.forward(x)?;
        let x = x.relu()?;
        let (x, _) = self.pool.forward(&x)?;
        
        // Second conv block
        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;
        let (x, _) = self.pool.forward(&x)?;
        
        // Flatten
        let batch_size = x.shape().dims()[0];
        let x = x.reshape(&[batch_size, 32 * 7 * 7])?;
        
        // FC layers
        let x = self.fc1.forward(&x)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        
        Ok(x)
    }
    
}

/// Cross-entropy loss for classification
fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Compute log_softmax: log(exp(x_i) / sum(exp(x_j)))
    let max_logits = logits.max_keepdim(1)?;
    let shifted_logits = logits.sub(&max_logits)?;
    let exp_logits = shifted_logits.exp()?;
    let sum_exp = exp_logits.sum_keepdim(1)?;
    let log_sum_exp = sum_exp.log()?;
    let log_softmax = shifted_logits.sub(&log_sum_exp)?;
    
    // Gather log probabilities for target classes
    // targets should be class indices
    let batch_size = logits.shape().dims()[0];
    let num_classes = logits.shape().dims()[1];
    
    // Create one-hot encoding of targets
    let targets_long = targets.to_dtype(DType::I64)?;
    let one_hot = Tensor::zeros(&[batch_size, num_classes], DType::F32, targets.device())?;
    let one_hot = one_hot.scatter(1, &targets_long.unsqueeze(1)?, &Tensor::ones(&[batch_size, 1], DType::F32, targets.device())?)?;
    
    // Compute negative log likelihood
    let nll = log_softmax.mul(&one_hot)?.sum()?;
    nll.neg()?.div_scalar(batch_size as f32)
}

#[test]
fn test_cnn_forward_backward() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Create model
    let model = SimpleCNN::new(device.clone())?;
    
    // Training parameters
    let batch_size = 2;
    
    println!("Starting CNN forward/backward test...");
    
    // Create dummy batch (batch_size, 1, 28, 28)
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size, 1, 28, 28]),
        0.0,
        0.1,
        device.clone()
    )?.requires_grad_(true);
    
    // Create one-hot targets (batch_size, 10)
    let mut target_data = vec![0.0f32; batch_size * 10];
    for i in 0..batch_size {
        let class = (i * 3) % 10; // Dummy class assignment
        target_data[i * 10 + class] = 1.0;
    }
    let targets = Tensor::from_vec(
        target_data,
        Shape::from_dims(&[batch_size, 10]),
        device.clone()
    )?;
    
    // Forward pass
    let output = model.forward(&input)?;
    let loss = cross_entropy_loss(&output, &targets)?;
    
    let loss_value = loss.to_vec()?[0];
    println!("Loss = {:.4}", loss_value);
    
    // Backward pass
    let grads = loss.backward()?;
    
    // Check that we got gradients for input
    assert!(grads.contains(input.id()), "No gradient for input");
    
    // Check that gradients for model parameters exist
    assert!(grads.contains(model.conv1.weight.id()), "No gradient for conv1 weight");
    assert!(grads.contains(model.conv2.weight.id()), "No gradient for conv2 weight");
    assert!(grads.contains(model.fc1.weight.id()), "No gradient for fc1 weight");
    assert!(grads.contains(model.fc2.weight.id()), "No gradient for fc2 weight");
    
    // Check that loss is finite
    assert!(!loss_value.is_nan(), "Loss is NaN");
    assert!(!loss_value.is_infinite(), "Loss is infinite");
    
    println!("CNN forward/backward test completed successfully!");
    Ok(())
}

#[test]
fn test_cnn_gradient_flow() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Smaller CNN for gradient checking
    let conv = Conv2d::new(1, 4, 3, 1, 0, device.clone())?;
    let pool = MaxPool2d::new(MaxPool2dConfig::new((2, 2)));
    let fc = Linear::new(4 * 4 * 4, 2, true, &device)?;
    
    // Small input
    let input = Tensor::randn(
        Shape::from_dims(&[1, 1, 10, 10]),
        0.0,
        0.1,
        device.clone()
    )?.requires_grad_(true);
    
    let target = Tensor::from_vec(
        vec![1.0, 0.0],
        Shape::from_dims(&[1, 2]),
        device.clone()
    )?;
    
    // Forward pass
    let x = conv.forward(&input)?;
    let x = x.relu()?;
    let (x, _) = pool.forward(&x)?;
    let x = x.reshape(&[1, 4 * 4 * 4])?;
    let output = fc.forward(&x)?;
    
    let loss = cross_entropy_loss(&output, &target)?;
    
    // Backward pass
    let grads = loss.backward()?;
    
    // Check that gradients exist for model weights
    let param_ids = vec![
        (conv.weight.id(), "conv.weight"),
        (fc.weight.id(), "fc.weight"),
    ];
    
    let mut grad_count = 0;
    
    for (param_id, name) in param_ids {
        if let Some(grad) = grads.get(param_id) {
            let grad_data = grad.to_vec()?;
            let grad_norm: f32 = grad_data.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            println!("{} gradient norm: {:.6}", name, grad_norm);
            
            // Check gradient is not zero (at least for some parameters)
            if grad_norm > 1e-8 {
                grad_count += 1;
            }
        }
    }
    
    assert!(grad_count > 0, "No non-zero gradients found!");
    println!("Gradient flow test passed - {} parameters have non-zero gradients", grad_count);
    
    Ok(())
}

#[test]
fn test_conv_pooling_dimensions() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Test various input sizes and configurations
    let test_cases = vec![
        // (input_shape, kernel, stride, padding, expected_after_conv, expected_after_pool)
        ([2, 3, 32, 32], (3, 3), (1, 1), (1, 1), [2, 8, 32, 32], [2, 8, 16, 16]),
        ([1, 1, 28, 28], (5, 5), (1, 1), (0, 0), [1, 4, 24, 24], [1, 4, 12, 12]),
        ([4, 16, 64, 64], (3, 3), (1, 1), (1, 1), [4, 32, 64, 64], [4, 32, 32, 32]),
    ];
    
    for (i, (input_shape, kernel, stride, padding, expected_conv, expected_pool)) in test_cases.iter().enumerate() {
        println!("Test case {}: {:?}", i, input_shape);
        
        let input = Tensor::randn(
            Shape::from_dims(input_shape),
            0.0,
            1.0,
            device.clone()
        )?;
        
        let conv = Conv2d::new(
            input_shape[1], 
            expected_conv[1], 
            kernel.0,
            stride.0,
            padding.0,
            device.clone()
        )?;
        
        let pool = MaxPool2d::new(MaxPool2dConfig::new((2, 2)));
        
        let conv_out = conv.forward(&input)?;
        assert_eq!(conv_out.shape().dims(), expected_conv, "Conv output shape mismatch");
        
        let (pool_out, _) = pool.forward(&conv_out)?;
        assert_eq!(pool_out.shape().dims(), expected_pool, "Pool output shape mismatch");
    }
    
    println!("Dimension test passed!");
    Ok(())
}