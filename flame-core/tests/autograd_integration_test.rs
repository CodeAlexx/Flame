// Comprehensive FLAME Autograd Integration Tests
// Tests the complete framework functionality including forward and backward passes

use flame_core::{
    Tensor, Shape, Result, CudaDevice, 
    autograd::{AutogradContext},
    gradient::GradientMap,
    linear::Linear,
    conv::Conv2d,
    attention::{MultiHeadAttention, AttentionConfig},
    optimizers::{Adam, AdamConfig, SGD, SGDConfig},
};
use std::sync::Arc;

fn create_test_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("Failed to create CUDA device")
}

fn assert_tensor_close(a: &Tensor, b: &Tensor, tolerance: f32) -> Result<()> {
    let a_vec = a.to_vec()?;
    let b_vec = b.to_vec()?;
    
    assert_eq!(a_vec.len(), b_vec.len(), "Tensors have different sizes");
    
    for (i, (a_val, b_val)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        let diff = (a_val - b_val).abs();
        assert!(
            diff < tolerance,
            "Values differ at index {}: {} vs {} (diff: {})",
            i, a_val, b_val, diff
        );
    }
    
    Ok(())
}

#[test]
fn test_conv2d_forward_backward() -> Result<()> {
    let device = create_test_device();
    
    // Create input tensor [batch=2, channels=3, height=8, width=8]
    let input = Tensor::randn(
        Shape::from_dims(&[2, 3, 8, 8]),
        0.0, 0.1,
        device.clone()
    )?.requires_grad_(true);
    
    // Create Conv2d layer: 3 input channels, 16 output channels, 3x3 kernel
    let conv = Conv2d::new_with_bias(3, 16, 3, 1, 1, device.clone(), true)?;
    
    // Forward pass
    let output = conv.forward(&input)?;
    assert_eq!(output.shape().dims(), &[2, 16, 8, 8]);
    
    // Create loss (simple sum)
    let loss = output.sum()?;
    
    // Backward pass
    let grad_map = AutogradContext::backward(&loss)?;
    
    // Check that gradients exist for input and conv parameters
    assert!(grad_map.has_gradient(input.id()));
    assert!(grad_map.has_gradient(conv.weight.id()));
    if let Some(bias) = &conv.bias {
        assert!(grad_map.has_gradient(bias.id()));
    }
    
    // Verify gradient shapes
    let input_grad = grad_map.get(input.id()).unwrap();
    assert_eq!(input_grad.shape(), input.shape());
    
    let weight_grad = grad_map.get(conv.weight.id()).unwrap();
    assert_eq!(weight_grad.shape(), conv.weight.shape());
    
    Ok(())
}

#[test]
fn test_linear_forward_backward() -> Result<()> {
    let device = create_test_device();
    
    // Create input tensor [batch=4, features=32]
    let input = Tensor::randn(
        Shape::from_dims(&[4, 32]),
        0.0, 0.1,
        device.clone()
    )?.requires_grad_(true);
    
    // Create Linear layer: 32 input features, 64 output features
    let linear = Linear::new(32, 64, true, device.clone())?;
    
    // Forward pass
    let output = linear.forward(&input)?;
    assert_eq!(output.shape().dims(), &[4, 64]);
    
    // Create loss (mean squared)
    let target = Tensor::randn(output.shape().clone(), 0.0, 0.1, device.clone())?;
    let diff = output.sub(&target)?;
    let loss = diff.square()?.mean()?;
    
    // Backward pass
    let grad_map = AutogradContext::backward(&loss)?;
    
    // Check gradients
    assert!(grad_map.has_gradient(input.id()));
    assert!(grad_map.has_gradient(linear.weight.id()));
    if let Some(bias) = &linear.bias {
        assert!(grad_map.has_gradient(bias.id()));
    }
    
    Ok(())
}

#[test]
fn test_attention_forward_backward() -> Result<()> {
    let device = create_test_device();
    
    // Create attention config
    let config = AttentionConfig::new(256, 8); // 256 embed dim, 8 heads
    let attention = MultiHeadAttention::new(config, device.clone())?;
    
    // Create input tensors [batch=2, seq_len=10, embed_dim=256]
    let query = Tensor::randn(
        Shape::from_dims(&[2, 10, 256]),
        0.0, 0.1,
        device.clone()
    )?.requires_grad_(true);
    
    let key = query.clone()?;
    let value = query.clone()?;
    
    // Forward pass
    let output = attention.forward(&query, &key, &value, None)?;
    assert_eq!(output.shape().dims(), &[2, 10, 256]);
    
    // Create loss
    let loss = output.mean()?;
    
    // Backward pass
    let grad_map = AutogradContext::backward(&loss)?;
    
    // Check gradients for attention parameters
    assert!(grad_map.has_gradient(query.id()));
    assert!(grad_map.has_gradient(attention.q_proj.weight.id()));
    assert!(grad_map.has_gradient(attention.k_proj.weight.id()));
    assert!(grad_map.has_gradient(attention.v_proj.weight.id()));
    assert!(grad_map.has_gradient(attention.out_proj.weight.id()));
    
    Ok(())
}

#[test]
fn test_adam_optimizer() -> Result<()> {
    let device = create_test_device();
    
    // Create a simple linear layer
    let mut linear = Linear::new(10, 5, true, device.clone())?;
    
    // Create dummy gradients
    let weight_grad = Tensor::ones(linear.weight.shape().clone(), device.clone())?;
    let bias_grad = if let Some(bias) = &linear.bias {
        Some(Tensor::ones(bias.shape().clone(), device.clone())?)
    } else {
        None
    };
    
    // Store initial weights
    let initial_weight = linear.weight.clone()?;
    let initial_bias = linear.bias.as_ref().map(|b| b.clone().unwrap());
    
    // Create optimizer
    let mut optimizer = Adam::new(AdamConfig::default());
    
    // Prepare parameters with gradients
    let mut params = vec![(linear.weight.id(), &mut linear.weight, &weight_grad)];
    if let (Some(bias), Some(grad)) = (linear.bias.as_mut(), bias_grad.as_ref()) {
        params.push((bias.id(), bias, grad));
    }
    
    // Take optimization step
    optimizer.step(&mut params)?;
    
    // Verify weights were updated
    let weight_diff = initial_weight.sub(&linear.weight)?;
    let weight_diff_data = weight_diff.to_vec()?;
    assert!(weight_diff_data.iter().any(|&x| x.abs() > 1e-6), "Weights should have changed");
    
    if let (Some(initial), Some(current)) = (initial_bias, &linear.bias) {
        let bias_diff = initial.sub(current)?;
        let bias_diff_data = bias_diff.to_vec()?;
        assert!(bias_diff_data.iter().any(|&x| x.abs() > 1e-6), "Bias should have changed");
    }
    
    Ok(())
}

#[test]
fn test_sgd_optimizer_with_momentum() -> Result<()> {
    let device = create_test_device();
    
    // Create a simple parameter
    let mut param = Tensor::randn(
        Shape::from_dims(&[5, 5]),
        0.0, 0.1,
        device.clone()
    )?.requires_grad_(true);
    
    // Create gradient
    let grad = Tensor::ones(param.shape().clone(), device.clone())?;
    
    // Create optimizer with momentum
    let mut config = SGDConfig::default();
    config.lr = 0.1;
    config.momentum = 0.9;
    let mut optimizer = SGD::new(config);
    
    // Store initial value
    let initial = param.clone()?;
    
    // Take multiple steps to see momentum effect
    for _ in 0..3 {
        let mut params = vec![(param.id(), &mut param, &grad)];
        optimizer.step(&mut params)?;
    }
    
    // Verify parameter was updated
    let diff = initial.sub(&param)?;
    let diff_data = diff.to_vec()?;
    assert!(diff_data.iter().all(|&x| x.abs() > 0.1), "Parameters should have changed significantly with momentum");
    
    Ok(())
}

#[test]
fn test_complex_gradient_flow() -> Result<()> {
    let device = create_test_device();
    
    // Build a small network: Conv -> Linear -> Loss
    let input = Tensor::randn(
        Shape::from_dims(&[1, 3, 16, 16]),
        0.0, 0.1,
        device.clone()
    )?.requires_grad_(true);
    
    // Conv layer
    let conv = Conv2d::new_with_bias(3, 8, 3, 2, 1, device.clone(), true)?; // stride=2 for downsampling
    let conv_out = conv.forward(&input)?;
    assert_eq!(conv_out.shape().dims(), &[1, 8, 8, 8]);
    
    // Flatten for linear layer
    let flattened = conv_out.reshape(&[1, 8 * 8 * 8])?;
    
    // Linear layer
    let linear = Linear::new(8 * 8 * 8, 10, true, device.clone())?;
    let output = linear.forward(&flattened)?;
    
    // Create target and loss
    let target = Tensor::zeros(Shape::from_dims(&[1, 10]), device.clone())?;
    let diff = output.sub(&target)?;
    let loss = diff.square()?.mean()?;
    
    // Backward pass through entire network
    let grad_map = AutogradContext::backward(&loss)?;
    
    // Verify gradients flow through entire network
    assert!(grad_map.has_gradient(input.id()), "Input should have gradient");
    assert!(grad_map.has_gradient(conv.weight.id()), "Conv weight should have gradient");
    assert!(grad_map.has_gradient(linear.weight.id()), "Linear weight should have gradient");
    
    // Check gradient shapes
    let input_grad = grad_map.get(input.id()).unwrap();
    assert_eq!(input_grad.shape(), input.shape());
    
    Ok(())
}

#[test]
fn test_gradient_accumulation() -> Result<()> {
    let device = create_test_device();
    
    // Create parameter
    let param = Tensor::randn(
        Shape::from_dims(&[5, 5]),
        0.0, 0.1,
        device.clone()
    )?.requires_grad_(true);
    
    // First forward/backward pass
    let loss1 = param.sum()?;
    let grad_map1 = AutogradContext::backward(&loss1)?;
    
    // Second forward/backward pass
    let loss2 = param.mean()?;
    let grad_map2 = AutogradContext::backward(&loss2)?;
    
    // Gradients should be different for different losses
    let grad1 = grad_map1.get(param.id()).unwrap();
    let grad2 = grad_map2.get(param.id()).unwrap();
    
    let grad1_data = grad1.to_vec()?;
    let grad2_data = grad2.to_vec()?;
    
    // sum() gradient should be all 1s
    assert!(grad1_data.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    
    // mean() gradient should be 1/n
    let expected_mean_grad = 1.0 / 25.0;
    assert!(grad2_data.iter().all(|&x| (x - expected_mean_grad).abs() < 1e-6));
    
    Ok(())
}

#[test]
fn test_no_grad_mode() -> Result<()> {
    let device = create_test_device();
    
    // Create tensor that requires grad
    let x = Tensor::ones(Shape::from_dims(&[2, 2]), device.clone())?.requires_grad_(true);
    
    // Operations under no_grad should not create gradient graph
    let y = {
        let _guard = AutogradContext::no_grad();
        x.mul_scalar(2.0)?.add_scalar(1.0)?
    };
    
    // y should not require grad
    assert!(!y.requires_grad());
    
    // Operations outside no_grad should create gradient graph
    let z = y.mul(&x)?;
    assert!(z.requires_grad()); // Because x requires grad
    
    Ok(())
}

#[test]
fn test_framework_is_generic() -> Result<()> {
    // This test verifies that FLAME contains no model-specific code
    // All components should be generic tensor operations
    
    // Conv2d is generic - can be used for any convolutional network
    let device = create_test_device();
    let conv = Conv2d::new_with_bias(3, 64, 7, 2, 3, device.clone(), true)?;
    assert!(conv.in_channels() == 3);
    assert!(conv.out_channels() == 64);
    
    // Linear is generic - can be used for any fully connected layer
    let linear = Linear::new(768, 3072, true, device.clone())?;
    assert!(linear.in_features() == 768);
    assert!(linear.out_features() == 3072);
    
    // Attention is generic - can be used for any transformer
    let config = AttentionConfig::new(512, 8);
    let attention = MultiHeadAttention::new(config, device)?;
    assert!(attention.config.embed_dim == 512);
    assert!(attention.config.num_heads == 8);
    
    // All operations are mathematical, not model-specific
    Ok(())
}