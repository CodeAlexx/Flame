//! Comprehensive gradient tests with real weights and operations
//! Tests the entire autograd system with actual computations

use flame_core::{Tensor, Shape, Result, autograd::AutogradContext, TensorGradExt};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// Helper to create a test device
fn test_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("Failed to create CUDA device")
}

/// Helper to setup test environment
fn setup_test() -> Arc<CudaDevice> {
    // Reset autograd context to ensure clean state
    AutogradContext::reset();
    test_device()
}

/// Helper to check if gradients are approximately equal
fn assert_grad_close(actual: &Tensor, expected: &[f32], tolerance: f32) {
    let actual_data = actual.to_vec().expect("Failed to get tensor data");
    assert_eq!(actual_data.len(), expected.len(), "Gradient shape mismatch");
    
    for (i, (&a, &e)) in actual_data.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tolerance,
            "Gradient mismatch at index {}: actual {} vs expected {}, diff {}",
            i, a, e, diff
        );
    }
}

#[test]
fn test_basic_gradient_flow() -> Result<()> {
    let device = setup_test();
    
    // Create input tensor with requires_grad
    let x = Tensor::from_vec(vec![2.0, 3.0, 4.0], Shape::from_dims(&[3]), device.clone())?;
    let x = x.requires_grad_(true);
    
    // y = x * 2
    let y = x.mul_scalar(2.0)?;
    
    // z = y + 3 = 2x + 3
    let z = y.add_scalar(3.0)?;
    
    // loss = sum(z) = sum(2x + 3) = 2*sum(x) + 9
    let loss = z.sum()?;
    
    // Compute gradients
    let grads = AutogradContext::backward(&loss)?;
    
    // d(loss)/dx = 2 for all elements
    let x_grad = x.grad(&grads).expect("Missing gradient for x");
    assert_grad_close(&x_grad, &[2.0, 2.0, 2.0], 1e-5);
    
    Ok(())
}

#[test]
fn test_matrix_multiplication_gradients() -> Result<()> {
    let device = setup_test();
    
    // A: [2, 3], B: [3, 2]
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    
    let a = Tensor::from_vec(a_data.clone(), Shape::from_dims(&[2, 3]), device.clone())?
        .requires_grad_(true);
    let b = Tensor::from_vec(b_data.clone(), Shape::from_dims(&[3, 2]), device.clone())?
        .requires_grad_(true);
    
    // C = A @ B
    let c = a.matmul(&b)?;
    
    // loss = sum(C)
    let loss = c.sum()?;
    
    // Compute gradients
    let grads = AutogradContext::backward(&loss)?;
    
    // Check gradient shapes
    let a_grad = a.grad(&grads).expect("Missing gradient for A");
    let b_grad = b.grad(&grads).expect("Missing gradient for B");
    
    assert_eq!(a_grad.shape(), a.shape());
    assert_eq!(b_grad.shape(), b.shape());
    
    // For loss = sum(A @ B):
    // dL/dA = ones @ B^T
    // dL/dB = A^T @ ones
    
    // B^T is [[0.5, 2.5, 4.5], [1.5, 3.5, 5.5]]
    // sum of each row: [0.5+2.5+4.5, 1.5+3.5+5.5] = [7.5, 10.5]
    // But we need to broadcast this to match A's shape
    // Actually: dL/dA = ones(2,2) @ B^T = [[1,1],[1,1]] @ [[0.5,2.5,4.5],[1.5,3.5,5.5]]
    // = [[2.0, 6.0, 10.0], [2.0, 6.0, 10.0]]
    let expected_a_grad = vec![
        2.0, 6.0, 10.0,  // First row
        2.0, 6.0, 10.0,  // Second row
    ];
    assert_grad_close(&a_grad, &expected_a_grad, 1e-4);
    
    // Expected gradient for B: sum of each column of A^T
    let expected_b_grad = vec![
        5.0, 5.0,   // First row
        7.0, 7.0,   // Second row
        9.0, 9.0,   // Third row
    ];
    assert_grad_close(&b_grad, &expected_b_grad, 1e-4);
    
    Ok(())
}

#[test]
fn test_activation_gradients() -> Result<()> {
    let device = setup_test();
    
    // Test ReLU gradient
    let x = Tensor::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        Shape::from_dims(&[5]),
        device.clone()
    )?.requires_grad_(true);
    
    let y = x.relu()?;
    let loss = y.sum()?;
    
    let grads = AutogradContext::backward(&loss)?;
    let x_grad = x.grad(&grads).expect("Missing gradient for x");
    
    // ReLU gradient: 1 if x > 0, 0 otherwise
    assert_grad_close(&x_grad, &[0.0, 0.0, 0.0, 1.0, 1.0], 1e-5);
    
    Ok(())
}

#[test]
fn test_conv2d_gradients() -> Result<()> {
    let device = setup_test();
    
    // Small conv2d test: [1, 1, 3, 3] input, [1, 1, 2, 2] kernel
    let input_data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    
    let kernel_data = vec![
        0.5, 1.0,
        1.5, 2.0,
    ];
    
    let input = Tensor::from_vec(
        input_data,
        Shape::from_dims(&[1, 1, 3, 3]),
        device.clone()
    )?.requires_grad_(true);
    
    let kernel = Tensor::from_vec(
        kernel_data,
        Shape::from_dims(&[1, 1, 2, 2]),
        device.clone()
    )?.requires_grad_(true);
    
    // Create Conv2d operation manually
    use flame_core::conv::Conv2d;
    let mut conv = Conv2d::new(1, 1, 2, 1, 0, device.clone())?;
    
    // Replace weight with our kernel (keep reference for gradient check)
    let kernel_ref = kernel.clone()?;
    conv.weight = kernel;
    let output = conv.forward(&input)?;
    
    let loss = output.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    
    // Check gradients exist
    let input_grad = input.grad(&grads).expect("Missing gradient for input");
    let kernel_grad = kernel_ref.grad(&grads).expect("Missing gradient for kernel");
    
    assert_eq!(input_grad.shape(), input.shape());
    assert_eq!(kernel_grad.shape(), kernel_ref.shape());
    
    Ok(())
}

#[test]
fn test_layer_norm_gradients() -> Result<()> {
    let device = setup_test();
    
    // Test LayerNorm gradient computation
    let batch_size = 2;
    let seq_len = 3;
    let hidden_size = 4;
    
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, hidden_size]),
        0.0,
        1.0,
        device.clone()
    )?.requires_grad_(true);
    
    let gamma = Tensor::ones(Shape::from_dims(&[hidden_size]), device.clone())?
        .requires_grad_(true);
    let beta = Tensor::zeros(Shape::from_dims(&[hidden_size]), device.clone())?
        .requires_grad_(true);
    
    // Manual layer norm computation for testing
    let eps = 1e-5;
    
    // Compute mean and variance along last dimension
    let mean = input.mean_dim(&[2], true)?;
    let centered = input.sub(&mean)?;
    let var = centered.square()?.mean_dim(&[2], true)?;
    let std = var.add_scalar(eps)?.sqrt()?;
    let normalized = centered.div(&std)?;
    
    // Apply affine transformation
    let output = normalized.mul(&gamma)?.add(&beta)?;
    
    let loss = output.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    
    // Check all gradients exist
    let input_grad = input.grad(&grads).expect("Missing gradient for input");
    let gamma_grad = gamma.grad(&grads).expect("Missing gradient for gamma");
    let beta_grad = beta.grad(&grads).expect("Missing gradient for beta");
    
    assert_eq!(input_grad.shape(), input.shape());
    assert_eq!(gamma_grad.shape(), gamma.shape());
    assert_eq!(beta_grad.shape(), beta.shape());
    
    // Beta gradient should be sum over batch and sequence dimensions
    let expected_beta_grad = vec![
        (batch_size * seq_len) as f32,
        (batch_size * seq_len) as f32,
        (batch_size * seq_len) as f32,
        (batch_size * seq_len) as f32,
    ];
    assert_grad_close(&beta_grad, &expected_beta_grad, 1e-4);
    
    Ok(())
}

#[test]
fn test_gradient_accumulation() -> Result<()> {
    let device = setup_test();
    
    // Test that gradients accumulate correctly across multiple backward passes
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), device.clone())?
        .requires_grad_(true);
    
    // First forward-backward pass
    let y1 = x.mul_scalar(2.0)?;
    let loss1 = y1.sum()?;
    let grads1 = AutogradContext::backward(&loss1)?;
    
    // Second forward-backward pass
    let y2 = x.mul_scalar(3.0)?;
    let loss2 = y2.sum()?;
    let grads2 = AutogradContext::backward(&loss2)?;
    
    // Gradients should be independent (not accumulated)
    let grad1 = x.grad(&grads1).expect("Missing gradient from pass 1");
    let grad2 = x.grad(&grads2).expect("Missing gradient from pass 2");
    
    assert_grad_close(&grad1, &[2.0, 2.0, 2.0], 1e-5);
    assert_grad_close(&grad2, &[3.0, 3.0, 3.0], 1e-5);
    
    Ok(())
}

#[test]
fn test_complex_computation_graph() -> Result<()> {
    let device = setup_test();
    
    // Build a more complex computation graph
    let x = Tensor::from_vec(vec![1.0, 2.0], Shape::from_dims(&[2, 1]), device.clone())?
        .requires_grad_(true);
    let w1 = Tensor::from_vec(vec![0.5, 1.5], Shape::from_dims(&[1, 2]), device.clone())?
        .requires_grad_(true);
    let w2 = Tensor::from_vec(vec![2.0, -1.0], Shape::from_dims(&[2, 1]), device.clone())?
        .requires_grad_(true);
    
    // Two-layer network: y = relu(x @ w1) @ w2
    let hidden = x.matmul(&w1)?.relu()?;
    let output = hidden.matmul(&w2)?;
    
    let loss = output.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    
    // Check all gradients exist
    let x_grad = x.grad(&grads).expect("Missing gradient for x");
    let w1_grad = w1.grad(&grads).expect("Missing gradient for w1");
    let w2_grad = w2.grad(&grads).expect("Missing gradient for w2");
    
    assert_eq!(x_grad.shape(), x.shape());
    assert_eq!(w1_grad.shape(), w1.shape());
    assert_eq!(w2_grad.shape(), w2.shape());
    
    // w2 gradient should be the hidden activations
    let hidden_vals = hidden.to_vec()?;
    assert_grad_close(&w2_grad, &hidden_vals, 1e-4);
    
    Ok(())
}

#[test]
fn test_batch_operations() -> Result<()> {
    let device = setup_test();
    
    // Test batch matrix multiplication
    let batch_size = 3;
    let a = Tensor::randn(
        Shape::from_dims(&[batch_size, 2, 3]),
        0.0,
        1.0,
        device.clone()
    )?.requires_grad_(true);
    
    let b = Tensor::randn(
        Shape::from_dims(&[batch_size, 3, 4]),
        0.0,
        1.0,
        device.clone()
    )?.requires_grad_(true);
    
    let c = a.bmm(&b)?;
    assert_eq!(c.shape().dims(), &[batch_size, 2, 4]);
    
    let loss = c.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    
    let a_grad = a.grad(&grads).expect("Missing gradient for a");
    let b_grad = b.grad(&grads).expect("Missing gradient for b");
    
    assert_eq!(a_grad.shape(), a.shape());
    assert_eq!(b_grad.shape(), b.shape());
    
    Ok(())
}

#[test]
fn test_broadcasting_gradients() -> Result<()> {
    let device = setup_test();
    
    // Test gradient computation with broadcasting
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?.requires_grad_(true);
    
    let b = Tensor::from_vec(
        vec![0.5, 1.5],
        Shape::from_dims(&[2]),
        device.clone()
    )?.requires_grad_(true);
    
    // Broadcast b to match a's shape
    let c = a.add(&b)?;  // Broadcasting should happen automatically
    
    let loss = c.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    
    let a_grad = a.grad(&grads).expect("Missing gradient for a");
    let b_grad = b.grad(&grads).expect("Missing gradient for b");
    
    // a's gradient should be all ones
    assert_grad_close(&a_grad, &[1.0, 1.0, 1.0, 1.0], 1e-5);
    
    // b's gradient should be summed across the broadcast dimension
    assert_grad_close(&b_grad, &[2.0, 2.0], 1e-5);
    
    Ok(())
}

#[test]
fn test_reshape_operations() -> Result<()> {
    let device = setup_test();
    
    // Test gradient flow through reshape operations
    let x = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone()
    )?.requires_grad_(true);
    
    // Reshape to [3, 2]
    let y = x.reshape(&[3, 2])?;
    
    // Some computation on reshaped tensor
    let z = y.mul_scalar(2.0)?;
    
    let loss = z.sum()?;
    let grads = AutogradContext::backward(&loss)?;
    
    let x_grad = x.grad(&grads).expect("Missing gradient for x");
    
    // Gradient should flow back through reshape
    assert_eq!(x_grad.shape(), x.shape());
    assert_grad_close(&x_grad, &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 1e-5);
    
    Ok(())
}

#[test]
fn test_memory_efficient_gradients() -> Result<()> {
    let device = setup_test();
    
    // Test gradient computation with memory-efficient operations
    let size = 100;
    let x = Tensor::randn(
        Shape::from_dims(&[size, size]),
        0.0,
        1.0,
        device.clone()
    )?.requires_grad_(true);
    
    // Chain of operations
    let y = x.relu()?.tanh()?.sigmoid()?;
    
    let loss = y.mean()?;
    let grads = AutogradContext::backward(&loss)?;
    
    let x_grad = x.grad(&grads).expect("Missing gradient for x");
    assert_eq!(x_grad.shape(), x.shape());
    
    // Check that gradient values are reasonable (not NaN or infinite)
    let grad_data = x_grad.to_vec()?;
    for &val in &grad_data {
        assert!(val.is_finite(), "Gradient contains NaN or infinite values");
    }
    
    Ok(())
}

#[test]
fn test_gradient_computation_for_squared_values() -> Result<()> {
    let device = setup_test();
    
    // Test gradient computation for x^2
    let x = Tensor::from_vec(
        vec![10.0, -20.0, 30.0],
        Shape::from_dims(&[3]),
        device.clone()
    )?.requires_grad_(true);
    
    // Operation that will produce large gradients
    let y = x.mul(&x)?;  // x^2
    let loss = y.sum()?;
    
    let grads = AutogradContext::backward(&loss)?;
    
    // Gradients should be 2*x = [20, -40, 60]
    let x_grad = x.grad(&grads).expect("Missing gradient for x");
    assert_grad_close(&x_grad, &[20.0, -40.0, 60.0], 1e-4);
    
    // Note: Actual gradient clipping functionality would require:
    // 1. A gradient clipper that modifies gradients in-place
    // 2. GPU kernel support for efficient clipping
    // This test only verifies gradient computation, not clipping
    
    Ok(())
}