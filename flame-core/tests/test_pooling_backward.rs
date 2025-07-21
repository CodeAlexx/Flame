use flame_core::{Tensor, Shape};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

#[test]
fn test_maxpool2d_forward_with_indices() {
    let device = CudaDevice::new(0).expect("CUDA device");
    
    // Create a simple test input
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input = Tensor::from_vec(input_data, Shape::from_dims(&[1, 1, 4, 4]), device.clone())
        .expect("Tensor creation should succeed");
    
    // MaxPool2d with 2x2 kernel, stride 2
    let kernel_size = (2, 2);
    let stride = (2, 2);
    let padding = (0, 0);
    
    // Use the CUDA kernel directly
    use flame_core::cuda_kernels::CudaKernels;
    let (output, indices) = CudaKernels::maxpool2d_forward_with_indices(&input, kernel_size, stride, padding)
        .expect("MaxPool2d forward should succeed");
    
    // Check output shape
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
    assert_eq!(indices.shape().dims(), &[1, 1, 2, 2]);
    
    // Check output values
    let output_data = output.to_vec().expect("to_vec should succeed");
    assert_eq!(output_data, vec![6.0, 8.0, 14.0, 16.0]); // Max values from each 2x2 window
    
    // Check indices (positions of max values in flattened input)
    let indices_data = indices.to_vec().expect("to_vec should succeed");
    // Expected indices: 5 (position of 6), 7 (position of 8), 13 (position of 14), 15 (position of 16)
    assert_eq!(indices_data, vec![5.0, 7.0, 13.0, 15.0]);
    
    println!("✅ MaxPool2d forward with indices works correctly");
}

#[test]
fn test_maxpool2d_backward_with_indices() {
    let device = CudaDevice::new(0).expect("CUDA device");
    
    // Create test input
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input = Tensor::from_vec(input_data, Shape::from_dims(&[1, 1, 4, 4]), device.clone())
        .expect("Tensor creation should succeed");
    
    // Forward pass
    use flame_core::cuda_kernels::CudaKernels;
    let (output, indices) = CudaKernels::maxpool2d_forward_with_indices(&input, (2, 2), (2, 2), (0, 0))
        .expect("Forward pass should succeed");
    
    // Create gradient for output (all ones for simplicity)
    let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], output.shape().clone(), device.clone())
        .expect("Grad tensor creation should succeed");
    
    // Backward pass
    let grad_input = CudaKernels::maxpool2d_backward_with_indices(&grad_output, &input, &indices)
        .expect("Backward pass should succeed");
    
    // Check gradient shape
    assert_eq!(grad_input.shape(), input.shape());
    
    // Check gradient values
    let grad_data = grad_input.to_vec().expect("to_vec should succeed");
    
    // Expected gradient: zeros except at positions that were max values
    let expected_grad = vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,  // 6 and 8 were max
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,  // 14 and 16 were max
    ];
    
    assert_eq!(grad_data, expected_grad);
    println!("✅ MaxPool2d backward with indices works correctly");
}

#[test]
fn test_avgpool2d_backward() {
    let device = CudaDevice::new(0).expect("CUDA device");
    
    // Create test input
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input = Tensor::from_vec(input_data, Shape::from_dims(&[1, 1, 4, 4]), device.clone())
        .expect("Tensor creation should succeed");
    
    // Forward pass (average pooling)
    use flame_core::cuda_kernels::CudaKernels;
    let output = CudaKernels::avgpool2d_forward(&input, (2, 2), (2, 2), (0, 0), false)
        .expect("AvgPool2d forward should succeed");
    
    // Check output values
    let output_data = output.to_vec().expect("to_vec should succeed");
    // Expected: average of each 2x2 window
    assert_eq!(output_data, vec![3.5, 5.5, 11.5, 13.5]);
    
    // Create gradient for output
    let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], output.shape().clone(), device.clone())
        .expect("Grad tensor creation should succeed");
    
    // Backward pass
    let grad_input = CudaKernels::avgpool2d_backward(&grad_output, &input, (2, 2), (2, 2), (0, 0), false)
        .expect("AvgPool2d backward should succeed");
    
    // Check gradient values
    let grad_data = grad_input.to_vec().expect("to_vec should succeed");
    
    // Each input contributes to one output, so gradient is 1/4 = 0.25
    let expected_grad = vec![
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
    ];
    
    // Check with tolerance due to floating point
    for (actual, expected) in grad_data.iter().zip(expected_grad.iter()) {
        assert!((actual - expected).abs() < 1e-6, "Gradient mismatch: {} vs {}", actual, expected);
    }
    
    println!("✅ AvgPool2d backward works correctly");
}

#[test]
fn test_maxpool2d_gradient_flow() {
    let device = CudaDevice::new(0).expect("CUDA device");
    
    // Create input that requires gradient
    let input = Tensor::from_vec(
        vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
        Shape::from_dims(&[1, 1, 2, 4]),
        device.clone()
    ).expect("Tensor creation should succeed")
     .requires_grad();
    
    // Forward pass
    use flame_core::cuda_kernels::CudaKernels;
    let (output, indices) = CudaKernels::maxpool2d_forward_with_indices(&input, (1, 2), (1, 2), (0, 0))
        .expect("Forward pass should succeed");
    
    // Simulate loss and backward
    let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0], output.shape().clone(), device.clone())
        .expect("Grad creation should succeed");
    
    let grad_input = CudaKernels::maxpool2d_backward_with_indices(&grad_output, &input, &indices)
        .expect("Backward pass should succeed");
    
    let grad_data = grad_input.to_vec().expect("to_vec should succeed");
    
    // Gradient should flow to max elements: positions 1, 3, 7
    let expected_grad = vec![
        0.0, 1.0, 0.0, 2.0,  // First row: gradients at positions 1 and 3
        0.0, 0.0, 0.0, 3.0,  // Second row: gradient at position 7
    ];
    
    assert_eq!(grad_data, expected_grad);
    println!("✅ MaxPool2d gradient flow works correctly");
}