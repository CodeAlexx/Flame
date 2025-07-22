use flame_core::{Tensor, Shape, CudaDevice, Result};
use std::sync::Arc;

fn test_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).unwrap()
}

/// Helper to check if two tensors are approximately equal
fn assert_tensor_approx_eq(a: &Tensor, b: &Tensor, tolerance: f32) -> Result<()> {
    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;
    
    assert_eq!(a_data.len(), b_data.len(), "Tensor sizes don't match");
    
    for (i, (a_val, b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
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
fn test_matmul_correctness() -> Result<()> {
    let device = test_device();
    
    // Test 1: Simple 2x2 matrix multiplication
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?;
    
    let b = Tensor::from_vec(
        vec![5.0, 6.0, 7.0, 8.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?;
    
    let c = a.matmul(&b)?;
    
    // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    //         = [[19, 22], [43, 50]]
    let expected = Tensor::from_vec(
        vec![19.0, 22.0, 43.0, 50.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?;
    
    assert_tensor_approx_eq(&c, &expected, 1e-5)?;
    
    // Test 2: Non-square matrices
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone()
    )?;
    
    let b = Tensor::from_vec(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        Shape::from_dims(&[3, 2]),
        device.clone()
    )?;
    
    let c = a.matmul(&b)?;
    assert_eq!(c.shape().dims(), &[2, 2]);
    
    // Expected: [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12], 
    //            [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
    //         = [[58, 64], [139, 154]]
    let expected = Tensor::from_vec(
        vec![58.0, 64.0, 139.0, 154.0],
        Shape::from_dims(&[2, 2]),
        device
    )?;
    
    assert_tensor_approx_eq(&c, &expected, 1e-5)?;
    
    println!("✅ Matrix multiplication tests passed!");
    Ok(())
}

#[test]
fn test_add_broadcast_correctness() -> Result<()> {
    let device = test_device();
    
    // Test 1: Same shape addition
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), device.clone())?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[4]), device.clone())?;
    let c = a.add(&b)?;
    
    let expected = Tensor::from_vec(vec![6.0, 8.0, 10.0, 12.0], Shape::from_dims(&[4]), device.clone())?;
    assert_tensor_approx_eq(&c, &expected, 1e-5)?;
    
    // Test 2: Broadcasting scalar
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?;
    let b = Tensor::from_vec(vec![10.0], Shape::from_dims(&[1]), device.clone())?;
    let c = a.add(&b)?;
    
    let expected = Tensor::from_vec(vec![11.0, 12.0, 13.0, 14.0], Shape::from_dims(&[2, 2]), device.clone())?;
    assert_tensor_approx_eq(&c, &expected, 1e-5)?;
    
    // Test 3: Broadcasting along dimension
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_dims(&[2, 3]), device.clone())?;
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], Shape::from_dims(&[1, 3]), device.clone())?;
    let c = a.add(&b)?;
    
    let expected = Tensor::from_vec(
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0], 
        Shape::from_dims(&[2, 3]), 
        device
    )?;
    assert_tensor_approx_eq(&c, &expected, 1e-5)?;
    
    println!("✅ Addition and broadcasting tests passed!");
    Ok(())
}

#[test]
fn test_mul_correctness() -> Result<()> {
    let device = test_device();
    
    // Test element-wise multiplication
    let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], Shape::from_dims(&[4]), device.clone())?;
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::from_dims(&[4]), device.clone())?;
    let c = a.mul(&b)?;
    
    let expected = Tensor::from_vec(vec![20.0, 60.0, 120.0, 200.0], Shape::from_dims(&[4]), device.clone())?;
    assert_tensor_approx_eq(&c, &expected, 1e-5)?;
    
    // Test scalar multiplication
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?;
    let c = a.mul_scalar(0.5)?;
    
    let expected = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], Shape::from_dims(&[2, 2]), device)?;
    assert_tensor_approx_eq(&c, &expected, 1e-5)?;
    
    println!("✅ Multiplication tests passed!");
    Ok(())
}

#[test]
fn test_relu_forward_backward() -> Result<()> {
    let device = test_device();
    
    // Test forward pass
    let input = Tensor::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        Shape::from_dims(&[5]),
        device.clone()
    )?;
    
    let output = input.relu()?;
    let expected = Tensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 2.0],
        Shape::from_dims(&[5]),
        device.clone()
    )?;
    
    assert_tensor_approx_eq(&output, &expected, 1e-5)?;
    
    // Test backward pass (gradient)
    let input = Tensor::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        Shape::from_dims(&[5]),
        device.clone()
    )?.requires_grad_(true);
    
    let output = input.relu()?;
    let loss = output.sum()?;
    let grads = loss.backward()?;
    
    let input_grad = grads.get(input.id()).unwrap();
    let grad_data = input_grad.to_vec()?;
    
    // ReLU gradient: 0 for negative inputs, 1 for positive
    assert_eq!(grad_data, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    
    println!("✅ ReLU forward and backward tests passed!");
    Ok(())
}

#[test]
fn test_gelu_activation() -> Result<()> {
    let device = test_device();
    
    let input = Tensor::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        Shape::from_dims(&[5]),
        device.clone()
    )?;
    
    let output = input.gelu()?;
    
    // GELU approximation should be smooth
    let output_data = output.to_vec()?;
    
    // Check that GELU(0) ≈ 0
    assert!((output_data[2] - 0.0).abs() < 0.01);
    
    // Check that negative values are dampened but not zero
    assert!(output_data[0] > -0.5 && output_data[0] < 0.0);
    assert!(output_data[1] > -0.5 && output_data[1] < 0.0);
    
    // Check that positive values are close to identity but slightly less
    assert!(output_data[3] > 0.5 && output_data[3] < 1.0);
    assert!(output_data[4] > 1.5 && output_data[4] < 2.0);
    
    println!("✅ GELU activation test passed!");
    Ok(())
}

#[test]
fn test_sum_reduction() -> Result<()> {
    let device = test_device();
    
    // Test 1: Simple sum
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), device.clone())?;
    let sum = a.sum()?;
    
    assert_eq!(sum.shape().dims(), &[1]);
    assert_eq!(sum.to_vec()?, vec![10.0]);
    
    // Test 2: 2D sum
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone()
    )?;
    let sum = a.sum()?;
    
    assert_eq!(sum.to_vec()?, vec![21.0]);
    
    // Test 3: Sum along specific dimension
    let sum_dim0 = a.sum_dim(0, false)?;
    assert_eq!(sum_dim0.shape().dims(), &[3]);
    let expected = Tensor::from_vec(vec![5.0, 7.0, 9.0], Shape::from_dims(&[3]), device.clone())?;
    assert_tensor_approx_eq(&sum_dim0, &expected, 1e-5)?;
    
    let sum_dim1 = a.sum_dim(1, false)?;
    assert_eq!(sum_dim1.shape().dims(), &[2]);
    let expected = Tensor::from_vec(vec![6.0, 15.0], Shape::from_dims(&[2]), device)?;
    assert_tensor_approx_eq(&sum_dim1, &expected, 1e-5)?;
    
    println!("✅ Sum reduction tests passed!");
    Ok(())
}

#[test]
fn test_mean_operation() -> Result<()> {
    let device = test_device();
    
    // Test 1: Simple mean
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), device.clone())?;
    let mean = a.mean()?;
    
    assert_eq!(mean.shape().dims(), &[1]);
    assert_eq!(mean.to_vec()?, vec![2.5]);
    
    // Test 2: 2D mean
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device
    )?;
    let mean = a.mean()?;
    
    assert_eq!(mean.to_vec()?, vec![3.5]);
    
    println!("✅ Mean operation tests passed!");
    Ok(())
}

#[test]
fn test_transpose_operation() -> Result<()> {
    let device = test_device();
    
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone()
    )?;
    
    let transposed = a.transpose()?;
    assert_eq!(transposed.shape().dims(), &[3, 2]);
    
    let expected = Tensor::from_vec(
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        Shape::from_dims(&[3, 2]),
        device
    )?;
    
    assert_tensor_approx_eq(&transposed, &expected, 1e-5)?;
    
    println!("✅ Transpose operation test passed!");
    Ok(())
}

#[test]
fn test_reshape_operation() -> Result<()> {
    let device = test_device();
    
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone()
    )?;
    
    // Reshape to [3, 2]
    let reshaped = a.reshape(&[3, 2])?;
    assert_eq!(reshaped.shape().dims(), &[3, 2]);
    
    // Data should remain the same
    assert_eq!(a.to_vec()?, reshaped.to_vec()?);
    
    // Reshape to [6]
    let flattened = a.reshape(&[6])?;
    assert_eq!(flattened.shape().dims(), &[6]);
    assert_eq!(a.to_vec()?, flattened.to_vec()?);
    
    // Reshape to [1, 6]
    let row = a.reshape(&[1, 6])?;
    assert_eq!(row.shape().dims(), &[1, 6]);
    
    println!("✅ Reshape operation tests passed!");
    Ok(())
}

#[test]
fn test_all_core_operations() -> Result<()> {
    // Run all tests to ensure core operations work
    test_matmul_correctness()?;
    test_add_broadcast_correctness()?;
    test_mul_correctness()?;
    test_relu_forward_backward()?;
    test_gelu_activation()?;
    test_sum_reduction()?;
    test_mean_operation()?;
    test_transpose_operation()?;
    test_reshape_operation()?;
    
    println!("\n🎉 ALL CORE OPERATION TESTS PASSED! 🎉");
    println!("FLAME has a solid foundation of working operations!");
    
    Ok(())
}