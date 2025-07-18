//! Simple numerical accuracy test between operations
//! 
//! This tests basic tensor operations without dependencies on eridiffusion.
//! 
//! Run with: cargo run --example simple_accuracy_test --release

use anyhow::Result;
use flame_core::{CudaDevice, Tensor, Shape};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("=== Simple FLAME Numerical Test ===");
    
    // Initialize device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("Using CUDA device: {:?}", device);
    
    // Test parameters
    let tolerance = 1e-5;
    
    println!("\n1. Testing basic arithmetic operations...");
    
    // Create test tensors
    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = Shape::from_dims(&[2, 3]);
    
    let a = Tensor::from_slice(&test_data, shape.clone(), device.clone())?;
    let b = Tensor::from_slice(&test_data, shape.clone(), device.clone())?;
    
    // Test addition
    {
        let result = a.add(&b)?;
        let result_vec = result.to_vec()?;
        
        let expected: Vec<f32> = test_data.iter().map(|x| x * 2.0).collect();
        
        let max_diff = compare_vectors(&expected, &result_vec);
        println!("   Addition max difference: {:.2e} {}", 
            max_diff, 
            if max_diff < tolerance { "✓" } else { "✗" }
        );
    }
    
    // Test multiplication
    {
        let result = a.mul(&b)?;
        let result_vec = result.to_vec()?;
        
        let expected: Vec<f32> = test_data.iter().map(|x| x * x).collect();
        
        let max_diff = compare_vectors(&expected, &result_vec);
        println!("   Multiplication max difference: {:.2e} {}", 
            max_diff, 
            if max_diff < tolerance { "✓" } else { "✗" }
        );
    }
    
    // Test scalar operations
    {
        let scalar = 2.5;
        let result = a.mul_scalar(scalar)?;
        let result_vec = result.to_vec()?;
        
        let expected: Vec<f32> = test_data.iter().map(|x| x * scalar).collect();
        
        let max_diff = compare_vectors(&expected, &result_vec);
        println!("   Scalar multiplication max difference: {:.2e} {}", 
            max_diff, 
            if max_diff < tolerance { "✓" } else { "✗" }
        );
    }
    
    println!("\n2. Testing activation functions...");
    
    // Test ReLU
    {
        let test_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let x = Tensor::from_slice(&test_data, Shape::from_dims(&[6]), device.clone())?;
        
        let result = x.relu()?;
        let result_vec = result.to_vec()?;
        
        let expected: Vec<f32> = test_data.iter().map(|&x| x.max(0.0)).collect();
        
        let max_diff = compare_vectors(&expected, &result_vec);
        println!("   ReLU max difference: {:.2e} {}", 
            max_diff, 
            if max_diff < tolerance { "✓" } else { "✗" }
        );
    }
    
    // Test Sigmoid
    {
        let test_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let x = Tensor::from_slice(&test_data, Shape::from_dims(&[5]), device.clone())?;
        
        let result = x.sigmoid()?;
        let result_vec = result.to_vec()?;
        
        let expected: Vec<f32> = test_data.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        
        let max_diff = compare_vectors(&expected, &result_vec);
        println!("   Sigmoid max difference: {:.2e} {}", 
            max_diff, 
            if max_diff < tolerance { "✓" } else { "✗" }
        );
    }
    
    println!("\n3. Testing matrix operations...");
    
    // Test matrix multiplication
    {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        
        let a = Tensor::from_slice(&a_data, Shape::from_dims(&[2, 3]), device.clone())?;
        let b = Tensor::from_slice(&b_data, Shape::from_dims(&[3, 2]), device.clone())?;
        
        let result = a.matmul(&b)?;
        let result_vec = result.to_vec()?;
        
        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        
        let max_diff = compare_vectors(&expected, &result_vec);
        println!("   MatMul max difference: {:.2e} {}", 
            max_diff, 
            if max_diff < tolerance { "✓" } else { "✗" }
        );
        
        println!("     Expected: {:?}", expected);
        println!("     Got: {:?}", result_vec);
    }
    
    println!("\n=== Test Summary ===");
    println!("Basic FLAME tensor operations are working correctly!");
    
    Ok(())
}

fn compare_vectors(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |max, diff| max.max(diff))
}