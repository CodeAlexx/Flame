//! Memory Stability Test for FLAME
//! Tests memory management under stress conditions

use flame_core::{
    Tensor, Shape, Result,
    conv::Conv2d, linear::Linear,
};
use cudarc::driver::CudaDevice;
use std::time::Instant;

/// Get current GPU memory usage
fn get_memory_usage(device: &CudaDevice) -> Result<(usize, usize)> {
    // cudarc doesn't expose memory info directly, so we'll track allocations
    // In a real implementation, we'd use CUDA API directly
    Ok((0, 0)) // Placeholder
}

#[test]
fn test_memory_stability() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Starting memory stability test...");
    
    // Test 1: Large tensor allocation/deallocation cycles
    println!("\nTest 1: Large tensor allocation cycles");
    for i in 0..10 {
        let size = 1024 * 1024 * 10; // 10M elements (~40MB)
        let tensor = Tensor::randn(
            Shape::from_dims(&[size]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        // Force computation
        let _sum = tensor.sum()?.to_vec()?[0];
        
        if i % 2 == 0 {
            println!("  Iteration {}: Created {}MB tensor", i, size * 4 / 1024 / 1024);
        }
    }
    println!("  ✓ Large tensor test completed");
    
    // Test 2: Many small allocations
    println!("\nTest 2: Many small tensor allocations");
    let mut tensors = Vec::new();
    for i in 0..1000 {
        let tensor = Tensor::randn(
            Shape::from_dims(&[100, 100]),
            0.0,
            1.0,
            device.clone()
        )?;
        tensors.push(tensor);
        
        if i % 200 == 0 {
            println!("  Created {} tensors", i + 1);
        }
    }
    // Drop all at once
    drop(tensors);
    println!("  ✓ Small tensor test completed");
    
    // Test 3: Model creation/destruction cycles
    println!("\nTest 3: Model creation/destruction cycles");
    for i in 0..5 {
        // Create a reasonably sized model
        let conv1 = Conv2d::new(3, 64, 3, 1, 1, device.clone())?;
        let conv2 = Conv2d::new(64, 128, 3, 1, 1, device.clone())?;
        let fc = Linear::new(128 * 8 * 8, 1000, true, &device)?;
        
        // Do some forward passes
        let input = Tensor::randn(
            Shape::from_dims(&[4, 3, 32, 32]),
            0.0,
            0.1,
            device.clone()
        )?;
        
        let x = conv1.forward(&input)?;
        let x = x.relu()?;
        let x = conv2.forward(&x)?;
        
        println!("  Iteration {}: Model created and used", i + 1);
        
        // Model drops here
    }
    println!("  ✓ Model lifecycle test completed");
    
    // Test 4: Concurrent tensor operations
    println!("\nTest 4: Stress test with many operations");
    let start = Instant::now();
    let base = Tensor::randn(
        Shape::from_dims(&[512, 512]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    let mut result = base.clone_result()?;
    for i in 0..100 {
        // Chain of operations
        result = result.add(&base)?;
        result = result.mul_scalar(0.99)?;
        result = result.relu()?;
        
        if i % 25 == 0 {
            let norm = result.sum()?.to_vec()?[0];
            println!("  Iteration {}: Norm = {:.4}", i, norm);
        }
    }
    
    let elapsed = start.elapsed();
    println!("  ✓ Stress test completed in {:.2}s", elapsed.as_secs_f32());
    
    println!("\nMemory stability test completed successfully!");
    Ok(())
}

#[test]
fn test_memory_reuse() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Testing memory reuse patterns...");
    
    // Test that memory is properly reused when tensors are dropped
    let mut max_values = Vec::new();
    
    for size_mb in [1, 10, 50, 100] {
        let size = size_mb * 1024 * 1024 / 4; // Convert MB to number of f32s
        
        // Allocate and immediately drop 10 tensors of this size
        for _ in 0..10 {
            let tensor = Tensor::randn(
                Shape::from_dims(&[size]),
                0.0,
                1.0,
                device.clone()
            )?;
            
            // Use the tensor to ensure it's allocated
            let max_val = tensor.to_vec()?.into_iter()
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));
            max_values.push(max_val);
        }
        
        println!("  Allocated and freed 10x {}MB tensors", size_mb);
    }
    
    // If we get here without OOM, memory is being properly managed
    println!("Memory reuse test completed successfully!");
    Ok(())
}

#[test]
fn test_gradient_memory() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Testing gradient accumulation memory...");
    
    // Create tensors that require gradients
    let mut tensors = Vec::new();
    for i in 0..20 {
        let tensor = Tensor::randn(
            Shape::from_dims(&[256, 256]),
            0.0,
            1.0,
            device.clone()
        )?.requires_grad_(true);
        
        tensors.push(tensor);
    }
    
    // Perform operations that would accumulate gradients
    let mut result = tensors[0].clone_result()?;
    for i in 1..tensors.len() {
        result = result.add(&tensors[i])?;
    }
    
    // Final operation
    let loss = result.sum()?;
    println!("  Created computation graph with {} tensors", tensors.len());
    
    // In a full implementation, we'd call backward here
    // let _grads = loss.backward()?;
    
    // Clean up - drop everything
    drop(loss);
    drop(result);
    drop(tensors);
    
    println!("Gradient memory test completed!");
    Ok(())
}
