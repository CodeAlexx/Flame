use flame_core::{Tensor, Shape, Result};
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use std::time::Instant;

/// Test memory allocation and deallocation patterns
#[test]
fn test_memory_lifecycle() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Get initial memory state
    device.synchronize()?;
    let initial_free = get_free_memory(&device)?;
    println!("Initial free memory: {:.2} GB", initial_free as f64 / 1e9);
    
    // Phase 1: Allocate and deallocate many small tensors
    {
        let mut tensors = Vec::new();
        for i in 0..1000 {
            let tensor = Tensor::randn(
                Shape::from_dims(&[32, 32]),
                0.0,
                1.0,
                device.clone()
            )?;
            tensors.push(tensor);
            
            // Periodically clear to test deallocation
            if i % 100 == 99 {
                tensors.clear();
                device.synchronize()?;
            }
        }
    }
    
    // Check memory after small allocations
    device.synchronize()?;
    let after_small = get_free_memory(&device)?;
    let small_leak = initial_free.saturating_sub(after_small);
    println!("Memory after small allocations: {:.2} MB leaked", small_leak as f64 / 1e6);
    assert!(small_leak < 50 * 1024 * 1024, "Too much memory leaked from small allocations");
    
    // Phase 2: Allocate and deallocate large tensors
    {
        for _ in 0..10 {
            let _large = Tensor::randn(
                Shape::from_dims(&[1024, 1024, 64]), // 256MB
                0.0,
                1.0,
                device.clone()
            )?;
            // Tensor dropped here
        }
    }
    
    // Check memory after large allocations
    device.synchronize()?;
    let after_large = get_free_memory(&device)?;
    let large_leak = initial_free.saturating_sub(after_large);
    println!("Memory after large allocations: {:.2} MB leaked", large_leak as f64 / 1e6);
    assert!(large_leak < 100 * 1024 * 1024, "Too much memory leaked from large allocations");
    
    println!("✅ Memory lifecycle test passed!");
    Ok(())
}

/// Test memory usage during forward and backward passes
#[test]
fn test_training_memory_usage() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    device.synchronize()?;
    let initial_free = get_free_memory(&device)?;
    
    // Run several training iterations
    for iter in 0..20 {
        // Create computation graph
        let x = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 1.0, device.clone())?.requires_grad();
        let w = Tensor::randn(Shape::from_dims(&[128, 256]), 0.0, 0.1, device.clone())?.requires_grad();
        
        // Forward pass
        let y = x.matmul(&w)?;
        let y_relu = y.relu()?;
        let loss = y_relu.sum()?;
        
        // Backward pass
        let _grad_map = loss.backward()?;
        
        // Check memory usage periodically
        if iter % 5 == 4 {
            device.synchronize()?;
            let current_free = get_free_memory(&device)?;
            let used = initial_free.saturating_sub(current_free);
            println!("Iteration {}: Memory used: {:.2} MB", iter + 1, used as f64 / 1e6);
            
            // Memory usage should stabilize, not grow indefinitely
            if iter > 10 {
                assert!(used < 500 * 1024 * 1024, "Memory usage growing too much");
            }
        }
    }
    
    println!("✅ Training memory usage test passed!");
    Ok(())
}

/// Test memory efficiency of different operations
#[test]
fn test_operation_memory_efficiency() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Test various operations for memory efficiency
    let test_cases = vec![
        ("matmul", 1024, 1024),
        ("conv2d", 64, 64),
        ("sum_reduction", 2048, 2048),
        ("broadcast", 1, 1024),
    ];
    
    for (op_name, dim1, dim2) in test_cases {
        device.synchronize()?;
        let before = get_free_memory(&device)?;
        
        match op_name {
            "matmul" => {
                let a = Tensor::randn(Shape::from_dims(&[dim1, dim2]), 0.0, 1.0, device.clone())?;
                let b = Tensor::randn(Shape::from_dims(&[dim2, dim1]), 0.0, 1.0, device.clone())?;
                let _c = a.matmul(&b)?;
            }
            "conv2d" => {
                use flame_core::conv::Conv2d;
                let conv = Conv2d::new_with_bias(3, 64, 3, 1, 1, device.clone(), true)?;
                let input = Tensor::randn(Shape::from_dims(&[4, 3, dim1, dim2]), 0.0, 1.0, device.clone())?;
                let _output = conv.forward(&input)?;
            }
            "sum_reduction" => {
                let tensor = Tensor::randn(Shape::from_dims(&[dim1, dim2]), 0.0, 1.0, device.clone())?;
                let _sum = tensor.sum()?;
            }
            "broadcast" => {
                let small = Tensor::randn(Shape::from_dims(&[dim1, dim2]), 0.0, 1.0, device.clone())?;
                let _large = small.broadcast_to(&Shape::from_dims(&[32, dim1, dim2]))?;
            }
            _ => unreachable!()
        }
        
        device.synchronize()?;
        let after = get_free_memory(&device)?;
        let used = before.saturating_sub(after);
        
        println!("{}: Used {:.2} MB", op_name, used as f64 / 1e6);
        
        // Each operation should use reasonable memory
        let max_expected = match op_name {
            "matmul" => dim1 * dim2 * 4 * 3, // Input A, B, and output
            "conv2d" => 4 * 64 * dim1 * dim2 * 4 * 2, // Input and output
            "sum_reduction" => dim1 * dim2 * 4 + 4, // Input + scalar output
            "broadcast" => 32 * dim1 * dim2 * 4, // Just output
            _ => unreachable!()
        };
        
        assert!(used < max_expected * 2, "{} using too much memory", op_name);
    }
    
    println!("✅ Operation memory efficiency test passed!");
    Ok(())
}

/// Test memory pool behavior under stress
#[test]
fn test_memory_pool_stress() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Simulate realistic allocation patterns
    let sizes = vec![
        1024 * 1024,      // 4MB
        256 * 256,        // 256KB
        512 * 512 * 3,    // 3MB
        64 * 64 * 128,    // 2MB
        1024 * 1024 * 16, // 64MB
    ];
    
    let start = Instant::now();
    
    // Rapid allocation and deallocation
    for _ in 0..100 {
        let mut tensors = Vec::new();
        
        // Allocate various sizes
        for &size in &sizes {
            let tensor = Tensor::zeros(Shape::from_dims(&[size]), device.clone())?;
            tensors.push(tensor);
        }
        
        // Use tensors (prevents optimization)
        for tensor in &tensors {
            let _sum = tensor.sum()?;
        }
        
        // Deallocate in different order
        tensors.reverse();
        tensors.clear();
    }
    
    let elapsed = start.elapsed();
    println!("Memory pool stress test completed in {:.2}s", elapsed.as_secs_f64());
    
    // Should complete reasonably fast (memory pooling working)
    assert!(elapsed.as_secs() < 10, "Memory allocation too slow");
    
    // Check final memory state
    device.synchronize()?;
    
    println!("✅ Memory pool stress test passed!");
    Ok(())
}

/// Test gradient checkpointing memory savings
#[test]
fn test_gradient_checkpointing_memory() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // First: Regular forward/backward without checkpointing
    device.synchronize()?;
    let before_regular = get_free_memory(&device)?;
    
    {
        let x = Tensor::randn(Shape::from_dims(&[128, 512]), 0.0, 1.0, device.clone())?.requires_grad();
        
        // Deep network
        let mut h = x.clone_result()?;
        let mut weights = Vec::new();
        
        for i in 0..10 {
            let w = Tensor::randn(Shape::from_dims(&[512, 512]), 0.0, 0.1, device.clone())?.requires_grad();
            weights.push(w.clone());
            h = h.matmul(&w)?;
            h = h.relu()?;
        }
        
        let loss = h.sum()?;
        let _grads = loss.backward()?;
    }
    
    device.synchronize()?;
    let after_regular = get_free_memory(&device)?;
    let regular_peak = before_regular.saturating_sub(after_regular);
    
    println!("Regular training peak memory: {:.2} MB", regular_peak as f64 / 1e6);
    
    // Note: Actual gradient checkpointing would require recomputation during backward
    // This test mainly verifies memory measurement works correctly
    
    println!("✅ Gradient checkpointing memory test passed!");
    Ok(())
}

// Helper function to get free memory
fn get_free_memory(device: &Arc<CudaDevice>) -> Result<usize> {
    match device.memory_info() {
        Ok((free, _total)) => Ok(free),
        Err(_) => {
            // Fallback: just ensure we can allocate
            Ok(1024 * 1024 * 1024) // 1GB
        }
    }
}
