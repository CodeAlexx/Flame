//! Test to find minimal case where autograd hangs

use flame_core::{Tensor, Shape, Result, conv::Conv2d};
use cudarc::driver::CudaDevice;

#[test] 
fn find_minimal_hanging_case() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("=== Finding minimal hanging case ===");
    
    // Test 1: Two operations
    println!("\nTest 1: Two operations (add + mul)");
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = x.add(&x)?;  // Op 1
    let z = y.mul(&y)?;  // Op 2
    
    println!("  Forward pass complete, testing backward...");
    match z.sum()?.backward_debug() {
        Ok(_) => println!("  ✓ 2-op graph works!"),
        Err(e) => {
            println!("  ✗ 2-op graph failed: {:?}", e);
            return Ok(());
        }
    }
    
    // Test 2: Three operations  
    println!("\nTest 2: Three operations (add + mul + add_scalar)");
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = x.add(&x)?;        // Op 1
    let z = y.mul(&y)?;        // Op 2  
    let w = z.add_scalar(1.0)?; // Op 3
    
    println!("  Forward pass complete, testing backward...");
    match w.sum()?.backward_debug() {
        Ok(_) => println!("  ✓ 3-op graph works!"),
        Err(e) => {
            println!("  ✗ 3-op graph failed: {:?}", e);
            return Ok(());
        }
    }
    
    // Test 3: Four operations with matmul
    println!("\nTest 3: Four operations with matmul");
    let x = Tensor::randn(Shape::from_dims(&[4, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = x.add(&x)?;      // Op 1
    let z = x.matmul(&y)?;   // Op 2
    let w = z.relu()?;       // Op 3
    let loss = w.sum()?;     // Op 4
    
    println!("  Forward pass complete, testing backward...");
    match loss.backward_debug() {
        Ok(_) => println!("  ✓ 4-op matmul graph works!"),
        Err(e) => {
            println!("  ✗ 4-op matmul graph failed: {:?}", e);
            return Ok(());
        }
    }
    
    // Test 4: Simple CNN operations
    println!("\nTest 4: Simple CNN forward pass");
    let x = Tensor::randn(Shape::from_dims(&[1, 1, 4, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let conv = Conv2d::new(1, 1, 2, 1, 0, device.clone())?;
    let y = conv.forward(&x)?;  // Op 1
    let z = y.sum()?;          // Op 2
    
    println!("  Forward pass complete, testing backward...");
    match z.backward_debug() {
        Ok(_) => println!("  ✓ Conv2d graph works!"),
        Err(e) => {
            println!("  ✗ Conv2d graph failed: {:?}", e);
            return Ok(());
        }
    }
    
    // Test 5: Multiple conv layers
    println!("\nTest 5: Two conv layers");
    let x = Tensor::randn(Shape::from_dims(&[1, 1, 8, 8]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let conv1 = Conv2d::new(1, 4, 3, 1, 1, device.clone())?;
    let conv2 = Conv2d::new(4, 1, 3, 1, 1, device.clone())?;
    
    let h1 = conv1.forward(&x)?;
    let h2 = h1.relu()?;
    let y = conv2.forward(&h2)?;
    let loss = y.sum()?;
    
    println!("  Forward pass complete, testing backward...");
    match loss.backward_debug() {
        Ok(_) => println!("  ✓ Two conv layer graph works!"),
        Err(e) => {
            println!("  ✗ Two conv layer graph failed: {:?}", e);
            return Ok(());
        }
    }
    
    println!("\n=== Test complete ===");
    Ok(())
}

#[test]
fn test_simple_add_backward() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("\n=== Testing simplest possible backward ===");
    let x = Tensor::randn(Shape::from_dims(&[2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = x.add_scalar(1.0)?;
    let loss = y.sum()?;
    
    println!("Forward complete, running backward_debug...");
    let grads = loss.backward_debug()?;
    
    println!("Backward complete!");
    println!("Got {} gradients", grads.len());
    
    Ok(())
}

#[test]
fn test_requires_grad_propagation() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("\n=== Testing requires_grad propagation ===");
    
    // Test that operations preserve requires_grad
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    println!("x.requires_grad = {}", x.requires_grad());
    
    let y = x.add(&x)?;
    println!("y.requires_grad = {}", y.requires_grad());
    
    let z = y.relu()?;
    println!("z.requires_grad = {}", z.requires_grad());
    
    assert!(x.requires_grad());
    assert!(y.requires_grad());
    assert!(z.requires_grad());
    
    println!("✓ requires_grad propagates correctly");
    
    Ok(())
}