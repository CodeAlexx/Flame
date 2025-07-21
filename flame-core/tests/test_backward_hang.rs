//! Test exact backward hanging case

use flame_core::{Tensor, Shape, Result};
use cudarc::driver::CudaDevice;

#[test]
fn test_mul_backward_alone() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Testing mul backward alone...");
    let a = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let b = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    
    let c = a.mul(&b)?;
    let loss = c.sum()?;
    
    println!("Running backward...");
    let grads = loss.backward()?;
    
    println!("Success! Got {} gradients", grads.len());
    Ok(())
}

#[test]
fn test_add_mul_backward_exact() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Creating x...");
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    println!("x id: {:?}", x.id());
    
    println!("Step 1: y = x + x");
    let y = x.add(&x)?;
    println!("y id: {:?}", y.id());
    
    println!("Step 2: z = y * y");
    let z = y.mul(&y)?;
    println!("z id: {:?}", z.id());
    
    println!("Step 3: loss = sum(z)");
    let loss = z.sum()?;
    println!("loss id: {:?}", loss.id());
    
    println!("Forward complete. Starting backward_debug...");
    let grads = loss.backward_debug()?;
    
    println!("Backward complete! Got {} gradients", grads.len());
    Ok(())
}

#[test]
fn test_check_saved_tensors() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Enable some internal debugging if possible
    println!("Testing saved tensors in operations...");
    
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    
    // First op
    println!("\nFirst op: add");
    let y = x.add(&x)?;
    
    // Second op - this might be where saved tensors get messed up
    println!("\nSecond op: mul");
    let z = y.mul(&y)?;
    
    println!("\nAll operations recorded. Checking backward...");
    let loss = z.sum()?;
    
    // Try backward
    println!("\nStarting backward...");
    let _grads = loss.backward_debug()?;
    
    Ok(())
}