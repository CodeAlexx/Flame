//! Minimal test to find where autograd hangs

use flame_core::{Tensor, Shape, Result};
use cudarc::driver::CudaDevice;

#[test]
fn test_two_ops_only() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Creating tensor...");
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    
    println!("First operation: add...");
    let y = x.add(&x)?;
    println!("  y shape: {:?}, requires_grad: {}", y.shape(), y.requires_grad());
    
    println!("Second operation: mul...");
    let z = y.mul(&y)?;
    println!("  z shape: {:?}, requires_grad: {}", z.shape(), z.requires_grad());
    
    println!("Sum to scalar...");
    let loss = z.sum()?;
    println!("  loss shape: {:?}, requires_grad: {}", loss.shape(), loss.requires_grad());
    
    println!("Starting backward_debug...");
    let grads = loss.backward_debug()?;
    
    println!("Backward complete! Got {} gradients", grads.len());
    Ok(())
}

#[test] 
fn test_single_op_only() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Testing single operation...");
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = x.mul_scalar(2.0)?;
    let loss = y.sum()?;
    
    println!("Running backward...");
    let grads = loss.backward_debug()?;
    
    println!("Success! Got {} gradients", grads.len());
    Ok(())
}