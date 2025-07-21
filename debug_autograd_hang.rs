#!/usr/bin/env rust-script
//! Simple script to debug autograd hanging
//! 
//! ```cargo
//! [dependencies]
//! flame-core = { path = "flame-core" }
//! cudarc = "0.11"
//! ```

use flame_core::{Tensor, Shape, Result};
use cudarc::driver::CudaDevice;

fn main() -> Result<()> {
    println!("=== Debugging Autograd Hanging ===");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Single operation backward (should work)
    println!("\nTest 1: Single operation backward");
    {
        let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
        let y = x.add_scalar(2.0)?;
        let loss = y.sum()?;
        
        println!("  Forward pass complete");
        println!("  Running backward...");
        let grads = loss.backward()?;
        println!("  ✓ Success! Got {} gradients", grads.len());
    }
    
    // Test 2: Two operations backward
    println!("\nTest 2: Two operations backward (add -> mul)");
    {
        let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
        println!("  x tensor created, id: {:?}", x.id());
        
        let y = x.add(&x)?;
        println!("  y = x + x created, id: {:?}", y.id());
        
        let z = y.mul(&y)?;
        println!("  z = y * y created, id: {:?}", z.id());
        
        let loss = z.sum()?;
        println!("  loss = sum(z) created, id: {:?}", loss.id());
        
        println!("  Forward pass complete");
        println!("  Running backward_debug...");
        
        // This is where it hangs
        let grads = loss.backward_debug()?;
        println!("  ✓ Success! Got {} gradients", grads.len());
    }
    
    println!("\n=== All tests complete ===");
    Ok(())
}