//! Find exact hanging point

use flame_core::{Tensor, Shape, Result};
use cudarc::driver::CudaDevice;

#[test]
fn test_add_operation_forward() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Creating tensors...");
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let x_data = x.to_vec()?;
    println!("x data: {:?}", x_data);
    
    println!("Performing add operation...");
    let y = x.add(&x)?;
    println!("Add complete!");
    
    let y_data = y.to_vec()?;
    println!("y data: {:?}", y_data);
    
    // Verify the add worked
    for i in 0..4 {
        assert!((y_data[i] - 2.0 * x_data[i]).abs() < 1e-6);
    }
    
    println!("Forward pass verified!");
    Ok(())
}

#[test]
fn test_add_then_sum_backward() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Step 1: Create tensor");
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    
    println!("Step 2: Add operation");
    let y = x.add(&x)?;
    
    println!("Step 3: Sum to scalar");
    let loss = y.sum()?;
    
    println!("Step 4: Start backward");
    println!("  Loss value: {}", loss.to_vec()?[0]);
    
    // Use regular backward first to see if it hangs
    println!("Step 5: Calling backward (not debug)...");
    let _grads = loss.backward()?;
    
    println!("Backward complete!");
    Ok(())
}

#[test]
fn test_mul_after_add() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Creating tensor x...");
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    
    println!("Operation 1: y = x + x");
    let y = x.add(&x)?;
    println!("  y created, shape: {:?}", y.shape());
    
    println!("Operation 2: z = y * y");
    println!("  About to call mul...");
    let z = y.mul(&y)?;
    println!("  z created, shape: {:?}", z.shape());
    
    println!("All forward operations complete!");
    Ok(())
}