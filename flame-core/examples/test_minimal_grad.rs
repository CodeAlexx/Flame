//! Minimal test to debug gradient computation

use flame_core::{Tensor, Shape, Result, CudaDevice, AutogradContext};

fn main() -> Result<()> {
    println!("Minimal gradient test...");
    
    // Clear any previous context
    AutogradContext::clear();
    
    let device = CudaDevice::new(0)?;
    
    // Create tiny tensors
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 1.0, 0.0, device.clone())?
        .requires_grad_(true);
    
    println!("Created x");
    
    // Simple operation: y = x * 2
    let y = x.mul_scalar(2.0)?;
    println!("Computed y = x * 2");
    
    // Sum to get scalar
    let loss = y.sum()?;
    let loss_val = loss.to_vec()?[0];
    println!("Loss: {}", loss_val);
    
    println!("About to call backward...");
    
    // Try backward
    let gradients = AutogradContext::backward(&loss)?;
    
    println!("Backward complete!");
    
    if let Some(x_grad) = gradients.get(x.id()) {
        println!("x gradient: {:?}", x_grad.to_vec()?);
    }
    
    Ok(())
}