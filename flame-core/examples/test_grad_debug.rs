use flame_core::{Tensor, Shape, autograd::AutogradContext, gradient::TensorGradExt};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() -> flame_core::Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Create input tensor with requires_grad
    let x = Tensor::from_vec(vec![2.0, 3.0], Shape::from_dims(&[2]), device.clone())?;
    println!("x.requires_grad before: {}", x.requires_grad());
    
    let x = x.requires_grad_(true);
    println!("x.requires_grad after: {}", x.requires_grad());
    println!("x.id: {:?}", x.id());
    
    // Multiply by itself
    let y = x.mul(&x)?;
    println!("y.requires_grad: {}", y.requires_grad());
    println!("y.id: {:?}", y.id());
    
    // Sum
    let loss = y.sum()?;
    println!("loss.requires_grad: {}", loss.requires_grad());
    println!("loss.id: {:?}", loss.id());
    
    // Backward
    let grads = AutogradContext::backward(&loss)?;
    println!("Number of gradients: {}", grads.len());
    
    // Check gradients
    if let Some(x_grad) = x.grad(&grads) {
        println!("x gradient found: {:?}", x_grad.to_vec()?);
    } else {
        println!("x gradient NOT found for id {:?}", x.id());
        println!("Available gradient ids:");
        for (id, _) in grads.iter() {
            println!("  {:?}", id);
        }
    }
    
    Ok(())
}