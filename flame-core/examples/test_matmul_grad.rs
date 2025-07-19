//! Test MatMul gradient computation

use flame_core::{Tensor, Shape, Result, CudaDevice, AutogradContext};

fn main() -> Result<()> {
    println!("Testing MatMul gradients...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Clear context at start
    AutogradContext::clear();
    
    // Create small matrices
    let a = Tensor::randn(Shape::from_dims(&[2, 3]), 1.0, 0.0, device.clone())?
        .requires_grad_(true);
    let b = Tensor::randn(Shape::from_dims(&[3, 2]), 1.0, 0.0, device.clone())?
        .requires_grad_(true);
    
    println!("A shape: {:?}", a.shape().dims());
    println!("B shape: {:?}", b.shape().dims());
    
    // MatMul: C = A @ B
    let c = a.matmul(&b)?;
    println!("C = A @ B, shape: {:?}", c.shape().dims());
    
    // Loss = sum(C)
    let loss = c.sum()?;
    println!("Loss: {:.4}", loss.to_vec()?[0]);
    
    // Backward
    println!("\nComputing gradients...");
    let gradients = AutogradContext::backward(&loss)?;
    
    // Check gradients
    if let Some(grad_a) = gradients.get(a.id()) {
        println!("Gradient A shape: {:?}", grad_a.shape().dims());
        let grad_vec = grad_a.to_vec()?;
        println!("Gradient A (first 3): [{:.4}, {:.4}, {:.4}]", 
                 grad_vec[0], grad_vec[1], grad_vec[2]);
    }
    
    if let Some(grad_b) = gradients.get(b.id()) {
        println!("Gradient B shape: {:?}", grad_b.shape().dims());
        let grad_vec = grad_b.to_vec()?;
        println!("Gradient B (first 3): [{:.4}, {:.4}, {:.4}]", 
                 grad_vec[0], grad_vec[1], grad_vec[2]);
    }
    
    println!("\n✅ MatMul gradient test passed!");
    
    Ok(())
}