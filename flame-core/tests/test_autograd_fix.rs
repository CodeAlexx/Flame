//! Test autograd fix for hanging issue

use flame_core::{Tensor, Shape, Result};
use cudarc::driver::CudaDevice;

#[test]
fn test_autograd_fix_two_ops() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("=== Testing autograd fix ===");
    
    // This was the minimal hanging case: x -> add -> mul -> sum -> backward
    let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    println!("Created x with id: {:?}", x.id());
    
    let y = x.add(&x)?;
    println!("Created y = x + x with id: {:?}", y.id());
    
    let z = y.mul(&y)?;
    println!("Created z = y * y with id: {:?}", z.id());
    
    let loss = z.sum()?;
    println!("Created loss = sum(z) with id: {:?}", loss.id());
    
    println!("\nStarting backward pass...");
    let grads = loss.backward_debug()?;
    
    println!("\n✓ SUCCESS! Backward pass completed without hanging!");
    println!("Got {} gradients", grads.len());
    
    // Verify we got gradient for x
    if let Some(x_grad) = grads.get(x.id()) {
        println!("x gradient shape: {:?}", x_grad.shape());
        
        // Expected gradient: d/dx of sum((x+x)*(x+x)) = 8x
        let x_data = x.to_vec()?;
        let grad_data = x_grad.to_vec()?;
        
        println!("\nGradient check:");
        for i in 0..4 {
            let expected = 8.0 * x_data[i];
            let actual = grad_data[i];
            let diff = (expected - actual).abs();
            println!("  Element {}: expected {:.4}, got {:.4}, diff {:.6}", 
                     i, expected, actual, diff);
            assert!(diff < 1e-4, "Gradient mismatch at element {}", i);
        }
    } else {
        panic!("No gradient found for input tensor x!");
    }
    
    println!("\n✓ All gradient checks passed!");
    Ok(())
}

#[test]
fn test_autograd_fix_complex() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("\n=== Testing more complex autograd graph ===");
    
    let x = Tensor::randn(Shape::from_dims(&[4, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let w = Tensor::randn(Shape::from_dims(&[4, 4]), 0.0, 0.1, device.clone())?.requires_grad_(true);
    
    // Complex computation: (x @ w + x) * 2.0
    let y = x.matmul(&w)?;
    let z = y.add(&x)?;
    let out = z.mul_scalar(2.0)?;
    let loss = out.sum()?;
    
    println!("Forward pass complete, running backward...");
    let grads = loss.backward()?;
    
    println!("✓ Complex graph backward completed!");
    println!("Got gradients for {} tensors", grads.len());
    
    assert!(grads.contains(x.id()), "Missing gradient for x");
    assert!(grads.contains(w.id()), "Missing gradient for w");
    
    Ok(())
}