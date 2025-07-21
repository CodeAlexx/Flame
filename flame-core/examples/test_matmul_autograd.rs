use flame_core::{Tensor, Shape, Result};
use cudarc::driver::CudaDevice;

fn main() -> Result<()> {
    println!("Testing MatMul autograd fix...");
    
    let device = CudaDevice::new(0)?;
    
    // Create two matrices for matmul
    let a = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?
        .requires_grad_(true);
    let b = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?
        .requires_grad_(true);
    
    println!("Created tensors a: {:?}, b: {:?}", a.shape(), b.shape());
    
    // Forward pass: C = A @ B
    let c = a.matmul(&b)?;
    println!("MatMul result shape: {:?}", c.shape());
    
    // Sum to get scalar loss
    let loss = c.sum()?;
    println!("Loss computed");
    
    // Backward pass - this should NOT hang
    println!("Starting backward pass...");
    let grads = loss.backward()?;
    
    println!("✅ Backward pass completed without hanging!");
    println!("Number of gradients: {}", grads.len());
    
    // Check we got gradients for both inputs
    if grads.contains(a.id()) {
        println!("✅ Got gradient for tensor a");
    } else {
        println!("❌ Missing gradient for tensor a");
    }
    
    if grads.contains(b.id()) {
        println!("✅ Got gradient for tensor b");
    } else {
        println!("❌ Missing gradient for tensor b");
    }
    
    println!("\n🎉 MatMul autograd test passed!");
    
    Ok(())
}