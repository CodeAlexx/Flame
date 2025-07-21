use flame_core::{Tensor, Shape};
use cudarc::driver::CudaDevice;

fn main() -> flame_core::Result<()> {
    println!("Testing gradient propagation...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Create tensors
    let x = Tensor::ones(Shape::from_dims(&[2, 2]), device.clone())?.requires_grad_(true);
    let y = Tensor::ones(Shape::from_dims(&[2, 2]), device.clone())?.requires_grad_(true);
    
    println!("x requires_grad: {}", x.requires_grad());
    println!("y requires_grad: {}", y.requires_grad());
    
    // Test operations
    let z = x.add(&y)?;
    println!("z = x + y, requires_grad: {}", z.requires_grad());
    
    let w = z.mul(&x)?;
    println!("w = z * x, requires_grad: {}", w.requires_grad());
    
    let loss = w.sum()?;
    println!("loss = sum(w), requires_grad: {}", loss.requires_grad());
    println!("loss shape: {:?}", loss.shape().dims());
    
    // Clear any previous tape
    flame_core::AutogradContext::clear();
    
    // Now test backward on a fresh computation
    println!("\nCreating fresh computation graph...");
    let a = Tensor::ones(Shape::from_dims(&[1]), device.clone())?.requires_grad_(true);
    let b = a.add_scalar(1.0)?;
    let c = b.mul_scalar(2.0)?;
    
    println!("a -> b -> c chain created");
    println!("c shape: {:?}, requires_grad: {}", c.shape().dims(), c.requires_grad());
    
    println!("Calling backward...");
    match c.backward() {
        Ok(grads) => {
            println!("✓ Backward succeeded! {} gradients", grads.len());
            for (id, grad) in grads.iter() {
                println!("  Gradient for tensor {:?}: shape {:?}", id, grad.shape().dims());
            }
        }
        Err(e) => {
            println!("✗ Backward failed: {:?}", e);
        }
    }
    
    Ok(())
}