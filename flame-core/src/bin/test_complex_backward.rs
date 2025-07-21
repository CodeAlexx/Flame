use flame_core::{Tensor, Shape};
use cudarc::driver::CudaDevice;

fn main() -> flame_core::Result<()> {
    println!("Testing FLAME complex backward pass...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Multiple operations
    println!("Test 1: Chain of operations");
    let x = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    
    let z = x.add(&y)?;
    let w = z.mul(&x)?;
    let loss = w.sum()?;
    
    println!("Forward pass complete, calling backward...");
    match loss.backward() {
        Ok(grads) => {
            println!("✓ Backward succeeded! {} gradients", grads.len());
        }
        Err(e) => {
            println!("✗ Backward failed: {:?}", e);
        }
    }
    
    // Test 2: Matrix multiplication
    println!("\nTest 2: Matrix multiplication");
    let a = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let b = Tensor::randn(Shape::from_dims(&[4, 5]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    
    let c = a.matmul(&b)?;
    let loss2 = c.sum()?;
    
    println!("MatMul forward complete, calling backward...");
    match loss2.backward() {
        Ok(grads) => {
            println!("✓ MatMul backward succeeded! {} gradients", grads.len());
        }
        Err(e) => {
            println!("✗ MatMul backward failed: {:?}", e);
        }
    }
    
    // Test 3: Activation functions
    println!("\nTest 3: Activation functions");
    let input = Tensor::randn(Shape::from_dims(&[10]), -1.0, 1.0, device.clone())?.requires_grad_(true);
    
    let relu_out = input.relu()?;
    let loss3 = relu_out.sum()?;
    
    println!("ReLU forward complete, calling backward...");
    match loss3.backward() {
        Ok(grads) => {
            println!("✓ ReLU backward succeeded! {} gradients", grads.len());
        }
        Err(e) => {
            println!("✗ ReLU backward failed: {:?}", e);
        }
    }
    
    println!("\nAll tests complete!");
    Ok(())
}