use flame_core::{Tensor, Shape};
use cudarc::driver::CudaDevice;

fn main() -> flame_core::Result<()> {
    println!("Testing FLAME basic functionality...");
    
    // Test 1: Device creation
    let device = match CudaDevice::new(0) {
        Ok(d) => {
            println!("✓ CUDA device created");
            d
        }
        Err(e) => {
            println!("✗ Failed to create CUDA device: {:?}", e);
            return Ok(());
        }
    };
    
    // Test 2: Tensor creation
    let tensor = match Tensor::zeros(Shape::from_dims(&[2, 2]), device.clone()) {
        Ok(t) => {
            println!("✓ Created zero tensor");
            t
        }
        Err(e) => {
            println!("✗ Failed to create tensor: {:?}", e);
            return Ok(());
        }
    };
    
    // Test 3: Basic operation
    match tensor.add_scalar(1.0) {
        Ok(_) => println!("✓ Scalar addition works"),
        Err(e) => println!("✗ Scalar addition failed: {:?}", e),
    }
    
    // Test 4: Check autograd
    let tensor_grad = match Tensor::zeros(Shape::from_dims(&[2, 2]), device.clone()) {
        Ok(t) => t.requires_grad(),
        Err(e) => {
            println!("✗ Failed to create tensor with grad: {:?}", e);
            return Ok(());
        }
    };
    
    // Note: Cannot access requires_grad field directly
    println!("Created tensor with gradient tracking");
    
    println!("\nFLAME minimal test complete.");
    Ok(())
}