use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};

fn main() -> flame_core::Result<()> {
    println!("Testing FLAME backward pass...\n");

    let device = CudaDevice::new(0)?;

    // Test simple operation
    println!("Creating tensors...");
    let x = Tensor::ones(Shape::from_dims(&[1]), device.clone())?.requires_grad_(true);
    println!("x created, requires_grad: {}", x.requires_grad());

    println!("Computing y = x + 1...");
    let y = x.add_scalar(1.0)?;
    println!(
        "y shape: {:?}, requires_grad: {}",
        y.shape().dims(),
        y.requires_grad()
    );

    println!("Calling backward...");
    match y.backward() {
        Ok(grads) => {
            println!("✓ Backward succeeded!");
            println!("Number of gradients: {}", grads.len());
        }
        Err(e) => {
            println!("✗ Backward failed: {:?}", e);
        }
    }

    println!("\nTest complete.");
    Ok(())
}
