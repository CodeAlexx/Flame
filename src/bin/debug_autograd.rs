use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};
use std::time::Instant;

fn main() -> flame_core::Result<()> {
    println!("=== Debugging Autograd Hang ===\n");

    let device = CudaDevice::new(0)?;

    // Test 1: Simple backward pass
    println!("Test 1: Simple scalar operation");
    let x = Tensor::randn(Shape::from_dims(&[1]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let y = x.add_scalar(1.0)?;

    println!("  Created tensors, starting backward...");
    let start = Instant::now();

    // Add timeout
    let handle = std::thread::spawn(move || match y.backward() {
        Ok(grads) => {
            println!("  ✓ Backward completed in {:?}", start.elapsed());
            println!("  Gradients: {} entries", grads.len());
        }
        Err(e) => {
            println!("  ✗ Backward failed: {:?}", e);
        }
    });

    // Simple wait since join_timeout doesn't exist in std
    std::thread::sleep(std::time::Duration::from_secs(2));
    println!("  Waiting for backward to complete...");
    if let Err(e) = handle.join() {
        println!("  ✗ Backward thread panicked: {:?}", e);
    }

    // Test 2: More complex operation
    println!("\nTest 2: Matrix multiplication");
    let a =
        Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let b =
        Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);

    let c = a.matmul(&b)?;
    let loss = c.sum()?;

    println!("  Created computation graph, starting backward...");
    let start = Instant::now();

    match loss.backward() {
        Ok(grads) => {
            println!("  ✓ Backward completed in {:?}", start.elapsed());
            println!("  Gradients: {} entries", grads.len());
        }
        Err(e) => {
            println!("  ✗ Backward failed: {:?}", e);
        }
    }

    Ok(())
}
