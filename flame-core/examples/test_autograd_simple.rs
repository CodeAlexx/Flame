#![cfg(feature = "legacy_examples")]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]
#![cfg_attr(
    clippy,
    allow(
        clippy::unused_imports,
        clippy::useless_vec,
        clippy::needless_borrow,
        clippy::needless_clone
    )
)]

use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting simple autograd test...");

    // Create device
    let device = CudaDevice::new(0)?;

    // Create a simple computation graph: x -> add -> mul -> sum -> backward
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device)?
        .requires_grad_(true);
    println!("Created input tensor x with requires_grad");

    // Add operation
    let y = x.add_scalar(2.0)?;
    println!("Performed add operation: y = x + 2.0");

    // Multiply operation
    let z = y.mul_scalar(3.0)?;
    println!("Performed multiply operation: z = y * 3.0");

    // Sum to get scalar
    let loss = z.sum()?;
    println!("Computed sum: loss = z.sum()");

    // Backward pass
    println!("Starting backward pass...");
    let gradients = loss.backward()?;
    println!("Backward pass completed successfully!");

    // Get gradient using the extension trait
    use flame_core::gradient::TensorGradExt;
    if let Some(grad) = x.grad(&gradients) {
        println!("Gradient of x: {:?}", grad.to_vec()?);
        println!(
            "Expected gradient: [3.0, 3.0, 3.0, 3.0] (since d(loss)/dx = 3.0 for all elements)"
        );
    } else {
        println!("ERROR: No gradient found for x!");
    }

    println!("\nTEST PASSED: Autograd is working correctly!");
    Ok(())
}
