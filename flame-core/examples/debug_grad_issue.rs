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

use flame_core::{autograd::AutogradContext, CudaDevice, Result, Shape, Tensor, TensorGradExt};

fn main() -> Result<()> {
    println!("🔍 Debugging gradient issue...\n");

    let device = CudaDevice::new(0)?;

    // Reset autograd context
    AutogradContext::reset();

    // Create simple tensor
    let x = Tensor::from_vec(vec![2.0, 3.0], Shape::from_dims(&[2]), device.clone())?
        .requires_grad_(true);

    println!("x.requires_grad: {}", x.requires_grad());
    println!("x.id: {:?}", x.id());

    // Simple operation
    let y = x.mul_scalar(2.0)?;
    println!("y.requires_grad: {}", y.requires_grad());
    println!("y.id: {:?}", y.id());

    // Sum to get scalar
    let loss = y.sum()?;
    println!("loss.requires_grad: {}", loss.requires_grad());
    println!("loss.id: {:?}", loss.id());

    // Check autograd state before backward
    println!("\nChecking autograd state:");
    // AutogradContext::set_enabled(true); // Make sure it's enabled

    // Compute gradients
    println!("\nComputing gradients...");
    let grads = AutogradContext::backward(&loss)?;

    println!("Number of gradients: {}", grads.len());
    println!("Gradient IDs:");
    for (id, grad) in grads.iter() {
        println!("  {:?} -> shape: {:?}", id, grad.shape());
    }

    // Check for x gradient
    if let Some(x_grad) = x.grad(&grads) {
        println!("\nx gradient found: {:?}", x_grad.to_vec()?);
    } else {
        println!("\n❌ x gradient NOT found!");
    }

    Ok(())
}
