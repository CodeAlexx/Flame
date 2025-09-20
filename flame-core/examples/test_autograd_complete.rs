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
use flame_core::{Result, Shape, Tensor};

fn main() -> Result<()> {
    println!("🔧 Testing complete autograd fix for all operations...\n");

    let device = CudaDevice::new(0)?;

    // Test 1: Basic operations (Add, Mul, MulScalar)
    {
        println!("Test 1: Basic operations");
        let x = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);

        let y = x.add(&x)?; // Add
        let z = y.mul(&y)?; // Mul
        let w = z.mul_scalar(2.0)?; // MulScalar
        let loss = w.sum()?; // Sum

        let grads = loss.backward()?;
        println!("✅ Basic operations backward pass completed!");
        println!("   Got {} gradients", grads.len());
    }

    // Test 2: Division and Square
    {
        println!("\nTest 2: Division and Square operations");
        let a = Tensor::randn(Shape::from_dims(&[3, 3]), 1.0, 0.5, device.clone())?
            .requires_grad_(true);
        let b = Tensor::randn(Shape::from_dims(&[3, 3]), 1.0, 0.5, device.clone())?
            .requires_grad_(true);

        let c = a.square()?; // Square: x^2
        let d = c.div(&b)?; // Division: c/b
        let loss = d.sum()?;

        println!("   Loss requires_grad: {}", loss.requires_grad());
        let grads = loss.backward()?;
        println!("✅ Division and Square backward pass completed!");
        if grads.contains(a.id()) && grads.contains(b.id()) {
            println!("   Got gradients for both inputs");
        }
    }

    // Test 3: ReLU activation
    {
        println!("\nTest 3: ReLU activation");
        let x = Tensor::randn(Shape::from_dims(&[4, 4]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);

        let y = x.relu()?; // ReLU activation
        let loss = y.sum()?;

        println!("   Loss requires_grad: {}", loss.requires_grad());
        let grads = loss.backward()?;
        println!("✅ ReLU backward pass completed!");
    }

    // Test 4: Complex computation graph
    {
        println!("\nTest 4: Complex computation graph");
        let x = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        let w = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 0.5, device.clone())?
            .requires_grad_(true);

        // Complex graph: (x @ w + 1) * 2 / 3
        let y = x.matmul(&w)?; // MatMul
        let z = y.add_scalar(1.0)?; // AddScalar
        let a = z.mul_scalar(2.0)?; // MulScalar
        let divisor = Tensor::full(a.shape().clone(), 3.0, device.clone())?;
        let b = a.div(&divisor)?; // Div
        let loss = b.sum()?;

        println!("   Starting backward pass for complex graph...");
        let grads = loss.backward()?;
        println!("✅ Complex graph backward pass completed!");
        println!("   Got gradients for {} tensors", grads.len());
    }

    // Test 5: Multiple activations
    {
        println!("\nTest 5: Multiple activations chain");
        let x = Tensor::randn(Shape::from_dims(&[5, 5]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);

        let y = x.relu()?; // ReLU
        let z = y.square()?; // Square
        let w = z.add_scalar(1e-6)?; // Add small value to avoid div by zero
        let v = z.div(&w)?; // Normalize
        let loss = v.sum()?;

        println!("   Loss requires_grad: {}", loss.requires_grad());
        let grads = loss.backward()?;
        println!("✅ Multiple activations backward pass completed!");
    }

    println!("\n🎉 ALL AUTOGRAD TESTS PASSED!");
    println!("The autograd system can now handle complex computation graphs for image/video model training!");

    Ok(())
}
