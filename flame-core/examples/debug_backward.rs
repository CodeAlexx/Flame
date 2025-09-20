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

//! Debug backward pass issues

use flame_core::{AutogradContext, CudaDevice, Result, Shape, Tensor};

fn main() -> Result<()> {
    println!("Debug backward pass...\n");

    let device = CudaDevice::new(0)?;

    // Test 1: Simple scalar operation
    println!("Test 1: Scalar operations");
    AutogradContext::clear();

    let x = Tensor::randn(Shape::from_dims(&[1]), 2.0, 0.0, device.clone())?.requires_grad_(true);
    let y = x.mul_scalar(3.0)?;
    let z = y.add_scalar(1.0)?;

    println!("Forward: x * 3 + 1");
    println!("z = {:.2}", z.to_vec()?[0]);

    let grads = AutogradContext::backward(&z)?;
    if let Some(x_grad) = grads.get(x.id()) {
        println!("x gradient: {:.2}", x_grad.to_vec()?[0]);
    }
    println!("✅ Scalar test passed\n");

    // Test 2: Simple add
    println!("Test 2: Add operation");
    AutogradContext::clear();

    let a =
        Tensor::randn(Shape::from_dims(&[2, 2]), 1.0, 0.0, device.clone())?.requires_grad_(true);
    let b =
        Tensor::randn(Shape::from_dims(&[2, 2]), 1.0, 0.0, device.clone())?.requires_grad_(true);
    let c = a.add(&b)?;
    let loss = c.sum()?;

    println!("Forward: sum(a + b)");
    println!("loss = {:.2}", loss.to_vec()?[0]);

    let grads = AutogradContext::backward(&loss)?;
    if let Some(a_grad) = grads.get(a.id()) {
        println!("a gradient shape: {:?}", a_grad.shape().dims());
    }
    println!("✅ Add test passed\n");

    // Test 3: Small MatMul
    println!("Test 3: Small MatMul");
    AutogradContext::clear();

    let m1 = Tensor::from_vec(vec![1.0, 2.0], Shape::from_dims(&[1, 2]), device.clone())?
        .requires_grad_(true);
    let m2 = Tensor::from_vec(vec![3.0, 4.0], Shape::from_dims(&[2, 1]), device.clone())?
        .requires_grad_(true);

    println!("m1: {:?}", m1.to_vec()?);
    println!("m2: {:?}", m2.to_vec()?);

    let result = m1.matmul(&m2)?;
    println!("m1 @ m2 = {:?}", result.to_vec()?);

    println!("About to backward...");
    let grads = AutogradContext::backward(&result)?;
    println!("Backward complete!");

    if let Some(m1_grad) = grads.get(m1.id()) {
        println!("m1 gradient: {:?}", m1_grad.to_vec()?);
    }

    Ok(())
}
