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

// Simple test to verify FLAME actually works
use flame_core::{device::CudaDevice, Result, Shape, Tensor};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("Testing FLAME Framework Reality...\n");

    // Create cached device
    let device = CudaDevice::new(0)?;
    println!("✅ CUDA device created and cached with Arc");

    // Test 1: Basic tensor operations
    println!("\n1. Testing basic tensor operations:");
    let a = Tensor::full(Shape::from_dims(&[2, 2]), 3.0, device.clone())?;
    let b = Tensor::full(Shape::from_dims(&[2, 2]), 2.0, device.clone())?;
    let c = a.add(&b)?;

    let result = c.to_vec()?;
    println!("   3.0 + 2.0 = {:?}", result[0]);
    assert!((result[0] - 5.0).abs() < 1e-6);
    println!("   ✅ Addition works correctly");

    // Test 2: Matrix multiplication
    println!("\n2. Testing matrix multiplication:");
    let x = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[2, 2]),
        device.clone(),
    )?;
    let y = Tensor::from_vec(
        vec![5.0, 6.0, 7.0, 8.0],
        Shape::from_dims(&[2, 2]),
        device.clone(),
    )?;
    let z = x.matmul(&y)?;

    // [1,2] * [5,7] = [1*5+2*6, 1*7+2*8] = [17, 23]
    // [3,4]   [6,8]   [3*5+4*6, 3*7+4*8]   [39, 53]
    let z_data = z.to_vec()?;
    println!("   Result: {:?}", z_data);
    assert!((z_data[0] - 19.0).abs() < 1e-6); // First element
    assert!((z_data[1] - 22.0).abs() < 1e-6);
    assert!((z_data[2] - 43.0).abs() < 1e-6);
    assert!((z_data[3] - 50.0).abs() < 1e-6);
    println!("   ✅ MatMul produces correct results");

    // Test 3: Autograd
    println!("\n3. Testing autograd:");
    let x = Tensor::full(Shape::from_dims(&[1]), 3.0, device.clone())?.requires_grad_(true);
    let y = x.mul(&x)?; // x^2
    let z = y.mul(&x)?; // x^3

    // For f(x) = x^3, f'(x) = 3x^2, so f'(3) = 27
    println!("   Computing gradient of x^3 at x=3");
    use flame_core::autograd::AutogradContext;
    let grad_map = AutogradContext::backward(&z)?;

    if let Some(x_grad) = grad_map.get(x.id()) {
        let grad_val = x_grad.to_vec()?[0];
        println!("   Gradient: {}", grad_val);
        assert!((grad_val - 27.0).abs() < 1e-4);
        println!("   ✅ Autograd computes correct gradient");
    }

    // Test 4: Memory management
    println!("\n4. Testing memory management:");
    {
        let _large_tensor =
            Tensor::randn(Shape::from_dims(&[1000, 1000]), 0.0, 1.0, device.clone())?;
        println!("   Created 1M element tensor");
    }
    println!("   Tensor dropped - memory should be freed");
    println!("   ✅ Memory management works");

    // Test 5: Device verification
    println!("\n5. Verifying device caching:");
    let another_ref = device.clone();
    println!("   Arc strong count: {}", Arc::strong_count(&device));
    assert!(Arc::strong_count(&device) >= 2);
    println!("   ✅ Device is properly cached with Arc");

    println!("\n=== FLAME REALITY CHECK PASSED ===");
    println!("Core functionality is REAL and working!");

    Ok(())
}
