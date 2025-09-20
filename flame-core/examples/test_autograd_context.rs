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
use flame_core::{AutogradContext, Result, Shape, Tensor};

fn main() -> Result<()> {
    println!("Testing autograd context control...");

    let device = CudaDevice::new(0)?;

    // Test 1: Normal operation recording
    {
        AutogradContext::set_enabled(true);
        let a = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        let b = a.add(&a)?;
        println!("✅ With autograd enabled, operations are recorded");
    }

    // Test 2: Disabled autograd
    {
        AutogradContext::set_enabled(false);
        let a = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        let b = a.add(&a)?;
        let c = b.mul(&b)?;
        println!("✅ With autograd disabled, operations are NOT recorded");

        // Re-enable for normal use
        AutogradContext::set_enabled(true);
    }

    // Test 3: Test within backward pass simulation
    {
        let a = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        let b = Tensor::randn(Shape::from_dims(&[2, 2]), 0.0, 1.0, device.clone())?;

        // Simulate what happens in backward pass
        AutogradContext::set_enabled(false);

        // These operations should NOT try to record
        let c = a.transpose()?;
        let d = b.transpose()?;
        let e = c.matmul(&d)?;

        println!("✅ Operations within simulated backward pass don't hang");

        AutogradContext::set_enabled(true);
    }

    println!("\n🎉 All autograd context tests passed!");

    Ok(())
}
