use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};

fn main() -> flame_core::Result<()> {
    println!("Testing sum operation backward...\n");

    let device = CudaDevice::new(0)?;

    // Test 1: Simple sum
    println!("Test 1: Simple 1D sum");
    let x = Tensor::randn(Shape::from_dims(&[5]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let sum_x = x.sum()?;

    println!("x shape: {:?}", x.shape().dims());
    println!("sum shape: {:?}", sum_x.shape().dims());
    println!("sum requires_grad: {}", sum_x.requires_grad());

    match sum_x.backward() {
        Ok(grads) => {
            println!("✓ 1D sum backward succeeded! {} gradients", grads.len());
        }
        Err(e) => {
            println!("✗ 1D sum backward failed: {:?}", e);
        }
    }

    // Test 2: 2D sum
    println!("\nTest 2: 2D tensor sum");
    flame_core::AutogradContext::clear();

    let y =
        Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let sum_y = y.sum()?;

    println!("y shape: {:?}", y.shape().dims());
    println!("sum shape: {:?}", sum_y.shape().dims());

    match sum_y.backward() {
        Ok(grads) => {
            println!("✓ 2D sum backward succeeded! {} gradients", grads.len());
        }
        Err(e) => {
            println!("✗ 2D sum backward failed: {:?}", e);
        }
    }

    // Test 3: Chain with sum
    println!("\nTest 3: Chain operations with sum");
    flame_core::AutogradContext::clear();

    let a = Tensor::ones(Shape::from_dims(&[2, 2]), device.clone())?.requires_grad_(true);
    let b = a.mul_scalar(2.0)?;
    let c = b.add_scalar(1.0)?;
    let loss = c.sum()?;

    println!("Chain: a -> mul(2) -> add(1) -> sum");
    println!("Calling backward...");

    match loss.backward() {
        Ok(grads) => {
            println!("✓ Chain backward succeeded! {} gradients", grads.len());
            // The gradient for 'a' should be 2.0 everywhere (due to mul by 2)
            if let Some(grad_a) = grads.get(a.id()) {
                let grad_values = grad_a.to_vec()?;
                println!("Gradient for a: {:?}", grad_values);
            }
        }
        Err(e) => {
            println!("✗ Chain backward failed: {:?}", e);
        }
    }

    Ok(())
}
