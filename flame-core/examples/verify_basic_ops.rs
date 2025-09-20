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
use std::sync::Arc;

fn main() -> Result<()> {
    println!("=== FLAME Basic Operations Verification ===\n");

    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0).map_err(|e| {
        flame_core::FlameError::Cuda(format!("Failed to create CUDA device: {:?}", e))
    })?);
    println!("✓ CUDA device initialized");

    // Test 1: Tensor creation
    println!("\n1. Testing tensor creation...");
    let shape = Shape::from_dims(&[2, 3]);
    let a = Tensor::ones(shape.clone(), device.clone())?;
    let b = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape.clone(),
        device.clone(),
    )?;
    println!("✓ Created tensors a (ones) and b (1-6)");

    // Test 2: Basic arithmetic
    println!("\n2. Testing arithmetic operations...");

    // Addition
    let c = a.add(&b)?;
    println!("✓ Addition: a + b completed");

    // Multiplication
    let d = b.mul(&b)?;
    println!("✓ Multiplication: b * b completed");

    // Scalar operations
    let e = b.mul_scalar(2.0)?;
    println!("✓ Scalar multiplication: b * 2.0 completed");

    let f = b.add_scalar(10.0)?;
    println!("✓ Scalar addition: b + 10.0 completed");

    // Test 3: Activations
    println!("\n3. Testing activation functions...");

    let g = b.relu()?;
    println!("✓ ReLU activation completed");

    let h = b.sigmoid()?;
    println!("✓ Sigmoid activation completed");

    let i = b.tanh()?;
    println!("✓ Tanh activation completed");

    // Test 4: Matrix operations
    println!("\n4. Testing matrix operations...");

    // Create matrices for matmul
    let mat_a = Tensor::ones(Shape::from_dims(&[3, 4]), device.clone())?;
    let mat_b = Tensor::ones(Shape::from_dims(&[4, 2]), device.clone())?;

    let mat_c = mat_a.matmul(&mat_b)?;
    println!("✓ Matrix multiplication [3,4] x [4,2] = [3,2] completed");

    // Test 5: Transpose
    let mat_t = mat_a.transpose()?;
    println!("✓ Transpose [3,4] -> [4,3] completed");

    // Test 6: Broadcasting
    println!("\n5. Testing broadcasting...");
    let scalar = Tensor::ones(Shape::from_dims(&[1]), device.clone())?;
    let broadcast_result = scalar.broadcast_to(&Shape::from_dims(&[2, 3]))?;
    println!("✓ Broadcast [1] -> [2,3] completed");

    println!("\n=== All basic operations completed successfully! ===");

    // Optional: Download and print some results
    println!("\nSample results (downloading from GPU):");

    // Helper function to download and print
    fn download_and_print(name: &str, tensor: &Tensor) -> Result<()> {
        let numel = tensor.shape().elem_count();
        let mut cpu_data = vec![0.0f32; numel];
        tensor
            .device()
            .dtoh_sync_copy_into(&**tensor.data(), &mut cpu_data)
            .map_err(|_| flame_core::FlameError::CudaDriver)?;

        println!(
            "{}: {:?} (first 6 elements)",
            name,
            &cpu_data[..6.min(numel)]
        );
        Ok(())
    }

    download_and_print("b", &b)?;
    download_and_print("b + 10", &f)?;
    download_and_print("b * 2", &e)?;
    download_and_print("sigmoid(b)", &h)?;

    Ok(())
}
