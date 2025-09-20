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

use flame_core::{CudaDevice, Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    // Test basic operations with CUDA kernels
    println!("\n=== Testing CUDA Kernels ===");

    // Create test tensors
    let shape = Shape::from_dims(&[4, 4]);
    let a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        shape.clone(),
        device.clone(),
    )?;

    let b = Tensor::from_vec(
        vec![
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ],
        shape.clone(),
        device.clone(),
    )?;

    // Test addition (CUDA kernel)
    println!("\nTesting addition (CUDA kernel):");
    let c = a.add(&b)?;
    let c_data = c.to_vec()?;
    println!("a + b = {:?}", &c_data[0..4]); // Should be all 17s

    // Test multiplication (CUDA kernel)
    println!("\nTesting multiplication (CUDA kernel):");
    let d = a.mul(&b)?;
    let d_data = d.to_vec()?;
    println!("a * b = {:?}", &d_data[0..4]);

    // Test scalar multiplication (CUDA kernel)
    println!("\nTesting scalar multiplication (CUDA kernel):");
    let e = a.mul_scalar(2.0)?;
    let e_data = e.to_vec()?;
    println!("a * 2 = {:?}", &e_data[0..4]); // Should be [2, 4, 6, 8]

    // Test ReLU (CUDA kernel)
    println!("\nTesting ReLU (CUDA kernel):");
    let neg_tensor = Tensor::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 5.0],
        Shape::from_dims(&[8]),
        device.clone(),
    )?;
    let relu_result = neg_tensor.relu()?;
    let relu_data = relu_result.to_vec()?;
    println!("ReLU([-2, -1, 0, 1, 2, 3, -4, 5]) = {:?}", relu_data);

    // Test sum reduction (CUDA kernel)
    println!("\nTesting sum reduction (CUDA kernel):");
    let sum_result = a.sum()?;
    let sum_data = sum_result.to_vec()?;
    println!("sum(a) = {:?}", sum_data); // Should be 136.0

    // Test weight update (CUDA kernel)
    println!("\nTesting weight update (CUDA kernel):");
    let mut weights = Tensor::from_vec(
        vec![10.0, 20.0, 30.0, 40.0],
        Shape::from_dims(&[4]),
        device.clone(),
    )?;
    let gradients = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        device.clone(),
    )?;

    println!("Initial weights: {:?}", weights.to_vec()?);
    weights.update_weights(&gradients, 0.1)?;
    println!("After update (lr=0.1): {:?}", weights.to_vec()?);
    // Should be [9.9, 19.8, 29.7, 39.6]

    println!("\nAll CUDA kernel tests completed!");

    Ok(())
}
