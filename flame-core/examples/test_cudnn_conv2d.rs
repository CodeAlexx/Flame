#![cfg(all(feature = "legacy_examples", feature = "cudnn"))]
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

// Test cuDNN Conv2D integration in FLAME
use cudarc::driver::CudaDevice;
use flame_core::cudnn::conv2d::cudnn_conv2d;
use flame_core::{Shape, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing cuDNN Conv2D in FLAME...");

    // Create CUDA device
    let device = CudaDevice::new(0)?;

    // Create test tensors - small test first
    let batch_size = 1;
    let in_channels = 3;
    let height = 32;
    let width = 32;
    let out_channels = 16;
    let kernel_size = 3;

    // Create input tensor
    let input_shape = Shape::from_dims(&[batch_size, in_channels, height, width]);
    let input = Tensor::randn(input_shape, 0.0, 1.0, device.clone())?;

    // Create weight tensor
    let weight_shape = Shape::from_dims(&[out_channels, in_channels, kernel_size, kernel_size]);
    let weight = Tensor::randn(weight_shape, 0.0, 0.1, device.clone())?;

    // Create bias tensor
    let bias_shape = Shape::from_dims(&[out_channels]);
    let bias = Tensor::zeros(bias_shape, device.clone())?;

    // Test Conv2D with cuDNN
    println!("Running cuDNN Conv2D...");
    let start = Instant::now();
    let output = cudnn_conv2d(&input, &weight, Some(&bias), 1, 1)?;
    device.synchronize()?;
    let elapsed = start.elapsed();

    println!("Output shape: {:?}", output.shape());
    println!("Time: {:.3}ms", elapsed.as_secs_f32() * 1000.0);

    // Now test with larger image (1024x1024 like in VAE)
    println!("\nTesting with VAE-sized input (1024x1024)...");
    let large_input_shape = Shape::from_dims(&[1, 3, 1024, 1024]);
    let large_input = Tensor::randn(large_input_shape, 0.0, 1.0, device.clone())?;

    // Warm up
    for _ in 0..3 {
        let _ = cudnn_conv2d(&large_input, &weight, Some(&bias), 1, 1)?;
        device.synchronize()?;
    }

    // Benchmark
    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        let _ = cudnn_conv2d(&large_input, &weight, Some(&bias), 1, 1)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    println!(
        "Average time for 1024x1024 Conv2D: {:.3}ms",
        elapsed.as_secs_f32() * 1000.0 / iterations as f32
    );

    println!("\n✅ cuDNN Conv2D test passed!");

    Ok(())
}
