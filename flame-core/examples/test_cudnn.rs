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

//! Test FLAME cuDNN integration

use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};
use std::time::Instant;

fn main() {
    println!("Testing FLAME cuDNN Integration");
    println!("================================");

    // Create CUDA device (not wrapped in Arc, CudaDevice is already Arc-like)
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");

    // Test a moderate size convolution
    let batch = 1;
    let in_channels = 128;
    let height = 512;
    let width = 512;
    let out_channels = 128;
    let kernel_size = 3;

    println!(
        "\nTest case: {}x{}x{}x{} -> {}x{}x?x? (kernel {}x{})",
        batch, in_channels, height, width, batch, out_channels, kernel_size, kernel_size
    );

    // Create input and weight tensors
    let input_shape = Shape::from_dims(&[batch, in_channels, height, width]);
    let weight_shape = Shape::from_dims(&[out_channels, in_channels, kernel_size, kernel_size]);

    // Tensor::randn takes (shape, mean, std, device) order
    let input =
        Tensor::randn(input_shape, 0.0, 1.0, device.clone()).expect("Failed to create input");
    let weight =
        Tensor::randn(weight_shape, 0.0, 0.1, device.clone()).expect("Failed to create weight");

    println!("Created input tensor: {:?}", input.shape());
    println!("Created weight tensor: {:?}", weight.shape());

    // Try convolution
    println!("\nAttempting convolution...");
    let start = Instant::now();

    match input.conv2d(&weight, None, 1, 1) {
        Ok(output) => {
            let elapsed = start.elapsed();
            println!("✅ Conv2d succeeded!");
            println!("  Output shape: {:?}", output.shape().dims());
            println!("  Time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

            // If this is fast (<10ms), cuDNN is working
            // If slow (>100ms), it fell back to im2col
            if elapsed.as_secs_f64() * 1000.0 < 10.0 {
                println!("  🚀 FAST! cuDNN is likely working!");
            } else if elapsed.as_secs_f64() * 1000.0 > 100.0 {
                println!("  🐌 SLOW! Likely using im2col fallback");
            } else {
                println!("  🤔 Medium speed - unclear which backend");
            }
        }
        Err(e) => {
            println!("❌ Conv2d failed: {:?}", e);
        }
    }

    // Calculate theoretical im2col memory usage
    let col_buffer_size =
        (kernel_size * kernel_size * in_channels * height * width * 4) / (1024 * 1024);
    println!("\nMemory comparison:");
    println!(
        "  Im2col would use: {} MB temporary buffer",
        col_buffer_size
    );
    println!("  cuDNN should use: <10 MB workspace");
}
