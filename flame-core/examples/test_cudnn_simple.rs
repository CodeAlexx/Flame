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

use flame_core::{DType, Device, Result, Shape, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("TESTING CUDNN ACCELERATION IN FLAME");
    println!("{}", "=".repeat(60));

    // Create CUDA device
    let device = Device::cuda(0)?;
    println!("\n✅ CUDA device initialized");

    // Test 1: Single Conv2D operation (1024x1024 image)
    println!("\n📊 TEST 1: Conv2D on 1024x1024 image");
    println!("{}", "-".repeat(40));

    let input = Tensor::randn(
        Shape::from_dims(&[1, 3, 1024, 1024]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?;

    let weight = Tensor::randn(
        Shape::from_dims(&[128, 3, 3, 3]), // 128 filters, 3 input channels, 3x3 kernel
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;

    // Warmup
    let _ = input.conv2d(&weight, None, 1, 1)?;

    // Benchmark
    let start = Instant::now();
    let output = input.conv2d(&weight, None, 1, 1)?;
    let elapsed = start.elapsed();

    println!("Input shape:  {:?}", input.shape().dims());
    println!("Weight shape: {:?}", weight.shape().dims());
    println!("Output shape: {:?}", output.shape().dims());
    println!("Time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    #[cfg(feature = "cudnn")]
    {
        println!("✅ cuDNN feature ENABLED");
        if elapsed.as_secs_f64() * 1000.0 < 10.0 {
            println!("⚡ Performance suggests cuDNN acceleration is working!");
        }
    }

    #[cfg(not(feature = "cudnn"))]
    println!("⚠️  cuDNN feature DISABLED");

    // Test 2: Multiple Conv2D layers (VAE-like)
    println!("\n📊 TEST 2: VAE-like encoding (8 Conv2D layers)");
    println!("{}", "-".repeat(40));

    let mut x = input.clone()?;
    let start_vae = Instant::now();

    // Layer 1: 3 -> 128 channels
    let w1 = Tensor::randn(
        Shape::from_dims(&[128, 3, 3, 3]),
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;
    x = x.conv2d(&w1, None, 1, 1)?.silu()?;

    // Layer 2: 128 -> 128 channels, stride 2 (downsample)
    let w2 = Tensor::randn(
        Shape::from_dims(&[128, 128, 3, 3]),
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;
    x = x.conv2d(&w2, None, 2, 1)?.silu()?;

    // Layer 3: 128 -> 256 channels
    let w3 = Tensor::randn(
        Shape::from_dims(&[256, 128, 3, 3]),
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;
    x = x.conv2d(&w3, None, 1, 1)?.silu()?;

    // Layer 4: 256 -> 256 channels, stride 2 (downsample)
    let w4 = Tensor::randn(
        Shape::from_dims(&[256, 256, 3, 3]),
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;
    x = x.conv2d(&w4, None, 2, 1)?.silu()?;

    // Layer 5: 256 -> 512 channels
    let w5 = Tensor::randn(
        Shape::from_dims(&[512, 256, 3, 3]),
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;
    x = x.conv2d(&w5, None, 1, 1)?.silu()?;

    // Layer 6: 512 -> 512 channels, stride 2 (downsample)
    let w6 = Tensor::randn(
        Shape::from_dims(&[512, 512, 3, 3]),
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;
    x = x.conv2d(&w6, None, 2, 1)?.silu()?;

    // Layer 7: 512 -> 512 channels
    let w7 = Tensor::randn(
        Shape::from_dims(&[512, 512, 3, 3]),
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;
    x = x.conv2d(&w7, None, 1, 1)?.silu()?;

    // Layer 8: 512 -> 8 channels (latent)
    let w8 = Tensor::randn(
        Shape::from_dims(&[8, 512, 3, 3]),
        0.0,
        0.02,
        device.cuda_device_arc(),
    )?;
    x = x.conv2d(&w8, None, 1, 1)?;

    let vae_time = start_vae.elapsed();

    println!("Final latent shape: {:?}", x.shape().dims());
    println!("Total time: {:.3} ms", vae_time.as_secs_f64() * 1000.0);

    // Performance analysis
    println!("\n📈 PERFORMANCE ANALYSIS");
    println!("{}", "-".repeat(40));

    let vae_ms = vae_time.as_secs_f64() * 1000.0;

    if vae_ms < 500.0 {
        println!("⚡ EXCELLENT: VAE encoding < 500ms");
        println!("   This indicates cuDNN acceleration is working!");
    } else if vae_ms < 2000.0 {
        println!("✅ GOOD: VAE encoding < 2s");
        println!("   Better than native, but not optimal");
    } else {
        println!("⚠️  SLOW: VAE encoding > 2s");
        println!("   cuDNN may not be active");
    }

    println!("\nExpected performance:");
    println!("  - With cuDNN:    ~100-200 ms");
    println!("  - Native FLAME:  ~2-5 seconds");
    println!("  - Original im2col: ~111 seconds");

    println!("\n✅ Test completed successfully!");

    Ok(())
}
