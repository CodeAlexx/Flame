use flame_core::{Tensor, Shape, Device};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=====================================");
    println!("DIRECT VAE CONV2D PERFORMANCE TEST");
    println!("=====================================");
    println!();
    println!("Testing the fix for 111+ second VAE encoding");
    println!("Grid size changed: 16384 -> 65535 blocks");
    println!();
    
    // Initialize CUDA
    let device = Device::cuda(0)?;
    println!("✅ CUDA device initialized");
    
    // Test a convolution similar to VAE first layer
    // VAE typically: 3 channels -> 128 channels on 512x512 image
    let batch = 1;
    let in_channels = 3;
    let out_channels = 128;
    let height = 512;
    let width = 512;
    let kernel = 3;
    
    println!("Creating test tensors...");
    println!("  Input: {}x{}x{}x{}", batch, in_channels, height, width);
    println!("  Weight: {}x{}x{}x{}", out_channels, in_channels, kernel, kernel);
    
    let input = Tensor::randn(
        Shape::from_dims(&[batch, in_channels, height, width]),
        0.0, 1.0, device.clone()
    )?;
    
    let weight = Tensor::randn(
        Shape::from_dims(&[out_channels, in_channels, kernel, kernel]),
        0.0, 1.0, device.clone()
    )?;
    
    // Warmup
    println!("\nWarming up CUDA...");
    let _ = input.conv2d(&weight, None, 1, 1)?;
    device.cuda_device()?.synchronize()?;
    
    // Time the actual operation
    println!("Running timed convolution...");
    let start = Instant::now();
    let _output = input.conv2d(&weight, None, 1, 1)?;
    device.cuda_device()?.synchronize()?;
    let elapsed = start.elapsed();
    
    let ms = elapsed.as_secs_f64() * 1000.0;
    println!("\n=====================================");
    println!("RESULT: Conv2d took {:.2}ms", ms);
    println!("=====================================");
    
    if elapsed.as_secs() >= 100 {
        println!("❌ CRITICAL FAILURE: >100 seconds!");
        println!("   The 111+ second bug is NOT fixed!");
        println!("   Grid size limit is still causing massive slowdown!");
    } else if elapsed.as_secs() >= 10 {
        println!("❌ FAILURE: >10 seconds!");
        println!("   Still way too slow, fix not working properly");
    } else if elapsed.as_secs() >= 1 {
        println!("⚠️  WARNING: >1 second ({:.2}s)", elapsed.as_secs_f64());
        println!("   Slower than target but better than 111 seconds");
    } else if ms > 100.0 {
        println!("✅ ACCEPTABLE: {:.0}ms (under 1 second)", ms);
        println!("   Fix is working! Was 111,000ms, now {:.0}ms", ms);
    } else {
        println!("🚀 EXCELLENT: {:.1}ms", ms);
        println!("   Optimal performance achieved!");
        println!("   Fix is working perfectly! Was 111,000ms, now {:.1}ms", ms);
    }
    
    // Estimate full VAE encoding time
    println!("\n=====================================");
    println!("ESTIMATED FULL VAE ENCODING TIME");
    println!("=====================================");
    let estimated_total = ms * 50.0 / 1000.0; // ~50 conv layers
    println!("~50 conv layers × {:.1}ms = {:.2} seconds total", ms, estimated_total);
    
    if estimated_total < 1.0 {
        println!("✅ Full VAE encoding should be <1 second!");
    } else if estimated_total < 5.0 {
        println!("✅ Full VAE encoding should be <5 seconds");
    } else {
        println!("⚠️  Full VAE might still take {:.1} seconds", estimated_total);
    }
    
    Ok(())
}