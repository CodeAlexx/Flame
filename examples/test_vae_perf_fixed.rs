use flame_core::{Tensor, Shape, Device};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("===========================================");
    println!("Testing VAE Conv2D Performance (Grid Fix)");
    println!("===========================================");
    println!("Previous: 111+ seconds per 512x512 image");
    println!("Expected: <1 second per image");
    println!("Grid size fixed: 16384 -> 65535 blocks");
    println!();
    
    // Initialize CUDA
    let device = Device::cuda(0)?;
    
    // Test VAE-like convolution (first layer: 3->128 channels on 512x512)
    println!("Testing VAE first layer (3->128 channels, 512x512):");
    let input = Tensor::randn(
        Shape::from_dims(&[1, 3, 512, 512]),
        0.0, 1.0, device.clone()
    )?;
    
    let weight = Tensor::randn(
        Shape::from_dims(&[128, 3, 3, 3]),
        0.0, 1.0, device.clone()
    )?;
    
    // Warmup
    println!("Warming up CUDA...");
    let _ = input.conv2d(&weight, None, 1, 1)?;
    device.cuda_device()?.synchronize()?;
    
    // Time single operation
    println!("Timing conv2d operation...");
    let start = Instant::now();
    let output = input.conv2d(&weight, None, 1, 1)?;
    device.cuda_device()?.synchronize()?;
    let elapsed = start.elapsed();
    
    let ms = elapsed.as_secs_f64() * 1000.0;
    println!("Time: {:.2}ms", ms);
    
    // Check performance
    if elapsed.as_secs() > 10 {
        println!("❌ CRITICAL FAILURE: >10 seconds! Grid fix not applied!");
        println!("   Check cuda_conv2d.rs lines 166 and 216");
    } else if elapsed.as_secs() >= 1 {
        println!("❌ FAILURE: Still >1 second ({}s)", elapsed.as_secs());
        println!("   Performance requirement not met!");
    } else if ms > 100.0 {
        println!("⚠️  WARNING: Slower than expected ({:.0}ms > 100ms)", ms);
        println!("   But acceptable (<1 second requirement met)");
    } else if ms > 10.0 {
        println!("✅ SUCCESS: Fast performance ({:.1}ms)", ms);
        println!("   Requirement met: <1 second ✓");
    } else {
        println!("🚀 EXCELLENT: Very fast ({:.1}ms < 10ms)", ms);
        println!("   Optimal performance achieved!");
    }
    
    // Test typical VAE layer (128->128 channels)
    println!("\nTesting typical VAE layer (128->128 channels, 256x256):");
    let input2 = Tensor::randn(
        Shape::from_dims(&[1, 128, 256, 256]),
        0.0, 1.0, device.clone()
    )?;
    
    let weight2 = Tensor::randn(
        Shape::from_dims(&[128, 128, 3, 3]),
        0.0, 1.0, device.clone()
    )?;
    
    let start = Instant::now();
    let _ = input2.conv2d(&weight2, None, 1, 1)?;
    device.cuda_device()?.synchronize()?;
    let elapsed = start.elapsed();
    
    println!("Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    
    // Calculate approximate full VAE encoding time
    println!("\n===========================================");
    println!("Estimated Full VAE Encoding Time:");
    println!("(~50 conv layers in VAE encoder)");
    let estimated_total = elapsed.as_secs_f64() * 50.0;
    println!("Estimated: {:.2} seconds", estimated_total);
    
    if estimated_total < 1.0 {
        println!("✅ EXCELLENT: Full VAE should be <1 second!");
    } else if estimated_total < 5.0 {
        println!("✅ GOOD: Full VAE should be <5 seconds");
    } else {
        println!("⚠️  WARNING: Full VAE might take {:.0} seconds", estimated_total);
    }
    
    println!("===========================================");
    
    Ok(())
}