use flame_core::{Tensor, Shape, Device};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("Testing Conv2D performance with fixed grid size...");
    println!("Grid size limit changed from 16384 to 65535 blocks");
    println!("");
    
    // Initialize CUDA
    let device = Device::cuda(0)?;
    
    // Test the specific size that was problematic
    println!("Testing 512x512 image encoding (was taking 111+ seconds):");
    
    // Typical VAE encoder first layer: 3 -> 128 channels
    let batch = 1;
    let in_channels = 3;
    let out_channels = 128;
    let height = 512;
    let width = 512;
    let kernel_h = 3;
    let kernel_w = 3;
    
    println!("Input: {}x{}x{}x{}", batch, in_channels, height, width);
    println!("Weight: {}x{}x{}x{}", out_channels, in_channels, kernel_h, kernel_w);
    
    // Create tensors
    let input = Tensor::randn(
        Shape::from_dims(&[batch, in_channels, height, width]),
        0.0, 1.0, device.clone()
    )?;
    
    let weight = Tensor::randn(
        Shape::from_dims(&[out_channels, in_channels, kernel_h, kernel_w]),
        0.0, 1.0, device.clone()
    )?;
    
    // Warmup
    println!("Warming up...");
    let _ = input.conv2d(&weight, None, 1, 1)?;
    device.cuda_device()?.synchronize()?;
    
    // Time single convolution
    println!("Timing single conv2d operation...");
    let start = Instant::now();
    let _ = input.conv2d(&weight, None, 1, 1)?;
    device.cuda_device()?.synchronize()?;
    let elapsed = start.elapsed();
    
    println!("Single conv2d took: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    
    if elapsed.as_secs_f64() > 1.0 {
        println!("ERROR: Still TOO SLOW! Was supposed to be <1 second!");
        println!("The grid size fix may not have been applied correctly.");
    } else if elapsed.as_secs_f64() > 0.1 {
        println!("WARNING: Slower than expected (>100ms)");
    } else {
        println!("SUCCESS: Fast performance achieved (<100ms)!");
    }
    
    // Test multiple sizes to see scaling
    println!("\nTesting different sizes:");
    let sizes = vec![
        (128, 128),
        (256, 256),
        (512, 512),
    ];
    
    for (h, w) in sizes {
        let input = Tensor::randn(
            Shape::from_dims(&[1, 128, h, w]),
            0.0, 1.0, device.clone()
        )?;
        
        let weight = Tensor::randn(
            Shape::from_dims(&[128, 128, 3, 3]),
            0.0, 1.0, device.clone()
        )?;
        
        let start = Instant::now();
        let _ = input.conv2d(&weight, None, 1, 1)?;
        device.cuda_device()?.synchronize()?;
        let elapsed = start.elapsed();
        
        println!("  {}x{}: {:.2}ms", h, w, elapsed.as_secs_f64() * 1000.0);
    }
    
    Ok(())
}