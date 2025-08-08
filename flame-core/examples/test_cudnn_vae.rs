use flame_core::{Device, DType, Result, Shape, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println\!("Testing cuDNN-accelerated VAE encoding...");
    
    // Create device
    let device = Device::cuda(0)?;
    
    // Create input tensor (batch=1, channels=3, height=1024, width=1024)
    let input = Tensor::randn(
        0.0, 1.0,
        Shape::from_dims(&[1, 3, 1024, 1024]),
        device.clone()
    )?;
    
    println\!("Input shape: {:?}", input.shape());
    
    // Test Conv2D operation (will use cuDNN if available)
    let weight = Tensor::randn(
        0.0, 0.02,
        Shape::from_dims(&[128, 3, 3, 3]), // out_channels=128, in_channels=3, kernel=3x3
        device.clone()
    )?;
    
    println\!("\nRunning Conv2D with cuDNN acceleration (if available)...");
    let start = Instant::now();
    
    // This will automatically use cuDNN if the feature is enabled
    let output = input.conv2d(&weight, None, 1, 1)?;
    
    let elapsed = start.elapsed();
    println\!("Conv2D completed in {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println\!("Output shape: {:?}", output.shape());
    
    // Test multiple Conv2D operations (simulate VAE encoder)
    println\!("\nSimulating VAE encoder (multiple Conv2D layers)...");
    let mut x = input.clone()?;
    let layers = vec\![
        (3, 128, 3),     // Conv2D(3 -> 128)
        (128, 128, 3),   // Conv2D(128 -> 128)
        (128, 256, 3),   // Conv2D(128 -> 256)
        (256, 256, 3),   // Conv2D(256 -> 256)
        (256, 512, 3),   // Conv2D(256 -> 512)
        (512, 512, 3),   // Conv2D(512 -> 512)
        (512, 512, 3),   // Conv2D(512 -> 512)
        (512, 8, 3),     // Conv2D(512 -> 8) for latents
    ];
    
    let start = Instant::now();
    
    for (i, (in_c, out_c, k)) in layers.iter().enumerate() {
        let current_in_channels = if i == 0 { 3 } else {
            match i {
                1 => 128,
                2 => 128,
                3 => 256,
                4 => 256,
                5 => 512,
                6 => 512,
                7 => 512,
                _ => 512,
            }
        };
        
        let weight = Tensor::randn(
            0.0, 0.02,
            Shape::from_dims(&[*out_c, current_in_channels, *k, *k]),
            device.clone()
        )?;
        
        // Apply convolution
        let stride = if i % 2 == 1 { 2 } else { 1 }; // Downsample every other layer
        x = x.conv2d(&weight, None, stride, 1)?;
        
        // Apply activation (SiLU)
        x = x.silu()?;
        
        println\!("  Layer {}: shape {:?}", i + 1, x.shape());
    }
    
    let total_time = start.elapsed();
    println\!("\nTotal VAE encoding time: {:.2}ms", total_time.as_secs_f64() * 1000.0);
    println\!("Final latent shape: {:?}", x.shape());
    
    #[cfg(feature = "cudnn")]
    println\!("\n✅ cuDNN acceleration is ENABLED");
    
    #[cfg(not(feature = "cudnn"))]
    println\!("\n⚠️  cuDNN acceleration is DISABLED");
    
    Ok(())
}
