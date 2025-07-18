use flame_core::{Tensor, CudaDevice, Shape, conv::{Conv2d, Conv2dConfig}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    // Create Conv2d layer: 3 input channels, 16 output channels, 3x3 kernel
    let config = Conv2dConfig {
        in_channels: 3,
        out_channels: 16,
        kernel_size: (3, 3),
        stride: (1, 1),
        padding: (1, 1), // Same padding
        groups: 1,
    };
    
    let conv = Conv2d::new(config, device.clone())?.with_bias(device.clone())?;
    println!("Created Conv2d layer: {} -> {} channels", 
             conv.config.in_channels, conv.config.out_channels);
    
    // Create input tensor: [batch=2, channels=3, height=32, width=32]
    let batch_size = 2;
    let height = 32;
    let width = 32;
    
    let input_shape = Shape::from_dims(&[batch_size, 3, height, width]);
    let input = Tensor::randn(input_shape, 0.0, 1.0, device.clone())?;
    println!("Input shape: {:?}", input.shape());
    
    // Forward pass
    let output = conv.forward(&input)?;
    println!("Output shape: {:?}", output.shape());
    
    // Verify output dimensions
    let out_dims = output.shape().dims();
    assert_eq!(out_dims[0], batch_size, "Batch size preserved");
    assert_eq!(out_dims[1], 16, "Output channels correct");
    assert_eq!(out_dims[2], 32, "Height preserved with padding");
    assert_eq!(out_dims[3], 32, "Width preserved with padding");
    
    println!("\n=== Testing different configurations ===");
    
    // Test with stride=2
    let config2 = Conv2dConfig {
        in_channels: 3,
        out_channels: 32,
        kernel_size: (3, 3),
        stride: (2, 2),
        padding: (1, 1),
        groups: 1,
    };
    
    let conv2 = Conv2d::new(config2, device.clone())?;
    let output2 = conv2.forward(&input)?;
    println!("With stride=2, output shape: {:?}", output2.shape());
    
    // Test 1x1 convolution (often used for channel mixing)
    let config3 = Conv2dConfig {
        in_channels: 3,
        out_channels: 64,
        kernel_size: (1, 1),
        stride: (1, 1),
        padding: (0, 0),
        groups: 1,
    };
    
    let conv3 = Conv2d::new(config3, device.clone())?;
    let output3 = conv3.forward(&input)?;
    println!("1x1 conv output shape: {:?}", output3.shape());
    
    // Test depthwise convolution (groups = in_channels) - TODO: Fix grouped convolution
    // let config4 = Conv2dConfig {
    //     in_channels: 3,
    //     out_channels: 3,
    //     kernel_size: (3, 3),
    //     stride: (1, 1),
    //     padding: (1, 1),
    //     groups: 3,
    // };
    // 
    // let conv4 = Conv2d::new(config4, device.clone())?;
    // let output4 = conv4.forward(&input)?;
    // println!("Depthwise conv output shape: {:?}", output4.shape());
    
    println!("\nConv2d demo completed successfully!");
    
    Ok(())
}