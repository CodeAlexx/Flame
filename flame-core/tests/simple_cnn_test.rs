//! Simple CNN forward pass test without autograd

use flame_core::{
    Tensor, Shape, Result,
    conv::Conv2d, pooling::{MaxPool2d, MaxPool2dConfig},
    linear::Linear,
};
use cudarc::driver::CudaDevice;

#[test]
fn test_cnn_forward_only() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Creating CNN layers...");
    
    // Simple CNN layers
    let conv1 = Conv2d::new(1, 16, 3, 1, 1, device.clone())?;
    let pool = MaxPool2d::new(MaxPool2dConfig::new((2, 2)));
    let fc = Linear::new(16 * 14 * 14, 10, true, &device)?;
    
    // Create input tensor
    let input = Tensor::randn(
        Shape::from_dims(&[1, 1, 28, 28]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    println!("Input shape: {:?}", input.shape().dims());
    
    // Forward pass
    let x1 = conv1.forward(&input)?;
    println!("After conv1: {:?}", x1.shape().dims());
    
    let x2 = x1.relu()?;
    println!("After relu: {:?}", x2.shape().dims());
    
    let (x3, _) = pool.forward(&x2)?;
    println!("After pool: {:?}", x3.shape().dims());
    
    // Flatten
    let batch_size = x3.shape().dims()[0];
    let x4 = x3.reshape(&[batch_size, 16 * 14 * 14])?;
    println!("After flatten: {:?}", x4.shape().dims());
    
    let output = fc.forward(&x4)?;
    println!("Final output: {:?}", output.shape().dims());
    
    // Check output shape
    assert_eq!(output.shape().dims(), &[1, 10]);
    
    println!("CNN forward pass test completed successfully!");
    Ok(())
}

#[test]
fn test_pooling_shapes() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Test different pooling configurations
    let pool2x2 = MaxPool2d::new(MaxPool2dConfig::new((2, 2)));
    let pool3x3 = MaxPool2d::new(MaxPool2dConfig {
        kernel_size: (3, 3),
        stride: Some((2, 2)),
        padding: (1, 1),
        dilation: (1, 1),
        return_indices: false,
    });
    
    // Test input
    let input = Tensor::randn(
        Shape::from_dims(&[2, 4, 16, 16]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    // Test 2x2 pooling
    let (out1, _) = pool2x2.forward(&input)?;
    assert_eq!(out1.shape().dims(), &[2, 4, 8, 8]);
    println!("2x2 pool: {:?} -> {:?}", input.shape().dims(), out1.shape().dims());
    
    // Test 3x3 pooling with stride 2
    let (out2, _) = pool3x3.forward(&input)?;
    // With padding 1 and stride 2: (16 + 2*1 - 3) / 2 + 1 = 8.5 -> 8
    assert_eq!(out2.shape().dims(), &[2, 4, 8, 8]);
    println!("3x3 pool stride 2: {:?} -> {:?}", input.shape().dims(), out2.shape().dims());
    
    println!("Pooling shape test completed!");
    Ok(())
}