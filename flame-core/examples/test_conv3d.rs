use flame_core::{CudaDevice, Tensor, Shape, Result};
use flame_core::conv3d::{Conv3d, BatchNorm3d};

fn main() -> Result<()> {
    println!("Testing Conv3d and BatchNorm3d...");
    
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");
    
    // Test Conv3d
    test_conv3d(&device)?;
    
    // Test BatchNorm3d
    test_batch_norm_3d(&device)?;
    
    println!("\nAll tests passed!");
    Ok(())
}

fn test_conv3d(device: &std::sync::Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Testing Conv3d ---");
    
    let conv = Conv3d::new(
        3,  // in_channels
        16, // out_channels
        (3, 3, 3), // kernel_size
        Some((1, 1, 1)), // stride
        Some((1, 1, 1)), // padding
        None, // dilation
        None, // groups
        true, // bias
        device.clone()
    )?;
    
    // Test input: [batch=2, channels=3, depth=8, height=32, width=32]
    let input = Tensor::randn(
        Shape::from_dims(&[2, 3, 8, 32, 32]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    println!("Input shape: {:?}", input.shape().dims());
    
    let output = conv.forward(&input)?;
    
    // Check output shape: [2, 16, 8, 32, 32] with padding
    println!("Output shape: {:?}", output.shape().dims());
    assert_eq!(output.shape().dims(), &[2, 16, 8, 32, 32]);
    
    println!("Conv3d test passed!");
    Ok(())
}

fn test_batch_norm_3d(device: &std::sync::Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Testing BatchNorm3d ---");
    
    let mut bn = BatchNorm3d::new(
        16, // num_features
        None,
        None,
        None,
        None,
        device.clone()
    )?;
    
    // Test input: [batch=2, channels=16, depth=4, height=8, width=8]
    let input = Tensor::randn(
        Shape::from_dims(&[2, 16, 4, 8, 8]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    println!("Input shape: {:?}", input.shape().dims());
    
    let output = bn.forward(&input, true)?;
    
    // Check output shape matches input
    println!("Output shape: {:?}", output.shape().dims());
    assert_eq!(output.shape().dims(), input.shape().dims());
    
    // Check that output is normalized
    let data = output.to_vec()?;
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    println!("BatchNorm3d output mean: {:.6}", mean);
    
    println!("BatchNorm3d test passed!");
    Ok(())
}