use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::norm::{GroupNorm, InstanceNorm2d, RMSNorm, RMSNorm1d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame normalization layers...");
    
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");
    
    // Create test input tensor [batch=2, channels=4, height=4, width=4]
    let input = Tensor::randn(
        Shape::from_dims(&[2, 4, 4, 4]),
        0.0,
        1.0,
        device.clone()
    )?;
    println!("Input shape: {:?}", input.shape().dims());
    
    // Test GroupNorm
    println!("\n--- Testing GroupNorm ---");
    
    // Group norm with 2 groups (each group has 2 channels)
    let group_norm = GroupNorm::new(
        2,       // num_groups
        4,       // num_channels
        1e-5,    // eps
        true,    // affine
        device.clone()
    )?;
    
    let gn_output = group_norm.forward(&input)?;
    println!("GroupNorm output shape: {:?}", gn_output.shape().dims());
    
    // Check that output has proper statistics
    let gn_data = gn_output.to_vec()?;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    
    // Check statistics for first group (channels 0-1)
    for n in 0..2 {
        for c in 0..2 {
            for h in 0..4 {
                for w in 0..4 {
                    let idx = n * 64 + c * 16 + h * 4 + w;
                    sum += gn_data[idx];
                    sum_sq += gn_data[idx] * gn_data[idx];
                }
            }
        }
    }
    
    let count = 2 * 2 * 4 * 4; // batch * channels_in_group * height * width
    let mean = sum / count as f32;
    let var = (sum_sq / count as f32) - mean * mean;
    println!("Group 1 - Mean: {:.6}, Variance: {:.6} (should be near 0 and 1)", mean, var);
    
    // Test InstanceNorm2d
    println!("\n--- Testing InstanceNorm2d ---");
    
    let mut instance_norm = InstanceNorm2d::new(
        4,       // num_features
        1e-5,    // eps
        0.1,     // momentum
        true,    // affine
        false,   // track_running_stats
        device.clone()
    )?;
    
    let in_output = instance_norm.forward(&input)?;
    println!("InstanceNorm2d output shape: {:?}", in_output.shape().dims());
    
    // Check that each instance-channel has mean ~0 and variance ~1
    let in_data = in_output.to_vec()?;
    
    for n in 0..2 {
        for c in 0..4 {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            
            for h in 0..4 {
                for w in 0..4 {
                    let idx = n * 64 + c * 16 + h * 4 + w;
                    sum += in_data[idx];
                    sum_sq += in_data[idx] * in_data[idx];
                }
            }
            
            let spatial_size = 16;
            let mean = sum / spatial_size as f32;
            let var = (sum_sq / spatial_size as f32) - mean * mean;
            println!("Instance {} Channel {} - Mean: {:.6}, Variance: {:.6}", n, c, mean, var);
        }
    }
    
    // Test with different input sizes
    println!("\n--- Testing with larger input ---");
    let larger_input = Tensor::randn(
        Shape::from_dims(&[4, 32, 8, 8]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    // GroupNorm with 8 groups
    let gn2 = GroupNorm::new(8, 32, 1e-5, false, device.clone())?;
    let gn2_output = gn2.forward(&larger_input)?;
    println!("Large GroupNorm output shape: {:?}", gn2_output.shape().dims());
    
    // InstanceNorm
    let mut in2 = InstanceNorm2d::new(32, 1e-5, 0.1, false, false, device.clone())?;
    let in2_output = in2.forward(&larger_input)?;
    println!("Large InstanceNorm output shape: {:?}", in2_output.shape().dims());
    
    // Test RMSNorm
    println!("\n--- Testing RMSNorm ---");
    
    let rms_norm = RMSNorm::new(
        vec![4, 4],     // Normalize over last 2 dimensions
        1e-5,           // eps
        true,           // affine
        device.clone()
    )?;
    
    let rms_output = rms_norm.forward(&input)?;
    println!("RMSNorm output shape: {:?}", rms_output.shape().dims());
    
    // Verify RMS normalization
    let rms_data = rms_output.to_vec()?;
    
    // Check first sample
    let mut sum_sq = 0.0f32;
    for i in 0..16 { // 4x4 spatial dimensions
        sum_sq += rms_data[i] * rms_data[i];
    }
    let rms = (sum_sq / 16.0).sqrt();
    println!("RMS of first normalized sample: {:.6} (should be close to 1)", rms);
    
    // Test RMSNorm1d
    println!("\n--- Testing RMSNorm1d ---");
    
    let input_1d = Tensor::randn(
        Shape::from_dims(&[8, 512]),  // [batch, features]
        0.0,
        1.0,
        device.clone()
    )?;
    
    let rms_norm_1d = RMSNorm1d::new(
        512,            // normalized_shape
        1e-5,           // eps
        device.clone()
    )?;
    
    let rms_1d_output = rms_norm_1d.forward(&input_1d)?;
    println!("RMSNorm1d input shape: {:?}", input_1d.shape().dims());
    println!("RMSNorm1d output shape: {:?}", rms_1d_output.shape().dims());
    
    // Check RMS for a few sequences
    let rms_1d_data = rms_1d_output.to_vec()?;
    for seq in 0..3 {
        let mut sum_sq = 0.0f32;
        for i in 0..512 {
            let idx = seq * 512 + i;
            sum_sq += rms_1d_data[idx] * rms_1d_data[idx];
        }
        let rms = (sum_sq / 512.0).sqrt();
        println!("Sequence {} RMS: {:.6}", seq, rms);
    }
    
    println!("\nAll normalization tests completed!");
    
    Ok(())
}