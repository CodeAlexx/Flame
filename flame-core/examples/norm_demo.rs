use flame_core::{Tensor, CudaDevice, Shape, norm::{BatchNorm2d, LayerNorm}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing BatchNorm2d ===");
    
    // Create BatchNorm2d layer for 16 channels
    let num_features = 16;
    let mut bn = BatchNorm2d::new(
        num_features,
        1e-5,      // eps
        0.1,       // momentum
        true,      // affine
        true,      // track_running_stats
        device.clone()
    )?;
    
    println!("Created BatchNorm2d with {} features", num_features);
    
    // Create input tensor: [batch=4, channels=16, height=8, width=8]
    let input_shape = Shape::from_dims(&[4, 16, 8, 8]);
    let input = Tensor::randn(input_shape, 0.0, 1.0, device.clone())?;
    println!("Input shape: {:?}", input.shape());
    
    // Forward pass in training mode
    let output_train = bn.forward(&input, true)?;
    println!("Output shape (training): {:?}", output_train.shape());
    
    // Check that output has same shape as input
    assert_eq!(output_train.shape().dims(), input.shape().dims());
    
    // Forward pass in evaluation mode
    let output_eval = bn.forward(&input, false)?;
    println!("Output shape (eval): {:?}", output_eval.shape());
    
    // Verify normalization worked
    let output_data = output_train.to_vec()?;
    let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
    let variance = output_data.iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>() / output_data.len() as f32;
    
    println!("Output statistics - Mean: {:.6}, Variance: {:.6}", mean, variance);
    println!("(Should be close to 0 and 1 respectively)");
    
    println!("\n=== Testing LayerNorm ===");
    
    // Create LayerNorm for the last two dimensions [height=8, width=8]
    let normalized_shape = vec![8, 8];
    let ln = LayerNorm::new_with_affine(
        normalized_shape.clone(),
        1e-5,      // eps
        true,      // elementwise_affine
        device.clone()
    )?;
    
    println!("Created LayerNorm with shape {:?}", normalized_shape);
    
    // Forward pass
    let output_ln = ln.forward(&input)?;
    println!("LayerNorm output shape: {:?}", output_ln.shape());
    
    // Test with different input shapes
    println!("\n=== Testing LayerNorm with different shapes ===");
    
    // 3D input: [batch=10, sequence=20, features=64]
    let input_3d = Tensor::randn(
        Shape::from_dims(&[10, 20, 64]),
        0.0, 1.0,
        device.clone()
    )?;
    
    let ln_3d = LayerNorm::new_with_affine(
        vec![64],  // Normalize over feature dimension
        1e-5,
        true,
        device.clone()
    )?;
    
    let output_3d = ln_3d.forward(&input_3d)?;
    println!("3D input shape: {:?} -> output: {:?}", 
             input_3d.shape(), output_3d.shape());
    
    // Verify LayerNorm statistics per sample
    let data_3d = output_3d.to_vec()?;
    let feature_size = 64;
    let num_samples = 10 * 20;
    
    let mut sample_means = Vec::new();
    let mut sample_vars = Vec::new();
    
    for i in 0..num_samples {
        let start = i * feature_size;
        let end = (i + 1) * feature_size;
        let sample = &data_3d[start..end];
        
        let mean = sample.iter().sum::<f32>() / feature_size as f32;
        let var = sample.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / feature_size as f32;
        
        sample_means.push(mean);
        sample_vars.push(var);
    }
    
    let avg_mean = sample_means.iter().sum::<f32>() / sample_means.len() as f32;
    let avg_var = sample_vars.iter().sum::<f32>() / sample_vars.len() as f32;
    
    println!("LayerNorm per-sample statistics:");
    println!("Average mean: {:.6}, Average variance: {:.6}", avg_mean, avg_var);
    println!("(Should be close to 0 and 1 respectively)");
    
    println!("\nNormalization demo completed successfully!");
    
    Ok(())
}