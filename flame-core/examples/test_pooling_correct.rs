use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::pooling::{
    MaxPool2d, MaxPool2dConfig,
    AvgPool2d, AvgPool2dConfig,
    AdaptiveAvgPool2d, GlobalAvgPool2d,
    AdaptiveMaxPool2d, GlobalMaxPool2d,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame pooling layers...");
    
    // Initialize CUDA device - CudaDevice::new already returns Arc<CudaDevice>
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");
    
    // Create test input tensor [batch=2, channels=3, height=8, width=8]
    let input = Tensor::randn(
        Shape::from_dims(&[2, 3, 8, 8]),
        0.0,
        1.0,
        device.clone()
    )?;
    println!("Input shape: {:?}", input.shape().dims());
    
    // Test MaxPool2d
    println!("\n--- Testing MaxPool2d ---");
    let max_config = MaxPool2dConfig::new((2, 2));
    let max_pool = MaxPool2d::new(max_config);
    let (max_output, indices) = max_pool.forward(&input)?;
    println!("MaxPool2d output shape: {:?}", max_output.shape().dims());
    println!("Has indices: {}", indices.is_some());
    
    // Test with stride and padding
    let mut max_config2 = MaxPool2dConfig::new((3, 3));
    max_config2.stride = Some((2, 2));
    max_config2.padding = (1, 1);
    max_config2.return_indices = true;
    let max_pool2 = MaxPool2d::new(max_config2);
    let (max_output2, indices2) = max_pool2.forward(&input)?;
    println!("MaxPool2d with stride/padding output shape: {:?}", max_output2.shape().dims());
    println!("Has indices: {}", indices2.is_some());
    
    // Test AvgPool2d
    println!("\n--- Testing AvgPool2d ---");
    let avg_config = AvgPool2dConfig::new((2, 2));
    let avg_pool = AvgPool2d::new(avg_config);
    let avg_output = avg_pool.forward(&input)?;
    println!("AvgPool2d output shape: {:?}", avg_output.shape().dims());
    
    // Test with different parameters
    let mut avg_config2 = AvgPool2dConfig::new((3, 3));
    avg_config2.stride = Some((1, 1));
    avg_config2.padding = (1, 1);
    avg_config2.count_include_pad = false;
    let avg_pool2 = AvgPool2d::new(avg_config2);
    let avg_output2 = avg_pool2.forward(&input)?;
    println!("AvgPool2d with stride/padding output shape: {:?}", avg_output2.shape().dims());
    
    // Test AdaptiveAvgPool2d
    println!("\n--- Testing AdaptiveAvgPool2d ---");
    let adaptive_avg = AdaptiveAvgPool2d::new((4, 4));
    let adaptive_avg_output = adaptive_avg.forward(&input)?;
    println!("AdaptiveAvgPool2d output shape: {:?}", adaptive_avg_output.shape().dims());
    
    // Test GlobalAvgPool2d
    println!("\n--- Testing GlobalAvgPool2d ---");
    let global_avg = GlobalAvgPool2d::new();
    let global_avg_output = global_avg.forward(&input)?;
    println!("GlobalAvgPool2d output shape: {:?}", global_avg_output.shape().dims());
    
    // Test AdaptiveMaxPool2d
    println!("\n--- Testing AdaptiveMaxPool2d ---");
    let adaptive_max = AdaptiveMaxPool2d::new((2, 2));
    let (adaptive_max_output, _adaptive_indices) = adaptive_max.forward(&input)?;
    println!("AdaptiveMaxPool2d output shape: {:?}", adaptive_max_output.shape().dims());
    
    // Test GlobalMaxPool2d
    println!("\n--- Testing GlobalMaxPool2d ---");
    let global_max = GlobalMaxPool2d::new();
    let (global_max_output, _global_indices) = global_max.forward(&input)?;
    println!("GlobalMaxPool2d output shape: {:?}", global_max_output.shape().dims());
    
    // Test backward pass
    println!("\n--- Testing backward passes ---");
    let grad_output = Tensor::ones(max_output.shape().clone(), device.clone())?;
    let grad_input = max_pool.backward(&grad_output, &input, indices.as_ref())?;
    println!("MaxPool2d backward output shape: {:?}", grad_input.shape().dims());
    
    let grad_output2 = Tensor::ones(avg_output.shape().clone(), device.clone())?;
    let grad_input2 = avg_pool.backward(&grad_output2, &input)?;
    println!("AvgPool2d backward output shape: {:?}", grad_input2.shape().dims());
    
    println!("\nAll pooling tests completed successfully!");
    
    Ok(())
}