use flame_core::{Tensor, CudaDevice, Shape, conv::{Conv2d, Conv2dConfig}};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Conv2d with CUDA Kernels ===");
    
    // Test 1: Simple convolution
    let config = Conv2dConfig {
        in_channels: 3,
        out_channels: 16,
        kernel_size: (3, 3),
        stride: (1, 1),
        padding: (1, 1),
        groups: 1,
    };
    
    let conv = Conv2d::new(config, device.clone())?;
    
    // Create test input: [batch=2, channels=3, height=32, width=32]
    let input = Tensor::randn(
        Shape::from_dims(&[2, 3, 32, 32]),
        0.0, 1.0,
        device.clone()
    )?;
    
    println!("Input shape: {:?}", input.shape().dims());
    
    // Forward pass
    let output = conv.forward(&input)?;
    println!("Output shape: {:?} (expected [2, 16, 32, 32])", output.shape().dims());
    
    // Test 2: Different configurations
    println!("\n=== Testing various Conv2d configurations ===");
    
    // 5x5 kernel with stride 2
    let config2 = Conv2dConfig {
        in_channels: 16,
        out_channels: 32,
        kernel_size: (5, 5),
        stride: (2, 2),
        padding: (2, 2),
        groups: 1,
    };
    
    let conv2 = Conv2d::new(config2, device.clone())?;
    let output2 = conv2.forward(&output)?;
    println!("5x5 stride 2: {:?} -> {:?}", output.shape().dims(), output2.shape().dims());
    
    // 1x1 convolution (pointwise)
    let config3 = Conv2dConfig {
        in_channels: 32,
        out_channels: 64,
        kernel_size: (1, 1),
        stride: (1, 1),
        padding: (0, 0),
        groups: 1,
    };
    
    let conv3 = Conv2d::new(config3, device.clone())?;
    let output3 = conv3.forward(&output2)?;
    println!("1x1 pointwise: {:?} -> {:?}", output2.shape().dims(), output3.shape().dims());
    
    // Test 3: Performance test
    println!("\n=== Performance Test ===");
    
    // Larger input for performance testing
    let large_input = Tensor::randn(
        Shape::from_dims(&[8, 3, 224, 224]),  // ImageNet size
        0.0, 1.0,
        device.clone()
    )?;
    
    let perf_config = Conv2dConfig {
        in_channels: 3,
        out_channels: 64,
        kernel_size: (7, 7),
        stride: (2, 2),
        padding: (3, 3),
        groups: 1,
    };
    
    let perf_conv = Conv2d::new(perf_config, device.clone())?;
    
    // Warmup
    for _ in 0..10 {
        let _ = perf_conv.forward(&large_input)?;
    }
    
    // Benchmark
    let num_iterations = 100;
    let start = Instant::now();
    
    for _ in 0..num_iterations {
        let _ = perf_conv.forward(&large_input)?;
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / num_iterations;
    
    println!("Conv2d (3->64, 7x7, stride 2) on 8x3x224x224:");
    println!("  Total time for {} iterations: {:?}", num_iterations, elapsed);
    println!("  Average time per forward pass: {:?}", avg_time);
    
    // Test 4: Verify correctness with known pattern
    println!("\n=== Correctness Test ===");
    
    // Create a simple pattern
    let test_input = Tensor::from_vec(
        vec![1.0; 1 * 1 * 3 * 3],  // All ones
        Shape::from_dims(&[1, 1, 3, 3]),
        device.clone()
    )?;
    
    let simple_config = Conv2dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: (3, 3),
        stride: (1, 1),
        padding: (0, 0),
        groups: 1,
    };
    
    let mut simple_conv = Conv2d::new(simple_config, device)?;
    
    // Set weights to all ones
    simple_conv.weight = Tensor::from_vec(
        vec![1.0; 1 * 1 * 3 * 3],
        Shape::from_dims(&[1, 1, 3, 3]),
        simple_conv.weight.device().clone()
    )?;
    
    let test_output = simple_conv.forward(&test_input)?;
    let output_data = test_output.to_vec()?;
    
    println!("Simple test (all ones):");
    println!("  Input: 3x3 matrix of ones");
    println!("  Kernel: 3x3 matrix of ones");
    println!("  Output: {:?} (should be [9.0])", output_data);
    
    println!("\nAll Conv2d CUDA tests completed!");
    
    Ok(())
}