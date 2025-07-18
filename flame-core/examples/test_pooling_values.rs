use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::pooling::{MaxPool2d, MaxPool2dConfig, AvgPool2d, AvgPool2dConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame pooling layers with known values...");
    
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");
    
    // Create test input tensor with known values
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    ];
    
    let input = Tensor::from_vec(
        input_data,
        Shape::from_dims(&[1, 1, 4, 4]),
        device.clone()
    )?;
    println!("Input tensor created");
    
    // Test MaxPool2d with 2x2 kernel
    println!("\n--- Testing MaxPool2d (2x2) ---");
    let max_config = MaxPool2dConfig::new((2, 2));
    let max_pool = MaxPool2d::new(max_config);
    let (max_output, _) = max_pool.forward(&input)?;
    
    let output_vec = max_output.to_vec()?;
    println!("MaxPool2d output: {:?}", output_vec);
    println!("Expected: [6.0, 8.0, 14.0, 16.0]");
    
    // Verify output
    let expected = vec![6.0, 8.0, 14.0, 16.0];
    let all_correct = output_vec.iter().zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    println!("MaxPool2d test: {}", if all_correct { "PASSED" } else { "FAILED" });
    
    // Test AvgPool2d with 2x2 kernel
    println!("\n--- Testing AvgPool2d (2x2) ---");
    let avg_config = AvgPool2dConfig::new((2, 2));
    let avg_pool = AvgPool2d::new(avg_config);
    let avg_output = avg_pool.forward(&input)?;
    
    let avg_vec = avg_output.to_vec()?;
    println!("AvgPool2d output: {:?}", avg_vec);
    println!("Expected: [3.5, 5.5, 11.5, 13.5]");
    
    // Verify output
    let expected_avg = vec![3.5, 5.5, 11.5, 13.5];
    let all_correct_avg = avg_vec.iter().zip(expected_avg.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    println!("AvgPool2d test: {}", if all_correct_avg { "PASSED" } else { "FAILED" });
    
    println!("\nAll tests completed!");
    
    Ok(())
}