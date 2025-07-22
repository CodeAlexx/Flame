use flame_core::{Tensor, Shape, CudaDevice, Result};
use flame_core::pooling::{MaxPool2d, MaxPool2dConfig, AvgPool2d, AvgPool2dConfig};

fn main() -> Result<()> {
    println!("🔬 Testing FLAME Pooling Operations...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: 2D Max Pooling
    {
        println!("Test 1: 2D Max Pooling");
        
        // Create a 4x4 input with known values
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        
        let input = Tensor::from_vec(
            input_data,
            Shape::from_dims(&[1, 1, 4, 4]), // [batch, channels, height, width]
            device.clone()
        )?;
        
        // Test 2x2 kernel with stride 2 (no overlap)
        let pool_config = MaxPool2dConfig {
            kernel_size: (2, 2),
            stride: Some((2, 2)),
            padding: (0, 0),
            dilation: (1, 1),
            return_indices: false,
        };
        let pool = MaxPool2d::new(pool_config);
        let (output, _indices) = pool.forward(&input)?;
        
        // Expected output: [[6, 8], [14, 16]]
        let expected = vec![6.0, 8.0, 14.0, 16.0];
        let output_data = output.to_vec()?;
        
        println!("  Input shape: {:?}", input.shape().dims());
        println!("  Output shape: {:?}", output.shape().dims());
        println!("  Output values: {:?}", output_data);
        
        for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-5, 
                "Max pooling mismatch at index {}: got {}, expected {}", i, actual, expected);
        }
        
        println!("  ✅ 2D Max pooling passed!\n");
    }
    
    // Test 2: 2D Average Pooling
    {
        println!("Test 2: 2D Average Pooling");
        
        // Create a 4x4 input
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        
        let input = Tensor::from_vec(
            input_data,
            Shape::from_dims(&[1, 1, 4, 4]),
            device.clone()
        )?;
        
        // Test 2x2 kernel with stride 2
        let pool_config = AvgPool2dConfig {
            kernel_size: (2, 2),
            stride: Some((2, 2)),
            padding: (0, 0),
            count_include_pad: true,
            divisor_override: None,
        };
        let pool = AvgPool2d::new(pool_config);
        let output = pool.forward(&input)?;
        
        // Expected output: [[3.5, 5.5], [11.5, 13.5]]
        let expected = vec![3.5, 5.5, 11.5, 13.5];
        let output_data = output.to_vec()?;
        
        println!("  Output values: {:?}", output_data);
        
        for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-5, 
                "Avg pooling mismatch at index {}: got {}, expected {}", i, actual, expected);
        }
        
        println!("  ✅ 2D Average pooling passed!\n");
    }
    
    // Test 3: Max Pooling with different strides
    {
        println!("Test 3: Max Pooling with stride != kernel_size");
        
        // Create a 5x5 input
        let mut input_data = Vec::new();
        for i in 0..25 {
            input_data.push(i as f32);
        }
        
        let input = Tensor::from_vec(
            input_data,
            Shape::from_dims(&[1, 1, 5, 5]),
            device.clone()
        )?;
        
        // 3x3 kernel with stride 1
        let pool_config = MaxPool2dConfig {
            kernel_size: (3, 3),
            stride: Some((1, 1)),
            padding: (0, 0),
            dilation: (1, 1),
            return_indices: false,
        };
        let pool = MaxPool2d::new(pool_config);
        let (output, _indices) = pool.forward(&input)?;
        
        println!("  Input shape: {:?}", input.shape().dims());
        println!("  Output shape: {:?}", output.shape().dims());
        
        // The output should be 3x3
        assert_eq!(output.shape().dims(), &[1, 1, 3, 3]);
        
        // Check center value (should be max of 3x3 region centered at (2,2))
        let output_data = output.to_vec()?;
        println!("  Output data: {:?}", output_data);
        // For 5x5 input [0-24], with 3x3 kernel and stride 1:
        // The center output position (1,1) looks at input region (1:4, 1:4)
        // Which contains values: 6,7,8,11,12,13,16,17,18
        // Max value is 18
        assert_eq!(output_data[4], 18.0, "Center value should be 18");
        
        println!("  ✅ Different stride pooling passed!\n");
    }
    
    // Test 4: Max Pooling with padding
    {
        println!("Test 4: Max Pooling with padding");
        
        // Create a 3x3 input
        let input_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        
        let input = Tensor::from_vec(
            input_data,
            Shape::from_dims(&[1, 1, 3, 3]),
            device.clone()
        )?;
        
        // 3x3 kernel with stride 2 and padding 1
        let pool_config = MaxPool2dConfig {
            kernel_size: (3, 3),
            stride: Some((2, 2)),
            padding: (1, 1),
            dilation: (1, 1),
            return_indices: false,
        };
        let pool = MaxPool2d::new(pool_config);
        let (output, _indices) = pool.forward(&input)?;
        
        println!("  Input shape: {:?}", input.shape().dims());
        println!("  Output shape: {:?}", output.shape().dims());
        
        // With padding, output should be 2x2
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        
        println!("  ✅ Padded pooling passed!\n");
    }
    
    // Test 5: Batch pooling
    {
        println!("Test 5: Batch pooling");
        
        // Create input with batch size 2
        let mut input_data = Vec::new();
        for _ in 0..2 {  // 2 batches
            for i in 0..16 {  // 4x4 each
                input_data.push(i as f32);
            }
        }
        
        let input = Tensor::from_vec(
            input_data,
            Shape::from_dims(&[2, 1, 4, 4]),
            device.clone()
        )?;
        
        // 2x2 kernel with stride 2
        let pool_config = MaxPool2dConfig {
            kernel_size: (2, 2),
            stride: Some((2, 2)),
            padding: (0, 0),
            dilation: (1, 1),
            return_indices: false,
        };
        let pool = MaxPool2d::new(pool_config);
        let (output, _indices) = pool.forward(&input)?;
        
        println!("  Input shape: {:?}", input.shape().dims());
        println!("  Output shape: {:?}", output.shape().dims());
        
        // Output should be [2, 1, 2, 2]
        assert_eq!(output.shape().dims(), &[2, 1, 2, 2]);
        
        // Both batches should have same pooling result
        let output_data = output.to_vec()?;
        let batch1 = &output_data[0..4];
        let batch2 = &output_data[4..8];
        
        for i in 0..4 {
            assert_eq!(batch1[i], batch2[i], "Batch results should be identical");
        }
        
        println!("  ✅ Batch pooling passed!\n");
    }
    
    // Test 6: Multi-channel pooling
    {
        println!("Test 6: Multi-channel pooling");
        
        // Create input with 3 channels
        let mut input_data = Vec::new();
        for c in 0..3 {  // 3 channels
            for i in 0..16 {  // 4x4 each
                input_data.push((c * 16 + i) as f32);
            }
        }
        
        let input = Tensor::from_vec(
            input_data,
            Shape::from_dims(&[1, 3, 4, 4]),
            device.clone()
        )?;
        
        // 2x2 kernel with stride 2
        let pool_config = MaxPool2dConfig {
            kernel_size: (2, 2),
            stride: Some((2, 2)),
            padding: (0, 0),
            dilation: (1, 1),
            return_indices: false,
        };
        let pool = MaxPool2d::new(pool_config);
        let (output, _indices) = pool.forward(&input)?;
        
        println!("  Input shape: {:?}", input.shape().dims());
        println!("  Output shape: {:?}", output.shape().dims());
        
        // Output should be [1, 3, 2, 2]
        assert_eq!(output.shape().dims(), &[1, 3, 2, 2]);
        
        println!("  ✅ Multi-channel pooling passed!\n");
    }
    
    // Test 7: 1D Pooling (if implemented)
    if false {  // Skip if not implemented
        println!("Test 7: 1D Pooling");
        
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::from_dims(&[1, 1, 8]),  // [batch, channels, length]
            device.clone()
        )?;
        
        // Test would go here
        println!("  ⚠️ 1D pooling not tested\n");
    }
    
    // Test 8: Adaptive pooling (if implemented)
    if false {  // Skip if not implemented
        println!("Test 8: Adaptive Average Pooling");
        
        // Adaptive pooling adjusts kernel size to produce specified output size
        let input = Tensor::randn(
            Shape::from_dims(&[1, 1, 7, 7]),
            0.0, 1.0, device.clone()
        )?;
        
        // Test would go here
        println!("  ⚠️ Adaptive pooling not tested\n");
    }
    
    println!("🎉 ALL POOLING TESTS PASSED! 🎉");
    
    Ok(())
}