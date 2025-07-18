use flame_core as flame;
use candle_core as candle;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing FLAME and Candle tensor operations...\n");
    
    // Initialize devices
    let flame_device = flame::CudaDevice::new(0)?;
    let candle_device = candle::Device::new_cuda(0)?;
    
    // Test data
    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    
    // Create tensors
    println!("1. Creating tensors from same data:");
    let flame_a = flame::Tensor::from_vec(data_a.clone(), flame::Shape::from_dims(&[2, 3]), flame_device.clone())?;
    let flame_b = flame::Tensor::from_vec(data_b.clone(), flame::Shape::from_dims(&[2, 3]), flame_device.clone())?;
    
    let candle_a = candle::Tensor::from_vec(data_a.clone(), (2, 3), &candle_device)?;
    let candle_b = candle::Tensor::from_vec(data_b.clone(), (2, 3), &candle_device)?;
    
    println!("   ✓ Tensors created");
    
    // Test addition
    println!("\n2. Testing addition:");
    let flame_add = flame_a.add(&flame_b)?;
    let candle_add = (&candle_a + &candle_b)?;
    
    let flame_add_vec = flame_add.to_vec_f32()?;
    let candle_add_vec = candle_add.to_vec1::<f32>()?;
    
    println!("   FLAME result:  {:?}", flame_add_vec);
    println!("   Candle result: {:?}", candle_add_vec);
    assert_eq!(flame_add_vec.len(), candle_add_vec.len());
    for i in 0..flame_add_vec.len() {
        assert!((flame_add_vec[i] - candle_add_vec[i]).abs() < 1e-6, 
                "Mismatch at index {}: {} vs {}", i, flame_add_vec[i], candle_add_vec[i]);
    }
    println!("   ✓ Results match!");
    
    // Test multiplication
    println!("\n3. Testing multiplication:");
    let flame_mul = flame_a.mul(&flame_b)?;
    let candle_mul = (&candle_a * &candle_b)?;
    
    let flame_mul_vec = flame_mul.to_vec_f32()?;
    let candle_mul_vec = candle_mul.to_vec1::<f32>()?;
    
    println!("   FLAME result:  {:?}", flame_mul_vec);
    println!("   Candle result: {:?}", candle_mul_vec);
    for i in 0..flame_mul_vec.len() {
        assert!((flame_mul_vec[i] - candle_mul_vec[i]).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}", i, flame_mul_vec[i], candle_mul_vec[i]);
    }
    println!("   ✓ Results match!");
    
    // Test scalar multiplication
    println!("\n4. Testing scalar multiplication (x * 2.5):");
    let flame_scalar = flame_a.mul_scalar(2.5)?;
    let scalar_tensor = candle::Tensor::new(2.5f32, &candle_device)?;
    let candle_scalar = candle_a.broadcast_mul(&scalar_tensor)?;
    
    let flame_scalar_vec = flame_scalar.to_vec_f32()?;
    let candle_scalar_vec = candle_scalar.to_vec1::<f32>()?;
    
    println!("   FLAME result:  {:?}", flame_scalar_vec);
    println!("   Candle result: {:?}", candle_scalar_vec);
    for i in 0..flame_scalar_vec.len() {
        assert!((flame_scalar_vec[i] - candle_scalar_vec[i]).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}", i, flame_scalar_vec[i], candle_scalar_vec[i]);
    }
    println!("   ✓ Results match!");
    
    // Test ReLU
    println!("\n5. Testing ReLU activation:");
    let test_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let flame_relu_input = flame::Tensor::from_vec(test_data.clone(), flame::Shape::from_dims(&[2, 3]), flame_device.clone())?;
    let candle_relu_input = candle::Tensor::from_vec(test_data, (2, 3), &candle_device)?;
    
    let flame_relu = flame_relu_input.relu()?;
    let candle_relu = candle_relu_input.relu()?;
    
    let flame_relu_vec = flame_relu.to_vec_f32()?;
    let candle_relu_vec = candle_relu.to_vec1::<f32>()?;
    
    println!("   FLAME result:  {:?}", flame_relu_vec);
    println!("   Candle result: {:?}", candle_relu_vec);
    for i in 0..flame_relu_vec.len() {
        assert!((flame_relu_vec[i] - candle_relu_vec[i]).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}", i, flame_relu_vec[i], candle_relu_vec[i]);
    }
    println!("   ✓ Results match!");
    
    // Test matrix multiplication
    println!("\n6. Testing matrix multiplication:");
    let flame_x = flame::Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                                          flame::Shape::from_dims(&[2, 3]), flame_device.clone())?;
    let flame_y = flame::Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                                          flame::Shape::from_dims(&[3, 2]), flame_device.clone())?;
    
    let candle_x = candle::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &candle_device)?;
    let candle_y = candle::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &candle_device)?;
    
    let flame_matmul = flame_x.matmul(&flame_y)?;
    let candle_matmul = candle_x.matmul(&candle_y)?;
    
    let flame_matmul_vec = flame_matmul.to_vec_f32()?;
    let candle_matmul_vec = candle_matmul.to_vec1::<f32>()?;
    
    println!("   FLAME result:  {:?}", flame_matmul_vec);
    println!("   Candle result: {:?}", candle_matmul_vec);
    for i in 0..flame_matmul_vec.len() {
        assert!((flame_matmul_vec[i] - candle_matmul_vec[i]).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}", i, flame_matmul_vec[i], candle_matmul_vec[i]);
    }
    println!("   ✓ Results match!");
    
    println!("\n✓ All comparisons passed! FLAME and Candle produce identical results.");
    
    Ok(())
}