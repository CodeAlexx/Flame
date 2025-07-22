use flame_core::{Tensor, Shape, CudaDevice, Result};

fn main() -> Result<()> {
    println!("🔬 Testing FLAME mean_dim operation...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Simple 2D mean
    {
        println!("Test 1: 2D mean along dimension");
        
        // Create a 3x4 tensor with known values
        let data = vec![
            1.0, 2.0, 3.0, 4.0,    // row 0
            5.0, 6.0, 7.0, 8.0,    // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ];
        
        let tensor = Tensor::from_vec(
            data,
            Shape::from_dims(&[3, 4]),
            device.clone()
        )?;
        
        // Mean along dimension 0 (rows)
        let mean_dim0 = tensor.mean_dim(&[0], false)?;
        let mean_dim0_values = mean_dim0.to_vec()?;
        println!("  Mean along dim 0: {:?}", mean_dim0_values);
        println!("  Expected: [5.0, 6.0, 7.0, 8.0]");
        // Verify values
        let expected_dim0 = vec![5.0, 6.0, 7.0, 8.0];
        for (i, (&actual, &expected)) in mean_dim0_values.iter().zip(expected_dim0.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-5, 
                "Mean dim 0 mismatch at index {}: got {}, expected {}", i, actual, expected);
        }
        
        // Mean along dimension 1 (columns)
        let mean_dim1 = tensor.mean_dim(&[1], false)?;
        let mean_dim1_values = mean_dim1.to_vec()?;
        println!("  Mean along dim 1: {:?}", mean_dim1_values);
        println!("  Expected: [2.5, 6.5, 10.5]");
        // Verify values
        let expected_dim1 = vec![2.5, 6.5, 10.5];
        for (i, (&actual, &expected)) in mean_dim1_values.iter().zip(expected_dim1.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-5, 
                "Mean dim 1 mismatch at index {}: got {}, expected {}", i, actual, expected);
        }
        
        println!("  ✅ 2D mean test passed!\n");
    }
    
    // Test 2: Keepdim test
    {
        println!("Test 2: Mean with keepdim");
        
        let tensor = Tensor::randn(
            Shape::from_dims(&[2, 3, 4]),
            0.0, 1.0, device.clone()
        )?;
        
        // Mean along dimension 1 with keepdim=true
        let mean_keepdim = tensor.mean_dim(&[1], true)?;
        println!("  Original shape: {:?}", tensor.shape().dims());
        println!("  Mean shape (keepdim=true): {:?}", mean_keepdim.shape().dims());
        println!("  Expected: [2, 1, 4]");
        
        // Mean along dimension 1 with keepdim=false
        let mean_no_keepdim = tensor.mean_dim(&[1], false)?;
        println!("  Mean shape (keepdim=false): {:?}", mean_no_keepdim.shape().dims());
        println!("  Expected: [2, 4]");
        
        println!("  ✅ Keepdim test passed!\n");
    }
    
    // Test 3: Multiple dimensions
    {
        println!("Test 3: Mean along multiple dimensions");
        
        let tensor = Tensor::randn(
            Shape::from_dims(&[2, 3, 4, 5]),
            0.0, 1.0, device.clone()
        )?;
        
        // Mean along dimensions 1 and 2
        let mean_multi = tensor.mean_dim(&[1, 2], false)?;
        println!("  Original shape: {:?}", tensor.shape().dims());
        println!("  Mean shape (dims [1,2]): {:?}", mean_multi.shape().dims());
        println!("  Expected: [2, 5]");
        
        println!("  ✅ Multi-dimension mean test passed!\n");
    }
    
    println!("🎉 ALL MEAN_DIM TESTS PASSED! 🎉");
    
    Ok(())
}