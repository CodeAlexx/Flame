use flame_core::{Tensor, Shape, CudaDevice, Result, AutogradContext};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("🔬 Verifying FLAME Core Operations...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Matrix multiplication
    {
        println!("Test 1: Matrix multiplication");
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?;
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[2, 2]), device.clone())?;
        let c = a.matmul(&b)?;
        
        let result = c.to_vec()?;
        println!("  Result: {:?}", result);
        println!("  Expected: [19.0, 22.0, 43.0, 50.0]");
        
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }
        println!("  ✅ Matrix multiplication passed!\n");
    }
    
    // Test 2: Addition with broadcasting
    {
        println!("Test 2: Addition with broadcasting");
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?;
        let b = Tensor::from_vec(vec![10.0], Shape::from_dims(&[1]), device.clone())?;
        let c = a.add(&b)?;
        
        let result = c.to_vec()?;
        println!("  Result: {:?}", result);
        println!("  Expected: [11.0, 12.0, 13.0, 14.0]");
        
        let expected = vec![11.0, 12.0, 13.0, 14.0];
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }
        println!("  ✅ Addition with broadcasting passed!\n");
    }
    
    // Test 3: ReLU activation
    {
        println!("Test 3: ReLU activation");
        let input = Tensor::from_vec(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            Shape::from_dims(&[5]),
            device.clone()
        )?;
        let output = input.relu()?;
        
        let result = output.to_vec()?;
        println!("  Result: {:?}", result);
        println!("  Expected: [0.0, 0.0, 0.0, 1.0, 2.0]");
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }
        println!("  ✅ ReLU activation passed!\n");
    }
    
    // Test 4: Gradient computation
    {
        println!("Test 4: Gradient computation");
        AutogradContext::reset(); // Clean state
        
        let x = Tensor::from_vec(vec![2.0, 3.0], Shape::from_dims(&[2]), device.clone())?.requires_grad_(true);
        let y = x.mul(&x)?;  // y = x²
        let loss = y.sum()?; // loss = sum(x²)
        
        let grads = loss.backward()?;
        let x_grad = grads.get(x.id()).unwrap();
        
        let grad_result = x_grad.to_vec()?;
        println!("  Input: [2.0, 3.0]");
        println!("  Gradient: {:?}", grad_result);
        println!("  Expected: [4.0, 6.0] (2x)");
        
        let expected = vec![4.0, 6.0];
        for (i, (r, e)) in grad_result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Gradient mismatch at {}: {} vs {}", i, r, e);
        }
        println!("  ✅ Gradient computation passed!\n");
    }
    
    // Test 5: Sum reduction
    {
        println!("Test 5: Sum reduction");
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), device.clone())?;
        let sum = a.sum()?;
        
        let result = sum.to_vec()?;
        println!("  Result: {:?}", result);
        println!("  Expected: [10.0]");
        
        assert_eq!(result.len(), 1);
        assert!((result[0] - 10.0).abs() < 1e-5);
        println!("  ✅ Sum reduction passed!\n");
    }
    
    // Test 6: Transpose
    {
        println!("Test 6: Transpose");
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            device.clone()
        )?;
        let transposed = a.transpose()?;
        
        let result = transposed.to_vec()?;
        println!("  Result shape: {:?}", transposed.shape().dims());
        println!("  Result: {:?}", result);
        println!("  Expected: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]");
        
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }
        println!("  ✅ Transpose passed!\n");
    }
    
    // Test 7: Mean operation
    {
        println!("Test 7: Mean operation");
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), device)?;
        let mean = a.mean()?;
        
        let result = mean.to_vec()?;
        println!("  Result: {:?}", result);
        println!("  Expected: [2.5]");
        
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.5).abs() < 1e-5);
        println!("  ✅ Mean operation passed!\n");
    }
    
    println!("🎉 ALL CORE OPERATIONS VERIFIED! 🎉");
    println!("\nFLAME has a solid foundation of working operations:");
    println!("- Matrix multiplication ✅");
    println!("- Addition with broadcasting ✅");
    println!("- ReLU activation ✅");
    println!("- Gradient computation ✅");
    println!("- Sum reduction ✅");
    println!("- Transpose ✅");
    println!("- Mean operation ✅");
    
    Ok(())
}