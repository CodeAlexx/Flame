// Comprehensive FLAME Tensor Functionality Tests

#[cfg(test)]
mod tests {
    use flame_core::{Tensor, Shape, FlameError, Result, CudaDevice};
    use std::sync::Arc;

    // Helper function to create a test device
    fn create_test_device() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("Failed to create CUDA device")
    }

    // Helper function to compare tensors with tolerance
    fn assert_tensor_eq(a: &Tensor, b: &Tensor, tolerance: f32) -> Result<()> {
        let a_vec = a.to_vec()?;
        let b_vec = b.to_vec()?;
        
        assert_eq!(a_vec.len(), b_vec.len(), "Tensors have different sizes");
        
        for (i, (a_val, b_val)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
            let diff = (a_val - b_val).abs();
            assert!(
                diff < tolerance,
                "Values differ at index {}: {} vs {} (diff: {})",
                i, a_val, b_val, diff
            );
        }
        
        Ok(())
    }

    #[test]
    fn test_tensor_creation_zeros() -> Result<()> {
        let device = create_test_device();
        let shape = Shape::from_dims(&[2, 3]);
        
        let tensor = Tensor::zeros(shape.clone(), device)?;
        
        assert_eq!(tensor.shape(), &shape);
        
        let data = tensor.to_vec()?;
        assert_eq!(data.len(), 6);
        assert!(data.iter().all(|&x| x == 0.0));
        
        Ok(())
    }

    #[test]
    fn test_tensor_creation_ones() -> Result<()> {
        let device = create_test_device();
        let shape = Shape::from_dims(&[3, 4]);
        
        let tensor = Tensor::ones(shape.clone(), device)?;
        
        assert_eq!(tensor.shape(), &shape);
        
        let data = tensor.to_vec()?;
        assert_eq!(data.len(), 12);
        assert!(data.iter().all(|&x| x == 1.0));
        
        Ok(())
    }

    #[test]
    fn test_tensor_creation_randn() -> Result<()> {
        let device = create_test_device();
        let shape = Shape::from_dims(&[5, 5]);
        let mean = 0.0;
        let std = 1.0;
        
        let tensor = Tensor::randn(shape.clone(), mean, std, device)?;
        
        assert_eq!(tensor.shape(), &shape);
        
        let data = tensor.to_vec()?;
        assert_eq!(data.len(), 25);
        
        // Check that values are not all the same (randomness)
        let first = data[0];
        assert!(!data.iter().all(|&x| x == first));
        
        // Check rough statistics (mean should be close to 0, std close to 1)
        let sum: f32 = data.iter().sum();
        let mean_actual = sum / data.len() as f32;
        assert!(mean_actual.abs() < 0.5, "Mean {} is too far from 0", mean_actual);
        
        Ok(())
    }

    #[test]
    fn test_tensor_from_vec() -> Result<()> {
        let device = create_test_device();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = Shape::from_dims(&[2, 3]);
        
        let tensor = Tensor::from_vec(data.clone(), shape.clone(), device)?;
        
        assert_eq!(tensor.shape(), &shape);
        
        let retrieved_data = tensor.to_vec()?;
        assert_eq!(retrieved_data, data);
        
        Ok(())
    }

    #[test]
    fn test_tensor_from_slice() -> Result<()> {
        let device = create_test_device();
        let data = [1.0, 2.0, 3.0, 4.0];
        let shape = Shape::from_dims(&[2, 2]);
        
        let tensor = Tensor::from_slice(&data, shape.clone(), device)?;
        
        assert_eq!(tensor.shape(), &shape);
        
        let retrieved_data = tensor.to_vec()?;
        assert_eq!(retrieved_data, data.to_vec());
        
        Ok(())
    }

    #[test]
    fn test_tensor_add() -> Result<()> {
        let device = create_test_device();
        let shape = Shape::from_dims(&[2, 2]);
        
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], shape.clone(), device.clone())?;
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], shape.clone(), device.clone())?;
        
        let c = a.add(&b)?;
        
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        let result = c.to_vec()?;
        assert_eq!(result, expected);
        
        Ok(())
    }

    #[test]
    fn test_tensor_sub() -> Result<()> {
        let device = create_test_device();
        let shape = Shape::from_dims(&[2, 2]);
        
        let a = Tensor::from_vec(vec![10.0, 8.0, 6.0, 4.0], shape.clone(), device.clone())?;
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], shape.clone(), device.clone())?;
        
        let c = a.sub(&b)?;
        
        let expected = vec![9.0, 6.0, 3.0, 0.0];
        let result = c.to_vec()?;
        assert_eq!(result, expected);
        
        Ok(())
    }

    #[test]
    fn test_tensor_mul() -> Result<()> {
        let device = create_test_device();
        let shape = Shape::from_dims(&[2, 3]);
        
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape.clone(), device.clone())?;
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0], shape.clone(), device.clone())?;
        
        let c = a.mul(&b)?;
        
        let expected = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let result = c.to_vec()?;
        assert_eq!(result, expected);
        
        Ok(())
    }

    #[test]
    fn test_tensor_reshape() -> Result<()> {
        let device = create_test_device();
        let original_shape = Shape::from_dims(&[2, 3]);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let tensor = Tensor::from_vec(data.clone(), original_shape, device)?;
        
        // Reshape to 3x2
        let reshaped = tensor.reshape(&[3, 2])?;
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        
        let reshaped_data = reshaped.to_vec()?;
        assert_eq!(reshaped_data, data);
        
        // Reshape to 1D
        let flattened = tensor.reshape(&[6])?;
        assert_eq!(flattened.shape().dims(), &[6]);
        
        let flattened_data = flattened.to_vec()?;
        assert_eq!(flattened_data, data);
        
        Ok(())
    }

    #[test]
    fn test_tensor_transpose() -> Result<()> {
        let device = create_test_device();
        let shape = Shape::from_dims(&[2, 3]);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let tensor = Tensor::from_vec(data, shape, device)?;
        let transposed = tensor.transpose()?;
        
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        
        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let result = transposed.to_vec()?;
        assert_eq!(result, expected);
        
        Ok(())
    }

    #[test]
    fn test_tensor_clone() -> Result<()> {
        let device = create_test_device();
        let shape = Shape::from_dims(&[2, 2]);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        
        let original = Tensor::from_vec(data.clone(), shape, device)?;
        let cloned = original.clone();
        
        // Check that shapes are equal
        assert_eq!(original.shape(), cloned.shape());
        
        // Check that data is equal
        let original_data = original.to_vec()?;
        let cloned_data = cloned.to_vec()?;
        assert_eq!(original_data, cloned_data);
        
        Ok(())
    }

    #[test]
    fn test_shape_mismatch_error() -> Result<()> {
        let device = create_test_device();
        
        let a = Tensor::from_vec(vec![1.0, 2.0], Shape::from_dims(&[2]), device.clone())?;
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], Shape::from_dims(&[3]), device.clone())?;
        
        // This should fail due to shape mismatch
        let result = a.add(&b);
        assert!(result.is_err());
        
        Ok(())
    }

    #[test]
    fn test_device_consistency() -> Result<()> {
        let device1 = create_test_device();
        let device2 = device1.clone(); // Same device through Arc
        
        let tensor1 = Tensor::zeros(Shape::from_dims(&[2, 2]), device1)?;
        let tensor2 = Tensor::ones(Shape::from_dims(&[2, 2]), device2)?;
        
        // Operations between tensors on the same device should work
        let result = tensor1.add(&tensor2)?;
        let data = result.to_vec()?;
        assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0]);
        
        Ok(())
    }
}