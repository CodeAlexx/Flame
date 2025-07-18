#[cfg(test)]
mod tests {
    use flame_core::{Tensor, CudaDevice, Shape};

    #[test]
    fn test_weight_update_actually_changes_weights() {
        // Initialize CUDA
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        
        // Create weight tensor with known values
        let initial_values = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut weight = Tensor::from_vec(
            initial_values.clone(),
            Shape::from_dims(&[2, 2]),
            device.clone()
        ).expect("Failed to create weight tensor");

        // Create gradient tensor
        let gradient_values = vec![0.1f32, 0.2, 0.3, 0.4];
        let gradient = Tensor::from_vec(
            gradient_values.clone(),
            Shape::from_dims(&[2, 2]),
            device.clone()
        ).expect("Failed to create gradient tensor");

        // Update weights
        let lr = 0.1;
        weight.update_weights(&gradient, lr).expect("Failed to update weights");

        // Check weights changed correctly
        let updated_weights = weight.to_vec().expect("Failed to copy to CPU");
        
        // Expected: weight - lr * gradient
        let expected = vec![
            1.0 - 0.1 * 0.1,  // 0.99
            2.0 - 0.1 * 0.2,  // 1.98
            3.0 - 0.1 * 0.3,  // 2.97
            4.0 - 0.1 * 0.4,  // 3.96
        ];

        for (i, (actual, expected)) in updated_weights.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Weight update incorrect at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_multiple_weight_updates_accumulate() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        
        // Start with zeros
        let mut weight = Tensor::zeros(Shape::from_dims(&[2, 2]), device.clone())
            .expect("Failed to create weight tensor");

        // Apply multiple updates
        let gradient = Tensor::from_vec(
            vec![1.0f32, 1.0, 1.0, 1.0],
            Shape::from_dims(&[2, 2]),
            device.clone()
        ).expect("Failed to create gradient");

        let lr = 0.1;
        for _ in 0..10 {
            weight.update_weights(&gradient, lr).expect("Update failed");
        }

        // After 10 updates of -0.1, weights should be -1.0
        let final_weights = weight.to_vec().expect("Failed to copy to CPU");
        for (i, &value) in final_weights.iter().enumerate() {
            assert!(
                (value - (-1.0)).abs() < 1e-6,
                "Weight at index {} should be -1.0, got {}",
                i, value
            );
        }
    }

    #[test]
    fn test_shape_mismatch_fails() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        
        let mut weight = Tensor::zeros(Shape::from_dims(&[2, 3]), device.clone())
            .expect("Failed to create weight");
        let gradient = Tensor::zeros(Shape::from_dims(&[3, 2]), device.clone())
            .expect("Failed to create gradient");

        // This should fail due to shape mismatch
        assert!(weight.update_weights(&gradient, 0.1).is_err());
    }

    #[test]
    fn test_matmul_correctness() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        
        // Simple 2x2 matmul
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::from_dims(&[2, 2]),
            device.clone()
        ).expect("Failed to create tensor A");

        let b = Tensor::from_vec(
            vec![5.0, 6.0, 7.0, 8.0],
            Shape::from_dims(&[2, 2]),
            device.clone()
        ).expect("Failed to create tensor B");

        let c = a.matmul(&b).expect("Matmul failed");
        let result = c.to_vec().expect("Failed to copy result");

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        for (i, (actual, expected)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Matmul incorrect at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_transpose_correctness() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        
        // Test 2x3 matrix transpose
        let mat = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            device.clone()
        ).expect("Failed to create matrix");
        
        let transposed = mat.transpose().expect("Transpose failed");
        
        // Check shape
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        
        // Check values
        let result = transposed.to_vec().expect("Failed to copy to CPU");
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        
        for (i, (actual, expected)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Transpose incorrect at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }
    
    #[test]
    fn test_zero_learning_rate_no_change() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        
        let initial = vec![1.0, 2.0, 3.0, 4.0];
        let mut weight = Tensor::from_vec(
            initial.clone(),
            Shape::from_dims(&[2, 2]),
            device.clone()
        ).expect("Failed to create weight");

        let gradient = Tensor::from_vec(
            vec![10.0, 20.0, 30.0, 40.0],
            Shape::from_dims(&[2, 2]),
            device.clone()
        ).expect("Failed to create gradient");

        // Update with lr=0
        weight.update_weights(&gradient, 0.0).expect("Update failed");

        // Weights should not change
        let final_weights = weight.to_vec().expect("Failed to copy");
        for (i, (actual, expected)) in final_weights.iter().zip(initial.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Weight changed at index {} with lr=0: expected {}, got {}",
                i, expected, actual
            );
        }
    }
}