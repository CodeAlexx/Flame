#[cfg(test)]
mod tensor_operations_tests {
    use super::*;
    use std::collections::HashMap;

    // Helper function for approximate equality
    fn assert_tensors_close(a: &Tensor, b: &Tensor, tolerance: f32) {
        assert_eq!(a.shape(), b.shape(), "Tensor shapes don't match");
        let diff = (a - b).abs().max().item::<f32>();
        assert!(diff < tolerance, "Tensors differ by {}, tolerance {}", diff, tolerance);
    }

    fn assert_tensor_finite(tensor: &Tensor) {
        let data = tensor.flatten().to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()), "Tensor contains NaN or Inf values");
    }

    #[test]
    fn test_basic_math_operations() {
        let device = Device::Cuda(0);
        let a = Tensor::ones(&[2, 3], device).unwrap();
        let b = Tensor::full(&[2, 3], 2.0, device).unwrap();

        // Addition
        let add_result = &a + &b;
        let expected_add = Tensor::full(&[2, 3], 3.0, device).unwrap();
        assert_tensors_close(&add_result, &expected_add, 1e-6);

        // Subtraction
        let sub_result = &b - &a;
        let expected_sub = Tensor::full(&[2, 3], 1.0, device).unwrap();
        assert_tensors_close(&sub_result, &expected_sub, 1e-6);

        // Multiplication
        let mul_result = &a * &b;
        let expected_mul = Tensor::full(&[2, 3], 2.0, device).unwrap();
        assert_tensors_close(&mul_result, &expected_mul, 1e-6);

        // Division
        let div_result = &b / &a;
        let expected_div = Tensor::full(&[2, 3], 2.0, device).unwrap();
        assert_tensors_close(&div_result, &expected_div, 1e-6);
    }

    #[test]
    fn test_matrix_operations() {
        let device = Device::Cuda(0);
        let a = Tensor::randn(&[3, 4], device).unwrap();
        let b = Tensor::randn(&[4, 5], device).unwrap();

        // Matrix multiplication
        let matmul_result = a.matmul(&b).unwrap();
        assert_eq!(matmul_result.shape(), &[3, 5]);
        assert_tensor_finite(&matmul_result);

        // Transpose
        let a_t = a.transpose(-2, -1).unwrap();
        assert_eq!(a_t.shape(), &[4, 3]);

        // Transpose should be reversible
        let a_t_t = a_t.transpose(-2, -1).unwrap();
        assert_tensors_close(&a, &a_t_t, 1e-6);
    }

    #[test]
    fn test_shape_operations() {
        let device = Device::Cuda(0);
        let original = Tensor::randn(&[2, 3, 4], device).unwrap();

        // Reshape
        let reshaped = original.reshape(&[6, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[6, 4]);

        // View (should be same as reshape for contiguous tensors)
        let viewed = original.view(&[24]).unwrap();
        assert_eq!(viewed.shape(), &[24]);

        // Squeeze (remove dimensions of size 1)
        let with_singleton = original.unsqueeze(1).unwrap(); // [2, 1, 3, 4]
        let squeezed = with_singleton.squeeze(1).unwrap();
        assert_eq!(squeezed.shape(), original.shape());

        // Unsqueeze (add dimension of size 1)
        let unsqueezed = original.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_indexing_operations() {
        let device = Device::Cuda(0);
        let tensor = Tensor::arange(0.0, 24.0, device).unwrap().reshape(&[2, 3, 4]).unwrap();

        // Slice along first dimension
        let slice_0 = tensor.slice(0, 0, 1).unwrap();
        assert_eq!(slice_0.shape(), &[1, 3, 4]);

        // Slice along last dimension
        let slice_last = tensor.slice(-1, 0, 2).unwrap();
        assert_eq!(slice_last.shape(), &[2, 3, 2]);

        // Select (remove a dimension)
        let selected = tensor.select(0, 0).unwrap();
        assert_eq!(selected.shape(), &[3, 4]);

        // Index select with indices
        let indices = Tensor::from_slice(&[0i64, 2], device).unwrap();
        let indexed = tensor.index_select(1, &indices).unwrap();
        assert_eq!(indexed.shape(), &[2, 2, 4]);
    }

    #[test]
    fn test_reduction_operations() {
        let device = Device::Cuda(0);
        let tensor = Tensor::ones(&[2, 3, 4], device).unwrap();

        // Sum all elements
        let sum_all = tensor.sum().unwrap();
        assert_eq!(sum_all.shape(), &[]);
        assert_eq!(sum_all.item::<f32>(), 24.0);

        // Sum along specific dimension
        let sum_dim0 = tensor.sum_dim(0, false).unwrap();
        assert_eq!(sum_dim0.shape(), &[3, 4]);

        // Sum with keepdim
        let sum_keepdim = tensor.sum_dim(1, true).unwrap();
        assert_eq!(sum_keepdim.shape(), &[2, 1, 4]);

        // Mean
        let mean_all = tensor.mean().unwrap();
        assert_eq!(mean_all.item::<f32>(), 1.0);

        // Max and Min
        let max_val = tensor.max().unwrap();
        let min_val = tensor.min().unwrap();
        assert_eq!(max_val.item::<f32>(), 1.0);
        assert_eq!(min_val.item::<f32>(), 1.0);
    }

    #[test]
    fn test_broadcasting() {
        let device = Device::Cuda(0);
        let a = Tensor::ones(&[3, 1], device).unwrap();
        let b = Tensor::full(&[1, 4], 2.0, device).unwrap();

        // Broadcasting addition
        let result = &a + &b;
        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(result.get(&[0, 0]).unwrap().item::<f32>(), 3.0);

        // Broadcasting with scalar
        let scalar_result = &a + 5.0;
        assert_eq!(scalar_result.shape(), &[3, 1]);
        assert_eq!(scalar_result.get(&[0, 0]).unwrap().item::<f32>(), 6.0);
    }

    #[test]
    fn test_device_transfers() {
        let cpu_device = Device::Cpu;
        let gpu_device = Device::Cuda(0);

        // Create tensor on CPU
        let cpu_tensor = Tensor::randn(&[2, 3], cpu_device).unwrap();
        assert_eq!(cpu_tensor.device(), cpu_device);

        // Transfer to GPU
        let gpu_tensor = cpu_tensor.to_device(gpu_device).unwrap();
        assert_eq!(gpu_tensor.device(), gpu_device);
        assert_tensors_close(&cpu_tensor, &gpu_tensor, 1e-6);

        // Transfer back to CPU
        let cpu_tensor2 = gpu_tensor.to_device(cpu_device).unwrap();
        assert_eq!(cpu_tensor2.device(), cpu_device);
        assert_tensors_close(&cpu_tensor, &cpu_tensor2, 1e-6);
    }

    #[test]
    fn test_data_type_conversions() {
        let device = Device::Cuda(0);
        let f32_tensor = Tensor::randn(&[2, 3], device).unwrap();
        assert_eq!(f32_tensor.dtype(), DataType::F32);

        // Convert to f16
        let f16_tensor = f32_tensor.to_dtype(DataType::F16).unwrap();
        assert_eq!(f16_tensor.dtype(), DataType::F16);

        // Convert back to f32
        let f32_tensor2 = f16_tensor.to_dtype(DataType::F32).unwrap();
        assert_eq!(f32_tensor2.dtype(), DataType::F32);

        // Should be close but not exact due to precision loss
        assert_tensors_close(&f32_tensor, &f32_tensor2, 1e-3);

        // Test integer conversions
        let int_tensor = Tensor::arange(0.0, 10.0, device).unwrap().to_dtype(DataType::I32).unwrap();
        assert_eq!(int_tensor.dtype(), DataType::I32);
    }

    #[test]
    fn test_memory_layout() {
        let device = Device::Cuda(0);
        let tensor = Tensor::randn(&[2, 3, 4], device).unwrap();

        // Should be contiguous initially
        assert!(tensor.is_contiguous());

        // Transpose should make it non-contiguous
        let transposed = tensor.transpose(0, 2).unwrap();
        assert!(!transposed.is_contiguous());

        // Contiguous should make it contiguous again
        let made_contiguous = transposed.contiguous().unwrap();
        assert!(made_contiguous.is_contiguous());
        assert_eq!(made_contiguous.shape(), transposed.shape());
    }

    #[test]
    fn test_tensor_creation_methods() {
        let device = Device::Cuda(0);
        let shape = &[2, 3];

        // Zeros
        let zeros = Tensor::zeros(shape, device).unwrap();
        assert_eq!(zeros.sum().unwrap().item::<f32>(), 0.0);

        // Ones
        let ones = Tensor::ones(shape, device).unwrap();
        assert_eq!(ones.sum().unwrap().item::<f32>(), 6.0);

        // Full
        let full = Tensor::full(shape, 3.14, device).unwrap();
        assert!((full.mean().unwrap().item::<f32>() - 3.14).abs() < 1e-6);

        // Arange
        let arange = Tensor::arange(0.0, 6.0, device).unwrap();
        assert_eq!(arange.shape(), &[6]);

        // Random tensors should have reasonable statistics
        let randn = Tensor::randn(&[1000], device).unwrap();
        let mean = randn.mean().unwrap().item::<f32>();
        let std = randn.std().unwrap().item::<f32>();
        assert!(mean.abs() < 0.1); // Should be close to 0
        assert!((std - 1.0).abs() < 0.1); // Should be close to 1
    }
}
