//! CUDA kernel implementations using type-safe launcher
//! 
//! This module provides CUDA kernel implementations for tensor operations
//! using the type-safe kernel launcher.

use crate::{Tensor, Shape, Result, FlameError};
use crate::kernel_launcher::{KernelLauncher, launch_configs, templates};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
use std::sync::Arc;

// Import the kernel_params macro
use crate::kernel_params;

/// Create a tensor from allocated CUDA memory
pub fn create_output_tensor(data: CudaSlice<f32>, shape: Shape, device: Arc<CudaDevice>) -> Tensor {
    Tensor {
        data: Arc::new(data),
        shape,
        device,
        id: crate::tensor::TensorId(0), // Will be set properly when added to graph
        requires_grad: false,
    }
}

/// CUDA kernels using type-safe launcher
pub struct CudaKernelsV2 {
    launcher: KernelLauncher,
}

impl CudaKernelsV2 {
    /// Create a new instance
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            launcher: KernelLauncher::new(device),
        }
    }
    
    /// Element-wise addition
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(FlameError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        
        let numel = a.shape.elem_count();
        let mut output = unsafe { a.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let kernel_code = templates::elementwise_binary("add_kernel", "+");
        let _ = self.launcher.prepare_kernel("add_kernel", &kernel_code)?;
        
        // Get the function directly from device
        let f = a.device.get_func("add_kernel", "add_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get add_kernel".into()))?;
        
        let config = launch_configs::elementwise(numel);
        unsafe {
            f.launch(config, (
                &mut output,
                &*a.data,
                &*b.data,
                numel as i32,
            ))?;
        }
        
        Ok(create_output_tensor(output, a.shape.clone(), a.device.clone()))
    }
    
    /// Element-wise multiplication
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(FlameError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        
        let numel = a.shape.elem_count();
        let mut output = unsafe { a.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let kernel_code = templates::elementwise_binary("mul_kernel", "*");
        let _ = self.launcher.prepare_kernel("mul_kernel", &kernel_code)?;
        
        let f = a.device.get_func("mul_kernel", "mul_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get mul_kernel".into()))?;
        
        let config = launch_configs::elementwise(numel);
        unsafe {
            f.launch(config, (
                &mut output,
                &*a.data,
                &*b.data,
                numel as i32,
            ))?;
        }
        
        Ok(create_output_tensor(output, a.shape.clone(), a.device.clone()))
    }
    
    /// ReLU activation
    pub fn relu(&self, input: &Tensor) -> Result<Tensor> {
        let numel = input.shape.elem_count();
        let mut output = unsafe { input.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let kernel_code = templates::elementwise_unary("relu_kernel", "fmaxf(0.0f, x)");
        let _ = self.launcher.prepare_kernel("relu_kernel", &kernel_code)?;
        
        let f = input.device.get_func("relu_kernel", "relu_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get relu_kernel".into()))?;
        
        let config = launch_configs::elementwise(numel);
        unsafe {
            f.launch(config, (
                &mut output,
                &*input.data,
                numel as i32,
            ))?;
        }
        
        Ok(create_output_tensor(output, input.shape.clone(), input.device.clone()))
    }
    
    /// GELU activation
    pub fn gelu(&self, input: &Tensor) -> Result<Tensor> {
        let numel = input.shape.elem_count();
        let mut output = unsafe { input.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let kernel_code = templates::elementwise_unary(
            "gelu_kernel",
            "0.5f * x * (1.0f + tanhf(0.797884560802865f * (x + 0.044715f * x * x * x)))"
        );
        let _ = self.launcher.prepare_kernel("gelu_kernel", &kernel_code)?;
        
        let f = input.device.get_func("gelu_kernel", "gelu_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get gelu_kernel".into()))?;
        
        let config = launch_configs::elementwise(numel);
        unsafe {
            f.launch(config, (
                &mut output,
                &*input.data,
                numel as i32,
            ))?;
        }
        
        Ok(create_output_tensor(output, input.shape.clone(), input.device.clone()))
    }
    
    /// Matrix multiplication with type-safe parameters
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Define kernel parameters structure
        kernel_params! {
            struct MatMulParams {
                m: i32,
                n: i32,
                k: i32,
            }
        }
        
        let a_dims = a.shape.dims();
        let b_dims = b.shape.dims();
        
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(FlameError::InvalidOperation(
                "MatMul requires 2D tensors".into()
            ));
        }
        
        let (m, k1) = (a_dims[0], a_dims[1]);
        let (k2, n) = (b_dims[0], b_dims[1]);
        
        if k1 != k2 {
            return Err(FlameError::ShapeMismatch {
                expected: Shape::from_dims(&[m, k1]),
                got: Shape::from_dims(&[k2, n]),
            });
        }
        
        let output_shape = Shape::from_dims(&[m, n]);
        let output_numel = output_shape.elem_count();
        let mut output = unsafe { a.device.alloc::<f32>(output_numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let kernel_code = r#"
extern "C" __global__ void matmul_kernel(
    float* output,
    const float* a,
    const float* b,
    int m,
    int n,
    int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        output[row * n + col] = sum;
    }
}"#;
        
        let _ = self.launcher.prepare_kernel("matmul_kernel", kernel_code)?;
        
        let f = a.device.get_func("matmul_kernel", "matmul_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get matmul_kernel".into()))?;
        
        let params = MatMulParams {
            m: m as i32,
            n: n as i32,
            k: k1 as i32,
        };
        
        let config = launch_configs::grid_2d(m, n, 16);
        unsafe {
            f.launch(config, (
                &mut output,
                &*a.data,
                &*b.data,
                params.m,
                params.n,
                params.k,
            ))?;
        }
        
        Ok(create_output_tensor(output, output_shape, a.device.clone()))
    }
    
    /// Sum reduction
    pub fn sum(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = unsafe { input.device.alloc_zeros::<f32>(1) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let kernel_code = templates::reduction_sum();
        let _ = self.launcher.prepare_kernel("reduction_sum", kernel_code)?;
        
        let f = input.device.get_func("reduction_sum", "reduction_sum")
            .ok_or_else(|| FlameError::Cuda("Failed to get reduction_sum".into()))?;
        
        let config = launch_configs::reduction(input.shape.elem_count());
        unsafe {
            f.launch(config, (
                &mut output,
                &*input.data,
                input.shape.elem_count() as i32,
            ))?;
        }
        
        Ok(create_output_tensor(output, Shape::from_dims(&[1]), input.device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kernel_launcher() -> Result<()> {
        let device: Arc<CudaDevice> = CudaDevice::new(0)?;
        let kernels = CudaKernelsV2::new(device.clone());
        
        // Test element-wise addition
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), device.clone())?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]), device.clone())?;
        
        let result = kernels.add(&a, &b)?;
        let result_data = result.to_vec()?;
        
        assert_eq!(result_data, vec![5.0, 7.0, 9.0]);
        
        Ok(())
    }
}