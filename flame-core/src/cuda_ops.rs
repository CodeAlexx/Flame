use crate::{Result, Tensor};
use crate::cuda_kernels::CudaKernels;
use std::sync::Arc;
use lazy_static::lazy_static;
use std::sync::Mutex;
use std::collections::HashMap;

lazy_static! {
    static ref KERNELS_CACHE: Mutex<HashMap<usize, Arc<CudaKernels>>> = Mutex::new(HashMap::new());
}

/// GPU operations using CUDA kernels with NVRTC compilation
pub struct GpuOps;

impl GpuOps {
    /// Get or create CudaKernels instance for a device
    fn get_kernels(device: &Arc<cudarc::driver::CudaDevice>) -> Result<Arc<CudaKernels>> {
        let device_id = device.ordinal();
        let mut cache = KERNELS_CACHE.lock().unwrap();
        
        if let Some(kernels) = cache.get(&device_id) {
            Ok(kernels.clone())
        } else {
            let kernels = Arc::new(CudaKernels::new(device.clone())?);
            cache.insert(device_id, kernels.clone());
            Ok(kernels)
        }
    }
    
    /// Element-wise addition
    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&a.device)?;
        kernels.add(a, b)
    }
    
    /// Element-wise multiplication  
    pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&a.device)?;
        kernels.mul(a, b)
    }
    
    /// Scalar multiplication
    pub fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.mul_scalar(tensor, scalar)
    }
    
    /// Scalar addition
    pub fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.add_scalar(tensor, scalar)
    }
    
    /// ReLU activation
    pub fn relu(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.relu(tensor)
    }
    
    /// Sigmoid activation
    pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.sigmoid(tensor)
    }
    
    /// GELU activation
    pub fn gelu(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.gelu(tensor)
    }
    
    /// SiLU activation
    pub fn silu(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.silu(tensor)
    }
    
    /// Tanh activation
    pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.tanh(tensor)
    }
    
    /// Sum reduction
    pub fn sum(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.sum(tensor)
    }
    
    /// Sum reduction along specific dimensions
    pub fn sum_dims(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.sum_dims(tensor, dims)
    }
    
    /// Transpose
    pub fn transpose(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.transpose(tensor)
    }
    
    /// Update weights
    pub fn update_weights(weights: &Tensor, gradients: &Tensor, lr: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&weights.device)?;
        kernels.update_weights(weights, gradients, lr)
    }
    
    /// Leaky ReLU
    pub fn leaky_relu(tensor: &Tensor, negative_slope: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.leaky_relu(tensor, negative_slope)
    }
    
    /// ELU
    pub fn elu(tensor: &Tensor, alpha: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.elu(tensor, alpha)
    }
    
    /// PReLU
    pub fn prelu(tensor: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.prelu(tensor, weight)
    }
    
    /// Broadcast
    pub fn broadcast(tensor: &Tensor, target_shape: &crate::Shape) -> Result<Tensor> {
        CudaKernels::broadcast(tensor, target_shape)
    }
    
    /// Element-wise division
    pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&a.device)?;
        kernels.div(a, b)
    }
    
    /// Max reduction along dimension
    pub fn max_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.max_dim(tensor, dim, keepdim)
    }
    
    /// Sum along dimension with keepdim
    pub fn sum_dim_keepdim(tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.sum_dim_keepdim(tensor, dim)
    }
    
    // Upsampling operations
    pub fn upsample2d_nearest(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.upsample2d_nearest(input, output_size)
    }
    
    pub fn upsample2d_bilinear(input: &Tensor, output_size: (usize, usize), align_corners: bool) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.upsample2d_bilinear(input, output_size, align_corners)
    }
    
    pub fn upsample2d_nearest_backward(grad_output: &Tensor, input_size: (usize, usize), output_size: (usize, usize)) -> Result<Tensor> {
        let kernels = Self::get_kernels(&grad_output.device)?;
        kernels.upsample2d_nearest_backward(grad_output, input_size, output_size)
    }
    
    pub fn upsample2d_bilinear_backward(grad_output: &Tensor, input_size: (usize, usize), output_size: (usize, usize), align_corners: bool) -> Result<Tensor> {
        let kernels = Self::get_kernels(&grad_output.device)?;
        kernels.upsample2d_bilinear_backward(grad_output, input_size, output_size, align_corners)
    }
    
    // Transposed convolution operations
    pub fn conv_transpose2d_forward(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        groups: usize,
        dilation: (usize, usize),
    ) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.conv_transpose2d_forward(input, weight, bias, stride, padding, output_padding, groups, dilation)
    }
    
    pub fn conv_transpose2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        groups: usize,
        dilation: (usize, usize),
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let kernels = Self::get_kernels(&grad_output.device)?;
        kernels.conv_transpose2d_backward(grad_output, input, weight, stride, padding, output_padding, groups, dilation)
    }
}