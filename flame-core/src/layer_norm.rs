//! Layer normalization for FLAME

use crate::{Tensor, Shape, Result, FlameError};
use crate::autograd::{AutogradContext, Op};
use crate::tensor_storage::TensorStorage;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig, LaunchAsync};

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// Layer normalization configuration
pub struct LayerNormConfig {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            normalized_shape: vec![],
            eps: 1e-5,
            elementwise_affine: true,
        }
    }
}

/// Layer normalization layer
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    /// Create new layer normalization
    pub fn new(normalized_shape: Vec<usize>, eps: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let param_shape = Shape::from_dims(&normalized_shape);
        
        // Initialize weight and bias
        let weight = Tensor::ones(param_shape.clone(), device.clone())?
            .requires_grad_(true);
        let bias = Tensor::zeros(param_shape, device)?
            .requires_grad_(true);
        
        Ok(Self {
            normalized_shape,
            eps,
            weight: Some(weight),
            bias: Some(bias),
        })
    }
    
    /// Create layer norm without affine parameters
    pub fn new_no_affine(normalized_shape: Vec<usize>, eps: f32) -> Self {
        Self {
            normalized_shape,
            eps,
            weight: None,
            bias: None,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Check if we can use cuDNN for this operation
        #[cfg(feature = "cudnn")]
        if crate::cudnn::is_cudnn_norm_compatible(input, "layer") {
            return crate::cudnn::cudnn_layer_norm(
                input,
                &self.normalized_shape,
                self.weight.as_ref(),
                self.bias.as_ref(),
                self.eps as f64,
            );
        }
        
        // Fallback to regular implementation
        layer_norm(
            input,
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }
}

/// CUDA kernel for layer normalization forward pass
pub const LAYER_NORM_FWD_KERNEL: &str = r#"
extern "C" __global__ void layer_norm_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float* mean_out,
    float* rstd_out,
    int batch_size,
    int norm_size,
    float eps
) {
    int idx = blockIdx.x;
    if (idx >= batch_size) return;
    
    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < norm_size; i++) {
        sum += input[idx * norm_size + i];
    }
    float mean = sum / norm_size;
    mean_out[idx] = mean;
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int i = 0; i < norm_size; i++) {
        float diff = input[idx * norm_size + i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / norm_size;
    float rstd = rsqrtf(var + eps);
    rstd_out[idx] = rstd;
    
    // Normalize and apply affine transform
    for (int i = 0; i < norm_size; i++) {
        int global_idx = idx * norm_size + i;
        float normalized = (input[global_idx] - mean) * rstd;
        
        if (weight != nullptr) {
            normalized = normalized * weight[i];
        }
        if (bias != nullptr) {
            normalized = normalized + bias[i];
        }
        
        output[global_idx] = normalized;
    }
}
"#;

/// CUDA kernel for layer normalization backward pass
pub const LAYER_NORM_BWD_KERNEL: &str = r#"
extern "C" __global__ void layer_norm_backward(
    const float* grad_output,
    const float* input,
    const float* mean,
    const float* rstd,
    const float* weight,
    float* grad_input,
    float* grad_weight,
    float* grad_bias,
    int batch_size,
    int norm_size
) {
    int idx = blockIdx.x;
    if (idx >= batch_size) return;
    
    float mean_val = mean[idx];
    float rstd_val = rstd[idx];
    
    // Calculate gradients for this batch element
    float grad_mean = 0.0f;
    float grad_var = 0.0f;
    
    // First pass: calculate grad_mean and grad_var
    for (int i = 0; i < norm_size; i++) {
        int global_idx = idx * norm_size + i;
        float grad_out = grad_output[global_idx];
        
        if (weight != nullptr) {
            grad_out = grad_out * weight[i];
        }
        
        grad_mean += grad_out;
        grad_var += grad_out * (input[global_idx] - mean_val);
    }
    
    grad_mean = grad_mean * (-rstd_val) / norm_size;
    grad_var = grad_var * (-0.5f) * rstd_val * rstd_val * rstd_val / norm_size;
    
    // Second pass: calculate grad_input
    for (int i = 0; i < norm_size; i++) {
        int global_idx = idx * norm_size + i;
        float grad_out = grad_output[global_idx];
        
        if (weight != nullptr) {
            grad_out = grad_out * weight[i];
        }
        
        float x_centered = input[global_idx] - mean_val;
        grad_input[global_idx] = rstd_val * grad_out + grad_mean + 2.0f * grad_var * x_centered / norm_size;
        
        // Accumulate weight and bias gradients
        if (idx == 0 && grad_weight != nullptr) {
            atomicAdd(&grad_weight[i], grad_output[global_idx] * (input[global_idx] - mean_val) * rstd_val);
        }
        if (idx == 0 && grad_bias != nullptr) {
            atomicAdd(&grad_bias[i], grad_output[global_idx]);
        }
    }
}
"#;

/// Functional layer normalization
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let device = input.device();
    
    // Ensure kernels are loaded
    use crate::cuda_kernels::CudaKernels;
    CudaKernels::ensure_kernel(device, "layer_norm_forward", LAYER_NORM_FWD_KERNEL)?;
    
    // Validate input shape
    let input_dims = input.shape().dims();
    if input_dims.len() < normalized_shape.len() {
        return Err(FlameError::InvalidOperation(
            "Input dimensions less than normalized_shape".into()
        ));
    }
    
    // Check that normalized_shape matches the last dimensions
    let offset = input_dims.len() - normalized_shape.len();
    for (i, &dim) in normalized_shape.iter().enumerate() {
        if input_dims[offset + i] != dim {
            return Err(FlameError::InvalidOperation(
                format!("Shape mismatch at dim {}: expected {}, got {}", 
                    offset + i, dim, input_dims[offset + i])
            ));
        }
    }
    
    // Calculate batch size and normalization size
    let batch_size: usize = input_dims[..offset].iter().product();
    let norm_size: usize = normalized_shape.iter().product();
    
    // Allocate output and intermediate buffers
    let output_data = crate::tensor::alloc_zeros_from_pool(&device, input.shape().elem_count())?;
    let mean_data = crate::tensor::alloc_zeros_from_pool(&device, batch_size)?;
    let rstd_data = crate::tensor::alloc_zeros_from_pool(&device, batch_size)?;
    
    // Launch kernel
    let f = device.get_func("layer_norm_forward", "layer_norm_forward")
        .ok_or_else(|| FlameError::Cuda("Failed to get layer_norm kernel".into()))?;
    
    // Handle optional weight and bias
    let has_weight = weight.is_some();
    let has_bias = bias.is_some();
    
    unsafe {
        use cudarc::driver::LaunchAsync;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Create launch parameters based on whether weight/bias exist
        match (weight, bias) {
        (Some(w), Some(b)) => {
            launch_kernel!(f, cfg,
                input.storage.try_as_slice_f32()?,
                w.storage.try_as_slice_f32()?,
                b.storage.try_as_slice_f32()?,
                &output_data,
                &mean_data,
                &rstd_data,
                batch_size as i32,
                norm_size as i32,
                eps
            )?;
        }
        (Some(w), None) => {
            launch_kernel!(f, cfg,
                input.storage.try_as_slice_f32()?,
                w.storage.try_as_slice_f32()?,
                0usize, // null pointer for bias
                &output_data,
                &mean_data,
                &rstd_data,
                batch_size as i32,
                norm_size as i32,
                eps
            )?;
        }
        (None, Some(b)) => {
            launch_kernel!(f, cfg,
                input.storage.try_as_slice_f32()?,
                0usize, // null pointer for weight
                b.storage.try_as_slice_f32()?,
                &output_data,
                &mean_data,
                &rstd_data,
                batch_size as i32,
                norm_size as i32,
                eps
            )?;
        }
        (None, None) => {
            launch_kernel!(f, cfg,
                input.storage.try_as_slice_f32()?,
                0usize, // null pointer for weight
                0usize, // null pointer for bias
                &output_data,
                &mean_data,
                &rstd_data,
                batch_size as i32,
                norm_size as i32,
                eps
            )?;
        }
        }
    }
    
    let mut output = Tensor {
        storage: TensorStorage::F32 { data: output_data, numel: input.shape().elem_count() },
        shape: input.shape().clone(),
        device: device.clone(),
        id: crate::tensor::TensorId::new(),
        requires_grad: false,
    };
    
    // Record operation for autograd
    if input.requires_grad || weight.map(|w| w.requires_grad).unwrap_or(false) || bias.map(|b| b.requires_grad).unwrap_or(false) {
        output.requires_grad = true;
        
        let mut saved_tensors = vec![(input.id, input.clone_result()?)];
        if let Some(w) = weight {
            saved_tensors.push((w.id, w.clone_result()?));
        }
        if let Some(b) = bias {
            saved_tensors.push((b.id, b.clone_result()?));
        }
        
        // Save mean and rstd for backward pass
        let mean_tensor = Tensor {
            storage: TensorStorage::F32 { data: mean_data, numel: batch_size },
            shape: Shape::from_dims(&[batch_size]),
            device: device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        let rstd_tensor = Tensor {
            storage: TensorStorage::F32 { data: rstd_data, numel: batch_size },
            shape: Shape::from_dims(&[batch_size]),
            device: device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        
        saved_tensors.push((mean_tensor.id, mean_tensor));
        saved_tensors.push((rstd_tensor.id, rstd_tensor));
        
        AutogradContext::record_op(
            output.id,
            Op::LayerNorm {
                input: input.id,
                normalized_shape: normalized_shape.to_vec(),
            },
            saved_tensors,
        );
    }
    
    // Cast to default dtype if needed, preserve requires_grad
    let dt = crate::config::default_dtype();
    if dt == crate::DType::F32 {
        Ok(output)
    } else {
        let mut out = output.to_dtype(dt)?;
        // Preserve autograd flag
        // Note: we didn't record a new op here; this is a dtype cast for numerics
        if input.requires_grad || weight.map(|w| w.requires_grad).unwrap_or(false) || bias.map(|b| b.requires_grad).unwrap_or(false) {
            out.requires_grad = true;
        }
        Ok(out)
    }
}

/// Add layer norm method to Tensor
impl Tensor {
    /// Apply layer normalization
    pub fn layer_norm(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        layer_norm(self, normalized_shape, weight, bias, eps)
    }
}
