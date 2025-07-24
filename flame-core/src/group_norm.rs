//! Group Normalization for FLAME
//! Required for all diffusion models (UNet, DiT, etc.)

use crate::{Tensor, Shape, Result, FlameError};
use crate::autograd::{AutogradContext, Op};
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// Group Normalization
/// Divides channels into groups and normalizes within each group
pub fn group_norm(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let shape = input.shape().dims();
    if shape.len() != 4 {
        return Err(FlameError::InvalidOperation(
            "GroupNorm expects 4D input [batch, channels, height, width]".into()
        ));
    }
    
    let batch_size = shape[0];
    let num_channels = shape[1];
    let height = shape[2];
    let width = shape[3];
    let spatial_size = height * width;
    
    if num_channels % num_groups != 0 {
        return Err(FlameError::InvalidOperation(
            format!("num_channels ({}) must be divisible by num_groups ({})", num_channels, num_groups)
        ));
    }
    
    let channels_per_group = num_channels / num_groups;
    
    // Validate weight and bias shapes
    if let Some(w) = weight {
        if w.shape().dims() != &[num_channels] {
            return Err(FlameError::ShapeMismatch {
                expected: Shape::from_dims(&[num_channels]),
                got: w.shape().clone(),
            });
        }
    }
    
    if let Some(b) = bias {
        if b.shape().dims() != &[num_channels] {
            return Err(FlameError::ShapeMismatch {
                expected: Shape::from_dims(&[num_channels]),
                got: b.shape().clone(),
            });
        }
    }
    
    // Compile kernel
    let kernel_code = get_group_norm_kernel();
    crate::cuda_kernels::CudaKernels::ensure_kernel(&input.device, "group_norm_forward", kernel_code)?;
    
    let f = input.device.get_func("group_norm_forward", "group_norm_forward")
        .ok_or_else(|| FlameError::Cuda("Failed to get group_norm_forward kernel".into()))?;
    
    // Allocate output and temporary buffers
    let output_data = crate::tensor::alloc_zeros_from_pool(&input.device, input.shape().elem_count())?;
    let mean_data = crate::tensor::alloc_zeros_from_pool(&input.device, batch_size * num_groups)?;
    let var_data = crate::tensor::alloc_zeros_from_pool(&input.device, batch_size * num_groups)?;
    
    // Launch kernel
    let threads_per_block = 256;
    let num_blocks = (input.shape().elem_count() + threads_per_block - 1) / threads_per_block;
    
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    
    // Handle optional weight and bias
    // Create dummy tensors for None cases to avoid allocations in kernel launch
    let dummy = crate::tensor::alloc_zeros_from_pool(&input.device, 1)?;
    let weight_ptr = weight
        .map(|w| w.storage.as_slice())
        .unwrap_or(&dummy);
    let bias_ptr = bias
        .map(|b| b.storage.as_slice())
        .unwrap_or(&dummy);
    
    // Pack dimensions into fewer parameters to avoid exceeding LaunchAsync limit
    let dims1 = (batch_size << 16) | num_channels;
    let dims2 = (num_groups << 16) | channels_per_group;
    
    launch_kernel!(f, cfg,
        input.storage.as_slice(),
        &output_data,
        weight_ptr,
        bias_ptr,
        &mean_data,
        &var_data,
        dims1 as i32,
        dims2 as i32,
        spatial_size as i32,
        eps,
        weight.is_some() as i32
    )?;
    
    let mut output = Tensor {
        storage: TensorStorage::F32 { data: output_data, numel: input.shape().elem_count() },
        shape: input.shape().clone(),
        device: input.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };
    
    // Record for autograd
    if input.requires_grad || (weight.is_some() && weight.unwrap().requires_grad) {
        output.requires_grad = true;
        
        let mut saved_tensors = vec![(input.id, input.clone()?)];
        if let Some(w) = weight {
            saved_tensors.push((w.id, w.clone()?));
        }
        if let Some(b) = bias {
            saved_tensors.push((b.id, b.clone()?));
        }
        
        // Save mean and var for backward
        let mean_tensor = Tensor {
            storage: TensorStorage::F32 { data: mean_data, numel: batch_size * num_groups },
            shape: Shape::from_dims(&[batch_size, num_groups]),
            device: input.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        let var_tensor = Tensor {
            storage: TensorStorage::F32 { data: var_data, numel: batch_size * num_groups },
            shape: Shape::from_dims(&[batch_size, num_groups]),
            device: input.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        saved_tensors.push((mean_tensor.id, mean_tensor));
        saved_tensors.push((var_tensor.id, var_tensor));
        
        AutogradContext::record_op(
            output.id,
            Op::GroupNorm {
                input: input.id,
                num_groups,
                weight: weight.map(|w| w.id),
                bias: bias.map(|b| b.id),
            },
            saved_tensors,
        );
    }
    
    Ok(output)
}

fn get_group_norm_kernel() -> &'static str {
    r#"
extern "C" __global__ void group_norm_forward(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    float* mean_out,
    float* var_out,
    int dims1,  // (batch_size << 16) | num_channels
    int dims2,  // (num_groups << 16) | channels_per_group
    int spatial_size,
    float eps,
    int has_affine
) {
    // Unpack dimensions
    int batch_size = dims1 >> 16;
    int num_channels = dims1 & 0xFFFF;
    int num_groups = dims2 >> 16;
    int channels_per_group = dims2 & 0xFFFF;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * spatial_size;
    
    if (idx >= total_elements) return;
    
    // Decompose index
    int n = idx / (num_channels * spatial_size);
    int c = (idx / spatial_size) % num_channels;
    int hw = idx % spatial_size;
    
    int g = c / channels_per_group;
    int group_idx = n * num_groups + g;
    
    // First pass: compute mean and variance (simplified - in production use shared memory)
    if (hw == 0 && c % channels_per_group == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int count = channels_per_group * spatial_size;
        
        for (int i = 0; i < channels_per_group; i++) {
            int ch = g * channels_per_group + i;
            for (int j = 0; j < spatial_size; j++) {
                int input_idx = n * num_channels * spatial_size + ch * spatial_size + j;
                float val = input[input_idx];
                sum += val;
                sum_sq += val * val;
            }
        }
        
        float mean = sum / count;
        float var = sum_sq / count - mean * mean;
        
        mean_out[group_idx] = mean;
        var_out[group_idx] = var;
    }
    
    __syncthreads();
    
    // Second pass: normalize
    float mean = mean_out[group_idx];
    float var = var_out[group_idx];
    float std = sqrtf(var + eps);
    
    float normalized = (input[idx] - mean) / std;
    
    if (has_affine) {
        normalized = normalized * weight[c] + bias[c];
    }
    
    output[idx] = normalized;
}
"#
}

/// Group Normalization module
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
    pub affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl GroupNorm {
    /// Create a new GroupNorm module
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        affine: bool,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(FlameError::InvalidOperation(
                format!("num_channels ({}) must be divisible by num_groups ({})", num_channels, num_groups)
            ));
        }
        
        let (weight, bias) = if affine {
            let weight = Tensor::from_vec(
                vec![1.0f32; num_channels],
                Shape::from_dims(&[num_channels]),
                device.clone(),
            )?;
            let bias = Tensor::zeros(Shape::from_dims(&[num_channels]), device)?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        
        Ok(Self {
            num_groups,
            num_channels,
            eps,
            affine,
            weight,
            bias,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        group_norm(
            input,
            self.num_groups,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }
}