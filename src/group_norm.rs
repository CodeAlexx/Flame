//! Group Normalization for FLAME
//! Required for all diffusion models (UNet, DiT, etc.)

use crate::autograd::{AutogradContext, Op};
#[cfg(feature = "bf16_u16")]
use crate::cuda_ops_bf16;
use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::{strict, DType, Error, Result, Shape, Tensor};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::convert::TryFrom;
use std::sync::Arc;

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
    assert_nhwc_bf16_public("group_norm::in", input)?;

    if let Some(w) = weight {
        if w.dtype() != DType::BF16 || w.storage_dtype() != DType::BF16 {
            eprintln!(
                "GroupNorm weight mismatch: dtype={:?} storage={:?} shape={:?} id={:?}",
                w.dtype(),
                w.storage_dtype(),
                w.shape(),
                w.id()
            );
            return Err(Error::InvalidInput(
                "group_norm expects BF16 weight storage".into(),
            ));
        }
    }

    if let Some(b) = bias {
        if b.dtype() != DType::BF16 || b.storage_dtype() != DType::BF16 {
            return Err(Error::InvalidInput(
                "group_norm expects BF16 bias storage".into(),
            ));
        }
    }

    let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
    let output_nchw = group_norm_nchw(&input_nchw, num_groups, weight, bias, eps)?;
    if strict::is_enabled() && output_nchw.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(
            "group_norm_nchw produced non-BF16 tensor under STRICT_BF16".into(),
        ));
    }
    let mut output = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&output_nchw)?;
    if output.dtype() != DType::BF16 {
        output = output.to_dtype(DType::BF16)?;
    }

    assert_nhwc_bf16_public("group_norm::out", &output)?;
    Ok(output)
}

fn group_norm_nchw(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let shape = input.shape().dims();
    if shape.len() != 4 {
        return Err(Error::InvalidOperation(
            "GroupNorm expects 4D input [batch, channels, height, width]".into(),
        ));
    }

    let batch_size = shape[0];
    let num_channels = shape[1];
    let height = shape[2];
    let width = shape[3];
    let spatial_size = height * width;

    if num_channels % num_groups != 0 {
        return Err(Error::InvalidOperation(format!(
            "num_channels ({}) must be divisible by num_groups ({})",
            num_channels, num_groups
        )));
    }

    let channels_per_group = num_channels / num_groups;

    if let Some(w) = weight {
        if w.shape().dims() != [num_channels] {
            eprintln!(
                "GroupNorm weight mismatch: x.shape={:?} num_channels={} w.shape={:?} id={:?}",
                input.shape(),
                num_channels,
                w.shape(),
                w.id()
            );
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(&[num_channels]),
                got: w.shape().clone(),
            });
        }
    }

    if let Some(b) = bias {
        if b.shape().dims() != [num_channels] {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(&[num_channels]),
                got: b.shape().clone(),
            });
        }
    }

    let artifacts = if input.dtype() == DType::BF16 && input.storage.dtype() == DType::BF16 {
        group_norm_forward_bf16(
            input,
            num_groups,
            weight,
            bias,
            batch_size,
            num_channels,
            channels_per_group,
            spatial_size,
            eps,
        )?
    } else {
        if strict::is_enabled() {
            return Err(Error::InvalidInput(
                "group_norm expects BF16 input under STRICT_BF16".into(),
            ));
        }
        group_norm_forward_f32(
            input,
            num_groups,
            weight,
            bias,
            batch_size,
            num_channels,
            channels_per_group,
            spatial_size,
            eps,
        )?
    };

    finalize_group_norm(input, weight, bias, num_groups, batch_size, artifacts)
}

struct GroupNormForwardArtifacts {
    output: Tensor,
    mean_data: CudaSlice<f32>,
    var_data: CudaSlice<f32>,
}

fn group_norm_forward_f32(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    batch_size: usize,
    num_channels: usize,
    channels_per_group: usize,
    spatial_size: usize,
    eps: f32,
) -> Result<GroupNormForwardArtifacts> {
    use crate::cuda_kernels::CudaKernels;

    let device = input.device();
    let kernel_code = get_group_norm_kernel();
    CudaKernels::ensure_kernel(&device, "group_norm_compute_stats", kernel_code)?;
    CudaKernels::ensure_kernel(&device, "group_norm_forward", kernel_code)?;

    let f_stats = device
        .get_func("group_norm_compute_stats", "group_norm_compute_stats")
        .ok_or_else(|| Error::Cuda("Failed to get group_norm_compute_stats kernel".into()))?;
    let f_norm = device
        .get_func("group_norm_forward", "group_norm_forward")
        .ok_or_else(|| Error::Cuda("Failed to get group_norm_forward kernel".into()))?;

    let output_data = crate::tensor::alloc_from_pool(&device, input.shape().elem_count())?;
    let mean_data = crate::tensor::alloc_zeros_from_pool(&device, batch_size * num_groups)?;
    let var_data = crate::tensor::alloc_zeros_from_pool(&device, batch_size * num_groups)?;

    let total_groups = batch_size * num_groups;
    let stats_threads = if spatial_size > 65_536 { 512 } else { 256 };
    let stats_cfg = LaunchConfig {
        grid_dim: (total_groups as u32, 1, 1),
        block_dim: (stats_threads as u32, 1, 1),
        shared_mem_bytes: (stats_threads * 2 * 4) as u32,
    };

    launch_kernel!(
        f_stats,
        stats_cfg,
        input.storage.try_as_slice_f32()?,
        &mean_data,
        &var_data,
        batch_size as i32,
        num_channels as i32,
        num_groups as i32,
        channels_per_group as i32,
        spatial_size as i32
    )?;

    let threads_per_block = if input.shape().elem_count() > 100_000_000 {
        512
    } else {
        256
    };
    let num_blocks = input.shape().elem_count().div_ceil(threads_per_block);

    let norm_cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let has_affine = (weight.is_some() && bias.is_some()) as i32;

    match (weight, bias) {
        (Some(w), Some(b)) => {
            launch_kernel!(
                f_norm,
                norm_cfg,
                input.storage.try_as_slice_f32()?,
                &output_data,
                w.storage.try_as_slice_f32()?,
                b.storage.try_as_slice_f32()?,
                &mean_data,
                &var_data,
                ((batch_size << 16) | num_channels) as i32,
                ((num_groups << 16) | channels_per_group) as i32,
                spatial_size as i32,
                eps,
                has_affine
            )?;
        }
        (Some(w), None) => {
            launch_kernel!(
                f_norm,
                norm_cfg,
                input.storage.try_as_slice_f32()?,
                &output_data,
                w.storage.try_as_slice_f32()?,
                0usize,
                &mean_data,
                &var_data,
                ((batch_size << 16) | num_channels) as i32,
                ((num_groups << 16) | channels_per_group) as i32,
                spatial_size as i32,
                eps,
                has_affine
            )?;
        }
        (None, Some(b)) => {
            launch_kernel!(
                f_norm,
                norm_cfg,
                input.storage.try_as_slice_f32()?,
                &output_data,
                0usize,
                b.storage.try_as_slice_f32()?,
                &mean_data,
                &var_data,
                ((batch_size << 16) | num_channels) as i32,
                ((num_groups << 16) | channels_per_group) as i32,
                spatial_size as i32,
                eps,
                has_affine
            )?;
        }
        (None, None) => {
            launch_kernel!(
                f_norm,
                norm_cfg,
                input.storage.try_as_slice_f32()?,
                &output_data,
                0usize,
                0usize,
                &mean_data,
                &var_data,
                ((batch_size << 16) | num_channels) as i32,
                ((num_groups << 16) | channels_per_group) as i32,
                spatial_size as i32,
                eps,
                has_affine
            )?;
        }
    }

    let output = Tensor {
        storage: TensorStorage::F32 {
            data: output_data.into(),
            numel: input.shape().elem_count(),
        },
        shape: input.shape().clone(),
        device: device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    Ok(GroupNormForwardArtifacts {
        output,
        mean_data,
        var_data,
    })
}

fn group_norm_forward_bf16(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    batch_size: usize,
    _num_channels: usize,
    _channels_per_group: usize,
    _spatial_size: usize,
    eps: f32,
) -> Result<GroupNormForwardArtifacts> {
    #[cfg(feature = "bf16_u16")]
    {
        let device = input.device();
        let mut mean_data = crate::tensor::alloc_zeros_from_pool(device, batch_size * num_groups)?;
        let mut var_data = crate::tensor::alloc_zeros_from_pool(device, batch_size * num_groups)?;

        let groups_i32 = i32::try_from(num_groups)
            .map_err(|_| Error::InvalidInput("group_norm: num_groups exceeds i32::MAX".into()))?;

        let output = cuda_ops_bf16::group_norm_bf16_with_stats(
            input,
            weight,
            bias,
            groups_i32,
            eps,
            &mut mean_data,
            &mut var_data,
        )?;
        Ok(GroupNormForwardArtifacts {
            output,
            mean_data,
            var_data,
        })
    }
    #[cfg(not(feature = "bf16_u16"))]
    {
        let _ = (
            input,
            num_groups,
            weight,
            bias,
            batch_size,
            _num_channels,
            _channels_per_group,
            _spatial_size,
            eps,
        );
        Err(Error::Unsupported(
            "group_norm BF16 path requires bf16_u16 feature".into(),
        ))
    }
}

fn finalize_group_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    num_groups: usize,
    batch_size: usize,
    artifacts: GroupNormForwardArtifacts,
) -> Result<Tensor> {
    let needs_grad = input.requires_grad
        || weight.map(|w| w.requires_grad).unwrap_or(false)
        || bias.map(|b| b.requires_grad).unwrap_or(false);

    let mut output = artifacts.output;

    if needs_grad {
        output.requires_grad = true;
        if AutogradContext::is_recording() {
            let mut saved_tensors = vec![(input.id, input.clone_result()?)];
            if let Some(w) = weight {
                saved_tensors.push((w.id, w.clone_result()?));
            }
            if let Some(b) = bias {
                saved_tensors.push((b.id, b.clone_result()?));
            }

            let mean_tensor = Tensor {
                storage: TensorStorage::F32 {
                    data: artifacts.mean_data.into(),
                    numel: batch_size * num_groups,
                },
                shape: Shape::from_dims(&[batch_size, num_groups]),
                device: input.device.clone(),
                id: TensorId::new(),
                requires_grad: false,
                custom_strides: None,
                view_offset: 0,

            };
            let var_tensor = Tensor {
                storage: TensorStorage::F32 {
                    data: artifacts.var_data.into(),
                    numel: batch_size * num_groups,
                },
                shape: Shape::from_dims(&[batch_size, num_groups]),
                device: input.device.clone(),
                id: TensorId::new(),
                requires_grad: false,
                custom_strides: None,
                view_offset: 0,

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
        } else {
            drop(artifacts.mean_data);
            drop(artifacts.var_data);
        }
    } else {
        drop(artifacts.mean_data);
        drop(artifacts.var_data);
    }

    Ok(output)
}

fn get_group_norm_kernel() -> &'static str {
    r#"
#include <cuda_bf16.h>

// Optimized kernel for computing group statistics with reduced memory access
extern "C" __global__ void group_norm_compute_stats(
    const float* input,
    float* mean_out,
    float* var_out,
    int batch_size,
    int num_channels,
    int num_groups,
    int channels_per_group,
    int spatial_size
) {
    // Use shared memory for partial sums
    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sum_sq = &shared_data[blockDim.x];
    
    int tid = threadIdx.x;
    int group_id = blockIdx.x;
    
    if (group_id >= batch_size * num_groups) return;
    
    int n = group_id / num_groups;
    int g = group_id % num_groups;
    
    // Each thread handles a portion of the spatial dimension
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int elements_per_thread = (spatial_size * channels_per_group + blockDim.x - 1) / blockDim.x;
    int start_idx = tid * elements_per_thread;
    int end_idx = min(start_idx + elements_per_thread, spatial_size * channels_per_group);
    
    for (int idx = start_idx; idx < end_idx; idx++) {
        int c_offset = idx / spatial_size;
        int hw_offset = idx % spatial_size;
        int ch = g * channels_per_group + c_offset;
        
        if (ch < num_channels) {
            int input_idx = n * num_channels * spatial_size + ch * spatial_size + hw_offset;
            float val = input[input_idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    // Store local sums in shared memory
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Reduce in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        int count = channels_per_group * spatial_size;
        float mean = shared_sum[0] / count;
        float var = shared_sum_sq[0] / count - mean * mean;
        
        mean_out[group_id] = mean;
        var_out[group_id] = var;
    }
}

extern "C" __global__ void group_norm_compute_stats_bf16(
    const __nv_bfloat16* input,
    float* mean_out,
    float* var_out,
    int batch_size,
    int num_channels,
    int num_groups,
    int channels_per_group,
    int spatial_size
) {
    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sum_sq = &shared_data[blockDim.x];

    int tid = threadIdx.x;
    int group_id = blockIdx.x;

    if (group_id >= batch_size * num_groups) return;

    int n = group_id / num_groups;
    int g = group_id % num_groups;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int elements_per_thread = (spatial_size * channels_per_group + blockDim.x - 1) / blockDim.x;
    int start_idx = tid * elements_per_thread;
    int end_idx = min(start_idx + elements_per_thread, spatial_size * channels_per_group);

    for (int idx = start_idx; idx < end_idx; idx++) {
        int c_offset = idx / spatial_size;
        int hw_offset = idx % spatial_size;
        int ch = g * channels_per_group + c_offset;

        if (ch < num_channels) {
            int input_idx = n * num_channels * spatial_size + ch * spatial_size + hw_offset;
            float val = __bfloat162float(input[input_idx]);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int count = channels_per_group * spatial_size;
        float mean = shared_sum[0] / count;
        float var = shared_sum_sq[0] / count - mean * mean;

        mean_out[group_id] = mean;
        var_out[group_id] = var;
    }
}

// Main normalization kernel
extern "C" __global__ void group_norm_forward(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* mean_in,
    const float* var_in,
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
    
    // Normalize using pre-computed statistics
    float mean = mean_in[group_idx];
    float var = var_in[group_idx];
    float std = sqrtf(var + eps);
    
    float normalized = (input[idx] - mean) / std;
    
    if (has_affine) {
        normalized = normalized * weight[c] + bias[c];
    }
    
    output[idx] = normalized;
}

extern "C" __global__ void group_norm_forward_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float* mean_in,
    const float* var_in,
    int dims1,
    int dims2,
    int spatial_size,
    float eps,
    int has_affine
) {
    int batch_size = dims1 >> 16;
    int num_channels = dims1 & 0xFFFF;
    int num_groups = dims2 >> 16;
    int channels_per_group = dims2 & 0xFFFF;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * spatial_size;

    if (idx >= total_elements) return;

    int n = idx / (num_channels * spatial_size);
    int c = (idx / spatial_size) % num_channels;
    int hw = idx % spatial_size;

    int g = c / channels_per_group;
    int group_idx = n * num_groups + g;

    float mean = mean_in[group_idx];
    float var = var_in[group_idx];
    float std = sqrtf(var + eps);

    float normalized = (__bfloat162float(input[idx]) - mean) / std;

    if (has_affine && weight != nullptr && bias != nullptr) {
        float gamma = __bfloat162float(weight[c]);
        float beta = __bfloat162float(bias[c]);
        normalized = normalized * gamma + beta;
    }

    output[idx] = __float2bfloat16_rn(normalized);
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
    ///
    /// IMPORTANT: Always pass `dtype` (e.g. DType::BF16) to ensure weights are initialized
    /// directly in the target precision. Initializing as F32 and casting can cause
    /// severe OOM issues on consumer hardware (e.g. 24GB VRAM) due to temporary F32 allocation.
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        affine: bool,
        dtype: DType,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(Error::InvalidOperation(format!(
                "num_channels ({}) must be divisible by num_groups ({})",
                num_channels, num_groups
            )));
        }

        let weight = if affine {
            Some(Tensor::ones_dtype(
                Shape::from_dims(&[num_channels]),
                dtype,
                device.clone(),
            )?)
        } else {
            None
        };
        let bias = if affine {
            Some(Tensor::zeros_dtype(
                Shape::from_dims(&[num_channels]),
                dtype,
                device,
            )?)
        } else {
            None
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

    /// Forward pass for NCHW input (no permutation)
    pub fn forward_nchw(&self, input: &Tensor) -> Result<Tensor> {
        group_norm_nchw(
            input,
            self.num_groups,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }
}
