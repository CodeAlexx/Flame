//! Fused CUDA kernels for optimized operations
//! 
//! This module provides fused kernels that combine multiple operations
//! into a single kernel launch for better performance.

use crate::{Tensor, Shape, Result, FlameError, CudaDevice};
use crate::cuda_kernels::CudaKernels;
use std::sync::Arc;
use cudarc::driver::{LaunchAsync, CudaFunction};

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// Fused operations for common patterns
pub struct FusedKernels;

impl FusedKernels {
    /// Fused bias + activation kernel
    pub fn bias_gelu(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void bias_gelu_kernel(
    float* output,
    const float* input,
    const float* bias,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_size;
    
    if (idx < total_elements) {
        int h_idx = idx % hidden_size;
        float x = input[idx] + bias[h_idx];
        
        // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        const float c0 = 0.7978845608f; // sqrt(2/π)
        const float c1 = 0.044715f;
        float x3 = x * x * x;
        float arg = c0 * (x + c1 * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(arg));
    }
}
"#;
        
        CudaKernels::ensure_kernel(&x.device, "bias_gelu_kernel", kernel_code)?;
        
        let f = x.device.get_func("bias_gelu_kernel", "bias_gelu_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get bias_gelu_kernel".into()))?;
        
        let mut output = Tensor::zeros(x.shape().clone(), x.device.clone())?;
        
        let dims = x.shape().dims();
        let batch_size = dims[0] as i32;
        let seq_len = dims[1] as i32;
        let hidden_size = dims[2] as i32;
        
        let total_elems = (batch_size * seq_len * hidden_size) as u32;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(total_elems);
        
        launch_kernel!(f, cfg,
            output.storage.as_slice(),
            x.storage.as_slice(),
            bias.storage.as_slice(),
            batch_size,
            seq_len,
            hidden_size
        )?;
        
        Ok(output)
    }
    
    /// Fused LayerNorm + Linear kernel
    pub fn layernorm_linear(
        x: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void layernorm_linear_kernel(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    const float* weight,
    const float* bias,
    int batch_size,
    int seq_len,
    int hidden_size,
    int out_features,
    float eps,
    bool has_bias
) {
    extern __shared__ float shared_data[];
    
    int batch_seq = blockIdx.x;
    int out_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_seq >= batch_size * seq_len || out_idx >= out_features) return;
    
    // First compute LayerNorm for this sequence position
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Load input to shared memory and compute statistics
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_seq * hidden_size + i;
        float val = input[idx];
        shared_data[i] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Reduce to get mean and variance
    __syncthreads();
    
    // Warp reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    
    // Write to shared memory
    if (tid % warpSize == 0) {
        atomicAdd(&shared_data[hidden_size], local_sum);
        atomicAdd(&shared_data[hidden_size + 1], local_sum_sq);
    }
    
    __syncthreads();
    
    float mean = shared_data[hidden_size] / hidden_size;
    float var = shared_data[hidden_size + 1] / hidden_size - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Now compute the fused layernorm + linear operation
    float sum = 0.0f;
    
    for (int i = 0; i < hidden_size; i++) {
        // LayerNorm: (x - mean) * inv_std * gamma + beta
        float normalized = (shared_data[i] - mean) * inv_std * gamma[i] + beta[i];
        
        // Linear: normalized @ weight[i, out_idx]
        sum += normalized * weight[out_idx * hidden_size + i];
    }
    
    // Add bias if present
    if (has_bias) {
        sum += bias[out_idx];
    }
    
    // Write output
    output[batch_seq * out_features + out_idx] = sum;
}
"#;
        
        CudaKernels::ensure_kernel(&x.device, "layernorm_linear_kernel", kernel_code)?;
        
        let f = x.device.get_func("layernorm_linear_kernel", "layernorm_linear_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get layernorm_linear_kernel".into()))?;
        
        let dims = x.shape().dims();
        let batch_size = dims[0] as i32;
        let seq_len = dims[1] as i32;
        let hidden_size = dims[2] as i32;
        let out_features = weight.shape().dims()[0] as i32;
        
        let mut output = Tensor::zeros(
            Shape::from_dims(&[batch_size as usize, seq_len as usize, out_features as usize]),
            x.device.clone()
        )?;
        
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: ((batch_size * seq_len) as u32, out_features as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: ((hidden_size + 2) * 4) as u32, // For input + stats
        };
        
        // For now, always require a bias tensor
        let dummy_bias;
        let bias_to_use = if let Some(b) = bias {
            b
        } else {
            // Create a dummy zero tensor for bias
            dummy_bias = Tensor::zeros(Shape::from_dims(&[out_features as usize]), x.device.clone())?;
            &dummy_bias
        };
        
        launch_kernel!(f, grid,
            output.storage.as_slice(),
            x.storage.as_slice(),
            gamma.storage.as_slice(),
            beta.storage.as_slice(),
            weight.storage.as_slice(),
            bias_to_use.storage.as_slice(),
            batch_size,
            seq_len,
            hidden_size,
            out_features,
            eps,
            bias.is_some()
        )?;
        
        Ok(output)
    }
    
    /// Fused attention: Q @ K^T * scale + mask + softmax @ V
    pub fn fused_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void fused_attention_kernel(
    float* output,
    const float* q,
    const float* k,
    const float* v,
    const float* mask,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool has_mask
) {
    // Optimized version using shared memory for better performance
    extern __shared__ float shared_mem[];
    
    // Shared memory layout:
    // - First head_dim floats: Q vector for current position
    // - Next seq_len floats: attention scores
    // - Next seq_len floats: attention weights (after softmax)
    float* shared_q = shared_mem;
    float* shared_scores = &shared_mem[head_dim];
    float* shared_weights = &shared_mem[head_dim + seq_len];
    
    int batch_head = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_head >= batch_size * num_heads || seq_idx >= seq_len) {
        return;
    }
    
    int batch_idx = batch_head / num_heads;
    int head_idx = batch_head % num_heads;
    
    // Load Q vector into shared memory
    for (int d = tid; d < head_dim; d += block_size) {
        int q_idx = batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim +
                   seq_idx * head_dim + d;
        shared_q[d] = q[q_idx];
    }
    __syncthreads();
    
    // Compute attention scores using all threads
    float max_score = -1e20f;
    
    // Each thread computes scores for a subset of K positions
    for (int kv_idx = tid; kv_idx <= seq_idx; kv_idx += block_size) {
        float score = 0.0f;
        
        // Compute dot product Q @ K^T
        for (int d = 0; d < head_dim; d++) {
            int k_idx = batch_idx * num_heads * seq_len * head_dim +
                       head_idx * seq_len * head_dim +
                       kv_idx * head_dim + d;
            score += shared_q[d] * k[k_idx];
        }
        score *= scale;
            
            if (has_mask) {
                int mask_idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + kv_idx;
                score += mask[mask_idx];
            }
            
            shared_mem[kv_idx] = score;
            max_score = fmaxf(max_score, score);
        }
    }
    
    __syncthreads();
    
    // Second pass: exp and sum
    float sum_exp = 0.0f;
    if (head_elem == 0) {
        for (int kv_idx = 0; kv_idx <= seq_idx; kv_idx++) {
            shared_mem[kv_idx] = expf(shared_mem[kv_idx] - max_score);
            sum_exp += shared_mem[kv_idx];
        }
    }
    
    __syncthreads();
    
    // Third pass: normalize and apply to V
    if (head_elem < head_dim) {
        float result = 0.0f;
        for (int kv_idx = 0; kv_idx <= seq_idx; kv_idx++) {
            float attn_weight = shared_mem[kv_idx] / sum_exp;
            
            int v_idx = batch_idx * num_heads * seq_len * head_dim +
                       head_idx * seq_len * head_dim +
                       kv_idx * head_dim + head_elem;
                       
            result += attn_weight * v[v_idx];
        }
        
        int out_idx = batch_idx * num_heads * seq_len * head_dim +
                     head_idx * seq_len * head_dim +
                     seq_idx * head_dim + head_elem;
                     
        output[out_idx] = result;
    }
}
"#;
        
        // Use Flash Attention for fused implementation
        let fa = crate::flash_attention::FlashAttention::new()
            .with_causal(mask.is_some());
        fa.forward(q, k, v, mask)
    }
    
    /// Fused residual + LayerNorm
    pub fn residual_layernorm(
        x: &Tensor,
        residual: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void residual_layernorm_kernel(
    float* output,
    float* residual_out,  // Store residual for next layer
    const float* input,
    const float* residual,
    const float* gamma,
    const float* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
) {
    extern __shared__ float shared_data[];
    
    int batch_seq = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_seq >= batch_size * seq_len) return;
    
    // Compute residual
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_seq * hidden_size + i;
        float val = input[idx] + residual[idx];
        residual_out[idx] = val;  // Save for next layer
        
        local_sum += val;
        local_sum_sq += val * val;
        
        if (tid < hidden_size) {
            shared_data[tid] = val;
        }
    }
    
    // Reduce to get mean and variance
    __syncthreads();
    
    // Warp reduction for sum and sum_sq
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    
    if (tid % warpSize == 0) {
        atomicAdd(&shared_data[hidden_size], local_sum);
        atomicAdd(&shared_data[hidden_size + 1], local_sum_sq);
    }
    
    __syncthreads();
    
    float mean = shared_data[hidden_size] / hidden_size;
    float var = shared_data[hidden_size + 1] / hidden_size - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Apply LayerNorm
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_seq * hidden_size + i;
        float normalized = (shared_data[i] - mean) * inv_std;
        output[idx] = normalized * gamma[i] + beta[i];
    }
}
"#;
        
        // For now, implement as separate operations
        let residual_sum = x.add(residual)?;
        // Create a LayerNorm instance and use it
        let layer_norm = crate::norm::LayerNorm {
            normalized_shape: vec![residual_sum.shape().dims().last().copied().unwrap_or(1)],
            eps,
            elementwise_affine: true,
            weight: Some(gamma.clone()?),
            bias: Some(beta.clone()?),
        };
        layer_norm.forward(&residual_sum)
    }
    
    /// Fused GELU backward
    pub fn gelu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void gelu_backward_kernel(
    float* grad_input,
    const float* grad_output,
    const float* input,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float x = input[idx];
    
    // GELU derivative
    const float c0 = 0.7978845608f; // sqrt(2/π)
    const float c1 = 0.044715f;
    
    float x3 = x * x * x;
    float arg = c0 * (x + c1 * x3);
    float tanh_arg = tanhf(arg);
    
    // d/dx[GELU(x)] = 0.5 * (1 + tanh(arg)) + 0.5 * x * sech²(arg) * c0 * (1 + 3*c1*x²)
    float sech_sq = 1.0f - tanh_arg * tanh_arg;
    float grad = 0.5f * (1.0f + tanh_arg) + 
                 0.5f * x * sech_sq * c0 * (1.0f + 3.0f * c1 * x * x);
    
    grad_input[idx] = grad_output[idx] * grad;
}
"#;
        
        CudaKernels::ensure_kernel(&input.device, "gelu_backward_kernel", kernel_code)?;
        
        let f = input.device.get_func("gelu_backward_kernel", "gelu_backward_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get gelu_backward_kernel".into()))?;
        
        let mut grad_input = Tensor::zeros(input.shape().clone(), input.device.clone())?;
        let numel = input.shape().elem_count() as i32;
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(numel as u32);
        
        launch_kernel!(f, cfg,
            grad_input.storage.as_slice(),
            grad_output.storage.as_slice(),
            input.storage.as_slice(),
            numel
        )?;
        
        Ok(grad_input)
    }
    
    /// Fused Adam optimizer step
    pub fn adam_step(
        param: &mut Tensor,
        grad: &Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: i32,
    ) -> Result<()> {
        let kernel_code = r#"
extern "C" __global__ void adam_step_kernel(
    float* param,
    float* m,
    float* v,
    const float* grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float g = grad[idx];
    
    // Update biased moments
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
    
    // Bias correction
    float m_hat = m[idx] / (1.0f - powf(beta1, step));
    float v_hat = v[idx] / (1.0f - powf(beta2, step));
    
    // Update parameters
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}
"#;
        
        CudaKernels::ensure_kernel(&param.device, "adam_step_kernel", kernel_code)?;
        
        let f = param.device.get_func("adam_step_kernel", "adam_step_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get adam_step_kernel".into()))?;
        
        let numel = param.shape().elem_count() as i32;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(numel as u32);
        
        launch_kernel!(f, cfg,
            param.storage.as_slice(),
            m.storage.as_slice(),
            v.storage.as_slice(),
            grad.storage.as_slice(),
            lr,
            beta1,
            beta2,
            eps,
            step,
            numel
        )?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bias_gelu() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        let batch_size = 2;
        let seq_len = 4;
        let hidden_size = 8;
        
        let x = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len, hidden_size]),
            0.0, 0.1, device.clone()
        )?;
        
        let bias = Tensor::randn(
            Shape::from_dims(&[hidden_size]),
            0.0, 0.01, device.clone()
        )?;
        
        let output = FusedKernels::bias_gelu(&x, &bias)?;
        
        assert_eq!(output.shape().dims(), x.shape().dims());
        
        println!("Fused bias+GELU test passed!");
        Ok(())
    }
}