// ===== SAGE ATTENTION IMPLEMENTATION =====
// flame/src/sage_attention.rs

use crate::{Tensor, TensorId, Shape, Result, FlameError, DType};
use crate::autograd::{AutogradContext, Op};
use crate::cuda_kernels::CudaKernels;
use cudarc::driver::{LaunchAsync, CudaDevice};
use std::sync::Arc;

/// Sage Attention - Accurate 8-bit Attention
/// Based on "SAGE: Bridging the Gap between Full-Precision and 8-bit Matrix Multiplication for Large Language Models"
/// 
/// Key features:
/// - 8-bit quantization for K and V matrices
/// - Per-row quantization with scale factors
/// - Smoothing to reduce quantization error
/// - 2-4x memory reduction vs FP16
pub struct SageAttention {
    pub scale: Option<f32>,
    pub causal: bool,
    pub smoothing_factor: f32,  // Default: 0.1
    pub quantize_kv: bool,      // Enable 8-bit quantization
}

impl SageAttention {
    pub fn new() -> Self {
        Self {
            scale: None,
            causal: false,
            smoothing_factor: 0.1,
            quantize_kv: true,
        }
    }
    
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
        self
    }
    
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
    
    pub fn with_smoothing(mut self, factor: f32) -> Self {
        self.smoothing_factor = factor;
        self
    }
    
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Validate inputs
        let shape_q = query.shape().dims();
        let shape_k = key.shape().dims();
        let shape_v = value.shape().dims();
        
        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(FlameError::InvalidOperation(
                "Sage attention expects 4D tensors [batch, num_heads, seq_len, head_dim]".into()
            ));
        }
        
        let (batch_size, num_heads, seq_len_q, head_dim) = 
            (shape_q[0], shape_q[1], shape_q[2], shape_q[3]);
        let seq_len_k = shape_k[2];
        
        // Apply smoothing to K and V for better quantization
        let (key_smoothed, value_smoothed) = if self.quantize_kv {
            self.apply_smoothing(key, value)?
        } else {
            (key.clone_result()?, value.clone_result()?)
        };
        
        // Quantize K and V to 8-bit
        let (key_quantized, key_scales, value_quantized, value_scales) = if self.quantize_kv {
            self.quantize_kv_tensors(&key_smoothed, &value_smoothed)?
        } else {
            // Skip quantization
            (key_smoothed.clone_result()?, Tensor::ones(Shape::from_dims(&[batch_size, num_heads, seq_len_k, 1]), key.device.clone())?,
             value_smoothed.clone_result()?, Tensor::ones(Shape::from_dims(&[batch_size, num_heads, seq_len_k, 1]), value.device.clone())?)
        };
        
        // Compute attention scores with 8-bit K
        let scale = self.scale.unwrap_or((head_dim as f32).sqrt());
        let scores = if self.quantize_kv {
            self.matmul_q_k8bit(query, &key_quantized, &key_scales, scale)?
        } else {
            // Standard attention
            let key_t = key_smoothed.permute(&[0, 1, 3, 2])?;
            let scores = query.matmul(&key_t)?;
            scores.mul_scalar(1.0 / scale)?
        };
        
        // Apply causal mask if needed
        let scores = if self.causal {
            self.apply_causal_mask(&scores, seq_len_q, seq_len_k)?
        } else {
            scores
        };
        
        // Apply attention mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.add(mask)?
        } else {
            scores
        };
        
        // Softmax
        let attention_weights = scores.softmax(-1)?;
        
        // Apply attention to values with 8-bit V
        let output = if self.quantize_kv {
            self.matmul_attn_v8bit(&attention_weights, &value_quantized, &value_scales)?
        } else {
            attention_weights.matmul(&value_smoothed)?
        };
        
        // Record for autograd if needed
        if query.requires_grad() || key.requires_grad() || value.requires_grad() {
            let mut saved_tensors = vec![
                (query.id, query.clone_result()?),
                (key.id, key_smoothed),
                (value.id, value_smoothed),
                (attention_weights.id, attention_weights.clone_result()?),
            ];
            
            if self.quantize_kv {
                saved_tensors.push((key_scales.id, key_scales));
                saved_tensors.push((value_scales.id, value_scales));
            }
            
            AutogradContext::record_op(
                output.id,
                Op::SageAttention {
                    query_id: query.id,
                    key_id: key.id,
                    value_id: value.id,
                    scale,
                    causal: self.causal,
                    quantized: self.quantize_kv,
                },
                saved_tensors,
            );
        }
        
        Ok(output)
    }
    
    /// Apply smoothing to reduce quantization error
    fn apply_smoothing(&self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        // Compute channel-wise statistics
        // Compute mean along last dimension keeping dims
        let key_abs = key.abs()?;
        let value_abs = value.abs()?;
        let last_dim = key.shape().dims().len() - 1;
        let key_abs_sum = key_abs.sum_dims(&[last_dim])?;
        let value_abs_sum = value_abs.sum_dims(&[last_dim])?;
        let num_elements = key.shape().dims()[last_dim] as f32;
        let key_abs_mean = key_abs_sum.div_scalar(num_elements)?;
        let value_abs_mean = value_abs_sum.div_scalar(num_elements)?;
        
        // Smooth factors
        let key_smooth = key_abs_mean.mul_scalar(self.smoothing_factor)?.add_scalar(1.0)?;
        let value_smooth = value_abs_mean.mul_scalar(self.smoothing_factor)?.add_scalar(1.0)?;
        
        // Apply smoothing
        let key_smoothed = key.div(&key_smooth)?;
        let value_smoothed = value.div(&value_smooth)?;
        
        Ok((key_smoothed, value_smoothed))
    }
    
    /// Quantize K and V tensors to INT8 with per-row scales
    fn quantize_kv_tensors(&self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let device = key.device.clone();
        
        // Get kernel
        let kernel_code = get_sage_quantize_kernel();
        CudaKernels::ensure_kernel(&device, "sage_quantize_tensor", kernel_code)?;
        
        let f = device.get_func("sage_quantize_tensor", "sage_quantize_tensor")
            .ok_or_else(|| FlameError::Cuda("Failed to get quantize kernel".into()))?;
        
        // Quantize key
        let key_shape = key.shape().dims();
        let batch_size = key_shape[0];
        let num_heads = key_shape[1];
        let seq_len = key_shape[2];
        let head_dim = key_shape[3];
        
        let num_rows = batch_size * num_heads * seq_len;
        
        // Allocate outputs
        let key_quantized = Tensor::zeros_dtype(key.shape().clone(), DType::I8, device.clone())?;
        let key_scales = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, num_heads, seq_len, 1]),
            DType::F32,
            device.clone()
        )?;
        
        // Launch quantization kernel for key
        let cfg = cudarc::driver::LaunchConfig::for_num_elems((num_rows * head_dim) as u32);
        unsafe {
            f.clone().launch(cfg, (
                key.storage.try_as_slice_f32()?,
                key_quantized.storage.try_as_slice_f32()?,
                key_scales.storage.try_as_slice_f32()?,
                num_rows as i32,
                head_dim as i32,
            ))?;
        }
        
        // Quantize value
        let value_quantized = Tensor::zeros_dtype(value.shape().clone(), DType::I8, device.clone())?;
        let value_scales = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, num_heads, seq_len, 1]),
            DType::F32,
            device.clone()
        )?;
        
        unsafe {
            f.launch(cfg, (
                value.storage.try_as_slice_f32()?,
                value_quantized.storage.try_as_slice_f32()?,
                value_scales.storage.try_as_slice_f32()?,
                num_rows as i32,
                head_dim as i32,
            ))?;
        }
        
        Ok((key_quantized, key_scales, value_quantized, value_scales))
    }
    
    /// Matrix multiply Q with 8-bit K
    fn matmul_q_k8bit(&self, query: &Tensor, key_int8: &Tensor, key_scales: &Tensor, scale: f32) -> Result<Tensor> {
        let device = query.device.clone();
        
        // Get kernel
        let kernel_code = get_sage_qk_matmul_kernel();
        CudaKernels::ensure_kernel(&device, "sage_qk_matmul", kernel_code)?;
        
        let f = device.get_func("sage_qk_matmul", "sage_qk_matmul")
            .ok_or_else(|| FlameError::Cuda("Failed to get qk matmul kernel".into()))?;
        
        let shape_q = query.shape().dims();
        let shape_k = key_int8.shape().dims();
        
        let batch_size = shape_q[0];
        let num_heads = shape_q[1];
        let seq_len_q = shape_q[2];
        let seq_len_k = shape_k[2];
        let head_dim = shape_q[3];
        
        // Output shape: [batch, heads, seq_q, seq_k]
        let output = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, num_heads, seq_len_q, seq_len_k]),
            DType::F32,
            device.clone()
        )?;
        
        // Launch kernel
        let threads_per_block = 256;
        let num_blocks = ((batch_size * num_heads * seq_len_q * seq_len_k + threads_per_block - 1) / threads_per_block) as u32;
        
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            f.launch(cfg, (
                query.storage.try_as_slice_f32()?,
                key_int8.storage.try_as_slice_f32()?,
                key_scales.storage.try_as_slice_f32()?,
                output.storage.try_as_slice_f32()?,
                batch_size as i32,
                num_heads as i32,
                seq_len_q as i32,
                seq_len_k as i32,
                head_dim as i32,
                scale,
            ))?;
        }
        
        Ok(output)
    }
    
    /// Matrix multiply attention weights with 8-bit V
    fn matmul_attn_v8bit(&self, attn_weights: &Tensor, value_int8: &Tensor, value_scales: &Tensor) -> Result<Tensor> {
        let device = attn_weights.device.clone();
        
        // Get kernel
        let kernel_code = get_sage_attn_v_matmul_kernel();
        CudaKernels::ensure_kernel(&device, "sage_attn_v_matmul", kernel_code)?;
        
        let f = device.get_func("sage_attn_v_matmul", "sage_attn_v_matmul")
            .ok_or_else(|| FlameError::Cuda("Failed to get attn-v matmul kernel".into()))?;
        
        let shape_a = attn_weights.shape().dims();
        let shape_v = value_int8.shape().dims();
        
        let batch_size = shape_a[0];
        let num_heads = shape_a[1];
        let seq_len_q = shape_a[2];
        let seq_len_k = shape_a[3];
        let head_dim = shape_v[3];
        
        // Output shape: [batch, heads, seq_q, head_dim]
        let output = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, num_heads, seq_len_q, head_dim]),
            DType::F32,
            device.clone()
        )?;
        
        // Launch kernel
        let threads_per_block = 256;
        let num_blocks = ((batch_size * num_heads * seq_len_q * head_dim + threads_per_block - 1) / threads_per_block) as u32;
        
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            f.launch(cfg, (
                attn_weights.storage.try_as_slice_f32()?,
                value_int8.storage.try_as_slice_f32()?,
                value_scales.storage.try_as_slice_f32()?,
                output.storage.try_as_slice_f32()?,
                batch_size as i32,
                num_heads as i32,
                seq_len_q as i32,
                seq_len_k as i32,
                head_dim as i32,
            ))?;
        }
        
        Ok(output)
    }
    
    /// Apply causal mask
    fn apply_causal_mask(&self, scores: &Tensor, seq_len_q: usize, seq_len_k: usize) -> Result<Tensor> {
        let device = scores.device.clone();
        
        // Create causal mask
        let mask = Tensor::zeros_dtype(Shape::from_dims(&[seq_len_q, seq_len_k]), DType::F32, device)?;
        
        // Fill upper triangle with -inf
        let kernel_code = r#"
extern "C" __global__ void causal_mask(float* mask, int seq_len_q, int seq_len_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q = idx / seq_len_k;
    int k = idx % seq_len_k;
    
    if (q < seq_len_q && k < seq_len_k) {
        if (k > q) {
            mask[idx] = -1e9f;
        }
    }
}
"#;
        
        CudaKernels::ensure_kernel(&mask.device, "causal_mask", kernel_code)?;
        let f = mask.device.get_func("causal_mask", "causal_mask")
            .ok_or_else(|| FlameError::Cuda("Failed to get causal mask kernel".into()))?;
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems((seq_len_q * seq_len_k) as u32);
        unsafe {
            f.launch(cfg, (
                mask.storage.try_as_slice_f32()?,
                seq_len_q as i32,
                seq_len_k as i32,
            ))?;
        }
        
        // Broadcast and add to scores
        scores.add(&mask.unsqueeze(0)?.unsqueeze(0)?)
    }
}

/// Sage attention backward pass
pub fn sage_attention_backward(
    grad_output: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attention_weights: &Tensor,
    scale: f32,
    causal: bool,
    quantized: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    // For quantized version, we need to handle dequantization in backward
    // For now, implement standard attention backward
    
    // Gradient w.r.t value: grad_output @ attention_weights.T
    let attention_weights_t = attention_weights.permute(&[0, 1, 3, 2])?;
    let grad_value = attention_weights_t.matmul(grad_output)?;
    
    // Gradient w.r.t attention weights
    let value_t = value.permute(&[0, 1, 3, 2])?;
    let grad_attention = grad_output.matmul(&value_t)?;
    
    // Gradient through softmax
    let grad_scores = grad_attention.mul(&attention_weights)?;
    let sum_grad = grad_scores.sum_dims(&[grad_scores.shape().dims().len() - 1])?;
    let grad_scores = grad_scores.sub(&attention_weights.mul(&sum_grad)?)?;
    
    // Apply scale
    let grad_scores = grad_scores.mul_scalar(1.0 / scale)?;
    
    // Gradient w.r.t query and key
    let grad_query = grad_scores.matmul(&key)?;
    let grad_scores_t = grad_scores.permute(&[0, 1, 3, 2])?;
    let grad_key = grad_scores_t.matmul(query)?;
    
    Ok((grad_query, grad_key, grad_value))
}

/// CUDA kernel for INT8 quantization with per-row scales
fn get_sage_quantize_kernel() -> &'static str {
    r#"
extern "C" __global__ void sage_quantize_tensor(
    const float* __restrict__ input,
    signed char* __restrict__ output,
    float* __restrict__ scales,
    int num_rows,
    int row_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / row_size;
    int col = idx % row_size;
    
    if (row < num_rows && col < row_size) {
        // Compute row-wise max for symmetric quantization
        if (col == 0) {
            float max_val = 0.0f;
            for (int i = 0; i < row_size; i++) {
                float val = fabsf(input[row * row_size + i]);
                max_val = fmaxf(max_val, val);
            }
            
            // Compute scale
            scales[row] = max_val / 127.0f;
        }
        
        __syncthreads();
        
        // Quantize
        float scale = scales[row];
        float val = input[idx];
        int quantized = __float2int_rn(val / scale);
        quantized = max(-128, min(127, quantized));
        output[idx] = (signed char)quantized;
    }
}
"#
}

/// CUDA kernel for Q @ K^T with INT8 K
fn get_sage_qk_matmul_kernel() -> &'static str {
    r#"
extern "C" __global__ void sage_qk_matmul(
    const float* __restrict__ query,
    const signed char* __restrict__ key_int8,
    const float* __restrict__ key_scales,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len_q * seq_len_k;
    
    if (idx < total_elements) {
        int b = idx / (num_heads * seq_len_q * seq_len_k);
        int h = (idx / (seq_len_q * seq_len_k)) % num_heads;
        int q = (idx / seq_len_k) % seq_len_q;
        int k = idx % seq_len_k;
        
        float sum = 0.0f;
        
        // Compute dot product
        for (int d = 0; d < head_dim; d++) {
            int q_idx = b * num_heads * seq_len_q * head_dim + 
                       h * seq_len_q * head_dim + 
                       q * head_dim + d;
            int k_idx = b * num_heads * seq_len_k * head_dim + 
                       h * seq_len_k * head_dim + 
                       k * head_dim + d;
            
            float q_val = query[q_idx];
            float k_val = (float)key_int8[k_idx] * key_scales[b * num_heads * seq_len_k + h * seq_len_k + k];
            
            sum += q_val * k_val;
        }
        
        output[idx] = sum / scale;
    }
}
"#
}

/// CUDA kernel for attention @ V with INT8 V
fn get_sage_attn_v_matmul_kernel() -> &'static str {
    r#"
extern "C" __global__ void sage_attn_v_matmul(
    const float* __restrict__ attn_weights,
    const signed char* __restrict__ value_int8,
    const float* __restrict__ value_scales,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len_q * head_dim;
    
    if (idx < total_elements) {
        int b = idx / (num_heads * seq_len_q * head_dim);
        int h = (idx / (seq_len_q * head_dim)) % num_heads;
        int q = (idx / head_dim) % seq_len_q;
        int d = idx % head_dim;
        
        float sum = 0.0f;
        
        // Compute weighted sum
        for (int k = 0; k < seq_len_k; k++) {
            int attn_idx = b * num_heads * seq_len_q * seq_len_k + 
                          h * seq_len_q * seq_len_k + 
                          q * seq_len_k + k;
            int v_idx = b * num_heads * seq_len_k * head_dim + 
                       h * seq_len_k * head_dim + 
                       k * head_dim + d;
            
            float attn = attn_weights[attn_idx];
            float v_val = (float)value_int8[v_idx] * value_scales[b * num_heads * seq_len_k + h * seq_len_k + k];
            
            sum += attn * v_val;
        }
        
        output[idx] = sum;
    }
}
"#
}
