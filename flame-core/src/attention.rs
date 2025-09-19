use crate::{Tensor, Shape, Result, FlameError, DType};
const MASK_NEG: f32 = -1e4;

#[inline]
fn flash_env_enabled() -> bool {
    std::env::var("FLAME_DISABLE_FLASH").map(|v| v != "1").unwrap_or(true)
}

#[inline]
fn prefer_flash() -> bool {
    cfg!(feature = "flash_attn") && flash_env_enabled()
}
use crate::linear::Linear;
use std::sync::Arc;

/// Multi-head attention configuration
#[derive(Clone)]
pub struct AttentionConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout: f32,
    pub bias: bool,
}

impl AttentionConfig {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
        Self {
            embed_dim,
            num_heads,
            head_dim: embed_dim / num_heads,
            dropout: 0.0,
            bias: true,
        }
    }
}

/// Multi-head attention layer
pub struct MultiHeadAttention {
    pub config: AttentionConfig,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(config: AttentionConfig, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let q_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, &device)?;
        let k_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, &device)?;
        let v_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, &device)?;
        let out_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, &device)?;
        
        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }
    
    /// Forward pass for multi-head attention
    /// query: [batch, seq_len, embed_dim]
    /// key: [batch, seq_len, embed_dim]
    /// value: [batch, seq_len, embed_dim]
    /// mask: Optional [batch, seq_len, seq_len] or [seq_len, seq_len]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let query_shape = query.shape().dims();
        if query_shape.len() != 3 {
            return Err(FlameError::InvalidOperation(
                format!("Expected 3D query tensor, got {:?}", query_shape)
            ));
        }
        
        let batch_size = query_shape[0];
        let seq_len = query_shape[1];
        let embed_dim = query_shape[2];
        
        if embed_dim != self.config.embed_dim {
            return Err(FlameError::InvalidOperation(
                format!("Expected embed_dim {}, got {}", self.config.embed_dim, embed_dim)
            ));
        }
        
        // Project Q, K, V
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;
        
        // Get key/value sequence length (may differ from query for cross-attention)
        let kv_seq_len = key.shape().dims()[1];
        
        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape(&[batch_size, seq_len, self.config.num_heads, self.config.head_dim])?;
        let k = k.reshape(&[batch_size, kv_seq_len, self.config.num_heads, self.config.head_dim])?;
        let v = v.reshape(&[batch_size, kv_seq_len, self.config.num_heads, self.config.head_dim])?;
        
        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;
        
        // Scaled dot-product attention
        let scale = (self.config.head_dim as f32).sqrt();
        let scores = self.scaled_dot_product_attention(&q, &k, &v, mask, scale)?;
        
        // Transpose back to [batch, seq_len, num_heads, head_dim]
        let scores = scores.permute(&[0, 2, 1, 3])?;
        
        // Reshape to [batch, seq_len, embed_dim]
        let scores = scores.reshape(&[batch_size, seq_len, self.config.embed_dim])?;
        
        // Output projection
        self.out_proj.forward(&scores)
    }
    
    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        // q: [batch, num_heads, q_seq_len, head_dim]
        // k, v: [batch, num_heads, kv_seq_len, head_dim]
        
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        // First transpose K: [batch, num_heads, head_dim, kv_seq_len]
        let k_t = k.transpose_dims(2, 3)?;
        
        // Batch matrix multiply: [batch, num_heads, q_seq_len, head_dim] @ [batch, num_heads, head_dim, kv_seq_len]
        // Result: [batch, num_heads, q_seq_len, kv_seq_len]
        let scores = q.bmm(&k_t)?;
        let scores = scores.mul_scalar(1.0 / scale)?;
        
        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            self.apply_attention_mask(&scores, mask)?
        } else {
            scores
        };
        
        // Softmax along last dimension
        let attention = scores.softmax(-1)?;
        
        // Apply attention to values: [batch, num_heads, q_seq_len, kv_seq_len] @ [batch, num_heads, kv_seq_len, head_dim]
        // Result: [batch, num_heads, q_seq_len, head_dim]
        let output = attention.bmm(v)?;
        
        // The backward pass will be handled by the individual operations
        // (bmm, mul_scalar, softmax) which already record their gradients
        
        Ok(output)
    }
    
    /// Extract a single head from the attention tensor
    fn extract_head(&self, tensor: &Tensor, batch: usize, head: usize, dim1: usize, dim2: usize) -> Result<Tensor> {
        let data = tensor.to_vec()?;
        let head_size = dim1 * dim2;
        let batch_stride = self.config.num_heads * head_size;
        let head_stride = head_size;
        
        let start = batch * batch_stride + head * head_stride;
        let end = start + head_size;
        
        Tensor::from_vec(
            data[start..end].to_vec(),
            Shape::from_dims(&[dim1, dim2]),
            tensor.device.clone()
        )
    }
    
    /// Apply attention mask
    fn apply_attention_mask(&self, scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // mask: 1 for positions to attend to, 0 for positions to mask
        // Convert to additive mask: 0 for attend, -inf for mask
        let mask_data = mask.to_vec()?;
        let scores_data = scores.to_vec()?;
        
        let mut result = vec![0.0f32; scores_data.len()];
        for i in 0..result.len() {
            let mask_idx = i % mask_data.len(); // Handle broadcasting
            result[i] = if mask_data[mask_idx] == 0.0 {
                scores_data[i] - 1e9 // Large negative value instead of -inf
            } else {
                scores_data[i]
            };
        }
        
        Tensor::from_vec(result, scores.shape().clone(), scores.device.clone())
    }
    
    /// Softmax along the last dimension
    fn softmax(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let shape = tensor.shape().dims();
        let data = tensor.to_vec()?;
        
        let mut result = vec![0.0f32; data.len()];
        let dim_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();
        
        for i in 0..outer_size {
            for j in 0..inner_size {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..dim_size {
                    let idx = i * dim_size * inner_size + k * inner_size + j;
                    max_val = max_val.max(data[idx]);
                }
                
                // Compute exp and sum
                let mut sum = 0.0;
                for k in 0..dim_size {
                    let idx = i * dim_size * inner_size + k * inner_size + j;
                    let exp_val = (data[idx] - max_val).exp();
                    result[idx] = exp_val;
                    sum += exp_val;
                }
                
                // Normalize
                for k in 0..dim_size {
                    let idx = i * dim_size * inner_size + k * inner_size + j;
                    result[idx] /= sum;
                }
            }
        }
        
        Tensor::from_vec(result, tensor.shape().clone(), tensor.device.clone())
    }
}

// --- Stable SDPA/FlashAttention helpers (public) ---

/// Validate Q/K/V basic shape relationships. For brevity, only dimensionality is checked here.
fn validate_qkv(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<()> {
    let qd = q.shape().dims();
    let kd = k.shape().dims();
    let vd = v.shape().dims();
    if !(qd.len() == 4 && kd.len() == 4 && vd.len() == 4) {
        return Err(FlameError::InvalidInput("Q,K,V must be [B,H,SEQ,D]".into()));
    }
    let (b, h, q_len, dq) = (qd[0], qd[1], qd[2], qd[3]);
    let (bk, hk, k_len, dk) = (kd[0], kd[1], kd[2], kd[3]);
    let (bv, hv, kv, dv) = (vd[0], vd[1], vd[2], vd[3]);
    if !(b == bk && b == bv && h == hk && h == hv && dq == dk && dk == dv && k_len == kv) {
        return Err(FlameError::InvalidInput("Q,K,V dimension mismatch".into()));
    }
    Ok(())
}

fn validate_qkv_shapes(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<(usize, usize, usize, usize, usize)> {
    validate_qkv(q, k, v)?;
    let qd = q.shape().dims();
    Ok((qd[0], qd[1], qd[2], k.shape().dims()[2], qd[3]))
}

fn validate_mask_shape(mask: &Tensor, b: usize, h: usize, q_len: usize, k_len: usize) -> Result<()> {
    let md = mask.shape().dims();
    if md.len() != 4 {
        return Err(FlameError::InvalidInput("Mask must be 4D: [B,H,Q,K] / [B,1,Q,K] / [1,1,Q,K]".into()));
    }
    let (mb, mh, mq, mk) = (md[0], md[1], md[2], md[3]);
    let ok_b = mb == b || mb == 1;
    let ok_h = mh == h || mh == 1;
    let ok_q = mq == q_len;
    let ok_k = mk == k_len;
    if !(ok_b && ok_h && ok_q && ok_k) {
        return Err(FlameError::InvalidInput("Mask dims must broadcast to [B,H,Q,K]".into()));
    }
    match mask.dtype() {
        DType::Bool | DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        _ => Err(FlameError::InvalidInput("Mask dtype must be bool or float (additive)".into())),
    }
}

fn ensure_fp32_compute(x: &Tensor) -> Result<Tensor> {
    if matches!(x.dtype(), DType::BF16) { x.to_dtype(DType::F32) } else { x.clone_result() }
}

fn maybe_downcast_to_input_dtype(y: &Tensor, reference: &Tensor) -> Result<Tensor> {
    if matches!(reference.dtype(), DType::BF16) { y.to_dtype(DType::BF16) } else { y.clone_result() }
}

/// Scaled dot-product attention with FP32 softmax/reductions. Accepts [B,H,S,D] tensors.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    crate::sdpa::forward(q, k, v, mask)
}

/// FlashAttention wrapper; computes in FP32 and casts to match Q input when needed.
#[cfg(feature = "flash_attn")]
pub fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let (b, h, q_len, k_len, _d) = validate_qkv_shapes(q, k, v)?;
    if let Some(m) = mask { validate_mask_shape(m, b, h, q_len, k_len)?; }
    let q32 = ensure_fp32_compute(q)?;
    let k32 = ensure_fp32_compute(k)?;
    let v32 = ensure_fp32_compute(v)?;
    // Build additive FP32 mask if present
    let add_mask_owned;
    let add_mask_ref = if let Some(m) = mask {
        add_mask_owned = if matches!(m.dtype(), DType::Bool) {
            m.to_dtype(DType::F32)?.mul_scalar(MASK_NEG)?
        } else { ensure_fp32_compute(m)? };
        Some(&add_mask_owned)
    } else { None };
    let y = crate::flash_attention::flash_attention_forward(&q32, &k32, &v32, add_mask_ref, None, false)?;
    maybe_downcast_to_input_dtype(&y, q)
}

/// Generic attend: prefer FlashAttention when feature enabled, else SDPA.
pub fn attend(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    if prefer_flash() {
        #[cfg(feature = "flash_attn")] 
        {
            if let Ok(y) = flash_attn(q, k, v, mask) { return Ok(y); }
        }
    }
    sdpa(q, k, v, mask)
}


/// GeGLU activation for transformer FFN
pub struct GeGLU {
    pub proj: Linear,
}

impl GeGLU {
    pub fn new(dim_in: usize, dim_out: usize, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        // Project to 2x dim_out for gating
        let proj = Linear::new(dim_in, dim_out * 2, true, &device)?;
        Ok(Self { proj })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Project and split
        let projected = self.proj.forward(x)?;
        let dims = projected.shape().dims();
        let last_dim = dims[dims.len() - 1];
        
        // Split into two halves
        let half_dim = last_dim / 2;
        let data = projected.to_vec()?;
        
        let mut gate_data = vec![0.0f32; data.len() / 2];
        let mut value_data = vec![0.0f32; data.len() / 2];
        
        let outer_size: usize = dims[..dims.len()-1].iter().product();
        
        for i in 0..outer_size {
            for j in 0..half_dim {
                let idx = i * last_dim + j;
                let gate_idx = i * half_dim + j;
                value_data[gate_idx] = data[idx];
                gate_data[gate_idx] = data[idx + half_dim];
            }
        }
        
        // Apply GELU to gate
        for i in 0..gate_data.len() {
            gate_data[i] = gelu(gate_data[i]);
        }
        
        // Multiply value by gated output
        let mut output_data = vec![0.0f32; gate_data.len()];
        for i in 0..output_data.len() {
            output_data[i] = value_data[i] * gate_data[i];
        }
        
        let mut output_shape = dims.to_vec();
        output_shape[dims.len() - 1] = half_dim;
        
        Tensor::from_vec(output_data, Shape::from_dims(&output_shape), projected.device.clone())
    }
}

/// GELU activation function
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Feed-forward network with GeGLU activation
pub struct FeedForward {
    pub project_in: GeGLU,
    pub linear: Linear,
}

impl FeedForward {
    pub fn new(
        dim: usize,
        dim_out: Option<usize>,
        mult: usize,
        device: Arc<cudarc::driver::CudaDevice>
    ) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        
        let project_in = GeGLU::new(dim, inner_dim, device.clone())?;
        let linear = Linear::new(inner_dim, dim_out, true, &device)?;
        
        Ok(Self { project_in, linear })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.project_in.forward(x)?;
        self.linear.forward(&x)
    }
}

/// Rotary Position Embedding (RoPE)
pub struct RotaryEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
    pub theta: f32,
    pub freqs_cos: Option<Tensor>,
    pub freqs_sin: Option<Tensor>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self {
            dim,
            max_seq_len,
            theta,
            freqs_cos: None,
            freqs_sin: None,
        }
    }
    
    /// Initialize frequency tensors
    pub fn init_freqs(&mut self, device: Arc<cudarc::driver::CudaDevice>) -> Result<()> {
        if self.dim % 2 != 0 {
            return Err(FlameError::InvalidOperation(
                format!("RoPE dim must be even, got {}", self.dim)
            ));
        }
        
        let half_dim = self.dim / 2;
        let mut freqs = vec![0.0f32; half_dim];
        
        for i in 0..half_dim {
            freqs[i] = 1.0 / self.theta.powf(2.0 * i as f32 / self.dim as f32);
        }
        
        // Create position indices
        let mut cos_data = Vec::new();
        let mut sin_data = Vec::new();
        
        for pos in 0..self.max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f32 * freqs[i];
                cos_data.push(angle.cos());
                cos_data.push(angle.cos()); // Duplicate for interleaved format
                sin_data.push(angle.sin());
                sin_data.push(angle.sin());
            }
        }
        
        self.freqs_cos = Some(Tensor::from_vec(
            cos_data,
            Shape::from_dims(&[self.max_seq_len, self.dim]),
            device.clone()
        )?);
        
        self.freqs_sin = Some(Tensor::from_vec(
            sin_data,
            Shape::from_dims(&[self.max_seq_len, self.dim]),
            device
        )?);
        
        Ok(())
    }
    
    /// Apply rotary embeddings to query or key tensors
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let freqs_cos = self.freqs_cos.as_ref()
            .ok_or(FlameError::InvalidOperation("RoPE not initialized".into()))?;
        let freqs_sin = self.freqs_sin.as_ref()
            .ok_or(FlameError::InvalidOperation("RoPE not initialized".into()))?;
        
        let dims = x.shape().dims();
        let batch_size = dims[0];
        let num_heads = dims[1];
        let head_dim = dims[3];
        
        if head_dim != self.dim {
            return Err(FlameError::InvalidOperation(
                format!("Expected head_dim {}, got {}", self.dim, head_dim)
            ));
        }
        
        let x_data = x.to_vec()?;
        let cos_data = freqs_cos.to_vec()?;
        let sin_data = freqs_sin.to_vec()?;
        
        let mut output_data = vec![0.0f32; x_data.len()];
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    for d in 0..head_dim / 2 {
                        let idx1 = b * num_heads * seq_len * head_dim
                                + h * seq_len * head_dim
                                + s * head_dim
                                + 2 * d;
                        let idx2 = idx1 + 1;
                        
                        let cos_idx = s * self.dim + 2 * d;
                        let sin_idx = cos_idx;
                        
                        let x1 = x_data[idx1];
                        let x2 = x_data[idx2];
                        
                        // Apply rotation
                        output_data[idx1] = x1 * cos_data[cos_idx] - x2 * sin_data[sin_idx];
                        output_data[idx2] = x1 * sin_data[sin_idx] + x2 * cos_data[cos_idx];
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, x.shape().clone(), x.device.clone())
    }
}

/// Basic Transformer Block
pub struct TransformerBlock {
    pub self_attn: MultiHeadAttention,
    pub cross_attn: Option<MultiHeadAttention>,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: Option<LayerNorm>,
}

impl TransformerBlock {
    pub fn new(
        dim: usize,
        num_heads: usize,
        ff_mult: usize,
        use_cross_attn: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let attn_config = AttentionConfig::new(dim, num_heads);
        
        let self_attn = MultiHeadAttention::new(attn_config.clone(), device.clone())?;
        let cross_attn = if use_cross_attn {
            Some(MultiHeadAttention::new(attn_config, device.clone())?)
        } else {
            None
        };
        
        let ff = FeedForward::new(dim, None, ff_mult, device.clone())?;
        
        let norm1 = LayerNorm::new(dim, 1e-5, true, device.clone())?;
        let norm2 = LayerNorm::new(dim, 1e-5, true, device.clone())?;
        let norm3 = if use_cross_attn {
            Some(LayerNorm::new(dim, 1e-5, true, device)?)
        } else {
            None
        };
        
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
            norm1,
            norm2,
            norm3,
        })
    }
    
    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        self_attn_mask: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self attention with residual
        let normed = self.norm1.forward(x)?;
        let self_attn_out = self.self_attn.forward(&normed, &normed, &normed, self_attn_mask)?;
        let x = x.add(&self_attn_out)?;
        
        // Cross attention if available
        let x = if let (Some(cross_attn), Some(norm3), Some(ctx)) = 
            (&self.cross_attn, &self.norm3, context) {
            let normed = norm3.forward(&x)?;
            let cross_attn_out = cross_attn.forward(&normed, ctx, ctx, cross_attn_mask)?;
            x.add(&cross_attn_out)?
        } else {
            x
        };
        
        // Feed-forward with residual
        let normed = self.norm2.forward(&x)?;
        let ff_out = self.ff.forward(&normed)?;
        x.add(&ff_out)
    }
}

/// Layer normalization
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    pub fn new(
        normalized_shape: usize,
        eps: f32,
        elementwise_affine: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let (weight, bias) = if elementwise_affine {
            let weight = Tensor::from_vec(
                vec![1.0f32; normalized_shape],
                Shape::from_dims(&[normalized_shape]),
                device.clone(),
            )?;
            let bias = Tensor::zeros(Shape::from_dims(&[normalized_shape]), device)?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        
        Ok(Self {
            normalized_shape: vec![normalized_shape],
            eps,
            elementwise_affine,
            weight,
            bias,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims();
        let normalized_dims = self.normalized_shape.len();
        
        if dims.len() < normalized_dims {
            return Err(FlameError::InvalidOperation(
                "Input dimensions less than normalized dimensions".into()
            ));
        }
        
        // Check that last dimensions match
        for i in 0..normalized_dims {
            if dims[dims.len() - normalized_dims + i] != self.normalized_shape[i] {
                return Err(FlameError::InvalidOperation(
                    "Shape mismatch in layer norm".into()
                ));
            }
        }
        
        let data = x.to_vec()?;
        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = data.len() / norm_size;
        
        let mut output = vec![0.0f32; data.len()];
        
        for b in 0..batch_size {
            let start = b * norm_size;
            let end = (b + 1) * norm_size;
            
            // Calculate mean
            let mut mean = 0.0f32;
            for i in start..end {
                mean += data[i];
            }
            mean /= norm_size as f32;
            
            // Calculate variance
            let mut var = 0.0f32;
            for i in start..end {
                let diff = data[i] - mean;
                var += diff * diff;
            }
            var /= norm_size as f32;
            
            // Normalize
            let std = (var + self.eps).sqrt();
            for i in start..end {
                output[i] = (data[i] - mean) / std;
            }
            
            // Apply affine transformation if enabled
            if let (Some(weight), Some(bias)) = (&self.weight, &self.bias) {
                let weight_data = weight.to_vec()?;
                let bias_data = bias.to_vec()?;
                
                for i in 0..norm_size {
                    let idx = start + i;
                    output[idx] = output[idx] * weight_data[i] + bias_data[i];
                }
            }
        }
        
        Tensor::from_vec(output, x.shape().clone(), x.device.clone())
    }
}
