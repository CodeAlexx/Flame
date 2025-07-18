//! Flux model building blocks
//! 
//! This module provides the core blocks for the Flux diffusion model,
//! including double stream blocks and single stream blocks.

use crate::{
    Tensor, Shape, Result,
    attention::{MultiHeadAttention, AttentionConfig},
    linear::Linear,
    modulated_blocks::{Modulation, QKNorm, RMSNorm},
    norm::{LayerNorm, RMSNorm as RMSNormModule}
};
use std::sync::Arc;

/// Configuration for Flux model
#[derive(Clone)]
pub struct FluxConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: f32,
    pub theta: f32,
    pub qkv_bias: bool,
    pub guidance_embed: bool,
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3072,
            num_heads: 24,
            head_dim: 128,
            mlp_ratio: 4.0,
            theta: 10_000.0,
            qkv_bias: true,
            guidance_embed: false,
        }
    }
}

/// Self-attention module for Flux
pub struct FluxSelfAttention {
    pub qkv: Linear,
    pub norm: QKNorm,
    pub proj: Linear,
    pub num_heads: usize,
}

impl FluxSelfAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = Linear::new(dim, dim * 3, qkv_bias, &device)?;
        let norm = QKNorm::new(head_dim, device.clone())?;
        let proj = Linear::new(dim, dim, true, &device)?;
        
        Ok(Self {
            qkv,
            norm,
            proj,
            num_heads,
        })
    }
    
    /// Extract Q, K, V from combined tensor
    pub fn qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.qkv.forward(x)?;
        let dims = qkv.shape().dims();
        let batch = dims[0];
        let seq_len = dims[1];
        let total_dim = dims[2];
        let head_dim = total_dim / 3 / self.num_heads;
        
        // Reshape to separate Q, K, V
        let qkv = qkv.reshape(&[batch, seq_len, 3, self.num_heads, head_dim])?;
        
        // Extract Q, K, V
        let data = qkv.to_vec()?;
        let qkv_size = seq_len * self.num_heads * head_dim;
        let mut q_data = vec![0.0f32; batch * qkv_size];
        let mut k_data = vec![0.0f32; batch * qkv_size];
        let mut v_data = vec![0.0f32; batch * qkv_size];
        
        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..head_dim {
                        let src_base = b * seq_len * 3 * self.num_heads * head_dim
                                     + s * 3 * self.num_heads * head_dim
                                     + h * head_dim + d;
                        let dst_idx = b * qkv_size + s * self.num_heads * head_dim + h * head_dim + d;
                        
                        q_data[dst_idx] = data[src_base];
                        k_data[dst_idx] = data[src_base + self.num_heads * head_dim];
                        v_data[dst_idx] = data[src_base + 2 * self.num_heads * head_dim];
                    }
                }
            }
        }
        
        // Create tensors with shape [batch, seq_len, num_heads, head_dim]
        let shape = Shape::from_dims(&[batch, seq_len, self.num_heads, head_dim]);
        let q = Tensor::from_vec(q_data, shape.clone(), qkv.device().clone())?;
        let k = Tensor::from_vec(k_data, shape.clone(), qkv.device().clone())?;
        let v = Tensor::from_vec(v_data, shape, qkv.device().clone())?;
        
        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;
        
        // Apply normalization to Q and K
        let (q_normed, k_normed) = self.norm.forward(&q, &k)?;
        
        Ok((q_normed, k_normed, v))
    }
    
    /// Forward pass with RoPE
    pub fn forward(&self, x: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (q, k, v) = self.qkv(x)?;
        
        // Apply RoPE
        let q = apply_rope(&q, pe)?;
        let k = apply_rope(&k, pe)?;
        
        // Attention
        let attn = scaled_dot_product_attention(&q, &k, &v)?;
        
        // Transpose back and reshape
        let attn = attn.transpose_dims(1, 2)?
            .flatten_from(2)?;
        
        // Output projection
        self.proj.forward(&attn)
    }
}

/// Apply rotary position embeddings
fn apply_rope(x: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    let batch = dims[0];
    let num_heads = dims[1];
    let seq_len = dims[2];
    let head_dim = dims[3];
    
    let x_data = x.to_vec()?;
    let pe_data = pe.to_vec()?;
    let mut output = vec![0.0f32; x_data.len()];
    
    // PE shape is [seq_len, head_dim]
    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim / 2 {
                    let idx = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + s * head_dim;
                    let pe_idx = s * head_dim;
                    
                    let cos_val = pe_data[pe_idx + 2 * d];
                    let sin_val = pe_data[pe_idx + 2 * d + 1];
                    
                    let x0 = x_data[idx + 2 * d];
                    let x1 = x_data[idx + 2 * d + 1];
                    
                    output[idx + 2 * d] = x0 * cos_val - x1 * sin_val;
                    output[idx + 2 * d + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
    
    Tensor::from_vec(output, x.shape().clone(), x.device().clone())
}

/// Scaled dot-product attention
fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let head_dim = q.shape().dims()[3];
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Q @ K^T
    let scores = q.bmm(&k.transpose_dims(2, 3)?)?;
    let scores = scores.scale(scale)?;
    
    // Softmax
    let attn = softmax_last_dim(&scores)?;
    
    // Attention @ V
    attn.bmm(v)
}

/// Softmax over last dimension
fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    let data = x.to_vec()?;
    let last_dim = dims[dims.len() - 1];
    let outer_size = data.len() / last_dim;
    
    let mut output = vec![0.0f32; data.len()];
    
    for i in 0..outer_size {
        let offset = i * last_dim;
        
        // Find max
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..last_dim {
            max_val = max_val.max(data[offset + j]);
        }
        
        // Exp and sum
        let mut sum = 0.0f32;
        for j in 0..last_dim {
            let exp_val = (data[offset + j] - max_val).exp();
            output[offset + j] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for j in 0..last_dim {
            output[offset + j] /= sum;
        }
    }
    
    Tensor::from_vec(output, x.shape().clone(), x.device().clone())
}

/// MLP module for Flux
pub struct FluxMLP {
    pub lin1: Linear,
    pub lin2: Linear,
}

impl FluxMLP {
    pub fn new(
        in_dim: usize,
        mlp_dim: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let lin1 = Linear::new(in_dim, mlp_dim, true, &device)?;
        let lin2 = Linear::new(mlp_dim, in_dim, true, &device)?;
        
        Ok(Self { lin1, lin2 })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.lin1.forward(x)?;
        let x = x.gelu()?;
        self.lin2.forward(&x)
    }
}

/// Double stream block for Flux
/// Processes image and text streams separately then combines
pub struct DoubleStreamBlock {
    pub img_mod: Modulation,
    pub img_norm1: LayerNorm,
    pub img_attn: FluxSelfAttention,
    pub img_norm2: LayerNorm,
    pub img_mlp: FluxMLP,
    
    pub txt_mod: Modulation,
    pub txt_norm1: LayerNorm,
    pub txt_attn: FluxSelfAttention,
    pub txt_norm2: LayerNorm,
    pub txt_mlp: FluxMLP,
}

impl DoubleStreamBlock {
    pub fn new(
        config: &FluxConfig,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let mlp_hidden = (hidden_size as f32 * config.mlp_ratio) as usize;
        
        // Image blocks
        let img_mod = Modulation::new(hidden_size, hidden_size, 2, device.clone())?;
        let img_norm1 = LayerNorm::new(hidden_size, 1e-6, device.clone())?;
        let img_attn = FluxSelfAttention::new(hidden_size, config.num_heads, config.qkv_bias, device.clone())?;
        let img_norm2 = LayerNorm::new(hidden_size, 1e-6, device.clone())?;
        let img_mlp = FluxMLP::new(hidden_size, mlp_hidden, device.clone())?;
        
        // Text blocks
        let txt_mod = Modulation::new(hidden_size, hidden_size, 2, device.clone())?;
        let txt_norm1 = LayerNorm::new(hidden_size, 1e-6, device.clone())?;
        let txt_attn = FluxSelfAttention::new(hidden_size, config.num_heads, config.qkv_bias, device.clone())?;
        let txt_norm2 = LayerNorm::new(hidden_size, 1e-6, device.clone())?;
        let txt_mlp = FluxMLP::new(hidden_size, mlp_hidden, device)?;
        
        Ok(Self {
            img_mod, img_norm1, img_attn, img_norm2, img_mlp,
            txt_mod, txt_norm1, txt_attn, txt_norm2, txt_mlp,
        })
    }
    
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Get modulations
        let img_mods = self.img_mod.forward(vec)?;
        let img_mod1 = &img_mods[0];
        let img_mod2 = &img_mods[1];
        
        let txt_mods = self.txt_mod.forward(vec)?;
        let txt_mod1 = &txt_mods[0];
        let txt_mod2 = &txt_mods[1];
        
        // Process image and text with cross-attention
        let img_normed = self.img_norm1.forward(img)?;
        let img_modulated = scale_shift(&img_normed, img_mod1)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;
        
        let txt_normed = self.txt_norm1.forward(txt)?;
        let txt_modulated = scale_shift(&txt_normed, txt_mod1)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;
        
        // Concatenate for cross-attention
        let q = cat_tensors(&[&txt_q, &img_q], 2)?;
        let k = cat_tensors(&[&txt_k, &img_k], 2)?;
        let v = cat_tensors(&[&txt_v, &img_v], 2)?;
        
        // Apply RoPE
        let q = apply_rope(&q, pe)?;
        let k = apply_rope(&k, pe)?;
        
        // Attention
        let attn = scaled_dot_product_attention(&q, &k, &v)?;
        
        // Split attention output
        let txt_seq_len = txt.shape().dims()[1];
        let txt_attn = slice_seq_dim(&attn, 0, txt_seq_len)?;
        let img_attn = slice_seq_dim(&attn, txt_seq_len, attn.shape().dims()[2])?;
        
        // Transpose and project
        let txt_attn = txt_attn.transpose_dims(1, 2)?.flatten_from(2)?;
        let txt_attn = self.txt_attn.proj.forward(&txt_attn)?;
        let txt_attn = gate(&txt_attn, txt_mod1)?;
        let txt = txt.add(&txt_attn)?;
        
        let img_attn = img_attn.transpose_dims(1, 2)?.flatten_from(2)?;
        let img_attn = self.img_attn.proj.forward(&img_attn)?;
        let img_attn = gate(&img_attn, img_mod1)?;
        let img = img.add(&img_attn)?;
        
        // MLP blocks
        let img_normed = self.img_norm2.forward(&img)?;
        let img_mlp = self.img_mlp.forward(&scale_shift(&img_normed, img_mod2)?)?;
        let img = img.add(&gate(&img_mlp, img_mod2)?)?;
        
        let txt_normed = self.txt_norm2.forward(&txt)?;
        let txt_mlp = self.txt_mlp.forward(&scale_shift(&txt_normed, txt_mod2)?)?;
        let txt = txt.add(&gate(&txt_mlp, txt_mod2)?)?;
        
        Ok((img, txt))
    }
}

/// Single stream block for Flux
/// Processes concatenated image and text
pub struct SingleStreamBlock {
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm: QKNorm,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub mlp_hidden: usize,
}

impl SingleStreamBlock {
    pub fn new(
        config: &FluxConfig,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let mlp_hidden = (hidden_size as f32 * config.mlp_ratio) as usize;
        let head_dim = config.head_dim;
        
        let linear1 = Linear::new(hidden_size, hidden_size * 3 + mlp_hidden, true, &device)?;
        let linear2 = Linear::new(hidden_size + mlp_hidden, hidden_size, true, &device)?;
        let norm = QKNorm::new(head_dim, device.clone())?;
        
        Ok(Self {
            linear1,
            linear2,
            norm,
            hidden_size,
            num_heads: config.num_heads,
            mlp_hidden,
        })
    }
    
    pub fn forward(&self, x: &Tensor, vec: &Tensor, pe: &Tensor) -> Result<Tensor> {
        // Modulation
        let mod_out = self.linear1.forward(vec)?;
        let dims = mod_out.shape().dims();
        let batch = dims[0];
        
        // Split modulation
        let data = mod_out.to_vec()?;
        let mut shift_data = vec![0.0f32; batch * self.hidden_size];
        let mut scale_data = vec![0.0f32; batch * self.hidden_size];
        let mut gate_data = vec![0.0f32; batch * self.hidden_size];
        
        for b in 0..batch {
            for i in 0..self.hidden_size {
                shift_data[b * self.hidden_size + i] = data[b * dims[1] + i];
                scale_data[b * self.hidden_size + i] = data[b * dims[1] + self.hidden_size + i];
                gate_data[b * self.hidden_size + i] = data[b * dims[1] + 2 * self.hidden_size + i];
            }
        }
        
        let shift = Tensor::from_vec(shift_data, Shape::from_dims(&[batch, self.hidden_size]), x.device().clone())?;
        let scale = Tensor::from_vec(scale_data, Shape::from_dims(&[batch, self.hidden_size]), x.device().clone())?;
        let gate_t = Tensor::from_vec(gate_data, Shape::from_dims(&[batch, self.hidden_size]), x.device().clone())?;
        
        // Normalize and modulate
        let x_mod = scale_shift(x, &scale)?;
        let x_mod = x_mod.add(&broadcast_to_seq_len(&shift, x.shape().dims()[1])?)?;
        
        // Extract QKV and MLP from linear1 output
        let qkv_mlp = self.linear1.forward(&x_mod)?;
        let qkv = slice_last_dim(&qkv_mlp, 0, 3 * self.hidden_size)?;
        let mlp = slice_last_dim(&qkv_mlp, 3 * self.hidden_size, self.mlp_hidden)?;
        
        // Process attention
        let head_dim = self.hidden_size / self.num_heads;
        let seq_len = x.shape().dims()[1];
        
        // Reshape QKV
        let qkv = qkv.reshape(&[batch, seq_len, 3, self.num_heads, head_dim])?;
        let q = extract_qkv(&qkv, 0)?;
        let k = extract_qkv(&qkv, 1)?;
        let v = extract_qkv(&qkv, 2)?;
        
        // Normalize Q and K
        let (q, k) = self.norm.forward(&q, &k)?;
        
        // Apply RoPE
        let q = apply_rope(&q, pe)?;
        let k = apply_rope(&k, pe)?;
        
        // Attention
        let attn = scaled_dot_product_attention(&q, &k, &v)?;
        let attn = attn.transpose_dims(1, 2)?.flatten_from(2)?;
        
        // Combine attention and MLP outputs
        let mlp_out = mlp.gelu()?;
        // Concatenate along last dimension (features)
        let combined = cat_last_dim(&attn, &mlp_out)?;
        let output = self.linear2.forward(&combined)?;
        
        // Apply gating and residual
        let output = gate(&output, &gate_t)?;
        x.add(&output)
    }
}

// Helper functions

/// Scale and shift modulation
fn scale_shift(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let x_dims = x.shape().dims();
    let batch = x_dims[0];
    let seq_len = x_dims[1];
    let hidden = x_dims[2];
    
    let x_data = x.to_vec()?;
    let scale_data = scale.to_vec()?;
    let mut output = vec![0.0f32; x_data.len()];
    
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..hidden {
                let idx = b * seq_len * hidden + s * hidden + h;
                let scale_idx = b * hidden + h;
                output[idx] = x_data[idx] * (1.0 + scale_data[scale_idx]);
            }
        }
    }
    
    Tensor::from_vec(output, x.shape().clone(), x.device().clone())
}

/// Apply gating
fn gate(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let x_dims = x.shape().dims();
    let batch = x_dims[0];
    let seq_len = x_dims[1];
    let hidden = x_dims[2];
    
    let x_data = x.to_vec()?;
    let gate_data = gate.to_vec()?;
    let mut output = vec![0.0f32; x_data.len()];
    
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..hidden {
                let idx = b * seq_len * hidden + s * hidden + h;
                let gate_idx = b * hidden + h;
                output[idx] = x_data[idx] * gate_data[gate_idx];
            }
        }
    }
    
    Tensor::from_vec(output, x.shape().clone(), x.device().clone())
}

/// Concatenate tensors along sequence dimension
fn cat_tensors(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(crate::FlameError::InvalidOperation("Cannot concatenate empty tensor list".into()));
    }
    
    let first = tensors[0];
    let dims = first.shape().dims();
    let device = first.device();
    
    // Validate all tensors have same shape except concat dimension
    for t in tensors {
        let t_dims = t.shape().dims();
        if t_dims.len() != dims.len() {
            return Err(crate::FlameError::InvalidOperation("All tensors must have same number of dimensions".into()));
        }
        for (i, (&d1, &d2)) in dims.iter().zip(t_dims.iter()).enumerate() {
            if i != dim && d1 != d2 {
                return Err(crate::FlameError::InvalidOperation(
                    format!("Dimension {} must match: {} vs {}", i, d1, d2)
                ));
            }
        }
    }
    
    // Calculate total size along concatenation dimension
    let mut total_size = 0;
    for t in tensors {
        total_size += t.shape().dims()[dim];
    }
    
    // Create output shape
    let mut out_dims = dims.to_vec();
    out_dims[dim] = total_size;
    
    // Handle different tensor dimensions
    match dims.len() {
        3 => {
            // [batch, seq, hidden] tensors
            let batch = dims[0];
            let hidden = dims[2];
            let mut output_data = vec![0.0f32; batch * total_size * hidden];
            
            let mut offset = 0;
            for t in tensors {
                let t_data = t.to_vec()?;
                let t_seq = t.shape().dims()[1];
                
                for b in 0..batch {
                    for s in 0..t_seq {
                        for h in 0..hidden {
                            let src_idx = b * t_seq * hidden + s * hidden + h;
                            let dst_idx = b * total_size * hidden + (offset + s) * hidden + h;
                            output_data[dst_idx] = t_data[src_idx];
                        }
                    }
                }
                offset += t_seq;
            }
            
            Tensor::from_vec(output_data, Shape::from_dims(&out_dims), device.clone())
        }
        4 => {
            // [batch, heads, seq, dim] tensors
            let batch = dims[0];
            let heads = dims[1];
            let head_dim = dims[3];
            let mut output_data = vec![0.0f32; batch * heads * total_size * head_dim];
            
            let mut offset = 0;
            for t in tensors {
                let t_data = t.to_vec()?;
                let t_seq = t.shape().dims()[2];
                
                for b in 0..batch {
                    for h in 0..heads {
                        for s in 0..t_seq {
                            for d in 0..head_dim {
                                let src_idx = b * heads * t_seq * head_dim 
                                            + h * t_seq * head_dim 
                                            + s * head_dim 
                                            + d;
                                let dst_idx = b * heads * total_size * head_dim 
                                            + h * total_size * head_dim 
                                            + (offset + s) * head_dim 
                                            + d;
                                output_data[dst_idx] = t_data[src_idx];
                            }
                        }
                    }
                }
                offset += t_seq;
            }
            
            Tensor::from_vec(output_data, Shape::from_dims(&out_dims), device.clone())
        }
        _ => Err(crate::FlameError::InvalidOperation(
            format!("cat_tensors not implemented for {} dimensions", dims.len())
        ))
    }
}

/// Slice along sequence dimension
fn slice_seq_dim(x: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    let batch = dims[0];
    let num_heads = dims[1];
    let head_dim = dims[3];
    let slice_len = end - start;
    
    let x_data = x.to_vec()?;
    let mut output_data = vec![0.0f32; batch * num_heads * slice_len * head_dim];
    
    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..slice_len {
                for d in 0..head_dim {
                    let src_idx = b * num_heads * dims[2] * head_dim
                                + h * dims[2] * head_dim
                                + (start + s) * head_dim
                                + d;
                    let dst_idx = b * num_heads * slice_len * head_dim
                                + h * slice_len * head_dim
                                + s * head_dim
                                + d;
                    output_data[dst_idx] = x_data[src_idx];
                }
            }
        }
    }
    
    Tensor::from_vec(
        output_data,
        Shape::from_dims(&[batch, num_heads, slice_len, head_dim]),
        x.device().clone()
    )
}

/// Broadcast to sequence length
fn broadcast_to_seq_len(x: &Tensor, seq_len: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    let batch = dims[0];
    let hidden = dims[1];
    
    let x_data = x.to_vec()?;
    let mut output_data = vec![0.0f32; batch * seq_len * hidden];
    
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..hidden {
                let src_idx = b * hidden + h;
                let dst_idx = b * seq_len * hidden + s * hidden + h;
                output_data[dst_idx] = x_data[src_idx];
            }
        }
    }
    
    Tensor::from_vec(
        output_data,
        Shape::from_dims(&[batch, seq_len, hidden]),
        x.device().clone()
    )
}

/// Slice last dimension
fn slice_last_dim(x: &Tensor, start: usize, size: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    let batch = dims[0];
    let seq_len = dims[1];
    
    let x_data = x.to_vec()?;
    let mut output_data = vec![0.0f32; batch * seq_len * size];
    
    for b in 0..batch {
        for s in 0..seq_len {
            for i in 0..size {
                let src_idx = b * seq_len * dims[2] + s * dims[2] + start + i;
                let dst_idx = b * seq_len * size + s * size + i;
                output_data[dst_idx] = x_data[src_idx];
            }
        }
    }
    
    Tensor::from_vec(
        output_data,
        Shape::from_dims(&[batch, seq_len, size]),
        x.device().clone()
    )
}

/// Concatenate two tensors along last dimension
fn cat_last_dim(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    
    if a_dims.len() != b_dims.len() {
        return Err(crate::FlameError::InvalidOperation(
            format!("Tensors must have same number of dimensions: {} vs {}", a_dims.len(), b_dims.len())
        ));
    }
    
    // Check all dimensions match except last
    for i in 0..a_dims.len()-1 {
        if a_dims[i] != b_dims[i] {
            return Err(crate::FlameError::InvalidOperation(
                format!("Dimension {} must match: {} vs {}", i, a_dims[i], b_dims[i])
            ));
        }
    }
    
    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;
    
    // Calculate output shape
    let mut out_dims = a_dims.to_vec();
    out_dims[a_dims.len()-1] = a_dims[a_dims.len()-1] + b_dims[b_dims.len()-1];
    
    // Calculate strides
    let last_dim_a = a_dims[a_dims.len()-1];
    let last_dim_b = b_dims[b_dims.len()-1];
    let last_dim_out = last_dim_a + last_dim_b;
    let outer_size: usize = a_dims[..a_dims.len()-1].iter().product();
    
    let mut output = vec![0.0f32; outer_size * last_dim_out];
    
    for i in 0..outer_size {
        // Copy from a
        for j in 0..last_dim_a {
            output[i * last_dim_out + j] = a_data[i * last_dim_a + j];
        }
        // Copy from b
        for j in 0..last_dim_b {
            output[i * last_dim_out + last_dim_a + j] = b_data[i * last_dim_b + j];
        }
    }
    
    Tensor::from_vec(output, Shape::from_dims(&out_dims), a.device().clone())
}

/// Extract Q, K, or V from combined QKV tensor
fn extract_qkv(qkv: &Tensor, idx: usize) -> Result<Tensor> {
    let dims = qkv.shape().dims();
    let batch = dims[0];
    let seq_len = dims[1];
    let num_heads = dims[3];
    let head_dim = dims[4];
    
    let qkv_data = qkv.to_vec()?;
    let mut output_data = vec![0.0f32; batch * num_heads * seq_len * head_dim];
    
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src_idx = b * seq_len * 3 * num_heads * head_dim
                                + s * 3 * num_heads * head_dim
                                + idx * num_heads * head_dim
                                + h * head_dim
                                + d;
                    let dst_idx = b * num_heads * seq_len * head_dim
                                + h * seq_len * head_dim
                                + s * head_dim
                                + d;
                    output_data[dst_idx] = qkv_data[src_idx];
                }
            }
        }
    }
    
    Tensor::from_vec(
        output_data,
        Shape::from_dims(&[batch, num_heads, seq_len, head_dim]),
        qkv.device().clone()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;
    
    #[test]
    fn test_flux_self_attention() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let hidden_size = 768;
        let num_heads = 12;
        let seq_len = 16;
        let batch = 2;
        
        let attn = FluxSelfAttention::new(hidden_size, num_heads, true, device.clone())?;
        
        let x = Tensor::randn(
            Shape::from_dims(&[batch, seq_len, hidden_size]),
            0.0,
            0.02,
            &device
        )?;
        
        let pe = Tensor::randn(
            Shape::from_dims(&[seq_len, hidden_size / num_heads]),
            0.0,
            0.02,
            device
        )?;
        
        let output = attn.forward(&x, &pe)?;
        assert_eq!(output.shape().dims(), &[batch, seq_len, hidden_size]);
        
        Ok(())
    }
    
    #[test]
    fn test_double_stream_block() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = FluxConfig {
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            ..Default::default()
        };
        
        let block = DoubleStreamBlock::new(&config, &device)?;
        
        let batch = 2;
        let img_seq = 16;
        let txt_seq = 8;
        
        let img = Tensor::randn(
            Shape::from_dims(&[batch, img_seq, config.hidden_size]),
            0.0,
            0.02,
            &device
        )?;
        
        let txt = Tensor::randn(
            Shape::from_dims(&[batch, txt_seq, config.hidden_size]),
            0.0,
            0.02,
            &device
        )?;
        
        let vec = Tensor::randn(
            Shape::from_dims(&[batch, config.hidden_size]),
            0.0,
            0.02,
            &device
        )?;
        
        let pe = Tensor::randn(
            Shape::from_dims(&[img_seq + txt_seq, config.head_dim]),
            0.0,
            0.02,
            device
        )?;
        
        let (img_out, txt_out) = block.forward(&img, &txt, &vec, &pe)?;
        
        assert_eq!(img_out.shape().dims(), &[batch, img_seq, config.hidden_size]);
        assert_eq!(txt_out.shape().dims(), &[batch, txt_seq, config.hidden_size]);
        
        Ok(())
    }
}