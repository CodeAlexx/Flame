//! MMDiT (Multimodal Diffusion Transformer) blocks for SD3.5
//! 
//! This module provides the core blocks for the SD3.5 diffusion model,
//! which uses a multimodal architecture processing image and text jointly.

use crate::{
    Tensor, Shape, Result,
    attention::{MultiHeadAttention, AttentionConfig},
    linear::Linear,
    modulated_blocks::{AdaLayerNorm, Modulation, QKNorm},
    norm::LayerNorm,
};
use std::sync::Arc;

/// Configuration for MMDiT model
#[derive(Clone)]
pub struct MMDiTConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub mlp_ratio: f32,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    pub pos_embed_max_size: usize,
}

impl Default for MMDiTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1536,  // SD3.5 Large
            num_heads: 24,
            depth: 38,
            mlp_ratio: 4.0,
            qkv_bias: false,
            qk_norm: true,
            pos_embed_max_size: 192,  // Max 192x192 patches
        }
    }
}

/// Joint attention module for MMDiT
/// Processes concatenated image and text sequences
pub struct JointAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub qkv: Linear,
    pub q_norm: Option<QKNorm>,
    pub k_norm: Option<QKNorm>,
    pub proj: Linear,
}

impl JointAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        qk_norm: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        
        let qkv = Linear::new(dim, dim * 3, qkv_bias, &device)?;
        let proj = Linear::new(dim, dim, true, &device)?;
        
        let (q_norm, k_norm) = if qk_norm {
            (
                Some(QKNorm::new(head_dim, device.clone())?),
                Some(QKNorm::new(head_dim, device.clone())?),
            )
        } else {
            (None, None)
        };
        
        Ok(Self {
            num_heads,
            head_dim,
            qkv,
            q_norm,
            k_norm,
            proj,
        })
    }
    
    pub fn forward(&self, x: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let batch = x.shape().dims()[0];
        let seq_len = x.shape().dims()[1];
        let dim = x.shape().dims()[2];
        
        // Generate QKV
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape(&[batch, seq_len, 3, self.num_heads, self.head_dim])?;
        
        // Split into Q, K, V
        let q = extract_qkv(&qkv, 0)?;
        let k = extract_qkv(&qkv, 1)?;
        let v = extract_qkv(&qkv, 2)?;
        
        // Apply QK normalization if enabled
        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            let (q_normed, k_normed) = q_norm.forward(&q, &k)?;
            (q_normed, k_normed)
        } else {
            (q, k)
        };
        
        // Add position embeddings
        let q = add_pos_embed(&q, pe)?;
        let k = add_pos_embed(&k, pe)?;
        
        // Attention
        let attn = scaled_dot_product_attention(&q, &k, &v)?;
        
        // Reshape back
        let attn = attn.transpose_dims(1, 2)?.flatten_from(2)?;
        
        // Output projection
        self.proj.forward(&attn)
    }
}

/// MMDiT block
pub struct MMDiTBlock {
    pub norm1: LayerNorm,
    pub attn: JointAttention,
    pub norm2: LayerNorm,
    pub mlp: MLP,
    pub adaLN_modulation: AdaLayerNorm,
}

impl MMDiTBlock {
    pub fn new(
        config: &MMDiTConfig,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        
        let norm1 = LayerNorm::new(hidden_size, 1e-6, device.clone())?;
        let norm2 = LayerNorm::new(hidden_size, 1e-6, device.clone())?;
        
        let attn = JointAttention::new(
            hidden_size,
            config.num_heads,
            config.qkv_bias,
            config.qk_norm,
            device.clone(),
        )?;
        
        let mlp = MLP::new(
            hidden_size,
            (hidden_size as f32 * config.mlp_ratio) as usize,
            device.clone(),
        )?;
        
        let adaLN_modulation = AdaLayerNorm::new(
            hidden_size,
            hidden_size,
            device,
        )?;
        
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            adaLN_modulation,
        })
    }
    
    pub fn forward(
        &self,
        x: &Tensor,
        pe: &Tensor,
        c: &Tensor,  // conditioning (timestep)
    ) -> Result<Tensor> {
        // Apply adaptive layer norm with conditioning
        let x_norm = self.adaLN_modulation.forward(&self.norm1.forward(x)?, c)?;
        
        // Self-attention
        let attn_out = self.attn.forward(&x_norm, pe)?;
        let x = x.add(&attn_out)?;
        
        // MLP
        let x_norm = self.adaLN_modulation.forward(&self.norm2.forward(&x)?, c)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        
        x.add(&mlp_out)
    }
}

/// Simple MLP for MMDiT
pub struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MLP {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let fc1 = Linear::new(in_features, hidden_features, true, &device)?;
        let fc2 = Linear::new(hidden_features, in_features, true, &device)?;
        
        Ok(Self { fc1, fc2 })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        self.fc2.forward(&x)
    }
}

/// 2D Sinusoidal position embedding
pub fn get_2d_sincos_pos_embed(
    embed_dim: usize,
    grid_size: (usize, usize),
    device: Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let (h, w) = grid_size;
    let total_patches = h * w;
    
    // Create grid
    let mut grid_h = vec![0.0f32; h * w];
    let mut grid_w = vec![0.0f32; h * w];
    
    for i in 0..h {
        for j in 0..w {
            grid_h[i * w + j] = i as f32;
            grid_w[i * w + j] = j as f32;
        }
    }
    
    let half_dim = embed_dim / 2;
    let omega: Vec<f32> = (0..half_dim / 2)
        .map(|i| 1.0 / 10000_f32.powf(2.0 * i as f32 / half_dim as f32))
        .collect();
    
    let mut pos_embed = vec![0.0f32; total_patches * embed_dim];
    
    // Height embeddings (first half of dimensions)
    for i in 0..total_patches {
        for j in 0..half_dim / 2 {
            let angle = grid_h[i] * omega[j];
            pos_embed[i * embed_dim + j] = angle.sin();
            pos_embed[i * embed_dim + j + half_dim / 2] = angle.cos();
        }
    }
    
    // Width embeddings (second half of dimensions)
    for i in 0..total_patches {
        for j in 0..half_dim / 2 {
            let angle = grid_w[i] * omega[j];
            pos_embed[i * embed_dim + half_dim + j] = angle.sin();
            pos_embed[i * embed_dim + half_dim + j + half_dim / 2] = angle.cos();
        }
    }
    
    Tensor::from_vec(
        pos_embed,
        Shape::from_dims(&[total_patches, embed_dim]),
        device,
    )
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

/// Add position embeddings to queries/keys
fn add_pos_embed(x: &Tensor, pe: &Tensor) -> Result<Tensor> {
    // x: [batch, num_heads, seq_len, head_dim]
    // pe: [seq_len, head_dim]
    let dims = x.shape().dims();
    let batch = dims[0];
    let num_heads = dims[1];
    let seq_len = dims[2];
    let head_dim = dims[3];
    
    let x_data = x.to_vec()?;
    let pe_data = pe.to_vec()?;
    let mut output = vec![0.0f32; x_data.len()];
    
    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let idx = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + s * head_dim
                            + d;
                    let pe_idx = s * head_dim + d;
                    output[idx] = x_data[idx] + pe_data[pe_idx];
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

/// Final layer for MMDiT
pub struct FinalLayer {
    pub norm_final: LayerNorm,
    pub linear: Linear,
    pub adaLN_modulation: AdaLayerNorm,
}

impl FinalLayer {
    pub fn new(
        hidden_size: usize,
        patch_size: usize,
        out_channels: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let norm_final = LayerNorm::new(hidden_size, 1e-6, device.clone())?;
        let linear = Linear::new(
            hidden_size,
            patch_size * patch_size * out_channels,
            true,
            &device,
        )?;
        let adaLN_modulation = AdaLayerNorm::new(hidden_size, hidden_size, device)?;
        
        Ok(Self {
            norm_final,
            linear,
            adaLN_modulation,
        })
    }
    
    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let x = self.adaLN_modulation.forward(&self.norm_final.forward(x)?, c)?;
        self.linear.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;
    
    #[test]
    fn test_joint_attention() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let hidden_size = 768;
        let num_heads = 12;
        let seq_len = 16;
        let batch = 2;
        
        let attn = JointAttention::new(hidden_size, num_heads, true, true, device.clone())?;
        
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
    fn test_mmdit_block() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = MMDiTConfig {
            hidden_size: 768,
            num_heads: 12,
            depth: 4,
            mlp_ratio: 4.0,
            qkv_bias: false,
            qk_norm: true,
            pos_embed_max_size: 32,
        };
        
        let block = MMDiTBlock::new(&config, &device)?;
        
        let batch = 2;
        let seq_len = 16;
        
        let x = Tensor::randn(
            Shape::from_dims(&[batch, seq_len, config.hidden_size]),
            0.0,
            0.02,
            &device
        )?;
        
        let pe = Tensor::randn(
            Shape::from_dims(&[seq_len, config.hidden_size / config.num_heads]),
            0.0,
            0.02,
            &device
        )?;
        
        let c = Tensor::randn(
            Shape::from_dims(&[batch, config.hidden_size]),
            0.0,
            0.02,
            device
        )?;
        
        let output = block.forward(&x, &pe, &c)?;
        assert_eq!(output.shape().dims(), &[batch, seq_len, config.hidden_size]);
        
        Ok(())
    }
    
    #[test]
    fn test_2d_sincos_pos_embed() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let embed_dim = 768;
        let grid_size = (8, 8);
        
        let pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, device)?;
        
        assert_eq!(pos_embed.shape().dims(), &[64, embed_dim]);
        
        // Check that embeddings have reasonable values
        let data = pos_embed.to_vec()?;
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
        
        println!("Pos embed mean: {:.6}, std: {:.6}", mean, std);
        // Note: Mean won't be zero due to grid coordinates being positive
        assert!(std > 0.3 && std < 0.8);  // Should have reasonable variance
        
        Ok(())
    }
}