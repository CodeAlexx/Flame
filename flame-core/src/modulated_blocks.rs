//! Modulated transformer blocks for diffusion models
//! 
//! This module provides modulated blocks used in models like Flux and SD3.5,
//! where the attention and FFN are modulated by timestep embeddings.

use crate::{Tensor, Shape, Result, attention::{MultiHeadAttention, AttentionConfig, LayerNorm}};
use crate::linear::Linear;
use std::sync::Arc;

/// Adaptive Layer Normalization with modulation
/// Used in DiT, SD3.5, and Flux models
pub struct AdaLayerNorm {
    pub norm: LayerNorm,
    pub linear: Linear,
    pub num_features: usize,
}

impl AdaLayerNorm {
    pub fn new(
        num_features: usize,
        cond_features: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let norm = LayerNorm::new(num_features, 1e-6, false, device.clone())?;
        // Project conditioning to scale and shift parameters
        let linear = Linear::new(cond_features, num_features * 2, true, &device)?;
        
        Ok(Self {
            norm,
            linear,
            num_features,
        })
    }
    
    /// Forward pass with conditioning
    /// x: input tensor [batch, seq_len, features]
    /// cond: conditioning tensor [batch, cond_features]
    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        // Get scale and shift from conditioning
        let scale_shift = self.linear.forward(cond)?;
        let dims = scale_shift.shape().dims();
        
        // Split into scale and shift
        let data = scale_shift.to_vec()?;
        let half = self.num_features;
        let batch_size = dims[0];
        
        let mut scale_data = vec![0.0f32; batch_size * half];
        let mut shift_data = vec![0.0f32; batch_size * half];
        
        for b in 0..batch_size {
            for i in 0..half {
                let idx = b * 2 * half + i;
                scale_data[b * half + i] = 1.0 + data[idx]; // 1 + scale for residual
                shift_data[b * half + i] = data[idx + half];
            }
        }
        
        // Apply normalization
        let normalized = self.norm.forward(x)?;
        
        // Apply scale and shift
        let x_dims = x.shape().dims();
        let seq_len = x_dims[1];
        let norm_data = normalized.to_vec()?;
        let mut output_data = vec![0.0f32; norm_data.len()];
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for f in 0..self.num_features {
                    let idx = b * seq_len * self.num_features + s * self.num_features + f;
                    let scale_idx = b * self.num_features + f;
                    output_data[idx] = norm_data[idx] * scale_data[scale_idx] + shift_data[scale_idx];
                }
            }
        }
        
        Tensor::from_vec(output_data, normalized.shape().clone(), normalized.device().clone())
    }
}

/// Modulation for transformer blocks
/// Projects conditioning to multiple modulation parameters
pub struct Modulation {
    pub linear: Linear,
    pub num_outputs: usize,
}

impl Modulation {
    pub fn new(
        cond_features: usize,
        hidden_features: usize,
        num_outputs: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let linear = Linear::new(cond_features, hidden_features * num_outputs, true, &device)?;
        
        Ok(Self {
            linear,
            num_outputs,
        })
    }
    
    /// Forward pass returning multiple modulation tensors
    pub fn forward(&self, cond: &Tensor) -> Result<Vec<Tensor>> {
        let modulations = self.linear.forward(cond)?;
        let dims = modulations.shape().dims();
        let batch_size = dims[0];
        let hidden_features = dims[1] / self.num_outputs;
        
        let data = modulations.to_vec()?;
        let mut outputs = Vec::new();
        
        for i in 0..self.num_outputs {
            let mut output_data = vec![0.0f32; batch_size * hidden_features];
            for b in 0..batch_size {
                for h in 0..hidden_features {
                    let src_idx = b * dims[1] + i * hidden_features + h;
                    let dst_idx = b * hidden_features + h;
                    output_data[dst_idx] = data[src_idx];
                }
            }
            
            outputs.push(Tensor::from_vec(
                output_data,
                Shape::from_dims(&[batch_size, hidden_features]),
                modulations.device().clone()
            )?);
        }
        
        Ok(outputs)
    }
}

/// Modulated Transformer Block (for DiT/MMDiT architectures)
pub struct ModulatedTransformerBlock {
    pub norm1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub norm2: LayerNorm,
    pub mlp: FeedForwardMLP,
    pub adaLN_modulation: Modulation,
}

impl ModulatedTransformerBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let norm1 = LayerNorm::new(hidden_size, 1e-6, false, device.clone())?;
        let norm2 = LayerNorm::new(hidden_size, 1e-6, false, device.clone())?;
        
        let attn_config = AttentionConfig::new(hidden_size, num_heads);
        let attn = MultiHeadAttention::new(attn_config, device.clone())?;
        
        let mlp_hidden = (hidden_size as f32 * mlp_ratio) as usize;
        let mlp = FeedForwardMLP::new(hidden_size, mlp_hidden, device.clone())?;
        
        // Modulation produces 6 outputs: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        let adaLN_modulation = Modulation::new(hidden_size, hidden_size, 6, device)?;
        
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            adaLN_modulation,
        })
    }
    
    /// Forward pass with timestep conditioning
    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        // Get modulation parameters
        let mods = self.adaLN_modulation.forward(c)?;
        let shift_msa = &mods[0];
        let scale_msa = &mods[1];
        let gate_msa = &mods[2];
        let shift_mlp = &mods[3];
        let scale_mlp = &mods[4];
        let gate_mlp = &mods[5];
        
        // Attention block with modulation
        let x_norm = self.norm1.forward(x)?;
        let x_mod = self.apply_modulation(&x_norm, scale_msa, shift_msa)?;
        let attn_out = self.attn.forward(&x_mod, &x_mod, &x_mod, None)?;
        let attn_out = self.apply_gate(&attn_out, gate_msa)?;
        let x = x.add(&attn_out)?;
        
        // MLP block with modulation
        let x_norm = self.norm2.forward(&x)?;
        let x_mod = self.apply_modulation(&x_norm, scale_mlp, shift_mlp)?;
        let mlp_out = self.mlp.forward(&x_mod)?;
        let mlp_out = self.apply_gate(&mlp_out, gate_mlp)?;
        
        x.add(&mlp_out)
    }
    
    /// Apply scale and shift modulation
    fn apply_modulation(&self, x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        let x_dims = x.shape().dims();
        let batch_size = x_dims[0];
        let seq_len = x_dims[1];
        let hidden_size = x_dims[2];
        
        let x_data = x.to_vec()?;
        let scale_data = scale.to_vec()?;
        let shift_data = shift.to_vec()?;
        let mut output = vec![0.0f32; x_data.len()];
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..hidden_size {
                    let idx = b * seq_len * hidden_size + s * hidden_size + h;
                    let mod_idx = b * hidden_size + h;
                    output[idx] = x_data[idx] * (1.0 + scale_data[mod_idx]) + shift_data[mod_idx];
                }
            }
        }
        
        Tensor::from_vec(output, x.shape().clone(), x.device().clone())
    }
    
    /// Apply gating
    fn apply_gate(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let x_dims = x.shape().dims();
        let batch_size = x_dims[0];
        let seq_len = x_dims[1];
        let hidden_size = x_dims[2];
        
        let x_data = x.to_vec()?;
        let gate_data = gate.to_vec()?;
        let mut output = vec![0.0f32; x_data.len()];
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..hidden_size {
                    let idx = b * seq_len * hidden_size + s * hidden_size + h;
                    let gate_idx = b * hidden_size + h;
                    output[idx] = x_data[idx] * gate_data[gate_idx];
                }
            }
        }
        
        Tensor::from_vec(output, x.shape().clone(), x.device().clone())
    }
}

/// Simple MLP for transformer blocks
pub struct FeedForwardMLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl FeedForwardMLP {
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

/// Timestep embedding with sinusoidal encoding
pub struct TimestepEmbedding {
    pub linear1: Linear,
    pub linear2: Linear,
    pub freq_embed_size: usize,
}

impl TimestepEmbedding {
    pub fn new(
        channels: usize,
        freq_embed_size: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let time_embed_dim = channels * 4;
        let linear1 = Linear::new(freq_embed_size, time_embed_dim, true, &device)?;
        let linear2 = Linear::new(time_embed_dim, channels, true, &device)?;
        
        Ok(Self {
            linear1,
            linear2,
            freq_embed_size,
        })
    }
    
    /// Create sinusoidal position embeddings
    pub fn sinusoidal_embedding(&self, timesteps: &Tensor) -> Result<Tensor> {
        let device = timesteps.device();
        let half_dim = self.freq_embed_size / 2;
        let exponent: Vec<f32> = (0..half_dim)
            .map(|i| -(i as f32 * std::f32::consts::LN_2 / half_dim as f32))
            .collect();
        
        let t_data = timesteps.to_vec()?;
        let batch_size = t_data.len();
        let mut embedding = vec![0.0f32; batch_size * self.freq_embed_size];
        
        for b in 0..batch_size {
            let t = t_data[b];
            for i in 0..half_dim {
                let freq = (exponent[i]).exp();
                let angle = t * freq;
                embedding[b * self.freq_embed_size + i] = angle.cos();
                embedding[b * self.freq_embed_size + half_dim + i] = angle.sin();
            }
        }
        
        Tensor::from_vec(
            embedding,
            Shape::from_dims(&[batch_size, self.freq_embed_size]),
            device.clone()
        )
    }
    
    /// Forward pass: timestep -> embedding
    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let t_emb = self.sinusoidal_embedding(timesteps)?;
        let t_emb = self.linear1.forward(&t_emb)?;
        let t_emb = t_emb.silu()?;
        self.linear2.forward(&t_emb)
    }
}

/// QK-Normalization for attention (used in Flux)
pub struct QKNorm {
    pub query_norm: RMSNorm,
    pub key_norm: RMSNorm,
}

impl QKNorm {
    pub fn new(dim: usize, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let query_norm = RMSNorm::new(dim, 1e-6, device.clone())?;
        let key_norm = RMSNorm::new(dim, 1e-6, device)?;
        
        Ok(Self {
            query_norm,
            key_norm,
        })
    }
    
    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_normed = self.query_norm.forward(q)?;
        let k_normed = self.key_norm.forward(k)?;
        Ok((q_normed, k_normed))
    }
}

/// RMSNorm for QK normalization
pub struct RMSNorm {
    pub scale: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let scale = Tensor::from_vec(
            vec![1.0f32; dim],
            Shape::from_dims(&[dim]),
            device.clone()
        )?;
        
        Ok(Self { scale, eps })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims();
        let last_dim = dims[dims.len() - 1];
        let outer_dims: usize = dims[..dims.len()-1].iter().product();
        
        let x_data = x.to_vec()?;
        let scale_data = self.scale.to_vec()?;
        let mut output = vec![0.0f32; x_data.len()];
        
        for i in 0..outer_dims {
            let offset = i * last_dim;
            
            // Calculate RMS
            let mut sum_sq = 0.0f32;
            for j in 0..last_dim {
                sum_sq += x_data[offset + j] * x_data[offset + j];
            }
            let rms = (sum_sq / last_dim as f32 + self.eps).sqrt();
            
            // Normalize and scale
            for j in 0..last_dim {
                output[offset + j] = (x_data[offset + j] / rms) * scale_data[j];
            }
        }
        
        Tensor::from_vec(output, x.shape().clone(), x.device().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;
    
    #[test]
    fn test_timestep_embedding() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        let embed = TimestepEmbedding::new(512, 256, device.clone())?;
        let timesteps = Tensor::from_vec(
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            Shape::from_dims(&[5]),
            device
        )?;
        
        let output = embed.forward(&timesteps)?;
        assert_eq!(output.shape().dims(), &[5, 512]);
        
        Ok(())
    }
    
    #[test]
    fn test_modulated_block() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        let block = ModulatedTransformerBlock::new(512, 8, 4.0, device.clone())?;
        
        let x = Tensor::randn(
            Shape::from_dims(&[2, 10, 512]),
            0.0,
            0.02,
            &device
        )?;
        
        let c = Tensor::randn(
            Shape::from_dims(&[2, 512]),
            0.0,
            0.02,
            device
        )?;
        
        let output = block.forward(&x, &c)?;
        assert_eq!(output.shape().dims(), &[2, 10, 512]);
        
        Ok(())
    }
}