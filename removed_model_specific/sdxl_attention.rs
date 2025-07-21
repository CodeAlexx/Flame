//! SDXL Attention implementation in FLAME

use crate::{
    Tensor, Shape, Result, FlameError, 
    linear::Linear,
    norm::GroupNorm,
    lora::{LoRALayer, LoRAConfig},
};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// Attention configuration for SDXL
#[derive(Debug, Clone)]
pub struct SDXLAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each head
    pub head_dim: usize,
    /// Hidden dimension (num_heads * head_dim)
    pub hidden_dim: usize,
    /// Context dimension for cross-attention
    pub context_dim: Option<usize>,
    /// Use Flash Attention if available
    pub use_flash_attn: bool,
}

/// SDXL Attention module with optional LoRA
pub struct SDXLAttention {
    /// Query projection
    pub to_q: Linear,
    /// Key projection
    pub to_k: Linear,
    /// Value projection
    pub to_v: Linear,
    /// Output projection
    pub to_out: Linear,
    /// Group normalization
    pub norm: Option<GroupNorm>,
    /// Configuration
    pub config: SDXLAttentionConfig,
    /// Optional LoRA adapters
    pub lora_q: Option<LoRALayer>,
    pub lora_k: Option<LoRALayer>,
    pub lora_v: Option<LoRALayer>,
    pub lora_out: Option<LoRALayer>,
    /// Device
    device: Arc<CudaDevice>,
}

impl SDXLAttention {
    /// Create new SDXL attention module
    pub fn new(
        in_channels: usize,
        config: SDXLAttentionConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let hidden_dim = config.hidden_dim;
        let context_dim = config.context_dim.unwrap_or(in_channels);
        
        // Create projections
        let to_q = Linear::new(in_channels, hidden_dim, true, &device)?;
        let to_k = Linear::new(context_dim, hidden_dim, true, &device)?;
        let to_v = Linear::new(context_dim, hidden_dim, true, &device)?;
        let to_out = Linear::new(hidden_dim, in_channels, true, &device)?;
        
        // Group norm (optional)
        let norm = if in_channels > 32 {
            Some(GroupNorm::new_with_affine(32, in_channels, 1e-6, true, device.clone())?)
        } else {
            None
        };
        
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            norm,
            config,
            lora_q: None,
            lora_k: None,
            lora_v: None,
            lora_out: None,
            device,
        })
    }
    
    /// Add LoRA adapters
    pub fn add_lora(&mut self, lora_config: LoRAConfig) -> Result<()> {
        let in_channels = self.to_q.in_features();
        let hidden_dim = self.config.hidden_dim;
        let context_dim = self.config.context_dim.unwrap_or(in_channels);
        
        // Add LoRA to each projection
        self.lora_q = Some(LoRALayer::new(
            in_channels,
            hidden_dim,
            lora_config.clone(),
            self.device.clone(),
        )?);
        
        self.lora_k = Some(LoRALayer::new(
            context_dim,
            hidden_dim,
            lora_config.clone(),
            self.device.clone(),
        )?);
        
        self.lora_v = Some(LoRALayer::new(
            context_dim,
            hidden_dim,
            lora_config.clone(),
            self.device.clone(),
        )?);
        
        self.lora_out = Some(LoRALayer::new(
            hidden_dim,
            in_channels,
            lora_config,
            self.device.clone(),
        )?);
        
        Ok(())
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = self.get_dims(x)?;
        
        // Apply group norm if present
        let h = if let Some(ref norm) = self.norm {
            norm.forward(x)?
        } else {
            x.clone()?
        };
        
        // Compute Q
        let q = if let Some(ref lora) = self.lora_q {
            lora.forward(&h, &self.to_q.weight)?
        } else {
            self.to_q.forward(&h)?
        };
        
        // Compute K and V (use context if cross-attention)
        let kv_input = context.unwrap_or(&h);
        
        let k = if let Some(ref lora) = self.lora_k {
            lora.forward(kv_input, &self.to_k.weight)?
        } else {
            self.to_k.forward(kv_input)?
        };
        
        let v = if let Some(ref lora) = self.lora_v {
            lora.forward(kv_input, &self.to_v.weight)?
        } else {
            self.to_v.forward(kv_input)?
        };
        
        // Reshape for multi-head attention
        let q = self.reshape_for_attention(&q, batch_size, seq_len)?;
        let k = self.reshape_for_attention(&k, batch_size, kv_input.shape().dims()[1])?;
        let v = self.reshape_for_attention(&v, batch_size, kv_input.shape().dims()[1])?;
        
        // Compute attention
        let attn_output = if self.config.use_flash_attn {
            self.flash_attention(&q, &k, &v)?
        } else {
            self.scaled_dot_product_attention(&q, &k, &v)?
        };
        
        // Reshape back
        let attn_output = self.reshape_from_attention(&attn_output, batch_size, seq_len)?;
        
        // Output projection
        let output = if let Some(ref lora) = self.lora_out {
            lora.forward(&attn_output, &self.to_out.weight)?
        } else {
            self.to_out.forward(&attn_output)?
        };
        
        // Add residual
        x.add(&output)
    }
    
    /// Get dimensions from input tensor
    fn get_dims(&self, x: &Tensor) -> Result<(usize, usize, usize)> {
        let dims = x.shape().dims();
        if dims.len() != 3 {
            return Err(FlameError::InvalidOperation(
                format!("Expected 3D tensor, got {}D", dims.len())
            ));
        }
        Ok((dims[0], dims[1], dims[2]))
    }
    
    /// Reshape tensor for multi-head attention
    fn reshape_for_attention(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        
        // [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]
        x.reshape(&[batch_size, seq_len, num_heads, head_dim])?
            .permute(&[0, 2, 1, 3])
    }
    
    /// Reshape tensor back from multi-head attention
    fn reshape_from_attention(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let hidden_dim = self.config.hidden_dim;
        
        // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        x.permute(&[0, 2, 1, 3])?
            .reshape(&[batch_size, seq_len, hidden_dim])
    }
    
    /// Scaled dot-product attention
    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let head_dim = self.config.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // Compute attention scores: Q @ K^T / sqrt(d)
        let k_dims = k.shape().dims();
        let k_rank = k_dims.len();
        let scores = q.matmul(&k.transpose_dims(k_rank - 2, k_rank - 1)?)?
            .mul_scalar(scale)?;
        
        // Apply softmax
        let attn_weights = scores.softmax(-1isize)?;
        
        // Apply attention to values: attn @ V
        attn_weights.matmul(v)
    }
    
    /// Flash attention (placeholder - would use optimized kernel)
    fn flash_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        // For now, fall back to standard attention
        // In production, this would call an optimized Flash Attention kernel
        self.scaled_dot_product_attention(q, k, v)
    }
    
    /// Get LoRA parameters if any
    pub fn lora_parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        
        if let Some(ref lora) = self.lora_q {
            params.extend(lora.parameters());
        }
        if let Some(ref lora) = self.lora_k {
            params.extend(lora.parameters());
        }
        if let Some(ref lora) = self.lora_v {
            params.extend(lora.parameters());
        }
        if let Some(ref lora) = self.lora_out {
            params.extend(lora.parameters());
        }
        
        params
    }
}

/// SDXL Cross-Attention block
pub struct SDXLCrossAttention {
    /// Self-attention
    pub self_attn: SDXLAttention,
    /// Cross-attention
    pub cross_attn: SDXLAttention,
    /// Layer norms
    pub norm1: GroupNorm,
    pub norm2: GroupNorm,
}

impl SDXLCrossAttention {
    /// Create new cross-attention block
    pub fn new(
        in_channels: usize,
        context_dim: usize,
        num_heads: usize,
        head_dim: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let hidden_dim = num_heads * head_dim;
        
        // Self-attention config
        let self_config = SDXLAttentionConfig {
            num_heads,
            head_dim,
            hidden_dim,
            context_dim: None,
            use_flash_attn: false,
        };
        
        // Cross-attention config
        let cross_config = SDXLAttentionConfig {
            num_heads,
            head_dim,
            hidden_dim,
            context_dim: Some(context_dim),
            use_flash_attn: false,
        };
        
        Ok(Self {
            self_attn: SDXLAttention::new(in_channels, self_config, device.clone())?,
            cross_attn: SDXLAttention::new(in_channels, cross_config, device.clone())?,
            norm1: GroupNorm::new_with_affine(32, in_channels, 1e-6, true, device.clone())?,
            norm2: GroupNorm::new_with_affine(32, in_channels, 1e-6, true, device)?,
        })
    }
    
    /// Add LoRA to both attention layers
    pub fn add_lora(&mut self, lora_config: LoRAConfig) -> Result<()> {
        self.self_attn.add_lora(lora_config.clone())?;
        self.cross_attn.add_lora(lora_config)?;
        Ok(())
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        // Self-attention
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward(&h, None)?;
        
        // Cross-attention
        let h = self.norm2.forward(&h)?;
        self.cross_attn.forward(&h, Some(context))
    }
}