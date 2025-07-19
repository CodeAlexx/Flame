//! SDXL UNet blocks implementation in FLAME

use crate::{
    Tensor, Shape, Result, FlameError,
    linear::Linear,
    conv::Conv2d,
    norm::GroupNorm,
    sdxl_attention::{SDXLCrossAttention, SDXLAttentionConfig},
    lora::LoRAConfig,
};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// Configuration for ResNet block
#[derive(Debug, Clone)]
pub struct ResNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub temb_channels: Option<usize>,
    pub groups: usize,
    pub dropout: f32,
}

/// ResNet block for SDXL
pub struct ResNetBlock {
    /// First normalization
    pub norm1: GroupNorm,
    /// First convolution
    pub conv1: Conv2d,
    /// Time embedding projection (optional)
    pub time_emb_proj: Option<Linear>,
    /// Second normalization
    pub norm2: GroupNorm,
    /// Second convolution
    pub conv2: Conv2d,
    /// Skip connection (if channels change)
    pub skip_connection: Option<Conv2d>,
    /// Configuration
    config: ResNetConfig,
    device: Arc<CudaDevice>,
}

impl ResNetBlock {
    pub fn new(config: ResNetConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let norm1 = GroupNorm::new_with_affine(
            config.groups,
            config.in_channels,
            1e-6,
            true,
            device.clone(),
        )?;
        
        let conv1 = Conv2d::new_with_bias(
            config.in_channels,
            config.out_channels,
            3,
            1,
            1,
            device.clone(),
            true,
        )?;
        
        let time_emb_proj = config.temb_channels.map(|temb_channels| {
            Linear::new(temb_channels, config.out_channels, true, &device)
        }).transpose()?;
        
        let norm2 = GroupNorm::new_with_affine(
            config.groups,
            config.out_channels,
            1e-6,
            true,
            device.clone(),
        )?;
        
        let conv2 = Conv2d::new_with_bias(
            config.out_channels,
            config.out_channels,
            3,
            1,
            1,
            device.clone(),
            true,
        )?;
        
        let skip_connection = if config.in_channels != config.out_channels {
            Some(Conv2d::new_with_bias(
                config.in_channels,
                config.out_channels,
                1,
                1,
                0,
                device.clone(),
                true,
            )?)
        } else {
            None
        };
        
        Ok(Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            skip_connection,
            config,
            device,
        })
    }
    
    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        // Save input for residual
        let residual = x;
        
        // First block
        let h = self.norm1.forward(x)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;
        
        // Add time embedding if provided
        let h = if let (Some(temb), Some(ref proj)) = (temb, &self.time_emb_proj) {
            let temb = temb.silu()?;
            let temb = proj.forward(&temb)?;
            // Add with broadcasting
            let temb = temb.unsqueeze(2)?.unsqueeze(3)?;
            h.add(&temb)?
        } else {
            h
        };
        
        // Second block
        let h = self.norm2.forward(&h)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;
        
        // Skip connection
        let residual = if let Some(ref skip) = self.skip_connection {
            skip.forward(residual)?
        } else {
            residual.clone()?
        };
        
        // Add residual
        residual.add(&h)
    }
}

/// SDXL Transformer block configuration
#[derive(Debug, Clone)]
pub struct TransformerBlockConfig {
    pub in_channels: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub context_dim: usize,
    pub dropout: f32,
}

/// SDXL Transformer block
pub struct TransformerBlock {
    /// Cross-attention module
    pub attention: SDXLCrossAttention,
    /// Feed-forward network
    pub ff: FeedForward,
    /// Layer norms
    pub norm1: GroupNorm,
    pub norm2: GroupNorm,
    pub norm3: GroupNorm,
    /// Configuration
    config: TransformerBlockConfig,
}

impl TransformerBlock {
    pub fn new(config: TransformerBlockConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let attention = SDXLCrossAttention::new(
            config.in_channels,
            config.context_dim,
            config.num_heads,
            config.head_dim,
            device.clone(),
        )?;
        
        let ff = FeedForward::new(
            config.in_channels,
            config.in_channels * 4,
            config.dropout,
            device.clone(),
        )?;
        
        let norm1 = GroupNorm::new_with_affine(32, config.in_channels, 1e-6, true, device.clone())?;
        let norm2 = GroupNorm::new_with_affine(32, config.in_channels, 1e-6, true, device.clone())?;
        let norm3 = GroupNorm::new_with_affine(32, config.in_channels, 1e-6, true, device.clone())?;
        
        Ok(Self {
            attention,
            ff,
            norm1,
            norm2,
            norm3,
            config,
        })
    }
    
    pub fn add_lora(&mut self, lora_config: LoRAConfig) -> Result<()> {
        self.attention.add_lora(lora_config)?;
        Ok(())
    }
    
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        // Reshape for attention if needed
        let (batch, channels, height, width) = self.get_conv_dims(x)?;
        let seq_len = height * width;
        
        // Reshape to [batch, seq_len, channels]
        let h = x.permute(&[0, 2, 3, 1])?
            .reshape(&[batch, seq_len, channels])?;
        
        // Attention block
        let h = self.norm1.forward(&h)?;
        let h = self.attention.forward(&h, context)?;
        
        // Feed-forward block
        let h = self.norm3.forward(&h)?;
        let h = self.ff.forward(&h)?;
        
        // Reshape back to conv format
        h.reshape(&[batch, height, width, channels])?
            .permute(&[0, 3, 1, 2])
    }
    
    fn get_conv_dims(&self, x: &Tensor) -> Result<(usize, usize, usize, usize)> {
        let dims = x.shape().dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("Expected 4D tensor, got {}D", dims.len())
            ));
        }
        Ok((dims[0], dims[1], dims[2], dims[3]))
    }
}

/// Feed-forward network
pub struct FeedForward {
    pub net: Vec<Box<dyn Module>>,
}

impl FeedForward {
    pub fn new(
        in_channels: usize,
        hidden_channels: usize,
        dropout: f32,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let net: Vec<Box<dyn Module>> = vec![
            Box::new(Linear::new(in_channels, hidden_channels, true, &device)?),
            Box::new(GELU),
            Box::new(Linear::new(hidden_channels, in_channels, true, &device)?),
        ];
        
        Ok(Self { net })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = Clone::clone(x);
        for module in &self.net {
            let h_new = module.forward(&h)?;
            h = h_new;
        }
        x.add(&h)
    }
}

/// Module trait for composability
pub trait Module: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Linear::forward(self, x)
    }
}

/// GELU activation
pub struct GELU;

impl Module for GELU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu()
    }
}

/// SDXL Down block
pub struct SDXLDownBlock {
    pub resnets: Vec<ResNetBlock>,
    pub attentions: Vec<Option<TransformerBlock>>,
    pub downsamplers: Vec<Option<Conv2d>>,
}

impl SDXLDownBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        add_attention: bool,
        attention_config: Option<TransformerBlockConfig>,
        add_downsample: bool,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();
        
        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            
            // ResNet block
            let config = ResNetConfig {
                in_channels: in_ch,
                out_channels,
                temb_channels: Some(temb_channels),
                groups: 32,
                dropout: 0.0,
            };
            resnets.push(ResNetBlock::new(config, device.clone())?);
            
            // Attention block (optional)
            if add_attention {
                if let Some(ref attn_config) = attention_config {
                    let mut config = attn_config.clone();
                    config.in_channels = out_channels;
                    attentions.push(Some(TransformerBlock::new(config, device.clone())?));
                } else {
                    attentions.push(None);
                }
            } else {
                attentions.push(None);
            }
        }
        
        // Downsampler (optional)
        let downsamplers = if add_downsample {
            vec![Some(Conv2d::new_with_bias(
                out_channels,
                out_channels,
                3,
                2,
                1,
                device,
                true,
            )?)]
        } else {
            vec![]
        };
        
        Ok(Self {
            resnets,
            attentions,
            downsamplers,
        })
    }
    
    pub fn forward(
        &self,
        x: &Tensor,
        temb: &Tensor,
        context: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut h = Clone::clone(x);
        let mut outputs = Vec::new();
        
        for (resnet, attention) in self.resnets.iter().zip(&self.attentions) {
            let h_new = resnet.forward(&h, Some(temb))?;
            h = h_new;
            
            if let (Some(attn), Some(ctx)) = (attention, context) {
                let h_attn = attn.forward(&h, ctx)?;
                h = h_attn;
            }
            
            outputs.push(Clone::clone(&h));
        }
        
        if let Some(Some(ref downsampler)) = self.downsamplers.get(0) {
            let h_down = downsampler.forward(&h)?;
            h = h_down;
            outputs.push(Clone::clone(&h));
        }
        
        Ok((h, outputs))
    }
    
    pub fn add_lora(&mut self, lora_config: LoRAConfig) -> Result<()> {
        for attention in &mut self.attentions {
            if let Some(ref mut attn) = attention {
                attn.add_lora(lora_config.clone())?;
            }
        }
        Ok(())
    }
}