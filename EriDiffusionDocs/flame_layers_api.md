# FLAME Layers API Documentation

This document describes FLAME's neural network layer implementations and how EriDiffusion should use them.

## Core Layer Types

```rust
use flame_core::{
    Linear, Conv2d, Conv2dConfig,
    LayerNorm, GroupNorm, RMSNorm,
    Dropout, 
    BatchNorm2d,
};
```

## Linear Layer

### Basic Usage

```rust
use flame_core::Linear;

// Create linear layer
let linear = Linear::new(
    in_features: 768,
    out_features: 1024,
    bias: true,
    &device
)?;

// Without bias
let linear_no_bias = Linear::new(768, 1024, false, &device)?;

// Forward pass
let output = linear.forward(&input)?;
// Input shape: [..., 768]
// Output shape: [..., 1024]

// Access weights and bias
let weight = &linear.weight;  // Shape: [1024, 768]
let bias = &linear.bias;      // Option<Tensor>, Shape: [1024]

// Get dimensions
let in_dim = linear.in_features();
let out_dim = linear.out_features();
```

### Linear Layer for LoRA

```rust
// LoRA decomposition example
pub struct LoRALinear {
    pub base: Linear,
    pub lora_a: Linear,  // down projection
    pub lora_b: Linear,  // up projection
    pub scale: f32,
}

impl LoRALinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        device: &Arc<CudaDevice>
    ) -> Result<Self> {
        let base = Linear::new(in_features, out_features, true, device)?;
        let lora_a = Linear::new(in_features, rank, false, device)?;
        let lora_b = Linear::new(rank, out_features, false, device)?;
        
        // Initialize LoRA weights
        // A: normal initialization
        // B: zero initialization
        lora_b.weight.zero_()?;
        
        Ok(Self {
            base,
            lora_a,
            lora_b,
            scale: 1.0,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base forward
        let base_out = self.base.forward(x)?;
        
        // LoRA forward: x @ A @ B * scale
        let lora_out = self.lora_a.forward(x)?;
        let lora_out = self.lora_b.forward(&lora_out)?;
        let lora_out = lora_out.mul_scalar(self.scale)?;
        
        // Combine
        base_out.add(&lora_out)
    }
}
```

## Convolution Layers

### Conv2d

```rust
use flame_core::{Conv2d, Conv2dConfig};

// Simple Conv2d
let conv = Conv2d::new(
    in_channels: 3,
    out_channels: 64,
    kernel_size: 3,
    stride: 1,
    padding: 1,
    device.clone()
)?;

// With bias control
let conv = Conv2d::new_with_bias(3, 64, 3, 1, 1, device.clone(), true)?;

// Advanced configuration
let config = Conv2dConfig {
    in_channels: 3,
    out_channels: 64,
    kernel_size: (3, 3),
    stride: (1, 1),
    padding: (1, 1),
    groups: 1,  // For grouped/depthwise convolution
};
let conv = Conv2d::from_config(config, device.clone())?;

// Forward pass
let output = conv.forward(&input)?;
// Input shape: [batch, 3, height, width]
// Output shape: [batch, 64, height, width]
```

### Depthwise and Grouped Convolutions

```rust
// Depthwise convolution (groups = in_channels)
let depthwise_config = Conv2dConfig {
    in_channels: 64,
    out_channels: 64,
    kernel_size: (3, 3),
    stride: (1, 1),
    padding: (1, 1),
    groups: 64,  // Depthwise
};
let depthwise = Conv2d::from_config(depthwise_config, device.clone())?;

// Grouped convolution
let grouped_config = Conv2dConfig {
    in_channels: 64,
    out_channels: 128,
    kernel_size: (3, 3),
    stride: (1, 1),
    padding: (1, 1),
    groups: 4,  // 4 groups
};
let grouped = Conv2d::from_config(grouped_config, device.clone())?;
```

## Normalization Layers

### LayerNorm

```rust
use flame_core::LayerNorm;

// Create LayerNorm
let ln = LayerNorm::new(
    normalized_shape: vec![768],  // Last dimension(s) to normalize
    eps: 1e-5,
    device.clone()
)?;

// For transformer models
let ln = LayerNorm::new(vec![hidden_dim], 1e-5, device.clone())?;

// Forward pass
let output = ln.forward(&input)?;
// Normalizes over the last dimension(s)

// Without affine parameters
let ln = LayerNorm::new_no_affine(vec![768], 1e-5, device.clone())?;

// Access parameters
let weight = &ln.weight;  // Option<Tensor>
let bias = &ln.bias;      // Option<Tensor>
```

### GroupNorm

```rust
use flame_core::GroupNorm;

// Create GroupNorm
let gn = GroupNorm::new(
    num_groups: 32,
    num_channels: 256,
    eps: 1e-5,
    affine: true,
    device.clone()
)?;

// Common configurations
let gn8 = GroupNorm::new(8, 64, 1e-5, true, device.clone())?;   // 8 groups
let gn32 = GroupNorm::new(32, 256, 1e-5, true, device.clone())?; // 32 groups

// Forward pass
let output = gn.forward(&input)?;
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels, height, width]
```

### RMSNorm (Root Mean Square Normalization)

```rust
use flame_core::RMSNorm;

// Used in many modern models (LLaMA, etc.)
let rms_norm = RMSNorm::new(
    hidden_size: 4096,
    eps: 1e-6,
    device.clone()
)?;

// Forward pass
let output = rms_norm.forward(&input)?;

// RMSNorm is more efficient than LayerNorm
// No mean subtraction, just RMS normalization
```

### BatchNorm2d

```rust
use flame_core::BatchNorm2d;

// Create BatchNorm2d
let bn = BatchNorm2d::new(
    num_features: 64,
    eps: 1e-5,
    momentum: 0.1,
    affine: true,
    track_running_stats: true,
    device.clone()
)?;

// Set training/eval mode
bn.train();  // Updates running stats
bn.eval();   // Uses running stats

// Forward pass
let output = bn.forward(&input)?;
```

## Activation Functions

Activations are typically methods on tensors, but FLAME provides layer wrappers:

```rust
use flame_core::{ReLU, GELU, SiLU, Tanh, Sigmoid};

// As layers
let relu = ReLU::new();
let gelu = GELU::new();
let silu = SiLU::new();  // Also known as Swish

// Forward pass
let output = relu.forward(&input)?;

// Or directly on tensors
let output = input.relu()?;
let output = input.gelu()?;
let output = input.silu()?;
let output = input.tanh()?;
let output = input.sigmoid()?;
```

## Dropout

```rust
use flame_core::Dropout;

// Create dropout layer
let dropout = Dropout::new(0.1);  // 10% dropout

// Set training/eval mode
dropout.train();  // Applies dropout
dropout.eval();   // No dropout

// Forward pass
let output = dropout.forward(&input)?;

// For reproducibility
let dropout = Dropout::new_with_seed(0.1, 42);
```

## Attention Layers

### Multi-Head Attention

```rust
use flame_core::MultiHeadAttention;

// Create attention layer
let mha = MultiHeadAttention::new(
    embed_dim: 768,
    num_heads: 12,
    dropout: 0.1,
    bias: true,
    device.clone()
)?;

// Forward pass
let output = mha.forward(
    query: &q,
    key: &k,
    value: &v,
    mask: None,  // Optional attention mask
    is_causal: false,
)?;

// With causal mask for autoregressive models
let output = mha.forward(&q, &k, &v, None, true)?;
```

### Flash Attention (Memory Efficient)

```rust
use flame_core::FlashAttention;

// More memory efficient attention
let flash_attn = FlashAttention::new(
    embed_dim: 768,
    num_heads: 12,
    dropout: 0.0,  // Dropout not supported in flash attention
    device.clone()
)?;

// Forward pass (same interface)
let output = flash_attn.forward(&q, &k, &v, None, false)?;
```

## Building Blocks for Diffusion Models

### ResNet Block

```rust
pub struct ResNetBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    norm1: GroupNorm,
    norm2: GroupNorm,
    shortcut: Option<Conv2d>,
}

impl ResNetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        device: Arc<CudaDevice>
    ) -> Result<Self> {
        let conv1 = Conv2d::new(in_channels, out_channels, 3, 1, 1, device.clone())?;
        let conv2 = Conv2d::new(out_channels, out_channels, 3, 1, 1, device.clone())?;
        
        let norm1 = GroupNorm::new(32, in_channels, 1e-5, true, device.clone())?;
        let norm2 = GroupNorm::new(32, out_channels, 1e-5, true, device.clone())?;
        
        let shortcut = if in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, 1, 1, 0, device)?)
        } else {
            None
        };
        
        Ok(Self { conv1, conv2, norm1, norm2, shortcut })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;
        
        let h = self.norm2.forward(&h)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;
        
        let shortcut = if let Some(conv) = &self.shortcut {
            conv.forward(x)?
        } else {
            x.clone()?
        };
        
        h.add(&shortcut)
    }
}
```

### Transformer Block

```rust
pub struct TransformerBlock {
    attn: MultiHeadAttention,
    mlp: MLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

pub struct MLP {
    fc1: Linear,
    fc2: Linear,
    act: GELU,
}

impl TransformerBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm architecture
        let h = self.norm1.forward(x)?;
        let h = self.attn.forward(&h, &h, &h, None, false)?;
        let x = x.add(&h)?;  // Residual
        
        let h = self.norm2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        x.add(&h)  // Residual
    }
}
```

## Layer Initialization

```rust
// Custom initialization
use flame_core::init::{xavier_uniform, kaiming_normal, constant, normal};

// Xavier/Glorot initialization
xavier_uniform(&mut linear.weight, gain: 1.0)?;

// Kaiming/He initialization  
kaiming_normal(&mut conv.weight, a: 0.0, mode: "fan_out")?;

// Constant initialization
constant(&mut bias, 0.0)?;

// Normal distribution
normal(&mut weight, mean: 0.0, std: 0.02)?;
```

## Best Practices

1. **Parameter initialization**: Use appropriate initialization for your architecture
2. **Normalization choice**: 
   - LayerNorm for transformers
   - GroupNorm for CNNs (batch-size independent)
   - RMSNorm for efficiency
3. **Activation functions**:
   - ReLU for CNNs
   - GELU/SiLU for transformers
4. **Dropout placement**: After activation, before residual connections
5. **Device consistency**: Ensure all layers are on the same device

## Integration Example for EriDiffusion

```rust
// Building a LoRA-enabled UNet block
pub struct LoRAUNetBlock {
    resnet1: ResNetBlock,
    resnet2: ResNetBlock,
    attn: Option<LoRAAttention>,
    downsample: Option<Conv2d>,
}

impl LoRAUNetBlock {
    pub fn new(config: &UNetConfig, device: Arc<CudaDevice>) -> Result<Self> {
        // Build layers with FLAME components
        let resnet1 = ResNetBlock::new(
            config.in_channels,
            config.out_channels,
            device.clone()
        )?;
        
        let resnet2 = ResNetBlock::new(
            config.out_channels,
            config.out_channels,
            device.clone()
        )?;
        
        let attn = if config.use_attention {
            Some(LoRAAttention::new(
                config.out_channels,
                config.attention_heads,
                config.lora_rank,
                device.clone()
            )?)
        } else {
            None
        };
        
        let downsample = if config.downsample {
            Some(Conv2d::new(
                config.out_channels,
                config.out_channels,
                3, 2, 1,  // stride 2 for downsampling
                device
            )?)
        } else {
            None
        };
        
        Ok(Self { resnet1, resnet2, attn, downsample })
    }
}
```