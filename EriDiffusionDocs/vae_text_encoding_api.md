# EriDiffusion VAE and Text Encoding API Documentation

## Overview

EriDiffusion provides comprehensive VAE (Variational Autoencoder) and text encoding functionality for diffusion models. The system supports multiple model architectures including SDXL, SD 3.5, and Flux, each with their specific VAE and text encoder requirements.

## VAE Encoding/Decoding

### 1. SDXLVAEEncoder (`src/trainers/vae_encoder.rs`)

Direct weight implementation for SDXL VAE without VarBuilder.

```rust
pub struct SDXLVAEEncoder {
    weights: std::collections::HashMap<String, Tensor>,
    device: Device,
}

impl SDXLVAEEncoder {
    /// Create new VAE encoder
    pub fn new(
        weights: std::collections::HashMap<String, Tensor>, 
        device: Device
    ) -> flame_core::Result<Self>;
    
    /// Encode image to latent space
    pub fn encode(&self, image: &Tensor) -> flame_core::Result<Tensor>;
    
    /// Decode latent to image
    pub fn decode(&self, latent: &Tensor) -> flame_core::Result<Tensor>;
}
```

**Parameters:**
- `image: &Tensor` - Input image tensor in range [0, 1], shape `[B, C, H, W]`
- `latent: &Tensor` - Latent tensor, shape `[B, 4, H/8, W/8]` for SDXL

**Returns:**
- `encode()` - Latent tensor scaled by 0.18215
- `decode()` - Image tensor in range [0, 1]

**Internal Methods:**
```rust
fn resnet_block(&self, x: &Tensor, block_idx: i32, layer_idx: usize) -> Result<Tensor>;
fn attention_block(&self, x: &Tensor) -> Result<Tensor>;
fn downsample(&self, x: &Tensor, block_idx: usize) -> Result<Tensor>;
fn upsample(&self, x: &Tensor, block_idx: usize) -> Result<Tensor>;
```

### 2. VAE Helper Functions

```rust
/// 2D convolution with bias
fn conv2d(
    x: &Tensor, 
    weight: &Tensor, 
    bias: &Tensor, 
    stride: usize, 
    padding: usize
) -> Result<Tensor>;

/// Group normalization
fn group_norm(
    x: &Tensor, 
    groups: usize, 
    scale: &Tensor, 
    bias: &Tensor
) -> Result<Tensor>;

/// Linear operation with optional bias
fn linear_op(
    x: &Tensor, 
    weight: &Tensor, 
    bias: Option<&Tensor>
) -> Result<Tensor>;
```

### 3. VAE Architecture Details

**SDXL VAE:**
- 4-channel latent space
- 8x downsampling factor
- Scaling factor: 0.18215
- Architecture: Encoder → Distribution → Sample → Decoder

**SD 3.5 VAE:**
- 16-channel latent space  
- Different scaling factor
- More complex architecture

**Flux VAE:**
- 16-channel latent space
- 2x2 patch processing (64 channels after patchify)
- Custom normalization

### Usage Example:

```rust
// Load VAE weights
let vae_weights = load_safetensors("/path/to/vae.safetensors", &device)?;
let vae = SDXLVAEEncoder::new(vae_weights, device)?;

// Encode image to latent
let image = load_image("/path/to/image.png")?; // [1, 3, 1024, 1024]
let latent = vae.encode(&image)?; // [1, 4, 128, 128]

// Decode back to image
let reconstructed = vae.decode(&latent)?; // [1, 3, 1024, 1024]
```

## Text Encoding

### 1. TextEncoders (`src/trainers/text_encoders.rs`)

Unified text encoding for multiple models.

```rust
pub struct TextEncoders {
    pub clip_l: Option<clip::ClipTextTransformer>,
    pub clip_g: Option<clip::ClipTextTransformer>,
    pub t5: Option<T5EncoderModel>,
    tokenizer_clip: Option<Tokenizer>,
    tokenizer_t5: Option<Tokenizer>,
    pub device: Device,
}

impl TextEncoders {
    /// Create new text encoders
    pub fn new(device: Device) -> Self;
    
    /// Load CLIP-L encoder
    pub fn load_clip_l(&mut self, model_path: &str) -> flame_core::Result<()>;
    
    /// Load CLIP-G encoder
    pub fn load_clip_g(&mut self, model_path: &str) -> flame_core::Result<()>;
    
    /// Load T5-XXL encoder
    pub fn load_t5(&mut self, model_path: &str) -> flame_core::Result<()>;
    
    /// Load tokenizers
    pub fn load_tokenizers(
        &mut self, 
        clip_tokenizer_path: &str, 
        t5_tokenizer_path: &str
    ) -> flame_core::Result<()>;
    
    /// Main encoding function
    pub fn encode(
        &mut self, 
        text: &str, 
        max_sequence_length: usize
    ) -> flame_core::Result<(Tensor, Tensor)>;
    
    /// SDXL-specific encoding
    pub fn encode_sdxl(
        &mut self, 
        text: &str, 
        _max_sequence_length: usize
    ) -> flame_core::Result<(Tensor, Tensor)>;
    
    /// Batch encoding
    pub fn encode_batch(
        &mut self, 
        texts: &[String], 
        max_sequence_length: usize
    ) -> flame_core::Result<(Tensor, Tensor)>;
    
    /// Unconditional encoding for CFG
    pub fn encode_unconditional(
        &mut self, 
        batch_size: usize, 
        max_sequence_length: usize
    ) -> flame_core::Result<(Tensor, Tensor)>;
}
```

**Returns:**
- First tensor: Context embeddings
  - SDXL: `[batch, 77, 2048]` (CLIP-L + CLIP-G concatenated)
  - SD3.5/Flux: `[batch, 77+max_seq_len, 4096]` (CLIP + T5)
- Second tensor: Pooled embeddings `[batch, pooled_dim]`

### 2. Model-Specific Encoding

**SDXL:**
```rust
// Uses CLIP-L (768 dim) + CLIP-G (1280 dim)
// Concatenated to 2048 dimensions
// Max sequence length: 77 tokens
// Unconditional: zeros
```

**SD 3.5 / Flux:**
```rust
// Uses CLIP-L + CLIP-G + T5-XXL
// CLIP embeddings padded to 2048, concatenated to 4096
// Then concatenated with T5 embeddings
// Unconditional: empty string
```

### 3. Tokenization Functions

```rust
fn tokenize_clip(&self, text: &str, max_length: usize) -> Result<Tensor>;
fn tokenize_t5(&self, text: &str, max_length: usize) -> Result<Tensor>;
```

### 4. Helper Functions

```rust
/// Pad tensor to target dimension
fn pad_to_dim(&self, tensor: &Tensor, target_dim: usize) -> Result<Tensor>;

/// Pool CLIP-G embeddings
fn pool_clip_g(&self, hidden_states: &Tensor) -> Result<Tensor>;
```

### Usage Example:

```rust
// Create and load encoders
let mut encoders = TextEncoders::new(Device::cuda(0)?);

// For SDXL
encoders.load_clip_l("/path/to/clip_l.safetensors")?;
encoders.load_clip_g("/path/to/clip_g.safetensors")?;
encoders.load_clip_tokenizer("/path/to/tokenizer.json")?;

// Encode text
let (context, pooled) = encoders.encode_sdxl(
    "a beautiful landscape", 
    77
)?;

// For SD3.5/Flux - also load T5
encoders.load_t5("/path/to/t5xxl.safetensors")?;
encoders.load_tokenizers(
    "/path/to/clip_tokenizer.json",
    "/path/to/t5_tokenizer.json"
)?;

let (context, pooled) = encoders.encode(
    "a beautiful landscape",
    256  // T5 max length
)?;

// Batch encoding
let prompts = vec![
    "a cat".to_string(),
    "a dog".to_string(),
];
let (batch_context, batch_pooled) = encoders.encode_batch(&prompts, 77)?;

// Unconditional for CFG
let (uncond_context, uncond_pooled) = encoders.encode_unconditional(4, 77)?;
```

## Advanced VAE Features

### 1. VAE Tiling (`src/trainers/vae_tiling.rs`)

For processing large images that don't fit in memory.

### 2. VAE Wrapper (`src/trainers/sdxl_vae_wrapper.rs`)

Provides a high-level interface for VAE operations.

### 3. Native VAE (`src/trainers/sdxl_vae_native.rs`)

Direct tensor operations for maximum performance.

## Model-Specific Configurations

### CLIP Configurations:

```rust
// CLIP-L (v1.5)
clip::Config::v1_5()

// CLIP-G (SDXL)
clip::Config::sdxl2()
```

### T5 Configuration:

```rust
T5Config {
    vocab_size: 32128,
    d_model: 4096,
    d_ff: 10240,
    num_layers: 24,
    num_heads: 64,
    relative_attention_num_buckets: 32,
    relative_attention_max_distance: 128,
    dropout_rate: 0.1,
    layer_norm_epsilon: 1e-6,
}
```

## Integration Notes

1. **Device Compatibility**: All operations support both CPU and CUDA devices
2. **Dtype Matching**: Automatically matches model dtype (F16, BF16, F32)
3. **Memory Efficiency**: Use tiling for large images
4. **Batch Processing**: All functions support batched inputs
5. **Model Differences**: 
   - SDXL uses zeros for unconditional
   - SD3.5/Flux use empty strings
   - Different latent channel counts (4 vs 16)

## Best Practices

1. **Precompute Latents**: Cache VAE outputs for faster training
2. **Batch Encoding**: Process multiple prompts together
3. **Match Dtypes**: Ensure consistent dtypes across models
4. **Handle Empty Prompts**: Use model-specific unconditional handling
5. **Memory Management**: Clear caches between large batches