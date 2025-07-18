# FLAME Architecture Guide

## Core Philosophy: Separation of Concerns

**FLAME = Tensor computation framework (like PyTorch core)**  
**Trainer = Model implementations and training logic**

## What Belongs Where

### FLAME Side (Framework)
```rust
// Core compute primitives
- Tensor operations
- Autograd engine  
- CUDA kernels
- Basic NN layers (Conv2d, Linear, Attention)
- Optimizers
- Memory management
```

### Trainer Side (Application)
```rust
// Model-specific components
- VAE (specific architecture)
- CLIP (specific model)
- Text Encoders (T5, CLIP text)
- UNet/DiT models
- Training loops
- Data loading
- Inference pipelines
```

## Why This Split Makes Sense

1. **FLAME stays focused** - It's a tensor framework, not a model zoo
2. **Reusability** - Other projects can use FLAME without pulling in specific models
3. **Lighter dependencies** - FLAME doesn't need tokenizers, image processing, etc.
4. **Cleaner API** - Framework vs application logic separated

## Example Structure

```
flame/
├── src/
│   ├── tensor.rs       # Core tensor ops
│   ├── nn/            # Generic layers
│   │   ├── conv.rs
│   │   ├── attention.rs
│   │   └── linear.rs
│   └── cuda/          # Low-level compute

diffusion-trainer/
├── src/
│   ├── models/
│   │   ├── vae.rs     # Uses flame::nn::Conv2d
│   │   ├── clip.rs    # Uses flame::nn::Linear
│   │   └── unet.rs    # Uses flame::nn::Attention
│   ├── inference.rs
│   └── training.rs
```

## The Interface

### FLAME provides building blocks:
```rust
// FLAME exports
pub use flame::nn::{Conv2d, Linear, MultiHeadAttention};
pub use flame::{Tensor, Optimizer};
```

### Trainer assembles them into models:
```rust
// In trainer
use flame::nn::{Conv2d, GroupNorm};

struct VAEDecoder {
    conv1: Conv2d,
    norm1: GroupNorm,
    // ... assembled from FLAME parts
}
```

## Implementation Guidelines

### What FLAME Should Provide

1. **Core Tensor Operations**
   - Basic math ops (add, mul, matmul, etc.)
   - Activation functions (relu, gelu, sigmoid, etc.)
   - Reduction ops (sum, mean, max, etc.)

2. **Neural Network Primitives**
   - Linear/Dense layers
   - Convolution layers (1D, 2D, 3D)
   - Normalization layers (BatchNorm, LayerNorm, GroupNorm, RMSNorm)
   - Attention mechanisms (scaled dot-product, multi-head)
   - Pooling layers

3. **Training Infrastructure**
   - Autograd/backward pass
   - Optimizers (SGD, Adam, AdamW, etc.)
   - Learning rate schedulers
   - Gradient clipping
   - Mixed precision training

4. **Memory Management**
   - CUDA memory pools
   - Gradient checkpointing
   - CPU offloading utilities

### What Trainers Should Implement

1. **Specific Model Architectures**
   - VAE encoder/decoder structures
   - CLIP vision/text encoders
   - UNet/DiT diffusion models
   - LoRA/ControlNet adapters

2. **Data Processing**
   - Image loading and preprocessing
   - Text tokenization
   - Batch collation
   - Data augmentation

3. **Training Loops**
   - Diffusion noise scheduling
   - Loss computation
   - Validation logic
   - Checkpoint saving/loading

4. **Inference Pipelines**
   - Sampling algorithms
   - Prompt processing
   - Image generation workflows

## Example: VAE Implementation

### In FLAME (generic components):
```rust
// flame/src/nn/conv.rs
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    // ... generic convolution
}

// flame/src/nn/norm.rs  
pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    // ... generic group normalization
}
```

### In Trainer (specific architecture):
```rust
// trainer/src/models/vae.rs
use flame::nn::{Conv2d, GroupNorm};

pub struct VAEDecoder {
    // Specific architecture for Stable Diffusion VAE
    conv_in: Conv2d,
    
    // Decoder blocks with specific channel counts
    blocks: Vec<DecoderBlock>,
    
    // Output layers
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl VAEDecoder {
    pub fn stable_diffusion_vae() -> Self {
        // Specific configuration for SD VAE
        Self {
            conv_in: Conv2d::new(4, 512, 3, 1, 1),
            // ... specific architecture
        }
    }
}
```

## Benefits of This Architecture

1. **Clean Dependencies**: FLAME only depends on CUDA/tensor libraries
2. **Testability**: Framework and models can be tested independently  
3. **Flexibility**: Easy to swap implementations or add new models
4. **Performance**: Framework can optimize without model-specific concerns
5. **Maintainability**: Clear boundaries make code easier to understand

This keeps FLAME lean and focused on what it does best - fast tensor computation with autograd.