# EriDiffusion Documentation

EriDiffusion is a pure Rust implementation of state-of-the-art diffusion model training, supporting SDXL, SD 3.5, Flux, and more.

## Overview

EriDiffusion provides:
- 🚀 High-performance training with minimal overhead
- 🎯 Multiple model architectures (SDXL, SD 3.5, Flux)
- 🔧 LoRA, DoRA, LoKr, and other efficient fine-tuning methods
- 📊 Integrated sampling during training
- 💾 Memory-efficient training with gradient checkpointing
- 🎛️ YAML-based configuration system

## Architecture

### Core Components

1. **Model Implementations** (`src/models/`)
   - SDXL U-Net implementation
   - SD 3.5 MMDiT (Multimodal Diffusion Transformer)
   - Flux hybrid architecture
   - VAE encoders/decoders
   - Text encoders (CLIP, T5)

2. **Training Pipelines** (`src/trainers/`)
   - Model-specific training loops
   - Gradient accumulation and checkpointing
   - Mixed precision training
   - Optimizer implementations

3. **Network Adapters** (`src/networks/`)
   - LoRA (Low-Rank Adaptation)
   - DoRA (Weight-Decomposed LoRA)
   - LoKr (Low-Rank Kronecker product)
   - Custom adapter implementations

4. **Data Loading** (`src/trainers/`)
   - Efficient image loading and preprocessing
   - Caption handling
   - Multi-resolution bucketing
   - Latent caching

5. **Inference & Sampling** (`src/inference/`)
   - Sampling during training
   - Various scheduler implementations
   - Model weight loading and conversion

## Model Support

### Stable Diffusion XL (SDXL)
- U-Net architecture with attention blocks
- Dual text encoder (CLIP-L + CLIP-G)
- Resolution: 1024x1024
- Channel configuration: 4-channel VAE

### Stable Diffusion 3.5
- MMDiT architecture (Multimodal Diffusion Transformer)
- Triple text encoding (CLIP-L + CLIP-G + T5-XXL)
- Flow matching objective
- 16-channel VAE
- Resolution: up to 1024x1024

### Flux
- Hybrid architecture with double and single stream blocks
- T5-XXL text encoder
- Guidance embedding support
- 16-channel VAE with 2x2 patches
- Resolution: flexible, typically 1024x1024

## Training Process

### 1. Configuration
Training is configured via YAML files:

```yaml
job: extension
config:
  name: "my_lora"
  process:
    - type: 'sd_trainer'
      device: cuda:0
      network:
        type: "lora"
        linear: 16
        linear_alpha: 16
      model:
        name_or_path: "/path/to/model.safetensors"
        arch: "sdxl"  # or "sd35", "flux"
```

### 2. Model Loading
Models are loaded from local safetensors files:
```rust
let model = load_model_weights(&config.model.name_or_path)?;
let vae = load_vae_weights(&config.model.vae_path)?;
```

### 3. Training Loop
The training loop follows these steps:
1. Load batch of images and captions
2. Encode images to latents (with optional caching)
3. Encode text prompts
4. Add noise using the scheduler
5. Predict noise with the model
6. Compute loss
7. Backward pass and parameter update
8. Optional: Generate samples

### 4. Sampling Integration
During training, samples can be generated to monitor progress:
```rust
if step % config.sample_every == 0 {
    let samples = generate_samples(&model, &vae, &text_encoder, &config)?;
    save_samples(&samples, step)?;
}
```

## Memory Optimization

### Gradient Checkpointing
Reduces memory usage by recomputing activations:
```rust
config.train.gradient_checkpointing = true;
```

### Mixed Precision
Train in FP16/BF16 for reduced memory:
```rust
config.train.dtype = "bf16";
```

### 8-bit Optimizers
Use memory-efficient optimizers:
```rust
config.train.optimizer = "adamw8bit";
```

### Latent Caching
Cache VAE encodings to disk:
```rust
config.datasets[0].cache_latents_to_disk = true;
```

## Performance Tuning

### Multi-Resolution Training
Train on multiple resolutions for better generalization:
```yaml
datasets:
  - resolution: [512, 768, 1024]
```

### Batch Size Optimization
Find optimal batch size for your GPU:
- 24GB VRAM: batch_size 1-4 depending on model
- Enable gradient accumulation for larger effective batches

### CPU Offloading
Offload optimizer states and gradients:
```rust
config.process[0].cpu_offload = true;
```

## Network Adapters

### LoRA (Low-Rank Adaptation)
Efficient fine-tuning by learning low-rank updates:
```yaml
network:
  type: "lora"
  linear: 16        # rank
  linear_alpha: 16  # alpha for scaling
```

### DoRA (Weight-Decomposed LoRA)
Improved LoRA with weight magnitude and direction decomposition:
```yaml
network:
  type: "dora"
  linear: 16
```

### LoKr (Low-Rank Kronecker)
Uses Kronecker product for even more parameter efficiency:
```yaml
network:
  type: "lokr"
  linear: 16
  lokr_factor: 4
```

## File Organization

```
eridiffusion/
├── src/
│   ├── bin/              # Training executables
│   ├── models/           # Model implementations
│   ├── trainers/         # Training logic
│   ├── networks/         # LoRA/adapter implementations
│   ├── inference/        # Inference and sampling
│   └── loaders/          # Model loading utilities
├── config/               # Example configurations
└── Cargo.toml
```

## Integration with FLAME

EriDiffusion uses FLAME as its tensor computation backend:

```rust
use flame_core::{Tensor, Shape};

// All computations use FLAME tensors
let latents = Tensor::randn(shape, 0.0, 1.0, device)?;
let noise_pred = unet.forward(&latents, &timestep, &encoder_hidden_states)?;
```

FLAME provides:
- Automatic differentiation for backpropagation
- GPU memory management
- Custom CUDA kernels for performance
- Type-safe tensor operations

## Next Steps

- [Training Your First LoRA](../training/first-lora.md)
- [Model-Specific Guides](../models/README.md)
- [Configuration Reference](../training/config-reference.md)
- [Troubleshooting Guide](../troubleshooting/README.md)