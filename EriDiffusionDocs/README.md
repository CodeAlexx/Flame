# EriDiffusion & FLAME Documentation

Welcome to the comprehensive documentation for EriDiffusion and FLAME - a pure Rust implementation of modern diffusion model training.

## Overview

**EriDiffusion** is a pure Rust trainer for state-of-the-art diffusion models including SDXL, SD 3.5, Flux, and more. It provides high-performance training with LoRA, DoRA, and other parameter-efficient fine-tuning methods.

**FLAME** (Fast Learning Accelerated Memory Engine) is a GPU-only tensor computation framework with automatic differentiation, designed specifically for deep learning in Rust.

## Documentation Structure

### 1. [Getting Started](./getting-started/README.md)
- Installation and setup
- Quick start guide
- First training run

### 2. [FLAME Framework](./flame/README.md)
- Architecture overview
- Tensor operations
- Automatic differentiation
- GPU memory management
- CUDA kernel development

### 3. [EriDiffusion Trainer](./eridiffusion/README.md)
- Training pipelines
- Model implementations
- Configuration system
- Data loading
- Sampling and inference

### 4. [Model Documentation](./models/README.md)
- [SDXL](./models/sdxl.md) - Stable Diffusion XL
- [SD 3.5](./models/sd35.md) - Stable Diffusion 3.5
- [Flux](./models/flux.md) - Flux Dev/Schnell
- Architecture details and training tips

### 5. [Training Guides](./training/README.md)
- [LoRA Training](./training/lora.md)
- [Full Fine-tuning](./training/full.md)
- [Configuration Reference](./training/config-reference.md)
- [Best Practices](./training/best-practices.md)

### 6. [API Reference](./api/README.md)
- FLAME API documentation
- EriDiffusion modules
- Public interfaces

### 7. [Development](./development/README.md)
- Contributing guidelines
- Adding new models
- Writing CUDA kernels
- Testing and debugging

### 8. [Troubleshooting](./troubleshooting/README.md)
- Common issues and solutions
- Performance optimization
- Memory management tips
- FAQ

## Key Features

### EriDiffusion
- ✅ Pure Rust implementation (no Python dependencies)
- ✅ Multiple model support: SDXL, SD 3.5, Flux, and more
- ✅ LoRA, DoRA, LoKr, and other efficient fine-tuning methods
- ✅ YAML-based configuration
- ✅ Integrated sampling during training
- ✅ Multi-GPU support
- ✅ Memory-efficient training with gradient checkpointing

### FLAME
- ✅ GPU-only tensor operations
- ✅ Automatic differentiation with backward pass
- ✅ Custom CUDA kernels
- ✅ Memory pooling and optimization
- ✅ Zero-copy tensor views
- ✅ Type-safe API
- ✅ Integration with cuDNN and cuBLAS

## System Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A5000, etc.)
- **CUDA**: 11.8 or higher
- **Rust**: 1.70 or higher
- **OS**: Linux (Ubuntu 20.04+ recommended)

## Quick Links

- [Installation Guide](./getting-started/installation.md)
- [Your First LoRA](./getting-started/first-lora.md)
- [Configuration Examples](./training/config-examples.md)
- [Performance Tuning](./troubleshooting/performance.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.