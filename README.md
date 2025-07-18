# FLAME (Fast Learning Accelerated Matrix Engine)

A pure Rust tensor computation framework designed for GPU-accelerated deep learning with full gradient support. FLAME is being developed to replace Candle in EriDiffusion, as Candle's architecture fundamentally prevents gradient modification necessary for training.

## Features

- **GPU-Only Design**: Built from the ground up for NVIDIA GPUs with no CPU fallbacks
- **Training Support**: Full autograd system with gradient tracking and automatic differentiation
- **Runtime Kernel Compilation**: Uses NVRTC for JIT compilation of CUDA kernels
- **Memory Efficient**: Arc-based tensor memory management with zero-copy operations
- **Pure Rust**: No Python dependencies or bindings required

## Why FLAME?

### The Candle Problem
Despite extensive efforts to modify Candle for training support, its fundamental architecture prevents gradient modification. Candle's `VarBuilder` returns immutable tensors, making it impossible to implement proper backpropagation for training neural networks.

### FLAME Solution
- **Native Gradient Support**: Built from the ground up with mutable gradients and autograd
- **Training Ready**: Designed specifically for training diffusion models in EriDiffusion
- **Drop-in Replacement**: Will integrate seamlessly into EriDiffusion's training pipeline
- **Custom Kernels**: Easy to add custom CUDA kernels via NVRTC
- **Type Safe**: Leverages Rust's type system for safe GPU programming

## Current Status

### ✅ Working
- Basic tensor operations (add, mul, matmul, etc.)
- Activation functions (ReLU, Sigmoid, GELU, SiLU, Tanh)
- Gradient tracking with `requires_grad`
- Manual gradient computation
- CUDA memory management
- NVRTC kernel compilation

### 🚧 In Progress
- Automatic differentiation API improvements
- Convolution operations
- Batch normalization
- Full model migration examples

## Quick Start

```rust
use flame_core::{Tensor, Shape, CudaDevice};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    
    // Create tensors
    let a = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?;
    let b = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?;
    
    // Perform operations
    let c = a.add(&b)?;
    let d = c.relu()?;
    
    println!("Result shape: {:?}", d.shape().dims());
    Ok(())
}
```

## Building

### Prerequisites
- Rust 1.70+
- CUDA Toolkit 11.0+
- NVIDIA GPU with compute capability 7.0+

### Build Commands
```bash
# Build the library
cargo build --release

# Run tests
cargo test --release

# Run examples
cargo run --bin test_basic_ops --release
```

## Architecture

FLAME consists of several key components:

- **flame-core**: Core tensor operations and autograd engine
- **CUDA Kernels**: GPU kernels compiled via NVRTC
- **Gradient System**: Separate gradient storage for clean API
- **Autograd Engine**: Tracks operations and computes gradients

## Examples

### Basic Operations
```rust
// Matrix multiplication
let x = Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 1.0, device.clone())?;
let w = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 0.02, device.clone())?;
let y = x.matmul(&w)?;

// Activation function
let activated = y.relu()?;

// Reduction
let sum = activated.sum()?;
```

### Training Simulation
```rust
// Initialize parameters
let mut weight = Tensor::randn(Shape::from_dims(&[10, 5]), 0.0, 0.02, device.clone())?;

// Training loop
for epoch in 0..num_epochs {
    // Forward pass
    let output = input.matmul(&weight)?;
    
    // Compute loss
    let loss = compute_mse_loss(&output, &target)?;
    
    // Manual gradient computation (autograd API in progress)
    let grad_weight = compute_gradients(&loss, &weight)?;
    
    // Update weights
    weight = weight.sub(&grad_weight.mul_scalar(learning_rate)?)?;
}
```

## Integration with EriDiffusion

FLAME is being developed as the tensor backend for EriDiffusion, replacing the Candle fork. The integration will enable:

- Full training support for diffusion models (SDXL, SD3.5, Flux, etc.)
- LoRA, DoRA, and other adapter training
- Gradient checkpointing for memory efficiency
- Custom CUDA kernels for diffusion-specific operations

## Roadmap

### Immediate (for EriDiffusion integration)
- [ ] Improve autograd API for easier usage
- [ ] Implement conv2d operations (required for UNet/DiT)
- [ ] Add layer normalization (required for transformers)
- [ ] Migrate EriDiffusion inference pipeline
- [ ] Migrate EriDiffusion training code

### Future
- [ ] Optimize kernel performance
- [ ] Add mixed precision (FP16) support
- [ ] Implement Flash Attention
- [ ] Add distributed training support

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Built with:
- [cudarc](https://github.com/coreylowman/cudarc) - Rust CUDA bindings
- NVIDIA CUDA Toolkit