# FLAME Framework Documentation

FLAME (Fast Learning Accelerated Memory Engine) is a GPU-only tensor computation framework with automatic differentiation, built from scratch in Rust.

## Architecture Overview

FLAME is designed as a lightweight, efficient alternative to PyTorch/TensorFlow for Rust applications, with a focus on:
- Zero-copy operations where possible
- Efficient GPU memory management
- Type-safe tensor operations
- Automatic differentiation

## Core Components

### 1. [Tensor System](./tensor.md)
- GPU-backed tensors with shape tracking
- Lazy evaluation and operation fusion
- Broadcasting and shape manipulation
- Memory-efficient views and slicing

### 2. [Automatic Differentiation](./autograd.md)
- Dynamic computation graph
- Gradient accumulation
- Custom backward operations
- Gradient checkpointing support

### 3. [CUDA Operations](./cuda-ops.md)
- Custom CUDA kernels
- Integration with cuBLAS and cuDNN
- Kernel compilation and caching
- Launch configuration optimization

### 4. [Memory Management](./memory.md)
- GPU memory pooling
- Automatic memory reclamation
- Zero-copy tensor views
- Memory profiling tools

### 5. [Neural Network Modules](./nn-modules.md)
- Linear layers
- Convolution (Conv2d, Conv3d)
- Normalization (LayerNorm, GroupNorm, RMSNorm)
- Attention mechanisms
- Activation functions

## Design Principles

### GPU-Only Design
FLAME is designed exclusively for GPU computation. This allows us to:
- Eliminate CPU-GPU synchronization overhead
- Optimize memory layouts for GPU access patterns
- Use GPU-specific optimizations throughout

### Type Safety
All tensor operations are type-checked at compile time:
```rust
let a = Tensor::randn([32, 768], 0.0, 1.0, device)?;
let b = Tensor::randn([768, 512], 0.0, 1.0, device)?;
let c = a.matmul(&b)?; // Shape: [32, 512]
```

### Zero-Copy Operations
Many operations return views rather than copying data:
```rust
let x = Tensor::randn([100, 100], 0.0, 1.0, device)?;
let view = x.slice(0, 10)?; // No data copy
let transposed = x.transpose()?; // No data copy
```

### Automatic Differentiation
All operations that require gradients are automatically tracked:
```rust
let x = Tensor::randn([32, 10], 0.0, 1.0, device)?.requires_grad();
let w = Tensor::randn([10, 5], 0.0, 0.1, device)?.requires_grad();
let y = x.matmul(&w)?;
let loss = y.sum()?;

let grads = loss.backward()?;
let x_grad = grads.get(x.id).unwrap();
let w_grad = grads.get(w.id).unwrap();
```

## Integration with EriDiffusion

FLAME provides the computational backend for EriDiffusion:
- All model computations run through FLAME tensors
- Automatic differentiation handles backpropagation
- Custom kernels optimize diffusion-specific operations
- Memory pooling enables large model training

## Performance Characteristics

### Strengths
- Minimal overhead compared to PyTorch
- Efficient memory usage through pooling
- Fast custom kernels for common operations
- Zero-copy views reduce memory bandwidth

### Current Limitations
- Limited to NVIDIA GPUs (CUDA)
- No CPU fallback
- Smaller operation set than mature frameworks
- Single-GPU focus (multi-GPU in development)

## Examples

### Basic Tensor Operations
```rust
use flame_core::{Tensor, Shape, CudaDevice};
use std::sync::Arc;

let device = Arc::new(CudaDevice::new(0)?);

// Create tensors
let a = Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 1.0, device.clone())?;
let b = Tensor::ones(Shape::from_dims(&[32, 64]), device.clone())?;

// Operations
let c = a.add(&b)?;
let d = c.relu()?;
let e = d.sum()?;
```

### Neural Network Layer
```rust
use flame_core::nn::{Linear, Module};

let linear = Linear::new(768, 512, true, device)?;
let output = linear.forward(&input)?;
```

### Custom CUDA Kernel
```rust
use flame_core::cuda_kernels::CudaKernels;

CudaKernels::ensure_kernel(&device, "custom_op", r#"
extern "C" __global__ void custom_op(
    float* output,
    const float* input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}
"#)?;
```

## Next Steps

- [Tensor Operations Guide](./tensor.md)
- [Autograd System](./autograd.md)
- [Writing Custom Kernels](./cuda-ops.md)
- [Memory Optimization](./memory.md)