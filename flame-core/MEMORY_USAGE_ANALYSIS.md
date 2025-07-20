# FLAME Memory Usage Analysis

## Expected Memory Characteristics

Based on code analysis of the FLAME tensor framework:

### Tensor Memory Footprint

1. **Base Tensor Structure**:
   ```rust
   pub struct Tensor {
       id: TensorId,              // 8 bytes
       data: Arc<CudaSlice<f32>>, // 16 bytes (Arc pointer)
       shape: Shape,              // ~40 bytes (Vec<usize> + metadata)
       device: Arc<CudaDevice>,   // 16 bytes (Arc pointer)
       requires_grad: bool,       // 1 byte + padding
       grad_fn: Option<...>,      // 24 bytes
   }
   ```
   - Per-tensor overhead: ~112 bytes
   - Actual data: 4 bytes per f32 element

2. **Memory Pool Implementation**:
   - Pre-allocated blocks to reduce allocation overhead
   - Reuses freed memory for same-sized allocations
   - Thread-local caching for small tensors

### Expected Memory Usage by Operation

1. **Matrix Multiplication (A @ B)**:
   - Input A: M×K elements
   - Input B: K×N elements
   - Output: M×N elements
   - Temporary: None (uses CuBLAS)
   - Total: (M×K + K×N + M×N) × 4 bytes

2. **Convolution (Conv2D)**:
   - Input: B×C×H×W
   - Weights: O×C×KH×KW
   - Output: B×O×H'×W'
   - Im2col buffer: B×(C×KH×KW)×(H'×W')
   - Total: Significant due to im2col transformation

3. **Attention Mechanism**:
   - Q, K, V: B×L×D each
   - Attention scores: B×H×L×L
   - Peak memory: ~4×(B×L×D) + B×H×L×L

### Memory Optimization Features

1. **Gradient Checkpointing** (planned):
   - Trade compute for memory
   - Only store layer outputs, recompute intermediates
   - Expected 2-4x memory reduction

2. **Mixed Precision**:
   - FP16 storage: 2 bytes per element
   - FP32 compute: Temporary upcasting
   - ~50% memory reduction for weights

3. **Tensor Views**:
   - Zero-copy operations for reshape, transpose
   - Slice operations share underlying storage
   - Broadcast operations don't duplicate data

### GPU Memory Management

1. **CUDA Memory Pools**:
   ```rust
   // From memory_pool.rs
   - Small pool: <1MB allocations
   - Medium pool: 1MB-10MB
   - Large pool: >10MB
   - Direct allocation: >100MB
   ```

2. **Allocation Strategy**:
   - Lazy allocation on first use
   - Aggressive caching of freed memory
   - Defragmentation on OOM

### Expected Peak Memory Usage

For typical diffusion model training:

1. **SDXL LoRA (rank 32)**:
   - Model weights: ~6.5GB (FP16)
   - LoRA parameters: ~50MB
   - Gradients: ~50MB
   - Optimizer state: ~100MB
   - Activations (batch=1): ~2GB
   - **Total: ~9GB**

2. **Flux LoRA (rank 32)**:
   - Model weights: ~11GB (FP16)
   - LoRA parameters: ~80MB
   - Gradients: ~80MB
   - Optimizer state: ~160MB
   - Activations (batch=1): ~3GB
   - **Total: ~14.5GB**

3. **SD3.5 LoRA (rank 32)**:
   - Model weights: ~8GB (FP16)
   - LoRA parameters: ~60MB
   - Gradients: ~60MB
   - Optimizer state: ~120MB
   - Activations (batch=1): ~2.5GB
   - **Total: ~11GB**

### Memory Profiling Hooks

The framework includes (planned) memory profiling:
```rust
// Usage tracking per operation
tensor.track_memory_usage();

// Peak memory watermark
CudaDevice::get_peak_memory();

// Memory timeline
MemoryProfiler::start_recording();
```

### Current Limitations

1. **No Gradient Accumulation**: All gradients stored simultaneously
2. **No CPU Offloading**: Everything stays on GPU
3. **No Quantization**: Only FP32/FP16 supported
4. **No Sharding**: Single GPU only

These limitations mean actual memory usage would be higher than theoretical minimums until optimization features are implemented.