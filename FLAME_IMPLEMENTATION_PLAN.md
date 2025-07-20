# FLAME Framework GPU & Backward Pass Implementation Plan

## Executive Summary

FLAME (Flexible Learning and Autograd Made Easy) is currently 40% complete with solid forward pass implementations but critically missing backward pass functionality required for training. This implementation plan prioritizes GPU operations and backward passes to enable training within 2 weeks.

**Current State**: 543 Candle references prevent compilation, but core FLAME functionality is solid.
**Goal**: Make FLAME functional for training diffusion models, starting with SDXL LoRA.
**Timeline**: 14 days of focused development.

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Priority 1: Critical Compilation Fixes](#priority-1-critical-compilation-fixes)
3. [Priority 2: GPU Backward Pass Implementation](#priority-2-gpu-backward-pass-implementation)
4. [Priority 3: Autograd System Consolidation](#priority-3-autograd-system-consolidation)
5. [Priority 4: GPU Optimization](#priority-4-gpu-optimization)
6. [Implementation Schedule](#implementation-schedule)
7. [Success Metrics](#success-metrics)
8. [Risk Mitigation](#risk-mitigation)

## Current State Analysis

### Working Components ✅
- **CUDA Device Management**: Proper GPU initialization and memory allocation
- **Forward Passes**: Conv2D, Linear, Attention, Pooling, Normalization
- **Tensor Operations**: Basic math ops, reshaping, broadcasting
- **Memory Management**: CUDA memory pools, allocation tracking
- **Custom CUDA Kernels**: Compiled and ready for use

### Missing Components ❌
- **Backward Passes**: Only 5 of 50+ operations have backward implemented
- **Autograd Integration**: 3 different versions not consolidated
- **Test Compilation**: 58+ errors due to Arc double-wrapping
- **Training Loop**: No complete example that actually trains

### Critical Issues
1. **Arc Double-Wrapping**: Tests expect `Arc<CudaDevice>` but get `Arc<Arc<CudaDevice>>`
2. **Missing Imports**: Test modules missing basic imports
3. **No Linear Backward**: Most fundamental operation missing backward
4. **Autograd Confusion**: Three implementations with no clear integration

## Priority 1: Critical Compilation Fixes

### Day 1: Arc Double-Wrapping Fix (Morning)

**Problem**: 
```rust
// Current (WRONG)
let device = Arc::new(CudaDevice::new(0)?);

// Should be
let device = CudaDevice::new(0)?; // Already returns Arc<CudaDevice>
```

**Implementation Steps**:
1. Search and replace all occurrences in test files
2. Update helper functions:
```rust
// Fix in all test files
fn create_test_device() -> Result<Arc<CudaDevice>> {
    CudaDevice::new(0) // Remove Arc::new wrapper
}
```
3. Run compilation to verify all 58 errors resolved

**Production Files to Update**:
- `/home/alex/diffusers-rs/flame/flame-core/src/lib.rs` (module exports)
- `/home/alex/diffusers-rs/flame/flame-core/src/tensor.rs` (if needed)
- All production code using CudaDevice::new()

### Day 1: Module Exports Fix (Afternoon)

**Problem**: Production code needs proper module exports and visibility

**Implementation**:
1. Fix module visibility in `src/lib.rs`:
```rust
// Ensure all necessary types are exported
pub use crate::autograd::{AutogradContext, Op};
pub use crate::gradient::GradientMap;
pub use crate::tensor::{Tensor, TensorId};
pub use crate::shape::Shape;
pub use cudarc::driver::CudaDevice;
```

2. Create convenience prelude for production use:
```rust
pub mod prelude {
    pub use crate::{
        Tensor, Shape, Result, FlameError,
        CudaDevice, AutogradContext, GradientMap,
    };
    pub use std::sync::Arc;
}
```

### Day 2: Verify Basic Operations

**Verification Protocol**:
1. Compile production code to ensure all imports work
2. Verify forward operations in training pipelines
3. Check memory allocation/deallocation in production use
4. Ensure CUDA kernels load properly in real scenarios

## Priority 2: GPU Backward Pass Implementation

### Day 3: Linear Layer Backward

**Current State**: Forward pass works, no backward implementation

**Implementation**:
```rust
// In src/linear.rs
impl Linear {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // ... existing forward code ...
        
        // Add autograd recording
        if input.requires_grad || self.weight.requires_grad {
            let saved = vec![
                (input.id, input.clone()?),
                (self.weight.id, self.weight.clone()?),
            ];
            
            AutogradContext::record_op(
                output.id,
                Op::Linear {
                    input: input.id,
                    weight: self.weight.id,
                    bias: self.bias.as_ref().map(|b| b.id),
                },
                saved,
            );
        }
        
        Ok(output)
    }
}

// In src/autograd_ops.rs
pub fn linear_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    has_bias: bool,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    // Gradient w.r.t input: grad_output @ weight
    let grad_input = grad_output.matmul(weight)?;
    
    // Gradient w.r.t weight: grad_output.T @ input
    let grad_output_reshaped = grad_output.reshape(&[-1, grad_output.shape().dims().last().unwrap()])?;
    let input_reshaped = input.reshape(&[-1, input.shape().dims().last().unwrap()])?;
    let grad_weight = grad_output_reshaped.transpose()?.matmul(&input_reshaped)?;
    
    // Gradient w.r.t bias: sum grad_output along batch dimensions
    let grad_bias = if has_bias {
        Some(grad_output_reshaped.sum_dim(0)?)
    } else {
        None
    };
    
    Ok((grad_input, grad_weight, grad_bias))
}
```

**CUDA Optimization**:
- Use cuBLAS for matrix multiplication
- Fuse bias gradient computation
- Implement memory-efficient transpose

### Days 4-5: Conv2D Backward Integration

**Current State**: CUDA kernels exist but not fully integrated

**Implementation Plan**:
1. Complete im2col/col2im kernels
2. Implement gradient computations:
   - Input gradient: col2im(weight @ grad_col)
   - Weight gradient: grad_col @ input_col.T
   - Bias gradient: sum(grad_output, dims=[0,2,3])

```rust
// In src/cuda_conv2d.rs
pub fn conv2d_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    // Existing implementation needs completion
    
    // 1. Gradient w.r.t input (using transposed convolution)
    let grad_input = conv_transpose2d(
        grad_output,
        weight,
        stride,
        padding,
        input.shape(),
    )?;
    
    // 2. Gradient w.r.t weight (using im2col)
    let input_col = im2col(input, weight.shape(), stride, padding)?;
    let grad_col = grad_output.reshape(&[-1, grad_output.shape().dims()[1]])?;
    let grad_weight = grad_col.transpose()?.matmul(&input_col)?;
    
    // 3. Gradient w.r.t bias
    let grad_bias = grad_output.sum_dims(&[0, 2, 3])?;
    
    Ok((grad_input, grad_weight, Some(grad_bias)))
}
```

**CUDA Kernel Optimization**:
- Optimize im2col for different stride/padding combinations
- Use shared memory for weight access
- Implement NHWC format support for better memory coalescing

### Days 6-7: Attention Mechanism Backward

**Implementation**:
```rust
// In src/attention.rs
pub fn scaled_dot_product_attention_backward(
    grad_output: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attention_weights: &Tensor,
    scale: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch_size = query.shape().dims()[0];
    let seq_len = query.shape().dims()[1];
    let head_dim = query.shape().dims()[2];
    
    // Gradient w.r.t value: attention_weights.T @ grad_output
    let grad_value = attention_weights.transpose_dims(-2, -1)?
        .matmul(&grad_output)?;
    
    // Gradient w.r.t attention weights
    let grad_attention = grad_output.matmul(&value.transpose_dims(-2, -1)?)?;
    
    // Gradient through softmax
    let grad_scores = softmax_backward(&grad_attention, &attention_weights)?;
    
    // Gradient w.r.t query and key
    let grad_scores_scaled = grad_scores.mul_scalar(scale)?;
    let grad_query = grad_scores_scaled.matmul(&key)?;
    let grad_key = grad_scores_scaled.transpose_dims(-2, -1)?.matmul(&query)?;
    
    Ok((grad_query, grad_key, grad_value))
}
```

**Flash Attention Backward**:
- Implement memory-efficient backward using block-wise computation
- Recompute attention weights on-the-fly
- Support for causal masks

## Priority 3: Autograd System Consolidation

### Days 8-9: Merge Autograd Implementations

**Decision**: Use `autograd_v3.rs` (thread-local) as base

**Consolidation Plan**:
1. Move best features from each version:
   - `autograd.rs`: Simple API
   - `autograd_v2.rs`: Efficient memory management
   - `autograd_v3.rs`: Thread-local safety

2. Create unified backward dispatch:
```rust
// In src/autograd_v3.rs
impl AutogradEngine {
    pub fn backward(&self, loss: &Tensor) -> Result<GradientMap> {
        let mut gradients = GradientMap::new();
        gradients.insert(loss.id, Tensor::ones_like(loss)?);
        
        // Process tape in reverse order
        let tape = self.tape.lock().unwrap();
        for entry in tape.iter().rev() {
            if let Some(grad_output) = gradients.get(&entry.output_id) {
                match &entry.op {
                    Op::Add { lhs, rhs } => {
                        let (grad_lhs, grad_rhs) = add_backward(grad_output)?;
                        accumulate_gradient(&mut gradients, *lhs, grad_lhs);
                        accumulate_gradient(&mut gradients, *rhs, grad_rhs);
                    }
                    Op::Linear { input, weight, bias } => {
                        let input_tensor = &entry.saved_tensors[input];
                        let weight_tensor = &entry.saved_tensors[weight];
                        let (grad_input, grad_weight, grad_bias) = 
                            linear_backward(grad_output, input_tensor, weight_tensor, bias.is_some())?;
                        accumulate_gradient(&mut gradients, *input, grad_input);
                        accumulate_gradient(&mut gradients, *weight, grad_weight);
                        if let Some(b) = bias {
                            accumulate_gradient(&mut gradients, *b, grad_bias.unwrap());
                        }
                    }
                    // ... other operations ...
                }
            }
        }
        
        Ok(gradients)
    }
}
```

### Day 10: Complete Missing Operations

**Implementation Priority**:
1. **Pooling Operations**:
   - MaxPool2d backward (using saved indices)
   - AvgPool2d backward (uniform distribution)

2. **Normalization**:
   - LayerNorm backward (already partially implemented)
   - RMSNorm backward
   - GroupNorm backward

3. **Activation Functions**:
   - ReLU: `grad * (input > 0)`
   - GELU: Use approximation formula
   - SiLU: `grad * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))`

## Priority 4: GPU Optimization

### Days 11-12: Kernel Optimization

**Fused Operations**:
```cuda
// fused_linear_bias_gelu.cu
__global__ void fused_linear_bias_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    // Compute linear + bias + GELU in one kernel
    // Use shared memory for weight tile
    // Minimize global memory accesses
}
```

**Memory Coalescing**:
- Ensure all kernels access memory in coalesced patterns
- Use texture memory for weight matrices
- Implement NHWC format for convolutions

### Days 13-14: Memory Management

**Gradient Buffer Reuse**:
```rust
// In src/memory_pool.rs
pub struct GradientBufferPool {
    buffers: HashMap<(Shape, DType), Vec<CudaBuffer>>,
    device: Arc<CudaDevice>,
}

impl GradientBufferPool {
    pub fn get_buffer(&mut self, shape: &Shape, dtype: DType) -> Result<CudaBuffer> {
        let key = (shape.clone(), dtype);
        if let Some(buffers) = self.buffers.get_mut(&key) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer);
            }
        }
        // Allocate new buffer
        self.device.alloc_zeros(shape.elem_count() * dtype.size())
    }
    
    pub fn return_buffer(&mut self, buffer: CudaBuffer, shape: Shape, dtype: DType) {
        self.buffers.entry((shape, dtype))
            .or_insert_with(Vec::new)
            .push(buffer);
    }
}
```

**Mixed Precision Support**:
```rust
// In src/mixed_precision.rs
pub struct MixedPrecisionConfig {
    pub compute_dtype: DType,  // FP16 or BF16
    pub master_dtype: DType,   // FP32
    pub loss_scale: f32,
}

impl Tensor {
    pub fn to_compute_dtype(&self, config: &MixedPrecisionConfig) -> Result<Tensor> {
        self.to_dtype(config.compute_dtype)
    }
}
```

## Implementation Schedule

| Day | Morning | Afternoon | Goal |
|-----|---------|-----------|------|
| 1 | Fix Arc wrapping | Add imports | Compilation working |
| 2 | Test basic ops | Verify CUDA | All tests compile |
| 3 | Linear backward | Test gradients | Linear training works |
| 4 | Conv2D kernels | Im2col/col2im | Conv forward/backward |
| 5 | Conv2D gradients | Integration | Conv2D complete |
| 6 | Attention forward | Attention backward | Basic attention works |
| 7 | Flash Attention | Causal masks | Attention optimized |
| 8 | Merge autograds | Unified dispatch | Single autograd system |
| 9 | Gradient accumulation | Checkpointing | Advanced autograd |
| 10 | Missing ops | Testing | All ops have backward |
| 11 | Fused kernels | Profiling | 20% speedup |
| 12 | Memory optimization | Buffer reuse | 30% memory reduction |
| 13 | Mixed precision | Loss scaling | FP16 training works |
| 14 | Integration test | SDXL LoRA | End-to-end training |

## Success Metrics

### Compilation Success
- [ ] All 50+ tests compile without errors
- [ ] No Arc double-wrapping issues
- [ ] Clean module imports

### Functional Success
- [ ] Linear layer trains correctly
- [ ] Conv2D produces correct gradients
- [ ] Attention mechanism works with masks
- [ ] All activation functions have gradients

### Performance Success
- [ ] Within 20% of PyTorch performance
- [ ] Memory usage comparable to PyTorch
- [ ] No memory leaks during training
- [ ] Mixed precision training stable

### Integration Success
- [ ] Can train simple CNN on MNIST
- [ ] Can train SDXL LoRA adapter
- [ ] Gradients match PyTorch (rel_error < 1e-5)
- [ ] Training loop example works

## Risk Mitigation

### Technical Risks

1. **Complex CUDA Kernels**
   - Mitigation: Start with simple implementations
   - Use cuDNN/cuBLAS where possible
   - Profile and optimize iteratively

2. **Numerical Stability**
   - Mitigation: Add gradient clipping
   - Use stable formulations (log-sum-exp)
   - Extensive gradient checking

3. **Memory Management**
   - Mitigation: Implement early checkpointing
   - Use gradient accumulation
   - Monitor memory usage closely

### Process Risks

1. **Scope Creep**
   - Mitigation: Stick to priority list
   - Defer optimizations to phase 2
   - Focus on correctness first

2. **Integration Issues**
   - Mitigation: Test each component individually
   - Maintain compatibility layer
   - Regular integration tests

## Validation Plan

### Unit Tests
Each backward implementation must have:
- Gradient check test (finite differences)
- Memory leak test
- Performance benchmark
- Edge case coverage

### Integration Tests
- Simple model training (MLP on synthetic data)
- CNN on CIFAR-10
- SDXL LoRA on small dataset
- Memory profiling under load

### Performance Tests
- Operation throughput (GFLOPS)
- Memory bandwidth utilization
- Comparison with PyTorch
- Scaling with batch size

## Next Steps

1. **Immediate Action**: Fix Arc double-wrapping (Day 1 AM)
2. **First Milestone**: Linear backward working (Day 3)
3. **Second Milestone**: Conv2D complete (Day 5)
4. **Third Milestone**: All ops have backward (Day 10)
5. **Final Milestone**: SDXL LoRA training (Day 14)

## Conclusion

This plan transforms FLAME from a forward-only framework to a complete training system in 14 days. By prioritizing GPU operations and backward passes, we ensure that the most critical functionality for diffusion model training is implemented first. The systematic approach ensures each component is properly tested before moving to the next, reducing integration risks.

Success is defined as being able to train an SDXL LoRA adapter end-to-end with performance comparable to PyTorch.