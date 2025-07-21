# FLAME Architecture Violations Report

## Critical Issue: Model-Specific Code in Framework

The following files violate the architectural separation and must be moved to EriDiffusion:

### Files to Move

1. **`sdxl_unet_blocks.rs`**
   - Contains SDXL-specific ResNet blocks
   - Should be in EriDiffusion under `models/sdxl/`

2. **`sdxl_attention.rs`**
   - Contains SDXL-specific cross-attention
   - Should be in EriDiffusion under `models/sdxl/`

3. **`mmdit_blocks.rs`**
   - Contains SD3.5 MMDiT transformer blocks
   - Should be in EriDiffusion under `models/sd35/`

4. **`flux_blocks.rs`** (if exists)
   - Contains Flux-specific blocks
   - Should be in EriDiffusion under `models/flux/`

5. **`tokenizer.rs`**
   - Text tokenization is application-specific
   - Should be in EriDiffusion under `data/`

6. **`modulated_blocks.rs`**
   - Contains AdaLayerNorm and other model-specific normalization
   - Should be in EriDiffusion under `models/`

### What Should Stay in FLAME

✅ **Generic Components** (these are correct):
- `tensor.rs` - Core tensor operations
- `autograd.rs` - Automatic differentiation engine
- `conv.rs` - Generic Conv2d layer
- `linear.rs` - Generic Linear/Dense layer
- `attention.rs` - Generic multi-head attention mechanism
- `norm.rs` - Generic normalization (LayerNorm, GroupNorm, RMSNorm)
- `optimizers.rs` - Adam, SGD, etc.
- `cuda_*` - CUDA kernels and operations
- `loss.rs` - Generic loss functions
- `activations.rs` - ReLU, GELU, SiLU, etc.

### Why This Matters

1. **Reusability**: FLAME should be usable for ANY deep learning task, not just diffusion models
2. **Maintenance**: Model-specific code changes frequently, framework code should be stable
3. **Dependencies**: FLAME shouldn't need tokenizers or know about specific model architectures
4. **Testing**: Framework tests should be mathematical, not model-specific

### Immediate Action Required

These files must be moved to EriDiffusion to maintain proper architectural separation. FLAME should remain a pure tensor computation framework like PyTorch's core, not a model zoo.

## Summary

**Current State**: FLAME contains ~8 files with model-specific code
**Required State**: FLAME should contain 0 model-specific files
**Impact**: This violation prevents FLAME from being a truly generic framework