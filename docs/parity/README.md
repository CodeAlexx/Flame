# PyTorch Parity Test Infrastructure

## Overview

Two files:
- `generate_pytorch_fixtures.py` — Python script that runs ops in PyTorch, saves input/output tensors as `.safetensors`, records timing as JSON
- `pytorch_parity.rs` — Rust test suite that loads the fixtures and compares flame-core's output against PyTorch's oracle

PyTorch is the source of truth. No circular comparisons.

## Setup

```bash
# Generate fixtures (run once, commit the output)
cd /home/alex/EriDiffusion
python generate_pytorch_fixtures.py --output-dir flame-core/tests/pytorch_fixtures/

# Run parity tests
cd flame-core
PYTORCH_FIXTURES=tests/pytorch_fixtures/ cargo test --release --test pytorch_parity

# Run perf comparison (prints table, doesn't assert on timing)
PYTORCH_FIXTURES=tests/pytorch_fixtures/ cargo test --release --test pytorch_parity perf_comparison -- --ignored --nocapture
```

## What's covered

### Per-op (correctness + timing)
| Category | Ops | Shapes | Variants |
|----------|-----|--------|----------|
| Unary | silu, gelu, gelu_exact, relu, sigmoid, tanh, abs, neg, square, exp, log, sqrt, rsqrt, recip | 10 real model shapes | contiguous + permuted |
| Binary | add, sub, mul, div, maximum, minimum | 10 shapes | contiguous + broadcast |
| Scalar | add_scalar, mul_scalar | 10 shapes | contiguous |
| Comparison | gt, ge, lt, le, eq, ne | 10 shapes | contiguous |
| SDPA | scaled_dot_product_attention | 5 attention shapes | contiguous + strided (simulates real DiT Q/K/V permute) |
| Matmul | linear (with/without bias), mm | 5 real model shapes | contiguous |
| Conv2d | conv2d with bias, padding=1 | 4 real model shapes | contiguous |

### Combined patterns (timing)
| Pattern | What it measures |
|---------|-----------------|
| klein_attention_path | QKV proj → split → permute → RoPE → SDPA → permute back → out proj |
| klein_mlp_path | Linear → SwiGLU (split + silu + mul) → Linear |
| dit_modulate_residual | RMSNorm → modulate (scale + shift) → gate + residual add |
| klein_double_block | Full attention + MLP + modulate + residual |
| permute_narrow_linear | Permute → narrow → permute back → linear (the view chain) |
| sdpa_backward | Forward + backward through SDPA (training path) |

### Fused op comparison
| Op | PyTorch (unfused) | flame-core (fused) |
|----|-------------------|--------------------|
| RMSNorm | variance → rsqrt → mul → mul (4 kernels) | fused_rms_norm_bf16 (1 kernel) |
| SwiGLU | chunk → silu → mul (3 kernels) | swiglu_fused_bf16 (1 kernel) |

These show where flame-core should be *faster* than PyTorch.

## Output structure

```
pytorch_fixtures/
  environment.json          # PyTorch version, CUDA, device, etc.
  timing_results.json       # All timing data in one file
  unary/
    silu/
      klein_hidden.safetensors
      klein_hidden_permuted.safetensors
      sdxl_hidden.safetensors
      ...
    gelu/
      ...
  binary/
    add/
      klein_hidden.safetensors
      klein_hidden_broadcast.safetensors
      ...
  scalar/
    add_scalar/
      ...
  comparison/
    gt/
      ...
  sdpa/
    klein_sdpa_contig.safetensors
    klein_sdpa_strided.safetensors
    ...
  matmul/
    klein_linear.safetensors
    klein_linear_nobias.safetensors
    ...
  conv2d/
    sdxl_conv_entry.safetensors
    ...
  patterns/
    klein_attention_path.safetensors
    klein_mlp_path.safetensors
    dit_modulate_residual.safetensors
    klein_double_block.safetensors
    permute_narrow_linear.safetensors
    sdpa_backward.safetensors
  fused/
    rmsnorm_klein_hidden.safetensors
    swiglu_klein_hidden.safetensors
    ...
```

## Adapting the Rust test

The Rust test has two stubs that need flame-core-specific implementation:

1. `load_safetensors_to_device()` — Load safetensors bytes into GPU Tensors. Replace the `todo!()` with flame-core's actual safetensors loading API.

2. `perf_comparison` test's per-op timing dispatch — Map op name strings to actual flame-core calls for timing. Currently returns 0.0 for flame-core timing.

## Regenerating fixtures

When PyTorch updates or you want different shapes:

```bash
# Regenerate everything
python generate_pytorch_fixtures.py --output-dir flame-core/tests/pytorch_fixtures/

# Regenerate specific ops only
python generate_pytorch_fixtures.py --output-dir flame-core/tests/pytorch_fixtures/ --op silu,gelu,add

# Regenerate patterns only
python generate_pytorch_fixtures.py --output-dir flame-core/tests/pytorch_fixtures/ --patterns-only
```

Commit the regenerated fixtures with a note about which PyTorch version produced them (recorded in `environment.json`).

## When a test fails

The test prints cos_sim and max_abs_diff. Common causes:

- **Bit-exact failure, cos_sim ≥ 0.9999**: BF16 reduction-order difference. Probably fine. Check if the max_abs_diff is ≤ 1 ULP of BF16.
- **cos_sim < 0.9999**: Real bug. The op is computing something different from PyTorch. Check: is the op using F32 round-trip when it should be native BF16? Is the op reading storage-linear instead of respecting strides?
- **cos_sim < 0.99**: Seriously wrong. Wrong formula, wrong indexing, or reading garbage memory.
