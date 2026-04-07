# flame-core CUDA kernel catalog

> Every CUDA kernel in flame-core, grouped by file, with one-line descriptions
> and (where known) layout assumptions and perf notes. Kernels are split into
> two pipelines:
>
> 1. **NVRTC kernels** — string consts in `.rs` files, compiled at runtime
>    via `cudarc::nvrtc::compile_ptx_with_opts`. Usually short, single-purpose,
>    and the easiest to add. The "fast path" for inference primitives.
> 2. **Build-time `.cu` kernels** — `.cu` files in `cuda/` and `src/cuda/`,
>    compiled by `build.rs` via `cc-rs/nvcc` into a static lib. Heavier
>    kernels (cuBLASLt wrappers, flash attention, conv2d). Two naming
>    conventions: `flame_*` (returns int status) and `fc_*` (returns
>    `fc_status_t`).
>
> See [`FLAME_CONVENTIONS.md`](./FLAME_CONVENTIONS.md) for "how to add a new
> kernel" templates.

---

## NVRTC kernels (runtime-compiled, in `.rs` files)

These are the inline `const CUDA_*: &str` blocks compiled at runtime. Each
kernel has a one-shot launcher `pub fn` in the same module that handles the
ensure / get_func / launch dance.

### `bf16_elementwise.rs` — broadcast + flat-path elementwise

| Kernel | Line | Purpose / notes |
|---|---|---|
| `add_bf16_flat_kernel` | `:97` | `__hadd2` vectorized BF16 add — flat fast path when shapes match. 2 elements/thread. |
| `mul_bf16_flat_kernel` | `:117` | `__hmul2` |
| `sub_bf16_flat_kernel` | `:136` | `__hsub2` |
| `div_bf16_flat_kernel` | `:155` | `__h2div` |
| `add_bf16_kernel` | `:201` | Generic 8-D broadcast path (slow, fallback). |
| `mul_bf16_kernel` | `:210` | Same broadcast path |
| `div_bf16_kernel` | `:219` | |
| `max_bf16_kernel` | `:228` | |
| `min_bf16_kernel` | `:237` | |
| `transpose2d_bf16_kernel` | `:252` | 2D BF16 transpose. Used by Klein/Mistral pre-transpose. |
| `cmp_bf16_kernel` | `:269` | Comparison ops returning u8 (ge/gt/le/lt/ne). |
| `softmax_lastdim_bf16_kernel` | `:414` | **Fused last-dim softmax** — one block per row, warp-cooperative max+sum reductions, no scratch tensor. Wired into `Tensor::softmax` BF16 fast path. Was 175× slower as a 5-step pipeline before this kernel landed. |
| `patchify_bf16_kernel` | `:789` | DiT patchify (raster → 2x2 patches → seq). |
| `unpatchify_bf16_kernel` | `:828` | Inverse. |

**Launcher convention**: `lc(n)` for 1-element-per-thread kernels,
`lc_pairs(n) = (n+1)/2` for the vectorized 2-element kernels.

### `bf16_ops.rs` — single-arg + RoPE + fused inference primitives

| Kernel | Line | Purpose / notes |
|---|---|---|
| `gelu_bf16_kernel` | `:22` | tanh-approx GELU. Vectorized with `__nv_bfloat162` (2 elements/thread). |
| `silu_bf16_kernel` | `:49` | SiLU = `x / (1 + exp(-x))`. Same vectorization. |
| `square_bf16_kernel` | `:73` | Element square. Vectorized. |
| `softmax_last_dim_bf16_kernel` | `:195` | Older fused softmax (one block per row). The 2026-04 `softmax_lastdim_bf16_kernel` in `bf16_elementwise.rs` is the preferred entry but this still exists and is called by `softmax_last_dim_bf16` pub fn. |
| `rope_fused_bf16_kernel` | `:343` | **Interleaved-pair RoPE** — `out[2i] = x[2i]*cos[i] - x[2i+1]*sin[i]`. Used by FLUX, Klein, LTX, Hunyuan, QwenImage, Chroma. |
| `rope_halfsplit_bf16_kernel` | `:376` | Halfsplit RoPE — first/second half rotation. Used by Z-Image, some Klein variants. |
| `modulate_pre_bf16_kernel` | `:580` | DiT modulate `(1 + scale) * x + shift`. |
| `gate_residual_bf16_kernel` | `:699` | `out = x + gate * attn_out`. |
| `swiglu_fused_bf16_kernel` | `:776` | `silu(gate) * up`. |

### `bf16_convert.rs` — BF16↔F32 cast

| Kernel | Line | Purpose / notes |
|---|---|---|
| `bf16_to_f32` | `:14` | `__bfloat1622float2` — 2 elements/thread vectorized. |
| `f32_to_bf16` | `:33` | `__floats2bfloat162_rn` — 2 elements/thread. |

### `bf16_normal.rs` / `bf16_factories.rs` / `bf16_clamp.rs` — RNG / factories

| Kernel | File:line | Purpose |
|---|---|---|
| `normal_bf16_kernel` | `bf16_normal.rs:19` | Box-Muller Gaussian, BF16 output |
| `uniform_bf16_kernel` | `bf16_factories.rs:48` | Uniform random BF16 |
| `clamp_bf16_kernel` | `bf16_clamp.rs:18` | Element clamp `[lo, hi]` |

### `conv3d_bf16.rs` — 3D conv

| Kernel | Line | Purpose |
|---|---|---|
| `im2vol_bf16` | `:35` | im2col-equivalent for 3D conv (im2vol). Builds the column matrix from `[N, C, D, H, W]` input. |
| `bias_add_bf16_conv3d` | `:106` | Per-channel bias add after the GEMM. |
| `copy_bf16` | `:132` | Helper memcopy (used for non-contiguous output paths). |

`Conv3dBF16::forward` does: `im2vol` → cuBLASLt GEMM → `bias_add` → optional `copy`.

### `rng/mod.rs` — F32 RNG

| Kernel | Line | Purpose |
|---|---|---|
| `fill_rand_f32` | `:18` | Per-thread Philox-style F32 random fill. Used by `Tensor::randn` for the F32 path. |

### `sgd/mod.rs` — F32 SGD step

| Kernel | Line | Purpose |
|---|---|---|
| `sgd_f32` | `:13` | `p -= lr * g`. Used by the F32 training SGD. |

### `cuda_kernels_gpu.rs` — F32 framework kernels (training)

Two notable broadcast kernels:

| Kernel | Line | Purpose |
|---|---|---|
| `mul_bc_kernel` | `:2421` | F32 broadcast multiply |
| `add_bc_kernel` | `:2548` | F32 broadcast add |

The full set of F32 NVRTC kernels in `cuda_kernels.rs` and
`cuda_kernels_gpu.rs` (~100+) are training-only — see those files directly.

---

## Build-time `.cu` kernels (compiled by `build.rs`)

`build.rs` lists the source files (search for `cuda_sources.push`). The two
locations are:
- **`cuda/`** at the repo root — older surface, `fc_*` symbols (status enum)
- **`src/cuda/`** — newer fused inference kernels, `flame_*` symbols (int)
- **`src/kernels/`** — additional `.cu` files
- **`kernels/`** at repo root — duplicates of `src/kernels/` (some legacy
  copies; check `build.rs` for which is actually compiled)

### `src/cuda/flash_attention_fwd.cu` — wmma flash attention (LIVE)

The single most-important file in this directory.

| Symbol | Line | Notes |
|---|---|---|
| `flash_attn_fwd_hd64 / hd96 / hd128` | `:95` (macro-generated) | wmma BF16 attention forward. Compile-time specialized per head_dim. Uses `nvcuda::wmma` 16x16x16 BF16 fragments + FP32 accumulation + online softmax. **SM_80+** required (3090 / Ada / Hopper). |
| `flame_flash_attention_bf16` | `:349` | C entry point: `(Q, K, V, O, batch_heads, seq_q, seq_kv, head_dim, stream)` — same signature as the legacy scalar version. |

**Tile sizes**: `BQ=32, BKV=64, NUM_WARPS=8` (was `BQ=64, NUM_WARPS=16` in the
original FA2; halved on SM_86 because of the 100 KB shared-mem opt-in limit).
Stage 3 (P@V) uses one warp per (qi_base, hd_base) tile — **no atomicAdd**
to s_O (the warp tile map proves zero collision).

**Stage 2 softmax**: warp-cooperative — one warp per row, lanes do `__shfl_xor_sync`
max+sum reductions across BKV. Was scalar (224/256 threads idle) before 2026-04.

**`load_tile_bf16`** (line 59): pads K/V to **buffer dimension** (BKV=64), not
valid_rows. Critical: an earlier dead-code zero-pad branch left padding rows
uninitialized, which made the kernel run-to-run nondeterministic. Fixed
2026-04 — pad to `buf_rows` is now mandatory.

### `src/cuda/fused_linear3d.cu` — cuBLASLt 3D linear (LIVE)

| Symbol | Line | Notes |
|---|---|---|
| `flame_linear3d_bf16` | `:24` | cuBLASLt BF16 matmul + bias epilogue. Weight is `[Cin, Cout]` row-major (pre-transposed). Used by Klein. |
| `flame_linear3d_bf16_native` | `:135` | Same but takes weight in standard PyTorch `[Cout, Cin]` row-major layout. Uses `TRANSA=T` so the transpose happens inside the GEMM. **This is what every FLUX/Chroma/QwenImage block forward calls.** Added 2026-04. |

Both use `CUBLAS_COMPUTE_32F` accumulation, BF16 inputs/outputs, and the
`CUBLASLT_EPILOGUE_BIAS` epilogue (so the bias add is fused into the GEMM —
no separate add kernel).

### `src/cuda/fused_rms_norm.cu` — fused RMSNorm

| Symbol | Line | Notes |
|---|---|---|
| `fused_rms_norm_bf16` (kernel) | `:26` | One block per row; sum-of-squares + rsqrt + scale, single kernel. |
| `flame_fused_rms_norm_bf16` (entry) | `:89` | C entry. Used by `ops::fused_inference::fused_rms_norm`. |

### `src/cuda/fused_norm_modulate.cu` — fused RMSNorm + modulate

| Symbol | Line | Notes |
|---|---|---|
| `fused_rms_norm_modulate_bf16_kernel` | `:19` | RMSNorm followed by `(1+scale) * x + shift` in one kernel. Saves a roundtrip vs calling them separately. |
| `flame_fused_rms_norm_modulate_bf16` | `:75` | C entry. |

### `src/cuda/fused_modulate.cu` — modulate alone

| Symbol | Line | Notes |
|---|---|---|
| `fused_modulate_bf16` | `:19` | Single-element/thread modulate. |
| `fused_modulate_bf16_vec2` | `:39` | Vectorized 2-element/thread variant. |
| `flame_fused_modulate_bf16` | `:70` | C entry. |

### `src/cuda/fused_residual_gate.cu` — gated residual

| Symbol | Line | Notes |
|---|---|---|
| `fused_residual_gate_bf16_kernel` | `:10` | `out = x + gate * attn_out` in one kernel. |
| `flame_fused_residual_gate_bf16` | `:28` | C entry. |

### `src/cuda/fp8_dequant.cu` — FP8 → BF16

| Symbol | Line | Notes |
|---|---|---|
| `fp8_to_bf16_kernel` | `:10` | E4M3 / E5M2 unpack. Used by FlameSwap FP8 paths. |
| `flame_fp8_to_bf16` | `:40` | C entry. |

### `src/cuda/fused_dequant_transpose.cu` — FP8 dequant + transpose fused

| Symbol | Line | Notes |
|---|---|---|
| `fp8_dequant_transpose_kernel` | `:17` | Dequant + transpose in one kernel. Used by `fp8_resident.rs` for the on-the-fly weight unpack path. |
| `flame_fused_dequant_transpose_bf16` | `:93` | C entry. |

### `src/cuda/narrow_strided.cu` / `narrow_strided_backward.cu`

| Symbol | Line | Notes |
|---|---|---|
| `flame_narrow_strided_launch` | `:58` | Generic narrow op with stride support. |
| `flame_narrow_backward_scatter_add_launch` | various | Scatter-add backward for narrow. ⚠️ Training only. |

### `src/cuda/pinned_host.cu` — pinned memory + async copy

| Symbol | Line | Notes |
|---|---|---|
| `flame_cuda_alloc_pinned_host(size, flags)` | top | Allocate pinned host buffer. |
| `flame_cuda_free_pinned_host(ptr)` | `:10` | |
| `flame_cuda_memcpy_async(dst, src, size, kind, stream)` | `:14` | |
| `flame_cuda_host_register(ptr, size, flags)` | `:33` | Register existing host memory as pinned. |
| `flame_cuda_host_unregister(ptr)` | `:42` | |

### `src/cuda/kernels.cu` — early F32 kernels (training)

| Symbol | Line | Purpose |
|---|---|---|
| `update_weights_f32` | `:4` | F32 SGD step (replaced by `sgd/mod.rs` NVRTC kernel) |
| `add_f32 / mul_f32 / mul_scalar_f32 / relu_f32 / relu_backward_f32 / mse_loss_f32 / mse_backward_f32 / fill_f32 / copy_f32` | `:17-141` | F32 ops, training-only. ⚠️ |

### `src/kernels/sdpa_kernels.cu` — SDPA chunk primitives

These are the chunked SDPA building blocks used by `sdpa_stream_bf16` and the
older non-flash path. Used by training and the LTX-2 d=64 audio attention
fallback.

| Symbol | Line | Purpose |
|---|---|---|
| `causal_mask_kernel` | `:34` | Apply causal mask to FP32 score tile. |
| `attn_mask_kernel` | `:63` | Apply additive bias mask. |
| `add_mask_tile_fp32_kernel` | `:81` | Tile-level bias add. |
| `softmax_from_lse_tile_kernel` | `:139` | Online softmax: write `exp(s - lse)` per row. |
| `lse_from_logits_tile_kernel` | `:159` | Compute LSE for an incoming tile. |
| `lse_merge_rows_kernel` | `:201` | Merge two LSE values + per-row scales. |
| `dropout_bf16_inplace_kernel` | `:217` | Inverted-dropout for training. |
| `flame_apply_causal_mask_fp32 / attn_mask_fp32 / sdpa_add_mask_tile_fp32 / sdpa_softmax_from_lse_tile / sdpa_lse_from_logits_tile / sdpa_lse_merge_rows / sdpa_dropout_bf16_inplace` | `:246-384` | C entries. |

### `src/kernels/rope_kernels.cu`

| Symbol | Line | Purpose |
|---|---|---|
| `rope_apply_kernel` | `:15` | F32 RoPE apply (legacy path). |
| `rope_copy_tail_kernel` | `:64` | Tail copy when RoPE dim < total head_dim. |
| `flame_rope_apply_bf16_fp32` | `:87` | C entry. |

### `src/kernels/geglu_kernels.cu`

| Symbol | Line | Purpose |
|---|---|---|
| `geglu_kernel` | `:14` | F32 GeGLU `gelu(gate) * up`. |
| `flame_geglu_pointwise_fp32` | `:30` | C entry. |

### `src/kernels/mul_bwd_bf16.cu` — BF16 mul backward
- Single-purpose backward for the BF16 mul op.

### `cuda/cuda_ops.cu` — `fc_*` BF16 ops surface

This is the largest single `.cu` file. All the `fc_*` BF16 op entries live here.

| Kernel | Line | Notes |
|---|---|---|
| `relu_kernel` | `:109` | Vectorized BF16 ReLU (2-elem/thread). |
| `silu_kernel` | `:128` | Vectorized BF16 SiLU. **Two implementations** of silu exist — this `fc_silu_bf16` is one, and `bf16_ops::silu_bf16` is the other. `Tensor::silu` calls the latter. |
| `gelu_kernel` | `:147` | Vectorized BF16 GELU. Same caveat. |
| `axpby_kernel` | `:169` | `y = a*x + b*y` |
| `rms_norm_kernel` | `:253` | **Block-per-row + parallel reduction RMSNorm**. Was 1-thread-per-row scalar before 2026-04. Wraps `fc_rms_norm_bf16` C entry. |
| `layer_norm_forward_bf16_kernel` | `:295` | LayerNorm forward (with optional gamma/beta). |
| `group_norm_compute_stats_bf16_kernel` | `:361` | GroupNorm 1st pass (mean/var per group). |
| `group_norm_forward_bf16_kernel` | `:418` | GroupNorm 2nd pass (apply). |
| `rms_norm_bf16_to_f32_kernel` | `:468` | RMSNorm with F32 output (for mixed-precision callers). |
| `fc_relu_bf16 / fc_gelu_bf16 / fc_silu_bf16` | `:206-214` | C entries |
| `fc_axpby_bf16` | `:218` | |
| `fc_rms_norm_bf16_to_f32` | `:489` | |
| `fc_rms_norm_bf16` | `:514` | |
| `fc_layer_norm_bf16` | `:547` | |
| `fc_group_norm_bf16` | `:659` | |

### `cuda/cuda_ops_common.cu` — workspace + memcpy

| Symbol | Line | Notes |
|---|---|---|
| `fc_ws_ensure_capacity` | `:42` | Workspace arena grow. |
| `fc_bf16_memcpy_async` | `:87` | Async BF16 memcpy. |

### `cuda/gemm_bf16_cublaslt.cu` — cuBLASLt BF16 GEMMs

| Symbol | Line | Notes |
|---|---|---|
| `fc_gemm_bf16` | `:357` | Standard 2D GEMM, BF16 in/out, FP32 accumulate, optional bias epilogue. |
| `fc_batched_gemm_bf16` | `:518` | Strided batched variant. |

### `cuda/gemm_bf16_fp32acc.cu`
Helper utilities for the BF16+FP32 accumulation path (not directly callable
from Rust; included in the `gemm_bf16_cublaslt.cu` translation unit).

### `cuda/conv2d_nhwc_bf16.cu` — BF16 conv2d (im2col + GEMM)

| Symbol | Line | Notes |
|---|---|---|
| `im2col_bf16_kernel` | `:20` | NHWC im2col |
| `fc_conv2d_bf16` | `:149` | C entry — top-level conv2d. |

### `cuda/repeat_bf16.cu`

| Symbol | Line | Notes |
|---|---|---|
| `repeat_nd_kernel` | `:37` | N-D repeat (broadcast-like) for BF16. |
| `repeat_nhwc_kernel` | `:64` | NHWC fast path. |
| `fc_bf16_repeat_nd` | `:124` | C entry. |

### `cuda/bf16_broadcast_repeat.cu`

| Symbol | Line | Notes |
|---|---|---|
| `broadcast_kernel` | `:61` | BF16 strided broadcast. |
| `repeat_axis_kernel` | `:85` | Single-axis repeat. |
| `fc_bf16_broadcast / fc_bf16_repeat_axis` | `:121, :178` | C entries. |

### `cuda/bf16_slice_index.cu`

| Symbol | Line | Notes |
|---|---|---|
| `slice_copy_kernel` | `:49` | Strided slice copy. |
| `index_select_kernel` | `:76` | Gather along axis 0. |
| `fc_bf16_slice / fc_bf16_index_select` | `:140, :196` | C entries. |

### `cuda/upsample_nearest.cu`

| Symbol | Line | Notes |
|---|---|---|
| `upsample2d_nearest_nchw_kernel` | `:17` | Nearest upsample, NCHW BF16. |
| `fc_upsample2d_nearest_bf16` | `:52` | BF16 entry |
| `fc_upsample2d_nearest_f32` | `:87` | F32 entry |

### `cuda/permute0213.cu`

| Symbol | Line | Notes |
|---|---|---|
| `permute0213_kernel` | `:15` | `[B, N, H, D] → [B, H, N, D]` — the attention reshape permute. |
| `permute021_kernel` | `:81` | `[B, M, N] → [B, N, M]`. |

### `cuda/sdpa_stream_bf16.cu` — chunked SDPA (legacy)

The streaming SDPA path used by `sdpa_stream_bf16`. ⚠️ This is the
catastrophically slow path for d=64 and causal — see PERF_SDPA_FLASH_KERNEL.md.

| Symbol | Line | Notes |
|---|---|---|
| `ker_cast_bf16_to_f32` | `:42` | Per-tile BF16 → FP32 cast. |
| `ker_cast_f32_to_bf16` | `:47` | Inverse. |
| `ker_apply_scale_mask_rowmax` | `:54` | Apply scale, mask, compute row max. |
| `ker_row_exp_and_sum` | `:93` | Compute exp + row sum. |
| `ker_update_l_and_factors` | `:109` | Update l_new and the per-row scale factors. |
| `ker_row_normalize` | `:137` | Normalize scores by 1/l. |
| `ker_scale_O_by_beta` | `:149` | Scale running O by beta. |
| `ker_fill_constant` | `:157` | Fill helper. |

### `cuda/streaming_attn_bf16.cu`
Older streaming attention scaffolding.

### `cuda/reduce_sum_bf16.cu`

| Symbol | Line | Notes |
|---|---|---|
| `sum_last_keepdim_bf16_kernel` | `:23` | Sum over last dim, keepdim. |

### `cuda/add_inplace.cu`

| Symbol | Line | Notes |
|---|---|---|
| `inplace_binary_kernel` | `:6` | Generic inplace binary. |
| `scalar_transform_kernel` | `:120` | Inplace scalar transform. |

### `cuda/add_same_shape.cu`

| Symbol | Line | Notes |
|---|---|---|
| `add_same_shape_kernel` | `:5` | Templated same-shape add (T type param). |

### `cuda/broadcast.cu`

| Symbol | Line | Notes |
|---|---|---|
| `broadcast_strided_kernel` | `:6` | Strided broadcast. |

### `cuda/tile_bc.cu`

| Symbol | Line | Notes |
|---|---|---|
| `tile_bc_to_bhwc_kernel` | `:4` | Tile/broadcast to BHWC layout. |

### `cuda/modulate_affine_bf16.cu`

| Symbol | Line | Notes |
|---|---|---|
| `modulate_affine_bf16_kernel` | `:29` | DiT modulate (alternative path to the `bf16_ops` version). |

### `cuda/gate_mul_bf16.cu`

| Symbol | Line | Notes |
|---|---|---|
| `gate_mul_bf16_kernel` | `:25` | Element gate × value. |

### `cuda/src/flame_bf16_utils.cu`

| Symbol | Line | Notes |
|---|---|---|
| `flame_k_zero_bf16` | `:5` | Zero-fill BF16 buffer. |
| `flame_k_copy_bf16` | `:12` | Copy BF16 buffer. |

### `cuda/src/flame_nhwc_adapters.cu`

| Symbol | Line | Notes |
|---|---|---|
| `flame_k_nhwc_to_nchw` | `:6` | Layout conversion. |
| `flame_k_nchw_to_nhwc` | `:26` | Inverse. |

### `cuda/src/flame_norm_bf16.cu` — norm backward

| Symbol | Line | Notes |
|---|---|---|
| `layer_norm_backward_kernel` | `:26` | LayerNorm backward (training). |
| `group_norm_backward_kernel` | `:82` | GroupNorm backward. |
| `fc_layer_norm_backward_bf16` | `:172` | C entry. |
| `fc_group_norm_backward_bf16` | `:203` | C entry. |

### `cuda/src/flame_conv2d_stub.cu` — extra conv2d helpers
- `depthwise_conv2d_bf16_kernel`, `apply_activation_kernel`,
  `bf16_matmul_bias_kernel`, `im2col_bf16_tile` — assorted helpers used by
  the BF16 conv2d path

### `cuda/src/flame_sdpa_stub.cu` — extra SDPA helpers (training)
- `qk_matmul_bf16_kernel`, `sdpa_reset_kernel`, `sdpa_block_accumulate_kernel`,
  `sdpa_finalize_kernel` — building blocks for the older SDPA training path

### `kernels/adaln_layernorm_bf16.cu`

| Symbol | Line | Notes |
|---|---|---|
| `layernorm_affine_bf16_nhwc_kernel` | `:9` | NHWC AdaLN-style LayerNorm with gamma/beta. |

---

## Perf-critical kernels — known characteristics

### Hot path on Z-Image / FLUX / Chroma / QwenImage at 1024² (~per call)

| Kernel | Per-call time | Notes |
|---|---|---|
| `flash_attn_fwd_hd128` | ~39 ms | After 2026-04 wmma + warp-coop softmax fixes. PyTorch is ~3.4 ms (cuDNN flash-attn-2) — still 11× gap. |
| `rms_norm_kernel` (`cuda_ops.cu`) | ~0.04–0.24 ms | Beats PyTorch. Block-per-row + parallel reduction. |
| `fc_layer_norm_bf16` | ~0.02–0.10 ms | Comparable to PyTorch. |
| `fused_linear3d_bf16_native` | ~5–13 ms | Within 1.4–1.7× of cuBLASLt peak. |
| `add_bf16_flat / mul_bf16_flat` | ~0.87 ms | 5–6× faster than the broadcast path. Still 8× PyTorch (kernel launch floor). |
| `softmax_lastdim_bf16_kernel` | ~1 ms (small) / 30 ms (large) | 175× faster than the previous 5-step pipeline. |
| `silu_bf16_kernel / gelu_bf16_kernel` | ~3.5 ms | Memory-bandwidth bound. ~12× PyTorch — kernel launch overhead floor. |

### Catastrophically slow (still need fixes)

| Kernel | Per-call time | Notes |
|---|---|---|
| `sdpa_stream_bf16` (causal d=64) | 110-215 ms | Blocks LTX-2 / Wan / HunyuanVideo temporal attention. Needs wmma + causal mask path. |
| `sdpa_stream_bf16` (with mask, d=64) | ~9 ms | T5 path. Same wmma fix would help. |
| `bf16_to_f32 / f32_to_bf16` | ~3.5 ms | Vectorized 2-elem/thread. 29-40× PyTorch — kernel launch floor. |

---

## Adding a new kernel — quick template

### NVRTC kernel (preferred for new BF16 inference primitives)

In `src/bf16_ops.rs` (or wherever fits):

```rust
const CUDA_MY_KERNEL: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void my_kernel_bf16(const __nv_bfloat16* X, __nv_bfloat16* Y, long n) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // ... your kernel ...
    }
}
"#;

pub fn my_op_bf16(x: &Tensor) -> Result<Tensor> {
    let n = x.shape().elem_count();
    // alloc output ...
    ensure(&x.device, "my_kernel_bf16", CUDA_MY_KERNEL)?;
    let f = x.device.get_func("my_kernel_bf16", "my_kernel_bf16")
        .ok_or_else(|| Error::Cuda("missing".into()))?;
    unsafe { f.launch(lc(n), (slice_ref(xs), ys, n as i64))?; }
    Ok(out)
}
```

For 2-element-per-thread kernels use `lc_pairs(n)` instead of `lc(n)`.

### Build-time `.cu` kernel (for cuBLASLt wrappers, larger kernels)

1. Create `src/cuda/my_kernel.cu`
2. Add `cuda_sources.push("src/cuda/my_kernel.cu");` in `build.rs`
3. Declare the C entry in `src/cuda/ffi.rs`:
   ```rust
   pub fn flame_my_kernel_bf16(...) -> i32;
   ```
4. Write the Rust wrapper in `src/ops/fused_inference.rs` or wherever fits.
5. The build script auto-rebuilds when the `.cu` file changes mtime, but you
   can `touch src/cuda/my_kernel.cu` to force.
