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
| `abs_bf16_kernel` | `:559` | BF16 abs via sign-bit clear (`x & 0x7FFF`). Replaces `square().sqrt()` decomposition that was 8.4× slower. |
| `softmax_lastdim_bf16_kernel` | `:472` | **Fused last-dim softmax** — 2-pass online softmax (Milakov & Gimelshein) with warp-shuffle reductions. Single block per row, no scratch tensor. 1.5× PyTorch (kernel is 147μs, rest is pool overhead). Launcher `softmax_lastdim_bf16` (`:152`) sizes `block_size` to the nearest multiple of 32 that covers `cols` (32/64/128/256, capped at 256) instead of the legacy fixed-128 — saves launch overhead for head_dim=64 rows (2026-05-12). Tail threads still no-op via `if (tid < cols)` so correctness holds. |
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
| `rope_halfsplit_bf16_kernel` | `:376` | Halfsplit RoPE — first/second half rotation. Used by Z-Image, some Klein variants, MagiHuman (via partial-rotation wrapper — see gotcha). |
| `swiglu_fused_bf16_vec2_kernel` | `:1819` | **Vectorized SwiGLU** — `__nv_bfloat162` pair loads, 2 elements/thread, F32 sigmoid. Selected by `swiglu_fused_bf16` when `total % 2 == 0` (CUDA pool buffers are 256-byte aligned so 2-elem alignment holds). Scalar tail handled by thread (0,0) when `total` is odd. `FLAME_SWIGLU_LEGACY=1` forces the scalar `swiglu_fused_bf16_kernel`. Added 2026-05-12. |

> **Gotcha (rope_halfsplit_bf16 / rope_fused_bf16, partial rotation):** both kernels rotate the *full* last dim of `x` (compute `half = d / 2` from `x.shape[-1]`). Models that rotate only a prefix of `head_dim` (e.g. MagiHuman: head_dim=128, ROPE_DIM=96, last 32 channels are passthrough) must split→rotate→cat manually: `narrow(3, 0, ROPE_DIM).contiguous() → rope_halfsplit_bf16 → cat with narrow(3, ROPE_DIM, head_dim - ROPE_DIM)`. Symptom if you don't: `Shape mismatch: expected [..., D/2_from_x], got [..., ROPE_DIM/2]` from the cos/sin reshape inside the kernel. See `inference-flame/src/models/magihuman_dit.rs::rope_partial_halfsplit` for the wrapper pattern.
| `modulate_pre_bf16_kernel` | `:580` | DiT modulate `(1 + scale) * x + shift`. |
| `gate_residual_bf16_kernel` | `:699` | `out = x + gate * attn_out`. |
| `swiglu_fused_bf16_kernel` | `:776` | `silu(gate) * up`. |
| `bf16_stoch_round_kernel` | end-of-file (~`:2700`) | **Stochastic-round F32→BF16**. One thread/element. Reads `src[i]` (F32) + `rng[i]` (u32); keeps high-16 bits, increments by 1 with probability `(src[i] & 0xFFFF) / 2^16` driven by the low-16 of `rng[i]`. Launched via `lc(n)` (1 thread/element). Standalone kernel — used for ad-hoc cast paths (e.g. final F32-master → BF16 store at save time). AdamW's BF16 stochastic-round path uses the inline-hash variants `adam_fused_bf16_f32grad_stoch_kernel` / `adam_fused_multi_bf16_f32grad_stoch_kernel` instead of round-tripping through this kernel. CPU reference in `bf16_convert::stochastic_round_to_bf16_cpu`. |

### `ops/deinterleave.rs` — last-dim pair deinterleave

| Kernel | Line | Purpose / notes |
|---|---|---|
| `deinterleave_pair_f32_kernel` | `:25` | **Last-dim deinterleave**, vectorized via `float2` reinterpret_cast. Splits `[..., 2K]` F32 into two `[..., K]` halves (even/odd columns). Used by interleaved-SwiGLU MLPs to skip the stride-2 generic gather (`materialize_strided_f32_kernel` was ~1.35 s for 18 M elements vs ~0.5 ms here on a 3090 Ti) — the host-side path went `reshape→narrow(stride 2)→.contiguous()` and lost on coalescing; this kernel does one vectorized 8-byte read + two 4-byte writes per thread. |

### `bf16_reduce.rs` — BF16-native scalar reductions (added 2026-05-12)

| Kernel | Line | Purpose / notes |
|---|---|---|
| `sum_bf16_to_f32_scalar_kernel` | `:50` | **BF16-native sum reduce.** Per-thread grid-stride F32 accumulator over BF16 input, shared-mem tree reduce per block, `atomicAdd` of block result into a single F32 scratch scalar. Block=256, grid capped at 4096 with grid-stride loop covering arbitrary `n`. Replaces the legacy F32 `sum_kernel` for BF16 inputs (Foundation fix #B), eliminating the BF16→F32 cast + F32-reduce + F32→BF16 cast triple pass. Note: legacy F32 `sum_kernel` (`cuda_kernel_sources.rs:260`) hard-caps grid at 1024 blocks and silently drops elements past `1024 * 256 = 262144` — this kernel's grid-stride loop is correct for any size. |
| `f32_scalar_to_bf16_kernel` | `:80` | 1-thread cast of an F32 scalar to BF16, with a fused `* scale` multiply so `mean_bf16` can pass `1/n` and keep the entire reduction on a single CUDA stream (no host D2H sync). |

### `bf16_convert.rs` — BF16↔F32 cast

| Kernel | Line | Purpose / notes |
|---|---|---|
| `bf16_to_f32` | `:14` | `__bfloat1622float2` — 2 elements/thread vectorized. |
| `f32_to_bf16` | `:33` | `__floats2bfloat162_rn` — 2 elements/thread. |

Public Rust wrappers `f32_to_bf16_u16` (`:100`) and `bf16_to_f32_u16` (`:119`)
take raw device pointers as `u64` / `&mut CudaSlice<f32>` so `Tensor::to_dtype`
can call the NVRTC kernel directly without going through the F32-staging
storage roundtrip. Both call `lc_pairs(n)` so each thread handles 2 elements.
See the `to_dtype` fast-path landmine in CONVENTIONS for the dispatch logic.

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

### `conv1d.rs` — 1D conv + transposed conv (BF16 via cuDNN)

No dedicated CUDA kernels — the 1D conv paths reshape `[B, C, L]` to
`[B, C, 1, L]` and call `cudnn_conv2d_bf16` with `(H=1, W=L)` descriptors. This
re-uses cuDNN's mature BF16 conv2d path with F32 accumulation.

| Function | File:line | Purpose |
|---|---|---|
| `conv1d(x, w, bias, stride, padding, dilation, groups)` | `conv1d.rs:17` | Forward 1D conv. Plumbs dilation through to `cudnn_conv2d_bf16` via the length-axis (`dilation_w`). |
| `conv1d_grouped(x, w, stride, padding, groups)` | `conv1d.rs` | Thin wrapper over `conv1d` for depthwise/grouped cases. |
| `conv_transpose1d(x, w, bias, stride, padding, output_padding, groups)` | `conv1d.rs:83` | 1D transposed conv. Implemented via zero-insert → regular cuDNN conv1d with flipped + C_in/C_out-transposed weight. |
| `conv_transpose1d_dilated(x, w, bias, stride, padding, output_padding, dilation, groups)` | `conv1d.rs` | Same, with explicit `dilation`. The non-`_dilated` variant forwards `dilation=1`. |

**`conv_transpose1d` math** (documented at the call site too):

> `ConvTranspose1d(x, w, s, p, op) ≡ Conv1d(zero_insert(x, s, right_pad=op), flip+transpose(w), padding_side=(K-1)·d - p)`

No dedicated CUDA kernel — the im2col + cuBLASLt GEMM + col2im path is a
potential optimization for large output lengths (see the BigVGAN vocoder
speed work in `handoff_ltx23_pure_rust_port.md`).

**`cudnn_conv2d_bf16` signature** is:
```rust
pub fn cudnn_conv2d_bf16(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),  // 2026-04: previously hardcoded (1, 1)
    groups: usize,
) -> Result<Tensor>
```

### `rng/mod.rs` — F32 RNG

| Kernel | Line | Purpose |
|---|---|---|
| `fill_rand_f32` | `:18` | Per-thread xorshift32 F32 random fill. Used by `Tensor::randn` for the F32 path. |
| `flame_randn_torch_f32` | `rng/torch_compat.rs` | **Bit-exact** `torch.randn` parity. Mirrors PyTorch's `distribution_elementwise_grid_stride_kernel` + `curand_normal4`: per-thread Philox4x32-10 seeded with `curand_init(seed, idx, 0)`, two Box-Muller pairs per quad, grid-stride loop with `unroll=4`. Grid sized to PyTorch's `calc_execution_policy`. Tested against torch fixtures at `tests/torch_randn_fixtures/`. |
| `flame_rand_torch_f32` | `rng/torch_compat.rs` | **Bit-exact** `torch.rand` parity (uniform [0, 1)). Mirrors `uniform_kernel` (DistributionTemplates.h:458) with from=0, to=1: same Philox4_32_10 state setup, `_curand_uniform(x) = x * 2^-32 + 2^-33` per element, then reverse-bound `if (u == 1.0) u = 0.0`. Tested at `tests/torch_compat_fixtures/`. |
| `flame_bernoulli_torch_f32` | `rng/torch_compat.rs` | **Bit-exact** `torch.empty(shape).bernoulli_(p, generator=g)` parity. Same uniform stream as `flame_rand_torch_f32`; emits `(u < p) ? 1.0 : 0.0` per element. Tested at `tests/torch_compat_fixtures/`. |
| `flame_randint_torch_i32` | `rng/torch_compat.rs` | **Bit-exact** `torch.randint(low, high, shape)` parity for ranges < 2^32. Mirrors `random_from_to_kernel`'s uint32 dispatch (DistributionTemplates.h:309) + `uniform_int_from_to(val, range, base) = (val % range) + base`. Output I32. Tested at `tests/torch_compat_fixtures/`. |
| `flame_uniform_torch_f32` | `rng/torch_compat.rs` | **Bit-exact** `tensor.uniform_(a, b, generator=g)` parity. Underlies `kaiming_uniform_torch` and `xavier_uniform_torch`. Same uniform stream as `flame_rand_torch_f32`; applies `value = rand * (to - from) + from` with reverse-bound `if (value == to) value = from`. Tested at `tests/torch_compat_fixtures/`. |

### `sgd/mod.rs` — F32 SGD step

| Kernel | Line | Purpose |
|---|---|---|
| `sgd_f32` | `:13` | `p -= lr * g`. Used by the F32 training SGD. |

### `adam.rs` — fused Adam / AdamW step

Six NVRTC kernels, concatenated into a single translation unit, compiled
once on first call, loaded into the `adam_fused` module. All kernels are
single-pass: read `(param, grad, m, v)`, write `(param, m, v)` in place,
no temporaries. All implement decoupled weight decay (AdamW).

**Per-tensor variants** (launch config `block=256, grid=(n+255)/256`):

| Kernel | Param / Grad dtype | Purpose |
|---|---|---|
| `adam_fused_bf16_kernel` | BF16 param, BF16 grad, F32 m/v | BF16-param fast path. |
| `adam_fused_f32grad_kernel` | BF16 param, F32 grad, F32 m/v | BF16-param with F32 grad (default path — `Parameter::set_grad` casts grads to F32). |
| `adam_fused_f32param_f32grad_kernel` | F32 param, F32 grad, F32 m/v | F32-param fast path (biases, F32 embeddings, F32 LoRA alphas). |
| `adam_fused_f32param_bf16grad_kernel` | F32 param, BF16 grad, F32 m/v | F32-param with BF16 grad for callers that bypass `Parameter::set_grad`. |

**Multi-tensor variants** (launch config `block=256, grid=n_tensors`,
"one block per tensor + grid-strided loop within the block"):

| Kernel | Param / Grad dtype | Purpose |
|---|---|---|
| `adam_fused_multi_bf16_f32grad_kernel` | BF16 param, F32 grad, F32 m/v | Klein 9B / BF16-param LoRA path. One launch covers every parameter via a 5-region packed pointer/size buffer (`[params \| grads \| ms \| vs \| sizes]`) staged via a single H2D copy. Buffer is held by `fused::MultiTensorMetaCache` (one per `Adam` instance) so the alloc cost is paid once at first step, not per step. |
| `adam_fused_multi_bf16_bf16grad_kernel` | BF16 param, BF16 grad, F32 m/v | Same launch shape as above, BF16 grad reads. |
| `adam_fused_multi_f32param_f32grad_kernel` (added 2026-05-12, Phase 1) | F32 param, F32 grad, F32 m/v | Zimage / no-quant LoRA path. Same 5-region packed-buffer pattern as the BF16 multi-tensor kernels, no BF16↔F32 casts (params + grads + m + v all F32). Bit-identical per-element math to `adam_fused_f32param_f32grad_kernel`. Auto-selected by `Adam::step` when all params are F32 + all grads are F32. |
| `adam_fused_bf16_f32grad_stoch_kernel` (added 2026-05-08) | BF16 param, F32 grad, F32 m/v | Single-tensor variant of the BF16-param + F32-grad fused step **with stochastic rounding** at the F32 → BF16 store. Per-element entropy from splitmix64 keyed on `(seed, idx)` where `seed` is supplied by the caller (typically the optimizer step counter). Same Adam math as `adam_fused_bf16_kernel` modulo the rounding. Selected by `Adam::step` when `Adam::set_stochastic_round(true)` was called and the param is BF16 + grad is F32. |
| `adam_fused_multi_bf16_f32grad_stoch_kernel` (added 2026-05-08) | BF16 param, F32 grad, F32 m/v | Multi-tensor variant of `*_stoch_kernel` above. Same 5-region packed buffer harness as `adam_fused_multi_bf16_f32grad_kernel`; tensor index is mixed into the seed to avoid lock-step rounding decisions across params at the same elementwise idx. |

`Adam::step` auto-selects multi-tensor when all params share a dtype
(BF16 or F32) and all grads are F32. The classifier returns
`Some(param_is_bf16)` for that case and routes to the appropriate
multi-tensor kernel via the `param_is_bf16: bool` discriminator on
`adam_fused_multi_tensor_step`. Otherwise it falls through to the
per-tensor loop. `FLAME_ADAM_NO_MULTI_TENSOR=1` forces the fallback.

Bit-exact parity is gated by `tests/adam_multi_tensor_parity.rs`: each
of the BF16 and F32 multi-tensor kernels matches its per-tensor
counterpart byte-for-byte on a 50-tensor 1-step toy. The F32 path stays
within 1e-7 relative tolerance across 100 sequential steps (same kernel
math, same accumulation order; tiny tolerance covers any future float
reassociation). BF16 stays within BF16 atol/rtol across 100 steps.
Multi-tensor only changes launch shape — kernel math is identical
including the DECOUPLED-WD ordering receipt.

### `adam8bit_kernel.rs` — block-wise 8-bit AdamW (bnb 0.49.2 parity)

Two NVRTC kernels in one translation unit, compiled once on first call,
loaded into the `adam8bit_fused` module:

| Kernel | Grad dtype | Purpose |
|---|---|---|
| `adam8bit_blockwise_kernel` | F32 grad | Per-parameter fused dequant + AdamW step + requant for one parameter tensor. F32 param required. |
| `adam8bit_blockwise_bf16grad_kernel` | BF16 grad | Same but reads grads as `__nv_bfloat16` and widens to F32 inside the kernel. Output (param F32, m/v u8 codes + F32 absmax) identical to F32-grad path modulo BF16 noise on `g`. |

**Launch geometry** (per parameter, per step):

- `grid_dim = (ceil(n / 256), 1, 1)` — one block per 256-element slab.
- `block_dim = (256, 1, 1)` — one thread per element.
- `shared_mem = 4096 bytes` = `2 × qmap[256] + s_m[256] + s_v[256]`, all `f32`.

**Per-block flow** (single launch, no host round-trip):

1. Co-load both 256-entry qmaps (signed for `m`, unsigned for `v`) into
   shared memory.
2. Each active thread dequants `(m_old, v_old)` via `qmap[code] * absmax[blk]`,
   runs AdamW math (`m_new = β₁·m_old + (1-β₁)·g`; `v_new = β₂·v_old + (1-β₂)·g²`;
   `p -= lr · m̂/(√v̂ + ε); p -= lr·wd·p`), writes back `param[i]`.
3. Stash `(m_new, v_new)` in shared, run block-wide max-abs tree reduction
   (`__syncthreads` + halving stride 128→64→…→1) to compute
   `(absmax_m_new, absmax_v_new)`.
4. Thread 0 stores the new absmax into `m_absmax[blk] / v_absmax[blk]`.
5. Each active thread requants by **linear scan** over the 256-entry
   shared qmap (256 abs-diff compares per element per moment, picks
   argmin). Linear scan because the signed qmap is monotonic but the
   unsigned one has a long flat-near-zero region from the negative
   half being absent (verified against bnb output); binary search would
   require care on tiebreaks and the scan is fast enough at this size.
6. Writes the chosen `u8` codes to `m_codes[i] / v_codes[i]`.

**Edge cases**:
- **Tail block** (when `n % 256 ≠ 0`): inactive threads load from the
  block base index (any in-range index — they don't write anywhere), and
  contribute `0` into the shared `s_m / s_v`, so they cannot drag the
  block absmax. Writes are guarded by the `active` predicate.
- **All-zero block**: post-reduction absmax falls back to `1e-12f` to
  avoid NaN on the normalize divide before requant (codes degrade to
  whatever LUT entry best approximates 0).
- **BF16 grad**: upcast via `__bfloat162float` once at load; all internal
  math is F32. No autograd hook (state slices are raw `CudaSlice`s; only
  the F32 param tensor bumps its version counter).

**Bit-exact equivalence** to bitsandbytes 0.49.2 `optim.AdamW8bit(block_wise=True)`
(non-paged) for the AdamW algorithm, modulo BF16 noise on the grad load.
Validates with `eridiffusion_core::AdamW8bit`'s `adamw8bit_close_to_adamw`
test (within 5e-4 abs vs F32 AdamW after 3 steps on a 4-elem param).
Decoupled WD ordering matches the receipt at `src/adam.rs:1-32`.

**Perf**: one kernel launch per parameter per step, no PCIe round-trips
(qmaps live on-device across all steps, state buffers are on-device for
the entire run). Roughly equivalent to plain `adam_fused_f32param_*` per
the 30-step real-data validation in `eridiffusion-core::AdamW8bit`'s
parity suite — the extra dequant / reduce / requant in shared memory is
small vs. the param/grad global-mem traffic.

**See also**: `FLAME_CONVENTIONS.md` — "Optimizer kernel placement"
(why this file sits at `src/adam8bit_kernel.rs` and not in `bf16_*.rs`),
"`TensorStorage` has no `U8` arm" (why the per-param U8 state is held as
raw `CudaSlice<u8>` rather than `Tensor`), "`Tensor::from_vec` is
F32-only" (how `upload_qmap` lands the 256-entry LUT on the device).

### `int8_weight_only_qt_kernel.rs` — torchao int8-QT cast (2026-05-17)

One NVRTC kernel, compiled once on first call, loaded into the
`int8_qt_cast` module:

| Kernel | Purpose |
|---|---|
| `i8_to_bf16_kernel` | Trivial sign-extending int8 → BF16 cast. No scale baked in (scale is multiplied in a separate BF16 elementwise op so autograd records the `Op::Mul` correctly). |

**Launch geometry** (per `codes_to_bf16` call):

- `grid_dim = (ceil(n / 256), 1, 1)` — one block per 256 elements.
- `block_dim = (256, 1, 1)` — one thread per element.
- `shared_mem = 0`.

**Per-thread**: sign-extend `i8 → int → float → __nv_bfloat16` via
`__float2bfloat16` (round-to-nearest-even). Tail-block bounds-checked.

**Usage**: invoked by `Int8QtWeight::codes_to_bf16` at every forward
pass (cheap; the resulting BF16 matrix feeds `Tensor::matmul`). Storing
the cast result instead of recomputing it would defeat the memory-saving
purpose of int8 weights — we trade ~1 NVRTC kernel + 2× BF16 memory per
layer for the int8 storage win.

### `ops/multi_tensor.rs` — foreach-style primitives (Phase 3, 2026-05-12)

| Kernel | Purpose |
|---|---|
| `multi_tensor_l2_norm_sq_stage1_f32_kernel` | Stage 1 of the multi-tensor L2 norm. One block per tensor, block-wide tree reduction in shared memory, writes per-tensor partial sum-of-squares to `partials[N]`. Used by `flame_core::ops::grad_norm::global_l2_norm` when every grad is F32 + contiguous. |
| `multi_tensor_l2_norm_sq_stage1_bf16_kernel` | BF16 stage 1 (Phase 4a, 2026-05-13). Reads `__nv_bfloat16` storage, widens to F32 inside the inner loop (`opmath_t`), accumulates in F32, writes F32 partials. Same block-wise tree-reduction shape as the F32 kernel. Used by `global_l2_norm` when every grad is BF16 + contiguous — keeps BF16 grads BF16 through gradient clipping under autograd v2 / `BF16_GRAD_DECISION.md` Option A. |
| `multi_tensor_l2_norm_sq_stage2_kernel` | Stage 2: single-block reduction across `partials[N]` → `out[1]`. `N` can exceed `blockDim` (256); loop strides until exhausted. Shared by BF16 and F32 stage-1 paths. |
| `multi_tensor_scale_inplace_f32_kernel` | In-place `x[i] *= scale` across a list of F32 tensors. One block per tensor, grid-stride inner loop. Pointwise — no reduction, so bit-identical to per-tensor `mul_scalar`. Added 2026-05-12 (Phase 2) for the trainer clip-grad path; default-off in callers via `FLAME_MT_SCALE=1`. |
| `multi_tensor_scale_inplace_bf16_kernel` | BF16 variant. Loads `__nv_bfloat16`, multiplies via `__bfloat162float → float * scale → __float2bfloat16`. Within 1 ULP of per-tensor BF16 `mul_scalar` because both go through the same cast chain. |

The 2-stage L2 norm pattern replaces `2N + (N-1) + 1 = 2N` legacy launches
(per-tensor `square().sum()` + serial fold-add + sqrt) with 3 launches.
Parity is tolerance-bound, not bit-identical, because parallel-tree
reduction sums in a different order than serial fold-add. Tests in
`tests/multi_tensor_l2_norm_parity.rs` document the ≤ 1e-5 absolute /
≤ 1e-6 relative bound. Env override `FLAME_MT_L2NORM=0` falls back to
legacy.

The in-place scale primitive replaces `N` per-parameter `mul_scalar`
launches with `1` grid-per-tensor launch when the trainer's clip-grad
scale fires. Pointwise math means F32 is bit-exact and BF16 is 1-ULP-
bound — both verified by `tests/multi_tensor_scale_parity.rs`.
Currently default-off in trainer call sites (`train_zimage.rs`,
`train_klein.rs`) because production grad-norms stay below the clip
threshold; enable via `FLAME_MT_SCALE=1`. See
`EriDiffusion-v2/HANDOFF_2026-05-12_PHASE2_SCALE_FOLLOWUP.md` for the
deferral rationale.

The shared `MultiTensorMetaCache` (`ops/multi_tensor.rs`) is separate
from `adam::MultiTensorMetaCache` — different region layouts (L2 norm
needs partials buffer + 2 pointer/size regions; Adam needs 5 regions).
Held behind a process-wide `Mutex` in `ops::grad_norm`.

### `norm.rs` — RMSNorm NVRTC kernels (training + inference)

Live RMSNorm path for both `norm::rms_norm` (training entry) and
`cuda_ops_bf16::rms_norm_bf16` (inference — delegates to `norm::rms_norm` as
of commit `d729ede`, 2026-05-12). Three NVRTC kernels, all 2026-05-12:

| Kernel | Source const / line | Purpose / notes |
|---|---|---|
| `rms_norm_forward_bf16_vec` | `RMS_NORM_FWD_KERNEL_BF16_VEC` (`:1368`) | **Vectorized forward.** Block per row, 256 threads/block, `bf16x4` (8-byte aligned) loads, warp-shuffle intra-warp reduction, smem inter-warp aggregation (`n_warps` floats). Replaces the legacy 1-thread-per-row scalar kernel. 13.5–16.1× faster on zimage/klein/chroma forward; 3.5× on qknorm (small `norm_size=128`). Requires `norm_size % 4 == 0`. |
| `rms_norm_backward_bf16_vec` | `RMS_NORM_BWD_KERNEL_BF16_VEC` (`:1522`) | Same 256-threads + vec=4 pattern; writes `grad_input` only. `grad_weight` moved to the separate cross-row reduction below. 9.5–14.8× faster than legacy. |
| `rms_norm_grad_weight_bf16_vec` | `RMS_NORM_GRAD_WEIGHT_KERNEL_BF16` (`:1644`) | Cross-row `grad_weight` reduction. Tile `COLS_PER_BLOCK=64` × `ROWS_PER_BLOCK=512`. Grid `(ceil(norm_size/64), ceil(batch_size/512), 1)`, block `(64, 1, 1)`. ~500× fewer atomicAdds than inline accumulation (qknorm: 12.6M atomics → ~25k). Fires only in the vec path; legacy kernel still accumulates inline. |

Dispatch in `rms_norm_forward_bf16` (`:735`) and `rms_norm_backward_bf16`
(`:957`): vec when `norm_size % 4 == 0` (production shapes 2560/3072/4096/128
all qualify). `FLAME_RMS_NORM_LEGACY=1` forces the scalar fallback for A/B
benchmarking. Bench: `cargo bench --features cuda --bench rms_norm_vec`.

### `cuda_kernel_sources.rs` — shared kernel source constants (NVRTC)

| Kernel | Purpose |
|---|---|
| `permute_generic_{f32,bf16}_kernel` | Generic 8-D permute via (in_strides, out_strides, perm) table. Dispatched by `GpuOps::permute_generic` as the fallback from `Tensor::permute`. Assumes row-major contiguous input. |
| `materialize_strided_{f32,bf16}_kernel` ⭐ | Generic strided-plus-offset gather into contiguous row-major output. Used by `Tensor::contiguous()` to realize narrow/chunk views (and any narrow-of-permute composition). `src_addr = in_offset + sum(out_coord[d] * in_strides[d])`. |

### `cuda_kernels_gpu.rs` — F32 framework kernels (training)

Two notable broadcast kernels:

| Kernel | Line | Purpose |
|---|---|---|
| `mul_bc_kernel` | `:2421` | F32 broadcast multiply |
| `add_bc_kernel` | `:2548` | F32 broadcast add |

The full set of F32 NVRTC kernels in `cuda_kernels.rs` and
`cuda_kernels_gpu.rs` (~100+) are training-only — see those files directly.

### `tensor_ops_extended.rs` — index_assign (TREAD scatter-back)

Two NVRTC kernels (one per dtype) that implement the
`Tensor::index_assign(dim, indices, values)` op. Output shape == self
shape; for each output element the kernel decomposes the flat index into
per-axis coords using `self_strides`, looks up `inverse_mask[coord_dim]`
(precomputed on host: `inverse_mask[k] = pos in indices` or `-1`), and
copies from `values` (at `(coords with axis-dim = pos)`) when masked,
else from `self` at the same flat index.

| Kernel | Symbol | Notes |
|---|---|---|
| `index_assign_f32_kernel` | `:1597` (const), `:1666` (launch in `index_assign_f32`) | F32 path. Single thread per output element. `total = self.elem_count`. |
| `index_assign_bf16_kernel` | `:1622` (const), `:1755` (launch in `index_assign_bf16`) | BF16 path (`u16` reads/writes). Inputs forced through `clone_result()` to materialize arena/view BF16 → owning storage. |

Used by `eridiffusion-core/training/features/tread.rs::TreadStep::scatter_routed`.

---

## Build-time `.cu` kernels (compiled by `build.rs`)

`build.rs` lists the source files (search for `cuda_sources.push`). The two
locations are:
- **`cuda/`** at the repo root — older surface, `fc_*` symbols (status enum)
- **`src/cuda/`** — newer fused inference kernels, `flame_*` symbols (int)
- **`src/kernels/`** — additional `.cu` files
- **`kernels/`** at repo root — duplicates of `src/kernels/` (some legacy
  copies; check `build.rs` for which is actually compiled)

### `src/cuda/flash_attention_fwd.cu` — FA2-style wmma flash attention (LIVE)

The single most-important file in this directory. Current kernel is a
templated K/V-reuse WMMA design swapped in 2026-04-17 (replacing the
Phase 1.6 `cp.async`-vectorized kernel, kept on disk as
`flash_attention_fwd.phase16.cu.bak`). Supports HD ∈ {64, 96, 128}
via runtime dispatch and accepts a runtime `causal` flag.

| Symbol | Line | Notes |
|---|---|---|
| `flash_attn_fwd_kernel<TILE_Q, TILE_KV, HD, NUM_WARPS>` | `:87` | Templated WMMA kernel — FP32 accumulators, online softmax, K/V-reuse shared buffer. |
| `flame_flash_attention_bf16` | `:333` | C entry point: `(Q, K, V, O, LSE, batch_heads, seq_q, seq_kv, head_dim, causal, stream)`. LSE output unused (backward path stores it separately). |
| `launch_fwd` | `:263` | Per-head_dim dispatch into the template specialization. |
| `mask_tail_2d_float_neg_inf` | `:71` | Softmax-safe tail mask — fills invalid cells with `-INFINITY`. See gotcha in CONVENTIONS. |

**Tile sizes**: HD64/HD96 use `TILE_Q=64, TILE_KV=64, NUM_WARPS=16`.
HD128 causal also uses `64x64`; HD128 non-causal uses `TILE_Q=64,
TILE_KV=32, NUM_WARPS=8` to track PyTorch FlashAttention's narrower K tile
for O1 full-suffix attention while staying inside Flame's shared-memory
`s_O` accumulator budget. Warp layout is row-groups x col-groups (each warp
owns one 16x16 QK^T block). PV matmul scatters across `row_group x hd_tile`
pairs with an atomicAdd into `s_O`.

**2026-05-21 HiDream-O1 FA2 parity work**: the forward kernel keeps raw QK
logits until softmax, scans K/V tiles right-to-left, applies runtime causal
masking, and mirrors PyTorch FlashAttention's `UNFUSE_FMA` softmax form:
`exp2(__fmul_rn(score, log2(e)/sqrt(head_dim)) - max_scaled)`. This tracks
PyTorch numerics more closely while preserving Flame's existing shared-memory
accumulator kernel. It is not yet a full CUTLASS/CUTE port.

Verification trap: `cargo test -p flame-core --features "cuda bf16_u16"
--test fa2_parity_naive -- --nocapture` passes for `N={512,4096}` and
`HD={64,128}` against a materialized FP32 reference (`cos_sim=0.999997`,
`max_abs<=3.906e-3`). `tests/sdpa_ragged_sk.rs` also passes for
`Sk={64,71,72,128,200}`. The O1 production parity gate still fails at
`layer00.attn_out`; `layer00.sdpa_out` is within the current threshold and
the sparse one-ULP attention differences are amplified by `o_proj`.

**PyTorch reference tile targets, deferred**: local PyTorch FlashAttention
dispatch uses SM8x HD64 non-causal `128x128`, HD96 non-causal `128x64`,
HD128 non-causal `128x32`, and HD96/HD128 causal `64x64`, typically with
4 warps and register-resident output accumulators. Flame remains `64x64`
with 16 warps for now because the current `s_O` shared-memory design cannot
drop in the HD128 `128x32` path without exceeding the SM86 budget. Finish
that exact tiled port only if O1 trainer parity still needs it after the
current gate.

**Shared-memory layout** (HD=128 worst case):
| Region | Dtype | Size (HD=128) | Role |
|---|---|---|---|
| `s_Q`  | BF16  | 16 KB  | Q tile, persists across KV iters |
| `s_K ≡ s_V` | BF16 | 16 KB | **Aliased** — K is used for QK^T then overwritten by V for PV |
| `s_S`  | FP32  | 16 KB  | QK^T scores for this iter |
| `s_P`  | BF16  | 8 KB   | Softmax probs for this iter (separate region) |
| `s_O`  | FP32  | 32 KB  | Running output accumulator |
| `s_m`, `s_l` | FP32 | 0.5 KB | Per-row running max & denom |

**K/V-reuse is the budget trick**: aliasing `s_K` and `s_V` (K is
dead after QK^T, V gets loaded into the same slot before PV) saves
one `TILE_KV*HD` BF16 region. That's exactly what fits HD=128,
TILE_KV=64 within SM_86's 100 KB opt-in budget.

**2026-04-19 fix — multi-tile softmax under-scale at Sk > BKV=64.**
Invalid K-tile tail columns were initially masked with `0.0f` before
the per-row online softmax. `exp(0 - new_max)` then contributed ~56
spurious terms to `tile_sum` at Sk=72 (kv_rows=8 in the second
tile), inflating the denominator ~25-30% and under-scaling output
~32%. Fix: mask invalid cells with `-INFINITY` via
`mask_tail_2d_float_neg_inf`. Regression:
`tests/sdpa_ragged_sk.rs` covers Sk ∈ {64, 71, 72, 128, 200}, all
cos_sim ≥ 0.999995 and mag_ratio = 1.0000.

**Stage 3 (P@V)** each warp computes a 16×16 output block into a
per-warp FP32 scratch placed at `s_S + warp_id * 256` (s_S is dead
for this iter after softmax consumed it). Lanes then atomicAdd their
scratch slots into `s_O`. The warp partitioning (`row_group × col_group`)
already guarantees disjoint s_O targets, so the atomicAdd is conservative
rather than necessary, but it keeps the code simple.

**Softmax (Stage 2)**: one thread per Q row in the valid-row loop
(`for qi = tid; qi < q_rows; qi += THREADS`). Each thread scans its
row's TILE_KV=64 cols serially to build `tile_max` and
`tile_sum = Σ exp(s_S[qi,j] - new_max)`. Since all invalid tail cols
are now `-INFINITY` (2026-04-19 fix), the serial scan can run the full
TILE_KV without branching on validity.

**Load helper `load_tile_bf16_zero_padded`** (`:37`): scalar BF16 load
with a `valid_rows` clamp. Invalid rows are written as `__float2bfloat16(0)`
— safe for Q (zero-padding doesn't contaminate the WMMA output since
corresponding softmax scores are also tail-masked) and for V (zero V
rows contribute zero through PV).

**LSE output**: unused by this kernel. The `LSE` argument is kept in
the C entry for signature compatibility with the prior backward
kernel. The backward was removed in Phase 2c (2026-04-23) — training
now uses `flame_cudnn_sdpa_bwd_bf16` instead, which reads its Stats
(LSE) from the cuDNN training-forward (`flame_cudnn_sdpa_bf16_train_fwd`)
rather than this kernel.

**Parity**: `tests/fa2_parity_naive.rs` validates the kernel against a
pure-Rust FP32-materialized reference. `tests/sdpa_ragged_sk.rs`
covers the Sk > BKV regression at Sk ∈ {64, 71, 72, 128, 200} —
the bug fixed on 2026-04-19.

### `src/cuda/cudnn_sdpa.cpp` — cuDNN v9 Flash SDPA forward shim (LIVE)

Host-only C++17 shim over NVIDIA's vendored `cudnn_frontend`. Two
entry points, both using the same graph-cache machinery keyed on
(shape, per-tensor strides, scale). Offsets are pointer-arithmetic
at execute-time so one graph handles any sliced/narrow view.

| Symbol | Notes |
|---|---|
| `flame_cudnn_sdpa_bf16` | Inference forward. `set_generate_stats(false)`. 12.1× faster than WMMA at Klein's shape (3.24 ms vs 39.26 ms per call). |
| `flame_cudnn_sdpa_bf16_train_fwd` | Training forward. `set_generate_stats(true)` — also writes Stats (per-row log-sum-exp). Added Phase 2c (2026-04-23). Output Stats layout: contig FP32 `[B*H, N_q]`, equivalent to 4D `[B, H, N_q, 1]` stride `[H*N_q, N_q, 1, 1]` — that's what the backward shim expects. |

BF16 in/out, FP32 intermediate + compute. Per-shape graph build is a
one-shot cost (~50–200 ms); steady-state dispatch is pure `execute`.
Separate graph caches for the two entry points (different topology
because of the Stats output).

**Parity**: `tests/cudnn_sdpa_parity.rs` (inference vs WMMA, Klein +
Chroma, cos_sim 1.000000), `tests/cudnn_sdpa_bwd_parity.rs::cudnn_sdpa_train_fwd_matches_inference_fwd`
(train-fwd O is bit-exact to inference-fwd O).

### `src/cuda/cudnn_sdpa_bwd.cpp` — cuDNN v9 SDPA backward shim (LIVE)

Companion to `cudnn_sdpa.cpp`. Added Phase 2c (2026-04-23) to replace
the decomposed-recompute backward that every trainer was actually
hitting (the old WMMA backward at `flash_attention_bwd.cu` was gated
behind an unused `flash_attn` feature). Measured: 30–50× faster than
decomposed recompute on typical DiT training shapes.

| Symbol | Notes |
|---|---|
| `flame_cudnn_sdpa_bwd_bf16` | `g.sdpa_backward(Q, K, V, O, dO, Stats) → [dQ, dK, dV]`. 9 BF16 tensors in/out plus the FP32 Stats input. Graph cached per (shape, all 8 BF16 stride vectors, scale). Stats stride is fixed by convention (`[H*N_q, N_q, 1, 1]`) so it's not part of the key. |

Shape support mirrors the forward: unmasked, head_dim ∈ {64, 96, 128},
4D `[B, H, N, D]` BF16. Shapes outside that fall through to the
decomposed recompute path in `autograd.rs::attention_backward_recompute`.

**Parity**: `tests/cudnn_sdpa_bwd_parity.rs` (HD=64 and HD=128 vs
decomposed FP32 recompute; cos_sim ≥ 0.999996, mean_rel ≤ 1.7e-3
on each of dQ, dK, dV).

### `src/cuda/fused_linear3d.cu` — cuBLASLt 3D linear (LIVE)

| Symbol | Line | Notes |
|---|---|---|
| `flame_linear3d_bf16` | `:24` | cuBLASLt BF16 matmul + bias epilogue. Weight is `[Cin, Cout]` row-major (pre-transposed). Used by Klein. |
| `flame_linear3d_bf16_native` | `:135` | Same but takes weight in standard PyTorch `[Cout, Cin]` row-major layout. Uses `TRANSA=T` so the transpose happens inside the GEMM. **This is what every FLUX/Chroma/QwenImage block forward calls.** Added 2026-04. |
| `flame_linear3d_bf16_pytorch_parity` | `:369` | Bit-exact mirror of PyTorch's biased `at::cuda::blas::gemm_and_bias<at::BFloat16>` path (added 2026-05-20). Same weight layout as `_native`. With `bias == NULL`, delegates to `_native` because PyTorch no-bias `F.linear` follows the matmul path; this is required for HiDream-O1 Full `x_embedder.proj1` parity and is faster than the biased heuristic mirror. Biased differences vs `_native` (sourced from `CUBLASLT_LOG_LEVEL=5` capture of PyTorch): (1) workspace = 1 MiB on the preference (not 4 MiB); (2) `BIAS_POINTER` set on descriptor *before* `cublasLtMatmulAlgoGetHeuristic`; (3) heuristic called **per-call**, not cached, because the bias pointer changes each invocation; (4) `BIAS_DATA_TYPE` not set explicitly (PyTorch leaves it default); (5) no `BATCH_COUNT` / `STRIDED_BATCH_OFFSET` on layouts; (6) no `MIN_ALIGNMENT_*` preferences. The descriptor + layouts are still cached per-shape (`g_linear_parity_cache` at `:365`); only the biased algo is heuristic-selected per call. ~1% perf overhead for biased calls, byte-identical output vs PyTorch. C entry wrapped by `ops::fused_inference::fused_linear3d_native_pytorch_parity` (Rust at `ops/fused_inference.rs:479`, FFI at `cuda/ffi.rs:906`). |

All three use `CUBLAS_COMPUTE_32F` accumulation, BF16 inputs/outputs, and the
`CUBLASLT_EPILOGUE_BIAS` epilogue (so the bias add is fused into the GEMM —
no separate add kernel).

### `src/cuda/grouped_mm.cu` — grouped BF16 matmul (MoE)

| Symbol | Line | Notes |
|---|---|---|
| `grouped_mm_bf16_kernel` | `:120` | Single fused kernel covering all E experts. Grid: `(ceil(N/128), ceil(T_max/128), E)`. Tile: `BM=128 BN=128 BK=32`, warp tile `64x64`, WMMA 16x16x16 BF16→FP32 fragments. 4 warps per block (128 threads). Matches `torch.nn.functional.grouped_mm(x, w, offs=offsets)`. |
| `flame_grouped_mm_bf16` | `:255` | C entry. Wrapped by `ops::grouped_mm::grouped_mm_bf16` (the previous doc reference to `Tensor::grouped_mm` was aspirational — there's no Tensor method, the wrapper takes `(x, w, offsets: &[i32], t_max)`). |

Offset semantics: `offsets` is **exclusive cumulative end indices** of length `E` (expert `e` covers rows `[offsets[e-1] .. offsets[e])`, with `offsets[-1] := 0`). Matches PyTorch's `F.grouped_mm`. The Rust wrapper takes a host `&[i32]` and HtoD-copies into a temp `CudaSlice<i32>`; flame-core's `DType::I32` tensors hold f32-bytes-relabeled and would feed nonsense to the kernel — see `FLAME_CONVENTIONS.md` for the full story.

Phase-1 perf (RTX 3090 Ti, T=32768 K=2048 N=2688 E=64 uniform, BF16):
- for-loop of 64 cuBLASLt matmuls: ~12 ms → ~30 TFLOPS
- `grouped_mm`:                    ~15 ms → ~23 TFLOPS (0.78x of cuBLASLt-per-expert)

The cuBLASLt-per-expert baseline is already close to tensor-core peak for this shape (~18% of hardware peak due to tall-skinny T_e=512 dimension), so a single fused kernel cannot provide the ≥5x speedup over the baseline that would apply against a naive launch-overhead-dominated for-loop. The win is removing 64 separate tensor allocations + 64 launches; at this shape, ~1-3 ms.

### `src/cuda/fused_gated_scatter_add.cu` — fused MoE unpermute

| Symbol | Line | Notes |
|---|---|---|
| `fused_gated_scatter_add_kernel` | `:30` | `accum[indices[t]] += expert_out[t] * gating[t]` in one kernel. F32 `atomicAdd` because multiple `t`s may collide on the same output row (MoE top-K with K>1). Grid: `(ceil(D/256), T, 1)`, block = 256. `expert_out` is BF16, `gating` and `accum` are F32, `indices` is I32. |
| `flame_fused_gated_scatter_add_bf16` | `:57` | C entry. Wrapped by `ops::fused_gated_scatter_add::fused_gated_scatter_add_bf16(expert_out, gating, indices: &[i32], accum)`. (Previous doc reference to `Tensor::fused_gated_scatter_add` was aspirational — no Tensor method exists; the function is a free `pub fn` that mutates `accum` in-place.) Same I32-tensor caveat as `grouped_mm`: indices come in as a host slice, HtoD'd inside. |

Phase-1 perf (RTX 3090 Ti, T=32768 D=2048 N=4096 BF16 → F32 accum):
- cast BF16→F32 + F32 broadcast-mul (no scatter): 2603 μs
- `fused_gated_scatter_add` (incl. scatter):        794 μs  →  **3.28× speedup, 845 GB/s**

### NVRTC: `fused_swiglu_bf16_kernel` (in `ops/fused_swiglu.rs`)

| Kernel | Purpose / notes |
|---|---|
| `fused_swiglu_bf16_kernel` | Takes `(..., 2I) BF16` with first `I` cols = `up` and last `I` cols = `gate`; returns `up * silu(gate) : (..., I) BF16`. FP32 sigmoid math with a BF16 round on `silu(gate)` between the sigmoid and the multiply so the output matches PyTorch eager `up * F.silu(gate)` bit-for-bit. Launch: `grid=(ceil(I/256), rows, 1)`, block=256, grid-strided along the inner dim. Used by MoE FFN forward after `grouped_mm` of the `gate_up_proj`. |

Phase-1 perf (RTX 3090 Ti, T=32768 I=2688 BF16):
- narrow + silu + mul (unfused):  3300 μs
- `fused_swiglu`:                  647 μs  →  **5.10× speedup, 817 GB/s**

### `src/cuda/fused_rms_norm.cu` — fused RMSNorm

| Symbol | Line | Notes |
|---|---|---|
| `fused_rms_norm_bf16` (kernel) | `:26` | One block per row; sum-of-squares + rsqrt + scale, single kernel. |
| `flame_fused_rms_norm_bf16` (entry) | `:89` | C entry. Used by `ops::fused_inference::fused_rms_norm`. |

> **Pattern (Gemma3 / MagiHuman `(weight + 1)` formulation):** the kernel computes `out = normed * weight`. For models that use `out = normed * (weight + 1)`, pre-add `1.0` to the weight tensor at layer-load time and pass that as `weight`. Saves one `add_scalar(1.0)` kernel launch per call. MagiHuman MM layers also pre-split `[hidden * 3]` weights into 3 per-modality contiguous chunks at load time so per-call forward can call `fused_rms_norm` 3 times directly (one per modality narrow + cat) instead of running the 6-op cascade. Empirical: replaced ~10 kernel launches at ~5 sec/call with one fused launch at ~1 ms (5000× speedup at L≈1086, hidden=5120).

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

### `src/cuda/fp8_quant.cu` — BF16 → FP8 E4M3 (activation offload)

| Symbol | Line | Notes |
|---|---|---|
| `f32_to_fp8_e4m3` (device) | `:19` | Per-element F32 → FP8 E4M3 with round-to-nearest and subnormal handling. Clamps to +-448 (E4M3 max). No inf/nan encoding. |
| `bf16_to_fp8_kernel` | `:59` | Grid-stride loop, 1 element/thread. `output[i] = fp8(bf16_to_f32(input[i]) * inv_scale)`. |
| `flame_bf16_to_fp8` | `:74` | C entry: `(input, output, inv_scale, n_elements, stream) -> int`. Block=256, grid capped at 65535. |

Pairs with `fp8_dequant.cu::flame_fp8_to_bf16` for the round-trip. Used by
`ActivationOffloadPool::push` when `OffloadCompression::FP8` is enabled.
The caller provides `inv_scale = 1.0 / scale` where `scale = absmax / 448.0`;
the pool currently uses a fixed scale assuming activation range [-8, 8].

### `src/cuda/fp8_dequant.cu` — FP8 → BF16

| Symbol | Line | Notes |
|---|---|---|
| `fp8_to_bf16_kernel` | `:10` | E4M3 / E5M2 unpack. Used by FlameSwap FP8 paths and `ActivationOffloadPool::pull`. |
| `flame_fp8_to_bf16` | `:40` | C entry. |

### `src/cuda/mxfp4_dequant.cu` — MXFP4 → BF16

MXFP4 = 32 FP4 (E2M1) values share one 8-bit E8M0 exponent scale.
FP4 LUT bit-exactly matches HuggingFace transformers `FP4_VALUES`
(`[+0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]`); see
`transformers/integrations/mxfp4.py::convert_moe_packed_tensors`.

Per-block layout: 16 packed bytes (2 nibbles each, low → even index, high → odd)
+ 1 E8M0 byte for the shared scale `2^(scale_byte - 127)`. Output: 32 BF16.

| Symbol | Line | Notes |
|---|---|---|
| `FP4_LUT` (constant) | `:35` | 16-entry FP4 magnitude table. Indexed by 4-bit nibble; high bit = sign. |
| `flame_mxfp4_to_bf16_kernel` | `:47` | Grid-stride loop, **one thread per 32-element block**. Each thread reads 16 bytes + 1 scale, writes 32 BF16. `#pragma unroll` on the 16-byte inner loop. |
| `flame_mxfp4_to_bf16` | `:90` | C entry. Block=256, grid capped at 65535. `(blocks, scales, out, rows_total, stream) → int`. |

Used by Lens M2 (GPT-OSS encoder MoE expert dequant). Pairs with the
forthcoming `BLOCKOFF_MXFP4_PINNED=1` BlockOffloader path so raw MXFP4 stays
in pinned host RAM and is dequantized on H2D prefetch.

### `src/cuda/fp16_to_bf16.cu` — FP16 (IEEE half) → BF16

| Symbol | Line | Notes |
|---|---|---|
| `fp16_to_bf16_kernel` | `:14` | Per-element `__half2float` → `__float2bfloat16`. In-place safe (both 2 bytes/elem). Used by FlameSwap for FP16 model weights (e.g. Wan2.2). |
| `flame_fp16_to_bf16` | `:27` | C entry. `(input, output, n_elements, stream) → int`. |

### `src/cuda/fused_dequant_transpose.cu` — FP8 dequant + transpose fused

| Symbol | Line | Notes |
|---|---|---|
| `fp8_dequant_transpose_kernel` | `:17` | Dequant + transpose in one kernel. Used by `fp8_resident.rs` for the on-the-fly weight unpack path. |
| `flame_fused_dequant_transpose_bf16` | `:93` | C entry. |

### `cuda/narrow_strided.cu` — Reference impl for SPEED_CONTRACT clause 1 (Class-E fix, commit `b552f61`)

The active narrow path. Rewritten 2026-05-12 to pass shape + strides
**inline as a kernel-arg struct** (`NarrowMeta`, sized
`FLAME_NARROW_MAX_RANK = 8`). Replaces the pre-fix shape of "per-call
`cudaMalloc + cudaMemcpyAsync + cudaStreamSynchronize + cudaFree` for the
metadata buffers" — Class E. Micro-bench: 1000 `cudaStreamSynchronize` →
0. Real-trainer (klein 9B): 268/step → 0/step. This is **Reference
Implementation #1** for the "small/fixed metadata fits in kernel-arg
space" pattern documented in
[`SPEED_CONTRACT.md`](./SPEED_CONTRACT.md) ("Two correct primitive
patterns").

| Symbol | Line | Notes |
|---|---|---|
| `struct NarrowMeta` | `:17` | `int64_t shape[8]` + `int64_t strides[8]`. Passed by value through kernel-arg space — no allocation, no copy, no sync. |
| `narrow_strided_kernel` | `:33` | Byte-copy gather; reads `meta.shape` / `meta.strides` from the inline struct. Block 256, grid `ceil(n_elements / 256)`. |
| `flame_narrow_strided_launch` | `:65` | C entry. Fills `NarrowMeta` on host, launches kernel, returns `cudaGetLastError()`. No `cudaMalloc` / `cudaMemcpyAsync` / `cudaStreamSynchronize` / `cudaFree`. |
| `narrow_backward_scatter_add_kernel` | `:105` | Inverse of the gather — byte-copy scatter from `grad_out` into `grad_in` at the narrow slice. Dtype-agnostic via `elem_size`. |
| `flame_narrow_backward_scatter_add_launch` | `:137` | C entry, same inline-meta pattern. Post-Class-B (`15d8ef8`) is called directly for BF16 + F32 — no more F32 cast detour. |

The matching `src/cuda/narrow_strided.cu` and
`src/cuda/narrow_strided_backward.cu` files are older alternative builds;
the active path is the build-time `cuda/narrow_strided.cu` listed above
(declared in `cuda/ffi.rs`, linked via `build.rs`).

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

### `src/cuda/tensor_iterator.cuh` + `src/cuda/activation_silu_iter.cu` — TensorIterator port, session 1 (2026-04-22)
- Port of PyTorch's `aten/src/ATen/cuda/detail/OffsetCalculator.cuh` + the minimal path from `aten/src/ATen/native/cuda/Loops.cuh` `gpu_kernel_impl_nocast`. Foundation for kernels 2..N of the strided-elementwise migration.
- `flame::iter::StridedOffsetCalc` — `tensor_iterator.cuh:41` — rank ≤ 6, NARGS=1 (single input arg). Plain divmod; IntegerDivider magic-constants deferred until perf becomes a gate.
- `flame::iter::launch_elementwise_strided_to_contig<InT, OutT, Op>` — `tensor_iterator.cuh:96` — templated host launcher; takes a device functor instead of a lambda to avoid `--extended-lambda`.
- `flame_silu_bf16_strided` — `activation_silu_iter.cu:55` — first migrated kernel. Entry for `ops::silu_iter::silu_bf16_iter`'s strided branch. Scalar BF16 (strided input is not guaranteed 2-aligned, so no `__nv_bfloat162` vectorization).
- `flame_gelu_bf16_strided` — `activation_gelu_iter.cu:40` — second migrated kernel (tanh approximation, matches `CUDA_GELU` in bf16_ops.rs). Same scaffolding; only the functor differs.
- ⭐ `flame_gelu_exact_bf16_kernel` — `src/cuda/unary/gelu_exact.cu:41` — **exact-erf BF16 GELU** (added 2026-05-21). Math: `y = 0.5 * x * (1 + erff(x/√2))`. Same Phase-4 scaffolding as `gelu.cu`; functor calls CUDA's built-in `erff(float)`. Matches PyTorch's bare `nn.GELU()` (default `approximate='none'`); the existing tanh-approx kernel only matches `approximate='tanh'`. Used by Cosmos-Predict2.5 inference (`inference-flame/src/models/cosmos_predict25_dit.rs::mlp`) — bare `nn.GELU()` in Python's `GPT2FeedForward` was setting a ~0.02%-per-block parity ceiling against the tanh-approx variant. **Forward-only**: no autograd registration (Cosmos inference-only; flame-core's existing `gelu_backward.cu` already implements the tanh-approx derivative — exact-erf backward would need a separate `2/√π * exp(-x²)` kernel which is unneeded). Routed via `Tensor::gelu_exact()` (BF16 contig fast path → `bf16_ops::gelu_exact_bf16_contig_direct`; else → `tensor_iterator::ops::unary::gelu_exact_bf16_iter`). Parity gate: `tests/tensor_iterator_gelu_exact_parity.rs` against PyTorch GPU reference at `tests/data/gelu_exact_ref.safetensors` (CUDA-generated by `gen_gelu_exact_ref.py` — NEVER CPU per CONTEXT.md). `Tensor::gelu` deliberately stays tanh-approx (used by Z-Image, Klein, Chroma, etc. — flipping it would invalidate trained LoRAs).
- `flame_square_bf16_strided` — `activation_square_iter.cu:35` — third migrated kernel (y = x*x, matches `CUDA_SQUARE`).
- `flame_add_bf16_strided` — `add_bf16_iter.cu:45` — first BINARY migrated kernel (session 4, 2026-04-22). Two strided input offsets, one contig output. Extends `tensor_iterator.cuh` with `launch_elementwise_strided_binary_to_contig`. Same-shape only this session (broadcast stays on legacy `launch_bf16_elementwise`).
- Perf note: contig paths are UNTOUCHED for all four (Rust dispatcher short-circuits `is_contiguous()` back to the existing vectorized NVRTC kernels). Strided path is correctness-first; 4096×4096 T-view unary ~270 µs (vs ~90-140 µs contig), binary ~470 µs. Per-kernel speedup does NOT materialize until enough kernels route through the iterator that `Tensor::narrow` can flip to view-return.
- **Gate fix (session 4)**: `bf16_elementwise::shapes_equal_no_broadcast` at `:721` now requires both operands `is_contiguous()`. Without this, `launch_bf16_flat` reads storage-base-linear on permuted BF16 views and produces semantically-wrong bytes — a live correctness bug that existed since the Phase 2a permute-as-view change. Fixed once at the gate, covers all four binary ops (add, sub, mul, div).

### `src/kernels/silu_backward.cu` — fused SiLU backward
- `flame_silu_backward_bf16` / `flame_silu_backward_f32` — single-kernel `g * sig(x) * (1 + x*(1-sig(x)))`. Same ABI as every fused unary backward kernel: `(grad_out, input, grad_in, n, stream) -> i32`.

### `src/kernels/swiglu_backward.cu` — fused SwiGLU backward
- `flame_swiglu_backward_bf16` — two outputs (`d_gate`, `d_up`) from a single kernel.

### `src/kernels/{relu,gelu,tanh,sigmoid}_backward.cu` (2026-04-18)
- Fused unary-activation backward kernels, BF16 + F32 entrypoints each.
- GELU uses the **tanh-approximation** derivative to match the forward path.
- Signatures: `flame_<op>_backward_{bf16,f32}(grad_out, input, grad_in, n, stream) -> i32`.
- Called from `autograd.rs::fused_unary_backward` (main compute path) and
  `autograd_ops.rs::launch_unary_backward` (the `BackwardOps` façade).
- Parity tests in `flame-core/tests/activation_backward_fused_kernels.rs`.

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
| `rms_norm_kernel` | `:253` | **Block-per-row + parallel reduction RMSNorm**. Was 1-thread-per-row scalar before 2026-04. Wraps `fc_rms_norm_bf16` C entry. Legacy path — `cuda_ops_bf16::rms_norm_bf16` now delegates to `norm::rms_norm` (see commit `d729ede`), so this kernel is only reached when `norm_size % 4 != 0`. |
| `layer_norm_forward_bf16_kernel` | `:295` | LayerNorm forward (with optional gamma/beta) — legacy smem-tree reduction path. |
| `layer_norm_forward_bf16_vec_kernel` | `:298` | **Vectorized LayerNorm forward** (2026-05-12). 256 threads/block (8 warps), `bf16x4_ln` 8-byte loads, single combined mean+variance pass via warp-shuffle + 2-slot smem (`2 * n_warps` floats). Selected by `fc_layer_norm_bf16` (`:825`) when `norm_size % 4 == 0`. `FLAME_LAYER_NORM_FWD_LEGACY=1` forces the smem-tree path. 1.10–1.36× faster than legacy on flux/zimage/klein production shapes (legacy was already parallel — this is a vec+shuffle upgrade, not a 1-thread fix). |
| `group_norm_compute_stats_bf16_kernel` | `:361` | GroupNorm 1st pass (mean/var per group) — legacy smem-tree. |
| `group_norm_compute_stats_bf16_vec_kernel` | `:476` | **Vectorized GroupNorm stats** (2026-05-12). One block per `(n, group)`, vec=4 `bf16x4_gn` loads along the contiguous spatial axis, warp-shuffle intra-warp + smem inter-warp reduction. Selected by `fc_group_norm_bf16` (`:967`) when `spatial_size % 4 == 0` (VAE blocks always satisfy). `FLAME_GROUP_NORM_STATS_LEGACY=1` forces the legacy path. 1.44–2.28× faster end-to-end on VAE shapes (larger spatial sizes win more — bandwidth-bound). The apply kernel `group_norm_forward_bf16_kernel` was unchanged (already coalesced; backward still has the 1-thread-per-group bug). |
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

### `cuda/gemm_bf16_fp32acc.cu` — BF16 strided-batched GEMM (trainer hot path)

| Symbol | Line | Notes |
|---|---|---|
| `gemm_bf16_fp32acc_stridedBatched` | `extern "C"` | BF16 in/out, FP32 accumulate, row-major, strided-batched. Used by `matmul_bf16_trans` and `bmm_bf16_fp32acc_out` — i.e. every Linear forward/backward and every SDPA Q@K^T / P@V in the BF16 trainer path. Per-shape cache (op desc + 3 layouts + preference + winning algo) keyed on `(m,n,k,lda,ldb,ldc,strideA,strideB,strideC,batchCount,opA,opB)`. Persistent per-device workspace pool (default 0; opt-in via `FLAME_GEMM_BF16_WORKSPACE_BYTES`). Cache rollback knob: `FLAME_HANDLE_TLS_DISABLE=1` reverts to build-everything-per-call. Cache pattern mirrors `gemm_bf16_cublaslt.cu::lt_matmul_run`. |

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
| `slice_copy_kernel` | `:49` | Strided slice copy. Generic up-to-8-D div/mod unravel per output element — ALU-bound (~225 GB/s) and pure waste when the result is a contiguous chunk. |
| `index_select_kernel` | `:76` | Gather along axis 0. |
| `fc_bf16_slice / fc_bf16_index_select` | `:140, :196` | C entries. |

**Leading-axis fast path** (2026-05-12, `fc_bf16_slice` `:190`): when `axis == 0`
AND input is fully row-major contiguous, the slice is a single contiguous chunk
of the input. Dispatch one `cudaMemcpyAsync(dst, src + start*inner, len*inner*2)`
instead of the generic kernel — hits ~800–850 GB/s (82–89% of RTX 3090 Ti peak
HBM), 3.4–4.0× faster than the legacy kernel on production shapes. Auditor
flagged 440–3206 calls/step depending on pipeline. `FLAME_SLICE_COPY_LEGACY=1`
forces the kernel for A/B benchmarking.

### `cuda/upsample_nearest.cu`

| Symbol | Line | Notes |
|---|---|---|
| `upsample2d_nearest_nchw_kernel` | `:17` | Nearest upsample, NCHW BF16. |
| `fc_upsample2d_nearest_bf16` | `:52` | BF16 entry |
| `fc_upsample2d_nearest_f32` | `:87` | F32 entry |

### `cuda/upsample_bilinear.cu`

2D bilinear upsample over NCHW tensors, matches PyTorch's
`F.interpolate(mode='bilinear')` (both `align_corners=True` and
`=False`). Templated over element type; BF16 inputs are upcast to F32
for the weighted-sum compute and rounded back on store (don't
accumulate 4 taps in a 7-bit mantissa). Parity vs a CPU reference in
`tests/upsample_bilinear_parity.rs`: F32 cos_sim = 1.0, BF16 cos_sim
= 0.999999 / max_abs ≤ 8e-3.

Reached through `CudaKernels::upsample2d_bilinear` (the F32/BF16
dtype-dispatch wrapper in `src/cuda_kernels.rs:1810`), which is what
`upsampling::Upsample2d { mode: Bilinear }` dispatches into.

| Symbol | Line | Notes |
|---|---|---|
| `upsample2d_bilinear_nchw_kernel` | `:29` | Templated BF16/F32 bilinear kernel. |
| `fc_upsample2d_bilinear_bf16` | `:108` | BF16 entry |
| `fc_upsample2d_bilinear_f32` | `:141` | F32 entry |

### `cuda/permute0213.cu`

| Symbol | Line | Notes |
|---|---|---|
| `permute0213_kernel` | `:16` | `[B, N, H, D] → [B, H, N, D]` — legacy scalar grid-strided fallback. |
| `permute0213_vec4_bf16_kernel` | `:66` | **Vectorized + 4D-grid permute0213** (2026-05-12). `bf16x4` 8-byte loads/stores along the `c` axis (head_dim). 4D grid `(ceil(C/4/TX), ceil(A/TY), N*B)` with block `(min(C_VEC, 32), min(A, 32), 1)` — no per-element divmod since `n` and `b` are decoded from `blockIdx.z` once. Selected by `launch_permute0213_bf16` (`:118`) when `C % 4 == 0` (production head_dims 64/128/256 all qualify). `FLAME_PERMUTE_LEGACY=1` forces the legacy kernel. Beats stock PyTorch by 1.5–1.8× on a 3090 Ti for standalone permutes (PyTorch only exceeds when the permute fuses into a cuBLAS matmul prelude). |
| `permute021_kernel` | `:143` | `[N, A, B] → [N, B, A]` — legacy scalar grid-strided. ~135 GB/s on attention shapes (1/8 of peak — within-warp coalesced on `b` but across-warp scatters across `a`-rows separated by stride `B`). |
| `permute021_tiled_kernel<Scalar>` | `:175` | **Tiled-transpose permute021** (2026-05-12, F32 + BF16 via template). 32×32 smem tile with `tile[32][33]` bank-conflict padding. Phase 1: coalesced read `input[n, a0+ty, b0+tx] → tile[ty][tx]` (warp varies in `b`, stride 1). Phase 2: coalesced write `output[n, b0+ty, a0+tx] = tile[tx][ty]` (warp varies in `a`, stride 1 in output). Selected by `launch_permute021_{f32,bf16}` when `A >= 16 AND B >= 16`. `FLAME_PERMUTE_LEGACY=1` forces the legacy. |
| `permute021_tiled_small_kernel<Scalar, TH, TW>` | `:298` | **Asymmetric-tile transpose021** (2026-05-12, late). Covers small-A or small-B cases (one dim ≤ 16) that the 32×32 path can't handle without wasting threads. Block dim `(TW, TH)`; tile `[TH][TW+1]`. Phase 1: coalesced read on the big-B side. Phase 2: writes are stride-A (partially uncoalesced) but phase 1 dominates. Two specializations dispatched by `launch_permute021_dispatch`: `TH=8, TW=64` for small-A and `TH=64, TW=8` for small-B. |
| `launch_permute10_{bf16,f32}` | `:412/:427` | **Rank-2 `[1,0]` matrix transpose entry points** (2026-05-12). Internally `launch_permute021_dispatch` with `N=1`. Covers the 840 calls/step of `permute_generic` `(rank=2, perm=[1,0])` in zimage backward (matmul weight-grad). Micro-bench: 3.4× faster than the generic scatter kernel on `[8, 3840]`/`[3840, 8]`/`[10240, 8]`/`[8, 10240]` shapes (8.98 → 2.62 µs/call). |
| `launch_permute0132_{bf16,f32}` | `:455/:481` | **Rank-4 `[0,1,3,2]` inner-2 swap entry points** (2026-05-12). Collapses to rank-3 `[0,2,1]` with `N' = N*A` (the outer two dims are contiguous so the flatten is exact). Covers 120 calls/step in zimage backward (`[1,30,1536,128]` QK^T transpose and `[1,30,1536,1536]` attention-weights transpose). Micro-bench: 1.63× faster on `[1, 30, 1536, 128]` (97.3 → 59.4 µs), 1.69× on `[1, 30, 1536, 1536]` (1241 → 732 µs). |

### `cuda/sdpa_stream_bf16.cu` — chunked SDPA (legacy) + cached workspace (Class-E fix, commit `542c531`)

The streaming SDPA path used by `sdpa_stream_bf16`. ⚠️ Still the
catastrophically slow attention path for d=64 and causal — see
PERF_SDPA_FLASH_KERNEL.md — but as of 2026-05-12 the *launch wrapper*
no longer allocates per call. Reference impl for SPEED_CONTRACT clause 1's
**"variable/large workspace, cached per-device buffer"** pattern (mirrors
`gemm_bf16_cublaslt.cu`'s cuBLASLt workspace cache).

Mask API: callers pass a binary keep mask, not an additive attention bias.
Mask values `>= 0.5` mean "attend"; values `< 0.5` mean "block". The
accepted mask shape is `[B|1, H|1, Q, K]` BF16/F32/Bool and is expanded by
zero stride over broadcast batch/head dimensions.

| Symbol | Line | Notes |
|---|---|---|
| `ker_cast_bf16_to_f32` | `:42` | Per-tile BF16 → FP32 cast. |
| `ker_cast_f32_to_bf16` | `:47` | Inverse. |
| `ker_apply_scale_mask_rowmax` | `:54` | Apply scale and binary keep mask (`1=attend`, `0=block`), then compute row max. |
| `ker_row_exp_and_sum` | `:93` | Compute exp + row sum. |
| `ker_update_l_and_factors` | `:109` | Update l_new and the per-row scale factors. |
| `ker_row_normalize` | `:137` | Normalize scores by 1/l. |
| `ker_scale_O_by_beta` | `:149` | Scale running O by beta. |
| `ker_fill_constant` | `:157` | Fill helper. |
| `struct SdpaStreamWorkspace` | `:197` | Per-device cache: device id + shape `(q_t, k_b, d)` + single fused buffer holding all 11 sub-allocations (`scores`, `row_sums`, `m_prev/curr`, `l_prev/curr`, `inv_l_new`, `beta`, `O_accum`, `scores_T`, `scores_norm_bf16`). 256-byte aligned per slice. |
| `sdpa_ws_layout` | `:228` | Computes total fused-buffer size + 11 sub-buffer offsets. Used both for grow-decision and pointer fixup after alloc. |
| `sdpa_ws_acquire` | `:253` | Lookup-or-grow the per-device entry under `std::mutex`. On shape grow: one `cudaMalloc(needed)`, one `cudaFree` of prior buffer. Steady state: zero allocations. |
| `sdpa_ws_get_cublas_handle` | `:319` | Process-singleton `cublasHandle_t` per device. `cublasCreate` is ~ms; called once per attention forward across every transformer block, that's real wall time. Now created once, reused. Freed only at process exit. |
| `sdpa_stream_bf16_launch` | `:345` | C entry. Acquires workspace, gets cached handle, runs the chunked online-softmax loop. **Was**: 11 `cudaMalloc` + 11 `cudaFree` + 1 `cublasCreate` per call. **Is**: zero allocations on the hot path. |

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
| `layer_norm_backward_kernel` | `:26` | LayerNorm backward (training) — legacy 1-thread-per-row scalar (same shape as the RMSNorm legacy that was just rewritten). 4 sequential passes per thread: mean, var, sum1+sum2, then dx + inline atomicAdd into both `dgamma[i]` and `dbeta[i]` per element per row (zimage [4096, 2560] = 10.5M atomicAdds × 2 targets per call). |
| `layer_norm_backward_bf16_vec_kernel` | `:40` | **Vectorized LayerNorm backward** (2026-05-12). 256 threads/block, `bf16x4_t` 8-byte loads, warp-shuffle + smem inter-warp reductions. 3-pass: (1) parallel mean+sum_sq, (2) `sum1=sum(dy*g)` and `sum2=sum(dy*g*xn)`, (3) vectorized `dx` write only. dgamma/dbeta deferred to the cross-row kernel below. Mean/var recomputed inline (no API change to plumb from forward). |
| `layer_norm_grad_weight_bias_bf16_vec_kernel` | `:189` | **Cross-row dgamma/dbeta reduction** (2026-05-12). Tile `COLS_PER_BLOCK=64` × `ROWS_PER_BLOCK=128`. Each block: 64 threads collaboratively reduce each row's mean/var (2-warp warp-shuffle + smem broadcast via `s_mean`/`s_inv_std`), then accumulate F32 dgamma/dbeta in a single atomicAdd per `(col, row_tile)` per target. ~500× fewer atomicAdds than the inline path. Per-row mean/var is still recomputed across col_tiles (future fix: precompute into `[batch_size]` scratch). |
| `group_norm_backward_kernel` | `:82` | GroupNorm backward — still the 1-thread-per-group legacy path (auditor flagged 20–50× available; same fix pattern as RMSNorm bwd, deferred). |
| `fc_layer_norm_backward_bf16` | `:172` | C entry. Dispatches to vec kernel pair when `norm_size % 4 == 0` (`:451`). `FLAME_LAYER_NORM_LEGACY=1` forces the scalar path. Measured 1.8–3.4× faster than legacy on production shapes (smaller multipliers than rms_norm_bwd because LN has two atomicAdd targets vs RMSNorm's one). |
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

Benchmarked 2026-04-12 on RTX 3090 Ti vs PyTorch 2.8.0 (100 warmup, 200 timed, CUDA events, BF16).

| Kernel | Flame (μs) | PyTorch (μs) | Ratio | Notes |
|---|---|---|---|---|
| `abs_bf16_kernel` | 7.2 | 17.4 | 0.4× | Sign-bit clear. 2.4× faster than PT. |
| `add_bf16_flat` | 11.3 | 17.4 | 0.6× | Vectorized BF16 add. Beats PT. |
| `mul_bf16_flat` | 11.3 | 17.4 | 0.6× | Vectorized BF16 mul. Beats PT. |
| `silu_bf16_kernel` | 34.8 | 32.6 | 1.07× | At parity. Was 24× before pool fix. |
| `gelu_bf16_kernel` | 36.9 | 31.7 | 1.16× | At parity. Was 24× before pool fix. |
| `fc_layer_norm_bf16` | 32.8 | 31.7 | 1.03× | At parity. |
| `softmax_lastdim_bf16_kernel` | 157 | 104 | 1.5× | Kernel itself is 147μs. Pool overhead adds ~10μs. |
| MatMul (proj, 3D×2D) | 61.4 | 70.4 | 0.9× | cuBLASLt. Beats PT. |
| MatMul (FFN) | 203 | 195 | 1.04× | At parity. |
| BMM (QK^T) | 154 | 119 | 1.3× | Acceptable. |
| BMM (@V) | 70 | 76 | 0.9× | Beats PT. |

**14/17 ops within 1.5× of PyTorch. 10 ops faster than PyTorch.**

### Catastrophically slow (still need fixes)

| Kernel | Per-call time | Notes |
|---|---|---|
| `sdpa_stream_bf16` (causal d=64) | 110-215 ms | Blocks LTX-2 / Wan / HunyuanVideo temporal attention. Needs wmma + causal mask path. |
| `sdpa_stream_bf16` (with mask, d=64) | ~9 ms | T5 path. Same wmma fix would help. |

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
