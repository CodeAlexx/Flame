# flame-core symbol index

> Flat list of public symbols → `file:line` + 1-line description, grouped by
> module. The first place to look when you need to know "where is X" or "is
> there already a function for Y."
>
> **Liveness**: ⭐ = used by `inference-flame` (live), ⚠️ = legacy /
> training-only / dead code, plain = utility/framework. There are ~1700
> public items in flame-core spread across ~80 files; this index covers the
> ones you actually need to find. For the rest, `grep -rn "pub fn name"` and
> the [`FLAME_MODULES.md`](./FLAME_MODULES.md) overview tell you which file
> to look in.

---

## Core types and re-exports

`lib.rs` re-exports everything you usually need:

| Symbol | Where it lives | Notes |
|---|---|---|
| ⭐ `Tensor` | `tensor.rs:135` | The central type. 114+ methods across 3 impl blocks. |
| ⭐ `TensorId` | `tensor.rs` | Newtype for autograd tape keying. |
| ⭐ `Shape, D` | `shape.rs:9+` | Shape vec wrapper, dim helper enum. |
| ⭐ `Strides, ShapeDims` | `shape.rs` | `SmallVec<[usize;6]>` — inline storage for dims/strides. `Shape::strides()` and `Tensor::strides()` return `Strides`, not `Vec<usize>`, so kernel launchers never heap-allocate to read a tensor's strides. |
| ⭐ `DType` | `dtype.rs:4+` | `BF16 / F16 / F32 / I32 / I64 / Bool / U8`. |
| ⭐ `Error, Result, FlameError` | `error.rs:7` | Single error enum, `Result<T, Error>`. |
| ⭐ `CudaDevice` | re-export of `cudarc::driver::CudaDevice` | |
| ⭐ `global_cuda_device()` | `device.rs:42` | Singleton `Arc<CudaDevice>` for device 0. |
| `Device, DeviceEnum` | `device.rs:56,152` | Device wrapper enum. |
| `init()` | `lib.rs:268` | Auto-runs at load via `#[ctor::ctor]`. |
| `Module` trait | `lib.rs:239` | Layer trait: `forward(&self, x) -> Result<Tensor>`. |

### Config / strict / telemetry

| Symbol | File:line | Notes |
|---|---|---|
| `default_dtype() / set_default_dtype` | `config.rs:23,32` | Process-wide default; defaults to BF16. Override via `FLAME_DEFAULT_DTYPE`. |
| `should_use_cudnn() / set_force_cudnn` | `config.rs:12,17` | cuDNN gating flag. `FLAME_FORCE_CUDNN=1`. |
| `optimizer_moment_dtype() / set_optimizer_moment_dtype` | `config.rs:42,52` | Optimizer state dtype (default F32). |
| `select_optimizer_state_dtype(param_dtype)` | `config.rs:70` | Helper for picking moment dtype per param. |
| `FlameConfig` | `config.rs:86` | Compound config struct. |
| `strict::is_enabled / scope / GuardMode` | `strict.rs` | "Strict mode" — bans implicit F32 fallbacks and clones. Toggle via env vars. |
| `strict::allow_clone / allow_f32_in_kernel` | `strict.rs` | RAII overrides. |
| `telemetry::TelemetrySnapshot / record_*` | `telemetry.rs` | Counters for dtype traps, tensor bytes. |
| `perf_telemetry` (module) | `perf_telemetry.rs` | Wider perf metrics. |

---

## `tensor.rs` — the central Tensor struct

114 methods. The most-used:

### Construction
- `Tensor::zeros(shape, device)` — F32 zeros
- `Tensor::zeros_dtype(shape, dtype, device)` — typed zeros
- `Tensor::empty_dtype(shape, dtype, device)` — uninitialized (use only after explicit fill)
- `Tensor::ones(shape, device)`
- `Tensor::randn(shape, mean, std, device)` — F32 (or default dtype)
- `Tensor::randn_seeded(shape, mean, std, seed, device)` — `tensor.rs:1128`.
  Deterministic Box-Muller sibling of `randn` using
  `rand::rngs::StdRng::seed_from_u64(seed)`. Two calls with identical args
  produce bit-identical output, independent of the global RNG state set by
  `rng::set_seed`. Use when matching a Python/torch reference (LanPaint,
  diffusers, element-wise parity tests). Output dtype mirrors `randn`.
- `Tensor::from_vec(data, shape, device)` — F32
- `Tensor::from_vec_dtype(data, shape, device, dtype)` — typed
- `Tensor::from_f32_to_bf16(data, shape, device)` — convenience
- `Tensor::from_slice / from_data` — variants
- `Tensor::rand_like / zeros_like` — match shape

### Shape / metadata
- `.shape() -> &Shape`
- `.dtype() -> DType`
- `.device() -> &Arc<CudaDevice>`
- `.numel() / .ndim() / .id()`

### View / shape ops (zero-copy when possible)
- `.reshape(&[usize])`
- `.view(&[isize])` — with -1 inference
- `.unsqueeze(dim)` / `.squeeze(Some(dim))` / `.squeeze_dim(dim)`
- `.permute(&[dims])` — uses `GpuOps::permute_generic` fallback for non-fast-path orders
- `.transpose() / .t() / .transpose_dims(d0, d1)`
- `.narrow(dim, start, len)`
- `.chunk(num, dim)` — returns `Vec<Tensor>`
- `.as_strided(shape, strides, offset)` ⭐ — zero-copy view primitive used by
  narrow/chunk and parity tests. No autograd; caller records op.
- `.cat(&[&Tensor], dim)` — `Tensor::cat` static
- `.expand(&[usize])` — broadcast view
- `.flatten / .flatten_to_2d`

### Math (most go through GpuOps or BF16 paths)
- `.add(&Tensor) / .sub / .mul / .div / .maximum / .minimum` — BF16 routes through the TensorIterator pipeline (`tensor_iterator::ops::binary::*_bf16_iter`); F32 routes through `GpuOps`
- `.add_scalar(f32) / .mul_scalar / .sub_scalar / .div_scalar / .mul_scalar_inplace` — BF16 through `tensor_iterator::ops::binary::{add,mul}_scalar_bf16_iter`
- `.matmul(&Tensor)` — 2D matmul (cuBLASLt for BF16)
- `.bmm(&Tensor)` — 3D batched matmul
- `.silu / .gelu / .relu / .sigmoid / .tanh / .neg / .abs / .square` — BF16 through `tensor_iterator::ops::unary::*_bf16_iter`
- `.exp / .log / .sqrt / .rsqrt / .recip` — BF16 through `tensor_iterator::ops::transcendentals::*_bf16_iter` (f32-opmath inside)
- `.ge / .gt / .le / .lt / .eq / .ne` — BF16 through `tensor_iterator::ops::comparison::*_bf16_iter` (output is BF16 0.0/1.0)
- `.softmax(dim)` — fast-path dispatches to `bf16_elementwise::softmax_lastdim_bf16` for BF16 last-dim
- `.clamp(min, max)` — `tensor_ops_extended.rs:677`. Element-wise clamp via
  `maximum`/`minimum`. Output dtype always equals source dtype (fix 2026-04:
  previously built min/max constants via `full_like`, which applied
  `default_dtype()` and broke F32 clamps when the workspace default was BF16).
- `.maximum(&Tensor) / .minimum(&Tensor)` — `tensor_ops_extended.rs:691,731`.
  Element-wise max/min with broadcasting. Require matching dtypes (no implicit
  cast).
- `.sum / .mean / .max / .min / .var / .std`
- `.sum_dim / .sum_dim_keepdim / .mean_dim / .max_dim`

### Cast
- `.to_dtype(DType)` — generic cast
- via `ops::cast::{cast_bf16_to_f32, cast_f32_to_bf16}` — explicit fast paths

### Materialize / read back
- `.to_vec() / to_vec_f32() / to_vec_bf16() / to_vec_i32()`
- `.item() -> f32` — scalar tensor → host scalar
- `.contiguous()` — force contig copy. Propagates `requires_grad` and records
  `Op::Reshape { new_shape = input shape }` as an identity-reshape backward, so
  autograd flows through `narrow → to_dtype` chains (fix 2026-04-23 Phase 2a)

### Storage / pointer access (low-level)
- `.as_device_ptr_bf16(label) -> *const u16`
- `.as_mut_device_ptr_bf16(label) -> *mut u16`
- `.storage_ref() / .storage_mut()`

⚠️ **Stride hazard**: these return the storage's offset-0 pointer
without honoring `view_offset` or `custom_strides`. Anyone launching
a kernel that reads via these MUST contiguify non-contig inputs first.
See [`FLAME_CONVENTIONS.md`](./FLAME_CONVENTIONS.md#stride-hazards-in-kernel-paths)
for the audited chokepoints (`fetch_saved`, `clone_result`,
`CudaKernels::{add,mul,div}`, `add_same_dtype`, `mul_same_dtype` —
all materialize views).

### View materialization
- `.is_contiguous() -> bool` — `custom_strides.is_none() && view_offset == 0`
- `.contiguous()` — propagates requires_grad and records identity-reshape
  for autograd. Routes views through `materialize_view` /
  `permute_generic` / fast-paths.
- `.clone_result() -> Result<Tensor>` — fallible deep clone. **Now safe
  for views** (commit 05f07f9): non-contig inputs are routed through
  `.contiguous()` first; pre-fix it was duplicating parent storage with
  the view's smaller logical shape, producing wrong addressing.
- `.alias() -> Tensor` — non-owning shallow view. Preserves
  `custom_strides` + `view_offset` (fix in commit 8678680; pre-fix it
  zeroed both, breaking save-for-backward of strided views).

### Autograd hooks
- `.requires_grad / .requires_grad_(bool)`
- `.backward() / .backward_with_grad()`
- `.detach()`
- See [`FLAME_MODULES.md`](./FLAME_MODULES.md) `autograd_v3` section for the active engine.
- ⭐ **`AutogradContext::retain_intermediate_grads(ids)` /
  `take_retained_intermediate_grads()`** — test-only API for probing
  intermediate gradients during backward. Used by
  `parity_klein_full_single_block_prod_diag` to bisect bug-#4-class
  hazards. See `src/autograd.rs:910-940`.

---

## Attention / SDPA — multiple paths!

This is a critical area with several implementations. **Use these for inference**:

### ⭐ The live API (use these)
- `flame_core::attention::sdpa(q, k, v, mask)` — `attention/sdpa.rs:521`
  Public dispatcher. Routes BF16 to wmma flash kernel (`flash_attention_fwd.cu`),
  F32 to fallback. **This is what `inference-flame` model files call.**
- `flame_core::attention::sdpa_with_bias(q, k, v, bias, scale)` — `attention/sdpa.rs:542`
  T5-style additive bias variant. Same dispatch but accepts a `[*, H|1, Q, K]` bias tensor.
- `flame_core::attention::attend(q, k, v, mask)` — `attention/sdpa.rs:534` — alias for sdpa
- `flame_core::attention::attention_impl(...)` — `attention/sdpa.rs:395` — lower-level impl
- `flame_core::sdpa::forward(q, k, v, mask)` — `sdpa.rs:94`
  Used directly by `inference-flame::vae::ldm_decoder` and `vae::wan21_vae`
  for cases where the dispatch overhead isn't wanted.
  **2026-04 update**: the BF16 path now auto-routes to the streaming kernel
  when `B * H * Q * K > FLAME_SDPA_STREAM_THRESHOLD` (default 2·10⁹
  elements). Materialized fallback would allocate a multi-GB F32 scores
  tensor and OOM on 24 GB cards for LTX-2 stage-2 self-attn (11 k tokens).
  The threshold is env-tunable. `FLAME_SDPA_FORCE_STREAM=1` still forces
  the stream for any shape.
- `flame_core::sdpa::forward_with_bias(...)` — `sdpa.rs:125`
- `flame_core::cuda_ops_bf16::sdpa_stream_bf16(q, k, v, mask, chunk, causal, scale)` — `cuda_ops_bf16.rs:1599`
  The chunked streaming SDPA used by LTX-2. Takes a `causal` flag and chunk size.
  **Note**: this is the catastrophically slow path for d=64 / causal — see
  PERF_SDPA_FLASH_KERNEL.md.

### ⚠️ Legacy / training-only
- `attention/sdpa_legacy.rs` — old impl, keep for reference, do NOT call
- `sdpa_legacy.rs` (top-level) — same
- `sage_attention.rs` — experimental sage attention
- `sdpa::forward_v4(...)` — `sdpa.rs:291` — gated on `autograd_v4` feature

### 🧠 Training path (autograd-recorded SDPA)
- `sdpa::forward_train(q, k, v, mask)` — `sdpa.rs:105`
  Called from `sdpa::forward` when `AutogradContext::is_recording()` and
  any input requires grad. Routes unmasked BF16 head_dim ∈ {64, 96, 128}
  through `flame_cudnn_sdpa_bf16_train_fwd` (emits O + Stats in one graph
  execute), records `Op::FlashAttention`. Backward then calls
  `flame_cudnn_sdpa_bwd_bf16` via `autograd::try_cudnn_sdpa_backward`.
  Unsupported shapes fall through to the decomposed recompute.

### Helper structs (in `attention/sdpa.rs`, used by training paths)
- `AttentionConfig` — `:83`
- `MultiHeadAttention` — `:108`
- `AttentionBuffers<'a>` — `:118`
- `RotaryEmbedding` — `:696`
- `TransformerBlock` — `:812` (training-only)
- `LayerNorm` — `:891` (legacy duplicate; prefer `layer_norm::LayerNorm`)
- `GeGLU` — `:561`
- `FeedForward` — `:597`

### RoPE
- `attention/rope.rs` — RoPE precompute + apply helpers
- ⭐ `bf16_ops::rope_fused_bf16(x, cos, sin)` — `bf16_ops.rs:476`
  The interleaved-pair (FLUX/Klein/LTX/HunyuanVideo/QwenImage/Chroma) format.
- `bf16_ops::rope_halfsplit_bf16(x, cos, sin)` — `bf16_ops.rs:656`
  The halfsplit (Z-Image/some Klein variants) format.
- ⚠️ **`Op::RoPePrecomputed` backward dispatches by `cos` shape**
  (commit dfe85b8): broadcast cos `[1,_,N,half]` →
  `rope_fused_bf16` (Interleaved); per-head cos `[BH,N,half]` →
  `rope_halfsplit_bf16` (Halfsplit). Pre-fix it unconditionally called
  halfsplit, giving cos_sim ≈ 0.05 backward gradients on Klein
  (orthogonal-direction at correct magnitude — magnitude-only checks
  hid this for sessions). See `src/autograd.rs:2515-2570`.
- ⚠️ **Interleaved `rope_fused_bf16` autograd-recording fix** (commit
  fa3291e). Pre-fix: output had `requires_grad: false` hardcoded and no
  `Op::RoPePrecomputed` recording, severing Q/K LoRA gradient chains in
  every trainer using interleaved RoPE (Klein, Z-Image, Chroma, Wan,
  FLUX). Pre-fix Klein and Z-Image LoRAs are corrupt — Q_B and K_B
  stayed exactly at zero-init while V_B (skips RoPE) trained normally.
  Halfsplit variant always recorded; only the interleaved variant was
  missing the recording. Re-train pre-fix LoRAs. See
  `src/bf16_ops.rs:735-757`.

---

## Norms

### LayerNorm
- ⭐ `layer_norm::layer_norm(x, weight, bias, normalized_shape, eps)` — `layer_norm.rs:308`
  Functional API. Used by Z-Image / SD3 model code.
- `layer_norm::layer_norm_into(...)` — `layer_norm.rs:426` — output-into variant
- `layer_norm::LayerNorm` (struct) — `layer_norm.rs:37`
- `layer_norm::LayerNormConfig` — `layer_norm.rs:20`
- ⭐ `cuda_ops_bf16::layer_norm_bf16(x, gamma, beta, eps)` — `cuda_ops_bf16.rs:316`
  Direct BF16 call (used by FLUX `linear_norm_no_affine` helper).
- `cuda_ops_bf16::layer_norm_bf16_with_stats / layer_norm_bf16_into_with_stats` — variants returning mean/rstd for backward
- `cuda_ops_bf16::layer_norm_backward_bf16` — backward (training)

### RMSNorm
- ⭐ `cuda_ops_bf16::rms_norm_bf16(x, weight, eps)` — `cuda_ops_bf16.rs:263`
  The main entry. Wraps `fc_rms_norm_bf16` (cuda_ops.cu). Has the
  block-per-row + parallel reduction kernel as of 2026-04 (was 1-thread-per-row scalar).
- `cuda_ops_bf16::rms_norm_bf16_to_f32(x, eps)` — `cuda_ops_bf16.rs:296` — F32 output variant
- ⭐ `ops::fused_inference::fused_rms_norm(x, weight, eps)` — `ops/fused_inference.rs:116`
  Direct call to `flame_fused_rms_norm_bf16` kernel (`src/cuda/fused_rms_norm.cu`).
  Used by Z-Image NextDiT.

### GroupNorm
- ⭐ `group_norm::group_norm(x, groups, gamma, beta, eps)` — `group_norm.rs:24`
  Functional. Used by SDXL UNet, Klein VAE, LDM VAE, LTX-2 audio VAE, LTX-2 upsampler.
- `group_norm::GroupNorm` (struct) — `group_norm.rs:674`
- `cuda_ops_bf16::group_norm_bf16(x, gamma, beta, groups, eps)` — `cuda_ops_bf16.rs:619`
  ⚠️ NHWC layout only — see CONVENTIONS for the layout trap.
- `cuda_ops_bf16::group_norm_bf16_with_stats` — for backward
- `cuda_ops_bf16::group_norm_backward_bf16` — training

### Other
- `norm.rs` — older norm wrappers (BatchNorm-style, training)

---

## Linear / GEMM / matmul

### ⭐ The live linear path (FLUX, Chroma, QwenImage, Klein, LTX-2)
- `ops::fused_inference::fused_linear3d(input, weight, bias)` — `ops/fused_inference.rs:190`
  cuBLASLt 3D linear. Weight must be **pre-transposed** to `[Cin, Cout]`.
- `ops::fused_inference::fused_linear3d_native(input, weight, bias)` — `ops/fused_inference.rs:275`
  **Same but takes weight in standard PyTorch `[Cout, Cin]` row-major layout.**
  Uses cuBLASLt `TRANSA=T` to do the transpose inside the GEMM. **This is what
  every FLUX/Chroma/QwenImage block forward calls.** Added 2026-04 to kill the
  per-call `transpose2d_bf16` cost.
- C side: `flame_linear3d_bf16` / `flame_linear3d_bf16_native` in
  `src/cuda/fused_linear3d.cu`.

### Other linear / GEMM
- `linear::Linear / linear::linear(in, out, bias, device)` — `linear.rs:11+` —
  the `nn::Linear` struct (training).
- `cuda_ops_bf16::gemm_bf16(x, w, bias)` — `cuda_ops_bf16.rs:1019` — wraps `fc_gemm_bf16`.
- `cuda_ops_bf16::gemm_bf16_into(...)` — output-into variant
- `blas::gemm_bf16_fp32(...)` — `blas.rs:6` — cuBLASLt BF16+FP32-acc raw call
- `ops::gemm` / `ops::gemm_bf16` — broadcast helpers around the above

### Matmul on Tensor (auto-route)
- `Tensor::matmul(&Tensor)` — 2D, autograd-aware
- `Tensor::bmm(&Tensor)` — 3D batched
- These dispatch to the BF16 path when both inputs are BF16.

---

## Conv

### ⭐ Live (used by inference-flame)
- `cuda_ops_bf16::conv2d_bf16(...)` — `cuda_ops_bf16.rs:1310` — top-level dispatcher.
  Has autotune cache and routes to cuDNN when available.
- `cudnn::cudnn_conv2d_bf16` (re-exported as `cudnn::conv2d::cudnn_conv2d_bf16`) — `cudnn/conv2d.rs:62`
  Direct cuDNN BF16 conv2d. Used by LTX-2 audio VAE and ltx2_upsampler.
- `conv::Conv2d` (struct) — `conv.rs:43` — the main Conv2d layer.
  - `Conv2d::new / new_with_bias / new_zeroed / new_with_bias_zeroed`
  - `Conv2d::forward(input)` — NCHW
  - `Conv2d::forward_nhwc(input)` — NHWC fast path
- `conv::Conv2dConfig` — `conv.rs:20`
- `conv::conv2d_forward(...)` — `conv.rs` — functional API
- ⭐ `conv1d::conv1d(x, w, bias, stride, padding, dilation, groups)` — `conv1d.rs:17`
  BF16 1D conv via cuDNN conv2d with H=1. `dilation` is plumbed through
  (fixed 2026-04 — previously silently dropped).
- ⭐ `conv1d::conv_transpose1d(x, w, bias, stride, padding, output_padding, groups)` — `conv1d.rs:83`
  BF16 1D transposed conv. Implemented via `zero_insert → cuDNN conv1d` with a
  flipped + transposed weight. Supports arbitrary `stride`, `padding`,
  `output_padding`, `dilation` (via `conv_transpose1d_dilated`), and `groups`.
  Bit-exact vs PyTorch (max|Δ| ≤ 0.008 BF16) across BigVGAN configs and
  grouped anti-alias filters.
- `conv1d::conv1d_grouped(x, w, stride, padding, groups)` — thin no-bias wrapper over `conv1d`.
- ⭐ `conv3d_bf16::Conv3dBF16` — `conv3d_bf16.rs:183` — 3D conv used by LTX-2 audio VAE +
  Wan / QwenImage 3D VAEs and the LatentUpsampler. `forward()` now dispatches to
  cuDNN first (2026-04), falls back to im2vol+GEMM only on cuDNN refusal.
  Supports `dilation` and `groups` (groups only via cuDNN; fallback rejects).
  - `Conv3dBF16::from_weights(..)` / `from_weights_with_config(..)` — new
    config ctor accepts `dilation` + `groups`.
- ⭐ `cudnn::cudnn_conv3d_bf16(input, weight, bias, stride, padding, dilation, groups)`
  — `cudnn/conv3d.rs` — direct cuDNN NCDHW BF16 Conv3d forward. FP32
  accumulate, algo cache keyed by full descriptor fingerprint, workspace
  capped by `FLAME_CUDNN_CONV3D_WS_LIMIT_MB` (default 256). Used by the
  Conv3dBF16 dispatch; call directly for lower-level control.
- `cudnn::descriptors::FilterDescriptor::set_nd(..)` / `ConvolutionDescriptor::set_nd(..)`
  — 5D descriptors needed for Conv3d.
- `conv3d_simple::*` — F32 conv3d fallback
- `conv3d::*` — older conv3d (training)

### ⚠️ Legacy / training-only
- `cuda_conv2d.rs / cuda_conv2d_direct.rs / cuda_conv2d_fast.rs / cuda_conv2d_kernels.rs` —
  multiple older conv2d implementations. Don't call directly; go through `conv::Conv2d`.
- `ops/conv2d.rs / ops/conv2d_bf16.rs / ops/conv2d_bf16_cudnn.rs` — alternative paths;
  feature-gated, mostly training.

---

## BF16 family — the inference hot path

These modules are the BF16 inference primitives. They live in
`src/bf16_*.rs` (NVRTC kernels in inline string consts) and
`src/cuda/fused_*.cu` (build-time compiled kernels).

### `bf16_elementwise.rs` — fused/structured survivors (post-TensorIterator)
Historically the flat-path elementwise home; after the Phase 1–11 TensorIterator
port, this file only hosts fused and memory-layout kernels. Pointwise
`add / sub / mul / div / max / min / ge / gt / le / lt / eq / ne` live under
`tensor_iterator::ops` now.
- ⭐ `softmax_lastdim_bf16(x)` — `:152` — fused last-dim softmax (no scratch
  alloc). Wired into `Tensor::softmax` BF16 fast path.
- ⭐ `transpose2d_bf16(t)` — `:232` — 2D BF16 transpose (used by Klein/Mistral
  pre-transpose).
- `patchify_bf16 / unpatchify_bf16` — `:374,426` — DiT patch ops.

### `tensor_iterator/ops/` — BF16 elementwise via PyTorch-style TensorIterator (Phases 4–11)
All entries are `pub fn <op>_bf16_iter(...)` and route through the shared
dispatch registry in `tensor_iterator/dispatch.rs`.
- `unary.rs` — ⭐ `silu_bf16_iter` `:58`, ⭐ `gelu_bf16_iter` `:102`,
  ⭐ `square_bf16_iter` `:144`, `abs_bf16_iter` `:186`,
  `relu_bf16_iter` `:228`, `sigmoid_bf16_iter` `:270`,
  `tanh_bf16_iter` `:312`, `neg_bf16_iter` `:354`.
- `transcendentals.rs` — `exp_bf16_iter` `:46`, `log_bf16_iter` `:88`,
  `sqrt_bf16_iter` `:130`, `rsqrt_bf16_iter` `:172`,
  `recip_bf16_iter` `:214`. (f32-opmath inside: bf16→f32→op→`__float2bfloat16_rn`.)
- `binary.rs` — ⭐ `add_bf16_iter` `:54`, ⭐ `sub_bf16_iter` `:97`,
  ⭐ `mul_bf16_iter` `:140`, `div_bf16_iter` `:183`,
  `maximum_bf16_iter` `:226`, `minimum_bf16_iter` `:269`,
  `mul_scalar_bf16_iter` `:289`, `add_scalar_bf16_iter` `:331`.
- `comparison.rs` — `ge_bf16_iter` `:48`, `gt_bf16_iter` `:91`,
  `le_bf16_iter` `:134`, `lt_bf16_iter` `:177`, `eq_bf16_iter` `:220`,
  `ne_bf16_iter` `:263`. Output dtype is BF16 0.0/1.0 (not u8), matching
  the pre-port `GpuOps::compare_binary` contract.

### `bf16_ops.rs` — fused inference primitives (+ oracle references)
- `gelu_bf16(x)` — `:133` — NVRTC contig fast path. Retained as the parity
  oracle for `tensor_iterator/ops/unary.rs::gelu_bf16_iter`; not on the live
  inference path.
- `square_bf16(x)` — `:170` — same role for `square_bf16_iter`.
- `silu_bf16(x)` — `:322` — same role for `silu_bf16_iter`.
- `softmax_last_dim_bf16(x)` — `:264` — older fused softmax (one block per row).
- ⭐ `rope_fused_bf16(x, cos, sin)` — `:476` — interleaved-pair RoPE.
- ⭐ `rope_fused_bf16_f32pe(x, cos, sin)` — `:567` — RoPE with F32 positional embeddings.
- `rope_halfsplit_bf16(x, cos, sin)` — `:656` — halfsplit RoPE.
- `modulate_pre_fused_bf16(...)` — `:895` — DiT shift+scale modulation.
- `modulate_pre_split_apply_bf16(...)` — `:961` — B.3 split+apply variant.
- ⭐ `gate_residual_fused_bf16(x, gate, attn_out)` — `:1089` — `x + gate * attn_out`.
- ⭐ `swiglu_fused_bf16(gate, up)` — `:1156` — `silu(gate) * up`.
- `attn_split_txt_img_bf16(...)` — `:1246` — attention output text/image split.
- `qkv_split_permute_bf16(...)` — `:1642` — QKV split + permute.

### `bf16_convert.rs` — BF16↔F32 cast
- `bf16_u16_to_f32(...)` — `:54` — vectorized via `__nv_bfloat162` (2-element/thread)
- `f32_to_bf16_u16(...)` — `:70`
- (The Rust call site is `ops::cast::cast_bf16_to_f32 / cast_f32_to_bf16`.)

### `bf16_normal.rs` — Gaussian noise generator
- `normal_bf16(...)` — Box-Muller in BF16 directly

### `bf16_factories.rs`
- `uniform_bf16(...)` — uniform random
- Other BF16 tensor factories

### `bf16_clamp.rs`
- `clamp_bf16(...)` — element clamp

### `bf16_support.rs` — feature gate / capability checks

---

## Fused inference primitives — `ops/fused_inference.rs`

The "kernel calls that bypass autograd entirely". Used by every FLUX-style block.

| Function | Line | What it does |
|---|---|---|
| ⭐ `dequant_fp8_to_bf16` | `:16` | FP8 → BF16 dequant (one shot) |
| ⭐ `dequant_fp8_to_bf16_into` | `:45` | Same, output-into |
| ⭐ `dequant_fp8_transpose_into` | `:78` | Dequant + transpose in one kernel |
| ⭐ `fused_rms_norm` | `:116` | RMSNorm with weight, single kernel |
| ⭐ `fused_modulate` | `:155` | `(1+scale) * x + shift` — DiT modulate |
| ⭐ `fused_linear3d` | `:190` | cuBLASLt 3D linear (pre-transposed weight) |
| ⭐ `fused_linear3d_native` | `:275` | cuBLASLt 3D linear (PyTorch weight layout, TRANSA=T) |
| ⭐ `fused_rms_norm_modulate` | `:350` | RMSNorm + modulate fused |
| ⭐ `fused_residual_gate` | `:388` | `x + gate * attn` fused |

**All of these go through `crate::cuda::ffi::flame_*_bf16` declarations and
the `.cu` files in `src/cuda/`.**

---

## CUDA infrastructure

### `cuda/ffi.rs` — Rust FFI declarations
The `extern "C"` block declaring all the C-side `flame_*` symbols. Look here
to see what kernels are linked in. Notable groups:
- `flame_narrow_strided_launch / flame_narrow_backward_scatter_add_launch` (`:10,15`) — narrow ops
- `flame_cuda_alloc_pinned_host / flame_cuda_free_pinned_host / flame_cuda_memcpy_async / flame_cuda_host_register / flame_cuda_host_unregister` (`:83-94`) — pinned memory + async copy
- `flame_rope_apply_bf16_fp32` (`:225`) — RoPE kernel (legacy, used by training)
- `flame_apply_causal_mask_fp32 / flame_apply_attn_mask_fp32` (`:238,249`) — SDPA mask kernels
- `flame_sdpa_add_mask_tile_fp32` / `flame_sdpa_softmax_from_lse_tile` / `flame_sdpa_lse_from_logits_tile` / `flame_sdpa_lse_merge_rows` / `flame_sdpa_dropout_bf16_inplace` (`:259-303`) — chunked SDPA primitives
- `flame_geglu_pointwise_fp32` (`:313`) — GeGLU
- `fc_upsample2d_nearest_bf16 / fc_upsample2d_nearest_f32` (`:382,394`) — VAE upsample
- `fc_upsample2d_bilinear_bf16 / fc_upsample2d_bilinear_f32` (`:509,522`) — bilinear 2D upsample (BF16 + F32), PyTorch-matching index math with `align_corners`. Added 2026-04-19 to unblock Cascade.
- `flame_fp8_to_bf16` (`:409`) — FP8 dequant
- `flame_fp16_to_bf16` (`:416`) — FP16 → BF16 conversion (in-place safe). Used by BlockOffloader for FP16 checkpoints.
- `flame_flash_attention_bf16` (`:424`) — wmma flash attention forward (LIVE, inference dead-code fallback only; training uses cuDNN)
- `flame_cudnn_sdpa_bf16` — cuDNN v9 SDPA inference forward (primary inference attention path; see `src/cuda/cudnn_sdpa.cpp`)
- `flame_cudnn_sdpa_bf16_train_fwd` — cuDNN v9 SDPA training forward. Emits O + Stats (per-row LSE) so backward can skip recompute. Added Phase 2c (2026-04-23).
- `flame_cudnn_sdpa_bwd_bf16` — cuDNN v9 SDPA backward (`src/cuda/cudnn_sdpa_bwd.cpp`). Reads Stats from train-fwd. Replaces the removed `flame_flash_attention_backward_bf16` WMMA kernel and the decomposed-recompute backward. Added Phase 2c.
- `flame_fused_rms_norm_modulate_bf16` (`:434`)
- `flame_fused_residual_gate_bf16` (`:448`)
- `flame_fused_rms_norm_bf16` (`:459`)
- `flame_fused_modulate_bf16` (`:471`)
- `flame_fused_dequant_transpose_bf16` (`:482`)
- `flame_linear3d_bf16` (`:494`)
- `flame_linear3d_bf16_native` (`:513`) — added 2026-04

### `cuda_ops_ffi.rs` — `fc_*` FFI symbols
The `fc_*` family is from `cuda/cuda_ops.cu` and friends. Different naming
convention (`fc_status_t` returns), different file generation:
- `fc_relu_bf16 / fc_gelu_bf16 / fc_silu_bf16` (`:90-92`)
- `fc_axpby_bf16` (`:93`)
- `fc_layer_norm_bf16` (`:100`) + backward
- `fc_group_norm_bf16` (`:123`) + backward
- `fc_rms_norm_bf16 / fc_rms_norm_bf16_to_f32` (`:148,155`)
- `fc_gemm_bf16 / fc_batched_gemm_bf16` (`:161,168`)
- `fc_conv2d_bf16` (`:175`)
- Workspace + arena: `fc_ws_ensure_capacity`, `flame_arena_alloc / record / destroy` (`:89,273-280`)
- Async copy: `flame_h2d_async / flame_d2h_async / flame_d2d_async / flame_bf16_zero_async / flame_bf16_copy_async` (`:281-300`)
- Autotune: `flame_conv2d_autotune_get_stats / reset_stats`, `flame_sdpa_autotune_get_stats / reset_stats / flush_cache` (`:306-310`)
- `flame_sdpa_chunked_bf16` (`:311`) — chunked SDPA C entry
- NHWC↔NCHW: `flame_nhwc_to_nchw_*` / `flame_nchw_to_nhwc_*` (`:331-358`)
- `flame_conv2d_nhwc_bf16` (`:367`)
- `flame_status_to_result(status, op)` (`:566`) — error mapper

### `cuda/device_lt.rs` — cuBLASLt + stream helpers
- `device_lt::stream_ptr(device)` — get the default stream pointer
- `device_lt::cublaslt_handle_ptr(device)` — get the cuBLASLt handle (cached per device)

### `cuda/dtype_tag.rs` — DType <-> CUDA dtype tags

### `cuda/utils.rs` — small CUDA helpers

### `cuda/kernels.rs` — early F32 kernel wrappers
- `mul_scalar / add / mul / fill / copy / mse_loss` — F32 only, training/legacy

### `cuda_kernels.rs` — `CudaKernels` struct (training)
- 64 `pub fn` methods. Wraps NVRTC-loaded F32 kernels for the training path.
- `CudaKernels::add / mul / mul_scalar / relu / relu_backward / mse_loss / mse_backward / fill / copy` etc.
- ⚠️ This is the F32 training-side. Inference uses BF16 paths.

### `cuda_kernels_gpu.rs` — F32 GPU kernels (alternative)
- 38 `pub fn` methods. Older `GpuOps` path. The `Tensor::add` etc. fallback when both inputs aren't BF16.
- ⚠️ Mostly training/legacy.

### `cuda_kernel_compiler.rs` / `cuda_kernel_sources.rs`
- NVRTC compiler wrapper + a list of kernel source string consts
- ⚠️ Older compile path; new BF16 NVRTC kernels use the inline-string-then-`compile_ptx_with_opts` pattern in each module

### `cuda_ops.rs` — `GpuOps` namespace
- 59 `pub fn` methods on `GpuOps`. F32 ops surface used by the autograd v3 engine.
- `GpuOps::add / sub / mul / div / matmul / sum_dim_keepdim / max_dim / mean_dim / permute_generic / materialize_view` etc.
- ⚠️ Most paths are training-only; `permute_generic` is the live fallback used by `Tensor::permute` for non-fast-path orders.
- `GpuOps::materialize_view` ⭐ — materializes any strided-plus-offset view
  into contiguous row-major. Called by `Tensor::contiguous()` when
  `view_offset != 0`. Dispatches to `materialize_strided_{f32,bf16}_kernel`.

### `cuda_ops_bf16.rs` — the BF16 op surface (LIVE)
- See "Norms" / "Conv" / "Linear" sections above for the live entries.
- Plus: `slice_axis_bf16`, `broadcast_to_bf16`, `repeat_axis_bf16`, `repeat_nd_bf16_into`,
  `index_select_bf16_into`, etc.
- `SdpaWorkspace` (`:49`) — pre-allocated workspace for `sdpa_stream_bf16`
- `Conv2dAutotuneStats` / `SdpaAutotuneStats` — perf telemetry

---

## Serialization

- ⭐ `serialization::load_file<P>(path, device)` — `:555` — load a safetensors file as `HashMap<String, Tensor>`
- ⭐ `serialization::load_file_filtered<P, F>(path, device, filter_fn)` — `:570` — same but a closure picks which keys to load
- ⭐ `serialization::save_file(tensors, path)` — `:690` — save a HashMap to safetensors
- ⭐ `serialization::save_tensors(tensors, path, format)` — `:61`
- `serialization::load_tensors(path, format, device)` — `:73`
- `serialization::save_tensor(tensor, path, format)` — `:41`
- `serialization::load_tensor(path, format, device)` — `:49`
- `serialization::SerializationFormat` — `:33` — `SafeTensors / Bincode`

---

## Memory / staging

### `memory_pool.rs` — F32 memory pool
- 15 pub fns, training/legacy primarily

### `pinned.rs` — pinned host memory
- `PinnedHostBuffer / PinnedHostBufferView / PinnedHostBufferViewMut`
- `PinnedAllocFlags`
- `StagingDeviceBuf` — staging buffer
- `register_slice_as_pinned / unregister_pinned` — register existing memory
- `memcpy_async_device_to_host / memcpy_async_host_to_device`

### `pinned_pool.rs`
- `PinnedPool` — pool of pinned host buffers (re-exported)

### `staging.rs` (BF16-only, gated)
- 16 pub fns. BF16 arena + async copy primitives. Used internally by `Tensor` for some hot paths.
- `bf16_copy_async / ArenaLease`

### `cuda_memory_alignment.rs`
- `alloc_aligned_f32(...)` — aligned F32 alloc (used by tensor.rs)

---

## Activation offload — `activation_offload.rs`

Push GPU activations to pinned host RAM during forward, pull them back during
backward. Foundation of the "offload instead of recompute" checkpoint path.

| Symbol | File:line | Notes |
|---|---|---|
| `ActivationOffloadPool` | `activation_offload.rs:319` | Pool of pinned host buffers with a dedicated non-blocking CUDA transfer stream. Construct once at training setup. |
| `OffloadHandle` | `activation_offload.rs:293` | Opaque `Copy` handle returned by `push`, consumed by `pull`. Carries slot index + epoch for stale-handle detection. |
| `OffloadCompression` | `activation_offload.rs:89` | `None` (raw BF16/F32) or `FP8` (halves pinned memory + PCIe via BF16-to-FP8 quantize on transfer stream). |
| `ActivationOffloadPool::push(tensor)` | `activation_offload.rs:465` | Async DtoH on transfer stream. Gates on default-stream event. Returns handle. |
| `ActivationOffloadPool::pull(handle)` | `activation_offload.rs:619` | Async HtoD on transfer stream. Makes default stream wait via ready event. Frees slot. |
| `ActivationOffloadPool::clear()` | `activation_offload.rs:742` | Reset all slots to Idle, bump epoch (invalidates all outstanding handles). No host sync. |
| `OffloadedTapeEntry` | `autograd.rs:339` | Sub-tape entry with saved tensors replaced by `OffloadHandle`s. |
| `AutogradContext::checkpoint_offload(inputs, f)` | `autograd.rs:1338` | Run forward, capture sub-tape, offload saved tensors, record `Op::CheckpointOffload`. |
| `set_activation_offload_pool(pool)` | `autograd.rs:56` | Install global pool once at training setup. |

---

## Autograd — multiple generations, **read carefully**

### Active engine (`autograd_v3.rs` per the comment in lib.rs:153)
- `autograd::AutogradContext / Op` (re-export from autograd.rs)
- `AutogradContext::record_op(out_id, op, saved_tensors)` — register a node on the tape
- `AutogradContext::set_enabled(bool)` — global on/off
- `Tensor::backward()` — entry point

### `autograd_v4` (feature gated)
- `autograd_v4::*` — newer experimental engine. Off by default.
- `autograd_v4::ops::sdpa` — SDPA backward via v4

### Legacy / dead
- ⚠️ `autograd.rs` (top-level) — types still re-exported
- ⚠️ `autograd_simple.rs` — early stub
- ⚠️ `autograd_engine.rs` — older engine
- ⚠️ `autograd_ops.rs / autograd_ops_complete.rs` — older op set
- ⚠️ `autograd_debug.rs` — debug helpers

### Activation offload (v2.1)
- `Op::CheckpointOffload { input, sub_tape }` — `autograd.rs:325` — captures
  the forward sub-tape and offloads all saved tensors to CPU. Backward pulls
  them back and walks the sub-tape (no recompute).
- `AutogradContext::checkpoint_offload(inputs, f)` — `autograd.rs:1338` —
  public entry. Runs closure with autograd, captures sub-tape, offloads saved
  tensors. Falls back to standard `checkpoint()` if pool unavailable.
- `set_activation_offload_pool(pool)` — `autograd.rs:56` — install global pool
  (once, at training setup). Used by `flame-diffusion/src/offload.rs`.
- `OffloadedTapeEntry` — `autograd.rs:339` — tape entry with saved tensors
  replaced by `OffloadHandle`s + optional `resident_fallback` for non-BF16.

### Block offloading (flame-diffusion)
- `BlockOffloader` — `flame-diffusion/src/block_offload.rs` — double-buffered pinned CPU→GPU block offloader
- `BlockFacilitator` trait — `flame-diffusion/src/block_offload.rs` — model geometry provider
- `prefetch_block(idx)` — async H2D to non-active slot
- `await_block(idx)` → `Arc<HashMap<String, Tensor>>` — wait + prepare
- `ensure_block(idx)` — sync API (prefetch + await)
- `KleinFacilitator` — `klein-trainer/src/facilitator.rs`
- `ChromaFacilitator` — `chroma-trainer/src/facilitator.rs`
- `WanFacilitator` — `wan-trainer/src/facilitator.rs`
- `Wan22Dit::load_shared_only` — `inference-flame/src/models/wan22_dit.rs` — shared-only constructor (no block weights)

### Gradient utilities
- `gradient::GradientMap / TensorGradExt` — re-exported as `GradientMap`
- `gradient_clip::*` — gradient clipping
- `gradient_checkpointing::*` — activation checkpointing helpers

---

## Optimizers

- `adam::AdamW` — re-exported as `nn::AdamW`. Standard AdamW with BF16 master / F32 moments; `set_lr()` supports runtime schedulers.
- `sgd::*` — basic SGD
- `parameter::Parameter` — re-exported as `Var` and `Parameter`. Wraps a `Tensor` with `requires_grad=true`.
- `nn::Optimizer` trait — `lib.rs:258` — `step()` + `zero_grad()`

---

## NN building blocks (mostly training; some used by inference)

- ⭐ `nn::Linear` (`linear.rs:Linear`)
- ⭐ `nn::Embedding` (`embedding.rs`)
- ⭐ `nn::LayerNorm` (`layer_norm.rs:LayerNorm`)
- ⭐ `nn::Conv2d` (`conv.rs:Conv2d`)
- `linear::linear(in, out, bias, device)` — functional Linear constructor
- `cuda_conv2d::conv2d(...)` — re-exported in `nn::conv2d`
- `activations::*` — element-wise activation fns
- `pooling::*` / `pooling_impl::*` — pooling layers
- `loss::*` — loss functions (training)
- `regularization::*` — dropout, etc.
- `samplers::*` — diffusion samplers (older Karras/Euler implementations)

---

## Misc

- `lora::*` — LoRA adapter helpers (training)
- `mixed_precision::*` — fp16/bf16 amp helpers
- `embedding::Embedding` — token embedding
- `image_ops_nhwc::*` — image space ops in NHWC
- `upsampling::*` — 2D upsample (nearest / bilinear — both BF16 + F32). Bilinear kernel `cuda/upsample_bilinear.cu` added 2026-04-19; backed `UpsampleMode::Bilinear` was an `Err("not yet implemented")` prior.
- `vae::autoencoder_kl::*` / `vae::zimage_decoder::*` — generic VAE pieces (Z-Image specific)
- `kernels::adaln::*` — AdaLN kernel (feature-gated)
- `fused_kernels::*` — older fused kernel registry (training)
- `fp16::*` — F16 conversion helpers
- `tensor_compute::*` — small compute helpers
- `tensor_ext.rs` — `to_owning_fp32_strong / slice_channels / pad_channels`
- `tensor_narrow.rs` — narrow helper
- `tensor_ops_extended.rs` — extra Tensor ops (57 pub fns)
- `tensor_ops_missing.rs` — fill-ins for missing ops (`upsample_nearest2d`, `div_scalar`, etc.)
- `ops_ext.rs` — small `OpResult`-typed helpers (`shape4 / matmul_tt / where_mask / mean_all_f32`)
- `ops/utils.rs` — helper utilities for the `ops` family
- `borrowed/mod.rs` — feature-gated borrowed-weight tensor variant
- `python/*` — feature-gated PyO3 bindings
- `capi.rs` — feature-gated C API surface
- `debug_device.rs` — `assert_cuda(tag, t) / log_device(tag, t)`
- `debug_finite.rs` — `FLAME_DEBUG_FINITE=1` tripwire. `is_enabled() / reset() / check(site, t)`. When enabled, `check` syncs + scans the tensor for NaN/±Inf, logs a finite-range summary per call-site, errors on the first non-finite and self-disables so the trace isn't spammed. Training bring-up diagnostic — used by `autograd::backward` to tag per-op grads automatically.
- `logging.rs` — logging setup
- `env_flags.rs` — env var caching
- `kernel_launcher.rs` — `LaunchConfig` helpers
- `bf16_support.rs` — capability check helpers
- `rng/mod.rs` — `global_rng() / set_seed(seed)` — RNG state
- `devtensor.rs` — old per-device tensor wrapper
- `cuda_tensor.rs / cuda_tensor_gpu.rs / cuda_tensor_with_cublas.rs` — old standalone CUDA tensor types
  ⚠️ These predate the unified `Tensor`, do not use.

---

## Bins (test/debug binaries — not for production)

`src/bin/*.rs`:
- `basic_ops_test.rs / minimal_test.rs / minimal_flame_test.rs` — sanity checks
- `debug_autograd.rs / test_backward.rs / test_complex_backward.rs / test_grad_propagation.rs / test_sum_backward.rs` — autograd tests
- `flame_backward_probe.rs` — backward debugging
- `perf_test.rs` — perf bench

---

## C / CUDA extern functions

See [`FLAME_KERNELS.md`](./FLAME_KERNELS.md) for the kernel inventory grouped
by `.cu` file with launch configs and perf notes.

---

## Quick lookup recipes

- **"Where is the BF16 fast-path matmul?"** → `ops::fused_inference::fused_linear3d_native`
- **"Where is the SDPA dispatcher I should call from a model?"** → `attention::sdpa`
- **"Where do I add a new BF16 elementwise op?"** → `tensor_iterator/ops/{unary,binary,transcendentals,comparison}.rs` +
  a `.cu` functor under `src/cuda/{unary,binary,cmp}/` — see CONVENTIONS for the template
- **"Where is the cuDNN SDPA shim?"** → `src/cuda/cudnn_sdpa.cpp` (inference + training fwd), `src/cuda/cudnn_sdpa_bwd.cpp` (backward)
- **"Where is the wmma flash attention kernel?"** → `src/cuda/flash_attention_fwd.cu` (forward only; bwd was deleted in Phase 2c)
- **"Where do I add a new fused C kernel?"** → `src/cuda/fused_*.cu` + `src/cuda/ffi.rs` declaration +
  `ops/fused_inference.rs` Rust wrapper
- **"Where is the load_file used by every inference binary?"** → `serialization::load_file_filtered`
- **"Where is the global RNG seed?"** → `rng::set_seed`
- **"Where is the FP8 dequant?"** → `ops::fused_inference::dequant_fp8_to_bf16` →
  `flame_fp8_to_bf16` → `src/cuda/fp8_dequant.cu`
  `flame_fp16_to_bf16` → `src/cuda/fp16_to_bf16.cu`
- **"Where is the activation offload pool?"** → `activation_offload::ActivationOffloadPool` →
  autograd integration via `autograd::checkpoint_offload` + `Op::CheckpointOffload`.
  FP8 quant kernel: `src/cuda/fp8_quant.cu`. Trainer setup: `flame-diffusion/src/offload.rs`.
- **"Where is the BF16→FP8 quantize kernel?"** → `flame_bf16_to_fp8` →
  `src/cuda/fp8_quant.cu` (used by activation offload FP8 compression)
- **"Where are the QwenImage trainer parity tests?"** →
  Forward: `flame-diffusion/qwenimage-trainer/src/bin/parity_test.rs` +
  `tools/dump_forward.py`.
  Training: `src/bin/train_parity_test.rs` + `tools/dump_training_steps.py`.
  Sampler: `tools/compare_sampler.py`. See CONVENTIONS §7-9 for bugs found.
