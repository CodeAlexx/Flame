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
| `device::trim_cuda_mempool(min_keep_bytes)` | `device.rs:42` | Release cached freed VRAM back to the driver. Pass 0 to release everything not in use. |
| `device::cuda_peek_last_error() -> i32` | `device.rs:51` | Non-clearing peek at the per-thread last cudaError_t. |
| `device::cuda_probe(tag) -> i32` | `device.rs:60` | Sync + read+clear: `cudaDeviceSynchronize` (catches async errors) THEN `cudaGetLastError` (catches latched launch-validation errors). Prints when nonzero, used to bisect which kernel set a sticky error. |
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
- `Tensor::eye(n, device)` — `n×n` F32 identity — `tensor_ops_extended.rs:1131`
- `Tensor::eye_dtype(n, dtype, device)` — typed identity (BF16/F16/F32) —
  `tensor_ops_extended.rs:1138`. Used by OFT-Neumann series
  `R = I + 2Q + 2Q^2 + ...` in the LyCORIS family.
- `Tensor::from_parts(storage, shape, device, id, custom_strides, view_offset) -> Tensor`
  — `pub(crate)` constructor added 2026-05-12 (Phase 2 groundwork). Builds a
  `Tensor` from already-owned parts without re-bumping inner Arcs that the
  caller already cloned. Mirrors `at::Tensor::Tensor(TensorImpl)`. Caller is
  responsible for `Arc::clone` semantics. Sets `requires_grad=false` (saved
  tensors are detached leaves in backward).

### Shape / metadata
- `.shape() -> &Shape`
- `.dtype() -> DType`
- `.device() -> &Arc<CudaDevice>`
- `.numel() / .ndim() / .id()`

### View / shape ops (zero-copy when possible)
- `.reshape(&[usize])`
- `.view(&[isize])` — with -1 inference
- `.unsqueeze(dim)` / `.squeeze(Some(dim))` / `.squeeze_dim(dim)`
- `.permute(&[dims])` — uses `GpuOps::permute_generic` fallback for non-fast-path orders. Fast-path (`CudaKernels::permute_fastpath`, `cuda_kernels.rs:3005`) routes rank-2 `[1,0]` (`launch_permute10_*`) and rank-4 `[0,1,3,2]` (`launch_permute0132_*`) to tuned tiled kernels; rank-3 `[0,2,1]` and rank-4 `[0,2,1,3]` are routed upstream by `Tensor::contiguous` to `GpuOps::permute_021` / `permute_0213`. Bypass via `FLAME_PERMUTE_FASTPATH=0`.
- `.transpose() / .t() / .transpose_dims(d0, d1)`
- `.narrow(dim, start, len)` — zero-copy view; Arc-clones parent storage
- `.narrow_owning(dim, start, len)` ⭐ — like `narrow` but materializes into
  fresh contiguous storage via `cuda_ops::GpuOps::materialize_view`. No
  short-circuit; result is independent of parent. Use in chunked-decode
  loops where keeping multi-GB parent storage alive would fragment GPU heap
- `.chunk(num, dim)` — returns `Vec<Tensor>`
- `.as_strided(shape, strides, offset)` ⭐ — zero-copy view primitive used by
  narrow/chunk and parity tests. No autograd; caller records op.
- `.cat(&[&Tensor], dim)` — `Tensor::cat` static
- `.expand(&[usize])` — broadcast view
- `.flatten / .flatten_to_2d`

### Indexing — gather / scatter / assign
- `.index_select(dim, &indices)` — `tensor_ops_extended.rs:568`. Gather rows
  along `dim`. BF16 fast path via `cuda_ops_bf16::index_select_bf16_into`,
  F32 via `GpuOps::index_select` + `INDEX_SELECT_KERNEL`. Backward via
  `Op::IndexSelect` (uses `cuda_kernels::scatter_add` to splat upstream).
- ⭐ `.index_assign(dim, &indices, &values)` — `tensor_ops_extended.rs:680`.
  Returns a NEW tensor where slices at `indices` along `dim` are replaced
  by the corresponding slices of `values`; non-indexed positions are
  copied from `self`. F32 + BF16 paths via NVRTC kernels
  `index_assign_f32_kernel` / `index_assign_bf16_kernel`. Backward
  `Op::IndexAssign`: grad_input = upstream with indexed rows zeroed
  (computed by re-applying `index_assign_no_grad` with zero values),
  grad_values = `index_select(upstream, dim, indices)`. Used by TREAD's
  scatter-back step in `eridiffusion-core/training/features/tread.rs`.
- `.index_assign_no_grad(dim, &indices, &values)` — forward-only variant
  used internally by autograd (`tensor_ops_extended.rs:706`).

### Math (most go through GpuOps or BF16 paths)
- `.add(&Tensor) / .sub / .mul / .div / .maximum / .minimum` — BF16 routes through the TensorIterator pipeline (`tensor_iterator::ops::binary::*_bf16_iter`); F32 routes through `GpuOps`
- `.add_scalar(f32) / .mul_scalar / .sub_scalar / .div_scalar / .mul_scalar_inplace` — BF16 through `tensor_iterator::ops::binary::{add,mul}_scalar_bf16_iter`
- `.matmul(&Tensor)` — 2D matmul (cuBLASLt for BF16)
- `.bmm(&Tensor)` — 3D batched matmul
- `.silu / .gelu / .relu / .sigmoid / .tanh / .neg / .abs / .square` — BF16 through `tensor_iterator::ops::unary::*_bf16_iter`
- `.silu_structured()` — Phase 4 exemplar (added 2026-05-12). Same forward+backward as `.silu()`; demonstrates PyTorch meta+impl split via `structured::SiluStructured`. Test: `tests/structured_silu_parity.rs`.
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
- `.to_dtype(DType)` — generic cast. `tensor.rs:752`. As of 2026-05-12
  (commit `1332019`) has direct-call fast paths for the two hot cases:
  BF16→F32 and F32→BF16 (both contiguous source). The fast path allocates
  the output buffer directly and dispatches a single `bf16_convert` kernel
  via `bf16_to_f32_u16` / `f32_to_bf16_u16`, skipping the legacy
  F32-staging round-trip (`alloc_aligned_f32` + `storage.to_f32` +
  `dtod_copy` + optional second conversion = 2–3 kernels + 2–3 allocs).
  ~16–34× faster on production cast shapes. All other dtype combinations
  still hit the staging path.
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
- `.detach() -> Result<Tensor>` — deep-copy storage, fresh `TensorId`,
  `requires_grad=false`, no `record_op`. Breaks the autograd tape; backward
  through the original does NOT flow into the detached copy. Used by DoRA
  (norm of `W_orig + ΔW` is detached per the paper). `tensor.rs:3052`.
- `.detach_leaf() -> Tensor` — Arc-bump (zero copy), fresh `TensorId`,
  `requires_grad=true`. Equivalent to PyTorch `detach_variable`; used by
  gradient checkpointing to make a leaf for a local recompute graph.
  `tensor.rs:3068`.
- See [`FLAME_MODULES.md`](./FLAME_MODULES.md) `autograd_v3` section for the active engine.
- ⭐ **`AutogradContext::retain_intermediate_grads(ids)` /
  `take_retained_intermediate_grads()`** — test-only API for probing
  intermediate gradients during backward. Used by
  `parity_klein_full_single_block_prod_diag` to bisect bug-#4-class
  hazards. See `src/autograd.rs:1395` / `:1420`.
- ⭐ **`AutogradContext::retain_intermediate_grads_add(ids)`** — additive
  variant of the above (2026-05-20). Extends the existing retain set
  instead of replacing. Required when probe IDs are registered *during*
  a checkpoint recompute closure: the outer-tape snapshot at
  `autograd.rs:2058` already fired before the closure runs, but the
  sub-tape walk re-reads `RETAINED_INTERMEDIATE_GRAD_IDS` inside
  `Op::Checkpoint` (`autograd.rs:3551`) and `Op::CheckpointOffloadBoundary`
  (`autograd.rs:3800`) backward. Gate the registration with
  `AutogradContext::is_checkpoint_recompute()` so the same forward path
  doesn't double-record. See `src/autograd.rs:1412` and
  [`FLAME_DIAGNOSTICS.md`](./FLAME_DIAGNOSTICS.md) §0 + §1.

---

## Attention / SDPA — multiple paths!

This is a critical area with several implementations. **Use these from model
code**:

### ⭐ The live API (use these)
- `flame_core::attention::sdpa(q, k, v, mask)`
  Public dispatcher. Unmasked BF16 head_dim ∈ {64, 96, 128} routes to cuDNN
  SDPA. In training it records `Op::FlashAttention` with saved O/Stats so
  backward can use cuDNN instead of decomposed recompute. `mask=Some(...)` is
  the generic compatibility path for true arbitrary masks, not the hot road for
  structured model attention.
- `flame_core::attention::sdpa_causal(q, k, v)`
  Structured top-left causal self-attention. Use this instead of building a
  lower-triangular binary mask.
- `flame_core::attention::sdpa_prefix_causal_full(q, k, v, prefix_len)`
  Structured mixed self-attention: prefix rows are causal, suffix rows are full.
  HiDream-O1 uses this for its AR/text + image-token pattern. The current
  training implementation records a single `PrefixCausalFullAttention` autograd
  op: forward uses prefix causal + suffix all-ones masked attention, while
  backward recomputes as one exact prefix-causal/full masked SDPA. This avoids
  the old two-SDPA shared-K/V gradient collapse without paying the full-mask
  cost on forward.
  **2026-05-21 HiDream-O1 parity note**: the forward hot path is intentionally
  flame-core-only and uses the in-tree FA2-style BF16 kernel for supported
  head dimensions `{64, 96, 128}`. The FFI now accepts a `causal` flag, and
  the kernel uses raw logits, `exp2`/log2 softmax scaling, and reverse K/V tile
  traversal to track PyTorch FlashAttention more closely. Exact PyTorch/CUTLASS
  tile-shape parity is deferred until after the O1 trainer gate; do not claim
  trainer validity until the O1 parity smoke passes.
- `flame_core::attention::sdpa_with_bias(q, k, v, bias, scale)` — `attention/sdpa.rs:542`
  T5-style additive bias variant. Same dispatch but accepts a `[*, H|1, Q, K]` bias tensor.
- `flame_core::attention::attend(q, k, v, mask)` — `attention/sdpa.rs:534` — alias for sdpa
- `flame_core::attention::attention_impl(...)` — `attention/sdpa.rs:395` — lower-level impl
- `flame_core::sdpa::forward(q, k, v, mask)`
  Used directly by `inference-flame::vae::ldm_decoder` and `vae::wan21_vae`
  for cases where the dispatch overhead isn't wanted.
  **2026-04 update**: the BF16 path now auto-routes to the streaming kernel
  when `B * H * Q * K > FLAME_SDPA_STREAM_THRESHOLD` (default 2·10⁹
  elements). Materialized fallback would allocate a multi-GB F32 scores
  tensor and OOM on 24 GB cards for LTX-2 stage-2 self-attn (11 k tokens).
  The threshold is env-tunable. `FLAME_SDPA_FORCE_STREAM=1` still forces
  the stream for any shape.
- `flame_core::sdpa::forward_causal(q, k, v)`
- `flame_core::sdpa::forward_prefix_causal_full(q, k, v, prefix_len)`
  Public core implementation for prefix-causal-full self-attention. Default
  path is the O1-safe single-op hybrid. `FLAME_PREFIX_CAUSAL_FULL_FULL_MASK=1`
  forces the explicit full-mask control, while
  `FLAME_PREFIX_CAUSAL_FULL_STRUCTURED=1`,
  `FLAME_PREFIX_CAUSAL_FULL_SUFFIX_ONLY=1`, and
  `FLAME_PREFIX_CAUSAL_FULL_SUFFIX_MASKED=1` are diagnostic toggles.
  `FLAME_PREFIX_CAUSAL_FULL_TRY_CUDNN=1` re-enables the experimental cuDNN
  full-suffix attempt; keep it off for O1 parity unless explicitly bisecting
  cuDNN plan behavior.
- `flame_core::sdpa::forward_with_bias(...)` — `sdpa.rs:125`
- `flame_core::sdpa::forward_with_sinks(q, k, v, mask, sinks)` — `sdpa.rs` ⭐
  GPT-OSS / StreamingLLM attention with per-head learned sink logits
  concatenated as an extra "virtual key" column before softmax (then
  dropped from the V matmul). Manual FP32 path mirroring
  `forward_with_bias`. `sinks` is `[H]` BF16 or F32; mask is the same
  `[*, *, Q, K]` keep-mask semantics as `forward()`.
- `flame_core::attention::sliding_window_causal_keep_mask(seq_len, window_size, device, dtype)`
  — `attention/sliding_window_mask.rs` ⭐
  Builds a `[1, 1, S, S]` keep-mask where position `q` attends to `k`
  iff `q.saturating_sub(window_size - 1) <= k <= q`. Pure host build +
  upload (same pattern as `causal_keep_mask` in `sdpa.rs`). Used by
  GPT-OSS alternating sliding/full attention layers.
- `flame_core::cuda_ops_bf16::sdpa_stream_bf16(q, k, v, mask, chunk, causal, scale)` — `cuda_ops_bf16.rs:1599`
  The chunked streaming SDPA used by LTX-2. Takes a `causal` flag and chunk size.
  **Note**: this is the catastrophically slow path for d=64 / causal — see
  PERF_SDPA_FLASH_KERNEL.md.
  **Post-Class-E (2026-05-12, commit `542c531`)** the C-side launch wrapper
  `sdpa_stream_bf16_launch` in `cuda/sdpa_stream_bf16.cu` caches an 11-buffer
  fused workspace (`SdpaStreamWorkspace`) per device + a process-singleton
  `cublasHandle_t` per device (`sdpa_ws_get_cublas_handle`). Was: 11
  `cudaMalloc` + 11 `cudaFree` + 1 `cublasCreate` per call. Now: zero
  allocations on the hot path (grown only on shape change). Mirrors the
  cuBLASLt workspace cache pattern in `gemm_bf16_cublaslt.cu`. Reference
  impl for SPEED_CONTRACT clause 1's "cached per-device workspace" pattern.

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
  execute). Non-64-aligned Q/KV lengths are physically padded before cuDNN and
  sliced back afterward; real lengths are saved on `Op::FlashAttention` and
  passed into cuDNN as padding-mask sequence lengths. Backward then calls
  `flame_cudnn_sdpa_bwd_bf16` via `autograd::try_cudnn_sdpa_backward`.
  Unsupported shapes and true arbitrary masks fall through to recompute.

### Stage 2 (2026-05-12) — cuDNN SDPA backward re-enabled
**Bug fixed**: `autograd.rs:4231-4240` shape-find heuristic returned Q as O
because `fetch_saved` materializes via `.contiguous()` → fresh `TensorId`s
break the id-exclusion (`t.id != query_tensor.id`). Replaced with direct
`fetch_saved(id)` lookup using new `Op::FlashAttention { output, stats, .. }`
fields (autograd.rs:332-348).
- New tests at `tests/sdpa_bwd_parity.rs` — 8 cases: 4 same-path
  determinism + 4 cuDNN-vs-decomposed cross-path. cos ≥ 0.99996,
  max_abs_diff ≤ 1.2e-2.
- **Q/KV 64-alignment handling** — cuDNN flash bwd still wants physical
  sequence lengths aligned to 64. The dispatcher pads unmasked BF16 attention
  to satisfy cuDNN and supplies real lengths through cuDNN padding-mask
  sequence tensors, so models no longer fall back solely because token counts
  are not 64-aligned.
- Loss bit-equal at 4 decimal places vs `FLAME_NO_CUDNN_SDPA_BWD=1`
  rollback path.
- Real savings: zimage -200 ms/step (1.8 → 1.6); klein 9B -100 ms (4.7 →
  4.6, capped by alignment constraint).

⚠️ **Hidden bug class to watch**: anywhere code does
`saved_tensors.iter().find(|t| t.id != some_id)` is broken because
`fetch_saved` materializes via `.contiguous()` producing fresh ids.
Always use the saved-by-id lookup (record the id at forward time, look up
directly at backward).

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
  The halfsplit (Z-Image/some Klein variants/MagiHuman) format.
- ⚠️ **Both `rope_fused_bf16` and `rope_halfsplit_bf16` rotate the FULL last
  dim of `x`** — they compute `half = x.shape[-1] / 2` internally. For models
  that rotate only a prefix of `head_dim` (e.g. MagiHuman: head_dim=128,
  ROPE_DIM=96, last 32 channels passthrough), wrap with split→rotate→cat.
  Symptom of misuse: `Shape mismatch: expected [..., D/2_from_x], got
  [..., ROPE_DIM/2]` from the cos/sin reshape inside the kernel. See
  `inference-flame/src/models/magihuman_dit.rs::rope_partial_halfsplit`.
- ⭐ **`Op::RoPePrecomputed` backward dispatches by explicit
  `RopeLayout` tag** (2026-05-20). Variants: `RopeLayout::Interleaved` →
  `rope_fused_bf16`; `RopeLayout::Halfsplit` → `rope_halfsplit_bf16`.
  See `src/autograd.rs:4207-4230` for the dispatch. Forward sites at
  `bf16_ops.rs:923` (`rope_fused_bf16`), `bf16_ops.rs:1068`
  (`rope_fused_bf16_f32pe`), `bf16_ops.rs:1183` (`rope_halfsplit_bf16`)
  pass the correct tag. Replaces the shape-sniffing fallback (commit
  dfe85b8), which mis-classified HiDream-O1 MRoPE cos `[1,S,half]`
  (rank-3 but Halfsplit) as Interleaved — that was the HiDream-O1 Q/K
  LoRA-B grad collapse pattern. See [`RopeLayout` doc at
  `src/autograd.rs:155-177`].
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
- ⭐ `cuda_ops_bf16::layer_norm_bf16(x, gamma, beta, eps)` — `cuda_ops_bf16.rs:316`. Forward dispatches to `layer_norm_forward_bf16_vec_kernel` when `norm_size % 4 == 0` (2026-05-12, commit `774d675`); `FLAME_LAYER_NORM_FWD_LEGACY=1` forces the smem-tree path. Backward dispatches to `layer_norm_backward_bf16_vec_kernel` + the cross-row `layer_norm_grad_weight_bias_bf16_vec_kernel` (commit `4d46832`); `FLAME_LAYER_NORM_LEGACY=1` forces the legacy scalar path.
  Direct BF16 call (used by FLUX `linear_norm_no_affine` helper).
- `cuda_ops_bf16::layer_norm_bf16_with_stats / layer_norm_bf16_into_with_stats` — variants returning mean/rstd for backward
- `cuda_ops_bf16::layer_norm_backward_bf16` — backward (training)

### RMSNorm
- ⭐ `norm::rms_norm(x, normalized_shape, weight, eps)` — `norm.rs:1100`
  **Canonical RMSNorm entry for both training and inference.** Records
  `Op::RMSNorm`. As of 2026-05-12 (commit `2ebc2d1`) dispatches three new
  vectorized NVRTC kernels when `norm_size % 4 == 0` (all production shapes
  qualify):
    - `RMS_NORM_FWD_KERNEL_BF16_VEC` at `:1368` — block per row, 256 threads,
      `bf16x4` loads, warp-shuffle reduction. 13.5–16.1× faster than legacy.
    - `RMS_NORM_BWD_KERNEL_BF16_VEC` at `:1522` — same shape, writes
      `grad_input` only. 9.5–14.8× faster.
    - `RMS_NORM_GRAD_WEIGHT_KERNEL_BF16` at `:1644` — cross-row dgamma kernel
      (`COLS_PER_BLOCK=64`, `ROWS_PER_BLOCK=512`). ~500× fewer atomicAdds.
  Legacy scalar kernels (`RMS_NORM_FWD_KERNEL_BF16` / `_BWD_KERNEL_BF16`)
  remain for the `norm_size % 4 != 0` fallback. `FLAME_RMS_NORM_LEGACY=1`
  forces scalar for A/B benchmarking. Bit-exact backward against the
  primitive F32 chain (cos = 1.000000 on Z-Image shapes) — see
  `tests/rms_norm_vs_primitive_zimage.rs`. EDv2 Z-Image's `primitive_rms_norm`
  wrapper delegates here.
- `rms_norm_backward_for_bench(grad_out, input, weight, inv_rms, batch_size, norm_size)` — `norm.rs:856`
  ⚠️ **Bench-only escape hatch.** `#[doc(hidden)]`, hidden from API
  consumers. Calls `rms_norm_backward` directly without the autograd
  machinery. Used by `benches/rms_norm_vec.rs` to time the backward kernel
  in isolation. Do NOT use in production code.
- ⭐ `cuda_ops_bf16::rms_norm_bf16(x, weight, eps)` — `cuda_ops_bf16.rs:241`
  Inference entry. As of 2026-05-12 (commit `d729ede`) **delegates to
  `norm::rms_norm`** so inference picks up the same vec kernel speedup
  without a second rewrite (closed a 2× gap vs PyTorch on the inference
  path). Does NOT record autograd (caller's `x` doesn't require grad in
  inference). The older `fc_rms_norm_bf16` smem-tree kernel in
  `cuda/cuda_ops.cu` remains as the fallback inside `norm::` for shapes
  where `norm_size % 4 != 0`.
- `cuda_ops_bf16::rms_norm_bf16_to_f32(x, eps)` — `cuda_ops_bf16.rs:296` — F32 output variant
- ⭐ `ops::fused_inference::fused_rms_norm(x, weight, eps)` — `ops/fused_inference.rs:202`
  Direct call to `flame_fused_rms_norm_bf16` kernel (`src/cuda/fused_rms_norm.cu`).
  Used by Z-Image NextDiT, MagiHuman MM/Shared transformer layers.
- 💡 **`(weight + 1)` precompute pattern** (Gemma3 / MagiHuman): the kernel
  computes `out = normed * weight`, but those models want `out = normed *
  (weight + 1)`. Pre-add 1.0 to the weight at layer-load time and pass the
  precomputed tensor — saves a per-call `add_scalar(1.0)` kernel launch.
  For multi-modality variants (per-modality gain), pre-split + pre-add the
  weights into N contiguous chunks at load time; per-call forward then does
  N narrows + N fused_rms_norm calls + 1 cat (vs the 14-op cascade of
  to_dtype + mul + mean_dim + sqrt + div + per-modality narrow + add + mul +
  cat). MagiHuman: replaced ~14 op cascade taking 5 sec/call with 1 fused
  kernel taking <1 ms/call (5000× speedup at L≈1086, hidden=5120).
  See `inference-flame/src/models/magihuman_dit.rs::{precompute_w_plus_1_bf16,
  mm_rms_norm_multi_fused, mm_rms_norm_single_fused}`.

### GroupNorm
- ⭐ `group_norm::group_norm(x, groups, gamma, beta, eps)` — `group_norm.rs:24`
  Functional. Used by SDXL UNet, Klein VAE, LDM VAE, LTX-2 audio VAE, LTX-2 upsampler.
- `group_norm::GroupNorm` (struct) — `group_norm.rs:674`
- `cuda_ops_bf16::group_norm_bf16(x, gamma, beta, groups, eps)` — `cuda_ops_bf16.rs:619`
  ⚠️ NHWC layout only — see CONVENTIONS for the layout trap. Stats kernel
  dispatches to `group_norm_compute_stats_bf16_vec_kernel` (vec=4 +
  warp-shuffle) when `spatial_size % 4 == 0` (2026-05-12, commit `f3b75bb`);
  `FLAME_GROUP_NORM_STATS_LEGACY=1` forces the smem-tree path. The apply
  kernel is unchanged. Backward still has the auditor-flagged 1-thread
  bug — separate fix.
- `cuda_ops_bf16::group_norm_bf16_with_stats` — for backward
- `cuda_ops_bf16::group_norm_backward_bf16` — training

### Other
- `norm.rs` — older norm wrappers (BatchNorm-style, training)

---

## Linear / GEMM / matmul

### ⭐ The live linear path (FLUX, Chroma, QwenImage, Klein, LTX-2)
- `ops::fused_inference::fused_linear3d(input, weight, bias)` — `ops/fused_inference.rs:276`
  cuBLASLt 3D linear. Weight must be **pre-transposed** to `[Cin, Cout]`.
- `ops::fused_inference::fused_linear3d_native(input, weight, bias)` — `ops/fused_inference.rs:357`
  **Same but takes weight in standard PyTorch `[Cout, Cin]` row-major layout.**
  Uses cuBLASLt `TRANSA=T` to do the transpose inside the GEMM. **This is what
  every FLUX/Chroma/QwenImage block forward calls.** Added 2026-04 to kill the
  per-call `transpose2d_bf16` cost.
- `ops::fused_inference::fused_linear3d_native_lora(input, weight, bias, lora_a, lora_b, scale)`
  — `ops/fused_inference.rs:489`
  Additive LoRA wrapper over `fused_linear3d_native`. With both LoRA tensors
  `None`, byte-identical to the base (same kernel, same autograd). With LoRA,
  computes `out = base + scale * (x @ A^T @ B^T)` where A=[rank,Cin], B=[Cout,rank].
  Autograd flows into A and B (frozen base weight). Added 2026-05 for HiDream-O1
  LoRA injection where decoder owns weights via `HashMap<String, Tensor>`
  (decoder.rs:346-419) rather than `Linear` modules.
- ⭐ `ops::fused_inference::fused_linear3d_native_pytorch_parity(input, weight, bias)`
  — `ops/fused_inference.rs:479` (added 2026-05-20). Same signature/semantics
  as `fused_linear3d_native`; biased calls mirror PyTorch's
  `at::cuda::blas::gemm_and_bias<at::BFloat16>` cuBLASLt configuration
  bit-exactly (1 MiB workspace, per-call heuristic, BIAS_POINTER set
  before `cublasLtMatmulAlgoGetHeuristic`, no BIAS_DATA_TYPE / batch /
  alignment attrs). No-bias calls delegate to `_native` to match
  PyTorch's matmul path. ~1% perf overhead vs the non-parity variant
  for biased calls. Use this where ai-toolkit / PyTorch bit-exact per-op parity matters
  (HiDream-O1 TimestepEmbedder, BottleneckPatchEmbed). For established
  Klein / Z-Image baselines, keep using `fused_linear3d_native` to
  preserve their existing parity numbers.
- C side: `flame_linear3d_bf16` / `flame_linear3d_bf16_native` /
  `flame_linear3d_bf16_pytorch_parity` in `src/cuda/fused_linear3d.cu`
  (FFI bindings at `src/cuda/ffi.rs:906` for the parity variant).

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

### `tensor_iterator/cache.rs` — Phase 1 geometry cache (added 2026-05-12)
- `cache::IterCacheKey` — keyed on `(operand_shapes, element-strides,
  dtypes, pending-output bitmap, num_outputs, static_dtype, static_shape,
  packed flags)`.
- `cache::CachedIterGeometry` — `(shape, perm, stride_bytes per operand,
  has_coalesced_dimensions, all_ops_same_shape, requires_32bit_indexing,
  common_dtype, fast_setup, target_dtypes, logical_output_shape)`.
- `cache::cache()` — `&'static Mutex<HashMap<IterCacheKey, CachedIterGeometry>>`.
- `cache::cache_disabled()` — `OnceLock<bool>` read of `FLAME_TI_CACHE_DISABLE`.
- Hit/miss counters: `cache::record_hit / record_miss` (`AtomicU64`).
- Inserted at the top of `TensorIteratorConfig::build`: hit short-circuits
  steps 2-4 + 6-7 (compute_shape / compute_strides / reorder / coalesce /
  32bit-indexing). Steps 1 + 5 always run.

### Phase 4 — Structured-kernel pattern (added 2026-05-12)
- `structured::StructuredKernel` trait — `Input<'a>` GAT, `meta(input) ->
  Tensor` (validate + allocate), `impl_(input, output) -> Tensor` (write
  kernel result into pre-allocated output), `dispatch(input) -> Tensor`
  (meta → autograd record → impl_).
- `structured::SiluStructured` — exemplar. Routes through
  `TensorIteratorBase::build_unary_op(Some(&out), x)` so the iterator
  short-circuits its alloc path.
- `Tensor::silu_structured(&self) -> Result<Tensor>` — public entrypoint.
  Bit-identical forward + backward to `Tensor::silu`. Test:
  `tests/structured_silu_parity.rs`.

### `tensor_iterator/ops/` — BF16 elementwise via PyTorch-style TensorIterator (Phases 4–11)
All entries are `pub fn <op>_bf16_iter(...)` and route through the shared
dispatch registry in `tensor_iterator/dispatch.rs`.
- `unary.rs` — ⭐ `silu_bf16_iter` `:58`, ⭐ `gelu_bf16_iter` `:102` (tanh-approx),
  ⭐ `gelu_exact_bf16_iter` `:147` (exact-erf, used by Cosmos-Predict2.5;
  forward-only, no backward — see `Tensor::gelu_exact` docstring),
  ⭐ `square_bf16_iter` `:193`, `abs_bf16_iter` `:239`,
  `relu_bf16_iter` `:281`, `sigmoid_bf16_iter` `:323`,
  `tanh_bf16_iter` `:369`, `neg_bf16_iter` `:411`.
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
- ⭐ `add_bf16_contig_direct(a, b)` / `mul_bf16_contig_direct(a, b)` /
  `mul_scalar_bf16_contig_direct(x, scalar)` / `silu_bf16_contig_direct(x)`
  / `gelu_bf16_contig_direct(x)` / `gelu_exact_bf16_contig_direct(x)` (added
  2026-05-21 for Cosmos-Predict2.5) — hot-path collapse helpers added
  2026-05-12. Direct C-FFI into `flame_{add,mul,mul_scalar,silu,gelu,gelu_exact}_bf16_kernel`
  with an inline-populated `IterMetadata` (1-D contig, same-shape, no
  broadcasting). Skip `TensorIteratorConfig::build()` / `build_iter_metadata()`
  on the hot path. Same kernel as the corresponding `*_iter` slow path, so
  output is bit-identical. Used by `Tensor::{silu,gelu,add,mul,mul_scalar}`
  when input(s) are BF16, contig, same-shape. Rollback knob:
  `FLAME_HOT_FAST_PATH_DISABLE=1` (see `env_flags::hot_fast_path_disabled`).
- `softmax_last_dim_bf16(x)` — `:264` — older fused softmax (one block per row).
- ⭐ `rope_fused_bf16(x, cos, sin)` — `:476` — interleaved-pair RoPE.
- ⭐ `rope_fused_bf16_f32pe(x, cos, sin)` — `:595` — RoPE with F32 positional embeddings. Records `Op::RoPePrecomputed` (saves BF16-cast cos/sin for backward dispatcher); see `feedback_rope_fused_autograd.md`.
- `rope_halfsplit_bf16(x, cos, sin)` — `:656` — halfsplit RoPE.
- `modulate_pre_fused_bf16(...)` — `:895` — DiT shift+scale modulation.
- `modulate_pre_split_apply_bf16(...)` — `:961` — B.3 split+apply variant.
- ⭐ `gate_residual_fused_bf16(x, gate, attn_out)` — `:1089` — `x + gate * attn_out`.
- ⭐ `swiglu_fused_bf16(gate, up)` — `:1156` — `silu(gate) * up`.
- `attn_split_txt_img_bf16(...)` — `:1246` — attention output text/image split.
- `qkv_split_permute_bf16(...)` — `:1642` — QKV split + permute.
- `stochastic_round_f32_to_bf16(src, rng)` — `bf16_ops.rs:~2700` — unbiased
  F32→BF16 rounding driven by per-element u32 RNG. Matches the CPU reference
  `bf16_convert::stochastic_round_to_bf16_cpu`. Standalone kernel — useful for
  ad-hoc post-processing (e.g. cast F32 master → BF16 storage at save time).
  The AdamW BF16 update path uses dedicated fused kernels (`adam_fused_bf16_f32grad_stoch_kernel`,
  `adam_fused_multi_bf16_f32grad_stoch_kernel`) that re-implement the same
  lower-16-bit hash logic inline so the AdamW kernel does not need a
  separate temp F32 → BF16 round-trip.

### `bf16_reduce.rs` — BF16-native scalar reductions (added 2026-05-12)
- ⭐ `sum_bf16(x)` — `:120` — sum reduction over all elements of a BF16
  tensor, producing a 0-dim BF16 scalar. Grid-stride F32-accumulator
  in-kernel + atomicAdd into single F32 scratch, then a 1-thread cast
  kernel writes the BF16 result. Replaces the legacy BF16→F32 cast +
  F32 reduce + F32→BF16 cast triple pass that `cuda_ops.rs::GpuOps::sum`
  used for BF16 inputs (Foundation fix #B). Bonus: legacy F32
  `sum_kernel` capped grid at 1024 blocks and silently dropped elements
  past `1024 * 256 = 262144`; the new BF16 kernel uses a grid-stride
  loop so it's correct for any tensor size. Gated by
  `FLAME_BF16_REDUCE_LEGACY=1` (default off).
- ⭐ `mean_bf16(x)` — `:200` — same reduce kernel, but the BF16 cast
  fuses the `* (1/n)` multiply (the cast kernel takes a `scale` arg)
  so the entire mean stays on a single CUDA stream — no host-side
  D2H sync. Wired into `Tensor::mean` BF16 fast path.

### `bf16_convert.rs` — BF16↔F32 cast
- `bf16_u16_to_f32(...)` — `:54` — vectorized via `__nv_bfloat162` (2-element/thread)
- `f32_to_bf16_u16(...)` — `:100` — wraps the `f32_to_bf16` NVRTC kernel; takes
  raw `dst: u64` so callers (e.g. `Tensor::to_dtype` fast path) can write into
  a pre-allocated BF16 buffer without going through `TensorStorage`.
- ⭐ `bf16_to_f32_u16(...)` — `:119` — direct BF16→F32 cast helper added
  2026-05-12 for the `Tensor::to_dtype` BF16→F32 fast path. Takes raw `src: u64`
  + `dst: &mut CudaSlice<f32>`. Eliminates the F32-staging round-trip that
  `to_dtype` did via `storage.to_f32 + dtod_copy + optional f32_to_bf16` —
  collapses 2–3 kernel launches into one.
- `stochastic_round_to_bf16_cpu(f, rng_u32)` — `:~125` — CPU reference for
  unbiased F32→BF16 rounding (GPU path is `bf16_ops::stochastic_round_f32_to_bf16`).
- (The high-level Rust call site is `ops::cast::cast_bf16_to_f32 / cast_f32_to_bf16`.)

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
| ⭐ `dequant_fp8_to_bf16` | `:98` | FP8 → BF16 dequant (one shot) |
| ⭐ `dequant_fp8_to_bf16_into` | `:131` | Same, output-into |
| ⭐ `dequant_fp8_transpose_into` | `:164` | Dequant + transpose in one kernel |
| ⭐ `dequant_mxfp4_to_bf16` | `:220` | MXFP4 → BF16 dequant (GPT-OSS experts; FP4 LUT matches transformers `convert_moe_packed_tensors`) |
| ⭐ `dequant_mxfp4_to_bf16_into` | `:281` | Same, output-into (zero-alloc) |
| ⭐ `fused_rms_norm` | `:202` | RMSNorm with weight, single kernel |
| ⭐ `fused_modulate` | `:241` | `(1+scale) * x + shift` — DiT modulate |
| ⭐ `fused_linear3d` | `:276` | cuBLASLt 3D linear (pre-transposed weight) |
| ⭐ `fused_linear3d_native` | `:357` | cuBLASLt 3D linear (PyTorch weight layout, TRANSA=T) |
| ⭐ `fused_linear3d_native_lora` | `:489` | Additive LoRA over `fused_linear3d_native`; byte-identical at LoRA=None |
| ⭐ `fused_linear3d_native_pytorch_parity` | `:479` | Bit-exact PyTorch linear mirror (biased `gemm_and_bias`, no-bias native matmul path; 2026-05-20); use when ai-toolkit per-op parity matters |
| ⭐ `fused_rms_norm_modulate` | `:552` | RMSNorm + modulate fused |
| ⭐ `fused_residual_gate` | `:599` | `x + gate * attn` fused |

**All of these go through `crate::cuda::ffi::flame_*_bf16` declarations and
the `.cu` files in `src/cuda/`.**

### Deinterleave — `ops/deinterleave.rs`

| Function | Line | What it does |
|---|---|---|
| ⭐ `deinterleave_pair_f32` | `:67` | NVRTC `float2`-vectorized split of `[..., 2K]` F32 into `[..., K]` even+odd halves; replaces `materialize_strided_*` for stride-2 gathers (interleaved-SwiGLU MLPs) |

---

## MoE primitives — `ops/{grouped_mm, fused_gated_scatter_add, moe_routing, nucleus_moe}.rs`

Wrappers around the build-time CUDA kernels in `src/cuda/grouped_mm.cu` and
`src/cuda/fused_gated_scatter_add.cu`, plus host-side routing + a SwiGLU
MoE forward composite. Used by Nucleus-Image (and queued for LLaDA2.0-Uni).

| Function | Line | What it does |
|---|---|---|
| ⭐ `grouped_mm_bf16` | `ops/grouped_mm.rs:42` | Grouped BF16 matmul. `x:(T,K) BF16` × `w:(E,K,N) BF16` → `y:(T,N) BF16` with one expert per `gridDim.z` slot. WMMA tensor-core path, FP32 accumulators, SM80+. Offsets are EXCLUSIVE cumulative end indices passed as `&[i32]` (host slice — see CONVENTIONS for why not `&Tensor`). |
| ⭐ `fused_gated_scatter_add_bf16` | `ops/fused_gated_scatter_add.rs:35` | MoE unpermute: `accum[indices[t]] += expert_out[t] * gating[t]` in-place. F32 atomicAdd on Ampere+ for the per-row collisions. Indices passed as `&[i32]` host slice. |
| ⭐ `expert_choice_route` | `ops/moe_routing.rs:65` | `(B, E, S) F32 affinity` + `capacity`, `route_scale` → `ExpertRoutingPlan` (offsets, global_token_indices, gating_flat). Top-C per (batch,expert) host-side, gating renormalised per-token, scaled by `route_scale`. Mirrors `NucleusMoELayer.forward`'s routing block. |
| ⭐ `permute_tokens` | `ops/moe_routing.rs:204` | `x: (B*S, D)` + plan → `(E*B*C, D)` expert-major, via `Tensor::index_select0`. |
| ⭐ `nucleus_moe_expert_forward` | `ops/nucleus_moe.rs:51` | Full SwiGLU MoE expert FFN: route + permute + grouped_mm(gate_up) + SwiGLU + grouped_mm(down) + weighted scatter-add. Caller owns router matmul, modulation, and shared-expert addition. |
| ⭐ `token_choice_route` | `ops/token_choice_routing.rs:127` | Dual of `expert_choice_route`: each of T tokens picks top-K experts. `(T, E) F32/BF16 router_logits` + `top_k` + `ScoreMode` (TopKSoftmax for GPT-OSS, SoftmaxRenorm for Mixtral/Nucleus-style) → `TokenChoiceRoutingPlan` with prefix-sum offsets (length E+1), expert-major permuted_token_indices (length T*K), expert_weights_flat (length T*K), plus token-major (T, K) expert_indices/expert_weights device tensors. Host-side topk + softmax. |
| ⭐ `permute_tokens_for_token_choice` | `ops/token_choice_routing.rs:360` | `x: (T, D)` + token-choice plan → `(T*K, D)` expert-major, each source token appearing K times. |

All five have `#[cfg(test)] mod tests` parity tests against hand-rolled
scalar Rust references. **First time these CUDA kernels actually ran** —
the two `.cu` kernels had FFI declarations since pre-history but no Rust
caller until 2026-04-29. Phase 4 toy parity (D=inter=64, B=1, E=4, S=8,
C=4) passed within BF16 tolerance.

---

## CUDA infrastructure

### `cuda/ffi.rs` — Rust FFI declarations
The `extern "C"` block declaring all the C-side `flame_*` symbols. Look here
to see what kernels are linked in. Notable groups:
- `flame_narrow_strided_launch / flame_narrow_backward_scatter_add_launch` (`:10,15`) — narrow ops. **Post-Class-E (2026-05-12, commit `b552f61`)**: metadata (shape + strides) passes inline via a `NarrowMeta` kernel-arg struct — no per-call `cudaMalloc` / `cudaMemcpyAsync` / `cudaStreamSynchronize` / `cudaFree`. Reference impl for SPEED_CONTRACT clause 1 ("Sync — primitives do not host-stall"). Real-trainer impact: klein 9B step went 268 → 0 `cudaStreamSynchronize` calls/step.
- `flame_cuda_alloc_pinned_host / flame_cuda_free_pinned_host / flame_cuda_memcpy_async / flame_cuda_host_register / flame_cuda_host_unregister` (`:83-94`) — pinned memory + async copy
- `flame_rope_apply_bf16_fp32` (`:225`) — RoPE kernel (legacy, used by training)
- `flame_apply_causal_mask_fp32 / flame_apply_attn_mask_fp32` (`:238,249`) — SDPA mask kernels
- `flame_sdpa_add_mask_tile_fp32` / `flame_sdpa_softmax_from_lse_tile` / `flame_sdpa_lse_from_logits_tile` / `flame_sdpa_lse_merge_rows` / `flame_sdpa_dropout_bf16_inplace` (`:259-303`) — chunked SDPA primitives
- `flame_geglu_pointwise_fp32` (`:313`) — GeGLU
- `fc_upsample2d_nearest_bf16 / fc_upsample2d_nearest_f32` (`:382,394`) — VAE upsample
- `fc_upsample2d_bilinear_bf16 / fc_upsample2d_bilinear_f32` (`:509,522`) — bilinear 2D upsample (BF16 + F32), PyTorch-matching index math with `align_corners`. Added 2026-04-19 to unblock Cascade.
- `flame_fp8_to_bf16` (`:409`) — FP8 dequant
- `flame_mxfp4_to_bf16` — MXFP4 dequant (32 FP4/block + E8M0 scale → 32 BF16). Used by `dequant_mxfp4_to_bf16{,_into}`. Source: `src/cuda/mxfp4_dequant.cu`.
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
- ⭐ `device_lt::stream_ptr(device)` — `:117` — default-stream pointer for the device. TLS-cached; first call per thread+ordinal hits a global `Mutex<HashMap>`, subsequent calls are lock-free `Cell` reads. Rollback: `FLAME_HANDLE_TLS_DISABLE=1` falls back to global-mutex-on-every-call.
- ⭐ `device_lt::cublaslt_handle_ptr(device)` — `:121` — cached cuBLASLt handle (process-singleton per device; `cublasLtCreate` runs exactly once). Same TLS cache + rollback knob as `stream_ptr`. Called by every BF16 GEMM, fused linear, fused modulate, fused RMS norm — Foundation-#C hot path.

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
- ⭐ `serialization::save_file(tensors, path)` — `:690` — save a HashMap to safetensors (atomic: writes to `{path}.tmp` then renames)
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

### `ring_alloc/mod.rs` — bidirectional ring allocator (Gap 1 Phase 1)

Faithful Rust port of OneTrainer's `StaticLayerAllocator` /
`StaticLayerTensorAllocator` (`/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py:37-222`).
Direction-typed allocator with two cursors over a slab list — forward
allocations advance `allocation_end`; backward retreat `allocation_start`.
Slabs are `cudaMalloc`-ed lazily on first touch. Per-allocation reclaim is
intentionally absent (matches OT semantics); bytes return at `reset()`.

Phase 1 ships the primitive + microbench only. No consumer wiring; no
trainer gate. Full design in `docs/RING_ALLOC_DESIGN.md`.

| Symbol | File:line | Notes |
|---|---|---|
| `RingAllocator` | `ring_alloc/mod.rs:99` | The allocator. Owns slabs + cursors. Not `Send`/`Sync`. |
| `RingPtr` | `ring_alloc/mod.rs:65` | Untyped byte range result. Borrowed view; no Drop. Valid until `reset()`. |
| `RingForwardHandle<'a>` | `ring_alloc/mod.rs:124` | Direction-typed handle for forward allocations. Borrows `&mut RingAllocator`. |
| `RingBackwardHandle<'a>` | `ring_alloc/mod.rs:131` | Direction-typed handle for backward allocations. |
| `RingAllocator::new(device, num_slabs, slab_bytes)` | `ring_alloc/mod.rs:143` | Construct. Slabs are NOT allocated upfront. |
| `RingAllocator::with_slabs(...)` | `ring_alloc/mod.rs:182` | Alias for `new` (convenience for sweeps). |
| `RingAllocator::num_slabs / slab_bytes / total_bytes` | `ring_alloc/mod.rs:190` | Inspection accessors. |
| `RingAllocator::allocation_start / allocation_end` | `ring_alloc/mod.rs:207` | Cursor inspection. |
| `RingAllocator::slabs_allocated / cuda_malloc_count` | `ring_alloc/mod.rs:217` | Lazy-allocation diagnostics. |
| `RingAllocator::reset()` | `ring_alloc/mod.rs:231` | Cursors back to ends; slabs stay mapped. Call between steps. |
| `RingAllocator::forward_handle(block_idx)` | `ring_alloc/mod.rs:239` | Begin a forward-pass scope. |
| `RingAllocator::backward_handle(block_idx)` | `ring_alloc/mod.rs:244` | Begin a backward-pass scope. |
| `RingForwardHandle::alloc(num_bytes)` | `ring_alloc/mod.rs:412` | Forward alloc. 16-byte pre-aligned. |
| `RingBackwardHandle::alloc(num_bytes)` | `ring_alloc/mod.rs:430` | Backward alloc. 16-byte pre-aligned. |
| `RingForwardHandle::allocation_end / allocation_start` | `ring_alloc/mod.rs:418` | Read-through cursor accessors (handle is live, allocator is mutably borrowed). |
| `RingBackwardHandle::allocation_end / allocation_start` | `ring_alloc/mod.rs:435` | Same. |
| `RingAllocator::device()` | `ring_alloc/mod.rs:246` | Backing `Arc<CudaDevice>`. Phase 2a adapter helper. |
| `RingAllocator::alloc_forward(num_bytes)` | `ring_alloc/mod.rs:252` | Handle-free forward alloc. Phase 2a adapter helper. |
| `RingAllocator::alloc_backward(num_bytes)` | `ring_alloc/mod.rs:258` | Handle-free backward alloc. Phase 2a adapter helper. |

Test: `flame-core/tests/ring_alloc_microbench.rs` (19 tests, GPU-real).

### `ring_alloc/pool_adapter.rs` — pool miss-route backend (Gap 1 Phase 2a)

Adapter exposing a `RingAllocator` through `cuda_alloc_pool::PoolMissAllocator`.
Wraps the ring in a `Mutex` for `Sync`; routes cache-miss allocations from
`pool_alloc_u16` / `pool_alloc_f32` into the ring's forward direction.
Returned slices are transmuted views into ring slabs; the pool tags
free-list entries as `is_external: true` so neither `clear_cache` nor
`pool_return_*` calls `cudaFree` on them.

| Symbol | File:line | Notes |
|---|---|---|
| `RingPoolAdapter` | `ring_alloc/pool_adapter.rs:99` | The adapter type. `Send + Sync`. |
| `RingPoolAdapter::new(ring)` | `ring_alloc/pool_adapter.rs:108` | Wrap a `RingAllocator`. |
| `RingPoolAdapter::reset()` | `ring_alloc/pool_adapter.rs:125` | Reset wrapped ring cursors. Call AFTER `clear_pool_cache`. |
| `RingPoolAdapter::slabs_allocated() / cuda_malloc_count() / num_slabs() / slab_bytes()` | `ring_alloc/pool_adapter.rs:133` | Inspection accessors. |
| `impl PoolMissAllocator for RingPoolAdapter` | `ring_alloc/pool_adapter.rs:151` | `alloc_u16` / `alloc_f32` route through ring forward direction. |

Test: `flame-core/tests/ring_pool_adapter_smoke.rs` (3 tests, GPU-real).

### `cuda_alloc_pool` — Phase 2a additions

| Symbol | File:line | Notes |
|---|---|---|
| `PoolMissAllocator` (trait) | `cuda_alloc_pool.rs:639` | Pluggable cache-miss backend trait. `alloc_u16` / `alloc_f32`. |
| `install_miss_allocator(allocator)` | `cuda_alloc_pool.rs:671` | Install. Returns previously installed (if any). |
| `uninstall_miss_allocator()` | `cuda_alloc_pool.rs:692` | Remove. Returns last installed (if any). |
| `CudaAllocPool::external_miss_count()` | `cuda_alloc_pool.rs:217` | Lock-free count of cache-misses served by external allocator. Diagnostic. |
| `CudaAllocPool::register_external_ptr(ptr)` | `cuda_alloc_pool.rs:266` | Phase 2: mark a device ptr as ring-/external-owned so `push_*` routes it through `reconstruct_and_forget` instead of caching the slice. Increments refcount (round-2 fix 2026-05-14). Now a back-compat shim that delegates to `external_memory::ExternalMemoryRegistry::register_exact` under `DEVICE_KEY_ANY`. |
| `CudaAllocPool::is_external_ptr(ptr)` | `cuda_alloc_pool.rs:249` | Test if `ptr` is in the external refcount map (count > 0). Delegates to registry. |
| `CudaAllocPool::unregister_external_ptr(ptr)` | `cuda_alloc_pool.rs:280` | Decrement refcount. Delegates to `ExternalMemoryRegistry::unregister_exact`. |
| `trap_record_bf16_live(ptr, call_site, bucket)` | `cuda_alloc_pool.rs` | R2c. Records a live BF16 range `[ptr, ptr + bucket*2)` in the trap's range table. Called from `pool_alloc_u16` direct + external-miss paths, `static_slab_v2::alloc_u16`, and `Tensor::from_bf16_slice_gpu`. `pub(crate)`. |
| `trap_record_bf16_released(ptr, call_site, bucket)` | `cuda_alloc_pool.rs` | R2c. Clears the live range. Called from `pool_return_u16` and `TensorStorage::Drop` (slab path). `pub(crate)`. |
| `PtrState::Live` (enum variant) | `cuda_alloc_pool.rs` | R2c. New state for range-aware trap. The trap fires only on `Freed` when no live range covers the ptr. |

---

## R1a — `external_memory.rs` (unified registry, 2026-05-15)

⭐ Process-wide `OnceLock` registry tracking GPU memory NOT owned by `cuda_alloc_pool` bucket free lists. Supports both half-open `[start, end)` ranges (slab) and exact-pointer refcounts (ring + back-compat). Single shared cudarc `install_external_ptr_hook` installation.

| Symbol | File:line | Notes |
|---|---|---|
| `ExternalMemoryRegistry::global()` | `external_memory.rs` | Singleton accessor. Returns `&'static ExternalMemoryRegistry`. |
| `ExternalRange { start, end, device_key, owner }` | `external_memory.rs` | Half-open range entry. `end` exclusive. |
| `ExternalOwner { Slab, Ring, PoolExact, BlockOffloader }` | `external_memory.rs` | Origin tag (metadata). |
| `RangeHandle(u64)` | `external_memory.rs` | Opaque `Copy` handle for `unregister_range`. NO `Drop` — explicit unregister required. |
| `DEVICE_KEY_ANY = 0` | `external_memory.rs` | Sentinel for the device-less back-compat shim path. Real `Arc::as_ptr` values are non-zero. |
| `register_range(range) -> RangeHandle` | `external_memory.rs` | Add a range entry. Inverted ranges (`start > end`) silently accepted and never match — by design, caller owns sanity. |
| `unregister_range(handle)` | `external_memory.rs` | Remove. Double-unregister is a silent no-op. |
| `register_exact(ptr, device_key, owner)` | `external_memory.rs` | Add or increment refcount on an exact ptr. |
| `unregister_exact(ptr) -> usize` | `external_memory.rs` | Decrement refcount; returns new value. Removes entry at 0. Device-agnostic — matches any registered device for that ptr. |
| `unregister_exact_keyed(ptr, device_key) -> usize` | `external_memory.rs` | Strict variant — only matches the specified device. |
| `should_skip_free(ptr, device_key) -> bool` | `external_memory.rs` | True if a registered range covers `ptr` for `device_key` (or `DEVICE_KEY_ANY`), OR an exact entry has refcount > 0. Used by the cudarc Drop hook. |
| `should_skip_free_any_device(ptr) -> bool` | `external_memory.rs` | Device-less query — for the cudarc `CudaSlice::drop` hook signature `fn(u64) -> bool`. |
| `ensure_hook_installed()` | `external_memory.rs` | Idempotent cudarc hook install via `compare_exchange(false→true)`. First caller wins. |

---

## R1b–R2a — `static_slab_v2.rs` (bump-allocator + RAII guard, 2026-05-15)

⭐ OneTrainer-style static-slab allocator that backs transient training step allocations. Resolves the BF16 use-after-free bug class structurally: one `cudaMalloc` per device lifetime, bump-cursor allocation, cursor reset at guard drop, never `cudaFree` during training.

| Symbol | File:line | Notes |
|---|---|---|
| `StaticSlabAllocator` | `static_slab_v2.rs` | Per-device bump allocator. Lazy materialization (no `cudaMalloc` until first `alloc_*`). |
| `StaticSlabAllocator::new(device, capacity_bytes)` | `static_slab_v2.rs` | Construct (no alloc fires). |
| `StaticSlabAllocator::alloc_u16(n) -> Result<CudaSlice<u16>>` | `static_slab_v2.rs` | Bump-allocate BF16 elements. `n == 0` is a no-op (no cursor bump, no `live_count` inc). Increments live_count on success. |
| `StaticSlabAllocator::alloc_f32_zeroed(n) -> Result<CudaSlice<f32>>` | `static_slab_v2.rs` | Bump-allocate F32, zero-init via `cudaMemsetAsync`. Preserves the F32-opt-out zero contract. |
| `StaticSlabAllocator::alloc_f32_uninit(n) -> Result<CudaSlice<f32>>` | `static_slab_v2.rs` | Bump-allocate F32, no init. Data is undefined. |
| `StaticSlabAllocator::reset() -> Result<()>` | `static_slab_v2.rs` | Strict — returns `Err` if `live_count != 0`. Zeros the cursor; does NOT drop the slab. |
| `StaticSlabAllocator::release() -> Result<()>` | `static_slab_v2.rs` | Strict-check + unregister range + drop slab `CudaSlice<u8>` (so `cudaFree` actually fires). Next `alloc_*` lazily re-materializes. |
| `StaticSlabAllocator::live_count() -> usize` | `static_slab_v2.rs` | Atomic load. Diagnostic. |
| `StaticSlabAllocator::used_bytes()` / `capacity_bytes()` | `static_slab_v2.rs` | Cursor / capacity introspection. |
| `slab_for_device(device) -> &'static Mutex<StaticSlabAllocator>` | `static_slab_v2.rs` | Per-device leaked-`'static` accessor. Keyed by `Arc::as_ptr(device) as usize`. |
| `slab_v2_return_if_owned(ptr, device_key) -> bool` | `static_slab_v2.rs` | Drop-path helper. If `ptr` is in a registered slab range, decrement live_count and return true. Called from `TensorStorage::Drop` AND from the `external_memory` cudarc hook. |
| `pool_alloc_u16_via_slab(n) -> Option<CudaSlice<u16>>` | `static_slab_v2.rs` | Dispatch helper — called from `cuda_alloc_pool::pool_alloc_u16` when env=1 + guard active. None on overflow, slab missing, or env disabled — caller falls back to legacy. |
| `pool_alloc_f32_via_slab(n) -> Option<CudaSlice<f32>>` | `static_slab_v2.rs` | F32 mirror. Calls `alloc_f32_zeroed` to preserve zero contract. |
| `StepSlabGuard` | `static_slab_v2.rs` | RAII guard activating slab dispatch for the current thread. |
| `StepSlabGuard::enter(device) -> Result<Self>` | `static_slab_v2.rs` | Enter scope. Nested-guard rejected with Err. Eagerly registers slab in `device_map`. |
| `StepSlabGuard::enter_default() -> Result<Self>` | `static_slab_v2.rs` | Convenience: uses `CudaDevice::new(0)`. **Footgun**: if the trainer holds a different `Arc<CudaDevice>`, dispatch routes through this one. Prefer explicit `enter(device.clone())`. |
| `StepSlabGuard::finish(self) -> Result<()>` | `static_slab_v2.rs` | Graceful close. Err on `live_count != 0`. Drop becomes no-op. |
| `StepSlabGuard::active_on_thread() -> bool` | `static_slab_v2.rs` | Cheap thread-local query. |
| `Drop for StepSlabGuard` | `static_slab_v2.rs` | Strict reset. Panics on `live_count != 0` UNLESS `thread::panicking()` is already true (avoids double-panic-during-unwind abort). **Footgun**: never `Send`/`mem::transmute` the guard across threads — the thread-local is per-thread; moving the guard strands the source thread's flag. |
| `FLAME_USE_STATIC_SLAB=1` (env) | `static_slab_v2.rs` | Per-call env check. Gate for `pool_alloc_*` slab dispatch. |
| `FLAME_STATIC_SLAB_BYTES_BF16` (env) | `static_slab_v2.rs` | Default 4 GiB. Klein 9B overflows at this size; raise or accept legacy-fallback. |
| `FLAME_STATIC_SLAB_BYTES_F32` (env) | `static_slab_v2.rs` | Default 4 GiB. |

### `AutogradContext` — checkpoint recompute hook (R2c-perf)

| Symbol | File:line | Notes |
|---|---|---|
| `AutogradContext::is_checkpoint_recompute() -> bool` | `autograd.rs` | True iff the current thread is inside a checkpoint closure's backward replay. Used by `BlockOffloader` (and consumers like Klein's per-block closure) to trigger direction-aware async prefetch only during backward. Thread-local depth counter. |

### `Optimizer` — state prewarm (R2b)

| Symbol | File:line | Notes |
|---|---|---|
| `Adam::ensure_state_initialized(&[Parameter]) -> Result<()>` | `adam.rs` | Pre-materialize F32 `m` / `v` slots for every parameter without advancing `t` or touching values. Must be called BEFORE `FLAME_USE_STATIC_SLAB=1` is set on a trainer using the slab. Idempotent. Empty list ok. |
| `AdamW::ensure_state_initialized(&[Parameter]) -> Result<()>` | `adam.rs` | AdamW variant. Same contract. |

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

### Phase 2 — `SavedRef` + storage version counter (added 2026-05-12)
- `saved_ref::SavedRef { id, tensor, version_counter: Arc<AtomicU32>, version_at_save: u32 }`
  — PyTorch `SavedVariable` analog. `capture(&Tensor) -> SavedRef` and
  `unpack/unpack_ref() -> Result<Tensor>` (errors on version mismatch).
- `saved_ref::legacy_saved_mode() -> bool` — `OnceCell` read of
  `FLAME_AUTOGRAD_SAVED_LEGACY`.
- `autograd::SavedRefs = SmallVec<[SavedRef; 4]>` — replacement for the
  legacy `Vec<(TensorId, Tensor)>` save-list.
- `autograd::TapeEntry` — dual-path: carries BOTH `saved_tensors` (legacy)
  and `saved_refs` (new). `record_op` picks one per call based on
  `legacy_saved_mode()`. `get_saved(id)`, `saved_keys`, `saved_count`,
  `saved_at(i)` expose a unified view.
- `tensor_storage::register_version / lookup_version / unregister_version /
  clear_version_table` — process-global `RwLock<HashMap<usize, Arc<AtomicU32>>>`
  side-table keyed on inner Arc-pointer address.
- `TensorStorage::version_key / version / version_handle / bump_version`
  — read/bump the storage version counter.
- `AutogradContext::clear()` — now also flushes the version table.
- Rollback: `FLAME_AUTOGRAD_SAVED_LEGACY=1` (read once at process start).

### Op variants (forward-recording sites + backward dispatchers)
Each variant has a forward `record_op` site and a backward arm in
`autograd::backward_op`. When adding a new training-path primitive:
1. Add the variant to `pub enum Op` (`autograd.rs:~120`).
2. Add it to the unary-input / binary-input pattern branch in the
   compact-index id collector (`autograd.rs:~1140-1210`).
3. Add a string in `op_tag` (`autograd.rs:~4119`).
4. Wire the forward to propagate `requires_grad` and call `record_op`.
5. Add a backward arm in `backward_op`.

Recently-added variants:
- ⭐ `Op::Conv2d` — forward at `ops/conv2d.rs::conv2d_forward`,
  backward dispatches to `cuda_conv2d::CudaConv2d::conv2d_backward` (F32-only;
  the dispatcher casts BF16 inputs to F32 at the call site).
- ⭐ `Op::Permute` — forward at
  `cuda_ops::GpuOps::permute_nchw_to_nhwc / permute_nhwc_to_nchw` and
  `Tensor::permute`, backward applies the inverse permutation.
- ⭐ `Op::UpsampleNearest2D` (added 2026-04-25) — forward at
  `cuda_ops::GpuOps::upsample2d_nearest`, backward at
  `cuda_kernels::CudaKernels::upsample2d_nearest_backward` (NVRTC F32
  atomicAdd kernel; BF16 grad_outputs are cast to F32 internally).
- ⭐ `Op::RoPePrecomputed { input, cos, sin, layout: RopeLayout }`
  — forward at `bf16_ops::rope_fused_bf16` / `rope_fused_bf16_f32pe` /
  `rope_halfsplit_bf16`. Carries an explicit `RopeLayout` tag (added
  2026-05-20) so backward never has to shape-sniff cos. Was the Q/K
  LoRA gradient blockade pre-2026-04-25 (recording missing), and the
  HiDream-O1 MRoPE shape-sniff mis-classification pre-2026-05-20.
- ⭐ `RopeLayout::{Interleaved, Halfsplit}` — `src/autograd.rs:177`.
  Public enum tagging fused-RoPE autograd variants. See doc at
  `src/autograd.rs:155-177` for which model uses which layout.

### `autograd_v4` (feature gated)
- `autograd_v4::*` — newer experimental engine. Off by default.
- `autograd_v4::ops::sdpa` — SDPA backward via v4

### Autograd v2 (Phase 1 — feature `autograd_v2`)

Clean-sheet PyTorch-style DAG autograd. Phase 1 ships the foundational
types/traits; engine wiring lands in Phase 2. See
[`AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`](./AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md).
No live behavior change yet — module compiles, types compose, all 13
Phase 1 tests pass under the `autograd_v2` feature.

- `autograd_v2::AutogradMetaV2` — `src/autograd_v2/meta.rs:42` — shared,
  interior-mutable per-tensor metadata. Holds `grad`, `grad_fn` (strong
  `Arc<dyn GradFn>`), `grad_accumulator` (**Weak** to break cycle per §1),
  `output_nr`, `requires_grad`, view-base slot, `hooks`.
- `autograd_v2::AutogradMetaRef` — `src/autograd_v2/meta.rs:198` — type
  alias `Arc<Mutex<AutogradMetaV2>>`.
- `autograd_v2::new_meta_ref(meta)` — `src/autograd_v2/meta.rs:201` —
  constructor for the canonical `Arc<Mutex<_>>` shape.
- `autograd_v2::NodeId` — `src/autograd_v2/node.rs:33` — monotonic atomic
  identifier.
- `autograd_v2::Edge { function, input_nr }` — `src/autograd_v2/node.rs:52`
  — `(grad_fn, input slot)` pair.
- `autograd_v2::GradFn` trait — `src/autograd_v2/node.rs:98` —
  `apply(grad_outputs, ctx) -> Result<Vec<Option<Tensor>>, AutogradV2Error>`,
  `num_inputs()`, `next_edges()`, `sequence_nr()`, `topological_nr()`,
  `node_id()`, `name()`, `hooks()` (default empty),
  `release_variables(&self)` (default no-op). Stored sequence/topological
  numbers (not recomputed walks).
- `autograd_v2::SavedTensor` — `src/autograd_v2/saved_tensor.rs:29` —
  `Arc<AtomicU32>` version handle + `expected_version` snapshot +
  `Mutex<Option<Tensor>>` data + `fw_grad_` slot for forward-mode AD.
  `save_named`, `unpack() -> Result<Tensor, AutogradV2Error>`,
  `reset(&self)`, `set_fw_grad/fw_grad`.
- `autograd_v2::InputBuffer` — `src/autograd_v2/input_buffer.rs:39` —
  `num_inputs` slots with in-place AND out-of-place accumulation. In-place
  fires when `create_graph=false` AND dtype/shape match.
- `autograd_v2::Hooks` — `src/autograd_v2/hooks.rs:32` —
  `pre_backward/post_backward/tensor_hooks` vecs. `Hooks::empty_ref()`
  returns the `OnceLock` sentinel for the no-hook fast path.
- `autograd_v2::AccumulateGrad` — `src/autograd_v2/accumulator.rs:25` —
  leaf gradient sink. `Weak` handle to the variable's meta breaks the
  cycle. `apply()` is Phase-1-placeholder (`NotImplementedYet`); Phase 2
  implements actual accumulation.
- `autograd_v2::DispatchCtx` / `DeviceStream` — `src/autograd_v2/dispatch.rs:39,73`
  — multi-device surface (§16). `(device, raw_stream_ptr)` passed by `&`
  to every `apply()` and `InputBuffer::add()`.
- `autograd_v2::AutogradV2Error` — `src/autograd_v2/error.rs:17` —
  `VersionMismatch / SavedTensorReleased / InputSlotOutOfBounds /
  DtypeMismatch / NotImplementedYet / FlameCore(Error)`. Round-trips into
  the crate's `Error::Autograd(String)` via `From`.
- `autograd_v2::V2Result<T>` — `src/autograd_v2/error.rs:71` — alias for
  `Result<T, AutogradV2Error>`.

### Autograd v2 (Phase 2 — engine + accumulator + checkpoint skeleton)

Phase 2 ships the engine driver on top of Phase 1's types. All under
feature `autograd_v2`; no behavior change in the rest of the crate
until Phase 3 wires forward op recording. Tests:
`tests/autograd_v2_engine.rs` (10 tests, all green).

- `autograd_v2::Engine` — `src/autograd_v2/engine.rs` (struct ~line
  140) — `new()`, `execute(root, ctx) -> Result<Vec<Option<Tensor>>>`.
  Stateless across calls; nested execute is just
  `Engine::new().execute(...)` from within a `GradFn::apply`.
- `autograd_v2::GraphRoot` — `src/autograd_v2/engine.rs` (struct ~line
  60) — builder: `new(outputs)`, `with_grad_outputs(g)`,
  `with_inputs(inputs)`, `with_create_graph(b)`, `with_retain_graph(b)`.
- `autograd_v2::gradient_edge(meta, seq) -> Edge` —
  `src/autograd_v2/meta.rs` — materialize-or-cache the leaf accumulator,
  or return `(grad_fn, output_nr)` for a non-leaf. Caches into
  `meta.grad_accumulator` via Weak.
- `autograd_v2::AccumulateGrad::apply` — `src/autograd_v2/accumulator.rs`
  — real Phase 2 impl: drop `None` silently, no-op when meta dropped,
  in-place accumulation into `meta.grad` honoring dtype.
- `autograd_v2::AccumulateGrad::upgrade_variable() -> Option<AutogradMetaRef>`
  — `src/autograd_v2/accumulator.rs` — used by `Engine::with_inputs`
  collection and tests.
- `autograd_v2::CheckpointGradFn` — `src/autograd_v2/checkpoint.rs` —
  skeleton; `apply()` returns `NotImplementedYet` pending Phase 3.
- `autograd_v2::GradFn::as_any` — `src/autograd_v2/node.rs` —
  type-erased downcast helper. `AccumulateGrad` is the Phase 2 user
  (engine's `with_inputs` path needs the downcast).
- ~~`autograd_v2::_v2_set_grad_fn` / `_v2_clear_tensor_meta`~~ —
  **REMOVED in Phase 3a.** The test-only thread-local side-table is
  gone; `Tensor::autograd_meta` is now a real field (Phase 3a)
  exposed via `Tensor::autograd_meta()` / `Tensor::set_autograd_meta()`.
- New `AutogradV2Error` variants — `src/autograd_v2/error.rs`:
  `NoGradFnOnOutput { index }`,
  `OutputGradLenMismatch { outputs, grad_outputs }`,
  `ApplyArityMismatch { op, expected, got }`.

### Autograd v2 (Phase 3a — recording surface + 5 math P0 ops)

Phase 3a wires real forward-op recording on top of Phase 2's engine.
- `Tensor::autograd_meta`: real field (cfg-gated on `autograd_v2`).
- The `record_v2` surface every Phase 3+ op uses.
- 5 P0 math ops: `add`, `mul`, `sum`, `matmul`, `silu`.
- AccumulateGrad / InputBuffer threading `create_graph` via `DispatchCtx`.
- Non-leaf grad collection in `Engine::with_inputs`.
- Validation of `grad_outputs[i].shape() == outputs[i].shape()`.
Tests: `tests/autograd_v2_ops.rs` (18 tests, incl. matmul per-element parity after bug-fixer follow-up `6ee385f`). Phase 2 engine tests
(`tests/autograd_v2_engine.rs`) refactored to use `record_v2`-style
linking; still 11 tests, all green.

- `Tensor::autograd_meta` field — `src/tensor.rs:185` —
  `Option<crate::autograd_v2::AutogradMetaRef>`, `cfg(feature =
  "autograd_v2")`-gated so the default build pays zero.
- `Tensor::autograd_meta()` accessor — `src/tensor.rs:236` — returns
  `Option<&AutogradMetaRef>`.
- `Tensor::set_autograd_meta(meta)` — `src/tensor.rs:246` — used by
  `record_v2` to install a fresh meta on a recorded output.
- `Tensor::detach_v2()` — `src/tensor.rs:261` — returns a fresh
  `Tensor` with `autograd_meta = None`. PyTorch parity.
- `autograd_v2::record_v2(grad_fn, outputs, ctx) -> Vec<Tensor>` —
  `src/autograd_v2/recording.rs:113` — wires a freshly-built
  `Arc<dyn GradFn>` to one or more output tensors by allocating an
  `AutogradMetaRef` per output. The canonical op-recording entry.
- `autograd_v2::gradient_edge_for_tensor(t) -> Edge` —
  `src/autograd_v2/recording.rs:58` — `Tensor`-level wrapper around
  `meta::gradient_edge`. Returns `Edge::null()` for no-meta /
  no-requires_grad / no-grad_fn tensors.
- `autograd_v2::next_sequence_nr() -> u64` —
  `src/autograd_v2/recording.rs:42` — monotonic op-creation counter
  used by every `*GradFn::new`.
- `autograd_v2::needs_grad(inputs: &[&Tensor]) -> bool` —
  `src/autograd_v2/recording.rs:87` — the gating predicate every
  Phase 3+ op forward wrapper uses to skip recording on inference.

Op modules (each in its own file under `src/autograd_v2/ops/`):

- `autograd_v2::ops::add::AddGradFn` — `src/autograd_v2/ops/add.rs:24` —
  backward: `(g, g)`. No saved tensors.
- `autograd_v2::ops::add::add_v2(a, b, ctx)` —
  `src/autograd_v2/ops/add.rs:91` — forward wrapper. Records iff any
  input has `requires_grad=true` or a `grad_fn`.
- `autograd_v2::ops::mul::MulGradFn` — `src/autograd_v2/ops/mul.rs:21`
  — backward: `(g*b, g*a)`. Saves `a, b`.
- `autograd_v2::ops::mul::mul_v2(a, b, ctx)` —
  `src/autograd_v2/ops/mul.rs:97`.
- `autograd_v2::ops::sum::SumGradFn` — `src/autograd_v2/ops/sum.rs:25`
  — backward: `broadcast(g, input_shape)`. Saves input shape + dtype
  only.
- `autograd_v2::ops::sum::sum_v2(a, ctx)` —
  `src/autograd_v2/ops/sum.rs:99`.
- `autograd_v2::ops::matmul::MatMulGradFn` —
  `src/autograd_v2/ops/matmul.rs:27` — backward: `(g @ b^T, a^T @ g)`
  (2D). Saves `a, b`.
- `autograd_v2::ops::matmul::matmul_v2(a, b, ctx)` —
  `src/autograd_v2/ops/matmul.rs:122`.
- `autograd_v2::ops::silu::SiLUGradFn` —
  `src/autograd_v2/ops/silu.rs:31` — backward: `g * sigmoid(x) * (1 +
  x * (1 - sigmoid(x)))`. Saves input.
- `autograd_v2::ops::silu::silu_v2(x, ctx)` —
  `src/autograd_v2/ops/silu.rs:111`.

Plumbing changes (Phase 3a follow-ups to Phase 2 bug-fixer feedback):

- `DispatchCtx::create_graph` field — `src/autograd_v2/dispatch.rs:84`
  — threaded by `Engine::execute` from `GraphRoot::create_graph` so
  per-node accumulation paths can pick recording vs in-place.
- `DispatchCtx::with_create_graph(b)` builder —
  `src/autograd_v2/dispatch.rs:104`.
- `InputBuffer::add_outofplace` — `src/autograd_v2/input_buffer.rs:131`
  — records the accumulation `+` as a v2 op when
  `ctx.create_graph=true` (so higher-order grads differentiate through
  the accumulation).
- `AccumulateGrad::apply` — `src/autograd_v2/accumulator.rs:107` —
  same `create_graph` threading; recording branch builds an
  `AddGradFn` and routes through `record_v2` directly (the gradients
  flowing through backward don't carry `requires_grad` themselves).
- `AutogradV2Error::GradOutputShapeMismatch` —
  `src/autograd_v2/error.rs:65` — `Engine::execute` validates
  `grad_outputs[i].shape() == outputs[i].shape()` at entry, replacing
  the Phase 2 silent broadcast.
- Non-leaf grad capture in `Engine::execute` —
  `src/autograd_v2/engine.rs` (search for `nonleaf_capture_keys`) —
  per-call `(NodeId, output_nr) -> Tensor` map; populated at each
  node's apply-time and read out at the end of execute for any
  `with_inputs` entry whose grad_fn isn't an `AccumulateGrad`.

### Autograd v2 (Phase 3b — view-autograd surface + AccumulateGrad hooks() fix)

Phase 3b ships 6 view ops (shape-only, no math) + a Phase 2 carryover
fix for `AccumulateGrad::hooks()`. Each view op carries a `*GradFn`
struct + a `*_v2` forward wrapper. View-op backwards are inverse shape
ops applied to grad_output; HAZARD-2026-05-13-1 discipline applies
(see §HAZARD-2026-05-13-1 in `AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`).

- `autograd_v2::ops::reshape::ReshapeGradFn` —
  `src/autograd_v2/ops/reshape.rs:31` — backward: `grad_output.reshape(input_shape)`.
  Saves input shape only.
- `autograd_v2::ops::reshape::reshape_v2(a, new_shape, ctx)` —
  `src/autograd_v2/ops/reshape.rs:97`.
- `autograd_v2::ops::reshape::view_v2(a, new_shape, ctx)` —
  `src/autograd_v2/ops/reshape.rs:109` — alias of `reshape_v2`
  (flame-core's `Tensor::view` is `reshape` internally).
- `autograd_v2::ops::transpose::TransposeGradFn` —
  `src/autograd_v2/ops/transpose.rs:33` — backward: `grad_output.transpose().contiguous()`.
  `.contiguous()` materialises the strided view (HAZARD-2026-05-13-1 +
  project-wide gemm-stride-ignore — mirrors Phase 3a matmul fix `6ee385f`).
- `autograd_v2::ops::transpose::transpose_v2(a, ctx)` —
  `src/autograd_v2/ops/transpose.rs:101`.
- `autograd_v2::ops::narrow::NarrowGradFn` —
  `src/autograd_v2/ops/narrow.rs:38` — backward allocates a fresh zero
  tensor + scatters via `narrow_backward_scatter_add_cuda` (sole-
  owner writeback). HAZARD-2026-05-13-1 CRITICAL: never writes through
  a `narrow()` view back into a parent.
- `autograd_v2::ops::narrow::narrow_v2(a, dim, start, length, ctx)` —
  `src/autograd_v2/ops/narrow.rs:143`.
- `autograd_v2::ops::squeeze::SqueezeGradFn` —
  `src/autograd_v2/ops/squeeze.rs:29` — backward: `grad_output.unsqueeze(dim)`.
  Saves dim only.
- `autograd_v2::ops::squeeze::squeeze_v2(a, dim, ctx)` —
  `src/autograd_v2/ops/squeeze.rs:92`.
- `autograd_v2::ops::unsqueeze::UnsqueezeGradFn` —
  `src/autograd_v2/ops/unsqueeze.rs:24` — backward: `grad_output.squeeze(dim)`.
  Saves dim only.
- `autograd_v2::ops::unsqueeze::unsqueeze_v2(a, dim, ctx)` —
  `src/autograd_v2/ops/unsqueeze.rs:89`.
- `autograd_v2::ops::permute::PermuteGradFn` —
  `src/autograd_v2/ops/permute.rs:29` — backward:
  `grad_output.permute(inverse_perm).contiguous()`. Computes the
  inverse permutation once at construction. HAZARD-2026-05-13-1 +
  gemm-stride-ignore: `.contiguous()` materialises the strided view.
- `autograd_v2::ops::permute::permute_v2(a, perm, ctx)` —
  `src/autograd_v2/ops/permute.rs:105`.

Phase 2 carryover fix:

- `AccumulateGrad::hooks` field — `src/autograd_v2/accumulator.rs:39`
  — type changed from `Hooks` to `OnceLock<Hooks>`. Default state is
  uninitialised so `hooks()` returns `Hooks::empty_ref()` and the
  engine's pointer-equality fast path
  (`std::ptr::eq(hooks_ref, Hooks::empty_ref())`) fires. Previously
  the per-struct `Hooks::default()` instance had a unique address per
  accumulator, so the fast path never matched.
- `AccumulateGrad::install_hooks(hooks)` —
  `src/autograd_v2/accumulator.rs:102` — single-shot Hooks installer.
  Returns `Err(Hooks)` on second call (the previously-installed Hooks
  bundle stays; multi-hook merge deferred until Hooks fields carry
  interior mutability).
- `AccumulateGrad::hooks()` impl — `src/autograd_v2/accumulator.rs`
  (the `GradFn::hooks` impl in the `impl GradFn for AccumulateGrad`
  block) — `self.hooks.get().unwrap_or_else(|| Hooks::empty_ref())`.

### Autograd v2 (Phase 3c1 — layer_norm + CheckpointGradFn)

Phase 3c1 ships `layer_norm` (production op delegating to the BF16 fused
forward + backward kernels) and the full `CheckpointGradFn::apply`
implementation (was `NotImplementedYet` through Phase 2). Forward-mode AD
across the 11 prior ops is deferred to Phase 3c2 — Phase 4 does not
depend on forward-mode AD.

- `autograd_v2::ops::layer_norm::LayerNormGradFn` —
  `src/autograd_v2/ops/layer_norm.rs:46` — affine LN backward.
  Saves `(x, weight?, bias?)` via `SavedTensor` + `(normalized_shape, eps)`.
  Forward delegates to `crate::cuda_ops_bf16::layer_norm_bf16`; backward to
  `crate::cuda_ops_bf16::layer_norm_backward_bf16` (kernel recomputes
  per-feature mean/rstd internally). BF16-only at the public boundary.
  `next_edges` are sized 1 + has_weight + has_bias; `apply()` returns the
  matching `Vec<Option<Tensor>>` per input slot.
- `autograd_v2::ops::layer_norm::layer_norm_v2(x, normalized_shape, weight, bias, eps, ctx)` —
  `src/autograd_v2/ops/layer_norm.rs:249`. Wraps `cuda_ops_bf16::layer_norm_bf16`
  + conditional `record_v2` install.
- `autograd_v2::CheckpointGradFn` — `src/autograd_v2/checkpoint.rs:79` —
  backward node for activation checkpointing. Carries
  `Arc<CheckpointForwardFn>` + `Vec<SavedTensor>` for inputs +
  per-input `next_edges`. `apply()` re-runs forward closure under v2
  recording (after detach + fresh `requires_grad=true` meta on each
  input clone), drives a nested `Engine::execute(...)` through the
  standard backward path, then harvests per-leaf grads off each
  reattached leaf's `meta.grad`. Reentrant-nested-execute safe.
- `autograd_v2::CheckpointForwardFn` — `src/autograd_v2/checkpoint.rs:63`
  — type alias: `dyn Fn(&[Tensor], &DispatchCtx) -> Result<Vec<Tensor>> + Send + Sync`.
- `autograd_v2::checkpoint_v2(forward_fn, inputs, ctx)` —
  `src/autograd_v2/checkpoint.rs:281` — user-facing entry. Runs
  `forward_fn` with detached input clones (so inner ops see
  `needs_grad=false` and skip recording — that is the memory saving),
  builds a `CheckpointGradFn` with saved inputs + per-input
  `next_edges`, installs it on every output via `record_v2`. Inference
  path (no input requires_grad) returns outputs unchanged.

Design choices documented in the module-level comment of
`checkpoint.rs`: detach-clones replace what a "no-record" thread-local
flag would have done; `Arc<dyn Fn>` allows reuse under
`retain_graph=true`; the standard leaf-AccumulateGrad backward path
inside `apply()` replaces `with_inputs` (which `expect`s a `grad_fn` on
inputs — Phase 3a never wired the leaf path).

Tests (4 layer_norm + 4 checkpoint = 8 new):
- `tests/autograd_v2_ops.rs`:
  - `layer_norm_v2_forward_backward_shapes`
  - `layer_norm_v2_no_grad_skips_recording`
  - `layer_norm_v2_weight_only`
  - `layer_norm_v2_analytical_dweight_dbias` — closed-form d_w/d_b parity
  - `layer_norm_v2_dx_symmetry_uniform_grad` — bug-fixer-added (`2d9cd0d`): symmetry-based d_x correctness via "uniform grad + zero-mean residual → d_x = 0"
- `tests/autograd_v2_checkpoint.rs` (new file):
  - `checkpoint_v2_same_grads_as_no_checkpoint` — bit-equal vs reference
  - `checkpoint_v2_skips_when_no_input_requires_grad` — inference path
  - `checkpoint_v2_reentrant_nested` — checkpoint-in-checkpoint, 2 nested engines
  - `checkpoint_v2_multi_output` — multi-output forward, same node id on each

Forward-mode AD (JVP) plumbing across the 11 prior ops shipped in
Phase 3c2 — see the next section.

### Autograd v2 (Phase 3c2 — forward-mode AD across 11 ops)

Phase 3c2 closes Phase 3 by populating the `Tensor`-level forward-mode
AD slot (`AutogradMetaV2::fw_grad`) inside every Phase 3a/3b op's
forward wrapper. The JVP plumbing is independent of the backward-mode
recording — `fw_grad` and `requires_grad` are orthogonal gates.

- `autograd_v2::AutogradMetaV2::fw_grad` field —
  `src/autograd_v2/meta.rs:87` — `Option<Tensor>`; defaults to `None`
  on every constructor (`leaf_no_grad`, `leaf_requires_grad`,
  `non_leaf`). Populated by `Tensor::set_fw_grad`; read by per-op
  forward wrappers via `Tensor::fw_grad`.
- `Tensor::fw_grad(&self) -> Option<Tensor>` —
  `src/tensor.rs:258` — cloned read of the installed tangent (or
  `None` when no `autograd_meta` / no tangent). Cheap-zero-overhead
  on the `None` path: skips lock acquisition entirely.
- `Tensor::set_fw_grad(&mut self, g: Tensor)` —
  `src/tensor.rs:276` — installs a tangent on this tensor. PyTorch
  parity: implicitly allocates a fresh `AutogradMetaV2::leaf_no_grad()`
  if the tensor doesn't yet have a meta (setting a tangent does NOT
  silently enable backward-mode tracking — `requires_grad` stays
  false).
- `autograd_v2::ops::fw_mode::any_fw_grad(inputs) -> bool` —
  `src/autograd_v2/ops/fw_mode.rs:34` — gating predicate for op
  wrappers; mirrors `needs_grad` but for forward-mode AD.
- `autograd_v2::ops::fw_mode::tangent_or_zero(t) -> Result<Tensor>` —
  `src/autograd_v2/ops/fw_mode.rs:49` — returns `t.fw_grad()` if set,
  otherwise a fresh `zeros_like_with_dtype(t.dtype())`. Materialises
  the JVP convention "no tangent ⇒ zero tangent" so downstream
  formulas can multiply / matmul against a real tensor.

Per-op JVP additions (formulas in each op file's doc comment):

- `add_v2` (`src/autograd_v2/ops/add.rs:97`) — JVP: `a_dot + b_dot`.
- `mul_v2` (`src/autograd_v2/ops/mul.rs:102`) — JVP product rule:
  `a_dot * b + a * b_dot`.
- `sum_v2` (`src/autograd_v2/ops/sum.rs:102`) — JVP: `sum(a_dot)`.
- `matmul_v2` (`src/autograd_v2/ops/matmul.rs:130`) — JVP product
  rule: `a_dot @ b + a @ b_dot`.
- `silu_v2` (`src/autograd_v2/ops/silu.rs:123`) — JVP:
  `x_dot * silu_deriv(x)` where
  `silu_deriv = sigmoid(x) * (1 + x*(1 - sigmoid(x)))`.
- `reshape_v2` / `view_v2`
  (`src/autograd_v2/ops/reshape.rs:101, 122`) — JVP applies the same
  reshape/view to the tangent.
- `transpose_v2` (`src/autograd_v2/ops/transpose.rs:106`) — JVP:
  `a_dot.transpose().contiguous()`. HAZARD-2026-05-13-1 +
  gemm-stride-ignore.
- `narrow_v2` (`src/autograd_v2/ops/narrow.rs:144`) — JVP:
  `a_dot.narrow(dim, start, length)`. Read-only slice; no write
  hazard.
- `squeeze_v2` (`src/autograd_v2/ops/squeeze.rs:96`) — JVP:
  `a_dot.squeeze(Some(dim))`.
- `unsqueeze_v2` (`src/autograd_v2/ops/unsqueeze.rs:93`) — JVP:
  `a_dot.unsqueeze(dim)`.
- `permute_v2` (`src/autograd_v2/ops/permute.rs:110`) — JVP:
  `a_dot.permute(perm).contiguous()`. HAZARD-2026-05-13-1 +
  gemm-stride-ignore.

`layer_norm_v2` forward-mode AD is **deferred** — its JVP formula
`out_fw = (x_fw - mean_fw)*rstd*w + x_hat*d_rstd_dx*w + x_hat*rstd*w_fw + b_fw`
is mechanical but non-trivial; an input tangent on `x` / `weight` /
`bias` is silently dropped on the LN output (matches "tangent
defaults to zero" semantics). The Phase 5 parity gate is the right
time to ship and validate the formula against
`torch.autograd.functional.jvp(layer_norm, ...)`.

Tests: `tests/autograd_v2_fw_mode.rs` (13 tests — one per JVP-
supporting op plus `set_fw_grad_implicitly_allocates_meta_on_untracked_tensor`).
F32-only; BF16 paths will be exercised in Phase 5 parity.

### Autograd v2 (Phase 4a — optimizer + BF16-grad migration partial)

Phase 4a ships the parameter / optimizer half of `BF16_GRAD_DECISION.md`
Option A. v2 trainers can now construct a `Parameter::new_v2(...)` whose
gradients stay BF16 end-to-end (no upcast on `set_grad`), and an
`AdamWV2` wrapper routes them through the existing BF16-grad fused Adam
kernels that were dead code before this phase. The `GradientMap`
rewrite + actual trainer-integration smoke are deferred to Phase 4b.
Tests: `tests/autograd_v2_phase4a.rs` (12 tests).

- `parameter::GradDtypePolicy` — `src/parameter.rs:31` — enum with
  variants `CastToF32` (v1/v3 default) and `MatchParamDtype` (v2). Re-
  exported as `flame_core::GradDtypePolicy`.
- `Parameter::new_v2(t)` — `src/parameter.rs:93` — constructs a parameter
  with `GradDtypePolicy::MatchParamDtype`; BF16 params keep BF16 grads
  through `set_grad`, F32 params keep F32 grads.
- `Parameter::grad_dtype_policy()` / `Parameter::set_grad_dtype_policy()`
  — `src/parameter.rs:112, 119` — policy accessors.
- `Parameter::grad_bf16_or_f32()` — `src/parameter.rs:243` — v2-facing
  grad accessor; returns the native-dtype grad without casting (mirrors
  `grad()` but names the contract explicitly).
- Phase 4a `Parameter::set_grad` / `Parameter::apply_update` — rewrites
  at `src/parameter.rs:199` and `src/parameter.rs:303` — preserve
  dtype under `MatchParamDtype`; unchanged under `CastToF32` so
  existing v3 trainers keep their F32-grad invariant.
- `adam::Adam::step` — `src/adam.rs:1107` — classifier extended to
  3 multi-tensor arms: `(BF16 param, F32 grad)`, `(BF16 param, BF16
  grad)`, `(F32 param, F32 grad)`. `(F32 param, BF16 grad)` routes
  through the per-param fused path which has the existing
  `adam_fused_f32param_bf16grad_kernel` arm.
- `ops::multi_tensor::multi_tensor_l2_norm_sq_bf16(cache, grads)` —
  `src/ops/multi_tensor.rs:447` — BF16 sibling for
  `multi_tensor_l2_norm_sq_f32`. Same 2-launch stage1+stage2 reduction,
  F32 only as `opmath_t` inside the kernel. Re-uses the stage2 reducer
  from the F32 module.
- `ops::grad_norm::global_l2_norm` — `src/ops/grad_norm.rs:99` —
  routes all-BF16-contiguous slices through
  `multi_tensor_l2_norm_sq_bf16`.
- `autograd_v2::OptimizerV2` trait — `src/autograd_v2/optim.rs:47` —
  `step(params, ctx)` + `zero_grad(params)`.
- `autograd_v2::AdamWV2` — `src/autograd_v2/optim.rs:68` — thin v2
  wrapper around `AdamW`; same state shape, same kernels, same
  checkpoint format. `inner_adamw()` / `inner_adamw_mut()` borrow the
  inner state for callers that need the full v1 surface.
- `autograd_v2::set_param_grad_v2(param, grad)` —
  `src/autograd_v2/optim.rs:143` — convenience setter; returns `Err` if
  `param` is on the v1 `CastToF32` policy (catches policy/contract
  mismatches early).

### Autograd v2 (Phase 4b — GradientMap dtype-preserving policy)

Phase 4b closes the second half of `BF16_GRAD_DECISION.md` Option A by
giving `GradientMap` a `MatchInsertedDtype` policy variant. v1 / v3
trainers using `GradientMap::new` / `with_index` see zero behavior
change; v2 callers opt in via `new_v2` / `with_index_v2`. Trainer
integration (flipping Z-Image's `loss.backward()` to a v2 grad-storage
path) is Phase 5 work — see `docs/BF16_GRAD_DECISION.md` "Deliverable C"
section. Tests: `tests/autograd_v2_gradientmap_v2.rs` (17 tests).

- `autograd::policy::GradStorePolicy` — `src/autograd/policy.rs:28` —
  enum gained the `MatchInsertedDtype` variant. Now derives `PartialEq`
  / `Eq` for tests.
- `gradient::GradientMap::new_v2(device)` — `src/gradient.rs:99` —
  constructs an autograd-v2 GradientMap (HashMap fallback mode).
- `gradient::GradientMap::with_index_v2(device, index)` —
  `src/gradient.rs:112` — constructs an autograd-v2 GradientMap with
  the Vec-fast-path compact index.
- `gradient::GradientMap::policy()` — `src/gradient.rs:124` — read-only
  policy accessor (used by the test suite).
- `gradient::GradientMap::set_ones_dtype(id, shape, dtype)` —
  `src/gradient.rs:153` — v2-friendly loss seed at an explicit dtype.
  Under v1 forces F32 regardless (legacy invariant preserved).
- `gradient::GradientMap::get_or_create_dtype(id, shape, dtype)` —
  `src/gradient.rs:403` — v2-friendly zero-pre-allocate at explicit
  dtype. Under v1 forces F32.
- `gradient::GradientMap::cast_all_to_dtype(dtype)` —
  `src/gradient.rs:461` — **Phase 5b bug-fix** (`a5da3d5`). Walks both
  `vec_store` and `overflow`, casts every stored gradient to `dtype` via
  fresh `to_dtype` (reassigns slot — NOT in-place despite the doc-comment
  wording). Fast-paths when the stored grad already matches. Used by
  `backward_v2` as a single post-loop pass to realize BF16-end-to-end
  without forcing v3 backward kernels to handle BF16 non-leaf grads.
- `gradient::GradientMap::get_public_grad` / `take_public_grads` —
  `src/gradient.rs:215, 234` — both now branch on policy; v2 returns
  native-dtype grads (no BF16 cast), v1 returns the BF16-cast output
  (unchanged from pre-Phase-4b).
- `gradient::GradientMap::insert` — `src/gradient.rs:274` — under v1
  still upcasts BF16 → F32; under v2 preserves dtype.
- `gradient::GradientMap::accumulate` — `src/gradient.rs:312` —
  dispatches on policy: `accumulate_v1` keeps the legacy deferred-
  upcast behavior; `accumulate_v2` keeps the storage dtype and errs on
  dtype mismatch.

### Autograd v2 (Phase 5a — per-op PyTorch parity tests)

Phase 5a ships Deliverables A + B from `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`
§Phase 5 and closes Phase 3c2's deferred `layer_norm` JVP. 26 tests:
13 ops × backward parity + 13 ops × forward-mode AD parity against
PyTorch-generated fixtures. Phase 5b (model parity) and 5c (perf
bench) remain open.

- `layer_norm_v2` — `src/autograd_v2/ops/layer_norm.rs:250` — forward
  wrapper now installs an output `fw_grad` when any of `x`/`weight`/`bias`
  carries an input tangent. Backward path is unchanged.
- `layer_norm_jvp` (private) — `src/autograd_v2/ops/layer_norm.rs:329` —
  the F32-internal LN JVP. Per-row reductions reproduce
  `torch.autograd.functional.jvp(layer_norm, ...)` bit-equal in F64.
  Final cast back to the primal's BF16 dtype before storage.
- `tests/autograd_v2_parity.rs` — 26 parity tests. Each loads a
  `.safetensors` fixture from `tests/fixtures/<op>_{backward,jvp}.safetensors`,
  runs the v2 op via `Engine::execute`, and compares against the
  fixture's `grad_input_*` / `out_fw` keys via
  `flame_core::parity::ParityHarness`. Tolerance bands by op family
  documented at the file header.
- `tests/fixtures/gen_v2_parity.py` — fixture generator. PyTorch
  reference at fixed seed 42, per-op section. Run from `flame-core/`
  root with `python3 tests/fixtures/gen_v2_parity.py`. Layer-norm
  fixtures use BF16 storage; other ops F32.
- `tests/fixtures/<op>_backward.safetensors` (13 files) — inputs +
  all-ones `grad_output` + per-input reference grads.
- `tests/fixtures/<op>_jvp.safetensors` (13 files) — inputs +
  per-input tangents + reference `out_fw`.

### Autograd v2 (Phase 5b — `backward_v2()` bridge for BF16 grads end-to-end)

Phase 5b ships Deliverable C Route (ii) from `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`
§Phase 5 — a flag-switch on `AutogradContext::backward` that constructs
a `GradientMap` under `GradStorePolicy::MatchInsertedDtype` and casts
each emitted gradient to the loss tensor's dtype before accumulation.
The v3 op-dispatch loop is unchanged; the only delta is the grad-map
policy + the dtype-unification cast.

- `AutogradContext::backward` — `src/autograd.rs:1297` — public v3 entry,
  now a one-line wrapper around `backward_impl(loss, InternalFP32_PublicBF16)`.
  Byte-equivalent to pre-Phase-5b behavior.
- `AutogradContext::backward_v2` — `src/autograd.rs:1322` — public v2 entry
  (gated on feature `autograd_v2`). Returns a `GradientMap` whose stored
  gradients honor `loss.dtype()` end-to-end (BF16 preserved for BF16 loss,
  per Option A of `docs/BF16_GRAD_DECISION.md`). Forward graph does NOT
  need to be authored in v2 ops to use this entry.
- `AutogradContext::backward_impl` (private) — `src/autograd.rs:1342` —
  shared body, accepts a `GradStorePolicy`. Three sites differ from the
  pre-Phase-5b body: (a) `GradientMap::with_index` vs `with_index_v2`,
  (b) `set_ones` vs `set_ones_dtype(loss.id, loss.shape, loss.dtype())`,
  (c) per-grad `to_dtype(loss.dtype())` cast at accumulate time under
  `MatchInsertedDtype`.
- `tests/autograd_v2_bridge.rs` — 3 tests: BF16-grads-emitted, F32
  parity vs v3 (cos≥1-1e-6, max_abs_ratio≤1e-5), BF16 tolerance vs v3
  (cos≥0.999, max_abs_ratio≤5e-3 per `BF16_GRAD_DECISION.md`).
- `tests/autograd_v2_klein_parity.rs` — Phase 5b follow-up. 1 `#[test]`
  driver (`klein_parity_v2_scenarios`) running 7 scenarios sequentially
  to avoid the `AUTOGRAD_CONTEXT` parallel-mode race: 6 Klein component
  parity v2 (head_rms_norm toy/prod, apply_rope_prod, rms_norm_direct_4d,
  rms_norm_contig_prod, attn_chain_prod) + 1 full-block parity from
  `klein_block_backward.safetensors` (Deliverable B, new fixture
  consumer). Tolerance band: `min_cos=0.99`, `max_abs_ratio=5e-2` —
  matches the v3 Klein test gate (cos > 0.99). All 7 pass; max observed
  abs_ratio 0.0093 on `dw_qkv` in attn_chain.
- `tests/autograd_v2_perf.rs` — Phase 5c Deliverable D. 3 `#[serial]
  #[test]` cells: `perf_synthetic_mlp`, `perf_klein_attn_chain_prod`,
  `perf_klein_double_block`. Each runs v3 / bridge / Class A configs ×
  5 warmup + 50 timed iters with the slowest 5 trimmed, prints
  median/P90/mean (ms) and grad-byte totals per config. Numbers feed
  `BF16_GRAD_DECISION.md` §Phase 5c. Reproduce:
  `FLAME_CUDA_GRAPH=0 cargo test --release --features autograd_v2 --test autograd_v2_perf -- --nocapture --test-threads=1`.
- EriDiffusion-v2 `train_zimage.rs` ships an opt-in `--use-autograd-v2`
  flag plumbed through a new `autograd_v2` feature on `eridiffusion-cli`.
  Default OFF — v3 path is byte-equivalent to pre-flag behavior.

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

### Block offloading — `offload::BlockOffloader` (moved into flame-core 2026-05-12, commit `df00c5f`)

⭐ Sole block-offloading mechanism for both training and inference. Was at
`flame-diffusion/src/block_offload.rs`; the entire ~1,579 LOC module is now
re-rooted at `flame_core::offload`. Every trainer + every inference model
that calls `prefetch_block` / `await_block` now imports from
`flame_core::offload::*`. See [`FLAME_MODULES.md`](./FLAME_MODULES.md#-offload--blockoffloader--blockfacilitator-moved-into-flame-core-2026-05-12)
for the module overview.

| Symbol | File:line | Notes |
|---|---|---|
| ⭐ `offload::BlockOffloader` | `offload/mod.rs:331` | Double-buffered pinned-CPU → GPU offloader. Two GPU slots, one dedicated CUDA transfer stream, per-slot CUDA events (`h2d_done`, `compute_done`) for stream-stream waits — no host stalls on the hot path. |
| ⭐ `offload::BlockFacilitator` trait | `offload/mod.rs:179` | Model-specific geometry provider. Two methods: `block_count(&self) -> usize`, `classify_key(&self, key: &str) -> Option<usize>`. Each trainer / inference model implements one. |
| ⭐ `offload::BlockHandle` | `offload/mod.rs:1491` | Scoped handle returned by `await_block_handle`. Records `compute_done` on Drop; next prefetch reusing the slot does `cudaStreamWaitEvent` instead of host sync. `Deref<Target = HashMap<String, Tensor>>`. Methods: `.weights()`, `.get(key)`, `.arc()`. |
| ⭐ `BlockOffloader::load(paths, facilitator, device)` | `offload/mod.rs:384` | Default ctor. Reads safetensors → pinned CPU memory, converts to BF16 at load. `BLOCKOFF_FP8_PINNED=1` keeps F8_E4M3 raw in pinned RAM, GPU-dequants in `prepare_weights`. |
| ⭐ `BlockOffloader::load_fp8_stream(paths, facilitator, device)` | `offload/mod.rs:404` | Same as `load` but always treats F8_E4M3 as keep-raw-pinned + GPU-dequant. For LTX-2/Wan2.2 FP8 checkpoints; halves pinned RAM. |
| ⭐ `BlockOffloader::load_streaming(paths, facilitator, device)` | `offload/mod.rs:680` | mmap-backed mode: pinned RAM = `2 × max_block_bf16_bytes` (typically <1.5 GB) instead of full-model. Two staging buffers fed from mmap on each prefetch. For Qwen-Image-2512-class models that don't fit in 62 GB pinned. F8_E4M3 not supported in this mode. |
| ⭐ `BlockOffloader::with_native_layout(self, native: bool) -> Self` | `offload/mod.rs:822` | Builder. `true` disables the `[Cout,Cin] → [Cin,Cout]` pre-transpose in `prepare_weights`. Required when the caller's forward uses `ops::fused_inference::fused_linear3d_native` (cuBLASLt TRANSA=T). |
| ⭐ `BlockOffloader::prefetch_block(&mut self, idx)` | `offload/mod.rs:835` | Start async H2D into the non-active slot. No-op if already resident. If a different prefetch is in flight, syncs it first (rare; usually waits on the slot's `compute_done` event so the next compute step's H2D can start without host stall). |
| ⭐ `BlockOffloader::await_block(&mut self, idx)` | `offload/mod.rs:1227` | Returns `Arc<HashMap<String, Tensor>>`. Legacy API — no scoped handle, so the next slot-reusing `prefetch_block` falls back to host-side `cudaDeviceSynchronize`. Migrate hot paths to `await_block_handle`. |
| ⭐ `BlockOffloader::await_block_handle(&mut self, idx)` | `offload/mod.rs:1337` | Returns a scoped [`BlockHandle`]. Slot reuse waits on the handle's `compute_done` event via `cudaStreamWaitEvent` instead of host sync. Preferred. |
| ⭐ `BlockOffloader::ensure_block(&mut self, idx)` | `offload/mod.rs:1357` | Backward-compat: `prefetch_block` + `await_block` in one call. |
| ⭐ `BlockOffloader::ensure_ring(&mut self, max_slot_bytes)` | `offload/mod.rs` (Phase 2) | Lazy-materialize a 4-slab `RingAllocator` sized `max(256 MiB, 2 × max_slot_bytes)`. No-op if already initialized. Phase 2 post-reboot: ring is the backing store for BF16 slot buffers on the prefetch path, enabling `FLAME_ALLOC_POOL=1` under offload. |
| ⭐ `BlockOffloader::alloc_bf16_via_ring(&mut self, num_elems)` | `offload/mod.rs` (Phase 2) | Allocate `num_elems` BF16 from the ring's `forward_handle(0)`, synth a `CudaSlice<u16>` via the cudarc 0.11.x mirror, register the pointer as external in the global pool so `push_u16` routes through `reconstruct_and_forget` on return. |
| `BlockOffloader::evict_block(&mut self)` | `offload/mod.rs:1369` | Drop GPU tensors from both slots. Drains in-flight transfers and does a single host sync before freeing — pre-shutdown / mode-switch path, not hot loop. |
| `BlockOffloader::block_count(&self)` | `offload/mod.rs:1387` | Accessor. |
| `BlockOffloader::pinned_bytes(&self)` | `offload/mod.rs:1392` | Total pinned CPU bytes (excludes GPU slots). |

Facilitator implementations live in trainer / inference-flame crates:
- `KleinFacilitator` — `klein-trainer/src/facilitator.rs`
- `ChromaFacilitator` — `chroma-trainer/src/facilitator.rs`
- `WanFacilitator` — `wan-trainer/src/facilitator.rs`
- `Gemma3Facilitator`, `MistralFacilitator`, `Flux1Facilitator`, etc. — `inference-flame/src/models/*`
- `Wan22Dit::load_shared_only` — `inference-flame/src/models/wan22_dit.rs` — shared-only constructor (no block weights) used alongside `BlockOffloader` for 14B+ Wan2.2 experts.

### Block offload telemetry — `offload::telemetry` (Phase 1 FlexTensor port, 2026-05-12)

⭐ Per-prefetch / per-step counters and an opt-in event ring buffer. Hooks
into `prefetch_block` / `await_block` are cheap when disabled (single
relaxed atomic load); enabled via env var `FLAME_OFFLOAD_TELEMETRY=on`
(counters) / `=trace` (counters + ring buffer), or via
`telemetry::global().set_enabled(true)`. See
`offload/telemetry.rs` for the registry-style API and
[`FLAME_MODULES.md`](./FLAME_MODULES.md#offload-telemetry--offloadtelemetry-phase-1-flextensor-port-2026-05-12)
for the module overview.

| Symbol | File:line | Notes |
|---|---|---|
| ⭐ `offload::telemetry::global()` | `offload/telemetry.rs:411` | Returns `&'static Telemetry`. Lazily initialized; environment defaults applied at first call. |
| `offload::telemetry::Telemetry` | `offload/telemetry.rs:136` | Process-global atomic counter bag + bounded ring buffer. `Send + Sync`. |
| `offload::telemetry::TelemetryCounters` | `offload/telemetry.rs:36` | Plain-data snapshot (`Clone`). Has `effective_h2d_bps()` and `await_hit_ratio()` accessors. |
| `offload::telemetry::TelemetryEvent` / `TelemetryEventKind` | `offload/telemetry.rs:91` | Per-event trace record (only populated in trace mode). |
| `Telemetry::is_enabled` / `is_trace_enabled` | `offload/telemetry.rs:177-181` | Cheap relaxed loads. |
| `Telemetry::set_enabled` / `set_event_log_capacity` / `reset` / `snapshot` | `offload/telemetry.rs:185-249` | Mode + state controls. `snapshot` is 7 atomic loads. |
| `Telemetry::record_prefetch_begin` / `record_prefetch_end` / `record_prefetch_already_resident` | `offload/telemetry.rs:268-302` | Hooks for `prefetch_block`. Pair: begin returns timer, end consumes it. |
| `Telemetry::record_await_begin` / `record_await_end_hit` / `record_await_end_miss` | `offload/telemetry.rs:305-336` | Hooks for `await_block`. "Hit" = slot already had the block resident; "miss" = await had to issue its own H2D. |
| `offload::telemetry::format_counters` | `offload/telemetry.rs:380` | Stable diagnostic string for `eprintln!` / log output. |

### Block offload transfer benchmark — `offload::transfer_benchmark` (Phase 1 FlexTensor port, 2026-05-12)

⭐ One-time PCIe H2D/D2H bandwidth sweep at process init. Builds a
log-log interpolator usable for future strategy code (Phase 2). Init-only,
not on the per-step path — uses `cudaEventSynchronize` for device-observed
wall time, which is exempt under clause 1 of `SPEED_CONTRACT.md` for init
time. See `offload/transfer_benchmark.rs` and
[`FLAME_MODULES.md`](./FLAME_MODULES.md#offload-transfer-benchmark--offloadtransfer_benchmark-phase-1-flextensor-port-2026-05-12)
for the module overview.

| Symbol | File:line | Notes |
|---|---|---|
| `offload::transfer_benchmark::run_benchmark(device, cfg)` | `offload/transfer_benchmark.rs:295` | Run the sweep. Allocates one pinned host buffer + one GPU buffer at `cfg.max_bytes`, reused across all samples. Returns `TransferBandwidthProfile`. |
| `offload::transfer_benchmark::BenchmarkConfig` | `offload/transfer_benchmark.rs:138` | Knobs: `min_bytes`, `max_bytes` (default 1 KiB → 256 MiB), `samples`, `trials`, `warmup_trials`, `measure_d2h`. |
| `offload::transfer_benchmark::TransferBandwidthProfile` | `offload/transfer_benchmark.rs:176` | Holds measured H2D/D2H curves and peak bandwidth fields. `Clone`. |
| `TransferBandwidthProfile::predict_h2d` / `predict_d2h` | `offload/transfer_benchmark.rs:198,205` | Log-log interpolation, byte size → `Duration`. Linear extrapolation beyond range using endpoint slope. |
| `TransferBandwidthProfile::format_table` | `offload/transfer_benchmark.rs:213` | Diagnostic table matching FlexTensor's `format_memory_transfer_table`. |
| `offload::transfer_benchmark::TransferMeasurement` / `TransferDirection` | `offload/transfer_benchmark.rs:115,121` | Per-point result. |

### Block offload strategy — `offload::strategy` (Phase 2 FlexTensor port, 2026-05-13)

⭐ Opt-in resident-set strategy layer for `BlockOffloader`. When no
strategy is attached (default) the offloader runs the pre-Phase-2 code
paths bit-for-bit unchanged — klein 9B loss invariant preserved.
Attached via `BlockOffloader::set_strategy` / `with_strategy`. Strategy
decisions are advisory: the offloader honors what fits the 2-slot
mechanic and emits `target_resident_bytes` + decision counts into the
Phase 1 telemetry sink.

| Symbol | File:line | Notes |
|---|---|---|
| `offload::strategy::Strategy` trait | `offload/strategy.rs` | `plan(&mut self, &OffloaderState) -> ResidentPlan`. `Send + Sync`. Pure host logic — no CUDA calls. |
| `offload::strategy::OffloaderState` | `offload/strategy.rs` | Cheap snapshot: block_count, block_sizes, resident, requested, access_history, free/total VRAM, AccessHints. |
| `offload::strategy::ResidentPlan` | `offload/strategy.rs` | `evict` / `keep` / `prefetch` Vec<usize> + `target_resident_bytes`. Has `keep_bytes(sizes)`. |
| `offload::strategy::TwoSlot` | `offload/strategy/two_slot.rs` | Default 2-slot ping-pong reformulated as a Strategy. Stateless. Regression gate `two_slot_matches_hardcoded_pre_phase2` proves bit-identical eviction picks vs the hardcoded path. |
| `offload::strategy::Knapsack` | `offload/strategy/knapsack.rs` | Value-based selection bounded by a byte budget. Greedy value-per-byte sort (~5% of optimal at <1 µs). Score = inverse recency + frequency + requested bonus. `with_budget(bytes)` / `unbounded()`. |
| `offload::strategy::Adaptive` | `offload/strategy/adaptive.rs` | VRAM-pressure-driven sizing. Shrinks above high_watermark (default 0.85), grows below low_watermark (default 0.60), holds in between (hysteresis). Wraps a Knapsack with a dynamic budget. |
| `offload::planner::PlanRequest` / `PlanResult` | `offload/planner.rs` | Plain capacity-bounded block selection. `plan_smallest_first` (maximize count) / `plan_largest_first` (fewer transfers) / `max_block_bytes`. Used by strategies and callers that want a "just fit" baseline. |
| ⭐ `BlockOffloader::set_strategy(Box<dyn Strategy>)` | `offload/mod.rs:911` | Attach strategy; opt-in only. Default = no strategy = bit-identical pre-Phase-2 behavior. |
| `BlockOffloader::with_strategy(self, …)` | `offload/mod.rs:916` | Builder-style variant. |
| `BlockOffloader::clear_strategy()` | `offload/mod.rs:923` | Detach. |
| `BlockOffloader::strategy_name()` | `offload/mod.rs:929` | `"none"` when unset, else the active strategy's `name()` (`"two_slot"`, `"knapsack"`, `"adaptive"`). |
| `BlockOffloader::block_sizes()` | `offload/mod.rs:940` | Per-block BF16 footprint, for sizing strategy budgets externally. |
| `BlockOffloader::resident_blocks()` | `offload/mod.rs:946` | Block IDs currently parked on a GPU slot. |
| `Telemetry::record_strategy_decision(name, evicted, kept, target_bytes)` | `offload/telemetry.rs:383` | Strategy decision hook (cheap when telemetry disabled). Adds `strategy_plans`, `strategy_eviction_decisions`, `strategy_keep_total`, `strategy_last_target_resident_bytes` to `TelemetryCounters`. |

### Block offload manager — `offload::manager` (Phase 3 FlexTensor port, 2026-05-12)

⭐ Wraps a `BlockOffloader` in a `NotInitialized → Discovery → Profiling →
Active` state machine. Auto-selects a `Strategy` at `activate()` time
based on observed VRAM headroom (`TwoSlot` when block fits comfortably,
`Adaptive` otherwise). Trainers opt in by constructing an
`OffloadManager` instead of using `BlockOffloader` directly. See
[`FLAME_MODULES.md`](./FLAME_MODULES.md#offload-manager--offloadmanager-phase-3-flextensor-port-2026-05-12).

| Symbol | Location | Purpose |
|---|---|---|
| ⭐ `offload::OffloadManager` | `offload/manager.rs:174` | State-machine wrapper around `BlockOffloader`. Re-exported at `flame_core::offload::OffloadManager`. |
| `offload::OffloadPhase` | `offload/manager.rs:82` | `NotInitialized` / `Discovery` / `Profiling` / `Active`. `as_str()` returns the stable name. |
| `offload::ManagerConfig` | `offload/manager.rs:118` | Tunables: `bench_config`, `force_strategy`, `cache_path`, `vram_headroom_bytes`, `low_pressure_keep_fraction`. |
| `offload::ForcedStrategy` | `offload/manager.rs:157` | `Auto` / `TwoSlot` / `Knapsack(budget)` / `Adaptive`. Setting anything other than `Auto` skips the auto-decision in `activate()`. |
| `OffloadManager::new(device, offloader)` | `offload/manager.rs:203` | Construct in `NotInitialized` with default config. |
| `OffloadManager::with_config(device, offloader, config)` | `offload/manager.rs:208` | Same with custom `ManagerConfig`. |
| `OffloadManager::load(...)` | `offload/manager.rs:226` | One-shot ctor: pass `BlockOffloader::load` args through. |
| `OffloadManager::phase()` | `offload/manager.rs:238` | Current `OffloadPhase`. |
| `OffloadManager::offloader()` / `offloader_mut()` | `offload/manager.rs:243-250` | Borrow the wrapped `BlockOffloader`. |
| `OffloadManager::block_sizes()` | `offload/manager.rs:256` | Forwarded accessor. |
| `OffloadManager::bandwidth_profile()` | `offload/manager.rs:262` | `Option<&TransferBandwidthProfile>` — `None` until `Profiling` finishes. |
| `OffloadManager::active_strategy_name()` | `offload/manager.rs:268` | `"none"` until `activate()`, then the chosen strategy's name. |
| `OffloadManager::into_offloader()` | `offload/manager.rs:275` | Tear the manager apart, return the underlying offloader. |
| `OffloadManager::discover()` | `offload/manager.rs:289` | `NotInitialized → Discovery`. Snapshots block geometry. |
| `OffloadManager::run_profile()` | `offload/manager.rs:309` | `Discovery → Profiling`. Loads cached profile from disk if present, else runs `transfer_benchmark::run_benchmark` and writes the result back. |
| `OffloadManager::activate()` | `offload/manager.rs:381` | `Profiling → Active`. Picks + installs a strategy. Reads `cuda_mem_get_info` once for the headroom test. |
| `OffloadManager::discover_profile_activate()` | `offload/manager.rs:418` | One-shot: do all three transitions back-to-back. |

### Block offload strategy constructors — `offload::strategy::*` (Phase 2 FlexTensor port, 2026-05-13)

Strategy impl builders for callers that want to attach a strategy manually
(via `BlockOffloader::set_strategy`) instead of relying on `OffloadManager`.

| Symbol | Location | Purpose |
|---|---|---|
| `strategy::TwoSlot::new()` | `offload/strategy/two_slot.rs:29` | Default ping-pong. Stateless. |
| `strategy::Knapsack::with_budget(bytes)` | `offload/strategy/knapsack.rs:84` | Greedy value-per-byte under a fixed byte budget. |
| `strategy::Knapsack::unbounded()` | `offload/strategy/knapsack.rs:94` | No byte cap; still emits a priority ordering. |
| `strategy::Knapsack::with_weights(w)` | `offload/strategy/knapsack.rs:103` | Override scoring weights. |
| `strategy::ValueWeights` | `offload/strategy/knapsack.rs:40` | `recency` / `frequency` / `requested_bonus` (`f32`s). |
| `strategy::Adaptive::new()` | `offload/strategy/adaptive.rs:60` | VRAM-pressure-driven (defaults: low=0.60, high=0.85, shrink=0.5). |
| `strategy::Adaptive::with_watermarks(low, high)` | `offload/strategy/adaptive.rs:72` | Builder. |
| `strategy::Adaptive::with_shrink_fraction(f)` | `offload/strategy/adaptive.rs:83` | Builder. |
| `strategy::Adaptive::with_value_weights(w)` | `offload/strategy/adaptive.rs:89` | Builder; forwarded to the inner Knapsack. |
| `strategy::Adaptive::last_observed_pressure()` | `offload/strategy/adaptive.rs:97` | Last VRAM-pressure ratio plan() saw. |
| `strategy::Adaptive::last_target_bytes()` | `offload/strategy/adaptive.rs:103` | Last target resident-set size. |

### Block offload state persistence — `offload::state` (Phase 3 FlexTensor port, 2026-05-12)

JSON serde for the `TransferBandwidthProfile`, so a profile sweep
amortizes across runs.

| Symbol | Location | Purpose |
|---|---|---|
| `offload::state::DEFAULT_PROFILE_FILENAME` | `offload/state.rs:44` | `"offload_profile.json"`. |
| `offload::state::PROFILE_PATH_ENV` | `offload/state.rs:47` | `"FLAME_OFFLOAD_PROFILE_PATH"`. |
| `offload::state::default_profile_path()` | `offload/state.rs:133` | Honors `$FLAME_OFFLOAD_PROFILE_PATH` → else `$XDG_CACHE_HOME/flame-core/offload_profile.json` → else `~/.cache/flame-core/...`. |
| `offload::state::save_profile(path, profile)` | `offload/state.rs:160` | Atomic-write JSON. Schema-versioned (`SCHEMA_VERSION = 1`). |
| `offload::state::load_profile(path)` | `offload/state.rs:193` | Read + version-check. Schema mismatch → error so the caller re-benches. |
| `offload::state::relative_error(a, b)` | `offload/state.rs:224` | Helper for parity tests. |

### Block offload telemetry export — `offload::telemetry` (Phase 4, 2026-05-12)

⭐ On-disk exporters that surface the in-process telemetry state without
source edits in every trainer. Atomic writes (tmp-file + rename); no CUDA
calls. See [`OFFLOAD_GETTING_STARTED.md`](./OFFLOAD_GETTING_STARTED.md).

| Symbol | Location | Purpose |
|---|---|---|
| `offload::telemetry::snapshot_to_file(path)` | `offload/telemetry.rs` | JSON-serialize the current counter snapshot to `path` atomically. |
| `offload::telemetry::ring_buffer_to_file(path)` | `offload/telemetry.rs` | JSON-lines dump of the per-event ring buffer. Returns count written. |
| `offload::telemetry::dump_all(dir)` | `offload/telemetry.rs` | Convenience pair-dump. Honors `$FLAME_OFFLOAD_TELEMETRY_DUMP_DIR` when `dir = None`. |
| `Telemetry::set_periodic_dump_interval(N)` | `offload/telemetry.rs` | Every `N` recorded events trigger a `dump_all` to the configured dir. `0` disables. |
| `Telemetry::periodic_dump_interval()` | `offload/telemetry.rs` | Read back the current interval. |
| `offload::telemetry::DUMP_SNAPSHOT_FILENAME` | `offload/telemetry.rs` | `"flame_offload_telemetry_snapshot.json"`. |
| `offload::telemetry::DUMP_EVENTS_FILENAME` | `offload/telemetry.rs` | `"flame_offload_telemetry_events.jsonl"`. |
| `offload::telemetry::DUMP_DIR_ENV` | `offload/telemetry.rs` | `"FLAME_OFFLOAD_TELEMETRY_DUMP_DIR"`. |
| `offload::telemetry::DUMP_INTERVAL_ENV` | `offload/telemetry.rs` | `"FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_EVENTS"` — counter ticks per event (record_prefetch_end / record_await_end_{hit,miss}), NOT per training step. Legacy `_STEPS` alias still accepted but deprecated. |
| `offload::telemetry::DUMP_INTERVAL_ENV_LEGACY` | `offload/telemetry.rs` | Deprecated alias `"FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_STEPS"`. Kept for back-compat. |
| `TelemetryCounters` / `TelemetryEvent` / `TelemetryEventKind` | `offload/telemetry.rs` | All three now derive `Serialize` / `Deserialize`. |

Deliberately *still* deferred (not in Phase 3 or 4): FlexTensor's
`tensor_discovery` / `trap_tensor_mode` (rely on PyTorch
`__torch_function__`), `memory_block_planner.py` adjacency-graph coloring
(flame-core's blocks are flat IDs from `BlockFacilitator`), `shm/`
cross-process plumbing (single-process trainers only).

### Gradient utilities
- `gradient::GradientMap / TensorGradExt` — re-exported as `GradientMap`
- `gradient_clip::*` — gradient clipping
- `gradient_checkpointing::*` — activation checkpointing helpers

---

## Optimizers

- `adam::AdamW` — re-exported as `nn::AdamW`. Standard AdamW with BF16 master / F32 moments; `set_lr()` supports runtime schedulers. DECOUPLED weight decay. Two fused-kernel paths:
  - Single-tensor kernels (`adam_fused_bf16_kernel` etc., `adam.rs:54-225`) — fallback for mixed-dtype slices or when `FLAME_ADAM_NO_MULTI_TENSOR=1`.
  - Multi-tensor BF16 kernel (`adam_fused_multi_bf16_f32grad_kernel`, `adam.rs:259+`) — auto-selected when **all** params are BF16 and **all** grads are F32 (Klein 9B / dominant LoRA case). One kernel launch covers every parameter. Backed by a cached device-side metadata buffer (`fused::MultiTensorMetaCache`).
  - Multi-tensor F32 kernel (`adam_fused_multi_f32param_f32grad_kernel`, `adam.rs:305-359`) — added Phase 1 of the 2026-05-12 launch-storm refactor. Auto-selected when **all** params are F32 and **all** grads are F32 (zimage LoRA / no-quant trainers). Same packed-buffer pattern as BF16 path, no casts, no stoch. Bit-identical to per-tensor `adam_fused_f32param_f32grad_kernel`.
  - **Stochastic-round variants** (added 2026-05-08): `adam_fused_bf16_f32grad_stoch_kernel` + `adam_fused_multi_bf16_f32grad_stoch_kernel`. Identical math to the round-to-nearest variants except the final F32 → BF16 store applies lower-16-bit hash-driven stochastic rounding seeded from the step counter. Toggled via `Adam::set_stochastic_round(true)` / `AdamW::set_stochastic_round(true)`. Off by default (byte-identical to prior). Only fires for BF16-storage params; F32-storage trainers automatically take the new F32-param multi-tensor path.
- `adam::adam_fused_multi_tensor_step` (re-exported, `adam.rs:858`) — direct launcher for parity-test access. Signature is `(cache, device, n, param_is_bf16: bool, grad_is_bf16: bool, packed, …, stoch_seed: Option<u64>)`. The `param_is_bf16` discriminator (added Phase 1, 2026-05-12) routes between BF16 and F32 multi-tensor kernels; `(F32, BF16)` combo returns Err — caller must route to per-param fallback. Production code uses `Adam::step` / `AdamW::step` instead.
- `adam::adam_fused_step_f32` (re-exported, `adam.rs:707`) — single-tensor F32 variant. Used as the parity baseline for the F32 multi-tensor kernel and the fallback for `(F32 param, BF16 grad)` combos.
- `adam::adam_fused_step` (re-exported, `adam.rs:546`) — single-tensor variant; same `stoch_seed: Option<u64>` addition.
- `adam::Adam::set_stochastic_round(bool)` / `Adam::is_stochastic_round() -> bool` — toggle and read of the stochastic-round flag (added 2026-05-08).
- `adam::AdamW::set_stochastic_round(bool)` / `AdamW::is_stochastic_round() -> bool` — same on AdamW (forwards to Adam).
- `adam::MultiTensorMetaCache` (re-exported, `adam.rs:347`) — cache type held by `Adam` for reuse across steps. Reallocates when n changes.
- `adam8bit_kernel::create_dynamic_map(signed) -> [f32; 256]` — `adam8bit_kernel.rs`. CPU-side port of bitsandbytes 0.49.2 `functional.create_dynamic_map` (the 8-bit dynamic-exponent LUT). Used by `eridiffusion_core::AdamW8bit`; allocate the two LUTs (signed for `m`, unsigned for `v`) once per device and upload via `upload_qmap`.
- `adam8bit_kernel::upload_qmap(device, &[f32; 256]) -> Result<CudaSlice<f32>>` — `adam8bit_kernel.rs`. Single H2D copy of a 256-entry F32 LUT.
- `adam8bit_kernel::alloc_state(device, numel) -> Result<(CudaSlice<u8>, CudaSlice<u8>, CudaSlice<f32>, CudaSlice<f32>)>` — `adam8bit_kernel.rs`. Allocates the per-parameter 8-bit state for one tensor: `(m_codes, v_codes, m_absmax, v_absmax)`, all zero-initialized. Absmax buffers are sized `ceil(numel / 256)`.
- `adam8bit_kernel::adam8bit_step_bnb(param, grad, m_codes, v_codes, m_absmax, v_absmax, qmap_signed, qmap_unsigned, lr, beta1, beta2, eps, wd, bc1, bc2)` — `adam8bit_kernel.rs`. Fused NVRTC kernel: dequant + AdamW step + requant in one launch per parameter, no host round-trip. F32 param required (use a master shadow for BF16 params); F32 or BF16 grad auto-routed. Bit-exact-equivalent (modulo BF16 noise) to bitsandbytes 0.49.2 `optim.AdamW8bit(block_wise=True)`. Block size = 256.
- `adam8bit_kernel::ADAM8BIT_BLOCK_SIZE` — `adam8bit_kernel.rs`. `const usize = 256`. Block size for both quantization and the block-wide max-abs reduction. Matches bnb `optimizer.py:478`.
- `int8_weight_only_qt_kernel::quantize_int8_qt(weight: &[f32], shape: [usize; 2], source_is_bf16: bool) -> (Vec<i8>, Vec<f32>)` — `int8_weight_only_qt_kernel.rs`. CPU-side port of torchao 0.14.1 `prototype.quantized_training.int8.quantize_int8_rowwise` (stochastic_rounding=False path). Symmetric per-row absmax INT8 quant with divisor=127 (not 127.5). Uses round-ties-to-even (matches PyTorch's `tensor.round()`, not Rust's default `f32::round`). When `source_is_bf16=true` simulates the BF16 absmax/scale round-trip torchao does on BF16 nn.Linear weights — SimpleTuner always passes BF16 base weights so this is the realistic path.
- `int8_weight_only_qt_kernel::INT8_QT_EPS` — `int8_weight_only_qt_kernel.rs`. `const f32 = 1e-12`. Eps applied to the `inv_scale` divisor only (the stored `scale` is left unguarded). Matches torchao default at int8.py:25.
- `int8_weight_only_qt_kernel::Int8QtWeight` — `int8_weight_only_qt_kernel.rs`. Device-side container for one int8-quantized linear weight: raw `CudaSlice<i8>` codes (TensorStorage has no I8 arm) + frozen F32 scales `Tensor` + `[out, in]` shape. Codes/scales never receive a gradient (LoRA-only training; base is frozen).
- `int8_weight_only_qt_kernel::Int8QtWeight::upload(codes, scales, shape, device) -> Result<Self>` — `int8_weight_only_qt_kernel.rs`. H2D upload of host-quantized codes + scales.
- `int8_weight_only_qt_kernel::Int8QtWeight::codes_to_bf16(&self) -> Result<Tensor>` — `int8_weight_only_qt_kernel.rs`. Lazy NVRTC int8→bf16 cast of the stored codes, returns a frozen `[out, in]` BF16 tensor. Compiles `i8_to_bf16_kernel` on first call per device.
- `int8_weight_only_qt_kernel::linear_int8_qt(x: &Tensor, w: &Int8QtWeight, bias: Option<&Tensor>) -> Result<Tensor>` — `int8_weight_only_qt_kernel.rs`. torchao `_Int8WeightOnlyLinear` parity forward: `y = (x @ codes_to_bf16(w).T) * scale + bias?`. Autograd-tracked through `x` (and `bias` if it requires grad); the int8 weight is frozen. Used by trainers selecting `--base-quant int8-torchao` for SimpleTuner parity. Validated to bit-exact codes+scales+forward+grad_x vs torchao 0.14.1 via `parity_int8_torchao_qt`.
- `sgd::*` — basic SGD
- `parameter::Parameter` — re-exported as `Var` and `Parameter`. Wraps a `Tensor` with `requires_grad=true`.
- `nn::Optimizer` trait — `lib.rs:258` — `step()` + `zero_grad()`
- ⭐ `ops::grad_norm::global_l2_norm(grads)` — `ops/grad_norm.rs:62`. Device-resident global L2 norm of a slice of gradient tensors. Returns 1-element FP32 device tensor; caller decides when (if ever) to `.item()`. Mixed-dtype (BF16 + FP32) supported, casts internally. **Phase 3 multi-tensor fast path (2026-05-12):** when every grad is F32 + contiguous, dispatches to `multi_tensor_l2_norm_sq_f32` (3 launches total instead of 2N+(N-1)+1). Falls through to legacy per-tensor fold otherwise. Env override: `FLAME_MT_L2NORM=0` forces legacy.
- ⭐ `ops::grad_norm::global_l2_norm_with_scale(grads, max_norm, eps)` — `ops/grad_norm.rs:103`. Same but also returns the clip-scale factor as a 1-element device tensor. One D2H sync at the end if logging needed.
- `ops::multi_tensor::multi_tensor_l2_norm_sq_f32(cache, &[&Tensor]) -> Tensor` — `ops/multi_tensor.rs`. Two-stage Apex-style reduction kernel. Stage 1 = block-per-tensor sum-of-squares in shared memory → partials[N]. Stage 2 = single-block reduction across partials → F32[1]. F32 grads + contiguous required; legacy fallback in caller handles BF16. Parity ≤ 1e-5 abs / 1e-6 rel vs legacy fold (parallel-tree reordering, not bit-exact).
- `ops::multi_tensor::MultiTensorMetaCache` — `ops/multi_tensor.rs`. Process-wide cache for the L2 norm packed buffer + per-tensor partials buffer. Held behind a `Mutex` in `ops::grad_norm` (`MT_L2_CACHE`). Reallocates when n_tensors changes (one-time on step 0 in steady training). Note: this is a **separate** cache from `adam::MultiTensorMetaCache` — region layouts differ.
- `ops::multi_tensor::multi_tensor_scale_inplace_packed(cache, dev, n, &packed, scale, is_bf16) -> Result<()>` — `ops/multi_tensor.rs`. Single-launch in-place `x[i] *= scale` across a packed list of F32 or BF16 tensors. Targets the trainer clip-grad path (`train_zimage.rs`, `train_klein.rs`): collapses N per-parameter `mul_scalar` launches into one grid-per-tensor launch when `total_norm > clip`. Packed layout = `[ptrs(n) | sizes(n)]` (2n u64 entries). F32 path is bit-exact vs per-tensor `mul_scalar`; BF16 within 1 ULP. **Default off in callers:** zimage and klein only enable via `FLAME_MT_SCALE=1` env var because production grad-norms stay below clip threshold (Phase 2 of launch-storm refactor, 2026-05-12; see `EriDiffusion-v2/HANDOFF_2026-05-12_PHASE2_SCALE_FOLLOWUP.md`).
- `Tensor::as_mut_device_ptr_f32(tag) -> Result<u64>` — `tensor.rs:~605`. Raw mutable F32 device pointer as a `u64`. Mirrors `as_mut_device_ptr_bf16`. Intended for callers building packed pointer buffers for multi-tensor kernels without taking cudarc as a direct dependency.

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
- `tensor_narrow.rs` — narrow helper. Public methods on `Tensor`:
  - `narrow_general_cuda(dim, start, length) -> Result<Tensor>` (`:10`) — strided narrow forward; invokes `flame_narrow_strided_launch` (kernel-arg meta, no host sync). Reference impl for SPEED_CONTRACT clause 1 — see `docs/SPEED_CONTRACT.md` "Two correct primitive patterns".
  - `narrow_cuda(dim, start, length) -> Result<Tensor>` (`:119`) — narrow assuming contiguous input.
  - ⭐ `Tensor::narrow_backward_scatter_add_cuda(grad_out, grad_in, dim, start, length) -> Result<()>` (`:169`) — scatter-add backward for narrow. **Post-Class-B (2026-05-12, commit `15d8ef8`)** the BF16 storage arm dispatches directly; pre-fix there was a BF16→F32→BF16 cast detour costing 3 extra kernel launches per call. F32 + BF16 are both first-class now (dtype-agnostic byte-copy kernel underneath). Used by every autograd backward that splits a tensor with `narrow` (concat backward, chunked attention backward, etc.).
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
- `rng/torch_compat.rs` — PyTorch-parity CUDA RNG primitives. Same `port-parity` SM-count caveat applies to all. ⭐ Used by `port-parity` flow.
  - `randn_torch(seed, shape, device) -> Tensor` — bit-exact `torch.randn(generator=manual_seed(seed))` for CUDA F32.
  - `rand_torch(seed, shape, device) -> Tensor` — bit-exact `torch.rand` (uniform [0, 1)) for CUDA F32.
  - `bernoulli_torch(seed, shape, p, device) -> Tensor` — bit-exact `torch.empty(shape).bernoulli_(p, generator=g)` for CUDA F32 (0.0/1.0 mask).
  - `randint_torch(seed, low, high, shape, device) -> Tensor` — bit-exact `torch.randint(low, high, shape)` for CUDA I32. Range constraint: `high - low < 2^32`.
  - `kaiming_uniform_torch(shape, a: f64, fan, nonlinearity, seed, device) -> Tensor` — bit-exact `torch.nn.init.kaiming_uniform_` (CUDA F32). `a` is f64 to match PyTorch's Python-float gain math: a narrowed-to-f32 `a` or `gain` perturbs the final `bound` by 1 ulp on non-power-of-2 fans (e.g. 137, 1280). Internal compute runs in f64; only the final `bound: f32` is handed to the uniform kernel.
  - `xavier_uniform_torch(shape, gain: f64, fan_in, fan_out, seed, device) -> Tensor` — bit-exact `torch.nn.init.xavier_uniform_` (CUDA F32). `gain` is f64 for the same reason: callers passing `math.sqrt(2.0)` (the standard relu gain) must not narrow to f32 before the call, or `bound` will be 1 ulp off PyTorch's.
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
- **"Where is the MXFP4 dequant?"** → `ops::fused_inference::dequant_mxfp4_to_bf16` →
  `flame_mxfp4_to_bf16` → `src/cuda/mxfp4_dequant.cu`. Used by Lens M2 (GPT-OSS encoder)
  `flame_fp16_to_bf16` → `src/cuda/fp16_to_bf16.cu`
- **"Where is the activation offload pool?"** → `activation_offload::ActivationOffloadPool` →
  autograd integration via `autograd::checkpoint_offload` + `Op::CheckpointOffload`.
  FP8 quant kernel: `src/cuda/fp8_quant.cu`. Trainer setup: `flame-diffusion/src/offload.rs`.
- **"Where is the BLOCK offloader (not activation)?"** → `flame_core::offload::BlockOffloader`
  in `src/offload/mod.rs`. Moved into flame-core 2026-05-12 (commit `df00c5f`).
  Pair with a `BlockFacilitator` trait impl per model. Prefer
  `await_block_handle` (scoped, event-driven) over `await_block` (host-sync).
- **"Where is the speed-contract audit gate?"** → `docs/SPEED_CONTRACT.md`. Tenets it derives from: `../TENETS.md`. Reference impls for clause 1: `cuda/narrow_strided.cu` (inline meta) and `cuda/sdpa_stream_bf16.cu` (cached workspace).
- **"Where is the BF16→FP8 quantize kernel?"** → `flame_bf16_to_fp8` →
  `src/cuda/fp8_quant.cu` (used by activation offload FP8 compression)
- **"Where are the QwenImage trainer parity tests?"** →
  Forward: `flame-diffusion/qwenimage-trainer/src/bin/parity_test.rs` +
  `tools/dump_forward.py`.
  Training: `src/bin/train_parity_test.rs` + `tools/dump_training_steps.py`.
  Sampler: `tools/compare_sampler.py`. See CONVENTIONS §7-9 for bugs found.
