# flame-core module map

> One paragraph per public module. Read this once at session start to know
> where things live. ⭐ marks modules that `inference-flame` actually depends
> on. ⚠️ marks legacy / training-only / dead code areas.

flame-core is a 100K+ line Rust+CUDA library that grew from a training
framework into a hybrid training/inference codebase. It has multiple
generations stacked on top of each other (autograd v3 / v4, sdpa /
sdpa_legacy / wmma flash, conv2d via four different paths). The active
inference path is mostly the BF16 + cuBLASLt + wmma combo described in
[`FLAME_INDEX.md`](./FLAME_INDEX.md). The training path is mostly the F32 +
NVRTC kernels in `cuda_kernels*.rs`.

---

## Core types

### ⭐ `tensor.rs` — the central `Tensor`
The whole library hangs off this one type. `Tensor` holds a `TensorStorage`
(BF16-as-u16 / F32 / F16 / I32 etc.), a `Shape`, an `Arc<CudaDevice>`, an
`AtomicUsize` `TensorId` for autograd-tape keying, and a `requires_grad` flag.
The 114 `pub fn` methods cover construction (zeros, randn, from_vec,
from_f32_to_bf16), shape ops (reshape, view, narrow, permute, chunk, cat,
unsqueeze, squeeze), math (add, mul, matmul, bmm, silu, gelu, softmax,
sum_dim_keepdim), cast (to_dtype), and read-back (to_vec, item). Math ops
auto-route to the BF16 fast path: pointwise ops go through
`tensor_iterator::ops::{unary,binary,transcendentals,comparison}` (PyTorch
TensorIterator port, Phases 1–11), softmax last-dim goes through
`bf16_elementwise::softmax_lastdim_bf16`, and fused ops (RoPE, swiglu,
gate_residual, modulate) go through `bf16_ops::*`. The struct is large but
its discoverability is good — start in [`FLAME_INDEX.md`](./FLAME_INDEX.md)
"tensor.rs" section.

**Training-critical fixes (2026-04-09):**
- `narrow()` BF16 path now records `Op::Slice` for autograd and preserves
  `requires_grad` (both BF16 fast path AND F32 slow path patched).
- `to_dtype()` records `Op::Cast` which properly reverses the cast in
  backward — this is how F32 LoRA master params receive correct gradients
  after BF16 casts in forward.
- `bmm()` F32 path works via `launch_gemm_strided_batched`; BF16 path via
  `bmm_bf16_fp32acc_out`. Both dispatch on `self.dtype()` — mismatched
  dtypes between `self` and `other` will error.

### ⭐ `tensor_storage.rs`
The `TensorStorage` enum that backs every `Tensor`: `BF16 { data: Arc<CudaSlice<u16>>, numel }`,
`F32(...)`, `F16(...)`, `I32(...)`, `Arena { ... }` (for the staging arena
path). Helpers `slice_ref`, `ensure_unique_slice`, `wrap_slice` for the
common pattern of reading/writing the underlying `CudaSlice` while keeping
the Arc semantics correct.

### ⭐ `shape.rs`, `dtype.rs`, `error.rs`, `device.rs`
Tiny core types. `Shape` is a Vec wrapper with `dims()`, `elem_count()`,
`from_dims(&[...])`. `DType` is an enum with the supported precisions
(BF16/F16/F32/I32/I64/Bool/U8). `Error` is a single enum re-exported as both
`Error` and `FlameError` (and `Result<T, Error>` is `flame_core::Result`).
`device.rs` exposes `global_cuda_device()` (singleton `Arc<CudaDevice>` for
device 0) and the `CudaStreamRawPtrExt` helper trait for getting raw stream
pointers.

### `tensor/contracts.rs`
Layout assertions like `assert_nhwc_bf16_public(...)` used at function
boundaries to catch wrong-layout inputs early.

### `tensor_ext.rs` / `tensor_narrow.rs` / `tensor_ops_extended.rs` / `tensor_ops_missing.rs` / `tensor_compute.rs`
Extension methods on `Tensor`. `tensor_ops_missing.rs` is the catch-all for
"PyTorch has this, we need it too" — `upsample_nearest2d`, `div_scalar`,
etc. `tensor_ops_extended.rs` has 57 pub fns of more elaborate ops.

---

## Configuration / introspection

### `config.rs`
Process-wide settings. `default_dtype()` (defaults to BF16, override with
`FLAME_DEFAULT_DTYPE`), `should_use_cudnn()` (override with `FLAME_FORCE_CUDNN`),
`optimizer_moment_dtype()`, `select_optimizer_state_dtype(param_dtype)`. These
are global mutable and read by hot-path code via cached `OnceLock` reads.

### `strict.rs`
"Strict mode" — when enabled (via env var), bans implicit F32 fallbacks and
implicit clones. Provides RAII guards `allow_clone` / `allow_f32_in_kernel`
for the rare cases that need to bypass it. Used by training code to catch
silent dtype downgrades.

### `telemetry.rs` / `perf_telemetry.rs`
Counters: dtype trap events, tensor bytes allocated, op timings. Used by
`SDPA_AUTOTUNE_*` and conv2d autotune. `TelemetrySnapshot` aggregates the
counters; `reset_counters()` zeros them.

### `env_flags.rs`
Cached env-var reads. Every `std::env::var(...)` is a syscall, so the BF16
hot path uses `OnceLock`-backed cached lookups for `ALLOC_LOG`,
`FLAME_DTYPE_TRACE`, `SDXL_DEBUG_SHAPES`, `FLAME_NO_FLASH_ATTN`, etc.

### `logging.rs`, `debug_device.rs`
`logging.rs` — env_logger setup. `debug_device.rs` — `assert_cuda(tag, t)`
and `log_device(tag, t)` for "is this tensor on the device I expect?"
checks during debugging.

---

## BF16 family — the inference hot path

### ⭐ `tensor_iterator/`
The PyTorch-style TensorIterator port (Phases 1–11). Shape/stride
coalescing, broadcast, dtype promotion, `OffsetCalculator<NARGS>` with
`IntDivider<uint32_t>` fast divmod, `gpu_kernel` CUDA templates, and a
`DispatchStub` registry keyed on dtype. All BF16 pointwise ops live under
`tensor_iterator/ops/{unary,binary,transcendentals,comparison}.rs`. CUDA
functors live under `src/cuda/{unary,binary,cmp}/*.cu`. `Tensor::add / mul /
silu / gelu / ge / ...` all route here when dtype is BF16. F32 paths still
use `GpuOps`. See [`FLAME_INDEX.md`](./FLAME_INDEX.md) "tensor_iterator/ops/"
for the full op list.

**Phase 1 cache (2026-05-12, dispatch refactor):**
`tensor_iterator/cache.rs` — `Lazy<Mutex<HashMap<IterCacheKey, CachedIterGeometry>>>`
keyed on `(operand shapes, element-strides, dtypes, pending-output bitmap,
num_outputs, static_dtype, static_shape, packed flag bundle)`. On cache hit,
`TensorIteratorConfig::build` skips steps 2-4 + 6-7 (compute_shape /
compute_strides / reorder / coalesce / 32bit-indexing). Steps 1
(populate_operands), 5 (allocate_or_resize_outputs) still run because they
touch live tensors / per-call allocations. Capacity bound 4096 (drop-all on
overflow). `FLAME_TI_CACHE_DISABLE=1` is the rollback knob. Hit/miss
`AtomicU64` counters; rate-limited `log::trace!`. Mirrors the cuDNN AlgoKey
pattern at `cudnn/conv3d.rs`. The cache stores `logical_output_shape`
(= `invert_perm(shape_)`) so the hit path can replicate
`allocate_or_resize_outputs` without relying on the (already coalesce-
invalidated) `perm_`.

### ⭐ `saved_ref.rs` (added 2026-05-12, Phase 2)
PyTorch `SavedVariable` analog. `SavedRef { id, tensor, version_counter:
Arc<AtomicU32>, version_at_save: u32 }` captures the storage version at
record time so backward fails loudly if the saved tensor was mutated in
place. `SavedRef::capture(&Tensor)` and `unpack/unpack_ref() -> Result<Tensor>`.
Implementation note: the version counter is held in a **side table** keyed
by inner Arc-pointer address, NOT inside `TensorStorage`. The existing
`TensorStorage::Drop` does a hand-rolled `ptr::read` + `Arc::try_unwrap` to
return memory to the alloc pool — wrapping the storage variants in
`Arc<StorageInner<_>>` would have broken that across 6 variants. Side
table is flushed at `AutogradContext::clear()`; captured `SavedRef`s carry
private `Arc<AtomicU32>` clones so their version check stays valid across
flushes. See `tensor_storage.rs:1-256` for the table + `bump_version`.
Rollback: `FLAME_AUTOGRAD_SAVED_LEGACY=1` routes `record_op` through the
pre-Phase-2 `Vec<(TensorId, Tensor)>` path.

### ⭐ `structured/` (added 2026-05-12, Phase 4 exemplar)
PyTorch's structured-kernel pattern (meta + impl split). `kernel.rs` defines
the `StructuredKernel` trait with associated `Input<'a>` GAT and three
methods: `meta(input) -> Tensor` (validate + allocate output, no GPU
compute), `impl_(input, output) -> Tensor` (write kernel result into the
pre-allocated output), and `dispatch(input)` (meta → autograd record →
impl_). `silu.rs` is the exemplar: `SiluStructured` allocates via
`Tensor::empty_dtype` in meta and routes through
`TensorIteratorBase::build_unary_op(Some(&out), x)` in impl_ so the iterator
short-circuits its alloc path. Exposed as `Tensor::silu_structured` — the
existing `Tensor::silu` is untouched. Parity tests in
`tests/structured_silu_parity.rs`. The current exemplar is op-count-
equivalent to the existing path; the value is the seam for future `meta`
caching (caller-supplied output, scratch reuse) without touching `impl_` or
call sites.

### ⭐ `bf16_elementwise.rs`
Post-TensorIterator this file only hosts fused/structured BF16 kernels that
aren't pointwise: `softmax_lastdim_bf16` (fused last-dim softmax — one
block per row, exp+reduce+div in a single pass), `transpose2d_bf16` (2D
BF16 transpose for Klein/Mistral pre-transpose), `patchify_bf16` /
`unpatchify_bf16` (DiT patch ops). All NVRTC. The flat-path elementwise
kernels that used to live here were moved to `tensor_iterator/` in
Phase 5b–11.

### ⭐ `bf16_ops.rs`
Fused BF16 inference primitives that don't fit in `fused_inference.rs`:
`rope_fused_bf16` (interleaved-pair RoPE for FLUX/Klein/LTX/Qwen/Chroma),
`rope_fused_bf16_f32pe` (F32 pos-emb variant), `rope_halfsplit_bf16`
(Z-Image variant), `gate_residual_fused_bf16` (`x + gate*attn`),
`swiglu_fused_bf16` (`silu(gate)*up`), `modulate_pre_fused_bf16` /
`modulate_pre_split_apply_bf16` (DiT modulate), `attn_split_txt_img_bf16`,
`qkv_split_permute_bf16`. Also retains `silu_bf16` / `gelu_bf16` /
`square_bf16` / `softmax_last_dim_bf16` as parity oracles for the
TensorIterator port (not on the live inference path — `Tensor::silu` etc.
route through `tensor_iterator::ops::unary::silu_bf16_iter` now). All NVRTC.

**Hot-path BF16-contig direct wrappers (added 2026-05-12, Fix #F):**
`silu_bf16_contig_direct` / `gelu_bf16_contig_direct` /
`add_bf16_contig_direct` / `mul_bf16_contig_direct` /
`mul_scalar_bf16_contig_direct`. These wrap the existing `flame_<op>_bf16_kernel`
C entry points the slow path uses, populating `IterMetadata` inline for a
1-D contig launch — bypassing the TensorIterator config/build chain.
Bit-identical to the slow path (same C kernel). Called from `Tensor::silu`
etc. as a `dtype()==BF16 && is_contiguous() && same-shape` fast path.
Rollback: `FLAME_HOT_FAST_PATH_DISABLE=1`.

### ⭐ `bf16_reduce.rs` (added 2026-05-12, Fix #B)
BF16-native scalar reductions. Two NVRTC kernels:
- `sum_bf16_to_f32_scalar_kernel` — per-thread grid-stride F32 accumulator
  over BF16 input, shared-mem tree reduce per block, single `atomicAdd` of
  block result into a F32 scratch. block=256, grid capped at 4096 with
  grid-stride loop covering any `n`. **Single launch, no input-sized F32
  temp buffer.**
- `f32_scalar_to_bf16_kernel` — 1-thread cast with fused `* scale` arg
  (`scale=1.0` for `sum`, `scale=1/n` for `mean` — divide happens on-device,
  no host D2H sync).

Public entry points: `sum_bf16(x) -> Tensor`, `mean_bf16(x) -> Tensor`.
Wired into `GpuOps::sum` and `Tensor::mean` BF16 paths; falls through to the
legacy cast-then-F32-reduce-then-cast path on `FLAME_BF16_REDUCE_LEGACY=1`.
Mirrors `pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu`'s
`(scalar_t in, acc_t acc)` `func_wrapper` pattern.

**Landmine documented in CONVENTIONS**: legacy F32 `SUM_KERNEL` silently
drops elements past index 262144 (no grid-stride loop). New BF16 kernel is
correct for any size.

### ⭐ `bf16_convert.rs`
BF16↔F32 cast kernels. `bf16_u16_to_f32` and `f32_to_bf16_u16` are the
backing fns called from `ops::cast::cast_bf16_to_f32` / `cast_f32_to_bf16`.
2-element-per-thread vectorized. The Rust callers use `lc_pairs(n)` for the
launch.

### `bf16_normal.rs` / `bf16_factories.rs` / `bf16_clamp.rs`
RNG and factories. `bf16_normal.rs` is Box-Muller Gaussian directly into
BF16. `bf16_factories.rs` has uniform random + tensor factories.
`bf16_clamp.rs` is element clamp.

### `bf16_support.rs`
Capability checks (BF16 hardware support) — small helper module.

---

## Fused inference primitives

### ⭐ `ops/fused_inference.rs`
The "kernel calls that bypass autograd entirely" module. Each function is a
thin wrapper around a `flame_*_bf16` C entry in `cuda::ffi`. Nine pub fns:
- `dequant_fp8_to_bf16`, `dequant_fp8_to_bf16_into`, `dequant_fp8_transpose_into` — FP8 unpack
- `fused_rms_norm` — RMSNorm in one kernel
- `fused_modulate` — `(1+scale)*x + shift` 
- `fused_linear3d` — cuBLASLt 3D linear with pre-transposed weight
- `fused_linear3d_native` — same but takes PyTorch `[Cout, Cin]` weight (added 2026-04, used by every FLUX/Chroma/QwenImage block forward)
- `fused_linear3d_native_lora` — additive LoRA over `fused_linear3d_native`; byte-identical at LoRA=None, otherwise computes `base + scale * (x @ A^T @ B^T)` with autograd flowing into A/B only (added 2026-05 for HiDream-O1 decoder which owns weights via `HashMap<String, Tensor>`)
- `fused_linear3d_native_pytorch_parity` — bit-exact PyTorch linear mirror (added 2026-05-20). Same signature as `_native`; biased calls mirror `gemm_and_bias<BF16>` with 1 MiB heuristic workspace and BIAS_POINTER set before `cublasLtMatmulAlgoGetHeuristic`, while no-bias calls delegate to `_native` to match PyTorch's matmul path. ~1% perf overhead for biased calls; use when ai-toolkit per-op parity is the contract (HiDream-O1 TimestepEmbedder, BottleneckPatchEmbed). FAQ on which to pick is in [`FLAME_CONVENTIONS.md`](./FLAME_CONVENTIONS.md).
- `fused_rms_norm_modulate` — RMSNorm + modulate fused
- `fused_residual_gate` — `x + gate*attn` fused

The corresponding `.cu` files live in `src/cuda/fused_*.cu`.

### ⭐ `ops/deinterleave.rs` (added 2026-05-01)
NVRTC-compiled `float2`-vectorized deinterleave for the last dim of a
contiguous F32 tensor. One `pub fn`:
- `deinterleave_pair_f32(x: &Tensor) -> Result<(Tensor, Tensor)>` — splits
  `[..., 2K]` F32 into two `[..., K]` halves (even-indexed columns and
  odd-indexed columns). Self-contained NVRTC source, ~50 LOC.

Why this exists separately: the generic
`materialize_strided_f32_kernel` path (the one `Tensor::contiguous()`
takes for arbitrary strided views) coalesces poorly on stride-2 access
patterns — ~1.35 s per layer for 18 M-element interleaved-SwiGLU
gathers in MagiHuman. The dedicated kernel reads one `float2` per
thread (a single 8-byte coalesced load), writes 4 bytes to each output
buffer. Memory-bound at full bandwidth on a 3090 Ti (~0.5 ms for the
same workload, ~2700× speedup). Use this for any model whose MLP runs
the FLUX/Klein/MagiHuman pattern of `linear → split-into-glu/up halves
on the last dim`.

### ⭐ `ops/grouped_mm.rs` (added 2026-04-29)
Wrapper around the build-time `flame_grouped_mm_bf16` kernel
(`src/cuda/grouped_mm.cu`). One `pub fn`:
- `grouped_mm_bf16(x: &Tensor, w: &Tensor, offsets: &[i32], t_max: usize) -> Result<Tensor>`

Used by Nucleus-Image MoE expert FFN (gate-up + down projections) and
queued as the dispatch core for LLaDA2.0-Uni's MoE backbone. Offsets are
host slices, not `&Tensor`, because flame-core's `DType::I32` is
f32-bytes-relabeled and the kernel reads real `int*`. Wrapper HtoD-copies
into a temp `CudaSlice<i32>`. See `FLAME_CONVENTIONS.md` for the
underlying I32-tensor caveat.

### ⭐ `ops/fused_gated_scatter_add.rs` (added 2026-04-29)
Wrapper around the build-time `flame_fused_gated_scatter_add_bf16` kernel
(`src/cuda/fused_gated_scatter_add.cu`). One `pub fn`:
- `fused_gated_scatter_add_bf16(expert_out: &Tensor, gating: &Tensor, indices: &[i32], accum: &mut Tensor) -> Result<()>`

In-place scatter-add: `accum[indices[t]] += expert_out[t] * gating[t]`,
F32 atomicAdd. The MoE-unpermute counterpart to `grouped_mm`. Same
host-slice-for-indices convention.

### ⭐ `ops/moe_routing.rs` (added 2026-04-29)
Host-side **expert-choice** routing for MoE. Each expert independently
picks its top-C tokens (uniform C across experts), so per-expert offsets
are constant `[B*C, 2*B*C, ..., E*B*C]` and no radix sort is needed —
the original Phase 3 plan called for thrust-style sort+cumsum but
expert-choice routing eliminates it entirely.

- `expert_choice_route(affinity, capacity, route_scale) -> ExpertRoutingPlan`
- `permute_tokens(x, plan) -> Tensor` — gathers `x_flat[B*S, D]` into
  expert-major `(E*B*C, D)` order via `Tensor::index_select0`.

The plan struct holds `offsets: Vec<i32>`, `global_token_indices: Vec<i32>`,
and `gating_flat: Vec<f32>` — each plugs straight into the relevant
downstream wrapper without further conversion. Top-K runs host-side
(download → partial sort → upload indices) for now; for batch=1
inference at S~1024 / E~64 the affinity matrix is ~64K f32 per layer per
step, sub-millisecond. Swap to a CUDA top-K later only if profiling
demands.

### ⭐ `ops/nucleus_moe.rs` (added 2026-04-29)
SwiGLU MoE expert FFN composite — the Phase 4 milestone of the MoE
kernel plan. One `pub fn`:
- `nucleus_moe_expert_forward(x_flat, affinity, gate_up_w, down_w, capacity, route_scale) -> Tensor`

Chains all the lower modules: `expert_choice_route` → `permute_tokens` →
`grouped_mm_bf16(gate_up)` → `Tensor::swiglu` → `grouped_mm_bf16(down)`
→ `fused_gated_scatter_add_bf16` weighted unpermute → BF16 cast.
Mirrors `SwiGLUExperts._run_experts_grouped_mm` + the surrounding
`NucleusMoELayer.forward` logic from `transformer_nucleusmoe_image.py`.

Toy parity test: matches a bit-faithful hand-rolled scalar Rust reference
within BF16 tolerance (max abs error < 0.10) on
`(B=1, E=4, S=8, D=64, inter=64, capacity=4)`. Caller still owns the
router matmul, modulation, and shared-expert addition — those are normal
Tensor ops, not new infra.

All four modules have `#[cfg(test)]` parity tests against hand-rolled
scalar Rust references. 7 tests total. First time the underlying
`grouped_mm.cu` and `fused_gated_scatter_add.cu` kernels actually ran.

---

## Attention / SDPA — multiple paths

### ⭐ `attention/sdpa.rs`
The public attention surface. `sdpa(q, k, v, mask)`, `sdpa_with_bias(q, k, v, bias, scale)`,
`sdpa_causal(q, k, v)`, `sdpa_prefix_causal_full(q, k, v, prefix_len)`,
`attend(...)`, `attention_impl(...)`. Routes unmasked BF16+head_dim∈{64,96,128}
to the fastest available SDPA backend; structured causal and prefix-causal/full
masks should use the structured helpers instead of materializing binary masks.
True arbitrary masks remain on the compatibility path. Also defines
`MultiHeadAttention`, `RotaryEmbedding`, `TransformerBlock`, `GeGLU`,
`FeedForward`, and (legacy duplicate) `LayerNorm` structs used by training
code.

### ⭐ `sdpa.rs` (top-level)
Lower-level dispatcher used by `attention::sdpa::sdpa` and called directly
by some inference code. `forward(q, k, v, mask)`, `forward_causal(...)`,
`forward_prefix_causal_full(...)`, `forward_with_bias(...)`, `forward_v4(...)`
(feature-gated). Dispatch order:
1. **cuDNN SDPA** — primary unmasked BF16 training path, with O/Stats and
   padded Q/KV sequence lengths for backward.
2. **In-tree FA2 BF16** (`flame_flash_attention_bf16`) — unmasked/causal
   forward path for supported head dimensions `{64, 96, 128}` when cuDNN is
   not applicable or is explicitly bypassed. The 2026-05-21 O1 path added the
   causal flag, raw-logit online softmax, `exp2`/log2 scaling, and reverse
   K/V tile traversal to mirror PyTorch FlashAttention more closely.
3. **Streaming SDPA** (`sdpa_stream_bf16`) — chunked path for large masked or
   oversized shapes.
4. **cuBLASLt materialized fallback** (`forward_bf16_fallback`) — tensor cores
   but full S×S matrix allocation.

The retained in-tree wmma reference path calls FFI `flame_flash_attention_bf16` whose
implementation (`src/cuda/flash_attention_fwd.cu`) uses **FA2-style tiling**
(BQ=64, with `s_KV` shared between K and V across stages to fit SM_86's
100 KB shared-mem budget) plus `cp.async.cg` loads with V-prefetch
overlapping the online-softmax pass. 1.36×–1.63× over the pre-Phase-1
BQ=32 kernel. Further double-buffering of K is an open investment —
two attempts (BKV=48, s_P/s_S fold) either regressed perf or hit
`store_matrix_sync` corruption; see `FA2_CP_ASYNC_DESIGN.md` for
the investigation trail.

**Deferred PyTorch tile parity**: local PyTorch/FlashAttention dispatch uses
SM8x tile shapes HD64 non-causal `128x128`, HD96 non-causal `128x64`,
HD128 non-causal `128x32`, and HD96/HD128 causal `64x64`, normally with
4 warps and register-resident output accumulators. Flame's current kernel is
still the shared-memory accumulator design (`TILE_Q=64`, `TILE_KV=64`,
`NUM_WARPS=16`) because it fits HD128 inside the SM86 shared-memory budget.
Porting the exact CUTLASS/CUTE-style tiled kernel is intentionally deferred
until the HiDream-O1 trainer parity gate is closed.

### `attention/rope.rs`
RoPE precompute + apply helpers. Most callers use the inline `rope_fused_bf16`
in `bf16_ops.rs` instead.

### ⚠️ `attention/sdpa_legacy.rs` / `sdpa_legacy.rs`
Older SDPA implementations. Kept for reference / training. Do not call.

### `attention/flash_ffi.rs` / `flash_impl.rs`
Feature-gated (`flash_attn`) FFI shim for an external flash-attention
library. Not used by default — the in-tree wmma kernel is the real flash
path.

### `flash_attention.rs` (top level)
Feature-gated `flash_attn` legacy interface, separate from `attention/`.

### `sage_attention.rs`
Experimental "sage attention" — an alternative variable-rank attention.
Not currently called.

---

## Norms

### ⭐ `layer_norm.rs`
The functional `layer_norm(x, weight, bias, normalized_shape, eps)` and
`layer_norm_into(...)` plus the `LayerNorm` struct (with optional `affine`).
The kernel itself is in `cuda/cuda_ops.cu` (`layer_norm_forward_bf16_kernel`),
with backward in `cuda/src/flame_norm_bf16.cu`. Used by Z-Image / SD3 model
code.

**Fix (2026-04-13):** Autograd backward now correctly passes saved weight/bias
tensors to `layer_norm_backward_bf16` instead of `None`. Previously affine
LayerNorm silently produced zero gradients for weight/bias parameters.
Found by Codex read-only audit.

### ⭐ `group_norm.rs`
`group_norm(x, groups, gamma, beta, eps)` functional and the `GroupNorm`
struct. Used by SDXL UNet, Klein VAE, LDM VAE, LTX-2 audio VAE, LTX-2
upsampler. ⚠️ Note: `cuda_ops_bf16::group_norm_bf16` (the lower-level entry)
takes NHWC layout, not NCHW.

### ⭐ `norm.rs`
**Canonical RMSNorm entry for both training and inference** (2026-05-12).
`norm::rms_norm(x, normalized_shape, weight, eps)` records `Op::RMSNorm`
and dispatches three vectorized NVRTC kernels (`rms_norm_forward_bf16_vec`,
`rms_norm_backward_bf16_vec`, `rms_norm_grad_weight_bf16_vec`) when
`norm_size % 4 == 0`, with the original scalar kernels as a `% 4 != 0`
fallback. `cuda_ops_bf16::rms_norm_bf16` (the inference entry) delegates
here as of commit `d729ede`, so both paths share the 13–16× forward and
9–15× backward speedup. `FLAME_RMS_NORM_LEGACY=1` forces the scalar path
for A/B benchmarking. Also exposes a `#[doc(hidden)]`
`rms_norm_backward_for_bench` escape hatch used by `benches/rms_norm_vec.rs`.
The module also retains older norm wrappers (BatchNorm, etc.) used by the
training path.

---

## Conv

### ⭐ `conv.rs`
The main `Conv2d` struct + `Conv2dConfig`. `Conv2d::new`, `new_with_bias`,
`new_zeroed`, `new_with_bias_zeroed`. `forward(input)` is NCHW; there's
also a `forward_nhwc(input)` fast path. `conv2d_forward(...)` is the
functional API. This is the only conv2d module you should be calling from
new code.

### ⭐ `conv3d_bf16.rs`
3D conv with `Conv3dBF16::from_weights(weight, bias, stride, padding)` +
`forward(input)` and the new `from_weights_with_config(..)` ctor that
accepts `dilation` + `groups`. NCDHW layout. Used by LTX-2 audio VAE,
LTX-2 LatentUpsampler, and the planned Wan / QwenImage 3D VAE ports.

`forward()` dispatches to `cudnn::cudnn_conv3d_bf16` first (FP32
accumulate, algo cache, workspace capped by
`FLAME_CUDNN_CONV3D_WS_LIMIT_MB`, default 256). The legacy
im2vol → cuBLASLt GEMM → bias add pipeline remains as a fallback and
is taken only when cuDNN refuses; `FLAME_CUDNN_CONV3D_STRICT=1` turns
the fallback into a hard error for parity verification. See
`FLAME_CONVENTIONS.md` for why (im2vol materialization OOMs at LTX-2
LatentUpsampler shapes).

### ⭐ `conv1d.rs`
1D conv + 1D transposed conv, both BF16-via-cuDNN. The `[B, C, L]` tensors
are reshaped to `[B, C, 1, L]` and routed through `cudnn_conv2d_bf16` with
`(H=1, W=L)`. Used by Mistral / T5 audio paths and the LTX-2.3 BigVGAN
vocoder.

- `conv1d(x, w, bias, stride, padding, dilation, groups)` — the `dilation`
  parameter is plumbed through to cuDNN as of 2026-04 (previously silently
  dropped, see `FLAME_INDEX.md` for the fix).
- `conv_transpose1d(x, w, bias, stride, padding, output_padding, groups)`
  and `conv_transpose1d_dilated` — implemented via zero-insert + regular
  cuDNN conv1d with a flipped + C_in↔C_out–transposed weight. Matches
  PyTorch `torch.nn.ConvTranspose1d` bit-exact in BF16 (verified against
  stride-5/k-11, stride-2/k-4, grouped anti-alias filters, and
  `output_padding>0`).

### `conv3d.rs` / `conv3d_simple.rs`
F32 conv3d alternatives (training).

### ⚠️ `cuda_conv2d.rs / cuda_conv2d_direct.rs / cuda_conv2d_fast.rs / cuda_conv2d_kernels.rs`
Multiple older conv2d implementations from before the unified `conv::Conv2d`.
Some still re-exported via `nn::conv2d`. Don't call directly.

### ⚠️ `ops/conv2d.rs / ops/conv2d_bf16.rs / ops/conv2d_bf16_cudnn.rs`
Alternative conv2d entry points. Feature-gated (`bf16_conv`). Mostly for
training experiments.

---

## CUDA infrastructure

### `cuda/mod.rs` + submodules
The "low-level CUDA glue" namespace. Submodules:
- `cuda::ffi` — every `extern "C"` declaration of the build-time `.cu` files
- `cuda::device_lt` — cuBLASLt handle + stream pointer accessors. **Updated 2026-05-12 (Fix #C)**: `thread_local! Cell<Option<(ordinal, stream, handle)>>` TLS front for `stream_ptr` / `cublaslt_handle_ptr`. Hot path is now a lock-free Cell read; global `Mutex<HashMap>` only touched on TLS miss (first call per thread+ordinal). Pattern follows `cudarc-pinctx`'s `cuCtxSetCurrent` TLS approach. Rollback: `FLAME_HANDLE_TLS_DISABLE=1`.
- `cuda::dtype_tag` — DType ↔ CUDA dtype tag
- `cuda::utils` — small CUDA helpers
- `cuda::kernels` — early F32 kernel wrappers (training)

### `cuda_ops_ffi.rs`
The "older" FFI declaration file for the `fc_*` family (in `cuda/cuda_ops.cu`
etc). Notable: the `flame_arena_*` and `flame_h2d_async / d2h_async / d2d_async`
families for async memcpy + arena management, plus the autotune query/reset
functions for conv2d and SDPA, and the NHWC↔NCHW layout converters.

### ⭐ `cuda_ops_bf16.rs`
The big BF16 ops surface (~70 pub fns). This is where the live kernels are
exposed: `relu_bf16`, `gelu_bf16`, `silu_bf16`, `axpby_inplace_bf16`,
`rms_norm_bf16` (delegates to `norm::rms_norm` as of 2026-05-12 — see
the `norm.rs` paragraph), `rms_norm_bf16_to_f32`, `layer_norm_bf16` (vec
forward + vec backward + cross-row dgamma/dbeta kernels as of 2026-05-12),
`layer_norm_bf16_with_stats`, `group_norm_bf16` (NHWC, vec stats as of
2026-05-12), `gemm_bf16`, `slice_axis_bf16`, `broadcast_to_bf16`,
`repeat_axis_bf16`, `index_select_bf16_into`, `conv2d_bf16` (auto-tunes
cuDNN), `sdpa_stream_bf16` (chunked SDPA), and the autotune stat accessors.

### ⚠️ `cuda_ops.rs`
The older `GpuOps` namespace (~59 pub fns). F32 ops surface used by the
autograd v3 engine for the training path. `GpuOps::add / sub / mul / div /
matmul / sum_dim_keepdim / max_dim / mean_dim / permute_generic` etc. The
one currently-live entry is `permute_generic`, which is the fallback that
`Tensor::permute` calls for non-fast-path orders. Otherwise this module is
training-only.

### ⚠️ `cuda_kernels.rs` / `cuda_kernels_gpu.rs`
The `CudaKernels` struct (~64 pub fns) and `cuda_kernels_gpu.rs` (~38 pub
fns of `GpuOps` extensions). F32 NVRTC kernels for the training path. Don't
call from inference code.

### `cuda_kernel_compiler.rs` / `cuda_kernel_sources.rs`
The older NVRTC compile path with a registry of kernel source string
constants. New BF16 kernels use the inline-string-then-`compile_ptx_with_opts`
pattern in each module instead.

### `cuda_memory_alignment.rs`
`alloc_aligned_f32(...)` for ensuring proper alignment when allocating F32
buffers (used by the staging arena and some Tensor factories).

### `cuda_tensor.rs / cuda_tensor_gpu.rs / cuda_tensor_with_cublas.rs`
⚠️ Older standalone `CudaTensor` types that predate the unified `Tensor`
design. Do not use for new code.

### `cuda_gradient_ops.rs`
F32 gradient kernels (training).

### `blas.rs`
Thin wrapper around cuBLASLt for the BF16+FP32-acc gemm path. `gemm_bf16_fp32(...)`
is the raw entry used by `ops::gemm_bf16` and the older `linear` paths.

---

## Memory / staging

### ⭐ `cuda_alloc_pool.rs`
General CUDA allocator pool used by every `Tensor::empty_dtype` /
`zeros_dtype` call path. Activated via `FLAME_ALLOC_POOL=1` (production
scripts currently use `0`).

**Updated 2026-05-12 (Fix #A):**
- **PyTorch BFC-style bucket rounding** (mirrors `c10/core/AllocatorConfig.h`):
  `< 1 MiB → 512 B`; `1 MiB–10 MiB → 2 MiB`; `>= 10 MiB → 2 MiB`. Cache
  key now `(device_ptr, bucket_elems, dtype)`. Returned slice keeps
  `len == original_request` so cudarc's `dtod_copy` length assertions hold.
- **F32 cache miss no longer zero-inits**: callers either explicitly memset
  (`alloc_zeros_from_pool`, `TensorStorage::zeros`) or fully overwrite via
  `dtod_copy`/`htod_copy_into`/a kernel that touches every element. New
  default is `unsafe device.alloc::<f32>()` (uninitialized, matches the
  pre-existing BF16 path and PT BFCAllocator). Rollback:
  `FLAME_F32_ZERO_INIT=1`.
- `hits` / `misses` / `bucket_saves` AtomicU64 counters; `log::trace!` per
  call + `eprintln!` summary in `print_pool_stats`.
- Side fix: main + pool=ON was OOMing at step 0 (pre-existing exact-match
  fragmentation bug). Pool=ON now usable end-to-end.

dispatch_overhead bench pool=ON delta: zimage block -24%, small -45%,
tiny -48%. zimage 12-step pool=ON: 1.9 → 1.8 s/step.

**Updated 2026-05-15 (R2c — range-aware BF16 trap):** the `PtrHistory`
trap that was exact-pointer-only got a parallel live-range table.
`trap_record_bf16_live(ptr, call_site, bucket)` records a live range
`[ptr, ptr + len*2)`; `trap_record_bf16_released` clears it. Used by
`pool_alloc_u16` (direct + external-miss paths), `static_slab_v2`,
and `TensorStorage::Drop`. Resolves false-positives where `Tensor::cat`
validated a view-pointer inside a live BF16 allocation but the exact-
pointer history showed only a stale `Freed` event from a prior
lifetime. The live-range table coexists with the exact-pointer
forensic history; range hit on lookup wins over a `Freed` exact-match.

### ⭐ `external_memory.rs` — unified registry for ranges + exact pointers (R1a, 2026-05-15)

Process-wide singleton (`OnceLock<ExternalMemoryRegistry>`) that
tracks GPU memory NOT owned by `cuda_alloc_pool`'s bucket free lists:

- **Ranges** (`Vec<(RangeHandle, ExternalRange)>`): half-open
  `[start, end)` keyed by `device_key = Arc::as_ptr(&device) as usize`.
  Used by `static_slab_v2` to register the whole slab once at
  materialization. Linear scan on lookup — the expected steady-state
  count is small (slab + ring + per-block-offloader).
- **Exact pointers** (`HashMap<(ptr, device_key), u32>`): refcounted.
  Used by the ring allocator and `BlockOffloader` external-ptr
  bookkeeping. Preserves the pre-R1a `register_external_ptr` /
  `unregister_external_ptr` / `is_external_ptr` semantics — the old
  public API in `cuda_alloc_pool` is now a back-compat shim that
  delegates to this registry under `DEVICE_KEY_ANY` (= 0 sentinel).
- **Hook installation**: `ensure_hook_installed()` calls
  `cudarc::driver::install_external_ptr_hook` exactly once via a
  `compare_exchange(false→true)` race. The closure consults
  `should_skip_free_any_device(ptr)` — cudarc's `CudaSlice::drop`
  signature is `fn(u64) -> bool` with no device parameter, so the
  hook-side query has to be device-less.

Updated 2026-05-15 (R2c): the hook closure now ALSO calls
`static_slab_v2::slab_v2_return_if_owned(ptr, device_key)` when the
ptr lands in a slab range. This catches raw `CudaSlice` scratch drops
(not wrapped in `TensorStorage`) so `live_count` decrements uniformly
across both code paths.

Public surface: `ExternalMemoryRegistry::global()`, `register_range` /
`unregister_range` returning `RangeHandle`, `register_exact` /
`unregister_exact` / `unregister_exact_keyed`, `should_skip_free` /
`should_skip_free_any_device`, `ensure_hook_installed`. 16 R1a-Skeptic
adversarial tests in `src/external_memory.rs::tests` lock down boundary
math, device isolation, exact-vs-range composition, and double-install
idempotency.

### ⭐ `static_slab_v2.rs` — bump-allocator slab for transient training scope (R1b–R2a, 2026-05-15)

Direct Rust port of OneTrainer's static-slab pattern
(`LayerOffloadConductor.py:122-321`). Replaces the "many small
`cuMemAllocAsync` + `cuMemFreeAsync` per step → cudart VA reuse →
use-after-free" failure mode that the original `cuda_alloc_pool`
bucket allocator hit on Klein 9B (`HANDOFF_2026-05-15_OT_STATIC_SLAB_REDESIGN.md`).

**Core type — `StaticSlabAllocator`** (R1b, commit `416be6d`):
- One backing `CudaSlice<u8>` per device, sized by
  `FLAME_STATIC_SLAB_BYTES_BF16` (default 4 GiB) and
  `FLAME_STATIC_SLAB_BYTES_F32` (default 4 GiB).
- **Lazy materialization** — `cudaMalloc` does not fire until the
  first `alloc_*` call. `release()` drops the backing slab; subsequent
  `alloc_*` re-materializes.
- **Bump-cursor allocation** with 16-byte alignment. `alloc_u16(n)`,
  `alloc_f32_zeroed(n)` (memsets via cudarc), and
  `alloc_f32_uninit(n)` (no zero-init, document as such). Returns
  synthesized `CudaSlice<T>` via the existing cudarc layout-mirror
  pattern — the same trick `offload::alloc_bf16_via_ring` uses.
- **`AtomicUsize` live_count** — incremented at every `alloc_*`,
  decremented at slice drop via either `TensorStorage::Drop`'s
  `slab_v2_try_claim` OR the external-memory drop hook (R2c, see
  `external_memory.rs`). `reset()` is **strict**: returns `Err`
  with the current count if `live_count != 0`.
- **Per-step exhaustion short-circuit** (R2c-perf, commit `8b494f9`):
  once an `alloc_*` overflows the slab capacity within a guarded
  step, a thread-local "slab exhausted" flag is set, and remaining
  `pool_alloc_*` calls in the same step skip the slab lookup
  entirely and go straight to legacy. Resets at guard enter / drop.
  Eliminates the ~9k warning-format-and-log per step that capped
  perf at 4.3 s/step pre-fix.
- **Per-device global accessor**: `slab_for_device(&Arc<CudaDevice>)`
  returns `&'static Mutex<StaticSlabAllocator>` from a leaked
  `OnceLock<Mutex<HashMap>>`. Two threads racing on `enter` for the
  same device key get the same `Mutex` (loser's `Box::leak` wastes
  a small amount of memory; cold path).

**Guard type — `StepSlabGuard`** (R2a, commit `79289d4`):
- RAII guard that turns ON slab dispatch for the current thread.
  `enter(device) -> Result<Self>`; rejects nested guards with `Err`.
  `finish() -> Result<()>` for graceful close; `Drop` for scope-end.
- Sets a thread-local `Cell<Option<usize>>` to the device key at
  enter; clears at drop/finish. `pool_alloc_u16` / `pool_alloc_f32`
  in `cuda_alloc_pool` check `StepSlabGuard::active_on_thread()` AND
  `FLAME_USE_STATIC_SLAB=1` at the very top of the function before
  any bucket lookup; if both are true, dispatch to
  `pool_alloc_*_via_slab(n)`. Misses fall through to legacy.
- **Strict reset on Drop**: if `live_count != 0`, panics with the
  count and cursor. The panic is suppressed if the thread is already
  unwinding (`std::thread::panicking()`) to avoid the double-panic-
  during-unwind abort.
- **Footgun**: do NOT `mem::forget`/`mem::transmute`/`Send` the
  guard across threads. The thread-local is per-thread; moving the
  guard strands the source thread's active flag. Locked down by
  `r2a_skeptic::sk_send_across_threads_corrupts_enter_thread_local`.

**TensorStorage::Drop integration** (R1b, commit `afb066e`; R1b-bf
`0d046e3`; R2c hook expansion `8e6cc7b`):
- F32 + BF16(u16) Drop arms call `slab_v2_try_claim(&slice, _)`
  BEFORE the legacy `pool_return_*` path. On success: `mem::forget`
  the slice (cudarc Drop hook intercepts cudaFree because the slab
  range is registered) and decrement `live_count`. On miss: fall
  through to `pool_return_*` for legacy pool ownership.
- The pool_disabled short-circuit was moved INSIDE each arm AFTER
  the slab check so `FLAME_ALLOC_POOL=0 + FLAME_USE_STATIC_SLAB=1`
  (R3-target config) works without the live_count leak (R1b-bf).

**Trainer wiring** — `EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs`:
- The guard MUST be the first allocation-creating local in the step
  body. Subsequent step-local tensors drop before the guard via
  Rust's reverse drop order. Validation/sample lives OUTSIDE the
  guard scope (they have lifetime patterns that don't fit per-step
  reset). Two `r2b_wiring_lint` tests in `train_klein.rs` enforce
  the pattern at build time.
- The optimizer's persistent state (Adam m/v) MUST be pre-warmed
  BEFORE the slab env-flag activates. The trainer calls
  `Optimizer::ensure_state_initialized(&params)` post-resume and
  pre-step-loop. Without prewarm, m/v allocate lazily on step 0
  inside the guard scope → live_count violation at step-end drop.
  See `Adam::ensure_state_initialized` / `AdamW::ensure_state_initialized`.

**Test coverage**: 14 R1b lib unit tests (`slab_*`), 23 R1b-Skeptic
adversarial integration tests, 11 R2a-Skeptic adversarial tests,
8 R2b tests (3 bf + 5 sk for the prewarm hook contract), 1 microbench,
1 drop-wiring regression. All tests serialize on a `TEST_LOCK` because
the per-device slab map is process-global; parallel test execution
without the lock cascades into futex deadlocks.

### `memory_pool.rs`
F32 device memory pool. ~15 pub fns. Mostly training, but the pool is also
used by some BF16 fast-path arena code.

### `pinned.rs`
Pinned host memory: `PinnedHostBuffer / PinnedHostBufferView /
PinnedHostBufferViewMut`, `PinnedAllocFlags`, `StagingDeviceBuf` (a paired
host pinned + device staging buffer for async H2D), `register_slice_as_pinned`,
`unregister_pinned`, and the `memcpy_async_device_to_host /
memcpy_async_host_to_device` helpers. Used by BlockOffloader and the safetensors
loader for fast H2D.

### `pinned_pool.rs`
`PinnedPool` — a pool of reusable pinned host buffers for the staging path.
Re-exported at `flame_core::PinnedPool`.

### `staging.rs` (BF16-only, gated)
~16 pub fns. BF16 arena + async copy primitives. Used internally by
`Tensor` for some hot paths via `bf16_copy_async` and `ArenaLease`.

### `ring_alloc/mod.rs` — bidirectional ring allocator (Gap 1 Phase 1, 2026-05-14)

Faithful Rust port of OneTrainer's `StaticLayerAllocator` +
`StaticLayerTensorAllocator` (`/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py:37-222`).
Bidirectional ring over a list of fixed-size GPU slabs. Forward
allocations advance `allocation_end`; backward allocations retreat
`allocation_start`. Slabs are `cudaMalloc`-ed lazily on first touch
(matches OT's `ensure_allocation`). Bytes don't return per-allocation —
they return at `reset()` between training steps.

The shape is structurally different from `cuda_alloc_pool`'s bucketed
free list: the two cursors make overlapping forward/backward allocations
impossible within a step, side-stepping the corruption mode that
forced `FLAME_ALLOC_POOL=0` on Klein 9B (see
`HANDOFF_2026-05-14_TRAINER_REGRESSION_FAILURE.md` and
`docs/OFFLOAD_GAPS_vs_ONETRAINER.md` Gap 1).

Phase 1 (2026-05-14) ships the primitive + 19-test microbench
(`tests/ring_alloc_microbench.rs`). No consumer wiring; no trainer
gate. Phase 2a (2026-05-14) ships `pool_adapter.rs` — opt-in
`PoolMissAllocator` trait + `RingPoolAdapter` — which routes
`cuda_alloc_pool` cache-miss allocations through a shared ring. Klein
trainer-side opt-in (env `KLEIN_POOL_RING=1`) tests the hypothesis that
ring-backed misses fix the Klein 9B step-2 `INVALID_VALUE` crash
without the `FLAME_ALLOC_POOL=0` workaround. Design and lifecycle in
`docs/RING_ALLOC_DESIGN.md`.

Direction is enforced at the type level via `RingForwardHandle` /
`RingBackwardHandle` (no bool flag). Handles borrow the allocator
mutably; the type system prevents simultaneous forward + backward
allocation on the same ring.

### `ring_alloc/pool_adapter.rs` — pool miss-route backend (Gap 1 Phase 2a)

`RingPoolAdapter` wraps a `RingAllocator` in a `Mutex` (`Send + Sync`)
and implements `cuda_alloc_pool::PoolMissAllocator`. When installed via
`cuda_alloc_pool::install_miss_allocator`, all cache-MISS allocations
of `pool_alloc_u16` / `pool_alloc_f32` flow through the ring's forward
direction; cache HITS still serve from the bucket free lists. The
adapter synthesizes a `CudaSlice<T>` view onto a ring slab via the
`CudaSliceMirror` transmute pattern; the pool tags free-list entries
as `is_external: true` so `clear_cache` / pool drop skip `cudaFree`
on those entries.

Lifecycle: every training step boundary should call
`cuda_alloc_pool::clear_pool_cache()` then `adapter.reset()` — in that
order — to drain bucket entries pointing into ring slabs before
cursors return to extremes.

---

## Autograd — multiple generations, **read carefully**

### `autograd_v3.rs` (the active engine)
The currently-active autograd engine per the comment in `lib.rs:153`. Keeps a
tape of `(out_id, Op, saved_inputs)` triples and walks it backward when
`Tensor::backward()` is called. `Op` is a wide enum covering all the ops
flame-core supports (Add, Sub, Mul, Div, Matmul, Bmm, Reshape, View, Permute,
Narrow, Cat, Softmax, Sdpa, LayerNorm, GroupNorm, Conv2d, Conv3d, ...).

**Autograd bug fixes (2026-04-13):**
- SDPA: always route to `forward_train` when autograd is recording (was gated on env var)
- RoPE: record `Op::RoPePrecomputed` when input requires_grad (was inference-only)
- RMSNorm: use autograd-aware `to_dtype` (not `to_dtype_no_grad`) when input requires_grad
- LayerNorm backward: pass saved weight/bias to backward kernel (was `None, None`)

**Autograd update (2026-05-20):**
- `Op::RoPePrecomputed` now carries an explicit `autograd::RopeLayout`
  (`Interleaved` / `Halfsplit`) tag instead of shape-sniffing the saved
  cos tensor at backward time. Fixes HiDream-O1 MRoPE which emits
  rank-3 cos `[1,S,half]` but uses `rope_halfsplit_bf16` — the
  shape-sniffer mis-classified it as Interleaved and applied the wrong
  rotation in backward. See `FLAME_INDEX.md` and `FLAME_CONVENTIONS.md`.
- `AutogradContext::retain_intermediate_grads_add(ids)` — additive
  retain-set variant for probes registered *during* checkpoint
  recompute (the outer-tape snapshot already fired). Sub-tape backward
  in `Op::Checkpoint` and `Op::CheckpointOffloadBoundary` re-reads
  `RETAINED_INTERMEDIATE_GRAD_IDS` so additions inside recompute are
  honored. See `FLAME_DIAGNOSTICS.md` §0 + §1.

**Training performance caveat (2026-04-09):** The tape-based backward is
synchronous and has ~1s overhead per entry on 3090 Ti (HashMap lookup + GPU
kernel launch + implicit sync + gradient accumulate). Klein 4B generates
~2700 tape entries → ~45 min/step. This makes raw tape-based training
impractical for DiT-scale models. **Gradient checkpointing per transformer
block is required** to reduce the tape to ~100 entries. See
`gradient_checkpointing.rs` and the klein-trainer `PLAN_AUTOGRAD.md` for
the concrete approach.

**Key gotchas found during klein-trainer development:**
- `Tensor::narrow` BF16 fast path (`cuda_ops_bf16::slice_axis_bf16`) was
  silently dropping `requires_grad` and not recording `Op::Slice`. **Fixed
  2026-04-09** (commit `12d1433` in flame-core). Without this fix, LoRA
  training silently produces zero gradients because every QKV split breaks
  the autograd chain.
- `BF16 TensorStorage` uses `CudaSlice<u16>` (NOT `Arc`-wrapped, unlike
  F32 which uses `Arc<CudaSlice<f32>>`). This means `Tensor::clone()` for
  BF16 is a full GPU memcpy, not a cheap ref bump. Every `record_op` that
  saves BF16 tensors via `clone()` or `clone_result()` allocates new GPU
  memory. With ~300 saves per forward pass this adds ~8 GB overhead.
  **Fix:** Arc-wrap BF16 storage to match F32 (the `shared_storage` feature
  flag exists but only covers `from_bf16_arena`).
- `Op::FlashAttention` has a backward handler (`attention_backward_recompute`)
  that only saves Q/K/V and recomputes scores during backward. Using this
  instead of decomposed `bmm + softmax + bmm` eliminates ~275 tape entries
  for Klein 4B. klein-trainer's `sdpa_train` records `Op::FlashAttention`
  directly for this reason.

### `autograd.rs` (top-level re-export)
Re-exports `AutogradContext`, `Op`, `NoGradGuard` from `autograd_v3` so
callers can say `flame_core::autograd::AutogradContext`. Key public API:
- `AutogradContext::record_op(output_id, op, saved_tensors)` — add to tape
- `AutogradContext::no_grad()` → `NoGradGuard` RAII guard (disables taping)
- `AutogradContext::backward(loss)` → `GradientMap`
- `AutogradContext::set_enabled(bool)` — manual enable/disable

### `autograd_v4` (feature gated)
A newer experimental engine with explicit `Gradients` and `graph` types.
Off by default. The SDPA backward in `autograd_v4/ops/sdpa.rs` is more
correct than the v3 one in some edge cases.

### `autograd_v2/` (feature `autograd_v2`, Phases 1 + 2 + 3a + 3b + 3c1 + 3c2 + 4a + 4b + 5a + 5b)
Clean-sheet PyTorch-style DAG autograd engine. Designed to replace v1/v3/v4
once Phase 5 parity gates pass. Phase 1 ships the foundational types
and traits; Phase 2 ships the engine driver, dependency counting, ready
queue, hook dispatch, real `AccumulateGrad::apply`, `gradient_edge()`,
and the `CheckpointGradFn` skeleton. Phase 3a wires the recording surface
(`Tensor::autograd_meta`, `record_v2`) + 5 math P0 ops. Phase 3b adds
6 view ops + HAZARD-2026-05-13-1 characterization. Phase 3c1 ships
`layer_norm` + the full `CheckpointGradFn::apply` (no longer
`NotImplementedYet`). Phase 3c2 closes Phase 3 by populating the
`Tensor`-level forward-mode AD slot (`AutogradMetaV2::fw_grad`) inside
every Phase 3a/3b op's forward wrapper: 11 ops × JVP formulas
(`add`/`mul`/`sum`/`matmul`/`silu`/`reshape`/`view`/`transpose`/
`narrow`/`squeeze`/`unsqueeze`/`permute`); `layer_norm` forward-mode
AD was deferred to the Phase 5a parity gate and **shipped there**
(per-row reductions in F32; bit-equal to `torch.autograd.functional.jvp`
in F64). `Tensor::fw_grad` /
`Tensor::set_fw_grad` accessors land on the public surface (`src/tensor.rs`);
shared JVP helpers (`any_fw_grad`, `tangent_or_zero`) live at
`src/autograd_v2/ops/fw_mode.rs`. `set_fw_grad` implicitly allocates
a fresh `leaf_no_grad()` meta when called on an untracked tensor
(PyTorch parity; backward-mode tracking is NOT silently enabled).
Phase 4a ships the optimizer surface (`OptimizerV2` trait +
`AdamWV2` wrapper at `optim.rs`), the `GradDtypePolicy::MatchParamDtype`
path in `Parameter`, the BF16-grad classifier arms in `src/adam.rs`,
and `multi_tensor_l2_norm_sq_bf16` in `src/ops/multi_tensor.rs`.
Phase 4b (this milestone) ships the `GradientMap` half of the v2
grad-storage path: a new `GradStorePolicy::MatchInsertedDtype` variant
+ `GradientMap::new_v2` / `with_index_v2` / `policy` / `set_ones_dtype`
/ `get_or_create_dtype` helpers. v1 / v3 trainers keep the default
`InternalFP32_PublicBF16` policy unchanged; v2 callers opt in
explicitly. Real-trainer integration (flipping Z-Image's
`loss.backward()` to a v2 grad-storage path) is **not** in Phase 4b —
the Z-Image forward graph still uses v3 ops and the v2 forward surface
is 13 ops; adding the missing ~20+ ops is Phase 5 work. See
`docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` for the phase roadmap and
`docs/BF16_GRAD_DECISION.md` for the dtype policy (Deliverable C
section explains the trainer-integration gap).

Phase 5b (this milestone) ships Deliverable C Route (ii) — the
`loss.backward_v2()` bridge. `AutogradContext::backward` is now a
1-line wrapper around a private `backward_impl(loss, policy)`, and a
new sibling `AutogradContext::backward_v2(loss)` calls
`backward_impl(loss, MatchInsertedDtype)`. The bridge keeps the v3
op-dispatch loop intact and adds a per-grad `to_dtype(loss.dtype())`
cast at accumulation time so the v2 `accumulate` contract (no dtype
mismatch) is satisfied. This realizes BF16-grad-end-to-end for any
forward graph — the v2 op surface is **not** required. EriDiffusion-v2's
`train_zimage` now ships an opt-in `--use-autograd-v2` flag (default
OFF; new `autograd_v2` feature on `eridiffusion-cli`). Tests:
`tests/autograd_v2_bridge.rs` (3 tests — BF16 emit, F32 parity,
BF16 tolerance per `BF16_GRAD_DECISION.md`).

**Phase 5b follow-up — Klein backward parity v2 vs PyTorch.**
`tests/autograd_v2_klein_parity.rs` adapts the six existing v3 Klein
component backward parity tests (`tests/pytorch_parity.rs:1008-1424`)
to exercise `AutogradContext::backward_v2()` against the same PyTorch
fixtures, plus a 7th scenario reconstructing the small-shape Klein
double-block from the previously-unused `klein_block_backward.safetensors`
fixture (qkv linear → narrow chunks → reshape/permute → SDPA → permute
back → out linear → gated residual → MLP (up linear → silu*chunk → down
linear) → gated residual). Single `#[test] fn klein_parity_v2_scenarios`
driver runs all 7 sequentially to dodge the `AUTOGRAD_CONTEXT` global
parallel-mode race (Phase 5b skeptic CONCERN #3 carryforward). Tolerance
`min_cos=0.99 / max_abs_ratio=5e-2` matches the v3 Klein gate; max
observed `abs_ratio` 0.0093 on `dw_qkv` in attn_chain.

**Phase 5c — perf bench (Deliverable D).** `tests/autograd_v2_perf.rs`
benches `backward_v2` overhead vs v3 `backward` on three workloads
(synthetic 4-layer MLP / Klein attn_chain prod / Klein double-block
backward) × three configs (v3 control / bridge alone / Class A with
`Parameter::new_v2` + `set_grad` round-trip). 5 warmup + 50 timed iters
per cell with the slowest 5 trimmed; wall-clock via `std::time::Instant`
+ `device.synchronize()` pre/post; `FLAME_CUDA_GRAPH=0` and
`FLAME_MT_SCALE` unset to apples-to-apples both arms (per Phase 5b
skeptic CONCERNs). Three `#[serial] #[test]` cells (one combined-sweep
test was tried first and failed — cross-cell CUDA state pollution that
only resets cleanly across `#[test]` binary boundaries). Headline:
bridge Δ +1.46% / +2.05% on Klein double-block / attn_chain prod;
Class A delivers 50.00% exact grad-storage savings (78 MB per backward
on attn_chain prod). Full table in `docs/BF16_GRAD_DECISION.md`
§Phase 5c.

Key Phase 1 design points:
- **Weak accumulator cycle break (§1):** `AutogradMetaV2::grad_accumulator`
  and `AccumulateGrad::variable` are both `Weak`, so a leaf with
  `requires_grad=true` does not leak its storage.
- **Result-typed `apply` (§3):** `GradFn::apply(grad_outputs, ctx) ->
  Result<Vec<Option<Tensor>>, AutogradV2Error>`. Version mismatches and
  released saved tensors are recoverable training errors, not panics.
- **`SavedTensor` carries the version handle (§4):** `Arc<AtomicU32>`
  clone of `TensorStorage::version_handle()`. Survives
  `AutogradContext::clear()` / version-table flushes.
- **`&self` release (§3):** `GradFn::release_variables(&self)` works
  because `SavedTensor` holds data behind `Mutex<Option<Tensor>>`.
- **`InputBuffer` in-place AND out-of-place (§Phase 1 clause 7):** in-place
  is the default when `create_graph=false` AND dtype/shape match;
  out-of-place when `create_graph=true` or the in-place predicate fails.
  Phase 1's predicate is conservative — `BF16 | F32` same-shape only.
- **Hooks surface from day one (§8 / clause 13):** `Hooks { pre/post/tensor }`
  bundles + `GradFn::hooks()` accessor. `Hooks::empty_ref()` is the
  OnceLock-backed singleton for the no-hook fast path. Phase 2 wires
  dispatch.
- **Forward-mode AD plumbing (§8 / clause 15):** `SavedTensor::fw_grad_`
  slot is present. Phase 3 per-op forward formulas populate it.
- **Multi-device surface (§16 / charter 2026-05-13):** `DispatchCtx`
  parameter at every entry point. `(device, raw_stream_ptr)`. Phase 1's
  only ctx today is `(global_cuda_device, default-stream)` but the trait
  shape locks the parameter so non-default streams / multi-device land
  without an ABI break.

Phase 2 additions:
- **`engine.rs`** — `GraphRoot` builder + `Engine::execute(root, ctx)`.
  Builds a per-node dependency count from the output `grad_fn`s by
  walking `next_edges()`, seeds output grads into per-node
  `InputBuffer`s at slot `output_nr`, drives a `BinaryHeap` ready queue
  ordered by `(topological_nr desc, sequence_nr desc, node_id desc)`,
  dispatches `apply()`, routes per-`next_edge` output grads, calls
  `release_variables(&self)` on each node when `retain_graph=false`.
  Returns `Result<Vec<Option<Tensor>>, AutogradV2Error>` — `Vec` is
  populated when `GraphRoot::with_inputs` is set (per-input grads in
  order via `AccumulateGrad` downcast), empty otherwise.
- **`checkpoint.rs`** — `CheckpointGradFn` full implementation
  (Phase 3c1, commit `2be9770`). `apply()` re-runs the saved forward
  closure under v2 recording with detached input clones + fresh
  `requires_grad=true` meta, drives a nested `Engine::execute(...)`
  via the standard leaf-AccumulateGrad backward path, then harvests
  per-leaf grads off each reattached leaf's `meta.grad`.
  `checkpoint_v2(forward_fn, inputs, ctx)` is the user-facing entry —
  detaches inputs before forward so inner ops skip recording (the
  memory saving). Reentrant-nested-execute safe per the Phase 2
  engine design.
- **`accumulator.rs::AccumulateGrad::apply`** — real impl. Drops
  `None` grad silently, no-ops if the leaf meta is dropped, in-place
  accumulates same-dtype/same-shape into `meta.grad`, returns
  `Err(DtypeMismatch)` on dtype divergence, shape mismatch surfaces
  through the in-place op as `Err(FlameCore(_))`.
- **`meta.rs::gradient_edge(meta, seq) -> Edge`** — non-leaf returns
  `(grad_fn, output_nr)`; leaf with `requires_grad=true` materializes-
  or-caches an `AccumulateGrad` via the Weak slot; leaf without
  `requires_grad` returns `Edge::null()`.
- **`node.rs::GradFn::as_any`** — type-erased downcast helper.
  Default returns a static no-op marker; `AccumulateGrad` overrides
  to return `&self` so the engine's `with_inputs` path can resolve
  leaf accumulators.
- ~~`engine.rs::_v2_set_grad_fn / _v2_clear_tensor_meta`~~ —
  **REMOVED in Phase 3a.** The test-only side-table is gone;
  `Tensor::autograd_meta` is now a real field and `record_v2` is the
  canonical install entry. Phase 2 engine tests refactored to use
  `record_v2`-style linking helpers.

Phase 3a additions:
- **`recording.rs`** — the public op-recording surface every Phase 3+
  op uses. `record_v2(grad_fn, outputs, ctx)`, `gradient_edge_for_tensor(t)`,
  `next_sequence_nr()`, `needs_grad(inputs)`. There is no global tape
  — the recording is the per-output `Tensor::autograd_meta` slot,
  populated by `record_v2`.
- **`ops/`** — submodule with one file per op: `ops/add.rs`,
  `ops/mul.rs`, `ops/sum.rs`, `ops/matmul.rs`, `ops/silu.rs`. Each
  exports a `*GradFn` struct + `*_v2` forward wrapper. Forward
  wrappers call the existing flame-core math op for the forward
  (`Tensor::add`, `Tensor::mul`, `Tensor::sum`, `Tensor::matmul`,
  `Tensor::silu`) and conditionally record onto v2 when
  `needs_grad(inputs)` is true.
- **`DispatchCtx::create_graph` field** — threaded by
  `Engine::execute` from `GraphRoot::create_graph`. The
  `AccumulateGrad::apply` and `InputBuffer::add_outofplace`
  branches read it to pick recording vs in-place accumulation.
- **`AutogradV2Error::GradOutputShapeMismatch`** — new error variant.
  `Engine::execute` validates `grad_outputs[i].shape() ==
  outputs[i].shape()` at entry.
- **Non-leaf grad capture in `Engine::execute`** — per-call
  `(NodeId, output_nr) -> Tensor` map populated at each node's
  apply-time, read at the end for any `with_inputs` entry whose
  grad_fn isn't an `AccumulateGrad`.

Phase 3b additions (view-autograd surface + AccumulateGrad hooks fix):
- **`ops/reshape.rs`** — `ReshapeGradFn` + `reshape_v2` + `view_v2`.
  Backward reshapes the upstream grad back to the saved input shape;
  no tensor data saved (shape-only). `view_v2` aliases `reshape_v2`
  because flame-core's `Tensor::view` is `reshape` internally.
- **`ops/transpose.rs`** — `TransposeGradFn` + `transpose_v2`. 2D-only
  (matches `Tensor::transpose`'s shape, see `tensor.rs:2958`).
  Backward: `grad_output.transpose().contiguous()`. The `.contiguous()`
  materialises the strided view to dodge HAZARD-2026-05-13-1 + the
  project-wide gemm-stride-ignore concern (Phase 3a matmul fix
  commit `6ee385f`).
- **`ops/narrow.rs`** — `NarrowGradFn` + `narrow_v2`. Backward
  allocates a FRESH zero buffer of input shape and scatter-adds the
  grad_output's slab via `narrow_backward_scatter_add_cuda`. CRITICAL:
  never writes through a `narrow()` view back into a parent — that's
  HAZARD-2026-05-13-1.
- **`ops/squeeze.rs`** — `SqueezeGradFn` + `squeeze_v2`. Saves the
  squeezed dim index; backward unsqueezes at that dim.
- **`ops/unsqueeze.rs`** — `UnsqueezeGradFn` + `unsqueeze_v2`. Mirror
  of squeeze: saves dim, backward squeezes.
- **`ops/permute.rs`** — `PermuteGradFn` + `permute_v2`. Computes
  inverse permutation once at construction. Backward:
  `grad_output.permute(inverse_perm).contiguous()`. Same `.contiguous()`
  discipline as `TransposeGradFn`.
- **`AccumulateGrad::hooks` field** — changed from `Hooks` (default-
  init) to `OnceLock<Hooks>`. Default state is uninitialised so
  `hooks()` returns `Hooks::empty_ref()`, restoring the engine's
  pointer-equality fast path. Phase 2 carryover bug.
- **`AccumulateGrad::install_hooks(hooks)`** — single-shot Hooks
  installer (Phase 4 may add multi-hook merge once Hooks fields carry
  interior mutability).

Phase 4a additions (optimizer + BF16-grad migration partial):
- **`optim.rs`** — `OptimizerV2` trait + `AdamWV2` thin wrapper around
  the existing `AdamW`. Same state shape, same kernels, same
  checkpoint format. `set_param_grad_v2` is a typed convenience that
  errors if the parameter is on the v1 `CastToF32` policy.
- **`crate::Parameter` extensions** (in `src/parameter.rs`) —
  `GradDtypePolicy` enum + `Parameter::new_v2`, `grad_bf16_or_f32`,
  policy-aware `set_grad` / `apply_update`. Phase 4a only touches the
  NEW `MatchParamDtype` path; the v3 `CastToF32` path is unchanged so
  existing trainers keep their F32-grad invariant.
- **`src/adam.rs` classifier extension** — 4-way `(param_dtype,
  grad_dtype)` classifier with the previously-dead
  `adam_fused_multi_bf16_bf16grad_kernel` and
  `adam_fused_f32param_bf16grad_kernel` arms activated.
- **`src/ops/multi_tensor.rs::multi_tensor_l2_norm_sq_bf16`** — BF16
  sibling for `multi_tensor_l2_norm_sq_f32`. Reused by
  `ops::grad_norm::global_l2_norm` for all-BF16 grad slices.

What Phase 4a explicitly does NOT do:
- **No `GradientMap` rewrite.** v2 grads land in `AutogradMetaV2::grad`
  (per `accumulator.rs`), bypassing GradientMap on the recording path.
  Phase 4b reconsiders this once trainer integration shows what shape
  trainers actually need.
- **No trainer integration smoke.** Klein crashes on this box per
  Phase 0 skeptic notes; Z-Image LoRA is the next-session target.

Phase 4b additions (`GradientMap` half of the v2 grad-storage path):
- **`GradStorePolicy::MatchInsertedDtype`** — new enum variant in
  `src/autograd/policy.rs`. v1 default is unchanged.
- **`GradientMap::new_v2` / `with_index_v2`** — `src/gradient.rs:99,
  112` — construct on the v2 policy.
- **`GradientMap::policy()` / `set_ones_dtype` / `get_or_create_dtype`**
  — explicit-dtype helpers for v2 callers.
- **`get_public_grad` / `take_public_grads` / `insert` / `accumulate`**
  — all branch on policy. v1 path is byte-equivalent to pre-Phase-4b
  behavior. v2 path preserves grad dtype end-to-end; `accumulate`
  errors on dtype mismatch (mirrors `AccumulateGrad::apply` in
  `accumulator.rs`).

What Phase 4b explicitly does NOT do:
- **No trainer-side integration of v2 GradientMap.** Z-Image's
  `loss.backward()` goes through `AutogradContext::backward` which
  constructs a default-policy GradientMap. Flipping the trainer
  requires either porting the model's forward graph to v2 ops (~30+
  ops missing — Phase 5 work) or adding a `loss.backward_v2()` entry
  that builds `with_index_v2` from the existing tape (~200-line
  duplication of v3 backward — out of scope per the additive
  constraint). See `BF16_GRAD_DECISION.md` "Deliverable C" section.

Tests added in Phase 4b: `tests/autograd_v2_gradientmap_v2.rs`
(17 tests = 4 constructor-policy contracts + 3 insert v1-vs-v2 +
3 accumulate v1-vs-v2 + 2 get_public_grad v1-vs-v2 + 3 set_ones_dtype
+ 2 get_or_create_dtype).

Files: `accumulator.rs`, `checkpoint.rs`, `dispatch.rs`, `engine.rs`,
`error.rs`, `hooks.rs`, `input_buffer.rs`, `meta.rs`, `mod.rs`,
`node.rs`, `optim.rs` (Phase 4a), `recording.rs`, `saved_tensor.rs`,
`ops/mod.rs`, `ops/fw_mode.rs` (Phase 3c2), `ops/add.rs`,
`ops/mul.rs`, `ops/sum.rs`, `ops/matmul.rs`, `ops/silu.rs`,
`ops/reshape.rs`, `ops/transpose.rs`, `ops/narrow.rs`,
`ops/squeeze.rs`, `ops/unsqueeze.rs`, `ops/permute.rs`,
`ops/layer_norm.rs` (Phase 3c1). Tests:
`tests/autograd_v2_types.rs` (13 tests, Phase 1) +
`tests/autograd_v2_engine.rs` (12 tests, Phase 2 + Phase 3b
`accumulate_grad_uses_empty_sentinel_when_no_hooks`) +
`tests/autograd_v2_ops.rs` (33 tests = Phase 3a 18 + Phase 3b 10 +
Phase 3c1 5 layer_norm) + `tests/autograd_v2_checkpoint.rs`
(4 tests, Phase 3c1) + `tests/autograd_v2_phase4a.rs` (12 tests,
Phase 4a: 5 Parameter v2, 3 Adam BF16-grad, 2 multi_tensor BF16,
2 AdamWV2 trait surface) +
`tests/autograd_v2_gradientmap_v2.rs` (17 tests, Phase 4b) +
`tests/autograd_v2_fw_mode.rs` (13 tests, Phase 3c2: one JVP per
non-LN op + `set_fw_grad_implicitly_allocates_meta` regression) +
`tests/autograd_v2_parity.rs` (26 tests, Phase 5a: 13 backward vs
PyTorch + 13 forward-mode vs PyTorch). All green (modulo a known
parallel-CUDA-context flake in autograd_v2_ops / autograd_v2_engine /
inplace_version_bump_audit pre-existing — re-run individually to confirm).

Phase 5a additions (Deliverables A + B from
`docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Phase 5; **Phase 5b
model parity** and **Phase 5c perf bench** remain open):
- **`layer_norm_jvp`** — `src/autograd_v2/ops/layer_norm.rs:329`. The
  LN JVP formula deferred from Phase 3c2 ships here. F32-internal
  compute through per-row `mean`/`var`/`rstd`/`rstd_fw`; cast back to
  the primal's BF16 dtype before storage. Bit-equal to
  `torch.autograd.functional.jvp(layer_norm, ...)` in F64 (verified
  in `tests/fixtures/gen_v2_parity.py`).
- **`tests/fixtures/gen_v2_parity.py`** — PyTorch fixture generator
  at fixed seed 42. Emits 26 `.safetensors` files
  (`<op>_{backward,jvp}.safetensors`) used by the parity tests. ≤512
  elements per op so each file stays a few KB; total fixture footprint
  ~104 KB. Regenerate with `python3 tests/fixtures/gen_v2_parity.py`
  from `flame-core/` root.
- **`tests/autograd_v2_parity.rs`** — 26 PyTorch-parity tests driven
  by `flame_core::parity::ParityHarness`. Tolerance bands: tight F32
  (`min_cos=0.99999`, `max_abs_ratio=1e-4`) for view + add/mul/sum/silu
  outside matmul; loose F32 (`min_cos=0.9999`, `max_abs_ratio=5e-3`)
  for matmul + scalar-reductions through cuBLAS; BF16-LN
  (`min_cos=0.999`, `max_abs_ratio=5e-2`) for layer_norm.

### ⚠️ `autograd_simple.rs / autograd_engine.rs / autograd_ops.rs / autograd_ops_complete.rs / autograd_debug.rs`
Older autograd attempts. Dead code; kept for reference.

### `gradient.rs`
`GradientMap` (re-exported as `GradientMap` and `GradStore`), `TensorGradExt`
trait. The collection that holds `tensor_id → grad_tensor` mappings during
backward.

As of Phase 4b, the map carries a `GradStorePolicy` (see
`src/autograd/policy.rs`):
- **`InternalFP32_PublicBF16`** (default; `GradientMap::new` /
  `with_index`) — v1 / v3 behavior. Loss seed is F32, every insert
  upcasts BF16 → F32, public reads convert F32 → BF16.
- **`MatchInsertedDtype`** (autograd v2; `GradientMap::new_v2` /
  `with_index_v2`) — preserves the inserted grad's native dtype.
  Helpers: `set_ones_dtype(id, shape, dtype)`,
  `get_or_create_dtype(id, shape, dtype)`. `accumulate` errors on
  dtype mismatch (the producer is expected to feed a consistent
  dtype).

The v3 default path is unchanged; no live trainer flips to v2 in
Phase 4b. See `docs/BF16_GRAD_DECISION.md`.

### `gradient_clip.rs`
Gradient clipping helpers (per-norm and per-value).

### `gradient_checkpointing.rs`
Activation checkpointing scaffolding. Contains `CheckpointManager` (global
singleton behind `CHECKPOINT_MANAGER` mutex), `CheckpointPolicy` enum
(`CPUOffload` / `Recompute` / `Adaptive`), `CheckpointedBlock<F>` wrapper,
and `CheckpointableModel` trait.

**Current status (2026-04-09):** scaffolding only, not production-wired.
- `CPUOffload` policy is a placeholder that clones to device (no actual
  CPU transfer — line 88-92: "Placeholder: clone tensor on device").
- `Recompute` requires explicit `set_recompute_for(id, closure)` calls
  that nobody invokes.
- `CheckpointableModel` trait has zero implementors.
- `record_op` at `autograd.rs:456` already calls
  `mgr.checkpoint_saved_tensor()` for every saved tensor — so the hook
  point exists; the policies just don't do anything useful yet.

**To make training work at DiT scale**, the plan is to wrap each
transformer block's forward in a `CheckpointedBlock` that:
1. Saves only block inputs (2 tensors: img + txt residual)
2. Runs block forward under `NoGradGuard` (no taping)
3. Records one `Op::Checkpoint` entry
4. At backward: re-runs the block forward WITH autograd, builds a
   per-block tape, backward through it, frees it, returns grad
This reduces the global tape from ~2700 to ~50 entries for Klein 4B.

---

## Optimizers + nn

### `adam.rs`
Canonical Adam: `src/adam.rs` (fused BF16 and F32 CUDA kernels). Canonical SGD: `src/sgd/mod.rs`.
All param dtypes except BF16 and F32 return an error — no silent fallback.
BF16 master weights use F32 moments; F32 params use F32 moments. Re-exported as
`nn::AdamW`. Includes `set_lr()` for step-wise schedulers.

Six NVRTC kernels are compiled once on first call and loaded into the
`adam_fused` module:

  - **Single-tensor**: `adam_fused_bf16_kernel`, `adam_fused_f32grad_kernel`
    (BF16 param + F32 grad), `adam_fused_f32param_f32grad_kernel`,
    `adam_fused_f32param_bf16grad_kernel`. One launch per parameter.
  - **Multi-tensor** (Fusion Sprint Phase 4 follow-up + launch-storm
    Phase 1 2026-05-12):
    `adam_fused_multi_bf16_f32grad_kernel`,
    `adam_fused_multi_bf16_bf16grad_kernel`, and
    `adam_fused_multi_f32param_f32grad_kernel` (Phase 1, 2026-05-12).
    One launch covers every parameter via a device-resident metadata
    buffer (`fused::MultiTensorMetaCache`). `Adam::step` auto-selects
    this path when every param shares a dtype (BF16 or F32) and every
    grad is F32 — covers both Klein 9B (BF16 params) and zimage
    (F32 LoRA params) cases. Set `FLAME_ADAM_NO_MULTI_TENSOR=1` to
    force the per-param fallback. The `adam_fused_multi_tensor_step`
    dispatcher takes `param_is_bf16: bool` + `grad_is_bf16: bool` to
    route between kernel variants.

Bit-exact parity: `tests/adam_multi_tensor_parity.rs` asserts the
multi-tensor kernels and their per-tensor counterparts produce
identical bytes on 50-tensor 1-step toys (BF16 and F32 separately) +
drift-free agreement across 100 steps. The F32 path is bit-identical
(no float reassociation, same kernel math, same accumulation order);
the BF16 path matches within BF16 atol across 100 steps. The
multi-tensor variant only changes launch shape — kernel math is
identical including the DECOUPLED-WD ordering receipt.

### `adam8bit_kernel.rs` (added 2026-05-17, bnb-port)
Fused NVRTC kernel for bitsandbytes-equivalent 8-bit AdamW (256-element
block-wise dynamic-LUT quant). Byte-equivalent to
`bitsandbytes 0.49.2 F.optimizer_update_8bit_blockwise(optimizer_name="adam")`
(see `bitsandbytes/functional.py:1488-1550` for the dispatch we mirror, and
`bitsandbytes/optim/optimizer.py:460-555` for the state layout).

Two 256-entry dynamic-exponent qmaps (signed for `m`, unsigned for `v`) via
`create_dynamic_map`, allocated once per device and uploaded with
`upload_qmap`. State per parameter is `(m_codes: u8[n], v_codes: u8[n],
m_absmax: f32[ceil(n/256)], v_absmax: f32[ceil(n/256)])` — held as raw
`CudaSlice`s on the optimizer struct (no Tensor wrapper; see CONVENTIONS
"TensorStorage has no U8 arm"). One fused kernel launch per parameter per
step does dequant + AdamW math + block-wide max-abs reduction + requant in
shared memory; no PCIe round-trips.

Public surface:
- `create_dynamic_map(signed: bool) -> [f32; 256]` — CPU-side LUT builder.
- `upload_qmap(device, &[f32; 256]) -> CudaSlice<f32>` — single H2D copy.
- `alloc_state(device, numel) -> (m_codes, v_codes, m_absmax, v_absmax)`.
- `adam8bit_step_bnb(param, grad, m_codes, v_codes, m_absmax, v_absmax,
  qmap_signed, qmap_unsigned, lr, β₁, β₂, ε, wd, bc1, bc2)` — F32 param
  required; F32 or BF16 grad auto-routed to the matching NVRTC kernel.
- `ADAM8BIT_BLOCK_SIZE` — `const usize = 256`. Matches bnb hardcoded
  `optimizer.py:478`; do not change without matching kernel + caller edits.

Consumed by `eridiffusion-core::training::training_features::optimizers::AdamW8bit`.
Trainers that select `--optimizer adamw8bit` (`train_u1`, `train_wan22`)
get this path automatically — Z-Image and Klein trainers remain BF16/F32
per the no-quant rule.

**Parity status**: byte-exact vs bnb across 5 envelope cases (single-step,
10-step, wd>0, tail-block where `n % 256 ≠ 0`, BF16-grad). Validated
2026-05-17 via `tests/parity/adam8bit_bnb_python_ref*.py` +
`bins/parity_adam8bit_bnb*.rs`.

**Deferred** (separate variants in bnb; not required for current trainers):
- `PagedAdamW8bit` — bnb CUDA unified-memory CPU-spill variant
  (`optimizer.py:71-103`).
- `optimizer_name` values other than `"adam"` — `lion`, `momentum`,
  `rmsprop`, `adagrad`. Each is a different bnb kernel.
- `percentile_clipping` / `gnorm_scale` / `skip_zeros` — bnb defaults
  (100, 1.0, False); trainers use external `flame_core::ops::grad_norm`
  clipping instead.

### `int8_weight_only_qt_kernel.rs` (added 2026-05-17, torchao-port)
Port of torchao 0.14.1 `int8_weight_only_quantized_training` — the
**training** prototype API (NOT the inference `int8_weight_only`). Used
by SimpleTuner under `--base-quant int8-torchao` when `model_type=lora`.

Numerical contract mirrors
`torchao/prototype/quantized_training/int8.py:23-52` (`quantize_int8_rowwise`,
stochastic_rounding=False path) and `int8.py:146-173`
(`_Int8WeightOnlyLinear.forward` + backward).

The torchao differentiator vs the inference path: `F.linear` against an
`Int8QuantizedTrainingLinearWeight` is **differentiable** through the
input — the wrapper's dispatch routes to a custom `torch.autograd.Function`
that defines both forward and backward. Under LoRA-only training (the
only path SimpleTuner exercises with `int8-torchao`) the int8 base is
frozen and only adapter deltas learn; we therefore implement forward +
`grad_input` only. `grad_weight` and the `aten.copy_` stochastic-round
writeback are NOT ported (deferred — see below).

Public surface:
- `quantize_int8_qt(weight: &[f32], shape: [usize; 2], source_is_bf16: bool) -> (Vec<i8>, Vec<f32>)`
  — symmetric per-row absmax INT8 quant, divisor 127, eps on inv_scale
  only. `source_is_bf16=true` mirrors torchao's BF16-input path (the
  realistic one — SimpleTuner always passes BF16 base weights).
- `Int8QtWeight { codes: CudaSlice<i8>, scales: Tensor, shape, device }`
  + `upload(...)` + `codes_to_bf16() -> Tensor` — device-side storage.
- `linear_int8_qt(x, w, bias?) -> Tensor` — autograd-tracked forward;
  composes `Tensor::matmul + Tensor::mul + Tensor::add` so the standard
  autograd graph back-propagates `grad_x` (and `grad_bias` if needed)
  while the frozen int8 weight stays out of the gradient flow.

One NVRTC kernel (`int8_qt_cast::i8_to_bf16_kernel`): sign-extending
int8 → BF16 cast, 256 threads/block, no shared mem. Compiled once per
device on first `codes_to_bf16()` call.

**Parity status**: bit-exact (codes mismatch=0, scales max|Δ|=0,
forward+grad_x max|Δ|=0 and cos=1.0) vs torchao 0.14.1 on a [256, 384]
synthetic weight at seed=42. Validated 2026-05-17 via
`tests/parity/int8_torchao_python_ref.py` + `bin/parity_int8_torchao_qt`.
Two gotchas worth recording (both fixed in the port):
1. PyTorch `tensor.round()` uses IEEE round-ties-to-even (banker's
   rounding); Rust's default `f32::round()` uses round-half-away-from-zero.
   Use `f32::round_ties_even` instead.
2. torchao quantizes a BF16 nn.Linear weight in BF16 (absmax + scale
   computed in BF16, then `.float()` for the multiply). An F32-only
   port would diverge on ~5% of codes by ±1.

**Deferred** (not required for SimpleTuner LoRA parity):
- `_Int8WeightOnlyLinear.backward` `grad_weight` (int8.py:169-171) —
  needed only if someone enables full base-grad training on the int8
  weight directly.
- `aten.copy_` stochastic-rounding writeback (int8.py:236-252) — used
  when optimizer steps run **on the quantized weight**. Under LoRA-only,
  the int8 base never sees an optimizer step.
- FP8-torchao variant — `convert_to_float8_training` is a separate
  helper at `int8.py`'s sibling path; would be a new module here.

### `sgd/mod.rs`
Basic SGD with momentum + weight decay. F32 implementation with an inline
NVRTC kernel.

### `ops/grad_norm.rs`
Async global gradient L2 norm + clip-scale. Replaces the per-tensor
`g.square().mean().to_vec()?[0]` loop in EriDiffusion-v2 trainers
(N D2H syncs/step on Klein 9B LoRA = 200+ stalls) with at most one D2H
sync at the end. Two helpers:

- `global_l2_norm(grads: &[&Tensor]) -> Result<Tensor>` — returns a
  1-element FP32 device tensor (the global L2 norm). **Phase 3 fast
  path (2026-05-12):** when every grad is F32 + contiguous, dispatches
  to `ops::multi_tensor::multi_tensor_l2_norm_sq_f32` (Apex-style
  2-stage reduction = 3 launches total). Otherwise falls through to the
  legacy per-tensor `square().sum()` + serial fold-add. Env override
  `FLAME_MT_L2NORM=0` forces the legacy path. Empty slice short-circuits
  to a zero scalar. BF16 grads still go through the legacy path with
  on-the-fly F32 cast.
- `global_l2_norm_with_scale(grads, max_norm, eps) -> Result<(Tensor, Tensor)>` —
  norm + `min(max_norm/(norm+eps), 1.0)` clip scale, both as 1-element
  FP32 device tensors. Caller does at most one `.item()` for logging.

Parity tested vs PyTorch oracle on 200 LoRA-shape tensors —
`tests/grad_norm_parity.rs` (6 tests, atol=1e-4). Multi-tensor parity
in `tests/multi_tensor_l2_norm_parity.rs`: F32 fast path matches legacy
within abs ≤ 1e-5 / rel ≤ 1e-6 (parallel-tree vs serial-fold reduction
order). The process-wide `MT_L2_CACHE: Mutex<MultiTensorMetaCache>`
amortizes the device-side metadata + partials buffer across steps —
one-time alloc on step 0.

### `ops/multi_tensor.rs` (added 2026-05-12, Phase 3 of launch-storm refactor)
Foreach-style multi-tensor primitives that collapse per-parameter
launch storms in the trainer hot path. Adam already has its own
multi-tensor packed-buffer launcher in `adam::adam_fused_multi_tensor_step`;
this module generalizes the pattern for other op families.

Current entries:

- `MultiTensorMetaCache` — device-side packed-buffer + per-tensor
  partials cache. Reallocates when n_tensors changes. **Separate** from
  `adam::MultiTensorMetaCache` (different region layouts).
- `multi_tensor_l2_norm_sq_f32(cache, grads) -> Tensor` — F32-only
  sum-of-squares across a tensor list, returned as a 1-element F32
  device tensor. Two NVRTC kernels: stage 1 (block-per-tensor → partial
  sums) and stage 2 (single-block reduction → scalar). Used by
  `ops::grad_norm::global_l2_norm` fast path.
- `multi_tensor_scale_inplace_packed(cache, dev, n, &packed, scale, is_bf16)`
  (added 2026-05-12, Phase 2 of launch-storm refactor) — single-launch
  in-place `x[i] *= scale` across a packed list of F32 or BF16 tensors.
  Pointwise math (no reduction) so F32 is bit-exact and BF16 is 1-ULP-
  bound vs per-tensor `mul_scalar`. Two NVRTC kernels
  (`multi_tensor_scale_inplace_f32_kernel`,
  `multi_tensor_scale_inplace_bf16_kernel`), both loaded under
  `MT_SCALE_MODULE = "multi_tensor_scale"`. Trainer call sites
  (`train_zimage.rs`, `train_klein.rs`) gate the dispatch behind
  `FLAME_MT_SCALE=1` because production grad-norms in those configs sit
  well below the `CLIP_GRAD_NORM = 1.0` threshold, so the clip path
  doesn't fire and the multi-tensor primitive saves nothing in steady
  state. See `EriDiffusion-v2/HANDOFF_2026-05-12_PHASE2_SCALE_FOLLOWUP.md`.

The packed-buffer layout is `[ptrs(n) | sizes(n)]` (2n u64 entries) for
both L2 norm and scale. Each kernel is compiled once on first use,
loaded into a per-primitive cudarc module (`MT_L2_NORM_MODULE`,
`MT_SCALE_MODULE`), looked up by name on subsequent launches.

Future expansion: other foreach-pattern targets (fp16-master scatter,
multi-tensor `set_grad` for BF16 grad casts, etc.) as they prove
valuable in profiles.

### `parameter.rs`
`Parameter` (a `Tensor` wrapper with `requires_grad=true`) — re-exported as
both `Var` and `Parameter`.

### `linear.rs`
The `Linear` nn layer (`nn::Linear`). Used in training; inference paths
mostly use `ops::fused_inference::fused_linear3d_native` directly.

### `embedding.rs`
`Embedding` table (`nn::Embedding`). Token embedding lookup.

### `loss.rs`
Loss functions: MSE, CE, BCE, etc. Training-only.

### `regularization.rs`
Dropout and other regularizers. Training.

### `samplers.rs`
Older diffusion samplers (Karras, Euler, DDIM). The active samplers are in
the model-specific `inference-flame/sampling/*` files.

### `activations.rs`
Element-wise activation function impls. Most are now superseded by the
`bf16_ops.rs` fused versions.

### `pooling.rs / pooling_impl.rs`
2D pooling (avgpool/maxpool). Training.

### `mixed_precision.rs`
F16/BF16 amp helpers. Training.

### `lora.rs`
LoRA adapter helpers — apply LoRA deltas to weights at load time. Used by
training and the LoRA-aware inference loaders.

---

## Serialization

### ⭐ `serialization.rs`
safetensors load/save. The main entry points are `load_file` (read all
tensors), `load_file_filtered` (read only keys matching a closure — used
heavily by the inference model loaders to skip unused branches), `save_file`,
and the older `save_tensors / load_tensors / save_tensor / load_tensor` API
that takes a `SerializationFormat` enum (SafeTensors / Bincode). Lazy by
default — tensors are loaded on demand when iterating over the result map.
This is the file that ALL inference binaries import from.

---

## VAE

### `vae/mod.rs` / `vae/autoencoder_kl.rs` / `vae/zimage_decoder.rs`
Generic VAE components used by Z-Image and the older training code. Most
inference VAE work happens in `inference-flame/src/vae/*` (LDM, Klein, Wan21,
LTX-2 audio) instead.

---

## Image / upsampling

### `image_ops_nhwc.rs`
Image-space ops in NHWC layout (resize, etc).

### `upsampling.rs`
2D nearest/bilinear upsample. Used by VAE decoders and the LTX-2 latent
upsampler.

Also hosts `ConvTranspose2d` (`forward` / `backward`). As of 2026-04-18 the
forward GPU path is real — implemented in `cuda_kernels.rs::conv_transpose2d_forward`
by reducing to `conv2d_forward` via **zero-insert on H/W** (per-stride dilation),
**transposed-conv math padding**, **spatial kernel flip**, and an **in/out channel
axis transpose** of the weight. Supports `groups=1` and `dilation=(1,1)` only;
other configurations return `Error::Unsupported`. Backward still routes to the
pre-existing `conv_transpose2d_backward` stub. BF16 and F32 tensors both flow
through unchanged — the path is dtype-agnostic.

---

## Misc

### `kernels/adaln.rs` (feature-gated)
AdaLN kernel module — only compiled with the `cuda` feature.

### `fused_kernels.rs`
Older fused kernel registry. Replaced by `bf16_ops.rs` and `ops/fused_inference.rs`.

### `fp16.rs`
F16 conversion + storage helpers. Mostly used by `tensor_storage::F16`.

### `ops_ext.rs`
Small `OpResult`-typed helpers: `shape4 / matmul_tt / where_mask / mean_all_f32 /
zeros_like / full_like / transpose_last2`. These return a custom `OpResult`
to avoid coupling to the main `Result<T, Error>`.

### `ops/utils.rs`
Helpers for the `ops` family (validation, shape derivation, etc.).

### `ops/elt.rs / ops/broadcast.rs / ops/cast.rs / ops/reduce.rs / ops/tile.rs`
The "ops" namespace under `ops/`. Each file has functional wrappers around
elementwise / broadcast / cast / reduce / tile kernels. `ops/cast.rs` is the
public BF16↔F32 cast entry.

### `ops/gemm.rs / ops/gemm_bf16.rs`
Functional GEMM wrappers around `cuda_ops_bf16::gemm_bf16` and the F32
matmul. `ops/gemm_bf16.rs` is what `Tensor::matmul` routes to for BF16.

### `ops/attn.rs`
Older functional attention wrapper (mostly empty now; the live entries are
in `attention/sdpa.rs`).

### `ops/cuda/mod.rs / ops/cuda/lt.rs`
cuBLASLt-specific helpers — descriptors, layout setup, algo selection.

### `borrowed/mod.rs` (feature-gated)
"Borrowed weights" feature for BlockOffloader-style streaming where the tensor
data is owned externally.

### `python/*` (feature-gated)
PyO3 bindings — `bridge.rs`, `tensor.rs`, `nn.rs`, `functional.rs`.

### `capi.rs` (feature-gated)
C API surface for non-Rust callers.

### `kernel_launcher.rs`
`LaunchConfig` helpers — block/grid sizing, occupancy hints.

### `rng/mod.rs`
Global RNG — `global_rng()`, `set_seed(seed)`. Used by `Tensor::randn`. Also
re-exports the PyTorch-parity primitives from `rng::torch_compat`: NVRTC
kernels that reproduce PyTorch's CUDA RNG output bit-for-bit. All share the
same Philox4x32-10 setup that mirrors `curand_init(seed, idx, 0)` and
PyTorch's `distribution_nullary_kernel` grid policy. Output bytes depend on
GPU SM count — see module docstring. Parity-tested against torch fixtures
stored under `tests/torch_randn_fixtures/` and `tests/torch_compat_fixtures/`.
F32 + I32 only (no BF16/F16/I64 paths yet).

Functions exposed:
- `randn_torch(seed, shape, device)` — `torch.randn` (normal via Box-Muller).
- `rand_torch(seed, shape, device)` — `torch.rand` (uniform [0, 1)).
- `bernoulli_torch(seed, shape, p, device)` — `torch.empty(...).bernoulli_(p)`,
  emits 0.0/1.0 F32 (typical dropout-mask shape).
- `randint_torch(seed, low, high, shape, device)` — `torch.randint(low, high,
  shape)` for ranges `< 2^32`, output I32 (`flame-core` has no I64 storage).
- `kaiming_uniform_torch(shape, a: f64, fan, nonlinearity, seed, device)` and
  `xavier_uniform_torch(shape, gain: f64, fan_in, fan_out, seed, device)` —
  PyTorch `nn.init.{kaiming,xavier}_uniform_` initialisers; both wrap an
  internal `uniform_torch(seed, shape, -bound, +bound, ...)` so the
  Philox stream is identical to `tensor.uniform_(a, b, generator=g)`.
  `a` / `gain` are f64 (matching PyTorch's Python-float semantics). The
  `bound = sqrt(3) * gain / sqrt(fan)` math runs entirely in f64; only the
  final `bound` is cast to f32 at the kernel-launch boundary. Narrowing
  `a` or `gain` to f32 at the API boundary would lose 29 bits of mantissa
  and break bit-exact parity on non-power-of-2 fan values (e.g. 137, 1280).

**Layout gotcha.** PyTorch's CUDA RNG is shape-dependent (per-thread
subsequence). Calling `randn_torch(seed=42, shape=(N,))` and then
reshape-permuting to a 4D layout does NOT produce the same per-position
values as `randn_torch(seed=42, shape=(B, C, H, W))` — the flat stream is
the same but the (position) → flat-index mapping differs. To match Python's
`torch.randn(L, C)` followed by reshape/permute, call `randn_torch(seed,
shape=(L, C), device)` first, THEN reshape. See `inference-flame/src/bin/
lance_t2v.rs` for a canonical example: torch.randn(L, C) → reshape (T, H,
W, C) → permute (C, T, H, W) → unsqueeze batch.

### `devtensor.rs`
⚠️ Old per-device tensor wrapper. Predates the unified `Tensor`.

### `cudnn/*` (feature-gated)
cuDNN integration — separate handle, conv2d, conv3d, layer_norm, attention.
Live inference entry points:
- `cudnn::cudnn_conv2d_bf16` (via `cudnn::conv2d`)
- `cudnn::cudnn_conv3d_bf16` (via `cudnn::conv3d`, 2026-04) — called by
  `Conv3dBF16::forward` as the preferred path. Algo cache, BF16 NCDHW
  tensors, FP32 accumulate, workspace capped by
  `FLAME_CUDNN_CONV3D_WS_LIMIT_MB`.

`descriptors` has been extended with `FilterDescriptor::set_nd` and
`ConvolutionDescriptor::set_nd` for the 5D descriptors Conv3d needs.
The other modules (`activation`, `algorithms`, `linear`, `matmul`,
`matmul_simple`, `norm`, `attention`) are training-side.

### `tests/*` and `bin/*`
Test modules and standalone test/debug binaries. See `bin/` for runnable
sanity checks.

#### Parity test infrastructure (Fusion Sprint Phase 0)

- `tests/parity_helpers/mod.rs` — shared comparator. Public API:
  `compare_tensor(got, expected, atol, rtol)`,
  `compare_bf16(got, expected)`,
  `compare_fp32_reduction(got, expected)`,
  `assert_parity_bf16(name, got, expected)`,
  `sha256_file(path)`. See FLAME_CONVENTIONS.md "PyTorch Parity Tests"
  for tolerance defaults and policy.
- `tests/parity_smoke.rs` — 4-test smoke that pins fixture SHA256,
  asserts add parity, asserts the comparator catches a known mismatch,
  and asserts FP32-reduction tol is tight (not silently widened to BF16).
- `tests/pytorch_parity.rs` — broad per-op parity suite (unary, binary,
  scalar, comparison, sdpa, matmul). Predates the Fusion Sprint; uses
  its own helpers (`cos_sim_f32`, `assert_cos_sim`). New phases SHOULD
  use `parity_helpers::compare_bf16` instead.
- `tests/pytorch_fixtures/` — generated `.safetensors` fixtures.
  `smoke/add_4x8_bf16.safetensors` is pinned by SHA256 in `parity_smoke.rs`.

---

## Activation offload

### `activation_offload.rs` — push/pull GPU activations to pinned host RAM

The "offload instead of recompute" path for gradient checkpointing.
`ActivationOffloadPool` owns a non-blocking CUDA transfer stream and a bank
of pinned host buffers (one per slot). During forward, `push(tensor)` enqueues
an async DtoH on the transfer stream, gated by a per-slot event recorded on
the default stream so the copy cannot start before the producer kernel finishes.
During backward, `pull(handle)` enqueues the reverse HtoD, records a ready
event, and makes the default stream wait on it before any consumer touches the
returned tensor. Same-stream ordering (transfer stream) ensures the DtoH
completes before the HtoD for the same slot.

Slot allocation is stack-based (LIFO), matching autograd backward's reverse
consumption order. A per-pool epoch counter invalidated by `clear()` makes
stale handles fail loudly instead of silently corrupting.

`OffloadCompression::FP8` quantizes BF16 activations to FP8 E4M3 on the
transfer stream before DtoH (via `flame_bf16_to_fp8` in `src/cuda/fp8_quant.cu`)
and dequantizes after HtoD (via `flame_fp8_to_bf16` in `src/cuda/fp8_dequant.cu`).
This halves pinned memory and PCIe bandwidth at ~0.1% relative error.

The autograd integration lives in `autograd.rs`:
- `set_activation_offload_pool(pool)` installs the global pool (call once at
  training setup).
- `AutogradContext::checkpoint_offload(inputs, f)` runs the forward closure
  with autograd enabled, captures the sub-tape, offloads every saved tensor
  to CPU, and records a single `Op::CheckpointOffload` on the outer tape.
  At backward, saved tensors are pulled from CPU and the sub-tape is walked
  -- no recompute needed. Falls back to standard `checkpoint()` (recompute)
  if the pool is not set or runs out of slots.
- `OffloadedTapeEntry` stores the offloaded sub-tape entries with
  `OffloadHandle` keys replacing the original saved tensors.

The trainer-side setup helper is `flame-diffusion/src/offload.rs`:
`setup_activation_offload(device, config)` computes slot count from block
count + headroom, constructs the pool, and installs it via
`set_activation_offload_pool`.

## `offload` — `BlockOffloader` + `BlockFacilitator` (moved into flame-core 2026-05-12)

### `offload/mod.rs` — block-level weight offloader

⭐ Double-buffered sequential block offloader. Moved from
`flame-diffusion/src/block_offload.rs` into flame-core (commit `df00c5f`,
~1,579 LOC) so that any flame-core crate — trainers, inference-flame,
parity tests — can use the same offloading primitive without a circular
dep on flame-diffusion. The trainer-side `flame-diffusion/src/offload.rs`
remains as the activation-offload pool installer; block offloading is now
purely a flame-core concern.

**What it does.** Loads ALL block weights from safetensors into
`cudaMallocHost` pinned CPU memory at init (BF16-converted at load).
Maintains TWO GPU buffer slots for prefetch / compute overlap: while
compute runs on block N, block N+1 is being H2D-copied on a dedicated
transfer stream. Per-slot CUDA events (`h2d_done`, `compute_done`)
serialize stream-stream waits — no host-side `cudaStreamSynchronize` on
the hot path (Phase 0 event-safety work).

**Phase 2 post-reboot (2026-05-13): ring-backed slot allocations.**
Both prefetch paths (pinned at `prefetch_block_inner`, streaming at
`prefetch_block_streaming_inner`) now route their BF16 slot allocations
through a lazily-materialized 4-slab `RingAllocator` owned by the
offloader (`BlockOffloader::ensure_ring` + `alloc_bf16_via_ring`).
Pointer handed to the new BF16 slot is synthesized via the cudarc
0.11.x `CudaSlice` mirror layout and tagged as external in the
global `cuda_alloc_pool` so `push_u16` routes it through
`reconstruct_and_forget` + `unregister_external_ptr` on return
(never `cudaFree`, never re-cached). Net effect: trainers can run
with `FLAME_ALLOC_POOL=1` (the default) under offload without the
prior corruption mode (cached BF16 slot pointers being re-issued by
the pool while ring `reset()` had already invalidated them).

**R2c-perf (2026-05-15, commit `be67185`): OT-style resident-set conductor.**
The hard-coded two-slot ping-pong + one-block-prefetch policy is replaced
with a conductor modelled on OneTrainer's `LayerOffloadConductor`:

- `plan_layer_access` knows the forward AND backward traversal order
  and maintains the desired resident set under a fraction-of-VRAM byte
  budget. Slot count is no longer fixed; the policy admits as many
  blocks as the budget allows. `FLAME_BLOCK_OFFLOAD_SLOTS` (default 2)
  controls the slot ceiling for the pinned-RAM mode pre-conductor
  fallback.
- **Multiple async prefetches in flight**, queued by per-slot CUDA
  events instead of the prior one-block global in-flight gate.
- **Direction-aware**: separate forward-visit and backward-visit
  lists. During backward replay the conductor pre-stages block N-1
  while the trainer's checkpoint closure recomputes block N. Hooks
  on `AutogradContext::is_checkpoint_recompute()` (flame-core
  thread-local) so forward passes don't trigger spurious backward-
  direction prefetches.
- **Lazy eviction**: stale resident slots stay until a missing desired
  block needs that slot. Eviction is then GPU-side via
  `cudaStreamWaitEvent` on the previous `compute_done` — no host-side
  `cudaEventSynchronize` on the hot path. Pre-fix the eager evictor
  was at 51.0s / 1275 calls (~2.04 s/step) in nsys; lazy-evict
  eliminates it.

Telemetry: `258 AwaitHit / 0 AwaitMiss` on a 6-step Klein 9B run
confirms the bucket prefetch is landing. Combined with the
R2c-perf `static_slab_v2` + QKV/SwiGLU backward fixes and the
frozen-weight-gradient skips in `autograd::Op::MatMul` /
`Op::Mul` / `norm::rms_norm_backward`, Klein 9B + `--offload` ran
at **2.30 s/step steady (100-step head-to-head)** vs OneTrainer's
**2.79 s/step** on identical hardware, dataset (118-sample eri2),
and hyperparameters (LoRA r16/α16, AdamW, lr 3e-5, bs 1, bf16, 512
resolution with aspect bucketing). Pure-training wall: 229 s vs 280 s
(EDv2 1.22× faster). Total wall including setup/cache: 271 s vs 574 s
(EDv2 2.12× faster — OT pays per-epoch latent re-caching that EDv2
amortizes into a one-time prep step). Pre-redesign baselines for
context: `FLAME_ALLOC_POOL=0` workaround was 4.6 s/step; 661f9e9
pre-regression was 3.4-3.8 s/step.

**Three load modes:**
- `BlockOffloader::load(paths, facilitator, device)` — default pinned
  path. Pinned RAM ≈ full block-weight size. Fastest hot path.
- `BlockOffloader::load_fp8_stream(paths, facilitator, device)` — same
  shape, but keeps F8_E4M3 tensors as raw FP8 bytes pinned + GPU-dequants
  to BF16 inside `prepare_weights`. Halves pinned RAM for FP8 checkpoints
  (LTX-2 distilled-fp8, Wan2.2 14B experts).
- `BlockOffloader::load_streaming(paths, facilitator, device)` — mmap
  source files at init; pinned RAM = only `2 × max_block_bf16_bytes`
  (typically <1.5 GB). Each `prefetch_block` copies that block's bytes
  from mmap into one of two pinned staging buffers, then issues async H2D
  from staging into a fresh GPU slot. For Qwen-Image-2512-class models
  (≈39 GB block weights) that don't fit in 62 GB pinned. F8_E4M3 not
  supported in this mode.

**Public hot-path API.** Indented to match the call shape from a trainer
or inference block loop:
- `prefetch_block(idx)` — start async H2D into the non-active slot. No-op
  if already resident.
- `await_block_handle(idx) -> BlockHandle` — **preferred**. Returns a
  scoped `BlockHandle` whose Drop records `compute_done` on the default
  stream. The next prefetch reusing the slot waits on that event via
  `cudaStreamWaitEvent` (stream-stream, GPU-side) — no host sync.
- `await_block(idx) -> Arc<HashMap<String, Tensor>>` — legacy bare-Arc
  API. Slot reuse falls back to host-side `cudaDeviceSynchronize`. Kept
  for backward compatibility; migrate hot paths to `await_block_handle`.
- `ensure_block(idx)` — `prefetch + await` in one call (sync). Used by
  the inference loop when overlap isn't needed.

**`BlockFacilitator` trait.** Model-specific geometry provider with just
two methods: `block_count()` and `classify_key(name) -> Option<usize>`.
Each trainer / inference model supplies its own.

**Builder knob.** `with_native_layout(true)` disables the
`[Cout,Cin] → [Cin,Cout]` pre-transpose that `prepare_weights` otherwise
applies. Required when the caller's forward uses
`ops::fused_inference::fused_linear3d_native` (cuBLASLt TRANSA=T).
Pre-transposing in that case silently feeds the wrong layout to the GEMM.

**Trainers using it:** klein, sdxl, flux, chroma, ernie, qwenimage,
sensenova_u1, nucleus, ltx2, wan22, helios, hunyuan15, magihuman (DiT,
MMDiT, MoE, and video-DiT trainers — see the `*-trainer` crates under
`EriDiffusion-v2/`).

**Inference-flame models using it:** Klein, Chroma, Flux1, Wan22, LTX2,
Gemma3, Mistral, HiDream-O1, MagiHuman, plus every other model with a
block-shaped weight layout (each defines its own `*Facilitator`).

### Offload telemetry — `offload::telemetry` (Phase 1 FlexTensor port, 2026-05-12)

⭐ Port of the **measurement-and-observation** half of NVIDIA FlexTensor's
`instrumentation/` package (registry + dumper) into a flame-core shape.
Strategy / state-machine parts of FlexTensor stay out of scope for
Phase 1 (see HANDOFF for Phase 2/3 plans).

**What it provides.** A process-global, lock-free `Telemetry` sink with:
- Atomic counters: `h2d_bytes_total`, `prefetch_wall_ns`, `await_wall_ns`,
  `await_hits`/`misses`, `prefetch_issued`/`already_resident`.
- An opt-in bounded ring buffer of per-event traces (block_idx, bytes,
  duration_ns, kind).
- `format_counters(&snapshot)` for `eprintln!` / log dumps.

**Hot-path cost.** Disabled mode is a single relaxed atomic load on entry
to `prefetch_block` / `await_block` plus a no-op `TelemetryTimer`. Enabled
mode is one atomic load + an `Instant::now()` at begin/end + several
relaxed atomic adds (≤ 100 ns total on commodity hardware). Trace mode
adds a single `Mutex` push of a 24-byte record — only touched when trace
is enabled.

**Activation.** Three options:
- Set `FLAME_OFFLOAD_TELEMETRY=on` (counters) or `=trace` (counters +
  ring buffer) in the environment at process start.
- Call `flame_core::offload::telemetry::global().set_enabled(true)` from
  code; for trace mode, also call `set_event_log_capacity(N)`.
- `FLAME_OFFLOAD_TELEMETRY_RING=N` overrides the ring buffer capacity.

**Hooks already wired into `BlockOffloader`:**
- `prefetch_block` start → `record_prefetch_begin`
- `prefetch_block` end → `record_prefetch_end` (with byte count of the
  block) or `record_prefetch_already_resident` (no-op fast path).
- `await_block` start → `record_await_begin`
- `await_block` end → `record_await_end_hit` (slot already had block) or
  `record_await_end_miss` (await issued its own H2D).

The "miss" branch deliberately calls a private `prefetch_block_inner` so
the await-charged miss isn't *also* counted as a `prefetch_issued`.

### Offload transfer benchmark — `offload::transfer_benchmark` (Phase 1 FlexTensor port, 2026-05-12)

⭐ Port of `flextensor/memory_transfer_benchmark.py` +
`memory_transfer_interpolator.py`. One-time PCIe H2D/D2H bandwidth sweep
across a geometric range of transfer sizes, plus a log-log interpolator
for `bytes → predicted Duration` (and inverse via a future overload).
Result is a `TransferBandwidthProfile` cached for the process lifetime.

**Why it lives in flame-core.** Bandwidth is hardware-specific, not
model-specific. Every BlockOffloader caller on a given box sees the same
PCIe bus — fix the measurement primitive once, every model can consult
the same profile (tenet §1).

**What it measures.** GPU-observed wall time via `cudaEvent` start/stop
bracketed around one `memcpy_async`, with `cudaEventSynchronize` on the
stop event. This is the standard CUDA timing pattern and is exempt under
clause 1 of the speed contract for *init-time* code.

**What it does NOT measure.** Per-step memory churn, prefetch overlap
quality, or kernel launch cost. Those belong to `offload::telemetry`
(host-observed) and Phase 2/3 work respectively.

**Configuration knobs (`BenchmarkConfig`).** `min_bytes` / `max_bytes`
(default 1 KiB → 256 MiB), geometric `samples` count, `trials` per point
(median is reported), `warmup_trials`, `measure_d2h`.

**Result accessors.** `profile.h2d()` / `profile.d2h()` — measurement
slices, ordered by size. `profile.predict_h2d(bytes)` / `predict_d2h` —
log-log interpolated `Duration` with endpoint-slope linear extrapolation.
`profile.peak_h2d_bps` / `peak_d2h_bps` — measured peak at the top of
the size range.

**Diagnostic output.** `profile.format_table()` matches the layout of
FlexTensor's `format_memory_transfer_table` (size, direction, duration
ms, bandwidth GB/s).

On a single 3090 Ti the smoke test (5 sizes, 64 KiB → 16 MiB) measured
peak ≈26 GB/s H2D / D2H — close to PCIe 4.0 x16 theoretical max for
unidirectional traffic.

### Offload manager — `offload::manager` (Phase 3 FlexTensor port, 2026-05-12)

⭐ State-machine wrapper around `BlockOffloader` that auto-selects a
strategy at activation time. Closes Phase 3 of the FlexTensor port:
Phase 1 (telemetry) measures, Phase 2 (strategy trait) supplies
primitives, Phase 3 ties them together with an autonomous policy.

**Motivation.** PCIe bandwidth is the hardware ceiling (Phase 1 measured
~26 GB/s on this 3090 Ti). Phase 3 is *not* about beating that — it is
about **operational reliability for heavy memory-pressure workloads**
(sensenova_u1 @ 2048², hidream-o1 32B mixed-precision experts,
ltx2/wan22 video DiTs). Trainers ask the manager to "figure it out"
instead of hand-configuring `set_strategy` per model.

**Lifecycle.** `OffloadPhase` is `NotInitialized → Discovery → Profiling
→ Active`. Each transition is explicit; the upstream FlexTensor
`OffloadPhase` had `INFERENCE` as the final phase — flame-core does not
distinguish inference vs training at this layer (both run through the
same slot mechanic), so `Active` subsumes both.

- `discover()` snapshots the offloader's block geometry. Flame-core has
  no `__torch_function__` tensor crawler; block IDs are stable inputs
  declared via `BlockFacilitator`.
- `run_profile()` loads a cached `TransferBandwidthProfile` from disk if
  available, or runs the PCIe sweep + writes the result back. Default
  cache path is `${XDG_CACHE_HOME:-$HOME/.cache}/flame-core/offload_profile.json`
  (honors `FLAME_OFFLOAD_PROFILE_PATH`).
- `activate()` installs a `Strategy` on the underlying `BlockOffloader`.
  Default decision: if `2 × max_block_bytes < 0.3 × (free_VRAM −
  vram_headroom_bytes)` → `TwoSlot`; otherwise → `Adaptive`. Override
  via `ManagerConfig::force_strategy`.

**Tenet alignment.** Strategy decisions are pure host logic — no CUDA
launches, no `cudaStreamSynchronize`. The single CUDA touchpoint is the
`cuda_mem_get_info` call inside `activate()`, which is a non-blocking
driver query. `BlockOffloader`'s existing public API is unchanged;
existing trainers (and Phase 2 manual `set_strategy` callers) are
bit-identical to pre-Phase-3 behavior.

**Phase 3 also ships `offload::state`** — serde JSON persistence of the
`TransferBandwidthProfile`. Schema-versioned (`SCHEMA_VERSION = 1`),
mismatched files force a re-bench. Other FlexTensor `state_handler.py`
surface (tensor-mode persistence, multi-process shm coordination) is
out of scope — flame-core has no equivalent.

### Offload telemetry export — `offload::telemetry` (Phase 4, 2026-05-12)

⭐ Production-grade JSON exporters on top of the Phase 1 telemetry sink.
Makes counters and event traces visible *outside* the process without
requiring source edits in every trainer. All export paths are atomic
(write-to-tmp + rename) and host-only — no CUDA calls.

**Three export entry points** in `offload/telemetry.rs`:

- `snapshot_to_file(path)` — single JSON document, the current
  `TelemetryCounters` snapshot. Cheap; a few hundred bytes per call.
- `ring_buffer_to_file(path)` — JSON-lines (one event per line) dump of
  the per-event ring buffer. Empty when trace mode is off.
- `dump_all(dir)` — convenience pair-dump. When `dir = None`, falls back
  to `$FLAME_OFFLOAD_TELEMETRY_DUMP_DIR`, then to
  `std::env::temp_dir()`. Returns the directory actually used.

**Periodic dump.** Set `FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_EVENTS=N`
(or call `Telemetry::set_periodic_dump_interval(N)` programmatically) and
every Nth recorded prefetch / await event triggers a `dump_all` to the
configured directory. The interval counts **events**, not training
steps — each `record_prefetch_end` / `record_await_end_{hit,miss}` ticks
the counter (klein 9B emits ~64 events/step in trace mode, so `=1000`
fires the dump every ~16 training steps). Legacy alias
`FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_STEPS` is recognized but
deprecated — its `STEPS` suffix was a misnomer. Cheap when disabled —
one relaxed atomic load per event. Failures are logged via `eprintln!`
and swallowed so a missing dump dir cannot break training.

**Serde derive on the data types.** `TelemetryCounters`, `TelemetryEvent`,
and `TelemetryEventKind` all `derive(Serialize, Deserialize)`. External
tools can read the JSON without depending on flame-core.

**Tenet alignment.** Clause 1 of `SPEED_CONTRACT.md` (no implicit syncs)
is satisfied trivially — the export path never touches a CUDA function.
The atomic-rename pattern (tmp file + `std::fs::rename`) means a process
crash mid-write leaves the previous file intact, so an external monitor
can read the file at any time.

See [`OFFLOAD_GETTING_STARTED.md`](./OFFLOAD_GETTING_STARTED.md) for a
consumer-facing tutorial covering BlockOffloader, strategies, manager,
and telemetry together.

### Remaining wiring (not yet done)

Three things need to happen before activation offload is live in training:

1. **Pool construction in each trainer's `main.rs`**. Call
   `flame_diffusion::offload::setup_activation_offload(device, config)` after
   model load, before the training loop. The `OffloadConfig::from_model()`
   helper computes slot count from block count + headroom. `seq_len` must be
   the MAXIMUM across the dataset (largest bucket), not a single sample.

2. **Activation offload integration** in each trainer's block loop. Wrap
   the block forward in `checkpoint_offload` to offload activations between
   forward and backward passes.

3. **`FLAME_ACTIVATION_OFFLOAD=1`** env var to switch the block loop from
   the standard forward path to `checkpoint_offload`. Both Wan and LTX-2
   trainers already check this variable and branch accordingly.

Without step 1, `checkpoint_offload` falls back to standard `checkpoint()`
(recompute, no offload) because no pool is installed. Without step 2, VMM
budget gating and eviction scoring are inert. Without step 3, the block
loop doesn't use `checkpoint_offload` at all.

### Architecture decisions and known gotchas

**Why Level 2 (no recompute) instead of Level 1 (offload input + recompute):**
Level 1 was attempted and abandoned. The recompute closure captures input
tensors via `Arc`, keeping GPU memory alive even after offloading to CPU.
Net effect: zero VRAM savings plus wasted HtoD on pull. Level 2 runs the
forward once with autograd enabled, stores the sub-tape, and offloads ALL
saved tensors. No closure captures, no recompute. The fallback when the
pool is full is standard `checkpoint()` (recompute, no offload) — NOT a
broken Level 1.

**Closure captures and `refresh_cache()`:** When `checkpoint_offload` falls
back to `checkpoint()`, the closure runs with autograd disabled on first
pass. LoRA adapters cache BF16 views without tape history, causing zero
gradients. Both Wan and LTX-2 closures call `bundle.refresh_cache()` as
the first statement inside the closure to prevent this. Any new trainer
wiring `checkpoint_offload` MUST do the same.

**FP8 compression fixed scale:** The pool uses `scale = 8.0 / 448.0` which
maps activation range `[-8, 8]` to FP8 E4M3 range. Values beyond `+/-8`
saturate. For typical transformer activations this is fine. If a model
shows clipping artifacts, replace with adaptive scale (absmax reduction
before push — not yet implemented, marked TODO in `activation_offload.rs`).

**Pool sizing for variable resolutions:** `OffloadConfig::from_model()`
takes a single `seq_len`. This must be the MAXIMUM across the entire
dataset (the largest bucket's token count). If a later sample exceeds
`slot_bytes`, `push()` returns an error and `checkpoint_offload` falls
back to `checkpoint()` for that block. Not a crash — just slower.

**`linear3d` auto-dispatch:** `klein.rs::linear3d` now detects
weight layout automatically. Pre-transposed `[in, out]` (resident path)
uses `matmul`. Non-transposed `[out, in]` (swap path) uses
`fused_linear3d_native` with cuBLASLt TRANSA=T — zero transpose allocation.
The detection relies on `w_shape[1] == in_features && w_shape[0] != in_features`.
For square weight matrices (e.g. 3072x3072) this is ambiguous — the swap
path passes `native_weights=true` explicitly via the `lin` closure inside
each block forward to avoid misdetection.

**`prepare_block` vs old `materialize`:** The old `materialize` function
transposed 2D weights (GPU alloc + kernel) and round-tripped 1D weights
through CPU (`to_vec_bf16` → `copy_from_bf16_slice`). Both are eliminated.
2D weights pass through as-is (TRANSA=T handles them). 1D weights use
`clone_result()` (D2D, ~2us). Validated bit-identical against sync path.

**`OffloadedTapeEntry` and sub-tape storage:** `Op::CheckpointOffload`
stores a `Vec<OffloadedTapeEntry>` — the sub-tape with saved tensors
replaced by `OffloadHandle`s. Non-BF16 tensors (F32 gradients etc) stay
GPU-resident in `resident_fallback`. During backward, the pool is locked
for the entire sub-tape walk (pulls happen under one mutex guard). This is
fine for single-stream training but would need batched-pull optimization
for pipeline parallelism.

**Note:** `conductor.rs` and `vram_budget.rs` have been DELETED. All block
offloading now uses `BlockOffloader` exclusively (two fixed GPU slots, no
eviction policy needed). The eviction scoring and VMM intelligence described
in earlier versions of this doc are no longer present.

### Files changed in this session (2026-04-12)

**flame-core:**
- `src/activation_offload.rs` — pool hardened (stack alloc, keep-alive, FP8)
- `src/cuda/fp8_quant.cu` — NEW: BF16→FP8 quantize kernel
- `src/cuda/ffi.rs` — `flame_bf16_to_fp8` FFI
- `build.rs` — registered fp8_quant.cu
- `src/autograd.rs` — `Op::CheckpointOffload`, `checkpoint_offload()`,
  `set_activation_offload_pool()`, `OffloadedTapeEntry`
- `docs/FLAME_MODULES.md`, `docs/FLAME_INDEX.md`, `docs/FLAME_KERNELS.md`

**flame-diffusion:**
- `src/offload.rs` — NEW: pool setup helper
- `src/lib.rs` — exports for offload
- `wan-trainer/src/forward_impl/forward.rs` — checkpoint_offload wiring
- `wan-trainer/src/model.rs` — WanLoraBundle Clone
- `wan-trainer/src/forward_impl/rope.rs` — WanRope Clone
- `ltx-trainer/src/forward_impl/forward.rs` — checkpoint_offload wiring
- `ltx-trainer/src/model.rs` — LtxLoraBundle Clone

**inference-flame:**
- `src/models/klein.rs` — killed materialize, TRANSA=T via `linear3d_nt`,
  `native_weights` flag on block forwards, `prepare_block` zero-alloc

---

## How to navigate this codebase

1. **Start with the live API**, not the file count. Most of the 80 files are
   training-side or legacy. The actual inference hot path is:
   - `tensor.rs` — Tensor type
   - `attention::sdpa` — SDPA dispatcher
   - `ops::fused_inference::*` — fused primitives (linear, RMSNorm, modulate, gate)
   - `tensor_iterator::ops::*` — BF16 pointwise (add/mul/silu/gelu/exp/ge/…)
   - `bf16_ops::*` — RoPE, swiglu, gate_residual, modulate, qkv_split
   - `bf16_elementwise::*` — softmax_lastdim, transpose2d, patchify/unpatchify
   - `cuda_ops_bf16::*` — BF16 op surface (norms, conv, sdpa_stream)
   - `serialization::*` — safetensors load/save
   - `cuda::ffi` — FFI declarations for the C kernels

2. **When grepping**, prefer file:line over symbol. Many symbols are
   duplicated across modules (LayerNorm exists in 3 places, sdpa in 2).
   The doc here marks which one is canonical.

3. **When in doubt about a symbol**, look it up in [`FLAME_INDEX.md`](./FLAME_INDEX.md).

4. **Before adding a kernel**, check [`FLAME_KERNELS.md`](./FLAME_KERNELS.md)
   to see if there's already one. There usually is.

5. **Before adding a new BF16 op**, check [`FLAME_CONVENTIONS.md`](./FLAME_CONVENTIONS.md)
   for the standard pattern.
