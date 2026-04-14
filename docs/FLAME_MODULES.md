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
auto-route to the BF16 fast path (`bf16_elementwise::*_bf16` for shape-equal
elementwise, `bf16_elementwise::softmax_lastdim_bf16` for last-dim softmax,
`bf16_ops::silu_bf16` for unary). The struct is large but its discoverability
is good — start in [`FLAME_INDEX.md`](./FLAME_INDEX.md) "tensor.rs" section.

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

### ⭐ `bf16_elementwise.rs`
Broadcast and flat-path BF16 elementwise ops. Has TWO paths: the slow
generic broadcast kernel (per-element 8-D offset computation) and the fast
flat path with `__hadd2`/`__hmul2`/`__hsub2`/`__h2div` vectorized over 2
elements per thread. `add_bf16` / `mul_bf16` / etc. dispatch to the flat
path when shapes match exactly. Also home of `transpose2d_bf16` (2D BF16
transpose), the comparison ops returning u8, the patchify/unpatchify
helpers, and `softmax_lastdim_bf16` (the fused last-dim softmax that
`Tensor::softmax` dispatches to for BF16).

### ⭐ `bf16_ops.rs`
Single-arg BF16 ops + the fused inference primitives that don't fit in
`fused_inference.rs`. `silu_bf16`, `gelu_bf16`, `square_bf16` (2-elem/thread
vectorized), `rope_fused_bf16` (interleaved-pair RoPE for FLUX/Klein/LTX/
Qwen/Chroma), `rope_halfsplit_bf16` (Z-Image variant), `gate_residual_fused_bf16`
(`x + gate*attn`), `swiglu_fused_bf16` (`silu(gate)*up`), `modulate_pre_fused_bf16`
(DiT modulate). All NVRTC, runtime-compiled.

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
thin wrapper around a `flame_*_bf16` C entry in `cuda::ffi`. Eight pub fns:
- `dequant_fp8_to_bf16`, `dequant_fp8_to_bf16_into`, `dequant_fp8_transpose_into` — FP8 unpack
- `fused_rms_norm` — RMSNorm in one kernel
- `fused_modulate` — `(1+scale)*x + shift` 
- `fused_linear3d` — cuBLASLt 3D linear with pre-transposed weight
- `fused_linear3d_native` — same but takes PyTorch `[Cout, Cin]` weight (added 2026-04, used by every FLUX/Chroma/QwenImage block forward)
- `fused_rms_norm_modulate` — RMSNorm + modulate fused
- `fused_residual_gate` — `x + gate*attn` fused

The corresponding `.cu` files live in `src/cuda/fused_*.cu`.

---

## Attention / SDPA — multiple paths

### ⭐ `attention/sdpa.rs`
The public attention surface. `sdpa(q, k, v, mask)`, `sdpa_with_bias(q, k, v, bias, scale)`,
`attend(...)`, `attention_impl(...)`. Routes BF16+head_dim∈{64,96,128} to the
wmma flash kernel; everything else falls back to the F32 path. Also defines
`MultiHeadAttention`, `RotaryEmbedding`, `TransformerBlock`, `GeGLU`,
`FeedForward`, and (legacy duplicate) `LayerNorm` structs used by training
code.

### ⭐ `sdpa.rs` (top-level)
Lower-level dispatcher used by `attention::sdpa::sdpa` and called directly
by some inference code (`vae::ldm_decoder`, `vae::wan21_vae`,
`ltx2_model.rs`). `forward(q, k, v, mask)`, `forward_with_bias(...)`,
`forward_v4(...)` (feature-gated). Chooses between the wmma flash kernel
(`forward_flash_bf16` → `flame_flash_attention_bf16` C entry), the cuBLASLt
materialized fallback (`forward_bf16_fallback`), and the streaming SDPA
(`sdpa_stream_bf16`) based on env vars and shape. Caches dispatch decisions
via `use_flash_attn()` / `force_stream_sdpa()` / `chunk_limit_from_env()`.

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

### `norm.rs`
Older norm wrappers (BatchNorm, etc.) — training-only.

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
`forward(input)`. NCDHW layout. Implementation uses an im2vol → cuBLASLt
GEMM → bias add pipeline. Used by LTX-2 audio VAE and the (planned)
Wan / QwenImage 3D VAE ports.

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
- `cuda::device_lt` — cuBLASLt handle + stream pointer accessors
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
`rms_norm_bf16` (the live RMSNorm entry), `rms_norm_bf16_to_f32`,
`layer_norm_bf16` (live), `layer_norm_bf16_with_stats`,
`group_norm_bf16` (NHWC), `gemm_bf16`, `slice_axis_bf16`, `broadcast_to_bf16`,
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

### `memory_pool.rs`
F32 device memory pool. ~15 pub fns. Mostly training, but the pool is also
used by some BF16 fast-path arena code.

### `pinned.rs`
Pinned host memory: `PinnedHostBuffer / PinnedHostBufferView /
PinnedHostBufferViewMut`, `PinnedAllocFlags`, `StagingDeviceBuf` (a paired
host pinned + device staging buffer for async H2D), `register_slice_as_pinned`,
`unregister_pinned`, and the `memcpy_async_device_to_host /
memcpy_async_host_to_device` helpers. Used by FlameSwap and the safetensors
loader for fast H2D.

### `pinned_pool.rs`
`PinnedPool` — a pool of reusable pinned host buffers for the staging path.
Re-exported at `flame_core::PinnedPool`.

### `staging.rs` (BF16-only, gated)
~16 pub fns. BF16 arena + async copy primitives. Used internally by
`Tensor` for some hot paths via `bf16_copy_async` and `ArenaLease`.

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

### ⚠️ `autograd_simple.rs / autograd_engine.rs / autograd_ops.rs / autograd_ops_complete.rs / autograd_debug.rs`
Older autograd attempts. Dead code; kept for reference.

### `gradient.rs`
`GradientMap` (re-exported as `GradientMap` and `GradStore`), `TensorGradExt`
trait. The collection that holds `tensor_id → grad_tensor` mappings during
backward.

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
`AdamW` implementation. BF16 master weights, F32 moments by default
(configurable via `select_optimizer_state_dtype`). Re-exported as `nn::AdamW`.
Includes `set_lr()` for step-wise schedulers.

### `optimizers.rs`
Additional optimizer variants (Lion, RAdam, etc.).

### `sgd/mod.rs`
Basic SGD with momentum + weight decay. F32 implementation with an inline
NVRTC kernel.

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
"Borrowed weights" feature for FlameSwap-style streaming where the tensor
data is owned externally.

### `python/*` (feature-gated)
PyO3 bindings — `bridge.rs`, `tensor.rs`, `nn.rs`, `functional.rs`.

### `capi.rs` (feature-gated)
C API surface for non-Rust callers.

### `kernel_launcher.rs`
`LaunchConfig` helpers — block/grid sizing, occupancy hints.

### `rng/mod.rs`
Global RNG — `global_rng()`, `set_seed(seed)`. Used by `Tensor::randn`.

### `devtensor.rs`
⚠️ Old per-device tensor wrapper. Predates the unified `Tensor`.

### `cudnn/*` (feature-gated)
cuDNN integration — separate handle, conv2d, layer_norm, attention. The
`cudnn::cudnn_conv2d_bf16` entry is used by inference code. The other modules
(`activation`, `algorithms`, `descriptors`, `linear`, `matmul`, `matmul_simple`,
`norm`, `attention`) are training-side.

### `tests/*` and `bin/*`
Test modules and standalone test/debug binaries. See `bin/` for runnable
sanity checks.

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

## VMM intelligence (flame-diffusion)

### `vram_budget.rs` — watermark-based VRAM budget manager

Queries `cudaMemGetInfo` for real driver-level VRAM usage and provides
predicates for prefetch/eviction decisions. Two watermarks divide VRAM into
three zones: below low (safe, stop evicting), between low and high (caution),
above high (must evict before prefetching). The `can_prefetch()` predicate
gates `SwapCoordinator::try_prefetch_next()` — when VRAM exceeds the high
watermark, prefetch pauses until eviction frees space.

Note: flame-core uses cudarc (not PyTorch), so `cudaMemGetInfo` is
authoritative. The mempool's "release threshold = MAX" policy means freed
allocations stay cached, making `used = total - free` conservatively high.
This is correct for eviction decisions: better to evict early than OOM.

### `conductor.rs` — SwapCoordinator with VMM

Extended from a simple cursor-based prefetcher to a VMM-aware conductor:

- **`with_budget(budget)`**: enables VRAM budget enforcement. Without this,
  legacy behavior (blind prefetch) is preserved.
- **`acquire_block(idx)` / `release_block(idx)`**: refcount protection. A
  block with refcount > 0 cannot be evicted. Trainers call acquire before
  block forward and release after.
- **`set_step(step)`**: updates the current training step for eviction scoring.
- **`best_eviction_candidate()`**: scores evictable blocks by
  `(staleness + distance_to_next_use) * size_bytes`. Highest score = best
  candidate. Only considers blocks with refcount == 0 and non-permanent.
- **`try_prefetch_next()`**: now checks `budget.can_prefetch()` before issuing
  H2D. Returns `Ok(false)` if budget is exhausted, pausing the pipeline.
- **`clear()`**: resets all refcounts to 0 between training phases.

The VMM layer is opt-in via `with_budget()`. Without it, the conductor
behaves identically to v1 — no VRAM queries, no eviction scoring.

### Remaining wiring (not yet done)

Three things need to happen before activation offload is live in training:

1. **Pool construction in each trainer's `main.rs`**. Call
   `flame_diffusion::offload::setup_activation_offload(device, config)` after
   model load, before the training loop. The `OffloadConfig::from_model()`
   helper computes slot count from block count + headroom. `seq_len` must be
   the MAXIMUM across the dataset (largest bucket), not a single sample.

2. **`acquire_block` / `release_block`** in each trainer's block loop. One
   line before block forward (`coord.acquire_block(i)`), one line after
   (`coord.release_block(i)`). Then pass `SwapCoordinator::new(...).with_budget(VramBudget::default_24gb())`
   at conductor construction to activate VMM gating.

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

**FlameSwap `linear3d` auto-dispatch:** `klein.rs::linear3d` now detects
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

**Eviction scoring is scaffolding:** `best_eviction_candidate()` is built
and correct but has no eviction ACTION wired. FlameSwap's pre-allocated
slots recycle automatically. The real eviction target is the `resident`
HashMap (permanently loaded blocks). An `evict_resident(block_idx)` that
removes from the map and drops the `Arc<HashMap<String, Tensor>>` is the
logical next step. Until then, `with_budget()` gates prefetch but relies
on slot recycling for VRAM relief.

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
- `src/vram_budget.rs` — NEW: VRAM watermark manager
- `src/conductor.rs` — VMM intelligence (budget, refcount, eviction)
- `src/lib.rs` — exports for offload + vram_budget
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
   - `bf16_ops::*` — RoPE, silu, gelu, swiglu, gate_residual
   - `bf16_elementwise::*` — flat elementwise + softmax
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
