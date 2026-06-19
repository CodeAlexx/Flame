# flame-core MAP

Wayfinding for cold-start. See `CLAUDE.md` (this directory) for hard rules and `docs/FLAME_README.md` for the docs index. This file is "where does X live".

## 1. Entry points

- Library root: `src/lib.rs`. Re-exports: `Tensor`, `TensorId`, `DType`, `Device`, `Shape`, `Error`/`Result`, `Parameter` (alias `Var`), `GradientMap`, `AutogradContext`, `Op`, `Module`, `nn::{Linear, Conv2d, LayerNorm, Embedding, AdamW}`. Auto-runs `init()` via `#[ctor]` (reads `FLAME_DEFAULT_DTYPE`, `FLAME_FORCE_CUDNN`).
- Version: `26.15.1`. Pinned by `EriDiffusion-v2` with `version = "=26.15.1"` — bumping breaks trainers.
- Downstream: `inference-flame` (features `["cuda","bf16_u16","cudnn","shared_storage"]`), `EriDiffusion-v2` (trainers).
- Optional: `src/capi.rs` (feature `capi`), `src/python/` (feature `python`).

## 2. Top-level `src/` layout

| Path | Purpose |
|---|---|
| `lib.rs`, `config.rs`, `device.rs`, `dtype.rs`, `error.rs`, `shape.rs` | Crate spine, dtype/cuDNN toggles, `Device`, `DType`, `Shape`. |
| `tensor.rs`, `tensor/`, `tensor_storage.rs`, `tensor_ext.rs`, `tensor_narrow.rs`, `tensor_ops_*.rs` | `Tensor` struct, storage, contracts, narrow/view ops. |
| `tensor_iterator/` | PyTorch-style dispatch registry for BF16 kernels. |
| `autograd.rs` (+ `autograd/policy.rs`) | **Live** autograd. Global tape, `AutogradContext`, `Op` enum (~47 variants). |
| `autograd_v2/`, `autograd_v3.rs`, `autograd_v4/`, `autograd_simple.rs`, `autograd_*.rs` | Legacy / experimental. See §3. |
| `gradient.rs`, `gradient_clip.rs`, `gradient_checkpointing.rs`, `diagnostics.rs`, `parity.rs` | `GradientMap`, clip, checkpoint; `diagnostics::assert_grad_flow`; `parity::ParityHarness`. |
| `parameter.rs`, `linear.rs`, `embedding.rs`, `layer_norm.rs`, `group_norm.rs`, `norm.rs`, `conv*.rs`, `pooling.rs`, `upsampling.rs`, `activations.rs`, `lora.rs`, `vae/` | `nn`-style modules. `conv::Conv2d` is canonical. |
| `attention/` (`sdpa.rs`, `sdpa_legacy.rs`, `rope.rs`, `sliding_window_mask.rs`), `sdpa.rs`, `sdpa_legacy.rs`, `sage_attention.rs` | SDPA dispatchers, RoPE, masks. **No Flash Attention port here.** |
| `ops/` | High-level: `attn.rs`, `gemm.rs`, `gemm_bf16.rs`, `grouped_mm.rs`, `moe_routing.rs`, `nucleus_moe.rs`, `token_choice_routing.rs`, `fused_inference.rs`, `fused_gated_scatter_add.rs`, `reduce.rs`, `broadcast.rs`, `cast.rs`, `tile.rs`, `deinterleave.rs`, `elt.rs`, `grad_norm.rs`, `multi_tensor.rs`, `conv2d*.rs`, `cuda/`. |
| `cuda/` | Build-time `.cu` kernels (cuBLASLt, cuDNN SDPA, fused_*.cu, FP8 quant, grouped_mm, PyTorch flash shim, fused_linear3d, fused_rms_norm, fused_modulate, fused_norm_modulate, fused_residual_gate). FFI in `cuda/ffi.rs`. |
| `kernels/` | NVRTC kernels: `adaln.rs`, `geglu_kernels.cu`, `rope_kernels.cu`, `sdpa_kernels.cu`, `mul_bwd_bf16.cu`. |
| `cuda_kernels*.rs`, `cuda_ops*.rs`, `cuda_gradient_ops.rs`, `cuda_conv2d*.rs`, `kernel_launcher.rs` | NVRTC compile, dispatch, F32 (`CudaKernels`) + BF16 (`cuda_ops_bf16`) surfaces. |
| `bf16_*.rs`, `fp16.rs`, `mixed_precision.rs` | BF16/FP16 primitives. `bf16_ops` is the live inference op family. |
| `cudnn/` | cuDNN wrappers (feature `cudnn`). |
| `blas.rs`, `cuda/device_lt.rs` | cuBLAS / cuBLASLt entry + LT handle. |
| `cuda_alloc_pool.rs`, `memory_pool.rs`, `pinned*.rs`, `staging.rs`, `ring_alloc/`, `static_slab_v2.rs`, `external_memory.rs` | Device + pinned host alloc. `FLAME_ALLOC_POOL=0` disables pool (see Gotchas). |
| `activation_offload.rs`, `offload/` | `BlockOffloader` + activation offload for large-model weight streaming. |
| `adam.rs`, `adam8bit_kernel.rs`, `sgd/`, `int8_weight_only_qt_kernel.rs` | Optimizers + quantized state. |
| `loss.rs`, `samplers.rs`, `regularization.rs`, `rng/` | Loss, schedulers, weight decay, RNG. |
| `serialization.rs` | safetensors I/O. |
| `image_ops_nhwc.rs`, `pooling_impl.rs`, `fused_kernels.rs` | Vision/fused helpers. |
| `telemetry.rs`, `perf_telemetry.rs`, `logging.rs`, `env_flags.rs`, `debug_*.rs`, `strict.rs` | Counters, env flags, strict-mode guard, finite checks. |
| `ffi/`, `borrowed/`, `structured/`, `saved_ref.rs` | C FFI helpers, borrowed-weight views (feature `borrowed_weights`), structured ops, autograd saved-ref. |
| `bin/` | 13 dev binaries (`op_bench_flame`, `perf_test`, `test_backward`, etc.). Not shipped. |
| `tests/` | In-crate tests (separate from top-level `tests/`). |

Other root dirs: `cuda/`, `kernels/`, `benches/`, `benchmarks/`, `tests/`, `inference-test/`, `scripts/`, `third_party/`, `unused/`.

## 3. Active vs legacy autograd

| Module | Status | Notes |
|---|---|---|
| `autograd.rs` | **Live** — re-export at `lib.rs:237` | Global mutex tape, `TapeEntry { output_id, op, saved_tensors, saved_refs }`. Doc: `docs/FLAME_AUTOGRAD_INTERNALS.md`. |
| `autograd_v3.rs` | Compiled but unused as primary | `lib.rs:177` comment says "Primary" — stale; re-export is from `autograd`, not v3. Legacy. |
| `autograd_v2/`, `autograd_v4/` | Feature-gated (off by default) | Burn-style. Bridge tests at `tests/autograd_v2_*.rs`. |
| `autograd_simple.rs`, `autograd_engine.rs`, `autograd_ops.rs`, `autograd_ops_complete.rs`, `autograd_debug.rs` | Dead/superseded, compiled | Reference only; new work goes in `autograd.rs`. |

Don't add ops to v3/v4. Append `Op` variants to `autograd.rs` and a backward arm.

## 4. Where to start

- **Add CUDA kernel (BF16 inference)**: NVRTC source as `const &str` in `src/bf16_*.rs` or `src/cuda_kernels*.rs`; dispatch from `cuda_ops_bf16.rs`; log in `docs/FLAME_KERNELS.md`.
- **Add CUDA kernel (cuBLASLt/cuDNN/.cu)**: drop `.cu` in `src/cuda/`, list in `build.rs`, expose via `src/cuda/ffi.rs`, wrap under `ops/`.
- **Add op autograd**: variant to `autograd.rs::Op`, save via `record_op`, implement bwd arm. Internals: `docs/FLAME_AUTOGRAD_INTERNALS.md`.
- **Dead LoRA-B / grad debug**: `FLAME_ASSERT_GRAD_FLOW=1` + `diagnostics::assert_grad_flow(&grads, &names)` at step 1 (not step 0). See `docs/TRAINER_DIAGNOSTICS.md`.
- **Parity vs PyTorch**: `flame_core::parity::ParityHarness` (per-layer cos). Reference Python streams per-layer; never CPU-vs-GPU compare.
- **Find a symbol**: `docs/FLAME_INDEX.md` (`symbol → file:line`, marks ⭐ live / ⚠️ legacy).
- **Find a kernel**: `docs/FLAME_KERNELS.md`. **Perf work**: `docs/SPEED_CONTRACT.md`. Mirror PyTorch ATen (`aten/src/ATen/native/cuda`) — don't guess.

## 5. Gotchas (project-wide invariants)

- `Tensor::cat` non-contig output. Add `.contiguous()` before any conv/kernel that strides naively. Open: `project_flame_fixit_cat_contig`.
- `Tensor::narrow` returns zero-copy view that pins parent storage. Use `narrow_owning` for ownership without pin (TurboVAED OOM root cause).
- `Tensor::softmax` BF16 fast path is **last-dim only**; non-last falls to 5-step pipeline.
- Two silu/gelu impls. `Tensor::silu` calls `bf16_ops::`, NOT `cuda/cuda_ops.cu::fc_silu_bf16`. Edit the wrong one and the change is a no-op.
- `cuda_ops_bf16::group_norm_bf16` is **NHWC**, not NCHW.
- `conv1d` k=1 fast path broken on non-contig permuted inputs — hand-roll with explicit `.contiguous()`.
- NVRTC: no `<cfloat>` / `<float.h>` — use literal constants. `#pragma unroll` doesn't survive macro expansion — use `_Pragma("unroll")`.
- `cudarc` pinned at 0.11.9. **No cuSOLVER bindings** — GPU linalg (SVD/QR) needs 8-version upgrade or hand-written FFI.
- **No Flash Attention** here. cuDNN SDPA + WMMA reference only. Don't propose FA2/FA3.
- **head_dim ∉ {64,96,128} (no flash) → `sdpa.rs` materialized fallback**, which **auto-Q-tiles** the score matrix when it exceeds `FLAME_SDPA_MATERIALIZE_BUDGET_MB` (default 256 MiB). This is what lets d=256 (e.g. Ideogram-4) run at 1024² without OOMing on a 1.22 GB `[L,L]` F32 scores tensor (commit `7b4e281`). Sub-budget shapes keep the bit-identical single-shot path. **Do NOT route d>128 through the chunked `sdpa_stream_bf16`/`flame_sdpa_stub.cu`** — it silently produces garbage at d>128 (should hard-error; tracked). The Q-tiled materialized path is the correct memory-efficient route for large d.
- **No F32 fallbacks in inference paths.** Missing BF16 path → write one.
- `FLAME_ALLOC_POOL=0` required for any dataset-prep binary (pool leaks ~1 GB/sample).
- BF16 fused kernels must register autograd backward, or LoRA-B stays zero. Default-on `FLAME_ASSERT_GRAD_FLOW=1` when running trainers.
- Avoid in new code: `sdpa_legacy`, `cuda_tensor*`, `cuda_kernels::CudaKernels` (F32 training), standalone `attention/sdpa.rs::LayerNorm`.

## 6. Related docs

- `CLAUDE.md` — session entry rules, hard constraints, two build pipelines.
- `TENETS.md` — design principles (fix-the-primitive, API-makes-right-easy).
- `docs/FLAME_README.md` — LLM docs index. `FLAME_INDEX.md` (symbol → file:line, ⭐/⚠️). `FLAME_MODULES.md` (paragraph per module). `FLAME_KERNELS.md` (every kernel + launch + perf). `FLAME_CONVENTIONS.md` (naming, layout, dispatch). `FLAME_AUTOGRAD_INTERNALS.md` (tape, Op, bwd dispatch). `TRAINER_DIAGNOSTICS.md` (assert_grad_flow + ParityHarness). `SPEED_CONTRACT.md` (perf audit gates).
