# F32→BF16 Storage Audit & Kernelization — flame-core

## Mission
Refactor every **elementwise/broadcast** kernel first, then the remaining math stacks, so public APIs accept **BF16 storage** while computing in **FP32** internally. No host-side `to_dtype(DType::F32)` casts are allowed in hot paths; BF16 tensors must stay BF16 on device memory. When a dependency insists on FP32 buffers (legacy cuDNN/cuBLAS helpers), add a BF16-facing wrapper that upcasts inside the kernel and stores BF16 on exit.

## Batch Breakdown

### 1. Elementwise + Broadcast (current focus)
- Sweep `cuda_ops.rs`, `cuda_kernels.rs`, and `cuda_kernels_gpu.rs` for pre-launch casts (`to_dtype(DType::F32)`, `try_as_slice_f32`, `DevicePtr<f32>`).
- Replace them with BF16-friendly launches using the existing `launch_add_inplace_bf16`, `launch_broadcast_bf16`, etc., or author thin wrappers that perform BF16→FP32→BF16 within the CUDA kernel.
- Consolidate broadcasting logic by reusing `ops::broadcast::broadcast_to_impl` where possible to avoid duplicate kernels.
- Tests: extend `tests/bf16_broadcast.rs` and `tests/cuda_kernel_contracts.rs` to assert storage dtype stays BF16 and values match FP32 reference (`abs_tol=1e-3`).
- Update: Elementwise `add`/`mul`, scalar add/mul, and `where_mask` now run entirely on BF16 storage (no host FP32 widening); validated with `cargo check --features "cuda,bf16_u16"`.

### 2. Norms (LayerNorm/GroupNorm/AdaLN)
- Inputs: BF16; inside-kernel mean/variance/affine in FP32; outputs BF16.
- Update `kernels/adaln.rs`, `ops/layernorm.rs`, `ops/groupnorm.rs` so no host-side widening occurs.
- Tests: parity vs FP32 reference, NHWC layout assertions.
- Update: LayerNorm/GroupNorm/AdaLN now keep BF16 storage end-to-end using BF16 NVRTC kernels and BF16-only fallbacks; `cargo check --features "cuda,bf16_u16"` clean.

### 3. Softmax & Attention Helpers
- Accept BF16 Q/K/V; compute logits/exp/sums in FP32 registers; write BF16 results.
- Reuse log-sum-exp stabilization; add tests under `tests/sdpa_bf16_parity.rs` with BF16 storage checks.
- Update: Softmax/max/sum reductions run on BF16 storage (BF16 kernels + BF16 SDPA path); `SDPA` no longer widens tensors.

### 4. GEMM & Convolution
- Configure cuBLASLt and cuDNN descriptors with `CUDA_R_16BF` inputs/outputs and `CUBLAS_COMPUTE_32F`/`CUDNN_DATA_FLOAT` compute.
- Remove any host casts before GEMM/conv; isolate unavoidable FP32 requirements in wrappers that still surface BF16 storage.
- Tests: GEMM/conv parity (small shapes), dtype + NHWC contracts.
- Update: Matmul now calls BF16-aware cuBLASLt wrappers directly (no host cast) and cuDNN paths return BF16 tensors after FP32 compute; sum/max reductions honor BF16 storage for softmax.

### 5. Guardrails & Tooling
- Add `tests/common/assert_storage_bf16.rs` helper to catch regressions.
- Consider feature-gated panic in tests if `to_dtype(DType::F32)` appears in CUDA hot paths.

## CI Checklist (run per batch)
```bash
cargo check  --manifest-path flame-core/Cargo.toml
cargo test   --manifest-path flame-core/Cargo.toml -- --nocapture
cargo clippy --manifest-path flame-core/Cargo.toml -- -D warnings
```
Record findings and remaining blockers in `docs/eridiffusion_handoff_*` as you progress.

### Outstanding Runtime Gaps
- `flame-core/src/ops_ext.rs`: `mean_all_f32` now reuses the BF16-aware `sum_all` path and no longer widens tensors on the host.
- `flame-core/src/tensor.rs`: `index_select0` gathers rows via GPU slicing/cat, eliminating the CPU FP32 gather; inputs stay in their original dtype.
- `flame-core/src/ops/broadcast.rs`: CPU fallback allocates FP32 buffers; consider adding a BF16 kernel or removing the fallback entirely.
- Optimizer helpers (`gradient.rs`, `sgd/mod.rs`, `parameter.rs`) retain FP32 conversions for CPU bookkeeping—evaluate whether these can run in BF16 or should remain as host-only diagnostics.
