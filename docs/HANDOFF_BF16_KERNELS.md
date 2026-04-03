# BF16 CUDA Kernel Handoff — SD-3.5 Modernization

**Worktree**: `/home/alex/EriDiffusion/flame-core`
**Goal**: Finish the BF16/NHWC GPU runtime so SD‑3.5 & SDXL can run without FP32 fallbacks.

---
## Current Layout

```
cuda/
  cuda_ops_common.cu          # Workspace arena (fc_ws_ensure_capacity)
  elt_bf16.cu                 # ReLU/GELU/SiLU/axpby BF16 kernels (grid-stride)
  norm_bf16.cu                # LayerNorm/GroupNorm/RMSNorm kernels (FP32 stats)
  streaming_attn_bf16.cu      # Legacy streaming attention (needs retrofit)
  ... existing legacy CUDA sources ...

src/
  cuda_ops.rs                 # Legacy GPU ops (Still FP32 fallback)
  cuda_ops_ffi.rs             # FFI structs/helpers (FcTensorView/Ws arena)
  cuda_ops_bf16.rs            # BF16 wrappers (relu/gelu/silu/norms)
```

Docs:
- `docs/BF16_KERNELS_IMPLEMENTATION_PROMPT.md` — detailed spec for missing kernels.
- `docs/HANDOFF_BF16_KERNELS.md` (this file).

---
## Completed
- Shared workspace arena (`fc_ws_ensure_capacity`) with async/sync fallbacks.
- Elementwise BF16 kernels (ReLU/GELU/SiLU/axpby) with FP32 accumulate.
- LayerNorm/GroupNorm/RMSNorm kernels (`norm_bf16.cu`).
- cuBLASLt GEMM (single + batched) wired through `fc_gemm_bf16` and Rust wrappers.
- NHWC Conv2d via im2col + GEMM now routed through `flame_conv2d_nhwc_bf16` with stream-ordered arena workspace, automatic tiling when workspace exceeds cap, Rust callers allocating via `FlameStreamArena`, a depthwise fast-path (groups = Cin = Cout), fused bias + activation (ReLU/SiLU/GELU) in CUDA, and grouped convolutions handled natively in the GEMM loop.
- Streaming SDPA (`sdpa_stream_bf16.cu`) now supports BF16 masks + causal gating; `ops::attn`/Flux runtime call it instead of the legacy path.
- Rust surface cleaned up end-to-end (no FP32 fallbacks) and Flux runtime chunking honours mask + causal flags.
- GPU regression tests cover GEMM/Conv/SDPA (tolerance checks, causal + explicit mask cases).
- Build script enumerates all new CUDA sources (`cuda_ops_common.cu`, `elt_bf16.cu`, `norm_bf16.cu`, `gemm_bf16_cublaslt.cu`, `conv2d_nhwc_bf16.cu`, `sdpa_stream_bf16.cu`).

---
## Outstanding Work
1. **Conv2d Enhancements**
   - Micro-autotune for `(tile_h, tile_w, GEMM config)`; persist cache per device/shape.
   - Extend validation to cover dilation>1 and stride>1 combos; add perf telemetry (TFLOPs, ms, workspace bytes).
   - Consider additional fused epilogues (e.g., bias-only vs. activation combos) and expose activation toggles to higher-level APIs.
2. **End-to-End Validation**
   - Re-run SD‑3.5 / SDXL harnesses with the BF16 stack and compare against FP32 baselines (tol ≤ 2e-3).
   - Collect streaming-attention latency after masking changes (`ATTN_LOG_MS=1`) and watch for regressions.

---
## Quick Start for New Engineer
1. Checkout worktree: `/home/alex/EriDiffusion/flame-core`
2. Read spec: `docs/BF16_KERNELS_IMPLEMENTATION_PROMPT.md`
3. Implement missing kernels in `cuda/` (see Outstanding Work).
4. Extend build.rs with new `.cu` files when ready.
5. Update Rust wrappers (cuda_ops_bf16.rs / cuda_ops.rs).
6. Add unit tests; run `cargo check -p flame-core` and GPU smoke tests.

Environment:
- CUDA toolchain under `/usr/local/cuda` (NVCC 12.x).
- Build commands: `cargo check -p flame-core`, `cargo test -p flame-core --tests`.

Contacts / Notes:
- Streaming attention legacy kernel still exists (`streaming_attn_bf16.cu`); keep or replace.
- Workspace arena already handles growth & cleanup; reuse for GEMM/Conv/SDPA temps.
- Masks should arrive as BF16 0/1 tensors (we currently infer them from padding when absent).
