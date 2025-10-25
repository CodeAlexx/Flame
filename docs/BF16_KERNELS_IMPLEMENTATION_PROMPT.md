# Codex Execution Prompt — Minimal BF16 CUDA Kernels

## Goal
Implement BF16 CUDA kernels (cuBLASLt GEMM, NHWC Conv2d, Streaming SDPA) and wire them to `flame-core/cuda/cuda_ops.h` so Rust runs BF16 I/O, FP32 compute, NHWC layout, stream-aware, with no FP32 escapes.

## Files to add
```
flame-core/cuda/
  cuda_ops_common.cu
  elt_bf16.cu
  norm_bf16.cu
  gemm_bf16_cublaslt.cu
  conv2d_nhwc_bf16.cu
  sdpa_stream_bf16.cu
```

## Rules
- I/O BF16 (u16); masks uint8 0/1.
- FP32 compute, BF16 store.
- NHWC layout; reductions over C.
- Stream aware; single workspace arena, no per-call cudaMalloc/free.
- Degenerate dims: write zeros, return OK.
- Errors: FC_ERR_* only.

## Implementation
1. `cuda_ops_common.cu`: `fc_ws_ensure_capacity` grows arena via cudaMallocAsync (fallback cudaMalloc) aligning to 1 MiB.
2. `elt_bf16.cu`: implement relu/gelu/silu/axpby.
3. `norm_bf16.cu`: implement layer/group/RMS norm in FP32.
4. `gemm_bf16_cublaslt.cu`: call cuBLASLt bf16 gemm with FP32 accum and optional bias.
5. `conv2d_nhwc_bf16.cu`: im2col NHWC + GEMM + bias add.
6. `sdpa_stream_bf16.cu`: chunked Q tiles, FP32 matmul, mask, softmax, matmul V.

## Build
- Compile with nvcc `-O3 --use_fast_math --std=c++17 -Xcompiler -fPIC` for sm80/86/89/90.
- Link: cudart, cublas, cublasLt.

## Tests
- GEMM [512x1024]@[1024x768] + bias.
- Conv2d NHWC 3x3 stride1 pad1.
- SDPA B=1,H=4,Q=128,K=128,Dh=64 chunk=64.
- Degenerate and OOM cases.

