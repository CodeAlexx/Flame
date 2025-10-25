# Codex Execution Prompt — Minimal BF16 CUDA Kernels (cuBLASLt GEMM, NHWC Conv2d via im2col, Streaming SDPA)

## Build BF16 kernels fast (no over-engineering)
Implement the missing **BF16 CUDA kernels** and wire them to `flame-core/cuda/cuda_ops.h` so Rust runs **BF16 I/O, FP32 compute**, **NHWC** layout, **stream-aware**, with **no FP32 escapes**. Keep it minimal and shippable.

---

## Files to add

```
flame-core/cuda/
  cuda_ops_common.cu          # workspace arena + helpers
  elt_bf16.cu                 # relu/gelu/silu/axpby
  norm_bf16.cu                # layer/group/rms norm
  gemm_bf16_cublaslt.cu       # cuBLASLt bf16×bf16→fp32acc→bf16 (+bias)
  conv2d_nhwc_bf16.cu         # NHWC conv2d (im2col + GEMM)
  sdpa_stream_bf16.cu         # streaming SDPA (bf16) with chunk+workspace
```

*(use existing `flame-core/cuda/cuda_ops.h` ABI — do NOT change signatures)*

---

## Core rules (apply to every op)
- **I/O:** BF16 tensors (u16). Masks: uint8 (0/1).
- **Compute:** FP32 in registers/shared; **cast back to BF16** on final store.
- **Layout:** NHWC for images; last-dim reductions (LN/RMS) over **C**.
- **Streams:** every op takes `cudaStream_t`; no host sync.
- **Workspace:** single **per-(device,stream) arena**; **no per-call cudaMalloc/free**.
- **Degenerate:** if any reduced axis len == 0 → write zeros, `FC_OK`.
- **Errors:** `OK`, `INVALID_ARGUMENT`, `UNSUPPORTED`, `OOM`, `LAUNCH`. No retry loops in kernels.

---

## 1) Workspace Arena — `cuda_ops_common.cu`
- `fc_ws_ensure_capacity(fc_workspace_t* arena, size_t bytes, cudaStream_t stream)`:
  - If `arena->bytes >= bytes` → `OK`.
  - Else grow to `round_up(bytes, 1<<20)`; prefer `cudaMallocAsync`, fallback `cudaMalloc`.
  - On failure: `FC_ERR_OOM`. Do **not** free here.

---

## 2) Elementwise — `elt_bf16.cu`
Exports:
- `fc_relu_bf16`, `fc_gelu_bf16`, `fc_silu_bf16`, `fc_axpby_bf16`
Implementation:
- Grid-stride loop; vectorize when aligned; BF16 load/store; FP32 math; BF16 store.

---

## 3) Norms — `norm_bf16.cu`
Exports:
- `fc_layer_norm_bf16`, `fc_group_norm_bf16`, `fc_rms_norm_bf16`
Implementation:
- Inputs/outputs BF16; **stats in FP32** (shared reduction). LN/RMS reduce across C; GN splits C into `groups` (C%groups==0).
- Degenerate reduction → zeros + `OK`.

---

## 4) GEMM — `gemm_bf16_cublaslt.cu`
Export:
- `fc_gemm_bf16(x:[M,K], w:[K,N], bias:[N] or null, y:[M,N], stream)`
Implementation:
- Use **cuBLASLt**: BF16 inputs + FP32 accum; epilogue: bias add; output **BF16** if supported.
- If Lt outputs FP32 only → cast kernel to BF16 into `y` on same stream.
- No large workspace; set preference small/zero.

---

## 5) Conv2d NHWC — `conv2d_nhwc_bf16.cu`
Export:
- `fc_conv2d_bf16(x:[N,H,W,C], w:[KH,KW,IC,OC], bias:[OC] or null, strides(1/2), pads(0/1), dils(1), y:[N,H',W',OC], ws, stream)`
Implementation:
- **im2col + GEMM**: columns `[N*H'*W', KH*KW*IC]` (BF16) in arena → GEMM with `fc_gemm_bf16`.
- Bias add after GEMM (or via epilogue). Degenerate dims → `OK`.

---

## 6) Streaming SDPA — `sdpa_stream_bf16.cu`
Export:
- `fc_sdpa_stream_bf16(Q:[B,H,Q,Dh], K:[B,H,K,Dh], V:[B,H,K,Dh], mask:[B,1/H,Q,K] uint8 or null, cfg{heads,head_dim,chunk,scale}, ws, O, stream)`
Implementation:
- Tile Q by `chunk` rows. For each tile:
  - `S = Q_tile @ K^T` (FP32 accum) → apply mask (0→-inf), softmax (stable, FP32).
  - `O_tile = softmax * V` (FP32 accum) → store BF16.
- **Workspace** ≈ `B*H*chunk*K*sizeof(float)` (+ small vectors). If required > arena → `FC_ERR_OOM`.
- No per-call cudaMalloc; masks must be uint8.

---

## 7) Build
- Compile with nvcc: `-O3 --use_fast_math --std=c++17 -Xcompiler -fPIC` + `-gencode` for sm_80/sm_86/sm_89/sm_90.
- Link: `cudart`, `cublas`, `cublasLt`.

---

## 8) Minimal Tests
- GEMM: `[512,1024]@[1024,768]→[512,768]` + bias; BF16 out; FP32 ref tol 2e-3.
- Conv2d: `N=1,H=W=32,C=32`, `KH=KW=3`, `OC=64`, stride=1 pad=1; shape check + FP32 ref tol 2e-3.
- SDPA: `B=1,H=4,Q=128,K=128,Dh=64`, `chunk=64`; mask on/off; FP32 ref tol 3e-3.
- Degenerate/OOM cases hit correct early returns; no per-call allocs.

---

## Acceptance
- All `cuda_ops.h` symbols implemented/exported.
- Rust wrappers **do not** upcast to FP32 on public paths.
- SD-3.5/SDXL runners execute with **BF16 I/O**, **NHWC**, stable VRAM (no per-launch mallocs).

**Deliverable:** PR with `.cu` sources + build changes + unit test logs.
