// sdpa_stream_bf16.h
// Minimal C API for the streaming BF16 SDPA path (FlashAttention-style accumulation).
// MIT License.
//
// This interface is intentionally C-compatible so Rust FFI can bind it easily.
// The implementation uses cuBLAS for GEMMs and custom CUDA kernels for row-wise
// softmax accumulation without materializing the full [B*H, Q, K] logits tensor.

#pragma once
#include <stdbool.h>
#include <stdint.h>

#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns true on success. On failure, returns false and writes a short reason
// into unsupported_reason (null-terminated). If unsupported_reason is NULL or
// reason_buflen==0, the message is dropped.
//
// Shapes follow: Q: [B,H,Q_len,d], K: [B,H,K_len,d], V: [B,H,K_len,d], O: [B,H,Q_len,d].
// d must be a multiple of 8 for tensor-core friendly math (BF16).
//
// Data types:
//  - Q,K,V,O are in BF16 (__nv_bfloat16) device pointers.
//  - All accumulations are FP32. Output is cast back to BF16.
//
// Mask:
//  - attn_mask may be NULL for "no mask".
//  - If provided, it must be a row-major [B,H,Q_len,K_len] BF16 array where values
//    >= 0.5f mark masked positions. Stride in elements given by mask_stride_ek between
//    successive K positions for a fixed (B,H,Q). Stride in rows (Q) is mask_stride_eq.
//    Stride between heads (H) is mask_stride_eh. Stride between batches (B) is
//    mask_stride_eb. You can pass zeros for *_stride_* to indicate contiguous [Q,K]
//    layout and contiguous heads/batches.
//
// Environment-equivalent knobs are exposed as args:
//  - head_tile: number of heads per streaming tile (e.g., 12)
//  - q_tile:    number of query positions per tile (e.g., 96)
//  - max_q_tile: upper bound for q_tile; if q_tile > max_q_tile => unsupported
//
// If stream is NULL, the default stream (0) is used.
bool sdpa_stream_bf16_launch(
    const void* dQ,               // __nv_bfloat16*
    const void* dK,               // __nv_bfloat16*
    const void* dV,               // __nv_bfloat16*
    void*       dO,               // __nv_bfloat16*
    int B, int H, int Q_len, int K_len, int d,
    float scale,                  // usually 1/sqrt(d)
    const __nv_bfloat16* attn_mask, // optional; interpreted as 0/1 BF16 mask
    int64_t mask_stride_ek,       // stride over K (columns), elements
    int64_t mask_stride_eq,       // stride over Q (rows),    elements
    int64_t mask_stride_eh,       // stride over H,           elements
    int64_t mask_stride_eb,       // stride over B,           elements
    int head_tile,                // e.g., 12
    int q_tile,                   // e.g., 96
    int max_q_tile,               // e.g., 96
    void* cuda_stream,            // cudaStream_t
    char* unsupported_reason,     // out message buffer
    int  reason_buflen            // buffer size
);

#ifdef __cplusplus
}
#endif
