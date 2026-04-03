#ifndef FLAME_CORE_CUDA_OPS_H
#define FLAME_CORE_CUDA_OPS_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FC_OK = 0,
  FC_ERR_INVALID_ARGUMENT = 1,
  FC_ERR_LAUNCH = 2,
  FC_ERR_OOM = 3,
  FC_ERR_UNSUPPORTED = 4,
  FC_STATUS_LT_FALLBACK = 5
} fc_status_t;

typedef struct {
  void*   data;      // device pointer: BF16 -> u16*, mask -> uint8_t*
  int64_t dims[8];   // up to rank-8
  int64_t strides[8];
  int32_t rank;
} fc_tensor_view_t;

typedef struct {
  void*  ptr;        // device pointer
  size_t bytes;      // bytes allocated
} fc_workspace_t;

// Ensure arena has >= bytes; grow using cudaMallocAsync when available
fc_status_t fc_ws_ensure_capacity(fc_workspace_t* arena, size_t bytes, cudaStream_t stream);

// ======================== Elementwise ========================
fc_status_t fc_relu_bf16 (const fc_tensor_view_t* x, fc_tensor_view_t* y, cudaStream_t stream);
fc_status_t fc_gelu_bf16 (const fc_tensor_view_t* x, fc_tensor_view_t* y, cudaStream_t stream);
fc_status_t fc_silu_bf16 (const fc_tensor_view_t* x, fc_tensor_view_t* y, cudaStream_t stream);
fc_status_t fc_axpby_bf16(const fc_tensor_view_t* x, float a, fc_tensor_view_t* y, float b, cudaStream_t stream);

// ======================== Norms ========================
fc_status_t fc_layer_norm_bf16(const fc_tensor_view_t* x,    // [..., C]
                               const fc_tensor_view_t* gamma, // [C]
                               const fc_tensor_view_t* beta,  // [C] or null
                               int64_t norm_size,
                               float eps,
                               fc_tensor_view_t* y,
                               float* mean_out,
                               float* rstd_out,
                               cudaStream_t stream);

fc_status_t fc_layer_norm_backward_bf16(const __nv_bfloat16* x,
                                        const __nv_bfloat16* dy,
                                        const __nv_bfloat16* gamma,
                                        int64_t outer_size,
                                        int64_t norm_size,
                                        float eps,
                                        __nv_bfloat16* dx,
                                        float* dgamma,
                                        float* dbeta,
                                        cudaStream_t stream);

fc_status_t fc_group_norm_bf16(const fc_tensor_view_t* x,    // [N,H,W,C]
                               const fc_tensor_view_t* gamma, // [C]
                               const fc_tensor_view_t* beta,  // [C]
                               int32_t groups,
                               float eps,
                               fc_tensor_view_t* y,
                               float* mean_out,
                               float* var_out,
                               cudaStream_t stream);

fc_status_t fc_group_norm_backward_bf16(const __nv_bfloat16* x,
                                        const __nv_bfloat16* dy,
                                        const __nv_bfloat16* gamma,
                                        int64_t batch_size,
                                        int64_t channels,
                                        int64_t spatial_size,
                                        int32_t group_count,
                                        float eps,
                                        __nv_bfloat16* dx,
                                        float* dgamma,
                                        float* dbeta,
                                        cudaStream_t stream);

fc_status_t fc_rms_norm_bf16  (const fc_tensor_view_t* x,    // [..., C]
                               const fc_tensor_view_t* weight,// [C] or null
                               float eps,
                               fc_tensor_view_t* y,
                               cudaStream_t stream);

// ======================== GEMM / Linear ========================
// cuBLASLt bf16×bf16 -> fp32acc -> bf16 out (+ optional bias)
fc_status_t fc_gemm_bf16(const fc_tensor_view_t* x,     // [M,K] BF16
                         const fc_tensor_view_t* w,     // [K,N] BF16
                         const fc_tensor_view_t* bias,  // [N] BF16 or null
                         fc_tensor_view_t* y,           // [M,N] BF16
                         cudaStream_t stream);

// Optional batched GEMM
fc_status_t fc_batched_gemm_bf16(const fc_tensor_view_t* a,   // [B,M,K]
                                 const fc_tensor_view_t* b,   // [B,K,N]
                                 const fc_tensor_view_t* bias,// [B,N] or [N] or null
                                 fc_tensor_view_t* c,         // [B,M,N]
                                 cudaStream_t stream);

// ======================== BF16 Tensor Ops ========================
// View or copy slice along axis (step=1). Provides view when contiguous; otherwise copies into y.
fc_status_t fc_bf16_slice(const fc_tensor_view_t* x,
                          int32_t axis,
                          int64_t start,
                          int64_t len,
                          fc_tensor_view_t* y_view_or_buf,
                          cudaStream_t stream);

// Gather along axis using device indices.
fc_status_t fc_bf16_index_select(const fc_tensor_view_t* x,
                                 int32_t axis,
                                 const float* d_indices,
                                 int64_t nidx,
                                 fc_tensor_view_t* y,
                                 cudaStream_t stream);

// Broadcast expand BF16 tensor into dense output.
fc_status_t fc_bf16_broadcast(const fc_tensor_view_t* x,
                              const int64_t* out_dims,
                              const int64_t* out_strides,
                              int32_t rank,
                              fc_tensor_view_t* y,
                              cudaStream_t stream);

// Repeat BF16 tensor along a single axis producing dense output.
fc_status_t fc_bf16_repeat_axis(const fc_tensor_view_t* x,
                                int32_t axis,
                                int64_t repeats,
                                fc_tensor_view_t* y,
                                cudaStream_t stream);
fc_status_t fc_bf16_repeat_nd(const fc_tensor_view_t* x,
                              const int64_t* repeats,
                              int32_t rank,
                              fc_tensor_view_t* y,
                              cudaStream_t stream);

// Device-to-device memcpy helper (BF16 aware wrappers may call this).
fc_status_t fc_bf16_memcpy_async(void* dst,
                                 const void* src,
                                 size_t bytes,
                                 cudaStream_t stream);

// ======================== Conv2d (NHWC) ========================
fc_status_t fc_conv2d_bf16(const fc_tensor_view_t* x,     // [N,H,W,C]
                           const fc_tensor_view_t* w,     // [KH,KW,IC,OC]
                           const fc_tensor_view_t* bias,  // [OC] or null
                           int32_t stride_h, int32_t stride_w,
                           int32_t pad_h,    int32_t pad_w,
                           int32_t dil_h,    int32_t dil_w,
                           fc_tensor_view_t* y,           // [N,H',W',OC]
                           fc_workspace_t* ws,
                           cudaStream_t stream);

// ======================== Attention (Streaming SDPA) ========================
typedef struct {
  int32_t heads;       // H
  int32_t head_dim;    // Dh
  int32_t chunk;       // tile rows of Q per block
  float   scale;       // 1/sqrt(Dh) (optional if fused)
  int32_t causal;      // 1 if causal masking enabled
} fc_sdpa_cfg_t;

// mask: [B,1/H,Q,K] (uint8 0/1) or null
fc_status_t fc_sdpa_stream_bf16(const fc_tensor_view_t* Q,   // [B,H,Q,Dh]
                                const fc_tensor_view_t* K,   // [B,H,K,Dh]
                                const fc_tensor_view_t* V,   // [B,H,K,Dh]
                                const fc_tensor_view_t* mask,// optional uint8
                                const fc_sdpa_cfg_t* cfg,
                                fc_workspace_t* ws,
                                fc_tensor_view_t* O,         // [B,H,Q,Dh]
                                cudaStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FLAME_CORE_CUDA_OPS_H
