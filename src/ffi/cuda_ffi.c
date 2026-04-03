#include "cuda_ffi.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../../cuda/include/flame_cuda_common.cuh"
#include "../../cuda/include/flame_bf16_utils.cuh"
#include "../../cuda/include/flame_nhwc_adapters.cuh"

FlameCudaStatus flame_conv2d_nhwc_bf16_impl(const __nv_bfloat16* x,
                                            const __nv_bfloat16* w,
                                            const __nv_bfloat16* bias,
                                            int N,
                                            int H,
                                            int W,
                                            int Cin,
                                            int Kh,
                                            int Kw,
                                            int stride_h,
                                            int stride_w,
                                            int pad_h,
                                            int pad_w,
                                            int dil_h,
                                            int dil_w,
                                            int Cout,
                                            int activation,
                                            int groups,
                                            __nv_bfloat16* y,
                                            void* workspace,
                                            size_t workspace_bytes,
                                            cudaStream_t stream);

FlameCudaStatus flame_sdpa_chunked_bf16_impl(const __nv_bfloat16* q,
                                             const __nv_bfloat16* k,
                                             const __nv_bfloat16* v,
                                             int B,
                                             int H,
                                             int Q,
                                             int K,
                                             int Dh,
                                             int Dv,
                                             float scale,
                                             int chunk,
                                             int causal,
                                             int mask_heads,
                                             const __nv_bfloat16* attn_mask,
                                             __nv_bfloat16* out,
                                             void* workspace,
                                             size_t workspace_bytes,
                                             cudaStream_t stream);

FlameCudaStatus flame_conv2d_autotune_get_stats_impl(FlameConv2dAutotuneStats* out);
FlameCudaStatus flame_conv2d_autotune_reset_stats_impl(void);
FlameCudaStatus flame_sdpa_autotune_get_stats_impl(FlameSdpaAutotuneStats* out);
FlameCudaStatus flame_sdpa_autotune_reset_stats_impl(void);

#ifdef __cplusplus
extern "C" {
#endif

FlameCudaStatus flame_arena_create(int device,
                                   void* stream,
                                   uint64_t capacity_bytes,
                                   FlameStreamArena** out) {
  return flame_arena_create_impl(device,
                                 (cudaStream_t)stream,
                                 (size_t)capacity_bytes,
                                 out);
}

FlameCudaStatus flame_arena_reset(FlameStreamArena* arena) {
  return flame_arena_reset_impl(arena);
}

FlameCudaStatus flame_arena_alloc(FlameStreamArena* arena,
                                  uint64_t bytes,
                                  uint64_t align,
                                  void** out_ptr) {
  return flame_arena_alloc_impl(arena, (size_t)bytes, (size_t)align, out_ptr);
}

FlameCudaStatus flame_arena_record_and_release(FlameStreamArena* arena) {
  return flame_arena_record_and_release_impl(arena);
}

FlameCudaStatus flame_arena_destroy(FlameStreamArena* arena) {
  return flame_arena_destroy_impl(arena);
}

FlameCudaStatus flame_h2d_async(void* dst_device,
                                const void* src_host,
                                uint64_t bytes,
                                void* stream) {
  return flame_h2d_async_impl(dst_device, src_host, (size_t)bytes, (cudaStream_t)stream);
}

FlameCudaStatus flame_d2h_async(void* dst_host,
                                const void* src_device,
                                uint64_t bytes,
                                void* stream) {
  return flame_d2h_async_impl(dst_host, src_device, (size_t)bytes, (cudaStream_t)stream);
}

FlameCudaStatus flame_d2d_async(void* dst_device,
                                const void* src_device,
                                uint64_t bytes,
                                void* stream) {
  return flame_d2d_async_impl(dst_device, src_device, (size_t)bytes, (cudaStream_t)stream);
}

FlameCudaStatus flame_bf16_zero_async(void* dst_bf16,
                                      uint64_t elems,
                                      void* stream) {
  return flame_bf16_zero_async_impl((__nv_bfloat16*)dst_bf16,
                                    (size_t)elems,
                                    (cudaStream_t)stream);
}

FlameCudaStatus flame_bf16_copy_async(void* dst_bf16,
                                      const void* src_bf16,
                                      uint64_t elems,
                                      void* stream) {
  return flame_bf16_copy_async_impl((__nv_bfloat16*)dst_bf16,
                                    (const __nv_bfloat16*)src_bf16,
                                    (size_t)elems,
                                    (cudaStream_t)stream);
}

FlameCudaStatus flame_nhwc_to_nchw_f32(const float* in,
                                       float* out,
                                       int N,
                                       int H,
                                       int W,
                                       int C,
                                       void* stream) {
  return flame_nhwc_to_nchw_f32_impl(in, out, N, H, W, C, (cudaStream_t)stream);
}

FlameCudaStatus flame_nchw_to_nhwc_f32(const float* in,
                                       float* out,
                                       int N,
                                       int C,
                                       int H,
                                       int W,
                                       void* stream) {
  return flame_nchw_to_nhwc_f32_impl(in, out, N, C, H, W, (cudaStream_t)stream);
}

FlameCudaStatus flame_nhwc_to_nchw_bf16(const void* in,
                                        void* out,
                                        int N,
                                        int H,
                                        int W,
                                        int C,
                                        void* stream) {
  return flame_nhwc_to_nchw_bf16_impl((const __nv_bfloat16*)in,
                                      (__nv_bfloat16*)out,
                                      N,
                                      H,
                                      W,
                                      C,
                                      (cudaStream_t)stream);
}

FlameCudaStatus flame_nchw_to_nhwc_bf16(const void* in,
                                        void* out,
                                        int N,
                                        int C,
                                        int H,
                                        int W,
                                        void* stream) {
  return flame_nchw_to_nhwc_bf16_impl((const __nv_bfloat16*)in,
                                      (__nv_bfloat16*)out,
                                      N,
                                      C,
                                      H,
                                      W,
                                      (cudaStream_t)stream);
}

FlameCudaStatus flame_conv2d_nhwc_bf16(const void* x,
                                       const void* w,
                                       const void* bias,
                                       int N,
                                       int H,
                                       int W,
                                       int Cin,
                                       int Kh,
                                       int Kw,
                                       int stride_h,
                                       int stride_w,
                                       int pad_h,
                                       int pad_w,
                                       int dil_h,
                                       int dil_w,
                                       int Cout,
                                       int activation,
                                       int groups,
                                       void* y,
                                       void* workspace,
                                       uint64_t workspace_bytes,
                                       void* stream) {
  return flame_conv2d_nhwc_bf16_impl((const __nv_bfloat16*)x,
                                     (const __nv_bfloat16*)w,
                                     (const __nv_bfloat16*)bias,
                                     N,
                                     H,
                                     W,
                                     Cin,
                                     Kh,
                                     Kw,
                                     stride_h,
                                     stride_w,
                                     pad_h,
                                     pad_w,
                                     dil_h,
                                     dil_w,
                                     Cout,
                                     activation,
                                     groups,
                                     (__nv_bfloat16*)y,
                                     workspace,
                                     (size_t)workspace_bytes,
                                     (cudaStream_t)stream);
}

FlameCudaStatus flame_sdpa_chunked_bf16(const void* q,
                                        const void* k,
                                        const void* v,
                                        int B,
                                        int H,
                                        int Q,
                                        int K,
                                        int Dh,
                                        int Dv,
                                        float scale,
                                        int chunk,
                                        int causal,
                                        int mask_heads,
                                        const void* mask,
                                        void* out,
                                        void* workspace,
                                        uint64_t workspace_bytes,
                                        void* stream) {
  return flame_sdpa_chunked_bf16_impl((const __nv_bfloat16*)q,
                                      (const __nv_bfloat16*)k,
                                      (const __nv_bfloat16*)v,
                                      B,
                                      H,
                                      Q,
                                      K,
                                      Dh,
                                      Dv,
                                      scale,
                                      chunk,
                                      causal,
                                      mask_heads,
                                      (const __nv_bfloat16*)mask,
                                      (__nv_bfloat16*)out,
                                      workspace,
                                      (size_t)workspace_bytes,
                                      (cudaStream_t)stream);
}

FlameCudaStatus flame_conv2d_autotune_get_stats(FlameConv2dAutotuneStats* out) {
  return flame_conv2d_autotune_get_stats_impl(out);
}

FlameCudaStatus flame_conv2d_autotune_reset_stats(void) {
  return flame_conv2d_autotune_reset_stats_impl();
}

FlameCudaStatus flame_sdpa_autotune_get_stats(FlameSdpaAutotuneStats* out) {
  return flame_sdpa_autotune_get_stats_impl(out);
}

FlameCudaStatus flame_sdpa_autotune_reset_stats(void) {
  return flame_sdpa_autotune_reset_stats_impl();
}

FlameCudaStatus flame_sdpa_autotune_flush_cache(void) {
  return flame_sdpa_autotune_flush_cache_impl();
}

#ifdef __cplusplus
}
#endif
