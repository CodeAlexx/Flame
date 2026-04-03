#pragma once

#include <stdint.h>

#include "../../cuda/include/flame_cuda_common.cuh"
#include "../../cuda/include/flame_cuda_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FlameStreamArena FlameStreamArena;

FlameCudaStatus flame_arena_create(int device,
                                   void* stream,
                                   uint64_t capacity_bytes,
                                   FlameStreamArena** out);
FlameCudaStatus flame_arena_reset(FlameStreamArena* arena);
FlameCudaStatus flame_arena_alloc(FlameStreamArena* arena,
                                  uint64_t bytes,
                                  uint64_t align,
                                  void** out_ptr);
FlameCudaStatus flame_arena_record_and_release(FlameStreamArena* arena);
FlameCudaStatus flame_arena_destroy(FlameStreamArena* arena);

FlameCudaStatus flame_h2d_async(void* dst_device,
                                const void* src_host,
                                uint64_t bytes,
                                void* stream);
FlameCudaStatus flame_d2h_async(void* dst_host,
                                const void* src_device,
                                uint64_t bytes,
                                void* stream);
FlameCudaStatus flame_d2d_async(void* dst_device,
                                const void* src_device,
                                uint64_t bytes,
                                void* stream);

FlameCudaStatus flame_bf16_zero_async(void* dst_bf16,
                                      uint64_t elems,
                                      void* stream);
FlameCudaStatus flame_bf16_copy_async(void* dst_bf16,
                                      const void* src_bf16,
                                      uint64_t elems,
                                      void* stream);

FlameCudaStatus flame_nhwc_to_nchw_f32(const float* in,
                                       float* out,
                                       int N,
                                       int H,
                                       int W,
                                       int C,
                                       void* stream);
FlameCudaStatus flame_nchw_to_nhwc_f32(const float* in,
                                       float* out,
                                       int N,
                                       int C,
                                       int H,
                                       int W,
                                       void* stream);
FlameCudaStatus flame_nhwc_to_nchw_bf16(const void* in,
                                        void* out,
                                        int N,
                                        int H,
                                        int W,
                                        int C,
                                        void* stream);
FlameCudaStatus flame_nchw_to_nhwc_bf16(const void* in,
                                        void* out,
                                        int N,
                                        int C,
                                        int H,
                                        int W,
                                        void* stream);

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
                                       void* stream);
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
                                        void* stream);

FlameCudaStatus flame_conv2d_autotune_get_stats(FlameConv2dAutotuneStats* out);
FlameCudaStatus flame_conv2d_autotune_reset_stats(void);
FlameCudaStatus flame_sdpa_autotune_get_stats(FlameSdpaAutotuneStats* out);
FlameCudaStatus flame_sdpa_autotune_reset_stats(void);
FlameCudaStatus flame_sdpa_autotune_flush_cache(void);

#ifdef __cplusplus
}
#endif
