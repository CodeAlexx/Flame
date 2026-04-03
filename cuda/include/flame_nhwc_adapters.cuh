#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "flame_cuda_status.h"

#ifdef __cplusplus
extern "C" {
#endif

FlameCudaStatus flame_nhwc_to_nchw_f32_impl(const float* in,
                                            float* out,
                                            int N,
                                            int H,
                                            int W,
                                            int C,
                                            cudaStream_t stream);
FlameCudaStatus flame_nchw_to_nhwc_f32_impl(const float* in,
                                            float* out,
                                            int N,
                                            int C,
                                            int H,
                                            int W,
                                            cudaStream_t stream);
FlameCudaStatus flame_nhwc_to_nchw_bf16_impl(const __nv_bfloat16* in,
                                             __nv_bfloat16* out,
                                             int N,
                                             int H,
                                             int W,
                                             int C,
                                             cudaStream_t stream);
FlameCudaStatus flame_nchw_to_nhwc_bf16_impl(const __nv_bfloat16* in,
                                             __nv_bfloat16* out,
                                             int N,
                                             int C,
                                             int H,
                                             int W,
                                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif
