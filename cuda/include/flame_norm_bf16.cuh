#pragma once

#include "flame_cuda_status.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

