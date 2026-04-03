#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stddef.h>

#include "flame_cuda_status.h"

#ifdef __cplusplus
extern "C" {
#endif

FlameCudaStatus flame_bf16_zero_async_impl(__nv_bfloat16* dst,
                                           size_t elems,
                                           cudaStream_t stream);
FlameCudaStatus flame_bf16_copy_async_impl(__nv_bfloat16* dst,
                                           const __nv_bfloat16* src,
                                           size_t elems,
                                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif
