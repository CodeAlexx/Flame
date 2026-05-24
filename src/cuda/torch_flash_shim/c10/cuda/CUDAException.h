#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define C10_CUDA_CHECK(EXPR)                                                    \
    do {                                                                        \
        cudaError_t flame_cuda_check_err = (EXPR);                              \
        if (flame_cuda_check_err != cudaSuccess) {                              \
            fprintf(stderr,                                                     \
                    "CUDA error (%s:%d): %s\n",                                 \
                    __FILE__,                                                   \
                    __LINE__,                                                   \
                    cudaGetErrorString(flame_cuda_check_err));                  \
        }                                                                       \
    } while (0)

// The flame wrapper calls cudaGetLastError() after launching so it can return
// a C ABI status code instead of throwing through PyTorch's C10 machinery.
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {} while (0)
