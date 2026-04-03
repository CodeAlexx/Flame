#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

void layernorm_affine_bf16_nhwc_forward(
    const void* x_bf16,
    void* y_bf16,
    const void* gamma_bf16,
    const void* beta_bf16,
    int B, int H, int W, int C,
    float eps,
    cudaStream_t stream);

void adaln_modulate_bf16_nhwc_forward(
    const void* x_bf16,
    void* y_bf16,
    const void* gamma_bf16,
    const void* beta_bf16,
    const void* mod_scale_bf16,
    const void* mod_shift_bf16,
    int B, int H, int W, int C,
    float eps,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
