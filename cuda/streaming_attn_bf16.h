#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" cudaError_t streaming_attn_bf16_fp32_launch(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    int B, int H, int S, int Dh, int Dv,
    long qB, long qH, long qS, long qD,
    long kB, long kH, long kS, long kD,
    long vB, long vH, long vS, long vD,
    __nv_bfloat16* O,
    long oB, long oH, long oS, long oD,
    float scale, int chunk_size, int causal,
    const uint8_t* k_padding_mask,
    cudaStream_t stream);
