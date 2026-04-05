// fused_modulate.cu
// Fused adaptive layer norm modulation: out = x * (1 + scale) + shift
// Replaces 4 kernel launches with 1.
//
// All tensors are BF16. Arithmetic in FP32 for precision.
//
// x:     [B, N, C]  — normalized hidden states
// scale: [B, N, C]  — per-token scale (from timestep embedding)
// shift: [B, N, C]  — per-token shift (from timestep embedding)
// out:   [B, N, C]
//
// Simple grid-stride loop. Each thread handles multiple elements.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void fused_modulate_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ output,
    const size_t n_elements
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n_elements; i += stride) {
        float x_val = __bfloat162float(x[i]);
        float s_val = __bfloat162float(scale[i]);
        float h_val = __bfloat162float(shift[i]);
        float result = x_val * (1.0f + s_val) + h_val;
        output[i] = __float2bfloat16(result);
    }
}

// Vectorized version: process 2 BF16 values at once via bfloat162
__global__ void fused_modulate_bf16_vec2(
    const __nv_bfloat162* __restrict__ x,
    const __nv_bfloat162* __restrict__ scale,
    const __nv_bfloat162* __restrict__ shift,
    __nv_bfloat162* __restrict__ output,
    const size_t n_pairs  // n_elements / 2
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n_pairs; i += stride) {
        __nv_bfloat162 x_val = x[i];
        __nv_bfloat162 s_val = scale[i];
        __nv_bfloat162 h_val = shift[i];

        float2 xf = __bfloat1622float2(x_val);
        float2 sf = __bfloat1622float2(s_val);
        float2 hf = __bfloat1622float2(h_val);

        float2 result;
        result.x = xf.x * (1.0f + sf.x) + hf.x;
        result.y = xf.y * (1.0f + sf.y) + hf.y;

        output[i] = __float22bfloat162_rn(result);
    }
}

// Entry point for flame-core FFI
//
// All pointers: BF16 data, same shape [total n_elements]
// Uses vec2 path when n_elements is even (almost always true for transformer dims)
int flame_fused_modulate_bf16(
    const void* x,
    const void* scale,
    const void* shift,
    void* output,
    size_t n_elements,
    void* stream
) {
    if (n_elements == 0) return 0;

    const int block_size = 256;

    if (n_elements % 2 == 0) {
        size_t n_pairs = n_elements / 2;
        int n_blocks = (int)((n_pairs + block_size - 1) / block_size);
        if (n_blocks > 65535) n_blocks = 65535;

        fused_modulate_bf16_vec2<<<n_blocks, block_size, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat162*)x,
            (const __nv_bfloat162*)scale,
            (const __nv_bfloat162*)shift,
            (__nv_bfloat162*)output,
            n_pairs
        );
    } else {
        int n_blocks = (int)((n_elements + block_size - 1) / block_size);
        if (n_blocks > 65535) n_blocks = 65535;

        fused_modulate_bf16<<<n_blocks, block_size, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)x,
            (const __nv_bfloat16*)scale,
            (const __nv_bfloat16*)shift,
            (__nv_bfloat16*)output,
            n_elements
        );
    }
    return cudaGetLastError();
}

} // extern "C"
