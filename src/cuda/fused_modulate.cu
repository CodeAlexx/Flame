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

// B.3 — fused modulate-split-apply.
//
// Replaces `narrow(mod, shift_idx) + narrow(mod, shift_idx+1) +
// fused_modulate_bf16` (2-3 slice_copy kernels + 1 modulate) with a single
// kernel that reads shift/scale from the combined [B, mod_dim, C] tensor
// via index lookup inside the loop. Every DiT block's modulate_pre fires
// this pair of narrows (shift_msa/scale_msa, shift_mlp/scale_mlp, etc.).
//
// x:          [B, N, C]
// modulation: [B, mod_dim, C]
// output:     [B, N, C]
// shift   = modulation[:, shift_idx,     :]  (broadcast per-sample across N)
// scale   = modulation[:, shift_idx + 1, :]  (broadcast per-sample across N)
// out[b, n, c] = x[b, n, c] * (1 + scale[b, c]) + shift[b, c]
__global__ void fused_modulate_split_apply_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ modulation,
    __nv_bfloat16* __restrict__ output,
    const int B,
    const int N,
    const int C,
    const int mod_dim,
    const int shift_idx
) {
    const size_t total = (size_t)B * N * C;
    const size_t idx0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t BN = (size_t)B * N;
    for (size_t i = idx0; i < total; i += stride) {
        const int c = (int)(i % (size_t)C);
        const size_t bn = i / (size_t)C;
        const int b = (int)(bn / (size_t)N);
        const size_t shift_off = (size_t)b * (size_t)mod_dim * (size_t)C
                               + (size_t)shift_idx * (size_t)C + (size_t)c;
        const size_t scale_off = shift_off + (size_t)C;
        float x_v = __bfloat162float(x[i]);
        float shift_v = __bfloat162float(modulation[shift_off]);
        float scale_v = __bfloat162float(modulation[scale_off]);
        output[i] = __float2bfloat16(x_v * (1.0f + scale_v) + shift_v);
        (void)BN;
    }
}

int flame_fused_modulate_split_apply_bf16(
    const void* x,
    const void* modulation,
    void* output,
    int B, int N, int C,
    int mod_dim,
    int shift_idx,
    void* stream
) {
    if (B <= 0 || N <= 0 || C <= 0 || mod_dim <= 0) return 0;
    if (shift_idx < 0 || shift_idx + 1 >= mod_dim) return -1; // need room for scale

    const int block_size = 256;
    const size_t total = (size_t)B * N * C;
    int n_blocks = (int)((total + block_size - 1) / block_size);
    if (n_blocks > 65535) n_blocks = 65535;

    fused_modulate_split_apply_bf16_kernel<<<n_blocks, block_size, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)modulation,
        (__nv_bfloat16*)output,
        B, N, C, mod_dim, shift_idx
    );
    return (int)cudaGetLastError();
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
