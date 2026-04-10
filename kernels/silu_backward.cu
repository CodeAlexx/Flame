#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Fused SiLU backward: grad_input = grad_output * (sig(x) + x*sig(x)*(1-sig(x)))
// where sig(x) = 1/(1+exp(-x))
// Single kernel replaces 7 separate GpuOps calls.

__global__ void silu_backward_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ grad_in,
    int64_t n)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __bfloat162float(grad_out[idx]);
    float x = __bfloat162float(input[idx]);

    // sig = 1/(1+exp(-x))
    float sig = 1.0f / (1.0f + expf(-x));
    // SiLU'(x) = sig + x*sig*(1-sig)
    float deriv = sig + x * sig * (1.0f - sig);

    grad_in[idx] = __float2bfloat16_rn(g * deriv);
}

__global__ void silu_backward_f32_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    int64_t n)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad_out[idx];
    float x = input[idx];

    float sig = 1.0f / (1.0f + expf(-x));
    float deriv = sig + x * sig * (1.0f - sig);

    grad_in[idx] = g * deriv;
}

extern "C" int flame_silu_backward_bf16(
    const void* grad_out,
    const void* input,
    void* grad_in,
    int64_t n,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_backward_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)grad_out,
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)grad_in,
        n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_silu_backward_f32(
    const void* grad_out,
    const void* input,
    void* grad_in,
    int64_t n,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_backward_f32_kernel<<<grid, block, 0, stream>>>(
        (const float*)grad_out,
        (const float*)input,
        (float*)grad_in,
        n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
