#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void sigmoid_backward_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ grad_in,
    int64_t n)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = __bfloat162float(grad_out[idx]);
    const float x = __bfloat162float(input[idx]);
    const float s = sigmoidf_fast(x);
    grad_in[idx] = __float2bfloat16_rn(g * s * (1.0f - s));
}

__global__ void sigmoid_backward_f32_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    int64_t n)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = grad_out[idx];
    const float x = input[idx];
    const float s = sigmoidf_fast(x);
    grad_in[idx] = g * s * (1.0f - s);
}

extern "C" int flame_sigmoid_backward_bf16(
    const void* grad_out,
    const void* input,
    void* grad_in,
    int64_t n,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    sigmoid_backward_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)grad_out,
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)grad_in,
        n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_sigmoid_backward_f32(
    const void* grad_out,
    const void* input,
    void* grad_in,
    int64_t n,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    sigmoid_backward_f32_kernel<<<grid, block, 0, stream>>>(
        (const float*)grad_out,
        (const float*)input,
        (float*)grad_in,
        n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
