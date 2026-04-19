#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu_tanh_derivative(float x) {
    const float c0 = 0.7978845608028654f; // sqrt(2/pi)
    const float c1 = 0.044715f;
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float arg = c0 * (x + c1 * x3);
    const float t = tanhf(arg);
    const float sech2 = 1.0f - t * t;
    return 0.5f * (1.0f + t)
         + 0.5f * x * sech2 * c0 * (1.0f + 3.0f * c1 * x2);
}

__global__ void gelu_backward_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ grad_in,
    int64_t n)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = __bfloat162float(grad_out[idx]);
    const float x = __bfloat162float(input[idx]);
    grad_in[idx] = __float2bfloat16_rn(g * gelu_tanh_derivative(x));
}

__global__ void gelu_backward_f32_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    int64_t n)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = grad_out[idx];
    const float x = input[idx];
    grad_in[idx] = g * gelu_tanh_derivative(x);
}

extern "C" int flame_gelu_backward_bf16(
    const void* grad_out,
    const void* input,
    void* grad_in,
    int64_t n,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    gelu_backward_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)grad_out,
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)grad_in,
        n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_gelu_backward_f32(
    const void* grad_out,
    const void* input,
    void* grad_in,
    int64_t n,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    gelu_backward_f32_kernel<<<grid, block, 0, stream>>>(
        (const float*)grad_out,
        (const float*)input,
        (float*)grad_in,
        n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
