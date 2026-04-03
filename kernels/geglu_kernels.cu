// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float gelu(float x) {
    const float kAlpha = 0.044715f;
    const float kSqrt = sqrtf(2.0f / 3.14159265358979323846f);
    float inner = kSqrt * (x + kAlpha * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void geglu_pointwise_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ value,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float g = gate[idx];
    float v = value[idx];
    out[idx] = gelu(g) * v;
}

} // namespace

extern "C" int flame_geglu_pointwise_fp32(
    const float* gate,
    const float* value,
    float* out,
    int n,
    void* stream_ptr)
{
    if (n <= 0) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    geglu_pointwise_kernel<<<blocks, threads, 0, stream>>>(gate, value, out, n);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
