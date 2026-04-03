#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

__device__ inline float gelu(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    float x3 = x * x * x;
    float inner = c * (x + 0.044715f * x3);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

__global__ void geglu_kernel(
    const float* __restrict__ gated,
    const float* __restrict__ value,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float g = gelu(gated[idx]);
    out[idx] = g * value[idx];
}

} // namespace

extern "C" int flame_geglu_pointwise_fp32(
    const float* gated,
    const float* value,
    float* out,
    int n,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 0) {
        geglu_kernel<<<grid, block, 0, stream>>>(gated, value, out, n);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
