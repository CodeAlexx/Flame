#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void add_same_shape_kernel(const T* __restrict__ a,
                                      const T* __restrict__ b,
                                      T* __restrict__ out,
                                      long long n)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" void launch_add_f32(
    const float* a,
    const float* b,
    float* out,
    long long n,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    add_same_shape_kernel<float><<<grid, block, 0, stream>>>(a, b, out, n);
}

extern "C" void launch_add_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    long long n,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    add_same_shape_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(a, b, out, n);
}
