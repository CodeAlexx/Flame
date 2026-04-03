// flame-core/cuda/permute0213.cu
// -----------------------------------------------------------------------------
// This CUDA kernel keeps Flux’ (N, heads, tokens, dim) -> (N, tokens, heads, dim)
// permutation entirely on the GPU.  The original Rust implementation fell back
// to `Tensor::to_vec()` which staged tensors on the CPU and assumed F32
// storage; that broke for BF16 tensors and violated the “GPU-only” Phase-4
// contract.  We provide both F32 and BF16 launch wrappers here so the Rust
// side can dispatch without ever touching host memory.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Generic implementation shared by the F32 / BF16 entry points.
template <typename Scalar>
__global__ void permute0213_kernel(
    const Scalar* __restrict__ x,
    Scalar* __restrict__ y,
    int N, int A, int B, int C)
{
    // Total number of elements; we walk the flattened index space and map each
    // element to its destination.  Using long long avoids overflow for large
    // tensors.
    const long long total = static_cast<long long>(N) * A * B * C;
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;

    // Decompose the flattened index back into (n, a, b, c).
    int c = static_cast<int>(idx % C);
    idx /= C;
    int b = static_cast<int>(idx % B);
    idx /= B;
    int a = static_cast<int>(idx % A);
    int n = static_cast<int>(idx / A);

    // Input offset: ((n * A + a) * B + b) * C + c
    const long long in_offset =
        ((((static_cast<long long>(n) * A) + a) * B) + b) * C + c;

    // Output offset corresponds to (n, b, a, c): ((n * B + b) * A + a) * C + c
    const long long out_offset =
        ((((static_cast<long long>(n) * B) + b) * A) + a) * C + c;

    y[out_offset] = x[in_offset];
}

// Launch helpers matching the narrow kernels style the crate already uses.
extern "C" void launch_permute0213_f32(
    const float* x,
    float* y,
    int N,
    int A,
    int B,
    int C,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * A * B * C;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    permute0213_kernel<float><<<grid, block, 0, stream>>>(x, y, N, A, B, C);
}

extern "C" void launch_permute0213_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int N,
    int A,
    int B,
    int C,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * A * B * C;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    permute0213_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(x, y, N, A, B, C);
}

// ----------------------------------------------------------------------------
// 3D variant: (N, A, B) -> (N, B, A)

template <typename Scalar>
__global__ void permute021_kernel(
    const Scalar* __restrict__ x,
    Scalar* __restrict__ y,
    int N, int A, int B)
{
    const long long total = static_cast<long long>(N) * A * B;
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;

    int b = static_cast<int>(idx % B);
    idx /= B;
    int a = static_cast<int>(idx % A);
    int n = static_cast<int>(idx / A);

    const long long in_offset = ((static_cast<long long>(n) * A) + a) * B + b;
    const long long out_offset = ((static_cast<long long>(n) * B) + b) * A + a;

    y[out_offset] = x[in_offset];
}

extern "C" void launch_permute021_f32(
    const float* x,
    float* y,
    int N,
    int A,
    int B,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * A * B;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    permute021_kernel<float><<<grid, block, 0, stream>>>(x, y, N, A, B);
}

extern "C" void launch_permute021_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int N,
    int A,
    int B,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * A * B;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    permute021_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(x, y, N, A, B);
}
