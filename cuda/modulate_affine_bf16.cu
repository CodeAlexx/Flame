#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHiddenTile = 64;
constexpr int kTokenTile = 4;

__device__ inline __nv_bfloat16 bf16_from_float(float v) {
#if __CUDA_ARCH__ >= 800
    return __float2bfloat16(v);
#else
    // Fallback conversion for older architectures (should not be hit for sm80+)
    unsigned int tmp = __float_as_uint(v);
    return reinterpret_cast<__nv_bfloat16&>(tmp);
#endif
}

__device__ inline float bf16_to_float(__nv_bfloat16 v) {
#if __CUDA_ARCH__ >= 800
    return __bfloat162float(v);
#else
    // Fallback path (should not be hit for sm80+)
    unsigned int tmp = reinterpret_cast<unsigned int&>(v);
    return __uint_as_float(tmp);
#endif
}

__global__ void modulate_affine_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ shift,
    const __nv_bfloat16* __restrict__ scale,
    int tokens,
    int hidden)
{
    int b = blockIdx.z;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= tokens || h >= hidden) {
        return;
    }

    const int hidden_stride = hidden;
    const int token_stride = tokens * hidden;
    int dst_index = b * token_stride + t * hidden_stride + h;
    int batch_hidden_index = b * hidden + h;

    float v = bf16_to_float(dst[dst_index]);
    float s = bf16_to_float(scale[batch_hidden_index]);
    float sh = bf16_to_float(shift[batch_hidden_index]);

    dst[dst_index] = bf16_from_float(fmaf(v, s, sh));
}

} // namespace

extern "C" void launch_modulate_affine_bf16(
    void* dst,
    const void* shift,
    const void* scale,
    int batch,
    int tokens,
    int hidden,
    cudaStream_t stream)
{
    if (batch <= 0 || tokens <= 0 || hidden <= 0) {
        return;
    }

    dim3 block(std::min(hidden, kHiddenTile), std::min(tokens, kTokenTile), 1);
    dim3 grid(
        (hidden + block.x - 1) / block.x,
        (tokens + block.y - 1) / block.y,
        batch);

    modulate_affine_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dst),
        static_cast<const __nv_bfloat16*>(shift),
        static_cast<const __nv_bfloat16*>(scale),
        tokens,
        hidden);
}

