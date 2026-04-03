#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHiddenTile = 64;
constexpr int kTokenTile = 4;

__device__ inline __nv_bfloat16 bf16_from_float(float v) {
#if __CUDA_ARCH__ >= 800
    return __float2bfloat16(v);
#else
    return reinterpret_cast<const __nv_bfloat16&>(v);  // fallback (should not trigger on sm80+)
#endif
}

__device__ inline float bf16_to_float(__nv_bfloat16 v) {
#if __CUDA_ARCH__ >= 800
    return __bfloat162float(v);
#else
    return reinterpret_cast<const float&>(v);  // fallback (should not trigger on sm80+)
#endif
}

__global__ void gate_mul_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ gate,
    int tokens,
    int hidden) {
    int b = blockIdx.z;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= tokens || h >= hidden) {
        return;
    }

    int token_stride = tokens * hidden;
    int dst_index = b * token_stride + t * hidden + h;
    int gate_index = b * hidden + h;

    float v = bf16_to_float(dst[dst_index]);
    float g = bf16_to_float(gate[gate_index]);
    dst[dst_index] = bf16_from_float(v * g);
}

}  // namespace

extern "C" void launch_gate_mul_bf16(
    void* dst,
    const void* gate,
    int batch,
    int tokens,
    int hidden,
    cudaStream_t stream) {
    if (batch <= 0 || tokens <= 0 || hidden <= 0) {
        return;
    }

    dim3 block(std::min(hidden, kHiddenTile), std::min(tokens, kTokenTile), 1);
    dim3 grid(
        (hidden + block.x - 1) / block.x,
        (tokens + block.y - 1) / block.y,
        batch);

    gate_mul_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dst),
        static_cast<const __nv_bfloat16*>(gate),
        tokens,
        hidden);
}

