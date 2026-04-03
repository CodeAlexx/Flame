#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tile_bc_to_bhwc_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B,
    int H,
    int W,
    int C
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)B * H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int b = idx / (C * H * W);

    dst[idx] = src[b * C + c];
}

extern "C" void launch_tile_bc_to_bhwc_f32(
    const float* src,
    float* dst,
    int B,
    int H,
    int W,
    int C,
    cudaStream_t stream
) {
    long long total = (long long)B * H * W * C;
    if (total == 0) {
        return;
    }
    const int block = 256;
    const int grid = (int)((total + block - 1) / block);
    tile_bc_to_bhwc_kernel<<<grid, block, 0, stream>>>(src, dst, B, H, W, C);
}
