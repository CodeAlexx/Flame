#include "flame_nhwc_adapters.cuh"

namespace {

template <typename T>
__global__ void flame_k_nhwc_to_nchw(const T* __restrict__ in,
                                     T* __restrict__ out,
                                     int N,
                                     int H,
                                     int W,
                                     int C) {
  int n = blockIdx.z;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N || h >= H || w >= W) {
    return;
  }
  int base = ((n * H + h) * W + w) * C;
  for (int c = 0; c < C; ++c) {
    int idx_nchw = ((n * C + c) * H + h) * W + w;
    out[idx_nchw] = in[base + c];
  }
}

template <typename T>
__global__ void flame_k_nchw_to_nhwc(const T* __restrict__ in,
                                     T* __restrict__ out,
                                     int N,
                                     int C,
                                     int H,
                                     int W) {
  int n = blockIdx.z;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N || h >= H || w >= W) {
    return;
  }
  for (int c = 0; c < C; ++c) {
    int idx_nchw = ((n * C + c) * H + h) * W + w;
    int idx_nhwc = ((n * H + h) * W + w) * C + c;
    out[idx_nhwc] = in[idx_nchw];
  }
}

inline void flame_launch_grid(int N, int H, int W, dim3* grid, dim3* block) {
  const int tile = 16;
  *block = dim3(tile, tile, 1);
  unsigned int gx = static_cast<unsigned int>((W + tile - 1) / tile);
  unsigned int gy = static_cast<unsigned int>((H + tile - 1) / tile);
  *grid = dim3(gx, gy, static_cast<unsigned int>(N));
}

}  // namespace

FlameCudaStatus flame_nhwc_to_nchw_f32_impl(const float* in,
                                            float* out,
                                            int N,
                                            int H,
                                            int W,
                                            int C,
                                            cudaStream_t stream) {
  if (in == nullptr || out == nullptr || N <= 0 || H <= 0 || W <= 0 || C <= 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  dim3 grid, block;
  flame_launch_grid(N, H, W, &grid, &block);
  flame_k_nhwc_to_nchw<float>
      <<<grid, block, 0, stream>>>(in, out, N, H, W, C);
  cudaError_t st = cudaGetLastError();
  return st == cudaSuccess ? FLAME_CUDA_OK : FLAME_CUDA_ERR_CUDA;
}

FlameCudaStatus flame_nchw_to_nhwc_f32_impl(const float* in,
                                            float* out,
                                            int N,
                                            int C,
                                            int H,
                                            int W,
                                            cudaStream_t stream) {
  if (in == nullptr || out == nullptr || N <= 0 || H <= 0 || W <= 0 || C <= 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  dim3 grid, block;
  flame_launch_grid(N, H, W, &grid, &block);
  flame_k_nchw_to_nhwc<float>
      <<<grid, block, 0, stream>>>(in, out, N, C, H, W);
  cudaError_t st = cudaGetLastError();
  return st == cudaSuccess ? FLAME_CUDA_OK : FLAME_CUDA_ERR_CUDA;
}

FlameCudaStatus flame_nhwc_to_nchw_bf16_impl(const __nv_bfloat16* in,
                                             __nv_bfloat16* out,
                                             int N,
                                             int H,
                                             int W,
                                             int C,
                                             cudaStream_t stream) {
  if (in == nullptr || out == nullptr || N <= 0 || H <= 0 || W <= 0 || C <= 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  dim3 grid, block;
  flame_launch_grid(N, H, W, &grid, &block);
  flame_k_nhwc_to_nchw<__nv_bfloat16>
      <<<grid, block, 0, stream>>>(in, out, N, H, W, C);
  cudaError_t st = cudaGetLastError();
  return st == cudaSuccess ? FLAME_CUDA_OK : FLAME_CUDA_ERR_CUDA;
}

FlameCudaStatus flame_nchw_to_nhwc_bf16_impl(const __nv_bfloat16* in,
                                             __nv_bfloat16* out,
                                             int N,
                                             int C,
                                             int H,
                                             int W,
                                             cudaStream_t stream) {
  if (in == nullptr || out == nullptr || N <= 0 || H <= 0 || W <= 0 || C <= 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  dim3 grid, block;
  flame_launch_grid(N, H, W, &grid, &block);
  flame_k_nchw_to_nhwc<__nv_bfloat16>
      <<<grid, block, 0, stream>>>(in, out, N, C, H, W);
  cudaError_t st = cudaGetLastError();
  return st == cudaSuccess ? FLAME_CUDA_OK : FLAME_CUDA_ERR_CUDA;
}
