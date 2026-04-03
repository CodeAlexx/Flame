#include "flame_bf16_utils.cuh"

namespace {

__global__ void flame_k_zero_bf16(__nv_bfloat16* dst, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = __float2bfloat16(0.0f);
  }
}

__global__ void flame_k_copy_bf16(__nv_bfloat16* dst,
                                  const __nv_bfloat16* src,
                                  size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = src[idx];
  }
}

inline dim3 make_grid(size_t elems, int block) {
  size_t grid = (elems + block - 1) / block;
  return dim3(static_cast<unsigned int>(grid), 1, 1);
}

}  // namespace

FlameCudaStatus flame_bf16_zero_async_impl(__nv_bfloat16* dst,
                                           size_t elems,
                                           cudaStream_t stream) {
  if (dst == nullptr || elems == 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  const int block = 256;
  flame_k_zero_bf16<<<make_grid(elems, block), block, 0, stream>>>(dst, elems);
  cudaError_t st = cudaGetLastError();
  return st == cudaSuccess ? FLAME_CUDA_OK : FLAME_CUDA_ERR_CUDA;
}

FlameCudaStatus flame_bf16_copy_async_impl(__nv_bfloat16* dst,
                                           const __nv_bfloat16* src,
                                           size_t elems,
                                           cudaStream_t stream) {
  if (dst == nullptr || src == nullptr || elems == 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  const int block = 256;
  flame_k_copy_bf16<<<make_grid(elems, block), block, 0, stream>>>(dst, src, elems);
  cudaError_t st = cudaGetLastError();
  return st == cudaSuccess ? FLAME_CUDA_OK : FLAME_CUDA_ERR_CUDA;
}
