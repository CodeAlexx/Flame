#include "flame_cuda_common.cuh"

#include <cuda.h>

namespace {

inline size_t round_up(size_t value, size_t align) {
  if (align == 0) {
    return value;
  }
  size_t mask = align - 1;
  return (value + mask) & ~mask;
}

}  // namespace

FlameCudaStatus flame_arena_create_impl(int device,
                                        cudaStream_t stream,
                                        size_t capacity_bytes,
                                        FlameStreamArena** out) {
  if (out == nullptr) {
    return FLAME_CUDA_ERR_INVALID;
  }
  *out = nullptr;
  FLAME_CUDA_TRY(cudaSetDevice(device));

  FlameStreamArena* arena = new FlameStreamArena();
  arena->device = device;
  arena->stream = stream;
  arena->capacity = capacity_bytes;
  arena->offset = 0;
  arena->base = nullptr;
  arena->owns_base = capacity_bytes > 0;

  if (arena->owns_base) {
    if (stream == nullptr) {
      FLAME_CUDA_TRY(cudaMalloc(&arena->base, capacity_bytes));
    } else {
      FLAME_CUDA_TRY(cudaMallocAsync(&arena->base, capacity_bytes, stream));
    }
  }

  FLAME_CUDA_TRY(cudaEventCreateWithFlags(&arena->fence, cudaEventDisableTiming));
  *out = arena;
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_arena_reset_impl(FlameStreamArena* arena) {
  if (arena == nullptr) {
    return FLAME_CUDA_ERR_INVALID;
  }
  FLAME_CUDA_TRY(cudaEventSynchronize(arena->fence));
  arena->offset = 0;
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_arena_alloc_impl(FlameStreamArena* arena,
                                       size_t bytes,
                                       size_t align,
                                       void** out_ptr) {
  if (arena == nullptr || out_ptr == nullptr || bytes == 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  *out_ptr = nullptr;

  if (!arena->owns_base) {
    FLAME_CUDA_TRY(cudaMallocAsync(out_ptr, bytes, arena->stream));
    return FLAME_CUDA_OK;
  }

  size_t alignment = align == 0 ? 16 : align;
  size_t offset = round_up(arena->offset, alignment);
  if (offset + bytes > arena->capacity) {
    return FLAME_CUDA_ERR_INVALID;
  }

  *out_ptr = static_cast<char*>(arena->base) + offset;
  arena->offset = offset + bytes;
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_arena_record_and_release_impl(FlameStreamArena* arena) {
  if (arena == nullptr) {
    return FLAME_CUDA_ERR_INVALID;
  }
  FLAME_CUDA_TRY(cudaEventRecord(arena->fence, arena->stream));
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_arena_destroy_impl(FlameStreamArena* arena) {
  if (arena == nullptr) {
    return FLAME_CUDA_ERR_INVALID;
  }
  FLAME_CUDA_TRY(cudaEventSynchronize(arena->fence));
  FLAME_CUDA_TRY(cudaEventDestroy(arena->fence));
  if (arena->owns_base && arena->base != nullptr) {
    if (arena->stream == nullptr) {
      FLAME_CUDA_TRY(cudaFree(arena->base));
    } else {
      FLAME_CUDA_TRY(cudaFreeAsync(arena->base, arena->stream));
    }
  }
  delete arena;
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_h2d_async_impl(void* dst_device,
                                     const void* src_host,
                                     size_t bytes,
                                     cudaStream_t stream) {
  if (dst_device == nullptr || src_host == nullptr || bytes == 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  FLAME_CUDA_TRY(
      cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice, stream));
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_d2h_async_impl(void* dst_host,
                                     const void* src_device,
                                     size_t bytes,
                                     cudaStream_t stream) {
  if (dst_host == nullptr || src_device == nullptr || bytes == 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  FLAME_CUDA_TRY(
      cudaMemcpyAsync(dst_host, src_device, bytes, cudaMemcpyDeviceToHost, stream));
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_d2d_async_impl(void* dst_device,
                                     const void* src_device,
                                     size_t bytes,
                                     cudaStream_t stream) {
  if (dst_device == nullptr || src_device == nullptr || bytes == 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  FLAME_CUDA_TRY(cudaMemcpyAsync(
      dst_device, src_device, bytes, cudaMemcpyDeviceToDevice, stream));
  return FLAME_CUDA_OK;
}
