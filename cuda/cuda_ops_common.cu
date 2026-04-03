#include "cuda_ops.h"

#include <cuda_runtime.h>

#if CUDART_VERSION < 11020
#include <cuda_runtime_api.h>
#endif

namespace {

constexpr size_t kWorkspaceAlignBytes = 1u << 20; // 1 MiB granularity

inline size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

inline fc_status_t release_workspace(fc_workspace_t* arena, cudaStream_t stream) {
    if (!arena || arena->ptr == nullptr) {
        return FC_OK;
    }
#if CUDART_VERSION >= 11020
    if (cudaFreeAsync(arena->ptr, stream) != cudaSuccess) {
        if (cudaFree(arena->ptr) != cudaSuccess) {
            return FC_ERR_LAUNCH;
        }
    }
#else
    if (cudaFree(arena->ptr) != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }
#endif
    arena->ptr = nullptr;
    arena->bytes = 0;
    return FC_OK;
}

}  // namespace

extern "C" fc_status_t fc_ws_ensure_capacity(fc_workspace_t* arena, size_t bytes, cudaStream_t stream) {
    if (!arena) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    if (bytes == 0) {
        return FC_OK;
    }

    const size_t required = align_up(bytes, kWorkspaceAlignBytes);
    if (arena->ptr != nullptr && arena->bytes >= required) {
        return FC_OK;
    }

    void* new_ptr = nullptr;
#if CUDART_VERSION >= 11020
    cudaError_t alloc_status = cudaMallocAsync(&new_ptr, required, stream);
    if (alloc_status == cudaErrorUnsupportedPtxVersion || alloc_status == cudaErrorNotSupported) {
        alloc_status = cudaMalloc(&new_ptr, required);
    }
#else
    cudaError_t alloc_status = cudaMalloc(&new_ptr, required);
#endif

    if (alloc_status != cudaSuccess) {
        return FC_ERR_OOM;
    }

    if (arena->ptr != nullptr) {
        fc_status_t free_status = release_workspace(arena, stream);
        if (free_status != FC_OK) {
#if CUDART_VERSION >= 11020
            cudaFreeAsync(new_ptr, stream);
#else
            cudaFree(new_ptr);
#endif
            return free_status;
        }
    }

    arena->ptr = new_ptr;
    arena->bytes = required;
    return FC_OK;
}

extern "C" fc_status_t fc_bf16_memcpy_async(void* dst,
                                            const void* src,
                                            size_t bytes,
                                            cudaStream_t stream) {
    if (!dst || !src) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (bytes == 0) {
        return FC_OK;
    }
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }
    return FC_OK;
}
