#include <cuda_runtime.h>
#include <stdint.h>

extern "C" void* flame_cuda_alloc_pinned_host(size_t size, unsigned int flags) {
    void* p = nullptr;
    cudaError_t status = cudaHostAlloc(&p, size, flags);
    return (status == cudaSuccess) ? p : nullptr;
}

extern "C" int flame_cuda_free_pinned_host(void* ptr) {
    return static_cast<int>(cudaFreeHost(ptr));
}

extern "C" int flame_cuda_memcpy_async(
    void* dst,
    const void* src,
    size_t bytes,
    int kind,
    void* stream_void
) {
    cudaMemcpyKind copy_kind = cudaMemcpyDefault;
    switch (kind) {
        case 1: copy_kind = cudaMemcpyHostToDevice; break;
        case 2: copy_kind = cudaMemcpyDeviceToHost; break;
        case 3: copy_kind = cudaMemcpyDeviceToDevice; break;
        default: copy_kind = cudaMemcpyDefault; break;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    cudaError_t status = cudaMemcpyAsync(dst, src, bytes, copy_kind, stream);
    return static_cast<int>(status);
}

extern "C" int flame_cuda_host_register(
    void* ptr,
    size_t bytes,
    unsigned int flags
) {
    cudaError_t status = cudaHostRegister(ptr, bytes, flags);
    return static_cast<int>(status);
}

extern "C" int flame_cuda_host_unregister(void* ptr) {
    cudaError_t status = cudaHostUnregister(ptr);
    return static_cast<int>(status);
}
