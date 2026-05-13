// flame/flame-core/cuda/narrow_strided.cu
// General byte-wise narrow gather and scatter-add for any dimension with
// strided input. Metadata (shape + strides) passes inline via kernel-arg
// space — no per-call cudaMalloc / cudaMemcpyAsync / cudaStreamSynchronize /
// cudaFree. This is Reference Implementation #1 for the flame-core speed
// contract: launch wrappers must not host-sync.

#include <cuda_runtime.h>
#include <stdint.h>

// Max supported tensor rank for narrow ops. Diffusion-model tensors
// in flame-core trainers stay well under this (typical 2-5).
#define FLAME_NARROW_MAX_RANK 8

// Shape and strides packed inline. Passed by value through the kernel
// argument space.
struct NarrowMeta {
    int64_t shape[FLAME_NARROW_MAX_RANK];
    int64_t strides[FLAME_NARROW_MAX_RANK];
};

static __device__ __forceinline__ void linear_to_indices(
    int64_t lin, const int64_t* shape, int rank, int64_t* idx)
{
    // Row-major unravel: idx[0]..idx[rank-1]
    for (int i = rank - 1; i >= 0; --i) {
        int64_t dim = shape[i];
        idx[i] = lin % dim;
        lin /= dim;
    }
}

extern "C" __global__
void narrow_strided_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int rank,
    NarrowMeta meta,
    int dim,
    int64_t start,
    int64_t elem_size,
    int64_t n_elements)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    int64_t idx_buf[FLAME_NARROW_MAX_RANK];
    linear_to_indices(tid, meta.shape, rank, idx_buf);

    int64_t src_offset_elems = 0;
    for (int i = 0; i < rank; ++i) {
        int64_t idx_i = idx_buf[i];
        if (i == dim) idx_i += start;
        src_offset_elems += idx_i * meta.strides[i];
    }

    const uint8_t* s = src + src_offset_elems * elem_size;
    uint8_t* d = dst + tid * elem_size;

    for (int64_t i = 0; i < elem_size; ++i) {
        d[i] = s[i];
    }
}

extern "C" int flame_narrow_strided_launch(
    const void* src,
    void* dst,
    int rank,
    const int64_t* out_shape_host,
    const int64_t* src_strides_host,
    const int64_t* out_strides_host,
    int dim,
    int64_t start,
    int64_t elem_size,
    int64_t n_elements,
    void* stream_void)
{
    (void)out_strides_host;  // ABI-compat: kernel computes output offset from tid

    if (rank < 0 || rank > FLAME_NARROW_MAX_RANK) {
        return -1;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    NarrowMeta meta = {};
    for (int i = 0; i < rank; ++i) {
        meta.shape[i] = out_shape_host[i];
        meta.strides[i] = src_strides_host[i];
    }

    int threads = 256;
    int blocks = (int)((n_elements + threads - 1) / threads);

    narrow_strided_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)src,
        (uint8_t*)dst,
        rank,
        meta,
        dim, start, elem_size, n_elements);

    return (int)cudaGetLastError();
}

extern "C" __global__
void narrow_backward_scatter_add_kernel(
    const uint8_t* __restrict__ grad_out,
    uint8_t* __restrict__ grad_in,
    int rank,
    NarrowMeta meta,
    int dim,
    int64_t start,
    int64_t elem_size,
    int64_t n_elements)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    int64_t idx_buf[FLAME_NARROW_MAX_RANK];
    linear_to_indices(tid, meta.shape, rank, idx_buf);

    int64_t in_offset_elems = 0;
    for (int i = 0; i < rank; ++i) {
        int64_t idx_i = idx_buf[i];
        if (i == dim) idx_i += start;
        in_offset_elems += idx_i * meta.strides[i];
    }

    const uint8_t* src = grad_out + tid * elem_size;
    uint8_t* dst = grad_in + in_offset_elems * elem_size;

    for (int64_t i = 0; i < elem_size; ++i) {
        dst[i] = src[i];
    }
}

extern "C" int narrow_backward_scatter_add_launch(
    const void* grad_out,
    void* grad_in,
    int rank,
    const int64_t* out_shape_host,
    const int64_t* in_strides_host,
    const int64_t* out_strides_host,
    int dim,
    int64_t start,
    int64_t elem_size,
    int64_t n_elements,
    void* stream_void)
{
    (void)out_strides_host;  // ABI-compat: kernel computes output offset from tid

    if (rank < 0 || rank > FLAME_NARROW_MAX_RANK) {
        return -1;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    NarrowMeta meta = {};
    for (int i = 0; i < rank; ++i) {
        meta.shape[i] = out_shape_host[i];
        meta.strides[i] = in_strides_host[i];
    }

    int threads = 256;
    int blocks = (int)((n_elements + threads - 1) / threads);

    narrow_backward_scatter_add_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)grad_out,
        (uint8_t*)grad_in,
        rank,
        meta,
        dim, start, elem_size, n_elements);

    return (int)cudaGetLastError();
}
