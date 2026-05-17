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

template <typename T>
__global__ void narrow_backward_scatter_lastdim_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_in,
    int64_t rows,
    int64_t length,
    int64_t input_last_dim,
    int64_t start)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * length;
    if (tid >= n) return;

    int64_t row = tid / length;
    int64_t col = tid - row * length;
    grad_in[row * input_last_dim + start + col] = grad_out[tid];
}

__global__ void narrow_backward_scatter_lastdim_vec16_kernel(
    const uint4* __restrict__ grad_out,
    uint8_t* __restrict__ grad_in,
    int64_t rows,
    int64_t row_vecs,
    int64_t input_last_bytes,
    int64_t start_bytes)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * row_vecs;
    if (tid >= n) return;

    int64_t row = tid / row_vecs;
    int64_t vec_col = tid - row * row_vecs;
    uint4* dst = reinterpret_cast<uint4*>(
        grad_in + row * input_last_bytes + start_bytes + vec_col * 16);
    *dst = grad_out[tid];
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

    // Hot path for the transformer case: a contiguous last-dimension narrow
    // scattered back into a freshly zeroed contiguous parent. The generic
    // kernel above pays rank-wide div/mod index reconstruction for every
    // element. This path reduces that to one row/column decomposition, and
    // uses 16-byte vector copies when the slice layout is aligned.
    if (rank >= 2 && dim == rank - 1 && in_strides_host[rank - 1] == 1) {
        int64_t length = out_shape_host[rank - 1];
        int64_t rows = (length > 0) ? (n_elements / length) : 0;
        int64_t input_last_dim = in_strides_host[rank - 2];
        int64_t start_bytes = start * elem_size;
        int64_t input_last_bytes = input_last_dim * elem_size;
        int64_t length_bytes = length * elem_size;

        if (rows > 0 && length > 0) {
            if ((start_bytes % 16) == 0 &&
                (input_last_bytes % 16) == 0 &&
                (length_bytes % 16) == 0) {
                int64_t row_vecs = length_bytes / 16;
                int threads = 256;
                int64_t vec_elems = rows * row_vecs;
                int blocks = (int)((vec_elems + threads - 1) / threads);
                narrow_backward_scatter_lastdim_vec16_kernel<<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const uint4*>(grad_out),
                    reinterpret_cast<uint8_t*>(grad_in),
                    rows,
                    row_vecs,
                    input_last_bytes,
                    start_bytes);
                return (int)cudaGetLastError();
            }

            int threads = 256;
            int blocks = (int)((n_elements + threads - 1) / threads);
            if (elem_size == 2) {
                narrow_backward_scatter_lastdim_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const uint16_t*>(grad_out),
                    reinterpret_cast<uint16_t*>(grad_in),
                    rows,
                    length,
                    input_last_dim,
                    start);
                return (int)cudaGetLastError();
            }
            if (elem_size == 4) {
                narrow_backward_scatter_lastdim_kernel<uint32_t><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const uint32_t*>(grad_out),
                    reinterpret_cast<uint32_t*>(grad_in),
                    rows,
                    length,
                    input_last_dim,
                    start);
                return (int)cudaGetLastError();
            }
        }
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
