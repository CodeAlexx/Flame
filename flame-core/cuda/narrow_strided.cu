// flame/flame-core/cuda/narrow_strided.cu
// General byte-wise narrow gather for any dimension with strided input.

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ void linear_to_indices(
    int64_t lin, const int64_t* __restrict__ shape, int rank, int64_t* __restrict__ idx)
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
    const int64_t* __restrict__ out_shape,     // [rank]
    const int64_t* __restrict__ src_strides,   // [rank], in elements
    const int64_t* __restrict__ out_strides,   // [rank], in elements (row-major)
    int dim,
    int64_t start,
    int64_t elem_size,
    int64_t n_elements)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    // Compute output multi-index
    int64_t idx_buf[8]; // supports rank <= 8
    linear_to_indices(tid, out_shape, rank, idx_buf);

    // Map to source multi-index (add start on the narrowed dim)
    int64_t src_offset_elems = 0;
    for (int i = 0; i < rank; ++i) {
        int64_t idx_i = idx_buf[i];
        if (i == dim) idx_i += start;
        src_offset_elems += idx_i * src_strides[i];
    }

    // Byte addresses
    const uint8_t* s = src + src_offset_elems * elem_size;
    uint8_t* d = dst + tid * elem_size;

    // Copy elem_size bytes
    int64_t n8 = elem_size / 8;
    int64_t rem = elem_size % 8;
    const uint64_t* s64 = reinterpret_cast<const uint64_t*>(s);
    uint64_t* d64 = reinterpret_cast<uint64_t*>(d);
#pragma unroll
    for (int64_t i = 0; i < n8; ++i) d64[i] = s64[i];
    for (int64_t i = 0; i < rem; ++i) d[n8 * 8 + i] = s[n8 * 8 + i];
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
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    // Copy tiny metadata arrays to device
    int64_t *d_shape = nullptr, *d_src_strides = nullptr, *d_out_strides = nullptr;
    size_t meta_sz = sizeof(int64_t) * static_cast<size_t>(rank);
    if (cudaMalloc(&d_shape, meta_sz) != cudaSuccess) return (int)cudaGetLastError();
    if (cudaMalloc(&d_src_strides, meta_sz) != cudaSuccess) { cudaFree(d_shape); return (int)cudaGetLastError(); }
    if (cudaMalloc(&d_out_strides, meta_sz) != cudaSuccess) { cudaFree(d_shape); cudaFree(d_src_strides); return (int)cudaGetLastError(); }
    cudaMemcpyAsync(d_shape, out_shape_host, meta_sz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_src_strides, src_strides_host, meta_sz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_strides, out_strides_host, meta_sz, cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (int)((n_elements + threads - 1) / threads);

    narrow_strided_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)src, (uint8_t*)dst,
        rank, d_shape, d_src_strides, d_out_strides,
        dim, start, elem_size, n_elements
    );

    cudaError_t err = cudaGetLastError();
    cudaFree(d_shape);
    cudaFree(d_src_strides);
    cudaFree(d_out_strides);
    return (err == cudaSuccess) ? 0 : (int)err;
}
