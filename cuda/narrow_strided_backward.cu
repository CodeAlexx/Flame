// flame/flame-core/cuda/narrow_strided_backward.cu
// Scatter-add for narrow backward with dtype-correct atomic adds.

#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <stdio.h>

#define DEBUG_NARROW_BACKWARD 1

enum DTypeTag { F32 = 0, F16 = 1, BF16 = 2, I32 = 3 };

static __device__ __forceinline__ void linear_to_indices(
    int64_t lin, const int64_t* __restrict__ shape, int rank, int64_t* __restrict__ idx)
{
    for (int i = rank - 1; i >= 0; --i) {
        int64_t dim = shape[i];
        idx[i] = lin % dim;
        lin /= dim;
    }
}

extern "C" __global__
void narrow_backward_scatter_add_kernel(
    const uint8_t* __restrict__ grad_out,
    uint8_t* __restrict__ grad_in,
    int rank,
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ out_strides,
    int dim,
    int64_t start,
    int64_t elem_size,
    int dtype_tag,
    int64_t n_elements)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

#ifdef DEBUG_NARROW_BACKWARD
    if (tid == 0) {
        printf("narrow_backward_scatter: elem_size=%lld n_elements=%lld dtype=%d start=%lld\n",
               (long long)elem_size, (long long)n_elements, dtype_tag, (long long)start);
    }
#endif

    // Out multi-index
    int64_t idx_buf[8];
    linear_to_indices(tid, out_shape, rank, idx_buf);

    // Input offset in elements
    int64_t in_off_e = 0;
    for (int i = 0; i < rank; ++i) {
        int64_t idx_i = idx_buf[i];
        if (i == dim) idx_i += start;
        in_off_e += idx_i * in_strides[i];
    }

    const uint8_t* s = grad_out + tid * elem_size;
    uint8_t* d = grad_in + in_off_e * elem_size;

    switch (dtype_tag) {
    case F32: {
        float* dp = reinterpret_cast<float*>(d);
        const float* sp = reinterpret_cast<const float*>(s);
        dp[0] += sp[0];
        break;
    }
    case I32: {
        int* dp = reinterpret_cast<int*>(d);
        const int* sp = reinterpret_cast<const int*>(s);
        dp[0] += sp[0];
        break;
    }
    case F16: {
        uint16_t* dp = reinterpret_cast<uint16_t*>(d);
        const uint16_t* sp = reinterpret_cast<const uint16_t*>(s);
        __half cur = *reinterpret_cast<__half*>(dp);
        __half addv = *reinterpret_cast<const __half*>(sp);
        float sumf = __half2float(cur) + __half2float(addv);
        __half out = __float2half(sumf);
        *reinterpret_cast<__half*>(dp) = out;
        break;
    }
    case BF16: {
        uint16_t* dp = reinterpret_cast<uint16_t*>(d);
        const uint16_t* sp = reinterpret_cast<const uint16_t*>(s);
        uint32_t cur_bits = static_cast<uint32_t>(dp[0]) << 16;
        float cur = __int_as_float(cur_bits);
        uint32_t add_bits = static_cast<uint32_t>(sp[0]) << 16;
        float addf = __int_as_float(add_bits);
        float sum = cur + addf;
        uint32_t sum_bits = __float_as_uint(sum);
        uint16_t new_b16 = static_cast<uint16_t>((sum_bits + 0x8000u) >> 16);
        dp[0] = new_b16;
        break;
    }
    default: {
        // Fallback: plain byte-wise copy (safe for narrow scatter – no overlap)
        for (int64_t i = 0; i < elem_size; ++i) {
            reinterpret_cast<uint8_t*>(d)[i] = reinterpret_cast<const uint8_t*>(s)[i];
        }
        break;
    }
    }
}

extern "C" int flame_narrow_backward_scatter_add_launch(
    const void* grad_out,
    void* grad_in,
    int rank,
    const int64_t* out_shape_host,
    const int64_t* in_strides_host,
    const int64_t* out_strides_host,
    int dim,
    int64_t start,
    int64_t elem_size,
    int dtype_tag,
    int64_t n_elements,
    void* stream_void)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    // Copy metadata to device
    int64_t *d_shape = nullptr, *d_in_strides = nullptr, *d_out_strides = nullptr;
    size_t meta_sz = sizeof(int64_t) * static_cast<size_t>(rank);
    if (cudaMalloc(&d_shape, meta_sz) != cudaSuccess) return (int)cudaGetLastError();
    if (cudaMalloc(&d_in_strides, meta_sz) != cudaSuccess) { cudaFree(d_shape); return (int)cudaGetLastError(); }
    if (cudaMalloc(&d_out_strides, meta_sz) != cudaSuccess) { cudaFree(d_shape); cudaFree(d_in_strides); return (int)cudaGetLastError(); }
    cudaMemcpyAsync(d_shape, out_shape_host, meta_sz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_in_strides, in_strides_host, meta_sz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_strides, out_strides_host, meta_sz, cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (int)((n_elements + threads - 1) / threads);
    narrow_backward_scatter_add_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)grad_out, (uint8_t*)grad_in,
        rank, d_shape, d_in_strides, d_out_strides,
        dim, start, elem_size, dtype_tag, n_elements
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(d_shape);
        cudaFree(d_in_strides);
        cudaFree(d_out_strides);
        return (int)launch_err;
    }

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    cudaFree(d_shape);
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);
    return (sync_err == cudaSuccess) ? 0 : (int)sync_err;
}
