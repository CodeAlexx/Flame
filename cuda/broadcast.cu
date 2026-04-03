#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template <typename T>
__global__ void broadcast_strided_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ in_stride,
    const int64_t* __restrict__ out_stride,
    int ndim,
    int64_t total_elems) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) {
        return;
    }

    int64_t rem = idx;
    int64_t in_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        const int64_t step = out_stride[d];
        const int64_t coord = (step > 0) ? (rem / step) : 0;
        rem = (step > 0) ? (rem % step) : rem;
        in_offset += in_stride[d] * coord;
    }

    out[idx] = in[in_offset];
}

template <typename T>
static inline void launch_broadcast_impl(
    const T* in,
    T* out,
    const int64_t* out_shape,
    const int64_t* in_stride,
    const int64_t* out_stride,
    int ndim,
    int64_t total,
    cudaStream_t stream) {
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    broadcast_strided_kernel<T><<<grid, block, 0, stream>>>(
        in, out, out_shape, in_stride, out_stride, ndim, total);
}

extern "C" void launch_broadcast_f32(
    const float* in,
    float* out,
    const int64_t* out_shape,
    const int64_t* in_stride,
    const int64_t* out_stride,
    int ndim,
    int64_t total,
    cudaStream_t stream) {
    launch_broadcast_impl<float>(in, out, out_shape, in_stride, out_stride, ndim, total, stream);
}

extern "C" void launch_broadcast_bf16(
    const __nv_bfloat16* in,
    __nv_bfloat16* out,
    const int64_t* out_shape,
    const int64_t* in_stride,
    const int64_t* out_stride,
    int ndim,
    int64_t total,
    cudaStream_t stream) {
    launch_broadcast_impl<__nv_bfloat16>(in, out, out_shape, in_stride, out_stride, ndim, total, stream);
}
