// flame-core/src/cuda/activation_square_iter.cu
//
// Phase 2 retarget. Functor unchanged from session 3; launcher now goes
// through `flame::iter::launch_gpu_kernel<1, SquareBF16Op>`.
//
// Math: y = bf16(fp32(x) * fp32(x)), bit-equivalent to the pre-migration
// `GpuOps::mul(self, self)` dispatch.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "tensor_iterator.cuh"

namespace {

struct SquareBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        return __float2bfloat16_rn(v * v);
    }
};

}  // namespace

extern "C" int flame_square_bf16_strided(
    const void*    x_ptr,
    int64_t        x_offset_elems,
    void*          y_ptr,
    int            rank,
    const int64_t* sizes,
    const int64_t* in_strides,
    int64_t        n_elements,
    void*          stream_void)
{
    if (rank < 0 || rank > flame::iter::FLAME_MAX_DIMS) {
        return 1;
    }
    if (x_ptr == nullptr || y_ptr == nullptr) {
        return 1;
    }
    if (n_elements < 0) {
        return 1;
    }

    flame::iter::IterMetadata meta = {};
    meta.ndim        = rank;
    meta.num_args    = 2;
    meta.num_outputs = 1;
    meta.numel       = n_elements;
    meta.is_contiguous = false;
    meta.requires_32bit_indexing = (n_elements < INT_MAX);

    int64_t out_stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        meta.sizes[i]          = sizes[i];
        meta.strides[0][i]     = out_stride;
        meta.strides[1][i]     = in_strides[i];
        out_stride            *= sizes[i];
    }
    meta.offsets_elems[0] = 0;
    meta.offsets_elems[1] = x_offset_elems;
    meta.data_ptrs[0]     = y_ptr;
    meta.data_ptrs[1]     = const_cast<void*>(x_ptr);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    flame::iter::launch_gpu_kernel<1, SquareBF16Op>(meta, SquareBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
