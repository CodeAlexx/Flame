// flame-core/src/cuda/add_bf16_iter.cu
//
// Phase 2 retarget. Binary functor unchanged from session 4; launcher now
// goes through `flame::iter::launch_gpu_kernel<2, AddBF16Op>`.
//
// Math: fp32 round-trip add (matches pre-migration `launch_bf16_elementwise`
// broadcast path).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "tensor_iterator.cuh"

namespace {

struct AddBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 a, __nv_bfloat16 b) const
    {
        float va = __bfloat162float(a);
        float vb = __bfloat162float(b);
        return __float2bfloat16_rn(va + vb);
    }
};

}  // namespace

extern "C" int flame_add_bf16_strided(
    const void*    a_ptr,
    int64_t        a_offset_elems,
    const int64_t* a_strides,
    const void*    b_ptr,
    int64_t        b_offset_elems,
    const int64_t* b_strides,
    void*          y_ptr,
    int            rank,
    const int64_t* sizes,
    int64_t        n_elements,
    void*          stream_void)
{
    if (rank < 0 || rank > flame::iter::FLAME_MAX_DIMS) {
        return 1;
    }
    if (a_ptr == nullptr || b_ptr == nullptr || y_ptr == nullptr) {
        return 1;
    }
    if (n_elements < 0) {
        return 1;
    }

    // PyTorch operand convention: [0] output, [1] a, [2] b. Rust caller
    // guarantees `a` and `b` have the same logical shape; output is fresh
    // contig row-major over `sizes`.
    flame::iter::IterMetadata meta = {};
    meta.ndim        = rank;
    meta.num_args    = 3;
    meta.num_outputs = 1;
    meta.numel       = n_elements;
    meta.is_contiguous = false;
    meta.requires_32bit_indexing = (n_elements < INT_MAX);

    int64_t out_stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        meta.sizes[i]          = sizes[i];
        meta.strides[0][i]     = out_stride;
        meta.strides[1][i]     = a_strides[i];
        meta.strides[2][i]     = b_strides[i];
        out_stride            *= sizes[i];
    }
    meta.offsets_elems[0] = 0;
    meta.offsets_elems[1] = a_offset_elems;
    meta.offsets_elems[2] = b_offset_elems;
    meta.data_ptrs[0]     = y_ptr;
    meta.data_ptrs[1]     = const_cast<void*>(a_ptr);
    meta.data_ptrs[2]     = const_cast<void*>(b_ptr);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    flame::iter::launch_gpu_kernel<2, AddBF16Op>(meta, AddBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
