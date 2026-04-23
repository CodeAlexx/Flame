// flame-core/src/cuda/activation_square_iter.cu
//
// Third migration target for the PyTorch-mirror TensorIterator port.
// Strided-input, contig-output BF16 square (y = x*x), reusing
// `flame::iter::StridedOffsetCalc` + the templated launcher in
// `tensor_iterator.cuh`.
//
// Contig callers of `Tensor::square` for BF16 are short-circuited in
// `ops::square_iter::square_bf16_iter` to `bf16_ops::square_bf16`, which
// is bit-equivalent to the prior `GpuOps::mul(self, self)` dispatch
// (both compute `bf16(fp32(x) * fp32(x))` with the same rounding).
// This file only fires on strided BF16 inputs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

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

    flame::iter::StridedOffsetCalc calc;
    calc.rank        = rank;
    calc.base_offset = x_offset_elems;
    for (int i = 0; i < flame::iter::FLAME_MAX_DIMS; ++i) {
        calc.sizes[i]   = (i < rank) ? sizes[i]      : 1;
        calc.strides[i] = (i < rank) ? in_strides[i] : 0;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    cudaError_t err = flame::iter::launch_elementwise_strided_to_contig<
        __nv_bfloat16, __nv_bfloat16, SquareBF16Op>(
            reinterpret_cast<const __nv_bfloat16*>(x_ptr),
            reinterpret_cast<__nv_bfloat16*>(y_ptr),
            n_elements,
            calc,
            SquareBF16Op{},
            stream);

    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
