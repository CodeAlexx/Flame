// flame-core/src/cuda/activation_gelu_iter.cu
//
// Second migration target for the PyTorch-mirror TensorIterator port.
// Strided-input, contig-output BF16 GELU (tanh approximation, matching
// `CUDA_GELU` in `src/bf16_ops.rs`), reusing
// `flame::iter::StridedOffsetCalc` + the templated launcher in
// `tensor_iterator.cuh`.
//
// IMPORTANT: contig callers of `Tensor::gelu` are short-circuited in
// `ops::gelu_iter::gelu_bf16_iter` back to the existing NVRTC
// `bf16_ops::gelu_bf16` vectorized `__nv_bfloat162` kernel. This file
// only fires on strided inputs (permute / narrow / as_strided).
//
// GELU math (tanh-approx):
//   c = 0.7978845608 * (x + 0.044715 * x^3)
//   y = 0.5 * x * (1 + tanhf(c))
// Same constants as `CUDA_GELU` in bf16_ops.rs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "tensor_iterator.cuh"

namespace {

struct GeluBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float c = 0.7978845608f * (v + 0.044715f * v * v * v);
        float g = 0.5f * v * (1.0f + tanhf(c));
        return __float2bfloat16_rn(g);
    }
};

}  // namespace

extern "C" int flame_gelu_bf16_strided(
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
        __nv_bfloat16, __nv_bfloat16, GeluBF16Op>(
            reinterpret_cast<const __nv_bfloat16*>(x_ptr),
            reinterpret_cast<__nv_bfloat16*>(y_ptr),
            n_elements,
            calc,
            GeluBF16Op{},
            stream);

    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
