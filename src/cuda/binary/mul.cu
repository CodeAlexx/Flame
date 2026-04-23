// flame-core/src/cuda/binary/mul.cu
//
// Phase 5b — migrate mul onto the TensorIterator pipeline. Mirrors
// src/cuda/binary/add.cu shape.
//
// Functor math: fp32 round-trip multiply. Derived from
// src/bf16_elementwise.rs CUDA_ADD_MUL_BF16 mul_bf16_kernel (L237):
//   O[tid] = __float2bfloat16(__bfloat162float(A[a_off]) * __bfloat162float(B[b_off]));
// Same rounding convention (round-to-nearest-even; __float2bfloat16
// defaults to _rn).
//
// Reference: PyTorch aten/src/ATen/native/cuda/BinaryMulKernel.cu.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct MulBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 a, __nv_bfloat16 b) const
    {
        float va = __bfloat162float(a);
        float vb = __bfloat162float(b);
        return __float2bfloat16_rn(va * vb);
    }
};

}  // namespace

extern "C" int flame_mul_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<2, MulBF16Op>(*meta, MulBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
