// flame-core/src/cuda/binary/minimum.cu
//
// Phase 5b — migrate minimum onto the TensorIterator pipeline.
//
// Functor math: fp32 pair min, rounded back to BF16. Derived from
// src/bf16_elementwise.rs CUDA_ADD_MUL_BF16 min_bf16_kernel (L265):
//   float a = __bfloat162float(A[a_off]), b = __bfloat162float(B[b_off]);
//   O[tid] = __float2bfloat16(a < b ? a : b);
// The branch is equivalent to fminf for finite inputs.
//
// Reference: PyTorch aten/src/ATen/native/cuda/MaxMinElementwiseKernel.cu.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct MinimumBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 a, __nv_bfloat16 b) const
    {
        float va = __bfloat162float(a);
        float vb = __bfloat162float(b);
        return __float2bfloat16_rn(fminf(va, vb));
    }
};

}  // namespace

extern "C" int flame_minimum_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<2, MinimumBF16Op>(*meta, MinimumBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
