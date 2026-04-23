// flame-core/src/cuda/cmp/ge.cu
//
// Phase 9 — migrate ge (greater-or-equal) onto the TensorIterator pipeline.
//
// Output-dtype note (flame-core-specific deviation from PyTorch):
//   PyTorch returns a kBool (1-byte) output for ge/gt/le/lt/eq/ne.
//   flame-core has no 1-byte tensor storage — its `DType::Bool` backend is
//   an f32 buffer, and `DType::U8` is unimplemented in TensorStorage. Adding
//   a byte-addressable storage class is out of Phase 9's scope.
//
//   So flame-core's convention for compare outputs mirrors the live path
//   `GpuOps::compare_binary` (src/cuda_ops.rs:188) that the current
//   `Tensor::ge`/etc. call: write a sentinel 0.0/1.0 value at the input
//   dtype. Here that dtype is BF16. Callers that want an F32 mask do
//   `.ge(...)?.to_dtype(DType::F32)?` (see autograd_v4/ops/sdpa.rs:632),
//   which is trivially lossless when the only live values are 0 and 1.
//
// Functor math: compare in fp32 (matches PyTorch's opmath_t=float for BF16
// inputs, and handles BF16's denormals uniformly with PyTorch). NaN
// semantics follow IEEE 754: NaN ≥ anything is false.
//
// Reference: PyTorch aten/src/ATen/native/cuda/CompareKernels.cu
// (CompareFunctor<scalar_t>::operator(), case OpType::GE).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct GeBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 a, __nv_bfloat16 b) const
    {
        float va = __bfloat162float(a);
        float vb = __bfloat162float(b);
        return __float2bfloat16_rn((va >= vb) ? 1.0f : 0.0f);
    }
};

}  // namespace

extern "C" int flame_ge_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<2, GeBF16Op>(*meta, GeBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
