// flame-core/src/cuda/binary/mul_scalar.cu
//
// Phase 5b — migrate mul_scalar onto the TensorIterator pipeline. NARGS=1
// (one tensor input, one captured scalar), arity at the functor level is 1.
//
// PyTorch parallel: `opmath_gpu_kernel_with_scalars` (Loops.cuh:200) —
// the scalar is captured as an opmath_t value inside the lambda/functor
// rather than materialised as a stride=0 tensor. flame-core matches
// that shape via a stateful functor `MulScalarBF16Op { float scalar_fp32 }`.
//
// Functor math: fp32 multiply rounded back to BF16. Derived from
// cuda/add_inplace.cu's `mul_scalar_convert` (L114):
//   return __float2bfloat16(__bfloat162float(value) * scalar);
// Same rounding convention — __float2bfloat16 defaults to _rn.
//
// This op does NOT go through the DispatchStub registry (there is no
// extra-scalar-arg variant of BF16ElementwiseKernel). The Rust wrapper
// in src/ops/mul_scalar_iter.rs calls this FFI directly.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct MulScalarBF16Op {
    float scalar_fp32;
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const
    {
        float v = __bfloat162float(x);
        return __float2bfloat16_rn(v * scalar_fp32);
    }
};

}  // namespace

extern "C" int flame_mul_scalar_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    float scalar,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    MulScalarBF16Op op{scalar};
    flame::iter::launch_gpu_kernel<1, MulScalarBF16Op>(*meta, op, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
