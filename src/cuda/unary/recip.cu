// flame-core/src/cuda/unary/recip.cu
//
// Phase 7 recip functor + kernel entry. Mirror of `unary/sigmoid.cu`.
//
// Functor math: bf16 → f32 → __frcp_rn(v) → __float2bfloat16_rn.
// Matches PyTorch's `at::native::reciprocal_kernel_cuda` in
// aten/src/ATen/native/cuda/UnaryOpsKernel.cu:
//   return static_cast<opmath_t>(1) / static_cast<opmath_t>(a);
// where `opmath_t` for BF16 is `float`. The f32 intermediate matches
// PyTorch. Pre-Phase-7 there was no `GpuOps::recip` — `Tensor::reciprocal`
// composed as `ones.div(self)`, which for BF16 materialized an F32 tensor
// of ones and ran f32 div. This is a new native path.
//
// Intrinsic choice: `__frcp_rn` (IEEE round-to-nearest reciprocal) is the
// fp32 reciprocal PyTorch's opmath path uses. For x=0 returns +∞, for
// x=-0 returns -∞, x=NaN→NaN, preserving IEEE semantics.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct RecipBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float r = __frcp_rn(v);
        return __float2bfloat16_rn(r);
    }
};

}  // namespace

extern "C" int flame_recip_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, RecipBF16Op>(*meta, RecipBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
