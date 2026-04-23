// flame-core/src/cuda/unary/rsqrt.cu
//
// Phase 7 rsqrt functor + kernel entry. Mirror of `unary/sigmoid.cu`.
//
// Functor math: bf16 → f32 → __frsqrt_rn(v) → __float2bfloat16_rn.
// Matches PyTorch's `at::native::rsqrt_kernel_cuda` in
// aten/src/ATen/native/cuda/UnaryOpsKernel.cu:
//   return ::rsqrt(static_cast<opmath_t>(a));
// where `opmath_t` for BF16 is `float`. The f32 intermediate matches
// PyTorch. Pre-Phase-7 there was no `GpuOps::rsqrt` — `Tensor::rsqrt`
// composed as `sqrt().reciprocal()`, which for BF16 ran two F32 roundtrips
// back-to-back. This is a single native path.
//
// Intrinsic choice: `__frsqrt_rn` (IEEE round-to-nearest reciprocal sqrt)
// matches the fp32 rsqrt PyTorch's opmath path uses. x=0→+∞, x<0→NaN,
// preserving IEEE semantics.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct RsqrtBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float r = __frsqrt_rn(v);
        return __float2bfloat16_rn(r);
    }
};

}  // namespace

extern "C" int flame_rsqrt_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, RsqrtBF16Op>(*meta, RsqrtBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
