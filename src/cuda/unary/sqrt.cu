// flame-core/src/cuda/unary/sqrt.cu
//
// Phase 7 sqrt functor + kernel entry. Mirror of `unary/sigmoid.cu`.
//
// Functor math: bf16 → f32 → __fsqrt_rn(v) → __float2bfloat16_rn.
// Matches PyTorch's `at::native::sqrt_kernel_cuda` in
// aten/src/ATen/native/cuda/UnaryOpsKernel.cu:
//   return ::sqrt(static_cast<opmath_t>(a));
// where `opmath_t` for BF16 is `float`. The f32 intermediate matches
// PyTorch; BF16 has no `__hsqrt` native intrinsic on storage type, so
// this is the canonical native-BF16 path.
//
// Pre-Phase-7 path: `GpuOps::sqrt` → `cast_to_f32_tensor` → f32 sqrt kernel
// (sqrtf) → `restore_dtype(f32, BF16)`. Intrinsic choice: `__fsqrt_rn`
// (IEEE round-to-nearest) matches the fp32 sqrt PyTorch performs via its
// opmath path. Output is bit-equivalent for all finite inputs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct SqrtBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float s = __fsqrt_rn(v);
        return __float2bfloat16_rn(s);
    }
};

}  // namespace

extern "C" int flame_sqrt_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, SqrtBF16Op>(*meta, SqrtBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
