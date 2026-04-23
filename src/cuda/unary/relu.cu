// flame-core/src/cuda/unary/relu.cu
//
// Phase 6 relu functor + kernel entry. Mirror of `unary/silu.cu`.
//
// Native BF16: max(x, 0). Semantically equivalent to the pre-Phase-6
// `fc_relu_bf16` kernel at cuda/cuda_ops.cu:109 — that kernel does the same
// comparison after a BF16→f32 round-trip via `__bfloat1622float2`. This
// functor expresses the comparison directly on BF16 without the round-trip,
// which is bit-exact for finite inputs (the sign/magnitude comparison is
// the same in BF16 and f32 for non-NaN, non-infinite values).
//
// For x=NaN: both paths propagate NaN. For x=0: both paths return 0.
// For x=-0: both paths return +0 (because -0 < 0 is false but -0 > 0 is
// also false, and we select the `0.0f` branch).
//
// PyTorch reference: relu is implemented via `clamp_min(x, 0)`; the CUDA
// functor reduces to `a > 0 ? a : 0` for non-complex dtypes.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct ReluBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        // Native BF16 comparison via __hgt (returns true if x > 0).
        // Zero literal constructed via __float2bfloat16_rn(0.0f) — compile-time
        // constant, no per-element cost.
        const __nv_bfloat16 zero = __float2bfloat16_rn(0.0f);
        return __hgt(x, zero) ? x : zero;
    }
};

}  // namespace

extern "C" int flame_relu_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, ReluBF16Op>(*meta, ReluBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
