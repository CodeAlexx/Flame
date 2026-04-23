// flame-core/src/cuda/unary/neg.cu
//
// Phase 6 neg functor + kernel entry. Mirror of `unary/abs.cu` — native BF16.
//
// Native BF16: flip the sign bit (bit 15). Bit-exact for all BF16 values
// including NaN (where sign-bit flip preserves NaN-ness).
//
// Why native rather than f32 round-trip: `neg` is one of only a few unaries
// with an exact bit-level representation at every input. `bf16_to_f32 →
// negate → f32_to_bf16_rn` preserves every value exactly for finite inputs,
// so the two paths are observationally identical on real data. The native
// path is strictly cheaper (one xor vs. two converts + one FP op).
//
// Pre-Phase-6 path: `Tensor::neg` was `mul_scalar(-1.0)` (composite, F32
// round-trip via `GpuOps::mul_scalar`). This functor replaces it.
//
// PyTorch reference: `at::native::neg_kernel_cuda` in
// aten/src/ATen/native/cuda/UnarySignKernels.cu — dispatches `return -a`
// for non-complex dtypes.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct NegBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        // Bit-exact sign flip. Matches PyTorch's `-a` semantics for
        // finite BF16 (including -0 → +0 and +0 → -0 via the bit flip,
        // matching IEEE-754 negation).
        unsigned short raw = *reinterpret_cast<const unsigned short*>(&x);
        unsigned short neg = raw ^ 0x8000u;
        return *reinterpret_cast<const __nv_bfloat16*>(&neg);
    }
};

}  // namespace

extern "C" int flame_neg_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, NegBF16Op>(*meta, NegBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
