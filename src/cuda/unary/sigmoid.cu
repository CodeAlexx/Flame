// flame-core/src/cuda/unary/sigmoid.cu
//
// Phase 6 sigmoid functor + kernel entry. Mirror of `unary/silu.cu`.
//
// Functor math: bf16 → f32 → 1/(1 + exp(-v)) → __float2bfloat16_rn.
// Matches PyTorch's `at::native::sigmoid_kernel_cuda` in
// aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu:
//   return static_cast<scalar_t>(one / (one + std::exp(-opmath_t{a})));
// where `opmath_t` for BF16 is `float`. The f32 intermediate matches
// PyTorch; this is "native BF16" in the same sense that PyTorch's BF16
// sigmoid is native — storage is BF16 end-to-end, math uses opmath_t=f32.
//
// Pre-Phase-6 path: `GpuOps::sigmoid` → `cast_to_f32_tensor` → f32 sigmoid
// kernel → `restore_dtype(f32, BF16)`. Output is bit-equivalent to this
// functor for contiguous inputs because the math is identical:
//   f32_path:   bf16 → f32 materialize → sigmoid_f32(v) → f32 store → bf16 store_rn
//   functor:    bf16 → f32 on load → sigmoid_f32(v) → bf16 store_rn
// Both compute `__float2bfloat16_rn(1.0f/(1.0f+__expf(-v)))` per element.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct SigmoidBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float s = 1.0f / (1.0f + __expf(-v));
        return __float2bfloat16_rn(s);
    }
};

}  // namespace

extern "C" int flame_sigmoid_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, SigmoidBF16Op>(*meta, SigmoidBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
