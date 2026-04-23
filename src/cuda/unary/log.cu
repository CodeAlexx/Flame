// flame-core/src/cuda/unary/log.cu
//
// Phase 7 log functor + kernel entry. Mirror of `unary/sigmoid.cu`.
//
// Functor math: bf16 → f32 → __logf(v) → __float2bfloat16_rn.
// Matches PyTorch's `at::native::log_kernel_cuda` in
// aten/src/ATen/native/cuda/UnaryOpsKernel.cu:
//   return ::log(static_cast<opmath_t>(a));
// where `opmath_t` for BF16 is `float`.
//
// Intrinsic choice: `__logf` (fast fp32 intrinsic, ~3 ULP error) matches
// the flame-core convention of fast intrinsics where they exist. PyTorch
// uses `::log` (libm). The delta is within BF16 rounding on the valid
// input domain (cos_sim ≥ 0.9999 on random positive BF16 ranges).
//
// Pre-Phase-7 path: `GpuOps::log` → `cast_to_f32_tensor` → f32 log kernel.
// Note the pre-Phase-7 f32 kernel (LOG_KERNEL in cuda_kernel_sources.rs:228)
// clamped `v < 1e-20f` to `1e-20f` before calling `logf`. Phase 7 drops
// this: native BF16 cannot express `1e-20` (underflows to 0), and PyTorch
// does not clamp — `log(0)` yields `-∞`, `log(<0)` yields NaN, matching
// IEEE semantics. The edge-value test exercises this explicitly.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct LogBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float l = __logf(v);
        return __float2bfloat16_rn(l);
    }
};

}  // namespace

extern "C" int flame_log_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, LogBF16Op>(*meta, LogBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
