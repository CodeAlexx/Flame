// flame-core/src/cuda/unary/exp.cu
//
// Phase 7 exp functor + kernel entry. Mirror of `unary/sigmoid.cu`.
//
// Functor math: bf16 → f32 → __expf(v) → __float2bfloat16_rn.
// Matches PyTorch's `at::native::exp_kernel_cuda` in
// aten/src/ATen/native/cuda/UnaryOpsKernel.cu:
//   return ::exp(static_cast<opmath_t>(a));
// where `opmath_t` for BF16 is `float`.
//
// Intrinsic choice: `__expf` (fast fp32 intrinsic, ~2 ULP error) matches
// the existing flame-core convention used in silu.cu and sigmoid.cu.
// PyTorch uses `::exp` (libm) which is more precise but slower; the delta
// is within BF16 rounding for all non-saturation inputs (cos_sim ≥ 0.9999
// on random BF16 ranges, same threshold as sigmoid/silu parity).
//
// Pre-Phase-7 path: `GpuOps::exp` → `cast_to_f32_tensor` → f32 exp kernel
// (expf libm) → `restore_dtype(f32, BF16)`. Output differs at most by
// last-ULP due to __expf vs expf; cos_sim threshold is the same as silu.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct ExpBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float e = __expf(v);
        return __float2bfloat16_rn(e);
    }
};

}  // namespace

extern "C" int flame_exp_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, ExpBF16Op>(*meta, ExpBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
