// flame-core/src/cuda/unary/tanh.cu
//
// Phase 6 tanh functor + kernel entry. Mirror of `unary/silu.cu`.
//
// Functor math: bf16 → f32 → tanhf(v) → __float2bfloat16_rn.
// Matches PyTorch's `at::native::tanh_kernel_cuda` in
// aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu:
//   return ::tanh(static_cast<opmath_t>(a));
// where `opmath_t` for BF16 is `float`. The f32 intermediate matches
// PyTorch; BF16 has no `__htanh` intrinsic, so this is the canonical
// native-BF16 path.
//
// Pre-Phase-6 path: `GpuOps::tanh` → `cast_to_f32_tensor` → f32 tanh kernel
// → `restore_dtype(f32, BF16)`. Output is bit-equivalent.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct TanhBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float t = tanhf(v);
        return __float2bfloat16_rn(t);
    }
};

}  // namespace

extern "C" int flame_tanh_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, TanhBF16Op>(*meta, TanhBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
