// flame-core/src/cuda/unary/gelu.cu
//
// Phase 4 GELU functor + kernel entry (see `unary/silu.cu` for shape rationale).
// Functor math copied bit-for-bit from src/cuda/activation_gelu_iter.cu:
//
//   c = 0.7978845608 * (x + 0.044715 * x^3)
//   y = 0.5 * x * (1 + tanhf(c))
//
// Matches `CUDA_GELU` in src/bf16_ops.rs up to element-wise equivalence.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct GeluBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float c = 0.7978845608f * (v + 0.044715f * v * v * v);
        float g = 0.5f * v * (1.0f + tanhf(c));
        return __float2bfloat16_rn(g);
    }
};

}  // namespace

extern "C" int flame_gelu_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, GeluBF16Op>(*meta, GeluBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
