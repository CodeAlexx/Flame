// flame-core/src/cuda/unary/gelu_exact.cu
//
// Exact-erf BF16 GELU functor + kernel entry. Mirrors the structure of
// `unary/gelu.cu` (tanh-approx) but uses `erff` for PyTorch-exact parity.
//
// PyTorch reference: `aten/src/ATen/native/cuda/ActivationGeluKernel.cu`
// with `approximate='none'` (the default for `torch.nn.GELU()`):
//
//   y = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// `sqrt(2)/2 = M_SQRT1_2 = 0.7071067811865475`. CUDA provides `erff(float)`
// as a device built-in (no extra header needed).
//
// Added 2026-05-21 for Cosmos-Predict2.5 which uses bare `nn.GELU()` in its
// MLP — the existing tanh-approx variant diverged by ~9e-4 per element.
//
// Forward-only: no backward registered. Cosmos is inference-only and
// flame-core's existing `gelu_backward.cu` uses the tanh-approx derivative.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct GeluExactBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        // sqrt(2)/2 == 1/sqrt(2)
        float y = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        return __float2bfloat16_rn(y);
    }
};

}  // namespace

extern "C" int flame_gelu_exact_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    // Drain any sticky cudaError from upstream so cudaGetLastError after our
    // launch only reflects this kernel's status. Mirrors PyTorch's
    // TORCH_CUDA_CHECK_AFTER_LAUNCH discipline.
    (void)cudaGetLastError();
    flame::iter::launch_gpu_kernel<1, GeluExactBF16Op>(*meta, GeluExactBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
