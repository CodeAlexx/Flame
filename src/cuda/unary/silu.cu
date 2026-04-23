// flame-core/src/cuda/unary/silu.cu
//
// Phase 4 target shape (per TENSORITERATOR_PORT_REFERENCE.md §5.1):
// a single functor struct + a single `extern "C"` entry point that takes an
// already-built `IterMetadata` POD by const pointer and dispatches to
// `flame::iter::launch_gpu_kernel<1, SiluBF16Op>`.
//
// The functor math is COPIED BIT-FOR-BIT from
// src/cuda/activation_silu_iter.cu (sessions 1–2 of the port). Klein
// byte-equal against pre-Phase-4 HEAD depends on identical device-side
// arithmetic — do not rewrite, do not simplify.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct SiluBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float s = v / (1.0f + __expf(-v));
        return __float2bfloat16_rn(s);
    }
};

}  // namespace

extern "C" int flame_silu_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, SiluBF16Op>(*meta, SiluBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
