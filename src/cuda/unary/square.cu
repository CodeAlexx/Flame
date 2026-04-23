// flame-core/src/cuda/unary/square.cu
//
// Phase 4 square functor + kernel entry (see `unary/silu.cu` for shape rationale).
// Functor math copied bit-for-bit from src/cuda/activation_square_iter.cu:
// y = bf16(fp32(x) * fp32(x)). Bit-equivalent to the pre-migration
// `GpuOps::mul(self, self)` dispatch for contig inputs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct SquareBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        return __float2bfloat16_rn(v * v);
    }
};

}  // namespace

extern "C" int flame_square_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, SquareBF16Op>(*meta, SquareBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
