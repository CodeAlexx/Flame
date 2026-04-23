// flame-core/src/cuda/binary/add.cu
//
// Phase 4 add functor + kernel entry. First binary op on the new pipeline;
// NARGS=2 at the functor level (two inputs) which becomes num_args=3 in
// `IterMetadata` (out + a + b). Functor math copied bit-for-bit from
// src/cuda/add_bf16_iter.cu: fp32 round-trip add.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct AddBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 a, __nv_bfloat16 b) const
    {
        float va = __bfloat162float(a);
        float vb = __bfloat162float(b);
        return __float2bfloat16_rn(va + vb);
    }
};

}  // namespace

extern "C" int flame_add_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<2, AddBF16Op>(*meta, AddBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
