// flame-core/src/cuda/cmp/ne.cu
// Phase 9 — not-equal. See ge.cu header for the output-dtype deviation note.
// NaN semantics: IEEE 754 — ne(NaN, NaN) = true.
// Reference: PyTorch aten/src/ATen/native/cuda/CompareEQKernel.cu, EqOpType::NE.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct NeBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 a, __nv_bfloat16 b) const
    {
        float va = __bfloat162float(a);
        float vb = __bfloat162float(b);
        return __float2bfloat16_rn((va != vb) ? 1.0f : 0.0f);
    }
};

}  // namespace

extern "C" int flame_ne_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<2, NeBF16Op>(*meta, NeBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
