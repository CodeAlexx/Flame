// flame-core/src/cuda/cmp/gt.cu
// Phase 9 — greater-than. See ge.cu header for the output-dtype deviation note.
// Reference: PyTorch aten/src/ATen/native/cuda/CompareKernels.cu, OpType::GT.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct GtBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 a, __nv_bfloat16 b) const
    {
        float va = __bfloat162float(a);
        float vb = __bfloat162float(b);
        return __float2bfloat16_rn((va > vb) ? 1.0f : 0.0f);
    }
};

}  // namespace

extern "C" int flame_gt_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<2, GtBF16Op>(*meta, GtBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
