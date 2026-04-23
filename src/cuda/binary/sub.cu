// flame-core/src/cuda/binary/sub.cu
//
// Phase 5b — migrate sub onto the TensorIterator pipeline. Mirrors
// src/cuda/binary/add.cu (Phase 4 pattern). NARGS=2 at the functor level
// (two inputs) -> num_args=3 in IterMetadata (out + a + b).
//
// Functor math: fp32 round-trip subtract. Derived from the elementwise
// broadcast path in src/bf16_elementwise.rs CUDA_ADD_MUL_BF16
// (add_bf16_kernel at L228 uses the same pattern), adapted for the
// missing sub_bf16_kernel in that legacy path (the broadcast path did
// not have a sub kernel — sub was composed via a + (-1 * b)).
// The contig fast-path in CUDA_ADD_MUL_BF16_FLAT (sub_bf16_flat_kernel
// at L163) uses __hsub2 for native BF16; this functor matches its
// semantics via an fp32 round-trip, same convention chosen for
// Phase 4's AddBF16Op in add.cu.
//
// Reference: PyTorch aten/src/ATen/native/cuda/BinarySubKernel.cu.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct SubBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 a, __nv_bfloat16 b) const
    {
        float va = __bfloat162float(a);
        float vb = __bfloat162float(b);
        return __float2bfloat16_rn(va - vb);
    }
};

}  // namespace

extern "C" int flame_sub_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<2, SubBF16Op>(*meta, SubBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
