// flame-core/src/cuda/binary/add_scalar.cu
//
// Phase 5b — migrate add_scalar onto the TensorIterator pipeline. NARGS=1
// (one tensor input, one captured scalar).
//
// PyTorch parallel: `opmath_gpu_kernel_with_scalars` (Loops.cuh:200).
// The scalar is captured inside the functor (stateful) rather than
// materialised as a stride=0 tensor.
//
// Functor math: fp32 add rounded back to BF16. Derived from
// cuda/add_inplace.cu's `add_scalar_convert` (L104):
//   return __float2bfloat16(__bfloat162float(value) + scalar);
//
// This op does NOT go through the DispatchStub registry (see
// mul_scalar.cu for rationale). The Rust wrapper in
// src/ops/add_scalar_iter.rs calls this FFI directly.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct AddScalarBF16Op {
    float scalar_fp32;
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const
    {
        float v = __bfloat162float(x);
        return __float2bfloat16_rn(v + scalar_fp32);
    }
};

}  // namespace

extern "C" int flame_add_scalar_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    float scalar,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    AddScalarBF16Op op{scalar};
    flame::iter::launch_gpu_kernel<1, AddScalarBF16Op>(*meta, op, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
