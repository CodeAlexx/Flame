// flame-core/src/cuda/unary/abs.cu
//
// Phase 6 abs functor + kernel entry. Mirror of `unary/silu.cu`.
//
// Native BF16: clear the sign bit (bit 15). Bit-equivalent to the pre-Phase-6
// `bf16_elementwise::abs_bf16` NVRTC kernel at src/bf16_elementwise.rs:310.
// That kernel also treats inputs as `unsigned short` and masks with `0x7FFF`;
// functionally identical, just expressed through the TensorIterator pipeline.
// No float math, no rounding mode to preserve.
//
// PyTorch reference: `at::native::abs_kernel_cuda` in
// aten/src/ATen/native/cuda/AbsKernel.cu — dispatches through
// `gpu_kernel(iter, lambda)` returning `std::abs(a)`. For BF16,
// `std::abs(__nv_bfloat16)` ultimately lowers to the same sign-bit clear,
// so the end result is identical.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../tensor_iterator.cuh"

namespace flame { namespace native {

namespace {

struct AbsBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        // Bit-exact sign-bit clear. Reinterpret through unsigned short to
        // avoid any fp round-trip. Matches CUDA_ABS_BF16 in bf16_elementwise.rs.
        unsigned short raw = *reinterpret_cast<const unsigned short*>(&x);
        unsigned short pos = raw & 0x7FFFu;
        return *reinterpret_cast<const __nv_bfloat16*>(&pos);
    }
};

}  // namespace

extern "C" int flame_abs_bf16_kernel(
    const flame::iter::IterMetadata* meta,
    void* stream_void)
{
    if (meta == nullptr) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    flame::iter::launch_gpu_kernel<1, AbsBF16Op>(*meta, AbsBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} }  // namespace flame::native
