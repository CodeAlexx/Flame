// flame-core/src/cuda/activation_silu_iter.cu
//
// Phase 2 retarget. Functor is unchanged from the session-1 original; the
// launcher now goes through `flame::iter::launch_gpu_kernel<1, SiluBF16Op>`
// (ported from aten/src/ATen/native/cuda/Loops.cuh::gpu_kernel) instead of
// the Phase-0-stub `launch_elementwise_strided_to_contig`.
//
// The Rust caller (`ops::silu_iter::silu_bf16_iter`) is unchanged — the
// `extern "C" flame_silu_bf16_strided` signature is stable across the Phase 2
// refactor. Only the body here is different.
//
// Math (matches `CUDA_SILU` in `src/bf16_ops.rs` and PyTorch's
// `aten/src/ATen/native/cuda/ActivationSiluKernel.cu:30` functor):
//   v = bfloat16_to_float(x)
//   s = v / (1 + __expf(-v))
//   y = float_to_bfloat16_rn(s)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "tensor_iterator.cuh"

namespace {

struct SiluBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float s = v / (1.0f + __expf(-v));
        return __float2bfloat16_rn(s);
    }
};

}  // namespace

extern "C" int flame_silu_bf16_strided(
    const void*    x_ptr,
    int64_t        x_offset_elems,
    void*          y_ptr,
    int            rank,
    const int64_t* sizes,
    const int64_t* in_strides,
    int64_t        n_elements,
    void*          stream_void)
{
    if (rank < 0 || rank > flame::iter::FLAME_MAX_DIMS) {
        return 1;  // FLAME_CUDA_ERR_INVALID
    }
    if (x_ptr == nullptr || y_ptr == nullptr) {
        return 1;
    }
    if (n_elements < 0) {
        return 1;
    }

    // Build IterMetadata. Output is always fresh contig row-major over
    // `sizes` (Rust allocated it that way); input carries arbitrary
    // strides + offset. PyTorch convention: operand 0 = output,
    // operand 1 = input (see TENSORITERATOR_PORT_REFERENCE.md §3).
    flame::iter::IterMetadata meta = {};
    meta.ndim        = rank;
    meta.num_args    = 2;
    meta.num_outputs = 1;
    meta.numel       = n_elements;
    meta.is_contiguous = false;  // Rust-side dispatch routes contig→bf16_ops::silu_bf16.
    meta.requires_32bit_indexing = (n_elements < INT_MAX);

    // Compute contiguous row-major strides for the output.
    int64_t out_stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        meta.sizes[i]          = sizes[i];
        meta.strides[0][i]     = out_stride;
        meta.strides[1][i]     = in_strides[i];
        out_stride            *= sizes[i];
    }
    meta.offsets_elems[0] = 0;                // fresh output → no view offset
    meta.offsets_elems[1] = x_offset_elems;   // input may carry view offset
    meta.data_ptrs[0]     = y_ptr;
    meta.data_ptrs[1]     = const_cast<void*>(x_ptr);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    flame::iter::launch_gpu_kernel<1, SiluBF16Op>(meta, SiluBF16Op{}, stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
