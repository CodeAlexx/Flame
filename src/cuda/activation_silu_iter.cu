// flame-core/src/cuda/activation_silu_iter.cu
//
// First migration target for the PyTorch-mirror TensorIterator port.
// Strided-input, contig-output BF16 SiLU, driven by the `StridedOffsetCalc`
// + templated launcher in `tensor_iterator.cuh`.
//
// IMPORTANT: the CONTIGUOUS fast path for `Tensor::silu` is unchanged — the
// Rust dispatch in `ops::silu_iter::silu_bf16_iter` short-circuits
// `x.is_contiguous()` callers back to the existing NVRTC
// `bf16_ops::silu_bf16` vectorized `__nv_bfloat162` kernel. This file is
// the slow, general fallback that activates only when the input has custom
// strides or a non-zero view offset.
//
// Maths mirror `CUDA_SILU` in `src/bf16_ops.rs`:
//   v  = bfloat16_to_float(x)
//   s  = v / (1 + __expf(-v))
//   y  = float_to_bfloat16_rn(s)
// on a per-element basis (no `bfloat162` vector load — strided input is not
// guaranteed to be 2-aligned).
//
// Parity gate for the strided path is cos_sim ≥ 0.9999 vs
// `bf16_ops::silu_bf16(view.contiguous()?)`, NOT bit-exact. Small rounding
// differences vs the vectorized reference are allowed; the vectorized
// kernel uses `__bfloat1622float2` / `__floats2bfloat162_rn` whereas this
// kernel uses the scalar `__bfloat162float` / `__float2bfloat16_rn`.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "tensor_iterator.cuh"

namespace {

// Device functor consumed by `launch_elementwise_strided_to_contig`.
// Mirrors PyTorch's `silu_kernel` lambda at
//   aten/src/ATen/native/cuda/ActivationSiluKernel.cu:30
struct SiluBF16Op {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float v = __bfloat162float(x);
        float s = v / (1.0f + __expf(-v));
        return __float2bfloat16_rn(s);
    }
};

}  // namespace

// Rust-visible entry. Output is always a fresh contig BF16 buffer; Rust
// allocates it before this call. Input carries arbitrary strides + offset.
//
// Metadata (`sizes`, `strides`) are host pointers, `rank` elements each.
// We copy them into a `StridedOffsetCalc` local and pass that struct by
// value into the kernel launch parameter buffer — no `cudaMalloc` per call,
// unlike the existing `narrow_strided_launch` which still goes through
// device-side metadata arrays.
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

    flame::iter::StridedOffsetCalc calc;
    calc.rank        = rank;
    calc.base_offset = x_offset_elems;
    for (int i = 0; i < flame::iter::FLAME_MAX_DIMS; ++i) {
        calc.sizes[i]   = (i < rank) ? sizes[i]      : 1;
        calc.strides[i] = (i < rank) ? in_strides[i] : 0;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    cudaError_t err = flame::iter::launch_elementwise_strided_to_contig<
        __nv_bfloat16, __nv_bfloat16, SiluBF16Op>(
            reinterpret_cast<const __nv_bfloat16*>(x_ptr),
            reinterpret_cast<__nv_bfloat16*>(y_ptr),
            n_elements,
            calc,
            SiluBF16Op{},
            stream);

    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
