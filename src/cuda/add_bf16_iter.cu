// flame-core/src/cuda/add_bf16_iter.cu
//
// Session 4, 2026-04-22: first BINARY op on the TensorIterator scaffolding.
// Same-shape elementwise BF16 add with strided inputs and a contiguous
// row-major output. Broadcast (different-shape inputs) stays on the
// existing `launch_bf16_elementwise` path — scope-deferred.
//
// Contig+contig same-shape callers short-circuit in
// `ops::add_iter::add_bf16_iter` straight back to
// `bf16_elementwise::add_bf16` (whose fast path uses the vectorized
// `__hadd2` kernel). This file only fires when at least one input is
// strided.
//
// Math: fp32 round-trip, matching `launch_bf16_elementwise`
// (the broadcast slow path that already handles arbitrary strides in
// pre-migration code). Gate is cos_sim ≥ 0.9999, not bit-exact vs the
// vectorized `__hadd2` flat kernel.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "tensor_iterator.cuh"

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

inline void fill_calc(
    flame::iter::StridedOffsetCalc& calc,
    int rank,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t base_offset)
{
    calc.rank        = rank;
    calc.base_offset = base_offset;
    for (int i = 0; i < flame::iter::FLAME_MAX_DIMS; ++i) {
        calc.sizes[i]   = (i < rank) ? sizes[i]    : 1;
        calc.strides[i] = (i < rank) ? strides[i]  : 0;
    }
}

}  // namespace

/// Rust-visible entry. Both inputs must have the same logical shape
/// (no broadcast). `a_strides`, `b_strides`, `sizes` are host pointers,
/// `rank` elements each. Output is always a fresh contig BF16 buffer
/// with element count `n_elements` = product(sizes).
extern "C" int flame_add_bf16_strided(
    const void*    a_ptr,
    int64_t        a_offset_elems,
    const int64_t* a_strides,
    const void*    b_ptr,
    int64_t        b_offset_elems,
    const int64_t* b_strides,
    void*          y_ptr,
    int            rank,
    const int64_t* sizes,
    int64_t        n_elements,
    void*          stream_void)
{
    if (rank < 0 || rank > flame::iter::FLAME_MAX_DIMS) {
        return 1;
    }
    if (a_ptr == nullptr || b_ptr == nullptr || y_ptr == nullptr) {
        return 1;
    }
    if (n_elements < 0) {
        return 1;
    }

    flame::iter::StridedOffsetCalc a_calc;
    flame::iter::StridedOffsetCalc b_calc;
    fill_calc(a_calc, rank, sizes, a_strides, a_offset_elems);
    fill_calc(b_calc, rank, sizes, b_strides, b_offset_elems);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    cudaError_t err = flame::iter::launch_elementwise_strided_binary_to_contig<
        __nv_bfloat16, __nv_bfloat16, AddBF16Op>(
            reinterpret_cast<const __nv_bfloat16*>(a_ptr),
            reinterpret_cast<const __nv_bfloat16*>(b_ptr),
            reinterpret_cast<__nv_bfloat16*>(y_ptr),
            n_elements,
            a_calc,
            b_calc,
            AddBF16Op{},
            stream);

    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
