// flame-core/src/cuda/tensor_iterator.cuh
//
// Port of the minimal path from PyTorch's TensorIterator infrastructure:
//   aten/src/ATen/cuda/detail/OffsetCalculator.cuh
//   aten/src/ATen/native/cuda/Loops.cuh            (gpu_kernel_impl_nocast)
//
// Scope this session (HANDOFF_2026-04-22_TENSORITERATOR_PORT):
//   * Rank ≤ 6 (matches flame-core's Shape SmallVec<[usize;6]>).
//   * One input, one contiguous row-major output (unary elementwise).
//   * Plain 32/64-bit divmod — no IntegerDivider magic constants yet. The
//     strided path is a correctness-first slow path; the contig fast path
//     does not go through this iterator.
//   * Metadata passed by value inside `StridedOffsetCalc` — no per-launch
//     `cudaMalloc`/`cudaMemcpyAsync` pair the way `narrow_strided.cu`
//     currently does. Struct size (≈108 B) fits comfortably in the CUDA
//     kernel-parameter buffer (max 4 KB).
//
// The next session migrates `gelu_bf16` on top of the same header.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace flame { namespace iter {

// Matches flame-core's `Strides = SmallVec<[usize; 6]>`. If the Rust-side
// shape rank ever grows past this, bump both sides — not this session.
constexpr int FLAME_MAX_DIMS = 6;

// Strided-input offset calculator (NARGS=1 in PyTorch parlance).
//
// The output is always contiguous row-major over `sizes`, so its offset is
// the linear thread index itself; we only need to compute the strided INPUT
// offset. Keeping it single-arg keeps the struct small and matches how
// activation/unary kernels consume data.
//
// Row-major unravel: dim 0 is outermost (slowest-varying), dim rank-1 is
// innermost (fastest-varying, stride 1 for contig). Iterating from innermost
// to outermost = dividing out inner sizes first.
struct StridedOffsetCalc {
    int     rank;
    int64_t sizes[FLAME_MAX_DIMS];
    int64_t strides[FLAME_MAX_DIMS];   // element strides, NOT byte strides
    int64_t base_offset;               // element offset added to every access

    // Linear-thread-index → element offset into the underlying storage.
    // Equivalent to PyTorch OffsetCalculator::get() but for a single arg
    // and with plain divmod.
    __host__ __device__ __forceinline__
    int64_t get(int64_t linear_idx) const {
        int64_t off = base_offset;
        // Fixed iteration count so nvcc can fully unroll; `continue` prunes
        // the tail for smaller ranks without a data-dependent branch.
        #pragma unroll
        for (int i = FLAME_MAX_DIMS - 1; i >= 0; --i) {
            if (i >= rank) continue;
            int64_t dim = sizes[i];
            int64_t idx_i = linear_idx % dim;
            linear_idx /= dim;
            off += idx_i * strides[i];
        }
        return off;
    }
};

// Elementwise kernel: strided input → contiguous output. Per-thread: read one
// element via `in_calc.get(tid)`, apply the functor, write to `y[tid]`.
//
// Mirrors the non-contig branch of PyTorch's gpu_kernel_impl_nocast
// (aten/src/ATen/native/cuda/CUDALoops.cuh ~L664) but specialised for the
// common "one strided input, one contig output" case we hit first on the
// migration path.
template <typename InT, typename OutT, typename Op>
__global__ void flame_elementwise_strided_to_contig(
    const InT* __restrict__ x_base,
    OutT*       __restrict__ y,
    int64_t              n_elements,
    StridedOffsetCalc    in_calc,
    Op                   op)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;
    int64_t in_off = in_calc.get(tid);
    y[tid] = op(x_base[in_off]);
}

// Host-side launch wrapper. Templating is here (not in the FFI surface) so
// each elementwise `.cu` file can instantiate the launcher with its own op
// functor; the `extern "C"` wrapper that Rust calls stays concrete.
template <typename InT, typename OutT, typename Op>
inline cudaError_t launch_elementwise_strided_to_contig(
    const InT*        x_base,
    OutT*             y,
    int64_t           n_elements,
    const StridedOffsetCalc& in_calc,
    Op                op,
    cudaStream_t      stream)
{
    if (n_elements <= 0) return cudaSuccess;
    const int threads = 256;
    int64_t blocks = (n_elements + threads - 1) / threads;
    // 32-bit grid limit is 2^31 - 1 on all our target arches; n_elements up
    // to ~5.5e11 is fine at threads=256. Any real DL tensor is well below.
    flame_elementwise_strided_to_contig<InT, OutT, Op>
        <<<(unsigned int)blocks, threads, 0, stream>>>(x_base, y, n_elements, in_calc, op);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Binary path (session 4, 2026-04-22): two strided inputs, contig output.
//
// Mirrors the non-contig branch of PyTorch's `gpu_kernel_impl_nocast` for
// binary ops — two input OffsetCalculators, one linear output. Same-shape
// only for now; broadcast (different-shape inputs with stride=0 on
// broadcasted dims) is trivially representable in `StridedOffsetCalc` but
// is scope-deferred to a later session.
// ---------------------------------------------------------------------------

template <typename InT, typename OutT, typename Op>
__global__ void flame_elementwise_strided_binary_to_contig(
    const InT* __restrict__ a_base,
    const InT* __restrict__ b_base,
    OutT*       __restrict__ y,
    int64_t              n_elements,
    StridedOffsetCalc    a_calc,
    StridedOffsetCalc    b_calc,
    Op                   op)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;
    int64_t a_off = a_calc.get(tid);
    int64_t b_off = b_calc.get(tid);
    y[tid] = op(a_base[a_off], b_base[b_off]);
}

template <typename InT, typename OutT, typename Op>
inline cudaError_t launch_elementwise_strided_binary_to_contig(
    const InT*        a_base,
    const InT*        b_base,
    OutT*             y,
    int64_t           n_elements,
    const StridedOffsetCalc& a_calc,
    const StridedOffsetCalc& b_calc,
    Op                op,
    cudaStream_t      stream)
{
    if (n_elements <= 0) return cudaSuccess;
    const int threads = 256;
    int64_t blocks = (n_elements + threads - 1) / threads;
    flame_elementwise_strided_binary_to_contig<InT, OutT, Op>
        <<<(unsigned int)blocks, threads, 0, stream>>>(
            a_base, b_base, y, n_elements, a_calc, b_calc, op);
    return cudaGetLastError();
}

}}  // namespace flame::iter
