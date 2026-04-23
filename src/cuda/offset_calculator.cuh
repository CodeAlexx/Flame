// flame-core/src/cuda/offset_calculator.cuh
//
// Port of `aten/src/ATen/cuda/detail/OffsetCalculator.cuh` (L21-L136),
// restricted to flame-core's Shape capacity (6 dims, not PyTorch's 25).
//
// Given a linear thread index, returns one element offset per operand
// (NARGS inputs + outputs baked into the same struct). The sizes are wrapped
// as `IntDivider<uint32_t>` for magic-constant fast divmod; strides are
// element strides (not byte strides — flame-core doesn't need the
// byte-strides mode because every elementwise op we handle has uniform
// element sizes per operand, and the Rust side always passes element
// strides).
//
// See TENSORITERATOR_PORT_REFERENCE.md §3 for the PyTorch cross-ref table.

#pragma once

#include <cstdint>

#if defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
#endif

#include "integer_divider.cuh"

#ifndef FLAME_HOST_DEVICE
#define FLAME_HOST_DEVICE __host__ __device__ __forceinline__
#endif

namespace flame {
namespace iter {

// Matches flame-core's `Strides = SmallVec<[usize; 6]>` on the Rust side.
// Do NOT bump without also widening `Shape`'s inline capacity — the Rust
// → FFI marshalling paths assume 6.
constexpr int FLAME_MAX_DIMS = 6;

// Output type of `OffsetCalculator<NARGS>::get`. `std::array` is not
// reliably host-device available in all CUDA toolchains, so we use a plain
// POD. `offsets[NARGS]` stores one per-operand element offset into the
// respective storage base. For NARGS=0 we size to 1 to avoid the
// zero-length-array warning (matches PyTorch's `std::max<int>(NARGS, 1)`
// in OffsetCalculator.cuh L31).
template <int NARGS, typename index_t = uint32_t>
struct OffsetArray {
    static constexpr int SIZE = (NARGS > 0) ? NARGS : 1;
    index_t offsets[SIZE];

    FLAME_HOST_DEVICE index_t& operator[](int i) { return offsets[i]; }
    FLAME_HOST_DEVICE const index_t& operator[](int i) const { return offsets[i]; }
};

// ---------------------------------------------------------------------------
// OffsetCalculator<NARGS, index_t>
//
// Port of PyTorch OffsetCalculator.cuh:21-91 with flame-core's smaller
// MAX_DIMS. NARGS includes outputs (PyTorch convention: argument 0 is the
// output, arguments 1..NARGS-1 are inputs). See `make_offset_calculator`.
// ---------------------------------------------------------------------------

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
    static_assert(NARGS >= 1 && NARGS <= 8, "NARGS must be in [1, 8]");

    using offset_type = OffsetArray<NARGS, index_t>;

    // Host-side constructor. `sizes` points to `dims` int64_t size values.
    // `strides` is an array of NARGS pointers, each pointing to `dims`
    // int64_t element strides (one per operand). Mirrors PyTorch's
    // signature at OffsetCalculator.cuh L35.
    OffsetCalculator(int dims_, const int64_t* sizes, const int64_t* const* strides)
        : dims(dims_)
    {
        // Initialize every slot (even past `dims`) to safe defaults so an
        // accidental out-of-range access doesn't divide by zero on device.
        for (int i = 0; i < FLAME_MAX_DIMS; i++) {
            int64_t s = (i < dims) ? sizes[i] : 1;
            sizes_[i] = IntDivider<index_t>(static_cast<index_t>(s));
            for (int arg = 0; arg < NARGS; arg++) {
                int64_t str = (i < dims) ? strides[arg][i] : 0;
                strides_[i][arg] = static_cast<index_t>(str);
            }
        }
    }

    // Device side. Mirrors OffsetCalculator.cuh L46-L86: iterate dims,
    // divmod out the current dim's contribution, accumulate into offsets.
    // Unroll with #pragma so nvcc flattens the inner loops.
    FLAME_HOST_DEVICE offset_type get(index_t linear_idx) const {
        offset_type offsets;

        #pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }

        #pragma unroll
        for (int dim = 0; dim < FLAME_MAX_DIMS; dim++) {
            if (dim == dims) {
                break;
            }
            DivMod<index_t> dm = sizes_[dim].divmod(linear_idx);
            linear_idx = dm.div;

            #pragma unroll
            for (int arg = 0; arg < NARGS; arg++) {
                offsets[arg] += dm.mod * strides_[dim][arg];
            }
        }
        return offsets;
    }

    int dims;
    IntDivider<index_t> sizes_[FLAME_MAX_DIMS];
    // [dim][arg]: element strides (not byte strides).
    index_t strides_[FLAME_MAX_DIMS][NARGS];
};

// ---------------------------------------------------------------------------
// TrivialOffsetCalculator<NARGS> — identity offsets.
//
// Port of OffsetCalculator.cuh:94-110. Used on the vectorized path when all
// operands are contiguous and same-shape (offset == linear_idx for every
// operand). Phase 2 ports it for API completeness even though the
// `launch_vectorized_kernel` path that actually uses it is stubbed in
// Phase 2 (routes to legacy).
// ---------------------------------------------------------------------------

template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
    using offset_type = OffsetArray<NARGS, index_t>;

    FLAME_HOST_DEVICE offset_type get(index_t linear_idx) const {
        offset_type offsets;
        #pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = linear_idx;
        }
        return offsets;
    }
};

}  // namespace iter
}  // namespace flame
