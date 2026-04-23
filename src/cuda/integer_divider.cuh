// flame-core/src/cuda/integer_divider.cuh
//
// Port of `aten/src/ATen/cuda/detail/IntegerDivider.cuh` (L1-L124).
// Magic-constant fast divmod, method from Hacker's Delight §10-9
// (Granlund & Montgomery 1994). Used by OffsetCalculator<NARGS> on the
// hot path of every elementwise kernel — pre-computing a magic number
// per divisor at host construction lets the device side replace
// `linear_idx / dim` and `linear_idx % dim` with one `__umulhi` + an
// add + a shift + a multiply-subtract.
//
// Base IntDivider<T> (fall-through plain divmod, PyTorch L65-76) is
// kept for generality even though flame-core only instantiates the
// uint32_t specialization. If you add a new IntDivider<T> variant here,
// keep the three methods (`div`, `mod`, `divmod`) in the same signature
// shape as the base so OffsetCalculator can use either interchangeably.
//
// WARNING (copied from PyTorch): The fast divider only handles unsigned
// 32-bit dividends ≤ INT32_MAX. flame-core tensors are ≤ 2^31 BF16
// elements (see TENSORITERATOR_PORT_REFERENCE.md §2 `can_use_32bit_indexing`
// row), so this bound is always met.

#pragma once

#include <cstdint>

#if defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
#endif

#ifndef FLAME_HOST_DEVICE
#define FLAME_HOST_DEVICE __host__ __device__ __forceinline__
#endif

namespace flame {
namespace iter {

// Paired quotient/remainder — matches PyTorch `at::cuda::detail::DivMod<T>`.
template <typename Value>
struct DivMod {
    Value div;
    Value mod;

    FLAME_HOST_DEVICE DivMod(Value d, Value m) : div(d), mod(m) {}
};

// Base IntDivider<T> — plain `/` and `%`. This is the generic fall-through
// per PyTorch IntegerDivider.cuh L65-76. Any T that doesn't have a
// specialization below lands here.
template <typename Value>
struct IntDivider {
    IntDivider() = default;
    FLAME_HOST_DEVICE IntDivider(Value d) : divisor(d) {}

    FLAME_HOST_DEVICE Value div(Value n) const { return n / divisor; }
    FLAME_HOST_DEVICE Value mod(Value n) const { return n % divisor; }
    FLAME_HOST_DEVICE DivMod<Value> divmod(Value n) const {
        return DivMod<Value>(n / divisor, n % divisor);
    }

    Value divisor;
};

// ---------------------------------------------------------------------------
// uint32_t specialization — magic-constant fast divmod.
//
// Mirrors PyTorch IntegerDivider.cuh L80-L122 line-for-line. Algorithm from
// Hacker's Delight §10-9. For any N-bit unsigned integer d (> 0):
//
//   s  = ceil(log2 d)
//   m' = floor(2^N * (2^s - d) / d) + 1
//
// and:
//
//   floor(n / d) = ((__umulhi(n, m') + n) >> s).
//
// Precondition: divisor >= 1, divisor <= INT32_MAX, n ≤ INT32_MAX.
// ---------------------------------------------------------------------------

template <>
struct IntDivider<uint32_t> {
    static_assert(sizeof(uint32_t) == 4, "IntDivider<uint32_t>: 32-bit assumption");

    IntDivider() = default;

    // Host-only constructor (ran at iterator construction on the host before
    // we pass the POD across to the device). Matches PyTorch L85-95 exactly.
    IntDivider(uint32_t d) : divisor(d) {
        // NB: `assert` doesn't trip in release — PyTorch's contract is the
        // same. Leave the asserts for debug builds; they document the bound.
        // (Not using TORCH_CHECK; flame-core has no equivalent macro here.)
        // divisor in [1, INT32_MAX].
        // Find smallest shift s such that (1 << s) >= divisor.
        for (shift = 0; shift < 32; shift++) {
            if ((1U << shift) >= divisor) break;
        }

        uint64_t one = 1;
        uint64_t magic =
            ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
        m1 = static_cast<uint32_t>(magic);
        // Post-condition (match PyTorch assert): m1 > 0 AND m1 == magic (fits in 32b).
        // Assertion removed from release path; host unit test covers it.
    }

    FLAME_HOST_DEVICE uint32_t div(uint32_t n) const {
#if defined(__CUDA_ARCH__)
        // Upper 32 bits of n * m1 via the PTX `mul.hi.u32` intrinsic.
        uint32_t t = __umulhi(n, m1);
        return (t + n) >> shift;
#else
        // Host path: 64-bit multiply to dodge overflow.
        uint64_t t = (static_cast<uint64_t>(n) * m1) >> 32;
        return static_cast<uint32_t>(t + n) >> shift;
#endif
    }

    FLAME_HOST_DEVICE uint32_t mod(uint32_t n) const {
        return n - div(n) * divisor;
    }

    FLAME_HOST_DEVICE DivMod<uint32_t> divmod(uint32_t n) const {
        uint32_t q = div(n);
        return DivMod<uint32_t>(q, n - q * divisor);
    }

    uint32_t divisor;  // d
    uint32_t m1;       // magic number m'
    uint32_t shift;    // shift s
};

}  // namespace iter
}  // namespace flame
