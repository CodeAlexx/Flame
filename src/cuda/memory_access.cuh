// flame-core/src/cuda/memory_access.cuh
//
// Phase 5: narrow port of the pieces of
// `aten/src/ATen/native/cuda/MemoryAccess.cuh` that the flame-core
// vectorized-contig path actually needs.
//
// Scope deliberately kept small:
//   - `aligned_vector<T, N>` POD — the thing that turns an N-wide load into
//     a single 32-bit / 64-bit instruction. L180-L184 of the reference.
//   - `load_vector<N>(base, index)` / `store_vector<N>(base, index, value)`
//     wrappers that do the reinterpret_cast for you. L186-L215 of the
//     reference (the non-ROCm branch).
//   - `can_vectorize_up_to<T>(ptr)` — runtime alignment query. L508-L533
//     of the reference (non-ROCm branch; no AMD gfx942 16-wide path).
//
// Not ported:
//   - policies::vectorized / policies::unroll structs — flame-core's contig
//     path is simple enough (offsets == linear index) that we inline the
//     per-thread loop directly in `vectorized_elementwise_kernel` in
//     `tensor_iterator.cuh`. See TENSORITERATOR_PORT_REFERENCE.md §3 L125
//     ("Phase 5 adds VectorizedPolicy<N> for BF16 pair-vec" — the functional
//     equivalent is inlined rather than templated into a policy struct).
//   - vec_size=8 path — needs sm_90+; our targets are sm_80/86/89.
//   - Heterogeneous-type vectorized_templated — Phase 8 dtype promotion
//     territory; not in Phase 5.
//
// For Phase 5 we only dispatch vec_size ∈ {1, 2, 4}. BF16 (sizeof=2) with
// vec_size=2 matches the old `__nv_bfloat162` pair-vec flat kernel's load
// pattern (a single 4-byte read) and is the critical case for the perf gate.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flame {
namespace iter {

// Port of MemoryAccess.cuh L181-L184. Alignment is precisely `sizeof(T) * N`
// so a pointer to `aligned_vector<T, N>` is guaranteed to produce a single
// `ld.global.vN` on CUDA. Don't change this to a power-of-two alignment —
// the whole point is that the compiler picks the widest load instruction
// that matches the struct size.
template <typename T, int N>
struct alignas(sizeof(T) * N) aligned_vector {
    T val[N];
};

// Port of MemoryAccess.cuh L186-L215 (non-ROCm branch). Reinterprets
// `base_ptr` as a pointer-to-`aligned_vector<T, N>`, indexes it, returns by
// value. On BF16 + N=2 this compiles to a `LDG.32` (single 32-bit load);
// on BF16 + N=4 it's `LDG.64`.
template <int N, typename T>
__device__ __forceinline__ aligned_vector<T, N>
load_vector(const T* base_ptr, uint32_t offset) {
    using vec_t = aligned_vector<T, N>;
    auto from = reinterpret_cast<const vec_t*>(base_ptr);
    return from[offset];
}

// Paired store. Mirrors the store half of `policies::vectorized::store` at
// MemoryAccess.cuh L353-L368.
template <int N, typename T>
__device__ __forceinline__ void
store_vector(T* base_ptr, uint32_t offset, const aligned_vector<T, N>& v) {
    using vec_t = aligned_vector<T, N>;
    auto to = reinterpret_cast<vec_t*>(base_ptr);
    to[offset] = v;
}

// Port of MemoryAccess.cuh L508-L533 (non-ROCm branch). Returns the widest
// power-of-two vec_size ≤ 4 that the pointer is aligned to.
//
// Phase 5 caps at 4 intentionally: vec_size=8 needs sm_90+ and a separate
// kernel instantiation per PyTorch L316-L318, and flame-core's hardware
// targets are sm_80/86/89.
template <typename scalar_t>
__host__ __device__ __forceinline__ int
can_vectorize_up_to_single(const void* pointer) {
    uint64_t address = reinterpret_cast<uint64_t>(pointer);
    constexpr int vec2_alignment = alignof(aligned_vector<scalar_t, 2>);
    constexpr int vec4_alignment = alignof(aligned_vector<scalar_t, 4>);
    if (address % vec4_alignment == 0) {
        return 4;
    }
    if (address % vec2_alignment == 0) {
        return 2;
    }
    return 1;
}

// Multi-pointer variant. Returns the min of `can_vectorize_up_to_single`
// across `npointers` data pointers. Port of the `can_vectorize_up_to<func_t>`
// helper at MemoryAccess.cuh L551-L562, restricted to the "same scalar type
// across all operands" case (Phase 5 is BF16-only for both inputs and
// output).
//
// `pointers[0]` is the output, `pointers[1..npointers]` are inputs, matching
// the PyTorch layout.
template <typename scalar_t>
inline int can_vectorize_up_to_bf16(const void* const* pointers, int npointers) {
    int result = 4;
    for (int i = 0; i < npointers; i++) {
        int p = can_vectorize_up_to_single<scalar_t>(pointers[i]);
        if (p < result) {
            result = p;
        }
    }
    return result;
}

}  // namespace iter
}  // namespace flame
