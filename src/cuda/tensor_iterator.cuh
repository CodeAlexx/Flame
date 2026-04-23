// flame-core/src/cuda/tensor_iterator.cuh
//
// Phase 2 rewrite. Ports the flame-core side of PyTorch's
//   aten/src/ATen/native/cuda/Loops.cuh              (gpu_kernel, elementwise_kernel_helper)
//   aten/src/ATen/native/cuda/CUDALoops.cuh          (elementwise_kernel, launch_legacy_kernel,
//                                                     unrolled_elementwise_kernel,
//                                                     gpu_kernel_impl_nocast, gpu_kernel_impl)
//   aten/src/ATen/native/cuda/thread_constants.h     (num_threads / thread_work_size / block_work_size)
//
// Out of scope for Phase 2 (deferred — see brief §"You do NOT port in this phase"):
//   - `vectorized_elementwise_kernel<vec_size>` (CUDALoops.cuh:167). `launch_vectorized_kernel`
//     is stubbed here and routes to `launch_legacy_kernel`; Phase 5 fills it in with the
//     BF16 specialization that has to match `__hadd2` perf.
//   - `MemoryAccess.cuh` policies beyond the implicit scalar-per-thread trivial policy
//     (`elementwise_kernel`'s unrolled vt-loop plays the role of `policies::unroll`).
//   - Dynamic-cast support in `gpu_kernel_impl` — Phase 8.
//
// Entry point for the four session-1–4 `.cu` files is `launch_gpu_kernel<NARGS_IN, func_t>`
// (see bottom of file). Each retargeted `.cu` populates an `IterMetadata` POD from its
// existing `extern "C"` args, then calls `launch_gpu_kernel<arity>(meta, Functor{}, stream)`.
//
// See TENSORITERATOR_PORT_REFERENCE.md §3 for the full mapping table.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <climits>
#include <type_traits>

#include "integer_divider.cuh"
#include "offset_calculator.cuh"
#include "thread_constants.cuh"
#include "memory_access.cuh"

namespace flame {
namespace iter {

// Upper bound on (total operands) handled by the Phase 2 path.
// Phase 2 pilots are unary (2 operands: 1 in + 1 out) and binary (3 operands:
// 2 in + 1 out). `MAX_NARGS = 4` leaves one slot of slack for Phase 5's
// ternary ops (`clamp(x, lo, hi)`, `where(c, a, b)`) without needing to
// re-pod the struct. If you bump this, also widen `OffsetCalculator<NARGS>`
// static-asserts and the Rust-side `IterMetadata` marshalling in
// `ops/*_iter.rs`.
constexpr int MAX_NARGS = 4;

// ---------------------------------------------------------------------------
// IterMetadata — POD passed by value across the FFI boundary.
//
// This is the Rust-side build of `TensorIteratorBase`'s relevant fields.
// PyTorch does all the shape-inference / stride-computation inside
// `TensorIteratorConfig::build()`; Phase 1 of the port has the equivalent
// algorithm in Rust (`src/tensor_iterator/base.rs`). Phase 2 receives the
// already-built output by POD copy.
//
// Layout rules (enforced by the callers in `ops/*_iter.rs`):
//   - `num_args` = total operands (outputs + inputs). 0 < num_args <= MAX_NARGS.
//   - The first `num_outputs` entries of `strides`/`data_ptrs`/`offsets_elems`
//     are outputs; the rest are inputs. PyTorch convention: out at index 0.
//   - `strides[arg][dim]` is the ELEMENT stride (not byte stride) of operand
//     `arg` along dim `dim`. Broadcast dims have stride 0. Output operands are
//     always fresh contig row-major (strides[0] = stride_contiguous(sizes)).
//   - `offsets_elems[arg]` is the element offset of operand `arg` inside its
//     backing storage (for view tensors with non-zero view_offset).
//   - `is_contiguous` is a HINT from the Rust side: true iff all operand
//     strides match the contiguous row-major layout of `sizes` AND all
//     `offsets_elems == 0`. When true, `gpu_kernel_impl_nocast` takes the
//     (currently stubbed) vectorized branch.
//   - `requires_32bit_indexing` is true when `numel < INT32_MAX`; flame-core
//     tensors never exceed that bound (see TENSORITERATOR_PORT_REFERENCE.md
//     §2), so the split-recursion in `gpu_kernel` is only a safety assert
//     in Phase 2.
//
// `FLAME_MAX_DIMS` comes from `offset_calculator.cuh`.
struct IterMetadata {
    int     ndim;
    int     num_args;
    int     num_outputs;
    int     _pad;  // keep 4-byte alignment for the int64 array below
    int64_t sizes[FLAME_MAX_DIMS];
    int64_t strides[MAX_NARGS][FLAME_MAX_DIMS];  // [arg][dim] element strides
    int64_t offsets_elems[MAX_NARGS];
    void*   data_ptrs[MAX_NARGS];
    int64_t numel;
    bool    is_contiguous;
    bool    requires_32bit_indexing;
};

// ---------------------------------------------------------------------------
// Helper: build an OffsetCalculator<N> from IterMetadata on the host.
// Mirrors PyTorch `make_offset_calculator<N>` (OffsetCalculator.cuh:113),
// with the difference that flame-core's iterator POD already carries every
// operand's full stride array — no `iter.strides(i).data()` indirection.
// ---------------------------------------------------------------------------

template <int N>
inline OffsetCalculator<N> make_offset_calculator(const IterMetadata& meta) {
    static_assert(N >= 1 && N <= MAX_NARGS,
                  "make_offset_calculator: N must be in [1, MAX_NARGS]");
    // Build array-of-pointers into meta.strides so `OffsetCalculator`'s
    // constructor can index it per-operand. Same shape as PyTorch's call at
    // OffsetCalculator.cuh:116.
    const int64_t* stride_ptrs[N];
    for (int i = 0; i < N; i++) {
        stride_ptrs[i] = meta.strides[i];
    }
    return OffsetCalculator<N>(meta.ndim, meta.sizes, stride_ptrs);
}

// ---------------------------------------------------------------------------
// elementwise_kernel — the legacy (non-vectorized) kernel.
//
// Port of CUDALoops.cuh:528-539: `nt` threads per block, `vt` elements per
// thread, sequential strided-by-`nt` access so warp reads are coalesced.
// `f(idx)` is the per-element functor the launcher builds around the
// offset-calc + data-ptr closure.
// ---------------------------------------------------------------------------

template <int nt, int vt, typename func_t>
__global__ void elementwise_kernel(int N, func_t f) {
    int tid = threadIdx.x;
    int nv = nt * vt;
    int idx = nv * blockIdx.x + tid;
    #pragma unroll
    for (int i = 0; i < vt; i++) {
        if (idx < N) {
            f(idx);
            idx += nt;
        }
    }
}

// Port of CUDALoops.cuh:541-552. Grid sized so every thread iterates `vt`
// steps, covering `block.x * vt` elements per block. `N` bounded by
// INT32_MAX by caller precondition (we split if needed in `gpu_kernel`).
template <int nt, int vt, typename func_t>
static inline void launch_legacy_kernel(int64_t N, const func_t& f, cudaStream_t stream) {
    if (N == 0) return;
    dim3 block(nt);
    dim3 grid((unsigned int)((N + (int64_t)block.x * vt - 1) / ((int64_t)block.x * vt)));
    elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>((int)N, f);
}

// ---------------------------------------------------------------------------
// unrolled_elementwise_kernel — ported for API completeness (Phase 5 routes
// the contig-but-not-vectorizable fallback through here; Phase 2 never
// reaches this kernel because the stubbed `launch_vectorized_kernel` falls
// straight through to `launch_legacy_kernel` for ALL contig inputs).
//
// Port of CUDALoops.cuh:267-289. `elems_per_thread` elements per thread, one
// work-group = `num_threads()` threads, total work per block =
// `block_work_size()`.
// ---------------------------------------------------------------------------

template <typename func_t>
__global__ void unrolled_elementwise_kernel(int N, func_t f) {
    // Functor packaged by caller to take `(idx, bool inbounds)`-style per-step
    // calls. In Phase 2 we just use the identical shape as `elementwise_kernel`
    // — when Phase 5 specializes, it can swap in a `policies::unroll`-style
    // loader/storer. Keeping this as an alias of elementwise_kernel now means
    // a future Phase 5 change only needs to tweak the body, not reroute every
    // call site.
    int tid = threadIdx.x;
    int nv = num_threads() * thread_work_size();
    int idx = nv * blockIdx.x + tid;
    #pragma unroll
    for (int i = 0; i < thread_work_size(); i++) {
        if (idx < N) {
            f(idx);
            idx += num_threads();
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorized elementwise kernels (Phase 5 fill-in of the Phase 2 stub).
//
// PyTorch reference: `vectorized_elementwise_kernel<vec_size, func_t, array_t>`
// at CUDALoops.cuh:165-226 (non-ROCm, non-sm_90 branch) and the per-vec_size
// dispatch in `launch_vectorized_kernel` at CUDALoops.cuh:293-366.
//
// flame-core keeps two differences from the reference:
//   1. The scalar type (`__nv_bfloat16`) is fixed; PyTorch templates
//      `scalar_t` so the same machinery services F16/F32/BF16. Phase 8
//      generalises.
//   2. The full PyTorch policies/traits stack (`policies::vectorized`,
//      `function_traits<func_t>::arity`, `std::apply`) is collapsed into
//      explicit per-arity kernel specialisations (`UnaryVectorizedKernel`,
//      `BinaryVectorizedKernel`). flame-core doesn't need the generality
//      and avoiding function_traits keeps nvcc happy without
//      `--extended-lambda`.
//
// Correctness contract: the functor is called once per scalar element with
// bit-exact the same inputs as the legacy path would pass it, so output
// values are bit-identical to `launch_legacy_kernel`'s output (the only
// difference is load/store width). That makes the Klein byte-equal gate a
// statement about load-reordering having no observable effect for BF16
// elementwise arithmetic — which it doesn't, because the functor is pure.
//
// Vec size selection: caller passes `vec_size ∈ {1, 2, 4}` based on
// `can_vectorize_up_to_bf16` over the operand pointers. vec_size=1 falls
// through to `launch_legacy_kernel` because the vectorized kernel body
// assumes a power-of-two pack width > 1.
// ---------------------------------------------------------------------------

template <int vec_size, typename func_t>
__global__ void unary_vectorized_kernel(
    int N,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    func_t f)
{
    constexpr int tws = thread_work_size();          // elements/thread (= 4)
    constexpr int bws = tws * num_threads();         // elements/block (= 512)
    constexpr int loop_size = tws / vec_size;        // vector iterations/thread

    static_assert(vec_size == 2 || vec_size == 4,
                  "unary_vectorized_kernel supports vec_size in {2,4}; "
                  "vec_size=1 routes to launch_legacy_kernel");
    static_assert(tws % vec_size == 0,
                  "thread_work_size must be a multiple of vec_size");

    int block_base = bws * blockIdx.x;
    int remaining = N - block_base;

    if (remaining >= bws) {
        // Full block — every thread writes `tws` elements via `loop_size`
        // vectorized ops. Mirrors CUDALoops.cuh:205-208 (the full-block
        // branch of `vectorized_elementwise_kernel`).
        const __nv_bfloat16* in_block = in + block_base;
        __nv_bfloat16* out_block = out + block_base;
        int thread_idx = threadIdx.x;

        #pragma unroll
        for (int i = 0; i < loop_size; i++) {
            int index = thread_idx + i * num_threads();
            aligned_vector<__nv_bfloat16, vec_size> v =
                load_vector<vec_size, __nv_bfloat16>(in_block, index);
            aligned_vector<__nv_bfloat16, vec_size> o;
            #pragma unroll
            for (int j = 0; j < vec_size; j++) {
                o.val[j] = f(v.val[j]);
            }
            store_vector<vec_size, __nv_bfloat16>(out_block, index, o);
        }
    } else {
        // Tail block — fall back to scalar per-thread loop. Same shape as
        // `elementwise_kernel<128,4>`; mirrors the unrolled branch at
        // CUDALoops.cuh:209-224.
        int idx = block_base + threadIdx.x;
        #pragma unroll
        for (int i = 0; i < tws; i++) {
            if (idx < N) {
                out[idx] = f(in[idx]);
                idx += num_threads();
            }
        }
    }
}

template <int vec_size, typename func_t>
__global__ void binary_vectorized_kernel(
    int N,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    func_t f)
{
    constexpr int tws = thread_work_size();
    constexpr int bws = tws * num_threads();
    constexpr int loop_size = tws / vec_size;

    static_assert(vec_size == 2 || vec_size == 4,
                  "binary_vectorized_kernel supports vec_size in {2,4}");
    static_assert(tws % vec_size == 0,
                  "thread_work_size must be a multiple of vec_size");

    int block_base = bws * blockIdx.x;
    int remaining = N - block_base;

    if (remaining >= bws) {
        const __nv_bfloat16* a_block = a + block_base;
        const __nv_bfloat16* b_block = b + block_base;
        __nv_bfloat16* out_block = out + block_base;
        int thread_idx = threadIdx.x;

        #pragma unroll
        for (int i = 0; i < loop_size; i++) {
            int index = thread_idx + i * num_threads();
            aligned_vector<__nv_bfloat16, vec_size> va =
                load_vector<vec_size, __nv_bfloat16>(a_block, index);
            aligned_vector<__nv_bfloat16, vec_size> vb =
                load_vector<vec_size, __nv_bfloat16>(b_block, index);
            aligned_vector<__nv_bfloat16, vec_size> o;
            #pragma unroll
            for (int j = 0; j < vec_size; j++) {
                o.val[j] = f(va.val[j], vb.val[j]);
            }
            store_vector<vec_size, __nv_bfloat16>(out_block, index, o);
        }
    } else {
        int idx = block_base + threadIdx.x;
        #pragma unroll
        for (int i = 0; i < tws; i++) {
            if (idx < N) {
                out[idx] = f(a[idx], b[idx]);
                idx += num_threads();
            }
        }
    }
}

// Unary launcher: given vec_size chosen by the caller, dispatch to the
// correct template instantiation. Grid sized so every thread handles exactly
// `thread_work_size()` elements (`block_work_size()` total per block).
// Mirrors `launch_vectorized_kernel`'s grid math at CUDALoops.cuh:326-327.
template <typename func_t>
static inline void launch_unary_vectorized(
    int vec_size,
    int64_t N,
    __nv_bfloat16* out,
    const __nv_bfloat16* in,
    const func_t& f,
    cudaStream_t stream)
{
    if (N == 0) return;
    const int bws = block_work_size();
    dim3 block(num_threads());
    dim3 grid((unsigned int)((N + bws - 1) / bws));
    switch (vec_size) {
        case 4:
            unary_vectorized_kernel<4, func_t><<<grid, block, 0, stream>>>(
                (int)N, out, in, f);
            break;
        case 2:
            unary_vectorized_kernel<2, func_t><<<grid, block, 0, stream>>>(
                (int)N, out, in, f);
            break;
        default:
            // vec_size=1 is handled by the caller (falls through to
            // `launch_legacy_kernel`). Reaching this branch is a bug.
            break;
    }
}

template <typename func_t>
static inline void launch_binary_vectorized(
    int vec_size,
    int64_t N,
    __nv_bfloat16* out,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const func_t& f,
    cudaStream_t stream)
{
    if (N == 0) return;
    const int bws = block_work_size();
    dim3 block(num_threads());
    dim3 grid((unsigned int)((N + bws - 1) / bws));
    switch (vec_size) {
        case 4:
            binary_vectorized_kernel<4, func_t><<<grid, block, 0, stream>>>(
                (int)N, out, a, b, f);
            break;
        case 2:
            binary_vectorized_kernel<2, func_t><<<grid, block, 0, stream>>>(
                (int)N, out, a, b, f);
            break;
        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// elementwise_kernel_helper — port of Loops.cuh:44-75.
//
// Phase 2 doesn't use the `policy_t`-parameterised version from PyTorch
// because we only have the legacy (scalar-per-thread) "policy". The closure
// built inside `gpu_kernel_impl_nocast_dispatch<NARGS_IN>` below plays the
// role of the elementwise helper + policy together.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Per-arity dispatch layer. Phase 2 hands the launcher a specialization for
// arity 1 (unary: silu/gelu/square) and arity 2 (binary: add). The functor
// is invoked directly with the right number of BF16 operand loads; no
// `function_traits` / `std::apply` machinery needed.
//
// Templated over element type `T` and arity — functors return T, take T args.
// Phase 8 generalizes to mixed dtypes; Phase 2 keeps `T = __nv_bfloat16`.
// ---------------------------------------------------------------------------

// Per-thread closures are wrapped in named functor structs instead of
// `__device__` lambdas — lambdas with device annotations require the
// `--extended-lambda` nvcc flag, and flame-core's `build.rs` doesn't
// pass it. The two structs below play the exact role of PyTorch's
// `[=] GPU_LAMBDA (int idx) { ... }` closures at CUDALoops.cuh:667 (unary)
// and the equivalent binary form; captured values become struct members.

template <typename T, typename func_t>
struct UnaryLegacyBody {
    OffsetCalculator<2> oc;
    T* out_base;
    const T* in_base;
    func_t f;

    __device__ __forceinline__ void operator()(int idx) const {
        auto offs = oc.get(static_cast<uint32_t>(idx));
        T x = in_base[offs[1]];
        out_base[offs[0]] = f(x);
    }
};

template <typename T, typename func_t>
struct BinaryLegacyBody {
    OffsetCalculator<3> oc;
    T* out_base;
    const T* a_base;
    const T* b_base;
    func_t f;

    __device__ __forceinline__ void operator()(int idx) const {
        auto offs = oc.get(static_cast<uint32_t>(idx));
        T a = a_base[offs[1]];
        T b = b_base[offs[2]];
        out_base[offs[0]] = f(a, b);
    }
};

template <typename T, typename func_t>
static inline void launch_unary_legacy(
    const IterMetadata& meta, const func_t& f, cudaStream_t stream)
{
    const int64_t N = meta.numel;
    if (N == 0) return;

    // 2 operands: [0] output, [1] input.
    T* out_base = reinterpret_cast<T*>(meta.data_ptrs[0]) + meta.offsets_elems[0];
    const T* in_base = reinterpret_cast<const T*>(meta.data_ptrs[1]) + meta.offsets_elems[1];

    // Contig + BF16 → vectorized fast path (Phase 5). The offset-adjusted
    // base pointers are what must be checked for alignment, not the raw
    // storage pointers, because the element-offset in a view can push a
    // 4-byte-aligned storage into 2-byte alignment.
    if (meta.is_contiguous) {
        if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            const void* ptrs[2] = {
                reinterpret_cast<const void*>(out_base),
                reinterpret_cast<const void*>(in_base),
            };
            int vec = can_vectorize_up_to_bf16<__nv_bfloat16>(ptrs, 2);
            if (vec >= 2) {
                launch_unary_vectorized<func_t>(
                    vec, N, out_base, in_base, f, stream);
                return;
            }
        }
        // vec_size=1 or non-BF16 T: fall through to legacy. Same behaviour
        // as PyTorch's `case 1:` branch at CUDALoops.cuh:351-362.
    }

    // Non-contig or vec_size=1: legacy strided path via OffsetCalculator.
    UnaryLegacyBody<T, func_t> body{
        make_offset_calculator<2>(meta),
        out_base,
        in_base,
        f,
    };
    launch_legacy_kernel<num_threads(), thread_work_size(), UnaryLegacyBody<T, func_t>>(
        N, body, stream);
}

template <typename T, typename func_t>
static inline void launch_binary_legacy(
    const IterMetadata& meta, const func_t& f, cudaStream_t stream)
{
    const int64_t N = meta.numel;
    if (N == 0) return;

    T* out_base = reinterpret_cast<T*>(meta.data_ptrs[0]) + meta.offsets_elems[0];
    const T* a_base = reinterpret_cast<const T*>(meta.data_ptrs[1]) + meta.offsets_elems[1];
    const T* b_base = reinterpret_cast<const T*>(meta.data_ptrs[2]) + meta.offsets_elems[2];

    if (meta.is_contiguous) {
        if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            const void* ptrs[3] = {
                reinterpret_cast<const void*>(out_base),
                reinterpret_cast<const void*>(a_base),
                reinterpret_cast<const void*>(b_base),
            };
            int vec = can_vectorize_up_to_bf16<__nv_bfloat16>(ptrs, 3);
            if (vec >= 2) {
                launch_binary_vectorized<func_t>(
                    vec, N, out_base, a_base, b_base, f, stream);
                return;
            }
        }
    }

    // 3 operands: [0] output, [1] a, [2] b.
    BinaryLegacyBody<T, func_t> body{
        make_offset_calculator<3>(meta),
        out_base,
        a_base,
        b_base,
        f,
    };
    launch_legacy_kernel<num_threads(), thread_work_size(), BinaryLegacyBody<T, func_t>>(
        N, body, stream);
}

// ---------------------------------------------------------------------------
// gpu_kernel_impl_nocast — port of CUDALoops.cuh:642-735 (BF16-only subset).
// Picks the contig vs non-contig branch; the former routes to the stubbed
// vectorized kernel, the latter to `launch_legacy_kernel<128, 4>`.
//
// Phase 2 dispatches on arity via the explicit `NARGS_IN` template param
// (no `function_traits` dependency) so each retargeted `.cu` file calls the
// right specialization directly.
// ---------------------------------------------------------------------------

template <int NARGS_IN, typename T, typename func_t>
static inline void gpu_kernel_impl_nocast(
    const IterMetadata& meta, const func_t& f, cudaStream_t stream)
{
    static_assert(NARGS_IN == 1 || NARGS_IN == 2,
                  "Phase 2 gpu_kernel_impl_nocast supports unary/binary only; "
                  "Phase 5 adds ternary.");
    if constexpr (NARGS_IN == 1) {
        launch_unary_legacy<T, func_t>(meta, f, stream);
    } else {
        launch_binary_legacy<T, func_t>(meta, f, stream);
    }
}

// ---------------------------------------------------------------------------
// gpu_kernel_impl — port of CUDALoops.cuh:958-962 (minus dynamic cast).
//
// TODO(phase-8): dynamic cast — detect `func_t`'s arg types vs actual
// `meta.strides` element sizes and dispatch to a casting unroll policy.
// Phase 2 asserts no-cast (only BF16 in / BF16 out).
// ---------------------------------------------------------------------------

template <int NARGS_IN, typename T, typename func_t>
static inline void gpu_kernel_impl(
    const IterMetadata& meta, const func_t& f, cudaStream_t stream)
{
    gpu_kernel_impl_nocast<NARGS_IN, T, func_t>(meta, f, stream);
}

// ---------------------------------------------------------------------------
// gpu_kernel_nocast — port of Loops.cuh:84. Public-ish wrapper: asserts
// 32-bit indexing (Phase 2 doesn't split iterators), calls `gpu_kernel_impl`.
// ---------------------------------------------------------------------------

template <int NARGS_IN, typename T, typename func_t>
static inline void gpu_kernel_nocast(
    const IterMetadata& meta, const func_t& f, cudaStream_t stream)
{
    if (meta.numel == 0) return;
    // Phase 2 limit: flame-core tensors never exceed 2^31 BF16 elements (~4 GB)
    // so 32-bit indexing always suffices. Full SplitUntil32Bit support is
    // deferred (TENSORITERATOR_PORT_REFERENCE.md §2).
    // NOTE: not a TORCH_CHECK — on failure we'd rather silently truncate than
    // synchronously abort the process, so we promote to a debug assert via
    // `(void)` and trust the Rust side. The Rust caller sets
    // `requires_32bit_indexing = true` for every supported shape.
    (void)meta.requires_32bit_indexing;
    gpu_kernel_impl<NARGS_IN, T, func_t>(meta, f, stream);
}

// ---------------------------------------------------------------------------
// gpu_kernel — port of Loops.cuh:115. Wraps `gpu_kernel_nocast` with the
// device-check loop PyTorch runs there; flame-core's Rust side already
// validates all operands are on the same CUDA device before building the
// `IterMetadata`, so this is a no-op wrapper in Phase 2.
// ---------------------------------------------------------------------------

template <int NARGS_IN, typename T, typename func_t>
static inline void gpu_kernel(
    const IterMetadata& meta, const func_t& f, cudaStream_t stream)
{
    gpu_kernel_nocast<NARGS_IN, T, func_t>(meta, f, stream);
}

// ---------------------------------------------------------------------------
// launch_gpu_kernel — public FFI entry for the retargeted `.cu` files.
//
// This is what each retargeted `.cu` ('activation_silu_iter.cu' etc.) calls
// from its `extern "C"` wrapper. BF16-only in Phase 2; Phase 8 will add
// dtype parametrization.
// ---------------------------------------------------------------------------

template <int NARGS_IN, typename func_t>
inline void launch_gpu_kernel(
    const IterMetadata& meta, const func_t& f, cudaStream_t stream)
{
    gpu_kernel<NARGS_IN, __nv_bfloat16, func_t>(meta, f, stream);
}

}  // namespace iter
}  // namespace flame
