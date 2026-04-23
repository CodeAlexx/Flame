# TensorIterator Port Reference

Line-trace mapping between PyTorch's elementwise-dispatch infrastructure
and flame-core's current state + migration target. Produced by Phase 0 of
the plan at `/home/alex/.claude/plans/plan-this-and-fix-encapsulated-hennessy.md`;
lives until Phase 11 deletes it.

Every phase's Builder agent reads this before writing code. Every
Skeptic agent cross-references this against the Builder's output. When
something in flame-core contradicts this doc, the doc is the source of
truth (and any divergence needs a commit message + a doc update).

---

## 1. Executive summary

### Where flame-core is parallel to PyTorch today

- Two build pipelines: NVRTC (runtime-compiled, in `.rs` const strings)
  + `cc-rs/nvcc` (build-time `.cu` files). Mirrors aten's `native/` vs
  `native/cuda/` split, though flame-core has the mix inverted relative
  to PyTorch.
- BF16 storage via `__nv_bfloat16` pair-vectorized kernels in
  `bf16_ops.rs` (silu/gelu/square use `__nv_bfloat162` 2-elem/thread) —
  parallels aten's `at::BFloat16` vectorized lowering.
- cuDNN Frontend v9 SDPA path at `src/cuda/cudnn_sdpa.cpp` — parallels
  aten's FlashAttention + cuDNN integration, though flame-core also
  has two legacy SDPA paths that PyTorch doesn't.
- Sessions 1–4 have a minimal TensorIterator-style scaffolding at
  `src/cuda/tensor_iterator.cuh` (NARGS=1 + NARGS=2 launchers with
  `StridedOffsetCalc` for offset compute). Direct descendant of
  `aten/src/ATen/cuda/detail/OffsetCalculator.cuh` at ~1% feature
  coverage.

### Where flame-core diverges

- **4 parallel BF16 elementwise dispatch paths** (`bf16_elementwise`,
  `bf16_ops` elementwise, `cuda_ops_bf16` elementwise,
  `ops/*_iter.rs`). PyTorch has one (`Dispatcher` → stub →
  `gpu_kernel(iter, lambda)`). This is the core problem the plan
  targets.
- Ad hoc fast-path gates like `bf16_elementwise::shapes_equal_no_broadcast`
  (`src/bf16_elementwise.rs:712`). PyTorch branches on
  `iter.is_contiguous()`, never on shape equality alone. (Session-4
  Pre-Phase-0 fixed this gate's contiguity check as a stop-gap;
  Phase 5 deletes the gate entirely.)
- Fragmented `TensorStorage` enum (BF16, BF16Arena, BF16View, F32,
  F32Arena, F32View, …). PyTorch has one `c10::Storage` + `Tensor
  = (Storage, offset, sizes, strides, dtype)`. The enum fragmentation
  is what forces every op to match-branch on dtype inside the hot
  path. Plan does NOT unify this in Phases 1–11 (out-of-scope; the
  port accepts the enum with a match at the iterator boundary).
- No DispatchKey / `REGISTER_DISPATCH` registration pattern. Every
  `Tensor::<op>` has its own dtype branching in `tensor.rs`. Phase 3
  ports this as Rust `OnceLock`-based registry.
- 10 BF16 ops currently do F32 round-trip on the BF16 hot path
  (sigmoid, tanh, exp, neg, log, sqrt, rsqrt, recip, mul_scalar,
  add_scalar). Violates CLAUDE.md's hard rule "NEVER use F32
  fallbacks in inference code." Phases 6–7 give all of these native
  BF16 paths.

### Blast radius

Full port target: flame-core's `src/bf16_elementwise.rs` (1167 LoC,
~900 of them elementwise), elementwise subsets of `src/bf16_ops.rs`
(~730 of 1706 LoC) and `src/cuda_ops_bf16.rs` (~250 of 2094 LoC),
and `src/ops/{silu,gelu,square,add}_iter.rs` (~465 LoC total) all
collapse into `src/tensor_iterator/` + `src/cuda/{unary,binary,cmp}/`
+ thin `Tensor::<op>` callers. Net deletion target **~2,300 LoC**
across 12 phases.

---

## 2. `TensorIteratorConfig` + `TensorIteratorBase` mapping (Phase 1)

| PyTorch entity | flame-core equivalent | Status | Port notes |
|---|---|---|---|
| `TensorIteratorConfig` (TensorIterator.h:783–993) | none | NEEDS-PORT (~300 LoC) | Fluent builder. Phase 1 ports `add_output`, `add_input`, `declare_static_dtype_and_device`, `check_all_same_dtype`, `allow_cpu_scalars`, `promote_inputs_to_common_dtype` (flag only in Phase 1; actual promotion in Phase 8). Lands at `src/tensor_iterator/config.rs`. |
| `TensorIteratorBase` (TensorIterator.h:248–734) | none | NEEDS-PORT (~500 LoC) | Central iterator struct. Holds `OperandInfo[]`, coalesced shape, rank, contiguity flag. Phase 1 ports the fields + accessors (`ndim`, `shape`, `numel`, `strides(arg)`, `dtype(arg)`, `is_contiguous`, `can_use_32bit_indexing`, `data_ptr(arg)`) but NOT the build pipeline — that's Phase 3. Lands at `src/tensor_iterator/base.rs`. |
| `OperandInfo` (TensorIterator.h:117–206) | implicit (`&Tensor` args + scattered metadata) | NEEDS-PORT (~80 LoC) | PyTorch has byte-strides, element-size, current/target dtype, owner flag. flame-core will use element-strides (match existing convention) + a separate `byte_strides()` accessor that multiplies by `sizeof(dtype)`. |
| `DimVector` (c10, 6-elem SmallVec) | `SmallVec<[usize; 6]>` at `src/shape.rs` | EXISTS | Same capacity, same small-vec semantics. flame-core uses `usize` where PyTorch uses `int64_t`; convert at FFI boundary. |
| Broadcast via stride=0 | `bf16_elementwise::make_broadcast_spec` at `src/bf16_elementwise.rs:22+` (BcSpec struct) | PARTIAL | flame-core already represents broadcast as stride=0 on broadcasted dims. `BcSpec` is the existing consumer. Phase 1 ports the computation into `TensorIteratorBase::compute_shape_and_strides` and the old `BcSpec` is deleted in Phase 5. |
| `coalesce_dimensions` (TensorIterator.cpp:1027+) | none | NEEDS-PORT (~120 LoC) | Merge adjacent dims where every operand's strides are compatible. PyTorch: reduces an `[N, 1, H, W]` tensor iterating sequentially to effective rank-2. Phase 1 ports the algorithm; deferred to optimizing later if perf-bounded elsewhere. |
| `permute_dimensions` / `reorder_dimensions` (TensorIterator.cpp:950+) | none | NEEDS-PORT (~80 LoC) | Sorts dims by stride-descending to make stride-1 the innermost. Crucial for vectorization fast-path. Phase 1 ports. |
| `can_use_32bit_indexing` (TensorIterator.h:306) | none | NEEDS-PORT (~30 LoC) | `numel * max_element_size < INT_MAX`. Flame-core tensors are typically ≤ 2³¹ elements (≤ ~4 GB at BF16), so this is usually true. Phase 1 ports; the split logic it gates is Phase 2. |
| `build_unary_op` / `build_binary_op` / `build_borrowing_binary_op` (TensorIterator.h:788–820) | none | NEEDS-PORT (~150 LoC in Phase 3) | Three public entry points the kernel files call. Phase 1 stubs them returning an error; Phase 3 fills them in after the DispatchStub machinery lands. |
| `allocate_or_resize_outputs` (TensorIterator.cpp:1450+) | `Tensor::zeros_dtype` / explicit pool alloc in session callers | EXISTS (different API) | Output allocation happens on the Rust side before the iterator is built. Phase 1's `TensorIteratorConfig::add_output(None)` signals "allocate via pool after shape is computed"; `add_output(Some(&tensor))` uses the supplied tensor. |
| `FastSetupType` enum (TensorIterator.h:238–243) | none | NEEDS-PORT (~30 LoC) | `{ NONE, CONTIGUOUS, CHANNELS_LAST, NON_OVERLAPPING_DENSE }`. Phase 1 ports but only uses `CONTIGUOUS` — channels-last support is out-of-scope per the plan's deferred list. |
| `SplitUntil32Bit` iterator (TensorIterator.h:998–1032) | none | NEEDS-PORT (~150 LoC, Phase 2) | Generator that yields sub-iterators each safe for 32-bit indexing. Rare path for >2³¹-element tensors; stub in Phase 1, fill in Phase 2 when `gpu_kernel` needs it. |

**Phase 1 deliverable files:**
- `src/tensor_iterator/mod.rs` (module root)
- `src/tensor_iterator/config.rs` (`TensorIteratorConfig`)
- `src/tensor_iterator/base.rs` (`TensorIteratorBase` + `OperandInfo`)
- `src/tensor_iterator/dim_vec.rs` (small-vec type aliases, compat with `Shape::Strides`)
- `src/tensor_iterator/broadcast.rs` (`compute_shape_and_strides` algorithm)
- `tests/tensor_iter_config.rs` (broadcast, coalesce, 32-bit-split flag, contig flag)

No CUDA code in Phase 1. No op migrations.

---

## 3. `OffsetCalculator` + `gpu_kernel` mapping (Phase 2)

| PyTorch entity (file:line) | flame-core equivalent | Status | Port notes |
|---|---|---|---|
| `OffsetCalculator<NARGS, index_t, signed_strides>` (OffsetCalculator.cuh:21–91) | `flame::iter::StridedOffsetCalc` at `src/cuda/tensor_iterator.cuh:41` | PARTIAL | Current: single-arg, plain divmod, rank ≤ 6. PyTorch: NARGS-templated, `IntDivider<uint32_t>` fast divmod, MAX_DIMS=25, supports signed strides. Phase 2 generalizes to NARGS 1..=8, adds IntDivider backing, keeps rank ≤ 6 (matches flame-core Shape capacity). Signed strides NOT ported (no `torch.flip` equivalent in flame-core scope). |
| `OffsetCalculator::get(linear_idx)` returning `std::array<stride_t, NARGS>` | `StridedOffsetCalc::get(linear_idx)` returning `int64_t` | PARTIAL | Current returns a single offset. Phase 2 version returns NARGS offsets; call sites update. |
| `TrivialOffsetCalculator<NARGS>` (OffsetCalculator.cuh:94–110) | none | NEEDS-PORT (~30 LoC) | Identity offset calculator used by `launch_vectorized_kernel` when all inputs are contig. Phase 2 ports verbatim. |
| `IntDivider<T>` / `IntDivider<uint32_t>` (IntegerDivider.cuh) | none | NEEDS-PORT (~150 LoC) | Magic-constant divmod precomputed at construction; device-side uses `__umulhi`. ~20–30% speedup on rank ≥ 3 strided ops vs plain divmod. `__umulhi` is sm_35+ so safe on flame-core's sm_80/86/89 targets. Lands at `src/cuda/integer_divider.cuh`. |
| `MAX_DIMS` constant | `FLAME_MAX_DIMS = 6` at `src/cuda/tensor_iterator.cuh:25` | EXISTS (different value) | PyTorch is 25, flame-core is 6 matching Shape capacity. Keep 6 — no DL tensor in the codebase exceeds rank 5. |
| `make_offset_calculator<N>(iter)` (OffsetCalculator.cuh:113) | none | NEEDS-PORT (~30 LoC) | Host-side helper building OffsetCalculator from TensorIterator metadata. |
| `elementwise_kernel_helper<reverted_idx, func_t, policy_t>` (Loops.cuh:44–75) | inline body of `flame_elementwise_strided_to_contig` at `src/cuda/tensor_iterator.cuh:77+` | PARTIAL | Current: hardcoded policy (scalar per-thread). PyTorch: templated over memory-access policy (vectorized/unrolled). Phase 2 ports the helper + `TrivialPolicy`; vectorized policy deferred to Phase 5 perf tuning. |
| `gpu_kernel_impl_nocast` (CUDALoops.cuh:643) | `launch_elementwise_strided_to_contig` / `_binary_to_contig` | PARTIAL | Current: strided-only path (no vectorized branch). PyTorch branches on `iter.is_contiguous()` → `launch_vectorized_kernel` vs `launch_legacy_kernel`. Phase 2 adds the contig branch — in BF16 this is where the `__hadd2` / `__nv_bfloat162` math lives. |
| `gpu_kernel_impl` (CUDALoops.cuh:959) | none | NEEDS-PORT (~80 LoC) | Wrapper around `_nocast` with dynamic-cast detection. Phase 2 ports but flags dynamic-cast-unsupported → BLOCKED (Phase 8 fills in). |
| `gpu_kernel_nocast(iter, func)` (Loops.cuh:84) | none | NEEDS-PORT (~60 LoC) | Public API. Phase 2 entry point for kernel files. |
| `gpu_kernel(iter, func)` (Loops.cuh:115) | none | NEEDS-PORT (~60 LoC) | Public API + 32-bit-split recursion. |
| `gpu_kernel_with_scalars(iter, func)` (Loops.cuh:254) | `GpuOps::{mul,add}_scalar` F32 round-trip | NEEDS-PORT (~100 LoC, Phase 5) | CPU scalar arg gets folded into the kernel's lambda as a captured value. Phase 5 uses this for `mul_scalar` / `add_scalar` native BF16 paths. |
| `opmath_gpu_kernel_with_scalars` (Loops.cuh:200) | none | NEEDS-PORT (~80 LoC, Phase 5) | Higher-precision variant: scalar loaded as opmath_t (BF16 → F32). Required for the `opmath_type<scalar_t>` idiom in the functors. |
| `launch_vectorized_kernel` (CUDALoops.cuh:293) | none | NEEDS-PORT (~200 LoC, Phase 5) | Vectorized fast path using `__nv_bfloat162`-style 2/4/8-wide loads. Phase 2 stubs it (uses `launch_legacy_kernel` everywhere for contig); Phase 5 fills in with BF16 specialization that matches `__hadd2` perf of the existing flat kernel. |
| `launch_legacy_kernel` (CUDALoops.cuh:545) | body of `launch_elementwise_strided_*_to_contig` at `src/cuda/tensor_iterator.cuh:97+` | PARTIAL | Current: 256 threads, one elem/thread. PyTorch: 128 threads, 4 elems/thread with `#pragma unroll`. Phase 2 updates flame-core's to match PyTorch's work-per-thread; benchmark against the existing flat kernel. |
| `vectorized_elementwise_kernel<vec_size>` (CUDALoops.cuh:167) | none | NEEDS-PORT (~150 LoC, Phase 5) | The actual vectorized kernel. Phase 5 implements the sm_80+ variant (vec_size ∈ {1,2,4}). |
| `unrolled_elementwise_kernel` (CUDALoops.cuh:276) | none | NEEDS-PORT (~100 LoC) | Fallback for non-vectorizable contig cases. Phase 2 ports. |
| `num_threads()` / `block_work_size()` / `elementwise_thread_work_size()` (thread_constants.h) | hardcoded `threads=256` at `src/cuda/tensor_iterator.cuh:105` | PARTIAL | Phase 2 ports as compile-time constants in `src/cuda/thread_constants.cuh` matching PyTorch's values (128 threads, 4 elems, 512 per block). Bench verifies this isn't worse on sm_86. |
| Memory access policies (MemoryAccess.cuh) | none | NEEDS-PORT (~300 LoC, Phase 5) | Templates for load/store strategy (vectorized, unrolled, per-thread). Phase 2 only ports `TrivialPolicy` (one elem/thread, no unroll); Phase 5 adds `VectorizedPolicy<N>` for BF16 pair-vec. |

**Phase 2 deliverable files:**
- `src/cuda/tensor_iterator.cuh` (rewritten, ~500 LoC of port core — keep the file name)
- `src/cuda/offset_calculator.cuh` (new — matches PyTorch's filename)
- `src/cuda/integer_divider.cuh` (new — matches PyTorch's filename)
- `src/cuda/thread_constants.cuh` (new)
- `tests/tensor_iter_kernel_smoke.rs` (copy kernel via new path bit-exact for rank 1..=6, contig + permuted)
- Session-1–4 `.cu` files (`activation_silu_iter.cu`, etc.) retargeted to new `launch_gpu_kernel<>` — zero behavioral change, existing parity tests must still be green

**Deletions at Phase 2 exit (~80 LoC):** old `StridedOffsetCalc` + old `launch_elementwise_strided_to_contig` + old `launch_elementwise_strided_binary_to_contig` in the previous `tensor_iterator.cuh`.

### Current `StridedOffsetCalc` vs target `OffsetCalculator<2>` — concrete diff

```
flame-core/src/cuda/tensor_iterator.cuh:41–65 (StridedOffsetCalc, Phase 0 state):
  int     rank;                          // 0..=6
  int64_t sizes[6];
  int64_t strides[6];                    // element strides, single-arg
  int64_t base_offset;
  get(lin) : plain divmod loop, returns int64_t (single offset)

PyTorch aten/src/ATen/cuda/detail/OffsetCalculator.cuh:21–91 (OffsetCalculator<NARGS>, target):
  int     dims;                          // 0..=25
  IntDivider<uint32_t> sizes_[25];       // magic-constant divmod
  index_t strides_[25][NARGS];           // element strides, NARGS-arg
  get(lin) : unrolled over all dims, returns std::array<stride_t, NARGS>
```

Phase 2 closes the delta to:
```
flame::iter::OffsetCalculator<NARGS> (Phase 2 target):
  int     dims;                          // 0..=6
  IntDivider<uint32_t> sizes_[6];        // magic-constant divmod
  int64_t strides_[6][NARGS];            // element strides, NARGS-arg
  get(lin) : unrolled, returns array of NARGS offsets
```

Three deltas:
1. **NARGS-templated** (replaces current single-arg struct).
2. **IntDivider-backed** divmod (replaces plain `%`/`/`).
3. **Array return** (replaces single int64_t).

---

## 4. `DispatchStub` + registration mapping (Phase 3)

Rust has no header-file static init equivalent to PyTorch's
`DECLARE_DISPATCH` / `DEFINE_DISPATCH` / `REGISTER_DISPATCH`. Port uses
`OnceLock`-based runtime registry.

| PyTorch entity | flame-core equivalent | Status | Port notes |
|---|---|---|---|
| `DispatchStub<FnPtr, T>` template (DispatchStub.h:87) | `struct StubEntry { cuda: OnceLock<FnPtr> }` | NEEDS-PORT (~150 LoC) | Per-op stub type. Phase 3 generates one per elementwise op via the `DECLARE_STUB!` macro. Lands at `src/tensor_iterator/dispatch.rs`. |
| `DECLARE_DISPATCH(fn_type, name)` (DispatchStub.h:389) | `declare_stub!(silu_stub, fn(&TensorIterBase))` macro | NEEDS-PORT (~50 LoC, macro) | Rust macro generating `static SILU_STUB: StubEntry = StubEntry::new();`. |
| `DEFINE_DISPATCH(name)` (DispatchStub.h:400) | included in `declare_stub!` (no separate define) | Rolled in | Rust doesn't need the header-vs-source split. |
| `REGISTER_DISPATCH(name, fn)` (DispatchStub.h:471) | `register_stub!(silu_stub, silu_bf16_kernel)` in `lib.rs` init | NEEDS-PORT (~30 LoC, macro) | Call inside `register_all_bf16_kernels()`. Phase 3 generates this + the init list. |
| `DispatchStubImpl` (DispatchStub.h:96, CPU capability dispatch) | none | SKIP | flame-core is CUDA-only — no AVX2/AVX512 variants. |
| `REGISTER_CUDA_DISPATCH(name, fn)` / `RegisterCUDADispatch<>` | `register_stub!` with `.cuda` field setter | Rolled in | |
| Static-init ordering (PyTorch global constructors run before main) | Explicit init at `lib.rs::fn register_all_bf16_kernels()`, called once via `ctor::ctor!` macro or `Lazy::force()` | NEEDS-PORT (~20 LoC) | flame-core already uses the `ctor` crate (see `Cargo.toml`). Init list runs once per process. |
| Forgetting to register → linker error | Forgetting to register → panic at first call (`OnceLock::get().unwrap()`) | NEEDS-PORT (see macro) | Phase 3's `declare_stub!` + `register_stub!` macros both expand into the same central init list — hard to forget one without the other. Add a compile-time test that every `declare_stub!` has a matching `register_stub!`. |

**Phase 3 deliverable files:**
- `src/tensor_iterator/dispatch.rs` (`StubEntry`, `declare_stub!`, `register_stub!` macros)
- `src/lib.rs` — adds `fn register_all_bf16_kernels()` called via `ctor::ctor!`
- `tests/tensor_iter_builders.rs` (`build_unary_op`, `build_binary_op` produce correct shape/strides/allocations; dispatch stub lookup works; double-register panics cleanly)

Also in Phase 3: implement `TensorIteratorBase::build_unary_op` and
`build_binary_op` (stubbed in Phase 1). These call the dispatch stub
lookup + invoke `gpu_kernel` (from Phase 2).

---

## 5. Per-kernel-file template mapping (Phases 4–9)

### 5.1 Unary shape reference

**PyTorch `aten/src/ATen/native/cuda/ActivationSiluKernel.cu` (61 lines):**

```cpp
namespace at::native {
namespace {
void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(), "silu_cuda",
      [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t x_acc = static_cast<opmath_t>(x);
          return x_acc / (opmath_t(1) + ::exp(-x_acc));
        });
      });
}
}
REGISTER_DISPATCH(silu_stub, &silu_kernel)
}
```

**flame-core current `src/cuda/activation_silu_iter.cu` (96 lines):**

```cpp
namespace { struct SiluBF16Op { ... }; }  // functor

extern "C" int flame_silu_bf16_strided(
    const void* x_ptr, int64_t x_offset_elems, void* y_ptr,
    int rank, const int64_t* sizes, const int64_t* in_strides,
    int64_t n_elements, void* stream_void) {
    // explicit rank/offset validation
    flame::iter::StridedOffsetCalc calc;
    calc.rank = rank; calc.base_offset = x_offset_elems;
    for (int i = 0; i < FLAME_MAX_DIMS; ++i) { ... fill sizes/strides ... }
    return launch_elementwise_strided_to_contig<...>(...);
}
```

**Phase 4 target shape (rewrite of both silu and gelu kernels):**

```cpp
namespace flame::native {
namespace {
void silu_bf16_kernel(TensorIteratorBase& iter) {
    gpu_kernel(iter, [] GPU_LAMBDA(__nv_bfloat16 x) -> __nv_bfloat16 {
        float v = __bfloat162float(x);
        return __float2bfloat16_rn(v / (1.0f + __expf(-v)));
    });
}
}
REGISTER_BF16_DISPATCH(silu_stub, silu_bf16_kernel)
}
```

Close to PyTorch's shape — ~20 LoC per kernel. No more extern "C"
boilerplate, no explicit `StridedOffsetCalc` fill. Dispatch routes via
`silu_stub` registration. BF16-only `__nv_bfloat16` (no `scalar_t`
templating since there's no F16 or F32 path in our migration scope —
those come in Phase 8).

### 5.2 Binary shape reference

**PyTorch `aten/src/ATen/native/cuda/BinaryMulKernel.cu` (~49 lines):**

```cpp
void mul_kernel_cuda(TensorIteratorBase& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_cuda",
        [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
                iter, binary_internal::MulFunctor<opmath_t>());
        });
}
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda)
```

**flame-core current `src/cuda/add_bf16_iter.cu` (103 lines):** see
session-4 commit 896fa0d. ~70 LoC of explicit config-fill + validation
+ extern "C" wrapper, ~30 LoC of actual functor.

**Phase 5 target shape:**

```cpp
namespace flame::native {
namespace { struct AddBF16Op { /* fp32-round-trip add */ }; }
void add_bf16_kernel(TensorIteratorBase& iter) {
    gpu_kernel(iter, AddBF16Op{});
}
REGISTER_BF16_DISPATCH(add_stub, add_bf16_kernel)
}
```

### 5.3 Bool-output shape reference

**PyTorch `aten/src/ATen/native/cuda/BinaryCompareKernel.cu`:** uses
`gpu_kernel` with a lambda returning `bool`; the iterator knows the
output dtype is `u8`.

**flame-core target (Phase 9):** same pattern via a new
`build_comparison_op` builder that pins output dtype to `u8`.

---

## 6. Full op inventory

Every BF16 elementwise op that needs a home in the new structure.
Tensor::method line numbers confirmed against the earlier
explore-agent recon (2026-04-22 session).

| Op | `Tensor::` site | Current BF16 path | Respects strides? | Target phase | LoC to delete |
|---|---|---|---|---|---|
| add | tensor.rs:1844 | `ops::add_iter::add_bf16_iter` → `flame_add_bf16_strided` OR `bf16_elementwise::add_bf16` (post-session-4) | **YES** (iter path) | Phase 5 (rewrite) | ~200 NVRTC + ~120 dispatch |
| sub | tensor.rs:1888 | `sub_bf16` via `add_bf16 + mul_scalar(-1)` | NO | Phase 5 | ~50 |
| mul | tensor.rs:1911 | `bf16_elementwise::mul_bf16` (flat + broadcast) | **via broadcast path only** (flat path ignores strides — fixed in post-session-4 gate fix) | Phase 5 | ~200 + 50 |
| div | tensor.rs (search) | `bf16_elementwise::div_bf16` | same as mul | Phase 5 | ~50 |
| mul_scalar | tensor.rs:1938 | `GpuOps::mul_scalar` (F32 round-trip) | NO | Phase 5 | ~30 |
| add_scalar | tensor.rs:1966 | `GpuOps::add_scalar` (F32 round-trip) | NO | Phase 5 | ~30 |
| silu | tensor.rs:2044 | `ops::silu_iter::silu_bf16_iter` (session 1, commit cb25acf) | **YES** | Phase 4 (rewrite on real TI) | ~100 |
| gelu | tensor.rs:2009 | `ops::gelu_iter::gelu_bf16_iter` (session 2, commit 1aed0d6) | **YES** | Phase 4 (rewrite) | ~100 |
| square | tensor.rs:2176 | `ops::square_iter::square_bf16_iter` (session 3, commit 248abf1) | **YES** | Phase 4 (rewrite) | ~100 |
| abs | tensor.rs (search) | `bf16_elementwise::abs_bf16` at :346 (NVRTC native BF16 — sign-bit clear) | NO (contig-only) | Phase 6 (rewrite on TI) | ~50 |
| relu | tensor.rs:1989 | `GpuOps::relu` → `cuda_ops_bf16::relu_bf16` → `fc_relu_bf16` FFI | NO | Phase 6 | ~80 |
| sigmoid | tensor.rs:2127 | `GpuOps::sigmoid` (F32 round-trip) | NO | Phase 6 | ~50 |
| tanh | tensor.rs:2107 | `GpuOps::tanh` (F32 round-trip) | NO | Phase 6 | ~50 |
| neg | tensor.rs (search) | `mul_scalar(-1.0)` (composite, F32 round-trip) | NO | Phase 6 | ~30 |
| exp | tensor.rs:2175 | `GpuOps::exp` (F32 round-trip) | NO | Phase 7 | ~50 |
| log | tensor.rs (search) | F32 round-trip | NO | Phase 7 | ~50 |
| sqrt | tensor.rs (search) | F32 round-trip | NO | Phase 7 | ~50 |
| rsqrt | tensor.rs (search) | F32 round-trip | NO | Phase 7 | ~50 |
| recip | tensor.rs (search) | F32 round-trip | NO | Phase 7 | ~50 |
| ge/gt/le/lt/eq/ne | tensor.rs (search) | `bf16_elementwise::cmp_bf16` family | **YES** via broadcast | Phase 9 | ~350 |

**Critical note on strides** (post-session-4 state): `bf16_elementwise`'s
flat fast path (`launch_bf16_flat` at `:733`) now requires both
operands contiguous (gate fix in session-4 commit 896fa0d). Strided
same-shape inputs correctly fall through to `launch_bf16_elementwise`
which respects per-arg strides via `BcSpec`. So "respects strides"
column above is accurate for the current HEAD. Phase 5 deletes both
paths and replaces with one TensorIterator path.

---

## 7. Fused ops that STAY (do-not-touch list for Phase 11)

Each of these is NOT elementwise (fused multi-op, reduction, or
structured kernel). Phase 11's Builder must NOT delete these files or
touch these functions during corpse cleanup. Source `:line` verified.

| Op | File:line | Why out-of-scope |
|---|---|---|
| `rope_fused_bf16` | bf16_ops.rs:417 | Interleaved-pair RoPE (complex rotation + gather). Structured memory access. |
| `rope_halfsplit_bf16` | bf16_ops.rs:500 | RoPE halfsplit variant. Same. |
| `rope_fused_bf16_f32pe` | bf16_ops.rs:554 | RoPE with F32 positional embeddings. Same. |
| `swiglu_fused_bf16` | bf16_ops.rs:1143 | silu(gate) * up. Dual-input gating, not uniform elementwise. |
| `modulate_pre_fused_bf16` | bf16_ops.rs:882 | DiT shift+scale modulation. Structured per-token broadcast. |
| `modulate_pre_split_apply_bf16` | bf16_ops.rs:948 | B.3 split+apply variant. |
| `gate_residual_fused_bf16` | bf16_ops.rs:1076 | x + gate * attn_out. Fused 3-op. |
| `fused_rms_norm_bf16` / `*_modulate` / `*_bf16_to_f32` | src/cuda/fused_rms_norm.cu | Reduction (compute rms) + normalization + modulation. |
| `softmax_last_dim_bf16` | bf16_ops.rs:247 | Row-wise reduction (max, sum) + exp + div. |
| `attn_split_txt_img_bf16` | bf16_ops.rs:1233 | Attention output text/image split. Structured. |
| `qkv_split_permute_bf16` | bf16_ops.rs:1629 | QKV split + permute. Structured. |
| `patchify_bf16` / `unpatchify_bf16` | bf16_ops.rs:866, :922 | DiT patch ops. Structured reshape + gather. |
| `transpose2d_bf16` | bf16_elementwise.rs (search) | 2D transpose. Memory-layout-specific kernel. |
| SDPA (cuDNN / WMMA / flash) | src/cuda/cudnn_sdpa.cpp, flash_attention_fwd.cu, attention/sdpa.rs | Multi-stage matmul + softmax + matmul. Own builder in Phase 6 of PyTorch's lineage, not this plan. |
| conv2d (all variants) | conv/ ops/conv2d* | Convolution. Not elementwise. |
| GEMM family (`fused_linear3d*`, `gemm_bf16*`) | src/cuda/fused_linear3d.cu, ops/gemm*.rs | Matmul via cuBLASLt. |
| All reductions (`reduce_sum*`, `*_keepdim`, mean, max) | cuda_kernels / bf16_ops | Reductions. Not elementwise. |
| All autograd backward kernels | src/kernels/{silu,gelu,swiglu,relu,tanh,sigmoid}_backward.cu | Backward is `g * sig(x) * (1 + x * (1 - sig(x)))`-style; not trivially representable as a unary lambda. Separate Phase-6-like path when we ever migrate backward to TensorIterator (explicitly NOT in current plan). |

If a PyTorch op of the same name exists, the flame-core version is
intentionally kept custom (perf-critical and/or structurally different
from pointwise). Phase 11's Builder verifies this list against actual
file contents and raises BLOCKED if any op is accidentally elementwise
and slipped into the custom list.

---

## 8. Build pipeline notes

### Build-time `.cu` pipeline (Phase 2+ kernels use this)

`build.rs` at flame-core root compiles every `.cu` in the `cuda_sources`
list with:
```
nvcc -std=c++17 -O3 --use_fast_math -Xcompiler -fPIC -rdc=true
     -gencode arch=compute_80,code=sm_80
     -gencode arch=compute_86,code=sm_86
     -gencode arch=compute_89,code=sm_89
     -I<cuda>/include -I cuda/include
```

All object files are device-linked into `libflame_cuda_kernels.a`.
`-rdc=true` is required for `OffsetCalculator<NARGS>` template
instantiations across translation units.

Phase 2 adds new `.cuh` headers (not separate `.o`s):
- `src/cuda/integer_divider.cuh`
- `src/cuda/offset_calculator.cuh`
- `src/cuda/thread_constants.cuh`
- Rewrite of `src/cuda/tensor_iterator.cuh`

Phase 2 does NOT add a new `.cu` — the existing session-1–4 files
re-point to the new headers.

### NVRTC (legacy, stays for fused ops)

`bf16_ops.rs`, `bf16_elementwise.rs`, `bf16_convert.rs`,
`cuda_kernels*.rs` use NVRTC (runtime compilation of `const &str` CUDA
source). Phase 1–11 does NOT touch NVRTC for fused ops. Elementwise
NVRTC kernels in `bf16_ops` / `bf16_elementwise` are deleted as their
ops migrate.

NVRTC limitations (from flame-core/CLAUDE.md):
- `<cfloat>` / `<float.h>` headers unavailable — use literal constants
- `#pragma unroll` doesn't survive macro expansion — use
  `_Pragma("unroll")`
- Templates "don't survive NVRTC reliably" — another reason Phase 2's
  `OffsetCalculator<NARGS>` lives in build-time `.cuh`, not NVRTC

---

## 9. Risk flags

| Risk | Phase | Severity | Mitigation |
|---|---|---|---|
| **R1.** `launch_vectorized_kernel`'s BF16 path may be >5% slower than the existing `__hadd2`/`__nv_bfloat162` flat kernels. | 2, 5 | HIGH | Phase 5 bench compares `add_bf16_iter` contig-path vs direct flat kernel. If >5% regression, keep flat kernel as `launch_vectorized_kernel`'s BF16 specialization. |
| **R2.** NVRTC cannot compile the new `OffsetCalculator<NARGS>` template. | 2 | MEDIUM | The Phase 2 code path is build-time `.cu` only; NVRTC is never invoked on these headers. Guard by compile-time build-system placement (headers live in `src/cuda/`, not in any `.rs` NVRTC const string). |
| **R3.** Rust `Arc<Tensor>` lifetime conflict with iterator's borrowed operands. | 1 | MEDIUM | `TensorIteratorBase::add_input(&Tensor)` takes a borrow; iterator lifetime is bounded by the call. No cross-call storage. If some phase needs it, enable the `shared_storage` feature (currently in Cargo default list, unclear if off in practice). |
| **R4.** Autograd tape interactions in Phase 10. | 10 | MEDIUM | Phase 10's Skeptic brief explicitly blocks any Builder inlining a tape-save inside `build_binary_op`. Tape wiring stays in the `Tensor::<op>` method body, not inside the iterator. |
| **R5.** Klein fixture ratchet — each phase regenerating silently drifts the oracle. | 4–11 | HIGH | Hardened rule per the plan: preserve previous fixture as `klein_seed42_prev_<hash>.png`, commit new fixture in SAME commit as code change, commit message must include diff-analysis stats AND code citation for why the new output is semantically correct. Alex approval required. |
| **R6.** `IntDivider<uint32_t>` perf delta is smaller than expected on Klein's actual shapes (mostly rank 3–4, small dims). | 2 | LOW | Phase 2 benches real Klein shapes (1024×4608 etc.), not synthetic. If IntDivider delta is <2%, the port is still structurally correct; keep it for PyTorch parity. |
| **R7.** Phase 6/7 strict-mode dormant bugs — last-ULP drift between old F32-round-trip unaries and new native-BF16 unaries. | 6, 7 | MEDIUM | Skeptic briefs for phases 6 and 7 specifically audit `tests/strict_bf16_harness.rs` + any other strict-mode-gated test. Any new failure BLOCKs the phase. |
| **R8.** Phase 11 Builder accidentally deletes non-elementwise ops in `bf16_elementwise.rs`. | 11 | MEDIUM | §7 of this doc is the authoritative do-not-touch list. Phase 11 Builder brief reiterates. Phase 11 verifier runs `git diff` against known-good filenames. |

---

## 10. File layout for Phase 1+

New module tree under `flame-core/src/`:
```
tensor_iterator/
  mod.rs            (pub re-exports)
  config.rs         (TensorIteratorConfig builder — Phase 1)
  base.rs           (TensorIteratorBase + OperandInfo — Phase 1)
  dim_vec.rs        (DimVec / StrideVec aliases — Phase 1)
  broadcast.rs      (compute_shape_and_strides — Phase 1)
  dispatch.rs       (StubEntry, declare_stub!, register_stub! — Phase 3)
  ops/
    unary.rs        (silu/gelu/square/abs/relu/sigmoid/tanh/neg Rust wrappers — Phases 4, 6)
    transcendentals.rs (exp/log/sqrt/rsqrt/recip Rust wrappers — Phase 7)
    binary.rs       (add/sub/mul/div/max/min/scalar-variants Rust wrappers — Phase 5)
    comparison.rs   (ge/gt/le/lt/eq/ne Rust wrappers — Phase 9)
```

New CUDA headers under `flame-core/src/cuda/`:
```
integer_divider.cuh       (Phase 2)
offset_calculator.cuh     (Phase 2)
thread_constants.cuh      (Phase 2)
tensor_iterator.cuh       (rewrite, Phase 2)
```

New CUDA `.cu` functor files under `flame-core/src/cuda/`:
```
unary/
  silu.cu gelu.cu square.cu abs.cu relu.cu sigmoid.cu tanh.cu neg.cu
  exp.cu log.cu sqrt.cu rsqrt.cu recip.cu
binary/
  add.cu sub.cu mul.cu div.cu maximum.cu minimum.cu
  mul_scalar.cu add_scalar.cu
cmp/
  ge.cu gt.cu le.cu lt.cu eq.cu ne.cu
```

Session-1–4 files (`activation_silu_iter.cu`, `activation_gelu_iter.cu`,
`activation_square_iter.cu`, `add_bf16_iter.cu`) are deleted at Phase 4
when the new `unary/silu.cu` etc. land.

---

## 11. Verification oracles

### Existing (reuse, keep green at every phase boundary)

- `tests/bf16_tensor_ops.rs` (3 tests)
- `tests/bf16_broadcast.rs` (1 test)
- `tests/cudnn_sdpa_parity.rs` (2 tests — orthogonal but catches regressions)
- `tests/materialize_view_parity.rs` (3 tests)
- `tests/activation_backward_fused_kernels.rs` (9 tests)
- `tests/tensor_iterator_silu_parity.rs` (3 tests, session 1)
- `tests/tensor_iterator_gelu_parity.rs` (3 tests, session 2)
- `tests/tensor_iterator_square_parity.rs` (4 tests, session 3)
- `tests/tensor_iterator_add_parity.rs` (4 tests, session 4)
- `cargo test --release --lib` (54 tests)
- `inference-flame/tests/fixtures/klein_seed42_baseline.png` (PNG
  oracle — regenerated at session-4 Pre-Phase-0 under the "live bug
  fix" protocol)

### New per phase

| Phase | Test file | What it checks |
|---|---|---|
| 1 | `tests/tensor_iter_config.rs` | Broadcast shape, coalesce, 32-bit-split flag, contig flag |
| 2 | `tests/tensor_iter_kernel_smoke.rs` | Copy kernel bit-exact rank 1..=6 contig + permuted |
| 3 | `tests/tensor_iter_builders.rs` | `build_unary_op`/`build_binary_op` produce correct shape/strides/allocations on matching, broadcast, permuted inputs; stub lookup works |
| 4 | `tests/tensor_iter_pilot_broadcast.rs` | `add` broadcast cases (deferred from session 4) |
| 5 | `tests/tensor_iter_binary_arith.rs` | sub/div/max/min broadcast + strided |
| 6 | `tests/tensor_iter_native_bf16_unary.rs` + `tests/strict_bf16_unary.rs` | 7 unaries match F32-computed reference at cos_sim ≥ 0.9999; zero F32-round-trip observed |
| 7 | `tests/tensor_iter_transcendental_bf16.rs` | 5 transcendentals native BF16 |
| 8 | `tests/tensor_iter_mixed_dtype.rs` | BF16+F32 add with promotion |
| 9 | `tests/tensor_iter_comparisons.rs` | ge/gt/le/lt/eq/ne output is u8, matches reference |

### Klein end-to-end

Long-running (`#[ignore]` test at `flame-core/tests/png_parity_klein.rs`).
Run at every phase boundary. Byte-equal gate against
`inference-flame/tests/fixtures/klein_seed42_baseline.png`. Regen
protocol: preserve old fixture → commit new with diff-analysis →
Alex approves.

### Benchmarks

- `benches/silu_iter_alloc.rs` (session 1) — baseline for unary contig
  overhead
- `benches/gelu_iter_alloc.rs` (session 2)
- `benches/square_iter_alloc.rs` (session 3)
- `benches/add_iter_alloc.rs` (session 4)
- `benches/tensor_iter_offset_calc_compare.rs` (Phase 2) — IntDivider
  vs plain divmod on real Klein shapes

---

## 12. Cargo features

No new features in Phases 1–3. `cuda` + `bf16_u16` (defaults) cover
everything. The plan does NOT use a feature flag to gate the new
TensorIterator path — Phase 4 flips the dispatch entirely, and there is
no opt-out fallback. This is intentional per Alex's direction: "replace,
don't add alongside."

If a phase hits a hard blocker during rollout, the escape is a BLOCKED
handoff + revert, not a feature flag.

---

## 13. PyTorch reference line numbers (quick-lookup)

| File | Function / struct | Lines |
|---|---|---|
| `aten/src/ATen/TensorIterator.h` | `OperandInfo` | 117–206 |
| `aten/src/ATen/TensorIterator.h` | `TensorIteratorBase` class | 248–734 |
| `aten/src/ATen/TensorIterator.h` | `TensorIteratorConfig` class | 783–993 |
| `aten/src/ATen/TensorIterator.h` | `SplitUntil32Bit` iterator | 998–1032 |
| `aten/src/ATen/TensorIterator.cpp` | `compute_shape` | ~870 |
| `aten/src/ATen/TensorIterator.cpp` | `compute_strides` | ~910 |
| `aten/src/ATen/TensorIterator.cpp` | `reorder_dimensions` | ~950 |
| `aten/src/ATen/TensorIterator.cpp` | `coalesce_dimensions` | ~1027 |
| `aten/src/ATen/TensorIterator.cpp` | `allocate_or_resize_outputs` | ~1450 |
| `aten/src/ATen/cuda/detail/OffsetCalculator.cuh` | `OffsetCalculator<NARGS>` | 21–91 |
| `aten/src/ATen/cuda/detail/OffsetCalculator.cuh` | `TrivialOffsetCalculator<NARGS>` | 94–110 |
| `aten/src/ATen/cuda/detail/OffsetCalculator.cuh` | `make_offset_calculator<N>` | 113 |
| `aten/src/ATen/cuda/detail/OffsetCalculator.cuh` | `make_element_offset_calculator<N>` | 124 |
| `aten/src/ATen/cuda/detail/IntegerDivider.cuh` | `IntDivider<T>` base | 65–76 |
| `aten/src/ATen/cuda/detail/IntegerDivider.cuh` | `IntDivider<uint32_t>` fast | 80–122 |
| `aten/src/ATen/native/cuda/Loops.cuh` | `elementwise_kernel_helper` | 44–75 |
| `aten/src/ATen/native/cuda/Loops.cuh` | `gpu_kernel_nocast` | 84 |
| `aten/src/ATen/native/cuda/Loops.cuh` | `gpu_kernel` | 115 |
| `aten/src/ATen/native/cuda/Loops.cuh` | `gpu_kernel_with_scalars` | 254 |
| `aten/src/ATen/native/cuda/CUDALoops.cuh` | `vectorized_elementwise_kernel` | 167 |
| `aten/src/ATen/native/cuda/CUDALoops.cuh` | `unrolled_elementwise_kernel` | 276 |
| `aten/src/ATen/native/cuda/CUDALoops.cuh` | `launch_vectorized_kernel` | 293 |
| `aten/src/ATen/native/cuda/CUDALoops.cuh` | `launch_legacy_kernel` | 545 |
| `aten/src/ATen/native/cuda/CUDALoops.cuh` | `gpu_kernel_impl_nocast` | 643 |
| `aten/src/ATen/native/cuda/CUDALoops.cuh` | `gpu_kernel_impl` | 959 |
| `aten/src/ATen/native/cuda/ActivationSiluKernel.cu` | full file | 1–61 |
| `aten/src/ATen/native/cuda/BinaryMulKernel.cu` | `mul_kernel_cuda` + `REGISTER_DISPATCH` | 1–49 |
| `aten/src/ATen/native/DispatchStub.h` | `DispatchStub<>` template | 87–330 |
| `aten/src/ATen/native/DispatchStub.h` | `DECLARE_DISPATCH` macro | 389 |
| `aten/src/ATen/native/DispatchStub.h` | `DEFINE_DISPATCH` macro | 400 |
| `aten/src/ATen/native/DispatchStub.h` | `REGISTER_DISPATCH` macro | 471 |

These line numbers are current as of the PyTorch checkout at
`/home/alex/pytorch/`. If PyTorch is updated, the line numbers shift
but the structural references (function names, template parameters)
remain valid anchors.

---

_Doc written 2026-04-22 by Phase 0 of the plan. Alive through Phase 11._
