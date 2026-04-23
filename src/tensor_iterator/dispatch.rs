// Origin: flame-core TensorIterator port, Phase 3.
// Reference:
//   pytorch/aten/src/ATen/native/DispatchStub.h L87–L330 (DispatchStub template),
//   L389 (DECLARE_DISPATCH), L400 (DEFINE_DISPATCH), L471 (REGISTER_DISPATCH).
//   flame-core docs/TENSORITERATOR_PORT_REFERENCE.md §4.
//
// Why this file exists
// --------------------
// PyTorch registers elementwise kernels against per-op "stubs" via three
// macros:
//
//   DECLARE_DISPATCH(fn_type, silu_stub);   // header: extern static stub
//   DEFINE_DISPATCH(silu_stub);             // one .cpp: static stub instance
//   REGISTER_DISPATCH(silu_stub, &silu_kernel_cuda);   // a .cu: register
//
// On library load, the `REGISTER_DISPATCH` expansion (a file-scope static
// constructor) writes the kernel function pointer into a template-specialised
// member of the stub. Call sites then look up the pointer and invoke the
// kernel.
//
// Rust has no header/source split and no compile-time equivalent of a
// C++ file-scope static constructor that reaches back into a template. The
// port uses:
//
//   * A plain `static StubEntry = StubEntry::new()` per op (same role as
//     `DEFINE_DISPATCH`). The `declare_stub!` macro emits that.
//   * A `OnceLock<FnPtr>` field inside `StubEntry` (same role as the
//     template-specialised CUDA pointer). `register_stub!` writes it.
//   * A central `register_all_bf16_kernels()` function whose body lists
//     every `register_stub!` call. The `ctor` crate's `#[ctor::ctor]`
//     runs it once at library init, mirroring PyTorch's global-constructor
//     behaviour.
//
// Scope restriction for Phase 3: `register_all_bf16_kernels()` ships with
// an empty body. Phase 4 populates it as the first op migrations land.
// An empty init list is intentional — Phase 3 wires the registration
// plumbing without migrating any op. The `build_unary_op`/`build_binary_op`
// geometry pipeline lives alongside this module (see `base.rs`).

use std::sync::OnceLock;

use crate::error::Result;

use super::base::TensorIteratorBase;

/// Entry-point type for a BF16 elementwise CUDA kernel.
///
/// The kernel receives an already-built `TensorIteratorBase` (shape,
/// strides, operand pointers populated) and is expected to:
///
///   1. Build an `IterMetadata` / `OffsetCalculator` from the iterator.
///   2. Invoke `launch_gpu_kernel<NARGS, func_t>` from
///      `src/cuda/tensor_iterator.cuh` (Phase 2 delivery).
///   3. Propagate errors via `Result`.
///
/// Matches PyTorch's `void silu_kernel(TensorIteratorBase& iter)` modulo
/// the `Result` return — flame-core propagates CUDA errors rather than
/// aborting.
pub type BF16ElementwiseKernel = fn(&mut TensorIteratorBase<'_>) -> Result<()>;

/// Per-op dispatch entry. One static instance is emitted per
/// `declare_stub!` invocation.
///
/// The `cuda_bf16` slot is the flame-core analogue of PyTorch's
/// `DispatchStub::cuda_dispatch_ptr` (`DispatchStub.h:201`). More slots
/// (F16, F32 CUDA; later CPU capability flavours) can be added as
/// additional `OnceLock` fields when those phases land. Phase 3 is
/// BF16-only — see `docs/TENSORITERATOR_PORT_REFERENCE.md` §12.
pub struct StubEntry {
    /// CUDA BF16 kernel entry. `None` until `register_cuda_bf16` has been
    /// called.
    pub cuda_bf16: OnceLock<BF16ElementwiseKernel>,
}

impl StubEntry {
    /// Construct an unregistered stub. `const` so that `declare_stub!`
    /// can emit the `static` directly without resorting to `lazy_static!`.
    pub const fn new() -> Self {
        Self {
            cuda_bf16: OnceLock::new(),
        }
    }

    /// Register a CUDA BF16 kernel against this stub. Must be called
    /// exactly once per stub — double-registration is a programming
    /// error (two kernels competing for the same op), not a silent
    /// overwrite, so this panics with a named message rather than
    /// returning `Result`.
    ///
    /// PyTorch's `REGISTER_DISPATCH` macro expands to a static
    /// initialiser that simply assigns the function pointer; the linker
    /// rejects multiple definitions. Rust macros have no linker stage,
    /// so the panic here is the enforcement path.
    pub fn register_cuda_bf16(&self, f: BF16ElementwiseKernel) {
        self.cuda_bf16
            .set(f)
            .unwrap_or_else(|_| panic!("StubEntry: cuda_bf16 already registered"));
    }

    /// Look up the registered CUDA BF16 kernel. Returns `None` if no
    /// `register_cuda_bf16` has run — callers surface that as a
    /// `NotImplemented` error at the build site, same contract as
    /// PyTorch's `DispatchStub::operator()` assert when no kernel is
    /// registered.
    #[inline]
    pub fn cuda_bf16(&self) -> Option<BF16ElementwiseKernel> {
        self.cuda_bf16.get().copied()
    }
}

/// Declare a dispatch stub at module scope.
///
/// Emits `$vis static $name: StubEntry = StubEntry::new();`. Keep visibility
/// `pub` when the stub must be reachable from another crate (which is the
/// common case in Phase 4+ when ops live in sibling modules).
///
/// The analogue of `DECLARE_DISPATCH(fn_ptr, silu_stub);` +
/// `DEFINE_DISPATCH(silu_stub);` rolled into one — Rust does not need the
/// header/source split so the two PyTorch macros collapse.
///
/// Example:
/// ```ignore
/// declare_stub!(pub SILU_STUB);
/// ```
#[macro_export]
macro_rules! declare_stub {
    ($vis:vis $name:ident) => {
        $vis static $name: $crate::tensor_iterator::dispatch::StubEntry =
            $crate::tensor_iterator::dispatch::StubEntry::new();
    };
}

/// Register a BF16 CUDA kernel against a previously-declared stub.
///
/// Intended to be called from `register_all_bf16_kernels()` (see below).
/// Panics if the stub already has a CUDA BF16 kernel registered — catch
/// duplicate registrations at process init rather than silently
/// overriding.
///
/// Analogue of `REGISTER_DISPATCH(silu_stub, &silu_kernel_cuda);`.
#[macro_export]
macro_rules! register_stub {
    ($stub:path, $kernel:path) => {
        $stub.register_cuda_bf16($kernel);
    };
}

/// Single central init point. Phase 3 leaves this empty on purpose — the
/// dispatch plumbing is wired but no ops have been migrated yet. Phase 4
/// will populate the body with `register_stub!` calls per migrated op:
///
/// ```ignore
/// register_stub!(SILU_STUB, silu_bf16_kernel);
/// register_stub!(GELU_STUB, gelu_bf16_kernel);
/// // ...
/// ```
///
/// Called exactly once at library load via the `#[ctor::ctor]` hook in
/// `src/lib.rs`. Matches PyTorch's behaviour where every
/// `REGISTER_DISPATCH` static constructor runs before `main` on library
/// link.
pub fn register_all_bf16_kernels() {
    // Phase 4 populates this with register_stub!(SILU_STUB, silu_bf16_kernel) etc.
}
