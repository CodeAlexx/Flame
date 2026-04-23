// Origin: flame-core TensorIterator port, Phase 4.
// Reference: mirror of `flame::iter::IterMetadata` defined in
//            flame-core/src/cuda/tensor_iterator.cuh:75-87.
//
// This is the FFI-level POD every Phase-4+ `.cu` kernel entry point receives
// by const pointer. The CUDA side reads the fields and routes to
// `flame::iter::launch_gpu_kernel<NARGS, Op>(*meta, Op{}, stream)`.
//
// Field layout must match the C struct byte-for-byte. Any change to
// `tensor_iterator.cuh`'s `IterMetadata` requires a sibling change here.
// Both sides use `int64_t` for stride/size/offset — flame-core's Rust
// layer converts `usize`/`i64` at the boundary here.

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

/// Must match `FLAME_MAX_DIMS` in `src/cuda/offset_calculator.cuh`.
pub const FLAME_MAX_DIMS: usize = 6;

/// Must match `MAX_NARGS` in `src/cuda/tensor_iterator.cuh`.
/// Phase 4 pilots: unary (NARGS=2 operands: out + in) and binary (NARGS=3:
/// out + a + b). `MAX_NARGS = 4` leaves one slot of slack for Phase 5+
/// ternary ops. Bumping this requires widening the C struct's arrays too.
pub const MAX_NARGS: usize = 4;

/// POD marshalling struct handed across the FFI to `flame_<op>_bf16_kernel`.
///
/// `#[repr(C)]` guarantees the layout matches the C++ struct. Rust-side
/// callers populate this from a built `TensorIteratorBase` (see
/// `TensorIteratorBase::build_iter_metadata_unary` /
/// `build_iter_metadata_binary`) and pass a `*const` pointer into the
/// kernel entry.
///
/// Layout rules (enforced by the builders in `TensorIteratorBase`):
///   - `num_args` = total operands (outputs + inputs). 0 < num_args <= MAX_NARGS.
///   - The first `num_outputs` entries of `strides`/`data_ptrs`/`offsets_elems`
///     are outputs; the rest are inputs. PyTorch convention: out at index 0.
///   - `strides[arg][dim]` is the ELEMENT stride (not byte stride) of operand
///     `arg` along dim `dim`. Broadcast dims have stride 0.
///   - `offsets_elems[arg]` is the element offset of operand `arg` inside its
///     backing storage (for view tensors with non-zero view_offset). Note:
///     flame-core's Phase-3 allocator always produces fresh contig outputs
///     with offset 0, and `TensorIteratorBase` absorbs any input view_offset
///     into the `data_ptr` rather than a separate field, so this is always
///     0 in Phase 4 — but we populate it with the tensor's `offset()` for
///     CUDA-side sanity and future-proofing.
///   - `is_contiguous` is true iff every operand's stride chain matches the
///     contiguous row-major layout of `sizes` (no broadcasting, no permute,
///     no narrow). In the Phase-4 pipeline, this comes from
///     `TensorIteratorBase::is_contiguous()` which is true only after
///     `coalesce_dimensions` has collapsed the iteration to ndim=1.
#[repr(C)]
#[derive(Debug)]
pub struct IterMetadata {
    /// Number of iteration dims (0..=FLAME_MAX_DIMS).
    pub ndim: i32,
    /// Total operands in use (1..=MAX_NARGS).
    pub num_args: i32,
    /// Outputs prefix (1..=num_args).
    pub num_outputs: i32,
    /// Pad so the i64 arrays below start on 8-byte boundary.
    /// Matches the `int _pad` in the `.cuh` struct.
    pub _pad: i32,

    /// Per-dim iteration sizes.
    pub sizes: [i64; FLAME_MAX_DIMS],

    /// `strides[arg][dim]` in element units.
    pub strides: [[i64; FLAME_MAX_DIMS]; MAX_NARGS],

    /// Per-operand element offset inside its backing storage.
    pub offsets_elems: [i64; MAX_NARGS],

    /// Per-operand device base pointer (unshifted by `offsets_elems`).
    pub data_ptrs: [*mut c_void; MAX_NARGS],

    /// Iteration `numel` (product of `sizes[0..ndim]`).
    pub numel: i64,

    /// Rust-side contig hint: every operand's stride chain is row-major
    /// over `sizes`. CUDA side uses this to select the (stubbed) vectorized
    /// branch.
    pub is_contiguous: bool,

    /// Always true for flame-core tensors (numel < 2^31). Field kept for
    /// byte-layout parity with the C struct.
    pub requires_32bit_indexing: bool,
}

impl IterMetadata {
    /// Construct a zero-initialised metadata struct. Callers populate the
    /// fields per-operand before handing the pointer to the `.cu` kernel.
    pub fn zeroed() -> Self {
        Self {
            ndim: 0,
            num_args: 0,
            num_outputs: 0,
            _pad: 0,
            sizes: [0; FLAME_MAX_DIMS],
            strides: [[0; FLAME_MAX_DIMS]; MAX_NARGS],
            offsets_elems: [0; MAX_NARGS],
            data_ptrs: [std::ptr::null_mut(); MAX_NARGS],
            numel: 0,
            is_contiguous: false,
            requires_32bit_indexing: true,
        }
    }
}

// Compile-time sanity: the layout should be stable. If the C struct grows,
// the `size_of` assert here fires at compile time instead of a runtime
// segfault. Each field count:
//   4 * sizeof(i32)  = 16   (ndim, num_args, num_outputs, _pad)
//   FLAME_MAX_DIMS * sizeof(i64)            = 48   (sizes)
//   MAX_NARGS * FLAME_MAX_DIMS * sizeof(i64) = 192  (strides)
//   MAX_NARGS * sizeof(i64)                 = 32   (offsets_elems)
//   MAX_NARGS * sizeof(*mut)                = 32   (data_ptrs, 64-bit targets)
//   1 * sizeof(i64)                         = 8    (numel)
//   2 * sizeof(bool) + 6 pad                = 8    (is_contiguous,
//                                                  requires_32bit_indexing,
//                                                  trailing align)
// Total = 16 + 48 + 192 + 32 + 32 + 8 + 8 = 336 bytes.
// The `cfg(target_pointer_width = "64")` gate is tight on flame-core's
// supported platforms (Linux CUDA). No other width is built.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<IterMetadata>() == 336);
