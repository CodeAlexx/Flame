// Origin: flame-core TensorIterator port, Phase 1.
// Reference: pytorch/c10/util/DimVector.h (DimVector) and
//             pytorch/aten/src/ATen/TensorIterator.h L118, L251 (StrideVector).
// Status: Phase 1 — type aliases + small helpers. No CUDA.
//
// PyTorch stores dims / element-strides as `SmallVector<int64_t, 6>` and
// OperandInfo's byte-strides as `SmallVector<int64_t, 6>`. flame-core's
// Rust-side convention is element-strides as `SmallVec<[usize; 6]>` (see
// `crate::shape::Strides`). We reuse that alias for shape / element-stride
// vectors and introduce a parallel `I64StrideVec` for the byte-stride view
// that PyTorch keeps in `OperandInfo::stride_bytes`. The `i64` width is kept
// because `byte_strides()` is the FFI-facing accessor (Phase 2 kernels take
// it as `int64_t*`), and because PyTorch's `can_use_32bit_indexing` bound
// (max offset in bytes) is an `int64_t` comparison against `INT_MAX`.

use smallvec::SmallVec;

/// Shape / element-stride small-vec. Same alias as `crate::shape::Strides`
/// (`SmallVec<[usize; 6]>`). Re-exported here so `tensor_iterator` callers
/// don't have to reach into `crate::shape` for the type.
pub type DimVec = crate::shape::Strides;

/// Element-stride alias. Same underlying type as `DimVec`; the distinct
/// name documents intent at the call site (shape dims vs. a stride array).
pub type StrideVec = crate::shape::Strides;

/// Byte-stride small-vec, matching PyTorch's `OperandInfo::stride_bytes`.
/// Stored as `i64` because:
///   1. PyTorch's analogue is `int64_t`, and the FFI layer Phase 2 passes
///      these straight to device-side `int64_t strides[NARGS][MAX_DIMS]`.
///   2. Broadcast introduces stride=0 (fine for `usize`) but stride
///      arithmetic in `can_use_32bit_indexing` multiplies `(shape-1) * stride`
///      and sums, which fits cleanly in `i64` up to ~9 EB offsets.
///   3. A future `torch.flip`-style op may need signed strides; reserving
///      the signedness now avoids a breaking change later.
pub type I64StrideVec = SmallVec<[i64; 6]>;

/// Compute element-strides (row-major contiguous) for a shape. Returned
/// as the same `DimVec` alias flame-core uses everywhere, for
/// interchangeability with `Shape::stride_contiguous`.
///
/// `stride[rank-1] = 1`, `stride[i] = stride[i+1] * shape[i+1]`. For
/// rank 0, returns an empty vec. Identical to `Shape::stride_contiguous`
/// except it takes a raw slice so callers don't need to wrap a temporary
/// shape (used by `compute_strides` when the iterator's shape has been
/// permuted/coalesced and no `Shape` exists).
#[inline]
pub fn contiguous_element_strides(dims: &[usize]) -> StrideVec {
    let rank = dims.len();
    let mut strides: StrideVec = smallvec::smallvec![1; rank];
    if rank <= 1 {
        return strides;
    }
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Convert element-strides (`usize`) to byte-strides (`i64`) by multiplying
/// by `elem_size`. Used by `OperandInfo::byte_strides()` and by
/// `compute_strides` to initialise the iterator's per-operand byte-stride
/// array.
#[inline]
pub fn element_strides_to_bytes(elem_strides: &[usize], elem_size: usize) -> I64StrideVec {
    let mut out: I64StrideVec = smallvec::smallvec![0i64; elem_strides.len()];
    let es = elem_size as i64;
    for (i, &s) in elem_strides.iter().enumerate() {
        out[i] = (s as i64) * es;
    }
    out
}
