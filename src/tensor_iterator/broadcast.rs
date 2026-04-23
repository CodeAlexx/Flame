// Origin: flame-core TensorIterator port, Phase 1.
// Port of:
//   - pytorch/aten/src/ATen/TensorIterator.cpp L1237 (compute_shape)
//   - pytorch/aten/src/ATen/TensorIterator.cpp L1277 (compute_strides)
//   - pytorch/aten/src/ATen/TensorIterator.cpp L232  (reorder_dimensions)
//   - pytorch/aten/src/ATen/TensorIterator.cpp L638  (coalesce_dimensions)
//   - pytorch/aten/src/ATen/TensorIterator.cpp L1300 (can_use_32bit_indexing)
//   - pytorch/aten/src/ATen/ExpandUtils.cpp L17     (infer_size_impl)
//
// Scope: pure-Rust geometry. No CUDA, no Tensor allocation, no autograd.
// Every op here operates on shape+stride arrays so the algorithms can be
// unit-tested without a live GPU.

use smallvec::smallvec;

use crate::error::{Error, Result};

use super::dim_vec::{DimVec, I64StrideVec};

/// Lightweight, pre-iterator view of one operand: its logical shape and
/// element-strides (same units as `Tensor::strides`). Borrowed by the
/// broadcast helpers so they can run against raw slices in tests.
///
/// `shape.len() == element_strides.len()`.
#[derive(Clone, Debug)]
pub struct OperandView<'a> {
    pub shape: &'a [usize],
    pub element_strides: &'a [usize],
    pub elem_size: usize,
}

/// PyTorch's `infer_size_impl` specialised to `DimVec`. Broadcasts `a`
/// against `b` right-aligned: the shorter shape is padded with 1s on the
/// left, then each dim pair must match, or one must be 1 (in which case
/// the other wins — including 0, matching PyTorch's rule).
///
/// Returns `Err(BroadcastIncompatible)` when a dim-pair disagrees and
/// neither is 1.
pub fn broadcast_pair(a: &[usize], b: &[usize]) -> Result<DimVec> {
    let dims_a = a.len() as isize;
    let dims_b = b.len() as isize;
    let ndim = std::cmp::max(dims_a, dims_b);
    let mut out: DimVec = smallvec![0usize; ndim as usize];

    for i in (0..ndim).rev() {
        let offset = ndim - 1 - i;
        let dim_a = dims_a - 1 - offset;
        let dim_b = dims_b - 1 - offset;
        let size_a = if dim_a >= 0 { a[dim_a as usize] } else { 1 };
        let size_b = if dim_b >= 0 { b[dim_b as usize] } else { 1 };

        let out_size = if size_a == size_b {
            size_a
        } else if size_a == 1 {
            size_b
        } else if size_b == 1 {
            size_a
        } else {
            return Err(Error::BroadcastIncompatible {
                lhs: crate::shape::Shape::from_dims(a),
                rhs: crate::shape::Shape::from_dims(b),
            });
        };
        out[i as usize] = out_size;
    }

    Ok(out)
}

/// Port of `TensorIteratorBase::compute_shape`. Walks the operand list
/// left-to-right, broadcasting pairwise. Matches PyTorch's behaviour: if
/// all operands have the same shape, returns that shape verbatim; if
/// any operand broadcasts, the broadcast shape is inferred. Output
/// operands in PyTorch are skipped in this computation because
/// `resize_outputs_` is true by default (plan §2 Phase 1: outputs with
/// `add_output(None)` participate only as allocation slots, not shape
/// contributors).
///
/// Returns a tuple of `(shape, all_ops_same_shape)`. The second component
/// mirrors `all_ops_same_shape_` in PyTorch and is used downstream by
/// fast-set-up heuristics (Phase 3+).
pub fn compute_shape(operands: &[OperandView<'_>]) -> Result<(DimVec, bool)> {
    let mut shape: DimVec = smallvec![];
    let mut any_seen = false;
    let mut all_same = true;

    for op in operands {
        if !any_seen {
            shape = DimVec::from_slice(op.shape);
            any_seen = true;
            continue;
        }

        if shape.as_slice() == op.shape {
            continue;
        }

        all_same = false;
        shape = broadcast_pair(shape.as_slice(), op.shape)?;
    }

    Ok((shape, all_same))
}

/// Port of `TensorIteratorBase::compute_strides`. For each operand,
/// produce a byte-stride array of length `bcast_shape.len()`:
///   - left-pad with zeros if the operand has fewer dims than bcast
///   - for each original dim i: if `original_shape[i] == 1` and
///     `bcast_shape[offset+i] != 1`, the dim is broadcast → `stride_bytes[i] = 0`.
///   - otherwise `stride_bytes[i] = original_element_stride[i] * elem_size`.
///
/// The per-operand result has `len() == bcast_shape.len()` always.
///
/// This lives at `TensorIterator.cpp:1277`. The stride-0 convention on
/// broadcast dims is what makes `gpu_kernel` treat a broadcast operand
/// "as if" the dim were expanded in-memory without actually expanding it.
pub fn compute_strides(
    bcast_shape: &[usize],
    operands: &[OperandView<'_>],
) -> Vec<I64StrideVec> {
    let ndim = bcast_shape.len();
    let mut out = Vec::with_capacity(operands.len());

    for op in operands {
        let original_nd = op.shape.len();
        let offset = ndim - original_nd; // number of leading zero-stride pad dims
        let es = op.elem_size as i64;
        let mut strides: I64StrideVec = smallvec![0i64; ndim];
        for i in 0..original_nd {
            // PyTorch L1290: broadcast dim if original size is 1 but bcast isn't.
            if op.shape[i] == 1 && bcast_shape[offset + i] != 1 {
                strides[offset + i] = 0;
            } else {
                strides[offset + i] = (op.element_strides[i] as i64) * es;
            }
        }
        out.push(strides);
    }

    out
}

/// Apply a permutation `perm` to `shape` and every per-operand stride
/// array. `new[i] = old[perm[i]]`. Port of
/// `TensorIteratorBase::permute_dimensions` (TensorIterator.cpp:723).
fn permute_in_place(
    perm: &[usize],
    shape: &mut DimVec,
    byte_strides: &mut [I64StrideVec],
) {
    debug_assert_eq!(perm.len(), shape.len());
    let ndim = perm.len();

    let mut new_shape: DimVec = smallvec![0usize; ndim];
    for i in 0..ndim {
        new_shape[i] = shape[perm[i]];
    }
    *shape = new_shape;

    for s in byte_strides.iter_mut() {
        if s.is_empty() {
            continue;
        }
        debug_assert_eq!(s.len(), ndim);
        let mut new_s: I64StrideVec = smallvec![0i64; ndim];
        for i in 0..ndim {
            new_s[i] = s[perm[i]];
        }
        *s = new_s;
    }
}

/// Port of `TensorIteratorBase::reorder_dimensions` (TensorIterator.cpp:232).
///
/// Sorts dims so that the *smallest* stride comes first (innermost).
/// Returns the permutation used (same semantics as PyTorch's `perm_`:
/// `perm[new_dim] = old_dim`), and permutes `shape` + all `byte_strides`
/// in place.
///
/// Algorithm (matching PyTorch verbatim):
///   1. Initialise perm = [n-1, n-2, ..., 1, 0] (reverse).
///   2. Insertion sort with `should_swap(a, b)`:
///        - for each operand, if strides at a and b disagree (and neither
///          is 0), the smaller stride goes earlier;
///        - if strides are equal and both non-zero, the *smaller shape*
///          goes later (tie-break that prefers big-dim innermost when
///          strides tie, matching PyTorch's exact ordering).
///        - if both strides are 0 OR either is 0, move to the next operand.
///        - returns 0 (ambiguous) if no operand disambiguates.
///   3. Ambiguous comparisons leave the current pair unchanged and
///      continue the insertion sort outward, matching PyTorch's
///      `TensorIterator.cpp:293-304`. Only a strict `comparison < 0`
///      breaks out of the inner loop.
///
/// For a rank-1 input this is a no-op returning `[0]`.
pub fn reorder_dimensions(
    shape: &mut DimVec,
    byte_strides: &mut [I64StrideVec],
) -> DimVec {
    let ndim = shape.len();

    if ndim <= 1 {
        // PyTorch returns perm=[0] for ndim==1 and does not permute.
        let mut perm: DimVec = smallvec![0usize; ndim];
        for i in 0..ndim {
            perm[i] = i;
        }
        return perm;
    }

    // Reverse-iota: perm = [ndim-1, ndim-2, ..., 1, 0].
    let mut perm: DimVec = smallvec![0usize; ndim];
    for i in 0..ndim {
        perm[i] = ndim - 1 - i;
    }

    let should_swap = |shape_ref: &DimVec,
                       strides_ref: &[I64StrideVec],
                       dim0: usize,
                       dim1: usize|
     -> i32 {
        for op_strides in strides_ref.iter() {
            // NOTE (will_resize unsupported): PyTorch TensorIterator.cpp:258
            // also skips operands where will_resize == true. flame-core has
            // no implicit-resize path (allocate_or_resize_outputs errors on
            // shape mismatch instead), so this skip is a no-op in the
            // current port and carries no runtime cost.
            if op_strides.is_empty() {
                continue;
            }
            let stride0 = op_strides[dim0];
            let stride1 = op_strides[dim1];
            if stride0 == 0 || stride1 == 0 {
                continue;
            } else if stride0 < stride1 {
                return -1;
            } else if stride0 > stride1 {
                return 1;
            } else {
                // Equal non-zero strides: tie-break on shape.
                let t0 = shape_ref[dim0];
                let t1 = shape_ref[dim1];
                if t0 > t1 {
                    return 1;
                }
            }
        }
        0
    };

    // Insertion sort over perm, keyed via should_swap on perm[i] indices.
    // Mirrors PyTorch `TensorIterator.cpp:293-304`:
    //   comparison > 0 → swap and update dim1 to new position of the bubbled element
    //   comparison < 0 → break out of the inner loop
    //   comparison == 0 (ambiguous) → leave the current pair unchanged and
    //     continue probing further outward with dim1 fixed (do NOT update dim1).
    for i in 1..ndim {
        let mut dim1 = i;
        let mut dim0 = i as isize - 1;
        while dim0 >= 0 {
            let cmp = should_swap(shape, byte_strides, perm[dim0 as usize], perm[dim1]);
            if cmp > 0 {
                perm.swap(dim0 as usize, dim1);
                dim1 = dim0 as usize;
                dim0 -= 1;
            } else if cmp < 0 {
                break;
            } else {
                // Ambiguous — per PyTorch TensorIterator.cpp:293-304, do not
                // swap and do not update dim1; keep probing the next outer
                // position.
                dim0 -= 1;
            }
        }
    }

    permute_in_place(&perm, shape, byte_strides);
    perm
}

/// Port of `TensorIteratorBase::coalesce_dimensions`
/// (TensorIterator.cpp:638).
///
/// Merges adjacent dims where every operand's strides satisfy
/// `shape[prev] * stride[prev] == stride[cur]`, or where `shape[prev] == 1`
/// or `shape[cur] == 1`. After coalescing, `shape` may be shorter and each
/// operand's byte-stride array is truncated to match. Returns `true` if
/// any coalescing happened.
///
/// Mutates in place. No-op for rank ≤ 1.
pub fn coalesce_dimensions(
    shape: &mut DimVec,
    byte_strides: &mut [I64StrideVec],
) -> bool {
    let ndim = shape.len();
    if ndim <= 1 {
        return false;
    }

    let can_coalesce = |shape_ref: &DimVec,
                        strides_ref: &[I64StrideVec],
                        dim0: usize,
                        dim1: usize|
     -> bool {
        let s0 = shape_ref[dim0];
        let s1 = shape_ref[dim1];
        if s0 == 1 || s1 == 1 {
            return true;
        }
        for op_strides in strides_ref.iter() {
            if op_strides.is_empty() {
                continue;
            }
            if (s0 as i64) * op_strides[dim0] != op_strides[dim1] {
                return false;
            }
        }
        true
    };

    let replace_stride = |strides: &mut [I64StrideVec], dim0: usize, dim1: usize| {
        for op_strides in strides.iter_mut() {
            if op_strides.is_empty() {
                continue;
            }
            op_strides[dim0] = op_strides[dim1];
        }
    };

    let mut prev_dim = 0usize;
    for dim in 1..ndim {
        if can_coalesce(shape, byte_strides, prev_dim, dim) {
            if shape[prev_dim] == 1 {
                replace_stride(byte_strides, prev_dim, dim);
            }
            shape[prev_dim] *= shape[dim];
        } else {
            prev_dim += 1;
            if prev_dim != dim {
                replace_stride(byte_strides, prev_dim, dim);
                shape[prev_dim] = shape[dim];
            }
        }
    }

    let new_ndim = prev_dim + 1;
    let coalesced = new_ndim < ndim;
    shape.truncate(new_ndim);
    for op_strides in byte_strides.iter_mut() {
        if op_strides.is_empty() {
            continue;
        }
        op_strides.truncate(new_ndim);
    }
    coalesced
}

/// Port of `TensorIteratorBase::can_use_32bit_indexing`
/// (TensorIterator.cpp:1300).
///
/// Returns false if either:
///   1. `numel > INT32_MAX`, or
///   2. for any operand, `1 + sum_i (shape[i] - 1) * stride_bytes[i]`
///      exceeds `INT32_MAX`.
///
/// Note: stride is `i64` (bytes), not elements; the sum may blow past
/// 2³¹ even for moderate-element tensors if element size is large (e.g.
/// F64). BF16's worst case is ~2³⁰ elements × 2B = 2³¹B, close to the
/// edge; most flame-core tensors stay well under.
pub fn can_use_32bit_indexing(
    shape: &[usize],
    byte_strides: &[I64StrideVec],
) -> bool {
    let max_value = i32::MAX as i64;
    let mut numel: i64 = 1;
    for &s in shape {
        // usize → i64: on 64-bit hosts a 2³¹+ shape dim saturates the
        // comparison correctly; the multiplication would overflow but
        // catch that via the saturated early-return below.
        if s > i32::MAX as usize {
            return false;
        }
        numel = numel.saturating_mul(s as i64);
        if numel > max_value {
            return false;
        }
    }

    let ndim = shape.len();
    for op_strides in byte_strides.iter() {
        if op_strides.is_empty() {
            continue;
        }
        debug_assert_eq!(op_strides.len(), ndim);
        let mut max_offset: i64 = 1;
        for dim in 0..ndim {
            // Match PyTorch TensorIterator.cpp:1307-1309 — iterate every
            // dim including size-0 (produces a negative contribution that
            // still passes the < i32::MAX check).
            max_offset = max_offset.saturating_add(
                ((shape[dim] as i64) - 1).saturating_mul(op_strides[dim]),
            );
            if max_offset > max_value {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(shape: &[usize], strides: &[usize], elem_size: usize) -> (Vec<usize>, Vec<usize>, usize) {
        (shape.to_vec(), strides.to_vec(), elem_size)
    }

    #[test]
    fn broadcast_pair_left_align() {
        // [3] vs [4,2,3] -> [4,2,3]
        let out = broadcast_pair(&[3], &[4, 2, 3]).unwrap();
        assert_eq!(out.as_slice(), &[4, 2, 3]);
    }

    #[test]
    fn broadcast_pair_rejects_mismatch() {
        assert!(broadcast_pair(&[4, 3], &[4, 2]).is_err());
    }

    #[test]
    fn compute_shape_multi_operand() {
        let a = mk(&[4, 1, 3], &[3, 3, 1], 2);
        let b = mk(&[2, 3], &[3, 1], 2);
        let operands = vec![
            OperandView { shape: &a.0, element_strides: &a.1, elem_size: a.2 },
            OperandView { shape: &b.0, element_strides: &b.1, elem_size: b.2 },
        ];
        let (shape, same) = compute_shape(&operands).unwrap();
        assert_eq!(shape.as_slice(), &[4, 2, 3]);
        assert!(!same);
    }

    #[test]
    fn compute_strides_stride_zero_on_broadcast() {
        // a: [4,1,3], strides [3,3,1]; b: [2,3], strides [3,1]
        // bcast = [4,2,3]. For a, dim 1 (size 1 vs bcast 2) -> stride_bytes = 0.
        // For b, left-padded: dim 0 of bcast (size 4) is new -> stride = 0.
        let a = mk(&[4, 1, 3], &[3, 3, 1], 2);
        let b = mk(&[2, 3], &[3, 1], 2);
        let operands = vec![
            OperandView { shape: &a.0, element_strides: &a.1, elem_size: a.2 },
            OperandView { shape: &b.0, element_strides: &b.1, elem_size: b.2 },
        ];
        let bcast = [4, 2, 3];
        let strides = compute_strides(&bcast, &operands);
        // a: dim 0 full -> 6; dim 1 broadcast -> 0; dim 2 full -> 2.
        assert_eq!(strides[0].as_slice(), &[6, 0, 2]);
        // b: dim 0 padded -> 0; dim 1 full -> 6; dim 2 full -> 2.
        assert_eq!(strides[1].as_slice(), &[0, 6, 2]);
    }

    #[test]
    fn reorder_contig_is_identity_under_perm_mapping() {
        // Row-major [2,3,4]: byte-strides = [12,4,1]*2 = [24,8,2].
        // PyTorch's perm reverses C-contig: perm = [2,1,0].
        // After applying: shape = [4,3,2], strides = [2,8,24] (stride-2
        // innermost, stride-24 outermost).
        let mut shape: DimVec = smallvec![2usize, 3, 4];
        let mut strides = vec![smallvec![24i64, 8i64, 2i64] as I64StrideVec];
        let perm = reorder_dimensions(&mut shape, &mut strides);
        assert_eq!(perm.as_slice(), &[2, 1, 0]);
        assert_eq!(shape.as_slice(), &[4, 3, 2]);
        assert_eq!(strides[0].as_slice(), &[2, 8, 24]);
    }

    #[test]
    fn reorder_permuted_view() {
        // Shape [2,3,4], elem-strides [4,1,12] → byte-strides (u16)=[8,2,24].
        // Expect: dim with stride=2 innermost; dim with stride=24 outermost.
        // PyTorch perm end state: [1,0,2] (dim1 innermost, dim0 middle, dim2 outermost).
        // After applying: shape = [3,2,4], strides = [2,8,24].
        let mut shape: DimVec = smallvec![2usize, 3, 4];
        let mut strides = vec![smallvec![8i64, 2i64, 24i64] as I64StrideVec];
        let perm = reorder_dimensions(&mut shape, &mut strides);
        assert_eq!(perm.as_slice(), &[1, 0, 2]);
        assert_eq!(shape.as_slice(), &[3, 2, 4]);
        assert_eq!(strides[0].as_slice(), &[2, 8, 24]);
    }

    #[test]
    fn coalesce_full_contig() {
        // Row-major [2,3,4] already reverse-sorted to [4,3,2] with strides
        // [2,8,24] should coalesce all the way down.
        let mut shape: DimVec = smallvec![4usize, 3, 2];
        let mut strides = vec![smallvec![2i64, 8i64, 24i64] as I64StrideVec];
        let changed = coalesce_dimensions(&mut shape, &mut strides);
        assert!(changed);
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0], 24);
        assert_eq!(strides[0].as_slice(), &[2]);
    }

    #[test]
    fn coalesce_blocked_by_broadcast_zero() {
        // A broadcast dim (stride-0) in the middle blocks coalescing.
        // Shape [3,2,4], strides_a=[2,0,6]. 3*2 = 6 != 0, so coalescing
        // halts between dim 0 and dim 1; 2*0 = 0 != 6, so also blocked
        // between dim 1 and dim 2. Must remain rank 3.
        let mut shape: DimVec = smallvec![3usize, 2, 4];
        let mut strides = vec![smallvec![2i64, 0i64, 6i64] as I64StrideVec];
        let _ = coalesce_dimensions(&mut shape, &mut strides);
        assert!(
            shape.len() >= 2,
            "coalesce should not collapse across a broadcast dim: shape={:?}",
            shape
        );
    }

    #[test]
    fn can_use_32bit_small() {
        // 100 elements, contiguous BF16 row-major.
        let shape = [10usize, 10];
        let strides = vec![smallvec![20i64, 2i64] as I64StrideVec];
        assert!(can_use_32bit_indexing(&shape, &strides));
    }

    #[test]
    fn can_use_32bit_rejects_large_numel() {
        // Shape whose numel exceeds INT32_MAX (~2.14e9). 1 x (INT32_MAX + 10).
        let huge = i32::MAX as usize + 10;
        if (usize::BITS as usize) < 64 {
            // 32-bit host: skip.
            return;
        }
        let shape = [1usize, huge];
        // Contiguous byte-strides: [huge*1, 1] for a u8 tensor.
        let strides = vec![smallvec![huge as i64, 1i64] as I64StrideVec];
        assert!(!can_use_32bit_indexing(&shape, &strides));
    }
}
