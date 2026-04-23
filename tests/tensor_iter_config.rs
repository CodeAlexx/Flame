//! Phase 1 exit tests for `flame_core::tensor_iterator`.
//!
//! Covers, per the port reference doc (`flame-core/docs/TENSORITERATOR_PORT_REFERENCE.md`
//! §10/§11) and plan-this-and-fix-encapsulated-hennessy.md Phase 1:
//!   1. broadcast shape inference (right-aligned, singleton-on-either-side);
//!   2. stride-0 emitted on broadcast dims;
//!   3. dim reorder (stride-ascending → innermost-first);
//!   4. dim coalesce (merge adjacent when stride chain connects);
//!   5. 32-bit indexing flag;
//!   6. `is_contiguous` behaviour on a standard contiguous tensor vs a
//!      permuted `as_strided` view.
//!
//! Phase 3 wired real `build_unary_op`/`build_binary_op` — the end-to-end
//! coverage for those lives in `tests/tensor_iter_builders.rs`. The
//! "Phase-3 stub" assertions that were here in Phase 1 are gone because
//! the stubs themselves are gone.
//!
//! These are deliberately *not* tautological — each test asserts a
//! property the spec dictates (PyTorch behaviour or flame-core
//! convention), not a constant baked into the implementation.

use flame_core::tensor_iterator::{
    broadcast_pair, can_use_32bit_indexing, coalesce_dimensions, compute_shape, compute_strides,
    reorder_dimensions, OperandView, TensorIteratorConfig,
};
use flame_core::Error;

use smallvec::{smallvec, SmallVec};

type I64StrideVec = SmallVec<[i64; 6]>;


// ---------- 1. Broadcast shape ----------

#[test]
fn broadcast_shape_left_align_41x3_and_2x3() {
    // `[4, 1, 3]` against `[2, 3]` → `[4, 2, 3]`. The shorter operand is
    // right-aligned; its missing leftmost dims are implicit 1s that
    // broadcast up to the other operand's size.
    let a_shape = [4usize, 1, 3];
    let a_strides = [3usize, 3, 1];
    let b_shape = [2usize, 3];
    let b_strides = [3usize, 1];
    let ops = [
        OperandView { shape: &a_shape, element_strides: &a_strides, elem_size: 2 },
        OperandView { shape: &b_shape, element_strides: &b_strides, elem_size: 2 },
    ];
    let (shape, all_same) = compute_shape(&ops).unwrap();
    assert_eq!(shape.as_slice(), &[4, 2, 3]);
    assert!(!all_same, "operands differ in shape, compute_shape must set all_ops_same_shape=false");
}

#[test]
fn broadcast_shape_rejects_mismatched_nonone_dim() {
    // `[4, 3]` vs `[4, 2]` — same rank, last dim disagrees and neither is 1.
    // PyTorch raises; flame-core returns Err(BroadcastIncompatible).
    let a_shape = [4usize, 3];
    let b_shape = [4usize, 2];
    let a_strides = [3usize, 1];
    let b_strides = [2usize, 1];
    let ops = [
        OperandView { shape: &a_shape, element_strides: &a_strides, elem_size: 2 },
        OperandView { shape: &b_shape, element_strides: &b_strides, elem_size: 2 },
    ];
    match compute_shape(&ops) {
        Err(Error::BroadcastIncompatible { .. }) => {}
        other => panic!("expected BroadcastIncompatible, got {:?}", other.err()),
    }
}

// ---------- 2. Stride-0 on broadcast ----------

#[test]
fn compute_strides_stride_zero_on_broadcast_dim() {
    // Operand A shape [4, 1, 3] element-strides [3, 3, 1] → byte-strides (BF16, 2B):
    //   bcast=[4,2,3]; dim 1 is a broadcast → stride_bytes=0.
    // Operand B shape [2, 3] → left-padded with a zero-stride dim at index 0;
    //   remaining dims come from element-strides * 2.
    let a_shape = [4usize, 1, 3];
    let a_strides = [3usize, 3, 1];
    let b_shape = [2usize, 3];
    let b_strides = [3usize, 1];
    let ops = [
        OperandView { shape: &a_shape, element_strides: &a_strides, elem_size: 2 },
        OperandView { shape: &b_shape, element_strides: &b_strides, elem_size: 2 },
    ];
    let bcast = [4usize, 2, 3];
    let s = compute_strides(&bcast, &ops);

    // A's broadcast middle dim → 0 in byte-strides.
    assert_eq!(s[0].as_slice(), &[6, 0, 2]);
    // B's left-padded leading dim → 0 in byte-strides.
    assert_eq!(s[1].as_slice(), &[0, 6, 2]);
}

// ---------- 3. Reorder dimensions ----------

#[test]
fn reorder_contig_perm_matches_pytorch_reverse() {
    // For a row-major contiguous [2, 3, 4] tensor, PyTorch's
    // `reorder_dimensions` returns `perm_ = [2, 1, 0]` — it inverts the
    // C-contiguous order so `strides[0]` is the smallest (innermost).
    // Verified by tracing TensorIterator.cpp:232-308 at commit-pinned
    // line numbers (see reference doc §13).
    let mut shape: SmallVec<[usize; 6]> = smallvec![2usize, 3, 4];
    let mut strides: Vec<I64StrideVec> = vec![smallvec![24i64, 8i64, 2i64]];
    let perm = reorder_dimensions(&mut shape, &mut strides);

    assert_eq!(perm.as_slice(), &[2, 1, 0], "PyTorch reverses C-contig perm");
    // Apply: new_shape[i] = old_shape[perm[i]] → [4, 3, 2]; strides [2, 8, 24].
    assert_eq!(shape.as_slice(), &[4, 3, 2]);
    assert_eq!(strides[0].as_slice(), &[2, 8, 24]);
}

#[test]
fn reorder_ambiguous_case_continues_like_pytorch() {
    // Regression for the P0-1 bug where the Rust port broke out of the
    // insertion-sort inner loop on ambiguous comparisons. PyTorch's
    // TensorIterator.cpp:293-304 keeps probing outward (dim0--) with
    // dim1 fixed, so a deeper operand disambiguation can still fire.
    //
    // Setup (2 operands, ndim=4, shape = [4,4,4,4]):
    //   op0 strides = [1, 1, 1, 1]  — always ambiguous on op0 alone.
    //   op1 strides = [5, 0, 7, 2]  — broadcast (stride=0) at dim 1.
    //
    // Hand-trace of PyTorch's algorithm (fall through on every equal
    // op0 stride since shape tie-break also ties):
    //   i=1: cmp(perm[0]=3, perm[1]=2): op0 equal, shape equal, op1 2<7 → -1, break.
    //        perm = [3,2,1,0].
    //   i=2: dim1=2, dim0=1. cmp(perm[1]=2, perm[2]=1): op0 equal, shape
    //        equal, op1 stride1=op1[1]=0 → skip op1. cmp=0. (dim0 keeps
    //        decrementing under the fix.) dim0=0: cmp(perm[0]=3, perm[2]=1):
    //        op0 equal, shape equal, op1 stride1=0 → skip. cmp=0. Exit loop.
    //        perm = [3,2,1,0].
    //   i=3: dim1=3, dim0=2. cmp(perm[2]=1, perm[3]=0): op1 stride0=op1[1]=0
    //        → skip. cmp=0. (FIX: dim0--). dim0=1: cmp(perm[1]=2, perm[3]=0):
    //        op1 stride0=op1[2]=7, stride1=op1[0]=5. 7>5 → return 1. swap
    //        perm[1] ↔ perm[3], perm = [3,0,1,2], dim1=1, dim0=0.
    //        cmp(perm[0]=3, perm[1]=0): op0 equal, shape equal, op1
    //        stride0=op1[3]=2 stride1=op1[0]=5. 2<5 → return -1. break.
    //        Final perm = [3,0,1,2].
    //
    // Under the bug (break on cmp==0), the i=3 inner loop would break
    // immediately at dim0=2 and leave perm = [3,2,1,0]. The fix's
    // perm = [3,0,1,2] is what PyTorch produces.
    let mut shape: SmallVec<[usize; 6]> = smallvec![4usize, 4, 4, 4];
    let mut strides: Vec<I64StrideVec> = vec![
        smallvec![1i64, 1, 1, 1], // op0: uniform, always ambiguous on its own
        smallvec![5i64, 0, 7, 2], // op1: broadcast at dim 1
    ];
    let perm = reorder_dimensions(&mut shape, &mut strides);

    assert_eq!(
        perm.as_slice(),
        &[3, 0, 1, 2],
        "PyTorch's ambiguous-compare path (TensorIterator.cpp:293-304) \
         continues the insertion sort on cmp==0 instead of breaking"
    );
    // Post-apply: new_shape[i] = old_shape[perm[i]] → [4,4,4,4].
    assert_eq!(shape.as_slice(), &[4, 4, 4, 4]);
    // op0 rotated by perm=[3,0,1,2] → same uniform [1,1,1,1].
    assert_eq!(strides[0].as_slice(), &[1, 1, 1, 1]);
    // op1 rotated: [op1[3], op1[0], op1[1], op1[2]] = [2, 5, 0, 7].
    assert_eq!(strides[1].as_slice(), &[2, 5, 0, 7]);
}

#[test]
fn reorder_permuted_view_moves_largest_stride_outermost() {
    // Strides [4, 1, 12] on [2, 3, 4] (a permute view). Byte-strides
    // (BF16, 2B) are [8, 2, 24]. PyTorch sorts ascending:
    //   stride 2 (dim 1) first, stride 8 (dim 0) middle, stride 24 (dim 2) last.
    // perm_ = [1, 0, 2]; after applying, shape = [3, 2, 4] and
    // byte-strides = [2, 8, 24].
    let mut shape: SmallVec<[usize; 6]> = smallvec![2usize, 3, 4];
    let mut strides: Vec<I64StrideVec> = vec![smallvec![8i64, 2i64, 24i64]];
    let perm = reorder_dimensions(&mut shape, &mut strides);

    assert_eq!(perm.as_slice(), &[1, 0, 2]);
    assert_eq!(shape.as_slice(), &[3, 2, 4]);
    assert_eq!(strides[0].as_slice(), &[2, 8, 24]);
}

// ---------- 4. Coalesce dimensions ----------

#[test]
fn coalesce_fully_collapses_contiguous() {
    // After reorder, a fully-contiguous tensor lives as `shape=[4,3,2],
    // strides=[2,8,24]`. The chain `shape[0]*stride[0] = 8 == stride[1]`
    // AND `shape[1]*stride[1] = 24 == stride[2]` lets coalesce_dimensions
    // merge everything down to rank 1 with numel=24.
    let mut shape: SmallVec<[usize; 6]> = smallvec![4usize, 3, 2];
    let mut strides: Vec<I64StrideVec> = vec![smallvec![2i64, 8i64, 24i64]];
    let changed = coalesce_dimensions(&mut shape, &mut strides);

    assert!(changed);
    assert_eq!(shape.as_slice(), &[24]);
    assert_eq!(strides[0].as_slice(), &[2]);
}

#[test]
fn coalesce_blocked_by_stride_zero_broadcast() {
    // Operand A is broadcast across the middle dim → stride_bytes=[2,0,6].
    // With a second, non-broadcast operand B that's fully contiguous
    // ([2,0,6] for A, [6,2,... wait let's just do single-operand as-is).
    // Use two operands so the "per-operand" check has something to
    // disagree about: op0 has stride-0 in the middle, op1 is plain
    // contig. Neither merger direction is consistent for op0, so
    // coalesce must preserve ndim > 1.
    let mut shape: SmallVec<[usize; 6]> = smallvec![3usize, 2, 4];
    let mut strides: Vec<I64StrideVec> = vec![
        smallvec![2i64, 0i64, 6i64], // op0 with broadcast middle
        smallvec![2i64, 6i64, 12i64], // op1 plain (shape[0]*stride[0]=6=stride[1]; shape[1]*stride[1]=12=stride[2])
    ];
    let _ = coalesce_dimensions(&mut shape, &mut strides);
    // op0 fails the coalesce check at every boundary (2 != 0 and 0 != 6),
    // so the output must keep >=2 dims. PyTorch's exact output rank is
    // dependent on where the first non-merge boundary falls; property
    // we care about: the broadcast dim was not merged away.
    assert!(
        shape.len() >= 2,
        "coalesce must not collapse a broadcast dim (shape={:?}, strides[0]={:?})",
        shape,
        strides[0]
    );
    // Stride-0 must still exist on op0 in some merged dim — meaning:
    // the broadcast boundary is still present after coalescing.
    assert!(
        strides[0].iter().any(|&s| s == 0),
        "broadcast dim's stride=0 must survive coalescing (op0 strides={:?})",
        strides[0]
    );
}

// ---------- 5. 32-bit indexing flag ----------

#[test]
fn can_use_32bit_indexing_true_for_small_contig() {
    // numel=100, contig row-major. Well under INT32_MAX.
    let shape = [10usize, 10];
    let strides: Vec<I64StrideVec> = vec![smallvec![20i64, 2i64]];
    assert!(can_use_32bit_indexing(&shape, &strides));
}

#[test]
fn can_use_32bit_indexing_false_for_huge_numel() {
    // numel > INT32_MAX. Reject.
    if usize::BITS < 64 {
        return;
    }
    let huge = i32::MAX as usize + 10;
    let shape = [1usize, huge];
    let strides: Vec<I64StrideVec> = vec![smallvec![huge as i64, 1i64]];
    assert!(!can_use_32bit_indexing(&shape, &strides));
}

// ---------- 6. `is_contiguous` via TensorIteratorBase ----------
//
// These tests construct a base iterator from a real tensor via the
// public config → build API, then run the Phase-1 geometry helper to
// populate shape/strides/coalesce before inspecting `is_contiguous`.

// These tests run the Phase-1 geometry pipeline (compute_shape →
// compute_strides → reorder_dimensions → coalesce_dimensions) on real
// tensors and check the post-coalesce state directly against the
// contract of `TensorIteratorBase::is_contiguous` — namely: true iff
// `numel == 1` or `ndim == 1` with every operand's inner byte-stride
// equal to its element size (TensorIterator.cpp:806).
//
// We operate on raw shape/stride arrays rather than reaching into the
// iterator's private state; Phase 3 replaces this with `build_*_op` +
// `it.is_contiguous()` directly.

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn run_geometry_pipeline(
    operands: &[OperandView<'_>],
) -> (SmallVec<[usize; 6]>, Vec<I64StrideVec>) {
    // 1. Broadcast shape.
    let (mut shape, _all_same) =
        compute_shape(operands).expect("compute_shape must succeed on equal-shape tensors");
    // 2. Per-operand byte-strides.
    let strides_vec = compute_strides(&shape, operands);
    let mut flat: Vec<I64StrideVec> = strides_vec.into_iter().collect();
    // 3. Reorder + coalesce in place.
    let _perm = reorder_dimensions(&mut shape, &mut flat);
    coalesce_dimensions(&mut shape, &mut flat);
    (shape, flat)
}

/// Replicates `TensorIteratorBase::is_contiguous` (base.rs:296, port of
/// TensorIterator.cpp:806) against raw coalesced shape + per-operand
/// byte-strides + per-operand elem-size.
fn is_contiguous_property(
    shape: &[usize],
    byte_strides: &[I64StrideVec],
    elem_sizes: &[usize],
) -> bool {
    let numel: usize = shape.iter().product();
    if numel == 1 {
        return true;
    }
    if shape.len() != 1 {
        return false;
    }
    for (op, &es) in byte_strides.iter().zip(elem_sizes.iter()) {
        if op.is_empty() {
            continue;
        }
        if op[0] != es as i64 {
            return false;
        }
    }
    true
}

#[test]
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn is_contiguous_true_on_row_major_bf16_input() {
    use flame_core::{DType, Shape, Tensor};

    let dev = flame_core::global_cuda_device();
    // Plain row-major BF16 tensor. .randn_seeded gives us a real device
    // buffer without materializing anything tricky; shape is arbitrary.
    let x = Tensor::randn_seeded(Shape::from_dims(&[2, 3, 4]), 0.0, 1.0, 42u64, dev.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
        .expect("construct BF16 input");
    let y_store = Tensor::zeros_dtype(Shape::from_dims(&[2, 3, 4]), DType::BF16, dev.clone())
        .expect("alloc output");

    // Phase-3 will wire these via `build_binary_op`; Phase 1 just checks
    // that the config accepts the operands.
    let _it = TensorIteratorConfig::new()
        .add_output(Some(&y_store))
        .add_input(&x)
        .build()
        .expect("config build must succeed for matching-shape BF16 ops");

    // Mirror the config's operand order: output first (y_store), input
    // second (x). Run the geometry pipeline against their real shapes +
    // element-strides.
    let y_strides = y_store.strides();
    let x_strides = x.strides();
    let elem = DType::BF16.size_in_bytes();
    let operands = [
        OperandView { shape: y_store.shape().dims(), element_strides: y_strides.as_slice(), elem_size: elem },
        OperandView { shape: x.shape().dims(), element_strides: x_strides.as_slice(), elem_size: elem },
    ];
    let (shape, flat) = run_geometry_pipeline(&operands);
    let elem_sizes = [elem, elem];

    assert!(
        is_contiguous_property(&shape, &flat, &elem_sizes),
        "row-major BF16 contig input must coalesce to is_contiguous (shape post-coalesce={:?})",
        shape
    );
    // After coalescing a fully contig tensor, ndim must be exactly 1.
    assert_eq!(shape.len(), 1, "contiguous tensor coalesces to rank 1");
    // And numel must equal the original element count.
    let numel: usize = shape.iter().product();
    assert_eq!(numel, 2 * 3 * 4);
}

#[test]
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn is_contiguous_false_on_permuted_as_strided_view() {
    use flame_core::{DType, Shape, Tensor};

    let dev = flame_core::global_cuda_device();
    // Build a contig base, then construct a permuted view via
    // `Tensor::as_strided`: shape [2,3,4] with element-strides
    // [4, 1, 12] (dim 2 now stride 12, dim 1 stride 1 — a permutation of
    // the C-contig [12,4,1]).
    let base_t = Tensor::randn_seeded(Shape::from_dims(&[2, 3, 4]), 0.0, 1.0, 7u64, dev.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
        .expect("construct BF16 base");
    let view = base_t
        .as_strided(&[2, 3, 4], &[4, 1, 12], 0)
        .expect("permuted view");
    let y_store = Tensor::zeros_dtype(Shape::from_dims(&[2, 3, 4]), DType::BF16, dev.clone())
        .expect("alloc output");

    let _it = TensorIteratorConfig::new()
        .add_output(Some(&y_store))
        .add_input(&view)
        .build()
        .expect("config build must succeed on permuted view");

    let y_strides = y_store.strides();
    let view_strides = view.strides();
    let elem = DType::BF16.size_in_bytes();
    let operands = [
        OperandView { shape: y_store.shape().dims(), element_strides: y_strides.as_slice(), elem_size: elem },
        OperandView { shape: view.shape().dims(), element_strides: view_strides.as_slice(), elem_size: elem },
    ];
    let (shape, flat) = run_geometry_pipeline(&operands);
    let elem_sizes = [elem, elem];

    // A permuted view doesn't coalesce into a single row-major dim —
    // some operand's stride chain disagrees. `is_contiguous` must be
    // false because the view and the contig output can't both be 1-D.
    assert!(
        !is_contiguous_property(&shape, &flat, &elem_sizes),
        "permuted view must NOT report is_contiguous: post-coalesce shape={:?} byte_strides[1]={:?}",
        shape,
        flat[1],
    );
}

// Phase 3 wired the real `build_unary_op`/`build_binary_op` — the
// previously-present "stub returns NotImplemented" assertions were
// deleted when Phase 3 landed. End-to-end coverage of the real builders
// lives in `tests/tensor_iter_builders.rs`.

// ---------- 7. Sanity: broadcast_pair leak-through ----------

#[test]
fn broadcast_pair_left_align_scalar_shape() {
    // `[3]` broadcasts against `[4, 2, 3]`. The single-dim shape is
    // right-aligned and implicit 1s fill the left.
    let out = broadcast_pair(&[3], &[4, 2, 3]).unwrap();
    assert_eq!(out.as_slice(), &[4, 2, 3]);
}

#[test]
fn broadcast_pair_zero_dim_propagates() {
    // PyTorch spec: a dim of size 1 maps to the other side even if
    // the other side is 0 (empty tensor). Match that.
    let out = broadcast_pair(&[1, 3], &[0, 1]).unwrap();
    assert_eq!(out.as_slice(), &[0, 3]);
}
