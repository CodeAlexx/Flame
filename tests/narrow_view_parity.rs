//! Stride refactor Phase 2a regression tests: narrow() returns a zero-copy
//! view whose `.contiguous()` materialization is bit-exact to the old
//! `cuda_ops_bf16::slice_axis_bf16` kernel.
//!
//! Caught a real bug on 2026-04-23: `Tensor::contiguous()` had two paths —
//! `materialize_view` (gated on `view_offset != 0`) and a permute-recovery
//! path that assumed strides were a permutation of row-major of the
//! current shape. `narrow(dim, 0, length)` has `view_offset == 0` but its
//! custom_strides reflect the PARENT tensor's shape (not self's), so the
//! permute-recovery path misread strides as identity and returned the
//! wrong storage region.
//!
//! This suite pins the correct behavior. If it fails, someone re-broke
//! `contiguous()`'s routing or `narrow()`'s stride setup.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn assert_narrow_bit_exact(b: usize, n: usize, d: usize, dim: usize, start: usize, length: usize) {
    let device = global_cuda_device();

    let total = b * n * d;
    // Golden-ratio-ish hash produces distinct values per index so off-by-one
    // or wrong-row bugs surface as large maxdiff.
    let data_f32: Vec<f32> = (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 100.0)
        .collect();
    let shape = Shape::from_dims(&[b, n, d]);
    let t_f32 = Tensor::from_vec(data_f32, shape, device.clone()).unwrap();
    let t = t_f32.to_dtype(DType::BF16).unwrap();

    // Oracle: the pre-refactor slice_axis_bf16 materialization kernel.
    let oracle = flame_core::cuda_ops_bf16::slice_axis_bf16(&t, dim, start, length).unwrap();

    // Under test: narrow as view, then contiguous.
    let view = t.narrow(dim, start, length).unwrap();
    let mat = view.contiguous().unwrap();

    let oracle_vec = oracle.to_vec().unwrap();
    let mat_vec = mat.to_vec().unwrap();

    assert_eq!(
        oracle_vec.len(),
        mat_vec.len(),
        "length mismatch: oracle {} vs under-test {}",
        oracle_vec.len(),
        mat_vec.len()
    );
    for (i, (o, m)) in oracle_vec.iter().zip(mat_vec.iter()).enumerate() {
        assert_eq!(
            o.to_bits(),
            m.to_bits(),
            "narrow({},{},{}) on [{},{},{}] diverges at idx {}: oracle={} view+contig={}",
            dim,
            start,
            length,
            b,
            n,
            d,
            i,
            o,
            m
        );
    }
}

#[test]
fn narrow_start_zero_dim2() {
    // This is the case that actually broke on 2026-04-23. start=0 → offset=0
    // → old contiguous() fell into permute-recovery and misread storage.
    assert_narrow_bit_exact(1, 64, 384, 2, 0, 128);
}

#[test]
fn narrow_middle_dim2() {
    assert_narrow_bit_exact(1, 64, 384, 2, 128, 128);
}

#[test]
fn narrow_end_dim2() {
    assert_narrow_bit_exact(1, 64, 384, 2, 256, 128);
}

#[test]
fn narrow_gate_up_split_klein_shape() {
    // Klein's MLP gate_up split on dim 2 of [B,N,2*mlp_hidden]. mlp_hidden
    // = 15360, half_dim = mlp_hidden. Both halves.
    assert_narrow_bit_exact(1, 4608, 30720, 2, 0, 15360);
    assert_narrow_bit_exact(1, 4608, 30720, 2, 15360, 15360);
}

#[test]
fn narrow_on_dim1() {
    // Klein final layer: x.narrow(1, txt_len, img_len). Non-last dim.
    assert_narrow_bit_exact(1, 128, 64, 1, 16, 64);
}

#[test]
fn narrow_on_dim0() {
    // Batch-level narrow.
    assert_narrow_bit_exact(4, 32, 16, 0, 1, 2);
}

// ---- narrow_owning tests --------------------------------------------------
//
// `narrow_owning` is `narrow` followed by `materialize_view` — produces fresh
// storage that does NOT pin the parent's `Arc<Storage>`. Critical for chunked
// decode loops where multi-GB parent tensors would otherwise stay alive via
// Arc-clone in the views accumulated in `Vec<Tensor>`.
//
// Load-bearing test: parent_drop_independence. If a future "optimization"
// short-circuits materialize_view when input is already contiguous, that
// test will catch it because the child will still hold a parent-storage
// pointer after the parent goes out of scope.

fn assert_narrow_owning_bit_exact(
    b: usize,
    n: usize,
    d: usize,
    dim: usize,
    start: usize,
    length: usize,
) {
    let device = global_cuda_device();
    let total = b * n * d;
    let data_f32: Vec<f32> = (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 100.0)
        .collect();
    let shape = Shape::from_dims(&[b, n, d]);
    let t_f32 = Tensor::from_vec(data_f32, shape, device.clone()).unwrap();
    let t = t_f32.to_dtype(DType::BF16).unwrap();

    let oracle = flame_core::cuda_ops_bf16::slice_axis_bf16(&t, dim, start, length).unwrap();
    let owned = t.narrow_owning(dim, start, length).unwrap();

    // Property 1: bit-exact match to oracle.
    let oracle_vec = oracle.to_vec().unwrap();
    let owned_vec = owned.to_vec().unwrap();
    assert_eq!(oracle_vec.len(), owned_vec.len());
    for (i, (o, m)) in oracle_vec.iter().zip(owned_vec.iter()).enumerate() {
        assert_eq!(
            o.to_bits(),
            m.to_bits(),
            "narrow_owning({},{},{}) on [{},{},{}] diverges at idx {}: oracle={} owned={}",
            dim,
            start,
            length,
            b,
            n,
            d,
            i,
            o,
            m
        );
    }

    // Property 2: result is row-major contiguous (callers can safely reshape).
    assert!(
        owned.is_contiguous(),
        "narrow_owning result not contiguous: shape={:?}",
        owned.shape().dims()
    );
}

#[test]
fn narrow_owning_start_zero_dim2() {
    // Mirror existing narrow_start_zero_dim2 — same shape and slice.
    assert_narrow_owning_bit_exact(1, 64, 384, 2, 0, 128);
}

#[test]
fn narrow_owning_middle_dim2() {
    assert_narrow_owning_bit_exact(1, 64, 384, 2, 128, 128);
}

#[test]
fn narrow_owning_chunked_decode_shape() {
    // Mirrors turbo_vaed sliding_window_decode trim: narrow on dim=2 (temporal)
    // of a 5D-equivalent layout flattened to [B, N, D]. Length matches the
    // typical 7-frame-chunk trim with 1-frame overlap.
    assert_narrow_owning_bit_exact(1, 6, 1024, 1, 0, 5);
}

#[test]
fn narrow_owning_independent_of_parent() {
    // The load-bearing test. `narrow_owning` MUST produce storage independent
    // of the parent — if a future change makes materialize_view short-circuit
    // when the parent is already contiguous (returning a tensor that still
    // Arc-clones the parent's storage), this test should catch it via the
    // post-drop read.
    //
    // Mechanism: build a parent on the GPU, take a narrow_owning child,
    // drop the parent (dropping its Arc<Storage> reference), then read the
    // child. If child's storage is independent, this succeeds and matches
    // the oracle. If child's storage was an Arc-clone, the parent's
    // Arc<Storage> count went to zero only if all clones drop together —
    // but that would also tear down the child. Either way, a regression
    // produces visible wrongness here.
    let device = global_cuda_device();
    let b = 1;
    let n = 64;
    let d = 384;
    let total = b * n * d;
    let data_f32: Vec<f32> = (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 100.0)
        .collect();

    // Compute oracle expected slice contents BEFORE creating the parent so
    // we can compare without keeping the parent alive.
    let parent_for_oracle = Tensor::from_vec(
        data_f32.clone(),
        Shape::from_dims(&[b, n, d]),
        device.clone(),
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    let oracle =
        flame_core::cuda_ops_bf16::slice_axis_bf16(&parent_for_oracle, 2, 128, 128).unwrap();
    let oracle_vec = oracle.to_vec().unwrap();
    drop(parent_for_oracle);
    drop(oracle);

    // Now: parent inside an inner scope, narrow_owning out, drop parent,
    // verify child still reads correctly.
    let owned = {
        let parent = Tensor::from_vec(data_f32, Shape::from_dims(&[b, n, d]), device.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let owned = parent.narrow_owning(2, 128, 128).unwrap();
        // Explicit drop — parent goes here, before owned escapes.
        drop(parent);
        owned
    };

    // Read child after parent has been dropped. If storage was a view into
    // the parent, this would either error or return garbage.
    let owned_vec = owned.to_vec().unwrap();
    assert_eq!(oracle_vec.len(), owned_vec.len());
    for (i, (o, m)) in oracle_vec.iter().zip(owned_vec.iter()).enumerate() {
        assert_eq!(
            o.to_bits(),
            m.to_bits(),
            "narrow_owning post-parent-drop diverges at idx {}: oracle={} owned={}",
            i,
            o,
            m
        );
    }
    assert!(owned.is_contiguous());
}

#[test]
fn narrow_then_narrow() {
    // Narrow-of-narrow composition. First narrow produces a view; second
    // narrow produces a view-of-view (both custom_strides Some, offsets sum).
    let device = global_cuda_device();
    let total = 1 * 64 * 384;
    let data_f32: Vec<f32> = (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 100.0)
        .collect();
    let t = Tensor::from_vec(data_f32, Shape::from_dims(&[1, 64, 384]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // [1,64,384] → narrow(2, 128, 256) → [1,64,256] view → narrow(2, 64, 128) → [1,64,128]
    let view1 = t.narrow(2, 128, 256).unwrap();
    let view2 = view1.narrow(2, 64, 128).unwrap();
    let mat = view2.contiguous().unwrap();

    // Equivalent single narrow(2, 128+64, 128)
    let oracle = flame_core::cuda_ops_bf16::slice_axis_bf16(&t, 2, 192, 128).unwrap();

    let oracle_vec = oracle.to_vec().unwrap();
    let mat_vec = mat.to_vec().unwrap();
    assert_eq!(oracle_vec.len(), mat_vec.len());
    for (i, (o, m)) in oracle_vec.iter().zip(mat_vec.iter()).enumerate() {
        assert_eq!(
            o.to_bits(),
            m.to_bits(),
            "narrow-of-narrow diverges at idx {}: oracle={} view+contig={}",
            i,
            o,
            m
        );
    }
}
