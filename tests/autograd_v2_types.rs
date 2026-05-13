//! Autograd v2 Phase 1 — type composition + behavior tests.
//!
//! These tests pin the Phase 1 contracts:
//!
//! 1. Public types compose (instantiate without lifetime / send-sync issues).
//! 2. `Hooks::default()` is empty.
//! 3. `SavedTensor::unpack` returns `Err(VersionMismatch)` after an in-place
//!    bump (no panic).
//! 4. `SavedTensor::unpack` returns `Err(SavedTensorReleased)` after `reset()`.
//! 5. `InputBuffer::add` takes the in-place path when `create_graph=false`
//!    AND dtype/shape match.
//! 6. `InputBuffer::add` takes the out-of-place path when `create_graph=true`.
//! 7. `InputBuffer::add` returns `Err(DtypeMismatch)` on dtype mismatch.
//! 8. The Weak accumulator design works — dropping the meta Arc makes
//!    `Weak::upgrade()` return None (no cycle, no leak).

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::sync::Arc;

use flame_core::autograd_v2::{
    new_meta_ref, AccumulateGrad, AutogradMetaRef, AutogradMetaV2, AutogradV2Error, DispatchCtx,
    Edge, GradFn, Hooks, InputBuffer, NodeId, SavedTensor,
};
use flame_core::{global_cuda_device, DType, Device, Shape, Tensor};

// -----------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------

fn make_f32(values: Vec<f32>, dims: &[usize]) -> Tensor {
    let device = global_cuda_device().clone();
    Tensor::from_vec(values, Shape::from_dims(dims), device).expect("from_vec")
}

fn make_bf16(values: Vec<f32>, dims: &[usize]) -> Tensor {
    make_f32(values, dims)
        .to_dtype(DType::BF16)
        .expect("to_dtype BF16")
}

fn default_ctx() -> DispatchCtx {
    let dev = Device::from_arc(global_cuda_device().clone());
    DispatchCtx::default_for(dev)
}

// -----------------------------------------------------------------------
// 1. types compose
// -----------------------------------------------------------------------

#[test]
fn types_compose() {
    // NodeId
    let n1 = NodeId::new();
    let n2 = NodeId::new();
    assert_ne!(n1, n2, "NodeId::new should be monotonic");

    // Edge::null
    let e = Edge::null();
    assert!(!e.is_valid());

    // Hooks default empty
    let h = Hooks::default();
    assert!(h.is_empty());

    // AutogradMetaV2 leaf forms
    let _ = AutogradMetaV2::leaf_no_grad();
    let _ = AutogradMetaV2::leaf_requires_grad();

    // AutogradMetaRef alias
    let _meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());

    // DispatchCtx
    let _ctx = default_ctx();
}

// -----------------------------------------------------------------------
// 2. Hooks default empty
// -----------------------------------------------------------------------

#[test]
fn hooks_default_empty() {
    let h = Hooks::default();
    assert!(h.pre_backward.is_empty());
    assert!(h.post_backward.is_empty());
    assert!(h.tensor_hooks.is_empty());
    assert!(h.is_empty());
}

#[test]
fn hooks_empty_sentinel_is_stable() {
    // The empty sentinel from `Hooks::empty_ref()` is a OnceLock-backed
    // singleton — two calls return the same address. This is what the
    // no-hook fast path in Phase 2's dispatcher will rely on.
    let a = Hooks::empty_ref() as *const _ as usize;
    let b = Hooks::empty_ref() as *const _ as usize;
    assert_eq!(a, b);
}

// -----------------------------------------------------------------------
// 3. SavedTensor version mismatch -> Err (no panic)
// -----------------------------------------------------------------------

#[test]
fn saved_tensor_version_mismatch_returns_err() {
    // Note: flame-core's default `shared_storage` feature uses
    // `Arc::make_mut` (COW) inside `ensure_unique_slice`, so calling
    // `add_inplace_same_dtype` on a tensor whose storage is also held
    // by a SavedTensor clone produces a *fresh* unique storage for the
    // mutator and leaves the saved clone's storage unchanged. This is
    // a flame-core-wide property — not specific to v2 — and is why
    // PyTorch's in-place mutators check for live SavedVariables before
    // mutating. The realistic v2 test case is a view (e.g. `narrow()`)
    // that genuinely shares the same storage Arc with its parent.
    //
    // For the Phase 1 type-test, we exercise the version-mismatch
    // detection mechanism directly via the `_test_bump_saved_version`
    // helper — that gates the `unpack()` error path without depending
    // on the COW semantics. Whether the in-place mutators *should*
    // bump-through-shared-storage for non-COW dtypes is a flame-core
    // contract question tracked separately (the existing
    // `inplace_version_bump_audit` tests already verify the bumps fire
    // on uniquely-owned storage).
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let saved = SavedTensor::save_named(&a, "TestOp");

    saved._test_bump_saved_version();

    let res = saved.unpack();
    match res {
        Err(AutogradV2Error::VersionMismatch {
            op,
            expected,
            actual,
        }) => {
            assert_eq!(op, "TestOp");
            assert!(actual > expected, "live version should advance");
        }
        Err(other) => panic!("expected VersionMismatch, got {other:?}"),
        Ok(_) => panic!("expected unpack to fail after version bump"),
    }
}

// -----------------------------------------------------------------------
// 4. SavedTensor released -> Err
// -----------------------------------------------------------------------

#[test]
fn saved_tensor_released_returns_err() {
    let a = make_f32(vec![1.0, 2.0], &[2]);
    let saved = SavedTensor::save(&a);

    // Sanity: fresh save unpacks fine.
    assert!(saved.unpack().is_ok());

    // Release through &self (the whole point of the contract — no
    // &mut needed).
    saved.reset();
    assert!(saved.is_released());

    let res = saved.unpack();
    match res {
        Err(AutogradV2Error::SavedTensorReleased) => {}
        Err(other) => panic!("expected SavedTensorReleased, got {other:?}"),
        Ok(_) => panic!("expected unpack to fail after reset"),
    }
}

// -----------------------------------------------------------------------
// 5. InputBuffer in-place when shape+dtype match AND create_graph=false
// -----------------------------------------------------------------------

#[test]
fn input_buffer_inplace_when_shapes_match() {
    let mut buf = InputBuffer::new(1, /* create_graph = */ false);
    let ctx = default_ctx();

    // First contributor — empty slot, no accumulation kind chosen.
    let g0 = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    buf.add(0, g0, &ctx).expect("first add");
    assert_eq!(buf.inplace_count(), 0);
    assert_eq!(buf.outofplace_count(), 0);

    // Second contributor — same shape/dtype, create_graph=false ⇒
    // in-place path must fire.
    let g1 = make_f32(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    buf.add(0, g1, &ctx).expect("second add");
    assert_eq!(
        buf.inplace_count(),
        1,
        "second add should have taken the in-place path"
    );
    assert_eq!(buf.outofplace_count(), 0);

    // Sanity: the buffered grad is the sum.
    let final_g = buf.peek(0).expect("slot 0 populated");
    let host = final_g.to_vec().expect("to_vec");
    assert_eq!(host, vec![11.0, 22.0, 33.0, 44.0]);
}

// -----------------------------------------------------------------------
// 6. InputBuffer out-of-place when create_graph=true
// -----------------------------------------------------------------------

#[test]
fn input_buffer_outofplace_when_create_graph_true() {
    let mut buf = InputBuffer::new(1, /* create_graph = */ true);
    let ctx = default_ctx();

    let g0 = make_f32(vec![1.0, 2.0], &[2]);
    buf.add(0, g0, &ctx).expect("first add");
    let g1 = make_f32(vec![10.0, 20.0], &[2]);
    buf.add(0, g1, &ctx).expect("second add");

    assert_eq!(
        buf.inplace_count(),
        0,
        "create_graph=true should never take in-place"
    );
    assert_eq!(buf.outofplace_count(), 1);

    let final_g = buf.peek(0).expect("slot 0 populated");
    let host = final_g.to_vec().expect("to_vec");
    assert_eq!(host, vec![11.0, 22.0]);
}

// -----------------------------------------------------------------------
// 7. InputBuffer dtype mismatch -> Err
// -----------------------------------------------------------------------

#[test]
fn input_buffer_dtype_mismatch_returns_err() {
    let mut buf = InputBuffer::new(1, false);
    let ctx = default_ctx();

    let g0 = make_f32(vec![1.0, 2.0], &[2]);
    buf.add(0, g0, &ctx).expect("first add");

    let g1 = make_bf16(vec![10.0, 20.0], &[2]);
    let res = buf.add(0, g1, &ctx);
    match res {
        Err(AutogradV2Error::DtypeMismatch { existing, incoming }) => {
            assert_eq!(existing, DType::F32);
            assert_eq!(incoming, DType::BF16);
        }
        Err(other) => panic!("expected DtypeMismatch, got {other:?}"),
        Ok(()) => panic!("expected dtype mismatch to fail"),
    }
}

// -----------------------------------------------------------------------
// 8. Weak accumulator design — no cycle, no leak
// -----------------------------------------------------------------------

#[test]
fn weak_accumulator_drops_with_variable() {
    // Build a meta Arc and a leaf accumulator pointing weakly at it.
    let meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let acc = AccumulateGrad::new(&meta, /* sequence_nr = */ 0);

    // Sanity: the variable is alive while `meta` exists.
    assert!(acc.variable_alive());

    // Drop the meta Arc. There must be no strong cycle keeping it
    // alive — the AccumulateGrad's `variable` field is `Weak`, and
    // we never installed `acc` into `meta.grad_accumulator`.
    drop(meta);

    // The Weak inside acc must now fail to upgrade.
    assert!(
        !acc.variable_alive(),
        "AccumulateGrad::variable should be Weak — meta drop should invalidate it"
    );
}

#[test]
fn weak_accumulator_in_meta_slot_no_cycle() {
    // Build a meta Arc. Build an AccumulateGrad pointing weakly at it.
    // Install the accumulator weakly into the meta's
    // grad_accumulator slot. Drop the strong handle to the
    // accumulator. The Weak in meta must NOT keep the Arc alive.
    let meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let acc: Arc<dyn GradFn> = Arc::new(AccumulateGrad::new(&meta, 0));

    // Install weakly.
    {
        let mut m = meta.lock().unwrap();
        m.grad_accumulator = Arc::downgrade(&acc);
    }

    // Strong count is 1 (the local Arc we hold).
    assert_eq!(Arc::strong_count(&acc), 1);

    // Drop the strong accumulator handle.
    drop(acc);

    // The weak in meta must not have kept it alive.
    let m = meta.lock().unwrap();
    assert!(
        m.grad_accumulator.upgrade().is_none(),
        "Weak in meta should not keep accumulator alive after strong drop"
    );
}

// -----------------------------------------------------------------------
// AccumulateGrad::apply with None grad is a no-op (Phase 2 semantics)
// -----------------------------------------------------------------------

#[test]
fn accumulate_grad_apply_none_grad_is_noop() {
    // Phase 2 changed the semantics: passing `None` returns Ok(empty
    // edges) instead of NotImplementedYet. Real accumulation logic is
    // exercised by `tests/autograd_v2_engine.rs` (engine drives end-
    // to-end).
    let meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let acc = AccumulateGrad::new(&meta, 0);
    let ctx = default_ctx();
    let res = GradFn::apply(&acc, vec![None], &ctx);
    let out = res.expect("None grad should be a silent no-op");
    assert!(out.is_empty(), "AccumulateGrad has no downstream edges");
    // meta.grad must still be None.
    let m = meta.lock().unwrap();
    assert!(m.grad.is_none(), "no grad should be stored for None input");
}

// -----------------------------------------------------------------------
// SavedTensor forward-mode AD companion slot
// -----------------------------------------------------------------------

#[test]
fn saved_tensor_fw_grad_roundtrip() {
    let a = make_f32(vec![1.0], &[1]);
    let saved = SavedTensor::save(&a);
    assert!(saved.fw_grad().is_none());
    let dual = make_f32(vec![7.0], &[1]);
    saved.set_fw_grad(dual);
    let read = saved.fw_grad().expect("fw_grad present after set");
    let host = read.to_vec().expect("to_vec");
    assert_eq!(host, vec![7.0]);
}

// -----------------------------------------------------------------------
// InputBuffer slot out of bounds
// -----------------------------------------------------------------------

#[test]
fn input_buffer_oob_slot_returns_err() {
    let mut buf = InputBuffer::new(2, false);
    let ctx = default_ctx();
    let g = make_f32(vec![1.0], &[1]);
    let res = buf.add(5, g, &ctx);
    match res {
        Err(AutogradV2Error::InputSlotOutOfBounds { slot, num_inputs }) => {
            assert_eq!(slot, 5);
            assert_eq!(num_inputs, 2);
        }
        other => panic!("expected InputSlotOutOfBounds, got {other:?}"),
    }
}
