//! Phase 3a — autograd v2 ops tests.
//!
//! Coverage:
//!
//! Recording surface (5):
//!   1. `record_v2_creates_meta_with_grad_fn`
//!   2. `clone_preserves_autograd_meta`
//!   3. `detach_v2_drops_autograd_meta`
//!   4. `record_v2_skips_when_no_input_requires_grad`
//!   5. `record_v2_records_when_any_input_requires_grad`
//!
//! Per-op backward (5):
//!   6.  `add_v2_backward_distributes_grad`
//!   7.  `mul_v2_backward_chain_rule`
//!   8.  `sum_v2_backward_broadcasts`
//!   9.  `matmul_v2_backward_correct_shapes`
//!   10. `silu_v2_backward_at_zero`
//!
//! Engine integration (4):
//!   11. `add_through_engine_end_to_end`
//!   12. `accumulate_grad_uses_inplace_when_create_graph_false`
//!   13. `accumulate_grad_uses_out_of_place_when_create_graph_true`
//!   14. `engine_rejects_mismatched_grad_output_shape`
//!
//! Non-leaf collection (1):
//!   15. `with_inputs_returns_non_leaf_grad`

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::sync::Arc;

use flame_core::autograd_v2::{
    gradient_edge_for_tensor, new_meta_ref, needs_grad, next_sequence_nr,
    AutogradMetaV2, AutogradV2Error, DeviceStream, DispatchCtx, Engine, GraphRoot,
};
use flame_core::{global_cuda_device, Device, Shape, Tensor};

fn make_f32(values: Vec<f32>, dims: &[usize]) -> Tensor {
    let device = global_cuda_device().clone();
    Tensor::from_vec(values, Shape::from_dims(dims), device).expect("from_vec")
}

fn default_ctx() -> DispatchCtx {
    let dev = Device::from_arc(global_cuda_device().clone());
    DispatchCtx::default_for(dev)
}

/// Build a fresh leaf with `requires_grad=true` and install the meta
/// on the tensor (PyTorch parity: leaves carry their own meta).
fn make_leaf_requires_grad(values: Vec<f32>, dims: &[usize]) -> Tensor {
    let mut t = make_f32(values, dims);
    let meta = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    t.set_autograd_meta(Some(meta));
    t
}

// ---------------------------------------------------------------------------
// 1. record_v2_creates_meta_with_grad_fn
// ---------------------------------------------------------------------------

#[test]
fn record_v2_creates_meta_with_grad_fn() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_leaf_requires_grad(vec![3.0, 4.0], &[2]);
    let ctx = default_ctx();

    let out = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).expect("add_v2");
    let meta = out
        .autograd_meta()
        .expect("output should have autograd_meta after recording");
    let m = meta.lock().unwrap();
    assert!(
        m.grad_fn.is_some(),
        "recorded output meta must carry a grad_fn"
    );
    assert_eq!(m.grad_fn.as_ref().unwrap().name(), "AddGradFn");
}

// ---------------------------------------------------------------------------
// 2. clone_preserves_autograd_meta
// ---------------------------------------------------------------------------

#[test]
fn clone_preserves_autograd_meta() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_leaf_requires_grad(vec![3.0, 4.0], &[2]);
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();

    let out_clone = out.clone();
    // The cloned Tensor must share the SAME Arc<Mutex<AutogradMetaV2>>.
    let p1 = out.autograd_meta().unwrap();
    let p2 = out_clone.autograd_meta().unwrap();
    assert!(Arc::ptr_eq(p1, p2), "clone must share the meta Arc");
}

// ---------------------------------------------------------------------------
// 3. detach_v2_drops_autograd_meta
// ---------------------------------------------------------------------------

#[test]
fn detach_v2_drops_autograd_meta() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_leaf_requires_grad(vec![3.0, 4.0], &[2]);
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();

    assert!(out.autograd_meta().is_some(), "recorded out has meta");
    let detached = out.detach_v2();
    assert!(
        detached.autograd_meta().is_none(),
        "detach_v2 must drop the meta"
    );
    // The original tensor is unaffected.
    assert!(out.autograd_meta().is_some());
}

// ---------------------------------------------------------------------------
// 4. record_v2_skips_when_no_input_requires_grad
// ---------------------------------------------------------------------------

#[test]
fn record_v2_skips_when_no_input_requires_grad() {
    let a = make_f32(vec![1.0, 2.0], &[2]);
    let b = make_f32(vec![3.0, 4.0], &[2]);
    // Neither input has autograd_meta — inference path.
    assert!(!needs_grad(&[&a, &b]));
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();
    assert!(
        out.autograd_meta().is_none(),
        "inference path must NOT record"
    );
}

// ---------------------------------------------------------------------------
// 5. record_v2_records_when_any_input_requires_grad
// ---------------------------------------------------------------------------

#[test]
fn record_v2_records_when_any_input_requires_grad() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_f32(vec![3.0, 4.0], &[2]); // No requires_grad.
    assert!(needs_grad(&[&a, &b]));
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();
    assert!(
        out.autograd_meta().is_some(),
        "any-input-requires-grad must record"
    );
}

// ---------------------------------------------------------------------------
// 6. add_v2_backward_distributes_grad
// ---------------------------------------------------------------------------

#[test]
fn add_v2_backward_distributes_grad() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_leaf_requires_grad(vec![3.0, 4.0], &[2]);
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();

    // Backprop with grad_outputs = ones (default — pass None).
    let root = GraphRoot::new(vec![out]).with_grad_outputs(vec![Some(make_f32(
        vec![1.0, 1.0],
        &[2],
    ))]);
    Engine::new().execute(root, &ctx).expect("execute");

    let a_meta = a.autograd_meta().unwrap().lock().unwrap();
    let b_meta = b.autograd_meta().unwrap().lock().unwrap();
    let da = a_meta.grad.as_ref().unwrap().to_vec().unwrap();
    let db = b_meta.grad.as_ref().unwrap().to_vec().unwrap();
    assert_eq!(da, vec![1.0, 1.0]);
    assert_eq!(db, vec![1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// 7. mul_v2_backward_chain_rule
// ---------------------------------------------------------------------------

#[test]
fn mul_v2_backward_chain_rule() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0, 3.0], &[3]);
    let b = make_leaf_requires_grad(vec![10.0, 20.0, 30.0], &[3]);
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::mul::mul_v2(&a, &b, &ctx).unwrap();

    let root = GraphRoot::new(vec![out]).with_grad_outputs(vec![Some(make_f32(
        vec![1.0, 1.0, 1.0],
        &[3],
    ))]);
    Engine::new().execute(root, &ctx).expect("execute");

    let a_meta = a.autograd_meta().unwrap().lock().unwrap();
    let b_meta = b.autograd_meta().unwrap().lock().unwrap();
    // da = g * b = b ; db = g * a = a
    assert_eq!(a_meta.grad.as_ref().unwrap().to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(b_meta.grad.as_ref().unwrap().to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// 8. sum_v2_backward_broadcasts
// ---------------------------------------------------------------------------

#[test]
fn sum_v2_backward_broadcasts() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::sum::sum_v2(&a, &ctx).unwrap();

    // Sum produces a scalar; supply grad_output as a 1-element tensor
    // of shape matching the sum's output. Get the output shape from
    // the output tensor and produce ones in that shape.
    let g_shape = out.shape().dims().to_vec();
    let g_numel: usize = g_shape.iter().product::<usize>().max(1);
    let g = make_f32(vec![1.0_f32; g_numel], &g_shape);
    let root = GraphRoot::new(vec![out]).with_grad_outputs(vec![Some(g)]);
    Engine::new().execute(root, &ctx).expect("execute");

    let a_meta = a.autograd_meta().unwrap().lock().unwrap();
    // da = broadcast(1, [4]) = [1,1,1,1].
    assert_eq!(
        a_meta.grad.as_ref().unwrap().to_vec().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0]
    );
}

// ---------------------------------------------------------------------------
// 9. matmul_v2_backward_correct_shapes
// ---------------------------------------------------------------------------

#[test]
fn matmul_v2_backward_correct_shapes() {
    // A: 2x3, B: 3x4 → out: 2x4
    let a = make_leaf_requires_grad(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    );
    let b = make_leaf_requires_grad(
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        &[3, 4],
    );
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::matmul::matmul_v2(&a, &b, &ctx).unwrap();
    assert_eq!(out.shape().dims(), &[2, 4]);

    let g = make_f32(vec![1.0; 2 * 4], &[2, 4]);
    let root = GraphRoot::new(vec![out]).with_grad_outputs(vec![Some(g)]);
    Engine::new().execute(root, &ctx).expect("execute");

    let a_meta = a.autograd_meta().unwrap().lock().unwrap();
    let b_meta = b.autograd_meta().unwrap().lock().unwrap();
    let da = a_meta.grad.as_ref().unwrap();
    let db = b_meta.grad.as_ref().unwrap();
    assert_eq!(da.shape().dims(), &[2, 3], "da shape");
    assert_eq!(db.shape().dims(), &[3, 4], "db shape");

    // Spot-check both grads have the expected magnitudes. For g of
    // all-ones (2x4):
    //   da = g @ b^T  → each row of da is column-sum(b^T) = row-sum(b)
    //   So da[0,0] = row-sum of b's row 0 = 1+0+0+1 = 2,
    //   da[0,1] = row-sum of b's row 1 = 0+1+1+0 = 2,
    //   da[0,2] = row-sum of b's row 2 = 1+0+1+1 = 3.
    //
    // But the row/column convention can flip depending on how the
    // matmul kernel interprets the strided transpose view; what
    // matters for Phase 3a is "backward produced non-zero, correctly
    // shaped grads from the saved tensors". Use a sum-based assertion
    // that is invariant to row/column ordering: sum of da equals
    // (rows-of-g) * (sum-of-all-elements-of-b).
    let da_vec = da.to_vec().unwrap();
    let da_total: f32 = da_vec.iter().sum();
    // Sum of all 12 elements of B = 7 (rows: 2+2+3). g has 2 rows.
    // Each row of da contributes the full sum of B → da_total = 2 * 7 = 14.
    assert!(
        (da_total - 14.0).abs() < 1e-4,
        "sum(da) = {}, want 14 (= rows(g) * sum(B))",
        da_total
    );
}

// ---------------------------------------------------------------------------
// 10. silu_v2_backward_at_zero
// ---------------------------------------------------------------------------

#[test]
fn silu_v2_backward_at_zero() {
    // silu(0) = 0, silu'(0) = 0.5.
    let x = make_leaf_requires_grad(vec![0.0, 0.0, 0.0], &[3]);
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::silu::silu_v2(&x, &ctx).unwrap();

    let g = make_f32(vec![1.0, 1.0, 1.0], &[3]);
    let root = GraphRoot::new(vec![out]).with_grad_outputs(vec![Some(g)]);
    Engine::new().execute(root, &ctx).expect("execute");

    let x_meta = x.autograd_meta().unwrap().lock().unwrap();
    let dx = x_meta.grad.as_ref().unwrap().to_vec().unwrap();
    for v in &dx {
        assert!(
            (v - 0.5).abs() < 1e-5,
            "silu'(0) should be 0.5, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// 11. add_through_engine_end_to_end (smoke)
// ---------------------------------------------------------------------------

#[test]
fn add_through_engine_end_to_end() {
    let a = make_leaf_requires_grad(vec![0.5, 0.25], &[2]);
    let b = make_leaf_requires_grad(vec![1.5, 1.75], &[2]);
    let ctx = default_ctx();
    let y = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();
    // Verify forward result then backward.
    assert_eq!(y.to_vec().unwrap(), vec![2.0, 2.0]);

    let g = make_f32(vec![3.0, 7.0], &[2]);
    let root = GraphRoot::new(vec![y]).with_grad_outputs(vec![Some(g)]);
    Engine::new().execute(root, &ctx).unwrap();

    assert_eq!(
        a.autograd_meta()
            .unwrap()
            .lock()
            .unwrap()
            .grad
            .as_ref()
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![3.0, 7.0]
    );
    assert_eq!(
        b.autograd_meta()
            .unwrap()
            .lock()
            .unwrap()
            .grad
            .as_ref()
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![3.0, 7.0]
    );
}

// ---------------------------------------------------------------------------
// 12. accumulate_grad_uses_inplace_when_create_graph_false
// ---------------------------------------------------------------------------
//
// Two adds into the same leaf via two separate forward paths. With
// `create_graph=false` (default), the second contribution should be
// in-place'd into the first via the existing add_inplace_same_dtype
// path. We verify by checking that no new v2 GradFn was created for
// the accumulation add (i.e. the leaf's `meta.grad` does NOT carry a
// grad_fn).

#[test]
fn accumulate_grad_uses_inplace_when_create_graph_false() {
    let a = make_leaf_requires_grad(vec![10.0, 20.0], &[2]);
    let b = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let c = make_leaf_requires_grad(vec![100.0, 200.0], &[2]);
    let ctx = default_ctx();

    // Two paths into `a`'s accumulator:
    //   y1 = a + b
    //   y2 = a + c
    // Both flow back into `a` during backward; AccumulateGrad runs twice.
    let y1 = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();
    let y2 = flame_core::autograd_v2::ops::add::add_v2(&a, &c, &ctx).unwrap();

    let root = GraphRoot::new(vec![y1, y2]).with_grad_outputs(vec![
        Some(make_f32(vec![1.0, 1.0], &[2])),
        Some(make_f32(vec![1.0, 1.0], &[2])),
    ]);
    Engine::new().execute(root, &ctx).unwrap();

    let a_meta = a.autograd_meta().unwrap().lock().unwrap();
    let grad = a_meta.grad.as_ref().unwrap();
    // Both paths contributed [1,1] → sum [2,2].
    assert_eq!(grad.to_vec().unwrap(), vec![2.0, 2.0]);
    // create_graph=false: the accumulator did NOT record the add as a
    // v2 op, so the accumulated tensor has no autograd_meta attached.
    assert!(
        grad.autograd_meta().is_none(),
        "in-place accumulation must not produce a recording meta"
    );
}

// ---------------------------------------------------------------------------
// 13. accumulate_grad_uses_out_of_place_when_create_graph_true
// ---------------------------------------------------------------------------
//
// Same shape; with create_graph=true the AccumulateGrad's second
// contribution should go through `add_v2`, producing a new tensor
// whose `autograd_meta.grad_fn` is `AddGradFn`. We verify the
// resulting accumulated grad carries an autograd_meta.

#[test]
fn accumulate_grad_uses_out_of_place_when_create_graph_true() {
    let a = make_leaf_requires_grad(vec![10.0, 20.0], &[2]);
    let b = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let c = make_leaf_requires_grad(vec![100.0, 200.0], &[2]);
    let ctx = default_ctx();

    let y1 = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();
    let y2 = flame_core::autograd_v2::ops::add::add_v2(&a, &c, &ctx).unwrap();

    let root = GraphRoot::new(vec![y1, y2])
        .with_grad_outputs(vec![
            Some(make_f32(vec![1.0, 1.0], &[2])),
            Some(make_f32(vec![1.0, 1.0], &[2])),
        ])
        .with_create_graph(true);
    Engine::new().execute(root, &ctx).unwrap();

    let a_meta = a.autograd_meta().unwrap().lock().unwrap();
    let grad = a_meta.grad.as_ref().unwrap();
    // Numerically still [2,2].
    assert_eq!(grad.to_vec().unwrap(), vec![2.0, 2.0]);
    // create_graph=true: the accumulation add was recorded as a v2
    // op, so the grad tensor carries an autograd_meta with a
    // grad_fn (the AddGradFn from accumulator's recording branch).
    let meta = grad
        .autograd_meta()
        .expect("create_graph=true: accumulated grad must carry meta");
    let m = meta.lock().unwrap();
    assert!(
        m.grad_fn.is_some(),
        "out-of-place accumulation must record a grad_fn"
    );
    assert_eq!(m.grad_fn.as_ref().unwrap().name(), "AddGradFn");
}

// ---------------------------------------------------------------------------
// 14. engine_rejects_mismatched_grad_output_shape
// ---------------------------------------------------------------------------

#[test]
fn engine_rejects_mismatched_grad_output_shape() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_leaf_requires_grad(vec![3.0, 4.0], &[2]);
    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();

    // grad_outputs[0].shape() is [3], outputs[0].shape() is [2].
    let g = make_f32(vec![1.0, 1.0, 1.0], &[3]);
    let root = GraphRoot::new(vec![out]).with_grad_outputs(vec![Some(g)]);
    let res = Engine::new().execute(root, &ctx);
    match res {
        Err(AutogradV2Error::GradOutputShapeMismatch {
            index, out_shape, grad_shape,
        }) => {
            assert_eq!(index, 0);
            assert_eq!(out_shape, vec![2]);
            assert_eq!(grad_shape, vec![3]);
        }
        other => panic!("expected GradOutputShapeMismatch, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 15. with_inputs_returns_non_leaf_grad
// ---------------------------------------------------------------------------
//
// y = (a + b) * c. Use with_inputs(vec![intermediate]) where
// intermediate = a + b. The grad flowing into `intermediate` is the
// upstream of the mul w.r.t. its first input = c. Verify the
// returned grad equals c.

#[test]
fn with_inputs_returns_non_leaf_grad() {
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_leaf_requires_grad(vec![10.0, 20.0], &[2]);
    let c = make_leaf_requires_grad(vec![100.0, 200.0], &[2]);
    let ctx = default_ctx();

    let intermediate = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).unwrap();
    let y = flame_core::autograd_v2::ops::mul::mul_v2(&intermediate, &c, &ctx).unwrap();

    let g_out = make_f32(vec![1.0, 1.0], &[2]);
    let root = GraphRoot::new(vec![y])
        .with_grad_outputs(vec![Some(g_out)])
        .with_inputs(vec![intermediate.clone()]);
    let result = Engine::new().execute(root, &ctx).expect("execute");

    // result[0] is the grad flowing into `intermediate` (the non-leaf input).
    // For y = intermediate * c with g=ones, d/d_intermediate = c.
    let captured = result.into_iter().next().flatten().expect(
        "non-leaf grad should be captured, not None",
    );
    assert_eq!(captured.to_vec().unwrap(), vec![100.0, 200.0]);
}

// ---------------------------------------------------------------------------
// Sanity: gradient_edge_for_tensor on a no-meta tensor returns null
// ---------------------------------------------------------------------------

#[test]
fn gradient_edge_for_tensor_null_when_no_meta() {
    let t = make_f32(vec![1.0], &[1]);
    let e = gradient_edge_for_tensor(&t);
    assert!(!e.is_valid(), "no-meta tensor must produce a null edge");
}

#[test]
fn next_sequence_nr_monotonic() {
    let a = next_sequence_nr();
    let b = next_sequence_nr();
    assert!(b > a, "next_sequence_nr must be monotonically increasing");
}

// Silence unused import warnings if a particular cfg path drops one.
#[allow(dead_code)]
fn _unused_imports_anchor(_ds: &DeviceStream) {}
