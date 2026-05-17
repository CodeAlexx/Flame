//! Phase 3c1 — `CheckpointGradFn` + `checkpoint_v2` tests.
//!
//! Tests:
//!
//! 1. `checkpoint_v2_same_grads_as_no_checkpoint` — wrap an op chain in
//!    `checkpoint_v2`, verify backward produces grads bit-equal to the
//!    non-checkpointed reference.
//! 2. `checkpoint_v2_skips_when_no_input_requires_grad` — inference path:
//!    returns outputs without installing a CheckpointGradFn.
//! 3. `checkpoint_v2_reentrant_nested` — checkpoint inside a checkpoint.
//!    Verify both layers participate in the gradient flow and no engine
//!    state corruption.
//! 4. `checkpoint_v2_multi_output` — a forward closure returning 2 outputs.
//!    Both must drive the inner engine cleanly.

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::sync::Arc;

use flame_core::autograd_v2::{
    checkpoint_v2, new_meta_ref, AutogradMetaV2, CheckpointForwardFn, DispatchCtx, Engine,
    GraphRoot,
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

fn make_leaf_requires_grad(values: Vec<f32>, dims: &[usize]) -> Tensor {
    let mut t = make_f32(values, dims);
    let meta = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    t.set_autograd_meta(Some(meta));
    t
}

// ---------------------------------------------------------------------------
// 1. checkpoint_v2_same_grads_as_no_checkpoint
// ---------------------------------------------------------------------------
//
// Forward closure:  y = (a + b) * a      (intentionally non-trivial chain
//                                          so backward exercises mul + add)
// With grad_outputs=ones, the expected gradients are:
//   d_a = (a + b) + a = 2a + b
//   d_b = a
//
// Run twice: once via checkpoint_v2, once via the plain ops. Compare
// element-wise.

#[test]
fn checkpoint_v2_same_grads_as_no_checkpoint() {
    let ctx = default_ctx();
    let a_vals = vec![1.0_f32, 2.0, 3.0];
    let b_vals = vec![10.0_f32, 20.0, 30.0];

    // --- Path A: no checkpoint ------------------------------------------------
    let a_ref = make_leaf_requires_grad(a_vals.clone(), &[3]);
    let b_ref = make_leaf_requires_grad(b_vals.clone(), &[3]);
    let s_ref = flame_core::autograd_v2::ops::add::add_v2(&a_ref, &b_ref, &ctx).unwrap();
    let y_ref = flame_core::autograd_v2::ops::mul::mul_v2(&s_ref, &a_ref, &ctx).unwrap();
    let g_ref = make_f32(vec![1.0, 1.0, 1.0], &[3]);
    let root_ref = GraphRoot::new(vec![y_ref]).with_grad_outputs(vec![Some(g_ref)]);
    Engine::new().execute(root_ref, &ctx).expect("ref backward");
    let da_ref = a_ref
        .autograd_meta()
        .unwrap()
        .lock()
        .unwrap()
        .grad
        .clone()
        .unwrap()
        .to_vec()
        .unwrap();
    let db_ref = b_ref
        .autograd_meta()
        .unwrap()
        .lock()
        .unwrap()
        .grad
        .clone()
        .unwrap()
        .to_vec()
        .unwrap();

    // --- Path B: checkpoint ---------------------------------------------------
    let a_ckpt = make_leaf_requires_grad(a_vals.clone(), &[3]);
    let b_ckpt = make_leaf_requires_grad(b_vals.clone(), &[3]);

    let forward: Arc<CheckpointForwardFn> = Arc::new(
        |inputs: &[Tensor], ctx: &DispatchCtx| -> flame_core::Result<Vec<Tensor>> {
            let a = &inputs[0];
            let b = &inputs[1];
            let s = flame_core::autograd_v2::ops::add::add_v2(a, b, ctx)?;
            let y = flame_core::autograd_v2::ops::mul::mul_v2(&s, a, ctx)?;
            Ok(vec![y])
        },
    );

    let outs = checkpoint_v2(forward, &[a_ckpt.clone(), b_ckpt.clone()], &ctx)
        .expect("checkpoint_v2 forward");
    assert_eq!(outs.len(), 1);
    let y_ckpt = outs.into_iter().next().unwrap();
    // The output must carry a CheckpointGradFn.
    let out_meta = y_ckpt.autograd_meta().expect("output should carry meta");
    let m = out_meta.lock().unwrap();
    assert_eq!(m.grad_fn.as_ref().unwrap().name(), "CheckpointGradFn");
    drop(m);

    let g_ckpt = make_f32(vec![1.0, 1.0, 1.0], &[3]);
    let root_ckpt = GraphRoot::new(vec![y_ckpt]).with_grad_outputs(vec![Some(g_ckpt)]);
    Engine::new()
        .execute(root_ckpt, &ctx)
        .expect("ckpt backward");
    let da_ckpt = a_ckpt
        .autograd_meta()
        .unwrap()
        .lock()
        .unwrap()
        .grad
        .clone()
        .unwrap()
        .to_vec()
        .unwrap();
    let db_ckpt = b_ckpt
        .autograd_meta()
        .unwrap()
        .lock()
        .unwrap()
        .grad
        .clone()
        .unwrap()
        .to_vec()
        .unwrap();

    // F32 forward should give bit-equal results between the two paths.
    for (i, (g_ref, g_ckpt)) in da_ref.iter().zip(da_ckpt.iter()).enumerate() {
        assert!(
            (g_ref - g_ckpt).abs() < 1e-5,
            "d_a[{i}]: ref={g_ref}, ckpt={g_ckpt}"
        );
    }
    for (i, (g_ref, g_ckpt)) in db_ref.iter().zip(db_ckpt.iter()).enumerate() {
        assert!(
            (g_ref - g_ckpt).abs() < 1e-5,
            "d_b[{i}]: ref={g_ref}, ckpt={g_ckpt}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. checkpoint_v2_skips_when_no_input_requires_grad
// ---------------------------------------------------------------------------
//
// Inputs without requires_grad → checkpoint_v2 returns the closure's
// outputs as-is (no CheckpointGradFn wired). This is the inference path.

#[test]
fn checkpoint_v2_skips_when_no_input_requires_grad() {
    let ctx = default_ctx();
    let a = make_f32(vec![1.0, 2.0], &[2]); // no requires_grad
    let b = make_f32(vec![3.0, 4.0], &[2]);

    let forward: Arc<CheckpointForwardFn> = Arc::new(
        |inputs: &[Tensor], ctx: &DispatchCtx| -> flame_core::Result<Vec<Tensor>> {
            let y = flame_core::autograd_v2::ops::add::add_v2(&inputs[0], &inputs[1], ctx)?;
            Ok(vec![y])
        },
    );

    let outs = checkpoint_v2(forward, &[a, b], &ctx).expect("checkpoint_v2");
    let y = outs.into_iter().next().unwrap();
    // No meta installed; inference path.
    assert!(
        y.autograd_meta().is_none(),
        "no input requires_grad → no CheckpointGradFn meta installed"
    );
    assert_eq!(y.to_vec().unwrap(), vec![4.0, 6.0]);
}

// ---------------------------------------------------------------------------
// 3. checkpoint_v2_reentrant_nested
// ---------------------------------------------------------------------------
//
// Outer closure wraps an inner closure that is ALSO a checkpoint_v2 call.
// The outer's backward re-runs forward; that re-run calls the inner
// checkpoint_v2; the inner backward inside the outer's apply() drives ANOTHER
// nested Engine::execute. Total reentrancy depth: 2 nested engines.

#[test]
fn checkpoint_v2_reentrant_nested() {
    let ctx = default_ctx();
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_leaf_requires_grad(vec![3.0, 4.0], &[2]);

    // Inner closure: s = a + b
    let inner: Arc<CheckpointForwardFn> = Arc::new(
        |inputs: &[Tensor], ctx: &DispatchCtx| -> flame_core::Result<Vec<Tensor>> {
            let s = flame_core::autograd_v2::ops::add::add_v2(&inputs[0], &inputs[1], ctx)?;
            Ok(vec![s])
        },
    );

    // Outer closure: y = checkpoint(inner)(a, b) * a
    let inner_clone = inner.clone();
    let outer: Arc<CheckpointForwardFn> = Arc::new(
        move |inputs: &[Tensor], ctx: &DispatchCtx| -> flame_core::Result<Vec<Tensor>> {
            let a = &inputs[0];
            let b = &inputs[1];
            let inner_outs = checkpoint_v2(inner_clone.clone(), &[a.clone(), b.clone()], ctx)?;
            let s = inner_outs.into_iter().next().unwrap();
            let y = flame_core::autograd_v2::ops::mul::mul_v2(&s, a, ctx)?;
            Ok(vec![y])
        },
    );

    let outs = checkpoint_v2(outer, &[a.clone(), b.clone()], &ctx).expect("outer checkpoint_v2");
    let y = outs.into_iter().next().unwrap();
    assert_eq!(y.shape().dims(), &[2]);
    // y = (a+b) * a = [4, 12]
    assert_eq!(y.to_vec().unwrap(), vec![4.0, 12.0]);

    let g = make_f32(vec![1.0, 1.0], &[2]);
    let root = GraphRoot::new(vec![y]).with_grad_outputs(vec![Some(g)]);
    Engine::new().execute(root, &ctx).expect("nested backward");

    // Expected: same as no-checkpoint path.
    //   y = (a+b)*a, d_y=1 → d_a = (a+b) + a = 2a+b, d_b = a
    //   a=[1,2], b=[3,4] → d_a = [5, 8], d_b = [1, 2]
    let da = a
        .autograd_meta()
        .unwrap()
        .lock()
        .unwrap()
        .grad
        .clone()
        .unwrap()
        .to_vec()
        .unwrap();
    let db = b
        .autograd_meta()
        .unwrap()
        .lock()
        .unwrap()
        .grad
        .clone()
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(da, vec![5.0, 8.0], "nested d_a");
    assert_eq!(db, vec![1.0, 2.0], "nested d_b");
}

// ---------------------------------------------------------------------------
// 4. checkpoint_v2_multi_output
// ---------------------------------------------------------------------------
//
// A closure that returns two outputs. Both must be wired with a
// CheckpointGradFn and both must drive the inner engine cleanly.

#[test]
fn checkpoint_v2_multi_output() {
    let ctx = default_ctx();
    let a = make_leaf_requires_grad(vec![1.0, 2.0], &[2]);
    let b = make_leaf_requires_grad(vec![10.0, 20.0], &[2]);

    let forward: Arc<CheckpointForwardFn> = Arc::new(
        |inputs: &[Tensor], ctx: &DispatchCtx| -> flame_core::Result<Vec<Tensor>> {
            let s = flame_core::autograd_v2::ops::add::add_v2(&inputs[0], &inputs[1], ctx)?;
            let p = flame_core::autograd_v2::ops::mul::mul_v2(&inputs[0], &inputs[1], ctx)?;
            Ok(vec![s, p])
        },
    );

    let outs =
        checkpoint_v2(forward, &[a.clone(), b.clone()], &ctx).expect("multi-output checkpoint");
    assert_eq!(outs.len(), 2);
    let s = outs[0].clone();
    let p = outs[1].clone();

    // Both outputs must carry CheckpointGradFn — the SAME node (since
    // record_v2 clones the Arc<dyn GradFn> across all outputs of one
    // GradFn).
    let s_meta = s.autograd_meta().unwrap();
    let p_meta = p.autograd_meta().unwrap();
    let s_node_id = s_meta.lock().unwrap().grad_fn.as_ref().unwrap().node_id();
    let p_node_id = p_meta.lock().unwrap().grad_fn.as_ref().unwrap().node_id();
    assert_eq!(s_node_id, p_node_id, "multi-output: same GradFn node");

    let gs = make_f32(vec![1.0, 1.0], &[2]);
    let gp = make_f32(vec![1.0, 1.0], &[2]);
    let root = GraphRoot::new(vec![s, p]).with_grad_outputs(vec![Some(gs), Some(gp)]);
    Engine::new()
        .execute(root, &ctx)
        .expect("multi-output backward");

    // Expected (with gs=gp=1):
    //   s = a+b → d_s/d_a=1, d_s/d_b=1
    //   p = a*b → d_p/d_a=b, d_p/d_b=a
    //   total: d_a = 1 + b = [11, 21], d_b = 1 + a = [2, 3]
    let da = a
        .autograd_meta()
        .unwrap()
        .lock()
        .unwrap()
        .grad
        .clone()
        .unwrap()
        .to_vec()
        .unwrap();
    let db = b
        .autograd_meta()
        .unwrap()
        .lock()
        .unwrap()
        .grad
        .clone()
        .unwrap()
        .to_vec()
        .unwrap();
    for (i, (g, w)) in da.iter().zip(&[11.0_f32, 21.0]).enumerate() {
        assert!((g - w).abs() < 1e-5, "d_a[{i}]: got {g}, want {w}");
    }
    for (i, (g, w)) in db.iter().zip(&[2.0_f32, 3.0]).enumerate() {
        assert!((g - w).abs() < 1e-5, "d_b[{i}]: got {g}, want {w}");
    }
}
