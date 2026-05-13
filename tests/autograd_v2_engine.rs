//! Autograd v2 Phase 2 — engine driver tests.
//!
//! These 9 tests cover the engine's contracts:
//!
//! 1. `single_leaf_sum` — one synthetic `IdentityGradFn` feeds an
//!    `AccumulateGrad`; backward must populate `meta.grad`.
//! 2. `two_branches_one_leaf` — two `IdentityGradFn` nodes share one
//!    leaf accumulator; both contributions must accumulate.
//! 3. `diamond_accumulation` — a → {b, c} → d; verify the leaf grad
//!    captures both diamond paths.
//! 4. `undefined_grad_slot` — a `GradFn` returns `None` for one output
//!    slot; the engine must NOT panic and must decrement the child's
//!    dep_count.
//! 5. `released_saved_tensor_error` — a `GradFn` whose `apply` reads a
//!    `SavedTensor` that was `reset()` — error propagates up
//!    `Engine::execute` cleanly.
//! 6. `version_mismatch_error` — same shape but the saved version
//!    was bumped; error propagates.
//! 7. `create_graph_2nd_order_toy` — sanity that
//!    `create_graph=true` doesn't panic and runs to completion.
//! 8. `reentrant_nested_execute` — a `GradFn::apply` calls
//!    `Engine::new().execute(...)` recursively. Outer engine resumes
//!    cleanly.
//! 9. `hooks_fire_in_order` — register pre/post/tensor hooks on a
//!    `GradFn` and verify each fires the expected number of times.

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use flame_core::autograd_v2::{
    gradient_edge, new_meta_ref, AutogradMetaRef, AutogradMetaV2,
    AutogradV2Error, DispatchCtx, Edge, Engine, GradFn, GraphRoot, Hooks, NodeId, SavedTensor,
};
use flame_core::{global_cuda_device, Device, Shape, Tensor};

/// Phase 3a test helper: link a `Tensor` to a `grad_fn` at `output_nr`
/// by installing a fresh `AutogradMetaRef`. Replaces Phase 2's
/// `_v2_set_grad_fn` test-only backdoor; the production op-recording
/// path is `flame_core::autograd_v2::record_v2`, but the engine tests
/// hand-roll synthetic `GradFn` impls that aren't shaped as
/// `record_v2(grad_fn, vec![output], ctx)` (they build pre-linked
/// graphs), so we install the meta directly here.
fn link_tensor_to_grad_fn(t: &mut Tensor, gf: std::sync::Arc<dyn GradFn>, output_nr: u32) {
    let meta = new_meta_ref(AutogradMetaV2::non_leaf(gf, output_nr));
    t.set_autograd_meta(Some(meta));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_f32(values: Vec<f32>, dims: &[usize]) -> Tensor {
    let device = global_cuda_device().clone();
    Tensor::from_vec(values, Shape::from_dims(dims), device).expect("from_vec")
}

fn default_ctx() -> DispatchCtx {
    let dev = Device::from_arc(global_cuda_device().clone());
    DispatchCtx::default_for(dev)
}

/// Sequence-number counter shared across one test scenario. The engine
/// orders the ready queue partly by `sequence_nr` so tests that build
/// multi-node graphs should pass increasing values.
fn next_seq() -> u64 {
    static C: AtomicUsize = AtomicUsize::new(0);
    C.fetch_add(1, Ordering::Relaxed) as u64
}

// ---------------------------------------------------------------------------
// Test-only GradFn impls.
// ---------------------------------------------------------------------------

/// Identity backward: passes its single input grad straight through to
/// its single next_edge.
#[derive(Debug)]
struct IdentityGradFn {
    next_edges: Vec<Edge>,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
    name: &'static str,
}

impl IdentityGradFn {
    fn new(next_edges: Vec<Edge>, topological_nr: u64, name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            next_edges,
            sequence_nr: next_seq(),
            topological_nr,
            node_id: NodeId::new(),
            name,
        })
    }
}

impl GradFn for IdentityGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // num_inputs = 1; arity check: apply returns one grad per
        // next_edge (= one grad per *input* of the forward op). Identity
        // forwards its single incoming output-grad to its single
        // upstream input-grad slot.
        let g = grad_outputs.into_iter().next().flatten();
        Ok(vec![g])
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
    fn sequence_nr(&self) -> u64 {
        self.sequence_nr
    }
    fn topological_nr(&self) -> u64 {
        self.topological_nr
    }
    fn node_id(&self) -> NodeId {
        self.node_id
    }
    fn name(&self) -> &'static str {
        self.name
    }
}

/// Adds two input-grads together to produce one downstream grad. Used
/// to build diamond-style accumulation patterns.
#[derive(Debug)]
struct AddGradFn {
    next_edges: Vec<Edge>,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
}

impl AddGradFn {
    fn new(next_edges: Vec<Edge>, topological_nr: u64) -> Arc<Self> {
        Arc::new(Self {
            next_edges,
            sequence_nr: next_seq(),
            topological_nr,
            node_id: NodeId::new(),
        })
    }
}

impl GradFn for AddGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // Forward had 1 output, so grad_outputs.len()==1. The op
        // consumed 2 inputs, so num_inputs==2 and we return 2 grads.
        // For test purposes, route the same grad to both inputs.
        let g = grad_outputs.into_iter().next().flatten();
        Ok(vec![g.clone(), g])
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
    fn sequence_nr(&self) -> u64 {
        self.sequence_nr
    }
    fn topological_nr(&self) -> u64 {
        self.topological_nr
    }
    fn node_id(&self) -> NodeId {
        self.node_id
    }
    fn name(&self) -> &'static str {
        "AddGradFn"
    }
}

/// Returns `None` for output slot 0 unconditionally. Used by
/// `undefined_grad_slot`.
#[derive(Debug)]
struct NoneOutputGradFn {
    next_edges: Vec<Edge>,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
}

impl NoneOutputGradFn {
    fn new(next_edges: Vec<Edge>, topological_nr: u64) -> Arc<Self> {
        Arc::new(Self {
            next_edges,
            sequence_nr: next_seq(),
            topological_nr,
            node_id: NodeId::new(),
        })
    }
}

impl GradFn for NoneOutputGradFn {
    fn apply(
        &self,
        _grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        Ok(vec![None])
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
    fn sequence_nr(&self) -> u64 {
        self.sequence_nr
    }
    fn topological_nr(&self) -> u64 {
        self.topological_nr
    }
    fn node_id(&self) -> NodeId {
        self.node_id
    }
    fn name(&self) -> &'static str {
        "NoneOutputGradFn"
    }
}

/// A GradFn whose `apply` reads a `SavedTensor`. Used for the
/// `released_saved_tensor_error` and `version_mismatch_error` tests.
#[derive(Debug)]
struct SavedTensorReadingGradFn {
    saved: SavedTensor,
    next_edges: Vec<Edge>,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
}

impl SavedTensorReadingGradFn {
    fn new(saved: SavedTensor, next_edges: Vec<Edge>, topological_nr: u64) -> Arc<Self> {
        Arc::new(Self {
            saved,
            next_edges,
            sequence_nr: next_seq(),
            topological_nr,
            node_id: NodeId::new(),
        })
    }
}

impl GradFn for SavedTensorReadingGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // Read the SavedTensor — error variants bubble up through `?`.
        let _x = self.saved.unpack()?;
        let g = grad_outputs.into_iter().next().flatten();
        Ok(vec![g])
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
    fn sequence_nr(&self) -> u64 {
        self.sequence_nr
    }
    fn topological_nr(&self) -> u64 {
        self.topological_nr
    }
    fn node_id(&self) -> NodeId {
        self.node_id
    }
    fn name(&self) -> &'static str {
        "SavedTensorReadingGradFn"
    }
}

/// A GradFn that calls `Engine::new().execute(...)` from its own
/// `apply()` — exercises the reentrant-nested-execute pattern.
#[derive(Debug)]
struct ReentrantGradFn {
    inner_leaf: AutogradMetaRef,
    next_edges: Vec<Edge>,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
    inner_call_count: Arc<AtomicUsize>,
}

impl ReentrantGradFn {
    fn new(
        inner_leaf: AutogradMetaRef,
        next_edges: Vec<Edge>,
        topological_nr: u64,
        inner_call_count: Arc<AtomicUsize>,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner_leaf,
            next_edges,
            sequence_nr: next_seq(),
            topological_nr,
            node_id: NodeId::new(),
            inner_call_count,
        })
    }
}

impl GradFn for ReentrantGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // Build a tiny nested graph: identity → AccumulateGrad on
        // `inner_leaf`. Drive it through a fresh Engine.
        let inner_edge = gradient_edge(&self.inner_leaf, next_seq());
        let inner_identity = IdentityGradFn::new(vec![inner_edge], 1, "InnerIdentity");
        let mut inner_out = make_f32(vec![100.0, 200.0], &[2]);
        link_tensor_to_grad_fn(&mut inner_out, inner_identity.clone() as Arc<dyn GradFn>, 0);

        let inner_root = GraphRoot::new(vec![inner_out]).with_grad_outputs(vec![Some(
            make_f32(vec![1.0, 1.0], &[2]),
        )]);
        // Recursion: outer Engine + inner Engine are independent.
        Engine::new().execute(inner_root, ctx)?;
        self.inner_call_count.fetch_add(1, Ordering::Relaxed);

        // Forward our own grad through. Test scenario uses this as
        // an intermediate node with one downstream edge.
        let g = grad_outputs.into_iter().next().flatten();
        Ok(vec![g])
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
    fn sequence_nr(&self) -> u64 {
        self.sequence_nr
    }
    fn topological_nr(&self) -> u64 {
        self.topological_nr
    }
    fn node_id(&self) -> NodeId {
        self.node_id
    }
    fn name(&self) -> &'static str {
        "ReentrantGradFn"
    }
}

/// Identity-like GradFn that carries a `Hooks` bundle so we can
/// exercise hook dispatch.
struct HookedGradFn {
    next_edges: Vec<Edge>,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
    hooks: Hooks,
}

impl std::fmt::Debug for HookedGradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookedGradFn")
            .field("node_id", &self.node_id)
            .field("hook_counts", &self.hooks)
            .finish()
    }
}

impl HookedGradFn {
    fn new(next_edges: Vec<Edge>, hooks: Hooks, topological_nr: u64) -> Arc<Self> {
        Arc::new(Self {
            next_edges,
            sequence_nr: next_seq(),
            topological_nr,
            node_id: NodeId::new(),
            hooks,
        })
    }
}

impl GradFn for HookedGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = grad_outputs.into_iter().next().flatten();
        Ok(vec![g])
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
    fn sequence_nr(&self) -> u64 {
        self.sequence_nr
    }
    fn topological_nr(&self) -> u64 {
        self.topological_nr
    }
    fn node_id(&self) -> NodeId {
        self.node_id
    }
    fn name(&self) -> &'static str {
        "HookedGradFn"
    }
    fn hooks(&self) -> &Hooks {
        &self.hooks
    }
}

// ---------------------------------------------------------------------------
// 1. single_leaf_sum
// ---------------------------------------------------------------------------

#[test]
fn single_leaf_sum() {
    // Leaf with requires_grad=true.
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());

    // Build the graph: out = Identity(leaf). The output tensor is just
    // a value; the graph topology lives in the side-table.
    let leaf_edge = gradient_edge(&leaf_meta, next_seq());
    let identity = IdentityGradFn::new(vec![leaf_edge], 1, "Identity");
    let mut out = make_f32(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    link_tensor_to_grad_fn(&mut out, identity.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out])
        .with_grad_outputs(vec![Some(make_f32(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]))]);
    Engine::new().execute(root, &ctx).expect("execute");

    // Leaf grad must be the upstream grad (identity forwarded it).
    let m = leaf_meta.lock().unwrap();
    let g = m.grad.as_ref().expect("leaf grad populated");
    assert_eq!(g.to_vec().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// 2. two_branches_one_leaf
// ---------------------------------------------------------------------------

#[test]
fn two_branches_one_leaf() {
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());

    // Branch A and Branch B both feed into the same leaf accumulator.
    // gradient_edge() must materialize-or-cache so both edges point at
    // the SAME accumulator instance.
    let edge_a = gradient_edge(&leaf_meta, next_seq());
    let edge_b = gradient_edge(&leaf_meta, next_seq());

    // Both edges should resolve to the same NodeId.
    let id_a = edge_a.function.as_ref().unwrap().node_id();
    let id_b = edge_b.function.as_ref().unwrap().node_id();
    assert_eq!(id_a, id_b, "gradient_edge must cache the accumulator");

    let identity_a = IdentityGradFn::new(vec![edge_a], 1, "IdA");
    let identity_b = IdentityGradFn::new(vec![edge_b], 1, "IdB");

    let mut out_a = make_f32(vec![1.0, 2.0], &[2]);
    let mut out_b = make_f32(vec![3.0, 4.0], &[2]);
    link_tensor_to_grad_fn(&mut out_a, identity_a.clone() as Arc<dyn GradFn>, 0);
    link_tensor_to_grad_fn(&mut out_b, identity_b.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out_a, out_b]).with_grad_outputs(vec![
        Some(make_f32(vec![10.0, 20.0], &[2])),
        Some(make_f32(vec![100.0, 200.0], &[2])),
    ]);
    Engine::new().execute(root, &ctx).expect("execute");

    let m = leaf_meta.lock().unwrap();
    let g = m.grad.as_ref().expect("leaf grad populated");
    // Two branches → accumulator must have summed.
    assert_eq!(g.to_vec().unwrap(), vec![110.0, 220.0]);
}

// ---------------------------------------------------------------------------
// 3. diamond_accumulation
// ---------------------------------------------------------------------------
//
// Graph: leaf -> b, leaf -> c, b + c -> out
// Backward: out.grad → AddGradFn → routes the same grad twice (once to
// b's input slot, once to c's input slot). b and c are both Identity,
// each forwards into the leaf accumulator. Final leaf grad = 2 * out_grad.

#[test]
fn diamond_accumulation() {
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());

    let leaf_edge_b = gradient_edge(&leaf_meta, next_seq());
    let leaf_edge_c = gradient_edge(&leaf_meta, next_seq());

    // Sanity: same accumulator on both sides of the diamond.
    let id_b = leaf_edge_b.function.as_ref().unwrap().node_id();
    let id_c = leaf_edge_c.function.as_ref().unwrap().node_id();
    assert_eq!(id_b, id_c);

    let identity_b = IdentityGradFn::new(vec![leaf_edge_b], 1, "DiamondB");
    let identity_c = IdentityGradFn::new(vec![leaf_edge_c], 1, "DiamondC");

    let b_edge = Edge::new(identity_b.clone() as Arc<dyn GradFn>, 0);
    let c_edge = Edge::new(identity_c.clone() as Arc<dyn GradFn>, 0);

    // AddGradFn has 2 next_edges (one to identity_b, one to identity_c);
    // its `apply` returns 2 grads — the same grad routed twice. Its
    // own num_inputs is 1 (one output of the diamond is summed up).
    let add = AddGradFn::new(vec![b_edge, c_edge], 2);

    let mut out = make_f32(vec![1.0, 2.0], &[2]);
    link_tensor_to_grad_fn(&mut out, add.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out])
        .with_grad_outputs(vec![Some(make_f32(vec![5.0, 7.0], &[2]))]);
    Engine::new().execute(root, &ctx).expect("execute");

    let m = leaf_meta.lock().unwrap();
    let g = m.grad.as_ref().expect("leaf grad populated");
    // Diamond accumulation: each of the two paths contributed `[5,7]`,
    // accumulator summed → [10, 14].
    assert_eq!(g.to_vec().unwrap(), vec![10.0, 14.0]);
}

// ---------------------------------------------------------------------------
// 4. undefined_grad_slot
// ---------------------------------------------------------------------------

#[test]
fn undefined_grad_slot() {
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let leaf_edge = gradient_edge(&leaf_meta, next_seq());

    // NoneOutputGradFn returns None for its single output grad. The
    // engine must not panic; the leaf accumulator should not be called
    // with a Some grad, so meta.grad stays None.
    let none_op = NoneOutputGradFn::new(vec![leaf_edge], 1);

    let mut out = make_f32(vec![0.0], &[1]);
    link_tensor_to_grad_fn(&mut out, none_op.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out])
        .with_grad_outputs(vec![Some(make_f32(vec![1.0], &[1]))]);
    Engine::new().execute(root, &ctx).expect("execute must not panic");

    let m = leaf_meta.lock().unwrap();
    assert!(m.grad.is_none(), "None grad should not populate leaf");
}

// ---------------------------------------------------------------------------
// 5. released_saved_tensor_error
// ---------------------------------------------------------------------------

#[test]
fn released_saved_tensor_error() {
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let leaf_edge = gradient_edge(&leaf_meta, next_seq());

    let x = make_f32(vec![1.0, 2.0], &[2]);
    let saved = SavedTensor::save_named(&x, "TestReleased");
    saved.reset(); // pre-release

    let op = SavedTensorReadingGradFn::new(saved, vec![leaf_edge], 1);

    let mut out = make_f32(vec![0.0, 0.0], &[2]);
    link_tensor_to_grad_fn(&mut out, op.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out])
        .with_grad_outputs(vec![Some(make_f32(vec![1.0, 1.0], &[2]))]);

    let res = Engine::new().execute(root, &ctx);
    match res {
        Err(AutogradV2Error::SavedTensorReleased) => {}
        other => panic!("expected SavedTensorReleased, got {other:?}"),
    }
    // Leaf grad must not have been written.
    let m = leaf_meta.lock().unwrap();
    assert!(m.grad.is_none());
}

// ---------------------------------------------------------------------------
// 6. version_mismatch_error
// ---------------------------------------------------------------------------

#[test]
fn version_mismatch_error() {
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let leaf_edge = gradient_edge(&leaf_meta, next_seq());

    let x = make_f32(vec![1.0, 2.0], &[2]);
    let saved = SavedTensor::save_named(&x, "TestVersion");
    saved._test_bump_saved_version();

    let op = SavedTensorReadingGradFn::new(saved, vec![leaf_edge], 1);

    let mut out = make_f32(vec![0.0, 0.0], &[2]);
    link_tensor_to_grad_fn(&mut out, op.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out])
        .with_grad_outputs(vec![Some(make_f32(vec![1.0, 1.0], &[2]))]);

    let res = Engine::new().execute(root, &ctx);
    match res {
        Err(AutogradV2Error::VersionMismatch { op, .. }) => {
            assert_eq!(op, "TestVersion");
        }
        other => panic!("expected VersionMismatch, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 7. create_graph_2nd_order_toy
// ---------------------------------------------------------------------------
//
// Phase 2 only ships the surface. The toy test verifies:
//   - GraphRoot::with_create_graph(true) is accepted.
//   - InputBuffer was constructed with create_graph=true (verified
//     indirectly via no in-place accumulation, which is harder to assert
//     here — we settle for "engine ran to completion with the flag on").
//   - A test-only GradFn that returns a recordable result completes.
//
// Full numerical second-order parity vs PyTorch lives in Phase 5.

#[test]
fn create_graph_2nd_order_toy() {
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let leaf_edge = gradient_edge(&leaf_meta, next_seq());

    let identity = IdentityGradFn::new(vec![leaf_edge], 1, "Identity2nd");
    let mut out = make_f32(vec![1.0, 2.0], &[2]);
    link_tensor_to_grad_fn(&mut out, identity.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out])
        .with_grad_outputs(vec![Some(make_f32(vec![1.0, 1.0], &[2]))])
        .with_create_graph(true);

    Engine::new()
        .execute(root, &ctx)
        .expect("create_graph=true should still run to completion");

    let m = leaf_meta.lock().unwrap();
    let g = m.grad.as_ref().expect("leaf grad populated");
    assert_eq!(g.to_vec().unwrap(), vec![1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// 8. reentrant_nested_execute
// ---------------------------------------------------------------------------

#[test]
fn reentrant_nested_execute() {
    // Outer leaf and inner leaf are separate, so we can verify both
    // engines wrote to the correct meta.
    let outer_leaf: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let inner_leaf: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());

    let outer_edge = gradient_edge(&outer_leaf, next_seq());
    let inner_call_count = Arc::new(AtomicUsize::new(0));

    // Reentrant node has the outer leaf as its downstream edge AND
    // captures the inner leaf for its own nested-engine drive.
    let reentrant = ReentrantGradFn::new(
        inner_leaf.clone(),
        vec![outer_edge],
        1,
        inner_call_count.clone(),
    );

    let mut out = make_f32(vec![1.0, 2.0], &[2]);
    link_tensor_to_grad_fn(&mut out, reentrant.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out])
        .with_grad_outputs(vec![Some(make_f32(vec![7.0, 9.0], &[2]))]);

    Engine::new()
        .execute(root, &ctx)
        .expect("outer engine must resume cleanly after nested call");

    // Inner engine was called once.
    assert_eq!(inner_call_count.load(Ordering::Relaxed), 1);

    // Inner leaf grad was sunk by the nested engine.
    {
        let m = inner_leaf.lock().unwrap();
        let g = m.grad.as_ref().expect("inner leaf grad populated");
        assert_eq!(g.to_vec().unwrap(), vec![1.0, 1.0]);
    }
    // Outer leaf grad was sunk by the outer engine after the nested
    // call returned.
    {
        let m = outer_leaf.lock().unwrap();
        let g = m.grad.as_ref().expect("outer leaf grad populated");
        assert_eq!(g.to_vec().unwrap(), vec![7.0, 9.0]);
    }
}

// ---------------------------------------------------------------------------
// 9. hooks_fire_in_order
// ---------------------------------------------------------------------------

#[test]
fn hooks_fire_in_order() {
    let observed: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

    // Pre-backward hook records "pre".
    let pre = {
        let o = observed.clone();
        Arc::new(move |_g: &[Option<Tensor>]| {
            o.lock().unwrap().push("pre");
        }) as flame_core::autograd_v2::PreBackwardHook
    };
    // Post-backward hook records "post".
    let post = {
        let o = observed.clone();
        Arc::new(move |_g: &[Option<Tensor>]| {
            o.lock().unwrap().push("post");
        }) as flame_core::autograd_v2::PostBackwardHook
    };
    // Tensor hook records "tensor" and returns the grad unchanged.
    let tens = {
        let o = observed.clone();
        Arc::new(move |t: &Tensor| -> Option<Tensor> {
            o.lock().unwrap().push("tensor");
            Some(t.clone())
        }) as flame_core::autograd_v2::TensorHook
    };

    let mut hooks = Hooks::new();
    hooks.pre_backward.push(pre);
    hooks.post_backward.push(post);
    hooks.tensor_hooks.push(tens);

    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let leaf_edge = gradient_edge(&leaf_meta, next_seq());

    let hooked = HookedGradFn::new(vec![leaf_edge], hooks, 1);

    let mut out = make_f32(vec![1.0, 2.0], &[2]);
    link_tensor_to_grad_fn(&mut out, hooked.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    let root = GraphRoot::new(vec![out])
        .with_grad_outputs(vec![Some(make_f32(vec![1.0, 1.0], &[2]))]);

    Engine::new().execute(root, &ctx).expect("execute");

    let log = observed.lock().unwrap().clone();
    // Order: tensor (rewrite) → pre (observe) → apply runs (not logged
    // by our hooks) → post (observe).
    assert_eq!(log, vec!["tensor", "pre", "post"], "got {log:?}");
}

// ---------------------------------------------------------------------------
// Bonus: AccumulateGrad placeholder also runs cleanly via the engine
// when grad is None (the leaf-no-contribution path).
// ---------------------------------------------------------------------------

#[test]
fn engine_finishes_when_leaf_never_receives_grad() {
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());

    // Scope the strong Arcs holding the recording op + accumulator edge
    // so the only remaining handle to the AccumulateGrad after the
    // block is the Weak in `leaf_meta.grad_accumulator`.
    {
        let leaf_edge = gradient_edge(&leaf_meta, next_seq());

        // The op returns None instead of a real grad. The accumulator
        // should be enqueued (dep_count was incremented by next_edge
        // walk and decremented when None routes through), but its
        // apply sees an empty buffer and does nothing.
        let none_op = NoneOutputGradFn::new(vec![leaf_edge], 1);

        let mut out = make_f32(vec![0.0], &[1]);
        link_tensor_to_grad_fn(&mut out, none_op.clone() as Arc<dyn GradFn>, 0);

        let ctx = default_ctx();
        let root = GraphRoot::new(vec![out])
            .with_grad_outputs(vec![Some(make_f32(vec![1.0], &[1]))]);

        Engine::new().execute(root, &ctx).expect("execute");

        let m = leaf_meta.lock().unwrap();
        assert!(m.grad.is_none());
    }

    // Phase 3a: the side-table is gone — `none_op`, `out`, and `root`
    // were all dropped at end of the inner block (root was consumed by
    // execute; the test-local `none_op` Arc went out of scope). The
    // AccumulateGrad's only remaining handle is the Weak in
    // `leaf_meta.grad_accumulator`, which fails to upgrade.
    let m = leaf_meta.lock().unwrap();
    assert!(
        m.grad_accumulator.upgrade().is_none(),
        "accumulator should drop after all strong Arcs are released"
    );
}

// ---------------------------------------------------------------------------
// Regression: output_node_is_descendant_of_another_output_no_double_fire
// ---------------------------------------------------------------------------
//
// BUG-FIX 2026-05-13 (bug-fixer audit on ee69d4a): when `outputs =
// [A, B]` and A's next_edges include B, the original seed logic pushed
// B into the ready queue unconditionally (dep_count was 1 from BFS but
// the seed loop ignored dep_count). When A ran and decremented B's
// dep_count 1→0, B was enqueued a SECOND time. The second pop saw an
// empty buffer (already drained) and fired `apply()` again — a
// correctness bug for ops with side effects (counters, hooks, saved
// tensor accesses).
//
// This regression test installs a counting `apply` on the shared-output
// node and asserts it fires exactly once.

use std::sync::atomic::{AtomicUsize as AtomicU, Ordering as Ord_};

#[derive(Debug)]
struct CountingIdentityGradFn {
    next_edges: Vec<Edge>,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
    apply_count: Arc<AtomicU>,
}

impl CountingIdentityGradFn {
    fn new(
        next_edges: Vec<Edge>,
        topological_nr: u64,
        apply_count: Arc<AtomicU>,
    ) -> Arc<Self> {
        Arc::new(Self {
            next_edges,
            sequence_nr: next_seq(),
            topological_nr,
            node_id: NodeId::new(),
            apply_count,
        })
    }
}

impl GradFn for CountingIdentityGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        self.apply_count.fetch_add(1, Ord_::Relaxed);
        let g = grad_outputs.into_iter().next().flatten();
        Ok(vec![g])
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
    fn sequence_nr(&self) -> u64 {
        self.sequence_nr
    }
    fn topological_nr(&self) -> u64 {
        self.topological_nr
    }
    fn node_id(&self) -> NodeId {
        self.node_id
    }
    fn name(&self) -> &'static str {
        "CountingIdentity"
    }
}

#[test]
fn output_node_is_descendant_of_another_output_no_double_fire() {
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let leaf_edge = gradient_edge(&leaf_meta, next_seq());

    // B is the intermediate: identity feeding the leaf.
    let b_apply_count = Arc::new(AtomicU::new(0));
    let b_node = CountingIdentityGradFn::new(vec![leaf_edge], 1, b_apply_count.clone());

    // A is the "above B" output: identity feeding B.
    let a_edge = Edge::new(b_node.clone() as Arc<dyn GradFn>, 0);
    let a_node = IdentityGradFn::new(vec![a_edge], 2, "A_above_B");

    // Both A and B are returned as outputs of the forward.
    let mut out_a = make_f32(vec![1.0, 2.0], &[2]);
    let mut out_b = make_f32(vec![10.0, 20.0], &[2]);
    link_tensor_to_grad_fn(&mut out_a, a_node.clone() as Arc<dyn GradFn>, 0);
    link_tensor_to_grad_fn(&mut out_b, b_node.clone() as Arc<dyn GradFn>, 0);

    let ctx = default_ctx();
    // grad seeds: A gets [3,5]; B gets [7,11]. Forward identities pass
    // them through. Total accumulation into leaf:
    //   from A's path: [3,5] → B → leaf = [3,5]
    //   from B's seed: [7,11] → leaf
    //   sum: [10, 16].
    let root = GraphRoot::new(vec![out_a, out_b]).with_grad_outputs(vec![
        Some(make_f32(vec![3.0, 5.0], &[2])),
        Some(make_f32(vec![7.0, 11.0], &[2])),
    ]);
    Engine::new().execute(root, &ctx).expect("execute");

    // B must fire EXACTLY ONCE, even though it is both seeded and a
    // descendant of A.
    assert_eq!(
        b_apply_count.load(Ord_::Relaxed),
        1,
        "B fired more than once — double-enqueue bug"
    );

    // Numerical accumulation must be correct.
    let m = leaf_meta.lock().unwrap();
    let g = m.grad.as_ref().expect("leaf grad populated");
    assert_eq!(g.to_vec().unwrap(), vec![10.0, 16.0]);
}

// ---------------------------------------------------------------------------
// Phase 3b carryover fix: AccumulateGrad::hooks() empty-sentinel fast path.
// ---------------------------------------------------------------------------
//
// Phase 2 bug (carryover): `AccumulateGrad::hooks()` returned `&self.hooks`,
// where `hooks: Hooks` was default-initialised to `Hooks::default()`. That
// produced a per-struct empty-but-not-sentinel Hooks instance whose address
// was unique per AccumulateGrad — the engine's pointer-equality fast path
// (`std::ptr::eq(hooks_ref, Hooks::empty_ref())`) never matched, so every
// backward step ran the (empty) for-loops in the hook-dispatch branch.
//
// Phase 3b fix: `hooks: OnceLock<Hooks>`. Default state (uninitialised)
// causes `hooks()` to return `Hooks::empty_ref()`, restoring the fast path.
// After `install_hooks(...)` the per-struct Hooks ref is returned and the
// fast path no longer fires for that node.
//
// This test pins the contract with `std::ptr::eq` on both branches.

#[test]
fn accumulate_grad_uses_empty_sentinel_when_no_hooks() {
    use flame_core::autograd_v2::AccumulateGrad;

    // Fresh accumulator, no hooks installed.
    let leaf_meta: AutogradMetaRef = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
    let acc = AccumulateGrad::new(&leaf_meta, next_seq());

    let h_ref = acc.hooks();
    let empty_ref = Hooks::empty_ref();
    assert!(
        std::ptr::eq(h_ref as *const Hooks, empty_ref as *const Hooks),
        "fresh AccumulateGrad.hooks() must point at the empty sentinel \
         (Phase 2 carryover fix). h_ref = {:p}, empty_ref = {:p}",
        h_ref,
        empty_ref,
    );

    // Install a hooks bundle; pointer must now diverge from sentinel.
    let installed = Hooks::new();
    acc.install_hooks(installed).expect("install_hooks should succeed once");
    let h_ref_after = acc.hooks();
    assert!(
        !std::ptr::eq(h_ref_after as *const Hooks, empty_ref as *const Hooks),
        "after install_hooks(), hooks() must return the per-accumulator \
         Hooks (NOT the empty sentinel)"
    );

    // Second install_hooks call returns Err (single-shot).
    let res = acc.install_hooks(Hooks::new());
    assert!(res.is_err(), "second install_hooks must be Err (single-shot)");
}
