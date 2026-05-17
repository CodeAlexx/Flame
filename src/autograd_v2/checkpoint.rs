//! `CheckpointGradFn` — gradient checkpointing.
//!
//! Per §clause 12 and §8 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`: gradient checkpointing
//! is the canonical user of *reentrant backward*. `torch.utils.checkpoint`
//! drops the forward activations of a sub-graph and re-runs forward
//! during backward to recompute them, then drives a nested
//! `autograd.backward()` on the recomputed graph.
//!
//! Phase 3c1 ships the full implementation:
//!
//! 1. `checkpoint_v2(forward_fn, inputs, ctx)` — the user-facing helper.
//!    Runs `forward_fn` with **detached** input clones so the inner
//!    forward ops do not record (saving activation memory is the whole
//!    point of checkpointing). Captures `(forward_fn, saved_inputs,
//!    next_edges_for_original_inputs)` into a `CheckpointGradFn` and
//!    installs it on every output via `record_v2`.
//!
//! 2. `CheckpointGradFn::apply` — at backward time, the implementation:
//!    Unpacks the saved inputs, detaches each clone and installs a fresh
//!    `requires_grad=true` leaf meta, re-runs the forward closure under
//!    v2 recording so each inner op records its GradFn, drives a fresh
//!    `Engine::execute(...)` over the recomputed sub-graph (no
//!    `with_inputs` — the standard leaf-accumulator path works because
//!    Engine is single-threaded and stateless across calls; reentrancy
//!    is safe per `tests/autograd_v2_engine.rs::reentrant_nested_execute`),
//!    then harvests per-leaf grads off each reattached input's
//!    `meta.grad` (sunk there by its AccumulateGrad) and returns them
//!    to the outer Engine, which routes them along `next_edges`.
//!
//! ## Design choices
//!
//! - **No thread-local "no-record" flag.** Earlier sketches threaded a
//!   thread-local through `needs_grad` to silence recording inside the
//!   outer forward. Instead, we detach inputs before the closure runs:
//!   `Tensor::detach_v2()` drops the autograd_meta, so `needs_grad`
//!   returns false at every inner op site. Cheap, local, no global state.
//!
//! - **`Arc<dyn Fn>` not `Box<dyn Fn>`.** The closure may be reused if
//!   `retain_graph=true` causes a second backward pass; `Arc` makes that
//!   case free. `Send + Sync` is required by `GradFn`'s trait bounds.
//!
//! - **Saved input identity preserved via storage handle**. The
//!   `SavedTensor` slot carries an `Arc` to the storage. Unpacking at
//!   backward returns a fresh `Tensor` handle to that storage; we then
//!   `detach_v2()` to get a meta-less leaf and install
//!   `requires_grad=true` so the recompute records. The original input's
//!   meta is untouched.

use std::sync::Arc;

use super::dispatch::DispatchCtx;
use super::engine::{Engine, GraphRoot};
use super::error::AutogradV2Error;
use super::meta::{new_meta_ref, AutogradMetaV2};
use super::node::{Edge, GradFn, NodeId};
use super::recording::{gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2};
use super::saved_tensor::SavedTensor;
use crate::tensor::Tensor;

/// User-facing forward closure type for `checkpoint_v2`.
///
/// Takes the (re-attached) input tensors + dispatch ctx and returns the
/// sub-graph's outputs. Must be deterministic — `apply()` re-runs this
/// closure during backward and the engine assumes the recomputed shape
/// / dtype matches the outputs saved at outer-forward time.
pub type CheckpointForwardFn =
    dyn Fn(&[Tensor], &DispatchCtx) -> crate::Result<Vec<Tensor>> + Send + Sync;

/// Backward node for activation checkpointing.
///
/// Holds:
/// - `forward_fn` — the user's deterministic forward closure.
/// - `saved_inputs` — version-checked handles to the original inputs.
/// - `next_edges` — one edge per input (= where each input's grad
///   should go after the nested engine call returns).
/// - `num_outputs` — number of outputs the forward closure produced.
///   The engine sizes our `apply` `grad_outputs.len()` from this.
pub struct CheckpointGradFn {
    forward_fn: Arc<CheckpointForwardFn>,
    saved_inputs: Vec<SavedTensor>,
    next_edges: Vec<Edge>,
    num_outputs: usize,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
}

impl CheckpointGradFn {
    fn new(
        forward_fn: Arc<CheckpointForwardFn>,
        saved_inputs: Vec<SavedTensor>,
        next_edges: Vec<Edge>,
        num_outputs: usize,
    ) -> Self {
        let seq = next_sequence_nr();
        Self {
            forward_fn,
            saved_inputs,
            next_edges,
            num_outputs,
            sequence_nr: seq,
            topological_nr: seq,
            node_id: NodeId::new(),
        }
    }
}

impl std::fmt::Debug for CheckpointGradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointGradFn")
            .field("node_id", &self.node_id)
            .field("num_outputs", &self.num_outputs)
            .field("saved_inputs", &self.saved_inputs.len())
            .field("next_edges", &self.next_edges.len())
            .finish()
    }
}

impl GradFn for CheckpointGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // -----------------------------------------------------------------
        // Step 1: unpack saved inputs (version-checked), then detach +
        // install fresh `requires_grad=true` meta on each. The fresh
        // meta is what makes the inner forward record into v2.
        // -----------------------------------------------------------------
        let mut reattached: Vec<Tensor> = Vec::with_capacity(self.saved_inputs.len());
        for saved in &self.saved_inputs {
            let raw = saved.unpack()?;
            // detach_v2 drops the meta (Arc is not shared with the
            // original input's meta after this) and assigns a fresh
            // TensorId. The storage is the same Arc.
            let mut leaf = raw.detach_v2();
            let meta = new_meta_ref(AutogradMetaV2::leaf_requires_grad());
            leaf.set_autograd_meta(Some(meta));
            reattached.push(leaf);
        }

        // -----------------------------------------------------------------
        // Step 2: re-run forward closure under v2 recording. Each inner
        // op sees `needs_grad=true` on its inputs (because our
        // reattached leaves have requires_grad meta), so v2 ops record
        // their backward nodes onto the recomputed outputs.
        // -----------------------------------------------------------------
        let outputs = (self.forward_fn)(&reattached, ctx).map_err(AutogradV2Error::FlameCore)?;

        if outputs.len() != self.num_outputs {
            return Err(AutogradV2Error::FlameCore(crate::Error::InvalidOperation(
                format!(
                    "CheckpointGradFn: forward_fn re-run produced {} outputs, expected {}",
                    outputs.len(),
                    self.num_outputs
                ),
            )));
        }
        if grad_outputs.len() != self.num_outputs {
            return Err(AutogradV2Error::FlameCore(crate::Error::InvalidOperation(
                format!(
                    "CheckpointGradFn: received {} grad_outputs, expected {}",
                    grad_outputs.len(),
                    self.num_outputs
                ),
            )));
        }

        // -----------------------------------------------------------------
        // Step 3: drive the recomputed sub-graph via a fresh Engine
        // (inline-mini-execute, §clause 12). The recomputed outputs each
        // carry a v2 grad_fn. We use the **standard backward path**
        // (no `with_inputs`) — each `reattached` leaf has
        // `requires_grad=true`, so the engine's `gradient_edge`
        // resolution lazily materializes an AccumulateGrad whose
        // `apply()` writes the leaf's gradient into its own
        // `meta.grad` slot. We read those slots back after the
        // engine returns.
        //
        // Why not `with_inputs`? Today's `with_inputs` path assumes the
        // input tensor has a `grad_fn` (non-leaf) — it `expect`s on
        // `grad_fn_of(inp)`. Leaves go through AccumulateGrad and
        // their grad is sunk in `meta.grad`. The conventional path
        // works for both.
        // -----------------------------------------------------------------
        let root = GraphRoot::new(outputs).with_grad_outputs(grad_outputs);
        Engine::new().execute(root, ctx)?;

        // -----------------------------------------------------------------
        // Step 4: harvest per-leaf grads off the reattached inputs.
        // Each leaf's `meta.grad` was written by its AccumulateGrad.
        // -----------------------------------------------------------------
        let mut inner_grads: Vec<Option<Tensor>> = Vec::with_capacity(reattached.len());
        for leaf in &reattached {
            let g = leaf
                .autograd_meta()
                .and_then(|m| m.lock().ok().and_then(|guard| guard.grad.clone()));
            inner_grads.push(g);
        }

        // -----------------------------------------------------------------
        // Route grads outward. Length must match outer next_edges (= number
        // of saved inputs).
        // -----------------------------------------------------------------
        if inner_grads.len() != self.next_edges.len() {
            return Err(AutogradV2Error::FlameCore(crate::Error::InvalidOperation(
                format!(
                    "CheckpointGradFn: inner engine produced {} leaf grads, expected {}",
                    inner_grads.len(),
                    self.next_edges.len()
                ),
            )));
        }
        Ok(inner_grads)
    }

    fn num_inputs(&self) -> usize {
        self.num_outputs
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
        "CheckpointGradFn"
    }

    fn release_variables(&self) {
        for s in &self.saved_inputs {
            s.reset();
        }
    }
}

/// User-facing entry point for activation checkpointing.
///
/// Runs `forward_fn(inputs)` under a **non-recording** wrapper so the
/// inner forward ops do not retain activations. Installs a
/// `CheckpointGradFn` on each output that, at backward time, will:
///
/// 1. Re-run `forward_fn` with autograd recording enabled.
/// 2. Drive a nested `Engine::execute(...)` on the recomputed sub-graph.
/// 3. Return per-input grads to the outer engine.
///
/// ## Contract
///
/// - `forward_fn` must be deterministic. Re-running it with the same
///   inputs must produce the same shape / dtype outputs as the outer
///   forward. Non-determinism (dropout, BN running stats, RNG) breaks
///   gradient correctness — handle those at the caller (use the same
///   RNG state, freeze BN, etc.).
/// - `forward_fn` is `Arc<dyn Fn ...>` so it can be reused if the
///   outer backward runs twice (e.g., `retain_graph=true`).
/// - The closure receives **fresh** leaf tensors during the re-run;
///   the original `inputs` are not the same handles. If your closure
///   captures tensor identity (TensorId compare), this will fail. Use
///   the closure's `&[Tensor]` parameter.
///
/// ## Memory savings
///
/// The whole point: between outer-forward return and outer-backward
/// entry, none of the inner activations are held — only the saved
/// inputs are retained (one `SavedTensor` per input, storage shared via
/// `Arc`). For a long sequential block (e.g., a 24-layer transformer
/// stack), this can reduce activation memory by ~Nx the per-layer
/// activation size.
pub fn checkpoint_v2(
    forward_fn: Arc<CheckpointForwardFn>,
    inputs: &[Tensor],
    ctx: &DispatchCtx,
) -> crate::Result<Vec<Tensor>> {
    // Outer forward: run with detached inputs so inner ops see
    // `needs_grad=false` and skip recording. Each output is fresh
    // storage with no meta.
    let detached: Vec<Tensor> = inputs.iter().map(|t| t.detach_v2()).collect();
    let outputs_raw = forward_fn(&detached, ctx)?;

    // Recording gate: if NONE of the original inputs requires grad,
    // skip wiring a CheckpointGradFn — the user is running inference
    // through a "checkpointed" path and we shouldn't pollute their
    // tensors with autograd nodes.
    let input_refs: Vec<&Tensor> = inputs.iter().collect();
    if !needs_grad(&input_refs) {
        return Ok(outputs_raw);
    }

    // Save inputs for backward recompute. SavedTensor::save_named
    // captures the storage Arc + version snapshot.
    let saved_inputs: Vec<SavedTensor> = inputs
        .iter()
        .map(|t| SavedTensor::save_named(t, "CheckpointGradFn:input"))
        .collect();

    // Build next_edges from the *original* inputs (the ones the user
    // handed in — these may carry meta with grad_fn or requires_grad).
    // gradient_edge_for_tensor returns the proper edge for each.
    let next_edges: Vec<Edge> = inputs.iter().map(gradient_edge_for_tensor).collect();

    let num_outputs = outputs_raw.len();
    let grad_fn: Arc<dyn GradFn> = Arc::new(CheckpointGradFn::new(
        forward_fn,
        saved_inputs,
        next_edges,
        num_outputs,
    ));
    let recorded = record_v2(grad_fn, outputs_raw, ctx);
    Ok(recorded)
}
