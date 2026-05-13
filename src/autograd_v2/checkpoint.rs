//! `CheckpointGradFn` — skeleton for activation checkpointing.
//!
//! Per §clause 12 and §8 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`: gradient checkpointing
//! is the canonical user of *reentrant backward*. `torch.utils.checkpoint`
//! drops the forward activations of a sub-graph and re-runs forward
//! during backward to recompute them, then drives a nested
//! `autograd.backward()` on the recomputed graph.
//!
//! flame-core's existing `flame_core::autograd::checkpoint` already
//! exhibits this pattern. v2's `CheckpointGradFn` reproduces it in the
//! v2 trait shape: `apply()` calls `Engine::new().execute(...)`
//! recursively on the recomputed sub-graph.
//!
//! Phase 2 ships the **skeleton** — the `apply()` returns
//! `NotImplementedYet` because the full implementation needs Phase 3's
//! forward-op recording machinery to rebuild the sub-graph on the v2
//! tape. The state save/restore boundaries are exercised today by the
//! `reentrant_nested_execute` test in `tests/autograd_v2_engine.rs`,
//! which uses a hand-rolled GradFn that calls `Engine::new().execute`
//! from its own `apply()`.
//!
//! When Phase 3 lands, `CheckpointGradFn::apply` becomes:
//!
//! 1. Re-run the user's forward closure with v2 recording enabled,
//!    capturing the produced sub-graph.
//! 2. Build a `GraphRoot` whose outputs are the recomputed sub-graph's
//!    outputs and whose grad_outputs are the gradients flowing into
//!    this checkpoint node.
//! 3. Drive `Engine::new().execute(root, ctx)` on it.
//! 4. Collect per-input grads via `with_inputs(saved_inputs.clone())`.
//! 5. Return those as our own output grads.
//!
//! The skeleton is here so Phase 3 has a concrete target to extend, and
//! so the engine's state save/restore semantics on nested execute are
//! exercised at the right boundary.

use super::dispatch::DispatchCtx;
use super::error::AutogradV2Error;
use super::node::{Edge, GradFn, NodeId};
use crate::tensor::Tensor;

/// Skeleton for activation checkpointing. Phase 2 wires the trait
/// shape; Phase 3 fills `apply()` with forward-recompute + nested
/// engine drive.
pub struct CheckpointGradFn {
    /// Edges out of this node — one per input of the checkpointed
    /// region.
    next_edges: Vec<Edge>,
    /// Number of inputs the checkpointed region consumed. Drives the
    /// `InputBuffer` size when this node is the target of an edge.
    num_inputs: usize,
    sequence_nr: u64,
    topological_nr: u64,
    node_id: NodeId,
}

impl CheckpointGradFn {
    pub fn new(
        next_edges: Vec<Edge>,
        num_inputs: usize,
        sequence_nr: u64,
        topological_nr: u64,
    ) -> Self {
        Self {
            next_edges,
            num_inputs,
            sequence_nr,
            topological_nr,
            node_id: NodeId::new(),
        }
    }
}

impl std::fmt::Debug for CheckpointGradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointGradFn")
            .field("node_id", &self.node_id)
            .field("num_inputs", &self.num_inputs)
            .field("next_edges_len", &self.next_edges.len())
            .finish()
    }
}

impl GradFn for CheckpointGradFn {
    fn apply(
        &self,
        _grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // Phase 3 implementation outline:
        //   1. Re-run the saved forward closure under v2 recording.
        //   2. Build a GraphRoot from the recomputed outputs +
        //      `grad_outputs`.
        //   3. Drive a nested `Engine::new().execute(root, _ctx)`.
        //   4. Return the per-input grads from `with_inputs`.
        Err(AutogradV2Error::NotImplementedYet(
            "CheckpointGradFn needs Phase 3 forward ops",
        ))
    }

    fn num_inputs(&self) -> usize {
        self.num_inputs
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
}
