//! `AccumulateGrad` — the leaf-tensor gradient sink.
//!
//! Per §1 and recommended-change 1 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`:
//!
//! - Holds a **Weak** handle to the leaf tensor's autograd metadata.
//!   Combined with `AutogradMetaV2::grad_accumulator` (also weak),
//!   this breaks the AccumulateGrad ↔ tensor reference cycle that
//!   would otherwise leak.
//! - Stored `sequence_nr` and `topological_nr` (PyTorch parity — not
//!   recomputed walks).
//! - `apply()` is a Phase-1 placeholder; Phase 2 implements the
//!   actual gradient sink semantics (accumulate into `meta.grad`
//!   respecting BF16-end-to-end / Option A under Phase 4).

use std::sync::Weak;

use super::dispatch::DispatchCtx;
use super::error::AutogradV2Error;
use super::hooks::Hooks;
use super::meta::AutogradMetaRef;
use super::node::{Edge, GradFn, NodeId};
use crate::tensor::Tensor;

pub struct AccumulateGrad {
    /// Weak handle to the leaf tensor's metadata. Upgrades to `Some`
    /// while the tensor is alive; once the tensor is dropped, the
    /// accumulator becomes a no-op (PyTorch reaches the same end via
    /// `set_grad_accumulator(weak)`).
    variable: Weak<std::sync::Mutex<super::meta::AutogradMetaV2>>,
    /// Edges out of the accumulator. For a leaf this is the empty
    /// slice — the accumulator is itself a terminal node.
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
    hooks: Hooks,
}

impl AccumulateGrad {
    /// Build a leaf accumulator pointing at `meta`. The weak handle
    /// breaks the cycle (§1).
    pub fn new(meta: &AutogradMetaRef, sequence_nr: u64) -> Self {
        Self {
            variable: std::sync::Arc::downgrade(meta),
            next_edges: Vec::new(),
            node_id: NodeId::new(),
            sequence_nr,
            // Leaves are always at topological_nr 0 in PyTorch's
            // numbering (engine sets non-leaf nodes' topo to
            // `1 + max(input_topo_nr)`). We mirror that.
            topological_nr: 0,
            hooks: Hooks::default(),
        }
    }

    /// Test/diagnostic: returns true iff the underlying tensor's
    /// metadata is still alive.
    pub fn variable_alive(&self) -> bool {
        self.variable.upgrade().is_some()
    }
}

impl std::fmt::Debug for AccumulateGrad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccumulateGrad")
            .field("node_id", &self.node_id)
            .field("sequence_nr", &self.sequence_nr)
            .field("variable_alive", &self.variable_alive())
            .finish()
    }
}

impl GradFn for AccumulateGrad {
    fn apply(
        &self,
        _grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // Phase 2 implements:
        //   1. Upgrade `self.variable`; if None, drop the grad.
        //   2. Lock the meta, accumulate `grad_outputs[0]` into
        //      `meta.grad` using the BF16-end-to-end path (Option A).
        //   3. Fire post hooks.
        //   4. Return empty (accumulator has no downstream edges).
        Err(AutogradV2Error::NotImplementedYet(
            "AccumulateGrad::apply (Phase 2)",
        ))
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
        "AccumulateGrad"
    }

    fn hooks(&self) -> &Hooks {
        &self.hooks
    }

    fn release_variables(&self) {
        // No saved tensors on AccumulateGrad. Default no-op suffices,
        // but we explicitly note it here so a future reader looking
        // for the saved-tensor release pattern doesn't think it was
        // forgotten.
    }
}
