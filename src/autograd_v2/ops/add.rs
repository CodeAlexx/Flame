//! v2 `add` — pointwise BF16/F32 addition with full autograd recording.
//!
//! Backward: `d(a+b)/da = 1`, `d(a+b)/db = 1`. No saved tensors needed —
//! the upstream gradient flows unchanged into both inputs.
//!
//! Phase 3a math is delegated to `Tensor::add`. The op is the canonical
//! Phase 3a template (`mul` / `matmul` / `sum` / `silu` all follow the
//! same skeleton).

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::Result;

use super::super::dispatch::DispatchCtx;
use super::super::error::AutogradV2Error;
use super::super::hooks::Hooks;
use super::super::node::{Edge, GradFn, NodeId};
use super::super::recording::{
    gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2,
};

#[derive(Debug)]
pub struct AddGradFn {
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl AddGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor, b: &Tensor) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            next_edges: vec![
                gradient_edge_for_tensor(a),
                gradient_edge_for_tensor(b),
            ],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for AddGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // grad_outputs.len() == num_inputs == 1 (one output of forward).
        let g = grad_outputs.into_iter().next().flatten();
        // Backward: route the same upstream grad to BOTH next_edges
        // (one per input). Clone the Tensor handle — cheap (Arc bump
        // on storage, plus the autograd_meta Arc bump).
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

    // Do NOT override `hooks()` — empty sentinel fast-path enabled.
}

/// v2 forward wrapper for `add`. Forwards math through
/// `Tensor::add`; records a v2 backward node iff at least one input
/// is being tracked through v2.
pub fn add_v2(a: &Tensor, b: &Tensor, ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.add(b)?;
    if needs_grad(&[a, b]) {
        let grad_fn = AddGradFn::new(a, b);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        Ok(recorded.into_iter().next().unwrap())
    } else {
        Ok(out)
    }
}
