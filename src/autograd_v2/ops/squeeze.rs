//! v2 `squeeze` — remove a unit dimension with full autograd recording.
//!
//! Forward: `Tensor::squeeze(Some(dim))` (`tensor_ops_extended.rs:75`)
//! removes the dim if its size is 1, otherwise returns a clone.
//!
//! Backward: `grad_input = grad_output.unsqueeze(dim)` — re-insert the
//! removed unit dim. We save only the dim index (a `usize`).
//!
//! Phase 3b ships the `squeeze_dim` flavor (explicit dim argument).
//! The `squeeze` overload that removes ALL unit dims (`Tensor::
//! squeeze(None)`) is rare in trainer code; can be added in Phase 3c
//! if needed.
//!
//! No HAZARD-2026-05-13-1 concern — shape-only op, no aliased writes.

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::Result;

use super::super::dispatch::DispatchCtx;
use super::super::error::AutogradV2Error;
use super::super::node::{Edge, GradFn, NodeId};
use super::super::recording::{gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2};
use super::fw_mode::{any_fw_grad, tangent_or_zero};

#[derive(Debug)]
pub struct SqueezeGradFn {
    dim: usize,
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl SqueezeGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor, dim: usize) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            dim,
            next_edges: vec![gradient_edge_for_tensor(a)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for SqueezeGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().flatten() {
            None => return Ok(vec![None]),
            Some(g) => g,
        };
        // Inverse of squeeze is unsqueeze at the same dim.
        let d_in = g.unsqueeze(self.dim).map_err(AutogradV2Error::FlameCore)?;
        Ok(vec![Some(d_in)])
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
        "SqueezeGradFn"
    }
}

/// v2 forward wrapper for `squeeze(Some(dim))`.
///
/// Phase 3c2 forward-mode AD: linear shape-op JVP — apply the same
/// squeeze to the tangent, `out_fw = a_dot.squeeze(Some(dim))`.
pub fn squeeze_v2(a: &Tensor, dim: usize, ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.squeeze(Some(dim))?;
    let any_fw = any_fw_grad(&[a]);
    let mut result = if needs_grad(&[a]) {
        let grad_fn = SqueezeGradFn::new(a, dim);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        let out_fw = a_dot.squeeze(Some(dim))?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
