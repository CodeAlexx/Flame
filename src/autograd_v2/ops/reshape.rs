//! v2 `reshape` / `view` — shape-only operation with full autograd
//! recording.
//!
//! Backward: `d/d_input reshape(input, new_shape) = grad_output.reshape(input.shape())`.
//!
//! No tensor data needs saving — the backward only needs the input's
//! original shape. `Tensor::view` aliases `Tensor::reshape` in this
//! codebase (`tensor.rs:3528`); both record through `ReshapeGradFn`.
//!
//! Phase 3b view-autograd surface (per `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`
//! §clause 14 / §Phase 3 view subset). HAZARD-2026-05-13-1 discipline:
//! reshape's backward is a pure shape op — no in-place writes, no
//! aliasing risk. No `.contiguous()` needed because reshape preserves
//! contiguity when the input is contiguous (and falls back to
//! materialisation when it isn't, via the existing `Tensor::reshape`
//! implementation).

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::Result;

use super::super::dispatch::DispatchCtx;
use super::super::error::AutogradV2Error;
use super::super::node::{Edge, GradFn, NodeId};
use super::super::recording::{gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2};
use super::fw_mode::{any_fw_grad, tangent_or_zero};

#[derive(Debug)]
pub struct ReshapeGradFn {
    input_shape: Vec<usize>,
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl ReshapeGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            input_shape: a.shape().dims().to_vec(),
            next_edges: vec![gradient_edge_for_tensor(a)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for ReshapeGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().flatten() {
            None => return Ok(vec![None]),
            Some(g) => g,
        };
        // Backward: reshape grad_output back to input shape. No
        // contiguity worry — reshape itself materialises when needed.
        let d_in = g
            .reshape(&self.input_shape)
            .map_err(AutogradV2Error::FlameCore)?;
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
        "ReshapeGradFn"
    }
}

/// v2 forward wrapper for `reshape`.
///
/// Phase 3c2 forward-mode AD: linear-shape-op JVP — apply the same
/// reshape to the tangent, `out_fw = a_dot.reshape(new_shape)`.
pub fn reshape_v2(a: &Tensor, new_shape: &[usize], ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.reshape(new_shape)?;
    let any_fw = any_fw_grad(&[a]);
    let mut result = if needs_grad(&[a]) {
        let grad_fn = ReshapeGradFn::new(a);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        let out_fw = a_dot.reshape(new_shape)?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}

/// v2 forward wrapper for `view` (alias of `reshape` in flame-core).
///
/// Phase 3c2 forward-mode AD: identical to `reshape` JVP.
pub fn view_v2(a: &Tensor, new_shape: &[usize], ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.view(new_shape)?;
    let any_fw = any_fw_grad(&[a]);
    let mut result = if needs_grad(&[a]) {
        let grad_fn = ReshapeGradFn::new(a);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        let out_fw = a_dot.view(new_shape)?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
