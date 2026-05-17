//! v2 `mul` — pointwise BF16/F32 multiply with full autograd recording.
//!
//! Backward: `d(a*b)/da = b`, `d(a*b)/db = a`. Both inputs are saved
//! via `SavedTensor` so the version-counter pattern catches in-place
//! mutation of either operand after forward.

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::Result;

use super::super::dispatch::DispatchCtx;
use super::super::error::AutogradV2Error;
use super::super::node::{Edge, GradFn, NodeId};
use super::super::recording::{gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2};
use super::super::saved_tensor::SavedTensor;
use super::fw_mode::{any_fw_grad, tangent_or_zero};

#[derive(Debug)]
pub struct MulGradFn {
    saved_a: SavedTensor,
    saved_b: SavedTensor,
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl MulGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor, b: &Tensor) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            saved_a: SavedTensor::save_named(a, "MulGradFn:a"),
            saved_b: SavedTensor::save_named(b, "MulGradFn:b"),
            next_edges: vec![gradient_edge_for_tensor(a), gradient_edge_for_tensor(b)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for MulGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().flatten() {
            None => return Ok(vec![None, None]),
            Some(g) => g,
        };
        let a = self.saved_a.unpack()?;
        let b = self.saved_b.unpack()?;
        // Backward: d_a = g * b, d_b = g * a.
        let da = g.mul(&b).map_err(AutogradV2Error::FlameCore)?;
        let db = g.mul(&a).map_err(AutogradV2Error::FlameCore)?;
        Ok(vec![Some(da), Some(db)])
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
        "MulGradFn"
    }

    fn release_variables(&self) {
        self.saved_a.reset();
        self.saved_b.reset();
    }
}

/// v2 forward wrapper for `mul`.
///
/// Phase 3c2 forward-mode AD: product-rule JVP — when any input has
/// a `fw_grad` installed, `out_fw = a_dot * b + a * b_dot`. Missing
/// tangents default to zero (zero contribution from the missing term).
pub fn mul_v2(a: &Tensor, b: &Tensor, ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.mul(b)?;
    let any_fw = any_fw_grad(&[a, b]);
    let mut result = if needs_grad(&[a, b]) {
        let grad_fn = MulGradFn::new(a, b);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        let b_dot = tangent_or_zero(b)?;
        // JVP(mul): out_fw = a_dot * b + a * b_dot (product rule).
        let term1 = a_dot.mul(b)?;
        let term2 = a.mul(&b_dot)?;
        let out_fw = term1.add(&term2)?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
