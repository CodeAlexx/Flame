//! v2 `silu` — `x * sigmoid(x)` with full autograd recording.
//!
//! Backward: `d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`.
//!
//! Decomposition (Phase 3a uses primitive elementwise ops; Phase 3c may
//! replace with a fused kernel call once forward-mode AD plumbing is in
//! place):
//! - `s = sigmoid(x)`
//! - `one_minus_s = 1 - s`     via `s * -1 + 1`
//! - `term = 1 + x * (1 - s)`  via `x * one_minus_s + 1`
//! - `factor = s * term`
//! - `d_x = g * factor`
//!
//! Input is saved via `SavedTensor`; version-counter pattern catches
//! in-place mutation.

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
pub struct SiLUGradFn {
    saved_x: SavedTensor,
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl SiLUGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(x: &Tensor) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            saved_x: SavedTensor::save_named(x, "SiLUGradFn:x"),
            next_edges: vec![gradient_edge_for_tensor(x)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for SiLUGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().flatten() {
            None => return Ok(vec![None]),
            Some(g) => g,
        };
        let x = self.saved_x.unpack()?;
        let s = x.sigmoid().map_err(AutogradV2Error::FlameCore)?;
        let one_minus_s = s
            .mul_scalar(-1.0)
            .map_err(AutogradV2Error::FlameCore)?
            .add_scalar(1.0)
            .map_err(AutogradV2Error::FlameCore)?;
        let x_times_oms = x.mul(&one_minus_s).map_err(AutogradV2Error::FlameCore)?;
        let term = x_times_oms
            .add_scalar(1.0)
            .map_err(AutogradV2Error::FlameCore)?;
        let factor = s.mul(&term).map_err(AutogradV2Error::FlameCore)?;
        let dx = g.mul(&factor).map_err(AutogradV2Error::FlameCore)?;
        Ok(vec![Some(dx)])
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
        "SiLUGradFn"
    }

    fn release_variables(&self) {
        self.saved_x.reset();
    }
}

/// v2 forward wrapper for `silu`.
///
/// Phase 3c2 forward-mode AD: `silu'(x) = sigmoid(x) * (1 + x*(1 -
/// sigmoid(x)))`. JVP — `out_fw = x_dot * silu'(x)`. The derivative is
/// computed the same way the backward does (see [`SiLUGradFn::apply`]):
///
/// ```text
/// s = sigmoid(x)
/// term = 1 + x * (1 - s)
/// silu_deriv = s * term
/// out_fw = x_dot * silu_deriv
/// ```
pub fn silu_v2(x: &Tensor, ctx: &DispatchCtx) -> Result<Tensor> {
    let out = x.silu()?;
    let any_fw = any_fw_grad(&[x]);
    let mut result = if needs_grad(&[x]) {
        let grad_fn = SiLUGradFn::new(x);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let x_dot = tangent_or_zero(x)?;
        // silu_deriv(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
        let s = x.sigmoid()?;
        let one_minus_s = s.mul_scalar(-1.0)?.add_scalar(1.0)?;
        let x_times_oms = x.mul(&one_minus_s)?;
        let term = x_times_oms.add_scalar(1.0)?;
        let silu_deriv = s.mul(&term)?;
        let out_fw = x_dot.mul(&silu_deriv)?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
