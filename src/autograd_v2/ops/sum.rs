//! v2 `sum` — full-tensor reduction with full autograd recording.
//!
//! Backward: `d(sum(x))/dx = broadcast(g, x.shape())` — the scalar
//! upstream gradient is broadcast back to the input shape.
//!
//! No SavedTensor needed — the only thing the backward needs is the
//! input's shape, which is stored as a `Vec<usize>` on the struct. The
//! input's dtype is also needed for the broadcast destination.

use std::sync::Arc;

use crate::dtype::DType;
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::Result;

use super::super::dispatch::DispatchCtx;
use super::super::error::AutogradV2Error;
use super::super::node::{Edge, GradFn, NodeId};
use super::super::recording::{gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2};
use super::fw_mode::{any_fw_grad, tangent_or_zero};

#[derive(Debug)]
pub struct SumGradFn {
    input_shape: Vec<usize>,
    input_dtype: DType,
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl SumGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            input_shape: a.shape().dims().to_vec(),
            input_dtype: a.dtype(),
            next_edges: vec![gradient_edge_for_tensor(a)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for SumGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().flatten() {
            None => return Ok(vec![None]),
            Some(g) => g,
        };
        // Ensure g's dtype matches the input dtype; broadcast then.
        let g_typed = if g.dtype() == self.input_dtype {
            g
        } else {
            g.to_dtype(self.input_dtype)
                .map_err(AutogradV2Error::FlameCore)?
        };
        let target = Shape::from_dims(&self.input_shape);
        let broadcast = g_typed
            .broadcast_to(&target)
            .map_err(AutogradV2Error::FlameCore)?;
        Ok(vec![Some(broadcast)])
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
        "SumGradFn"
    }
}

/// v2 forward wrapper for `sum`.
///
/// Phase 3c2 forward-mode AD: linear-op JVP — `out_fw = sum(a_dot)`.
pub fn sum_v2(a: &Tensor, ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.sum()?;
    let any_fw = any_fw_grad(&[a]);
    let mut result = if needs_grad(&[a]) {
        let grad_fn = SumGradFn::new(a);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        // JVP(sum): out_fw = sum(a_dot).
        let out_fw = a_dot.sum()?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
