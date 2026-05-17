//! v2 `transpose` — 2D row/column swap with full autograd recording.
//!
//! Phase 3b: `Tensor::transpose()` is 2D-only in flame-core (see
//! `tensor.rs:2958`); general N-D transpose-by-axes goes through
//! `permute`. Backward: `grad_input = grad_output.transpose()` — same
//! op applied to the upstream grad.
//!
//! No SavedTensor needed (no tensor data goes into backward — just the
//! shape op).
//!
//! HAZARD-2026-05-13-1 + gemm stride-ignore discipline: `Tensor::
//! transpose` is a zero-copy view that swaps strides. Project-wide,
//! gemm / bmm kernels ignore strides and walk physical row-major
//! storage — passing a transpose view straight into matmul produces
//! scrambled output (Phase 3a matmul fix `6ee385f`). To make this
//! backward safe to feed into any downstream op (matmul especially),
//! we call `.contiguous()` on the transposed grad_output before
//! returning. One contiguity copy per backward call.

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::Result;

use super::super::dispatch::DispatchCtx;
use super::super::error::AutogradV2Error;
use super::super::node::{Edge, GradFn, NodeId};
use super::super::recording::{gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2};
use super::fw_mode::{any_fw_grad, tangent_or_zero};

#[derive(Debug)]
pub struct TransposeGradFn {
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl TransposeGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            next_edges: vec![gradient_edge_for_tensor(a)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for TransposeGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().flatten() {
            None => return Ok(vec![None]),
            Some(g) => g,
        };
        // HAZARD-2026-05-13-1 + gemm-stride-ignore: `.transpose()` is a
        // strided view; materialise contiguous before returning so any
        // downstream gemm/bmm consumer sees the correctly-laid-out
        // memory. See Phase 3a matmul fix (commit 6ee385f) for the
        // original symptom.
        let d_in = g
            .transpose()
            .and_then(|t| t.contiguous())
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
        "TransposeGradFn"
    }
}

/// v2 forward wrapper for 2D `transpose`.
///
/// Phase 3c2 forward-mode AD: linear shape-op JVP — apply `.transpose()`
/// to the tangent, then `.contiguous()` per HAZARD-2026-05-13-1 +
/// gemm-stride-ignore (same discipline as the backward formula above).
pub fn transpose_v2(a: &Tensor, ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.transpose()?;
    let any_fw = any_fw_grad(&[a]);
    let mut result = if needs_grad(&[a]) {
        let grad_fn = TransposeGradFn::new(a);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        // HAZARD-2026-05-13-1 + gemm-stride-ignore: materialise
        // contiguous so a downstream gemm/bmm consumer sees the
        // correctly-laid-out memory.
        let out_fw = a_dot.transpose()?.contiguous()?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
