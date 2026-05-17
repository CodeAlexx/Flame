//! v2 `permute` — N-D axis reorder with full autograd recording.
//!
//! Forward: `Tensor::permute(dims)` (`tensor.rs:3619`) — zero-copy view
//! with reordered strides.
//!
//! Backward: `grad_input = grad_output.permute(inverse_perm)`. We save
//! only the permutation `Vec<usize>` (or equivalently the inverse).
//!
//! HAZARD-2026-05-13-1 + gemm stride-ignore discipline: `Tensor::
//! permute` returns a strided view; gemm/bmm consumers would read
//! that view as if it were row-major, scrambling the result. The
//! backward calls `.contiguous()` on the permuted grad before
//! returning, mirroring the Phase 3a matmul fix (`6ee385f`) and the
//! Phase 3b `TransposeGradFn` discipline.

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::Result;

use super::super::dispatch::DispatchCtx;
use super::super::error::AutogradV2Error;
use super::super::node::{Edge, GradFn, NodeId};
use super::super::recording::{gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2};
use super::fw_mode::{any_fw_grad, tangent_or_zero};

#[derive(Debug)]
pub struct PermuteGradFn {
    inverse_perm: Vec<usize>,
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl PermuteGradFn {
    /// Build a PermuteGradFn from the forward permutation `perm`. The
    /// inverse is computed once at construction; backward just applies it.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor, perm: &[usize]) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        // inverse_perm[perm[i]] = i, i.e. perm composed with inverse_perm = identity.
        let mut inverse_perm = vec![0usize; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inverse_perm[p] = i;
        }
        Arc::new(Self {
            inverse_perm,
            next_edges: vec![gradient_edge_for_tensor(a)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for PermuteGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().flatten() {
            None => return Ok(vec![None]),
            Some(g) => g,
        };
        // HAZARD-2026-05-13-1 + gemm-stride-ignore: `.permute()` is a
        // strided view; materialise contiguous before returning so any
        // downstream gemm/bmm/elementwise consumer sees correctly-
        // laid-out memory. See Phase 3a matmul fix (commit 6ee385f).
        let d_in = g
            .permute(&self.inverse_perm)
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
        "PermuteGradFn"
    }
}

/// v2 forward wrapper for `permute`.
///
/// Phase 3c2 forward-mode AD: linear shape-op JVP — apply the same
/// permute to the tangent, then `.contiguous()` per HAZARD-2026-05-13-1
/// + gemm-stride-ignore (same discipline as the backward formula).
pub fn permute_v2(a: &Tensor, perm: &[usize], ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.permute(perm)?;
    let any_fw = any_fw_grad(&[a]);
    let mut result = if needs_grad(&[a]) {
        let grad_fn = PermuteGradFn::new(a, perm);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        let out_fw = a_dot.permute(perm)?.contiguous()?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
