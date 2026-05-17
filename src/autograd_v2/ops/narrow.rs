//! v2 `narrow` — slice along a dim with full autograd recording.
//!
//! Forward: `Tensor::narrow(dim, start, length)` returns a view into
//! `self`'s storage (zero-copy).
//!
//! Backward: scatter the upstream `grad_output` into a fresh zero
//! tensor of the input's shape at `[start..start+length)` along `dim`.
//! Uses `narrow_backward_scatter_add_cuda` (`src/tensor_narrow.rs:169`)
//! which is dtype-agnostic byte-copy.
//!
//! HAZARD-2026-05-13-1 discipline (CRITICAL):
//!
//! - The backward MUST scatter-add into a FRESH zero tensor. NEVER
//!   write through a `narrow()` view back into a parent — under
//!   `shared_storage` (default), the in-place mutation paths COW the
//!   inner CudaSlice when refcount > 1, silently detaching the view's
//!   storage from the parent and leaving the parent untouched. See
//!   §HAZARD-2026-05-13-1 of `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`.
//! - `narrow_backward_scatter_add_cuda` takes `grad_in: &mut Tensor`
//!   (the zero buffer allocated here) — it writes into that buffer's
//!   storage directly. No view-aliasing involved.

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
pub struct NarrowGradFn {
    input_shape: Vec<usize>,
    input_dtype: DType,
    dim: usize,
    start: usize,
    length: usize,
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl NarrowGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor, dim: usize, start: usize, length: usize) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            input_shape: a.shape().dims().to_vec(),
            input_dtype: a.dtype(),
            dim,
            start,
            length,
            next_edges: vec![gradient_edge_for_tensor(a)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for NarrowGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().flatten() {
            None => return Ok(vec![None]),
            Some(g) => g,
        };
        // HAZARD-2026-05-13-1: allocate a FRESH zero tensor (sole
        // owner of its storage); scatter-add the grad_output's slab
        // into it at the saved [start..start+length) along self.dim.
        // The kernel writes through the unique `&mut grad_in` handle
        // — no aliasing risk.
        let target_shape = Shape::from_dims(&self.input_shape);
        let mut grad_in = Tensor::zeros_dtype(target_shape, self.input_dtype, g.device().clone())
            .map_err(AutogradV2Error::FlameCore)?;
        // Ensure grad dtype matches (the kernel is byte-copy and would
        // copy wrong-sized elements if the dtypes differed).
        let g_typed = if g.dtype() == self.input_dtype {
            g
        } else {
            g.to_dtype(self.input_dtype)
                .map_err(AutogradV2Error::FlameCore)?
        };
        Tensor::narrow_backward_scatter_add_cuda(
            &g_typed,
            &mut grad_in,
            self.dim,
            self.start,
            self.length,
        )
        .map_err(AutogradV2Error::FlameCore)?;
        Ok(vec![Some(grad_in)])
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
        "NarrowGradFn"
    }
}

/// v2 forward wrapper for `narrow`.
///
/// HAZARD-2026-05-13-1: forward returns a view aliasing `a`'s storage.
/// Downstream consumers that need an OWNING handle should use
/// `narrow_owning` (which materialises a fresh contiguous copy) — that
/// path is for memory-management hygiene rather than autograd
/// correctness. The autograd path itself is hazard-safe because the
/// backward `apply` allocates a fresh zero tensor for the writeback;
/// the forward view's storage aliasing doesn't affect grad correctness.
pub fn narrow_v2(
    a: &Tensor,
    dim: usize,
    start: usize,
    length: usize,
    ctx: &DispatchCtx,
) -> Result<Tensor> {
    let out = a.narrow(dim, start, length)?;
    let any_fw = any_fw_grad(&[a]);
    let mut result = if needs_grad(&[a]) {
        let grad_fn = NarrowGradFn::new(a, dim, start, length);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        // JVP(narrow): out_fw = a_dot.narrow(dim, start, length). This
        // is a READ-only slice of the tangent (a fresh view onto
        // a_dot's storage); HAZARD-2026-05-13-1 does NOT apply because
        // no in-place write happens here (the hazard concerns writing
        // through a narrow view into a parent). The forward JVP only
        // reads, never writes.
        let out_fw = a_dot.narrow(dim, start, length)?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
