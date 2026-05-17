//! v2 `matmul` — 2D matrix multiply with full autograd recording.
//!
//! Backward (2D case):
//! - `d_a = g @ b^T`
//! - `d_b = a^T @ g`
//!
//! Both inputs are saved via `SavedTensor` so the version-counter
//! pattern catches in-place mutation. Phase 3a only validates the 2D
//! case (the test uses 2D matrices); Phase 3b/4 may extend for
//! batched / N-D matmul where the backward must reduce-broadcast extra
//! dims.

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
pub struct MatMulGradFn {
    saved_a: SavedTensor,
    saved_b: SavedTensor,
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl MatMulGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(a: &Tensor, b: &Tensor) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        Arc::new(Self {
            saved_a: SavedTensor::save_named(a, "MatMulGradFn:a"),
            saved_b: SavedTensor::save_named(b, "MatMulGradFn:b"),
            next_edges: vec![gradient_edge_for_tensor(a), gradient_edge_for_tensor(b)],
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for MatMulGradFn {
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
        // d_a = g @ b^T  ; d_b = a^T @ g
        //
        // Bug-fixer 2026-05-13: `Tensor::transpose()` returns a
        // non-contiguous view (swapped strides, same storage), but
        // `ops::gemm::launch_gemm` reads `storage.try_as_slice_*` and
        // ignores strides — treating the view as row-major over the
        // physical storage layout. That produces a row/column-scrambled
        // backward (the original test masked this by checking only
        // sum-of-grad, which is invariant under that scrambling).
        //
        // Force contiguous materialization of both transposes so
        // launch_gemm sees the correctly-laid-out memory.
        let b_t = b
            .transpose()
            .and_then(|t| t.contiguous())
            .map_err(AutogradV2Error::FlameCore)?;
        let a_t = a
            .transpose()
            .and_then(|t| t.contiguous())
            .map_err(AutogradV2Error::FlameCore)?;
        let da = g.matmul(&b_t).map_err(AutogradV2Error::FlameCore)?;
        let db = a_t.matmul(&g).map_err(AutogradV2Error::FlameCore)?;
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
        "MatMulGradFn"
    }

    fn release_variables(&self) {
        self.saved_a.reset();
        self.saved_b.reset();
    }
}

/// v2 forward wrapper for `matmul`.
///
/// Phase 3c2 forward-mode AD: product-rule JVP — `out_fw =
/// a_dot @ b + a @ b_dot`. Missing tangents default to zero. The
/// zero-tensor materialisation in `tangent_or_zero` ensures both
/// matmul invocations always run on real tensors (zero matmul is
/// cheap on tiny shapes; in production a future micro-opt could
/// skip the zero-side term).
pub fn matmul_v2(a: &Tensor, b: &Tensor, ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.matmul(b)?;
    let any_fw = any_fw_grad(&[a, b]);
    let mut result = if needs_grad(&[a, b]) {
        let grad_fn = MatMulGradFn::new(a, b);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        recorded.into_iter().next().unwrap()
    } else {
        out
    };
    if any_fw {
        let a_dot = tangent_or_zero(a)?;
        let b_dot = tangent_or_zero(b)?;
        // JVP(matmul): out_fw = a_dot @ b + a @ b_dot.
        let term1 = a_dot.matmul(b)?;
        let term2 = a.matmul(&b_dot)?;
        let out_fw = term1.add(&term2)?;
        result.set_fw_grad(out_fw);
    }
    Ok(result)
}
