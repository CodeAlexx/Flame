//! v2 `layer_norm` — affine layer normalization with full autograd recording.
//!
//! Forward delegates to `crate::cuda_ops_bf16::layer_norm_bf16` (BF16
//! fast kernel). Backward delegates to
//! `crate::cuda_ops_bf16::layer_norm_backward_bf16`, which recomputes
//! per-feature `mean` / `rstd` internally — so this op only needs to
//! save `(x, weight, bias)` plus `(normalized_shape, eps)`.
//!
//! BF16-only at the public boundary. F32 inputs return
//! `AutogradV2Error::FlameCore(Error::InvalidInput)` — matches the v3
//! `layer_norm` policy. Phase 4 may add an F32 path if a trainer needs
//! it; today's Z-Image / Klein / Wan trainers all run BF16 LN.
//!
//! Backward formula (verified vs the BF16 fused kernel, which mirrors
//! PyTorch's `aten/src/ATen/native/layer_norm.cpp::LayerNormBackwardCPU`):
//!
//! ```text
//! x_hat = (x - mean) * rstd       (per feature)
//! d_y_hat = d_y * weight
//! d_x = (1/N) * rstd * (
//!         N * d_y_hat - sum(d_y_hat, axes) - x_hat * sum(d_y_hat * x_hat, axes)
//!       )
//! d_weight = sum(d_y * x_hat, batch_axes)
//! d_bias = sum(d_y, batch_axes)
//! ```
//!
//! The kernel does this internally; the Rust side only forwards the
//! saved primal tensors back into it. We unit-test the *contract*
//! (shapes, finite-difference parity on a tiny example), not the
//! kernel arithmetic which is already exercised by v3 tests.

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::Result;

use super::super::dispatch::DispatchCtx;
use super::super::error::AutogradV2Error;
use super::super::node::{Edge, GradFn, NodeId};
use super::super::recording::{
    gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2,
};
use super::super::saved_tensor::SavedTensor;

#[derive(Debug)]
pub struct LayerNormGradFn {
    saved_x: SavedTensor,
    saved_weight: Option<SavedTensor>,
    saved_bias: Option<SavedTensor>,
    normalized_shape: Vec<usize>,
    eps: f32,
    /// `next_edges[0]` = x; `next_edges[1]` = weight (if any);
    /// `next_edges[2]` = bias (if any). The slot layout drives both
    /// `num_inputs` (= number of outputs = 1) and the routing in
    /// `apply()`.
    next_edges: Vec<Edge>,
    /// How many of (weight, bias) are present, in {0,1,2}. Used to
    /// shape the output `Vec<Option<Tensor>>` from `apply`.
    has_weight: bool,
    has_bias: bool,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
}

impl LayerNormGradFn {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        x: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        normalized_shape: Vec<usize>,
        eps: f32,
    ) -> Arc<dyn GradFn> {
        let seq = next_sequence_nr();
        let mut edges = vec![gradient_edge_for_tensor(x)];
        if let Some(w) = weight {
            edges.push(gradient_edge_for_tensor(w));
        }
        if let Some(b) = bias {
            edges.push(gradient_edge_for_tensor(b));
        }
        Arc::new(Self {
            saved_x: SavedTensor::save_named(x, "LayerNormGradFn:x"),
            saved_weight: weight.map(|w| SavedTensor::save_named(w, "LayerNormGradFn:weight")),
            saved_bias: bias.map(|b| SavedTensor::save_named(b, "LayerNormGradFn:bias")),
            normalized_shape,
            eps,
            next_edges: edges,
            has_weight: weight.is_some(),
            has_bias: bias.is_some(),
            node_id: NodeId::new(),
            sequence_nr: seq,
            topological_nr: seq,
        })
    }
}

impl GradFn for LayerNormGradFn {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> std::result::Result<Vec<Option<Tensor>>, AutogradV2Error> {
        // Output count is 1; collect the single grad_output.
        let dy = match grad_outputs.into_iter().next().flatten() {
            None => {
                // No upstream grad; route None into every input slot.
                let mut out = vec![None];
                if self.has_weight {
                    out.push(None);
                }
                if self.has_bias {
                    out.push(None);
                }
                return Ok(out);
            }
            Some(g) => g,
        };

        let x = self.saved_x.unpack()?;
        let weight = match &self.saved_weight {
            Some(s) => Some(s.unpack()?),
            None => None,
        };
        let bias = match &self.saved_bias {
            Some(s) => Some(s.unpack()?),
            None => None,
        };

        // The BF16 backward kernel expects BF16 dy / x / gamma / beta.
        // Cast dy if the upstream sent F32 (e.g., from the engine's
        // ones_like default for F32-scalar outputs — not applicable here
        // since LN output is BF16, but the conversion is cheap defensively).
        let dy_bf16_owned;
        let dy_ref = if dy.dtype() == crate::DType::BF16 {
            &dy
        } else {
            dy_bf16_owned = dy
                .to_dtype(crate::DType::BF16)
                .map_err(AutogradV2Error::FlameCore)?;
            &dy_bf16_owned
        };

        // Contiguity: the bwd kernel reads raw device pointers as
        // `[outer * norm_size]`. Mirror the v3 fix.
        let x_contig_owned;
        let x_ref = if x.is_contiguous() {
            &x
        } else {
            x_contig_owned = x.contiguous().map_err(AutogradV2Error::FlameCore)?;
            &x_contig_owned
        };
        let dy_contig_owned;
        let dy_ref = if dy_ref.is_contiguous() {
            dy_ref
        } else {
            dy_contig_owned = dy_ref
                .contiguous()
                .map_err(AutogradV2Error::FlameCore)?;
            &dy_contig_owned
        };

        let (dx, dweight, dbias) = crate::cuda_ops_bf16::layer_norm_backward_bf16(
            x_ref,
            dy_ref,
            weight.as_ref(),
            bias.as_ref(),
            &self.normalized_shape,
            self.eps,
        )
        .map_err(AutogradV2Error::FlameCore)?;

        // Per the kernel: dweight / dbias come back as F32. Cast them to
        // BF16 so they're shape+dtype-compatible with the leaf's storage
        // (weight/bias are BF16 affine params).
        let dweight_bf16 = match dweight {
            Some(dw) if dw.dtype() == crate::DType::BF16 => Some(dw),
            Some(dw) => Some(
                dw.to_dtype(crate::DType::BF16)
                    .map_err(AutogradV2Error::FlameCore)?,
            ),
            None => None,
        };
        let dbias_bf16 = match dbias {
            Some(db) if db.dtype() == crate::DType::BF16 => Some(db),
            Some(db) => Some(
                db.to_dtype(crate::DType::BF16)
                    .map_err(AutogradV2Error::FlameCore)?,
            ),
            None => None,
        };

        // Layout: dx, [dweight], [dbias] — only push the slots that
        // correspond to recorded inputs (== have a SavedTensor).
        let mut out = vec![Some(dx)];
        if self.has_weight {
            out.push(dweight_bf16);
        }
        if self.has_bias {
            out.push(dbias_bf16);
        }
        Ok(out)
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
        "LayerNormGradFn"
    }

    fn release_variables(&self) {
        self.saved_x.reset();
        if let Some(s) = &self.saved_weight {
            s.reset();
        }
        if let Some(s) = &self.saved_bias {
            s.reset();
        }
    }
}

/// v2 forward wrapper for layer-norm. Wraps `cuda_ops_bf16::layer_norm_bf16`.
///
/// Requires BF16 storage on `x`, `weight`, `bias` (when present). For F32
/// LN, use the v3 path (or add an F32 path here in Phase 4 if a trainer
/// needs it).
///
/// `normalized_shape` is the trailing axes to normalize over — same
/// PyTorch contract as the v3 helper.
pub fn layer_norm_v2(
    x: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
    ctx: &DispatchCtx,
) -> Result<Tensor> {
    // Forward computation. Reuse the v3 BF16 fast kernel — it already
    // does all the shape / dtype validation and the contiguity fix.
    let out = crate::cuda_ops_bf16::layer_norm_bf16(x, weight, bias, eps)?;

    // Recording gate: any input (x, weight, bias) with requires_grad
    // triggers v2 recording. The boilerplate `needs_grad` only takes a
    // `&[&Tensor]` so we build one inline here.
    let mut grad_inputs: Vec<&Tensor> = vec![x];
    if let Some(w) = weight {
        grad_inputs.push(w);
    }
    if let Some(b) = bias {
        grad_inputs.push(b);
    }

    if needs_grad(&grad_inputs) {
        let grad_fn = LayerNormGradFn::new(x, weight, bias, normalized_shape.to_vec(), eps);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        Ok(recorded.into_iter().next().unwrap())
    } else {
        Ok(out)
    }
}
