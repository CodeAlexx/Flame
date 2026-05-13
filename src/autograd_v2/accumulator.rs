//! `AccumulateGrad` — the leaf-tensor gradient sink.
//!
//! Per §1 and recommended-change 1 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`:
//!
//! - Holds a **Weak** handle to the leaf tensor's autograd metadata.
//!   Combined with `AutogradMetaV2::grad_accumulator` (also weak),
//!   this breaks the AccumulateGrad ↔ tensor reference cycle that
//!   would otherwise leak.
//! - Stored `sequence_nr` and `topological_nr` (PyTorch parity — not
//!   recomputed walks).
//! - Phase 2 `apply()` implements the real sink semantics: drop empty
//!   grad inputs silently, accumulate into `meta.grad` honoring the
//!   parameter's dtype (Option A from `docs/BF16_GRAD_DECISION.md` —
//!   partial in Phase 2: only the grad dtype is preserved end-to-end
//!   here; the full optimizer/GradientMap migration is Phase 4).

use std::sync::Weak;

use super::dispatch::DispatchCtx;
use super::error::AutogradV2Error;
use super::hooks::Hooks;
use super::meta::AutogradMetaRef;
use super::node::{Edge, GradFn, NodeId};
use crate::tensor::Tensor;

pub struct AccumulateGrad {
    /// Weak handle to the leaf tensor's metadata. Upgrades to `Some`
    /// while the tensor is alive; once the tensor is dropped, the
    /// accumulator becomes a no-op (PyTorch reaches the same end via
    /// `set_grad_accumulator(weak)`).
    variable: Weak<std::sync::Mutex<super::meta::AutogradMetaV2>>,
    /// Edges out of the accumulator. For a leaf this is the empty
    /// slice — the accumulator is itself a terminal node.
    next_edges: Vec<Edge>,
    node_id: NodeId,
    sequence_nr: u64,
    topological_nr: u64,
    hooks: Hooks,
}

impl AccumulateGrad {
    /// Build a leaf accumulator pointing at `meta`. The weak handle
    /// breaks the cycle (§1).
    pub fn new(meta: &AutogradMetaRef, sequence_nr: u64) -> Self {
        Self {
            variable: std::sync::Arc::downgrade(meta),
            next_edges: Vec::new(),
            node_id: NodeId::new(),
            sequence_nr,
            // Leaves are always at topological_nr 0 in PyTorch's
            // numbering (engine sets non-leaf nodes' topo to
            // `1 + max(input_topo_nr)`). We mirror that.
            topological_nr: 0,
            hooks: Hooks::default(),
        }
    }

    /// Test/diagnostic: returns true iff the underlying tensor's
    /// metadata is still alive.
    pub fn variable_alive(&self) -> bool {
        self.variable.upgrade().is_some()
    }

    /// Upgrade the held `Weak<Mutex<AutogradMetaV2>>` to its strong form
    /// if the variable is still alive. Used by the engine's leaf-grad
    /// collection path and by `gradient_edge()`.
    pub fn upgrade_variable(&self) -> Option<AutogradMetaRef> {
        self.variable.upgrade()
    }
}

impl std::fmt::Debug for AccumulateGrad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccumulateGrad")
            .field("node_id", &self.node_id)
            .field("sequence_nr", &self.sequence_nr)
            .field("variable_alive", &self.variable_alive())
            .finish()
    }
}

impl GradFn for AccumulateGrad {
    /// Sink one input gradient into the leaf tensor's `meta.grad` slot.
    ///
    /// Semantics:
    /// - `grad_outputs[0] == None` → no-op (the engine routed no grad here).
    /// - Variable already dropped (Weak fails to upgrade) → no-op (the
    ///   leaf is gone; accumulating would write into nothing).
    /// - First contributor → store `g` directly.
    /// - Subsequent contributors with same dtype + shape → in-place
    ///   accumulation via `ops::elt::add_inplace_same_dtype`.
    /// - Dtype mismatch → `Err(DtypeMismatch)`.
    /// - Shape mismatch → `Err(FlameCore(Error::ShapeMismatch))` via the
    ///   in-place op itself.
    ///
    /// Phase 2 always takes the in-place path when dtypes/shapes match
    /// because `create_graph=true` is not yet plumbed to per-node state
    /// (it lives on `GraphRoot` and `InputBuffer`). When Phase 3 wires
    /// recordable adds, AccumulateGrad will pick its path the same way
    /// `InputBuffer::add` does today.
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        let g = match grad_outputs.into_iter().next().and_then(|opt| opt) {
            None => return Ok(Vec::new()),
            Some(g) => g,
        };

        // If the leaf's meta has been dropped, silently discard the
        // grad (the parameter is gone; nothing to write into).
        let meta_arc = match self.variable.upgrade() {
            None => return Ok(Vec::new()),
            Some(m) => m,
        };

        let mut meta = meta_arc
            .lock()
            .map_err(|_| AutogradV2Error::NotImplementedYet(
                "AccumulateGrad: poisoned meta mutex",
            ))?;

        match meta.grad.take() {
            None => {
                meta.grad = Some(g);
            }
            Some(existing) => {
                if existing.dtype() != g.dtype() {
                    return Err(AutogradV2Error::DtypeMismatch {
                        existing: existing.dtype(),
                        incoming: g.dtype(),
                    });
                }
                #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
                {
                    let mut e = existing;
                    crate::ops::elt::add_inplace_same_dtype(&mut e, &g)
                        .map_err(AutogradV2Error::FlameCore)?;
                    meta.grad = Some(e);
                }
                #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
                {
                    let _ = existing;
                    let _ = g;
                    return Err(AutogradV2Error::NotImplementedYet(
                        "AccumulateGrad::apply requires cuda+bf16_u16 features",
                    ));
                }
            }
        }

        Ok(Vec::new())
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
        "AccumulateGrad"
    }

    fn hooks(&self) -> &Hooks {
        &self.hooks
    }

    fn release_variables(&self) {
        // No saved tensors on AccumulateGrad. Default no-op suffices,
        // but we explicitly note it here so a future reader looking
        // for the saved-tensor release pattern doesn't think it was
        // forgotten.
    }

    fn as_any(&self) -> &dyn std::any::Any {
        // Override the default so the Phase 2 engine can recognize
        // leaf accumulators for the `with_inputs` grad-collection path.
        self
    }
}
