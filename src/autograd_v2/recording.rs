//! Phase 3a recording surface — the API every Phase 3+ op uses.
//!
//! Per `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Phase 3 and the
//! Phase 3a port-agent prompt:
//!
//! - [`record_v2`] wires a freshly-built `GradFn` to one or more output
//!   tensors by allocating an `AutogradMetaRef` per output and stashing
//!   it in `Tensor::autograd_meta`.
//! - [`gradient_edge_for_tensor`] returns the `Edge` to backprop a
//!   gradient through when this tensor is consumed as an input by some
//!   downstream op. Wraps the existing `meta::gradient_edge` helper with
//!   a `Tensor`-level entry point.
//! - [`next_sequence_nr`] returns a monotonic counter for op creation
//!   order — every newly-built `GradFn` should call this once at
//!   construction. The engine breaks ready-queue ties on
//!   `sequence_nr`.
//! - [`needs_grad`] returns `true` iff at least one input tensor's
//!   autograd_meta has `requires_grad=true` — the gating predicate
//!   every op's forward wrapper uses to skip recording for inference.
//!
//! The recording surface is the highest-value piece of Phase 3a: every
//! Phase 3+ op forward wrapper (`add_v2`, `mul_v2`, `matmul_v2`,
//! `silu_v2`, `sum_v2`, plus all future ops) goes through this. There
//! is no global tape — the recording IS the per-output
//! `Tensor::autograd_meta` slot, populated by `record_v2`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::meta::{gradient_edge, new_meta_ref, AutogradMetaV2};
use super::node::{Edge, GradFn};
use crate::tensor::Tensor;

/// Monotonic op-creation-order counter. Used to break ties in the
/// engine's ready queue per PyTorch parity.
///
/// Phase 3a uses the same value for `sequence_nr` and `topological_nr`.
/// Phase 3b may refine the topological ordering for the view-autograd
/// pruning path; the simple identity is correct for math ops because
/// the BFS-derived dependency count already enforces the right firing
/// order.
pub fn next_sequence_nr() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Resolve the [`Edge`] that a downstream op should attach to in order
/// to backprop a gradient through `t`.
///
/// - `t.autograd_meta == None` → `Edge::null()` (no recording on this
///   input; the upstream op's grad is dropped at this slot).
/// - meta present but `requires_grad=false` AND `grad_fn=None` →
///   `Edge::null()`.
/// - Otherwise → delegates to [`meta::gradient_edge`], which lazily
///   materializes an `AccumulateGrad` for leaves with
///   `requires_grad=true` and uses the existing `grad_fn` for
///   non-leaves.
pub fn gradient_edge_for_tensor(t: &Tensor) -> Edge {
    let meta = match t.autograd_meta() {
        None => return Edge::null(),
        Some(m) => m,
    };

    // Fast-reject: leaf with neither requires_grad nor a grad_fn. We
    // need a separate read of the guard because `gradient_edge` itself
    // would take the lock; do a cheap up-front read so the common
    // inference path skips even the lock acquisition.
    {
        let m = meta
            .lock()
            .expect("autograd_v2: meta mutex poisoned in gradient_edge_for_tensor");
        if m.grad_fn.is_none() && !m.requires_grad {
            return Edge::null();
        }
    }

    gradient_edge(meta, next_sequence_nr())
}

/// Returns `true` iff at least one of `inputs` is being tracked through
/// v2 with `requires_grad=true` OR has a `grad_fn` attached (i.e. is a
/// downstream result of some recorded op).
///
/// Every Phase 3+ op forward wrapper should call this once and skip
/// recording entirely when it returns `false`. The inference path pays
/// zero overhead.
pub fn needs_grad(inputs: &[&Tensor]) -> bool {
    inputs.iter().any(|t| match t.autograd_meta() {
        None => false,
        Some(meta) => match meta.lock() {
            Ok(m) => m.requires_grad || m.grad_fn.is_some(),
            Err(_) => false,
        },
    })
}

/// Wire a fresh `grad_fn` to a list of `outputs`, allocating a new
/// `AutogradMetaRef` for each output and stashing it in
/// `output.autograd_meta`. `output_nr` matches the index in `outputs`.
///
/// Phase 3a contract:
/// - The returned `Vec<Tensor>` is the linked tensors (same vec, just
///   threaded through to give the call site a fluent shape).
/// - Each output's `autograd_meta` slot is `Some(_)` after this call.
/// - The shared `grad_fn` is `Arc`-cloned once per output for the
///   non-leaf meta; the strong count is `1 + outputs.len()`.
///
/// `ctx` is currently unused — kept in the signature so a Phase 4+
/// revisit can record per-stream/per-device hooks without a breaking
/// trait change.
pub fn record_v2(
    grad_fn: Arc<dyn GradFn>,
    mut outputs: Vec<Tensor>,
    _ctx: &super::dispatch::DispatchCtx,
) -> Vec<Tensor> {
    for (i, t) in outputs.iter_mut().enumerate() {
        let meta = new_meta_ref(AutogradMetaV2::non_leaf(grad_fn.clone(), i as u32));
        t.set_autograd_meta(Some(meta));
    }
    outputs
}
