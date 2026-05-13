//! `NodeId`, `Edge`, `GradFn` — the node-graph API.
//!
//! Per §6 and recommended-change 6 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`:
//!
//! - `NodeId` is a monotonic atomic counter (cheap engine bookkeeping).
//! - `Edge` carries a `(function, input_nr)` pair like PyTorch's
//!   `torch::autograd::Edge`.
//! - `GradFn::num_inputs()` is mandatory — the engine sizes downstream
//!   `InputBuffer`s from it.
//! - `apply()` returns `Result<Vec<Option<Tensor>>, AutogradV2Error>`
//!   so version-mismatch / released-saved-tensor surface as recoverable
//!   training errors.
//! - `release_variables(&self)` clears saved tensors via interior
//!   mutability (no `&mut self`).
//! - `sequence_nr()` and `topological_nr()` are **stored fields** on the
//!   concrete impl, not recomputed walks (PyTorch parity).
//! - `hooks() -> &Hooks` default returns the empty sentinel — Phase 2
//!   wires dispatch.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::dispatch::DispatchCtx;
use super::error::AutogradV2Error;
use super::hooks::Hooks;
use crate::tensor::Tensor;

/// Monotonic node identifier. Cheap to construct (single relaxed atomic
/// fetch_add). Used by the engine for ready-queue keying and by hooks
/// for per-node dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// `(function, input_nr)` pair. `function` is `None` for an edge that
/// terminates at a non-leaf tensor whose grad is not retained (the
/// engine drops the gradient instead of forwarding it).
#[derive(Clone)]
pub struct Edge {
    pub function: Option<Arc<dyn GradFn>>,
    /// Which input slot of `function` this gradient feeds into.
    pub input_nr: u32,
}

impl Edge {
    pub fn new(function: Arc<dyn GradFn>, input_nr: u32) -> Self {
        Self {
            function: Some(function),
            input_nr,
        }
    }

    pub fn null() -> Self {
        Self {
            function: None,
            input_nr: 0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.function.is_some()
    }
}

impl std::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Edge")
            .field(
                "function",
                &self
                    .function
                    .as_ref()
                    .map(|fun| fun.name())
                    .unwrap_or("<null>"),
            )
            .field("input_nr", &self.input_nr)
            .finish()
    }
}

/// The autograd v2 backward-node trait. One impl per op.
///
/// Phase 1 ships the trait contract; the engine that drives it is
/// Phase 2. Phase 3 fills the long-tail per-op impls.
pub trait GradFn: Send + Sync + std::fmt::Debug {
    /// Compute the gradients flowing *into* this node's inputs given
    /// the gradients flowing *out* of its outputs.
    ///
    /// `grad_outputs.len() == self.num_outputs_known()` (Phase 2 will
    /// formalize a `num_outputs()` accessor; Phase 1's contract leaves
    /// it to the concrete impl).
    ///
    /// Returns one `Option<Tensor>` per input slot — `None` means "no
    /// gradient to propagate to this slot" (e.g. dtype is not
    /// differentiable, or the slot is a non-tensor argument).
    ///
    /// Errors (`AutogradV2Error::VersionMismatch`,
    /// `SavedTensorReleased`, `DtypeMismatch`, `NotImplementedYet`)
    /// bubble up through `Engine::execute` as recoverable training
    /// failures.
    ///
    /// `ctx` carries the active device + stream (per §16 / charter
    /// decision 2026-05-13). Phase 1's only device today is the default
    /// stream of the global CUDA device; the parameter is here so
    /// Phase 2+ can dispatch across multiple streams/devices without
    /// changing the trait signature.
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error>;

    /// Number of input slots this node consumes (= the size of the
    /// `InputBuffer` the engine allocates for its predecessors to
    /// write into). Required by §6.
    fn num_inputs(&self) -> usize;

    /// Edges out of this node (one per input slot in
    /// PyTorch-equivalent order — `next_edges()[i]` is the edge for the
    /// grad routed back from input `i`). Phase 2's dependency counter
    /// walks these.
    fn next_edges(&self) -> &[Edge];

    /// Stored creation-order sequence number. Used by Phase 2's
    /// engine to deterministically break ties in the ready queue.
    fn sequence_nr(&self) -> u64;

    /// Stored topological number. Set by graph construction in Phase
    /// 2 (NOT recomputed). Used for DAG pruning when only a subset
    /// of outputs is asked for via `grad(...)`.
    fn topological_nr(&self) -> u64;

    /// Stable node identity. Used by hook dispatch and engine
    /// bookkeeping.
    fn node_id(&self) -> NodeId;

    /// Human-readable name for diagnostics and hook dispatch.
    fn name(&self) -> &'static str;

    /// Hook bundle. Default returns the empty sentinel — the no-hook
    /// fast path is a `Hooks::empty_ref()` pointer comparison.
    ///
    /// **DO NOT** override this method unless your op actually carries
    /// hooks. Overriding to return a non-sentinel `Hooks` reference
    /// defeats the engine's pointer-compare fast-path (`std::ptr::eq`
    /// in `engine::execute`) and forces the empty `for`-loops to run
    /// for every backward step. Carry an `Option<Hooks>` slot on the
    /// concrete op and only override when the slot is populated, if
    /// you must support per-op hook registration without the fast-path
    /// cost.
    fn hooks(&self) -> &Hooks {
        Hooks::empty_ref()
    }

    /// Drop saved tensors. The trait takes `&self` (per §3) so a
    /// shared `Arc<dyn GradFn>` can release without `&mut`. Concrete
    /// impls store saved tensors behind interior mutability
    /// (`SavedTensor::reset()` is `&self`).
    ///
    /// Default no-op; ops with no saved tensors don't override.
    fn release_variables(&self) {}

    /// Type-erased downcast helper for engine internals. The default
    /// impl returns `None`; concrete ops that the engine needs to
    /// downcast to (today: `AccumulateGrad` for the `with_inputs` leaf
    /// grad collection path) override to `Some(self)`.
    ///
    /// We add this rather than a blanket `Any` supertrait because
    /// `dyn GradFn` is held via `Arc<dyn GradFn>` and adding `Any`
    /// would force every `Arc` cast site to know about it. A scoped
    /// `as_any(&self) -> &dyn Any` is the minimal surface that lets
    /// the engine do safe downcast checks without polluting the trait
    /// bounds.
    fn as_any(&self) -> &dyn std::any::Any {
        // Default: no downcast. Concrete impls that the engine needs
        // to recognize (AccumulateGrad) override this.
        //
        // SAFETY: returning a reference to `self as &dyn Any` is sound
        // because every concrete `GradFn` impl is `Sized + 'static`
        // (the trait bound `Send + Sync + Debug` is satisfied by
        // Phase 2's hand-rolled ops, all of which are `'static`).
        //
        // To make this default safe we cannot return `self` here — `self`
        // is `&Self`, and `Self: ?Sized` at the trait declaration. We
        // return a static no-op marker; impls that want downcast
        // override.
        static MARKER: &str = "<no-downcast>";
        &MARKER
    }
}
