//! `AutogradMetaV2` — shared, interior-mutable autograd metadata.
//!
//! Per §2 and recommended-change 2 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`:
//!
//! - Stored as `Arc<Mutex<AutogradMetaV2>>` so a tensor's metadata can
//!   be shared across clones (PyTorch `.clone()` semantics — `clone()`
//!   is differentiable).
//! - `grad_accumulator` is a **Weak** pointer to break the
//!   AccumulateGrad ↔ tensor cycle (§1).
//! - `grad_fn` is **Strong** (non-leaf node ownership flows through the
//!   tensor; the engine drops it when the tensor is dropped).
//! - `is_view` + view-base slot are reserved for Phase 3 view-autograd;
//!   Phase 1 just leaves the fields present.
//! - `hooks` defaults to the empty sentinel.
//!
//! Tensor::clone() semantics (documented here because Phase 1 ships the
//! contract, even though `Tensor` doesn't yet carry an
//! `Arc<Mutex<AutogradMetaV2>>` field — that wiring is part of the
//! Phase 3 op-migration plan):
//!
//! - `.clone()` SHALL preserve `Arc<Mutex<AutogradMetaV2>>` by cloning
//!   the Arc (cheap — single ref bump). The two tensor handles share
//!   one set of autograd metadata: gradient accumulates into the same
//!   slot, `grad_fn` is the same node.
//! - `.detach()` SHALL allocate a fresh `AutogradMetaV2` with
//!   `requires_grad=false` and `grad_fn=None`. The detached handle is
//!   no longer connected to the source graph.
//! - `to_dtype()` / `contiguous()` are differentiable; they record as
//!   `GradFn`s and the result tensor allocates fresh metadata pointing
//!   at the new node.

use std::sync::{Arc, Mutex, Weak};

use super::hooks::Hooks;
use super::node::GradFn;
use crate::tensor::Tensor;

/// Shared autograd metadata for a tensor. Wrap in
/// `Arc<Mutex<AutogradMetaV2>>` at the tensor field (the wrapping is
/// the `AutogradMetaRef` type alias).
pub struct AutogradMetaV2 {
    /// Accumulated gradient for this tensor. Populated by the engine
    /// at backward time for leaf tensors with `requires_grad=true` and
    /// by user-installed retain_grad hooks for non-leaf tensors.
    pub grad: Option<Tensor>,

    /// The backward node that produced this tensor. `None` for leaves
    /// (the AccumulateGrad node is lazily materialized via the
    /// `grad_accumulator` weak slot below).
    pub grad_fn: Option<Arc<dyn GradFn>>,

    /// Weak reference to the leaf accumulator for this tensor. PyTorch
    /// uses a weak slot so the AccumulateGrad ↔ tensor cycle (each
    /// holds a strong ref to the other) is broken — see §1.
    /// `gradient_edge()` upgrades on demand; if the upgrade fails, a
    /// fresh AccumulateGrad is allocated and stored back here.
    pub grad_accumulator: Weak<dyn GradFn>,

    /// Which output index of `grad_fn` this tensor is. Multi-output
    /// nodes (e.g. `split`, `chunk`) need this to disambiguate which
    /// downstream gradient slot to feed.
    pub output_nr: u32,

    /// Set by the forward pass when a tensor is built from one or more
    /// trainable inputs. Drives whether grad_fn / grad_accumulator
    /// materialization happens.
    pub requires_grad: bool,

    /// Phase 3 view-autograd hook. `true` iff the tensor was produced
    /// by a view op (e.g. `view`, `narrow`, `transpose`). Phase 1
    /// ships the slot; Phase 3 wires the per-op formulas.
    pub is_view: bool,

    /// Weak reference back to the base tensor's autograd meta, for
    /// view chains. Phase 1 reserves the slot; Phase 3 populates it.
    pub view_base: Option<Weak<Mutex<AutogradMetaV2>>>,

    /// Hook bundle. Default empty.
    pub hooks: Hooks,
}

impl AutogradMetaV2 {
    /// Build a fresh meta for a leaf with `requires_grad = false`. The
    /// most common case: a constant or a tensor explicitly detached
    /// from the graph.
    pub fn leaf_no_grad() -> Self {
        Self {
            grad: None,
            grad_fn: None,
            grad_accumulator: WEAK_SENTINEL.with(|w| w.clone()),
            output_nr: 0,
            requires_grad: false,
            is_view: false,
            view_base: None,
            hooks: Hooks::default(),
        }
    }

    /// Build a fresh meta for a leaf with `requires_grad = true`. The
    /// `grad_accumulator` is left empty (a `Weak::new()`-style
    /// sentinel); the engine materializes it on first
    /// `gradient_edge()` call.
    pub fn leaf_requires_grad() -> Self {
        Self {
            grad: None,
            grad_fn: None,
            grad_accumulator: WEAK_SENTINEL.with(|w| w.clone()),
            output_nr: 0,
            requires_grad: true,
            is_view: false,
            view_base: None,
            hooks: Hooks::default(),
        }
    }

    /// Build a fresh meta for a non-leaf tensor produced by some
    /// `grad_fn` at output index `output_nr`.
    pub fn non_leaf(grad_fn: Arc<dyn GradFn>, output_nr: u32) -> Self {
        Self {
            grad: None,
            grad_fn: Some(grad_fn),
            grad_accumulator: WEAK_SENTINEL.with(|w| w.clone()),
            output_nr,
            requires_grad: true,
            is_view: false,
            view_base: None,
            hooks: Hooks::default(),
        }
    }

    /// True iff this meta represents a leaf (`grad_fn == None`). Note
    /// that `requires_grad == false` leaves are also "leaves" in this
    /// sense — PyTorch reserves the term for any tensor without an
    /// outgoing backward node.
    pub fn is_leaf(&self) -> bool {
        self.grad_fn.is_none()
    }
}

thread_local! {
    /// A `Weak<dyn GradFn>` whose strong count is permanently zero —
    /// used as the initial `grad_accumulator` for a freshly-built
    /// `AutogradMetaV2`. `Weak::new()` doesn't exist for `dyn Trait`
    /// without an explicit type hint; we materialize one by downgrading
    /// a never-stored Arc to a stub impl.
    ///
    /// The stub never actually executes — `Weak::upgrade()` on this
    /// will always return `None`, which is the intended sentinel
    /// semantic. The `Arc` itself is dropped immediately after the
    /// `Weak::downgrade()`, so the strong count goes to 0 right away.
    static WEAK_SENTINEL: Weak<dyn GradFn> = {
        let stub: Arc<dyn GradFn> = Arc::new(SentinelGradFn);
        Arc::downgrade(&stub)
    };
}

/// Stub `GradFn` used only to construct the `WEAK_SENTINEL`. Never
/// reachable from a live graph — the only `Arc` of it is dropped
/// inside the thread-local initializer.
#[derive(Debug)]
struct SentinelGradFn;

impl GradFn for SentinelGradFn {
    fn apply(
        &self,
        _grad_outputs: Vec<Option<Tensor>>,
        _ctx: &super::dispatch::DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, super::error::AutogradV2Error> {
        Err(super::error::AutogradV2Error::NotImplementedYet(
            "SentinelGradFn — should not be reachable",
        ))
    }
    fn num_inputs(&self) -> usize {
        0
    }
    fn next_edges(&self) -> &[super::node::Edge] {
        &[]
    }
    fn sequence_nr(&self) -> u64 {
        0
    }
    fn topological_nr(&self) -> u64 {
        0
    }
    fn node_id(&self) -> super::node::NodeId {
        super::node::NodeId(u64::MAX)
    }
    fn name(&self) -> &'static str {
        "<sentinel>"
    }
}

/// Shared, mutable autograd metadata handle. Cheap to clone (single
/// Arc bump). The Mutex is contended only at metadata-write points
/// (grad accumulation, grad_fn install) and at view-base linkage —
/// not on the hot tensor-op path.
pub type AutogradMetaRef = Arc<Mutex<AutogradMetaV2>>;

/// Wrap an `AutogradMetaV2` in the canonical `Arc<Mutex<_>>` shape.
pub fn new_meta_ref(meta: AutogradMetaV2) -> AutogradMetaRef {
    Arc::new(Mutex::new(meta))
}
