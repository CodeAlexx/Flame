//! Autograd v2 — clean-sheet PyTorch-style DAG autograd engine.
//!
//! **Status**: Phases 1 + 2 shipped. No live behavior change in the
//! rest of the crate — no forward op records through v2 until Phase 3
//! wires forward primitives through `gradient_edge()`. The engine
//! driver, ready queue, hook dispatch, and AccumulateGrad sink are
//! all functional and tested via synthetic test-only GradFn impls
//! (see `tests/autograd_v2_engine.rs`). Both the `autograd_v2`
//! feature build and the default build (without the feature) are green.
//!
//! ## Phase roadmap
//!
//! - **Phase 0 (shipped)** — feature flag + empty module + in-place
//!   version-bump audit + `BF16_GRAD_DECISION.md`.
//! - **Phase 1 (this file)** — metadata, traits, saved tensor, input
//!   buffer, hooks, accumulator skeleton, multi-device dispatch
//!   surface.
//! - **Phase 2** — engine skeleton (GraphRoot, dependency counting,
//!   ready queue), nested execute, hook dispatch.
//! - **Phase 3** — first real ops (add, mul, matmul, ...) + view
//!   backward + forward-mode AD per-op formulas.
//! - **Phase 4** — optimizer + trainer integration (BF16
//!   end-to-end for BF16 params — Class A path).
//! - **Phase 5** — parity gate before retiring v1.
//!
//! See `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` for the design
//! review and recommended-change list, and
//! `docs/BF16_GRAD_DECISION.md` for the Phase-4 dtype policy.
//!
//! ## Phase 1 design highlights
//!
//! - **Weak accumulator**: `AutogradMetaV2::grad_accumulator` and
//!   `AccumulateGrad::variable` are both `Weak`, breaking the cycle
//!   that would otherwise leak the leaf tensor's storage.
//! - **`Result`-typed apply**: `GradFn::apply` returns
//!   `Result<Vec<Option<Tensor>>, AutogradV2Error>`. Version
//!   mismatches and released saved tensors are recoverable training
//!   errors, not panics.
//! - **Version-handle saved tensors**: `SavedTensor` stores an
//!   `Arc<AtomicU32>` clone of the storage's version counter; the
//!   check at `unpack()` survives `AutogradContext::clear()`.
//! - **&self release**: `GradFn::release_variables(&self)` works
//!   without `&mut` because `SavedTensor` carries interior
//!   mutability on its data and `fw_grad_` slots.
//! - **In-place AND out-of-place accumulation**: `InputBuffer::add`
//!   picks the in-place path when `create_graph=false` AND the
//!   conservative predicate holds; otherwise it falls back to an
//!   out-of-place `+` that future Phase 3 ops will re-record under
//!   autograd_v2.
//! - **Multi-device surface**: `DispatchCtx { stream: DeviceStream }`
//!   is passed by `&` to every dispatch entry point so Phase 2+ can
//!   route on non-default streams or across devices without a
//!   breaking trait change. Today's only ctx is the default stream
//!   of the global CUDA device.
//! - **Forward-mode AD plumbing**: `SavedTensor::fw_grad_` slot is
//!   present per §clause 15; Phase 3 ops populate it.
//! - **Hooks surface**: `Hooks` struct + `GradFn::hooks()` default
//!   returns the empty sentinel. Phase 2 wires dispatch.
//!
//! ## Tensor::clone() / detach() contract (documented for Phase 3
//! op-migration)
//!
//! - `.clone()` SHALL preserve the `Arc<Mutex<AutogradMetaV2>>` by
//!   cloning the Arc — both handles share metadata, gradients flow
//!   into the same slot, `grad_fn` is the same node.
//! - `.detach()` SHALL allocate a fresh `AutogradMetaV2` with
//!   `requires_grad=false`. The detached handle is disconnected from
//!   the source graph.
//!
//! `Tensor` does not yet carry an `Arc<Mutex<AutogradMetaV2>>` field
//! — that wiring is part of Phase 3's op-migration plan. Phase 1
//! ships the contract; Phase 3 enforces it at the tensor level.

// `AutogradV2Error::FlameCore` wraps the crate's `Error` enum
// (~136 bytes) so version-mismatch / released-saved-tensor errors
// surface through the wider flame-core `Result<T, Error>` story
// (per §3 of `AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`). Boxing the
// wrapped error would mean every error allocates; the existing
// `Result<T, Error>` paths in flame-core don't box, and matching the
// crate idiom is correct here. The clippy lint flags it everywhere
// `Result<_, AutogradV2Error>` appears in this module — silence it
// at the module level.
#![allow(clippy::result_large_err)]

mod accumulator;
mod checkpoint;
mod dispatch;
mod engine;
mod error;
mod hooks;
mod input_buffer;
mod meta;
mod node;
pub mod ops;
mod optim;
mod recording;
mod saved_tensor;

pub use accumulator::AccumulateGrad;
pub use checkpoint::{checkpoint_v2, CheckpointForwardFn, CheckpointGradFn};
pub use dispatch::{DeviceStream, DispatchCtx};
pub use engine::{Engine, GraphRoot};
pub use error::{AutogradV2Error, V2Result};
pub use hooks::{Hooks, PostBackwardHook, PreBackwardHook, TensorHook};
pub use input_buffer::InputBuffer;
pub use meta::{gradient_edge, new_meta_ref, AutogradMetaRef, AutogradMetaV2};
pub use node::{Edge, GradFn, NodeId};
pub use optim::{set_param_grad_v2, AdamWV2, OptimizerV2};
pub use recording::{gradient_edge_for_tensor, needs_grad, next_sequence_nr, record_v2};
pub use saved_tensor::SavedTensor;
