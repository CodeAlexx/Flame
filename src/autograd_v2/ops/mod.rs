//! Phase 3a v2 op modules.
//!
//! Per `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Phase 3 (3a subset):
//!
//! - Each op has its own file under `autograd_v2/ops/`.
//! - Each op carries a `*GradFn` struct + a `*_v2` forward wrapper.
//! - Forward wrappers call the existing flame-core math op for the
//!   forward computation, then conditionally record into v2 via
//!   [`super::recording::record_v2`].
//! - Recording is gated on `super::recording::needs_grad` — inference
//!   pays zero overhead.
//! - BF16/F32 dtype preservation end-to-end. Per Option A (see
//!   `docs/BF16_GRAD_DECISION.md`).
//! - SavedTensors carry an op-named string via `SavedTensor::save_named`
//!   so version-mismatch error messages identify the failing op.
//!
//! Phase 3a ops:
//! - [`add`] — pointwise add. Saves no tensors (backward: g, g).
//! - [`mul`] — pointwise multiply. Saves both inputs.
//! - [`sum`] — full-tensor reduce. Saves input shape only.
//! - [`matmul`] — 2D/N-D matrix multiply. Saves both inputs.
//! - [`silu`] — `x * sigmoid(x)` activation. Saves input.
//!
//! Phase 3b view ops (shape-only; no tensor data saved):
//! - [`reshape`] — reshape / view alias. Backward reshape-back.
//! - [`transpose`] — 2D transpose. Backward transposes + `.contiguous()`
//!   (HAZARD-2026-05-13-1 + gemm-stride-ignore).
//! - [`narrow`] — slice along a dim. Backward writes into a FRESH zero
//!   tensor via `narrow_backward_scatter_add_cuda`; NEVER mutates
//!   through a `narrow()` view back into a parent (HAZARD-2026-05-13-1).
//! - [`squeeze`] — remove a unit dim. Backward unsqueezes.
//! - [`unsqueeze`] — insert a unit dim. Backward squeezes.
//! - [`permute`] — N-D axis reorder. Backward applies the inverse
//!   permutation + `.contiguous()` (HAZARD-2026-05-13-1 + gemm-stride-
//!   ignore).
//!
//! Phase 3c adds layer_norm, CheckpointGradFn::apply, and per-op
//! forward-mode AD formulas.

pub mod add;
pub mod matmul;
pub mod mul;
pub mod narrow;
pub mod permute;
pub mod reshape;
pub mod silu;
pub mod squeeze;
pub mod sum;
pub mod transpose;
pub mod unsqueeze;
