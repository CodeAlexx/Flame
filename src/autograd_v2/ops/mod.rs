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
//! Phase 3c1 added layer_norm + CheckpointGradFn::apply.
//!
//! Phase 3c2 adds per-op forward-mode AD (JVP) formulas across the
//! 11 non-LN ops. [`fw_mode`] hosts the shared helpers (`any_fw_grad`,
//! `tangent_or_zero`). Each op's forward wrapper now computes the
//! JVP when at least one input has a `fw_grad` set, installs the
//! tangent on the output via `Tensor::set_fw_grad`. JVP formulas:
//!
//! - Linear ops (`add`, `sum`, `silu` (in its `g * silu'(x)` shape),
//!   reshape/view/transpose/narrow/squeeze/unsqueeze/permute): apply
//!   the same forward op to the tangent.
//! - Product-rule ops (`mul`, `matmul`): `out_fw = a_dot*b + a*b_dot`.
//!
//! `transpose` and `permute` forward-mode tangents are `.contiguous()`-
//! materialised before storage, per HAZARD-2026-05-13-1 +
//! gemm-stride-ignore (same discipline as their backward formulas).
//!
//! `layer_norm` forward-mode AD is DEFERRED to Phase 5 or post-v2
//! polish. The JVP formula
//! `out_fw = (x_fw - mean_fw)*rstd*w + x_hat*d_rstd_dx*w + x_hat*rstd*w_fw + b_fw`
//! is mechanical but non-trivial; the Phase 5 parity gate will
//! validate it against `torch.autograd.functional.jvp(layer_norm, ...)`
//! once the parity harness ships. Phase 3c2 records `LayerNormGradFn`
//! for backward only; an input tangent on `x`, `weight`, or `bias`
//! is silently ignored (no JVP install on the output) — matches
//! "tangent vector defaults to zero" semantics if the formula were
//! not implemented.

pub mod add;
pub mod fw_mode;
pub mod layer_norm;
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
