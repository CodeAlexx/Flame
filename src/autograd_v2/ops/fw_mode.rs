//! Phase 3c2 — shared helpers for per-op forward-mode AD (JVP).
//!
//! Every Phase 3a/3b op forward wrapper that supports forward-mode AD
//! follows the same skeleton:
//!
//! 1. Compute the primal output.
//! 2. Record the backward `*GradFn` (existing Phase 3a/3b code path).
//! 3. **NEW**: if any input has a `fw_grad` set, compute the JVP and
//!    install the resulting tangent on the output via
//!    `Tensor::set_fw_grad`.
//!
//! Helpers in this module factor the boilerplate so the per-op
//! changes are minimal:
//!
//! - [`any_fw_grad`] returns `true` iff at least one input tensor has
//!   a `fw_grad` installed. Mirrors the gating role of
//!   [`super::super::recording::needs_grad`] but for forward-mode AD.
//! - [`tangent_or_zero`] returns the input's `fw_grad` if present,
//!   otherwise a fresh zero tensor matching the input's shape /
//!   dtype / device. Used to fill in "no tangent ⇒ zero tangent" per
//!   the standard JVP convention.
//!
//! Per `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §clause 15 +
//! §Phase 3.

use crate::tensor::Tensor;
use crate::Result;

/// Returns `true` iff at least one of `inputs` has a forward-mode
/// tangent (`fw_grad`) installed. The gating predicate every op's
/// forward wrapper uses to skip JVP work when nothing upstream has a
/// tangent set.
#[inline]
pub fn any_fw_grad(inputs: &[&Tensor]) -> bool {
    inputs.iter().any(|t| t.fw_grad().is_some())
}

/// Return the tangent for `t`. If a tangent is installed, returns a
/// clone of it (cheap — storage Arc bump). Otherwise allocates a
/// fresh zero tensor with the same shape, dtype, and device.
///
/// JVP convention: an input without an explicit tangent contributes
/// zero to the output tangent. Materialising the zero is convenient
/// for downstream JVP formulas that need a real tensor to do math
/// against (e.g. `a_dot @ b + a @ b_dot` — when `b_dot` is None we
/// still need the second term to be a real all-zeros tensor with
/// matching shape).
#[inline]
pub fn tangent_or_zero(t: &Tensor) -> Result<Tensor> {
    if let Some(g) = t.fw_grad() {
        Ok(g)
    } else {
        t.zeros_like_with_dtype(t.dtype())
    }
}
