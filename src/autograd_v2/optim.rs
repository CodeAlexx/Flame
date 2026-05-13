//! Autograd v2 optimizer surface.
//!
//! Phase 4a deliverable. Thin v2-facing wrapper over the existing
//! fused-Adam machinery in `src/adam.rs`. The internal kernels
//! (`adam_fused_step`, `adam_fused_step_f32`, `adam_fused_multi_tensor_step`)
//! already accept BF16 grads; the only piece preventing BF16 grads from
//! flowing through end-to-end was that `Parameter::set_grad` cast
//! everything to F32 before the optimizer saw it. Phase 4a fixed that
//! via [`crate::parameter::GradDtypePolicy::MatchParamDtype`] +
//! [`crate::Parameter::new_v2`].
//!
//! This module adds the OptimizerV2 trait + `AdamWV2` wrapper as the
//! public v2 surface. Internally `AdamWV2` delegates to `AdamW` — same
//! state shape, same kernels, same checkpoint format. The v2 wrapper
//! exists to give trainers a clear "this is the autograd v2 optimizer
//! entry point" import path without re-plumbing the optimizer state
//! machinery.
//!
//! ## What's NOT here (Phase 4b)
//!
//! - `GradientMap` rewrite to a `MatchParamDtype` policy variant —
//!   v2 grads land in `AutogradMetaV2::grad` (per
//!   `src/autograd_v2/accumulator.rs:160-209`), bypassing GradientMap
//!   entirely on the recording path. The GradientMap path is still v3
//!   and serves trainers that use the old `backward()` collection API.
//!   When Phase 4b introduces v2 trainer integration we'll either add
//!   the policy variant or settle on routing v2 grads through the
//!   `meta.grad` slot exclusively.
//! - Trainer integration smoke. Z-Image LoRA is the planned target;
//!   Klein crashes on the current box per Phase 0 skeptic notes.

use crate::{Error, Parameter, Result};

use super::dispatch::DispatchCtx;
use super::error::AutogradV2Error;

/// Autograd v2 optimizer trait.
///
/// `step(params, ctx)` pulls each parameter's current grad off the
/// `Parameter::grad_bf16_or_f32` accessor (the v2-aware native-dtype
/// reader) and routes it through a fused kernel. The caller is
/// responsible for `zero_grad` between steps.
///
/// The trait is intentionally minimal — Phase 4a; Phase 4b can extend
/// it once trainer integration shows what shape callers actually need
/// (loss-scaler hooks, lr schedule per-step, m/v inspection, …).
pub trait OptimizerV2 {
    /// Apply one optimizer step.
    ///
    /// Reads grads off each parameter's native-dtype accessor and
    /// writes the param tensor in-place. `ctx` is the v2 dispatch
    /// context; today it carries only the default-stream descriptor.
    fn step(
        &mut self,
        params: &[Parameter],
        ctx: &DispatchCtx,
    ) -> std::result::Result<(), AutogradV2Error>;

    /// Clear gradients on every supplied parameter.
    fn zero_grad(&mut self, params: &[Parameter]);
}

/// AdamW wrapper for autograd v2 callers.
///
/// Internally delegates to [`crate::adam::AdamW`]. Construct with the same
/// hyperparameter set; the wrapper just routes through the
/// [`OptimizerV2`] trait surface.
pub struct AdamWV2 {
    inner: crate::adam::AdamW,
}

impl AdamWV2 {
    /// Construct with the standard AdamW hyperparameters. Same defaults
    /// as PyTorch: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
    /// weight_decay=0.0.
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            inner: crate::adam::AdamW::new(lr, beta1, beta2, eps, weight_decay),
        }
    }

    /// Update the learning rate used for subsequent steps. Same
    /// semantics as `AdamW::set_lr`.
    pub fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    /// Toggle stochastic BF16-rounding on the param store. Affects the
    /// `(BF16 param, F32 grad)` fused step only; the `(BF16 param, BF16
    /// grad)` kernel currently uses round-to-nearest (see
    /// `adam.rs::adam_fused_multi_tensor_step` dispatch matrix).
    pub fn set_stochastic_round(&mut self, on: bool) {
        self.inner.set_stochastic_round(on);
    }

    /// Borrow the inner AdamW for checkpoint accessors that v2 doesn't
    /// re-expose yet (state_for, set_state, t, lr, …). Phase 4a doesn't
    /// re-plumb the entire checkpoint surface through the v2 wrapper —
    /// callers go through `inner_adamw()` when they need it.
    pub fn inner_adamw(&self) -> &crate::adam::AdamW {
        &self.inner
    }

    /// Mutable borrow of the inner AdamW. Same rationale as
    /// [`AdamWV2::inner_adamw`].
    pub fn inner_adamw_mut(&mut self) -> &mut crate::adam::AdamW {
        &mut self.inner
    }
}

impl OptimizerV2 for AdamWV2 {
    fn step(
        &mut self,
        params: &[Parameter],
        _ctx: &DispatchCtx,
    ) -> std::result::Result<(), AutogradV2Error> {
        // `AdamW::step` reads grads from `Parameter::grad()`, which under
        // `GradDtypePolicy::MatchParamDtype` returns the native-dtype
        // grad set by the v2 caller. The fused Adam classifier in
        // `Adam::step` then selects the appropriate kernel via the
        // (param_dtype, grad_dtype) tuple — including the BF16-grad
        // arms activated in Phase 4a.
        self.inner.step(params).map_err(AutogradV2Error::FlameCore)
    }

    fn zero_grad(&mut self, params: &[Parameter]) {
        self.inner.zero_grad(params);
    }
}

/// Set a tensor's gradient on its parameter, going through the v2
/// dtype-preserving path.
///
/// Convenience for trainers that pull grads off
/// `AutogradMetaV2::grad` after backward and want to feed them into
/// the parameter slot without thinking about the dtype contract.
/// Equivalent to calling `param.set_grad(grad)` when `param` was
/// constructed via `Parameter::new_v2` (or had its policy set to
/// `MatchParamDtype`).
///
/// Returns `Err(InvalidOperation)` if the parameter is on the v1
/// `CastToF32` policy — preserves the explicit-policy contract.
pub fn set_param_grad_v2(param: &Parameter, grad: crate::Tensor) -> Result<()> {
    use crate::parameter::GradDtypePolicy;
    if param.grad_dtype_policy() != GradDtypePolicy::MatchParamDtype {
        return Err(Error::InvalidOperation(
            "set_param_grad_v2: parameter is on CastToF32 policy; \
             construct via Parameter::new_v2 or set_grad_dtype_policy(MatchParamDtype)"
                .into(),
        ));
    }
    param.set_grad(grad)
}
