//! Runtime diagnostics for trainers.
//!
//! Helpers that catch silent failure modes the autograd graph would
//! otherwise hide.  The recurring bug class on this stack:
//!
//! > A BF16 fused inference op gets used in a trainer's forward path
//! > without autograd registration → no tape edge from the LoRA-B
//! > parameter back to the loss → backward never reaches it →
//! > `grad == None` (or `grad.abs().sum() == 0`) → the optimizer takes
//! > zero updates on that parameter → training silently produces a LoRA
//! > full of zeros at most adapter sites.
//!
//! Memory entry: `feedback_flame_core_bf16_fused_autograd.md`.  Klein,
//! Z-Image, Chroma, and `rope_fused_bf16` all hit this same bug; in each
//! case we trained for thousands of steps before noticing the LoRA was
//! useless.
//!
//! [`check_grad_flow`] / [`assert_grad_flow`] iterate `(name, parameter)`
//! pairs after `loss.backward()?` and report any param whose gradient is
//! missing or zero.  Intended to be called once at step 0 (or step 1)
//! per training run.

use crate::env_flags;
use crate::gradient::GradientMap;
use crate::parameter::Parameter;
use crate::{DType, Result, Tensor};

/// Outcome of a [`check_grad_flow`] pass.
///
/// A parameter is "dead" if either:
/// - it has no gradient entry in the [`GradientMap`] (the autograd op
///   that consumes it was never recorded), or
/// - its gradient is present but `abs().sum()` evaluates to zero or a
///   non-finite value (the op is recorded but the path through it
///   produces a zero / NaN gradient — typically a dtype-cast or shape-
///   mismatch path that silently emits a zero tensor).
#[derive(Debug, Clone, Default)]
pub struct GradFlowReport {
    /// Names of parameters with no entry in the GradientMap.
    pub missing: Vec<String>,
    /// Names of parameters whose gradient sums to zero or non-finite.
    pub zero: Vec<String>,
    /// Count of healthy parameters; names elided to keep reports compact
    /// when there are hundreds.
    pub ok_count: usize,
}

impl GradFlowReport {
    /// Total number of dead parameters (missing + zero).
    pub fn dead_count(&self) -> usize {
        self.missing.len() + self.zero.len()
    }

    /// True when every checked parameter has a non-zero finite gradient.
    pub fn is_clean(&self) -> bool {
        self.dead_count() == 0
    }

    /// Multi-line summary suitable for `log::warn!` or panic messages.
    pub fn summary(&self) -> String {
        if self.is_clean() {
            return format!("[grad-flow] clean ({} params)", self.ok_count);
        }
        let mut out = format!(
            "[grad-flow] {} dead / {} ok\n",
            self.dead_count(),
            self.ok_count,
        );
        if !self.missing.is_empty() {
            out.push_str("  missing-grad (op never recorded):\n");
            for n in &self.missing {
                out.push_str(&format!("    - {n}\n"));
            }
        }
        if !self.zero.is_empty() {
            out.push_str("  zero-grad (recorded but identically zero / non-finite):\n");
            for n in &self.zero {
                out.push_str(&format!("    - {n}\n"));
            }
        }
        out
    }

    /// Panic with [`summary`] if any parameter is dead.  No-op when clean.
    ///
    /// Use this when you want hard-fail-fast semantics.  Use
    /// [`assert_grad_flow`] instead if you want the panic gated by
    /// `FLAME_ASSERT_GRAD_FLOW=1`.
    pub fn assert_clean(&self) {
        if !self.is_clean() {
            panic!("{}", self.summary());
        }
    }
}

/// Compute `grad.abs().sum()` for each `(name, &Parameter)` pair and
/// return a [`GradFlowReport`] naming any dead parameter.
///
/// `grads` is the [`GradientMap`] returned by `tensor.backward()`.
/// `named_params` is a slice of `(display_name, &Parameter)` pairs;
/// most trainers can derive this from
/// `model.bundle.parameter_names().iter().zip(model.parameters().iter())`.
///
/// Cost is O(P × G) where P is parameter count and G is per-param grad
/// size — for a typical LoRA training run (~1500 params × rank-16
/// matrices) this is well under 100 ms on a 3090.  Do NOT call every
/// step; call once at step 0 or step 1.
///
/// The function never panics or aborts; callers control the response.
/// See [`assert_grad_flow`] for the env-flag-gated panicking variant.
pub fn check_grad_flow<S: AsRef<str>>(
    grads: &GradientMap,
    named_params: &[(S, &Parameter)],
) -> Result<GradFlowReport> {
    let mut report = GradFlowReport::default();
    for (name, param) in named_params {
        let id = param.id();
        match grads.get(id) {
            None => report.missing.push(name.as_ref().to_string()),
            Some(g) => {
                if grad_is_dead(g)? {
                    report.zero.push(name.as_ref().to_string());
                } else {
                    report.ok_count += 1;
                }
            }
        }
    }
    Ok(report)
}

/// Variant of [`check_grad_flow`] that consults
/// [`env_flags::assert_grad_flow_enabled`] (`FLAME_ASSERT_GRAD_FLOW=1`)
/// and panics with the report when both:
/// - the env flag is set, and
/// - the report is not clean.
///
/// When the flag is unset the function still returns the report so the
/// caller may log it at `info`/`warn` level; this lets you keep the
/// instrumentation in production trainers without making CI brittle.
///
/// Recommended insertion point: immediately after `let grads =
/// loss.backward()?;`, before any grad-clip / optimizer step.
pub fn assert_grad_flow<S: AsRef<str>>(
    grads: &GradientMap,
    named_params: &[(S, &Parameter)],
) -> Result<GradFlowReport> {
    let report = check_grad_flow(grads, named_params)?;
    if env_flags::assert_grad_flow_enabled() && !report.is_clean() {
        panic!("{}", report.summary());
    }
    Ok(report)
}

fn grad_is_dead(g: &Tensor) -> Result<bool> {
    // Compute abs-sum in F32 regardless of grad dtype.  GradientMap
    // stores grads as F32 by contract (see gradient.rs:184), but we
    // cast defensively in case a future caller wires a BF16 grad in.
    let g_f32 = if g.dtype() == DType::F32 {
        g.clone()
    } else {
        g.to_dtype(DType::F32)?
    };
    let s = g_f32.abs()?.sum_all()?.to_vec()?;
    let v = s.first().copied().unwrap_or(0.0);
    Ok(!v.is_finite() || v == 0.0)
}
