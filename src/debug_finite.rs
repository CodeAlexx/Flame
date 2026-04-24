//! Lightweight NaN/Inf tripwire for training bring-up.
//!
//! Enable with `FLAME_DEBUG_FINITE=1`. Then call [`check`] at strategic
//! points during forward/backward. Every call logs the site name, tensor
//! shape, dtype, and finite range. The first call that observes a NaN or
//! Inf logs a red line and returns an error, so the stack unwinds at the
//! first bad value rather than cascading into downstream garbage.
//!
//! Performance: each `check` call triggers a D2H sync + copy. This is
//! deliberately expensive — the flag is meant for one-step repros where
//! you want to know *where* numerics went wrong, not a production path.
//! When `FLAME_DEBUG_FINITE` is unset the check is a single cached atomic
//! load.
//!
//! Typical use:
//! ```ignore
//! flame_core::debug_finite::reset();
//! let pred = model.forward(...)?;
//! flame_core::debug_finite::check("model.forward out", &pred)?;
//! let loss = mse_loss_bf16(&pred, &target)?;
//! flame_core::debug_finite::check("loss", &loss)?;
//! let grads = loss.backward()?;  // autograd.rs::backward inserts its
//!                                 // own per-op checks when enabled
//! ```
//!
//! Keep the site names short and stable — they appear in log output and
//! are grepped from debug sessions.

use crate::{Error, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

static ENABLED: OnceLock<bool> = OnceLock::new();
static FIRED: AtomicBool = AtomicBool::new(false);

/// Whether `FLAME_DEBUG_FINITE=1` is set for this process.
#[inline]
pub fn is_enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("FLAME_DEBUG_FINITE").ok().as_deref() == Some("1"))
}

/// Clear the "already fired" flag. Call at the top of each training step
/// so the tripwire rearms — otherwise a non-finite value on step 0 would
/// silently disable checks on step 1+.
pub fn reset() {
    FIRED.store(false, Ordering::Relaxed);
}

/// Check a tensor for NaN / ±Inf and log a finite-range summary. No-op
/// when `FLAME_DEBUG_FINITE` is unset or when the tripwire has already
/// fired this step.
///
/// On failure: logs a line tagged `[FLAME_DEBUG_FINITE]` and returns an
/// `Error::Training`. The caller's `?` unwinds and the failing site is
/// visible at the top of the resulting backtrace.
pub fn check(site: &str, t: &Tensor) -> Result<()> {
    if !is_enabled() || FIRED.load(Ordering::Relaxed) {
        return Ok(());
    }
    // `to_vec` upcasts to F32 on CPU via cast+D2H; works for every dtype
    // we actually produce in training. Slow, which is fine — see module
    // docs.
    let data = t.to_vec()?;
    let (nan, pos_inf, neg_inf, min, max) = scan(&data);
    if nan > 0 || pos_inf > 0 || neg_inf > 0 {
        FIRED.store(true, Ordering::Relaxed);
        eprintln!(
            "[FLAME_DEBUG_FINITE] ❌ {site}: shape={:?} dtype={:?} nan={nan} +inf={pos_inf} -inf={neg_inf} finite=[{:.3e}, {:.3e}]",
            t.shape().dims(),
            t.dtype(),
            min,
            max,
        );
        return Err(Error::Training(format!(
            "{site}: non-finite values (nan={nan} +inf={pos_inf} -inf={neg_inf})"
        )));
    }
    eprintln!(
        "[FLAME_DEBUG_FINITE] ✓ {site}: shape={:?} dtype={:?} range=[{:.3e}, {:.3e}]",
        t.shape().dims(),
        t.dtype(),
        min,
        max,
    );
    Ok(())
}

fn scan(data: &[f32]) -> (usize, usize, usize, f32, f32) {
    let mut nan = 0usize;
    let mut pos_inf = 0usize;
    let mut neg_inf = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in data {
        if v.is_nan() {
            nan += 1;
        } else if v == f32::INFINITY {
            pos_inf += 1;
        } else if v == f32::NEG_INFINITY {
            neg_inf += 1;
        } else {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
    }
    if min == f32::INFINITY {
        // All values were non-finite; report 0.0 to avoid confusing log output.
        min = 0.0;
        max = 0.0;
    }
    (nan, pos_inf, neg_inf, min, max)
}
