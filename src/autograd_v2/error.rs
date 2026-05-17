//! Autograd v2 error type.
//!
//! All `apply()` / `unpack()` / `InputBuffer::add` operations return
//! `Result<_, AutogradV2Error>` so that version mismatches, released
//! saved tensors, and dtype mismatches are recoverable training errors
//! instead of panics. See
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §3.
//!
//! `AutogradV2Error` can be converted into the crate's top-level
//! `Error` enum via `From<AutogradV2Error> for Error`, mapping each
//! variant to `Error::Autograd(_)` with a structured message.

use crate::dtype::DType;
use crate::error::Error;

#[derive(Debug, thiserror::Error)]
pub enum AutogradV2Error {
    /// Saved tensor's underlying storage was mutated in-place after save.
    #[error(
        "autograd_v2: saved tensor version mismatch in {op}: expected v{expected}, got v{actual}"
    )]
    VersionMismatch {
        op: &'static str,
        expected: u32,
        actual: u32,
    },

    /// `release_variables()` was called on this saved tensor before the
    /// backward pass reached `unpack()`.
    #[error("autograd_v2: saved tensor has been released")]
    SavedTensorReleased,

    /// `InputBuffer::add` called with a slot index out of bounds.
    #[error("autograd_v2: input buffer slot {slot} out of bounds (num_inputs={num_inputs})")]
    InputSlotOutOfBounds { slot: usize, num_inputs: usize },

    /// Incoming gradient's dtype does not match the existing buffered grad.
    /// Under the Phase 4 Option A policy, gradients are stored at the
    /// parameter's dtype end-to-end; a mismatch indicates a caller bug.
    #[error(
        "autograd_v2: dtype mismatch in accumulation: existing {existing:?}, incoming {incoming:?}"
    )]
    DtypeMismatch { existing: DType, incoming: DType },

    /// A Phase 1 placeholder for engine code that lands in Phase 2.
    #[error("autograd_v2: not implemented yet in Phase 1: {0}")]
    NotImplementedYet(&'static str),

    /// The engine could not find the source tensor's `grad_fn` /
    /// `grad_accumulator` — typically because the tensor was never
    /// produced through `autograd_v2`-recorded ops and has no edge to
    /// kick off backward from.
    #[error("autograd_v2: output[{index}] has no grad_fn — nothing to backprop")]
    NoGradFnOnOutput { index: usize },

    /// `Engine::execute` was called with mismatched output / grad-output
    /// vector lengths.
    #[error("autograd_v2: outputs.len()={outputs} != grad_outputs.len()={grad_outputs}")]
    OutputGradLenMismatch { outputs: usize, grad_outputs: usize },

    /// `Engine::execute` got a user-supplied `grad_outputs[i]` whose
    /// shape doesn't match `outputs[i]`. Phase 3a validates at entry —
    /// the existing code path silently broadcast the size into the
    /// downstream InputBuffer, masking the caller bug. Bug-fixer
    /// audit flagged this; the validation lives at `Engine::execute`.
    #[error(
        "autograd_v2: grad_outputs[{index}] shape {grad_shape:?} != outputs[{index}] shape {out_shape:?}"
    )]
    GradOutputShapeMismatch {
        index: usize,
        out_shape: Vec<usize>,
        grad_shape: Vec<usize>,
    },

    /// A `GradFn::apply` returned a `Vec<Option<Tensor>>` whose length
    /// doesn't match its declared `num_inputs()`. Indicates a per-op
    /// implementation bug.
    #[error("autograd_v2: {op} apply returned {got} grads but declared num_inputs={expected}")]
    ApplyArityMismatch {
        op: &'static str,
        expected: usize,
        got: usize,
    },

    /// Pass-through for any wrapped flame-core error.
    #[error(transparent)]
    FlameCore(#[from] Error),
}

impl From<AutogradV2Error> for Error {
    fn from(e: AutogradV2Error) -> Self {
        // Keep the v2 error message verbatim under Error::Autograd so the
        // crate's existing Result<T, Error> consumers don't need to know
        // about the v2 enum.
        match e {
            AutogradV2Error::FlameCore(inner) => inner,
            other => Error::Autograd(other.to_string()),
        }
    }
}

/// Crate-local Result alias for autograd v2 paths. Engine and trait
/// methods that already return `Result<_, Error>` (because they bubble
/// through the wider flame-core error story) keep using `Result<_>`; the
/// internal v2-only entry points use this alias to surface structured
/// v2 errors.
pub type V2Result<T> = std::result::Result<T, AutogradV2Error>;
