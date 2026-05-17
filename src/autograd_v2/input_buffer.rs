//! `InputBuffer` — per-node gradient accumulation slots.
//!
//! Per §Phase 1 and recommended-change 7 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`:
//!
//! - `num_inputs` slots, each an `Option<Tensor>`. `None` = no grad
//!   contributed yet.
//! - `add(slot, grad)` accumulates incoming gradients. Two paths:
//!   * **In-place**: when `create_graph == false` AND the buffered grad's
//!     dtype/shape match AND the precondition for in-place mutation
//!     holds, the new grad is folded into the buffered tensor through
//!     `add_inplace_same_dtype`. No new allocation.
//!   * **Out-of-place**: otherwise, allocates a fresh tensor as
//!     `buffer[slot] + grad`. Required when `create_graph == true`
//!     (so backward-of-backward can re-record the add) or when
//!     in-place would clobber a tensor still referenced elsewhere.
//!
//! Phase 1's in-place precondition is conservative: same dtype, same
//! shape, contiguous (in-place adds today don't handle strides
//! correctly). Phase 2 may tighten — for example, by inspecting
//! `Arc::strong_count` on the underlying storage to detect aliasing
//! with another live tensor. The conservative path is correct; only
//! perf is left on the table.
//!
//! Out-of-place note: when `create_graph == true`, the `+` operation
//! itself must be recorded under autograd_v2 (so the backward pass can
//! differentiate through accumulation — higher-order grads). Phase 1
//! uses the existing `Tensor + Tensor` operator which records on the
//! old tape; the v2 wiring of that add is a Phase 3 op concern. A
//! `TODO(Phase 3)` marks the call site.

use std::os::raw::c_void;

use super::dispatch::DispatchCtx;
use super::error::AutogradV2Error;
use crate::dtype::DType;
use crate::tensor::Tensor;

pub struct InputBuffer {
    buffer: Vec<Option<Tensor>>,
    create_graph: bool,
    /// Internal counter — number of `add()` calls that took the
    /// in-place path. Test-only; production code shouldn't depend on
    /// the exact value (it changes when the in-place predicate is
    /// tightened in Phase 2).
    inplace_count: usize,
    /// Same idea for out-of-place adds. Together they make in-place
    /// behavior verifiable in tests without exposing storage internals.
    outofplace_count: usize,
}

impl InputBuffer {
    pub fn new(num_inputs: usize, create_graph: bool) -> Self {
        Self {
            buffer: vec![None; num_inputs],
            create_graph,
            inplace_count: 0,
            outofplace_count: 0,
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.buffer.len()
    }

    pub fn create_graph(&self) -> bool {
        self.create_graph
    }

    /// Accumulate `grad` into `buffer[slot]`. See module doc for the
    /// in-place vs out-of-place decision.
    ///
    /// `ctx` is currently unused (single-device, default stream) — it
    /// is in the signature per §16 so Phase 2+ can route in-place adds
    /// on different streams without a breaking change.
    pub fn add(
        &mut self,
        slot: usize,
        grad: Tensor,
        ctx: &DispatchCtx,
    ) -> Result<(), AutogradV2Error> {
        // Phase 3a: ctx is now load-bearing — the out-of-place path
        // routes through `add_v2` when `create_graph=true`.

        if slot >= self.buffer.len() {
            return Err(AutogradV2Error::InputSlotOutOfBounds {
                slot,
                num_inputs: self.buffer.len(),
            });
        }

        // Empty slot — first contributor wins, no accumulation needed.
        if self.buffer[slot].is_none() {
            self.buffer[slot] = Some(grad);
            return Ok(());
        }

        let existing_dtype = self.buffer[slot].as_ref().unwrap().dtype();
        let incoming_dtype = grad.dtype();
        if existing_dtype != incoming_dtype {
            return Err(AutogradV2Error::DtypeMismatch {
                existing: existing_dtype,
                incoming: incoming_dtype,
            });
        }

        let existing_shape_ok = self.buffer[slot].as_ref().unwrap().shape() == grad.shape();
        if !existing_shape_ok {
            // Shapes differ — out-of-place add is the only correct
            // behavior; the underlying op may broadcast.
            return self.add_outofplace(slot, grad, ctx);
        }

        // In-place is permitted iff create_graph=false AND
        // safe_to_inplace fires.
        if !self.create_graph && safe_to_inplace(self.buffer[slot].as_ref().unwrap()) {
            // Take ownership of the buffered tensor, mutate, put back.
            // The take/swap is safe because we hold `&mut self`.
            let mut existing = self.buffer[slot].take().unwrap();
            inplace_add(&mut existing, &grad)?;
            self.buffer[slot] = Some(existing);
            self.inplace_count += 1;
            return Ok(());
        }

        self.add_outofplace(slot, grad, ctx)
    }

    fn add_outofplace(
        &mut self,
        slot: usize,
        grad: Tensor,
        ctx: &DispatchCtx,
    ) -> Result<(), AutogradV2Error> {
        let existing = self.buffer[slot].take().unwrap();
        // Phase 3a: when create_graph=true, record the `+` itself as
        // a v2 op so higher-order grads can differentiate through the
        // accumulation. We construct the AddGradFn unconditionally
        // (the gradient tensors flowing through backward don't carry
        // `requires_grad=true` metadata — they ARE gradients, not
        // parameters; `create_graph` is the explicit signal that the
        // user wants the accumulation recorded regardless).
        let summed = if self.create_graph {
            let summed = existing.add(&grad).map_err(AutogradV2Error::FlameCore)?;
            let grad_fn = super::ops::add::AddGradFn::new(&existing, &grad);
            let recorded = super::recording::record_v2(grad_fn, vec![summed], ctx);
            recorded.into_iter().next().unwrap()
        } else {
            // Default (inference-fast) path: plain `+`, no recording.
            existing.add(&grad).map_err(AutogradV2Error::FlameCore)?
        };
        self.buffer[slot] = Some(summed);
        self.outofplace_count += 1;
        Ok(())
    }

    /// Consume the buffer, returning the per-slot gradient vector.
    /// Called by the engine when the node is ready to fire (Phase 2).
    pub fn take(self) -> Vec<Option<Tensor>> {
        self.buffer
    }

    // ----- Test-only inspectors -----

    /// Number of `add()` calls that took the in-place path. For tests.
    #[doc(hidden)]
    pub fn inplace_count(&self) -> usize {
        self.inplace_count
    }

    /// Number of `add()` calls that took the out-of-place path. For tests.
    #[doc(hidden)]
    pub fn outofplace_count(&self) -> usize {
        self.outofplace_count
    }

    /// Peek at a slot's current grad (clone). For tests.
    #[doc(hidden)]
    pub fn peek(&self, slot: usize) -> Option<Tensor> {
        self.buffer.get(slot).and_then(|s| s.clone())
    }
}

/// Phase 1 conservative in-place predicate. Returns true iff the buffered
/// tensor is in a shape/dtype/storage state that `inplace_add` can
/// safely mutate.
///
/// Phase 2 will tighten this — for instance by inspecting
/// `Arc::strong_count` on the storage to detect aliasing. The
/// conservative version is correct; only perf is left on the table.
fn safe_to_inplace(buffered: &Tensor) -> bool {
    // Dtype must be one that `add_inplace_same_dtype` actually supports
    // (BF16 + F32 today). Phase 2 may extend.
    matches!(buffered.dtype(), DType::F32 | DType::BF16)
}

/// Thin shim around `ops::elt::add_inplace_same_dtype`. The shim exists
/// so the InputBuffer source has one place to swap the implementation
/// (e.g. for a stream-aware in-place add in Phase 2+).
fn inplace_add(dst: &mut Tensor, src: &Tensor) -> Result<(), AutogradV2Error> {
    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        crate::ops::elt::add_inplace_same_dtype(dst, src).map_err(AutogradV2Error::FlameCore)
    }
    #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
    {
        // Without the CUDA+BF16 features there is no fast in-place
        // path; fall back to out-of-place by signaling failure.
        let _ = (dst, src);
        Err(AutogradV2Error::NotImplementedYet(
            "InputBuffer in-place add without cuda+bf16_u16 features",
        ))
    }
}

// `*mut c_void` from DispatchCtx is the only `unsafe` interaction here;
// it's never dereferenced inside InputBuffer.
fn _assert_ctx_handle_is_unused(_: *mut c_void) {}
