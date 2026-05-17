//! `SavedTensor` — the v2 analog of PyTorch's `SavedVariable`.
//!
//! Per §4 and recommended-change 4 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`:
//!
//! - Stores an `Arc<AtomicU32>` **handle** to the storage's version
//!   counter, NOT a bare `u32`. The handle survives
//!   `AutogradContext::clear()` / version-table flushes, so the version
//!   check at `unpack()` is always valid.
//! - `unpack()` returns `Result` (no panics) for version-mismatch and
//!   released-saved-tensor cases.
//! - `release_variables()` works through `&self` (interior mutability
//!   on the data slot).
//! - Carries an optional `fw_grad_` companion field for forward-mode
//!   autodiff plumbing (§clause 15 — phase 3 ops record forward
//!   formulas alongside backward).
//!
//! Compare with `crate::saved_ref::SavedRef`. The v1 `SavedRef` already
//! uses the version-handle pattern; `SavedTensor` extends it with
//! `&self`-driven release and `fw_grad_` storage. Once v2 supplants v1
//! the `SavedRef` type can be retired.

use crate::tensor::{Tensor, TensorId};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use super::error::AutogradV2Error;

pub struct SavedTensor {
    /// Identity for diagnostics and tape keying.
    id: TensorId,
    /// Interior-mutable data slot. `None` after `reset()` /
    /// `release_variables()`. `Mutex` (not `RwLock`) because reads are
    /// rare-but-exclusive and writes are rarer; the lock is contended
    /// only between the engine thread reading `unpack()` and a `&self`
    /// release path clearing the slot.
    data: Mutex<Option<Tensor>>,
    /// Shared handle to the storage's version counter. Cloned from
    /// `TensorStorage::version_handle()` at save time. Kept alive
    /// across version-table flushes so the version check at `unpack()`
    /// is meaningful even after `AutogradContext::clear()`.
    saved_version: Arc<AtomicU32>,
    /// Snapshot of the version counter at save time. Compared against
    /// the live counter at `unpack()`.
    expected_version: u32,
    /// True iff the saved tensor was a leaf when captured. Today
    /// informational only; Phase 3 view-autograd will use this to
    /// decide whether to re-materialize from base or unpack a view.
    is_leaf: bool,
    /// Forward-mode AD companion. Phase 3 ops write a forward-AD
    /// gradient here when running under a `dual_level()` context;
    /// `fw_grad()` reads it back. Phase 1 only ships the slot.
    fw_grad_: Mutex<Option<Tensor>>,
    /// Static op name carried for error messages (e.g. "MatmulBackward").
    /// Phase 1 stores `"<unnamed>"` by default; per-op SavedTensor
    /// constructors override it.
    op_name: &'static str,
}

impl SavedTensor {
    /// Snapshot a tensor for backward use. The version handle is
    /// captured eagerly so subsequent version-table flushes do not
    /// invalidate the check.
    pub fn save(t: &Tensor) -> Self {
        Self::save_named(t, "<unnamed>")
    }

    /// Like `save`, but carries a static op name for richer error
    /// messages on version mismatch.
    pub fn save_named(t: &Tensor, op_name: &'static str) -> Self {
        let counter = t.storage_ref().version_handle();
        let expected = counter.load(Ordering::Relaxed);
        Self {
            id: t.id(),
            data: Mutex::new(Some(t.clone())),
            saved_version: counter,
            expected_version: expected,
            is_leaf: !t.requires_grad(),
            fw_grad_: Mutex::new(None),
            op_name,
        }
    }

    /// Return the saved tensor. Errors if:
    /// - the underlying storage's version was bumped after save
    ///   (in-place mutation through some other handle), or
    /// - `release_variables()` / `reset()` cleared the data slot.
    pub fn unpack(&self) -> Result<Tensor, AutogradV2Error> {
        let live = self.saved_version.load(Ordering::Relaxed);
        if live != self.expected_version {
            return Err(AutogradV2Error::VersionMismatch {
                op: self.op_name,
                expected: self.expected_version,
                actual: live,
            });
        }
        let guard = self
            .data
            .lock()
            .expect("autograd_v2: SavedTensor data mutex poisoned");
        match &*guard {
            Some(t) => Ok(t.clone()),
            None => Err(AutogradV2Error::SavedTensorReleased),
        }
    }

    /// Drop the saved tensor's data and any forward-mode companion.
    /// Designed to be callable through `&self` so a `GradFn::release_variables(&self)`
    /// can clear its saved tensors without `&mut self`.
    pub fn reset(&self) {
        if let Ok(mut g) = self.data.lock() {
            *g = None;
        }
        if let Ok(mut g) = self.fw_grad_.lock() {
            *g = None;
        }
    }

    /// Forward-mode AD: set the dual companion. Phase 3 op forward
    /// formulas call this when a primal saved tensor has an active
    /// forward gradient under `dual_level()`.
    pub fn set_fw_grad(&self, g: Tensor) {
        if let Ok(mut slot) = self.fw_grad_.lock() {
            *slot = Some(g);
        }
    }

    /// Forward-mode AD: read the dual companion (cloned out).
    pub fn fw_grad(&self) -> Option<Tensor> {
        self.fw_grad_.lock().ok().and_then(|g| g.clone())
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// True iff `release_variables()` / `reset()` has been called and
    /// the data slot is empty.
    pub fn is_released(&self) -> bool {
        self.data.lock().map(|g| g.is_none()).unwrap_or(true)
    }

    /// Test-only: bump the underlying saved version counter directly,
    /// simulating an in-place mutation on a tensor that genuinely
    /// shares this storage. Used by the Phase 1 type test to verify
    /// `unpack()`'s version-mismatch error path without depending on
    /// the COW-on-`Arc::make_mut` behavior of `ensure_unique_slice`,
    /// which clones storage instead of mutating shared storage under
    /// the default `shared_storage` feature.
    ///
    /// In production, the version bump happens via in-place mutators
    /// (e.g. `ops::elt::add_inplace_same_dtype`) when storage is
    /// genuinely shared (e.g. via a view created by `narrow()`).
    #[doc(hidden)]
    pub fn _test_bump_saved_version(&self) {
        self.saved_version.fetch_add(1, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for SavedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SavedTensor")
            .field("id", &self.id)
            .field("op_name", &self.op_name)
            .field("expected_version", &self.expected_version)
            .field("live_version", &self.saved_version.load(Ordering::Relaxed))
            .field("released", &self.is_released())
            .finish()
    }
}
