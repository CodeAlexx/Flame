//! Phase 2 — `SavedRef`: lightweight reference to a tensor saved for backward.
//!
//! PyTorch's `SavedVariable` stores a (storage, version_counter, view metadata)
//! triple instead of a full `Tensor` to avoid an Arc bump on the device-pointer
//! field per recorded op. flame-core's `Tensor::clone` is already cheap (shape
//! is an inline SmallVec, storage clone is a single Arc bump), but the
//! per-record `vec![...]` heap allocation + the extra Arc bump on `device:
//! Arc<CudaDevice>` add up across the ~2660 tape entries per training step
//! for klein 4B.
//!
//! `SavedRef` is the equivalent of PyTorch's `SavedVariable`:
//!   - holds the saved tensor's `id` (for backward keying)
//!   - holds an `Arc<AtomicU32>` clone of the storage's version counter,
//!     snapshotted at save time. `unpack()` errors if the live counter has
//!     advanced (i.e. the tensor was modified in-place after the save).
//!   - holds a `Tensor` so backward consumers can still read activations.
//!
//! The Tensor field is retained for the initial migration so backward
//! consumers don't need to be rewritten in lock-step. A future change can
//! split this into a `storage + shape + strides` rebuild to drop the
//! `Arc<CudaDevice>` bump on `Tensor::clone`.
//!
//! Rollback knob: `FLAME_AUTOGRAD_SAVED_LEGACY=1` keeps the old
//! `(TensorId, Tensor)` tape entries instead of `SavedRef`. Bench harnesses
//! use this to A/B the change.

use crate::tensor::TensorId;
use crate::{Error, Result, Tensor};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Lightweight saved-tensor handle for the autograd tape.
#[derive(Clone)]
pub struct SavedRef {
    pub id: TensorId,
    /// The saved tensor itself. Backward consumers read shape/storage/device
    /// from this. Cheap to clone (single inner Arc bump + device Arc bump +
    /// shape SmallVec copy). Future optimization: replace with raw
    /// storage+shape+strides triple, drop the device Arc bump.
    pub tensor: Tensor,
    /// Snapshot of the storage's version counter at save time. The Arc keeps
    /// the counter alive even if the side-table is flushed (e.g. at
    /// `AutogradContext::clear`) so the check at unpack is always valid.
    pub version_counter: Arc<AtomicU32>,
    pub version_at_save: u32,
}

impl SavedRef {
    /// Snapshot a tensor for backward use. Captures the storage's version
    /// counter; later in-place modifications to the same storage will fail
    /// the `unpack` check.
    #[inline]
    pub fn capture(t: &Tensor) -> Self {
        let counter = t.storage_ref().version_handle();
        let version = counter.load(Ordering::Relaxed);
        Self {
            id: t.id(),
            tensor: t.clone(),
            version_counter: counter,
            version_at_save: version,
        }
    }

    /// Materialize the saved tensor for backward use. Errors if the
    /// underlying storage was modified in-place after the save (version
    /// mismatch). This is the empirical safety net that the spec calls for.
    #[inline]
    pub fn unpack(&self) -> Result<Tensor> {
        let live = self.version_counter.load(Ordering::Relaxed);
        if live != self.version_at_save {
            return Err(Error::InvalidOperation(format!(
                "SavedRef::unpack: tensor id={:?} was modified in-place after save \
                 (saved v{} != live v{}) — backward would be silently wrong",
                self.id, self.version_at_save, live
            )));
        }
        Ok(self.tensor.clone())
    }

    /// Borrowing accessor — for cases where backward only needs a read-only
    /// reference to the saved tensor. Performs the version check.
    #[inline]
    pub fn unpack_ref(&self) -> Result<&Tensor> {
        let live = self.version_counter.load(Ordering::Relaxed);
        if live != self.version_at_save {
            return Err(Error::InvalidOperation(format!(
                "SavedRef::unpack_ref: tensor id={:?} was modified in-place after save \
                 (saved v{} != live v{})",
                self.id, self.version_at_save, live
            )));
        }
        Ok(&self.tensor)
    }
}

impl std::fmt::Debug for SavedRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SavedRef")
            .field("id", &self.id)
            .field("version_at_save", &self.version_at_save)
            .field("shape", self.tensor.shape())
            .finish()
    }
}

/// Cached rollback knob. Read once at process start; subsequent reads are
/// a single atomic load. `FLAME_AUTOGRAD_SAVED_LEGACY=1` forces the legacy
/// `(TensorId, Tensor)` tape path (for A/B testing).
#[inline]
pub fn legacy_saved_mode() -> bool {
    static CACHED: once_cell::sync::OnceCell<bool> = once_cell::sync::OnceCell::new();
    *CACHED
        .get_or_init(|| std::env::var("FLAME_AUTOGRAD_SAVED_LEGACY").ok().as_deref() == Some("1"))
}
