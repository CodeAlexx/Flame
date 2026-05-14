//! Phase 4 of OFFLOAD_NEXT_GEN_DESIGN — `RingSlabAllocator`.
//!
//! Bidirectional ring allocator for offload-shaped workloads. Replaces
//! `cuda_alloc_pool`'s role for tensors that participate in the
//! forward/backward block cycle. The original pool's free-list ordering
//! corrupts under the forward-extend / backward-extend / forward-extend
//! pattern that offload training produces; this allocator's two cursors
//! and slab-list design sidestep the failure mode by design.
//!
//! Per OFFLOAD_NEXT_GEN_DESIGN §C and the Phase 4 phase-plan gate.
//!
//! # Design
//!
//! Each slab is a fixed-size GPU buffer. The allocator holds a list of
//! slabs and two cursors:
//!
//! - `allocation_start` — low watermark. Backward direction shrinks this.
//! - `allocation_end`   — high watermark. Forward direction grows this.
//!
//! In **Forward** direction, `alloc(n)` bumps `allocation_end += n` and
//! returns a slice at the previous `end`. In **Backward** direction,
//! `alloc(n)` bumps `allocation_start -= n` (well, advances by `n` and
//! returns the slice starting at the new start). When a cursor reaches
//! its slab's boundary, the allocator jumps to the next slab in the ring.
//!
//! Sub-allocations live until their `DeviceSlab` handle drops; drops
//! retire the most recent end (forward) or start (backward) range.
//! Out-of-order drops collapse retired ranges into the active cursor in
//! a separate pass to avoid fragmentation.
//!
//! # Phase 4 scope (this commit)
//!
//! - `RingSlabAllocator` struct + cursors + slab list.
//! - `alloc(num_bytes)` direction-aware (no ring wrap yet — Phase 4b).
//! - `DeviceSlab` RAII handle.
//! - `set_direction` / `reset_cursors`.
//! - Unit tests for forward-only, backward-only, and mixed sequences
//!   that don't cross a slab boundary.
//!
//! # Deferred to Phase 4b (separate commit when needed)
//!
//! - Slab-boundary wraparound (right now, oversize allocations error).
//! - Out-of-order drop coalescing (right now, only LIFO drops free).
//! - Microbench replicating the Klein corruption pattern.
//! - Wiring `Tensor::drop` to consult the allocator.
//!
//! Today this allocator is unused by trainer code — it ships as a
//! library type that Phase 6/7 trainers can adopt opt-in.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

use crate::{Error, Result};

pub use crate::offload::coordinator::AutogradDirection;

/// A bidirectional ring allocator over a list of fixed-size GPU slabs.
pub struct RingSlabAllocator {
    /// Backing device. Slabs are allocated from this.
    device: Arc<CudaDevice>,
    /// Slab list. First slab is allocated at construction; more grow
    /// on demand (Phase 4b — currently fixed-count from `new`).
    slabs: Vec<CudaSlice<u8>>,
    /// Per-slab byte capacity. Constant after construction.
    slab_bytes: usize,
    /// Index of the slab currently servicing forward allocations.
    forward_slab: usize,
    /// Offset within `forward_slab` where the next forward `alloc`
    /// will land. Grows with each forward alloc; resets when crossing
    /// a slab boundary.
    forward_offset: usize,
    /// Index of the slab currently servicing backward allocations.
    backward_slab: usize,
    /// Offset within `backward_slab`. Same growth semantics as forward;
    /// in Phase 4b this will retreat instead of advance.
    backward_offset: usize,
    /// Currently selected direction. `alloc` consults this; trainer
    /// flips it at the forward→backward transition via `set_direction`.
    direction: AutogradDirection,
    /// Stack of allocation lengths in forward direction. LIFO drop
    /// retires the top. Phase 4b adds out-of-order coalescing.
    forward_stack: Vec<usize>,
    /// Same for backward direction.
    backward_stack: Vec<usize>,
}

impl RingSlabAllocator {
    /// Construct with `num_slabs` slabs of `slab_bytes` each. Allocates
    /// all slabs eagerly on the device — Phase 4b adds lazy growth.
    pub fn new(
        device: Arc<CudaDevice>,
        num_slabs: usize,
        slab_bytes: usize,
    ) -> Result<Self> {
        if num_slabs == 0 {
            return Err(Error::InvalidInput(
                "RingSlabAllocator: num_slabs must be > 0".into(),
            ));
        }
        if slab_bytes == 0 {
            return Err(Error::InvalidInput(
                "RingSlabAllocator: slab_bytes must be > 0".into(),
            ));
        }

        let mut slabs = Vec::with_capacity(num_slabs);
        for _ in 0..num_slabs {
            let slab = unsafe { device.alloc::<u8>(slab_bytes) }
                .map_err(|e| Error::CudaDriver(format!("RingSlabAllocator slab alloc: {e:?}")))?;
            slabs.push(slab);
        }

        Ok(Self {
            device,
            slabs,
            slab_bytes,
            forward_slab: 0,
            forward_offset: 0,
            backward_slab: 0,
            backward_offset: 0,
            direction: AutogradDirection::Forward,
            forward_stack: Vec::new(),
            backward_stack: Vec::new(),
        })
    }

    /// Current direction. The next `alloc` will service this direction.
    pub fn direction(&self) -> AutogradDirection {
        self.direction
    }

    /// Switch direction. Forward-to-backward at the start of the
    /// backward pass; backward-to-forward at the start of the next step.
    /// Idempotent — setting the same direction is a no-op.
    pub fn set_direction(&mut self, d: AutogradDirection) {
        self.direction = d;
    }

    /// Reset both cursors to 0 and clear the LIFO drop stacks. Call at
    /// the start of training or between epochs if the allocator
    /// shouldn't carry state across.
    pub fn reset_cursors(&mut self) {
        self.forward_slab = 0;
        self.forward_offset = 0;
        self.backward_slab = 0;
        self.backward_offset = 0;
        self.forward_stack.clear();
        self.backward_stack.clear();
    }

    /// Number of bytes currently allocated in the forward direction
    /// across all forward slabs (sum of `forward_stack`). Diagnostic.
    pub fn forward_bytes_in_use(&self) -> usize {
        self.forward_stack.iter().copied().sum()
    }

    /// Same for backward.
    pub fn backward_bytes_in_use(&self) -> usize {
        self.backward_stack.iter().copied().sum()
    }

    /// Allocate `num_bytes` from the current direction's cursor. Returns
    /// a `DeviceSlab` RAII handle whose Drop retires the range from the
    /// LIFO drop stack.
    ///
    /// Phase 4 limitation: if `num_bytes` doesn't fit in the current
    /// slab, returns an error. Phase 4b adds wraparound to the next
    /// slab. Today, callers must size `slab_bytes` to fit their peak
    /// single-allocation.
    pub fn alloc(&mut self, num_bytes: usize) -> Result<DeviceSlab> {
        if num_bytes == 0 {
            return Err(Error::InvalidInput(
                "RingSlabAllocator::alloc: num_bytes must be > 0".into(),
            ));
        }
        if num_bytes > self.slab_bytes {
            return Err(Error::InvalidInput(format!(
                "RingSlabAllocator::alloc: {num_bytes} bytes exceeds slab capacity {} \
                 (Phase 4 doesn't yet wrap across slabs; size slab_bytes to fit)",
                self.slab_bytes
            )));
        }

        match self.direction {
            AutogradDirection::Forward => {
                let (slab_idx, offset) = self.alloc_forward(num_bytes)?;
                self.forward_stack.push(num_bytes);
                Ok(DeviceSlab {
                    slab_idx,
                    offset,
                    bytes: num_bytes,
                    direction: AutogradDirection::Forward,
                    base_ptr: self.slab_ptr(slab_idx),
                    allocator: None,
                })
            }
            AutogradDirection::Backward => {
                let (slab_idx, offset) = self.alloc_backward(num_bytes)?;
                self.backward_stack.push(num_bytes);
                Ok(DeviceSlab {
                    slab_idx,
                    offset,
                    bytes: num_bytes,
                    direction: AutogradDirection::Backward,
                    base_ptr: self.slab_ptr(slab_idx),
                    allocator: None,
                })
            }
        }
    }

    /// Allocate via an `Arc<Mutex<Self>>`. The returned `DeviceSlab`'s
    /// Drop auto-retires the range to this allocator. Use this when the
    /// allocation isn't manually retired by the caller.
    pub fn alloc_handle(
        this: &Arc<Mutex<Self>>,
        num_bytes: usize,
    ) -> Result<DeviceSlab> {
        let weak = Arc::downgrade(this);
        let mut guard = this
            .lock()
            .map_err(|_| Error::InvalidOperation("RingSlabAllocator mutex poisoned".into()))?;
        let mut slab = guard.alloc(num_bytes)?;
        slab.allocator = Some(weak);
        Ok(slab)
    }

    fn alloc_forward(&mut self, n: usize) -> Result<(usize, usize)> {
        let remaining = self.slab_bytes - self.forward_offset;
        if n > remaining {
            // Phase 4: jump to next slab. (Phase 4b will reuse the
            // freed tail of the previous slab via coalescing.)
            self.forward_slab += 1;
            self.forward_offset = 0;
            if self.forward_slab >= self.slabs.len() {
                return Err(Error::InvalidOperation(format!(
                    "RingSlabAllocator: forward direction exhausted slabs \
                     ({} of capacity {}) — Phase 4b will grow",
                    self.slabs.len(),
                    self.slab_bytes
                )));
            }
        }
        let slab_idx = self.forward_slab;
        let offset = self.forward_offset;
        self.forward_offset += n;
        Ok((slab_idx, offset))
    }

    fn alloc_backward(&mut self, n: usize) -> Result<(usize, usize)> {
        let remaining = self.slab_bytes - self.backward_offset;
        if n > remaining {
            self.backward_slab += 1;
            self.backward_offset = 0;
            if self.backward_slab >= self.slabs.len() {
                return Err(Error::InvalidOperation(format!(
                    "RingSlabAllocator: backward direction exhausted slabs \
                     ({} of capacity {}) — Phase 4b will grow",
                    self.slabs.len(),
                    self.slab_bytes
                )));
            }
        }
        let slab_idx = self.backward_slab;
        let offset = self.backward_offset;
        self.backward_offset += n;
        Ok((slab_idx, offset))
    }

    fn slab_ptr(&self, slab_idx: usize) -> u64 {
        *self.slabs[slab_idx].device_ptr()
    }

    /// Internal — retire a forward allocation. Called by
    /// `DeviceSlab::drop` (when allocated via `alloc_handle`) or by
    /// tests directly. LIFO-fast path advances the cursor back;
    /// out-of-order drops drop the matching stack entry but don't
    /// advance the cursor (leaks until the stack drains, then the
    /// cursor resets — see below).
    ///
    /// **Cursor reset on empty stack**: when the last forward
    /// allocation retires, both `forward_slab` and `forward_offset`
    /// reset to 0. This makes practical workloads possible without
    /// infinite slab growth — BlockOffloader-style code drops a whole
    /// batch of tensors at once (in HashMap iteration order, not
    /// reverse-alloc order), and we want the next batch to start fresh.
    pub(crate) fn retire_forward(&mut self, bytes: usize) {
        let fast = self.forward_stack.last().copied() == Some(bytes);
        if fast {
            self.forward_stack.pop();
            if self.forward_offset >= bytes {
                self.forward_offset -= bytes;
            } else {
                self.forward_offset = 0;
            }
        } else if let Some(pos) = self.forward_stack.iter().rposition(|&b| b == bytes) {
            self.forward_stack.remove(pos);
        }
        if self.forward_stack.is_empty() {
            self.forward_offset = 0;
            self.forward_slab = 0;
        }
    }

    /// Internal — retire a backward allocation. Same shape as
    /// `retire_forward`.
    pub(crate) fn retire_backward(&mut self, bytes: usize) {
        let fast = self.backward_stack.last().copied() == Some(bytes);
        if fast {
            self.backward_stack.pop();
            if self.backward_offset >= bytes {
                self.backward_offset -= bytes;
            } else {
                self.backward_offset = 0;
            }
        } else if let Some(pos) = self.backward_stack.iter().rposition(|&b| b == bytes) {
            self.backward_stack.remove(pos);
        }
        if self.backward_stack.is_empty() {
            self.backward_offset = 0;
            self.backward_slab = 0;
        }
    }
}

/// RAII handle to a slab allocation. Holds a raw GPU pointer and the
/// allocator-side metadata needed to retire the range on Drop.
///
/// **Lifetime safety**: the handle does NOT borrow the allocator. The
/// allocator manages slab-level lifetimes — slabs are not freed until
/// the allocator drops. So a `DeviceSlab`'s pointer is valid for as
/// long as the allocator is alive.
///
/// **Drop semantics**: `Drop` consults a thread-local hook to retire the
/// allocation back to its allocator. Phase 4 ships this struct without
/// the hook (drops are no-ops at the allocator); the hook lands in
/// Phase 4b when Tensor::drop integration is wired.
pub struct DeviceSlab {
    /// Which slab in the allocator's list backs this allocation.
    pub(crate) slab_idx: usize,
    /// Byte offset within the slab.
    pub(crate) offset: usize,
    /// Size of the allocation in bytes.
    pub(crate) bytes: usize,
    /// Direction the allocation was made in. Drives whether `Drop`
    /// retires forward or backward.
    pub(crate) direction: AutogradDirection,
    /// Raw device pointer of the slab's base.
    pub(crate) base_ptr: u64,
    /// Weak reference to the owning allocator. When the slab's Drop
    /// runs, it upgrades and calls the appropriate retire. `None`
    /// means "the slab was constructed without an Arc-owned allocator"
    /// (legacy `RingSlabAllocator::alloc(&mut self)` path used by
    /// pre-Phase-4b tests); in that case Drop is a no-op and callers
    /// must retire manually.
    pub(crate) allocator: Option<std::sync::Weak<std::sync::Mutex<RingSlabAllocator>>>,
}

impl DeviceSlab {
    /// Raw device pointer to the start of this allocation.
    pub fn device_ptr(&self) -> u64 {
        self.base_ptr + self.offset as u64
    }

    /// Allocation size in bytes.
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// True if the allocation has zero size. Always false because
    /// `alloc` rejects zero-byte requests.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Direction the allocation was made in.
    pub fn direction(&self) -> AutogradDirection {
        self.direction
    }

    /// Slab index within the allocator's list. Mostly for tests.
    pub fn slab_idx(&self) -> usize {
        self.slab_idx
    }

    /// Offset within the slab. Mostly for tests.
    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl Drop for DeviceSlab {
    fn drop(&mut self) {
        // Auto-retire when constructed via the Arc-aware path
        // (`alloc_handle` or `alloc_with_registry`). When constructed
        // via the bare `RingSlabAllocator::alloc(&mut self)` path,
        // `allocator` is `None` and Drop is a no-op — those tests
        // call `retire_forward`/`retire_backward` manually.
        if let Some(weak) = self.allocator.take() {
            if let Some(arc) = weak.upgrade() {
                if let Ok(mut a) = arc.lock() {
                    match self.direction {
                        AutogradDirection::Forward => a.retire_forward(self.bytes),
                        AutogradDirection::Backward => a.retire_backward(self.bytes),
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn try_cuda() -> Option<Arc<CudaDevice>> {
        let dev = Device::cuda(0).ok()?;
        Some(dev.cuda_device().clone())
    }

    #[test]
    fn rejects_zero_slabs() {
        let Some(dev) = try_cuda() else { return };
        assert!(RingSlabAllocator::new(dev, 0, 1024).is_err());
    }

    #[test]
    fn rejects_zero_slab_bytes() {
        let Some(dev) = try_cuda() else { return };
        assert!(RingSlabAllocator::new(dev, 1, 0).is_err());
    }

    #[test]
    fn forward_alloc_advances_cursor() -> Result<()> {
        let Some(dev) = try_cuda() else { return Ok(()) };
        let mut a = RingSlabAllocator::new(dev, 2, 4096)?;
        let s1 = a.alloc(256)?;
        assert_eq!(s1.slab_idx(), 0);
        assert_eq!(s1.offset(), 0);
        let s2 = a.alloc(512)?;
        assert_eq!(s2.slab_idx(), 0);
        assert_eq!(s2.offset(), 256);
        assert_eq!(a.forward_bytes_in_use(), 768);
        Ok(())
    }

    #[test]
    fn backward_alloc_uses_separate_cursor() -> Result<()> {
        let Some(dev) = try_cuda() else { return Ok(()) };
        let mut a = RingSlabAllocator::new(dev, 2, 4096)?;
        let _f = a.alloc(256)?; // forward, slab 0, offset 0
        a.set_direction(AutogradDirection::Backward);
        let b = a.alloc(128)?;
        assert_eq!(b.slab_idx(), 0);
        // Backward starts at 0, not contaminated by forward's cursor.
        assert_eq!(b.offset(), 0);
        assert_eq!(a.direction(), AutogradDirection::Backward);
        Ok(())
    }

    #[test]
    fn forward_crosses_slab_boundary() -> Result<()> {
        let Some(dev) = try_cuda() else { return Ok(()) };
        let mut a = RingSlabAllocator::new(dev, 2, 1024)?;
        let s1 = a.alloc(800)?;
        assert_eq!(s1.slab_idx(), 0);
        let s2 = a.alloc(800)?;
        assert_eq!(s2.slab_idx(), 1);
        assert_eq!(s2.offset(), 0);
        Ok(())
    }

    #[test]
    fn forward_exhausts_all_slabs_errors() -> Result<()> {
        let Some(dev) = try_cuda() else { return Ok(()) };
        let mut a = RingSlabAllocator::new(dev, 1, 1024)?;
        let _s1 = a.alloc(800)?;
        // Won't fit; only 1 slab exists.
        assert!(a.alloc(800).is_err());
        Ok(())
    }

    #[test]
    fn rejects_oversize_alloc() -> Result<()> {
        let Some(dev) = try_cuda() else { return Ok(()) };
        let mut a = RingSlabAllocator::new(dev, 2, 1024)?;
        assert!(a.alloc(2048).is_err());
        Ok(())
    }

    #[test]
    fn reset_clears_cursors() -> Result<()> {
        let Some(dev) = try_cuda() else { return Ok(()) };
        let mut a = RingSlabAllocator::new(dev, 2, 4096)?;
        let _s1 = a.alloc(256)?;
        let _s2 = a.alloc(512)?;
        a.set_direction(AutogradDirection::Backward);
        let _b1 = a.alloc(128)?;
        a.reset_cursors();
        assert_eq!(a.forward_bytes_in_use(), 0);
        assert_eq!(a.backward_bytes_in_use(), 0);
        // Next forward alloc starts fresh.
        a.set_direction(AutogradDirection::Forward);
        let s = a.alloc(100)?;
        assert_eq!(s.offset(), 0);
        Ok(())
    }

    #[test]
    fn device_slab_pointer_is_offset_from_base() -> Result<()> {
        let Some(dev) = try_cuda() else { return Ok(()) };
        let mut a = RingSlabAllocator::new(dev, 1, 4096)?;
        let s1 = a.alloc(256)?;
        let base1 = s1.device_ptr();
        let s2 = a.alloc(128)?;
        let base2 = s2.device_ptr();
        assert_eq!(base2, base1 + 256);
        Ok(())
    }
}
