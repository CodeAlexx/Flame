//! Bidirectional ring allocator over a list of fixed-size GPU slabs.
//!
//! Phase 1 of the Gap 1 workstream from
//! [`docs/OFFLOAD_GAPS_vs_ONETRAINER.md`]. Faithful port of
//! `StaticLayerAllocator` + `StaticLayerTensorAllocator` from OneTrainer's
//! `modules/util/LayerOffloadConductor.py` (lines 37-222).
//!
//! Full design rationale, OT line citations, invariants, and what is
//! deliberately NOT in this phase are documented in
//! [`docs/RING_ALLOC_DESIGN.md`].
//!
//! # Quick model
//!
//! Slabs are concatenated into one logical byte space. Two cursors walk
//! that space from opposite ends:
//!
//! - Forward allocations advance `allocation_end` (low-to-high).
//! - Backward allocations retreat `allocation_start` (high-to-low).
//!
//! Slabs are `cudaMalloc`-ed lazily on first touch (OT line 187-197).
//! Bytes are never freed per-allocation; the cursors reset between steps
//! and the slabs stay mapped.
//!
//! # Why this exists
//!
//! `cuda_alloc_pool` (PyTorch-style bucketed free list) corrupts under
//! BlockOffloader + checkpoint replay on Klein 9B at step 2
//! (`project_klein9b_step2_crash_isolation`). The ring's two-cursor design
//! makes overlapping forward / backward allocations structurally impossible
//! within a step — the corruption window doesn't exist.
//!
//! Phase 1 ships the primitive + a microbench. Phase 2 wires Klein and
//! drops the `FLAME_ALLOC_POOL=0` workaround.

pub mod pool_adapter;
pub use pool_adapter::RingPoolAdapter;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

use crate::{Error, Result};

// ---------------------------------------------------------------------------
// Alignment helpers — direct port of OT lines 37-42.
// ---------------------------------------------------------------------------

/// Round `n` up to the next multiple of 16. OT line 37-38.
#[inline]
const fn ceil_16(n: usize) -> usize {
    (n + 15) & !15
}

/// Round `n` down to a multiple of 16. OT line 41-42.
#[inline]
const fn floor_16(n: usize) -> usize {
    n & !15
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An untyped device byte range produced by a `RingAllocator`.
///
/// **Not RAII.** Dropping a `RingPtr` does not free bytes — the underlying
/// slab is owned by the allocator and persists across allocations. The
/// range is valid until the matching `RingAllocator::reset()` runs. After
/// reset, future allocations may reuse these bytes; holding a stale
/// `RingPtr` past a reset is a use-after-free.
#[derive(Debug, Clone, Copy)]
pub struct RingPtr {
    /// Raw CUDA device pointer to the start of the allocation.
    /// Always 16-byte aligned.
    pub device_ptr: u64,
    /// Length of the allocation in bytes (as requested by the caller).
    pub len_bytes: usize,
    /// Index of the slab this allocation came out of.
    pub slab_idx: usize,
    /// Byte offset within `slab_idx` where the allocation starts.
    pub intra_offset: usize,
}

/// A bidirectional ring allocator over a list of fixed-size GPU slabs.
///
/// One ring per training step is the canonical lifetime: forward pass
/// fills via `forward_handle`, backward pass drains via `backward_handle`,
/// and `reset()` between steps puts the cursors back at the ends. Slabs
/// stay mapped across resets.
///
/// **Not `Sync` / `Send`-safe** — alloc paths require `&mut self`. This
/// matches flame-core's single-threaded autograd model.
///
/// Faithful Rust port of OT's `StaticLayerAllocator`
/// (`/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py:122-222`).
/// Differences from OT:
///
/// - Direction is enforced at the type level via `RingForwardHandle` /
///   `RingBackwardHandle`, instead of a `bool` parameter.
/// - Slab list is `Vec<Option<CudaSlice<u8>>>` (lazy `cudaMalloc` on
///   first touch), matching OT's `cache_tensors: list[Tensor | None]`.
/// - No deallocation method exposed: per-allocation reclaim would break
///   the cursor invariant (OT's `deallocate(deallocate_forward)` is a
///   no-op equivalent — see OT lines 113-119).
pub struct RingAllocator {
    /// Backing CUDA device. All slabs are allocated on this device.
    device: Arc<CudaDevice>,
    /// Slab list. `None` entries are slabs not yet `cudaMalloc`-ed.
    /// First touch in either direction materializes the slab.
    slabs: Vec<Option<CudaSlice<u8>>>,
    /// Per-slab byte capacity. Constant after construction. Rounded up
    /// to a multiple of 16 internally so intra-slab offsets stay aligned.
    slab_bytes: usize,
    /// Logical low watermark, in bytes, across the concatenated slab
    /// space. Backward allocations decrease this. OT `allocation_start`.
    allocation_start: usize,
    /// Logical high watermark. Forward allocations increase this.
    /// OT `allocation_end`.
    allocation_end: usize,
    /// Cached `slab_bytes * num_slabs`.
    total_bytes: usize,
    /// Monotonic counter of slabs we have materialized. Equals the number
    /// of `cudaMalloc` calls the ring has issued. Exposed for the
    /// microbench's lazy-allocation assertion.
    cuda_malloc_count: u64,
}

/// Direction-typed handle for forward-pass allocations within a block.
///
/// Borrows the allocator mutably. Drop before requesting a backward handle.
/// Mirrors OT's `StaticLayerTensorAllocator(allocate_forward=True, ...)`.
pub struct RingForwardHandle<'a> {
    alloc: &'a mut RingAllocator,
    /// Layer index this handle was issued for — recorded for diagnostics
    /// only; does not affect allocation semantics.
    #[allow(dead_code)]
    layer_idx: usize,
}

/// Direction-typed handle for backward-pass allocations within a block.
pub struct RingBackwardHandle<'a> {
    alloc: &'a mut RingAllocator,
    #[allow(dead_code)]
    layer_idx: usize,
}

// ---------------------------------------------------------------------------
// Construction & inspection
// ---------------------------------------------------------------------------

impl RingAllocator {
    /// Construct with `num_slabs` × `slab_bytes` (rounded up to a multiple
    /// of 16) total capacity.
    ///
    /// **Lazy allocation.** No slab is `cudaMalloc`-ed by this call. The
    /// first `alloc` in each slab triggers its allocation.
    ///
    /// Errors if `num_slabs == 0` or `slab_bytes == 0`.
    pub fn new(device: Arc<CudaDevice>, num_slabs: usize, slab_bytes: usize) -> Result<Self> {
        if num_slabs == 0 {
            return Err(Error::InvalidInput(
                "RingAllocator: num_slabs must be > 0".into(),
            ));
        }
        if slab_bytes == 0 {
            return Err(Error::InvalidInput(
                "RingAllocator: slab_bytes must be > 0".into(),
            ));
        }

        // Force 16-byte alignment of slab size so intra-slab offsets stay
        // 16-aligned for every allocation. OT does not do this (it relies
        // on the caller picking a sensible size); we make it impossible to
        // mis-size by construction.
        let slab_bytes = ceil_16(slab_bytes);
        let total_bytes = slab_bytes.checked_mul(num_slabs).ok_or_else(|| {
            Error::InvalidInput("RingAllocator: num_slabs * slab_bytes overflows usize".into())
        })?;

        let slabs = (0..num_slabs).map(|_| None).collect();

        Ok(Self {
            device,
            slabs,
            slab_bytes,
            allocation_start: total_bytes, // backward cursor starts at the top
            allocation_end: 0,             // forward cursor starts at the bottom
            total_bytes,
            cuda_malloc_count: 0,
        })
    }

    /// Convenience alias for `new` (Spec API parity — sweeps may want this
    /// name).
    pub fn with_slabs(
        device: Arc<CudaDevice>,
        num_slabs: usize,
        slab_bytes: usize,
    ) -> Result<Self> {
        Self::new(device, num_slabs, slab_bytes)
    }

    /// Number of slabs configured (allocated or not).
    #[inline]
    pub fn num_slabs(&self) -> usize {
        self.slabs.len()
    }

    /// Bytes per slab (16-aligned).
    #[inline]
    pub fn slab_bytes(&self) -> usize {
        self.slab_bytes
    }

    /// Total bytes across all slabs.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Current backward (high) watermark in the global byte space.
    #[inline]
    pub fn allocation_start(&self) -> usize {
        self.allocation_start
    }

    /// Current forward (low) watermark in the global byte space.
    #[inline]
    pub fn allocation_end(&self) -> usize {
        self.allocation_end
    }

    /// Count of slabs materialized so far. Equals `cuda_malloc_count`.
    #[inline]
    pub fn slabs_allocated(&self) -> usize {
        self.slabs.iter().filter(|s| s.is_some()).count()
    }

    /// Monotonic count of `cudaMalloc` calls this allocator has issued.
    /// Used by the microbench to verify lazy allocation.
    #[inline]
    pub fn cuda_malloc_count(&self) -> u64 {
        self.cuda_malloc_count
    }

    /// Backing `CudaDevice`. Useful for wrapping a `RingAllocator` in a
    /// `PoolMissAllocator` adapter where the alloc surfaces need the
    /// device handle anyway.
    #[inline]
    pub fn device(&self) -> &Arc<cudarc::driver::CudaDevice> {
        &self.device
    }

    /// Allocate `num_bytes` bytes forward without a handle. Convenience
    /// for adapters (e.g. `PoolMissAllocator`) that don't carry a
    /// `RingForwardHandle`. Semantics identical to
    /// `forward_handle(0).alloc(num_bytes)`.
    pub fn alloc_forward(&mut self, num_bytes: usize) -> Result<RingPtr> {
        self.alloc_forward_impl(num_bytes)
    }

    /// Allocate `num_bytes` bytes backward without a handle. Convenience
    /// for adapters; see `alloc_forward`.
    pub fn alloc_backward(&mut self, num_bytes: usize) -> Result<RingPtr> {
        self.alloc_backward_impl(num_bytes)
    }

    /// Reset cursors. Forward starts at byte 0, backward at `total_bytes`.
    /// Slabs are NOT deallocated — they stay mapped for reuse next step.
    pub fn reset(&mut self) {
        self.allocation_start = self.total_bytes;
        self.allocation_end = 0;
    }

    // -------- Per-block handles --------

    /// Begin a forward-pass scope for `block_idx`. The returned handle
    /// borrows the allocator mutably; drop before requesting a backward
    /// handle on the same allocator.
    pub fn forward_handle(&mut self, block_idx: usize) -> RingForwardHandle<'_> {
        RingForwardHandle {
            alloc: self,
            layer_idx: block_idx,
        }
    }

    /// Begin a backward-pass scope for `block_idx`.
    pub fn backward_handle(&mut self, block_idx: usize) -> RingBackwardHandle<'_> {
        RingBackwardHandle {
            alloc: self,
            layer_idx: block_idx,
        }
    }

    // -------- Internal helpers --------

    /// Materialize slab `idx` if it isn't yet. OT line 187-197 equivalent.
    ///
    /// Bumps `cuda_malloc_count` on a real allocation; no-op otherwise.
    fn ensure_slab(&mut self, idx: usize) -> Result<()> {
        if self.slabs[idx].is_some() {
            return Ok(());
        }
        // SAFETY: `device.alloc` returns uninitialized memory; we make no
        // assumptions about the contents — every allocator hand-out is
        // expected to be fully written by the caller before any read.
        let slab = unsafe { self.device.alloc::<u8>(self.slab_bytes) }.map_err(|e| {
            Error::CudaDriver(format!("RingAllocator slab[{idx}] cudaMalloc: {e:?}"))
        })?;
        self.slabs[idx] = Some(slab);
        self.cuda_malloc_count += 1;
        Ok(())
    }

    /// Look up the device pointer for a (slab_idx, intra_offset) pair.
    /// Assumes `ensure_slab(slab_idx)` was called.
    #[inline]
    fn slab_device_ptr(&self, slab_idx: usize, intra_offset: usize) -> u64 {
        let slab = self.slabs[slab_idx]
            .as_ref()
            .expect("ensure_slab must be called before slab_device_ptr");
        *slab.device_ptr() + intra_offset as u64
    }

    /// Forward allocation — port of OT lines 65-89.
    fn alloc_forward_impl(&mut self, num_bytes: usize) -> Result<RingPtr> {
        if num_bytes == 0 {
            return Err(Error::InvalidInput(
                "RingAllocator: num_bytes must be > 0".into(),
            ));
        }
        if num_bytes > self.slab_bytes {
            return Err(Error::InvalidInput(format!(
                "RingAllocator: alloc({num_bytes}B) exceeds slab_bytes ({})",
                self.slab_bytes
            )));
        }

        // OT line 71-72: current slab index and the 16-aligned start
        // within that slab.
        let cur_slab_idx = self.allocation_end / self.slab_bytes;
        let cur_intra = ceil_16(self.allocation_end % self.slab_bytes);

        // OT line 74-77: if it doesn't fit in the remainder of this slab,
        // jump to slab 0 of the next.
        let (cand_slab_idx, cand_intra) = if cur_intra + num_bytes > self.slab_bytes {
            (cur_slab_idx + 1, 0_usize)
        } else {
            (cur_slab_idx, cur_intra)
        };

        // OT line 78-82: cyclic wrap if we walked off the end.
        let wrapped = cand_slab_idx >= self.slabs.len();
        let (slab_idx, intra) = if wrapped {
            // Refuse the wrap when backward is active — the wrap would
            // silently lap a live forward allocation and/or collide with
            // the backward region. Per design doc §4 invariant 2: wrap-
            // induced violations error rather than silently overlap. Wrap
            // is permitted only when allocation_start == total_bytes
            // (i.e., no backward state to conflict with), which is the
            // documented Phase 1 wrap-when-only-forward-active mode.
            if self.allocation_start < self.total_bytes {
                return Err(Error::OutOfMemory(format!(
                    "RingAllocator exhausted: forward wrap with backward \
                     active would lap live allocations (allocation_end={}, \
                     allocation_start={}, total={}, slabs={}). Increase \
                     ring size or reset between steps.",
                    self.allocation_end,
                    self.allocation_start,
                    self.total_bytes,
                    self.slabs.len(),
                )));
            }
            (0_usize, 0_usize)
        } else {
            (cand_slab_idx, cand_intra)
        };

        // Invariant check (no OT analog — OT trusts sizing). If the new
        // forward end would cross into the backward-allocated region, the
        // ring is exhausted; error rather than silently overlap.
        let new_global_end = slab_idx * self.slab_bytes + intra + num_bytes;
        self.check_no_overlap_forward(new_global_end)?;

        self.ensure_slab(slab_idx)?;
        let device_ptr = self.slab_device_ptr(slab_idx, intra);

        // OT line 83 (effective): record the new end.
        // OT updates `self.__allocation_end = cache_tensor_index *
        // cache_tensor_size + cache_tensor_allocation_end;` BEFORE adding
        // num_bytes (line 83), then adds num_bytes after (line 88). Net
        // effect is what we do here in one step.
        self.allocation_end = new_global_end;

        Ok(RingPtr {
            device_ptr,
            len_bytes: num_bytes,
            slab_idx,
            intra_offset: intra,
        })
    }

    /// Backward allocation — port of OT lines 90-109.
    fn alloc_backward_impl(&mut self, num_bytes: usize) -> Result<RingPtr> {
        if num_bytes == 0 {
            return Err(Error::InvalidInput(
                "RingAllocator: num_bytes must be > 0".into(),
            ));
        }
        if num_bytes > self.slab_bytes {
            return Err(Error::InvalidInput(format!(
                "RingAllocator: alloc({num_bytes}B) exceeds slab_bytes ({})",
                self.slab_bytes
            )));
        }

        // OT line 91-92: current slab and top-of-allocation within it.
        // `allocation_start` points at the first allocated byte (high
        // watermark of the free region from the top). The new allocation
        // sits BELOW it.
        //
        // Edge case: when `allocation_start == total_bytes` (initial state
        // after `new` or `reset`), `cur_slab_idx` would be `num_slabs`,
        // i.e., one past the last slab. Treat that as "we're about to
        // step into the last slab from above" — set `cur_intra` to
        // `slab_bytes` (top of last slab) and `cur_slab_idx` to last.
        let (cur_slab_idx, cur_intra) = if self.allocation_start == self.total_bytes {
            (self.slabs.len() - 1, self.slab_bytes)
        } else {
            let s = self.allocation_start / self.slab_bytes;
            let i = self.allocation_start % self.slab_bytes;
            (s, i)
        };

        // OT line 94-97: if it doesn't fit at the head of the current
        // slab, jump to the END of the previous slab.
        let (cand_slab_idx_signed, cand_intra_top) = if cur_intra < num_bytes {
            (cur_slab_idx as isize - 1, self.slab_bytes)
        } else {
            (cur_slab_idx as isize, cur_intra)
        };

        // OT line 98-101: cyclic wrap to the last slab if we walked
        // below slab 0.
        let wrapped = cand_slab_idx_signed < 0;
        let (slab_idx, intra_top) = if wrapped {
            // Refuse the wrap when forward is active — symmetric to the
            // forward path above. Backward wrap would silently lap a live
            // backward allocation in the last slab. Only permitted when
            // allocation_end == 0 (no forward state present).
            if self.allocation_end > 0 {
                return Err(Error::OutOfMemory(format!(
                    "RingAllocator exhausted: backward wrap with forward \
                     active would lap live allocations (allocation_start={}, \
                     allocation_end={}, total={}, slabs={}). Increase \
                     ring size or reset between steps.",
                    self.allocation_start,
                    self.allocation_end,
                    self.total_bytes,
                    self.slabs.len(),
                )));
            }
            (self.slabs.len() - 1, self.slab_bytes)
        } else {
            (cand_slab_idx_signed as usize, cand_intra_top)
        };

        // OT line 103: floor-16 the new low watermark.
        let new_intra = floor_16(intra_top - num_bytes);
        let new_global_start = slab_idx * self.slab_bytes + new_intra;

        // Invariant check: new backward low must not cross below the
        // forward high. The ring exhausted error.
        self.check_no_overlap_backward(new_global_start)?;

        self.ensure_slab(slab_idx)?;
        let device_ptr = self.slab_device_ptr(slab_idx, new_intra);

        // OT line 107-109: record the new start.
        self.allocation_start = new_global_start;

        Ok(RingPtr {
            device_ptr,
            len_bytes: num_bytes,
            slab_idx,
            intra_offset: new_intra,
        })
    }

    /// Forward bidirectional non-overlap check.
    ///
    /// Phase 1 semantics (per design doc §4 invariant 2 and §8 question 1):
    /// while we are still in the linear regime (forward hasn't wrapped
    /// past slab 0 yet), the new forward end must not exceed
    /// `allocation_start`. After a cyclic wrap (cand_slab_idx >=
    /// num_slabs), we currently allow the wrap unconditionally — this
    /// matches OT semantics. The microbench Test 3 covers the linear
    /// regime (where the invariant bites); wrap behavior is exercised
    /// only in Tests 4-5 where forward and backward are not active
    /// simultaneously after wrap.
    fn check_no_overlap_forward(&self, new_end: usize) -> Result<()> {
        // Only the simple linear case: backward hasn't moved from
        // top-of-ring AND forward stays below backward.
        // If allocation_start is at `total_bytes`, no backward has fired
        // — forward can grow freely up to total_bytes.
        if self.allocation_start < self.total_bytes && new_end > self.allocation_start {
            return Err(Error::OutOfMemory(format!(
                "RingAllocator exhausted: forward end {new_end} would cross \
                 backward start {} (total {} bytes across {} slabs)",
                self.allocation_start,
                self.total_bytes,
                self.slabs.len(),
            )));
        }
        Ok(())
    }

    /// Backward bidirectional non-overlap check.
    fn check_no_overlap_backward(&self, new_start: usize) -> Result<()> {
        // If allocation_end is at 0, no forward has fired — backward can
        // shrink freely down to 0.
        if self.allocation_end > 0 && new_start < self.allocation_end {
            return Err(Error::OutOfMemory(format!(
                "RingAllocator exhausted: backward start {new_start} would cross \
                 forward end {} (total {} bytes across {} slabs)",
                self.allocation_end,
                self.total_bytes,
                self.slabs.len(),
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Handle alloc surfaces
// ---------------------------------------------------------------------------

impl<'a> RingForwardHandle<'a> {
    /// Allocate `num_bytes` bytes in the forward direction (16-byte
    /// pre-aligned). See `RingAllocator` docs for the algorithm.
    pub fn alloc(&mut self, num_bytes: usize) -> Result<RingPtr> {
        self.alloc.alloc_forward_impl(num_bytes)
    }

    /// Current forward cursor (read-through to allocator). Useful for
    /// in-loop invariant checks while the handle is live.
    #[inline]
    pub fn allocation_end(&self) -> usize {
        self.alloc.allocation_end
    }

    /// Current backward cursor (read-through). The bidirectional
    /// invariant is `allocation_end <= allocation_start`.
    #[inline]
    pub fn allocation_start(&self) -> usize {
        self.alloc.allocation_start
    }
}

impl<'a> RingBackwardHandle<'a> {
    /// Allocate `num_bytes` bytes in the backward direction.
    pub fn alloc(&mut self, num_bytes: usize) -> Result<RingPtr> {
        self.alloc.alloc_backward_impl(num_bytes)
    }

    /// Current forward cursor (read-through).
    #[inline]
    pub fn allocation_end(&self) -> usize {
        self.alloc.allocation_end
    }

    /// Current backward cursor (read-through).
    #[inline]
    pub fn allocation_start(&self) -> usize {
        self.alloc.allocation_start
    }
}

// ---------------------------------------------------------------------------
// Unit tests (CPU-side reasoning, GPU touch via cudarc).
//
// End-to-end correctness lives in `tests/ring_alloc_microbench.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_16_is_correct() {
        assert_eq!(ceil_16(0), 0);
        assert_eq!(ceil_16(1), 16);
        assert_eq!(ceil_16(15), 16);
        assert_eq!(ceil_16(16), 16);
        assert_eq!(ceil_16(17), 32);
        assert_eq!(ceil_16(1023), 1024);
    }

    #[test]
    fn floor_16_is_correct() {
        assert_eq!(floor_16(0), 0);
        assert_eq!(floor_16(1), 0);
        assert_eq!(floor_16(15), 0);
        assert_eq!(floor_16(16), 16);
        assert_eq!(floor_16(31), 16);
        assert_eq!(floor_16(32), 32);
    }
}
