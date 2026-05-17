//! Adapter that lets a `RingAllocator` serve as the cache-miss backend
//! for `flame_core::cuda_alloc_pool`.
//!
//! Phase 2a of the Gap 1 workstream from
//! [`docs/OFFLOAD_GAPS_vs_ONETRAINER.md`]. Phase 1 shipped the standalone
//! ring; this adapter is the opt-in glue that wires the ring as the
//! pool's miss-route allocator.
//!
//! ## Why an adapter, not direct integration
//!
//! `cuda_alloc_pool::pool_alloc_*` returns `CudaSlice<T>` whose `Drop`
//! must NOT call `cudaFree` when the backing memory is ring-owned. We
//! synthesize that via `CudaSliceMirror` transmute, then tag the pool's
//! free-list entries as `is_external: true` so `clear_cache` / pool drop
//! skip `cudaFree` on those entries. The pool side of the contract is in
//! `cuda_alloc_pool.rs`.
//!
//! ## Lifecycle
//!
//! 1. Trainer init: construct `RingAllocator`, wrap in
//!    `Arc::new(RingPoolAdapter::new(ring))`, call
//!    `cuda_alloc_pool::install_miss_allocator(adapter.clone())`.
//! 2. Per training step body runs normally.
//! 3. After `AutogradContext::clear()` at step boundary: call
//!    `cuda_alloc_pool::clear_pool_cache()` THEN `adapter.reset()`.
//!    Order matters — `clear_pool_cache` first drops all bucket entries
//!    that point into ring slabs (via the `is_external` tag, no cudaFree);
//!    `reset()` then puts the cursors back at extremes for the next step.
//! 4. Trainer shutdown: optionally `cuda_alloc_pool::uninstall_miss_allocator()`.
//!
//! ## Use-after-reset hazard (Skeptic Phase 1 CONCERN #2)
//!
//! Any `CudaSlice<T>` synthesized through this adapter is a borrowed view
//! into a ring slab. After `reset()`, future ring allocations will reuse
//! those slab bytes. **Holding a slice past reset is a use-after-free.**
//!
//! The pool's `clear_pool_cache` ensures all bucket free-list entries
//! pointing into ring slabs are dropped without `cudaFree`. Live tensors
//! (whose storage holds a not-yet-returned `CudaSlice<T>`) are the
//! caller's responsibility. In the Klein trainer:
//!
//! - Step-scope tensors die when `AutogradContext::clear()` releases the
//!   autograd graph's hold on activations + when the step body's locals
//!   go out of scope. Then `clear_pool_cache` + `reset()` is safe.
//! - Cross-step tensors (parameters, optimizer state, EMA weights) do NOT
//!   go through `pool_alloc_*` — they use `device.alloc_zeros::<T>`
//!   directly. So they never touch the ring.
//!
//! See `docs/FLAME_CONVENTIONS.md` "Ring allocator as pool backend".

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::cuda_alloc_pool::PoolMissAllocator;
use crate::ring_alloc::{RingAllocator, RingPtr};
use crate::{Error, Result};

// ---------------------------------------------------------------------------
// CudaSlice synthesis from RingPtr — mirror struct must match cudarc 0.11.x
// layout. This duplicates the mirror in `cuda_alloc_pool.rs`; both are pinned
// to cudarc 0.11.9 by the workspace Cargo.lock. Any cudarc bump must update
// both copies in lockstep.
// ---------------------------------------------------------------------------

struct CudaSliceMirror<T> {
    cu_device_ptr: u64,
    len: usize,
    device: Arc<CudaDevice>,
    host_buf: Option<std::pin::Pin<Vec<T>>>,
}

/// Synthesize a `CudaSlice<T>` of length `len` pointing at `ptr` on
/// `device`. The slice does NOT own the memory; dropping it WILL run
/// `cudaFree` on the pointer if the mirror layout is correct, so callers
/// MUST `mem::forget` or transmute-back-and-forget the result rather than
/// let it drop.
///
/// # Safety
///
/// `ptr` must be a live device pointer on `device` with at least
/// `len * size_of::<T>()` bytes of valid memory. The caller assumes
/// responsibility for the eventual `mem::forget`.
unsafe fn synth_slice<T>(ptr: u64, len: usize, device: Arc<CudaDevice>) -> CudaSlice<T> {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: ptr,
        len,
        device,
        host_buf: None,
    };
    std::mem::transmute(mirror)
}

// ---------------------------------------------------------------------------
// Adapter type
// ---------------------------------------------------------------------------

/// Adapter exposing a shared `RingAllocator` through the
/// `PoolMissAllocator` trait.
///
/// Wraps the ring in a `Mutex` so the adapter is `Send + Sync`; the
/// pool's miss-route is single-threaded per process today (one allocator
/// per trainer process), but the pool trait requires `Sync`.
///
/// Always allocates **forward**. The ring's bidirectional invariant is
/// not exploited at this stage — see Phase 2b notes in
/// `OFFLOAD_NEXT_GEN_DESIGN.md`.
pub struct RingPoolAdapter {
    /// The wrapped ring. `Mutex` makes the adapter `Sync`; pool callers
    /// acquire this briefly per miss.
    ring: Mutex<RingAllocator>,
}

impl RingPoolAdapter {
    /// Wrap a freshly-constructed `RingAllocator` so it can be installed
    /// as the pool's cache-miss backend.
    pub fn new(ring: RingAllocator) -> Self {
        Self {
            ring: Mutex::new(ring),
        }
    }

    /// Reset the wrapped ring's cursors. Slabs stay mapped. Call after
    /// `cuda_alloc_pool::clear_pool_cache()` at the step boundary.
    ///
    /// **Lifecycle contract**: all `CudaSlice<T>` views handed out by this
    /// adapter must be dropped or returned to the pool (which then
    /// reconstructs-and-forgets the mirror via the `is_external` tag)
    /// BEFORE `reset()` is called. Failure to do so is undefined behavior
    /// — the ring will hand out the same bytes to the next forward
    /// allocation.
    pub fn reset(&self) {
        if let Ok(mut ring) = self.ring.lock() {
            ring.reset();
        }
    }

    /// Snapshot inspection: number of slabs the ring has materialized so
    /// far (== `cuda_malloc_count`).
    pub fn slabs_allocated(&self) -> usize {
        self.ring.lock().map(|r| r.slabs_allocated()).unwrap_or(0)
    }

    /// Snapshot inspection: number of `cudaMalloc` calls the ring has
    /// issued. Should equal `slabs_allocated()` (one per slab, lazy).
    pub fn cuda_malloc_count(&self) -> u64 {
        self.ring.lock().map(|r| r.cuda_malloc_count()).unwrap_or(0)
    }

    /// Inspection: number of slabs configured (allocated or not).
    pub fn num_slabs(&self) -> usize {
        self.ring.lock().map(|r| r.num_slabs()).unwrap_or(0)
    }

    /// Inspection: per-slab byte capacity.
    pub fn slab_bytes(&self) -> usize {
        self.ring.lock().map(|r| r.slab_bytes()).unwrap_or(0)
    }
}

impl PoolMissAllocator for RingPoolAdapter {
    fn alloc_u16(&self, _device: &Arc<CudaDevice>, bucket_elems: usize) -> Result<CudaSlice<u16>> {
        let bytes = bucket_elems
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                Error::InvalidInput("RingPoolAdapter::alloc_u16: bucket_elems * 2 overflows".into())
            })?;
        let mut ring = self
            .ring
            .lock()
            .map_err(|_| Error::Cuda("RingPoolAdapter: ring mutex poisoned".into()))?;
        let RingPtr { device_ptr, .. } = ring.alloc_forward(bytes)?;
        let dev = ring.device().clone();
        // SAFETY: `device_ptr` points into a ring slab valid until the
        // next `reset()`. Caller (`pool_alloc_u16`) `mem::forget`s the
        // mirror via the `is_external` tag path.
        let slice = unsafe { synth_slice::<u16>(device_ptr, bucket_elems, dev) };
        Ok(slice)
    }

    fn alloc_f32(&self, _device: &Arc<CudaDevice>, _bucket_elems: usize) -> Result<CudaSlice<f32>> {
        // Phase 2a bug-fix (post-Builder): f32 routing DISABLED.
        //
        // The Builder's first commit (f82ab9b) attempted f32 routing through
        // the ring, made safe via a `cudarc-pinctx::install_external_ptr_hook`
        // that consults the pool's `external_ptrs` set to skip `cudaFree`
        // on ring-owned pointers. The 5-step Klein 9B smoke FALSIFIED that
        // approach: backward-graph teardown panics with
        // `CUDA_ERROR_INVALID_VALUE` inside `CudaSlice<f32>::drop`
        // (`/tmp/klein9b_phase2a_5step.log`).
        //
        // Root cause hypothesis: f32 cache-MISS slices that get sliced /
        // narrowed / cloned through flame-core's many `CudaSlice<f32>`
        // intermediaries (autograd_v3 gradient bufs, fused_linear3d_native
        // workspaces, broadcast metadata buffers, etc.) escape the
        // `external_ptrs` registration window — the slice's ptr is in the
        // set only when reconstructed from `decompose_slice` in
        // `pool_alloc_f32`, but cudarc internals may produce derived
        // `CudaSlice<f32>` values whose `cu_device_ptr` is an offset into
        // the same slab and is NOT in the set. Those derived slices then
        // call `cudaFree` on a mid-slab offset at drop time.
        //
        // The Builder's commit message documented the SAFE behavior
        // ("alloc_f32 deliberately returns Err and falls back to
        // device.alloc::<f32>"); the code shipped the unsafe behavior.
        // This is the alignment between commit-message intent and code.
        //
        // u16 routing stays ON: all `pool_alloc_u16` callers wrap the
        // slice in `TensorStorage::BF16 { data: Arc<CudaSlice<u16>> }`
        // and `TensorStorage::Drop` routes through `pool_return_u16`,
        // which respects the external tag. No derived-slice path exists.
        Err(Error::Cuda(
            "RingPoolAdapter::alloc_f32: f32 routing disabled (Phase 2a \
             smoke gate falsified ring-backed f32; fall back to \
             device.alloc::<f32>). See pool_adapter.rs for details."
                .into(),
        ))
    }
}
