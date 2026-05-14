//! Static slab + bump-cursor allocator. Port of OneTrainer's
//! `StaticLayerAllocator` / `StaticActivationAllocator` (see
//! `OneTrainer/modules/util/LayerOffloadConductor.py:122-321`).
//!
//! # Why
//!
//! flame-core's `cuda_alloc_pool` does per-tensor `cudaMallocAsync` +
//! `cudaFreeAsync` with a free-list cache layer on top. The cudart driver's
//! mempool internals accumulate state across ~hundreds of alloc/free pairs
//! per training step and at some point return a "valid-looking" ptr that the
//! same driver later rejects with `CUDA_ERROR_INVALID_VALUE` on the next
//! `cuMemcpy`/`cuMemset` — even after `cuStreamSynchronize`,
//! `cuCtxSynchronize`, and `bind_to_thread`. Compute-sanitizer flags
//! nothing. We've burned a session trying to chase this in-band; the
//! direct answer is to **stop using cudart's mempool in the hot path**.
//!
//! # Design
//!
//! - One `cudaMalloc::<u8>(slab_bytes)` per slab at construction time.
//! - `bump_*<T>(n)` returns a `CudaSlice<T>` view into the slab at the
//!   current cursor, advances the cursor by `align_up(n * sizeof T, 16)`.
//! - `reset()` sets cursor = 0. No `cudaFree` during the run.
//! - Slab `Drop` calls `cudaFree` exactly once on the backing storage.
//!
//! # Lifetime contract
//!
//! `bump_*` returns a `CudaSlice<T>` whose backing memory is owned by the
//! `StaticSlab`. The slice MUST NOT outlive the slab — caller must drop the
//! slice before `reset()` or before the slab itself drops. The slice's
//! `Drop` is suppressed via the global `EXTERNAL_PTR_HOOK` (registered by
//! [`install_slab_drop_hook`]) so dropping the slice doesn't `cudaFree`
//! the slab byte.
//!
//! Concretely:
//! 1. Construct slab → `cudaMalloc(bytes)`, register every byte's ptr range
//!    as "external" via [`register_slab_range`].
//! 2. `bump_f32(n)` → returns slice with cursor-offset ptr inside the slab.
//!    Slice's Drop hits hook → "is external" → skip cudaFree.
//! 3. `reset()` → cursor=0, slab memory unchanged. Slices made before reset
//!    that are still alive ALIAS new bump allocations — caller must not
//!    bump until all prior bump slices are dropped (this is the OT
//!    invariant: deallocate the layer before starting the next).
//! 4. `Drop for StaticSlab` → unregister the range, `cudaFree` the slab.

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

use crate::{Error, Result};

// ─────────────────────────────────────────────────────────────────────
// Global slab-range registry — the cudarc Drop hook consults this to
// decide whether to skip cudaFree on a CudaSlice.
//
// Each entry is (start_ptr, end_ptr) of a slab. A device ptr P is
// "owned by a slab" iff some entry has start <= P < end.
// ─────────────────────────────────────────────────────────────────────

struct SlabRange {
    start: u64,
    end: u64,
}

fn registry() -> &'static Mutex<Vec<SlabRange>> {
    static REG: OnceLock<Mutex<Vec<SlabRange>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Vec::new()))
}

/// Register a slab byte range as "external memory" so the cudarc
/// `CudaSlice::drop` hook skips `cudaFree` on any pointer that falls
/// within `[start, start+bytes)`. Called by `StaticSlab::new`.
fn register_slab_range(start: u64, bytes: usize) {
    if let Ok(mut r) = registry().lock() {
        r.push(SlabRange {
            start,
            end: start + bytes as u64,
        });
    }
}

fn unregister_slab_range(start: u64) {
    if let Ok(mut r) = registry().lock() {
        r.retain(|sr| sr.start != start);
    }
}

/// Hook for `cudarc::install_external_ptr_hook`. Returns true if `ptr`
/// belongs to any registered slab range (i.e. cudaFree must be skipped
/// on drop). Falls through to the cuda_alloc_pool's external-ptr set
/// for ptrs that belong to that allocator (e.g. ring-backed slot
/// tensors from `BlockOffloader::alloc_bf16_via_ring`).
fn slab_external_ptr_hook(ptr: u64) -> bool {
    // Slab range check first (cheaper, fewer entries).
    if let Ok(r) = registry().try_lock() {
        if r.iter().any(|sr| ptr >= sr.start && ptr < sr.end) {
            return true;
        }
    }
    // Fall through to cuda_alloc_pool's external-ptr set so the
    // BlockOffloader ring path keeps working.
    crate::cuda_alloc_pool::global_pool().is_external_ptr(ptr)
}

/// Install the slab hook. Idempotent; safe to call multiple times.
/// Composes with `cuda_alloc_pool::install_miss_allocator` — both hooks
/// (slab + pool) coexist via the hook's "first match wins" semantics in
/// cudarc-pinctx, but in practice we install only one of them at a time.
pub fn install_slab_drop_hook() {
    cudarc::driver::install_external_ptr_hook(slab_external_ptr_hook);
}

// ─────────────────────────────────────────────────────────────────────
// CudaSlice mirror — match cudarc 0.11.x's CudaSlice layout so we can
// transmute raw (ptr, len, device) into a CudaSlice<T> view.
// ─────────────────────────────────────────────────────────────────────

struct CudaSliceMirror<T> {
    cu_device_ptr: u64,
    len: usize,
    device: Arc<CudaDevice>,
    host_buf: Option<std::pin::Pin<Vec<T>>>,
}

unsafe fn synth_slice<T>(ptr: u64, len: usize, device: Arc<CudaDevice>) -> CudaSlice<T> {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: ptr,
        len,
        device,
        host_buf: None,
    };
    std::mem::transmute(mirror)
}

// ─────────────────────────────────────────────────────────────────────
// StaticSlab — single slab, bump cursor.
// ─────────────────────────────────────────────────────────────────────

#[inline]
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// A single GPU-side slab of `capacity` bytes with a bump cursor. Allocate
/// inside via `bump_f32` / `bump_u16`; reset the cursor with `reset`.
///
/// Caller is responsible for ensuring all slices handed out by this slab
/// are dropped before `reset` or `Drop`.
pub struct StaticSlab {
    /// Backing allocation. Kept alive for the slab's lifetime; dropped at
    /// `Drop`, which `cudaFree`s the whole block. We never touch the slab
    /// from Rust as a typed slice — we only need its device pointer.
    #[allow(dead_code)]
    storage: CudaSlice<u8>,
    /// Cached base device pointer (= `*storage.device_ptr()`).
    base_ptr: u64,
    /// Capacity in bytes.
    capacity: usize,
    /// Current bump cursor in bytes from `base_ptr`.
    cursor: usize,
    /// Device handle, cloned into each handed-out slice.
    device: Arc<CudaDevice>,
}

impl StaticSlab {
    /// Allocate a new slab of `capacity` bytes on `device`. One
    /// `cudaMalloc` call total, no further allocs during the slab's life.
    pub fn new(device: Arc<CudaDevice>, capacity: usize) -> Result<Self> {
        let storage = unsafe { device.alloc::<u8>(capacity) }
            .map_err(|e| Error::CudaDriver(format!("StaticSlab::new({capacity}): {e:?}")))?;
        let base_ptr = *storage.device_ptr();
        register_slab_range(base_ptr, capacity);
        Ok(Self {
            storage,
            base_ptr,
            capacity,
            cursor: 0,
            device,
        })
    }

    /// Bytes currently allocated.
    #[inline]
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity - self.cursor
    }

    /// Slab capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset the bump cursor. All slices previously returned from this
    /// slab become aliased to new bump allocations after this call —
    /// caller must ensure no prior slice is in use.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Bump-allocate `n` `f32` elements. Returns a `CudaSlice<f32>` view
    /// into the slab.
    pub fn bump_f32(&mut self, n: usize) -> Result<CudaSlice<f32>> {
        let bytes = n.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            Error::InvalidInput(format!("StaticSlab::bump_f32({n}): overflow"))
        })?;
        let aligned = align_up(bytes, 16);
        if self.cursor.checked_add(aligned).map_or(true, |end| end > self.capacity) {
            return Err(Error::InvalidInput(format!(
                "StaticSlab::bump_f32({n}): cursor {} + {} bytes > capacity {}",
                self.cursor, aligned, self.capacity
            )));
        }
        let ptr = self.base_ptr + self.cursor as u64;
        self.cursor += aligned;
        Ok(unsafe { synth_slice::<f32>(ptr, n, self.device.clone()) })
    }

    /// Bump-allocate `n` `u16` (BF16) elements.
    pub fn bump_u16(&mut self, n: usize) -> Result<CudaSlice<u16>> {
        let bytes = n.checked_mul(std::mem::size_of::<u16>()).ok_or_else(|| {
            Error::InvalidInput(format!("StaticSlab::bump_u16({n}): overflow"))
        })?;
        let aligned = align_up(bytes, 16);
        if self.cursor.checked_add(aligned).map_or(true, |end| end > self.capacity) {
            return Err(Error::InvalidInput(format!(
                "StaticSlab::bump_u16({n}): cursor {} + {} bytes > capacity {}",
                self.cursor, aligned, self.capacity
            )));
        }
        let ptr = self.base_ptr + self.cursor as u64;
        self.cursor += aligned;
        Ok(unsafe { synth_slice::<u16>(ptr, n, self.device.clone()) })
    }
}

impl Drop for StaticSlab {
    fn drop(&mut self) {
        // Unregister BEFORE `storage` drops — once unregistered, the
        // cudarc hook will see this ptr as non-external and `storage`'s
        // own Drop will call cudaFree on it (which is what we want for
        // the slab itself).
        unregister_slab_range(self.base_ptr);
        // `self.storage` drops here automatically and calls cudaFree.
    }
}

// ─────────────────────────────────────────────────────────────────────
// LoadScratchSlab — global slab for `Tensor::from_vec` callers in the
// sample-load hot path. Sized once on first use to fit the largest F32
// tensor we expect to load. Reset between training steps via
// `reset_load_scratch()`.
//
// This is the targeted fix for the Klein 9B step-2 INVALID_VALUE crash:
// the failing call was `cuMemcpyHtoDAsync` on a fresh `cuMemAllocAsync`
// ptr returned by the cudart mempool. Routing those allocs through a
// pre-allocated slab eliminates cudart's mempool from that path
// entirely.
// ─────────────────────────────────────────────────────────────────────

static LOAD_SCRATCH: OnceLock<Mutex<Option<StaticSlab>>> = OnceLock::new();

fn load_scratch() -> &'static Mutex<Option<StaticSlab>> {
    LOAD_SCRATCH.get_or_init(|| Mutex::new(None))
}

/// Bump-allocate `n` `f32` elements out of the global load-scratch slab,
/// growing the slab on first use OR on insufficient capacity. Drop of
/// the returned slice does NOT free; the slab is reset between training
/// steps by [`reset_load_scratch`].
///
/// Returns `Err` if the slab cannot fit `n` f32s even after growth.
pub fn bump_load_scratch_f32(
    device: &Arc<CudaDevice>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    install_slab_drop_hook();
    let bytes_needed = n * std::mem::size_of::<f32>();
    let aligned = align_up(bytes_needed, 16);

    let mut guard = load_scratch()
        .lock()
        .map_err(|_| Error::Training("load_scratch mutex poisoned".into()))?;

    // First-use: size the slab generously. 128 MB default — enough for
    // the largest text-embedding tensor we expect (24 MB) plus headroom.
    // Grows by 2× if a single bump exceeds capacity.
    if guard.is_none() {
        let initial = bytes_needed.max(128 * 1024 * 1024);
        *guard = Some(StaticSlab::new(device.clone(), initial)?);
    }

    // If a single alloc + existing cursor exceeds capacity, grow.
    {
        let slab = guard.as_ref().unwrap();
        if slab.cursor + aligned > slab.capacity {
            // Drop and reallocate. Caller must have ensured all prior
            // slices are gone (which the per-step reset pattern
            // guarantees — see `reset_load_scratch`).
            let new_cap = (slab.capacity.max(aligned) * 2).max(bytes_needed);
            *guard = None; // drop old slab (cudaFree)
            *guard = Some(StaticSlab::new(device.clone(), new_cap)?);
        }
    }

    let slab = guard.as_mut().unwrap();
    let slice = slab.bump_f32(n)?;
    // Also register the per-tensor ptr with the cuda_alloc_pool's external
    // ptr set so `pool_return_f32` tags the eventual FreeEntry as external
    // (and `clear_cache` calls `reconstruct_and_forget` instead of trying
    // to `cudaFree` a slab-offset ptr).
    crate::cuda_alloc_pool::global_pool()
        .register_external_ptr(*cudarc::driver::DevicePtr::device_ptr(&slice));
    Ok(slice)
}

/// Reset the load-scratch slab's bump cursor. Call at training step
/// boundary AFTER all tensors loaded via [`bump_load_scratch_f32`] have
/// been consumed (typically: after `latent` and `text_embedding` are
/// copied into BF16 / further-processed tensors, the F32 source views
/// are dropped, and the step proceeds).
///
/// Safe to call when slab is uninitialized (no-op).
pub fn reset_load_scratch() {
    if let Ok(mut guard) = load_scratch().lock() {
        if let Some(slab) = guard.as_mut() {
            slab.reset();
        }
    }
}

/// `FLAME_USE_LOAD_SCRATCH=1` opts F32 `Tensor::from_vec` into the slab
/// path. **Default OFF** until the slab-vs-autograd-saved-tensor
/// lifetime contract is solid — currently the slab views can be held as
/// `saved_tensors` in the autograd tape past the next step's reset,
/// which scrambles gradients. See `HANDOFF_2026-05-14_OT_PORT_PLAN.md`
/// (Phase A) for the design work that still needs to ship.
#[inline]
pub fn load_scratch_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("FLAME_USE_LOAD_SCRATCH").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}

// ─────────────────────────────────────────────────────────────────────
// GradScratchSlab — global slab for backward-pass F32 grad accumulators
// + transient F32 casts. Reset at end-of-step AFTER
// `AutogradContext::clear()` (same rule as LoadScratch: tape
// saved_tensors can hold references through backward).
//
// Initial size 1 GiB (Klein 9B's largest grad accumulator is ~256 MB
// for the qkv-proj fused weight; 1 GiB carries ~4× headroom). Grows 2×
// on overflow.
//
// This is the biggest Class-A win per `HEADTOHEAD_2026-05-12_ROOT_CAUSE.md`:
// ~9,500 bf16_to_f32 + ~9,700 f32_to_bf16 launches per step on klein 9B,
// each currently going through a fresh `pool_alloc_f32` → cudart mempool.
// Routing all of them through one pre-allocated slab eliminates the
// alloc/free churn entirely.
// ─────────────────────────────────────────────────────────────────────

static GRAD_SCRATCH: OnceLock<Mutex<Option<StaticSlab>>> = OnceLock::new();

fn grad_scratch() -> &'static Mutex<Option<StaticSlab>> {
    GRAD_SCRATCH.get_or_init(|| Mutex::new(None))
}

/// Initial size (bytes) for the GradScratch slab. Overridable via
/// `FLAME_GRAD_SCRATCH_BYTES`. Default 4 GiB — Klein 9B's full backward
/// fits comfortably under 2 GiB of transient F32 buffers; 4 GiB carries
/// 2× headroom so no growth is needed during a single backward pass.
///
/// Growth during a backward scope is UB (would drop a slab while
/// saved_tensors hold views into it). Sizing generously up-front is
/// the simple solution; chained-slab growth is future work.
fn grad_scratch_initial_bytes() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("FLAME_GRAD_SCRATCH_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4 * 1024 * 1024 * 1024)
    })
}

/// Bump-allocate `n` `f32` elements out of the global grad-scratch slab.
///
/// Pre-sized to `FLAME_GRAD_SCRATCH_BYTES` (default 4 GiB). Errors with
/// a clear message on overflow — DO NOT grow during a scope (would
/// dangle saved tensor views). Drop of the returned slice does NOT
/// free; the slab is reset at end-of-step via [`reset_grad_scratch`].
pub fn bump_grad_scratch_f32(
    device: &Arc<CudaDevice>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    install_slab_drop_hook();
    let bytes_needed = n * std::mem::size_of::<f32>();
    let aligned = align_up(bytes_needed, 16);

    let mut guard = grad_scratch()
        .lock()
        .map_err(|_| Error::Training("grad_scratch mutex poisoned".into()))?;

    if guard.is_none() {
        let initial = grad_scratch_initial_bytes().max(bytes_needed);
        *guard = Some(StaticSlab::new(device.clone(), initial)?);
    }

    {
        let slab = guard.as_ref().unwrap();
        if slab.cursor + aligned > slab.capacity {
            return Err(Error::InvalidInput(format!(
                "GradScratch slab overflow: cursor={} + {} bytes > capacity={}. \
                 Raise FLAME_GRAD_SCRATCH_BYTES (current={}). Growth during a \
                 backward scope is unsafe — see static_slab.rs for rationale.",
                slab.cursor, aligned, slab.capacity, grad_scratch_initial_bytes(),
            )));
        }
    }

    let slab = guard.as_mut().unwrap();
    let slice = slab.bump_f32(n)?;
    crate::cuda_alloc_pool::global_pool()
        .register_external_ptr(*cudarc::driver::DevicePtr::device_ptr(&slice));
    Ok(slice)
}

/// Reset the grad-scratch slab's bump cursor. Call at training step
/// boundary AFTER `AutogradContext::clear()` so the tape's saved_tensors
/// have been released first.
pub fn reset_grad_scratch() {
    if let Ok(mut guard) = grad_scratch().lock() {
        if let Some(slab) = guard.as_mut() {
            slab.reset();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// ActivationSlab — forward-pass intermediates (inter-layer tensors).
// Reset at end-of-step AFTER backward consumes them. Initial size 2 GiB
// (klein 9B's largest activation is ~1.2 GB at 1024² — headroom for
// re-use during recompute).
// ─────────────────────────────────────────────────────────────────────

static ACTIVATION_SLAB: OnceLock<Mutex<Option<StaticSlab>>> = OnceLock::new();

fn activation_slab() -> &'static Mutex<Option<StaticSlab>> {
    ACTIVATION_SLAB.get_or_init(|| Mutex::new(None))
}

fn activation_slab_initial_bytes() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("FLAME_ACTIVATION_SLAB_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4 * 1024 * 1024 * 1024)
    })
}

pub fn bump_activation_f32(
    device: &Arc<CudaDevice>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    install_slab_drop_hook();
    let bytes_needed = n * std::mem::size_of::<f32>();
    let aligned = align_up(bytes_needed, 16);

    let mut guard = activation_slab()
        .lock()
        .map_err(|_| Error::Training("activation_slab mutex poisoned".into()))?;

    if guard.is_none() {
        let initial = activation_slab_initial_bytes().max(bytes_needed);
        *guard = Some(StaticSlab::new(device.clone(), initial)?);
    }

    {
        let slab = guard.as_ref().unwrap();
        if slab.cursor + aligned > slab.capacity {
            return Err(Error::InvalidInput(format!(
                "Activation slab overflow: cursor={} + {} bytes > capacity={}. \
                 Raise FLAME_ACTIVATION_SLAB_BYTES (current={}).",
                slab.cursor, aligned, slab.capacity, activation_slab_initial_bytes(),
            )));
        }
    }

    let slab = guard.as_mut().unwrap();
    let slice = slab.bump_f32(n)?;
    crate::cuda_alloc_pool::global_pool()
        .register_external_ptr(*cudarc::driver::DevicePtr::device_ptr(&slice));
    Ok(slice)
}

pub fn reset_activation_slab() {
    if let Ok(mut guard) = activation_slab().lock() {
        if let Some(slab) = guard.as_mut() {
            slab.reset();
        }
    }
}

/// Reset ALL Phase B slabs in one call. Use at training step boundary
/// AFTER `AutogradContext::clear()`. Convenience wrapper.
pub fn reset_all_slabs() {
    reset_load_scratch();
    reset_grad_scratch();
    reset_activation_slab();
}

// ─────────────────────────────────────────────────────────────────────
// Phase B-1: Region enum + thread-local routing
//
// OneTrainer's LayerOffloadConductor carries an implicit "which layer/
// direction am I in" context — its allocator (`StaticLayerAllocator`)
// reads that context to pick the right cache_tensor. flame-core has no
// conductor object yet; we approximate the same effect by tagging hot-
// path callers with a `Region` via a thread-local + scope helper, and
// dispatching `pool_alloc_f32` to the matching slab.
//
// The five regions correspond to the OT pattern:
//   * LoadScratch  — sample-load + serialization F32 buffers (Phase A)
//   * Activation   — forward-pass intermediates (inter-layer tensors)
//   * GradScratch  — backward-pass grad accumulators + transient F32
//                    casts (the biggest Class-A win per HEADTOHEAD root
//                    cause: ~1.5 s/step of grad-storage F32 roundtrips
//                    + cast pairs on klein 9B)
//   * Layer        — bump-allocated layer weight scratch (matches OT's
//                    StaticLayerAllocator)
//   * None         — no scope set; legacy pool path
//
// Status: scaffolding only. The Region thread-local is here so that
// callers can opt in via `with_region(...)` scopes; `pool_alloc_f32`
// reads it via `try_bump_region_f32`. Default behavior is unchanged
// when `FLAME_REGION_DISPATCH` is unset OR no scope is active — the
// thread-local is `Region::None` everywhere, and `try_bump_region_f32`
// returns `Ok(None)` so the pool falls through to its legacy path.
// ─────────────────────────────────────────────────────────────────────

use std::cell::Cell;

/// Alloc region for routing F32 allocations to per-region slabs.
///
/// The order of variants matches the OT pattern's logical phases:
/// load → forward → backward → layer-weights → unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    /// No region set — `pool_alloc_f32` falls through to the legacy
    /// `cuda_alloc_pool` path. This is the default.
    None,
    /// Sample-load + serialization F32 buffers. Reset between steps via
    /// `reset_load_scratch`. Already wired via [`bump_load_scratch_f32`].
    LoadScratch,
    /// Forward-pass intermediates (activation tensors between layers).
    /// Reset between steps via `reset_activation_slab`.
    Activation,
    /// Backward-pass grad accumulators + transient F32 casts. Reset
    /// between steps via `reset_grad_scratch_slab`.
    GradScratch,
    /// Layer-weight scratch (matches OT's StaticLayerAllocator).
    /// Reset per-layer-direction (forward/backward bidir cursors).
    Layer,
}

thread_local! {
    /// Per-thread current region for F32 alloc routing.
    static CURRENT_REGION: Cell<Region> = const { Cell::new(Region::None) };
}

/// Read the current thread's alloc region. Default `Region::None`.
#[inline]
pub fn current_region() -> Region {
    CURRENT_REGION.with(|r| r.get())
}

/// Run `body` with the current thread's alloc region set to `region`.
/// On scope exit (normal return OR panic via the Drop guard), the prior
/// region is restored. Nested scopes save/restore the inner→outer chain.
///
/// Caller-side pattern (OT-equivalent of entering a layer's allocation
/// context):
/// ```ignore
/// with_region(Region::Activation, || {
///     // any pool_alloc_f32 inside here routes to the Activation slab
///     // (when FLAME_REGION_DISPATCH=1 + slab is initialized)
///     primitive_forward(...)
/// })
/// ```
pub fn with_region<R>(region: Region, body: impl FnOnce() -> R) -> R {
    struct Guard {
        prior: Region,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            CURRENT_REGION.with(|c| c.set(self.prior));
        }
    }
    let prior = CURRENT_REGION.with(|c| c.replace(region));
    let _guard = Guard { prior };
    body()
}

/// Master switch for the Phase B region-dispatch path. Default OFF.
/// When unset, [`try_bump_region_f32`] is a no-op (`Ok(None)`) — the
/// thread-local is consulted but no slab is created or used.
///
/// Setting `FLAME_REGION_DISPATCH=1` enables routing F32 allocs from
/// `pool_alloc_f32` into the matching region slab IF a `with_region`
/// scope is active AND that region has a slab wired.
#[inline]
pub fn region_dispatch_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("FLAME_REGION_DISPATCH").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}

/// Dispatch entry called from `pool_alloc_f32`. Returns:
/// * `Ok(Some(slice))` — routed to a region slab.
/// * `Ok(None)`        — region routing not applicable (master switch
///                       off, no scope active, or region has no slab
///                       wired yet). Caller falls through to legacy
///                       pool path.
/// * `Err(...)`        — slab routing was attempted but failed (e.g.
///                       OOM during slab growth). Caller may surface
///                       or fall through depending on policy; current
///                       policy is to fall through (slab errors should
///                       not crash a trainer that would otherwise work
///                       on the legacy path).
pub fn try_bump_region_f32(
    device: &Arc<CudaDevice>,
    size: usize,
) -> Result<Option<CudaSlice<f32>>> {
    if !region_dispatch_enabled() {
        return Ok(None);
    }
    match current_region() {
        Region::None => Ok(None),
        // Phase A handles LoadScratch separately via Tensor::from_vec.
        // If a caller explicitly enters a LoadScratch scope, route here
        // for consistency.
        Region::LoadScratch => bump_load_scratch_f32(device, size).map(Some),
        Region::Activation => bump_activation_f32(device, size).map(Some),
        Region::GradScratch => bump_grad_scratch_f32(device, size).map(Some),
        // Layer slab is OT's StaticLayerAllocator analog — not yet
        // implemented because the BlockOffloader already has its own
        // 2-slot ping-pong path that mostly fills this role.
        Region::Layer => Ok(None),
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;

    #[test]
    fn current_region_defaults_to_none() {
        assert_eq!(current_region(), Region::None);
    }

    #[test]
    fn with_region_sets_and_restores() {
        assert_eq!(current_region(), Region::None);
        with_region(Region::GradScratch, || {
            assert_eq!(current_region(), Region::GradScratch);
        });
        assert_eq!(current_region(), Region::None);
    }

    #[test]
    fn with_region_nests_correctly() {
        with_region(Region::Activation, || {
            assert_eq!(current_region(), Region::Activation);
            with_region(Region::GradScratch, || {
                assert_eq!(current_region(), Region::GradScratch);
            });
            assert_eq!(current_region(), Region::Activation);
        });
        assert_eq!(current_region(), Region::None);
    }

    #[test]
    fn with_region_restores_on_panic() {
        let _ = std::panic::catch_unwind(|| {
            with_region(Region::GradScratch, || {
                assert_eq!(current_region(), Region::GradScratch);
                panic!("test panic");
            });
        });
        assert_eq!(current_region(), Region::None);
    }
}
