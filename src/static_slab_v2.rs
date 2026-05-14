//! Static slab allocator (Phase R1b of the OT static-slab redesign).
//!
//! ## What this is
//!
//! A **bump allocator** that carves transient training-step tensors out of
//! ONE big `CudaSlice<u8>` slab per (device, dtype). Each `alloc_*` call
//! bumps a cursor; each `reset()` rewinds to zero. The slab itself is
//! materialised lazily on the first `alloc_*` call and registered as a
//! single range with [`crate::external_memory::ExternalMemoryRegistry`], so
//! mid-slab pointers (from `narrow`/`view`/`permute`) are protected from
//! `cudaFree` by the cudarc Drop hook installed in R1a.
//!
//! ## Scope of R1b
//!
//! This module is the **primitive**. The transient-scope dispatch (i.e.
//! "when does `pool_alloc_u16` route here?") is Phase R2a's
//! [`StepSlabGuard`] — NOT in this file. R1b builds the allocator + the
//! per-device global accessor + the `TensorStorage::Drop` hook that
//! decrements `live_count` when a slab-owned slice drops. Nothing in the
//! trainer or in `pool_alloc_*` knows about this module yet.
//!
//! ## Reference
//!
//! OneTrainer's `StaticLayerAllocator` /
//! `LayerOffloadConductor.py:122-321` (Apache 2.0, citable). Same idea:
//! one big slab, bump cursor, reset between steps.
//!
//! ## Lifecycle
//!
//! ```text
//!   StaticSlabAllocator::new(device, capacity)       // cheap, no cudaMalloc
//!     │
//!     ├── alloc_u16 / alloc_f32_* (first call materialises slab + registers range)
//!     │   ├── increments live_count
//!     │   └── synthesises CudaSlice<T> at base+cursor
//!     │
//!     ├── (caller drops the slice)
//!     │   ├── TensorStorage::Drop → slab_v2_return_if_owned(ptr, dev_key)
//!     │   ├── slab-range hit → decrement live_count, forget slice (no cudaFree)
//!     │   └── return true → pool_return_* short-circuited
//!     │
//!     ├── reset()  // STRICT: errs if live_count != 0
//!     │   └── cursor → 0, slab + range still alive
//!     │
//!     └── release()  // tears down the slab
//!         ├── STRICT: errs if live_count != 0
//!         ├── unregister_range  (hook stops protecting)
//!         └── drop slab CudaSlice<u8> → real cudaFree
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};

use crate::error::{Error, Result};
use crate::external_memory::{
    ExternalMemoryRegistry, ExternalOwner, ExternalRange, RangeHandle,
};

// ---------------------------------------------------------------------------
// CudaSliceMirror — synthesize a CudaSlice<T> from raw parts.
//
// Same trick used in `cuda_alloc_pool.rs:29-34` and `offload/mod.rs:1050-1055`.
// Layout matches cudarc 0.11.x; if that version pin moves we need to revisit.
//
// SAFETY: this is `unsafe` to construct AND to transmute. Caller must
// guarantee the ptr is a valid device pointer with `len` elements of T.
// ---------------------------------------------------------------------------
struct CudaSliceMirror<T> {
    cu_device_ptr: u64,
    len: usize,
    device: Arc<CudaDevice>,
    host_buf: Option<std::pin::Pin<Vec<T>>>,
}

/// Synthesise a `CudaSlice<T>` at `(ptr, len, device)` without allocating.
///
/// # Safety
/// `ptr` must be a valid device pointer on `device` with at least `len`
/// elements of `T` allocated. The synthesized slice's `Drop` will call
/// `cudaFree(ptr)` — caller must either:
/// - Register `ptr` (or a range containing it) with `ExternalMemoryRegistry`
///   so the cudarc Drop hook skips the free, OR
/// - Eventually consume the slice via `forget_slice` (below).
unsafe fn synth_slice<T>(ptr: u64, len: usize, device: Arc<CudaDevice>) -> CudaSlice<T> {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: ptr,
        len,
        device,
        host_buf: None,
    };
    std::mem::transmute(mirror)
}

/// Consume a `CudaSlice<T>` without invoking its `Drop` (no `cudaFree`).
///
/// # Safety
/// Caller must guarantee the backing memory is owned by another allocator
/// (slab, ring) so leaking the slice doesn't strand memory.
unsafe fn forget_slice<T>(slice: CudaSlice<T>) {
    let mirror: CudaSliceMirror<T> = std::mem::transmute(slice);
    std::mem::forget(mirror);
}

// ---------------------------------------------------------------------------
// Env knobs
// ---------------------------------------------------------------------------

/// Default 4 GiB.
const DEFAULT_BF16_SLAB_BYTES: usize = 4 * 1024 * 1024 * 1024;
/// Default 4 GiB.
const DEFAULT_F32_SLAB_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Env name for BF16 (u16) slab capacity, in bytes.
pub const ENV_BF16_SLAB_BYTES: &str = "FLAME_STATIC_SLAB_BYTES_BF16";
/// Env name for F32 slab capacity, in bytes.
pub const ENV_F32_SLAB_BYTES: &str = "FLAME_STATIC_SLAB_BYTES_F32";
/// Env name for debug backtrace ring buffer.
pub const ENV_SLAB_DEBUG_BACKTRACE: &str = "FLAME_STATIC_SLAB_DEBUG_BACKTRACE";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str) -> bool {
    matches!(std::env::var(name).ok().as_deref(), Some("1") | Some("true"))
}

// ---------------------------------------------------------------------------
// Test-only memset counter
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) static MEMSET_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
fn record_memset() {
    MEMSET_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline(always)]
fn record_memset() {}

// ---------------------------------------------------------------------------
// 16-byte alignment
//
// DECISION: every cursor bump aligns to 16 bytes.
//
// Why 16:
// - CUDA's natural alignment for `cudaMalloc` is 256-byte boundaries; any
//   offset within that block is still well-aligned for f32/u16/bf16/u8
//   element access as long as the offset is at least element-aligned.
// - 16-byte alignment is sufficient for `float4` / vectorized 4xf32 loads
//   and is the largest alignment any NVRTC kernel in this crate assumes for
//   bulk-load-aligned pointers (search `aligned(16)` in the kernel sources).
// - Larger alignments (e.g. 256) would waste cursor space for small
//   allocations during a training step.
// - Stays consistent with the pool's `round_bytes_up` bucket boundary (which
//   uses 512-byte minimums), which means slab-allocated pointers will never
//   land at a less-aligned address than a fresh `cudaMalloc` of the same size.
//
// Locked down by `slab_alloc_advances_cursor` (checks alignment of returned ptrs).
// ---------------------------------------------------------------------------
const SLAB_ALIGN_BYTES: usize = 16;

#[inline]
fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// StaticSlabAllocator
// ---------------------------------------------------------------------------

/// One bump allocator per (device, dtype-family). Carves transient
/// training-step tensors out of a single `CudaSlice<u8>` slab.
///
/// **Lifecycle**:
/// - `new()` is cheap — does NOT call `cudaMalloc`.
/// - First `alloc_*` materialises the slab (`cudaMalloc(capacity_bytes)`),
///   registers `[base, base+capacity)` with [`ExternalMemoryRegistry`], and
///   ensures the cudarc Drop hook is installed.
/// - Each `alloc_*` bumps the cursor by an aligned size and synthesizes a
///   `CudaSlice<T>` pointing at `base + cursor`.
/// - `live_count` increments on alloc; the per-device drop hook
///   [`slab_v2_return_if_owned`] decrements it when a slab-owned slice flows
///   through `TensorStorage::Drop`.
/// - `reset()` rewinds the cursor — STRICT, errs if `live_count != 0`.
/// - `release()` tears down the slab — STRICT, errs if `live_count != 0`.
///
/// **NOT thread-safe by itself**: callers must serialize via the
/// `&'static Mutex<StaticSlabAllocator>` returned by [`slab_for_device`].
pub struct StaticSlabAllocator {
    device: Arc<CudaDevice>,
    device_key: usize,
    /// Lazily materialised. `Some` after the first `alloc_*`.
    slab: Option<CudaSlice<u8>>,
    /// Range registration handle. `Some` iff `slab.is_some()`.
    slab_handle: Option<RangeHandle>,
    /// Base device pointer of the materialised slab. `Some` iff `slab.is_some()`.
    slab_base: Option<u64>,
    capacity_bytes: usize,
    cursor: usize,
    live_count: AtomicUsize,
    debug_backtrace: bool,
}

impl StaticSlabAllocator {
    /// Construct an allocator. **Does NOT** call `cudaMalloc`. The slab is
    /// materialised lazily on the first `alloc_*` call.
    pub fn new(device: Arc<CudaDevice>, capacity_bytes: usize) -> Self {
        let device_key = Arc::as_ptr(&device) as usize;
        Self {
            device,
            device_key,
            slab: None,
            slab_handle: None,
            slab_base: None,
            capacity_bytes,
            cursor: 0,
            live_count: AtomicUsize::new(0),
            debug_backtrace: env_bool(ENV_SLAB_DEBUG_BACKTRACE),
        }
    }

    /// Currently-materialised base ptr of the slab (`None` until the first
    /// `alloc_*` call).
    pub fn slab_base(&self) -> Option<u64> {
        self.slab_base
    }

    /// Total slab capacity in bytes.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Bytes currently allocated (aligned-up sum of all `alloc_*` calls since
    /// the last `reset()`).
    pub fn used_bytes(&self) -> usize {
        self.cursor
    }

    /// Number of slices currently live (allocated and not yet dropped).
    pub fn live_count(&self) -> usize {
        self.live_count.load(Ordering::Acquire)
    }

    /// The device-key (`Arc::as_ptr as usize`) this allocator is bound to.
    pub fn device_key(&self) -> usize {
        self.device_key
    }

    /// Allocate `n` BF16 elements. The returned slice's memory is
    /// **uninitialised** (matches PT BFCAllocator + `pool_alloc_u16`).
    ///
    /// Returns `Err(Error::OutOfMemory)` on overflow with a structured
    /// message containing requested bytes, capacity, current cursor, dtype,
    /// and the env override name.
    pub fn alloc_u16(&mut self, n: usize) -> Result<CudaSlice<u16>> {
        // DECISION: n=0 is a no-op fast path — returns a zero-length slice
        // by allocating from cudart directly (NOT from the slab). live_count
        // is NOT incremented; cursor is NOT bumped. Rationale: a zero-length
        // CudaSlice from cudart drops cleanly without touching the hook, and
        // the slab live_count invariant stays simple. Locked down by
        // `slab_alloc_zero_elements`.
        if n == 0 {
            return unsafe { self.device.alloc::<u16>(0) }
                .map_err(|e| Error::CudaDriver(format!("alloc::<u16>(0): {e:?}")));
        }
        let bytes = n.checked_mul(std::mem::size_of::<u16>()).ok_or_else(|| {
            Error::InvalidInput(format!("alloc_u16: n*2 overflows usize (n={n})"))
        })?;
        let (base, offset) = self.bump_cursor(bytes, "BF16", ENV_BF16_SLAB_BYTES)?;
        let ptr = base + offset as u64;
        // SAFETY: `ptr` is in `[base, base+capacity)`, the range is registered
        // with ExternalMemoryRegistry, the cudarc Drop hook skips cudaFree
        // for ptrs in that range. The synthesized slice's drop path is
        // routed to `slab_v2_return_if_owned` via `TensorStorage::Drop`,
        // which decrements live_count.
        let slice: CudaSlice<u16> =
            unsafe { synth_slice::<u16>(ptr, n, self.device.clone()) };
        self.live_count.fetch_add(1, Ordering::AcqRel);
        Ok(slice)
    }

    /// Allocate `n` F32 elements. Memory is **NOT zero-initialised** —
    /// caller is responsible for initialisation. Matches `pool_alloc_f32`'s
    /// default opt-out semantics.
    pub fn alloc_f32_uninit(&mut self, n: usize) -> Result<CudaSlice<f32>> {
        // DECISION: same n=0 fast path as alloc_u16. Locked down by
        // `slab_alloc_zero_elements`.
        if n == 0 {
            return unsafe { self.device.alloc::<f32>(0) }
                .map_err(|e| Error::CudaDriver(format!("alloc::<f32>(0): {e:?}")));
        }
        let bytes = n.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            Error::InvalidInput(format!("alloc_f32_uninit: n*4 overflows usize (n={n})"))
        })?;
        let (base, offset) = self.bump_cursor(bytes, "F32", ENV_F32_SLAB_BYTES)?;
        let ptr = base + offset as u64;
        // SAFETY: see alloc_u16.
        let slice: CudaSlice<f32> =
            unsafe { synth_slice::<f32>(ptr, n, self.device.clone()) };
        self.live_count.fetch_add(1, Ordering::AcqRel);
        Ok(slice)
    }

    /// Allocate `n` F32 elements. Memory IS zero-initialised via
    /// `cudaMemsetAsync` (through cudarc's `memset_zeros`).
    pub fn alloc_f32_zeroed(&mut self, n: usize) -> Result<CudaSlice<f32>> {
        let slice = self.alloc_f32_uninit(n)?;
        if n > 0 {
            // SAFETY: we synthesise a *temporary* mutable mirror over the
            // same ptr+len, call memset_zeros, then forget the mirror so
            // its Drop doesn't run (would double-protect the same ptr via
            // the hook, harmless, but cleaner to forget). The actual
            // CudaSlice returned to the caller is `slice`, unchanged.
            let ptr = *slice.device_ptr();
            let dev = self.device.clone();
            let mut mirror: CudaSlice<f32> = unsafe { synth_slice::<f32>(ptr, n, dev) };
            self.device.memset_zeros(&mut mirror).map_err(|e| {
                Error::CudaDriver(format!("memset_zeros for alloc_f32_zeroed: {e:?}"))
            })?;
            unsafe { forget_slice(mirror) };
            record_memset();
        }
        Ok(slice)
    }

    /// Rewind the cursor to zero. STRICT: returns `Err` if `live_count != 0`.
    ///
    /// Does NOT free the slab. Subsequent `alloc_*` reuses the same backing
    /// memory.
    pub fn reset(&mut self) -> Result<()> {
        let live = self.live_count.load(Ordering::Acquire);
        if live != 0 {
            return Err(Error::InvalidOperation(format!(
                "StaticSlabAllocator::reset: refusing — {} live allocation(s) outstanding (cursor={}, capacity={}). \
                 Drop all slab-owned tensors before reset(). \
                 (Set {}=1 for alloc-site backtraces.)",
                live, self.cursor, self.capacity_bytes, ENV_SLAB_DEBUG_BACKTRACE,
            )));
        }
        self.cursor = 0;
        Ok(())
    }

    /// Tear down the slab. STRICT: returns `Err` if `live_count != 0`.
    ///
    /// Unregisters the range from [`ExternalMemoryRegistry`] BEFORE dropping
    /// the backing `CudaSlice<u8>`, so the real `cudaFree` fires (the hook
    /// will no longer protect ptrs in the released range).
    ///
    /// Subsequent `alloc_*` calls lazily re-materialise the slab.
    pub fn release(&mut self) -> Result<()> {
        let live = self.live_count.load(Ordering::Acquire);
        if live != 0 {
            return Err(Error::InvalidOperation(format!(
                "StaticSlabAllocator::release: refusing — {} live allocation(s) outstanding (cursor={}, capacity={}).",
                live, self.cursor, self.capacity_bytes,
            )));
        }
        // Order matters:
        // 1. Unregister the range FIRST so the hook stops protecting it.
        // 2. Drop the slab so cudaFree actually fires.
        if let Some(handle) = self.slab_handle.take() {
            ExternalMemoryRegistry::global().unregister_range(handle);
        }
        // Drop the slab CudaSlice<u8>. cudaFree will fire (range no longer
        // protected).
        self.slab = None;
        self.slab_base = None;
        self.cursor = 0;
        Ok(())
    }

    /// Internal: materialise the slab on demand. Idempotent — subsequent
    /// calls are a no-op.
    fn ensure_materialised(&mut self) -> Result<u64> {
        if let Some(base) = self.slab_base {
            return Ok(base);
        }
        // SAFETY: `device.alloc_zeros::<u8>(capacity_bytes)` returns a
        // freshly-allocated zero-initialised buffer. We zero-init the slab
        // so any subsequent allocation handed out via `alloc_f32_zeroed`
        // could in principle skip the memset for the FIRST pass (but we
        // don't bother — `alloc_f32_zeroed` always memsets to be safe across
        // resets).
        //
        // We choose `alloc_zeros` (vs `alloc`) for the slab body so that
        // any caller that *accidentally* reads uninitialised slab memory
        // (e.g. before a kernel has written it) sees zeros, not stale GPU
        // memory. This matches OneTrainer's `StaticLayerAllocator` which
        // memsets the slab on construction.
        let slab = self
            .device
            .alloc_zeros::<u8>(self.capacity_bytes)
            .map_err(|e| {
                Error::OutOfMemory(format!(
                    "StaticSlabAllocator::ensure_materialised: cudaMalloc({} bytes) failed: {:?}",
                    self.capacity_bytes, e,
                ))
            })?;
        let base = *slab.device_ptr();
        let end = base.saturating_add(self.capacity_bytes as u64);
        // Register the whole slab as a single range. The hook will skip
        // cudaFree for any ptr in `[base, end)`.
        let handle = ExternalMemoryRegistry::global().register_range(ExternalRange {
            start: base,
            end,
            device_key: self.device_key,
            owner: ExternalOwner::Slab,
        });
        // Ensure the cudarc Drop hook is installed (idempotent — first
        // caller wins, subsequent are no-ops).
        ExternalMemoryRegistry::ensure_hook_installed();

        self.slab = Some(slab);
        self.slab_handle = Some(handle);
        self.slab_base = Some(base);
        Ok(base)
    }

    /// Internal: bump the cursor by `bytes` (aligned up to `SLAB_ALIGN_BYTES`).
    /// Returns `(slab_base, aligned_offset_before_bump)`. Materialises the
    /// slab if needed.
    fn bump_cursor(
        &mut self,
        bytes: usize,
        dtype: &'static str,
        env_name: &'static str,
    ) -> Result<(u64, usize)> {
        let base = self.ensure_materialised()?;
        let aligned_bytes = align_up(bytes, SLAB_ALIGN_BYTES);
        // Cursor is always at an aligned address (initially 0, every bump
        // adds aligned_bytes which is a multiple of SLAB_ALIGN_BYTES).
        // Locked down by `slab_alloc_advances_cursor`.
        let offset = self.cursor;
        let new_cursor = offset.checked_add(aligned_bytes).ok_or_else(|| {
            Error::InvalidInput(format!(
                "StaticSlabAllocator::bump_cursor: cursor+aligned_bytes overflow (cursor={}, aligned_bytes={})",
                offset, aligned_bytes,
            ))
        })?;
        if new_cursor > self.capacity_bytes {
            return Err(Error::OutOfMemory(format!(
                "StaticSlabAllocator overflow: dtype={dtype} requested_bytes={bytes} aligned_bytes={aligned_bytes} cursor={offset} capacity={capacity} env_override={env_name}",
                capacity = self.capacity_bytes,
            )));
        }
        self.cursor = new_cursor;
        if self.debug_backtrace {
            // DECISION: ring buffer of last-N backtraces is a NIT in R1b
            // (the spec marks it "best-effort"). We log via `log::trace!`
            // with the call site instead — captures the equivalent
            // diagnostic info without the alloc churn of a Backtrace::new()
            // on every alloc. If the trainer ever needs the full backtrace
            // the env knob is still observable so we can wire a proper
            // ring buffer in a follow-up.
            log::trace!(
                "slab_alloc dtype={dtype} bytes={bytes} aligned={aligned_bytes} offset={offset} new_cursor={new_cursor} capacity={}",
                self.capacity_bytes,
            );
        }
        Ok((base, offset))
    }

    /// True if `ptr` falls inside this slab's `[base, base+capacity)` range.
    /// Used by [`slab_v2_return_if_owned`]; not part of the public spec.
    fn ptr_in_slab(&self, ptr: u64) -> bool {
        match self.slab_base {
            Some(base) => ptr >= base && ptr < base + (self.capacity_bytes as u64),
            None => false,
        }
    }
}

impl Drop for StaticSlabAllocator {
    /// Drop the allocator. If the slab has been materialised and there are
    /// no live allocations, this performs an orderly teardown.
    ///
    /// **WARNING**: if `live_count != 0` at drop time, we cannot panic
    /// (the slab may be inside a Mutex inside a process-wide map, and
    /// panicking on global teardown is the wrong shape). We log a `warn!`
    /// and leak the slab (no `cudaFree`); the cudarc Drop hook will still
    /// skip the free for any in-flight slices.
    fn drop(&mut self) {
        let live = self.live_count.load(Ordering::Acquire);
        if live != 0 {
            log::warn!(
                "StaticSlabAllocator::drop with live_count={} (cursor={}, capacity={}). \
                 Slab leaked; cudaFree skipped to avoid use-after-free.",
                live, self.cursor, self.capacity_bytes,
            );
            // Forget the slab to skip cudaFree. Range stays registered;
            // hook still protects any in-flight slices. This is a leak —
            // process exit reclaims it.
            if let Some(slab) = self.slab.take() {
                unsafe { forget_slice(slab) };
            }
            // Keep the range registration alive (intentional leak — the
            // alternative is unregistering and then immediately tripping
            // the hook on the slab's own drop).
            let _ = self.slab_handle.take();
            return;
        }
        // Orderly teardown: unregister range, then drop slab → real cudaFree.
        if let Some(handle) = self.slab_handle.take() {
            ExternalMemoryRegistry::global().unregister_range(handle);
        }
        // self.slab drops here (Option<CudaSlice<u8>>::None replaces it on
        // method exit; CudaSlice<u8>'s Drop calls cudaFree, now unprotected).
    }
}

// ---------------------------------------------------------------------------
// Per-device global accessor
//
// DECISION: storage strategy
//
// The spec says `slab_for_device(device) -> &'static Mutex<StaticSlabAllocator>`.
// For the `'static` lifetime to work, we leak the Mutex.
//
// Implementation:
// - A `OnceLock<Mutex<HashMap<usize, &'static Mutex<StaticSlabAllocator>>>>`
//   guards a map keyed by `Arc::as_ptr(device) as usize`.
// - On miss, we `Box::leak(Box::new(Mutex::new(StaticSlabAllocator::new(...))))`
//   to materialise a `&'static Mutex<...>` and stash it.
// - Capacity for the new slab is read from `FLAME_STATIC_SLAB_BYTES_BF16`
//   (default 4 GiB). Each `(device,)` pair gets ONE slab; the BF16/F32 split
//   is at the alloc-call level (`alloc_u16` vs `alloc_f32_*`), all coming out
//   of the same bump cursor.
//
// Rationale for shared cursor (vs spec's "per-device, per-dtype global"):
// re-reading the spec, R1b says "One bump allocator per (device, dtype)" —
// but the `slab_for_device` accessor signature is per-device only. The R2a
// `pool_alloc_u16` / `pool_alloc_f32` dispatch routes BOTH dtypes to the
// SAME accessor; if we kept two separate slabs we'd need two accessors. For
// R1b, one allocator per device is what the public API supports; the
// dtype-family slab capacity envs (`..._BF16` / `..._F32`) configure the
// PER-CALL alloc path semantics (BF16 alignment vs F32 alignment), not
// separate slabs. If R2a/R3 needs two slabs per device, that's a follow-up;
// the surface area we expose today doesn't preclude it (add a second
// accessor `slab_f32_for_device`).
//
// Locked down by `slab_multi_device_isolation`.
// ---------------------------------------------------------------------------

type DeviceMap = HashMap<usize, &'static Mutex<StaticSlabAllocator>>;

fn device_map() -> &'static Mutex<DeviceMap> {
    static MAP: OnceLock<Mutex<DeviceMap>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Per-device, process-wide accessor. Returns a `&'static Mutex` so callers
/// can stash it across step boundaries without lifetime gymnastics.
///
/// The first call for a given `Arc::as_ptr(device) as usize` materialises
/// a new [`StaticSlabAllocator`] with capacity from
/// `FLAME_STATIC_SLAB_BYTES_BF16` (default 4 GiB). The slab itself is NOT
/// allocated until the first `alloc_*` call.
///
/// **Identity contract**: two `Arc<CudaDevice>` handles for the same
/// physical device but obtained via separate `CudaDevice::new(0)` calls
/// have DIFFERENT `Arc::as_ptr` values, so they get DIFFERENT slabs. This
/// matches the pool's per-Arc identity rule (see
/// `cuda_alloc_pool.rs:387` note about per-Arc identity).
pub fn slab_for_device(device: &Arc<CudaDevice>) -> &'static Mutex<StaticSlabAllocator> {
    let key = Arc::as_ptr(device) as usize;
    let map = device_map();
    {
        let g = map.lock().expect("slab device_map poisoned");
        if let Some(slab) = g.get(&key) {
            return *slab;
        }
    }
    // Miss: build a new allocator. Use the BF16 env name for the capacity
    // since R1b has one slab per device (see DECISION above). Falling-back
    // to the F32 env when BF16 is unset lets callers configure either name
    // without breaking the other.
    let capacity_bytes = std::env::var(ENV_BF16_SLAB_BYTES)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| {
            std::env::var(ENV_F32_SLAB_BYTES)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or(DEFAULT_BF16_SLAB_BYTES.max(DEFAULT_F32_SLAB_BYTES));
    let new_slab = StaticSlabAllocator::new(device.clone(), capacity_bytes);
    let boxed: &'static Mutex<StaticSlabAllocator> =
        Box::leak(Box::new(Mutex::new(new_slab)));
    let mut g = map.lock().expect("slab device_map poisoned");
    // Race window: if another thread already inserted, prefer the existing
    // entry and drop ours (leak the freshly-boxed mutex; benign on cold path).
    *g.entry(key).or_insert(boxed)
}

// ---------------------------------------------------------------------------
// Slab return hook
//
// Called from `TensorStorage::Drop` BEFORE the existing `pool_return_*`
// logic. Returns `true` if the slice was slab-owned and live_count was
// decremented; the caller then SKIPS the rest of pool_return.
//
// Why this lives here (not in cuda_alloc_pool.rs): the slab live_count
// is part of the slab's invariant, not the pool's. The slab must "see"
// every drop of every slice it handed out, even if the pool's caching
// path would otherwise reconstruct-and-forget the slice anyway. We
// centralise the decision here so `TensorStorage::Drop` has ONE
// per-slab-tensor hook to call.
// ---------------------------------------------------------------------------

/// If `ptr` is owned by ANY slab on `device_key`, decrement that slab's
/// `live_count` and return `true`. The caller MUST then:
/// 1. Skip the rest of pool-return logic (do NOT call `pool_return_*`).
/// 2. `std::mem::forget` the `CudaSlice<T>` whose ptr we just claimed —
///    its memory is owned by the slab, not the slice. Letting the slice
///    drop normally is also OK (the cudarc hook will skip cudaFree
///    because the slab range is registered), but it costs a hook lookup
///    per drop. `forget` is the fast path.
///
/// Returns `false` if `ptr` is not slab-owned; caller continues with the
/// existing pool path.
///
/// # Arguments
/// - `ptr`: the device pointer being dropped.
/// - `device_key`: `Arc::as_ptr(&device) as usize` for the slice's device.
///
/// # Concurrency
/// Acquires the per-process device_map lock briefly, then the per-device
/// slab lock briefly. Safe under contention from concurrent `alloc_*` /
/// `slab_v2_return_if_owned` calls.
pub fn slab_v2_return_if_owned(ptr: u64, device_key: usize) -> bool {
    // Look up the slab for this device_key. We can't take the device_map
    // lock while ALSO holding a slab lock, so we copy the slab pointer out
    // first.
    let slab_ref: Option<&'static Mutex<StaticSlabAllocator>> = {
        let g = match device_map().lock() {
            Ok(g) => g,
            Err(_) => return false, // poisoned: best-effort, fall back to pool
        };
        g.get(&device_key).copied()
    };
    let Some(slab_mutex) = slab_ref else {
        return false;
    };
    // Lock the slab, check if the ptr falls in its range, decrement
    // live_count if so.
    let g = match slab_mutex.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if g.ptr_in_slab(ptr) {
        // live_count > 0 by construction (we incremented on alloc; if a
        // slab-owned slice is being dropped, it MUST have come from an
        // alloc that incremented). Guard against logic bugs by clamping
        // at zero.
        let prev = g.live_count.fetch_sub(1, Ordering::AcqRel);
        if prev == 0 {
            // Drop without matching alloc. Restore and warn — caller
            // still needs to skip cudaFree (ptr is in slab range).
            g.live_count.fetch_add(1, Ordering::AcqRel);
            log::warn!(
                "slab_v2_return_if_owned: ptr 0x{ptr:x} in slab range but live_count was 0; skipping decrement"
            );
        }
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Test helpers (doc-hidden)
// ---------------------------------------------------------------------------

/// Test-only: drop the per-device map. Used to ensure independent test
/// runs don't see each other's slabs.
#[doc(hidden)]
pub fn reset_device_map_for_testing() {
    if let Ok(mut g) = device_map().lock() {
        for (_, slab_ref) in g.drain() {
            // Try to release cleanly. If a slab still has live tensors,
            // release will error; we log and leak.
            if let Ok(mut s) = slab_ref.lock() {
                let _ = s.release();
            }
            // We intentionally do NOT free the leaked Box<Mutex<...>>:
            // there's no safe way to reclaim it without proving no other
            // thread still holds the &'static reference.
        }
    }
}

// ===========================================================================
// Tests
//
// Many tests require a CUDA device. We gate via FLAME_SKIP_GPU_TESTS=1 and
// also bail gracefully when `CudaDevice::new(0)` fails (CI sandbox, no GPU).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Process-wide test lock — slab + registry are global singletons.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    fn skip_if_no_gpu() -> Option<Arc<CudaDevice>> {
        if std::env::var("FLAME_SKIP_GPU_TESTS").ok().as_deref() == Some("1") {
            eprintln!("[slab tests] FLAME_SKIP_GPU_TESTS=1 — skipping");
            return None;
        }
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("[slab tests] CudaDevice::new(0) failed: {e:?} — skipping");
                None
            }
        }
    }

    /// Helper: build an allocator with a 16 MiB slab (small enough to be
    /// cheap, big enough for the test allocs).
    fn fresh_slab(device: Arc<CudaDevice>) -> StaticSlabAllocator {
        StaticSlabAllocator::new(device, 16 * 1024 * 1024)
    }

    /// 1. Allocate three BF16 tensors; cursor advances by aligned totals.
    #[test]
    fn slab_alloc_advances_cursor() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let mut slab = fresh_slab(device);

        // 1K BF16 elems = 2048 bytes (already 16-aligned).
        let s1 = slab.alloc_u16(1024).unwrap();
        assert_eq!(slab.live_count(), 1);
        assert_eq!(slab.used_bytes(), align_up(2048, SLAB_ALIGN_BYTES));

        // 2K BF16 elems = 4096 bytes.
        let s2 = slab.alloc_u16(2048).unwrap();
        assert_eq!(slab.live_count(), 2);
        assert_eq!(
            slab.used_bytes(),
            align_up(2048, SLAB_ALIGN_BYTES) + align_up(4096, SLAB_ALIGN_BYTES)
        );

        // 3K BF16 elems = 6144 bytes.
        let s3 = slab.alloc_u16(3072).unwrap();
        assert_eq!(slab.live_count(), 3);
        let total = align_up(2048, SLAB_ALIGN_BYTES)
            + align_up(4096, SLAB_ALIGN_BYTES)
            + align_up(6144, SLAB_ALIGN_BYTES);
        assert_eq!(slab.used_bytes(), total);

        // Pointers are increasing and 16-aligned.
        let p1 = *s1.device_ptr();
        let p2 = *s2.device_ptr();
        let p3 = *s3.device_ptr();
        assert!(p1 < p2 && p2 < p3);
        assert_eq!(p1 % SLAB_ALIGN_BYTES as u64, 0);
        assert_eq!(p2 % SLAB_ALIGN_BYTES as u64, 0);
        assert_eq!(p3 % SLAB_ALIGN_BYTES as u64, 0);

        // Drop slices; live_count goes to 0 (via TensorStorage::Drop? No —
        // these CudaSlices were synthesized OUTSIDE TensorStorage, so their
        // Drop runs cudarc's normal path → hook skips cudaFree → BUT no
        // slab_v2_return_if_owned is called. To decrement live_count we
        // must explicitly call the return hook.
        //
        // This is the integration the trainer (R2b) gets for free via
        // TensorStorage::Drop. At the unit-test level we exercise the
        // explicit return.
        let dev = s1.device().clone();
        let key = Arc::as_ptr(&dev) as usize;
        let len1 = DeviceSlice::len(&s1);
        let len2 = DeviceSlice::len(&s2);
        let len3 = DeviceSlice::len(&s3);
        // Forget the slices so cudarc's Drop doesn't fire (we'll route
        // through slab_v2_return_if_owned manually).
        unsafe {
            forget_slice(s1);
            forget_slice(s2);
            forget_slice(s3);
        }
        // Wire each ptr through slab_v2_return_if_owned. Need the per-device
        // map to see this slab — but `slab` is a local, not registered with
        // `slab_for_device`. To test the return hook we register manually.
        //
        // Quick trick: insert `slab` into the global map under a synthetic
        // key for the duration of this test. Use a different key per test
        // so independent tests don't interfere.
        let synth_key = 0xDEAD_BEEF_0001_usize;
        let slab_static: &'static Mutex<StaticSlabAllocator> =
            Box::leak(Box::new(Mutex::new(slab)));
        {
            let mut g = device_map().lock().unwrap();
            // Overwrite real key with our slab so slab_v2_return_if_owned
            // finds it. Save the old entry if present.
            g.insert(key, slab_static);
            g.insert(synth_key, slab_static); // dummy second insertion
        }
        // Suppress unused-len warnings — the new signature drops them.
        let _ = (len1, len2, len3);
        assert!(slab_v2_return_if_owned(p1, key));
        assert!(slab_v2_return_if_owned(p2, key));
        assert!(slab_v2_return_if_owned(p3, key));
        let final_live = {
            let g = slab_static.lock().unwrap();
            g.live_count()
        };
        assert_eq!(final_live, 0);

        // Cleanup the global map (remove our synthetic entries).
        {
            let mut g = device_map().lock().unwrap();
            g.remove(&key);
            g.remove(&synth_key);
        }
        // Release the slab cleanly (live_count is 0 now).
        let mut s = slab_static.lock().unwrap();
        s.release().unwrap();
    }

    /// 2. Lazy materialisation: `new()` does NOT cudaMalloc; first alloc does.
    #[test]
    fn slab_lazy_materialization() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let slab = StaticSlabAllocator::new(device.clone(), 4 * 1024 * 1024);

        // Pre-alloc: slab_base is None, range_count for this device is 0.
        assert!(slab.slab_base().is_none());

        let pre_ranges = ExternalMemoryRegistry::global().range_count();
        let mut slab = slab;
        let s = slab.alloc_u16(64).unwrap();
        let post_ranges = ExternalMemoryRegistry::global().range_count();
        assert_eq!(post_ranges, pre_ranges + 1, "first alloc registers slab range");
        assert!(slab.slab_base().is_some());

        // Cleanup.
        unsafe { forget_slice(s) };
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 3. Overflow returns Err with structured info.
    #[test]
    fn slab_overflow_fails_clean() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        // Tiny slab — 4 KiB.
        let mut slab = StaticSlabAllocator::new(device, 4 * 1024);
        // Fill 2 KiB.
        let s1 = slab.alloc_u16(1024).unwrap();
        // Request 4 KiB more — overflow.
        let err = slab.alloc_u16(2048).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("StaticSlabAllocator overflow"));
        assert!(msg.contains("dtype=BF16"));
        assert!(msg.contains(ENV_BF16_SLAB_BYTES));
        assert!(msg.contains("capacity="));
        assert!(msg.contains("cursor="));

        // Cleanup.
        unsafe { forget_slice(s1) };
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 4. reset() fails when live_count > 0; ptr is still valid (slab not torn down).
    #[test]
    fn slab_reset_with_live_allocation_fails() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let mut slab = fresh_slab(device);
        let s = slab.alloc_u16(128).unwrap();
        let ptr_before = *s.device_ptr();
        let err = slab.reset().unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("refusing"));
        assert!(msg.contains("live"));
        // ptr is still in the slab range (hook still protects).
        assert!(ExternalMemoryRegistry::global().should_skip_free_any_device(ptr_before));

        // Cleanup.
        unsafe { forget_slice(s) };
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 5. reset() after drops succeeds; cursor rewinds; next alloc starts at base.
    #[test]
    fn slab_reset_after_drop_succeeds() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let mut slab = fresh_slab(device);
        let s = slab.alloc_u16(1024).unwrap();
        let p1 = *s.device_ptr();
        // Drop the slice through the slab decrement path (test shim).
        unsafe { forget_slice(s) };
        slab.live_count.store(0, Ordering::Release);

        assert!(slab.reset().is_ok());
        assert_eq!(slab.used_bytes(), 0);
        assert_eq!(slab.live_count(), 0);

        // Next alloc returns ptr at base.
        let s2 = slab.alloc_u16(1024).unwrap();
        let p2 = *s2.device_ptr();
        assert_eq!(p1, p2, "after reset, next alloc returns slab base");

        // Cleanup.
        unsafe { forget_slice(s2) };
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 6. alloc_f32_zeroed produces zero-initialised memory.
    #[test]
    fn slab_f32_zeroed_reads_zeros() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let mut slab = fresh_slab(device.clone());

        let n = 256;
        let s = slab.alloc_f32_zeroed(n).unwrap();
        // Copy back to host.
        let host = device.dtoh_sync_copy(&s).unwrap();
        assert_eq!(host.len(), n);
        for (i, v) in host.iter().enumerate() {
            assert_eq!(*v, 0.0_f32, "slot {i} not zero: {v}");
        }

        // Cleanup.
        unsafe { forget_slice(s) };
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 7. alloc_f32_uninit does not invoke memset.
    #[test]
    fn slab_f32_uninit_is_fast() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let mut slab = fresh_slab(device);

        // Reset the counter (other tests may have run already).
        MEMSET_INVOCATIONS.store(0, Ordering::Release);
        let s = slab.alloc_f32_uninit(1024).unwrap();
        let count_after = MEMSET_INVOCATIONS.load(Ordering::Acquire);
        assert_eq!(count_after, 0, "uninit must NOT invoke memset");

        // Sanity: alloc_f32_zeroed DOES invoke memset.
        let s2 = slab.alloc_f32_zeroed(1024).unwrap();
        let count_after2 = MEMSET_INVOCATIONS.load(Ordering::Acquire);
        assert_eq!(count_after2, 1, "zeroed alloc invokes memset once");

        unsafe {
            forget_slice(s);
            forget_slice(s2);
        }
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 8. Mid-slab pointer is still protected by the hook (offset/narrow scenario).
    #[test]
    fn slab_hook_covers_offset_ptr() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let mut slab = fresh_slab(device);

        let s = slab.alloc_u16(1024).unwrap();
        let base = *s.device_ptr();
        // Simulate a narrow: ptr+1024 bytes inside the slab.
        let mid_ptr = base + 1024;
        assert!(
            ExternalMemoryRegistry::global().should_skip_free_any_device(mid_ptr),
            "mid-slab ptr must be protected by the slab range registration"
        );
        // Off-range ptr beyond capacity is NOT protected.
        let out_of_range = slab.slab_base().unwrap() + slab.capacity_bytes() as u64;
        assert!(
            !ExternalMemoryRegistry::global().should_skip_free_any_device(out_of_range),
            "ptr beyond slab range must NOT be protected"
        );

        unsafe { forget_slice(s) };
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 9. Multi-device isolation: different Arc handles → different slabs.
    #[test]
    fn slab_multi_device_isolation() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(dev_a) = skip_if_no_gpu() else { return };
        // Try to get a second `Arc<CudaDevice>` for the same physical
        // device. If the runtime returns the same Arc (refcount-shared),
        // we can't exercise the isolation invariant — bail gracefully.
        let Ok(dev_b) = CudaDevice::new(0) else {
            eprintln!("[slab tests] second CudaDevice::new(0) failed — skipping isolation");
            return;
        };
        if Arc::as_ptr(&dev_a) == Arc::as_ptr(&dev_b) {
            eprintln!("[slab tests] CudaDevice::new(0) returns shared Arc — can't test multi-device-key isolation");
            return;
        }
        let key_a = Arc::as_ptr(&dev_a) as usize;
        let key_b = Arc::as_ptr(&dev_b) as usize;
        assert_ne!(key_a, key_b);

        // Clean global map to ensure isolated test.
        reset_device_map_for_testing();

        let slab_a = slab_for_device(&dev_a);
        let slab_b = slab_for_device(&dev_b);
        // Different `&'static Mutex<...>` pointers.
        assert!(!std::ptr::eq(slab_a, slab_b));

        // Allocations from a's slab are NOT visible in b's.
        let ptr_a = {
            let mut g = slab_a.lock().unwrap();
            let s = g.alloc_u16(64).unwrap();
            let p = *s.device_ptr();
            unsafe { forget_slice(s) };
            g.live_count.store(0, Ordering::Release);
            p
        };
        {
            let g = slab_b.lock().unwrap();
            // slab_b never materialised — base is None.
            assert!(g.slab_base().is_none());
            assert!(!g.ptr_in_slab(ptr_a));
        }

        // Cleanup.
        {
            let mut a = slab_a.lock().unwrap();
            let _ = a.release();
        }
        reset_device_map_for_testing();
    }

    /// 10. release() then alloc re-materialises.
    #[test]
    fn slab_release_then_realloc() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let mut slab = fresh_slab(device);
        let s = slab.alloc_u16(64).unwrap();
        let base1 = slab.slab_base().unwrap();
        unsafe { forget_slice(s) };
        slab.live_count.store(0, Ordering::Release);
        // Release while live_count is 0.
        assert!(slab.release().is_ok());
        assert!(slab.slab_base().is_none());

        // Subsequent alloc re-materialises a NEW slab (likely at a different ptr).
        let s2 = slab.alloc_u16(64).unwrap();
        let base2 = slab.slab_base().unwrap();
        // We can't strongly assert base1 != base2 (cudart MAY reuse the
        // freed VA — and on some drivers it does). We assert that the
        // slab is materialised and the range is registered fresh.
        let _ = base1;
        assert_eq!(base2, *s2.device_ptr());
        assert!(ExternalMemoryRegistry::global().should_skip_free_any_device(base2));

        unsafe { forget_slice(s2) };
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 11. alloc_u16(0) is a no-op: no cursor bump, no live_count increment.
    #[test]
    fn slab_alloc_zero_elements() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let mut slab = fresh_slab(device);
        let s = slab.alloc_u16(0).unwrap();
        assert_eq!(DeviceSlice::len(&s), 0);
        // No bump, no live_count change.
        assert_eq!(slab.used_bytes(), 0);
        assert_eq!(slab.live_count(), 0);
        // Slab is NOT materialised on zero-element alloc — `n==0` short-
        // circuits before `ensure_materialised`.
        assert!(slab.slab_base().is_none());
        // Drop the zero-length cudart slice normally — no hook intercept needed.
        drop(s);

        // Same for f32.
        let s2 = slab.alloc_f32_uninit(0).unwrap();
        assert_eq!(DeviceSlice::len(&s2), 0);
        assert_eq!(slab.live_count(), 0);
        drop(s2);

        // alloc_f32_zeroed(0) also no-ops the memset.
        MEMSET_INVOCATIONS.store(0, Ordering::Release);
        let s3 = slab.alloc_f32_zeroed(0).unwrap();
        assert_eq!(DeviceSlice::len(&s3), 0);
        assert_eq!(MEMSET_INVOCATIONS.load(Ordering::Acquire), 0);
        drop(s3);
    }

    // --- Extra invariant tests (not in the 11-test spec, but lock down
    // edge cases the Bug Fixer would otherwise need to write themselves).

    /// align_up is correct for the alignments used in this module.
    #[test]
    fn align_up_unit() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(2048, 16), 2048);
    }

    /// Env knob is read by `slab_for_device`. We can't unset env vars in
    /// a multi-test process safely, but we CAN verify the constant + the
    /// default path returns a usable allocator.
    #[test]
    fn env_default_capacity_is_usable() {
        let _g = TEST_LOCK.lock().unwrap();
        // Don't reset env; just verify slab_for_device returns something
        // with a non-zero capacity.
        let Some(dev) = skip_if_no_gpu() else { return };
        reset_device_map_for_testing();
        let slab_mu = slab_for_device(&dev);
        let g = slab_mu.lock().unwrap();
        assert!(g.capacity_bytes() > 0);
        // Slab is NOT yet materialised.
        assert!(g.slab_base().is_none());
        drop(g);
        reset_device_map_for_testing();
    }

    /// slab_v2_return_if_owned returns false for non-slab pointer.
    #[test]
    fn return_hook_returns_false_for_non_slab_ptr() {
        let _g = TEST_LOCK.lock().unwrap();
        let Some(device) = skip_if_no_gpu() else { return };
        let key = Arc::as_ptr(&device) as usize;
        // Fresh map.
        reset_device_map_for_testing();
        let owned = slab_v2_return_if_owned(0xDEAD_BEEF, key);
        assert!(!owned);
    }
}
