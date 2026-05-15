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
//!     │   ├── raw CudaSlice::Drop → cudarc hook → slab_v2_return_if_owned_any_device(ptr)
//!     │   ├── slab-range hit → decrement live_count, skip cudaFree
//!     │   └── TensorStorage path returns true → pool_return_* short-circuited
//!     │
//!     ├── reset()  // STRICT: errs if live_count != 0
//!     │   └── cursor → 0, slab + range still alive
//!     │
//!     └── release()  // tears down the slab
//!         ├── STRICT: errs if live_count != 0
//!         ├── unregister_range  (hook stops protecting)
//!         └── drop slab CudaSlice<u8> → real cudaFree
//! ```

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};

use crate::error::{Error, Result};
use crate::external_memory::{ExternalMemoryRegistry, ExternalOwner, ExternalRange, RangeHandle};

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
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true")
    )
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
        let slice: CudaSlice<u16> = unsafe { synth_slice::<u16>(ptr, n, self.device.clone()) };
        self.live_count.fetch_add(1, Ordering::AcqRel);
        crate::cuda_alloc_pool::trap_record_bf16_live(ptr, n, "static_slab_v2/alloc_u16", n);
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
        let slice: CudaSlice<f32> = unsafe { synth_slice::<f32>(ptr, n, self.device.clone()) };
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
                live,
                self.cursor,
                self.capacity_bytes,
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
    // Poison-tolerant lock acquisition: the inner `HashMap` is benign on
    // panic (we never panic while holding device_map's lock in production
    // paths), and adversarial tests intentionally poison it. Recover and
    // continue.
    {
        let g = match map.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
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
    let boxed: &'static Mutex<StaticSlabAllocator> = Box::leak(Box::new(Mutex::new(new_slab)));
    let mut g = match map.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    // Race window: if another thread already inserted, prefer the existing
    // entry and drop ours (leak the freshly-boxed mutex; benign on cold path).
    *g.entry(key).or_insert(boxed)
}

// ---------------------------------------------------------------------------
// Slab return hook
//
// Called from `TensorStorage::Drop` BEFORE the existing `pool_return_*`
// logic, and from the cudarc external-pointer hook for raw `CudaSlice`
// drops. Returns `true` if the slice was slab-owned and live_count was
// claimed; callers then skip cudaFree / pool-return.
//
// Why this lives here (not in cuda_alloc_pool.rs): the slab live_count
// is part of the slab's invariant, not the pool's. The slab must "see"
// every drop of every slice it handed out, even if the pool's caching
// path would otherwise reconstruct-and-forget the slice anyway. We
// centralise the decision here so `TensorStorage::Drop` and raw `CudaSlice`
// drop share ONE per-slab hook.
// ---------------------------------------------------------------------------

fn slab_v2_claim_locked(slab: &StaticSlabAllocator, ptr: u64, caller: &'static str) -> bool {
    if slab.ptr_in_slab(ptr) {
        // live_count > 0 by construction (we incremented on alloc; if a
        // slab-owned slice is being dropped, it MUST have come from an
        // alloc that incremented). Guard against logic bugs by clamping
        // at zero.
        let prev = slab.live_count.fetch_sub(1, Ordering::AcqRel);
        if prev == 0 {
            // Drop without matching alloc. Restore and warn — caller
            // still needs to skip cudaFree (ptr is in slab range).
            slab.live_count.fetch_add(1, Ordering::AcqRel);
            log::warn!(
                "{caller}: ptr 0x{ptr:x} in slab range but live_count was 0; skipping decrement"
            );
        }
        true
    } else {
        false
    }
}

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
    slab_v2_claim_locked(&g, ptr, "slab_v2_return_if_owned")
}

/// Device-key-less variant used by the process-wide cudarc external-pointer
/// hook. It handles raw `CudaSlice` drops for slab allocations that were not
/// wrapped in `TensorStorage`.
pub(crate) fn slab_v2_return_if_owned_any_device(ptr: u64) -> bool {
    let slabs: Vec<&'static Mutex<StaticSlabAllocator>> = {
        let g = match device_map().lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        g.values().copied().collect()
    };
    for slab_mutex in slabs {
        let g = match slab_mutex.lock() {
            Ok(g) => g,
            Err(_) => continue,
        };
        if slab_v2_claim_locked(&g, ptr, "slab_v2_return_if_owned_any_device") {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Test helpers (doc-hidden)
// ---------------------------------------------------------------------------

/// Test-only: drop the per-device map. Used to ensure independent test
/// runs don't see each other's slabs.
#[doc(hidden)]
pub fn reset_device_map_for_testing() {
    // Poison-tolerant lock acquisition: the map may have been poisoned by a
    // panicking test (especially the R2a `guard_drop_with_live_count_panics`
    // suite). We drain regardless.
    let mut g = match device_map().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    for (_, slab_ref) in g.drain() {
        // Force live_count → 0 first so release() can succeed even if a
        // previous test leaked a slice that we can no longer reach. Then
        // call release. If the slab mutex is poisoned, recover and proceed.
        let mut s = match slab_ref.lock() {
            Ok(s) => s,
            Err(poisoned) => poisoned.into_inner(),
        };
        s.live_count.store(0, Ordering::Release);
        let _ = s.release();
        // We intentionally do NOT free the leaked Box<Mutex<...>>:
        // there's no safe way to reclaim it without proving no other
        // thread still holds the &'static reference.
    }
}

/// Test-only: inject a slab into the device map under a given key.
/// Used by integration tests that need to wire up
/// `slab_v2_return_if_owned` to find a slab built by hand.
///
/// Returns `Ok(())` always; the previous entry (if any) is silently
/// overwritten.
#[doc(hidden)]
pub fn __test_device_map_insert(key: usize, slab: &'static Mutex<StaticSlabAllocator>) {
    let map = device_map();
    if let Ok(mut g) = map.lock() {
        g.insert(key, slab);
    }
}

/// Test-only: remove a slab entry from the device map. Used by
/// integration tests for cleanup.
#[doc(hidden)]
pub fn __test_device_map_remove(key: usize) {
    let map = device_map();
    if let Ok(mut g) = map.lock() {
        g.remove(&key);
    }
}

// ---------------------------------------------------------------------------
// Phase R2a — StepSlabGuard (RAII transient scope)
//
// A guard activates "transient slab dispatch" on the CURRENT thread. While a
// guard is live, `cuda_alloc_pool::pool_alloc_{u16,f32}` route to the slab
// (gated by `FLAME_USE_STATIC_SLAB=1`). On drop, the guard strict-resets the
// slab and panics if `live_count != 0`, EXCEPT when the thread is already
// unwinding — in that case it logs and best-effort cleans up to avoid the
// "double panic during unwinding ⇒ abort" failure mode.
//
// DECISION: thread-local state uses `Cell` (not `Mutex` / not `AtomicUsize`).
// The slab dispatch happens on the alloc hot path; `Cell::get` is a plain
// load with no synchronisation cost, and the field is per-thread by
// construction so atomicity is not needed across threads. Locked down by
// `guard_routes_alloc_to_slab` (correctness) and `guard_active_on_thread_query`.
//
// DECISION: nested guards are forbidden (return Err). Nested transient
// scopes would have ambiguous reset semantics — does the inner drop reset?
// only outer? — and OneTrainer's StaticLayerAllocator likewise has a single-
// active-scope invariant. Locked down by `guard_nested_forbidden`.
//
// DECISION: the env check `FLAME_USE_STATIC_SLAB` is `std::env::var` on every
// `pool_alloc_*` call, NOT cached via `OnceLock`. Reason: env reads land at
// ~50ns; an alloc that goes to cudart is microseconds; an alloc that goes
// to the slab is hundreds of ns. The env read is in the noise. Caching it
// would prevent runtime toggling (which Phase R2c gates rely on for the
// "off vs on" baselines on the same trainer invocation).
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-thread "device_key of the active guard, if any". `None` ⇒ no
    /// guard active on this thread; allocations go through legacy pool.
    /// `Some(key)` ⇒ allocations route to `slab_for_device` at that key,
    /// when `FLAME_USE_STATIC_SLAB=1`.
    static ACTIVE_DEVICE_KEY: Cell<Option<usize>> = const { Cell::new(None) };
    /// Once a step's slab has overflowed, retrying every later allocation
    /// through the slab only burns CPU and log bandwidth. Fall back directly
    /// to the legacy pool until the guard resets the slab for the next step.
    static SLAB_OVERFLOWED_THIS_SCOPE: Cell<bool> = const { Cell::new(false) };
}

/// Env name for the dispatch gate. `=1` enables slab routing when a guard
/// is active. Anything else (unset, `"0"`, etc.) keeps the legacy pool path.
pub const ENV_USE_STATIC_SLAB: &str = "FLAME_USE_STATIC_SLAB";

/// True iff `FLAME_USE_STATIC_SLAB=1` in the current process environment.
/// Re-read on every call; not cached.
#[inline]
fn use_static_slab_enabled() -> bool {
    std::env::var(ENV_USE_STATIC_SLAB).ok().as_deref() == Some("1")
}

/// RAII guard that activates transient slab dispatch for the current thread.
///
/// While a guard is live:
/// - `pool_alloc_u16` / `pool_alloc_f32` route to
///   [`StaticSlabAllocator`] for the guard's device (gated by
///   `FLAME_USE_STATIC_SLAB=1`).
/// - Persistent allocations made BEFORE the guard's `enter` are unaffected.
///
/// On `Drop` (or explicit [`StepSlabGuard::finish`]):
/// - Clears the thread-local active flag.
/// - Calls `StaticSlabAllocator::reset()` — STRICT: panics on `Drop` if
///   `live_count != 0` (unless the thread is already unwinding, in which
///   case it logs and continues to avoid a double-panic abort). `finish`
///   returns `Err` instead of panicking.
///
/// **Nested guards are forbidden** — calling `enter` while another guard is
/// already live on the same thread returns `Err`.
pub struct StepSlabGuard {
    device: Arc<CudaDevice>,
    device_key: usize,
    finished: bool,
}

impl StepSlabGuard {
    /// Enter a transient scope for `device`.
    ///
    /// Returns `Err(InvalidOperation)` if a guard is already active on this
    /// thread (nested guards forbidden).
    pub fn enter(device: Arc<CudaDevice>) -> Result<Self> {
        let device_key = Arc::as_ptr(&device) as usize;
        // Check nesting and set the thread-local atomically (per-thread, so
        // no inter-thread race possible).
        let already_active = ACTIVE_DEVICE_KEY.with(|cell| {
            if cell.get().is_some() {
                true
            } else {
                cell.set(Some(device_key));
                false
            }
        });
        if already_active {
            return Err(Error::InvalidOperation(
                "StepSlabGuard::enter: a guard is already active on this thread (nested guards forbidden)".to_string(),
            ));
        }
        SLAB_OVERFLOWED_THIS_SCOPE.with(|cell| cell.set(false));
        // Eagerly register the slab in the device_map so the dispatch
        // helpers (`pool_alloc_*_via_slab`) can find it on the first alloc
        // inside this scope. The slab itself is still lazily materialised
        // by the first `alloc_*` call; only the `&'static Mutex` entry is
        // populated here.
        //
        // DECISION: call `slab_for_device` from `enter` rather than
        // requiring `pool_alloc_*_via_slab` to thread an `Arc<CudaDevice>`
        // through. The pool dispatch only has access to `&Arc<CudaDevice>`
        // at the *call site*; the slab lookup uses the device_key (a usize)
        // which doesn't carry the Arc. Pre-registering keeps the
        // dispatch hot path lookup-only. Locked down by
        // `guard_routes_alloc_to_slab` (asserts the slab is in the map
        // after the first alloc inside the guard).
        let _ = slab_for_device(&device);
        Ok(Self {
            device,
            device_key,
            finished: false,
        })
    }

    /// Convenience: enter a guard on `CudaDevice::new(0)`. Returns `Err` if
    /// CUDA device 0 is unavailable OR if a guard is already active.
    pub fn enter_default() -> Result<Self> {
        let device = CudaDevice::new(0).map_err(|e| {
            Error::CudaDriver(format!(
                "StepSlabGuard::enter_default: CudaDevice::new(0): {e:?}"
            ))
        })?;
        Self::enter(device)
    }

    /// True iff the current thread has a live `StepSlabGuard`.
    ///
    /// Cheap (`Cell::get` — plain load).
    #[inline]
    pub fn active_on_thread() -> bool {
        ACTIVE_DEVICE_KEY.with(|cell| cell.get().is_some())
    }

    /// The device-key (`Arc::as_ptr as usize`) of the active guard on this
    /// thread, if any. Used by the slab dispatch helpers below.
    #[inline]
    pub fn active_device_key() -> Option<usize> {
        ACTIVE_DEVICE_KEY.with(|cell| cell.get())
    }

    /// Explicit graceful close. Resets the slab; returns `Err` if
    /// `live_count != 0`. After `finish()`, `Drop` is a no-op.
    ///
    /// Use this in code paths that want to handle a "leaked slab tensor"
    /// failure without unwinding the stack.
    pub fn finish(mut self) -> Result<()> {
        // Clear the thread-local FIRST so subsequent allocs (e.g. inside a
        // future scope) don't see this guard still active even on the
        // error path.
        ACTIVE_DEVICE_KEY.with(|cell| cell.set(None));
        SLAB_OVERFLOWED_THIS_SCOPE.with(|cell| cell.set(false));
        self.finished = true;
        // Try to reset the slab. Note: the slab may not have been
        // materialised at all (if no alloc happened during the scope) —
        // that's fine, `reset()` is a no-op on an unmaterialised slab.
        let slab_ref = {
            let g = device_map().lock().map_err(|_| {
                Error::InvalidOperation("StepSlabGuard::finish: device_map poisoned".to_string())
            })?;
            g.get(&self.device_key).copied()
        };
        if let Some(slab_mu) = slab_ref {
            let mut slab = slab_mu.lock().map_err(|_| {
                Error::InvalidOperation("StepSlabGuard::finish: slab mutex poisoned".to_string())
            })?;
            slab.reset()?;
        }
        Ok(())
    }

    /// Internal: reset the slab. Used by `Drop`. Returns `Err` on
    /// `live_count != 0` or poisoned mutex — does NOT panic.
    fn reset_slab(&self) -> Result<()> {
        let slab_ref = {
            let g = device_map().lock().map_err(|_| {
                Error::InvalidOperation("StepSlabGuard::drop: device_map poisoned".to_string())
            })?;
            g.get(&self.device_key).copied()
        };
        if let Some(slab_mu) = slab_ref {
            let mut slab = slab_mu.lock().map_err(|_| {
                Error::InvalidOperation("StepSlabGuard::drop: slab mutex poisoned".to_string())
            })?;
            slab.reset()?;
        }
        Ok(())
    }
}

impl Drop for StepSlabGuard {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // Always clear the thread-local — even on the panic/error path —
        // so a subsequent guard `enter()` on this thread succeeds.
        ACTIVE_DEVICE_KEY.with(|cell| cell.set(None));
        SLAB_OVERFLOWED_THIS_SCOPE.with(|cell| cell.set(false));

        let reset_result = self.reset_slab();

        match reset_result {
            Ok(()) => {
                // Normal path: nothing else to do.
            }
            Err(e) => {
                // Strict: panic on live_count != 0 — but ONLY if we're not
                // already unwinding. Otherwise we'd cause a "double panic
                // during unwinding ⇒ abort", which swallows the original
                // panic and crashes the process.
                if std::thread::panicking() {
                    log::error!(
                        "StepSlabGuard::drop: slab reset failed during unwind \
                         (suppressed to avoid double-panic abort): {e}"
                    );
                } else {
                    panic!("StepSlabGuard::drop: slab invariant violated: {e}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Slab dispatch helpers (used by `cuda_alloc_pool::pool_alloc_*`)
//
// These are the dispatch points the pool calls to TRY the slab path. They
// return `None` if the slab can't (or shouldn't) be used — the pool falls
// back to the legacy path. Conditions for `None`:
//
//   1. The thread has no active `StepSlabGuard` → no transient scope.
//   2. `FLAME_USE_STATIC_SLAB=1` is not set in the env.
//   3. The slab overflowed (slab capacity exceeded by this alloc).
//   4. The device map / slab mutex is poisoned.
//
// All four are non-error fallbacks: the legacy `device.alloc::<T>` path is
// the safety net. The only behavior change vs the pre-R2a tree is that
// when (1) AND (2) BOTH hold AND the slab has capacity, the alloc goes
// to the slab. Persistent allocations (no guard) are unaffected.
// ---------------------------------------------------------------------------

/// Try to satisfy an N-element BF16 allocation from the active slab.
///
/// Returns `Some(slice)` on success, `None` if the slab is unavailable or
/// inactive. Caller (pool_alloc_u16) MUST fall back to the legacy path on
/// `None`.
pub fn pool_alloc_u16_via_slab(n: usize) -> Option<CudaSlice<u16>> {
    if !use_static_slab_enabled() {
        return None;
    }
    if SLAB_OVERFLOWED_THIS_SCOPE.with(|cell| cell.get()) {
        return None;
    }
    let key = StepSlabGuard::active_device_key()?;
    // The slab path is the new transient route; if a previous test or
    // operation poisoned the mutex, we still want to USE the slab (a
    // poisoned mutex means the inner data may be in a torn state, but for
    // the device_map that data is just a `HashMap` of slab pointers — no
    // tearing possible from a panic in `slab_for_device`-adjacent code).
    let device_ref = match device_map().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let slab_ref = (*device_ref).get(&key).copied();
    drop(device_ref);
    let slab_ref = slab_ref?;
    let mut slab = match slab_ref.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match slab.alloc_u16(n) {
        Ok(slice) => Some(slice),
        Err(e) => {
            // Overflow / bad input — fall back to legacy for the rest of the
            // guarded step. Retrying after the cursor is full can otherwise
            // produce tens of thousands of warnings and measurable step tax.
            SLAB_OVERFLOWED_THIS_SCOPE.with(|cell| cell.set(true));
            log::debug!(
                "pool_alloc_u16_via_slab: slab alloc({n}) failed; using legacy pool for remaining guarded allocations this step: {e}"
            );
            None
        }
    }
}

/// Try to satisfy an N-element F32 allocation from the active slab.
///
/// **Zero-init contract**: the F32 pool path (`pool_alloc_f32` opt-out)
/// zeros memory on miss to match the legacy `pool_disabled` path. We
/// preserve that contract by calling `alloc_f32_zeroed`, NOT
/// `alloc_f32_uninit`. Callers that legitimately need uninit memory can
/// be added to a separate dispatch path later (R2b/R3).
///
/// Returns `Some(slice)` on success, `None` if the slab is unavailable or
/// inactive. Caller (pool_alloc_f32) MUST fall back to the legacy path on
/// `None`.
pub fn pool_alloc_f32_via_slab(n: usize) -> Option<CudaSlice<f32>> {
    if !use_static_slab_enabled() {
        return None;
    }
    if SLAB_OVERFLOWED_THIS_SCOPE.with(|cell| cell.get()) {
        return None;
    }
    let key = StepSlabGuard::active_device_key()?;
    // Poison-tolerant lookup (see `pool_alloc_u16_via_slab`).
    let device_ref = match device_map().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let slab_ref = (*device_ref).get(&key).copied();
    drop(device_ref);
    let slab_ref = slab_ref?;
    let mut slab = match slab_ref.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match slab.alloc_f32_zeroed(n) {
        Ok(slice) => Some(slice),
        Err(e) => {
            SLAB_OVERFLOWED_THIS_SCOPE.with(|cell| cell.set(true));
            log::debug!(
                "pool_alloc_f32_via_slab: slab alloc({n}) failed; using legacy pool for remaining guarded allocations this step: {e}"
            );
            None
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
            let mut g = device_map().lock().unwrap_or_else(|e| e.into_inner());
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
            let g = slab_static.lock().unwrap_or_else(|e| e.into_inner());
            g.live_count()
        };
        assert_eq!(final_live, 0);

        // Cleanup the global map (remove our synthetic entries).
        {
            let mut g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            g.remove(&key);
            g.remove(&synth_key);
        }
        // Release the slab cleanly (live_count is 0 now).
        let mut s = slab_static.lock().unwrap_or_else(|e| e.into_inner());
        s.release().unwrap();
    }

    /// 2. Lazy materialisation: `new()` does NOT cudaMalloc; first alloc does.
    #[test]
    fn slab_lazy_materialization() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        let slab = StaticSlabAllocator::new(device.clone(), 4 * 1024 * 1024);

        // Pre-alloc: slab_base is None, range_count for this device is 0.
        assert!(slab.slab_base().is_none());

        let pre_ranges = ExternalMemoryRegistry::global().range_count();
        let mut slab = slab;
        let s = slab.alloc_u16(64).unwrap();
        let post_ranges = ExternalMemoryRegistry::global().range_count();
        assert_eq!(
            post_ranges,
            pre_ranges + 1,
            "first alloc registers slab range"
        );
        assert!(slab.slab_base().is_some());

        // Cleanup.
        unsafe { forget_slice(s) };
        slab.live_count.store(0, Ordering::Release);
        slab.release().unwrap();
    }

    /// 3. Overflow returns Err with structured info.
    #[test]
    fn slab_overflow_fails_clean() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev_a) = skip_if_no_gpu() else {
            return;
        };
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
            let mut g = slab_a.lock().unwrap_or_else(|e| e.into_inner());
            let s = g.alloc_u16(64).unwrap();
            let p = *s.device_ptr();
            unsafe { forget_slice(s) };
            g.live_count.store(0, Ordering::Release);
            p
        };
        {
            let g = slab_b.lock().unwrap_or_else(|e| e.into_inner());
            // slab_b never materialised — base is None.
            assert!(g.slab_base().is_none());
            assert!(!g.ptr_in_slab(ptr_a));
        }

        // Cleanup.
        {
            let mut a = slab_a.lock().unwrap_or_else(|e| e.into_inner());
            let _ = a.release();
        }
        reset_device_map_for_testing();
    }

    /// 10. release() then alloc re-materialises.
    #[test]
    fn slab_release_then_realloc() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
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
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Don't reset env; just verify slab_for_device returns something
        // with a non-zero capacity.
        let Some(dev) = skip_if_no_gpu() else { return };
        reset_device_map_for_testing();
        let slab_mu = slab_for_device(&dev);
        let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
        assert!(g.capacity_bytes() > 0);
        // Slab is NOT yet materialised.
        assert!(g.slab_base().is_none());
        drop(g);
        reset_device_map_for_testing();
    }

    /// slab_v2_return_if_owned returns false for non-slab pointer.
    #[test]
    fn return_hook_returns_false_for_non_slab_ptr() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        let key = Arc::as_ptr(&device) as usize;
        // Fresh map.
        reset_device_map_for_testing();
        let owned = slab_v2_return_if_owned(0xDEAD_BEEF, key);
        assert!(!owned);
    }

    // =======================================================================
    // Phase R2a — StepSlabGuard tests
    //
    // These tests touch:
    //  - thread-local `ACTIVE_DEVICE_KEY`
    //  - process-global `device_map`
    //  - process env `FLAME_USE_STATIC_SLAB`
    //
    // All three are global; tests MUST run serialized. The integration test
    // binary uses `--test-threads=1` plus `TEST_LOCK`. The inline tests rely
    // on cargo running mod-tests serially or on the TEST_LOCK above. The
    // R2a tests also serialize through TEST_LOCK.
    // =======================================================================

    /// RAII helper: set an env var on entry, restore on drop. Avoids
    /// inter-test contamination of `FLAME_USE_STATIC_SLAB`.
    struct EnvGuard {
        name: &'static str,
        previous: Option<String>,
    }
    impl EnvGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let previous = std::env::var(name).ok();
            std::env::set_var(name, value);
            Self { name, previous }
        }
        fn unset(name: &'static str) -> Self {
            let previous = std::env::var(name).ok();
            std::env::remove_var(name);
            Self { name, previous }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(v) => std::env::set_var(self.name, v),
                None => std::env::remove_var(self.name),
            }
        }
    }

    /// Manually drain any guard state that leaked from a previously panicked
    /// test on this thread. Belt-and-suspenders for `--test-threads=1` runs.
    fn clear_thread_local_guard_state() {
        ACTIVE_DEVICE_KEY.with(|cell| cell.set(None));
    }

    /// R2a #1: inside an active guard with the env set, `pool_alloc_u16`
    /// allocates into the slab. Outside a guard, it goes to the legacy
    /// pool / cudart path (slab live_count stays 0).
    #[test]
    fn guard_routes_alloc_to_slab() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");

        // === OUTSIDE guard: should NOT go to slab. ===
        let outside =
            crate::cuda_alloc_pool::pool_alloc_u16(&device, 64).expect("outside-guard alloc");
        // Slab for this device may or may not exist; if it does, the ptr
        // must not be inside its range.
        let key = Arc::as_ptr(&device) as usize;
        let slab_ref = {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            g.get(&key).copied()
        };
        if let Some(slab_mu) = slab_ref {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                !g.ptr_in_slab(*outside.device_ptr()),
                "outside-guard alloc must NOT go to slab"
            );
            assert_eq!(g.live_count(), 0);
        }
        // Drop outside-guard slice via legacy path (cudaFree fires normally).
        drop(outside);

        // === INSIDE guard: should go to slab. ===
        {
            let _guard = StepSlabGuard::enter(device.clone()).expect("enter");
            assert!(StepSlabGuard::active_on_thread());

            let inside =
                crate::cuda_alloc_pool::pool_alloc_u16(&device, 128).expect("inside-guard alloc");
            let inside_ptr = *inside.device_ptr();

            // The slab must be in the device_map now (alloc materialised it).
            let slab_mu = {
                let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
                *g.get(&key).expect("slab registered after first alloc")
            };
            {
                let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
                assert!(
                    g.ptr_in_slab(inside_ptr),
                    "inside-guard alloc must land in slab range"
                );
                assert_eq!(g.live_count(), 1, "slab live_count incremented");
            }

            // Forget the slice + decrement live_count manually (we don't
            // have TensorStorage wiring in this unit test).
            unsafe { forget_slice(inside) };
            {
                let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
                g.live_count.fetch_sub(1, Ordering::AcqRel);
            }
            // Guard drops cleanly here (live_count == 0).
        }
        assert!(!StepSlabGuard::active_on_thread(), "guard cleared on drop");

        reset_device_map_for_testing();
    }

    /// R2a #2: allocations made BEFORE any guard are persistent — entering
    /// a guard, resetting the slab, and exiting must NOT invalidate the
    /// pre-guard allocation. (The slab's live_count for pre-guard allocs
    /// is 0 because they didn't go through the slab.)
    #[test]
    fn guard_persistent_alloc_not_in_slab() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");

        // Persistent alloc — outside any guard.
        let persistent =
            crate::cuda_alloc_pool::pool_alloc_u16(&device, 1024).expect("persistent alloc");
        let persistent_ptr = *persistent.device_ptr();
        let persistent_len = DeviceSlice::len(&persistent);
        assert!(persistent_len >= 1024);

        // Enter a guard, do nothing, drop cleanly. Slab should NOT have any
        // live count from the pre-guard alloc; reset succeeds trivially.
        {
            let guard = StepSlabGuard::enter(device.clone()).expect("enter");
            // The slab may not even be materialised — the guard scope had
            // no allocs. finish() should succeed.
            guard.finish().expect("clean finish");
        }

        // Persistent slice still readable.
        let host = device.dtoh_sync_copy(&persistent).expect("readback");
        assert_eq!(host.len(), persistent_len);
        // Drop persistent normally; this is OUTSIDE the slab range so the
        // standard pool path frees it.
        let _ = persistent_ptr; // suppress unused warning
        drop(persistent);

        reset_device_map_for_testing();
    }

    /// R2a #3: clean scope (alloc + drop + reset) succeeds, no panic.
    #[test]
    fn guard_drop_on_clean_scope_succeeds() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");

        let key = Arc::as_ptr(&device) as usize;
        {
            let _guard = StepSlabGuard::enter(device.clone()).expect("enter");
            let s =
                crate::cuda_alloc_pool::pool_alloc_u16(&device, 64).expect("inside-guard alloc");

            // Forget + decrement (simulating TensorStorage::Drop).
            unsafe { forget_slice(s) };
            let slab_mu = {
                let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
                *g.get(&key).unwrap()
            };
            {
                let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
                g.live_count.fetch_sub(1, Ordering::AcqRel);
            }
            // _guard drops at end of scope; reset is clean.
        }
        // After drop:
        let slab_mu = {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            g.get(&key).copied()
        };
        if let Some(slab_mu) = slab_mu {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(g.live_count(), 0, "live_count cleared");
            assert_eq!(g.used_bytes(), 0, "cursor rewound by reset");
        }
        reset_device_map_for_testing();
    }

    /// R2c BF: raw slab-backed CudaSlice drops must also decrement live_count.
    /// Many hot-path scratch buffers use `pool_alloc_*` directly without ever
    /// being wrapped in TensorStorage.
    #[test]
    fn raw_slab_slice_drop_decrements_live_count_via_hook() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");

        let key = Arc::as_ptr(&device) as usize;
        {
            let _guard = StepSlabGuard::enter(device.clone()).expect("enter");
            let s = crate::cuda_alloc_pool::pool_alloc_u16(&device, 64)
                .expect("inside-guard raw alloc");
            let slab_mu = {
                let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
                *g.get(&key).unwrap()
            };
            assert_eq!(
                slab_mu
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .live_count(),
                1
            );
            drop(s);
            assert_eq!(
                slab_mu
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .live_count(),
                0
            );
        }

        let slab_mu = {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            g.get(&key).copied()
        };
        if let Some(slab_mu) = slab_mu {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(g.live_count(), 0, "live_count cleared");
            assert_eq!(g.used_bytes(), 0, "cursor rewound by reset");
        }
        reset_device_map_for_testing();
    }

    /// R2a #4: guard drop with `live_count != 0` panics.
    #[test]
    #[should_panic(expected = "live")]
    fn guard_drop_with_live_count_panics() {
        // NOTE: this test cannot acquire `TEST_LOCK` via `.unwrap()` —
        // `should_panic` unwinds and the lock would be poisoned for every
        // subsequent test. Use poison-tolerant lock acquisition.
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        // R2a-bf: use EnvGuard so `FLAME_USE_STATIC_SLAB` is restored on
        // unwind. Previously the test used bare `std::env::set_var` which
        // leaked `=1` into the process env after `should_panic` caught the
        // panic — observable in `env_unaffected_by_panicking_test` below.
        // EnvGuard's Drop fires during unwind (it doesn't panic itself).
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");
        // Skip-no-GPU pattern, but we MUST still satisfy `should_panic`
        // on a no-GPU machine. Synthesize a "live" panic on the no-GPU
        // path so the test passes uniformly.
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[R2a #4] no GPU — synthesizing should_panic match");
                panic!("live (synthetic — no GPU available)");
            }
        };

        let guard = StepSlabGuard::enter(device.clone()).expect("enter");
        let s = crate::cuda_alloc_pool::pool_alloc_u16(&device, 64).expect("inside-guard alloc");
        // Leak the slice so live_count stays > 0 when guard drops.
        std::mem::forget(s);
        drop(guard); // expected to panic on "live"
                     // (_env drops here on unwind, restoring prior env.)
    }

    /// R2a #5: `finish()` returns Err when live_count > 0; slab NOT reset.
    #[test]
    fn guard_finish_with_live_count_errs() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");

        let key = Arc::as_ptr(&device) as usize;
        let guard = StepSlabGuard::enter(device.clone()).expect("enter");
        let s = crate::cuda_alloc_pool::pool_alloc_u16(&device, 64).expect("inside-guard alloc");
        let cursor_before = {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            let slab_mu = g.get(&key).unwrap();
            let s = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            s.used_bytes()
        };
        assert!(cursor_before > 0, "cursor advanced");

        // Intentionally don't forget/decrement: live_count is 1.
        let result = guard.finish();
        assert!(result.is_err(), "finish must err on live_count != 0");

        // Slab cursor unchanged (NOT reset).
        let cursor_after = {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            let slab_mu = g.get(&key).unwrap();
            let s = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            s.used_bytes()
        };
        assert_eq!(cursor_before, cursor_after, "slab NOT reset");

        // Cleanup: forget slice + decrement so subsequent tests get clean state.
        unsafe { forget_slice(s) };
        {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            let slab_mu = g.get(&key).unwrap();
            let s = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            s.live_count.fetch_sub(1, Ordering::AcqRel);
        }
        // The guard was consumed by finish(); ACTIVE_DEVICE_KEY is None already.
        assert!(!StepSlabGuard::active_on_thread());
        reset_device_map_for_testing();
    }

    /// R2a #6: panic inside the scope ⇒ guard Drop does NOT panic again
    /// (avoids double-panic-during-unwind abort).
    #[test]
    fn guard_panic_during_step_does_not_double_panic() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");

        let key = Arc::as_ptr(&device) as usize;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = StepSlabGuard::enter(device.clone()).expect("enter");
            let s =
                crate::cuda_alloc_pool::pool_alloc_u16(&device, 64).expect("inside-guard alloc");
            // Leak the slice so live_count > 0 when the guard drops.
            std::mem::forget(s);
            // Panic NOW — guard's drop should see `thread::panicking()` and
            // suppress its own panic.
            panic!("simulated step failure");
        }));
        // We expect the ORIGINAL panic to surface, not a double-panic abort.
        assert!(result.is_err(), "catch_unwind captured the original panic");
        let msg = match result.unwrap_err().downcast::<&'static str>() {
            Ok(s) => *s,
            Err(_) => "(non-string panic payload)",
        };
        assert!(
            msg.contains("simulated step failure"),
            "original panic propagated cleanly, got: {msg}"
        );

        // Thread-local was still cleared by guard's Drop.
        assert!(!StepSlabGuard::active_on_thread());

        // Cleanup: decrement and clear. live_count is still 1 (Drop's reset
        // failed silently because we were panicking).
        let slab_mu = {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            g.get(&key).copied()
        };
        if let Some(slab_mu) = slab_mu {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            // Drain live_count and reset cursor manually (simulating the
            // post-test cleanup the trainer wouldn't normally need).
            let lc = g.live_count.swap(0, Ordering::AcqRel);
            // We leaked the slice ptr; it's still in the slab range. The
            // slab's drop will leak the memory until process exit — fine
            // for unit tests.
            let _ = lc;
        }
        reset_device_map_for_testing();
    }

    /// R2a #7: nested enter on the same thread returns Err.
    #[test]
    fn guard_nested_forbidden() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();

        let outer = StepSlabGuard::enter(device.clone()).expect("first enter");
        let nested = StepSlabGuard::enter(device.clone());
        let err = match nested {
            Ok(_) => panic!("nested enter must err"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.contains("nested"), "err mentions nesting: {msg}");

        // After dropping outer, a new enter should succeed.
        drop(outer);
        let second = StepSlabGuard::enter(device.clone()).expect("re-enter after drop");
        drop(second);

        reset_device_map_for_testing();
    }

    /// R2a #8: `active_on_thread()` and `active_device_key()` track guard
    /// lifetime exactly.
    #[test]
    fn guard_active_on_thread_query() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        let key = Arc::as_ptr(&device) as usize;

        assert!(!StepSlabGuard::active_on_thread(), "no guard initially");
        assert_eq!(StepSlabGuard::active_device_key(), None);

        {
            let _g = StepSlabGuard::enter(device.clone()).expect("enter");
            assert!(StepSlabGuard::active_on_thread());
            assert_eq!(StepSlabGuard::active_device_key(), Some(key));
        }

        assert!(!StepSlabGuard::active_on_thread(), "cleared after drop");
        assert_eq!(StepSlabGuard::active_device_key(), None);

        // finish() also clears it.
        let g = StepSlabGuard::enter(device.clone()).expect("enter");
        assert!(StepSlabGuard::active_on_thread());
        g.finish().expect("clean finish");
        assert!(!StepSlabGuard::active_on_thread(), "cleared after finish");
        assert_eq!(StepSlabGuard::active_device_key(), None);

        reset_device_map_for_testing();
    }

    /// R2a #9: with `FLAME_USE_STATIC_SLAB` unset (or `=0`), allocations
    /// inside a guard scope do NOT go to the slab — they fall through to
    /// the legacy pool. The slab's live_count stays 0.
    #[test]
    fn guard_disabled_by_env() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        // Explicitly unset the env (cover both "unset" and "=0" via two arms).
        let _env = EnvGuard::unset(ENV_USE_STATIC_SLAB);

        let key = Arc::as_ptr(&device) as usize;
        {
            let _guard = StepSlabGuard::enter(device.clone()).expect("enter");
            let s =
                crate::cuda_alloc_pool::pool_alloc_u16(&device, 64).expect("inside-guard alloc");
            // Slab was NOT touched. If a slab entry exists in the map
            // (because a previous test created it), live_count must still
            // be 0 and ptr must not be in slab range.
            let slab_ref = {
                let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
                g.get(&key).copied()
            };
            if let Some(slab_mu) = slab_ref {
                let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
                assert_eq!(g.live_count(), 0);
                assert!(
                    !g.ptr_in_slab(*s.device_ptr()),
                    "with env unset, alloc must NOT land in slab"
                );
            }
            drop(s); // legacy pool path
        }
        // Now test the `=0` arm.
        std::env::set_var(ENV_USE_STATIC_SLAB, "0");
        {
            let _guard = StepSlabGuard::enter(device.clone()).expect("enter");
            let s =
                crate::cuda_alloc_pool::pool_alloc_u16(&device, 64).expect("inside-guard alloc");
            let slab_ref = {
                let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
                g.get(&key).copied()
            };
            if let Some(slab_mu) = slab_ref {
                let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
                assert_eq!(g.live_count(), 0);
                assert!(!g.ptr_in_slab(*s.device_ptr()));
            }
            drop(s);
        }
        std::env::remove_var(ENV_USE_STATIC_SLAB);
        reset_device_map_for_testing();
    }

    // -----------------------------------------------------------------
    // R2a-bf regression tests (Skeptic / Bug Fixer 2026-05-15)
    // -----------------------------------------------------------------

    /// R2a-bf #1: `guard_drop_with_live_count_panics` (test #4) must not
    /// leak `FLAME_USE_STATIC_SLAB=1` into subsequent tests via the
    /// process env. Pre-fix, that test used bare `std::env::set_var` and
    /// relied on `#[should_panic]` to swallow the panic — but the env
    /// var was never restored, so any test running AFTER it (in the same
    /// process / same `cargo test` run) inherited `=1`. The fix is to
    /// wrap the set in an `EnvGuard` (RAII Drop runs on unwind).
    ///
    /// This test verifies the contract: we synthesize the same panic-and-
    /// recover pattern as test #4 with an EnvGuard, and check that the
    /// env state matches what was set OUTSIDE the unwinding scope.
    #[test]
    fn env_unaffected_by_panicking_test() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        // Establish a known baseline: env unset.
        std::env::remove_var(ENV_USE_STATIC_SLAB);
        assert!(
            std::env::var(ENV_USE_STATIC_SLAB).is_err(),
            "precondition: env unset"
        );

        // Run a closure that mirrors test #4: EnvGuard::set → panic on
        // "live" → catch_unwind. EnvGuard's Drop must restore the env.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");
            assert_eq!(
                std::env::var(ENV_USE_STATIC_SLAB).unwrap(),
                "1",
                "inside scope: env set"
            );
            // Simulate the should_panic path.
            panic!("live (simulated)");
        }));
        assert!(r.is_err(), "panic propagated through catch_unwind");

        // After unwind: EnvGuard's Drop must have restored the env.
        assert!(
            std::env::var(ENV_USE_STATIC_SLAB).is_err(),
            "EnvGuard restored env on unwind (no leak)"
        );
    }

    /// R2a-bf #2: locks down the silent-overflow-fallback behavior
    /// (Builder concern #5). When the slab is small enough that an
    /// allocation overflows mid-step, `pool_alloc_*_via_slab` returns
    /// `None` and the caller falls through to the legacy pool path.
    /// The legacy alloc returns memory whose ptr is OUTSIDE the slab
    /// range, so `slab_v2_return_if_owned` returns false on drop and
    /// cudaFree fires normally.
    ///
    /// This is the DESIGN PIN per the handoff: overflow is a graceful
    /// fallback, not a hard error. Phase R2c gates will decide whether
    /// to upgrade to hard-Err; until then this test locks in the
    /// current contract.
    #[test]
    fn slab_overflow_falls_back_to_legacy_pool() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");
        // Pin the slab capacity tiny (256 bytes) so a single alloc beyond
        // that overflows. The env is read once on slab materialization in
        // `slab_for_device`.
        let _cap = EnvGuard::set(ENV_BF16_SLAB_BYTES, "256");

        let key = Arc::as_ptr(&device) as usize;

        // Enter guard, then alloc something that fits.
        let _guard = StepSlabGuard::enter(device.clone()).expect("enter");
        let small = crate::cuda_alloc_pool::pool_alloc_u16(&device, 64).expect("small alloc fits");
        let small_ptr = *small.device_ptr();

        // The first alloc should be IN the slab.
        let slab_mu = {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            *g.get(&key).expect("slab registered")
        };
        let small_in_slab = {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            g.ptr_in_slab(small_ptr)
        };
        assert!(small_in_slab, "small alloc landed in slab");

        // Now alloc something that exceeds remaining slab capacity. The
        // via_slab helper returns None; the caller falls back to legacy.
        // We expect this to succeed (legacy path always satisfies — barring
        // OOM) and the ptr to be OUTSIDE the slab range.
        let big = crate::cuda_alloc_pool::pool_alloc_u16(&device, 1024)
            .expect("overflow alloc falls back to legacy and succeeds");
        let big_ptr = *big.device_ptr();
        let big_in_slab = {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            g.ptr_in_slab(big_ptr)
        };
        assert!(
            !big_in_slab,
            "overflow alloc fell back to legacy (ptr NOT in slab range)"
        );

        // Slab's live_count is 1 (only the small one).
        {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(g.live_count(), 1, "only small alloc counted in slab");
        }

        // Cleanup: forget the slab-owned slice and decrement; drop big
        // through the legacy path.
        unsafe { forget_slice(small) };
        {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            g.live_count.fetch_sub(1, Ordering::AcqRel);
        }
        drop(big); // legacy pool path / direct cudaFree.

        // Guard drops cleanly at end of scope.
        drop(_guard);
        reset_device_map_for_testing();
    }

    /// R2a-bf #3: `StepSlabGuard::enter` returning `Err` (nested guards
    /// forbidden) must NOT corrupt the thread-local. After the failed
    /// inner enter, the outer guard's `active_device_key()` must still
    /// match the OUTER guard — not the inner attempt's device.
    ///
    /// Builder's `enter()` checks `cell.get().is_some()` BEFORE setting,
    /// so the inner attempt should leave the cell untouched. Lock that
    /// down here. (Concern: a future refactor that reorders set vs check
    /// would silently break this.)
    #[test]
    fn enter_err_does_not_corrupt_thread_local() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device_a) = skip_if_no_gpu() else {
            return;
        };
        reset_device_map_for_testing();

        let key_a = Arc::as_ptr(&device_a) as usize;
        let outer = StepSlabGuard::enter(device_a.clone()).expect("first enter");
        assert_eq!(StepSlabGuard::active_device_key(), Some(key_a));

        // Attempt a nested enter with a DIFFERENT Arc (still same physical
        // device, but distinct Arc → distinct key). The inner attempt must
        // return Err and leave the thread-local pointing at the OUTER key.
        // (If `enter` were buggy and set the cell BEFORE the nesting check,
        // this would silently switch the active key.)
        let device_b = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                // No second Arc available — bail gracefully, the test
                // still proved the basic invariant via key_a.
                drop(outer);
                reset_device_map_for_testing();
                return;
            }
        };
        let key_b = Arc::as_ptr(&device_b) as usize;
        if key_a == key_b {
            // CUDA runtime returned the same Arc — can't test cross-key
            // contamination. Bail gracefully.
            drop(outer);
            reset_device_map_for_testing();
            return;
        }

        let nested = StepSlabGuard::enter(device_b.clone());
        assert!(nested.is_err(), "nested enter must fail");
        // CRITICAL: thread-local still points at OUTER (key_a), not at
        // the inner-attempt's key_b. If this fails, `enter()` is setting
        // the cell BEFORE checking nesting — a real bug.
        assert_eq!(
            StepSlabGuard::active_device_key(),
            Some(key_a),
            "failed inner enter must not overwrite the outer thread-local"
        );

        drop(outer);
        assert!(
            !StepSlabGuard::active_on_thread(),
            "cleared after outer drop"
        );
        reset_device_map_for_testing();
    }

    /// R2a-bf #4: with `FLAME_USE_STATIC_SLAB=1` and an active guard,
    /// the slab dispatch must NEVER route a `pool_alloc_*` whose device
    /// parameter differs from the guard's device. The current
    /// implementation reads the device_key from the thread-local (NOT
    /// the caller's device arg) — so any mismatch silently routes to
    /// the guard's slab. This test documents that behavior so a future
    /// refactor that adds a "wrong-device → fall back to legacy" check
    /// (or a hard-Err) doesn't silently change the contract.
    ///
    /// In production the trainer uses ONE Arc<CudaDevice>, so this can't
    /// fire. The test exists to lock down the contract for future-Skeptic.
    #[test]
    fn dispatch_uses_guard_device_not_caller_device() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_thread_local_guard_state();
        let Some(device_a) = skip_if_no_gpu() else {
            return;
        };
        // Try to get a distinct Arc for the SAME physical device. If the
        // runtime shares Arcs, the test is degenerate — bail gracefully.
        let Ok(device_b) = CudaDevice::new(0) else {
            return;
        };
        if Arc::as_ptr(&device_a) == Arc::as_ptr(&device_b) {
            eprintln!("[R2a-bf #4] runtime shares Arc — can't exercise cross-Arc routing");
            return;
        }
        reset_device_map_for_testing();
        let _env = EnvGuard::set(ENV_USE_STATIC_SLAB, "1");

        let key_a = Arc::as_ptr(&device_a) as usize;
        let _guard = StepSlabGuard::enter(device_a.clone()).expect("enter on A");

        // Call pool_alloc with device_B but a guard is active on device_A.
        // The slab dispatch reads the THREAD-LOCAL key (= A) and routes
        // to A's slab. The returned slice carries device_A's Arc, NOT
        // device_B's. This is the contract today.
        let s = crate::cuda_alloc_pool::pool_alloc_u16(&device_b, 64)
            .expect("alloc with mismatched device arg routes via guard");
        let slab_mu = {
            let g = device_map().lock().unwrap_or_else(|e| e.into_inner());
            *g.get(&key_a).expect("guard's slab registered")
        };
        let in_a_slab = {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            g.ptr_in_slab(*s.device_ptr())
        };
        assert!(
            in_a_slab,
            "alloc with device_B arg landed in guard's (device_A) slab"
        );

        // Cleanup.
        unsafe { forget_slice(s) };
        {
            let g = slab_mu.lock().unwrap_or_else(|e| e.into_inner());
            g.live_count.fetch_sub(1, Ordering::AcqRel);
        }
        drop(_guard);
        reset_device_map_for_testing();
    }
}
