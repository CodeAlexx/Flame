//! CUDA caching allocator for flame-core.
//!
//! Eliminates per-op `cudaMalloc`/`cudaFree` during backward by maintaining
//! power-of-2 bucketed free lists of GPU memory. Same strategy as PyTorch's
//! `CUDACachingAllocator`, simplified for single-device use.
//!
//! Integration: [`alloc_aligned_f32`](crate::cuda_memory_alignment::alloc_aligned_f32)
//! routes through [`pool_alloc_f32`], and [`Tensor::drop`](crate::tensor::Tensor)
//! returns slices via [`pool_return_f32`].

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Mirror struct matching cudarc 0.11.x CudaSlice<T> layout.
//
// CudaSlice<T> is:
//   cu_device_ptr: CUdeviceptr (u64),
//   len: usize,
//   device: Arc<CudaDevice>,
//   host_buf: Option<Pin<Vec<T>>>,
//
// We reconstruct CudaSlice from raw parts via transmute.  This is safe as
// long as the struct layout hasn't changed (pinned to cudarc 0.11.9).
// ---------------------------------------------------------------------------
// Must NOT be #[repr(C)] — must match CudaSlice's default Rust layout.
struct CudaSliceMirror<T> {
    cu_device_ptr: u64,
    len: usize,
    device: Arc<CudaDevice>,
    host_buf: Option<std::pin::Pin<Vec<T>>>,
}

/// Entry stored in the free list — raw device pointer + metadata.
struct FreeEntry {
    ptr: u64,
    len: usize, // element count (f32 elements, not bytes)
    device: Arc<CudaDevice>,
}

/// Cached env check for FLAME_PROFILE=1.
#[inline]
fn profiling_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_PROFILE").ok().as_deref() == Some("1"))
}

/// Cached env check for FLAME_ALLOC_POOL=0 (disable pool).
#[inline]
pub fn pool_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_ALLOC_POOL").ok().as_deref() == Some("0"))
}

/// Cached env check for FLAME_F32_ZERO_INIT=1 (opt-in: restore legacy
/// behavior where pool_alloc_f32 zero-initializes the buffer on cache miss).
///
/// **Default = OFF (uninitialized).** This matches the BF16 path
/// (`pool_alloc_u16` already returns uninitialized memory) and PyTorch's
/// BFCAllocator semantics: callers are responsible for initialization.
///
/// Audit (2026-05-12): every caller of `alloc_aligned_f32` /
/// `pool_alloc_f32` in flame-core either (a) explicitly memsets afterward
/// via `alloc_zeros_from_pool` / `TensorStorage::zeros` (their previous
/// implicit zero was redundant), or (b) fully overwrites the buffer via
/// `dtod_copy` / `htod_copy_into` / a kernel that writes every element.
/// The implicit zero-init was wasted work in every case.
///
/// Set `FLAME_F32_ZERO_INIT=1` to revert to legacy behavior if a hidden
/// caller is discovered.
#[inline]
fn f32_zero_init_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_F32_ZERO_INIT").ok().as_deref() == Some("1"))
}

// ---------------------------------------------------------------------------
// Pool statistics
// ---------------------------------------------------------------------------
#[derive(Debug)]
pub struct PoolStats {
    pub alloc_count: usize,
    pub reuse_count: usize,
    pub return_count: usize,
    pub peak_bytes: usize,
    pub current_cached_bytes: usize,
    pub current_cached_entries: usize,
}

// ---------------------------------------------------------------------------
// CudaAllocPool — the global caching allocator
// ---------------------------------------------------------------------------

/// Maximum bucket size: 2 GiB (2^31 bytes = 536_870_912 f32 elements).
const MAX_POOL_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// Maximum elements per size-class free list to prevent unbounded growth.
const MAX_FREE_PER_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// PyTorch-style size-bucket rounding.
//
// Mirrors `/home/alex/pytorch/c10/core/AllocatorConfig.h:13-24` and
// `c10/cuda/CUDACachingAllocator.cpp::round_size` / `::get_allocation_size`.
// Without rounding, every unique-size temporary (broadcast scratch, narrow
// scratch, KV intermediate, seq-len-dependent shapes) misses the pool and
// hits cudaMalloc — observed as 370 cudaMalloc/free per step on plain-LoRA
// zimage. Rounding maps slight shape variations to a shared bucket key.
//
// Constants are PyTorch's defaults; do not change without re-benchmarking.
// ---------------------------------------------------------------------------
/// All sizes are rounded to at least this many bytes (PT `kMinBlockSize`).
const K_MIN_BLOCK_BYTES: usize = 512;
/// Largest "small" allocation in bytes (PT `kSmallSize`). Requests at or
/// below this round to multiples of `K_MIN_BLOCK_BYTES`.
const K_SMALL_SIZE_BYTES: usize = 1 * 1024 * 1024;
/// Threshold above which we use the larger 2 MiB bucket (PT `kMinLargeAlloc`).
const K_MIN_LARGE_ALLOC_BYTES: usize = 10 * 1024 * 1024;
/// Bucket granularity for large allocations in bytes (PT `kRoundLarge` and
/// `kSmallBuffer`).
const K_ROUND_LARGE_BYTES: usize = 2 * 1024 * 1024;

/// Round `bytes` up to the next bucket boundary using PyTorch's three-tier
/// strategy. Returns the rounded byte count.
#[inline]
fn round_bytes_up(bytes: usize) -> usize {
    if bytes <= K_SMALL_SIZE_BYTES {
        // Round to next multiple of K_MIN_BLOCK_BYTES (= 512).
        if bytes < K_MIN_BLOCK_BYTES {
            K_MIN_BLOCK_BYTES
        } else {
            (bytes + K_MIN_BLOCK_BYTES - 1) & !(K_MIN_BLOCK_BYTES - 1)
        }
    } else if bytes < K_MIN_LARGE_ALLOC_BYTES {
        // 1 MiB < x < 10 MiB → round to 2 MiB.
        (bytes + K_ROUND_LARGE_BYTES - 1) & !(K_ROUND_LARGE_BYTES - 1)
    } else {
        // x >= 10 MiB → round to next 2 MiB.
        (bytes + K_ROUND_LARGE_BYTES - 1) & !(K_ROUND_LARGE_BYTES - 1)
    }
}

/// Round element count up to the next bucket boundary for `T`-sized elements.
#[inline]
fn round_elems_up<T>(n: usize) -> usize {
    let elem_size = std::mem::size_of::<T>();
    debug_assert!(elem_size > 0);
    if n == 0 {
        return 0;
    }
    let req_bytes = n.saturating_mul(elem_size);
    let bucket_bytes = round_bytes_up(req_bytes);
    // Convert back to element count. Buckets are multiples of 512 bytes
    // or 2 MiB; both are multiples of every type alignment we use
    // (sizeof(f32)=4, sizeof(u16)=2), so this divides evenly.
    bucket_bytes / elem_size
}

/// Cache key. The device pointer disambiguates `CudaDevice` instances —
/// tests sometimes construct fresh `CudaDevice::new(0)` Arcs which carry
/// distinct streams; memory pooled from one device cannot be safely
/// reused on another without an event-sync handshake. Production code
/// uses a single global Arc (`global_cuda_device`) so this discriminator
/// has no cost (same key always).
#[derive(Eq, PartialEq, Hash, Clone, Copy)]
struct CacheKey {
    device_ptr: usize,
    bucket: usize,
    is_u16: bool,
}

pub struct CudaAllocPool {
    /// Free lists keyed by `(device, bucket-rounded element count, dtype)`.
    /// Bucket rounding lets slightly-different sizes share a free list,
    /// matching PyTorch's BFC allocator strategy.
    free_lists: Mutex<HashMap<CacheKey, Vec<FreeEntry>>>,
    /// Whether the pool is accepting returns (set false during shutdown).
    active: AtomicBool,
    // --- stats (only updated when profiling_enabled()) ---
    alloc_count: AtomicUsize,
    reuse_count: AtomicUsize,
    return_count: AtomicUsize,
    peak_bytes: AtomicUsize,
    current_bytes: AtomicUsize,
    // --- hit/miss/bucket counters (always live, lock-free; minimal cost) ---
    hits: AtomicU64,
    misses: AtomicU64,
    bucket_saves: AtomicU64,
}

impl CudaAllocPool {
    fn new() -> Self {
        Self {
            free_lists: Mutex::new(HashMap::new()),
            active: AtomicBool::new(true),
            alloc_count: AtomicUsize::new(0),
            reuse_count: AtomicUsize::new(0),
            return_count: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            current_bytes: AtomicUsize::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bucket_saves: AtomicU64::new(0),
        }
    }

    /// Snapshot of hit/miss/bucket-save counters. `bucket_saves` is incremented
    /// every time a request maps to a different bucket than its exact size
    /// (i.e., would have been a miss without rounding but became a hit). The
    /// counters are best-effort (`Relaxed` ordering) and always-on.
    pub fn hit_miss_counts(&self) -> (u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.bucket_saves.load(Ordering::Relaxed),
        )
    }

    /// Round up to next power of 2 (element count). **Unused in production;**
    /// kept for the existing unit test. The real bucketing is `round_elems_up`.
    #[inline]
    fn bucket_size(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        n.next_power_of_two()
    }

    /// Try to pop a cached f32 allocation matching `(device, bucket)`.
    /// The popped entry's `len` is the actual underlying allocation size
    /// (= `bucket`).
    fn try_pop(&self, device: &Arc<CudaDevice>, bucket: usize) -> Option<FreeEntry> {
        let key = CacheKey {
            device_ptr: Arc::as_ptr(device) as *const () as usize,
            bucket,
            is_u16: false,
        };
        let mut lists = self.free_lists.lock().ok()?;
        let list = lists.get_mut(&key)?;
        let entry = list.pop();
        if entry.is_some() && profiling_enabled() {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            let bytes = bucket * std::mem::size_of::<f32>();
            self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }
        entry
    }

    /// Push a freed f32 allocation back into the pool, keyed by
    /// `(device, bucket-rounded element count)`.
    fn push_f32(&self, entry: FreeEntry) {
        if !self.active.load(Ordering::Relaxed) {
            unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
            return;
        }

        let size = entry.len; // already bucket-sized (allocator invariant)
        let bytes = size * std::mem::size_of::<f32>();

        // Don't cache huge allocations (>2 GiB).
        if bytes > MAX_POOL_BYTES {
            unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
            return;
        }

        let key = CacheKey {
            device_ptr: Arc::as_ptr(&entry.device) as *const () as usize,
            bucket: size,
            is_u16: false,
        };

        if let Ok(mut lists) = self.free_lists.lock() {
            let list = lists.entry(key).or_insert_with(Vec::new);
            if list.len() >= MAX_FREE_PER_SIZE {
                drop(lists);
                unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
                return;
            }
            list.push(entry);

            if profiling_enabled() {
                self.return_count.fetch_add(1, Ordering::Relaxed);
                let cur = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
                let mut peak = self.peak_bytes.load(Ordering::Relaxed);
                while cur > peak {
                    match self.peak_bytes.compare_exchange_weak(
                        peak,
                        cur,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }
            }
        } else {
            unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
        }
    }

    /// Push a u16 (BF16) allocation back into the pool.
    fn push_u16(&self, entry: FreeEntry) {
        if !self.active.load(Ordering::Relaxed) {
            unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
            return;
        }

        let size = entry.len; // already bucket-sized
        let bytes = size * std::mem::size_of::<u16>();
        if bytes > MAX_POOL_BYTES {
            unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
            return;
        }

        let key = CacheKey {
            device_ptr: Arc::as_ptr(&entry.device) as *const () as usize,
            bucket: size,
            is_u16: true,
        };

        if let Ok(mut lists) = self.free_lists.lock() {
            let list = lists.entry(key).or_insert_with(Vec::new);
            if list.len() >= MAX_FREE_PER_SIZE {
                drop(lists);
                unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
                return;
            }
            list.push(entry);

            if profiling_enabled() {
                self.return_count.fetch_add(1, Ordering::Relaxed);
                let cur = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
                let mut peak = self.peak_bytes.load(Ordering::Relaxed);
                while cur > peak {
                    match self.peak_bytes.compare_exchange_weak(
                        peak,
                        cur,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }
            }
        } else {
            unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
        }
    }

    /// Try to pop a cached u16 allocation matching `(device, bucket)`.
    fn try_pop_u16(&self, device: &Arc<CudaDevice>, bucket: usize) -> Option<FreeEntry> {
        let key = CacheKey {
            device_ptr: Arc::as_ptr(device) as *const () as usize,
            bucket,
            is_u16: true,
        };
        let mut lists = self.free_lists.lock().ok()?;
        let list = lists.get_mut(&key)?;
        let entry = list.pop();
        if entry.is_some() && profiling_enabled() {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            let bytes = bucket * std::mem::size_of::<u16>();
            self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }
        entry
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let (cached_bytes, cached_entries) = if let Ok(lists) = self.free_lists.lock() {
            let mut bytes = 0usize;
            let mut entries = 0usize;
            for (key, list) in lists.iter() {
                let elem_bytes = if key.is_u16 { 2 } else { 4 };
                bytes += key.bucket * elem_bytes * list.len();
                entries += list.len();
            }
            (bytes, entries)
        } else {
            (0, 0)
        };

        PoolStats {
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            reuse_count: self.reuse_count.load(Ordering::Relaxed),
            return_count: self.return_count.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            current_cached_bytes: cached_bytes,
            current_cached_entries: cached_entries,
        }
    }

    /// Free all cached memory. Call between training steps or on OOM retry.
    pub fn clear_cache(&self) {
        let entries: Vec<(CacheKey, Vec<FreeEntry>)> = {
            let mut lists = match self.free_lists.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            lists.drain().collect()
        };
        // Now free everything outside the lock.
        for (key, list) in entries {
            for entry in list {
                unsafe {
                    if key.is_u16 {
                        reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device);
                    } else {
                        reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device);
                    }
                }
            }
        }
        self.current_bytes.store(0, Ordering::Relaxed);
    }
}

impl Drop for CudaAllocPool {
    fn drop(&mut self) {
        self.active.store(false, Ordering::SeqCst);
        self.clear_cache();
    }
}

// ---------------------------------------------------------------------------
// Unsafe helpers — reconstruct / decompose CudaSlice<T>
// ---------------------------------------------------------------------------

/// Reconstruct a `CudaSlice<T>` from raw parts and let it drop (calling cudaFree).
///
/// # Safety
/// `ptr` must be a valid device pointer allocated by the same `device`,
/// with `len` elements of type T.
unsafe fn reconstruct_and_drop<T>(ptr: u64, len: usize, device: Arc<CudaDevice>) {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: ptr,
        len,
        device,
        host_buf: None,
    };
    let slice: CudaSlice<T> = std::mem::transmute(mirror);
    drop(slice); // runs cudaFree
}

/// Reconstruct a `CudaSlice<T>` from raw parts WITHOUT dropping.
///
/// # Safety
/// Same preconditions as `reconstruct_and_drop`.
unsafe fn reconstruct_slice<T>(ptr: u64, len: usize, device: Arc<CudaDevice>) -> CudaSlice<T> {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: ptr,
        len,
        device,
        host_buf: None,
    };
    std::mem::transmute(mirror)
}

/// Decompose a `CudaSlice<T>` into raw parts, consuming it without cudaFree.
///
/// # Safety
/// Caller must eventually either reconstruct the slice or manually free the ptr.
unsafe fn decompose_slice<T>(slice: CudaSlice<T>) -> (u64, usize, Arc<CudaDevice>) {
    let ptr = *slice.device_ptr();
    let len = DeviceSlice::len(&slice);
    // We need the device Arc. Read it from the mirror layout.
    let mirror: CudaSliceMirror<T> = std::mem::transmute(slice);
    // mirror won't drop (no Drop impl), so ptr stays live.
    let device = mirror.device.clone();
    // Forget mirror to prevent any implicit cleanup.
    // (CudaSliceMirror has no Drop, but Arc<CudaDevice> clone keeps it alive.)
    std::mem::forget(mirror);
    (ptr, len, device)
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static POOL: OnceLock<CudaAllocPool> = OnceLock::new();

/// Get the global allocation pool.
#[inline]
pub fn global_pool() -> &'static CudaAllocPool {
    POOL.get_or_init(CudaAllocPool::new)
}

// ---------------------------------------------------------------------------
// Public API — f32
// ---------------------------------------------------------------------------

/// Allocate at least `size` f32 elements from the caching pool.
///
/// **Bucket rounding (PT BFC allocator parity):** the request is rounded
/// up to the next bucket boundary; allocations within the same bucket
/// share a free list. The returned `CudaSlice<f32>` has
/// `len() == round_elems_up::<f32>(size)` (>= `size`) — callers must use
/// their original requested element count to compute kernel grids, or
/// honor `slice.len()` directly. The `TensorStorage::*` paths track
/// `numel` (= request) separately from the slice len, so they are
/// unaffected.
///
/// **Initialization:** the slice is **not** zeroed on either hit or miss.
/// This matches the BF16 path (`pool_alloc_u16`) and PT's BFCAllocator
/// semantics: callers are responsible for initializing the buffer. Set
/// `FLAME_F32_ZERO_INIT=1` to revert to the legacy zero-on-miss behavior
/// if a hidden caller is discovered.
pub fn pool_alloc_f32(device: &Arc<CudaDevice>, size: usize) -> crate::Result<CudaSlice<f32>> {
    if pool_disabled() || size == 0 {
        // Non-pool path unchanged: legacy callers expect zero-init.
        return device
            .alloc_zeros::<f32>(size)
            .map_err(|e| crate::Error::CudaDriver(format!("{e:?}")));
    }

    let pool = global_pool();
    let bucket = round_elems_up::<f32>(size);

    if profiling_enabled() {
        pool.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    // Try cache hit at the bucket size.
    if let Some(entry) = pool.try_pop(device, bucket) {
        pool.hits.fetch_add(1, Ordering::Relaxed);
        if bucket != size {
            pool.bucket_saves.fetch_add(1, Ordering::Relaxed);
        }
        // Reconstruct slice with len = original request. The underlying
        // memory is `bucket` elements; cudaFree only needs the pointer.
        // Reporting `len = size` preserves the historical caller contract
        // that `slice.len() == requested_size` (callers use this for grid
        // math + cudarc's dtod_copy asserts src.len()==dst.len()).
        let slice = unsafe { reconstruct_slice::<f32>(entry.ptr, size, entry.device) };
        log::trace!(
            "pool: f32 hit size={} bucket={} hits={} misses={} bucket_saves={}",
            size,
            bucket,
            pool.hits.load(Ordering::Relaxed),
            pool.misses.load(Ordering::Relaxed),
            pool.bucket_saves.load(Ordering::Relaxed),
        );
        return Ok(slice);
    }

    // Cache miss — fresh allocation at the bucket size.
    pool.misses.fetch_add(1, Ordering::Relaxed);
    log::trace!(
        "pool: f32 miss size={} bucket={} hits={} misses={}",
        size,
        bucket,
        pool.hits.load(Ordering::Relaxed),
        pool.misses.load(Ordering::Relaxed),
    );

    let zero_init = f32_zero_init_enabled();

    let alloc_once = |dev: &Arc<CudaDevice>, n: usize| -> crate::Result<CudaSlice<f32>> {
        if zero_init {
            dev.alloc_zeros::<f32>(n)
                .map_err(|e| crate::Error::CudaDriver(format!("alloc_zeros::<f32>({n}): {e:?}")))
        } else {
            unsafe { dev.alloc::<f32>(n) }
                .map_err(|e| crate::Error::CudaDriver(format!("alloc::<f32>({n}): {e:?}")))
        }
    };

    // Allocate `bucket` elements of underlying memory, but hand back a
    // CudaSlice with len=size. The slice's len affects cudarc's
    // dtod_copy/htod_copy assertions; the underlying capacity stays at
    // `bucket` for free-list reuse purposes.
    let result = match alloc_once(device, bucket) {
        Ok(s) => s,
        Err(_) => {
            pool.clear_cache();
            alloc_once(device, bucket).map_err(|e| {
                crate::Error::CudaDriver(format!(
                    "f32 alloc({bucket}) after pool.clear_cache: {e:?}"
                ))
            })?
        }
    };
    if bucket == size {
        Ok(result)
    } else {
        // Reconstruct with truncated len. The underlying memory is `bucket`
        // elements; only the slice header changes. cudaFree on drop just
        // uses the pointer.
        let (ptr, _, dev) = unsafe { decompose_slice(result) };
        Ok(unsafe { reconstruct_slice::<f32>(ptr, size, dev) })
    }
}

/// Return a `CudaSlice<f32>` to the caching pool instead of freeing it.
///
/// The slice's `len()` is the originally-requested element count. The
/// underlying allocation is at the bucket size; we re-round to recover
/// the bucket key for the free list.
///
/// # Safety
/// The slice must have been allocated by `pool_alloc_f32` or cudarc's
/// `device.alloc_zeros`. After this call, `slice` is consumed and the
/// caller must not use it.
pub fn pool_return_f32(slice: CudaSlice<f32>) {
    if pool_disabled() {
        drop(slice); // normal cudaFree
        return;
    }

    let len = DeviceSlice::len(&slice);
    if len == 0 {
        drop(slice);
        return;
    }

    let (ptr, elem_len, device) = unsafe { decompose_slice(slice) };
    // Bucket key = round of the user-visible len. Same function applied to
    // the same input always yields the same bucket, so alloc and free see
    // matching keys.
    let bucket = round_elems_up::<f32>(elem_len);

    global_pool().push_f32(FreeEntry {
        ptr,
        len: bucket,
        device,
    });
}

// ---------------------------------------------------------------------------
// Public API — u16 (BF16)
// ---------------------------------------------------------------------------

/// Allocate at least `size` u16 (BF16) elements from the caching pool.
///
/// **Bucket rounding (PT BFC allocator parity):** the request is rounded
/// up to the next bucket boundary; allocations within the same bucket
/// share a free list. The returned `CudaSlice<u16>` has
/// `len() == round_elems_up::<u16>(size)` (>= `size`).
///
/// Always returns uninitialized memory (matches PT BFCAllocator semantics).
pub fn pool_alloc_u16(device: &Arc<CudaDevice>, size: usize) -> crate::Result<CudaSlice<u16>> {
    if pool_disabled() || size == 0 {
        return unsafe {
            device
                .alloc::<u16>(size)
                .map_err(|e| crate::Error::CudaDriver(format!("{e:?}")))
        };
    }

    let pool = global_pool();
    let bucket = round_elems_up::<u16>(size);

    if profiling_enabled() {
        pool.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    if let Some(entry) = pool.try_pop_u16(device, bucket) {
        pool.hits.fetch_add(1, Ordering::Relaxed);
        if bucket != size {
            pool.bucket_saves.fetch_add(1, Ordering::Relaxed);
        }
        // Reconstruct with len = requested size, not bucket (see f32 path).
        let slice = unsafe { reconstruct_slice::<u16>(entry.ptr, size, entry.device) };
        log::trace!(
            "pool: u16 hit size={} bucket={} hits={} misses={} bucket_saves={}",
            size,
            bucket,
            pool.hits.load(Ordering::Relaxed),
            pool.misses.load(Ordering::Relaxed),
            pool.bucket_saves.load(Ordering::Relaxed),
        );
        return Ok(slice);
    }

    pool.misses.fetch_add(1, Ordering::Relaxed);
    log::trace!(
        "pool: u16 miss size={} bucket={} hits={} misses={}",
        size,
        bucket,
        pool.hits.load(Ordering::Relaxed),
        pool.misses.load(Ordering::Relaxed),
    );

    // Fresh allocation at the bucket size (uninitialized).
    let result = unsafe {
        device
            .alloc::<u16>(bucket)
            .map_err(|e| crate::Error::CudaDriver(format!("alloc::<u16>({bucket}): {e:?}")))
    };

    let allocated = result.or_else(|_| {
        pool.clear_cache();
        unsafe {
            device
                .alloc::<u16>(bucket)
                .map_err(|e| crate::Error::CudaDriver(format!(
                    "alloc::<u16>({bucket}) after pool.clear_cache: {e:?}"
                )))
        }
    })?;

    if bucket == size {
        Ok(allocated)
    } else {
        let (ptr, _, dev) = unsafe { decompose_slice(allocated) };
        Ok(unsafe { reconstruct_slice::<u16>(ptr, size, dev) })
    }
}

/// Return a `CudaSlice<u16>` to the caching pool.
pub fn pool_return_u16(slice: CudaSlice<u16>) {
    if pool_disabled() {
        drop(slice);
        return;
    }

    let len = DeviceSlice::len(&slice);
    if len == 0 {
        drop(slice);
        return;
    }

    let (ptr, elem_len, device) = unsafe { decompose_slice(slice) };
    let bucket = round_elems_up::<u16>(elem_len);

    global_pool().push_u16(FreeEntry {
        ptr,
        len: bucket,
        device,
    });
}

// ---------------------------------------------------------------------------
// Convenience: print stats summary
// ---------------------------------------------------------------------------

/// Print pool stats to stderr (gated on FLAME_PROFILE=1).
pub fn print_pool_stats() {
    if !profiling_enabled() {
        return;
    }
    let pool = global_pool();
    let s = pool.stats();
    let (hits, misses, bucket_saves) = pool.hit_miss_counts();
    let reuse_pct = if s.alloc_count > 0 {
        (s.reuse_count as f64) / (s.alloc_count as f64) * 100.0
    } else {
        0.0
    };
    let hit_pct = if hits + misses > 0 {
        (hits as f64) / ((hits + misses) as f64) * 100.0
    } else {
        0.0
    };
    eprintln!(
        "[alloc_pool] allocs={} reuses={} ({:.1}%) returns={} peak_cached={:.1}MB current_cached={:.1}MB entries={}",
        s.alloc_count,
        s.reuse_count,
        reuse_pct,
        s.return_count,
        s.peak_bytes as f64 / (1024.0 * 1024.0),
        s.current_cached_bytes as f64 / (1024.0 * 1024.0),
        s.current_cached_entries,
    );
    eprintln!(
        "[alloc_pool] hits={} misses={} ({:.1}%) bucket_saves={}",
        hits, misses, hit_pct, bucket_saves,
    );
}

/// Clear all cached GPU memory. Call on OOM or between phases.
pub fn clear_pool_cache() {
    global_pool().clear_cache();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_size() {
        assert_eq!(CudaAllocPool::bucket_size(0), 1);
        assert_eq!(CudaAllocPool::bucket_size(1), 1);
        assert_eq!(CudaAllocPool::bucket_size(2), 2);
        assert_eq!(CudaAllocPool::bucket_size(3), 4);
        assert_eq!(CudaAllocPool::bucket_size(5), 8);
        assert_eq!(CudaAllocPool::bucket_size(1000), 1024);
        assert_eq!(CudaAllocPool::bucket_size(1024), 1024);
        assert_eq!(CudaAllocPool::bucket_size(1025), 2048);
    }

    #[test]
    fn test_pool_disabled_env() {
        // Just verify the function doesn't panic.
        let _ = pool_disabled();
        let _ = profiling_enabled();
    }

    #[test]
    fn test_alloc_return_reuse() -> crate::Result<()> {
        // Allocate, return, allocate again — should get the same pointer.
        let device = CudaDevice::new(0)?;
        let size = 1024usize;

        let slice1 = pool_alloc_f32(&device, size)?;
        let ptr1 = *slice1.device_ptr();
        assert_eq!(DeviceSlice::len(&slice1), size);

        // Return to pool (this consumes slice1 without cudaFree).
        pool_return_f32(slice1);

        // Allocate again — should reuse the cached entry.
        let slice2 = pool_alloc_f32(&device, size)?;
        let ptr2 = *slice2.device_ptr();
        assert_eq!(ptr1, ptr2, "expected pool reuse — same device pointer");
        assert_eq!(DeviceSlice::len(&slice2), size);

        // Clean up — return and then clear cache (which does cudaFree).
        pool_return_f32(slice2);
        global_pool().clear_cache();

        Ok(())
    }

    #[test]
    fn test_round_bytes_up() {
        // Small allocs round to 512 bytes.
        assert_eq!(round_bytes_up(1), 512);
        assert_eq!(round_bytes_up(511), 512);
        assert_eq!(round_bytes_up(512), 512);
        assert_eq!(round_bytes_up(513), 1024);
        assert_eq!(round_bytes_up(4097), 4608);

        // At kSmallSize boundary (1 MiB), still rounds to 512.
        assert_eq!(round_bytes_up(K_SMALL_SIZE_BYTES), K_SMALL_SIZE_BYTES);

        // Mid range: 1 MiB < x < 10 MiB rounds to 2 MiB.
        assert_eq!(round_bytes_up(K_SMALL_SIZE_BYTES + 1), K_ROUND_LARGE_BYTES);
        assert_eq!(round_bytes_up(2 * 1024 * 1024), 2 * 1024 * 1024);
        assert_eq!(round_bytes_up(3_142_727), 4 * 1024 * 1024);
        assert_eq!(round_bytes_up(9 * 1024 * 1024), 10 * 1024 * 1024);

        // Large: >= 10 MiB rounds to 2 MiB.
        assert_eq!(round_bytes_up(K_MIN_LARGE_ALLOC_BYTES), K_MIN_LARGE_ALLOC_BYTES);
        assert_eq!(round_bytes_up(K_MIN_LARGE_ALLOC_BYTES + 1), K_MIN_LARGE_ALLOC_BYTES + K_ROUND_LARGE_BYTES);
    }

    #[test]
    fn test_round_elems_up_f32() {
        // 1 elem * 4 bytes = 4 bytes → rounds to 512 bytes = 128 f32 elems.
        assert_eq!(round_elems_up::<f32>(1), 128);
        // 1024 elems * 4 = 4096 bytes → 4096 bytes → 1024 elems (no change).
        assert_eq!(round_elems_up::<f32>(1024), 1024);
        // 0 → 0.
        assert_eq!(round_elems_up::<f32>(0), 0);
        // 3 MiB worth of f32 (786_433 elems) → rounds to 4 MiB.
        assert_eq!(round_elems_up::<f32>(786_433), 4 * 1024 * 1024 / 4);
    }

    #[test]
    fn test_round_elems_up_u16() {
        assert_eq!(round_elems_up::<u16>(1), 256);
        assert_eq!(round_elems_up::<u16>(0), 0);
        // 2048 elems * 2 = 4096 bytes → 2048 elems.
        assert_eq!(round_elems_up::<u16>(2048), 2048);
    }

    #[test]
    fn test_bucket_reuse_different_request_sizes() -> crate::Result<()> {
        // Two slightly-different-size requests should map to the same bucket
        // and reuse memory. This is the core mechanism Fix #A2 enables.
        let device = CudaDevice::new(0)?;
        // Pick two sizes that fall in the same 2 MiB bucket.
        // 600_000 f32 = 2.4 MiB → rounds to 4 MiB → bucket of 1_048_576 elems.
        // 700_000 f32 = 2.8 MiB → rounds to 4 MiB → same bucket.
        let size_a = 600_000usize;
        let size_b = 700_000usize;
        let bucket_a = round_elems_up::<f32>(size_a);
        let bucket_b = round_elems_up::<f32>(size_b);
        assert_eq!(bucket_a, bucket_b, "test setup: pick sizes in same bucket");

        let slice_a = pool_alloc_f32(&device, size_a)?;
        let ptr_a = *slice_a.device_ptr();
        assert_eq!(DeviceSlice::len(&slice_a), size_a);
        pool_return_f32(slice_a);

        let slice_b = pool_alloc_f32(&device, size_b)?;
        let ptr_b = *slice_b.device_ptr();
        assert_eq!(ptr_a, ptr_b, "expected bucket reuse — same device pointer");
        assert_eq!(DeviceSlice::len(&slice_b), size_b);

        pool_return_f32(slice_b);
        global_pool().clear_cache();
        Ok(())
    }
}
