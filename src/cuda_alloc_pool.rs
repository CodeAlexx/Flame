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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

pub struct CudaAllocPool {
    /// Free lists keyed by exact element count.
    /// Exact-match gives 90%+ reuse in backward (same shapes repeat each step).
    free_lists: Mutex<HashMap<usize, Vec<FreeEntry>>>,
    /// Whether the pool is accepting returns (set false during shutdown).
    active: AtomicBool,
    // --- stats (only updated when profiling_enabled()) ---
    alloc_count: AtomicUsize,
    reuse_count: AtomicUsize,
    return_count: AtomicUsize,
    peak_bytes: AtomicUsize,
    current_bytes: AtomicUsize,
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
        }
    }

    /// Round up to next power of 2 (element count).
    #[inline]
    fn bucket_size(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        n.next_power_of_two()
    }

    /// Try to pop a cached f32 allocation of exactly `size` elements.
    fn try_pop(&self, size: usize) -> Option<FreeEntry> {
        let mut lists = self.free_lists.lock().ok()?;
        let list = lists.get_mut(&size)?;
        let entry = list.pop();
        if entry.is_some() && profiling_enabled() {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            let bytes = size * std::mem::size_of::<f32>();
            self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }
        entry
    }

    /// Push a freed f32 allocation back into the pool, keyed by exact element count.
    fn push_f32(&self, entry: FreeEntry) {
        if !self.active.load(Ordering::Relaxed) {
            unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
            return;
        }

        let size = entry.len;
        let bytes = size * std::mem::size_of::<f32>();

        // Don't cache huge allocations (>2 GiB).
        if bytes > MAX_POOL_BYTES {
            unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
            return;
        }

        if let Ok(mut lists) = self.free_lists.lock() {
            let list = lists.entry(size).or_insert_with(Vec::new);
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

        let size = entry.len;
        let bytes = size * std::mem::size_of::<u16>();
        if bytes > MAX_POOL_BYTES {
            unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
            return;
        }

        if let Ok(mut lists) = self.free_lists.lock() {
            // Use a separate key space for u16 by setting high bit.
            let key = size | (1 << 63);
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

    /// Try to pop a cached u16 allocation of exactly `size` elements.
    fn try_pop_u16(&self, size: usize) -> Option<FreeEntry> {
        let key = size | (1 << 63);
        let mut lists = self.free_lists.lock().ok()?;
        let list = lists.get_mut(&key)?;
        let entry = list.pop();
        if entry.is_some() && profiling_enabled() {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            let bytes = size * std::mem::size_of::<u16>();
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
                let is_u16 = (*key >> 63) == 1;
                let elem_count = *key & !(1usize << 63);
                let elem_bytes = if is_u16 { 2 } else { 4 };
                bytes += elem_count * elem_bytes * list.len();
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
        let entries: Vec<(usize, Vec<FreeEntry>)> = {
            let mut lists = match self.free_lists.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            lists.drain().collect()
        };
        // Now free everything outside the lock.
        for (key, list) in entries {
            let is_u16 = (key >> 63) == 1;
            for entry in list {
                unsafe {
                    if is_u16 {
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

/// Allocate `size` f32 elements from the caching pool.
///
/// Returns a `CudaSlice<f32>` with `len() == size`.  Exact-size matching
/// ensures the slice length invariant that callers rely on.  The slice is
/// **not** zeroed on cache hit; call `device.memset_zeros` if needed.
pub fn pool_alloc_f32(device: &Arc<CudaDevice>, size: usize) -> crate::Result<CudaSlice<f32>> {
    if pool_disabled() || size == 0 {
        return device
            .alloc_zeros::<f32>(size)
            .map_err(|_| crate::Error::CudaDriver);
    }

    let pool = global_pool();

    if profiling_enabled() {
        pool.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    // Try cache hit (exact size match).
    if let Some(entry) = pool.try_pop(size) {
        let slice = unsafe { reconstruct_slice::<f32>(entry.ptr, entry.len, entry.device) };
        return Ok(slice);
    }

    // Cache miss — fresh allocation at exact requested size.
    device
        .alloc_zeros::<f32>(size)
        .map_err(|_| {
            // On OOM, try clearing the cache and retrying once.
            pool.clear_cache();
            crate::Error::CudaDriver
        })
        .or_else(|_| {
            device
                .alloc_zeros::<f32>(size)
                .map_err(|_| crate::Error::CudaDriver)
        })
}

/// Return a `CudaSlice<f32>` to the caching pool instead of freeing it.
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

    global_pool().push_f32(FreeEntry {
        ptr,
        len: elem_len,
        device,
    });
}

// ---------------------------------------------------------------------------
// Public API — u16 (BF16)
// ---------------------------------------------------------------------------

/// Allocate `size` u16 elements from the caching pool.
pub fn pool_alloc_u16(device: &Arc<CudaDevice>, size: usize) -> crate::Result<CudaSlice<u16>> {
    if pool_disabled() || size == 0 {
        return unsafe {
            device
                .alloc::<u16>(size)
                .map_err(|_| crate::Error::CudaDriver)
        };
    }

    let pool = global_pool();

    if profiling_enabled() {
        pool.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    if let Some(entry) = pool.try_pop_u16(size) {
        let slice = unsafe { reconstruct_slice::<u16>(entry.ptr, entry.len, entry.device) };
        return Ok(slice);
    }

    // Fresh allocation at exact requested size.
    let result = unsafe {
        device
            .alloc::<u16>(size)
            .map_err(|_| crate::Error::CudaDriver)
    };

    result.or_else(|_| {
        pool.clear_cache();
        unsafe {
            device
                .alloc::<u16>(size)
                .map_err(|_| crate::Error::CudaDriver)
        }
    })
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

    global_pool().push_u16(FreeEntry {
        ptr,
        len: elem_len,
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
    let s = global_pool().stats();
    let reuse_pct = if s.alloc_count > 0 {
        (s.reuse_count as f64) / (s.alloc_count as f64) * 100.0
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
}
