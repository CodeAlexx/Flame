//! External memory registry — unified ptr/range tracking for the cudarc
//! `CudaSlice::drop` hook.
//!
//! ## Why this module exists
//!
//! Before R1a, `cuda_alloc_pool::external_ptrs: HashMap<u64, u32>` tracked
//! externally-owned pointers by exact base address only. That works fine for
//! the `BlockOffloader` ring-allocator path (each `alloc_bf16_via_ring` hands
//! out a fresh slab base) and for the `PoolMissAllocator` route. It breaks
//! the moment a tensor sourced from a static slab is `narrow`ed, `view`ed, or
//! `permute`d: the resulting `CudaSlice` carries a mid-slab `ptr`, the
//! exact-pointer map has no entry for it, the cudarc Drop hook returns false,
//! and `cudaFree` is called on an offset into a slab — `CUDA_ERROR_INVALID_VALUE`.
//!
//! This registry adds a range-aware path on top of the existing exact-pointer
//! path. The slab allocator registers ONE range covering its whole capacity at
//! materialization; mid-slab pointers from derived slices are protected
//! transitively. Exact-pointer refcount semantics for the ring/pool path are
//! preserved unchanged via the back-compat shim in `cuda_alloc_pool`.
//!
//! ## API surface
//!
//! ```ignore
//! use flame_core::external_memory::{ExternalMemoryRegistry, ExternalOwner, ExternalRange};
//!
//! let reg = ExternalMemoryRegistry::global();
//!
//! // Range-based (slab / block-offloader use case).
//! let handle = reg.register_range(ExternalRange {
//!     start: 0x1000, end: 0x2000, device_key: dev_id, owner: ExternalOwner::Slab,
//! });
//! // ... allocations happen, the slab is in use ...
//! reg.unregister_range(handle);
//!
//! // Exact-pointer (back-compat path for ring/pool allocator).
//! reg.register_exact(ptr, dev_id, ExternalOwner::Ring);
//! let new_count = reg.unregister_exact(ptr);
//! ```
//!
//! ## Hook installation
//!
//! `ensure_hook_installed()` is the single entry point that installs the
//! process-wide cudarc Drop hook. Idempotent: first caller wins, subsequent
//! calls are cheap no-ops. The slab allocator (R1b) and the BlockOffloader
//! ring (existing) both call it lazily.
//!
//! ## Owner taxonomy
//!
//! `ExternalOwner` is metadata only — the registry's protection decision does
//! not branch on it. Useful for diagnostics ("which subsystem is keeping this
//! ptr alive?") when adding logging. Values: `Slab`, `Ring`, `PoolExact`,
//! `BlockOffloader`.

use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};

/// Sentinel device-key used by the back-compat shim in `cuda_alloc_pool`
/// when the caller did not supply a device.
///
/// Real `Arc::as_ptr(&device) as usize` is always non-zero (Arc allocates on
/// the heap), so `0` cannot collide with a legitimate device-key. An entry
/// registered with `device_key = DEVICE_KEY_ANY` matches any device on the
/// hook decision — see `should_skip_free_any_device`.
pub const DEVICE_KEY_ANY: usize = 0;

/// Origin tag for an external entry. Metadata-only; the registry's protect-
/// or-free decision does not branch on this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalOwner {
    /// `StaticSlabAllocator` (Phase R1b).
    Slab,
    /// `RingAllocator` (Phase 2 BlockOffloader ring path).
    Ring,
    /// `cuda_alloc_pool` exact-pointer back-compat shim.
    PoolExact,
    /// `BlockOffloader::alloc_bf16_via_ring` direct registration.
    BlockOffloader,
}

/// Half-open `[start, end)` range of device-pointer values owned by one
/// external allocator on one device. The hook decision is `ptr ∈ [start, end)`
/// (NOT inclusive of `end` — exclusive upper bound).
#[derive(Debug, Clone, Copy)]
pub struct ExternalRange {
    /// Base device pointer (inclusive).
    pub start: u64,
    /// One-past-the-last device pointer (exclusive).
    pub end: u64,
    /// `Arc::as_ptr(&device) as usize` for the owning device. Use
    /// [`DEVICE_KEY_ANY`] for wildcard.
    pub device_key: usize,
    /// Origin tag (metadata).
    pub owner: ExternalOwner,
}

/// Opaque handle returned by `register_range`. Pass to `unregister_range` to
/// remove the entry. Copy/Clone so callers can stash it in a `Drop` impl.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RangeHandle(u64);

/// Unified registry of external-owned device memory.
///
/// Two storage tiers:
/// - **Ranges**: `Vec<(RangeHandle, ExternalRange)>`. Few entries (one per
///   slab + one per ring + per-block-offloader), so linear scan is fine and
///   cheaper than a more complex interval tree.
/// - **Exact**: `HashMap<(ptr, device_key), u32>`. Refcounted to preserve
///   the existing ring-wrap behaviour (see `cuda_alloc_pool::external_ptrs`
///   docs and `test_external_ptr_refcount_under_ring_wrap`).
///
/// The hook closure (installed once, process-wide) consults `should_skip_free_any_device`
/// because the cudarc `CudaSlice::drop` signature is `fn(u64) -> bool` — no
/// device key. False-positive risk is negligible: different CUDA contexts
/// have disjoint virtual address spaces in practice.
pub struct ExternalMemoryRegistry {
    inner: Mutex<RegistryInner>,
    hook_installed: AtomicBool,
}

struct RegistryInner {
    next_handle: u64,
    ranges: Vec<(RangeHandle, ExternalRange)>,
    exact: std::collections::HashMap<(u64, usize), u32>,
}

impl ExternalMemoryRegistry {
    fn new() -> Self {
        Self {
            inner: Mutex::new(RegistryInner {
                // Start handles at 1 so 0 is reserved as a sentinel/never-issued.
                next_handle: 1,
                ranges: Vec::new(),
                exact: std::collections::HashMap::new(),
            }),
            hook_installed: AtomicBool::new(false),
        }
    }

    /// Process-wide singleton accessor.
    pub fn global() -> &'static ExternalMemoryRegistry {
        static REG: OnceLock<ExternalMemoryRegistry> = OnceLock::new();
        REG.get_or_init(ExternalMemoryRegistry::new)
    }

    /// Register a contiguous half-open range `[range.start, range.end)`.
    /// Returns an opaque handle that must be passed back to
    /// [`unregister_range`](Self::unregister_range) to remove the entry.
    ///
    /// Zero-length ranges (start == end) are accepted but never match any
    /// pointer — useful for testing edge cases.
    pub fn register_range(&self, range: ExternalRange) -> RangeHandle {
        let mut g = self.inner.lock().expect("ExternalMemoryRegistry poisoned");
        let h = RangeHandle(g.next_handle);
        g.next_handle = g.next_handle.wrapping_add(1);
        // DECISION: also guard against zero-handle if `next_handle` ever wraps
        // around; 0 stays reserved. In practice 2^64 registrations is
        // unreachable, but the guard is cheap.
        if g.next_handle == 0 {
            g.next_handle = 1;
        }
        g.ranges.push((h, range));
        h
    }

    /// Remove a previously-registered range. No-op if the handle is unknown
    /// (already removed, or never issued by this registry).
    pub fn unregister_range(&self, handle: RangeHandle) {
        let mut g = self.inner.lock().expect("ExternalMemoryRegistry poisoned");
        g.ranges.retain(|(h, _)| *h != handle);
    }

    /// Increment the refcount for exact-pointer `(ptr, device_key)`. Pair
    /// with [`unregister_exact`](Self::unregister_exact).
    ///
    /// `owner` is metadata only (not stored against the count — we keep the
    /// storage shape identical to the pre-R1a `HashMap<u64, u32>` for
    /// back-compat-shim simplicity).
    pub fn register_exact(&self, ptr: u64, device_key: usize, _owner: ExternalOwner) {
        let mut g = self.inner.lock().expect("ExternalMemoryRegistry poisoned");
        *g.exact.entry((ptr, device_key)).or_insert(0) += 1;
    }

    /// Decrement refcount for `(ptr, device_key)`. Removes the entry when
    /// the count reaches zero. Saturates at zero (extra unregisters after
    /// the count is already 0 are silently ignored).
    ///
    /// Returns the new refcount after the decrement (0 if the entry was
    /// removed or never existed).
    pub fn unregister_exact(&self, ptr: u64) -> usize {
        // DECISION: The spec API is `unregister_exact(&self, ptr: u64) -> usize`,
        // no device_key. To preserve the pre-R1a back-compat semantics
        // (existing `unregister_external_ptr` is keyed by ptr only), this
        // removes the entry matching `ptr` regardless of device. If the same
        // ptr was registered under two different device keys (an
        // architecturally suspicious situation — different CUDA contexts
        // sharing a host-visible numeric address), we decrement whichever
        // is hit first by HashMap iteration order. Tests must avoid that
        // shape; the dedicated device-key API is `unregister_exact_keyed`.
        let mut g = self.inner.lock().expect("ExternalMemoryRegistry poisoned");
        // Find any entry whose key.0 == ptr.
        let key_to_touch = g
            .exact
            .keys()
            .find(|(p, _)| *p == ptr)
            .copied();
        match key_to_touch {
            Some(k) => {
                if let Some(c) = g.exact.get_mut(&k) {
                    if *c > 1 {
                        *c -= 1;
                        *c as usize
                    } else {
                        g.exact.remove(&k);
                        0
                    }
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Strictly-keyed unregister — decrement refcount for the exact
    /// `(ptr, device_key)` pair. Used by callers that have the device key.
    pub fn unregister_exact_keyed(&self, ptr: u64, device_key: usize) -> usize {
        let mut g = self.inner.lock().expect("ExternalMemoryRegistry poisoned");
        let k = (ptr, device_key);
        if let Some(c) = g.exact.get_mut(&k) {
            if *c > 1 {
                *c -= 1;
                *c as usize
            } else {
                g.exact.remove(&k);
                0
            }
        } else {
            0
        }
    }

    /// Hook decision: should cudarc skip `cudaFree` for `ptr` on `device_key`?
    ///
    /// Returns true iff:
    /// 1. Any registered range `[start, end)` with matching `device_key` (or
    ///    `DEVICE_KEY_ANY`) covers `ptr`; **OR**
    /// 2. An exact entry `(ptr, device_key)` or `(ptr, DEVICE_KEY_ANY)`
    ///    exists with non-zero refcount.
    pub fn should_skip_free(&self, ptr: u64, device_key: usize) -> bool {
        let g = self.inner.lock().expect("ExternalMemoryRegistry poisoned");
        for (_, r) in &g.ranges {
            if (r.device_key == device_key || r.device_key == DEVICE_KEY_ANY)
                && ptr >= r.start
                && ptr < r.end
            {
                return true;
            }
        }
        if g.exact.get(&(ptr, device_key)).copied().unwrap_or(0) > 0 {
            return true;
        }
        if device_key != DEVICE_KEY_ANY
            && g.exact.get(&(ptr, DEVICE_KEY_ANY)).copied().unwrap_or(0) > 0
        {
            return true;
        }
        false
    }

    /// Hook decision when the caller has no device context — checks if ANY
    /// registered entry (range OR exact, on ANY device_key) covers `ptr`.
    ///
    /// Used by the cudarc `CudaSlice::drop` hook, whose signature is
    /// `fn(u64) -> bool` and does not pass a device_key. Also used by the
    /// back-compat `cuda_alloc_pool::is_external_ptr(ptr)` shim.
    pub fn should_skip_free_any_device(&self, ptr: u64) -> bool {
        let g = self.inner.lock().expect("ExternalMemoryRegistry poisoned");
        for (_, r) in &g.ranges {
            if ptr >= r.start && ptr < r.end {
                return true;
            }
        }
        // Exact-map lookup: any device.
        for ((p, _), c) in &g.exact {
            if *p == ptr && *c > 0 {
                return true;
            }
        }
        false
    }

    /// Install the cudarc Drop hook (once per process). Idempotent: subsequent
    /// calls are cheap no-ops. Safe to call from `StaticSlabAllocator` init,
    /// from `BlockOffloader::ensure_ring`, or from `install_miss_allocator` —
    /// whichever fires first wins, the rest are no-ops.
    pub fn ensure_hook_installed() {
        let reg = ExternalMemoryRegistry::global();
        // Fast-path: already installed.
        if reg.hook_installed.load(Ordering::Acquire) {
            return;
        }
        // Race-free: only the thread that flips false→true installs.
        if reg
            .hook_installed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            cudarc::driver::install_external_ptr_hook(external_ptr_hook_global);
        }
    }

    // ---- Diagnostics (test-only, doc-hidden) ----

    /// Returns the number of distinct (ptr, device_key) entries currently
    /// tracked in the exact-pointer map.
    #[doc(hidden)]
    pub fn exact_entry_count(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.exact.len())
            .unwrap_or(0)
    }

    /// Returns the current refcount for exact-pointer `(ptr, device_key)`.
    #[doc(hidden)]
    pub fn exact_refcount(&self, ptr: u64, device_key: usize) -> u32 {
        self.inner
            .lock()
            .map(|g| g.exact.get(&(ptr, device_key)).copied().unwrap_or(0))
            .unwrap_or(0)
    }

    /// Returns the number of currently-registered ranges.
    #[doc(hidden)]
    pub fn range_count(&self) -> usize {
        self.inner.lock().map(|g| g.ranges.len()).unwrap_or(0)
    }

    /// Test-only reset of internal state. Does NOT uninstall the cudarc
    /// hook (which is global and cannot be safely uninstalled mid-process).
    #[doc(hidden)]
    pub fn reset_for_testing(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.ranges.clear();
            g.exact.clear();
            // `next_handle` intentionally NOT reset — handle uniqueness must
            // hold across resets so a stale handle from before the reset
            // doesn't accidentally collide with a fresh one.
        }
    }

    /// Test-only inspection of the `hook_installed` flag.
    #[doc(hidden)]
    pub fn hook_installed_flag(&self) -> bool {
        self.hook_installed.load(Ordering::Acquire)
    }
}

/// Global hook closure registered with cudarc. Consults the global
/// registry's `should_skip_free_any_device` path because cudarc's hook
/// signature is `fn(u64) -> bool` (no device parameter).
fn external_ptr_hook_global(ptr: u64) -> bool {
    ExternalMemoryRegistry::global().should_skip_free_any_device(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Tests share a process-wide singleton, so they must run with a guard
    /// that serializes mutation and resets state between tests. Standard
    /// pattern across `cuda_alloc_pool` and `offload` tests in this crate.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    fn fresh() -> &'static ExternalMemoryRegistry {
        let reg = ExternalMemoryRegistry::global();
        reg.reset_for_testing();
        reg
    }

    /// 1. Register exact ptr → hook returns true; unregister → hook returns false.
    #[test]
    fn registry_exact_pointer_skip_free() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr: u64 = 0x1_0000_0000;
        let dev = 0xAB_usize;

        assert!(!reg.should_skip_free(ptr, dev));
        assert!(!reg.should_skip_free_any_device(ptr));

        reg.register_exact(ptr, dev, ExternalOwner::PoolExact);
        assert!(reg.should_skip_free(ptr, dev));
        assert!(reg.should_skip_free_any_device(ptr));

        let new_count = reg.unregister_exact(ptr);
        assert_eq!(new_count, 0);
        assert!(!reg.should_skip_free(ptr, dev));
        assert!(!reg.should_skip_free_any_device(ptr));
    }

    /// 2. Register range `[0x1000, 0x2000)`; mid-range hits, edges miss.
    #[test]
    fn registry_range_covers_offset_ptr() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0x42_usize;

        let h = reg.register_range(ExternalRange {
            start: 0x1000,
            end: 0x2000,
            device_key: dev,
            owner: ExternalOwner::Slab,
        });

        // Inside the range.
        assert!(reg.should_skip_free(0x1000, dev), "start inclusive");
        assert!(reg.should_skip_free(0x1500, dev), "midpoint");
        assert!(reg.should_skip_free(0x1FFF, dev), "last byte");

        // Outside.
        assert!(!reg.should_skip_free(0x0FFF, dev), "one below start");
        assert!(!reg.should_skip_free(0x2000, dev), "end exclusive");

        // Cleanup leaves the registry clean.
        reg.unregister_range(h);
        assert!(!reg.should_skip_free(0x1500, dev));
        assert_eq!(reg.range_count(), 0);
    }

    /// 3. Range + exact compose: unregistering the exact doesn't shrink the
    /// range's protection.
    #[test]
    fn registry_range_and_exact_compose() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0x99_usize;
        let mid_ptr: u64 = 0x1500;

        let h = reg.register_range(ExternalRange {
            start: 0x1000,
            end: 0x2000,
            device_key: dev,
            owner: ExternalOwner::Slab,
        });
        reg.register_exact(mid_ptr, dev, ExternalOwner::PoolExact);

        assert!(reg.should_skip_free(mid_ptr, dev));

        // Drop the exact entry: range still protects.
        let new_count = reg.unregister_exact(mid_ptr);
        assert_eq!(new_count, 0);
        assert!(
            reg.should_skip_free(mid_ptr, dev),
            "range entry must still protect after exact entry drops"
        );

        // Drop the range: now ptr is free-able.
        reg.unregister_range(h);
        assert!(!reg.should_skip_free(mid_ptr, dev));
    }

    /// 4. Same ptr on two different device_keys does not cross-protect.
    #[test]
    fn registry_device_isolation() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr: u64 = 0xCAFE_0000;
        let dev_a = 0x1111_usize;
        let dev_b = 0x2222_usize;

        // Range registered on dev_a only.
        let h = reg.register_range(ExternalRange {
            start: ptr,
            end: ptr + 0x1000,
            device_key: dev_a,
            owner: ExternalOwner::Slab,
        });
        assert!(reg.should_skip_free(ptr, dev_a));
        assert!(
            !reg.should_skip_free(ptr, dev_b),
            "ptr on dev_b must NOT be protected by a dev_a range"
        );

        reg.unregister_range(h);

        // Exact entry on dev_a only.
        reg.register_exact(ptr, dev_a, ExternalOwner::Ring);
        assert!(reg.should_skip_free(ptr, dev_a));
        assert!(
            !reg.should_skip_free(ptr, dev_b),
            "exact entry on dev_a must NOT cross-protect dev_b"
        );

        // ...but `should_skip_free_any_device` matches on either (used by
        // the cudarc hook, which has no device context).
        assert!(reg.should_skip_free_any_device(ptr));
    }

    /// 5. Hook install is idempotent — flag flips once, subsequent calls
    /// are cheap no-ops.
    #[test]
    fn registry_hook_idempotent_install() {
        let _g = TEST_LOCK.lock().unwrap();
        // Note: hook flag is process-wide; other tests in the suite may
        // have flipped it already. The contract is "no panic, no double
        // install" — we verify the flag stays `true` after repeated calls.
        ExternalMemoryRegistry::ensure_hook_installed();
        let after_first = ExternalMemoryRegistry::global().hook_installed_flag();
        assert!(after_first, "first install sets the flag");

        // Subsequent calls must be safe.
        ExternalMemoryRegistry::ensure_hook_installed();
        ExternalMemoryRegistry::ensure_hook_installed();
        ExternalMemoryRegistry::ensure_hook_installed();

        let after_more = ExternalMemoryRegistry::global().hook_installed_flag();
        assert!(after_more, "repeated installs keep the flag set");
    }

    /// 6. Exact-pointer refcount: register twice, unregister once → still
    /// protected; second unregister → no longer protected.
    #[test]
    fn registry_exact_refcount() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr: u64 = 0xDEADBEEF;
        let dev = 0x33_usize;

        reg.register_exact(ptr, dev, ExternalOwner::Ring);
        reg.register_exact(ptr, dev, ExternalOwner::Ring);
        assert_eq!(reg.exact_refcount(ptr, dev), 2);
        assert!(reg.should_skip_free(ptr, dev));

        let new_count = reg.unregister_exact(ptr);
        assert_eq!(new_count, 1, "first unregister returns new count = 1");
        assert!(
            reg.should_skip_free(ptr, dev),
            "ptr still protected at refcount=1"
        );

        let new_count = reg.unregister_exact(ptr);
        assert_eq!(new_count, 0, "final unregister returns new count = 0");
        assert!(
            !reg.should_skip_free(ptr, dev),
            "ptr no longer protected after final unregister"
        );

        // Extra unregister after the count is already 0 is a no-op and
        // returns 0.
        let new_count = reg.unregister_exact(ptr);
        assert_eq!(new_count, 0);
    }

    // --- Additional small coverage tests (not in the 6-test spec but
    // protect the lesser-used code paths) ---

    /// Zero-length range never matches.
    #[test]
    fn registry_zero_length_range_never_matches() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0xA_usize;
        let h = reg.register_range(ExternalRange {
            start: 0x1000,
            end: 0x1000,
            device_key: dev,
            owner: ExternalOwner::Slab,
        });
        assert!(!reg.should_skip_free(0x1000, dev));
        assert!(!reg.should_skip_free(0x0FFF, dev));
        reg.unregister_range(h);
    }

    /// DEVICE_KEY_ANY wildcard matches any device-key on `should_skip_free`.
    #[test]
    fn registry_device_key_any_wildcard() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr: u64 = 0xBABE_0000;
        reg.register_exact(ptr, DEVICE_KEY_ANY, ExternalOwner::PoolExact);
        assert!(reg.should_skip_free(ptr, 0x1234));
        assert!(reg.should_skip_free(ptr, 0x5678));
        reg.unregister_exact(ptr);
        assert!(!reg.should_skip_free(ptr, 0x1234));
    }

    /// `unregister_range` with an unknown handle is a no-op.
    #[test]
    fn registry_unregister_unknown_range_handle() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        reg.unregister_range(RangeHandle(0xFFFF_FFFF));
        // Should not panic and registry stays empty.
        assert_eq!(reg.range_count(), 0);
    }

    /// `unregister_exact_keyed` strict semantics.
    #[test]
    fn registry_unregister_exact_keyed_strict() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr: u64 = 0x7777;
        let dev_a = 0x10_usize;
        let dev_b = 0x20_usize;
        reg.register_exact(ptr, dev_a, ExternalOwner::Ring);
        reg.register_exact(ptr, dev_b, ExternalOwner::Ring);
        // Decrementing dev_a does NOT touch dev_b's count.
        let after = reg.unregister_exact_keyed(ptr, dev_a);
        assert_eq!(after, 0);
        assert_eq!(reg.exact_refcount(ptr, dev_a), 0);
        assert_eq!(reg.exact_refcount(ptr, dev_b), 1);
        reg.unregister_exact_keyed(ptr, dev_b);
        assert_eq!(reg.exact_refcount(ptr, dev_b), 0);
    }
}
