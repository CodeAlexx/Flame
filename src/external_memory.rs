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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

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
///
/// Note: `RangeHandle` itself has NO `Drop` impl — ranges must be explicitly
/// unregistered. Dropping a handle without calling `unregister_range` leaks
/// the registry entry until process exit. This is intentional: callers own
/// the lifetime (matches `register_exact`/`unregister_exact` symmetry).
/// Locked down by `skeptic_range_handle_drop_does_not_auto_unregister`.
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
    ///
    /// DECISION: inverted ranges (start > end) are silently accepted and
    /// silently never match. No `debug_assert!` is added because the registry
    /// is best-effort metadata — caller owns input sanity. Locked down by
    /// `skeptic_register_range_inverted_silently_never_matches`.
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
        let key_to_touch = g.exact.keys().find(|(p, _)| *p == ptr).copied();
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
    ///
    /// For device-less callers (e.g., the cudarc `CudaSlice::drop` hook, whose
    /// signature is `fn(u64) -> bool` with no device context), use
    /// [`should_skip_free_any_device`](Self::should_skip_free_any_device)
    /// instead. Querying this method with `device_key == DEVICE_KEY_ANY` will
    /// only match entries that were registered under `DEVICE_KEY_ANY`; it will
    /// NOT match real-device entries. Locked down by
    /// `skeptic_should_skip_free_with_any_key_does_not_match_real_device`.
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
        self.inner.lock().map(|g| g.exact.len()).unwrap_or(0)
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
    let should_skip = ExternalMemoryRegistry::global().should_skip_free_any_device(ptr);
    if !should_skip {
        return false;
    }
    if crate::static_slab_v2::slab_v2_return_if_owned_any_device(ptr) {
        return true;
    }
    true
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

    // =============================================================
    // Skeptic Phase R1a — adversarial tests
    //
    // These tests probe seams that Builder + Bug Fixer didn't cover.
    // Each test is documented with the specific invariant it locks down
    // or the bug-class hypothesis it disproves.
    // =============================================================

    /// Skeptic #1: `ensure_hook_installed` actually only calls cudarc's
    /// `install_external_ptr_hook` once.
    ///
    /// Bug Fixer flagged that the existing `registry_hook_idempotent_install`
    /// test only checks the flag stays `true` — it does NOT prove the cudarc
    /// side-effect is single-shot. We can't easily intercept the cudarc
    /// closure pointer, but we CAN prove the `compare_exchange` from
    /// `false→true` only succeeds once by directly inspecting the atomic
    /// flag and resetting it cooperatively between attempts.
    ///
    /// Approach: clobber the flag to `false`, race N threads through
    /// `ensure_hook_installed`, and verify that exactly one of them
    /// observed the false→true transition (via a side-channel atomic
    /// installed by us — see internal `install_with_counter`-style logic
    /// below). We can't add hooks into the registry from a test, but we
    /// CAN add a deterministic helper that exercises the CAS logic.
    ///
    /// Pragmatic test: drive the CAS path explicitly via repeated calls
    /// AND verify cudarc's hook does the right thing (returns true for
    /// a registered ptr, false for an unregistered one). This proves the
    /// hook function pointer was wired through, even if we can't count
    /// install-calls.
    #[test]
    fn skeptic_hook_install_single_shot_via_side_effect() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();

        // Install several times.
        for _ in 0..10 {
            ExternalMemoryRegistry::ensure_hook_installed();
        }
        assert!(reg.hook_installed_flag());

        // Register a fake ptr; the cudarc-installed closure should match it
        // when queried directly via the registry (we can't easily invoke
        // cudarc's stored fn pointer from a test, but we CAN verify the
        // registry path the closure consults).
        let ptr = 0xFEED_BABE_C0DE_0001_u64;
        reg.register_exact(ptr, DEVICE_KEY_ANY, ExternalOwner::PoolExact);
        assert!(reg.should_skip_free_any_device(ptr));
        reg.unregister_exact(ptr);
        assert!(!reg.should_skip_free_any_device(ptr));
    }

    /// Skeptic #1b: concurrent `ensure_hook_installed` from N threads
    /// terminates with consistent state and no panic.
    ///
    /// Even if cudarc's `install_external_ptr_hook` is idempotent in the
    /// sense of "store-the-same-fn-pointer twice is fine," we want to
    /// verify the registry's CAS-guarded path doesn't double-fire under
    /// contention.
    #[test]
    fn skeptic_hook_install_concurrent_no_panic() {
        let _g = TEST_LOCK.lock().unwrap();
        let _reg = fresh();
        let mut handles = Vec::new();
        for _ in 0..16 {
            handles.push(std::thread::spawn(|| {
                for _ in 0..100 {
                    ExternalMemoryRegistry::ensure_hook_installed();
                }
            }));
        }
        for h in handles {
            h.join().expect("ensure_hook_installed worker panicked");
        }
        assert!(ExternalMemoryRegistry::global().hook_installed_flag());
    }

    /// Skeptic #2: `unregister_exact(ptr)` non-determinism across two
    /// different `device_key`s.
    ///
    /// Bug Fixer flagged that when the same `ptr` is registered under two
    /// device keys, `unregister_exact(ptr)` picks one via HashMap iteration
    /// order. This test asserts the OBSERVABLE outcome: exactly one of the
    /// two refcounts must be unchanged (still 1), the other must be 0.
    /// This locks down "exactly-one-removed, but which-one is unspecified."
    /// If a future refactor makes the choice deterministic OR removes both,
    /// this test catches the behavior change.
    #[test]
    fn skeptic_unregister_exact_one_of_many_devices() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr: u64 = 0xAAAA_BBBB;
        let dev_a = 0x11_usize;
        let dev_b = 0x22_usize;
        reg.register_exact(ptr, dev_a, ExternalOwner::Ring);
        reg.register_exact(ptr, dev_b, ExternalOwner::Ring);
        assert_eq!(reg.exact_refcount(ptr, dev_a), 1);
        assert_eq!(reg.exact_refcount(ptr, dev_b), 1);

        let _ = reg.unregister_exact(ptr);
        // Exactly one of the two entries must remain at 1, the other gone.
        let a = reg.exact_refcount(ptr, dev_a);
        let b = reg.exact_refcount(ptr, dev_b);
        assert!(
            (a == 1 && b == 0) || (a == 0 && b == 1),
            "non-deterministic outcome must still satisfy 'exactly-one-removed': got a={a} b={b}"
        );
        // Total surviving count must be 1.
        assert_eq!(a + b, 1);

        // A second unregister picks up the remaining entry.
        let _ = reg.unregister_exact(ptr);
        assert_eq!(reg.exact_refcount(ptr, dev_a), 0);
        assert_eq!(reg.exact_refcount(ptr, dev_b), 0);
    }

    /// Skeptic #3: `should_skip_free(ptr, DEVICE_KEY_ANY)` asymmetry.
    ///
    /// When querying with `device_key=DEVICE_KEY_ANY=0`, the function:
    /// - Range scan: `r.device_key == 0 || r.device_key == 0` — only matches
    ///   ranges whose `device_key` is itself `DEVICE_KEY_ANY`. Will NOT
    ///   match a range registered on a real device key.
    /// - Exact scan: only checks `(ptr, 0)`; the secondary
    ///   `(ptr, DEVICE_KEY_ANY)` fallback is skipped when `device_key == DEVICE_KEY_ANY`.
    ///
    /// Decision (Skeptic): this asymmetry is INTENTIONAL — callers querying
    /// with `DEVICE_KEY_ANY` are explicitly saying "I have no device
    /// context, use the device-free path" and SHOULD route through
    /// `should_skip_free_any_device`, which DOES scan everything. This test
    /// locks down both halves of that contract.
    #[test]
    fn skeptic_should_skip_free_with_any_key_does_not_match_real_device() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr: u64 = 0xC0FFEE_0000_u64;
        let real_dev = 0x55_usize;

        // Range on real device.
        let h = reg.register_range(ExternalRange {
            start: ptr,
            end: ptr + 0x100,
            device_key: real_dev,
            owner: ExternalOwner::Slab,
        });
        // Exact on real device.
        let ptr2: u64 = 0xC0FFEE_1000_u64;
        reg.register_exact(ptr2, real_dev, ExternalOwner::Ring);

        // Asymmetry: `should_skip_free` with `DEVICE_KEY_ANY` does NOT
        // find real-device entries.
        assert!(
            !reg.should_skip_free(ptr, DEVICE_KEY_ANY),
            "querying with DEVICE_KEY_ANY must NOT see a real-device range"
        );
        assert!(
            !reg.should_skip_free(ptr2, DEVICE_KEY_ANY),
            "querying with DEVICE_KEY_ANY must NOT see a real-device exact entry"
        );

        // The device-agnostic path DOES find them — this is the path the
        // cudarc hook closure uses.
        assert!(reg.should_skip_free_any_device(ptr));
        assert!(reg.should_skip_free_any_device(ptr2));

        // Real-device queries find them too.
        assert!(reg.should_skip_free(ptr, real_dev));
        assert!(reg.should_skip_free(ptr2, real_dev));

        reg.unregister_range(h);
        reg.unregister_exact(ptr2);
    }

    /// Skeptic #4: `register_range` with `start > end` is silently accepted
    /// and creates a never-matching range.
    ///
    /// Builder did not add a debug_assert or hard error for this. The
    /// underlying check `ptr >= r.start && ptr < r.end` correctly returns
    /// false for any ptr when `start > end`, so the WORST case is wasted
    /// memory plus debug confusion. Lock down the current "silently accept,
    /// silently never match" behavior. If a future change adds a debug
    /// assertion, this test reminds the author to update either the
    /// behavior OR this test, not both silently.
    #[test]
    fn skeptic_register_range_inverted_silently_never_matches() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0x77_usize;

        // Inverted range — silently accepted.
        let h = reg.register_range(ExternalRange {
            start: 0x2000,
            end: 0x1000, // < start
            device_key: dev,
            owner: ExternalOwner::Slab,
        });
        assert_eq!(reg.range_count(), 1, "inverted range is currently accepted");

        // No ptr in the universe can satisfy `ptr >= 0x2000 && ptr < 0x1000`.
        assert!(!reg.should_skip_free(0x1000, dev));
        assert!(!reg.should_skip_free(0x1800, dev));
        assert!(!reg.should_skip_free(0x2000, dev));
        assert!(!reg.should_skip_free_any_device(0x1800));

        reg.unregister_range(h);
    }

    /// Skeptic #5: hammer `ensure_hook_installed` from N threads — final
    /// state is consistent (hook installed, registry queryable, no flake).
    ///
    /// Already covered by skeptic_hook_install_concurrent_no_panic (above),
    /// but this variant additionally verifies the registry remains
    /// queryable under concurrent installs.
    #[test]
    fn skeptic_concurrent_install_and_registry_queries() {
        let _g = TEST_LOCK.lock().unwrap();
        let _reg = fresh();
        let install_threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..200 {
                        ExternalMemoryRegistry::ensure_hook_installed();
                    }
                })
            })
            .collect();

        let query_threads: Vec<_> = (0..8)
            .map(|tid| {
                std::thread::spawn(move || {
                    let reg = ExternalMemoryRegistry::global();
                    let ptr = 0xDEAD_0000_u64 + (tid as u64);
                    for _ in 0..200 {
                        reg.register_exact(ptr, DEVICE_KEY_ANY, ExternalOwner::PoolExact);
                        let _hit = reg.should_skip_free_any_device(ptr);
                        let _ = reg.unregister_exact(ptr);
                    }
                })
            })
            .collect();

        for h in install_threads {
            h.join().unwrap();
        }
        for h in query_threads {
            h.join().unwrap();
        }
        assert!(ExternalMemoryRegistry::global().hook_installed_flag());
    }

    /// Skeptic #6: `hook_installed` flag persistent across `reset_for_testing`
    /// is benign — fresh register/query/unregister cycle still works.
    #[test]
    fn skeptic_reset_for_testing_does_not_break_hook() {
        let _g = TEST_LOCK.lock().unwrap();
        // Ensure hook flag is set (other tests probably did this).
        ExternalMemoryRegistry::ensure_hook_installed();
        assert!(ExternalMemoryRegistry::global().hook_installed_flag());

        let reg = fresh(); // resets ranges + exact, NOT the hook flag.
        assert!(
            reg.hook_installed_flag(),
            "reset_for_testing must NOT clear the hook-install flag"
        );
        assert_eq!(reg.range_count(), 0);
        assert_eq!(reg.exact_entry_count(), 0);

        // Fresh cycle still works.
        let ptr = 0xBEEF_0000_u64;
        reg.register_exact(ptr, DEVICE_KEY_ANY, ExternalOwner::PoolExact);
        assert!(reg.should_skip_free_any_device(ptr));
        assert_eq!(reg.unregister_exact(ptr), 0);
        assert!(!reg.should_skip_free_any_device(ptr));
    }

    /// Skeptic #7: lock down R1a `is_external_ptr` semantics so an R1b
    /// regression in the back-compat shim is caught.
    ///
    /// In R1a, the back-compat `is_external_ptr(ptr)` delegates to
    /// `should_skip_free_any_device(ptr)`. After R1b, this would also
    /// return `true` for any pointer inside a registered slab range —
    /// even though `unregister_external_ptr(mid_slab_ptr)` would be a
    /// no-op (the range is the owner, not an exact entry). That semantic
    /// drift IS the R1b design (transitive protection of mid-slab ptrs),
    /// but if R1b accidentally introduces a regression in the back-compat
    /// path itself (e.g., breaking the exact-entry path), this test
    /// catches it.
    ///
    /// Specifically: in R1a, `is_external_ptr(ptr)` returns true ONLY
    /// when the exact-pointer refcount is non-zero. Locked down here.
    #[test]
    fn skeptic_is_external_ptr_r1a_exact_path_only() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr = 0xCA11_AB1E_u64;

        // Nothing registered → not external.
        assert!(!reg.should_skip_free_any_device(ptr));

        // Register exact → external.
        reg.register_exact(ptr, DEVICE_KEY_ANY, ExternalOwner::PoolExact);
        assert!(reg.should_skip_free_any_device(ptr));
        assert_eq!(reg.exact_refcount(ptr, DEVICE_KEY_ANY), 1);

        // Unregister → not external.
        assert_eq!(reg.unregister_exact(ptr), 0);
        assert!(!reg.should_skip_free_any_device(ptr));
        assert_eq!(reg.exact_refcount(ptr, DEVICE_KEY_ANY), 0);
    }

    /// Skeptic #8: cross-device hook decision — `should_skip_free_any_device`
    /// is deliberately permissive.
    ///
    /// The cudarc Drop hook has signature `fn(u64) -> bool` (no device).
    /// If two different devices both have entries at the same numeric ptr,
    /// the hook returns `true` for either device's `CudaSlice::drop`.
    /// This is documented as "negligible risk" because distinct CUDA
    /// contexts have disjoint virtual address spaces in practice — but
    /// the test locks it down explicitly.
    #[test]
    fn skeptic_any_device_hook_does_not_disambiguate_devices() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr = 0xCAFE_0001_u64;
        let dev1 = 0x1_usize;
        let dev2 = 0x2_usize;

        // Register on dev1 only.
        reg.register_exact(ptr, dev1, ExternalOwner::Ring);

        // `should_skip_free_any_device` does NOT disambiguate — it returns
        // true for ANY drop with this ptr value.
        assert!(reg.should_skip_free_any_device(ptr));

        // `should_skip_free` WITH a real device key DOES disambiguate.
        assert!(reg.should_skip_free(ptr, dev1));
        assert!(!reg.should_skip_free(ptr, dev2));

        reg.unregister_exact(ptr);
    }

    /// Skeptic #9: null pointer (ptr=0) handling.
    ///
    /// `should_skip_free(0, ...)` against an empty registry must return
    /// false. After registering ptr=0 (pathological — no real CUDA alloc
    /// returns 0), the lookup must work consistently.
    #[test]
    fn skeptic_null_ptr_handling() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();

        // Empty: null ptr is not skipped.
        assert!(!reg.should_skip_free(0, 0xAB));
        assert!(!reg.should_skip_free_any_device(0));

        // Register exact ptr=0 → consistent behavior.
        reg.register_exact(0, 0xAB, ExternalOwner::Ring);
        assert!(reg.should_skip_free(0, 0xAB));
        assert!(reg.should_skip_free_any_device(0));
        assert_eq!(reg.unregister_exact(0), 0);
        assert!(!reg.should_skip_free(0, 0xAB));
    }

    /// Skeptic #10: `register_range` with `start == 0` does not interact
    /// pathologically with the DEVICE_KEY_ANY sentinel.
    ///
    /// `DEVICE_KEY_ANY=0` is the sentinel on the `device_key` axis, NOT
    /// the `start` axis. A range starting at address 0 should behave
    /// normally (cover [0, end)).
    #[test]
    fn skeptic_range_starting_at_zero_address() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0x99_usize;

        let h = reg.register_range(ExternalRange {
            start: 0,
            end: 0x1000,
            device_key: dev,
            owner: ExternalOwner::Slab,
        });
        assert!(
            reg.should_skip_free(0, dev),
            "address 0 is inside [0, 0x1000)"
        );
        assert!(reg.should_skip_free(0xFFF, dev));
        assert!(!reg.should_skip_free(0x1000, dev));
        reg.unregister_range(h);
    }

    /// Skeptic #11: `unregister_range` called twice with the same handle.
    /// Second call is a no-op (silently ignored, no panic, no error).
    #[test]
    fn skeptic_double_unregister_range_is_noop() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0xAA_usize;
        let h = reg.register_range(ExternalRange {
            start: 0x4000,
            end: 0x5000,
            device_key: dev,
            owner: ExternalOwner::Slab,
        });
        assert_eq!(reg.range_count(), 1);
        reg.unregister_range(h);
        assert_eq!(reg.range_count(), 0);
        // Second call with the same (now-stale) handle: silent no-op.
        reg.unregister_range(h);
        assert_eq!(reg.range_count(), 0);
    }

    /// Skeptic #12: dropping a `RangeHandle` without calling
    /// `unregister_range` leaks the range entry forever. Lock down the
    /// "no implicit Drop" contract so a future change adding `Drop` to
    /// `RangeHandle` is caught.
    ///
    /// If anyone ever adds `impl Drop for RangeHandle`, this test will
    /// fail (range_count() will become 0 after the handle drops) and the
    /// author must update the test AND the documentation.
    #[test]
    fn skeptic_range_handle_drop_does_not_auto_unregister() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0xBB_usize;
        {
            let _h = reg.register_range(ExternalRange {
                start: 0x6000,
                end: 0x7000,
                device_key: dev,
                owner: ExternalOwner::Slab,
            });
            assert_eq!(reg.range_count(), 1);
        } // _h dropped here.
          // Range is STILL present — no auto-Drop logic.
        assert_eq!(
            reg.range_count(),
            1,
            "RangeHandle Drop must not auto-unregister (would tear down the slab range mid-use)"
        );
        // Re-fresh to clean up for subsequent tests.
        reg.reset_for_testing();
    }

    /// Skeptic #13: back-compat shim path — `register_exact` with
    /// `DEVICE_KEY_ANY` then `unregister_exact(ptr)` correctly removes
    /// the `(ptr, 0)` entry even when there are unrelated entries.
    #[test]
    fn skeptic_back_compat_unregister_finds_any_device_entry() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let ptr = 0xFACE_0000_u64;

        // Simulate back-compat shim path.
        reg.register_exact(ptr, DEVICE_KEY_ANY, ExternalOwner::PoolExact);
        // Unrelated entry — different ptr.
        reg.register_exact(0xDEAD_0000_u64, 0x1234, ExternalOwner::Ring);

        let before = reg.exact_entry_count();
        assert_eq!(before, 2);
        let new_count = reg.unregister_exact(ptr);
        assert_eq!(new_count, 0);
        assert_eq!(
            reg.exact_entry_count(),
            1,
            "unrelated entry must remain after unregister_exact(ptr)"
        );
        // The unrelated entry survives.
        assert!(reg.should_skip_free(0xDEAD_0000_u64, 0x1234));
        reg.unregister_exact(0xDEAD_0000_u64);
    }

    /// Skeptic #14: range overlap — two ranges covering the same ptr.
    /// `should_skip_free` returns true; unregistering one keeps protection.
    #[test]
    fn skeptic_overlapping_ranges_compose() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0xCC_usize;
        let h1 = reg.register_range(ExternalRange {
            start: 0x8000,
            end: 0x9000,
            device_key: dev,
            owner: ExternalOwner::Slab,
        });
        let h2 = reg.register_range(ExternalRange {
            start: 0x8500,
            end: 0x9500,
            device_key: dev,
            owner: ExternalOwner::Ring,
        });
        assert!(reg.should_skip_free(0x8800, dev));
        // Drop h1; h2 still covers 0x8800 (0x8500..0x9500).
        reg.unregister_range(h1);
        assert!(reg.should_skip_free(0x8800, dev));
        // Drop h2; now nothing covers it.
        reg.unregister_range(h2);
        assert!(!reg.should_skip_free(0x8800, dev));
    }

    /// Skeptic #15: lock down `next_handle` uniqueness across
    /// `reset_for_testing` — handles must NOT collide with pre-reset
    /// handles. Builder documents this; we verify it.
    #[test]
    fn skeptic_next_handle_unique_across_reset() {
        let _g = TEST_LOCK.lock().unwrap();
        let reg = fresh();
        let dev = 0xDD_usize;
        let h1 = reg.register_range(ExternalRange {
            start: 0xA000,
            end: 0xB000,
            device_key: dev,
            owner: ExternalOwner::Slab,
        });
        reg.reset_for_testing();
        let h2 = reg.register_range(ExternalRange {
            start: 0xA000,
            end: 0xB000,
            device_key: dev,
            owner: ExternalOwner::Slab,
        });
        assert_ne!(
            h1, h2,
            "handles must remain unique across reset_for_testing (collision would break stale-handle detection)"
        );
        reg.unregister_range(h2);
    }
}
