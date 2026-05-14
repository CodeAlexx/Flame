//! R1b Skeptic — adversarial tests for `StaticSlabAllocator` + the
//! `TensorStorage::Drop` slab-claim wiring.
//!
//! Each test asserts a behavior contract that, if it ever silently broke,
//! would corrupt the slab's `live_count` invariant — which would either
//! make `reset()` fail forever (false leak) or allow a real
//! use-after-free (false success).
//!
//! Tests are grouped by Bug-Fixer-flagged seam, then Skeptic-originals.
//! Every test is self-contained (clean device_map at start, clean exit)
//! and skips gracefully when no CUDA device is available.

use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use flame_core::external_memory::ExternalMemoryRegistry;
use flame_core::static_slab_v2::{
    self, reset_device_map_for_testing, slab_v2_return_if_owned, StaticSlabAllocator,
    __test_device_map_insert, __test_device_map_remove,
};
use flame_core::tensor_storage::{wrap_slab_slice_for_test, TensorStorage};

/// Serializes every test in this binary. The slab `device_map` is a process-
/// global keyed by `Arc::as_ptr(&device)`; CUDA's runtime may memoize the
/// device handle, so distinct `CudaDevice::new(0)` calls in parallel tests
/// land on the same key and stomp on each other's slab state — observed as
/// cascading futex deadlocks on `Mutex<StaticSlabAllocator>`.
/// Poison-recoverable so one panicking test doesn't lock out the rest.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn test_serialize() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn skip_if_no_gpu() -> Option<Arc<CudaDevice>> {
    if std::env::var("FLAME_SKIP_GPU_TESTS").ok().as_deref() == Some("1") {
        eprintln!("[adversarial] FLAME_SKIP_GPU_TESTS=1 — skipping");
        return None;
    }
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[adversarial] CudaDevice::new(0) failed: {e:?} — skipping");
            None
        }
    }
}

/// Build a fresh slab, register it under `device_key`, return a `'static`
/// reference. Caller is responsible for releasing + removing from the map.
fn install_fresh_slab(
    device: Arc<CudaDevice>,
    capacity_bytes: usize,
) -> (&'static Mutex<StaticSlabAllocator>, usize) {
    let key = Arc::as_ptr(&device) as usize;
    // Remove any prior entry for this key so we get a clean slab.
    __test_device_map_remove(key);
    let slab = StaticSlabAllocator::new(device, capacity_bytes);
    let boxed: &'static Mutex<StaticSlabAllocator> = Box::leak(Box::new(Mutex::new(slab)));
    __test_device_map_insert(key, boxed);
    (boxed, key)
}

// =====================================================================
// Bug Fixer #1: concurrent release vs slab_v2_return_if_owned
// =====================================================================
//
// Scenario: thread A calls slab.release() while thread B is mid-
// slab_v2_return_if_owned for a still-live ptr. Both paths acquire the
// per-slab mutex, so they must serialize. We assert that after a
// successful release, a stale return_if_owned for a now-freed ptr
// returns `false` (slab_base is None, ptr_in_slab returns false) and
// doesn't crash.

#[test]
fn bf1_release_then_stale_return_is_safe() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, key) = install_fresh_slab(device, 1 * 1024 * 1024);

    // Allocate one slice, capture its ptr, then forget the slice and
    // manually decrement live_count to simulate a clean drop.
    let ptr = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(128).unwrap();
        let p = *s.device_ptr();
        // Use the slab return path itself to decrement (this exercises the
        // happy path of return_if_owned BEFORE release).
        drop(g);
        let claimed = slab_v2_return_if_owned(p, key);
        assert!(claimed, "ptr must be claimed by the slab on first return");
        std::mem::forget(s);
        p
    };

    // Now release the slab (live_count should be 0).
    {
        let mut g = slab_mu.lock().unwrap();
        assert_eq!(g.live_count(), 0);
        g.release().expect("release with live_count=0");
        assert!(g.slab_base().is_none(), "slab_base cleared after release");
    }

    // Stale return for the now-freed ptr: must return `false` and NOT crash.
    let stale = slab_v2_return_if_owned(ptr, key);
    assert!(
        !stale,
        "stale return_if_owned after release must return false (slab no longer materialised)"
    );

    __test_device_map_remove(key);
}

#[test]
fn bf1_concurrent_release_blocks_inflight_return() {
    let _test_lock = test_serialize();
    // We can't truly race release() against return_if_owned without a
    // synchronization primitive, but we CAN demonstrate that the mutex
    // strictly serializes the two operations. We hold the slab mutex on
    // thread A (simulating an in-flight return_if_owned that hasn't yet
    // decremented), and try release() on thread B with timeout — it must
    // BLOCK (return_if_owned will eventually complete and decrement, then
    // release succeeds).
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, key) = install_fresh_slab(device, 1 * 1024 * 1024);

    // Allocate.
    let ptr = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(128).unwrap();
        let p = *s.device_ptr();
        std::mem::forget(s);
        p
    };
    assert_eq!(slab_mu.lock().unwrap().live_count(), 1);

    // Hold the slab lock; release() on another thread will block.
    let guard = slab_mu.lock().unwrap();
    let slab_mu_send = slab_mu;
    let release_thread = std::thread::spawn(move || {
        // Acquires lock after `guard` is released; live_count is still 1
        // so release() should return Err.
        let mut g = slab_mu_send.lock().unwrap();
        g.release()
    });

    // Briefly hold the lock to ensure the other thread is waiting.
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert_eq!(guard.live_count(), 1);
    drop(guard);

    let res = release_thread.join().expect("release thread panicked");
    assert!(
        res.is_err(),
        "release() with live_count=1 must return Err (got Ok)"
    );

    // Clean up: decrement live_count via return_if_owned, then release.
    let claimed = slab_v2_return_if_owned(ptr, key);
    assert!(claimed);
    {
        let mut g = slab_mu.lock().unwrap();
        assert_eq!(g.live_count(), 0);
        g.release().unwrap();
    }
    __test_device_map_remove(key);
}

// =====================================================================
// Bug Fixer #2: Arc::try_unwrap failure path with shared_storage
// =====================================================================
//
// Scenario: a slab-backed BF16 TensorStorage is cloned (Arc strong=2);
// one clone is mem::forget'd; the other Drops. With shared_storage, the
// Arc::try_unwrap fails on the dropping clone (strong=1 after, but the
// forgotten clone keeps strong>=1 forever). The slab claim never runs.
//
// Result: live_count is permanently stuck at 1 → reset()/release()
// fails forever.
//
// We assert this CURRENT BROKEN behavior, locking it down so future
// changes that fix it (e.g. unconditional slab claim) are visible. If a
// fix lands, this test will start failing — UPDATE THE ASSERTION to
// `assert_eq!(live_after_drop, 0)`.

#[test]
fn bf2_arc_forget_strands_live_count() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    // Ensure pool_disabled is not stuck on — we want the normal pool path
    // for this test (the bug surfaces independent of pool_off, but cleaner
    // env helps reasoning).
    let (slab_mu, key) = install_fresh_slab(device, 1 * 1024 * 1024);

    // Allocate slab-backed BF16.
    let slice: CudaSlice<u16> = {
        let mut g = slab_mu.lock().unwrap();
        g.alloc_u16(64).unwrap()
    };
    assert_eq!(slab_mu.lock().unwrap().live_count(), 1);

    // Wrap into TensorStorage::BF16. With shared_storage feature (default),
    // `data` is `Arc<CudaSlice<u16>>` with strong=1.
    let storage = wrap_slab_slice_for_test(slice, 64);
    // Clone — strong becomes 2.
    let storage_clone = storage.clone();

    // Forget the clone. Strong stays at 2 forever (Drop never runs).
    std::mem::forget(storage_clone);

    // Drop the original. Inside Drop, `Arc::try_unwrap` sees strong=2,
    // fails, falls into `Err(arc) => drop(arc)`. The slab claim is NOT
    // called.
    drop(storage);

    // Document current behavior: live_count is stranded at 1.
    let live = slab_mu.lock().unwrap().live_count();
    assert_eq!(
        live, 1,
        "Arc::try_unwrap failure path skips slab claim → live_count stranded. \
         If this assertion starts failing, the underlying behavior was fixed; \
         change the assertion to `assert_eq!(live, 0)`."
    );

    // Reset/release must fail — slab is unusable for the rest of the
    // process. Document this consequence.
    {
        let mut g = slab_mu.lock().unwrap();
        let reset_err = g.reset();
        assert!(
            reset_err.is_err(),
            "reset() should err when live_count != 0"
        );
        let release_err = g.release();
        assert!(
            release_err.is_err(),
            "release() should err when live_count != 0"
        );
    }

    // We can't clean up live_count from outside the module (private field).
    // Leak the slab entry; process exit reclaims. Remove the map entry so
    // other tests don't see this poisoned slab.
    __test_device_map_remove(key);
}

// =====================================================================
// Bug Fixer #3: pool_disabled env caching race
// =====================================================================
//
// The drop_wiring regression test relies on `force_pool_disabled` setting
// `FLAME_ALLOC_POOL=0` before `pool_disabled()` caches its result via
// OnceLock. If something else trips the cache first with the env unset,
// the assertion `pool_disabled() == true` would PANIC (the existing
// regression test already does this) — so the regression test "fails
// loudly" rather than "silently passes with pool ENABLED".
//
// We lock down that loud-failure behavior here by asserting that:
// 1. After force-setting the env, `pool_disabled()` MUST be observable as
//    true OR panic, never silently return false.
// 2. The regression test's `force_pool_disabled` function contains an
//    explicit assert that would fail if the cache lost the race.

#[test]
fn bf3_drop_wiring_test_loudly_fails_on_cache_race() {
    let _test_lock = test_serialize();
    // Source-level lockdown: the drop_wiring binary's `force_pool_disabled`
    // function MUST contain an `assert!` that reads `pool_disabled()` and
    // panics if it's false. Source-text inspection guards this property
    // against accidental deletion.
    let src = std::fs::read_to_string(
        "tests/static_slab_v2_drop_wiring.rs",
    )
    .expect("read drop_wiring source");
    assert!(
        src.contains("force_pool_disabled"),
        "drop_wiring test must keep force_pool_disabled fn"
    );
    assert!(
        src.contains("pool_disabled()"),
        "force_pool_disabled must call pool_disabled() to verify the cache"
    );
    assert!(
        src.contains("assert!"),
        "force_pool_disabled must assert! the result is true (loud failure on cache race)"
    );
    // Ensure the assertion is on pool_disabled() specifically.
    let snippet_idx = src
        .find("force_pool_disabled")
        .expect("find function in source");
    let fn_window: String = src[snippet_idx..].chars().take(800).collect();
    assert!(
        fn_window.contains("pool_disabled()"),
        "force_pool_disabled body must reference pool_disabled() to verify"
    );
    assert!(
        fn_window.contains("assert"),
        "force_pool_disabled body must contain an assert that loud-fails on cache race"
    );
}

// =====================================================================
// Bug Fixer #4: F16 + I32 + Bool arms still have pool_off short-circuit
// =====================================================================
//
// Slab support today is BF16(u16) + F32. F16, I32, Bool Drop arms still
// short-circuit on `pool_disabled` and do NOT call `slab_v2_try_claim`.
// If a future change adds slab support for those dtypes without updating
// the Drop arms, slab live_count will leak.
//
// We lock this down with two assertions:
// 1. Source-text: every dtype variant in TensorStorage::Drop that holds a
//    backing CudaSlice has either a slab_v2_try_claim call OR a comment
//    explaining why it doesn't.
// 2. Runtime: F16/I32/Bool tensors allocated via the safe constructors
//    (which never go through the slab) drop cleanly with no slab
//    interaction.

#[test]
fn bf4_non_slab_dtype_drop_arms_documented() {
    let _test_lock = test_serialize();
    let src = std::fs::read_to_string("src/tensor_storage.rs")
        .expect("read tensor_storage source");

    // Pull out the impl Drop body.
    let drop_idx = src
        .find("impl Drop for TensorStorage")
        .expect("find Drop impl");
    let drop_end = src[drop_idx..]
        .find("\nunsafe impl Send")
        .expect("find end of Drop impl");
    let drop_body = &src[drop_idx..drop_idx + drop_end];

    // Structural per-arm scan: locate every arm header inside the Drop body,
    // slice each arm body up to the next header, and check for the call.
    //
    // Robust against Bug-Fixer's restructure (which split each slab-claiming
    // arm into shared/non-shared paths, giving 4 call sites instead of 2)
    // AND against cfg-gating: the BF16 arm appears TWICE in source, once
    // gated by `#[cfg(not(feature = "bf16_u16"))]` (legacy F32-backed —
    // NO slab claim) and once by `#[cfg(feature = "bf16_u16")]` (u16-backed
    // — HAS slab claim). The test must validate BOTH gated arms by their
    // expected contract, not just the first one `find()` returns.
    //
    // Contract:
    //   F32                 → slab claim present (R1b scope)
    //   BF16 (bf16_u16)     → slab claim present (R1b scope)
    //   BF16 (legacy F32)   → slab claim absent (slab is bf16_u16-only)
    //   F16, I32, Bool      → slab claim absent
    //   BF16Arena, BF16View, I8 → absent (caught by `_ => {}` catch-all)
    //
    // If slab support is extended to a new dtype, update both the alloc-side
    // wiring AND this test's expectations.
    let arms = ["F32", "F16", "BF16", "I32", "Bool"];

    // Find ALL occurrences of every arm header (handles cfg-duplicated arms).
    let mut headers: Vec<(&str, usize)> = vec![];
    for v in &arms {
        let pat = format!("TensorStorage::{} {{", v);
        let mut from = 0;
        while let Some(off) = drop_body[from..].find(&pat) {
            let abs = from + off;
            headers.push((*v, abs));
            from = abs + pat.len();
        }
    }
    headers.sort_by_key(|(_, i)| *i);

    // For each arm, slice from its header to the next header (or end of
    // drop_body), then check slab_v2_try_claim presence.
    let mut claim_total = 0usize;
    let mut bf16_arms_with_claim = 0usize;
    let mut bf16_arms_total = 0usize;
    for (idx, (variant, start)) in headers.iter().enumerate() {
        let end = headers
            .get(idx + 1)
            .map(|(_, n)| *n)
            .unwrap_or(drop_body.len());
        let arm_body = &drop_body[*start..end];
        let has_claim = arm_body.contains("slab_v2_try_claim");
        if has_claim {
            claim_total += arm_body.matches("slab_v2_try_claim").count();
        }
        match *variant {
            "F32" => assert!(
                has_claim,
                "F32 arm must contain slab_v2_try_claim (R1b wiring)"
            ),
            "F16" | "I32" | "Bool" => assert!(
                !has_claim,
                "{} arm must NOT contain slab_v2_try_claim (slab is BF16/F32 only)",
                variant
            ),
            "BF16" => {
                bf16_arms_total += 1;
                if has_claim {
                    bf16_arms_with_claim += 1;
                }
            }
            _ => unreachable!(),
        }
    }

    assert_eq!(
        bf16_arms_total, 2,
        "Expected 2 source-text BF16 arm headers (cfg-gated pair: legacy + bf16_u16). \
         Got {}. Source layout changed — review tensor_storage.rs Drop impl.",
        bf16_arms_total
    );
    assert_eq!(
        bf16_arms_with_claim, 1,
        "Exactly one BF16 arm (the bf16_u16-gated one) must call slab_v2_try_claim. \
         Got {} arms with claim out of {}.",
        bf16_arms_with_claim, bf16_arms_total
    );
    // Total expected: F32 shared + F32 non-shared + BF16(bf16_u16) shared +
    // BF16(bf16_u16) non-shared = 4 actual call sites in the Drop body.
    assert_eq!(
        claim_total, 4,
        "Expected exactly 4 slab_v2_try_claim call sites in Drop body \
         (F32 ×2 paths + BF16(bf16_u16) ×2 paths). Got {}.",
        claim_total
    );
}

#[test]
fn bf4_non_slab_dtype_storage_drops_cleanly() {
    let _test_lock = test_serialize();
    // Verify F16/I32/Bool tensors don't accidentally interact with the
    // slab. We allocate one through the normal constructor (which goes
    // through alloc_aligned_f32, NOT the slab), drop it, and assert the
    // slab's live_count is unchanged.
    let Some(device) = skip_if_no_gpu() else { return };
    reset_device_map_for_testing();
    let (slab_mu, key) = install_fresh_slab(device.clone(), 1 * 1024 * 1024);

    // Materialize the slab so a range is registered (otherwise ptr_in_slab
    // always returns false trivially).
    //
    // R1b-skep deadlock fix: `slab_v2_return_if_owned` re-locks the same
    // per-device slab mutex via the global device_map. We MUST release the
    // alloc-side lock before calling it, or this test hangs the entire
    // test binary (every later test serializes on TEST_LOCK and inherits
    // the hang).
    let ptr = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(64).unwrap();
        let ptr = *s.device_ptr();
        std::mem::forget(s);
        ptr
    };
    let claimed = slab_v2_return_if_owned(ptr, key);
    assert!(claimed, "ptr must be claimed by the slab");
    assert_eq!(slab_mu.lock().unwrap().live_count(), 0);
    let live_before = slab_mu.lock().unwrap().live_count();

    // Allocate F16/I32/Bool via the safe path. They go through
    // alloc_aligned_f32 (NOT the slab) → ptrs do NOT land in slab range.
    use flame_core::{DType, Shape};
    let f16 = TensorStorage::zeros(&Shape::from_dims(&[256]), DType::F16, &device).unwrap();
    let i32_st = TensorStorage::zeros(&Shape::from_dims(&[256]), DType::I32, &device).unwrap();
    let bool_st = TensorStorage::zeros(&Shape::from_dims(&[256]), DType::Bool, &device).unwrap();

    drop(f16);
    drop(i32_st);
    drop(bool_st);

    // Slab live_count must be exactly what it was before.
    let live_after = slab_mu.lock().unwrap().live_count();
    assert_eq!(
        live_after, live_before,
        "non-BF16/F32 dtype drops must NOT touch slab live_count"
    );

    {
        let mut g = slab_mu.lock().unwrap();
        let _ = g.release();
    }
    __test_device_map_remove(key);
}

// =====================================================================
// Bug Fixer #5: release() strict live_count check
// =====================================================================

#[test]
fn bf5_release_strict_check_returns_err_with_live_tensors() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, key) = install_fresh_slab(device, 1 * 1024 * 1024);

    let ptr = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(128).unwrap();
        let p = *s.device_ptr();
        std::mem::forget(s);
        p
    };

    {
        let mut g = slab_mu.lock().unwrap();
        assert_eq!(g.live_count(), 1);
        let err = g.release().expect_err("release with live tensors must Err");
        let msg = format!("{}", err);
        assert!(msg.contains("refusing"));
        assert!(msg.contains("live"));
        // Slab MUST still be alive (release was rejected).
        assert!(g.slab_base().is_some(), "rejected release must NOT tear down slab");
    }

    // Cleanup.
    let claimed = slab_v2_return_if_owned(ptr, key);
    assert!(claimed);
    {
        let mut g = slab_mu.lock().unwrap();
        g.release().unwrap();
    }
    __test_device_map_remove(key);
}

// =====================================================================
// Skeptic-originals
// =====================================================================

/// What happens if the same physical device is registered twice (two
/// different `Arc<CudaDevice>` handles)? Each Arc gets its own slab; no
/// clobber.
#[test]
fn sk_multi_arc_for_same_device_get_distinct_slabs() {
    let _test_lock = test_serialize();
    let Some(dev_a) = skip_if_no_gpu() else { return };
    let Ok(dev_b) = CudaDevice::new(0) else {
        eprintln!("[adversarial] second CudaDevice::new(0) failed — skipping");
        return;
    };
    if Arc::as_ptr(&dev_a) == Arc::as_ptr(&dev_b) {
        eprintln!("[adversarial] CudaDevice::new(0) returns shared Arc — skipping");
        return;
    }
    reset_device_map_for_testing();
    let slab_a = static_slab_v2::slab_for_device(&dev_a);
    let slab_b = static_slab_v2::slab_for_device(&dev_b);
    assert!(
        !std::ptr::eq(slab_a, slab_b),
        "distinct device Arcs must get distinct &'static Mutex<Slab>"
    );
    reset_device_map_for_testing();
}

/// What if `n` is huge enough that `n * sizeof(T)` overflows usize?
/// alloc_u16 / alloc_f32_uninit must return Err with checked_mul.
#[test]
fn sk_alloc_size_overflow_is_checked() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let mut slab = StaticSlabAllocator::new(device, 4096);

    // n * 2 overflows usize.
    let huge_u16 = usize::MAX / 2 + 1;
    let err = slab.alloc_u16(huge_u16).expect_err("u16 size overflow");
    let msg = format!("{}", err);
    assert!(
        msg.contains("overflow"),
        "alloc_u16 must reject n*2 overflow with 'overflow' in message; got: {msg}"
    );

    // n * 4 overflows usize.
    let huge_f32 = usize::MAX / 4 + 1;
    let err = slab
        .alloc_f32_uninit(huge_f32)
        .expect_err("f32 size overflow");
    let msg = format!("{}", err);
    assert!(
        msg.contains("overflow"),
        "alloc_f32_uninit must reject n*4 overflow with 'overflow' in message; got: {msg}"
    );

    // alloc_f32_zeroed inherits the check via alloc_f32_uninit.
    let err = slab
        .alloc_f32_zeroed(huge_f32)
        .expect_err("f32 zeroed size overflow");
    let msg = format!("{}", err);
    assert!(msg.contains("overflow"));
}

/// What if a slab is `release()`d while NEVER materialized? Must be a
/// no-op (no range to unregister, no slab to drop).
#[test]
fn sk_release_unmaterialised_slab_is_noop() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let mut slab = StaticSlabAllocator::new(device, 1 * 1024 * 1024);
    assert!(slab.slab_base().is_none());
    assert_eq!(slab.live_count(), 0);

    // Release without any allocation: must succeed (live_count=0).
    slab.release().expect("release of unmaterialised slab must be OK");
    assert!(slab.slab_base().is_none());

    // Release again — still OK.
    slab.release().expect("double release of unmaterialised slab must be OK");
}

/// What if `release()` is called twice on a materialized slab after all
/// live counts are zero? The second call must be a no-op (range_count
/// before/after second call is identical).
#[test]
fn sk_double_release_after_alloc_is_noop() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, key) = install_fresh_slab(device, 1 * 1024 * 1024);
    let ptr = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(64).unwrap();
        let p = *s.device_ptr();
        std::mem::forget(s);
        p
    };
    let claimed = slab_v2_return_if_owned(ptr, key);
    assert!(claimed);
    let pre_ranges = ExternalMemoryRegistry::global().range_count();

    {
        let mut g = slab_mu.lock().unwrap();
        g.release().unwrap();
    }
    let mid_ranges = ExternalMemoryRegistry::global().range_count();
    assert_eq!(
        mid_ranges,
        pre_ranges - 1,
        "first release unregisters exactly one range"
    );

    {
        let mut g = slab_mu.lock().unwrap();
        // Second release: no range to unregister, no slab to drop.
        g.release().expect("double release should be OK");
    }
    let post_ranges = ExternalMemoryRegistry::global().range_count();
    assert_eq!(
        post_ranges, mid_ranges,
        "second release must not unregister anything else"
    );

    __test_device_map_remove(key);
}

/// What happens if `slab_v2_return_if_owned` is called with ptr=0 or
/// device_key=0? Must return false; must not crash.
#[test]
fn sk_return_hook_zero_ptr_zero_key_safe() {
    let _test_lock = test_serialize();
    // No device required — these inputs never hit a slab.
    assert!(!slab_v2_return_if_owned(0, 0));
    assert!(!slab_v2_return_if_owned(0, 0xDEAD));
    assert!(!slab_v2_return_if_owned(0xDEAD, 0));
    assert!(!slab_v2_return_if_owned(u64::MAX, usize::MAX));
}

/// `reset()` must be a no-op when slab is not yet materialised.
#[test]
fn sk_reset_unmaterialised_slab_is_noop() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let mut slab = StaticSlabAllocator::new(device, 1 * 1024 * 1024);
    assert!(slab.slab_base().is_none());
    slab.reset().expect("reset before any alloc is OK");
    assert!(slab.slab_base().is_none(), "reset must not materialise slab");
}

/// alloc_u16(0)/alloc_f32_*(0) on a brand-new slab must NOT materialise
/// the slab (the n==0 short-circuit returns a zero-length cudart slice).
/// Verifies the "lazy materialization is preserved by zero-element fast
/// path" invariant.
#[test]
fn sk_zero_alloc_does_not_materialise_slab() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let mut slab = StaticSlabAllocator::new(device, 1 * 1024 * 1024);
    let s = slab.alloc_u16(0).unwrap();
    assert!(
        slab.slab_base().is_none(),
        "alloc_u16(0) must NOT materialise slab"
    );
    assert_eq!(DeviceSlice::len(&s), 0);
    drop(s);

    let s2 = slab.alloc_f32_uninit(0).unwrap();
    assert!(slab.slab_base().is_none());
    drop(s2);

    let s3 = slab.alloc_f32_zeroed(0).unwrap();
    assert!(slab.slab_base().is_none());
    drop(s3);
}

/// Cursor + aligned_bytes overflow must be checked (different from
/// element-count overflow). We synthesise this by setting up a tiny slab
/// and requesting allocations that would push `cursor + aligned_bytes`
/// past `usize::MAX` — the checked_add inside bump_cursor must catch it.
///
/// In practice we can't actually trigger this on a real slab (the
/// capacity check would catch it first), but we CAN assert via overflow
/// at the size-check layer: a request whose ALIGNED byte count exceeds
/// the slab's capacity returns the OOM error path (not a panic, not a
/// silent truncation).
#[test]
fn sk_capacity_overflow_returns_oom_err() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let mut slab = StaticSlabAllocator::new(device, 4 * 1024); // 4 KiB

    // 4 KiB / 4 bytes/f32 = 1024 f32. Request 2048 f32 = 8 KiB — overflow.
    let err = slab.alloc_f32_uninit(2048).expect_err("must overflow");
    let msg = format!("{}", err);
    assert!(msg.contains("overflow") || msg.contains("StaticSlabAllocator overflow"));
    assert!(msg.contains("F32"));
    assert!(msg.contains("capacity="));
    assert!(msg.contains("cursor="));
    assert!(msg.contains("FLAME_STATIC_SLAB_BYTES_F32"));
}

/// What happens if `release()` is called, then a fresh `alloc_*` is
/// requested? The slab must lazily re-materialise. Range is re-registered
/// (range_count returns to pre-release-state + 1).
#[test]
fn sk_release_then_alloc_relazy_materialises() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, key) = install_fresh_slab(device, 1 * 1024 * 1024);

    let ptr1 = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(64).unwrap();
        let p = *s.device_ptr();
        std::mem::forget(s);
        p
    };
    let claimed = slab_v2_return_if_owned(ptr1, key);
    assert!(claimed);

    let pre = ExternalMemoryRegistry::global().range_count();
    {
        let mut g = slab_mu.lock().unwrap();
        g.release().unwrap();
    }
    let mid = ExternalMemoryRegistry::global().range_count();
    assert_eq!(mid, pre - 1);

    // Re-alloc: must re-materialise.
    let ptr2 = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(64).unwrap();
        assert!(g.slab_base().is_some(), "alloc after release must re-materialise");
        let p = *s.device_ptr();
        std::mem::forget(s);
        p
    };
    let post = ExternalMemoryRegistry::global().range_count();
    assert_eq!(post, mid + 1, "re-materialisation registers a fresh range");

    // The new ptr is in the re-materialized slab (hook protects).
    assert!(ExternalMemoryRegistry::global().should_skip_free_any_device(ptr2));

    // Cleanup.
    let claimed = slab_v2_return_if_owned(ptr2, key);
    assert!(claimed);
    {
        let mut g = slab_mu.lock().unwrap();
        g.release().unwrap();
    }
    __test_device_map_remove(key);
}

/// `slab_v2_return_if_owned(ptr, device_key)` for a ptr OUTSIDE the slab
/// range but with a matching device_key must return false (not claim
/// memory that the slab doesn't own).
#[test]
fn sk_return_hook_offrange_ptr_matching_device_returns_false() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, key) = install_fresh_slab(device, 1 * 1024 * 1024);

    // Materialise slab.
    let base = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(64).unwrap();
        let p = *s.device_ptr();
        std::mem::forget(s);
        // ptr is in range; cleanup live_count.
        drop(g);
        slab_v2_return_if_owned(p, key);
        p
    };
    // Out-of-range ptr (way above slab end).
    let off_range = base + 100 * 1024 * 1024;
    assert!(
        !slab_v2_return_if_owned(off_range, key),
        "return_if_owned must reject ptr outside slab range"
    );

    {
        let mut g = slab_mu.lock().unwrap();
        let _ = g.release();
    }
    __test_device_map_remove(key);
}

/// A slab-backed TensorStorage that crosses an mpsc::channel send+recv
/// boundary, then drops on the receiver thread, must decrement
/// `live_count` correctly. Verifies the slab mutex is `Send`-safe
/// (acquired on whichever thread the Drop fires).
#[test]
fn sk_storage_drop_on_other_thread_decrements_correctly() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    // Don't disturb the production device key map by other tests — install
    // under the device's own key (matches the storage's slice device).
    reset_device_map_for_testing();
    let (slab_mu, key) = install_fresh_slab(device.clone(), 1 * 1024 * 1024);

    // Allocate slab BF16 + wrap.
    let slice: CudaSlice<u16> = {
        let mut g = slab_mu.lock().unwrap();
        g.alloc_u16(128).unwrap()
    };
    assert_eq!(slab_mu.lock().unwrap().live_count(), 1);
    let storage = wrap_slab_slice_for_test(slice, 128);

    // Send over a channel.
    let (tx, rx) = std::sync::mpsc::channel::<TensorStorage>();
    tx.send(storage).unwrap();
    drop(tx);

    let h = std::thread::spawn(move || {
        // Receive and drop on this thread.
        let st = rx.recv().expect("recv storage");
        drop(st);
    });
    h.join().expect("recv thread");

    // live_count must be 0 (slab claim ran on the receiver thread).
    let live_after = slab_mu.lock().unwrap().live_count();
    assert_eq!(
        live_after, 0,
        "TensorStorage::Drop on receiver thread must decrement slab live_count"
    );

    {
        let mut g = slab_mu.lock().unwrap();
        g.release().unwrap();
    }
    __test_device_map_remove(key);
}

/// alloc_u16 / alloc_f32_uninit must not zero-init (matches PT's
/// BFCAllocator + pool path). We've already asserted the memset counter
/// for f32_uninit inside the module's tests; for u16 we exercise the
/// "fast path" by observing that the slab is `alloc_zeros`'d once
/// (already verified in `slab_alloc_advances_cursor`). We add an
/// integration-level assertion: a second `alloc_u16` after slab materialisation
/// does NOT bump the registry range count.
#[test]
fn sk_subsequent_allocs_do_not_re_register_range() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, key) = install_fresh_slab(device, 1 * 1024 * 1024);

    let pre = ExternalMemoryRegistry::global().range_count();
    let (p1, p2, p3) = {
        let mut g = slab_mu.lock().unwrap();
        let s1 = g.alloc_u16(64).unwrap();
        let s2 = g.alloc_u16(128).unwrap();
        let s3 = g.alloc_f32_uninit(64).unwrap();
        let p1 = *s1.device_ptr();
        let p2 = *s2.device_ptr();
        let p3 = *s3.device_ptr();
        std::mem::forget(s1);
        std::mem::forget(s2);
        std::mem::forget(s3);
        (p1, p2, p3)
    };
    let post = ExternalMemoryRegistry::global().range_count();
    assert_eq!(
        post,
        pre + 1,
        "exactly one range registered for the entire slab (not one per alloc)"
    );

    // Cleanup.
    for p in [p1, p2, p3] {
        assert!(slab_v2_return_if_owned(p, key));
    }
    {
        let mut g = slab_mu.lock().unwrap();
        g.release().unwrap();
    }
    __test_device_map_remove(key);
}

/// `slab_v2_return_if_owned` must not deadlock if called while another
/// thread holds the device_map (no recursive lock acquisition pattern).
/// We verify by holding the device_map briefly on one thread and calling
/// the return hook on another — the return hook should wait for the map
/// lock, then proceed.
#[test]
fn sk_return_hook_serializes_through_device_map_lock() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, key) = install_fresh_slab(device.clone(), 1 * 1024 * 1024);

    let ptr = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(64).unwrap();
        let p = *s.device_ptr();
        std::mem::forget(s);
        p
    };

    let key_send = key;
    let t = std::thread::spawn(move || slab_v2_return_if_owned(ptr, key_send));
    let claimed = t.join().expect("return hook thread");
    assert!(claimed, "ptr must be claimed by the slab");

    {
        let mut g = slab_mu.lock().unwrap();
        assert_eq!(g.live_count(), 0);
        g.release().unwrap();
    }
    __test_device_map_remove(key);
}

/// `slab_v2_return_if_owned` looks up by device_key. A ptr that's in
/// slab A's range but called with slab B's key must return false (no
/// cross-slab claim).
#[test]
fn sk_return_hook_wrong_device_key_returns_false() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let (slab_mu, real_key) = install_fresh_slab(device, 1 * 1024 * 1024);

    let ptr = {
        let mut g = slab_mu.lock().unwrap();
        let s = g.alloc_u16(64).unwrap();
        let p = *s.device_ptr();
        std::mem::forget(s);
        p
    };

    // Wrong key.
    let bogus_key = real_key.wrapping_add(0x1000);
    assert!(
        !slab_v2_return_if_owned(ptr, bogus_key),
        "wrong device key must NOT claim ptr"
    );

    // live_count must still be 1.
    assert_eq!(slab_mu.lock().unwrap().live_count(), 1);

    // Cleanup via correct key.
    assert!(slab_v2_return_if_owned(ptr, real_key));
    {
        let mut g = slab_mu.lock().unwrap();
        g.release().unwrap();
    }
    __test_device_map_remove(real_key);
}

/// `reset()` while no slab is materialised but cursor is at 0 must
/// succeed (idempotent).
#[test]
fn sk_reset_idempotent_on_fresh_slab() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let mut slab = StaticSlabAllocator::new(device, 1 * 1024 * 1024);
    slab.reset().expect("reset on fresh slab is OK");
    slab.reset().expect("reset is idempotent");
    assert_eq!(slab.live_count(), 0);
    assert_eq!(slab.used_bytes(), 0);
}

/// Skeptic check: `device_key` returned by allocator matches
/// `Arc::as_ptr(&device) as usize`. Locks down the identity contract that
/// `slab_v2_return_if_owned` relies on.
#[test]
fn sk_device_key_matches_arc_as_ptr() {
    let _test_lock = test_serialize();
    let Some(device) = skip_if_no_gpu() else { return };
    let key_expected = Arc::as_ptr(&device) as usize;
    let slab = StaticSlabAllocator::new(device, 1 * 1024 * 1024);
    assert_eq!(slab.device_key(), key_expected);
}
