#![cfg(feature = "cuda")]

//! Regression test for the F32 pool cache opt-out shipped 2026-05-15.
//!
//! Asserts the behavioral invariants that protect the Phase 3 workaround
//! removal (TENETS §4 measurement gate, plan
//! `~/.claude/plans/mossy-discovering-ember.md`):
//!
//! 1. F32 allocations succeed with `FLAME_F32_POOL_CACHE` unset (default
//!    opt-out path). Confirms `pool_alloc_f32` doesn't error or panic on
//!    the new path.
//! 2. F32 allocations are zero-initialized on the opt-out path. Mirrors
//!    the `pool_disabled` path's "legacy callers expect zero-init"
//!    contract — required for Phase 3 (workaround removal) to be a
//!    no-observable-behavior-change ship.
//! 3. `pool_return_f32` accepts the slice without panic when the cache
//!    is opt-out (no free-list to push to; drop runs cudaFree directly).
//! 4. External-pointer paths still skip cudaFree (lifetime contract
//!    preserved across the opt-out branch).

use cudarc::driver::{CudaDevice, DevicePtr, DeviceSlice};
use flame_core::cuda_alloc_pool::{
    global_pool, pool_alloc_f32, pool_return_f32,
};
use std::sync::Arc;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME and LD_LIBRARY_PATH per project docs.",
    )
}

/// F32 alloc succeeds on the opt-out path (FLAME_F32_POOL_CACHE unset).
#[test]
fn pool_alloc_f32_succeeds_under_opt_out() {
    let dev = cuda_device();
    let s = pool_alloc_f32(&dev, 1024).expect("alloc 1024 f32");
    assert_eq!(DeviceSlice::len(&s), 1024);
    pool_return_f32(s);
}

/// F32 alloc returns zero-initialized memory on the opt-out path.
/// Mirrors the `pool_disabled` path's behavior. Phase 3 workaround
/// removal preserves this guarantee for callers that depend on it.
#[test]
fn pool_alloc_f32_zero_init_under_opt_out() {
    let dev = cuda_device();
    let n = 4096;
    let s = pool_alloc_f32(&dev, n).expect("alloc 4096 f32");
    let mut host = vec![1.0f32; n]; // poison: non-zero default to catch any leak
    dev.dtoh_sync_copy_into(&s, &mut host)
        .expect("dtoh copy back");
    let nonzero = host.iter().any(|&v| v != 0.0);
    assert!(
        !nonzero,
        "F32 opt-out path must return zero-init memory (mirrors pool_disabled). \
         First nonzero value index: {:?}",
        host.iter().position(|&v| v != 0.0),
    );
    pool_return_f32(s);
}

/// `pool_return_f32` accepts the slice cleanly under opt-out (no
/// free-list push, just drop). Run twice to catch any state leak.
#[test]
fn pool_return_f32_clean_under_opt_out() {
    let dev = cuda_device();
    for _ in 0..2 {
        let s = pool_alloc_f32(&dev, 2048).expect("alloc 2048 f32");
        pool_return_f32(s);
    }
}

/// Sanity: many sequential alloc+return cycles don't accumulate state.
/// If the F32 free-list were reachable on the opt-out path (it isn't),
/// after N cycles the pool's `current_cached_entries` would grow. Under
/// opt-out, `current_cached_entries` (F32 buckets) MUST stay at 0.
#[test]
fn pool_no_f32_cache_growth_under_opt_out() {
    let dev = cuda_device();
    let pool = global_pool();
    let cached_before = pool.stats().current_cached_entries;
    for _ in 0..32 {
        let s = pool_alloc_f32(&dev, 8192).expect("alloc 8192 f32");
        pool_return_f32(s);
    }
    let cached_after = pool.stats().current_cached_entries;
    // The opt-out path should not push F32 entries into the cache. The
    // BF16 path may have its own cached entries (e.g. from other tests
    // run in the same process) — what we care about is that this loop
    // didn't grow the cache.
    assert!(
        cached_after <= cached_before,
        "F32 opt-out path must not push to free-list: before={} after={}",
        cached_before,
        cached_after,
    );
}

/// External-pointer path still skips cudaFree on the opt-out branch.
/// Manually registers a ptr as external, then drops a slice with that
/// ptr through `pool_return_f32`. The pool must recognize it and
/// `reconstruct_and_forget` instead of cudaFree.
///
/// This test does NOT verify the cudaFree was actually skipped (the
/// CudaSlice's Drop is suppressed via mem::forget); it verifies the
/// external-ptr branch was taken by checking the external_ptrs
/// HashMap was decremented (via `unregister_external_ptr`).
#[test]
fn pool_return_f32_external_ptr_path_under_opt_out() {
    let dev = cuda_device();
    let pool = global_pool();
    // Allocate via the opt-out path (clean cudart alloc, no pool entry).
    let s = pool_alloc_f32(&dev, 512).expect("alloc 512 f32");
    let ptr = *DevicePtr::device_ptr(&s);
    // Manually register the ptr as external. (In production this is
    // done by ring_alloc::RingPoolAdapter; here we synthesize it.)
    pool.register_external_ptr(ptr);
    let refcount_before = pool.external_ptr_refcount(ptr);
    assert_eq!(
        refcount_before, 1,
        "after register, refcount should be 1"
    );
    // Return through the opt-out path. The external-ptr branch should
    // unregister and skip cudaFree.
    pool_return_f32(s);
    let refcount_after = pool.external_ptr_refcount(ptr);
    assert_eq!(
        refcount_after, 0,
        "after pool_return_f32 on opt-out path, external_ptrs should be \
         decremented to 0 (refcount_before={}, refcount_after={})",
        refcount_before, refcount_after,
    );
}
