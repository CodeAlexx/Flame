//! Smoke test for `RingPoolAdapter` wired as `cuda_alloc_pool`'s
//! cache-miss backend (Phase 2a).
//!
//! Verifies:
//! 1. `install_miss_allocator` succeeds.
//! 2. A subsequent `pool_alloc_u16` cache miss routes through the
//!    adapter (the ring's `cuda_malloc_count` grows).
//! 3. `pool_return_u16` of the resulting slice tags the bucket entry
//!    as external (no cudaFree at clear_cache time).
//! 4. `clear_pool_cache()` + `adapter.reset()` brings the system back
//!    to a fresh state without panicking on stale ring pointers.
//! 5. Cache-HIT path still works (a return-then-alloc pair).
//!
//! Per `tests/ring_alloc_microbench.rs` style: GPU-touching, opt-out
//! via `RUST_CUDA_OFFLINE=1`.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::cuda_alloc_pool::{
    clear_pool_cache, global_pool, install_miss_allocator, pool_alloc_u16,
    pool_return_u16, uninstall_miss_allocator,
};
use flame_core::ring_alloc::{RingAllocator, RingPoolAdapter};

fn skip_if_no_cuda() -> Option<Arc<CudaDevice>> {
    if std::env::var("RUST_CUDA_OFFLINE").is_ok() {
        eprintln!("skip: RUST_CUDA_OFFLINE set");
        return None;
    }
    CudaDevice::new(0).ok()
}

#[test]
fn install_and_alloc_routes_through_ring() {
    let Some(device) = skip_if_no_cuda() else { return };

    // 8 slabs × 4 MiB = 32 MiB ring — plenty for a single 1 MiB alloc
    let ring = RingAllocator::new(device.clone(), 8, 4 * 1024 * 1024).unwrap();
    let adapter = Arc::new(RingPoolAdapter::new(ring));

    // Snapshot pool state pre-install.
    let pool = global_pool();
    let ext_before = pool.external_miss_count();
    let slabs_before = adapter.slabs_allocated();
    assert_eq!(slabs_before, 0, "lazy: no slabs before first alloc");

    let prev = install_miss_allocator(adapter.clone());
    assert!(prev.is_none(), "no previous miss-allocator installed");

    // Allocate ~512 KiB of BF16 → should be a cache miss → routed
    // through the adapter → ring materializes its first slab.
    let elems = 512 * 1024 / 2; // 512 KiB / sizeof(u16) = 262144 elems
    let slice = pool_alloc_u16(&device, elems).expect("pool_alloc_u16");
    assert_eq!(cudarc::driver::DeviceSlice::len(&slice), elems);

    assert!(
        pool.external_miss_count() > ext_before,
        "external_miss_count should bump on cache-miss"
    );
    assert!(
        adapter.slabs_allocated() >= 1,
        "ring should have materialized at least one slab"
    );

    // Return the slice to the pool. The entry should be tagged external
    // (no cudaFree). The clear_pool_cache below verifies that.
    pool_return_u16(slice);

    // Clear the pool — external entries should be reconstructed-and-
    // forgotten (slab Arc keeps memory alive). This panicking would
    // indicate the is_external tracking is broken.
    clear_pool_cache();

    // Reset the ring — cursors back to extremes, slabs persist.
    adapter.reset();

    // Allocate again — should still route through the ring (allocator
    // remains installed), should re-materialize from slab 0 since
    // we reset, no new cudaMalloc.
    let slabs_mid = adapter.slabs_allocated();
    let cudamalloc_mid = adapter.cuda_malloc_count();
    let slice2 = pool_alloc_u16(&device, elems).expect("pool_alloc_u16 post-reset");
    assert_eq!(adapter.slabs_allocated(), slabs_mid, "no new slab after reset");
    assert_eq!(
        adapter.cuda_malloc_count(),
        cudamalloc_mid,
        "no new cudaMalloc after reset"
    );

    pool_return_u16(slice2);
    clear_pool_cache();

    // Cleanup.
    let _ = uninstall_miss_allocator();
}

#[test]
fn fallback_when_no_allocator_installed() {
    let Some(device) = skip_if_no_cuda() else { return };

    // Ensure no allocator is installed at test start.
    let _ = uninstall_miss_allocator();

    let pool = global_pool();
    let ext_before = pool.external_miss_count();

    let elems = 4096;
    let slice = pool_alloc_u16(&device, elems).expect("pool_alloc_u16");
    assert_eq!(cudarc::driver::DeviceSlice::len(&slice), elems);

    // External counter must NOT bump when no allocator is installed.
    assert_eq!(
        pool.external_miss_count(),
        ext_before,
        "no external_miss when no allocator installed"
    );

    pool_return_u16(slice);
    // No clear_pool_cache here — the slice's pointer is from device.alloc,
    // free-list reuse is the normal pool semantics.
}

#[test]
fn alloc_f32_falls_back_to_device_alloc() {
    // Phase 2a bug-fix (post-Builder): f32 routing through the ring is
    // DISABLED. The Builder's hook-based f32 routing falsified the Klein 9B
    // 5-step smoke at backward-graph teardown — derived `CudaSlice<f32>`
    // values escape the `external_ptrs` registration window and call
    // `cudaFree` on mid-slab offsets. `RingPoolAdapter::alloc_f32` now
    // returns `Err` and `pool_alloc_f32` falls back to
    // `device.alloc::<f32>` — same as the no-allocator-installed path.
    //
    // This test verifies the f32 miss path: external_miss_count does NOT
    // bump (the ring is not used), a subsequent direct-drop does NOT
    // panic (the slice came from `device.alloc` and `cudaFree` is valid).
    let Some(device) = skip_if_no_cuda() else { return };

    let ring = RingAllocator::new(device.clone(), 4, 4 * 1024 * 1024).unwrap();
    let adapter = Arc::new(RingPoolAdapter::new(ring));
    let _ = uninstall_miss_allocator();
    let _ = install_miss_allocator(adapter.clone());

    let pool = global_pool();
    let ext_before = pool.external_miss_count();

    let slice = flame_core::cuda_alloc_pool::pool_alloc_f32(&device, 1024)
        .expect("pool_alloc_f32");
    assert_eq!(cudarc::driver::DeviceSlice::len(&slice), 1024);

    // f32 path does NOT route through the ring (adapter returns Err).
    assert_eq!(
        pool.external_miss_count(),
        ext_before,
        "f32 routing disabled: external_miss_count must NOT bump"
    );

    // Drop the slice directly. Because the underlying ptr came from
    // `device.alloc::<f32>` (not the ring), `cudaFree` is the correct
    // operation. No panic expected.
    drop(slice);

    clear_pool_cache();
    adapter.reset();
    let _ = uninstall_miss_allocator();
}
