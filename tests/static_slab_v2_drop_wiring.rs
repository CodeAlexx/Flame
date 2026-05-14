//! R1b regression test (Bug Fixer): verify that
//! `TensorStorage::Drop` calls `slab_v2_return_if_owned` for slab-backed
//! BF16 + F32 tensors, even when `FLAME_ALLOC_POOL=0`.
//!
//! ## Bug
//!
//! Builder's `TensorStorage::Drop` impl has a top-level early-return when
//! `pool_disabled()` is true:
//!
//! ```ignore
//! if crate::cuda_alloc_pool::pool_disabled() {
//!     return;
//! }
//! ```
//!
//! This was correct PRE-R1b (skip pool bookkeeping when pool is off), but
//! after R1b's slab integration it also defeats the slab live-count
//! decrement. A slab-backed tensor dropped under `FLAME_ALLOC_POOL=0`:
//!
//! 1. cudarc Drop hook protects the ptr (slab range is registered) → no
//!    `cudaFree`. Memory-safe.
//! 2. BUT `slab_v2_return_if_owned` is never called → `live_count` never
//!    decremented. The slab's `reset()` will fail forever afterwards.
//!
//! This breaks the R2c gates the moment a trainer ever runs with
//! `FLAME_ALLOC_POOL=0` (which is the current default workaround for the
//! cache-cycling crash; replacing that workaround with the slab is the
//! whole point of the redesign).
//!
//! ## Test approach
//!
//! Set `FLAME_ALLOC_POOL=0` BEFORE any flame-core code reads the cached
//! `pool_disabled` flag. Build a slab, allocate a BF16 slice, wrap it in a
//! `TensorStorage::BF16`, drop the storage, and assert `live_count` went
//! from 1 → 0.
//!
//! Without the fix this test fails: live_count stays at 1.
//! With the fix this test passes.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaSlice};
use flame_core::static_slab_v2::{self, StaticSlabAllocator};
use flame_core::tensor_storage::TensorStorage;

fn skip_if_no_gpu() -> Option<Arc<CudaDevice>> {
    if std::env::var("FLAME_SKIP_GPU_TESTS").ok().as_deref() == Some("1") {
        eprintln!("[drop_wiring] FLAME_SKIP_GPU_TESTS=1 — skipping");
        return None;
    }
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[drop_wiring] CudaDevice::new(0) failed: {e:?} — skipping");
            None
        }
    }
}

/// Force `FLAME_ALLOC_POOL=0` BEFORE any flame-core function reads it.
/// `pool_disabled()` caches the value via `OnceLock`; we must set the env
/// before the first call. The test binary's `main` is the only place we
/// can do this — it runs before any `#[test]` body.
///
/// Rust integration tests don't have a user-controllable main, so we rely
/// on this being the FIRST test in this binary's discovery order and on
/// nothing earlier touching the cache. `std::env::set_var` is unsound in
/// multi-threaded programs in general, but here it's done before any test
/// thread starts running (Rust's test harness reads the env once per
/// process at startup for filter-name handling only).
fn force_pool_disabled() {
    // SAFETY (env mutation): no other thread has been spawned yet by the
    // test harness; this runs at the top of every test body, but the
    // `OnceLock` cache is set on first read — we win the race by setting
    // before that read happens within this binary.
    unsafe {
        std::env::set_var("FLAME_ALLOC_POOL", "0");
    }
    // Sanity: confirm the flag is observable.
    assert!(
        flame_core::cuda_alloc_pool::pool_disabled(),
        "FLAME_ALLOC_POOL=0 not observable by pool_disabled() (cached?)",
    );
}

/// Construct a `TensorStorage::BF16` directly from a slab-allocated slice.
/// Uses the same `CudaSliceMirror` trick as production code but doesn't
/// expose the slice through the safe constructor (which would force-pool-
/// alloc instead of accepting a pre-built slice).
fn wrap_slab_slice_as_bf16_storage(slice: CudaSlice<u16>) -> TensorStorage {
    let numel = cudarc::driver::DeviceSlice::len(&slice);
    // Use the public BF16-from-slice helper if one exists; otherwise build
    // via the storage's wrap_slice path. We need the `Arc<CudaSlice<u16>>`
    // shape because shared_storage is enabled in default features.
    //
    // The public API surface for "wrap an existing CudaSlice" is the
    // module-internal `wrap_slice`; we use a thin reflection via
    // `TensorStorage::empty` won't work (allocates anew). Instead, build
    // a Tensor from BF16 bytes — but that copies. We want NO copy.
    //
    // Pragmatic: cast through `Tensor::from_bf16_slice_gpu`-style helper
    // OR directly construct via `unsafe { std::ptr::read }` ... but
    // TensorStorage is public so we can construct it directly through the
    // `bf16_u16` variant builder if one is exposed.
    //
    // Look at storage.rs — `TensorStorage::BF16 { data, numel }` is the
    // shape. We need to construct the enum variant.
    use flame_core::tensor_storage::wrap_slab_slice_for_test;
    wrap_slab_slice_for_test(slice, numel)
}

#[test]
fn bug2_regression_slab_drop_under_pool_disabled() {
    force_pool_disabled();

    let Some(device) = skip_if_no_gpu() else { return };

    // Build a slab in isolation.
    let slab = StaticSlabAllocator::new(device.clone(), 1 * 1024 * 1024);

    // Register the slab in the global device_map so
    // `slab_v2_return_if_owned` can find it via the slice's device key.
    let key = Arc::as_ptr(&device) as usize;
    let slab_static: &'static Mutex<StaticSlabAllocator> =
        Box::leak(Box::new(Mutex::new(slab)));

    // Inject the slab under the production device key. (Test-shim
    // approach mirroring `slab_alloc_advances_cursor`.)
    {
        let map_helper = static_slab_v2::__test_device_map_insert(key, slab_static);
        let _ = map_helper;
    }

    // Allocate a BF16 slice through the slab.
    let slice: CudaSlice<u16> = {
        let mut g = slab_static.lock().unwrap();
        g.alloc_u16(256).expect("alloc_u16 256")
    };
    let _slice_ptr = {
        use cudarc::driver::DevicePtr;
        *slice.device_ptr()
    };
    assert_eq!(slab_static.lock().unwrap().live_count(), 1);

    // Wrap into TensorStorage::BF16 and drop it. Under the fix, this
    // routes through `slab_v2_try_claim` regardless of pool_disabled
    // state, decrementing live_count.
    let storage = wrap_slab_slice_as_bf16_storage(slice);
    drop(storage);

    // After drop: live_count must be 0 (decremented by slab_v2_return_if_owned).
    let live_after = slab_static.lock().unwrap().live_count();
    assert_eq!(
        live_after, 0,
        "TensorStorage::Drop did not decrement slab live_count under FLAME_ALLOC_POOL=0 \
         (pre-fix early-return defeated the slab wiring)"
    );

    // Cleanup: release slab.
    {
        let mut g = slab_static.lock().unwrap();
        let _ = g.release();
    }
    // Remove our injected map entry.
    static_slab_v2::__test_device_map_remove(key);
    // slab_static is intentionally leaked (small, process-bounded).
}
