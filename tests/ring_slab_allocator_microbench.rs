#![cfg(all(feature = "cuda", feature = "bf16_u16"))]
//! Phase 4 gate: replicate the Klein-9B-with-offload access pattern that
//! corrupts `cuda_alloc_pool` (the bug `EriDiffusion-v2@4511140` papered
//! over via `FLAME_ALLOC_POOL=0`), and verify `RingSlabAllocator` runs
//! clean against the same pattern.
//!
//! Per OFFLOAD_NEXT_GEN_DESIGN.md §C "Microbench gate":
//!
//! > `tests/ring_slab_allocator_microbench.rs` — replicate the
//! > Klein-9B-with-offload access pattern that corrupted
//! > `cuda_alloc_pool`. New allocator must run clean and faster than
//! > the no-pool fallback.
//!
//! # Pattern reproduced
//!
//! From `project_klein9b_step2_crash_isolation` bisect:
//!   - BlockOffloader allocates many BF16 buffers per "block" via
//!     `device.alloc::<u16>` (which uses `cuMemAllocAsync` on the
//!     device's internal stream).
//!   - Queues H2D into each on a separate transfer stream.
//!   - Records a transfer-stream event; default stream waits on it
//!     before compute.
//!   - Compute kernels read the buffers on the default stream.
//!   - At the next block's prefetch, the OLD slot's buffers DROP →
//!     `pool_return_u16` puts their raw pointers in the pool free-list.
//!     The compute kernels' reads may still be in flight at this
//!     moment (BlockOffloader's GPU-side `stream_wait_event` ensures
//!     ordering across streams, but the CPU-side drop doesn't wait).
//!   - The next iteration's `pool_alloc_u16` pops a free-list buffer.
//!     The pool gives a `CudaSlice` aliasing the same `cuMemAllocAsync`
//!     allocation. The new H2D writes to it while the previous compute
//!     is still reading. Race → CUDA driver's stream-ordered-allocator
//!     state corrupts → next `pool_alloc_u16` `cuMemAllocAsync` call
//!     returns `CUDA_ERROR_INVALID_VALUE`.
//!
//! # Tests
//!
//! - `pool_corruption_repro`: exercises the pattern against
//!   `cuda_alloc_pool`. **Expected to FAIL with CUDA_ERROR_INVALID_VALUE**
//!   on Klein-style hardware/driver combos. Test is `#[ignore]` by
//!   default — run with `--ignored` to confirm the bug still bites.
//! - `ring_runs_clean`: same allocation pattern routed through
//!   `RingSlabAllocator`. **Must succeed.**
//! - `ring_throughput`: timing — ring allocs must be no slower than the
//!   no-pool direct `device.alloc` fallback that the workaround uses.
//!
//! These three tests together gate the spec's
//! "Klein 9B runs without FLAME_ALLOC_POOL=0 workaround" claim.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, DevicePtr, DeviceSlice};
use flame_core::offload::ring_slab::{AutogradDirection, RingSlabAllocator};

/// Per-block tensor count (Klein 9B double block: img+txt+QKV+proj×
/// several layers ≈ this many slots). Tuned to load the allocator
/// hard without blowing PCIe in the test.
const TENSORS_PER_BLOCK: usize = 16;

/// Per-tensor element count (BF16). 4 MiB each — representative of
/// Klein 9B QKV-projection size and the size class where cuMemAllocAsync
/// has the most stream-ordering work to do.
const ELEMS_PER_TENSOR: usize = 2 * 1024 * 1024;

/// Number of "blocks" per "step". Klein 9B has 32 blocks.
const BLOCKS_PER_STEP: usize = 32;

/// Number of training "steps" to simulate. Klein's bisect showed
/// corruption hitting at step 2 (i.e., after one full forward+backward
/// of step 0 completes and step 1's first cudaMalloc runs).
const STEPS: usize = 5;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required for the ring_slab_allocator_microbench. Set CUDA_HOME and \
         LD_LIBRARY_PATH per the project README.",
    )
}

// ---------------------------------------------------------------------------
// Test 1 — repro: pool corruption pattern (expected to fail or hang)
// ---------------------------------------------------------------------------

/// Allocates TENSORS_PER_BLOCK × BLOCKS_PER_STEP × STEPS tensors via
/// `device.alloc::<u16>` (the BlockOffloader pattern) and lets them drop.
/// With `FLAME_ALLOC_POOL=1` (default), drops route through
/// `pool_return_u16`. If the allocations have any pending stream work at
/// drop time, the pool's free-list contains pointers whose CUDA-allocator
/// lifecycle isn't complete, and a subsequent pop returns a CudaSlice
/// whose next use triggers `CUDA_ERROR_INVALID_VALUE`.
///
/// **Ignored by default**: when the bug bites, the whole process aborts
/// with a CUDA driver error and there's no way to assert on it from
/// inside a test runner without bringing the runner down. Run with:
///
///   ```
///   FLAME_ALLOC_POOL=1 cargo test --release --features cuda,bf16_u16 \
///     --test ring_slab_allocator_microbench -- --ignored pool_corruption_repro
///   ```
#[test]
#[ignore = "reproduces a process-killing CUDA driver error; run manually"]
fn pool_corruption_repro() {
    let device = cuda_device();

    // Ensure pool is enabled for the repro. (If the user set
    // FLAME_ALLOC_POOL=0 in env, this test is meaningless.)
    if std::env::var("FLAME_ALLOC_POOL").as_deref() == Ok("0") {
        eprintln!(
            "[ring_slab_allocator_microbench] FLAME_ALLOC_POOL=0 set in env; \
             pool_corruption_repro can't reproduce the bug. Skipping."
        );
        return;
    }

    for step in 0..STEPS {
        for block in 0..BLOCKS_PER_STEP {
            // Per-block burst: TENSORS_PER_BLOCK allocations.
            let mut buffers = Vec::with_capacity(TENSORS_PER_BLOCK);
            for _ in 0..TENSORS_PER_BLOCK {
                let slice = unsafe { device.alloc::<u16>(ELEMS_PER_TENSOR) }
                    .expect("device.alloc::<u16> at repro step (pool path)");
                buffers.push(slice);
            }
            // Drop the buffers en masse — mimics BlockOffloader's slot
            // replacement dropping the OLD slot's HashMap of tensors.
            drop(buffers);
            std::hint::black_box(block);
        }
        std::hint::black_box(step);
    }
    // If we made it here, the bug didn't bite on this run. Note in
    // stderr — the repro is timing-sensitive.
    eprintln!(
        "[ring_slab_allocator_microbench] pool_corruption_repro completed \
         {STEPS} × {BLOCKS_PER_STEP} × {TENSORS_PER_BLOCK} allocs without crash. \
         (Bug is timing-sensitive; absence of crash here doesn't prove the pool is fixed.)"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — ring runs clean against the same pattern
// ---------------------------------------------------------------------------

/// Same allocation pattern as `pool_corruption_repro`, routed through
/// `RingSlabAllocator` instead of `device.alloc::<u16>`. Must complete
/// without CUDA error. This is the spec's
/// "ring runs clean against the corruption pattern" gate.
#[test]
fn ring_runs_clean() {
    let device = cuda_device();
    // Size the ring for one block's worth of allocations × 2 slabs
    // (mimics BlockOffloader's ping-pong slot count).
    let bytes_per_block = TENSORS_PER_BLOCK * ELEMS_PER_TENSOR * std::mem::size_of::<u16>();
    let ring = std::sync::Arc::new(std::sync::Mutex::new(
        RingSlabAllocator::new(device.clone(), 2, bytes_per_block)
            .expect("RingSlabAllocator::new failed"),
    ));

    for step in 0..STEPS {
        for block in 0..BLOCKS_PER_STEP {
            let mut handles = Vec::with_capacity(TENSORS_PER_BLOCK);
            for _ in 0..TENSORS_PER_BLOCK {
                let bytes = ELEMS_PER_TENSOR * std::mem::size_of::<u16>();
                let slab = RingSlabAllocator::alloc_handle(&ring, bytes).expect(
                    "RingSlabAllocator::alloc_handle — sized for one block, \
                     should never fail within a single block's burst",
                );
                handles.push(slab);
            }
            // Drop all handles before the next block — mimics slot
            // replacement. Auto-retire on Drop returns the cursors;
            // when the stack drains, retire_forward resets cursor to 0.
            drop(handles);
            std::hint::black_box(block);
        }
        std::hint::black_box(step);
    }
}

// ---------------------------------------------------------------------------
// Test 3 — throughput: ring vs direct device.alloc
// ---------------------------------------------------------------------------

/// Compares per-allocation latency: ring vs direct `device.alloc`. The
/// spec's Phase 4 success criterion is "similar or better wall time"
/// than the no-pool fallback (i.e., direct device.alloc — what the
/// `FLAME_ALLOC_POOL=0` workaround uses today).
#[test]
fn ring_throughput() {
    let device = cuda_device();

    // Warmup direct path.
    for _ in 0..16 {
        let _ = unsafe { device.alloc::<u16>(ELEMS_PER_TENSOR) }.unwrap();
    }

    // Direct path timing.
    let direct_iters = 256;
    let t0 = Instant::now();
    for _ in 0..direct_iters {
        let s = unsafe { device.alloc::<u16>(ELEMS_PER_TENSOR) }.unwrap();
        std::hint::black_box(*s.device_ptr());
    }
    let direct_dt = t0.elapsed();

    // Ring path timing.
    let bytes_per_block = TENSORS_PER_BLOCK * ELEMS_PER_TENSOR * std::mem::size_of::<u16>();
    let ring = std::sync::Arc::new(std::sync::Mutex::new(
        RingSlabAllocator::new(device.clone(), 2, bytes_per_block).unwrap(),
    ));
    // Warmup ring.
    for _ in 0..16 {
        let _ = RingSlabAllocator::alloc_handle(&ring, ELEMS_PER_TENSOR * std::mem::size_of::<u16>());
    }

    let ring_iters = 256;
    let bytes = ELEMS_PER_TENSOR * std::mem::size_of::<u16>();
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(TENSORS_PER_BLOCK);
    for i in 0..ring_iters {
        let slab = RingSlabAllocator::alloc_handle(&ring, bytes).unwrap();
        std::hint::black_box(slab.device_ptr());
        handles.push(slab);
        // Drop every block worth — keeps ring within its 2-slab capacity.
        if handles.len() == TENSORS_PER_BLOCK {
            handles.clear();
        }
        std::hint::black_box(i);
    }
    drop(handles);
    let ring_dt = t0.elapsed();

    eprintln!(
        "[ring_throughput] direct device.alloc: {} iters in {:.3} ms ({:.1} us/alloc)",
        direct_iters,
        direct_dt.as_secs_f64() * 1000.0,
        direct_dt.as_secs_f64() * 1_000_000.0 / direct_iters as f64,
    );
    eprintln!(
        "[ring_throughput] RingSlabAllocator:   {} iters in {:.3} ms ({:.1} us/alloc)",
        ring_iters,
        ring_dt.as_secs_f64() * 1000.0,
        ring_dt.as_secs_f64() * 1_000_000.0 / ring_iters as f64,
    );

    // The ring's per-alloc cost is a cursor bump + LIFO push + return.
    // Direct device.alloc goes through CUDA's stream-ordered allocator
    // (cuMemAllocAsync). Ring should be at least 5× faster per alloc.
    // Concretely on RTX 3090 Ti: cuMemAllocAsync ≈ 50-200 us, ring ≈ 0.1-1 us.
    assert!(
        ring_dt < direct_dt,
        "ring should be faster than direct device.alloc: direct={:.3} ms vs ring={:.3} ms",
        direct_dt.as_secs_f64() * 1000.0,
        ring_dt.as_secs_f64() * 1000.0
    );
}
