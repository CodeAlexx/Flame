//! Phase 1 microbench for `flame_core::ring_alloc::RingAllocator`.
//!
//! Strategy (per `docs/RING_ALLOC_DESIGN.md` §6): Tests 1-4 prove the ring's
//! invariants via end-to-end allocation sequences and pointer arithmetic
//! checks. Test 5 runs the Klein-pattern shape sequence (32 forward
//! 12 MB BF16 block I/O tensors, 32 backward of the same) and asserts the
//! ring runs clean with the bidirectional invariant holding throughout.
//!
//! Direct cross-comparison vs `cuda_alloc_pool`'s known corruption is
//! Phase 2 scope — it requires a `BlockOffloader` + autograd backward
//! harness, not standalone allocator calls. Phase 1 proves the ring's
//! structural guarantee: with the two-cursor design exercised under the
//! Klein shape sequence, overlap is impossible by construction.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::ring_alloc::{RingAllocator, RingPtr};

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export \
         LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

#[inline]
fn assert_aligned_16(p: &RingPtr) {
    assert_eq!(
        p.device_ptr % 16,
        0,
        "device_ptr 0x{:x} not 16-byte aligned (intra_offset={}, slab_idx={})",
        p.device_ptr,
        p.intra_offset,
        p.slab_idx,
    );
    assert_eq!(
        p.intra_offset % 16,
        0,
        "intra_offset {} not 16-byte aligned",
        p.intra_offset,
    );
}

// ---------------------------------------------------------------------------
// Test 1 — forward-only allocation sequence
// ---------------------------------------------------------------------------

#[test]
fn forward_only_sequence_advances_monotonically_and_aligns() {
    let device = cuda_device();
    // 4 slabs × 256 KiB = 1 MiB total.
    let slab_bytes = 256 * 1024;
    let num_slabs = 4;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes)
        .expect("RingAllocator::new");

    // Mix of sizes — none of which is a multiple of 16 — to stress
    // ceil_16 alignment.
    let sizes: &[usize] = &[
        1_000, 7_777, 31_111, 65_536, 100_000, 130_000,
        17, 333, 4_096, 8_193, 50_000,
    ];

    let mut last_global_end = 0_usize;
    let mut seen_slab: Option<usize> = None;
    let mut allocations = Vec::new();
    {
        let mut h = ring.forward_handle(0);
        for &n in sizes {
            let p = h.alloc(n).expect("forward alloc");
            assert_eq!(p.len_bytes, n, "len_bytes echoes request");
            assert_aligned_16(&p);

            let global = p.slab_idx * slab_bytes + p.intra_offset;

            // Within a slab, monotonically non-decreasing.
            match seen_slab {
                Some(prev) if prev == p.slab_idx => {
                    assert!(
                        global >= last_global_end,
                        "forward: alloc overlaps previous (global {global} < prev_end {last_global_end})"
                    );
                }
                _ => {
                    // Crossed slab boundary; new slab starts at intra=0.
                    assert_eq!(p.intra_offset, 0, "slab jump lands at offset 0");
                }
            }

            seen_slab = Some(p.slab_idx);
            last_global_end = global + n;
            allocations.push(p);
        }
    }

    // Cursors moved.
    assert!(ring.allocation_end() > 0, "forward cursor advanced");
    assert_eq!(ring.allocation_start(), ring.total_bytes(), "backward cursor untouched");

    // Lazy allocation: cuda_malloc_count equals slabs_allocated and
    // matches the count of distinct slabs we landed on.
    let distinct_slabs: usize = {
        let mut v: Vec<usize> = allocations.iter().map(|a| a.slab_idx).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    assert_eq!(ring.slabs_allocated(), distinct_slabs, "lazy alloc count");
    assert_eq!(ring.cuda_malloc_count(), distinct_slabs as u64, "cudaMalloc count matches slabs touched");
}

// ---------------------------------------------------------------------------
// Test 2 — backward-only allocation sequence
// ---------------------------------------------------------------------------

#[test]
fn backward_only_sequence_retreats_monotonically_and_aligns() {
    let device = cuda_device();
    let slab_bytes = 256 * 1024;
    let num_slabs = 4;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes)
        .expect("RingAllocator::new");

    let sizes: &[usize] = &[
        1_000, 7_777, 31_111, 65_536, 100_000, 17, 4_096, 8_193, 50_000,
    ];

    // Top of last slab is the implicit starting top — initial
    // allocation_start = total_bytes.
    let initial_start = ring.allocation_start();
    assert_eq!(initial_start, ring.total_bytes());

    let mut last_top: Option<(usize, usize)> = None; // (slab_idx, intra_top_after_alloc)
    let mut allocations = Vec::new();
    {
        let mut h = ring.backward_handle(0);
        for &n in sizes {
            let p = h.alloc(n).expect("backward alloc");
            assert_eq!(p.len_bytes, n);
            assert_aligned_16(&p);

            // Within a slab, intra_offset of successive allocations must
            // be monotonically non-increasing (we retreat).
            if let Some((prev_slab, prev_intra_top)) = last_top {
                if prev_slab == p.slab_idx {
                    assert!(
                        p.intra_offset + n <= prev_intra_top,
                        "backward: alloc overlaps previous in slab {} \
                         (new {}+{} > prev top {})",
                        p.slab_idx, p.intra_offset, n, prev_intra_top,
                    );
                } else {
                    // Crossed boundary to previous slab — new alloc's top
                    // must be at or below slab_bytes.
                    assert!(p.intra_offset + n <= slab_bytes);
                }
            }
            last_top = Some((p.slab_idx, p.intra_offset));
            allocations.push(p);
        }
    }

    assert!(ring.allocation_start() < initial_start, "backward cursor advanced");
    assert_eq!(ring.allocation_end(), 0, "forward cursor untouched");

    let distinct_slabs: usize = {
        let mut v: Vec<usize> = allocations.iter().map(|a| a.slab_idx).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    assert_eq!(ring.slabs_allocated(), distinct_slabs);
    assert_eq!(ring.cuda_malloc_count(), distinct_slabs as u64);
}

// ---------------------------------------------------------------------------
// Test 3 — bidirectional alternating, invariant holds until cursors meet
// ---------------------------------------------------------------------------

#[test]
fn bidirectional_alternating_preserves_invariant_and_errors_on_meet() {
    let device = cuda_device();
    // Smaller ring to make it easy to fill: 2 slabs × 64 KiB = 128 KiB.
    let slab_bytes = 64 * 1024;
    let num_slabs = 2;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes)
        .expect("RingAllocator::new");

    // Alternate forward(n) / backward(n) until one of them errors
    // (ring exhausted). At every step, allocation_end ≤ allocation_start.
    let n = 8_192;
    let mut alloc_count = 0;
    let mut last_err: Option<flame_core::Error> = None;

    for step in 0..200 {
        // Forward
        {
            let mut hf = ring.forward_handle(step);
            match hf.alloc(n) {
                Ok(p) => {
                    assert_aligned_16(&p);
                    alloc_count += 1;
                }
                Err(e) => {
                    last_err = Some(e);
                    break;
                }
            }
        }
        assert!(
            ring.allocation_end() <= ring.allocation_start(),
            "invariant broken after forward: end={} > start={}",
            ring.allocation_end(),
            ring.allocation_start(),
        );

        // Backward
        {
            let mut hb = ring.backward_handle(step);
            match hb.alloc(n) {
                Ok(p) => {
                    assert_aligned_16(&p);
                    alloc_count += 1;
                }
                Err(e) => {
                    last_err = Some(e);
                    break;
                }
            }
        }
        assert!(
            ring.allocation_end() <= ring.allocation_start(),
            "invariant broken after backward: end={} > start={}",
            ring.allocation_end(),
            ring.allocation_start(),
        );
    }

    // We must have done at least a few allocations.
    assert!(
        alloc_count >= 4,
        "expected several successful allocations before exhaustion, got {alloc_count}"
    );
    // And we must have eventually exhausted (cursors met).
    let err = last_err.expect("ring should eventually exhaust");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("exhausted") || msg.contains("cross"),
        "expected exhaustion error, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Test 4 — slab-boundary stress: every alloc forces a jump
// ---------------------------------------------------------------------------

#[test]
fn slab_boundary_stress_advances_slab_idx() {
    let device = cuda_device();
    // Slab of 64 KiB; allocate exactly 40 KiB each — second alloc forces
    // a jump (40 + 40 > 64) and so on.
    let slab_bytes = 64 * 1024;
    let num_slabs = 5;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes)
        .expect("RingAllocator::new");

    let alloc_size = 40 * 1024;

    let mut prev_slab: Option<usize> = None;
    {
        let mut h = ring.forward_handle(0);
        for i in 0..num_slabs {
            let p = h.alloc(alloc_size).expect("forward alloc");
            assert_aligned_16(&p);
            // Each allocation lands on a fresh slab — start of new slab.
            assert_eq!(p.intra_offset, 0, "alloc {i} should be at slab start");
            if let Some(prev) = prev_slab {
                assert_eq!(
                    p.slab_idx,
                    prev + 1,
                    "slab index should increment on boundary jump"
                );
            }
            prev_slab = Some(p.slab_idx);
        }
    }

    // Now all 5 slabs are touched and cudaMalloc'd.
    assert_eq!(ring.slabs_allocated(), num_slabs);
    assert_eq!(ring.cuda_malloc_count(), num_slabs as u64);

    // 6th forward alloc must error: backward hasn't fired so
    // allocation_start is at total_bytes — but the wrap to slab 0 would
    // attempt to overlap with the already-allocated region [0, used].
    // Phase 1 spec: linear-regime forward overlap with backward errors;
    // wrap-and-reuse without a reset is the ring-exhausted case.
    //
    // Practically: forward_end is now 4*slab_bytes + alloc_size. A 6th
    // alloc bumps to slab 5 (== num_slabs), which wraps to slab 0,
    // offset 0. Backward cursor is at total_bytes (untouched), so the
    // forward-overlap check passes (allocation_start == total_bytes
    // → check skipped). The wrap silently happens.
    //
    // Therefore the 6th alloc SUCCEEDS but lands back at slab 0,
    // intra=0 — reusing memory from the first alloc. That's the
    // documented Phase 1 wrap-when-only-forward-active behavior
    // (design doc §3 "Cyclic wrap" + §8 q1). The microbench just
    // verifies the wrap is observable; Phase 2 wires a reset boundary
    // before this can bite.
    let result = {
        let mut h = ring.forward_handle(0);
        h.alloc(alloc_size)
    };
    let wrapped = result.expect("wrap-around forward alloc should succeed");
    assert_eq!(wrapped.slab_idx, 0, "wrap returned to slab 0");
    assert_eq!(wrapped.intra_offset, 0);
    // No new cudaMalloc (slab 0 already materialized).
    assert_eq!(ring.cuda_malloc_count(), num_slabs as u64);
}

// ---------------------------------------------------------------------------
// Test 5 — Klein-pattern repro: 32 forward + 32 backward of 12 MB BF16
// ---------------------------------------------------------------------------
//
// Klein 9B block I/O activation shape: `[1, 1520, 4096]` BF16
//   = 1 * 1520 * 4096 * 2 = 12_451_840 bytes per tensor.
//
// Sizing: 4 blocks per slab → slab_bytes ≈ 4 * 12_451_840 = 49_807_360
// (we round up). 8 slabs → 8 * 4 = 32 blocks fwd, fitting backwards into
// the same byte space from the other end after a reset between fwd and
// bwd phase IS NOT what we want here — we want fwd and bwd to coexist
// in one step. So size for 64 simultaneous live blocks: 16 slabs × 4
// blocks = 64 slots. We allocate fwd first, then bwd, on the same ring
// without resetting. Bidirectional invariant must hold throughout.

#[test]
fn klein_pattern_32_block_fwd_bwd_bidirectional_invariant() {
    let device = cuda_device();

    let bytes_per_tensor: usize = 1 * 1520 * 4096 * 2; // BF16
    assert_eq!(bytes_per_tensor, 12_451_840);

    // 4 tensors per slab → slab_bytes = ceil_16(4 * 12_451_840) =
    // 49_807_360 (already 16-aligned). 16 slabs total → 64 tensor
    // capacity, which we split 32 fwd + 32 bwd. Total ring footprint
    // = 16 * 49_807_360 ≈ 760 MiB.
    let slab_bytes = 4 * bytes_per_tensor;
    let num_slabs = 16;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes)
        .expect("RingAllocator::new");

    let total_mb = (ring.total_bytes() as f64) / (1024.0 * 1024.0);
    println!(
        "klein_pattern: ring footprint = {} slabs × {} MiB = {:.1} MiB total",
        num_slabs,
        slab_bytes / (1024 * 1024),
        total_mb,
    );

    // Phase A: 32 forward allocations.
    let mut fwd_ptrs = Vec::with_capacity(32);
    {
        let mut h = ring.forward_handle(0);
        for block in 0..32_usize {
            let p = h.alloc(bytes_per_tensor)
                .unwrap_or_else(|e| panic!("fwd alloc for block {block} failed: {e:?}"));
            assert_aligned_16(&p);
            fwd_ptrs.push(p);
        }
    }

    // Invariant after fwd: cursors must still satisfy end ≤ start.
    let end_after_fwd = ring.allocation_end();
    let start_after_fwd = ring.allocation_start();
    assert!(
        end_after_fwd <= start_after_fwd,
        "invariant broken after fwd: end={end_after_fwd} > start={start_after_fwd}",
    );
    println!(
        "klein_pattern: after 32 fwd allocs — end={} MiB, start={} MiB, slabs={}/{}",
        end_after_fwd / (1024 * 1024),
        start_after_fwd / (1024 * 1024),
        ring.slabs_allocated(),
        num_slabs,
    );

    // Phase B: 32 backward allocations interleaved-with-nothing
    // (real trainer interleaves with checkpoint replay, but the allocator
    // pattern is what we are testing).
    let mut bwd_ptrs = Vec::with_capacity(32);
    {
        let mut h = ring.backward_handle(0);
        for block in 0..32_usize {
            let p = h.alloc(bytes_per_tensor)
                .unwrap_or_else(|e| panic!("bwd alloc for block {block} failed: {e:?}"));
            assert_aligned_16(&p);
            bwd_ptrs.push(p);

            // Spot-check the invariant after every backward alloc, via
            // the handle's read-through accessors (the allocator itself
            // is mutably borrowed by `h`).
            assert!(
                h.allocation_end() <= h.allocation_start(),
                "invariant broken after bwd block {block}: end={} > start={}",
                h.allocation_end(),
                h.allocation_start(),
            );
        }
    }

    let end_after_bwd = ring.allocation_end();
    let start_after_bwd = ring.allocation_start();
    println!(
        "klein_pattern: after 32 bwd allocs — end={} MiB, start={} MiB",
        end_after_bwd / (1024 * 1024),
        start_after_bwd / (1024 * 1024),
    );

    // Disjoint ranges check: every fwd ptr's [intra, intra+len) lies in
    // [0, end_after_fwd) — call it the "low half" — and every bwd ptr's
    // [intra, intra+len) lies in [start_after_bwd, total_bytes) — the
    // "high half". The two halves don't overlap.
    for fp in &fwd_ptrs {
        let global = fp.slab_idx * slab_bytes + fp.intra_offset;
        assert!(
            global + fp.len_bytes <= end_after_fwd,
            "fwd ptr beyond forward cursor: global+len={} > end_after_fwd={}",
            global + fp.len_bytes, end_after_fwd,
        );
    }
    for bp in &bwd_ptrs {
        let global = bp.slab_idx * slab_bytes + bp.intra_offset;
        assert!(
            global >= start_after_bwd,
            "bwd ptr below backward cursor: global={} < start_after_bwd={}",
            global, start_after_bwd,
        );
    }

    // Lazy allocation: exactly the slabs touched were cudaMalloc'd.
    // For 32 + 32 = 64 tensors at 4 per slab, expect 16 slabs touched.
    assert_eq!(ring.slabs_allocated(), 16, "should touch all 16 slabs");
    assert_eq!(
        ring.cuda_malloc_count(),
        16,
        "cudaMalloc count equals slab count, not allocation count"
    );

    // Reset between steps — slabs stay mapped.
    ring.reset();
    assert_eq!(ring.allocation_end(), 0);
    assert_eq!(ring.allocation_start(), ring.total_bytes());
    assert_eq!(
        ring.cuda_malloc_count(),
        16,
        "reset() does not allocate or free"
    );

    // A second step should reuse the same 16 slabs.
    {
        let mut h = ring.forward_handle(0);
        let _p = h.alloc(bytes_per_tensor).expect("step-2 fwd alloc");
    }
    assert_eq!(
        ring.cuda_malloc_count(),
        16,
        "step 2 forward reuses slabs — no new cudaMalloc"
    );

    println!(
        "klein_pattern: PASS — bidirectional invariant held, \
         {} cudaMalloc total, {} slabs allocated, peak {} MiB",
        ring.cuda_malloc_count(),
        ring.slabs_allocated(),
        (ring.slabs_allocated() * ring.slab_bytes()) / (1024 * 1024),
    );
}

// ---------------------------------------------------------------------------
// Negative cases — alloc rejects zero and oversized requests
// ---------------------------------------------------------------------------

#[test]
fn alloc_rejects_zero_bytes() {
    let device = cuda_device();
    let mut ring = RingAllocator::new(device, 2, 1024).expect("new");
    {
        let mut h = ring.forward_handle(0);
        assert!(h.alloc(0).is_err(), "forward zero-byte alloc must error");
    }
    {
        let mut h = ring.backward_handle(0);
        assert!(h.alloc(0).is_err(), "backward zero-byte alloc must error");
    }
}

#[test]
fn alloc_rejects_oversized() {
    let device = cuda_device();
    let mut ring = RingAllocator::new(device, 2, 1024).expect("new");
    {
        let mut h = ring.forward_handle(0);
        assert!(
            h.alloc(2048).is_err(),
            "alloc > slab_bytes must error (single alloc cannot span slabs)"
        );
    }
}

#[test]
fn new_rejects_zero_slabs() {
    let device = cuda_device();
    assert!(RingAllocator::new(device, 0, 1024).is_err());
}

#[test]
fn new_rejects_zero_slab_bytes() {
    let device = cuda_device();
    assert!(RingAllocator::new(device, 2, 0).is_err());
}
