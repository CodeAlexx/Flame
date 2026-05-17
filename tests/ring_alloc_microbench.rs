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
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("RingAllocator::new");

    // Mix of sizes — none of which is a multiple of 16 — to stress
    // ceil_16 alignment.
    let sizes: &[usize] = &[
        1_000, 7_777, 31_111, 65_536, 100_000, 130_000, 17, 333, 4_096, 8_193, 50_000,
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
    assert_eq!(
        ring.allocation_start(),
        ring.total_bytes(),
        "backward cursor untouched"
    );

    // Lazy allocation: cuda_malloc_count equals slabs_allocated and
    // matches the count of distinct slabs we landed on.
    let distinct_slabs: usize = {
        let mut v: Vec<usize> = allocations.iter().map(|a| a.slab_idx).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    assert_eq!(ring.slabs_allocated(), distinct_slabs, "lazy alloc count");
    assert_eq!(
        ring.cuda_malloc_count(),
        distinct_slabs as u64,
        "cudaMalloc count matches slabs touched"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — backward-only allocation sequence
// ---------------------------------------------------------------------------

#[test]
fn backward_only_sequence_retreats_monotonically_and_aligns() {
    let device = cuda_device();
    let slab_bytes = 256 * 1024;
    let num_slabs = 4;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("RingAllocator::new");

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
                        p.slab_idx,
                        p.intra_offset,
                        n,
                        prev_intra_top,
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

    assert!(
        ring.allocation_start() < initial_start,
        "backward cursor advanced"
    );
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
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("RingAllocator::new");

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
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("RingAllocator::new");

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
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("RingAllocator::new");

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
            let p = h
                .alloc(bytes_per_tensor)
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
            let p = h
                .alloc(bytes_per_tensor)
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
            global + fp.len_bytes,
            end_after_fwd,
        );
    }
    for bp in &bwd_ptrs {
        let global = bp.slab_idx * slab_bytes + bp.intra_offset;
        assert!(
            global >= start_after_bwd,
            "bwd ptr below backward cursor: global={} < start_after_bwd={}",
            global,
            start_after_bwd,
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

// ---------------------------------------------------------------------------
// Bug Fixer Phase 1 — additional coverage tests
// ---------------------------------------------------------------------------
//
// These tests target gaps and edge cases in the Phase 1 ring allocator.
// Each test was written against commit 2927038 and either reproduces a
// genuine bug (failing) or verifies behavior the existing 5 tests do not
// cover (passing).

/// Bug: forward wrap silently lands BELOW backward and overwrites live
/// forward allocations when backward has fired but `allocation_start` is
/// at the top half of the ring.
///
/// Setup: 4 slabs × 1024 bytes. backward issues a small alloc (sits high).
/// Forward fills nearly to slab 3 end, then a final forward alloc that
/// would jump to slab 4 wraps to slab 0 — overwriting the still-live
/// forward allocation at slab 0.
///
/// The invariant check `new_end > allocation_start` (where new_end is now
/// small, post-wrap, and allocation_start is in the high half) silently
/// passes, causing forward to "lap" itself.
#[test]
fn forward_wrap_with_backward_active_does_not_lap_silently() {
    // Construct exact scenario where forward cursor sits in the LAST slab
    // with some intra offset, and the next alloc forces a slab-jump past
    // the last slab into a wrap. Meanwhile backward sits in the middle of
    // the same last slab. After wrap, the new alloc lands at (slab 0,
    // intra 0) — silently overlapping a live forward alloc from earlier.
    //
    // The fwd check `new_end > allocation_start` trivially passes because
    // new_end is now small (post-wrap, back near 0) and allocation_start
    // is in the last slab.
    //
    // Expected (correct) behavior: the wrap-with-backward-active scenario
    // must error because forward is about to overwrite its own live
    // allocations. Per design doc §4 invariant 2:
    //   "After a wrap, the ring monitors allocation_end <= allocation_start
    //    (mod total_bytes); violations error rather than silently overlap."
    //
    // 2 slabs × 1024 bytes = 2048 total.
    let device = cuda_device();
    let slab_bytes = 1024;
    let num_slabs = 2;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("new");

    // Backward alloc 100 → cur=(_,_), special-case to (1, 1024). 1024 >=
    // 100 no jump. new_intra = floor_16(1024-100)=912. allocation_start =
    // 1*1024 + 912 = 1936.
    {
        let mut hb = ring.backward_handle(0);
        let _ = hb.alloc(100).expect("backward initial");
    }
    assert_eq!(ring.allocation_start(), 1936);

    // Forward fills slab 0 with two allocations leaving cursor in slab 1
    // at intra 528 (so next alloc forces a jump):
    let p0 = {
        let mut hf = ring.forward_handle(0);
        // alloc 500 → (0, 0), end=500
        let p0 = hf.alloc(500).expect("a");
        assert_eq!((p0.slab_idx, p0.intra_offset), (0, 0));
        // alloc 500 → cur=(0, ceil_16(500)=512). 512+500=1012<=1024. (0,512). end=1012.
        let _p1 = hf.alloc(500).expect("b");
        // alloc 500 → cur=(0, ceil_16(1012)=1024). 1024+500>1024 → jump (1,0).
        //   1>=2? No. new_end = 1024+0+500=1524. check 1524>1936? No. OK. (1,0).
        let _p2 = hf.alloc(500).expect("c");
        p0
    };
    assert_eq!(ring.allocation_end(), 1524);
    // Now allocation_end=1524, allocation_start=1936.

    // The danger alloc: size 700.
    // cur=(1, ceil_16(500)=512). 512+700=1212>1024 → jump (2,0).
    // 2>=2 → WRAP → (0,0). new_end = 0+0+700=700.
    // BUG: check 700 > 1936? NO. Silent pass.
    // Returned ptr: (slab 0, intra 0). Overlaps p0 ([0, 500)) byte-for-byte!
    let lapping_result = {
        let mut hf = ring.forward_handle(1);
        hf.alloc(700)
    };
    assert!(
        lapping_result.is_err(),
        "forward wrap-with-backward-active that would silently overlap \
         a live forward allocation must error. Got Ok({:?}); p0 was at \
         (slab {}, intra {}, len {}); cursor before was end=1524, start=1936.",
        lapping_result
            .as_ref()
            .map(|p| (p.slab_idx, p.intra_offset, p.len_bytes))
            .ok(),
        p0.slab_idx,
        p0.intra_offset,
        p0.len_bytes,
    );
}

/// Bug (mirror): backward wrap silently lands in the LAST slab and
/// overwrites a live backward allocation made earlier.
///
/// Symmetric to `forward_wrap_with_backward_active_does_not_lap_silently`.
/// Construct a state where backward's cursor sits in slab 0 with some
/// intra offset and a subsequent backward alloc requires jumping to "slab
/// -1" → wrap to last slab. The wrap puts backward at the TOP of the
/// last slab, overwriting an earlier backward alloc there.
///
/// The `new_start < allocation_end` check trivially passes because
/// allocation_end is small (forward only made a tiny alloc) while
/// new_start is now in the last slab.
#[test]
fn backward_wrap_with_forward_active_does_not_lap_silently() {
    let device = cuda_device();
    let slab_bytes = 1024;
    let num_slabs = 2;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("new");

    // Forward alloc 100 → (0, 0), end=100. allocation_end=100.
    {
        let mut hf = ring.forward_handle(0);
        let _ = hf.alloc(100).expect("forward initial");
    }
    assert_eq!(ring.allocation_end(), 100);

    // Backward fills last slab + most of slab 0 from the high side:
    let p_n = {
        let mut hb = ring.backward_handle(0);
        // 500: special-case (1, 1024). 1024>=500. new_intra=floor_16(524)=512. global=1*1024+512=1536.
        let p_n = hb.alloc(500).expect("bwd 1");
        assert_eq!((p_n.slab_idx, p_n.intra_offset), (1, 512));
        // 500: cur=(1, 512). 512>=500 no jump. new_intra=floor_16(12)=0. global=1024. ✓ start=1024.
        let _p1 = hb.alloc(500).expect("bwd 2");
        // 500: cur=(1, 0). 0<500 → jump (0, 1024). 0 not<0 → no wrap. new_intra=floor_16(524)=512. global=512.
        // check: 512 < 100? NO → OK. start=512.
        let _p2 = hb.alloc(500).expect("bwd 3");
        p_n
    };
    assert_eq!(ring.allocation_start(), 512);

    // The danger alloc: size 700. cur=(0, 512). 512<700 → jump prev: (-1, 1024).
    // -1 < 0 → WRAP: (last slab=1, intra_top=slab_bytes=1024).
    // new_intra = floor_16(1024-700) = 320. new_global_start = 1*1024+320 = 1344.
    // BUG: check 1344 < 100? NO → silent pass. But (slab 1, intra 320, len 700)
    // overlaps pN at (slab 1, intra 512, len 500) — they share [512, 1020).
    let lapping_result = {
        let mut hb = ring.backward_handle(1);
        hb.alloc(700)
    };
    assert!(
        lapping_result.is_err(),
        "backward wrap-with-forward-active that would silently overlap \
         a live backward allocation must error. Got Ok({:?}); p_n was at \
         (slab {}, intra {}, len {}); cursor before was start=512, end=100.",
        lapping_result
            .as_ref()
            .map(|p| (p.slab_idx, p.intra_offset, p.len_bytes))
            .ok(),
        p_n.slab_idx,
        p_n.intra_offset,
        p_n.len_bytes,
    );
}

/// Bug: lazy slab allocation from the backward direction.
///
/// Forward never touches slab N. Backward wraps into it. The
/// `ensure_slab(slab_idx)` must materialize that slab on first backward
/// touch. Verify cuda_malloc_count is incremented from the backward path.
#[test]
fn lazy_slab_alloc_from_backward_only() {
    let device = cuda_device();
    let slab_bytes = 1024;
    let num_slabs = 4;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("new");

    // No forward allocations at all.
    assert_eq!(ring.cuda_malloc_count(), 0);
    assert_eq!(ring.slabs_allocated(), 0);

    // First backward alloc: starts at allocation_start = total_bytes.
    // cur_slab_idx = num_slabs (one past), cur_intra = slab_bytes (per
    // the special case in alloc_backward_impl). Then since cur_intra
    // (slab_bytes) is NOT < num_bytes (256) → no slab jump. cand_idx =
    // num_slabs, cand_intra_top = slab_bytes. Then... HOLD ON:
    // cand_slab_idx_signed = num_slabs as isize, NOT < 0 → no wrap.
    // slab_idx = num_slabs (out of bounds!). Then
    // self.ensure_slab(num_slabs) → panics on slabs[num_slabs].
    //
    // This is a real bug: the initial-state special case sets
    // cur_slab_idx = self.slabs.len() - 1 (line 386), which is correct.
    // Re-reading: yes, line 386 sets it to last. OK, then we land in
    // last slab, intra = slab_bytes - 256 = 768. Good.
    let p = {
        let mut hb = ring.backward_handle(0);
        hb.alloc(256).expect("backward first alloc")
    };
    assert_eq!(
        p.slab_idx,
        num_slabs - 1,
        "first backward lands in last slab"
    );
    assert_eq!(ring.cuda_malloc_count(), 1, "cudaMalloc fired exactly once");
    assert_eq!(ring.slabs_allocated(), 1);

    // Continue backward into slab num_slabs-2 — another cudaMalloc.
    let _ = {
        let mut hb = ring.backward_handle(0);
        // Fill rest of last slab to force jump.
        hb.alloc(slab_bytes - 256)
            .expect("bwd fills rest of last slab")
    };
    assert_eq!(
        ring.cuda_malloc_count(),
        1,
        "still in last slab; no new malloc"
    );
    let _ = {
        let mut hb = ring.backward_handle(0);
        hb.alloc(256).expect("bwd into previous slab")
    };
    assert_eq!(
        ring.cuda_malloc_count(),
        2,
        "second cudaMalloc on first touch of slab num_slabs-2"
    );
    assert_eq!(ring.slabs_allocated(), 2);
}

/// Bug: oversized alloc rejection off-by-one.
/// The existing `alloc_rejects_oversized` uses size = 2*slab_bytes which
/// is far beyond slab_bytes. Test the exact boundary:
/// - num_bytes = slab_bytes exactly: must succeed (single slab fits)
/// - num_bytes = slab_bytes + 1: must error
#[test]
fn alloc_size_boundary_at_slab_bytes_exact_and_off_by_one() {
    let device = cuda_device();
    let slab_bytes = 1024;
    let mut ring = RingAllocator::new(device, 4, slab_bytes).expect("new");

    // Exactly slab_bytes — must fit.
    {
        let mut h = ring.forward_handle(0);
        let p = h
            .alloc(slab_bytes)
            .expect("alloc of exactly slab_bytes must succeed (fills one slab)");
        assert_eq!(p.len_bytes, slab_bytes);
        assert_eq!(p.intra_offset, 0);
    }

    // slab_bytes + 1 — must error.
    {
        let mut h = ring.forward_handle(0);
        let r = h.alloc(slab_bytes + 1);
        assert!(
            r.is_err(),
            "alloc of slab_bytes+1 must error; got {:?}",
            r.map(|p| (p.slab_idx, p.intra_offset, p.len_bytes))
        );
    }
}

/// Verify that drop semantics of `RingForwardHandle` are correctly
/// "no-op" — dropping the handle does not advance/retreat any cursor.
///
/// Per design doc §5: "Bytes return to the pool when the matching
/// RingAllocator::reset() runs. ... Dropping a RingPtr does nothing".
/// The handle similarly does not change cursor state on Drop.
#[test]
fn forward_handle_drop_is_noop_for_cursors() {
    let device = cuda_device();
    let mut ring = RingAllocator::new(device, 2, 1024).expect("new");

    let end_before = ring.allocation_end();
    let start_before = ring.allocation_start();

    // Create a handle but don't alloc.
    {
        let _h = ring.forward_handle(0);
        // Drop here on scope exit.
    }
    assert_eq!(ring.allocation_end(), end_before, "drop must not move end");
    assert_eq!(
        ring.allocation_start(),
        start_before,
        "drop must not move start"
    );

    // Same for backward.
    {
        let _h = ring.backward_handle(0);
    }
    assert_eq!(ring.allocation_end(), end_before);
    assert_eq!(ring.allocation_start(), start_before);

    // And: dropping the handle after an alloc should leave cursors at the
    // post-alloc state (no rollback of the alloc).
    let post_alloc_end;
    {
        let mut h = ring.forward_handle(0);
        let _p = h.alloc(256).expect("fwd alloc");
        post_alloc_end = h.allocation_end();
    }
    assert_eq!(
        ring.allocation_end(),
        post_alloc_end,
        "alloc persists after handle drop"
    );
}

// ---------------------------------------------------------------------------
// Skeptic Phase 1 — additional adversarial coverage tests
// ---------------------------------------------------------------------------
//
// These tests target gaps not covered by Builder + Bug Fixer:
// - symmetric case of the bug-fix (forward fills first, then backward, then
//   forward attempts wrap)
// - OT-parity byte positions for the backward direction (Bug Fixer only
//   covered forward)
// - reset() after backward + forward both fired — both cursors reset to ends
// - reset semantics around wrap-once forward-only case
// - forward-then-backward, forward attempts wrap with backward in opposite
//   half — symmetric to Bug Fixer's test that had backward go first

/// Symmetric to `forward_wrap_with_backward_active_does_not_lap_silently`:
/// forward fires FIRST (filling several slabs), THEN backward fires, THEN
/// forward attempts to wrap.
///
/// Bug Fixer's `forward_wrap_with_backward_active...` test sets up the
/// scenario backward-first. The same fwd-overlap-with-bwd hazard exists in
/// the order: forward, backward, forward wrap. The fix `allocation_start <
/// total_bytes` is order-independent — but let's prove it.
#[test]
fn forward_wrap_after_forward_then_backward_also_errors() {
    let device = cuda_device();
    let slab_bytes = 1024;
    let num_slabs = 2;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("new");

    // Forward fills slab 0 partially.
    {
        let mut hf = ring.forward_handle(0);
        let _ = hf.alloc(500).expect("a");
        let _ = hf.alloc(500).expect("b"); // end = 1012
        let _ = hf.alloc(500).expect("c"); // jumps to slab 1, end = 1524
    }
    assert_eq!(ring.allocation_end(), 1524);
    assert_eq!(ring.allocation_start(), 2048);

    // Now backward fires — moves allocation_start down.
    {
        let mut hb = ring.backward_handle(0);
        let _ = hb.alloc(100).expect("backward after forward");
    }
    let start_before_wrap_attempt = ring.allocation_start();
    assert!(
        start_before_wrap_attempt < ring.total_bytes(),
        "backward must have moved cursor down"
    );

    // Forward attempts a 700-byte alloc that forces wrap.
    // Per Bug Fixer's invariant, this must error (allocation_start <
    // total_bytes is the trigger), NOT silently wrap to slab 0.
    let result = {
        let mut hf = ring.forward_handle(2);
        hf.alloc(700)
    };
    assert!(
        result.is_err(),
        "forward wrap with backward having previously fired must error \
         regardless of fwd-bwd interleaving order. Got Ok({:?}); \
         allocation_start={}, allocation_end={}",
        result
            .as_ref()
            .map(|p| (p.slab_idx, p.intra_offset, p.len_bytes))
            .ok(),
        ring.allocation_start(),
        ring.allocation_end(),
    );
}

/// OT parity (backward): hand-computed byte positions from OT lines 90-109
/// applied to backward direction. Bug Fixer only covered the forward case.
///
/// OT line 91-92: backward index = start/size, intra = start%size.
/// Line 94-97: if cur_intra < num_bytes, jump prev (idx-1, slab_bytes).
/// Line 103: new_intra = floor_16(intra_top - num_bytes).
/// Line 107: new start = idx*size + new_intra.
///
/// Trace for slab_bytes=1024, num_slabs=4, sequence [200, 300, 600, 100]:
/// initial start = 4*1024 = 4096.
/// 1. start=4096. Special-case: cur=(3, 1024). 1024 >= 200, no jump.
///    new_intra = floor_16(1024 - 200) = 816. global = 3*1024+816 = 3888.
/// 2. start=3888: idx=3, intra=816. 816 >= 300, no jump.
///    new_intra = floor_16(816 - 300) = floor_16(516) = 512. global = 3*1024+512 = 3584.
/// 3. start=3584: idx=3, intra=512. 512 < 600 → jump (2, 1024).
///    new_intra = floor_16(1024 - 600) = floor_16(424) = 416. global = 2*1024+416 = 2464.
/// 4. start=2464: idx=2, intra=416. 416 >= 100, no jump.
///    new_intra = floor_16(416 - 100) = floor_16(316) = 304. global = 2*1024+304 = 2352.
#[test]
fn ot_parity_backward_sequence_byte_positions() {
    let device = cuda_device();
    let slab_bytes = 1024;
    let num_slabs = 4;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("new");

    let expected: &[(usize, usize, usize)] = &[
        // (slab_idx, intra_offset, allocation_start AFTER alloc)
        (3, 816, 3888),
        (3, 512, 3584),
        (2, 416, 2464), // jumped
        (2, 304, 2352),
    ];
    let sizes = [200, 300, 600, 100];

    let mut h = ring.backward_handle(0);
    for (i, (&n, &(exp_slab, exp_intra, exp_start))) in
        sizes.iter().zip(expected.iter()).enumerate()
    {
        let p = h.alloc(n).expect("alloc");
        assert_eq!(
            (p.slab_idx, p.intra_offset),
            (exp_slab, exp_intra),
            "step {i}: backward alloc({n}) slab/intra mismatch — got ({}, {}), expected ({exp_slab}, {exp_intra})",
            p.slab_idx, p.intra_offset
        );
        assert_eq!(
            h.allocation_start(),
            exp_start,
            "step {i}: allocation_start mismatch — got {}, expected {exp_start}",
            h.allocation_start()
        );
    }
}

/// reset() after both cursors moved — both must return to extremes, and
/// next allocations behave as a fresh ring.
///
/// Test 5 covers reset after fwd-then-bwd but only spot-checks the count.
/// This test verifies the full post-reset state and that a subsequent
/// fwd-then-bwd round respects the bug-fix invariant cleanly.
#[test]
fn reset_after_both_cursors_moved_restores_invariants() {
    let device = cuda_device();
    let slab_bytes = 1024;
    let num_slabs = 4;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("new");

    // Step 1: forward + backward both fire.
    {
        let mut hf = ring.forward_handle(0);
        let _ = hf.alloc(500).expect("fwd");
    }
    {
        let mut hb = ring.backward_handle(0);
        let _ = hb.alloc(500).expect("bwd");
    }
    let mid_end = ring.allocation_end();
    let mid_start = ring.allocation_start();
    assert!(mid_end > 0);
    assert!(mid_start < ring.total_bytes());
    let slab_count_before_reset = ring.cuda_malloc_count();

    // Reset.
    ring.reset();
    assert_eq!(ring.allocation_end(), 0, "reset puts fwd cursor at 0");
    assert_eq!(
        ring.allocation_start(),
        ring.total_bytes(),
        "reset puts bwd cursor at total_bytes"
    );
    assert_eq!(
        ring.cuda_malloc_count(),
        slab_count_before_reset,
        "reset does not free or alloc slabs"
    );

    // Step 2: invariants must hold cleanly again.
    {
        let mut hf = ring.forward_handle(0);
        let p = hf.alloc(500).expect("step-2 fwd alloc");
        assert_aligned_16(&p);
        // Forward starts fresh at slab 0, intra 0 — confirms reset cleared
        // mid-state and didn't leave stale wrap-related state behind.
        assert_eq!(
            (p.slab_idx, p.intra_offset),
            (0, 0),
            "step-2 fwd starts at fresh slab 0, intra 0"
        );
    }
    {
        let mut hb = ring.backward_handle(0);
        let p = hb.alloc(500).expect("step-2 bwd alloc");
        assert_aligned_16(&p);
        // Backward starts fresh at top of last slab.
        assert_eq!(p.slab_idx, num_slabs - 1, "step-2 bwd starts at last slab");
    }

    // Step-2 cudaMalloc count unchanged (slabs already materialized).
    assert_eq!(
        ring.cuda_malloc_count(),
        slab_count_before_reset,
        "step-2 reuses existing slabs"
    );
}

/// The "forward-only wrap silently laps live forward" hazard is documented
/// (Test 4) but worth pinning EXPLICITLY: with no reset and only-forward
/// activity, a wrap returns a pointer that aliases an earlier forward
/// allocation's bytes. This is the structural OT behavior; this test
/// pins it so any future Phase 2 detection regression is caught.
///
/// Tagged with the design-doc-§4 NOTE: forward-only wrap WITHOUT reset
/// silently aliases. Phase 2 callers MUST reset between steps.
#[test]
fn forward_only_wrap_aliases_prior_forward_alloc_without_reset() {
    let device = cuda_device();
    let slab_bytes = 1024;
    let num_slabs = 2;
    let mut ring = RingAllocator::new(device, num_slabs, slab_bytes).expect("new");

    // Forward fills both slabs.
    let p0 = {
        let mut hf = ring.forward_handle(0);
        let p0 = hf.alloc(1024).expect("a"); // (0, 0, 1024)
        let _p1 = hf.alloc(1024).expect("b"); // (1, 0, 1024)
        p0
    };
    assert_eq!((p0.slab_idx, p0.intra_offset), (0, 0));

    // Backward cursor untouched, allocation_start == total_bytes.
    assert_eq!(ring.allocation_start(), ring.total_bytes());

    // Wrap-around alloc — succeeds because backward is idle.
    // Returns (slab 0, intra 0): EXACTLY p0's range.
    let wrap = {
        let mut hf = ring.forward_handle(2);
        hf.alloc(512)
            .expect("wrap-around succeeds in forward-only mode")
    };
    assert_eq!(
        wrap.slab_idx, 0,
        "wrap returns to slab 0 (Phase 1 documented OT-faithful behavior)"
    );
    assert_eq!(
        wrap.intra_offset, 0,
        "wrap returns to intra 0 — silently overwrites p0's range"
    );
    assert_eq!(
        wrap.device_ptr, p0.device_ptr,
        "wrap's device_ptr ALIASES p0's device_ptr — caller-visible \
         silent alias. Phase 2 must reset between steps to avoid this."
    );
}

/// OT parity: the exact byte-position of a sequence of forward allocs
/// should match what OT's `allocate_like` would produce given the same
/// num_bytes sequence and the same slab_bytes.
///
/// OT line 71-72: forward index = end/size, intra = ceil_16(end%size).
/// Line 74-77: if cur_intra + num_bytes > size, jump.
/// Line 83-88: end = idx*size + intra + num_bytes (after add).
///
/// Trace for slab_bytes=1024, sequence [200, 300, 600, 100]:
/// 1. end=0: idx=0, intra=ceil_16(0)=0. fits (0+200<=1024). end → 200.
/// 2. end=200: idx=0, intra=ceil_16(200)=208. fits (208+300=508). end → 508.
/// 3. end=508: idx=0, intra=ceil_16(508)=512. 512+600=1112 > 1024 → jump.
///    idx=1, intra=0. end → 1*1024 + 0 + 600 = 1624.
/// 4. end=1624: idx=1, intra=ceil_16(1624%1024)=ceil_16(600)=608. fits.
///    end → 1*1024 + 608 + 100 = 1732.
#[test]
fn ot_parity_forward_sequence_byte_positions() {
    let device = cuda_device();
    let slab_bytes = 1024;
    let mut ring = RingAllocator::new(device, 4, slab_bytes).expect("new");

    let expected: &[(usize, usize, usize)] = &[
        // (slab_idx, intra_offset, ending_allocation_end)
        (0, 0, 200),
        (0, 208, 508),
        (1, 0, 1624), // jumped
        (1, 608, 1732),
    ];
    let sizes = [200, 300, 600, 100];

    let mut h = ring.forward_handle(0);
    for (i, (&n, &(exp_slab, exp_intra, exp_end))) in sizes.iter().zip(expected.iter()).enumerate()
    {
        let p = h.alloc(n).expect("alloc");
        assert_eq!(
            (p.slab_idx, p.intra_offset),
            (exp_slab, exp_intra),
            "step {i}: alloc({n}) slab/intra mismatch — got ({}, {}), expected ({exp_slab}, {exp_intra})",
            p.slab_idx, p.intra_offset
        );
        assert_eq!(
            h.allocation_end(),
            exp_end,
            "step {i}: allocation_end mismatch — got {}, expected {exp_end}",
            h.allocation_end()
        );
    }
}
