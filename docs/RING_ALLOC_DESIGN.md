# Ring Allocator Core — Phase 1 design

**Date**: 2026-05-14
**Status**: Phase 1 (this doc) — core data structure + microbench. No consumer wiring.
**Spec source**: `flame-core/docs/OFFLOAD_GAPS_vs_ONETRAINER.md` §Gap 1.
**Port target**: `/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py:37-222`
(`ceil_16` / `floor_16` / `StaticLayerTensorAllocator` / `StaticLayerAllocator`).
**Replaces**: the deleted Phase 4 skeleton at flame-core@`98dbebc` (off-spec —
used independent forward/backward offsets instead of OT's shared cursor pair,
which is what makes the bidirectional invariant work).

This document is the spec for a Builder agent to implement. It is reviewable
in isolation: it captures the data structure, algorithm, OT line citations,
public API, invariants, microbench strategy, and explicit non-goals.

---

## 1. Why a ring, not a bucket free-list

The current `flame_core::cuda_alloc_pool` is a power-of-2 bucketed free list
(PyTorch `CUDACachingAllocator` clone). It corrupts under
BlockOffloader + checkpoint replay on Klein 9B: at step 2, the first
`cudaMalloc` after a step boundary returns `CUDA_ERROR_INVALID_VALUE`
(`project_klein9b_step2_crash_isolation`). The current workaround
(`FLAME_ALLOC_POOL=0`) costs 0.7-1.0 s/step.

The bucket allocator's failure mode is structural: it has no notion of
forward-vs-backward direction. The forward pass extends the working set; the
backward pass extends it from the other end while inputs from forward are
still live. Free-list reuse across this pattern produces overlapping
allocations whose lifetimes don't actually fit the free-list's LIFO model,
and the result is a stream-state divergence that surfaces at the next
`cudaMalloc`.

OneTrainer's `StaticLayerAllocator` sidesteps this by being a ring buffer
with two cursors:

- `allocation_end` grows during forward pass
- `allocation_start` shrinks during backward pass
- The two never cross because they walk the same circular byte space from
  opposite ends; if they meet, the next allocation errors instead of
  silently overlapping.

This is a different allocator shape, not a bug fix to the bucket one. Phase 1
ships the shape as a flame-core primitive (`flame_core::ring_alloc`). Phase 2
(not in scope for this PR) wires consumers and gates the Klein no-workaround
run.

---

## 2. Data structure

```text
slabs:        [Slab0]                [Slab1]                 ...   [SlabN-1]
              ┌────────────────┐     ┌────────────────┐           ┌────────────────┐
              │................│     │................│   ...     │................│
              └────────────────┘     └────────────────┘           └────────────────┘
              0       SlabBytes      SlabBytes  2*SlabBytes       (N-1)*SB  N*SB

global byte space (logical):
   0 ─────────────────────── total_bytes = N * SlabBytes
       ▲                                                  ▲
       allocation_start (backward grows backward from end)  allocation_end (forward grows from start)
```

A slab is a fixed-size GPU `CudaSlice<u8>` allocated via `device.alloc::<u8>(slab_bytes)`.
Slabs are owned by the `RingAllocator`. The logical byte space is the
concatenation of all slabs: byte offset `B` in global space maps to
slab `B / slab_bytes` at intra-slab offset `B % slab_bytes`.

**Two cursors, NOT four.** OT uses exactly one pair: `allocation_start` and
`allocation_end`. The previous skeleton used `forward_slab` /
`forward_offset` / `backward_slab` / `backward_offset` — that was off-spec
because it lets forward and backward overlap on the same slab silently. The
correct model is one global byte cursor per direction; the slab index is
derived (`cursor / slab_bytes`).

**Alignment.** All allocations are 16-byte aligned. Forward uses
`ceil_16` of the candidate start; backward uses `floor_16` of the candidate
start. (OT lines 37-42; see §3.)

**Lazy slab allocation.** A slab is only `cudaMalloc`-ed the first time it
is touched. This matches OT's `ensure_allocation(cache_tensor_index)`
(line 187-197): the slab list begins as `[None; N]`, and a `cudaMalloc` only
fires on the first allocation that lands in slab `i`. This bounds peak
GPU bytes to the working-set high-water mark, not the static slab cap.

**Slab-boundary jump.** When an allocation would cross a slab boundary,
the cursor "jumps" to the start (forward) or end (backward) of the next
slab. The skipped tail bytes of the current slab are wasted for this
forward pass; they become available again when cursors reset between
steps. OT does this at lines 74-77 (forward) and 94-97 (backward).

**Cyclic wrap.** When forward reaches the end of the last slab, it wraps to
slab 0 (OT line 78-82). When backward reaches before slab 0, it wraps to
the last slab (OT line 98-101). This is what makes the cursors meet but
not cross: if forward wraps and backward hasn't yet vacated slab 0, the
next forward allocation in slab 0 sees `allocation_end >= allocation_start`
— which is the invariant violation we detect and error on.

---

## 3. Allocation algorithm

### Forward (`alloc_forward(num_bytes) -> RingPtr`)

Direct port of OT lines 65-89 (`allocate_like` forward branch).

```rust
// Capacity check is at slab-relative level (OT lines 74-77).
let cur_slab_idx = self.allocation_end / self.slab_bytes;
let cur_intra   = ceil_16(self.allocation_end % self.slab_bytes);

let (slab_idx, intra) = if cur_intra + num_bytes > self.slab_bytes {
    // Doesn't fit in remainder of current slab — jump to next slab.
    (cur_slab_idx + 1, 0_usize)
} else {
    (cur_slab_idx, cur_intra)
};

// Cyclic wrap if we walked off the last slab.
let (slab_idx, intra) = if slab_idx * self.slab_bytes + intra + num_bytes
                        > self.total_bytes {
    (0_usize, 0_usize)
} else {
    (slab_idx, intra)
};

// Invariant: the new range [start, end) must not overlap with
// the backward-allocated range [allocation_start, total_bytes) or
// (after wrap) [0, prev_allocation_end). See §4.
let new_global_end = slab_idx * self.slab_bytes + intra + num_bytes;
// (Optional debug check; not present in OT.)

self.allocation_end = new_global_end;
self.ensure_slab(slab_idx)?;  // cudaMalloc on first touch
let device_ptr = self.slabs[slab_idx].as_ref().unwrap().device_ptr() + intra;
return RingPtr { device_ptr, len: num_bytes, slab_idx, intra_offset: intra };
```

### Backward (`alloc_backward(num_bytes) -> RingPtr`)

Direct port of OT lines 90-109 (`allocate_like` backward branch).

```rust
let cur_slab_idx = self.allocation_start / self.slab_bytes;
let cur_intra   = self.allocation_start % self.slab_bytes;

let (slab_idx, intra_top) = if cur_intra < num_bytes {
    // Doesn't fit in head of current slab — jump to previous slab.
    (cur_slab_idx.wrapping_sub(1), self.slab_bytes)
} else {
    (cur_slab_idx, cur_intra)
};

// Cyclic wrap if we walked before slab 0.
let (slab_idx, intra_top) = if slab_idx == usize::MAX /* underflow */ {
    (self.num_slabs - 1, self.slab_bytes)
} else {
    (slab_idx, intra_top)
};

let new_intra = floor_16(intra_top - num_bytes);
self.allocation_start = slab_idx * self.slab_bytes + new_intra;
self.ensure_slab(slab_idx)?;
let device_ptr = self.slabs[slab_idx].as_ref().unwrap().device_ptr() + new_intra;
return RingPtr { device_ptr, len: num_bytes, slab_idx, intra_offset: new_intra };
```

### Helpers

```rust
#[inline] fn ceil_16(n: usize) -> usize { (n + 15) & !15 }
#[inline] fn floor_16(n: usize) -> usize { n & !15 }
```

(OT lines 37-42; the bit-mask form is Rust-idiomatic and equivalent because
all inputs are non-negative.)

---

## 4. Invariants

The ring guarantees these properties that `cuda_alloc_pool` does not:

1. **Direction-typed allocation.** A `RingForwardHandle` can only call
   `alloc(n)` that advances forward. A `RingBackwardHandle` can only call
   `alloc(n)` that retreats backward. No bool flag. Misuse is a type error
   at compile time. (Better than OT, which threads `allocate_forward: bool`.)

2. **Bidirectional non-overlap (within a step).** Until cursors reset, all
   bytes returned by a forward allocation lie in `[prev_end, allocation_end)`
   and all bytes returned by a backward allocation lie in
   `[allocation_start, prev_start)`. These ranges are disjoint by
   construction in the **linear (no-wrap) regime**. After a wrap, the
   ring monitors `allocation_end <= allocation_start (mod total_bytes)`;
   violations error rather than silently overlap.

3. **16-byte alignment.** Every returned pointer is `device_ptr % 16 == 0`
   (assuming the slab base pointer is 16-aligned, which CUDA guarantees
   for `cudaMalloc`). Forward uses `ceil_16`, backward uses `floor_16`.

4. **No `cudaFree` during a step.** Slabs are allocated lazily on first
   touch, never freed. Cursors reset between steps via `reset()`;
   slabs remain mapped. This kills the per-call malloc/free churn the
   no-pool path has, and avoids the bucketed pool's stream-state
   corruption window.

5. **Lazy slab allocation observable.** `cudaMalloc` calls during the
   lifetime of a `RingAllocator` equal `# distinct slabs touched`, not
   `# allocations`. Verifiable via the test in §6.

These invariants are testable. Invariant 2 is the one the bucket
allocator violates structurally.

---

## 5. Public API surface

Module path: `flame_core::ring_alloc`. (Sibling to `cuda_alloc_pool`. Not
inside `offload/` because the ring is a general-purpose direction-typed
allocator; offload is a downstream consumer. Phase 2 may decide to colocate;
this is a Phase 1 placement choice.)

### Types

```rust
/// A bidirectional ring allocator over a list of fixed-size GPU slabs.
///
/// Models the working-set memory of a forward-then-backward compute pass.
/// Forward allocations grow `allocation_end`; backward allocations retreat
/// `allocation_start`. Slabs are allocated lazily on first touch.
pub struct RingAllocator { /* private */ }

/// A direction-typed handle for forward-pass allocations within a single block.
///
/// Mirrors OT's `StaticLayerTensorAllocator(allocate_forward=True, layer_index=L)`.
/// Constructed via `RingAllocator::forward_handle(layer_idx)`.
pub struct RingForwardHandle<'a> { /* borrows &'a mut RingAllocator */ }

/// A direction-typed handle for backward-pass allocations within a single block.
pub struct RingBackwardHandle<'a> { /* borrows &'a mut RingAllocator */ }

/// An untyped device byte range returned by the ring allocator.
///
/// Not RAII-freeing on Drop (slabs are owned by the allocator).
/// Bytes return to the pool when the matching `RingAllocator::reset()` runs.
/// Callers cast `device_ptr` to a typed view via `as_cuda_slice::<T>()`.
pub struct RingPtr {
    pub device_ptr: u64,
    pub len_bytes: usize,
    pub slab_idx: usize,
    pub intra_offset: usize,
    // (Optional) handle id for debug provenance.
}
```

### Constructors

```rust
impl RingAllocator {
    /// Construct with `num_slabs` × `slab_bytes` logical capacity.
    ///
    /// Slabs are NOT allocated upfront; the first `alloc_forward` or
    /// `alloc_backward` that lands in slab `i` triggers its `cudaMalloc`.
    ///
    /// Both `num_slabs` and `slab_bytes` must be > 0. `slab_bytes` is
    /// rounded up internally to a multiple of 16.
    pub fn new(
        device: Arc<CudaDevice>,
        num_slabs: usize,
        slab_bytes: usize,
    ) -> Result<Self>;

    /// Equivalent OT default: 10 slabs of size = `ceil(target / 10)`.
    /// Provided as a convenience for sweeps; spec callers go through `new`.
    pub fn with_slabs(
        device: Arc<CudaDevice>,
        num_slabs: usize,
        slab_bytes: usize,
    ) -> Result<Self> { Self::new(device, num_slabs, slab_bytes) }
}
```

### Per-block handles

```rust
impl RingAllocator {
    /// Begin a forward-pass allocation scope for the given block index.
    ///
    /// Returned handle borrows the allocator mutably. Drop the handle
    /// before requesting a `backward_handle` for the same step.
    pub fn forward_handle(&mut self, block_idx: usize) -> RingForwardHandle<'_>;

    /// Begin a backward-pass allocation scope for the given block index.
    pub fn backward_handle(&mut self, block_idx: usize) -> RingBackwardHandle<'_>;
}

impl<'a> RingForwardHandle<'a> {
    /// Allocate `num_bytes` bytes in the forward direction.
    ///
    /// Advances `allocation_end` past the returned range, with 16-byte
    /// pre-alignment. If the current slab is full, jumps to the next slab
    /// (cyclically; see §3).
    pub fn alloc(&mut self, num_bytes: usize) -> Result<RingPtr>;
}

impl<'a> RingBackwardHandle<'a> {
    /// Allocate `num_bytes` bytes in the backward direction.
    pub fn alloc(&mut self, num_bytes: usize) -> Result<RingPtr>;
}
```

### Inspection / lifecycle

```rust
impl RingAllocator {
    pub fn num_slabs(&self) -> usize;
    pub fn slab_bytes(&self) -> usize;
    pub fn total_bytes(&self) -> usize;
    pub fn allocation_start(&self) -> usize;
    pub fn allocation_end(&self) -> usize;
    pub fn slabs_allocated(&self) -> usize; // count of materialized slabs (lazy)
    pub fn cuda_malloc_count(&self) -> u64;  // monotonic counter for tests

    /// Reset cursors to (0, total_bytes). Slabs stay materialized.
    /// Call between training steps.
    pub fn reset(&mut self);
}
```

### Drop / return semantics — explicit

**The ring allocator does NOT free per-allocation.** OT's
`StaticLayerTensorAllocator.deallocate(deallocate_forward)` (lines 113-119)
does not free either — it simply moves the outer cursor to wherever the
sub-allocator's cursor landed. Slabs persist for the life of the allocator.

A returned `RingPtr` is a borrowed view onto a slab the allocator owns. Its
lifetime extends until the next `reset()`. Dropping a `RingPtr` does nothing
— it has no Drop impl. **This is intentional.** Per-allocation reclaim would
break the "cursor walks linearly through the byte space" invariant.

The cost: a long-lived `RingPtr` past a `reset()` becomes a use-after-free.
Phase 2 callers must ensure all `RingPtr` views are dead before the per-step
`reset()` runs. The natural fit is: forward pass produces `RingPtr`s that
backward pass consumes, then step boundary resets.

Note: this differs from the previous Phase 4 skeleton at flame-core@`98dbebc`,
which exposed a `DeviceSlab` RAII handle that retired bytes on Drop. That
matched neither OT's semantics nor the corruption-avoiding ring shape.
Removed deliberately.

---

## 6. Microbench strategy

File: `flame-core/tests/ring_alloc_microbench.rs`.

The microbench is this phase's only consumer. It must exercise every public
function end-to-end and either (a) reproduce the BlockOffloader fwd/bwd
replay pattern that corrupts `cuda_alloc_pool` and show the ring runs
clean, or (b) prove the bidirectional invariant tested under the same
shape sequence. Strategy chosen: **both** — Tests 1-4 prove the
invariants; Test 5 runs the Klein-pattern shape sequence end-to-end
through the ring and asserts the ring runs clean. Direct cross-comparison
against `cuda_alloc_pool` is left to Phase 2 (we know from
`project_klein9b_step2_crash_isolation` that the pool crashes; replicating
that here would require linking the same `BlockOffloader` + autograd
backward path the trainer uses, which is Phase 2 scope).

### Tests

| # | Name | Purpose |
|---|---|---|
| 1 | `forward_only_sequence` | Allocate 100 forward chunks of varying sizes; verify each `RingPtr.device_ptr` is 16-aligned, monotonically non-decreasing within a slab, jumps cleanly across slab boundaries. |
| 2 | `backward_only_sequence` | Mirror: 100 backward chunks; verify `device_ptr` monotonically non-increasing within a slab, jumps cleanly to previous slab. |
| 3 | `bidirectional_alternating` | Alternate `alloc_forward(n)` / `alloc_backward(n)`. Verify `allocation_start >= allocation_end` (no overlap) at all times until cursors meet, at which point further allocs error. |
| 4 | `slab_boundary_stress` | Sizes chosen so each allocation forces a slab-boundary jump. Verify the jump is detected (`slab_idx` increments) and no allocation crosses the in-slab boundary. |
| 5 | `klein_pattern_repro` | 32 forward allocations of size `1 * 1520 * 4096 * 2 = 12_451_840` bytes (Klein BF16 block I/O), then 32 backward allocations of the same size. With slabs sized to hold ~4 blocks each, this stresses lazy allocation, slab transitions, and cyclic wrap. Verify: clean run, `cuda_malloc_count` equals `slabs_allocated`, bidirectional invariant holds at every step. |

All tests have hard pass/fail assertions, not just prints.

### Negative cases included in Tests 1-5

- Zero `num_bytes` → `Err`.
- `num_bytes > slab_bytes` → `Err` (single allocation cannot span slabs).
- Cursors meet → `Err` ("ring exhausted").

### What the microbench does NOT do

- Cross-comparison against `cuda_alloc_pool` corruption. Requires running
  the actual `BlockOffloader` + autograd backward path; that's Phase 2.
- Wall-clock performance vs alternatives. Phase 1 is correctness. Phase 2's
  Klein trainer run produces the perf number per SPEED_CONTRACT Clause 5.

---

## 7. What this phase does NOT do

Explicitly out of scope:

1. **No consumer migration.** No call site in flame-core or EriDiffusion-v2
   is changed to use `RingAllocator`. Existing tensors still allocate
   through `cuda_alloc_pool` and `device.alloc_zeros::<T>`.
2. **No Klein trainer gate.** The "Klein 9B runs without `FLAME_ALLOC_POOL=0`"
   gate from Gap 1 is Phase 2's payload.
3. **No `Tensor::drop` integration.** Phase 2 may add a registry hook;
   Phase 1 does not.
4. **No autograd-direction inference.** `RingAllocator` is driven by
   explicit `forward_handle` / `backward_handle` calls. Reading the
   autograd thread-local for implicit direction is Phase 2 polish.
5. **No coordinator / fraction strategy.** Gaps 3-4 are separate workstreams.
6. **No FP8 / pinned-host variant.** The ring is purely device-side.
   Activation offload (host-side pinned ring) is a separate primitive
   (`GrowOnDemandActivationCache`, already shipped at flame-core@`4f0d026`).

---

## 8. Open questions for Bug Fixer / Skeptic

These are spec ambiguities the Builder is to flag if they bite during
implementation. None block Phase 1 as written, but they shape Phase 2.

1. **Wrap-around vs error on cursor meet.** OT cyclically wraps when a
   cursor walks off the end of the slab list. This is fine because OT
   sizes slabs to comfortably hold one full forward and one full backward
   pass. If our sizing is wrong, wrap silently produces overlap. Current
   spec: detect wrap-into-other-side and **error**. Should we instead
   abort the step or trigger a `reset()`? Defer to Phase 2.

2. **Slab size auto-tuning.** OT computes `cache_tensor_size` from
   `max(target / N, max_tensor_bytes × 2)` + slack (line 169-181).
   Phase 1 takes `slab_bytes` from the caller. Auto-tuning needs a
   profiling pass and lives in the coordinator (Phase 2). OK.

3. **Multi-step lifetimes.** A tensor produced in step N's forward pass
   that's still live in step N+1 (e.g., EMA, optimizer state) cannot be
   ring-allocated — it'd be a use-after-`reset`. Phase 1 doesn't allocate
   anything user-facing, so this is non-issue here. Phase 2 wiring must
   restrict ring allocation to per-step working set only.

4. **Thread safety.** `RingAllocator` is `!Send` / `!Sync` in Phase 1
   (the handle pattern requires `&mut self`). Backward pass in flame-core
   runs on the same thread as forward, so this is fine. If autograd ever
   threads, the lock-free design is open. Phase 1 = single-threaded.

5. **CudaSlice reconstruction.** Returning a typed `CudaSlice<T>` view
   onto a slab requires the `CudaSliceMirror` transmute pattern from
   `cuda_alloc_pool.rs:29-34`. Phase 1 ships `RingPtr` (raw ptr + len)
   and lets Phase 2 build the typed-view helper. Microbench uses
   `RingPtr` directly.

6. **Reset interleaving with active handles.** A `RingForwardHandle`
   borrows `&mut self`. After it drops, the cursors are advanced. Calling
   `reset()` while a handle is live is a borrow-checker error — type
   system handles it. Same for `backward_handle` cross-borrowing.

---

## 9. Doc updates that ship with the implementation commit

Per `flame-core/CLAUDE.md`:

- `docs/FLAME_INDEX.md` — entries for `RingAllocator`, `RingForwardHandle`,
  `RingBackwardHandle`, `RingPtr`, and every `pub fn` on them.
- `docs/FLAME_MODULES.md` — paragraph for `ring_alloc/`.
- `docs/FLAME_CONVENTIONS.md` — entry: "Ring allocator: bytes are not
  reclaimed per-`Drop`; they return to the pool at `reset()`. `RingPtr` is
  a borrowed view, not RAII. Match scope: one ring per training step."

---

## 10. Acceptance checklist (Builder → Bug Fixer hand-off)

- [ ] `cargo build --release --lib` clean.
- [ ] `cargo test --release --features cuda --test ring_alloc_microbench` —
      all 5 tests pass.
- [ ] Doc updates listed in §9 included in the same commit.
- [ ] Verification log at `/tmp/builder_phase1_verification.md` filled.
- [ ] No call site outside the new module or tests modified.

If any check fails, do not commit. Escalate to Skeptic / Bug Fixer per the
session protocol.
