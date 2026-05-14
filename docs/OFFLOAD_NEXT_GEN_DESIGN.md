# flame-core offload next-gen — design doc

**Date**: 2026-05-13 (initial), 2026-05-14 (scope reduction)
**Replaces**: today's `ActivationOffloadPool` (fixed slots, doesn't fit Klein). The `cuda_alloc_pool` corruption under offload is now tracked separately at `HANDOFF_2026-05-14_TRAINER_REGRESSION_FAILURE.md`.
**Borrows from**: OneTrainer's `docs/RamOffloading.md` + `/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py`.
**Goal**: medium models (Klein 9B, Wan 14B+14B-LoRA) that almost fit in 24 GB but don't. FlexTensor handles the truly large.

## Status (2026-05-14)

| Phase | Status | Commits |
|---|---|---|
| 1: `GrowOnDemandActivationCache` | **DELIVERED** | flame-core@`4f0d026`, fix@`6b5d0a5` |
| 2a: `checkpoint_offload_boundary` API | **DELIVERED** | flame-core@`c71890c` |
| 2b: Cache-replay backward correctness | **DELIVERED** | flame-core@`6b5d0a5` |
| 6 (real): klein.rs → `checkpoint_offload_boundary` | **DELIVERED** | EriDiffusion-v2@`1994cac`, `cff0a60` |
| 3: `OffloadCoordinator` skeleton | **DELETED 2026-05-14** | was @`e2ec9f1` — no real consumer beyond a one-line wrapper, BlockGuard::Drop was a stub. v2 now installs the cache directly. |
| 4: `RingSlabAllocator` skeleton | **DELETED 2026-05-14** | was @`98dbebc`, microbench@`31e550b` — off-spec wiring (BlockOffloader weight slots, not the activation path the spec called for), verified ineffective for the Klein step-2 crash. |
| 4 (restart Phase 1): `flame_core::ring_alloc` faithful OT port | **SHIPPED 2026-05-14** | flame-core@`2927038` (core: design doc + impl + 9 tests), `828c98f` (Bug Fixer: silent wrap-and-lap fix + 6 tests), `e472712` (Skeptic: 4 adversarial tests, 19 total). Faithful port of OT `StaticLayerAllocator`/`StaticLayerTensorAllocator` (LayerOffloadConductor.py:37-222). |
| 4 (restart Phase 2a): `PoolMissAllocator` + `RingPoolAdapter` opt-in | **SHIPPED 2026-05-14** | flame-core (this commit) + EriDiffusion-v2 Klein opt-in (`KLEIN_POOL_RING=1`). Tests whether routing `cuda_alloc_pool` cache MISSES through a ring fixes the Klein 9B step-2 `INVALID_VALUE` crash without `FLAME_ALLOC_POOL=0`. Pool cache HITs unchanged. 3 GPU smoke tests; full Klein 5-step trainer smoke is the gate. Phase 2b (30-step run, default-on, removal of workaround) gated on the 5-step result. |
| 5: Fraction knob + 3-case strategy | **DELETED 2026-05-14** | was @`7d3348d`/@`8f13318` — library type with no production consumer. The fraction-cycle pattern from OT is already implemented inside `BlockOffloader`'s existing strategies (TwoSlot/Knapsack/Adaptive). A second strategy surface added complexity without value. |
| 7+: Other trainer migrations | **DEFERRED** | Per-trainer wiring of `checkpoint_offload_boundary` (the Phase 6 pattern) — Wan22, Chroma, Flux, Ernie, SD35, SDXL — to be done as those trainers are touched. No coordinator wrapping needed; direct cache install via `set_grow_activation_cache`. |
| 8: HostRamBudget + telemetry | **DEFERRED** | Not blocking any current goal. |

**Live design**: Phase 1 + 2 (the activation-side primitives) shipped and exercised end-to-end in Klein 9B. Layer-side offload remains driven by `BlockOffloader`'s existing strategies, which already match OT's RamOffloading.md fraction-cycle pattern. The Phase 3-5 "coordinator + ring + strategy" axis turned out to be redundant infrastructure when measured against the working trainer code — deleted rather than carried as weight.

The remainder of this doc preserves the original design as historical context for anyone re-evaluating the coordinator pattern (e.g., when adding a model that doesn't fit BlockOffloader's existing surface).

This is a design, not an implementation. Each component below is one PR/session (per SPEED_CONTRACT measurement-first rule).

---

## What we keep from OneTrainer

| OT mechanism | Why it works |
|---|---|
| **Narrow checkpoint save scope** (block I/O only, internal recomputes) | Bounds the per-block save to ~3 tensors vs our ~80. Pool/cache never overflows on block-shaped models. |
| **Grow-on-demand activation cache** (reserve_cache appends slabs) | No fixed slot sizing. Allocates exactly what's needed; appends a new cache tensor when full. Self-tuning. |
| **Ring buffer for layer weights** (allocation_start / allocation_end cursors) | Bidirectional fwd/bwd access without fragmentation. Fixes the alloc-pool corruption we papered over with FLAME_ALLOC_POOL=0. |
| **Three CUDA streams** (train, layer_transfer, activations_transfer) | Compute and transfer overlap. We already do this with one transfer stream; second one for activations would split contention. |
| **CUDA event wrapper** (SyncEvent: record/wait/synchronize) | Clean abstraction. We have the same primitives directly on cudarc; sugar on top would be nicer. |
| **before_layer / after_layer hooks** | Clear separation: trainer says "I'm about to execute layer i," conductor handles offload state. |
| **Layer offload fraction knob** (0..1, user knob) | Single tunable. Sweeps cleanly from "everything resident" to "everything offloaded." |
| **3 transition cases** (fwd→bwd, fwd→fwd, bwd→fwd) per-layer resident-set precompute | Optimal prefetch lookahead per direction. |

## What we do BETTER than OneTrainer

These exploit Rust + the existing flame-core primitives:

| Improvement | How |
|---|---|
| **RAII lifecycle guard** | `coordinator.before_block(idx)` returns a `BlockGuard` whose Drop records `compute_done` event. Trainer can't forget `after_block` — compile error if forgotten. OT relies on runtime call ordering (Python). |
| **Type-safe direction** | Separate `alloc_forward(n)` and `alloc_backward(n)` methods, NOT a bool parameter. OT threads `allocate_forward: bool` through every call site. |
| **Auto direction from autograd state** | Thread-local autograd direction (forward-recording vs backward-traversing). Ring allocator reads it; trainer doesn't need to pass direction. |
| **FP8 compression preserved** | Our existing `ActivationOffloadPool::OffloadCompression::FP8` already works (halves pinned bytes, quant kernel on transfer stream). Carry forward; OT doesn't have this. |
| **Single tenet-compliant API surface** | One `OffloadCoordinator` owns both weight blocks and activations. OT has separate `LayerOffloadConductor` + two allocators + manual coordination. We collapse to one struct. |
| **Shared host RAM budget** | Block-pinned + activation-pinned consult a `HostRamBudget` so they don't fight for memory. OT relies on user to size correctly. |
| **Zero host stall in per-block path** | SPEED_CONTRACT Clause 1: GPU-event chains only, no `cudaStreamSynchronize` in steady-state. OT's Python wrapper has interpreter overhead per call; we don't pay that. |

---

## The system at a glance

```
┌──────────────────────────────────────────────────────────────┐
│ OffloadCoordinator (flame_core::offload::coordinator)        │
│                                                              │
│ ┌─────────────────┐  ┌────────────────────────────────────┐  │
│ │ BlockOffloader  │  │ GrowOnDemandActivationCache        │  │
│ │ (weight blocks) │  │ (block I/O activations)            │  │
│ │ — exists today  │  │ — replaces ActivationOffloadPool   │  │
│ └─────────────────┘  └────────────────────────────────────┘  │
│         │                            │                       │
│         ▼                            ▼                       │
│ ┌────────────────────────────────────────────────────────┐   │
│ │ RingSlabAllocator (device)                             │   │
│ │ — replaces cuda_alloc_pool's role                      │   │
│ │ — bidirectional, slab-list, ring cursor                │   │
│ └────────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐   │
│ │ HostRamBudget (shared budget for both pinned pools)    │   │
│ └────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
        ▲                                       ▲
        │ before_block(i) → BlockGuard          │ checkpoint_offload_boundary
        │ (RAII, records compute_done on drop)  │ (narrow scope, block I/O only)
        │                                       │
   ┌────┴──────────────────────────────────────┴───────────┐
   │ Trainer (train_klein.rs etc.)                          │
   │   for block_idx in 0..num_blocks {                     │
   │       let _g = coord.before_block(block_idx);          │
   │       let out = checkpoint_offload_boundary(           │
   │           &[input], move || block.forward(input)       │
   │       )?;                                              │
   │       // _g drops here, records compute_done event     │
   │   }                                                    │
   └────────────────────────────────────────────────────────┘
```

---

## Component-by-component design

### A. `GrowOnDemandActivationCache` (replaces fixed-slot pool)

**Borrows from OT**: `StaticActivationAllocator.reserve_cache(tensors)` allocates exactly `sum(tensor.numel × element_size) + padding` bytes; appends a new cache tensor when no existing one has room. Cursor advances per push; resets each backward pass.

**Better than OT**: pre-flight isn't strictly required — `push(tensor)` can grow the cache on first miss. We retain OT's `reserve` hint for warmup-time pre-allocation when the trainer knows tensor sizes ahead of time.

**API sketch** (Rust):

```rust
pub struct GrowOnDemandActivationCache {
    device: Arc<CudaDevice>,
    slabs: Vec<PinnedHostBuffer<u8>>,
    slab_size_hint: usize,        // default 256 MB; growth granularity
    cursor: usize,                // current slab index
    offset: usize,                // bytes used in current slab
    transfer: TransferStream,     // dedicated CUDA stream
    push_events: HashMap<HandleId, CudaEvent>, // DtoH done
    pull_events: HashMap<HandleId, CudaEvent>, // HtoD done
    keep_alive: HashMap<HandleId, Tensor>,     // GPU tensor lifetime
    compression: OffloadCompression,           // None | FP8
    epoch: u64,                                // invalidate stale handles on reset
}

pub struct CacheHandle { id: HandleId, epoch: u64 }

impl GrowOnDemandActivationCache {
    pub fn new(device: Arc<CudaDevice>, slab_size_hint: usize, compression: OffloadCompression) -> Self;
    pub fn reserve_hint(&mut self, total_bytes: usize) -> Result<()>;
    pub fn push(&mut self, src: &Tensor) -> Result<CacheHandle>;
    pub fn pull(&mut self, handle: CacheHandle, dtype: DType, shape: &[usize]) -> Result<Tensor>;
    pub fn reset(&mut self);  // call between fwd and the next fwd; bumps epoch
    pub fn host_bytes(&self) -> usize;
}
```

**Tenet checks**:
- Tenet 1 (fix the primitive): in flame-core. Every trainer that needs activation offload uses this same primitive.
- Tenet 2 (API makes right easy): no slot-size guessing; one method per push. Wrong thing is hard.
- Clause 1 (no host stall): all H2D and D2H on `transfer` stream; default-stream waits via events.
- Clause 5 (bandwidth-bound): pinned host, async, single stream — bandwidth-limited by PCIe.

**Microbench gate**: `tests/grow_on_demand_cache_microbench.rs` — push N tensors of varying sizes, verify (a) no slab miss when sized correctly, (b) graceful growth when not, (c) `transfer` stream IS bandwidth-bound (≥ 80% of PCIe theoretical).

**Effort**: 1 PR, 1-2 sessions.

### B. Narrow-scope `checkpoint_offload_boundary` (replaces current `checkpoint_offload`)

**Borrows from OT**: OT saves only block I/O (input + output activations) and recomputes everything internal. Our current `checkpoint_offload` extracts the whole sub-tape — 25× more tensors than OT saves.

**Better than OT**: leverage `AutogradContext::checkpoint` machinery we already have. The new function is `checkpoint` + offload-the-input-and-output to pinned RAM, that's it.

**Semantics**:
- Forward: run closure under detached autograd (no sub-tape recording, same as `checkpoint`). Record a recompute closure. Push the closure's inputs and outputs to `GrowOnDemandActivationCache`.
- Backward: pull inputs from cache → recompute closure with autograd enabled → backward through the recomputed sub-tape.

**API**:

```rust
impl AutogradContext {
    pub fn checkpoint_offload_boundary<F>(inputs: &[Tensor], f: F) -> Result<Tensor>
    where F: Fn(&[Tensor]) -> Result<Tensor> + Send + Sync + 'static;
}
```

Closure now takes `&[Tensor]` (the pulled inputs at recompute time) instead of zero args. Cleaner than current `checkpoint` because the closure doesn't have to capture inputs.

**Klein expected save count**: 2 tensors per block (input + output) × 32 blocks = 64 saves/step. Each ~12 MB BF16 (`[1, 1520, 4096]`). Total pinned: ~768 MB raw, ~384 MB FP8. Fits a 24 GB system comfortably.

**Tenet checks**:
- Tenet 5 (reject wrong-place fix): the existing trainer-side workaround (wrap each block's qkv+mlp inside) is incorrect by Tenet 1. The fix is in autograd, where the scope is.
- Tenet 3 (dispatcher): this is autograd dispatch policy, lives in the autograd module.

**Effort**: 1 PR, 1-2 sessions. Lower bound if autograd_v2 bridge is enough; upper if integration with v3's `checkpoint` needs care.

### C. `RingSlabAllocator` (replaces `cuda_alloc_pool`'s role for offload-shaped workloads)

**Borrows from OT**: `StaticLayerAllocator` allocates fixed-count cache tensors (slabs), per-layer sub-allocations via `StaticLayerTensorAllocator` with `allocation_start` / `allocation_end` cursors. Forward advances `end`; backward retreats `start`. Crossing a slab boundary jumps to next slab; wraps cyclically.

**Better than OT**:
- Type-safe direction (`alloc_forward` vs `alloc_backward`, not a bool flag).
- Drop-based deallocation; no manual `deallocate(deallocate_forward)` call.
- Slab size auto-tuned at first use to peak observed allocation in one pass + headroom.

**API**:

```rust
pub struct RingSlabAllocator {
    device: Arc<CudaDevice>,
    slabs: Vec<CudaSlice<u8>>,
    slab_bytes: usize,
    allocation_start: usize,
    allocation_end: usize,
    direction: AutogradDirection,  // Forward | Backward
}

pub enum AutogradDirection { Forward, Backward }

impl RingSlabAllocator {
    pub fn new(device: Arc<CudaDevice>, num_slabs: usize, slab_bytes: usize) -> Result<Self>;
    pub fn alloc(&mut self, num_bytes: usize) -> Result<DeviceSlab>;  // direction-aware
    pub fn set_direction(&mut self, d: AutogradDirection);
    pub fn reset_cursors(&mut self);
}

pub struct DeviceSlab { /* RAII handle; Drop returns range to allocator */ }
```

**Where it plugs in**: `Tensor::drop` consults the allocator when the tensor was allocated through it. Existing `cuda_alloc_pool` callers can migrate by changing their alloc path from `pool_alloc_f32` to `RingSlabAllocator::alloc`, but `cuda_alloc_pool` stays for non-offload-shaped workloads (training tensors that aren't part of the block ring).

**Tenet checks**:
- Tenet 1: in flame-core. Every model that uses offload inherits the bidirectional-safe allocator.
- Tenet 4 (measurement): microbench against current `cuda_alloc_pool` on synthetic bidirectional fwd/bwd patterns. Target: no fragmentation, no corruption, similar or better wall time.

**Microbench gate**: `tests/ring_slab_allocator_microbench.rs` — replicate the Klein-9B-with-offload access pattern that corrupted `cuda_alloc_pool`. New allocator must run clean and faster than the no-pool fallback.

**Effort**: 1 PR, 2-3 sessions. Higher than A or B because it's a new memory subsystem with tape-direction integration.

### D. `OffloadCoordinator` (unifies the surface)

**Borrows from OT**: `LayerOffloadConductor` is the single entry point trainers talk to. Wraps the two allocators + strategy.

**Better than OT**:
- `BlockGuard` is RAII — `before_block(idx)` returns a guard, Drop on the guard records `compute_done`. Trainer literally cannot forget to record the event.
- Shared `HostRamBudget`.
- Single `offload_fraction: f32` knob that the trainer passes once; coordinator computes resident sets for all 3 transition cases internally.

**API**:

```rust
pub struct OffloadCoordinator {
    blocks: BlockOffloader,
    activations: GrowOnDemandActivationCache,
    strategy: BlockOffloadStrategy,  // 3-case resident-set precompute
    budget: HostRamBudget,
}

pub struct BlockGuard<'a> {
    coordinator: &'a mut OffloadCoordinator,
    block_idx: usize,
    direction: AutogradDirection,
}

impl OffloadCoordinator {
    pub fn new(
        device: Arc<CudaDevice>,
        weight_pool: BlockOffloader,
        activation_cache: GrowOnDemandActivationCache,
        layer_offload_fraction: f32,
        budget: HostRamBudget,
    ) -> Result<Self>;

    pub fn before_block(&mut self, idx: usize) -> Result<BlockGuard>;
    // BlockGuard::Drop records compute_done event automatically
}
```

Klein trainer usage (per-step):

```rust
for block_idx in 0..num_blocks {
    let _guard = coordinator.before_block(block_idx)?;
    let out = AutogradContext::checkpoint_offload_boundary(
        &[input.clone()],
        |inputs| block.forward_inner(&inputs[0])
    )?;
    input = out;
    // _guard drops at end of scope → records compute_done event
}
```

Compared to today's Klein code, this is ~5 lines per block loop and the trainer doesn't touch the offloader or pool directly.

**Tenet checks**:
- Tenet 2 (API makes right easy): `before_block` returning a guard is the only correct spelling. Wrong is hard.
- Tenet 3 (dispatcher): consolidates the 3 strategies (TwoSlot, Knapsack, Adaptive) into one fraction-driven strategy. Tenet 3 explicitly calls this out as the dispatcher's job.

**Effort**: 1 PR, 1 session (mostly wiring; the substance is in A, B, C).

### E. Other trainers: wire `OffloadCoordinator`

Each trainer that uses `BlockOffloader` today gets migrated to `OffloadCoordinator`. Klein is the proving ground. Then Wan22, Chroma, Flux, Ernie, SD35, SDXL.

**Effort**: 1 PR per trainer, ~0.5 session each. 4 sessions total.

---

## Phase plan + measurement gates

| Phase | Component | Sessions | Gate measurement | Closes gap |
|---|---|---|---|---|
| 1 | A — `GrowOnDemandActivationCache` | 1-2 | Microbench + Klein 9B → push count goes from `0 succeed / N fail` to `N succeed`. Pinned bytes auto-tunes. | Gap 2 (real fix), partially Gap 6 |
| 2 | B — `checkpoint_offload_boundary` | 1-2 | Klein 9B: 5-step run with `--activation-offload`. Step time ≤ baseline + 5%, peak GPU −400 to −800 MB vs baseline. | Gap 2 (full fix) |
| 3 | D — `OffloadCoordinator` skeleton | 1 | Compile; no perf gate. Just the type definitions and BlockGuard. | (foundation) |
| 4 | C — `RingSlabAllocator` | 2-3 | Microbench replicates Klein corruption pattern; ring runs clean. Klein 9B runs without `FLAME_ALLOC_POOL=0` workaround. | Gap 1 |
| 5 | D — `OffloadCoordinator` finish (fraction + 3-case strategy) | 1-2 | Klein 9B fraction sweep: {0.3, 0.5, 0.7, 0.9}. Step time monotone with fraction; peak GPU monotone inversely. | Gap 3 + Gap 4 |
| 6 | E — Trainer migration (Klein first) | 0.5 | Klein 9B 30-step run with Coordinator vs current. Loss bit-equal. | (wiring) |
| 7 | E — Other 7 trainers | 4 | Per-trainer 5-step parity smoke. | Gap 2 across fleet |
| 8 | Polish: HostRamBudget enforcement, telemetry, doc | 1 | Telemetry dump shows shared budget honored. | Gap 6 polish |

**Aggregate**: ~13-18 sessions to ship the full design. Same order of magnitude as the gap analysis estimate (13-23), but with a clearer dependency graph.

## What stays vs what's replaced

| Today | After | Notes |
|---|---|---|
| `flame_core::offload::BlockOffloader` | Same — owned by `OffloadCoordinator` | No API change |
| `flame_core::offload::strategy::{TwoSlot, Knapsack, Adaptive}` | One unified `BlockOffloadStrategy` | Old strategies removable after migration |
| `flame_core::activation_offload::ActivationOffloadPool` | `GrowOnDemandActivationCache` | Old pool deletable after Phase 1 |
| `flame_core::autograd::AutogradContext::checkpoint_offload` | `checkpoint_offload_boundary` | Old call site updates to new signature |
| `flame_core::cuda_alloc_pool` | Stays for non-offload allocs; `RingSlabAllocator` for offload-shaped | Both coexist |
| `eridiffusion-core::training::offload::setup_activation_offload` | Deleted; `OffloadCoordinator::new` is the only setup | Cleaner trainer-side surface |

## Risks + mitigations

1. **Risk**: narrow-scope checkpoint changes loss bit-equivalence. Mitigation: bit-equal smoke gate on Phase 2. We already have that gate from the current Klein 5-step matching pre-Gap-2 baseline.

2. **Risk**: RingSlabAllocator's tape-direction integration breaks under autograd_v2 bridge. Mitigation: design with both v3 and v2 backward paths; gate via feature flag during migration.

3. **Risk**: `BlockGuard` RAII pattern + autograd recording interact in unexpected ways. Mitigation: explicit `Drop` ordering test, single-threaded only (matches our autograd model).

4. **Risk**: GrowOnDemandActivationCache's slab growth heuristic produces churn (alloc/free pinned RAM mid-training). Mitigation: hint API for warmup-time pre-allocation; never shrink mid-training.

## Why this is better than just porting OT

If we transliterated OT to Rust:
- Same Python-style call ordering risks (`before_layer` / `after_layer` could be forgotten or out of order).
- Two allocators not coordinated on host RAM.
- No type-safe direction.
- No FP8.
- Bool-parameter API.

This design gives us OT's correctness AND Rust's safety AND our existing FP8 path AND a single coordinator surface. The flame-core tenets (1: fix the primitive; 2: API makes right easy; 3: dispatcher; 4: measurement; 5: reject wrong-place fixes) all hold.

## Decision point before kickoff

(a) Approve the design as written and start on Phase 1 (`GrowOnDemandActivationCache`)?
(b) Adjust scope first — e.g., do Phase 1+2 only this quarter, defer the ring allocator?
(c) Different direction — pick a subset or substitute components?

Phase 1 is the cheapest test of the design's premise: if `GrowOnDemandActivationCache` lands and Klein's push-count-fail rate drops to 0, the rest of the design is on solid ground.
