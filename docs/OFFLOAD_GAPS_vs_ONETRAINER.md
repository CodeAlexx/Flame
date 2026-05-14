# flame-core offload — gap analysis vs OneTrainer

**Date**: 2026-05-13
**Sources read**:
- OT: `/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py` (949 lines, complete)
- OT: `/home/alex/OneTrainer/docs/RamOffloading.md`
- flame-core: `src/offload/` (4623 lines), `src/activation_offload.rs` (979 lines), `src/cuda_alloc_pool.rs`, `src/autograd.rs:69-100,2008-2150`
- flame-core: `crates/eridiffusion-core/src/training/offload.rs` (wires the activation pool, called by `train_qwenimage` only)

**Scope per tenets**: target medium models that almost fit 24 GB (Klein 9B, Wan 14B+14B-LoRA, future similar). FlexTensor handles the truly large. **Fixes live in flame-core primitives**, not per-trainer.

---

## What OT actually does

Three coordinated CUDA streams:
- `train_stream` (default compute stream)
- `layer_transfer_stream` (H2D/D2H for layer weights)
- `activations_transfer_stream` (H2D/D2H for saved activations)

Five primitives in OT, ranked by what we have vs miss:

| # | OT primitive | OT location | What we have |
|---|---|---|---|
| 1 | `StaticLayerAllocator` — ring buffer over slab tensors | `LayerOffloadConductor.py:122-222` | ❌ `cuda_alloc_pool` (PyTorch-style bucket/free-list, NOT ring buffer); just had to be disabled for Klein 9B+offload |
| 2 | `StaticActivationAllocator` — grow-on-demand pinned slab for activations | `LayerOffloadConductor.py:224-321` | ✅ `ActivationOffloadPool` (slots + pinned host buffers + transfer stream + done/ready events + FP8); shape differs (fixed slots) but functionally equivalent |
| 3 | `SyncEvent` — wraps CUDA events for cross-stream wait | `LayerOffloadConductor.py:323-373` | ✅ `flame_core::offload::*` uses cudarc `CudaEvent` and `stream_wait_event` directly; same semantics |
| 4 | `LayerOffloadStrategy` — per-position load/offload lists with 3 transition cases (fwd-bwd, fwd-fwd, bwd-fwd) | `LayerOffloadConductor.py:376-522` | ⚠️ `BlockOffloader` prefetches the literal next block (slot ping-pong); has TwoSlot/Knapsack/Adaptive strategies but none compute the multi-layer subset OT does |
| 5 | `LayerOffloadConductor` — orchestrates 1+2+3+4, `before_layer`/`after_layer` hooks | `LayerOffloadConductor.py:524-949` | ⚠️ `BlockOffloader` + `OffloadManager` orchestrate weight blocks only; activations live in a parallel `ActivationOffloadPool` invoked via `AutogradContext::checkpoint_offload`. Not unified. |

Plus three control knobs:

| Knob | OT default/effect | flame-core equivalent |
|---|---|---|
| `layer_offload_fraction: float ∈ [0,1]` | OT: "fraction of layers offloaded to RAM at any time" | ❌ no per-fraction knob; `FLAME_OFFLOAD_ADAPTIVE=1` is binary-ish |
| `enable_activation_offloading: bool` | OT: separate from layer offload | ⚠️ implicit: setting up the pool + calling `checkpoint_offload` = on; nothing = off |
| `enable_async_offloading: bool` | OT: async = use the transfer streams, sync = inline copy | ✅ implicit: BlockOffloader is always async; no sync fallback |

---

## Gap-by-gap (ranked by value for "medium models that almost fit")

### Gap 1 — **No ring buffer allocator** (Phase 1 + 2a SHIPPED 2026-05-14, full closure pending 5-step smoke + 30-step gate)

**Phase 1 SHIPPED 2026-05-14**: `flame_core::ring_alloc` — bidirectional ring allocator with direction-typed handles. 19 tests pass. No consumer wiring.

**Phase 2a SHIPPED 2026-05-14**: `flame_core::ring_alloc::pool_adapter` + `cuda_alloc_pool::PoolMissAllocator` trait. Opt-in routing of `cuda_alloc_pool` cache-MISS allocations through a shared ring (`KLEIN_POOL_RING=1` on the Klein trainer). 3 GPU smoke tests. Klein 5-step trainer smoke gate pending.

**Phase 2b pending**: Klein 9B 30-step run, removal of `FLAME_ALLOC_POOL=0` auto-disable, loss-curve and steady-state s/step gate.

**OT mechanism**: `StaticLayerAllocator` pre-allocates `num_cache_tensors` slab tensors of `cache_tensor_size` each (heuristic: `max(target/N, max_tensor_bytes×2, ≥10% headroom)`, capped at 10 slabs). Each layer requests a sub-allocation via `StaticLayerTensorAllocator`. Allocations move `allocation_end` forward (forward pass) or `allocation_start` backward (backward pass). When the cursor crosses a slab boundary, jumps to the next slab; wraps cyclically. Bidirectional → forward and backward never fragment.

**flame-core today**: `cuda_alloc_pool.rs` is a power-of-2 bucketed free list (PyTorch CUDACachingAllocator clone). It corrupts under BlockOffloader+checkpoint replay on Klein 9B — had to disable with `FLAME_ALLOC_POOL=0` this session (`EDv2@4511140`). Phase 2a routes cache MISSES through `flame_core::ring_alloc::RingAllocator` via the `PoolMissAllocator` trait while preserving the pool's bucket-cache hit path. Bidirectional invariant currently unused (all routes are forward-only); Phase 2b polish wires autograd direction.

**Where the fix lives**: `flame_core::cuda_alloc_pool` (replace) or new `flame_core::ring_alloc` (parallel). API surface unchanged: `pool_alloc_f32` / `pool_return_f32` callers don't move. Internals: slab-list with `allocation_start` / `allocation_end` cursors, bidirectional advance.

**Measurement target** (per SPEED_CONTRACT Clause 1+5):
- Microbench: `tests/cuda_alloc_pool_microbench.rs` (new). Compare bucket-allocator vs ring-allocator on a synthetic alternating fwd/bwd pattern with rank-mixed sizes. Target: ring path matches bucket path in fragmentation-free regime; ring path **doesn't corrupt** under the BlockOffloader replay pattern that broke the current pool.
- Real-trainer: Klein 9B 30-step run without `FLAME_ALLOC_POOL=0` workaround → should run clean.
- Wall-time target: at most +1% step time vs current (no-pool) baseline; ideal: −5-10% by avoiding the cudaMalloc/cudaFree churn the no-pool path forces.

**Estimated effort**: 4-7 focused sessions. Ring buffer + bidirectional cursor is the design exercise; integration with `Tensor::drop` is mechanical. Includes a parity microbench and the Klein no-workaround validation.

### Gap 2 — **Activation offload not wired in 13 of 14 trainers** (HIGH for the named use case)

**OT mechanism**: Every block forward calls `conductor.before_layer(i, call_index, activations)` and `conductor.after_layer(i, call_index, activations)`. `after_layer` during forward saves activations to the pinned-RAM activation pool via `_schedule_activations_to_device(activations, temp_device, ...)`. `before_layer` during backward reloads them via the activation transfer stream + wait event.

**flame-core today**: `ActivationOffloadPool` is shipped, sophisticated (FP8 compression option, LIFO free stack matching backward order, per-slot done/ready events). `AutogradContext::checkpoint_offload` wraps it. `crates/eridiffusion-core/src/training/offload.rs::setup_activation_offload` is the convenience helper. **But only `train_qwenimage.rs:582` uses it.** Klein, Z-Image, Chroma, Flux, etc. don't.

**Where the fix lives**: per-trainer wiring (NOT a flame-core primitive change). Each trainer that uses BlockOffloader for weights also needs to call `setup_activation_offload` and switch its block forward from `AutogradContext::checkpoint` to `AutogradContext::checkpoint_offload`.

**Measurement target**:
- Klein 9B step time: ~4.4 s/step today (`EDv2@4511140` baseline). Target with activation offload: **−400 to −800 ms/step** (matches the activation memory delta freed by offloading saved tensors to pinned RAM; lets us either go batch=2 or fit a larger seq_len).
- Memory: peak GPU should drop by ~(num_blocks × max_block_activation_bytes); on Klein 9B that's hundreds of MB.

**Estimated effort**: ~1 session per trainer to wire + smoke. Reasonable to batch with the existing `--use-autograd-v2` wiring pattern. Total: 6-9 sessions to cover the trainers that actually offload weights (Klein, Wan22, Chroma, Flux, Ernie, SD35, SDXL — Z-Image fits resident at typical configs).

**Tenet check**: Tenet 5 ("Reject wrong-place fixes") doesn't apply here — the per-trainer wiring is exactly trainer-level policy (Tenet 1 corollary). The primitive (`ActivationOffloadPool`) already lives in flame-core where it should.

### Gap 3 — **No `layer_offload_fraction` knob** (MEDIUM)

**OT mechanism**: User sets `config.layer_offload_fraction ∈ [0,1]`. `LayerOffloadStrategy` precomputes 3 lists per layer position: `forward_backward_loaded_layers[i]`, `forward_forward_loaded_layers[i]`, `backward_forward_loaded_layers[i]`. Each list is "layers that should be resident on GPU when we're about to execute layer i, going to direction X next." Computed by `__get_layers_below` walking from `i` forward/backward (cyclic or not) until the cumulative byte budget hits `target_loaded_bytes = total_bytes × (1.0 − layer_offload_fraction)`.

**flame-core today**: `BlockOffloader::set_strategy(Adaptive::new())` opts into a growth/shrink-by-VRAM-headroom strategy. There's no "load N% of layers, offload rest" mode. Closest analog: `Knapsack::with_budget(byte_budget)` selects a value-per-byte subset within a byte cap, but it's static.

**Where the fix lives**: `flame_core::offload::strategy::Fraction` (new). API: `Fraction::new(layer_offload_fraction: f32)`. Internal: same 3-list precompute as OT. Wired in `BlockOffloader::set_strategy`.

**Measurement target**:
- 9B model on 24 GB: fraction=0.0 means everything resident (OOM today). fraction=0.5 means half resident. fraction=1.0 means everything offloaded. Validate by sweeping fraction ∈ {0.3, 0.5, 0.7, 0.9} and recording peak GPU + step time. Sweet spot is where step time stops dropping (more H2D doesn't help further).
- Reference: OT's docs use a 9-layer example with fraction=0.33 → 3 layers resident at any time.

**Estimated effort**: 2-3 sessions. The 3-list precompute is the design work; the rest is wiring.

### Gap 4 — **Strategy doesn't compute multi-block resident subsets** (MEDIUM, related to Gap 3)

**OT mechanism**: `LayerOffloadStrategy.get_layers_to_load(i, is_forward, ...)` returns a multi-block list to bring resident before executing block i. `get_layers_to_offload(i, ...)` returns the multi-block list to send to RAM. Walks adjacency in execution direction so the next-block prefetch always overlaps current-block compute.

**flame-core today**: `BlockOffloader::prefetch_block(i+1)` brings one block forward, evicting via slot ping-pong. Strategies (`TwoSlot`, `Knapsack`, `Adaptive`) decide eviction policy but the per-call API still moves one block at a time.

**Where the fix lives**: extend the `Strategy` trait with `next_resident_set(current_block, direction) -> Vec<usize>` and `next_offload_set(...)`. Default impl preserves current behavior (single-block prefetch). `Fraction` strategy from Gap 3 implements the multi-block subset.

**Measurement target**: only meaningful in conjunction with Gap 3. Same sweep.

**Estimated effort**: 1-2 sessions on top of Gap 3.

### Gap 5 — **Fused back pass** (LoRA-IRRELEVANT — skip for now)

**OT mechanism**: For full fine-tuning, optimizer step runs immediately after each layer's gradient calc, while the weight is still in VRAM. Then the weight is offloaded with grad=None and the next backward block proceeds. Saves a full H2D round-trip per parameter.

**flame-core today**: backward → `param.set_grad` → `opt.step(&params)` → `opt.zero_grad(&params)`. Optimizer step is one phase after backward completes.

**Why skip**: LoRA params are tiny and always resident. The fused back pass payoff is for full fine-tuning where the optimizer must touch the same multi-GB weights that just got their grads. Not in the "medium models that almost fit" wheelhouse.

**Revisit if**: we move to full fine-tuning workloads. Estimated effort then: 3-5 sessions (touches optimizer API + backward sequencing).

### Gap 6 — **Activations + weights share a host RAM budget but currently fight for it** (LOW)

**OT mechanism**: separate allocators (`StaticLayerAllocator` for weights, `StaticActivationAllocator` for activations) but they share the host's pinned RAM. OT doesn't enforce a joint cap — relies on user knobs being sane.

**flame-core today**: `BlockOffloader::pinned_bytes()` reports its budget. `ActivationOffloadPool::host_bytes()` reports its budget. **No joint cap; nothing prevents both from claiming all available RAM.**

**Where the fix lives**: a small `HostRamBudget` coordinator in `flame_core::offload`. Both pools consult it at allocation time.

**Estimated effort**: 1-2 sessions. Low impact in the current single-trainer-per-process setup; matters if/when multi-trainer batching happens.

---

## Cross-cutting: does our work obey the contract?

Per `flame-core/docs/SPEED_CONTRACT.md` Clause 1 (no host stall) and Clause 5 (bandwidth-bound H2D):

- ✅ All proposed fixes use the same async-on-transfer-stream + GPU-event-wait pattern flame-core already uses. No new `cudaStreamSynchronize` in any of them.
- ✅ All fixes are flame-core primitives (Tenet 1). Gap 2 is the only trainer-level work and it's wiring, not workaround.
- ✅ All fixes have measurement targets (Tenet 4) — microbench or trainer step-time delta.
- ⚠️ No PR ships without the named measurement landing first.

## Aggregate effort to close ALL gaps (Gaps 1-4 + 6)

| Gap | Sessions |
|---|---|
| 1. Ring buffer allocator | 4-7 |
| 2. Activation offload trainer wiring (8 trainers) | 6-9 |
| 3. layer_offload_fraction strategy | 2-3 |
| 4. Strategy multi-block subset | 1-2 (folded into Gap 3) |
| 6. Host RAM budget coordinator | 1-2 |
| **Total** | **~13-23 sessions** |

Lower bound assumes things go smoothly. Per Tenet 4, each gap closure needs a measurement landing alongside, so this is realistic — not padded.

## Recommended order

1. **Gap 2 first.** Highest user-visible payoff, lowest risk. Activation offload trainer wiring lets Klein 9B and others use the existing ActivationOffloadPool that's already battle-tested in `train_qwenimage`. Quick wins. ~6 sessions.

2. **Gap 3 second.** `layer_offload_fraction` is the OT knob users actually tune. Most of the win on 24 GB cards is here. 2-3 sessions.

3. **Gap 1 third.** Ring buffer allocator is the deepest and biggest. Closing it fixes the corruption permanently, but it's a flame-core primitive rewrite — only attempt after Gap 2+3 have shipped and proven the activation-offload path. 4-7 sessions.

4. **Gap 4 folded into Gap 3.**

5. **Gap 6 last.** Polish, only matters at scale.

6. **Gap 5 (fused back pass) deferred indefinitely until full-fine-tune lands.**

## Decision points before any of this starts

- (a) Does the user want all 13-23 sessions, or a subset? E.g., "Gap 2 only" is a 6-session commit; "Gap 2+3" is 8-12.
- (b) Klein 9B is the natural proving-ground per memory `project_v2_klein9b_proving_ground`. All measurement targets should run there first.
- (c) `train_qwenimage` is the reference for Gap 2 wiring — read its setup_activation_offload call before wiring others.
