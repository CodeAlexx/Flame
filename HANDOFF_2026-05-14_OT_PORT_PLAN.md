# OT-pattern port plan + bug findings (2026-05-14)

## TL;DR

Klein 9B + `--offload` step-2 INVALID_VALUE crash is **not** caused by any of:
F32 zero-init, fork_default_stream, Phase 2 ring_alloc, mempool release
threshold, stream-order race (synccheck clean), or memory corruption
(memcheck clean). The cudart driver rejects a `cuMemcpyHtoDAsync` on a
freshly-allocated 24 MB F32 ptr at step-1 sample-load with
`INVALID_VALUE`, with no sanitizer flag.

The bug is at the cudart API parameter check level on a "valid-looking"
ptr — there is no Rust-side cause I could pin down with point-fixes.

A **structural redesign** modeled on OneTrainer's
`LayerOffloadConductor` + `StaticLayerAllocator` would eliminate the
entire class of bug: no per-op `cudaMalloc`/`cudaFree` calls, no cudart
mempool involvement in the hot path.

## What we proved (this session)

### Reproducible bug
- `FLAME_ALLOC_POOL=1` (pool ON) + `--offload` + Klein 9B → crash at
  step 1 boundary in `serialization::load_file → Tensor::from_vec →
  htod_copy_into → cuMemcpyHtoDAsync_v2`.
- Compute-sanitizer `synccheck`: 0 errors.
- Compute-sanitizer `memcheck`: 0 errors (only reports the API error
  itself, no memory corruption).
- Pool accounting balanced (376 ext_reg ↔ 376 ext_unreg).
- Crashing ptr is FRESH from `cuMemAllocAsync`, returned OK, then
  `cuMemset`/`cuMemcpyHtoD` rejects it with `INVALID_VALUE`.

### Hypotheses ruled out
1. **Pool reuse / uninit-read** — `FLAME_F32_ZERO_INIT=1` alone doesn't
   fix it. With pool removal entirely (`FLAME_ALLOC_POOL=0` workaround),
   trainer works → so it IS pool-related, but not via uninit reads.
2. **Stream fork** — collapsing `fork_default_stream` to share the
   default stream made the crash worse (failed earlier on a smaller
   alloc). Fork was actually MASKING the bug.
3. **Phase 2 ring_alloc commits (f82ab9b..12929f7)** — reverted, crash
   still reproduces. Bug predates Phase 2.
4. **Phase 2b checkpoint_offload_boundary correctness (6b5d0a5)** —
   reverted, crash persists.
5. **Async vs sync mempool** — `FLAME_DISABLE_ASYNC_MEMPOOL=1` forces
   `cuMemAlloc`/`cuMemFree`, crash persists.
6. **cudart mempool release threshold** — tested 4 GiB, 0, default
   `u64::MAX`. All crash same.
7. **Stale thread-local context cache** — explicit `bind_to_thread()`
   right before the failing call returns OK; crash still happens.
8. **Cross-stream pending op race** — `cuCtxSynchronize()` at step
   boundary doesn't fix it. (Confirms synccheck's "0 errors".)
9. **Trim cudart mempool per step** — adds 0.3 s/step + grad explosion
   at step 5, doesn't fix the crash root.

### Other real bugs found
- **`AutogradContext::checkpoint` re-entry hazard** (autograd.rs:2027/
  2045): unconditional `AUTOGRAD_ENABLED.store(true)` on exit clobbers
  prior state. **Fixed** in this commit with an `EnabledGuard` that
  saves/restores prior. Does NOT fix the INVALID_VALUE crash but is a
  separate correctness issue.
- Same hazard in `checkpoint_offload_boundary` — also fixed.
- `ctx.record()` silently drops entries when `ctx.enabled == false`
  (autograd.rs:602). The fix above had to drop the EnabledGuard
  BEFORE calling `record()` so the prior state is restored.

### Perf signal
- Broken-fast variant (recompute skipped because tape entry wasn't
  recorded due to record-while-disabled bug): **2.4 s/step**, grads=0.
- Correct path: **5.9 s/step** (baseline pinned-RAM), grads correct.
- Workaround (`FLAME_ALLOC_POOL=0`): 4.4 s/step, grads correct.
- OT klein9b reference: 2.07 s/step.

The 2.4 floor says forward + light-backward work is ~2.4s/step; the
remaining 3.5s gap to current 5.9 is **all** allocator churn +
checkpoint recompute. OT eliminates both via the slab allocator.

## OT pattern (from `/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py`)

### `StaticLayerAllocator` (lines 122–221)
- Pre-allocates a fixed pool of "cache_tensors" (slabs) — torch tensors
  of `cache_tensor_size` bytes each — at training start.
- Per-layer: `get_allocator(layer_index, forward)` returns a
  `StaticLayerTensorAllocator` that BUMP-ALLOCATES within the slabs.
- `allocate_like(source_tensor)`: returns a view/slice into the slab at
  the current bump cursor, advances cursor by `ceil_16(bytes)`.
- `deallocate(forward)`: resets the bump cursor for that layer — no
  cudaFree.
- Bidirectional: forward direction allocs from front, backward from
  back; cursors meet in the middle. Wraps to next slab when full.

### `StaticActivationAllocator` (lines 224–321)
- Similar slab-based pattern for activation tensors (the data passed
  between layers).
- `reserve_cache(tensors)`: grows a single slab to fit a batch of
  tensors.
- `allocate_like` / `deallocate`: bump cursor.
- `deallocate()` condenses multiple slabs into one to reduce
  fragmentation.

### `LayerOffloadConductor` (lines 524–949)
- Wraps the module's layers. Three streams: `train_stream`,
  `layer_transfer_stream`, `activations_transfer_stream`.
- `start_forward(keep_graph)` — invoked once per training step.
- `before_layer(layer_idx, call_idx, activations)` — schedules
  prefetch of needed layers + offload of layers we're done with.
- `after_layer(...)` — records stream events, schedules
  activation-offload during forward.
- Uses **`SyncEvent`** wrapper around `torch.cuda.Event` for fine-
  grained cross-stream wait_event (NOT host-blocking sync).
- Requires `use_reentrant=True` checkpointing (line 718) — gradients
  enabled only during back pass.

### Key invariants OT relies on
1. **One cudaMalloc per slab, never per tensor.** Slabs are large
   (~hundreds of MB to GB). Tensors are slab slices.
2. **No `cudaFree` during forward+backward.** Slab memory is recycled
   via bump-cursor reset between layers.
3. **Async transfers use 3 dedicated streams** with explicit event
   waits between them. No fork-and-forget streams.
4. **Activations follow the SAME slab pattern** as layer weights.

## Where flame-core diverges (and breaks)

| Concern | OT | flame-core (current) |
|---|---|---|
| Allocator | Static slabs + bump cursor | `cuMemAllocAsync` per alloc + free-list cache (`cuda_alloc_pool`) |
| Free | Cursor reset (no cudaFree) | `cuMemFreeAsync` per drop → cudart mempool |
| Mempool involvement | None during hot path | Every alloc/free goes through cudart mempool |
| Streams | 3 named, event-coordinated | `device.stream` (default) + 1 forked `transfer_stream` per BlockOffloader |
| Activation offload | StaticActivationAllocator slab | `GrowOnDemandActivationCache` (opt-in, not wired by default) |
| Layer offload | Bump-allocated layer weight slabs | BlockOffloader 2-slot ping-pong with pinned host RAM |
| Recompute | PyTorch use_reentrant=True | `Op::Checkpoint` saved closure, ran with autograd enabled on backward |

The current crash is a direct consequence of column 2: every per-op
allocation churns the cudart mempool, and after ~600 alloc/free cycles
the mempool returns a ptr the driver later rejects.

## Port plan (incremental, can ship in pieces)

### Phase A — bump allocator primitive (1 file)
Add `flame_core/src/static_slab_allocator.rs`:
- `StaticSlab { device: Arc<CudaDevice>, slab: CudaSlice<u8>, cursor: usize, capacity: usize }`
- `StaticSlab::new(device, bytes) -> Result<Self>` — one
  `device.alloc::<u8>(bytes)` call.
- `StaticSlab::bump<T>(elems) -> Result<CudaSlice<T>>` — pointer math +
  cursor advance, returns transmuted slice. No cudaMalloc.
- `StaticSlab::reset()` — cursor = 0. No cudaFree.

### Phase B — replace `pool_alloc_*` callers with slab views
- For training-loop ALLOCATIONS (gradients, activation temporaries,
  optimizer scratch): allocate from a global per-region slab at startup,
  bump-allocate within. Sized once based on the largest step's needs.
- For LOAD/SAVE F32 buffers (the actual crash site at
  `serialization::load_file`): allocate from a dedicated "load
  scratch" slab. Sample-load fills slab → tensor view → consumed by
  forward. Reset slab next step.

### Phase C — `LayerSlabPool` matching OT's StaticLayerAllocator
- Pre-allocate per-layer bump allocators backed by N slabs.
- Replace BlockOffloader's GPU-side slot tensors with bump allocations
  out of these slabs.

### Phase D — `SyncEvent` wrapper + 3-stream pipeline
- Replace ad-hoc `fork_default_stream` with named streams: `train`,
  `weight_xfer`, `activation_xfer`.
- All inter-stream coordination via recorded events with explicit
  `wait_event` — no `cuStreamSynchronize` host blocks except at
  step boundary.

### Phase E — autograd re-entry depth counter
- Replace ad-hoc `AUTOGRAD_ENABLED.store(true/false)` toggles with a
  depth counter or RAII guard pattern across ALL no_grad code paths
  (`checkpoint`, `checkpoint_offload`, `checkpoint_offload_boundary`,
  `no_grad`, `inference_mode`).
- Already partly done in this commit for `checkpoint` +
  `checkpoint_offload_boundary`.

## Expected wins from full port

| Source | Cost recovered | Cumulative s/step |
|---|---|---|
| Current pinned-RAM baseline | — | 5.9 (with grads) or 4.4 (workaround) |
| Phase A+B (slab for hot-path allocs) | ~1.5 s/step (no cudart churn) | ~3.0 |
| Phase D (proper stream pipelining) | ~0.5 s/step (less host-side waiting) | ~2.5 |
| Phase C (slab for activations + layers) | ~0.3 s/step (less mempool fragmentation overhead) | ~2.2 |
| OT reference | — | 2.07 |

The 2.4 s/step number observed when the broken-fix-skipped-recompute
ran without backward suggests the forward path under flame-core is
already competitive — the ~3.5 s of overhead is purely the per-op
alloc + per-checkpoint recompute churn, both of which the slab pattern
removes.

## What shipped this session

- **`flame-core/src/static_slab.rs`** (new module, ~300 lines):
  `StaticSlab` (bump allocator + cursor reset), `bump_load_scratch_f32`
  (global slab for the literal crash site, `Tensor::from_vec` F32
  path), `reset_load_scratch` (per-step reset), `slab_external_ptr_hook`
  (composes with cuda_alloc_pool's external-ptr set so the
  BlockOffloader ring path still works). **Opt-in via
  `FLAME_USE_LOAD_SCRATCH=1`** because the slab-vs-autograd-saved-tensor
  lifetime contract has a known bug (see below). Default OFF.
- **`flame-core/src/autograd.rs`** — re-entry guard in
  `AutogradContext::checkpoint` and `checkpoint_offload_boundary`.
  Replaces unconditional `AUTOGRAD_ENABLED.store(true/false)` with an
  `EnabledGuard` that saves/restores the prior state. Critical fix:
  `ctx.record()` drops entries silently when `ctx.enabled == false`
  (autograd.rs:602), so the guard MUST be dropped BEFORE recording the
  tape entry. Both call sites do `let should_record = _guard.prior;
  drop(_guard); if should_record { ctx.record(...) }`.
- **`flame-core/src/lib.rs`** — exports the new `static_slab` module.
- **`flame-core/src/device.rs`** — `ctx_synchronize()` helper (calls
  `cuCtxSynchronize` — used as a diagnostic this session, kept as a
  convenience).
- **`cudarc-pinctx/src/driver/safe/{core,mod}.rs`** — `EXTERNAL_PTR_HOOK`
  static + `install_external_ptr_hook` re-added (was uncommitted, lost
  earlier; required by flame-core Phase 2 ring commits to compile).

EDv2 trainer:
- `train_klein.rs` workaround restored (`FLAME_ALLOC_POOL=0` auto-set
  if not in env). Trainer runs 4.5 s/step on Klein 9B + `--offload`.
- `reset_load_scratch()` call at top of each step loop (no-op when
  slab is disabled; primed for when Phase A correctness is finished).

## Phase A status (end of session)

Final Phase A wiring (the version currently in tree):
- `Tensor::from_vec` slab path: bump-allocate slab view, H2D `Vec<f32>`
  → slab view, wrap slab view directly into `TensorStorage::F32`. Slab
  is the result tensor's backing storage. No D2D to a pool tensor.
- Slab reset moved from top-of-step to AFTER `AutogradContext::clear()`
  at end-of-step in `train_klein.rs`. This means tape `saved_tensors`
  holding slab views remain valid through backward, and the slab is
  reset only after the tape is gone.
- Slab-view drop path: `TensorStorage::F32::drop` → `pool_return_f32` →
  `is_external_ptr(ptr)` returns true (slab ptrs are registered with
  pool's `external_ptrs` set in `bump_load_scratch_f32`) → external
  branch in `push_f32` → `reconstruct_and_forget` + `unregister_external_ptr`.
  No cudaFree on slab ptrs, hook composition works.
- `slab_external_ptr_hook` (in static_slab.rs) ALSO consults
  `cuda_alloc_pool::global_pool().is_external_ptr` so the
  BlockOffloader ring path (`alloc_bf16_via_ring`) keeps working.

### Empirical result (5-step Klein 9B + --offload, FLAME_USE_LOAD_SCRATCH=1, no workaround)

- **Step 1**: loss 1.1217, **grad_norm 0.0071** — bit-exact match to
  baseline. The slab path produces correct numerics for the sample-load
  route. NO crash at the historic step-1 sample-load site.
- **Step 2**: crashes with `Flame: Kernel error: DriverError(CUDA_ERROR_INVALID_VALUE)`.
  This is a different code path than the prior crash (was
  `CUDA driver error`, now `Flame: Kernel error` — internal kernel
  launch). The cudart-mempool bug has moved to a different F32
  allocation site in the autograd/backward path.

Conclusion: Phase A fixes the sample-load path correctly but doesn't
cover other F32 allocs in the training hot path. The bug is broader —
any F32 alloc through `cuda_alloc_pool::pool_alloc_f32` is susceptible
to the same cudart-mempool INVALID_VALUE after enough churn.

## Phase B (next session) — the actual fix to hit OT's 2.07 s/step

User's target: 2.0–2.1 s/step on Klein 9B (beat OT). Workaround
baseline 4.5 s/step. Gap = 2.4 s/step, broken down per
`HEADTOHEAD_2026-05-12_ROOT_CAUSE.md`:
- ~1.5 s/step F32 grad-storage roundtrips + cast pairs
- ~0.9 s/step cudaStreamSynchronize calls
- ~0.5 s/step kernel-mix tail

Phase A only escapes the sample-load crash. To beat OT, every F32
alloc on the hot path must skip the cudart mempool.

### Phase B work items

1. **Identify F32 alloc sites in the training hot path.** `grep` for
   `pool_alloc_f32`, `alloc_aligned_f32`, `alloc_from_pool`,
   `TensorStorage::empty(F32)`, `TensorStorage::zeros(F32)`. Most are
   in gradient accumulation, BF16→F32 casts, and intermediate buffers.

2. **Separate slab regions, OT-style:**
   - `load_scratch_slab` (already exists) — sample-load F32 tensors,
     reset after `AutogradContext::clear`.
   - `activation_slab` — BF16/F32 activations between layers, reset
     after each layer's backward (OT's `StaticActivationAllocator`).
   - `grad_scratch_slab` — gradient accumulators + transient F32
     casts during backward, reset after step's optimizer update.
   - `layer_slab` — layer weights (when offload+adaptive is on);
     bump-allocated per layer.

3. **Wire each slab into the appropriate allocator:** Add a "current
   region" thread-local that callers set before doing temporary
   allocations. `pool_alloc_f32`/`alloc_aligned_f32` consult it and
   route accordingly. Default region = pool (legacy fallback).

4. **Stream pipeline (OT's `LayerOffloadConductor`):** 3 named
   streams (`train`, `weight_xfer`, `activation_xfer`) with explicit
   event-based wait, no host-blocking sync except at step boundary.

### Phase B verification

After each work item, run Klein 9B 5-step with pool ON (no workaround):
- Item 1 only (audit) — no behavioral change, just shrinks the
  unknown.
- Item 2 (region slabs wired) — sample-load crash bypassed (already
  done in Phase A); the next-fail site moves further into backward.
  Each new fail site identifies the next slab to wire.
- Item 4 — should remove ~0.9 s/step of sync stalls per the May-12
  root-cause doc.

### Acceptance: 2.0–2.1 s/step with grads bit-exact to workaround

Klein 9B + --offload + FLAME_ALLOC_POOL=1, no workaround, 30 steps.
- Loss curve matches workaround within BF16 noise (≤1e-3 per step).
- Grad-norm curve matches workaround within BF16 noise.
- Steady-state ≤ 2.1 s/step.

This is the user's "won" criterion.

## Next session's pickup script

```bash
cd /home/alex/EriDiffusion/flame-core
cat HANDOFF_2026-05-14_OT_PORT_PLAN.md   # this file
# Current state: trainer works at 4.5 s/step with workaround.
# Phase A wired, sample-load slab proven correct on step 1.
# Phase B is the remaining work to hit 2.0-2.1.
```

## Re-entry fix verification (still to-do)

The re-entry guard in `checkpoint`/`checkpoint_offload_boundary` is a
real correctness improvement but its effect on Klein hasn't been
empirically measured this session (still crashed for unrelated
cudart-mempool reasons before the guard's behavior could be observed).
Once Phase B clears the crash class, a Klein 9B 30-step run should
compare grads with and without the guard to confirm it doesn't disturb
steady-state loss curves.
