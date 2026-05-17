# Codex Review - OT Static-Slab Redesign (2026-05-15)

## Decision

Conditional approval for the direction, not for the R2/R3 wiring as written.

The static-slab primitive is the right structural move: one long-lived CUDA allocation with bump allocation removes the allocator free/reuse path that is implicated in the current crash. But replacing `pool_alloc_u16` / `pool_alloc_f32` globally and resetting after `AutogradContext::clear()` is not safe yet. The design needs explicit lifetime tracking, range-aware external pointer handling, and narrower routing before implementation.

Recommended kickoff shape:

1. Land R1 only after adding a range-based external memory registry and an outstanding-allocation reset guard.
2. Land R2 behind an explicit transient-allocation scope, not a global `pool_alloc_*` replacement.
3. Defer R3 cache deletion until Klein, Z-Image, and at least one inference/phase-boundary workload pass with the new mode.

## Reviewed Inputs

- `HANDOFF_2026-05-15_OT_STATIC_SLAB_REDESIGN.md`
- `TENETS.md`
- `docs/SPEED_CONTRACT.md`
- `docs/OFFLOAD_NEXT_GEN_DESIGN.md`
- `/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py:122-321`
- `/tmp/cuda_bug_websearch_2026-05-15.md`
- `/tmp/regression_audit_2026-05-15.md`
- `src/cuda_alloc_pool.rs`
- `src/tensor_storage.rs`
- `src/autograd.rs`
- `src/ring_alloc/pool_adapter.rs`
- `src/offload/mod.rs`
- `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs`
- `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-core/src/models/klein.rs`

## Blocking Findings

### P0: `reset_global()` after `AutogradContext::clear()` is not a safe lifetime boundary

The handoff proposes resetting the slab in `train_klein.rs` immediately after `AutogradContext::clear()`.

That does not prove all slab-backed tensors are dead. `AutogradContext::clear()` only clears the tape and checkpoint closures:

- `src/autograd.rs:607-610`
- `src/autograd.rs:1173-1182`

It does not drop step-local tensors such as `pred`, `pred_f32`, `target_f32`, `raw_loss`, `loss`, `noise`, `noisy`, `target`, `timestep`, validation temporaries, or sampler temporaries. In `train_klein.rs`, the proposed reset point would sit around the current clear call:

- `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs:1420-1438`

Rust drops locals at the end of their scope, not when they are last mentioned. Resetting the slab before the loop body ends can make the next allocation alias a still-live tensor.

Required change:

- Add an outstanding-allocation counter to the slab allocator.
- Make `reset_global()` return an error or panic in strict mode if any slab allocation is still live.
- Restructure trainer code so transient tensors live inside an explicit step scope:

```rust
let metrics = flame_core::static_slab_v2::with_step_slab(|| -> anyhow::Result<StepMetrics> {
    // forward, loss, backward, optimizer step
    AutogradContext::clear();
    Ok(metrics)
})?;
// with_step_slab resets after closure locals have dropped.
```

Do not rely on convention. Make reset impossible while live allocations exist.

### P0: `external_ptrs` is exact-pointer refcounting, not a slab range registry

The handoff says slab ranges should be registered with `cuda_alloc_pool::register_external_ptr`, but the current API records only exact pointer values:

- `src/cuda_alloc_pool.rs:485-505`
- `src/cuda_alloc_pool.rs:1104-1114`

This is insufficient for slab memory. Any `CudaSlice` whose pointer is an offset inside the slab will miss the hook and call `cudaFree` on a mid-slab address. The repo already hit this class for ring-backed F32; `RingPoolAdapter::alloc_f32` is deliberately disabled because derived/offset F32 slices can escape exact-pointer registration.

The reverted Phase A implementation had the right shape here: it registered slab ranges and made the hook check `start <= ptr < end`.

Required change:

- Introduce one central external allocation registry that supports ranges:

```rust
struct ExternalRange {
    start: u64,
    end: u64,
    device_key: usize,
    owner: ExternalOwner,
}
```

- Install one composable cudarc hook that checks the range registry and the existing exact-pointer refcount map.
- Register the whole slab range at slab creation for drop-hook safety.
- Separately track per-allocation exact pointers/refcounts for live-allocation accounting and `pool_return_*` cleanup.

Do not try to model a range by calling `register_external_ptr` for every allocation start. That only catches exact starts and does not protect offset views.

### P0: Global `pool_alloc_*` dispatch will capture persistent tensors

The R2 plan routes `pool_alloc_u16` and `pool_alloc_f32` to the slab whenever `FLAME_USE_STATIC_SLAB=1`.

That is too broad. Many persistent tensors are allocated through these APIs:

- `TensorStorage::empty` uses `pool_alloc_u16` for BF16 and `alloc_aligned_f32` for F32: `src/tensor_storage.rs:323-345`
- `TensorStorage::zeros` uses the same route: `src/tensor_storage.rs:358-399`
- `alloc_aligned_f32` routes to `pool_alloc_f32`: `src/cuda_memory_alignment.rs:155-157`
- casts allocate through the pool: `src/tensor.rs:972-1003`, `src/tensor.rs:1061-1063`

The current Klein trainer sets allocator-related env before model load:

- `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs:380-388`

If slab dispatch is globally enabled at that time, model weights, LoRA params, optimizer state, prompt/cache tensors, or sample setup tensors can become slab-backed. Resetting the slab per step would then corrupt persistent state.

Required change:

- Do not make static slab the unconditional backend for `pool_alloc_*`.
- Add an explicit transient allocation scope or allocator mode:

```rust
static_slab_v2::with_transient_scope(|| {
    // Only allocations inside this scope use the step slab.
})
```

- Keep persistent allocations on direct cudart or a non-resetting allocator.
- If global dispatch is still desired later, it must classify allocation lifetime first. Without lifetime classification, R2 is unsafe.

### P0: F32 zero-initialization semantics would change

Current default F32 behavior is zero-initialized when F32 free-list caching is off:

- `src/cuda_alloc_pool.rs:1161-1198`

`TensorStorage::empty` also currently documents that the F32/F16/I32 path is effectively zeroed because `alloc_aligned_f32` goes through a zeroing path:

- `src/tensor_storage.rs:310-318`

The proposed `alloc_f32` API is a bump allocation and appears uninitialized. That silently changes behavior for F32, F16, and I32 storage paths that reuse `alloc_aligned_f32`.

Required change:

- Either preserve zero-init for `pool_alloc_f32` slab dispatch, or split the allocator API into explicit zeroed and uninitialized variants.
- Update `TensorStorage::empty` only after auditing every F32 caller that depends on current zero-init behavior.
- Add a regression test proving `Tensor::zeros(..., DType::F32)` and F32 opt-out parity still hold under slab mode.

## High-Risk Design Gaps

### P1: `clear_pool_cache()` cannot simply become a no-op globally

The handoff proposes making `clear_pool_cache` a no-op in slab mode. That is only safe for slab-owned transient memory. The wider tree uses `clear_pool_cache()` as a phase-boundary memory release primitive in training, model setup, sampling, and inference.

Examples from the current tree:

- `src/autograd.rs:3044-3045`
- `src/autograd.rs:3196-3197`
- `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-core/src/models/klein.rs:484`
- `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs:526`

There are many more calls across `EriDiffusion-v2` and `inference-flame`.

Required change:

- Keep `clear_pool_cache()` semantics for legacy free-list/direct fallback memory.
- Do not use `clear_pool_cache()` to reset static slabs.
- Add separate APIs:

```rust
static_slab_v2::reset_step_slab()    // fails if live allocations exist
static_slab_v2::release_all_slabs()  // explicit phase-boundary free, not hot path
```

Inference should remain on the existing allocator path until it has dedicated tests.

### P1: `inference-flame` makes the no-op risk concrete

Follow-up pass over `/home/alex/EriDiffusion/inference-flame` found this is not an abstract concern. `inference-flame` is a separate crate over the same `flame-core`, with many CLI binaries plus `inference_ui` workers. It uses `clear_pool_cache()` and `trim_cuda_mempool(0)` aggressively at model/phase boundaries; a repo-wide search found 174 such call sites across bins, UI workers, and model code.

Examples:

- `inference-flame/inference_ui/src/worker/flux.rs`
- `inference-flame/inference_ui/src/worker/qwenimage.rs`
- `inference-flame/inference_ui/src/worker/klein.rs`
- `inference-flame/src/bin/magihuman_infer.rs`
- `inference-flame/src/bin/helios_infer.rs`
- `inference-flame/src/models/ltx2_model.rs`
- `inference-flame/src/models/flux1_dit.rs`

Implication: static slab mode must be opt-in for a scoped training lifetime at first. A global env such as `FLAME_USE_STATIC_SLAB=1` that changes all `pool_alloc_*` and `clear_pool_cache()` behavior would risk breaking inference workloads that currently depend on explicit phase-boundary memory release.

### P1: `inference-flame` Turbo/VMM is a useful precedent

`inference-flame/src/turbo` already implements a stronger version of the ownership pattern the slab redesign needs:

- `TurboBlockLoader` maps block regions through a VMM `SlabAllocator`.
- It publishes non-owning BF16 tensors through `Tensor::from_bf16_device_ptr_non_owning`.
- The tensors use `TensorStorage::BF16View`, which explicitly does not free on drop.
- `TurboBlock` owns an `Arc<ResidentHandle>` so the mapped region outlives all view tensors.
- `ResidentHandle::Drop` records an event on the consumer stream before decrementing the region refcount.
- Eviction waits for recorded events before unmapping.

Relevant files:

- `inference-flame/src/turbo/loader.rs`
- `inference-flame/src/turbo/block.rs`
- `inference-flame/src/turbo/vmm/allocator.rs`
- `inference-flame/src/turbo/vmm/handle.rs`
- `inference-flame/src/turbo/vmm/eviction.rs`
- `flame-core/src/tensor_storage.rs:203` (`BF16View`)
- `flame-core/src/tensor.rs:1455` (`from_bf16_device_ptr_non_owning`)

Relevant tests:

- `inference-flame/tests/turbo_tensor_over_vmm.rs`
- `inference-flame/tests/turbo_reader_outlives_prefetch.rs`

Recommendation: use the same conceptual shape for static slab validation. The slab allocator should not depend only on cudarc drop hooks for lifetime. It should have explicit owner/lease state, non-owning views where appropriate, live refcounts, and reset/eviction barriers that fail loudly when views are still live.

### P1: The slab needs a hard live-allocation invariant, not just trap diagnostics

The handoff notes that stale tensors after reset would corrupt data and that the trap infrastructure may catch it. That should be a runtime invariant, not a post-failure diagnostic.

Required change:

- Increment a live counter on every slab allocation.
- Decrement only when `pool_return_*` or direct hook cleanup proves the slab slice is no longer live.
- `reset()` must fail if `live_count != 0`.
- During validation, default to strict mode. A stale live tensor should fail at the reset call, not later inside `Tensor::cat`.

### P1: Hook installation must be unified

`cudarc::driver::install_external_ptr_hook` installs a process-wide hook. Current users include:

- `cuda_alloc_pool::install_miss_allocator`: `src/cuda_alloc_pool.rs:1094-1100`
- `BlockOffloader::ensure_ring`: `src/offload/mod.rs:48-55`, `src/offload/mod.rs:1009`

The slab path must not install a hook that forgets ring-owned pointers, and ring setup must not overwrite slab range handling.

Required change:

- Install a single flame-core hook once.
- Make that hook consult a unified registry that knows both slab ranges and ring/external exact pointers.

### P1: Default slab sizing and initialization timing can cause OOM

The proposed defaults are 4 GiB BF16 plus 4 GiB F32. On a 24 GB card, allocating both before model weights are dropped/offloaded or while sample setup tensors are still resident can OOM.

Required change:

- Materialize slabs lazily on first transient-scope allocation, not at process start.
- Initialize slab mode after persistent model setup/offload, not before model load.
- Add telemetry for peak requested BF16/F32 slab bytes before picking defaults.
- Fail with a message that includes requested bytes, capacity, dtype, current cursor, and the env override.

### P1: Multi-device and thread safety need to be part of R1

The global pool keys free lists by device `Arc` identity. A global static slab must do the same. Tests and some tools create fresh `CudaDevice::new(0)` handles, and a slab allocated on one device handle must not satisfy another handle's allocation.

Required change:

- Store global slab allocators in a per-device map.
- Protect each slab cursor with a `Mutex`.
- Keep `StaticSlabAllocator` itself simple, but make the global dispatch path thread-safe.

## Notes On The Root Cause

The static-slab direction is still reasonable even though the web-search note argues against a CUDA driver bug. The strongest local evidence is a userspace ownership failure: a pointer returns to the pool/free path while another tensor can still later drop the same pointer. A slab removes `cudaFree` from the hot path, but it does not automatically restore Rust ownership. Without a reset guard, it merely changes the failure mode from `CUDA_ERROR_INVALID_VALUE` to silent aliasing.

So the implementation goal should be:

- No hot-path `cudaFree`.
- No reset while slab-backed tensors are live.
- No slab backing for persistent tensors.
- No exact-pointer-only hook for range-owned memory.

## Required Test Gates

Add these before the Klein/Z-Image trainer smokes:

1. `static_slab_reset_with_live_allocation_panics_or_errors`
   - Allocate one BF16 tensor from the slab.
   - Call reset while it is live.
   - Assert strict failure.

2. `static_slab_reset_after_drop_succeeds`
   - Allocate and drop BF16/F32 slab tensors.
   - Reset.
   - Assert cursor rewinds and live count is zero.

3. `static_slab_hook_covers_offset_ptr`
   - Register a slab range.
   - Synthesize a `CudaSlice` at a non-base offset.
   - Drop it.
   - Assert no `cudaFree` is attempted for the offset.

4. `static_slab_f32_zero_semantics`
   - With slab mode enabled, `Tensor::zeros(..., DType::F32)` must read back zeros.
   - Cover F16/I32 if they still reuse F32 storage.

5. `static_slab_persistent_alloc_not_in_step_slab`
   - Allocate a tensor outside the transient scope.
   - Enter/reset a transient scope.
   - Assert the outside tensor remains valid and its pointer was not slab-owned.

6. `static_slab_multi_device_isolation`
   - If the test environment allows multiple device handles, ensure allocations from different `Arc<CudaDevice>` values do not share a slab entry incorrectly.

7. Existing pool tests
   - Keep `pool_f32_opt_out_*` and `ring_pool_adapter_smoke` green.
   - The new hook must not regress ring-owned BF16 behavior.

Trainer gates after unit tests:

1. Klein 9B, no validation/sample, 15 steps:
   - `FLAME_ALLOC_POOL=1`
   - static slab enabled only inside transient scope
   - strict reset guard on
   - no trap fire

2. Klein 9B with validation/sample enabled:
   - catches tensors that survive past the nominal training-step boundary.

3. Z-Image 50 steps:
   - same parity and speed check as the handoff.

4. One inference/phase-boundary workload:
   - proves `clear_pool_cache()` and phase memory release still work outside trainer transient scopes.

## Suggested Revised Phase Plan

### R1a - External Memory Registry

Create a central registry used by the cudarc drop hook:

- range entries for slab-owned memory
- exact-pointer refcounts for ring/pool external allocations
- one installed hook
- tests for offset pointers and ring compatibility

### R1b - StaticSlabAllocator Primitive

Implement the allocator with:

- per-device global map
- BF16/F32 slabs or typed slab groups
- 16-byte alignment
- live allocation counter
- strict reset guard
- zero/uninit allocation distinction
- clear overflow errors

### R2a - Transient Scope Dispatch

Add a scope-local or thread-local marker so only allocations inside an explicit step scope use the slab. Leave persistent allocations untouched.

### R2b - Klein Wiring

Restructure `train_klein` so the slab reset happens after all transient tensors have dropped. Do not set `FLAME_USE_STATIC_SLAB=1` globally before model load.

### R2c - Smoke And Parity

Run the Klein/Z-Image gates. Keep diagnostic traps enabled during this phase.

### R3 - Default Policy

Only after the above passes:

- consider making transient slab mode default for training
- keep inference default unchanged
- keep direct fallback and explicit `release_all_slabs()`
- postpone deleting free-list code until phase-boundary/inference use cases have a replacement

## Bottom Line

Static slab is the right allocator family for this bug class, but the handoff currently under-specifies ownership. The two non-negotiable changes are:

1. `reset()` must be guarded by a live-allocation invariant.
2. slab-owned memory must be represented as ranges, not exact external pointers.

Without those, the redesign can pass a short smoke test and still silently corrupt training when a tensor crosses the reset boundary.
