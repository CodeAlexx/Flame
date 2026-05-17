# HANDOFF — OT static-slab redesign (2026-05-15, revised post-Codex)

**Status**: Codex review absorbed (`HANDOFF_2026-05-15_OT_STATIC_SLAB_REDESIGN_CODEX_REVIEW.md`).
RAII guard pattern chosen over closure-wrapping (Alex 2026-05-15). Ready for phase R1a kickoff.

**How to use this doc**: the Agent Teams block below is the per-phase template.
The Plan block lists six phases (R1a → R3). Run all three agents in sequence on
**one phase per prompt**. Each phase exits with green tests before the next
phase starts. The Context block at the bottom is shared across all phases.

---

# Agent Teams

All three agents run in sequence per step. Every step exits with green tests.
One phase per prompt — verify, then next.

---

## AGENT 1: Builder

You are the implementation agent. Your job is to write the code described in
the plan below.

Rules:
- Implement exactly what is specified — no extras, no "while I'm here" additions.
- Write tests for every public function and every non-trivial code path.
- If something is ambiguous in the spec, pick the simplest interpretation and
  leave a `// DECISION:` comment explaining what you chose and why.
- Do NOT reference any proprietary codebases as implementation reference. Use
  only the files in this repo, the spec below, and MIT/Apache 2.0 open-source
  references (OneTrainer `LayerOffloadConductor.py` is Apache 2.0 — citable).
- This repo's conventions: BF16 inference primitives go through NVRTC
  (`bf16_elementwise.rs` etc.); cuBLAS/cuDNN ops go through `.cu` files. See
  `flame-core/CLAUDE.md` for canonical-vs-legacy paths.
- Commit after each logical unit compiles and tests pass. Commit message format:
  `feat(<module>): R<phase> — <one-line summary>`.
- Run `cargo test -p flame-core --release` before declaring a phase done.

When you finish a phase, hand off to Bug Fixer with a one-line summary of
files touched and tests added.

---

## AGENT 2: Bug Fixer

You are the bug hunting agent. You run AFTER Builder completes each phase.

Your job:
1. Read every file Builder touched or created.
2. For each file, check for:
   - **CUDA-specific**: missing `cudaStreamSynchronize` before reads of just-written
     memory; missing `.contiguous()` after `cat`/`narrow`/`permute`; non-power-of-2
     alignment on bump cursors; double-`cudaFree` via the hook path; mid-slab
     offsets escaping range registration.
   - **Lifetime**: any `Drop` impl that touches a global, where the global may be
     poisoned or torn down at process exit; any `Arc<CudaDevice>` captured by
     value vs by clone (identity-equality breaks).
   - Off-by-one errors in indexing, ranges, and loop bounds (slab cursor math).
   - Race conditions (multiple threads bumping the same cursor, hook installation
     order between `BlockOffloader::ensure_ring` and slab init).
   - Resource leaks (slab handed back without decrementing live_count; error
     paths that early-return without `register/unregister` symmetry).
   - Error paths that `panic!`/`.unwrap()` instead of propagating (allocation
     overflow MUST return a `Result` with capacity/cursor/requested-bytes in
     the message, per Codex P1).
   - State leaks between calls (a slab cursor that doesn't reset; a registry
     entry that survives `release_all_slabs`).
   - Type mismatches or silent truncation (`usize` slab offset → `u64` device
     ptr, BF16 element-count vs byte-count confusion).
   - Edge cases: empty allocation (n=0), single-element, slab-capacity-exact,
     slab-capacity-plus-one.
   - Missing or wrong test assertions (tests that pass but don't actually
     verify what they claim to verify — e.g., a "no cudaFree called" test
     that doesn't actually instrument the hook).

3. For every bug found:
   - Write a regression test that fails on the current code.
   - Fix the bug.
   - Confirm the regression test now passes.
   - Confirm all existing tests still pass (`cargo test -p flame-core --release`).

4. If you find zero bugs, explicitly state "Bug Fixer: zero issues found" —
   don't invent problems.

When you finish, hand off to Skeptic.

---

## AGENT 3: Skeptic

You are the adversarial review agent. You run AFTER Bug Fixer passes.

Your job is to challenge assumptions and test edge cases that neither Builder
nor Bug Fixer considered.

For each piece of functionality implemented:
1. Ask "what happens if..." questions specific to the domain:
   - What happens if this runs twice in a row? (slab reset after reset, double
     hook install)
   - What happens if the input is valid but pathological (slab of size 0,
     allocation request of `usize::MAX`, BF16 tensor with 1-byte residue, an
     `Arc<CudaDevice>` that has been dropped elsewhere)?
   - What happens if a dependency fails mid-operation? (cudaMalloc fails during
     lazy slab init; the cudarc hook is uninstalled between alloc and drop)
   - What happens if this interacts with `BlockOffloader`, `AutogradContext`,
     `cuda_alloc_pool::clear_cache`, or `cuda_graph` capture — does the contract
     hold?
   - What happens under the opposite conditions the code was clearly written
     for? (slab mode OFF during transient scope; transient scope active during
     persistent allocation)
   - What happens if the trainer panics inside the step body — does
     `StepSlabGuard::drop` panic-on-live-count-nonzero create a
     double-panic-during-unwinding abort?

2. Write adversarial tests for every question. These should be the tests that
   would catch a regression six months from now when someone changes something
   upstream (cudarc bump, autograd refactor, new offload strategy).

3. If any test fails, hand back to Builder with the failing test attached.
   If all pass, explicitly state "Skeptic: all adversarial tests pass."

Do NOT rubber-stamp. If you can't find a real concern, look harder at the
seams: the cudarc drop hook ↔ slab registry ↔ ring registry interaction;
the `AutogradContext::clear` ordering vs the guard's drop; the persistent-
vs-transient classification boundary. The bugs that ship are always at the seams.

---

# Plan

Six phases, sequential. Each phase has: **Goal**, **Files**, **API/Spec**,
**Tests**, **Acceptance**. Builder reads only its current phase. Bug Fixer
and Skeptic see all phases up to and including current.

---

## Phase R1a — External Memory Registry

**Goal**: Replace `cuda_alloc_pool::external_ptrs` (exact-pointer HashMap) with
a unified registry that knows both ranges (slab-owned) and exact pointers
(ring/pool exact allocations). Install ONE process-wide cudarc drop hook that
consults the unified registry.

**Why this is first**: P0 #2. Without a range-aware registry, any view/narrow
on a slab tensor produces a CudaSlice at a non-base offset; current exact-
pointer hook misses it; `cudaFree` is called on a mid-slab address.

**Files**:
- New: `flame-core/src/external_memory.rs`
- Modify: `flame-core/src/cuda_alloc_pool.rs` (delegate `register_external_ptr` /
  `is_external_ptr` / `unregister_external_ptr` to the new registry; keep the
  old public API as thin wrappers for source compatibility).
- Modify: `flame-core/src/offload/mod.rs::ensure_ring` to register through the
  new registry; idempotent install of the cudarc hook.
- Modify: `cudarc-pinctx`: the existing `install_external_ptr_hook` stays,
  but flame-core's hook closure now consults `ExternalMemoryRegistry`.

**API** (`flame-core::external_memory`):

```rust
pub enum ExternalOwner { Slab, Ring, PoolExact, BlockOffloader }

pub struct ExternalRange {
    pub start: u64,
    pub end: u64,         // exclusive
    pub device_key: usize, // Arc<CudaDevice> identity (Arc::as_ptr as usize)
    pub owner: ExternalOwner,
}

pub struct ExternalMemoryRegistry { /* internal */ }

impl ExternalMemoryRegistry {
    pub fn global() -> &'static ExternalMemoryRegistry;

    /// Register a contiguous range. Returns an opaque handle for later removal.
    pub fn register_range(&self, range: ExternalRange) -> RangeHandle;
    pub fn unregister_range(&self, handle: RangeHandle);

    /// Exact-pointer refcount (back-compat for ring/pool callers).
    pub fn register_exact(&self, ptr: u64, device_key: usize, owner: ExternalOwner);
    pub fn unregister_exact(&self, ptr: u64) -> usize; // returns new refcount

    /// Hook decision: should cudarc skip cudaFree for `ptr` on `device_key`?
    pub fn should_skip_free(&self, ptr: u64, device_key: usize) -> bool;
}
```

**Hook installation**: idempotent. First caller (slab init OR ring init,
whichever fires first) installs the closure that calls
`ExternalMemoryRegistry::global().should_skip_free`.

**Tests** (`flame-core/src/external_memory.rs` + integration):

1. `registry_exact_pointer_skip_free` — register exact ptr, hook returns
   `skip=true`; unregister, hook returns `skip=false`.
2. `registry_range_covers_offset_ptr` — register range `[0x1000, 0x2000)`;
   `should_skip_free(0x1500, _)` returns `true`; `0x0fff` and `0x2000`
   return `false`.
3. `registry_range_and_exact_compose` — register a range AND an exact ptr
   inside that range; unregister the exact ptr; range still protects the
   address.
4. `registry_device_isolation` — same ptr value registered on two different
   `device_key`s does not cross-protect.
5. `registry_hook_idempotent_install` — calling `ensure_hook_installed`
   twice does not panic, does not double-register; second call is a no-op.
6. `registry_exact_refcount` — register same ptr twice, unregister once,
   `should_skip_free` still returns `true`; second unregister returns 0
   and hook returns `false`.

**Acceptance**:
- All 6 unit tests green.
- Existing `pool_f32_opt_out_*` and `ring_pool_adapter_smoke` tests still
  green after the back-compat shim.
- `cargo build -p flame-core --release` clean.
- Klein 9B 1-step smoke with no other changes still runs (workaround default
  still on; this phase changes infrastructure only).

---

## Phase R1b — StaticSlabAllocator primitive

**Goal**: One bump allocator per (device, dtype). Lazy slab materialization.
Live-allocation counter. Strict reset guard. Zero-init vs uninit split.

**Files**:
- New: `flame-core/src/static_slab_v2.rs`
- New: `flame-core/tests/static_slab_v2_unit.rs`

**API**:

```rust
pub struct StaticSlabAllocator {
    device: Arc<CudaDevice>,
    device_key: usize,
    slab: Option<CudaSlice<u8>>,  // lazily materialized
    slab_handle: Option<RangeHandle>,
    capacity_bytes: usize,
    cursor: usize,
    live_count: AtomicUsize,
}

impl StaticSlabAllocator {
    pub fn new(device: Arc<CudaDevice>, capacity_bytes: usize) -> Self;

    /// Allocate `n` BF16 elements. Materializes the slab on first call.
    /// Returns Err on overflow with a message containing requested bytes,
    /// capacity, current cursor, dtype, and the env override name.
    pub fn alloc_u16(&mut self, n: usize) -> Result<CudaSlice<u16>>;

    /// Allocate `n` F32 elements. Memory is NOT zero-initialized.
    pub fn alloc_f32_uninit(&mut self, n: usize) -> Result<CudaSlice<f32>>;

    /// Allocate `n` F32 elements. Memory IS zero-initialized via cudaMemsetAsync.
    pub fn alloc_f32_zeroed(&mut self, n: usize) -> Result<CudaSlice<f32>>;

    /// Reset cursor. STRICT: fails if live_count != 0.
    pub fn reset(&mut self) -> Result<()>;

    pub fn live_count(&self) -> usize;
    pub fn used_bytes(&self) -> usize;
    pub fn capacity_bytes(&self) -> usize;

    /// Explicit teardown for phase boundaries. Frees the slab; subsequent
    /// allocs lazily re-materialize.
    pub fn release(&mut self) -> Result<()>;
}

/// Per-device, per-dtype global accessor.
pub fn slab_for_device(device: &Arc<CudaDevice>) -> &'static Mutex<StaticSlabAllocator>;
```

**Slab lifetime contract**:
- `alloc_*` synthesizes a `CudaSlice` via the existing `CudaSliceMirror` trick
  (see `offload/mod.rs:1053`, `cuda_alloc_pool.rs:920-960`).
- Each allocation increments `live_count`.
- The slice's underlying ptr lands in the registry's range (registered at
  slab materialization, NOT per-alloc).
- When the slice drops, the cudarc hook sees the ptr-in-range and skips
  `cudaFree`. Slab's `pool_return_*`-equivalent path is the live_count
  decrement, wired through the existing `TensorStorage::Drop` chain.
- `reset()` fails if `live_count != 0`. Error message includes the count
  and a hint about the most recent allocation site (best-effort: a debug
  ring buffer of last-N alloc backtraces, gated by env).

**Env knobs**:
- `FLAME_STATIC_SLAB_BYTES_BF16` (default: 4 GiB)
- `FLAME_STATIC_SLAB_BYTES_F32` (default: 4 GiB)
- `FLAME_STATIC_SLAB_DEBUG_BACKTRACE` (default off; if on, ring-buffer alloc
  backtraces for reset diagnostics)

**Tests**:

1. `slab_alloc_advances_cursor` — alloc 3 BF16 tensors of sizes 1K, 2K, 3K;
   cursor advances by aligned-up totals; `live_count == 3`.
2. `slab_lazy_materialization` — `StaticSlabAllocator::new` does not call
   `cudaMalloc`; first `alloc_*` does.
3. `slab_overflow_fails_clean` — alloc up to capacity, then one more byte;
   `Err` with requested/capacity/cursor in message.
4. `slab_reset_with_live_allocation_fails` — alloc one slice; call `reset()`;
   assert `Err` and that the ptr is still valid.
5. `slab_reset_after_drop_succeeds` — alloc and drop slices; reset; cursor
   rewinds; `live_count == 0`; subsequent alloc returns ptr at base.
6. `slab_f32_zeroed_reads_zeros` — `alloc_f32_zeroed(N)`, copy back, all zeros.
7. `slab_f32_uninit_is_fast` — `alloc_f32_uninit(N)` does not invoke memset
   (instrument via counter).
8. `slab_hook_covers_offset_ptr` — alloc, take a narrow/offset slice, drop
   it; assert no `cudaFree` (instrument via the registry's hook decision).
9. `slab_multi_device_isolation` — if two `Arc<CudaDevice>` handles for the
   same device-0 exist, allocations from different `Arc`s go to different
   `Mutex<StaticSlabAllocator>` entries (per-device by `Arc::as_ptr` key).
10. `slab_release_then_realloc` — `release()` then `alloc_*` re-materializes.
11. `slab_alloc_zero_elements` — `alloc_u16(0)` returns a zero-length slice
    without bumping the cursor and without incrementing live_count
    (or: increments and decrements symmetrically — pick one, document it
    as a `// DECISION:` comment).

**Acceptance**:
- All 11 unit tests green.
- `cargo test -p flame-core --release static_slab_v2` clean.
- Microbench (`tests/static_slab_v2_microbench.rs`): 1000 BF16 allocs of
  varying sizes complete with zero `cudaFree` calls (instrumented by
  hook counter), zero `cudaMalloc` after slab materialization.

---

## Phase R2a — StepSlabGuard (RAII transient scope)

**Goal**: A scope object that marks "allocations inside this scope are
transient." The slab is consulted only when a guard is active on the
current thread. On drop, the guard resets the slab and panics if the
live-count invariant is violated.

**Files**:
- Modify: `flame-core/src/static_slab_v2.rs` (add `StepSlabGuard` + thread-local)
- Modify: `flame-core/src/cuda_alloc_pool.rs::pool_alloc_u16` and
  `pool_alloc_f32`: BEFORE the bucket lookup, if `StepSlabGuard::active_on_thread()`
  AND `FLAME_USE_STATIC_SLAB=1`, route to `StaticSlabAllocator`.

**API**:

```rust
/// RAII guard that activates transient slab dispatch for the current thread.
/// Drop resets the slab in strict mode: panics if live_count != 0.
pub struct StepSlabGuard {
    device: Arc<CudaDevice>,
    finished: bool,
}

impl StepSlabGuard {
    /// Enter a transient scope. Returns Err if a guard is already active on
    /// this thread (nested guards forbidden).
    pub fn enter(device: Arc<CudaDevice>) -> Result<Self>;

    /// Convenience for the common case (uses the default training device).
    pub fn enter_default() -> Result<Self>;

    /// Explicit graceful close. Resets slab; returns Err if live_count != 0.
    /// After `finish()`, `Drop` is a no-op.
    pub fn finish(mut self) -> Result<()>;

    /// True if the current thread has an active guard.
    pub fn active_on_thread() -> bool;
}

impl Drop for StepSlabGuard {
    fn drop(&mut self) {
        if self.finished { return; }
        // Strict: panic on live_count != 0, UNLESS already unwinding
        // (avoid double-panic-during-unwind abort).
        if std::thread::panicking() {
            // Best-effort cleanup; do not panic during unwind.
            let _ = self.reset_or_log();
        } else {
            self.reset_or_panic();
        }
    }
}
```

**Dispatch rule** (P0 #3 — non-negotiable):
- `pool_alloc_u16` / `pool_alloc_f32` route to slab ONLY when
  `StepSlabGuard::active_on_thread() == true` AND `FLAME_USE_STATIC_SLAB=1`.
- Outside the guard scope, allocations go through the existing pool path
  unchanged. Persistent tensors (model weights, optimizer state, LoRA
  params, sampler buffers) are allocated outside any guard.

**Tests**:

1. `guard_routes_alloc_to_slab` — outside guard: `pool_alloc_u16` goes to
   cudart. Inside guard: same call goes to slab (instrumented).
2. `guard_persistent_alloc_not_in_slab` — alloc outside guard; enter guard;
   reset; outside tensor still readable; cursor rewinds normally.
3. `guard_drop_on_clean_scope_succeeds` — enter, alloc, drop slice, drop
   guard; no panic; live_count == 0; cursor rewound.
4. `guard_drop_with_live_count_panics` — enter, alloc, leak (mem::forget),
   drop guard; assert panic (`#[should_panic]`).
5. `guard_finish_with_live_count_errs` — enter, alloc, leak, call `finish()`;
   assert `Err`; slab NOT reset.
6. `guard_panic_during_step_does_not_double_panic` — enter, panic inside
   the scope; guard's Drop sees `thread::panicking()`, logs instead of
   panicking; process unwinds cleanly to the catch.
7. `guard_nested_forbidden` — enter, enter again; second call returns `Err`.
8. `guard_active_on_thread_query` — `active_on_thread()` returns false
   outside, true inside, false after drop.
9. `guard_disabled_by_env` — with `FLAME_USE_STATIC_SLAB=0`, allocations
   inside the guard still go to the legacy pool (slab dispatch is gated).

**Acceptance**:
- All 9 unit tests green.
- Existing pool tests still green.
- Allocations made BEFORE any guard exists are unaffected (verified by
  test 2).

---

## Phase R2b — Klein trainer wiring

**Goal**: Wrap the per-step body of `train_klein.rs` in `StepSlabGuard`,
declared as the FIRST local in the loop body so it drops last. Move
validation and sampling OUT of the guard scope (they have different
lifetime patterns and may retain tensors across what the trainer treats
as a step boundary).

**Files**:
- Modify: `EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs`
  around lines 1420-1438 (where `AutogradContext::clear()` currently is).
- Audit: `train_klein.rs:380-388` — confirm `FLAME_USE_STATIC_SLAB` is NOT
  set before model load. The env should be set AFTER model + LoRA + optimizer
  state allocations are done.

**Required wiring pattern** (per Alex 2026-05-15):

```rust
for step in start_step..args.steps {
    // Guard is the FIRST local — drops LAST at end of loop body.
    // Reverse drop order means step-local tensors drop before the guard.
    let _slab_step = flame_core::static_slab_v2::StepSlabGuard::enter(
        device.clone(),
    )?;

    // All transient training allocations live here:
    //   - batch load
    //   - forward pass (pred, ...)
    //   - loss computation (pred_f32, target_f32, raw_loss, loss)
    //   - backward (autograd)
    //   - optimizer step
    //   - AutogradContext::clear()
    //
    // End of scope: step-local tensors drop, then guard drops.
    // Guard's drop resets the slab; panics if any slab tensor leaked.
}

// Validation / sampling happen OUTSIDE the per-step guard scope:
if should_validate(step) {
    // Validation has its own lifetime patterns. Wrap it in its own
    // StepSlabGuard only if validation allocations are also transient.
    run_validation(&model, ...)?;
}
```

**Hard constraints** (encode as comments in the trainer):
1. `StepSlabGuard::enter` MUST be the first allocation-creating local in the
   loop body. If something must be alive across the guard boundary, it lives
   in the outer scope (before the loop) or in a separate block after the
   guard scope.
2. `AutogradContext::clear()` is called INSIDE the guard scope, near the end.
   This is necessary but not sufficient — the guard's drop is the real
   boundary.
3. `clear_pool_cache()` is NOT called inside the guard scope. The legacy
   `clear_pool_cache()` API still exists for phase-boundary use cases
   (Codex P1 #1) — those happen outside training step boundaries.

**Tests** (mostly trainer-side smokes, not unit tests):

1. `guard_first_local_lint` — a compile-time-ish check (a doc test or a
   `#[test]` that re-reads `train_klein.rs` and greps for the pattern
   `for step in .* \{\s*let _slab_step =`). This catches future drift
   where someone inserts a `let batch = ...;` above the guard.
2. (See R2c for end-to-end gates.)

**Acceptance**:
- Trainer compiles.
- `FLAME_USE_STATIC_SLAB=0`: trainer runs exactly as before (regression-free).
- `FLAME_USE_STATIC_SLAB=1` + workaround off: trainer enters R2c smoke gate.

---

## Phase R2c — Smoke + parity gates

**Goal**: Validate the slab path on real workloads. No code changes — pure
testing. Failure at any gate hands back to Builder.

**Gates** (run in order, each must pass before the next):

1. **Klein 9B, no validation/sample, 15 steps**:
   ```bash
   FLAME_ALLOC_POOL=1 FLAME_USE_STATIC_SLAB=1 \
   FLAME_POOL_TRAP_BF16=1 FLAME_POOL_TRAP_BACKTRACE=1 \
   cargo run --release --bin train_klein -- configs/klein_9b_lora.toml \
       --offload --steps 15
   ```
   - Pass: all 15 steps clean, no trap fire, no guard panic.
   - Steady-state s/step ≤ 4.5 (matches `FLAME_ALLOC_POOL=0` workaround
     within BF16 noise; faster than that is bonus, not required at this gate).

2. **Klein 9B with validation + sample every 5 steps**:
   - Same env, 15 steps with validation/sample enabled.
   - Catches tensors that survive the nominal step boundary into validation.
   - Pass: no panic, no parity drift vs the no-val baseline at step
     boundaries.

3. **Z-Image 50 steps**:
   ```bash
   FLAME_ALLOC_POOL=1 FLAME_USE_STATIC_SLAB=1 \
   cargo run --release --bin train_zimage -- configs/zimage_lora.toml \
       --steps 50
   ```
   - Pass: clean run, loss curve within BF16 noise of the
     `FLAME_ALLOC_POOL=0` baseline (compare against `board.db`).

4. **One inference / phase-boundary workload**:
   - Run `inference-flame` Klein-LoRA inference with `FLAME_USE_STATIC_SLAB=1`
     in the env but no guard (inference doesn't use the trainer's step
     loop; the slab dispatch should be inert).
   - Pass: output bit-identical to `FLAME_USE_STATIC_SLAB=0` baseline.
   - Confirms persistent allocations are unaffected by slab env when no
     guard is active.

5. **Existing test suite**:
   - `cargo test -p flame-core --release` clean.
   - `cargo test -p eridiffusion-core --release` clean.
   - Ring pool adapter smokes still green.

**Acceptance**:
- All 5 gates green. Hand off to Phase R3.

---

## Phase R3 — Default policy

**Goal**: Decide what becomes default once R2c is stable. NO code changes
until R2c has run for at least one full Klein training run end-to-end
(>= 500 steps) without regression.

**Tasks** (each one optional, ordered by safety):

1. **Remove `FLAME_ALLOC_POOL=0` auto-set in `train_klein.rs:main`**.
   - Pre-req: 500-step Klein 9B run with `FLAME_USE_STATIC_SLAB=1` clean.
   - Replace with `FLAME_USE_STATIC_SLAB=1` auto-set.
   - Document in trainer header comment that direct override is supported
     for debugging.

2. **Make slab default for Z-Image trainer**.
   - Pre-req: 1000-step Z-Image run clean.
   - Mirror Klein wiring in `train_zimage.rs`.

3. **DO NOT delete `clear_pool_cache()` legacy code** (Codex P1 #1).
   - It's still used at phase boundaries in EDv2 + inference.
   - The slab path uses `StaticSlabAllocator::release()` for phase boundaries;
     `clear_pool_cache()` stays for direct-fallback memory not routed
     through any allocator.

4. **DO NOT delete the bucketed free-list code yet** (Codex defer to R3+).
   - Inference path may still rely on it.
   - Mark with TODO + condition for deletion ("once inference workloads
     verified against slab mode for a full release cycle").

5. **Restore EDv2 `klein.rs` to production state**:
   - Re-apply commit `1994cac`'s migration to `checkpoint_offload_boundary`.
   - This was reverted during diagnosis (see Context > State of tree).

**Acceptance** (this phase is the final sign-off):
- All R2c gates still green after the default flips.
- Klein 9B + `--offload` at 3.5-3.8 s/step (matches `661f9e9` pre-regression
  baseline) or better. Closing the gap to OT's 2.07 s/step is separate
  workstream — NOT in scope here.
- Z-Image ≤ 2.0 s/step.

---

# Context

Shared across all phases. Read once at session start.

## Reading list (in order)

1. `flame-core/TENETS.md` — non-negotiables. Tenets 1, 2, 5 directly relevant.
2. `flame-core/docs/SPEED_CONTRACT.md` — Clauses 1, 5 are the gates.
3. `flame-core/CLAUDE.md` — canonical vs legacy paths in flame-core.
4. `flame-core/docs/OFFLOAD_NEXT_GEN_DESIGN.md` — original conductor plan.
   Phases 3 + 5 deleted as redundant; THIS handoff supersedes Phase 4.
5. `OneTrainer/modules/util/LayerOffloadConductor.py:122-321` — reference
   implementation. `StaticLayerAllocator` + `StaticActivationAllocator`.
   Apache 2.0 — citable in commit messages.
6. `flame-core/HANDOFF_2026-05-15_OT_STATIC_SLAB_REDESIGN_CODEX_REVIEW.md` —
   Codex's review of the pre-revision handoff. All four P0s + five P1s
   from that review are baked into this Plan. Read it for the reasoning,
   not for the API spec (this doc is the spec).
7. `/tmp/cuda_bug_websearch_2026-05-15.md` — web-search findings. Confirms
   userspace bug, not cudart.

## The bug class (one paragraph)

A live `Tensor` holds a `CudaSlice<u16>` whose backing memory has already
been `cudaFree`'d by `cuda_alloc_pool::clear_cache`. When `Tensor::cat`
issues `cuMemcpy2DAsync_v2` against the stale ptr, cudart rejects it with
`CUDA_ERROR_INVALID_VALUE`. The 2D variant is strict enough to catch this;
1D `cuMemcpyAsync` would silently corrupt. Reproduces deterministically at
step 13-14 on Klein 9B + `--offload` + `FLAME_ALLOC_POOL=1`. Trap forensics
show a ptr cycling `InCache → Freed → InCache → Freed` without an
intervening `pool_alloc` — proves a duplicate Arc was alive past the first
`cudaFree`. The duplicate drops inside `compute_gradients` (SmallVec
gradient temporary).

## What was ruled out

| Hypothesis | Test | Result |
|---|---|---|
| cudart async mempool VA reuse | `CUDARC_FORCE_SYNC_ALLOC=1` | Still crashed |
| Phase 6 `checkpoint_offload_boundary` migration | reverted `1994cac` | Still crashed |
| Phase 2 ring-backed BF16 slot allocs | `FLAME_OFFLOAD_RING_DISABLE=1` | Still crashed |
| F32 free-list bug | `FLAME_F32_POOL_CACHE=0` (default) | Still crashed |
| Same-bucket concurrent double-push | defensive iter check in `push_u16` | 0 hits |
| Mempool release threshold | `cuMemPoolTrimTo` per step | Doesn't fix; +0.3 s/step + grad explosion |

What works: `FLAME_BF16_POOL_CACHE=0 FLAME_F32_POOL_CACHE=0` — 15/15 clean
at 4.4 s/step. Confirms bug is in cache cycling, not raw cudart allocs.

## State of the tree

### flame-core HEAD = `d04bb5f`

```
d04bb5f fix(tests): pool_stats_smoke is robust to parallel test execution
4397635 test(cuda_alloc_pool): Skeptic Phase 1 — adversarial tests for F32 opt-out
a6886fb fix(cuda_alloc_pool): F32 opt-out path zero-inits to match workaround
eb7cabf feat(cuda_alloc_pool): F32 free-list caching defaults OFF
1c2776d revert(static_slab,...): remove off-spec slab dispatch path
a10df80 revert(autograd): drop Phase B with_region wrap from backward_impl
88a7239 fix(cuda_alloc_pool): downgrade slab-overflow warn to trace
5fa2061 feat(static_slab,cuda_alloc_pool,autograd): Phase B — region-dispatch (reverted)
1c09f97 feat(static_slab,autograd): Phase A — bump allocator (reverted)
```

### Uncommitted diagnostic infrastructure (THIS session)

**flame-core**:
- `src/autograd.rs` — `FLAME_SKIP_POOL_CLEAR=1` escape hatch
- `src/cuda_alloc_pool.rs` — `PtrState/PtrEvent/PtrHistory` trap +
  `bf16_pool_cache_enabled` opt-out
- `src/offload/mod.rs` — `FLAME_OFFLOAD_RING_DISABLE=1` escape hatch +
  `trap_record_external`
- `src/tensor.rs` — `trap_record_external` in `from_bf16_slice_gpu`
- `src/tensor_ops_extended.rs` — `trap_validate_bf16_ptr` in `Tensor::cat`

**cudarc-pinctx**:
- `src/driver/safe/core.rs` — `CUDARC_FORCE_SYNC_ALLOC=1` escape hatch
  (verified does NOT fix the bug)

**EriDiffusion-v2**:
- `crates/eridiffusion-core/src/models/klein.rs` — `1994cac` partial revert
  (verified does NOT fix the bug)

### First commit before R1a starts

Commit the diagnostic stash as ONE preserved commit (will be needed for
R2c validation):

```
feat(cuda_alloc_pool,offload,cudarc-pinctx): instrumentation + escape
hatches for use-after-free diagnosis
```

Then revert the EDv2 `klein.rs` change — restore `1994cac`'s
`checkpoint_offload_boundary` migration. That's the production path.

## Reuse-this infrastructure

- `flame_core::cuda_alloc_pool::register_external_ptr` / `is_external_ptr` /
  `unregister_external_ptr` — back-compat shim after R1a; underlying storage
  moves to `ExternalMemoryRegistry`.
- `cudarc::driver::install_external_ptr_hook` — already shipped in
  `cudarc-pinctx@6608a7a`. Used by `BlockOffloader::ensure_ring`; slab
  init shares the same hook via the new registry.
- `CudaSliceMirror` trick — `cuda_alloc_pool.rs:920-960`, `offload/mod.rs:1053`.
  Same pattern for slab.

## Open items NOT in this redesign

- Klein 9B → OT 2.07 s/step gap. R3 restores the 3.4-3.8 baseline; closing
  to 2.07 is Class A F32 grad-storage roundtrips + Class E syncs (see
  `HEADTOHEAD_2026-05-12_ROOT_CAUSE.md`). Separate workstream.
- Z-Image OT-match (target ≤ 1.5 s/step). Same separate workstream.
- BlockOffloader per-step IO costs.

## Constraints

- **Tenet 1**: in flame-core, fix the primitive. No trainer-side workarounds
  beyond the R2b guard (which IS the fix's interface, not a workaround).
- **Tenet 2**: API makes right easy. `StepSlabGuard` is the only entry point
  for transient slab dispatch.
- **Tenet 4**: every clause has a measurement gate. R2c spells them out.
- **Tenet 5**: the long-term goal is to remove `FLAME_ALLOC_POOL=0` as the
  default trainer-side workaround. R3 task 1.
- **No regressions to inference**: R2c gate 4 is non-negotiable.
- **No regressions to the F32 path**: R1b zero-init/uninit split is the
  contract. Existing F32 opt-out tests stay green.

— Claude (Opus 4.7, 2026-05-15, post-Codex revision)
