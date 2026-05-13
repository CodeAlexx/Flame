# HANDOFF — Autograd v2 Implementation, Fresh Context

**Date:** 2026-05-13
**Purpose:** Next session picks up autograd v2 Phase 1 implementation cold. Everything in this file is what survived /clear. Read this AFTER soul.md, endearment.md (if present), and `flame-core/TENETS.md`, in that order.

---

## STANDING RULE for all port/build agents (added 2026-05-13 after Phase 1 narrative defect)

Phase 1's port agent reported "a previous session had left an essentially complete Phase 1 implementation in the working tree" — verifiably false against `git ls-tree 31e54b9 src/autograd_v2/` (only the stub `mod.rs` existed). Agent wrote everything itself and mis-narrated. Code was correct; narration was not.

**Mandatory for every future port/build agent prompt:**

1. **BEFORE starting work**, the agent MUST run and log verbatim in its report:
   ```
   git status --short
   git ls-tree HEAD <target_dir>
   git rev-parse HEAD
   ```
2. **AFTER finishing**, the agent MUST report "code I wrote" via:
   ```
   git diff <start-commit>..HEAD --stat
   git diff <start-commit>..HEAD --shortstat
   ```
   — **NOT** from descriptive recall.

Reporting-integrity defects compound. Code correctness alone is not the goal — accurate reports of process are required for verifier agents to do their job.

---

## TL;DR — where you are landing

- **Phase 0 of autograd v2 is GREEN and pushed.** Four commits on `flame-core` main, three verifier agents (builder + bug-fixer + skeptic) all closed clean.
- **Phases 1–5 are the next 7–9 weeks of focused work**, per the design doc the previous session wrote and committed.
- **Your immediate next step:** spawn the Phase 1 port agent. Phase 1 = metadata + core types. Spec lives in `flame-core/docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`.
- **Hard rule from the user:** "after Phase 0 green and only after, start on build for v2." Phase 0 IS green. v2 build is unblocked.

---

## Read these first, in order

1. `/home/alex/soul.md` — required, personal continuity
2. `/home/alex/endearment.md` — if present, relational
3. `/home/alex/EriDiffusion/flame-core/TENETS.md` — required, design principles
4. `/home/alex/EriDiffusion/flame-core/docs/SPEED_CONTRACT.md` — the 5-clause audit gate
5. `/home/alex/EriDiffusion/flame-core/docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` — the full v2 plan, all 16 spec changes, Phase 1–5 detail
6. `/home/alex/EriDiffusion/flame-core/docs/BF16_GRAD_DECISION.md` — Option A documented (BF16 grads end-to-end), F32-coercion sites listed for Phase 4

Then this file for the operational state. Then the task list for what's queued.

---

## What landed this session — flame-core

13 commits on `main`, pushed to `https://github.com/CodeAlexx/Flame`:

| Commit | What | Verified |
|---|---|---|
| `b552f61` | Class E narrow primitive — 268 syncs/step → 0 on klein 9B (inline kernel-arg metadata replaces per-call cudaMalloc + cudaStreamSynchronize + cudaFree). 15× per-call latency reduction in microbench. | nsys-confirmed |
| `dbcb262` | Speed contract + handoff docs + autograd_v2 design review | docs |
| `542c531` | Class E sdpa_stream_bf16 — cached workspace + cuBLAS handle (11 cudaMalloc + 11 cudaFree per call → 0) | tests green |
| `15d8ef8` | Class B narrow_backward — BF16 cast detour removed, BF16 passes through byte-copy kernel. 569 µs → 467 µs per call. | klein 8-step bit-identical |
| `df00c5f` | BlockOffloader moved from eridiffusion-core to flame-core/src/offload.rs | EDv2 shim re-exports |
| `0cfe5c0` | FLAME_INDEX/MODULES/KERNELS/CONVENTIONS updated for Class E + Class B + offload move | docs |
| `e84a335` | FlexTensor port Phase 1 — telemetry + transfer_benchmark | 5+2 tests pass, 26 GB/s measured |
| `e2ef4cf` | telemetry smoke test (regression gate) | |
| `b5a818a` | FlexTensor Phase 2 — Strategy trait + TwoSlot/Knapsack/Adaptive + planner. Default behavior unchanged (Strategy is opt-in). | 31 tests pass |
| `cd2718b` | **Class C** — sum_dim_keepdim_bf16 cooperative reduction. 1219 µs → 59 µs per call (~20× speedup), bit-exact parity across 9 shape/dim combinations. | parity + perf gate |
| `fc70412` | Dead .cu cleanup — 7 files (599 lines deleted) | build green |
| `6ade2d1` | FlexTensor Phase 3 — OffloadManager state machine + state cache | 10 manager smoke tests |
| `5484e2c` | autograd v2 doc — build-in scope expansion (reentrant, hooks, create_graph, view, forward-AD all in v0 surface) | docs |
| `bc7b759` | autograd v2 multi-device decision — build the surface, defer plumbing | docs |
| `e1197e5` | telemetry export — snapshot_to_file / ring_buffer_to_file / dump_all (Phase 4 of FlexTensor port) | new test green |
| `cde77a4` | FLAME_INDEX/MODULES updated for Phase 2+3 + Phase 4 telemetry export surface (105+38 lines added) | docs |
| `d4e5094` | OFFLOAD_GETTING_STARTED.md tutorial — 232 lines, 10 sections | docs |
| `9481b67` | telemetry env var rename `_STEPS` → `_EVENTS` (skeptic caught misnomer) + back-compat alias + FLAME_INDEX line-cite fix | tests green |
| `d3a0917` | BlockOffloader pre-flight log (SIGKILL-during-alloc visibility) | build green |
| `26e97da` | **Phase 0 prereq #1** — in-place version-bump audit + 13 tests across 11 in-place sites | 13/13 tests |
| `c6e91b8` | **Phase 0 prereq #2** — BF16_GRAD_DECISION.md (Option A documented + F32-coercion sites listed) | docs |
| `8e1cf81` | **Phase 0 prereq #3** — autograd_v2 feature flag wiring (Cargo.toml + lib.rs + empty module) | both feature builds green |
| `1f5365c` | **Phase 0 follow-up** — bug-fixer caught 2 missed sites (sgd::step_inplace, rng::rand_fill_). Both fixed + 4 more tests. 13 → 17 tests. | 17/17 pass |

## What landed this session — EriDiffusion-v2

7 commits on `master`, pushed to `https://github.com/CodeAlexx/EriDiffusion.git`:

| Commit | What |
|---|---|
| `0fcb90a` | training/block_offload becomes a shim → re-exports from flame_core::offload |
| `8cd7c3d` | qwenimage opt-in to Adaptive via FLAME_OFFLOAD_ADAPTIVE=1 |
| `f5156d4` | chroma opt-in (smoke-verified: 5-step run, no OOM, normal loss curve) |
| `805d4cd` | wan22 opt-in (build only, no smoke dataset) |
| `6b33852` | sensenova_u1 opt-in (build only) |
| `a362b5c` | ernie opt-in (build only) |
| `8478b9d` | **klein opt-in + time test** — Adaptive shows consistent ~3% step-time win (4.50 → 4.36 s/step steady-state, bit-identical loss across 20 steps) |

---

## Phase 0 verification — definitive proof

| Verifier | Verdict | Evidence |
|---|---|---|
| **Builder** | PASS | Both feature builds clean; 37+ tests green across 8 files (inplace_version_bump_audit 13/13, offload_telemetry_export 1/1, offload_telemetry_smoke 1/1, offload_strategy_smoke 8/8, offload_manager_smoke 10/10, narrow_sync_microbench 1/1, sum_dim_keepdim_bf16_class_c 2/2, narrow_bf16_debug 1/1); train_klein + train_qwenimage build clean; klein step-1 loss `1.1217` reproduces bit-identical pre/post Phase 0 |
| **Bug-fixer** | PASS-with-follow-up | 11 claimed bump sites verified PASS, 5 "don't need" exclusions verified PASS, feature flag wiring + BF16 grad doc clean PASS. Test coverage CONCERN (6/11 sites untested but SavedTensor pattern test covers failure mode in principle). **Area 2 FOUND 2 MISSED SITES** — sgd::step_inplace, rng::rand_fill_. **Both fixed in commit `1f5365c`** (+4 tests, 13 → 17 green) |
| **Skeptic** | REPRODUCES | Both feature builds compile, 17/17 audit tests pass, 20 regression tests still pass. Klein parity NOT-RUN — Skeptic note overstated the crash framing; per the Phase 0 shipped section (Phase 0 §1, lines 451-454 of `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`): `train_klein --config configs/klein9b_alina.json --rank 4 --steps 1` produces bit-identical step-1 loss `1.1217` pre/post Phase 0, and **only step 2+** hits `CUDA_ERROR_INVALID_VALUE` (pre-existing infrastructure issue, not a Phase 0 defect). Working tree restored to main. |

**Verdict: Phase 0 officially green. v2 implementation is unblocked.**

---

## Phase 1–5 plan (from `AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Suggested Implementation Order)

Scope note: **v0 ships with surface for reentrant + hooks + create_graph + view autograd + forward-mode AD + multi-device** (per user directive 2026-05-13: "if we come a day when we need it, i don't want to add it, i want it built in" + "we will need ... not now -- multi gpu fyi"). Feature-complete coverage lands in waves on top.

### Phase 1: Metadata and Core Types (~2 weeks)

- Shared `AutogradMetaV2` (interior-mutable, weak accumulator) — addresses §1 + §2 of the design review
- `Edge`, `GradFn` trait (with `hooks() -> &Hooks` accessor + `num_inputs()`), `NodeId`, sequence + topological numbers as STORED fields (not recomputed)
- Weak leaf accumulator cache (Weak<dyn GradFn> per §1 — breaks the Tensor → meta → Arc → Tensor cycle)
- `SavedTensor` with version handle (Arc<AtomicU32> like today's SavedRef — §4) + optional `fw_grad_` companion field for forward-mode AD
- `InputBuffer` with Option<Tensor> + `num_inputs()` + **both in-place AND out-of-place accumulation paths** (in-place when create_graph=false + dtype/shape match + unique storage; out-of-place otherwise)
- `Hooks` struct: pre/post/tensor callback vecs
- **Multi-device surface**: Engine and InputBuffer don't hardcode single-stream / single-device — stream and device are explicit parameters at every dispatch point (~5 days within Phase 1; feature-complete NCCL plumbing is post-v2 separately)

### Phase 2: Engine Skeleton (~1.5 weeks)

- `GraphRoot`
- `AccumulateGrad` — **BF16-throughout** per Class A (F32 only as `opmath_t` inside kernel). Out-of-place path gated on `create_graph=true`
- Dependency counting (uses topological numbers for O(1) DAG-pruning)
- Ready queue
- `Result<Vec<Option<Tensor>>>` everywhere — version mismatch and released-saved-tensor become recoverable training errors, not panics (§3)
- **Accept `create_graph=true`** — engine permits recording during backward
- **Support nested execute()** via inline-mini-execute pattern (CheckpointGradFn drives a local sub-graph through a mini-engine — single-threaded reentrant, no thread pool)
- **Wire hook dispatch** at GradFn entry/exit

Toy tests (Phase 2 acceptance):
- single leaf sum, two branches into one leaf, diamond accumulation, undefined grad slots, released saved tensor error, version mismatch error, `create_graph=true` (backward-of-backward on a simple op produces correct second-order grad), reentrant (nested Engine::execute inside a GradFn::apply returns cleanly), hooks (pre/post/tensor fire in expected order)

### Phase 3: First Real Ops + View Backward + Forward-Mode (~3 weeks)

P0/P1 ops:
- add, mul, sum, reshape, transpose, matmul/linear, silu, layer_norm
- view, squeeze, unsqueeze, permute (view-autograd surface — first-class GradFn)
- Checkpoint, CheckpointOffload (preserve `flame_core::autograd::checkpoint` semantics bit-equal — the reentrant users)

Each op needs:
- forward wiring under `autograd_v2` feature
- backward struct (the `GradFn` impl)
- **forward-mode AD formula** alongside backward (uses `SavedTensor::fw_grad_`)
- PyTorch fixture parity test (backward direction)
- forward-AD parity vs PT's `torch.autograd.functional.jvp`
- dtype assertion
- no unwanted `.to(F32)` in autograd_v2

Long-tail unary ops:
- Hand-written for P0/P1 path
- Permit a later `derivatives!` proc macro for `sin, cos, exp, log, sqrt, rsqrt, abs, neg, pow, …` once trait + SavedTensor + forward-wrapper patterns have stabilized

### Phase 4: Optimizer + Trainer Integration (~1 week)

Cross-cutting BF16-grad migration per `docs/BF16_GRAD_DECISION.md`:

| File | Sites to migrate |
|---|---|
| `src/gradient.rs` | lines 99, 145-170, 184-189, 209-213, 218-223, 245, 253 — stop coercing to F32 |
| `src/parameter.rs` | lines 119-123, 189-197 — preserve param dtype on set_grad |
| `src/adam.rs` | lines 1075-1082, 1106, 1112, 1273-1283 — accept param-dtype grads |
| `src/ops/grad_norm.rs` | lines 12, 90, 108, 111 — BF16-aware reductions |

Then:
- Route v2 grads into Parameter
- Add grad-norm and clipping tests
- One-step model parity before long runs
- Verify `flame_core::autograd::checkpoint` semantics preserved on v2

### Phase 5: Parity Gate (~1 week)

Do NOT retire v1 until:
- Per-op backward fixture parity passes for all P0/P1 ops
- Per-op forward-mode AD fixture parity passes for all P0/P1 ops
- Model parity passes for klein/zimage/ernie/qwen/chroma (bit-equal loss)
- No ms/step regression on klein 4B / 9B
- BF16 grad policy has optimizer parity
- In-place mutation tests are green
- Reentrant test: training run using `enable_checkpointing` matches v1 bit-equal at step 1+
- Hooks test: simple forward and backward hook fires expected callback count per step

---

## How to spawn Phase 1

Follow the established pattern: **port agent** does the work, then three **verifier agents** (builder, bug-fixer, skeptic) in parallel.

Suggested phase-1 port prompt skeleton:

```
You're implementing Phase 1 of autograd v2 per
`flame-core/docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Suggested
Implementation Order. Phase 0 is verified green (4 commits, 17-test
audit suite passes). Phase 1 = metadata + core types, ~2 weeks of
work.

Read:
- TENETS.md
- docs/SPEED_CONTRACT.md
- docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md (especially §1–§9 blocking
  issues, §8 build-in scope decisions, §Phase 1)
- docs/BF16_GRAD_DECISION.md
- src/saved_ref.rs (today's version-handle pattern — match it)
- src/tensor_storage.rs (bump_version + version() surface)
- src/autograd.rs + src/autograd_v3.rs + src/autograd_v4/ (the 3 prior
  generations — understand what NOT to repeat)

Build (all inside src/autograd_v2/ behind the `autograd_v2` feature):
- AutogradMetaV2 (Arc<Mutex<>> or split TensorInner — both valid)
- Edge, GradFn trait with hooks() + num_inputs() + apply() returning
  Result<Vec<Option<Tensor>>>
- NodeId, sequence_nr + topological_nr as STORED fields
- Weak leaf accumulator cache (no Tensor cycle)
- SavedTensor with Arc<AtomicU32> version handle + optional fw_grad_
- InputBuffer with in-place + out-of-place paths (create_graph gate)
- Hooks struct
- Multi-device explicit-stream/device surface (no hardcoded
  single-stream assumptions)

Tests:
- types compile alone (no engine yet)
- Hooks default impl is empty
- InputBuffer in-place path takes when refcount allows
- SavedTensor version mismatch returns Err (do not panic)

Verification:
- cargo build --features "cuda,heavy_kernels,bf16_u16"
- cargo build --features "cuda,heavy_kernels,bf16_u16,autograd_v2"
- new tests under tests/autograd_v2_types.rs
- no regression on inplace_version_bump_audit (17), offload_* tests,
  narrow_sync_microbench, sum_dim_keepdim_bf16_class_c

Commit as: "feat(autograd_v2): Phase 1 — metadata + core types"

Phases 2-5 NOT in scope for this agent.
```

Then verifier trio same shape as Phase 0's. The Phase-0 verifier prompts are good templates — adapt the file lists and test names.

**One known cost:** klein **multi-step** parity is blocked by `CUDA_ERROR_INVALID_VALUE` at step 2+ on this box's smoke config. Klein step 1 is usable — produces deterministic loss `1.1217` — so single-step parity gates (Phase 5 Deliverable C model parity) ARE available on Klein. Multi-step smokes (loss curves over 100+ steps) need a workaround for the step-2 infra issue or a different model. That's pre-Phase-0, not any v2 phase's defect. Build + unit tests + autograd_v2 feature build remain the meaningful gates.

**Known parallel-test flakes (verified pre-existing at parent commits)** — re-run individually under `--test-threads=1` if hit:
- `tests/autograd_v2_bridge.rs::bridge_scenarios` (1 consolidated test post-`a5da3d5`; pre-fix split as 3 tests and flaked under default parallel mode due to global `AUTOGRAD_CONTEXT` mutex)
- `tests/autograd_v2_engine.rs::single_leaf_sum`
- `tests/autograd_v2_ops.rs::engine_rejects_mismatched_grad_output_shape`
- `tests/autograd_v2_ops.rs::silu_v2_backward_at_zero` / `transpose_v2_backward` / `matmul_v2_backward_correct_shapes` / `sum_v2_backward_broadcasts`
- `tests/inplace_version_bump_audit.rs::copy_underscore_bumps_version`

All pass individually. Root cause: v3 `AUTOGRAD_CONTEXT` is process-global (`src/autograd.rs:56`); tests that touch v3 backward contaminate each other's tape state. Fix is structural (per-test reset already exists; consolidate into single drivers like Phase 5b's `bridge_scenarios`).

---

## Operational notes for the next session

### Trainer migrations already landed

All 6 canonical training trainers opt into Adaptive via `FLAME_OFFLOAD_ADAPTIVE=1`:
qwenimage, chroma, wan22, sensenova_u1, ernie, klein. Default behavior (env unset) unchanged. Heavy-OOM models (sensenova 2048², hidream-o1 when it exists, ltx2, wan22 video) are the intended Adaptive users — klein is just along for the regression-gate ride.

### Open hazard (must be addressed before Phase 3 view-autograd)

**HAZARD-2026-05-13-1: view + in-place silently detaches under `shared_storage`.**
`view = parent.narrow(...)` then `view.add_inplace_same_dtype(...)` does NOT mutate the parent — `Arc::make_mut` inside `ensure_unique_slice` silently COWs the view's storage when refcount > 1. Pre-existing flame-core base bug; no live trainer code hits the pattern today (verified via `rg -n "narrow\([^)]*\)\.add_inplace\|narrow\([^)]*\)\.copy_"` → 0 hits across EriDiffusion). Full mechanism + fix options A/B/C in `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §High-Risk → HAZARD-2026-05-13-1. **Phase 3 view-autograd backward will trip this unless fixed first.**

### Deferred workstreams (do NOT do as part of v2)

- **Inference path Adaptive migration** (Task #21) — ~12 inference binaries (sample_chroma, sample_flux, ltx2_generate, nucleus_infer, etc.), separate ~week-sized workstream
- **FlexTensor Phase 4 trainer auto-selection rollout** — replace env-var pattern with `OffloadManager::discover_profile_activate()` across trainers. Not started, natural next step after v2
- **Multi-device feature-complete (NCCL bindings + DDP)** — ~3-week workstream after v2
- **wan22 / sensenova_u1 / ernie / hidream-o1 smoke validation** — need datasets, can run any time

### What the user repeatedly emphasized this session

These came up multiple times and constrain v2 design:

1. **flame-core is one framework. Per-call inefficiency multiplies across every model. Fix the primitive once.** (tenets §1)
2. **APIs make the right thing easy and the wrong thing hard.** If a caller can accidentally take a slow path, the primitive's API is wrong. (tenets §2)
3. **Measurement beats assertion.** Don't claim a fix without nsys / microbench evidence. The previous-session `autograd.rs:1493` "sync source" claim was dead code; the discipline is to verify before claiming. (tenets §4)
4. **Reject fixes in the wrong place.** A model-level fix that ships in a trainer is a regression against tenet 1 even if the model gets faster. (tenets §5)
5. **Build in surface for everything we'd ever want later** — retrofit cost > upfront cost for autograd specifically.
6. **The winner is the one built better.** OneTrainer is fast BEFORE a new model is added. flame-core's goal is the same: any new model plugs in and is fast out of the gate.

### Outstanding cosmetic items

These were noted by the verifiers but considered non-blocking:

- Bug-fixer area 4 CONCERN: 6 of the 11 in-place sites lack dedicated unit tests; SavedTensor-pattern test covers the failure mode in principle. Adding per-site tests is a low-priority follow-up.
- Skeptic concern 2 from Team 2: periodic-dump firing is invisible from filesystem alone (atomic-replace overwrites same files). Adding a `dump_count` counter to the snapshot would close this. Low priority.

---

## Anchors

- **Last flame-core commit:** `1f5365c` on `main`, pushed to GitHub
- **Last EriDiffusion-v2 commit:** `8478b9d` on `master`, pushed to GitHub
- **Task #36** (the queued v2 Phases 1-5 task) is your trigger to start
- **All other tasks** in the task list are either completed or deferred-to-a-separate-workstream
- **GPU should be free** at session start (verify with `nvidia-smi`)

When you spawn the Phase 1 port agent, mark task #36 as `in_progress`. After it commits, spawn the builder + bug-fixer + skeptic verifier trio (parallel — they handle their own GPU contention via nvidia-smi probes).

---

## One last word

The user spent this session systematically tearing down maintainability and performance debt across flame-core's framework primitives (Class B/C/E narrow + sdpa_stream + sum_dim_keepdim, dead .cu cleanup, BlockOffloader move + FlexTensor Phases 1+2+3+4 port, trainer migrations) so that autograd v2 lands on a clean foundation. Don't squander that.

Phase 1 is the most expensive phase in dollar terms (~2 weeks). It's also the most consequential — every type/trait shape decided in Phase 1 constrains Phases 2-5. The design doc has 16 spec changes that have to be respected. Read it carefully before spawning the port agent.

Good luck.
