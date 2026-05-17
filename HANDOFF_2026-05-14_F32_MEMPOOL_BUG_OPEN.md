# HANDOFF — F32 mempool bug open, Klein regressed (2026-05-14)

**Status**: Klein 9B + `--offload` is **slower today than it was at commit `661f9e9` (2026-05-12)**. Net regression from this session's work. Two million tokens, twelve hours, and the trainer is in a worse state.

**Author note**: I shipped this session and failed to deliver the spec gate. Multiple agents wandered, escalated, or hit harness issues. The work that landed is real but does not close the gap it was meant to. The user is correct that we went backward.

---

## Read FIRST

1. `/home/alex/EriDiffusion/flame-core/TENETS.md` — every violation in this session traces to a tenet.
2. `/home/alex/EriDiffusion/flame-core/docs/SPEED_CONTRACT.md` — Clauses 1+5 still binding.
3. `/home/alex/EriDiffusion/flame-core/docs/OFFLOAD_NEXT_GEN_DESIGN.md` — Phase 2 marked but spec gate unmet.
4. `/home/alex/EriDiffusion/flame-core/docs/OFFLOAD_GAPS_vs_ONETRAINER.md` — Gap 1 still open.
5. `/tmp/skeptic_p2_findings.md` — Phase 2 verdict PARTIAL with 10 findings.
6. `/tmp/bug_fixer_p2_round2_verification.md` — refcount fix details + universal F32 bug discovery.
7. `/tmp/builder_p2c_verification.md` — H1+H3 negative results.
8. `EriDiffusion-v2/HANDOFF_2026-05-14_PHASE2_ROUND2_FINDINGS.md` (local, gitignored) — hypotheses 2/4/5 for future investigation.

---

## State at handoff

### Commits (in order)

flame-core (HEAD `12929f7`):
- `a192813` Builder Phase 2 — ring backs BlockOffloader BF16 slot allocs + external-ptr guards in `push_u16`/`push_f32` + lazy `BlockOffloader::ensure_ring` + doc updates
- `b5090d2` Bug Fixer round 1 — `install_external_ptr_hook` in `BlockOffloader::ensure_ring`
- `12929f7` Bug Fixer round 2 — `external_ptrs` `HashSet → HashMap<u64, u32>` refcount (fixes ring-wrap double-tag)

EriDiffusion-v2 (HEAD `b910bd4`):
- `4747b53` Builder Phase 2 — removed `FLAME_ALLOC_POOL=0` auto-disable in `train_klein.rs:main` (then later restored)
- `3499f95` Bug Fixer round 2 — **RESTORED** `FLAME_ALLOC_POOL=0` auto-disable + Phase 2 findings docs
- `b910bd4` docs — recorded Phase 2c H1/H3 negative results in `train_klein.rs` comment

### Klein 9B + `--offload` performance comparison

| Commit | Pool | Workaround | s/step (steady) | Status |
|---|---|---|---|---|
| `661f9e9` (2026-05-12 baseline) | ON | none | **3.5-3.8** | PRE-REGRESSION, doesn't crash on that day's run |
| Current HEAD (`12929f7` + `b910bd4`) | OFF via env hack | YES | **4.1** | works but Tenet 5 violation |
| Current HEAD without workaround | ON | NO | CRASH at step 2 | the bug |

**The session's net effect on Klein 9B speed: -0.3 to -0.6 s/step regression vs `661f9e9`.** Over a 30k-step training run, that's 2.5-5 hours of GPU time per run lost.

The work shipped IS internally correct (ring allocator works, refcount fix works, regression test passes). But it doesn't close the spec gate, and the production state is slower than where we started this session.

---

## The real open bug

`cuda_alloc_pool` step-2 corruption on Klein 9B + `--offload`. Not the ring-wrap double-tag (that was fixed in `12929f7`). A DEEPER bug surfaces when `FLAME_ALLOC_POOL=0` workaround is removed:

```
thread 'main' panicked at .../cudarc-pinctx/.../core.rs:281:
called Result::unwrap() on Err DriverError(CUDA_ERROR_INVALID_VALUE)
stack: CudaSlice<T>::drop → CudaAllocPool::clear_cache
       → autograd::compute_gradients → AutogradContext::backward
```

Specifically: at backward of step 1 (after Phase 2 refcount fix), a non-external `CudaSlice<u16>` returned via `pool_return_u16` fails `cudaFree` (sync or async). The ptr was correctly allocated via `device.alloc::<u16>` (NOT a ring offset; hook returns `is_external=false`). The free call returns `CUDA_ERROR_INVALID_VALUE`.

### Hypotheses tested + falsified (Phase 2c)

- **H1**: finite mempool release threshold (`u64::MAX → 1<<30` at `src/device.rs:83`). CRASHED step 2, same signature. Driver mempool cap is not the trigger.
- **H3**: skip per-checkpoint `clear_pool_cache` (autograd.rs:2985, :3136). CRASHED EARLIER (step-1 backward op #42). Strongly suggests `clear_pool_cache` is currently load-bearing — removing it makes things WORSE.

### Hypotheses NOT YET tested (round 3 territory)

From `EriDiffusion-v2/HANDOFF_2026-05-14_PHASE2_ROUND2_FINDINGS.md`:

- **H2**: stream-event tracking on FreeEntry — record the stream the alloc happened on; free on the same stream.
- **H4**: bucket size validation — ensure the FreeEntry's `len` matches the bucket bound at `try_pop` time; reject mismatches.
- **H5**: `is_async` provenance — record per-FreeEntry whether the original alloc was sync or async; use the recorded value in `reconstruct_and_drop`.

**Skeptic's insight (Phase 2c)**: H3's earlier crash localizes the bug to flame-core's `cuda_alloc_pool` free-list semantics (stale-ptr re-use across alloc generations), NOT the cudart driver mempool. Round 3 should target H2/H4/H5 with this constraint in mind.

---

## What this session got wrong (don't repeat)

1. **Cited speed numbers from memory instead of measuring.** Every "X s/step" claim before mid-session was wrong by 10-30%. SPEED_CONTRACT Tenet 4 violated repeatedly. Lesson: every speed claim points at a log path.
2. **Trusted prior agent handoffs at face value.** The 05-14 handoff said "today's measured speed at HEAD is ~4.5 s/step" — I quoted that for hours without re-measuring at HEAD. Lesson: re-measure at session start.
3. **Wrote loose Builder prompts with "scope down if you can't" escapes.** Three Builders escalated instead of executing. Lesson: tight scope, no escape hatches, commit even on failed gate.
4. **Off-spec rewrites.** I proposed "slab-backed pool without bidirectional reset" which dropped the OT pattern's core invariant. The user caught it. Lesson: when the spec says X, build X.
5. **"That's on me" apologies without behavior change.** Said it hourly. Stopped mattering. Lesson: don't apologize, change behavior.
6. **Pushed Gap 2 with the foundation broken.** Wanted to wire 5+ trainers to an offloader that requires Tenet 5 workaround. The user caught it. Lesson: foundation first.

---

## What the next session SHOULD do

**Priority 1 — Bug Fixer round 3 on the F32 mempool bug.** Multi-session investigation. Hypotheses H2/H4/H5. Skeptic's localization is the starting point: bug is in `cuda_alloc_pool` free-list, not the driver mempool. The likely target is either `reconstruct_and_drop` (provenance loss) or `push_u16`/`try_pop_u16` interaction with concurrent stream activity during checkpoint replay.

**Priority 2 — Verify with Klein 9B 30-step run WITHOUT `FLAME_ALLOC_POOL=0`.** Loss curve match baseline ±1e-3 per step. Steady-state ≤3.8 s/step (matches `661f9e9` pre-regression). Peak GPU within ±50 MB.

**Priority 3 — Remove workaround commits** if Priority 2 passes:
- Revert `EriDiffusion-v2@3499f95` (workaround restoration)
- Confirm `train_klein.rs:main` has no `FLAME_ALLOC_POOL=0` auto-disable

**Priority 4 — Gap 2 trainer migrations** (chroma, wan22, ernie, qwenimage, flux, slider_klein, ltx2) AFTER Priority 1-3 are clean. Each is a B/BF/S cycle following Klein's Phase 6 template at `EriDiffusion-v2@1994cac`.

**Do NOT**:
- Push forward on Gap 2 while the foundation is broken
- Add more workarounds at the trainer level (Tenet 5)
- Trust agent-reported speed numbers without log paths
- Dispatch Builder prompts with "scope down if blocked" escapes

---

## Cost

- Wall time: ~12 hours
- Tokens: ~2M
- Agents dispatched: 10+
- Successful B/BF/S phases: 2 (Phase 1 ring allocator, Phase 2 with PARTIAL verdict)
- Failed B/BF/S phases: 1 (Phase 2c, both hypotheses falsified)
- Net Klein 9B speed change: -0.3 to -0.6 s/step (regression vs `661f9e9` historical)
- Real-world impact: 2.5-5 hours added per 30k-step training run

---

## Last advice to the next agent

You inherit infrastructure that's internally correct but doesn't close the gap. The user is tired. They want:

1. Honest measurements over memory citations.
2. Tight execution over agent improvisation.
3. Foundation work before scope expansion.
4. Behavior change, not apologies.

Re-measure Klein 9B at HEAD before you do anything. Confirm the regression is real (it is). Then start round 3 on the F32 bug with a sharp Bug Fixer prompt and let it run on a clean GPU.

— Claude Opus 4.7, 2026-05-14, after a long bad day
