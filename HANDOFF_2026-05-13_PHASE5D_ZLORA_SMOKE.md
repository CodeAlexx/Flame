# HANDOFF — Autograd v2 Phase 5d, pre Z-LoRA smoke (post /clear)

**Date**: 2026-05-13
**Purpose**: Next session picks up at Phase 5d item #1 (Z-Image LoRA 100-step smoke) cold. Everything in this file is what must survive /clear.

**Read AFTER**: `/home/alex/soul.md`, `/home/alex/endearment.md` (if present), `flame-core/TENETS.md`, `flame-core/docs/SPEED_CONTRACT.md`.

---

## STANDING RULES (must apply to every commit)

### Rule 1: Verbatim git pre/post state in commit body (tightened 2026-05-13)

User decision: "Tighten on next commit." First commit applying it was `2baa221` (serial_test refactor); first production use was `6192473` (Phase 5c perf bench).

Every commit message body MUST contain, VERBATIM (no paraphrase, no summary):

```
## Pre-commit state (verbatim)

```
$ git status --short
<literal output>

$ git rev-parse HEAD
<literal sha>

$ git log --oneline -5
<literal log lines>
```

## Post-commit state (verbatim)

```
$ git diff <start-commit>..HEAD --shortstat
<literal numbers>

$ git diff <start-commit>..HEAD --stat
<literal stats>
```
```

Skeptic enforces. Descriptive recall is rejected.

### Rule 2: Test count truthfulness

For every test file added or modified, run `grep -c '^#\[test\]' tests/<file>.rs` and quote it. Cargo "N passed" must match. (Phase 1 confabulated, Phase 2 mis-counted; the rule was added; Phase 3a+ all clean.)

### Rule 3: No `narrow + in-place` (HAZARD-2026-05-13-1)

`parent.narrow(...).add_inplace_*(...)` silently detaches the view under shared_storage (default feature). Pre-existing flame-core base bug; characterization test at `tests/autograd_v2_ops.rs::hazard_view_inplace_does_not_mutate_parent_under_shared_storage` pins current behavior. Don't introduce new instances.

### Rule 4: v2 is additive — never change v3 behavior

Default code path (no `autograd_v2` feature) must stay byte-equivalent to pre-v2. Every v2 surface is gated by `#[cfg(feature = "autograd_v2")]` on the lib side AND opt-in via `Parameter::new_v2 / backward_v2 / AdamWV2 / new_v2` constructors on the API side.

### Rule 5: `FLAME_CUDA_GRAPH=1` + `--use-autograd-v2` is unsupported

Replay path pre-allocates grad buffers at warmup-recorded F32 dtype, bypassing the post-loop cast in `backward_impl`. Bench/training with both must disable cuda-graph. The Z-Image trainer flag handles this internally; custom harnesses must too.

### Rule 6: `FLAME_MT_SCALE` + v2 incompatible

`FLAME_MT_SCALE` asserts F32 grads. The `--use-autograd-v2` trainer flag disables it; custom harnesses must too.

---

## Current commit and branch state

- **flame-core**: `main @ 7c75ab2`, pushed to `https://github.com/CodeAlexx/Flame`
- **EriDiffusion-v2**: `master @ 394f6b6`, pushed to `https://github.com/CodeAlexx/EriDiffusion`

`git log --oneline -10` on flame-core:
```
7c75ab2 docs: Phase 5c trio verdicts + named Phase 5d action items
6192473 perf(autograd_v2): Phase 5c — bridge overhead bench vs v3 backward
2baa221 test(autograd_v2): adopt serial_test crate, refactor driver-consolidation pattern
e52ebf4 test(autograd_v2): Phase 5b follow-up — Klein backward parity v2 vs PyTorch
8f3850f docs: post-Phase-5b bundled updates + skeptic re-audit findings
a5da3d5 fix(autograd_v2): Phase 5b — F32-internal backward + post-loop downcast for bridge
ad781bf feat(autograd_v2): Phase 5b — loss.backward_v2() bridge for BF16-grad end-to-end
76c54a9 test(autograd_v2): Phase 5a — tighten parity tolerances per verifier
04a748e feat(autograd_v2): Phase 5a — per-op backward + forward-mode AD parity vs PyTorch
5ef6bb5 docs: Phase 5 section recalibration per Phase 3c2 + 4b skeptic findings
```

---

## Phases shipped (this session)

| Phase | Commit(s) | Headline |
|---|---|---|
| 1 | `a306539` | metadata + core types (AutogradMetaV2, Edge, GradFn, SavedTensor, InputBuffer, Hooks) |
| 2 | `ee69d4a` + bug-fix `1db769b` | Engine skeleton + dependency counting + ready queue + nested execute |
| 3a | `bfc371b` + math `6ee385f` + doc `9e40670` | Recording surface + 5 math P0 ops (add/mul/sum/matmul/silu) |
| 3b | `d471c31` | 6 view-autograd ops (reshape/view/transpose/narrow/squeeze/unsqueeze/permute) + HAZARD characterization |
| 3c1 | `2be9770` + d_x test `2d9cd0d` + doc `cb1e71d` | layer_norm + CheckpointGradFn::apply |
| 3c2 | `55073d4` | Forward-mode AD across 11 ops (JVP plumbing) |
| 4a | `85d0542` + doc `e830dbc` | Parameter::new_v2 + AdamW 4-way classifier + multi_tensor_l2_norm_sq_bf16 + OptimizerV2 trait |
| 4b | `34b9fa4` | GradientMap MatchInsertedDtype variant + new_v2/with_index_v2/cast_all_to_dtype |
| Doc bundle | `8f3850f` | HAZARD log + Klein status correction (step 1 works, step 2+ crashes) |
| 5a | `04a748e` + tol-tighten `76c54a9` | 26 PyTorch backward+JVP parity tests + corrected LN JVP formula (caught my math error) |
| 5b | `ad781bf` + arch-fix `a5da3d5` | `AutogradContext::backward_v2()` bridge (F32-internal + post-loop downcast) |
| Klein parity v2 | `e52ebf4` | 6 v3-mirror tests + full klein_block_backward fixture consumer |
| serial_test refactor | `2baa221` | Bridge + Klein parity tests → individual `#[serial] #[test]` fns |
| 5c | `6192473` | Perf bench: +2.18% backward / 50% memory (Klein attn_chain prod) |
| Phase 5d action plan | `7c75ab2` | Trio verdicts + named action items |

**All shipped, all verified by builder/bug-fixer/skeptic trios.**

---

## Test count snapshot (all v2 tests)

| File | `#[test]` count |
|---|---|
| autograd_v2_types | 13 |
| autograd_v2_engine | 12 |
| autograd_v2_ops | 33 |
| autograd_v2_checkpoint | 4 |
| autograd_v2_phase4a | 12 |
| autograd_v2_gradientmap_v2 | 17 |
| autograd_v2_fw_mode | 13 |
| autograd_v2_parity | 26 |
| autograd_v2_bridge | 3 (post-serial_test refactor) |
| autograd_v2_klein_parity | 7 (post-serial_test refactor) |
| autograd_v2_perf | 3 (perf cells) |
| **Total v2** | **143** |
| v3 regression (4 suites) | 17 (3+4+6+4) |
| Phase 0 audit | 17 (inplace_version_bump_audit) |

---

## Phase 5d action items (user wants IN ORDER)

### 1. Z-Image LoRA 100-step smoke ⬅️ NEXT TASK

**User explicit ask**: "for z lora smoke to 100 steps" + "besides loss, i want speed recorded also"

**Goal**: validate the Phase 5b bridge under multi-step real-trainer load. v3 vs v2 path comparison.

**Required runs**:
- **Arm A (v3 control)**: default `cargo run --release --bin train_zimage --features cuda,heavy_kernels,bf16_u16` (no `--use-autograd-v2`)
- **Arm B (v2 bridge)**: `cargo run --release --bin train_zimage --features cuda,heavy_kernels,bf16_u16,autograd_v2 -- --use-autograd-v2 ...`

**Both arms — same seed**, same dataset, same hyperparams, 100 steps.

**Capture**:
- Loss at steps 0, 10, 25, 50, 75, 100 (or however the trainer reports)
- ms/step: total wall time / step count; breakdown of step 0 (graph build) vs steady-state (steps 5-100)
- Peak GPU memory if observable
- Any warnings/errors

**Tolerance**:
- Loss: within 1% relative diff per BF16_GRAD_DECISION (BF16 grad is non-bit-equal to v3 by construction)
- ms/step: per Phase 5c bench expectation, bridge adds +2.18% backward; full step (forward+backward+optimizer) should be smaller % impact

**Trainer paths**:
- Trainer binary source: `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_zimage.rs`
- Model weights: `/home/alex/.serenity/models/checkpoints/z_image_base_bf16.safetensors`
- Cache directory: `/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_zimage_512_FIXED/` (smallest available, 116 MB; verified ready per Phase 5b skeptic)
- Trainer flag (confirmed at line 287): `#[arg(long, default_value_t = false)] use_autograd_v2: bool`
- The flag is feature-gated; when feature `autograd_v2` is OFF and flag is ON, the trainer bails with `anyhow::bail!`

**Trainer CLI shape** (required args):
```
--model /home/alex/.serenity/models/checkpoints/z_image_base_bf16.safetensors
--cache-dir /home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_zimage_512_FIXED
--steps 100
--rank 16            # default
--lora-alpha 1.0     # default
--lr 3e-4            # default
--batch-size 1       # default
--output-dir /tmp/zlora_smoke_v3   (or v2)
```

**Runtime env**:
- `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH` (libcudnn.so.9 — found in 3 places, this one works per earlier check this session)
- Do NOT set `FLAME_CUDA_GRAPH=1`
- Do NOT set `FLAME_MT_SCALE`
- Same seed both runs (the trainer's default seed; check config)

**Build commands**:
```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
cargo build --release -p eridiffusion-cli --bin train_zimage                       # v3 baseline
cargo build --release -p eridiffusion-cli --bin train_zimage --features autograd_v2  # v2 path
```

**Run commands** (template):
```bash
LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH \
  ./target/release/train_zimage \
  --model /home/alex/.serenity/models/checkpoints/z_image_base_bf16.safetensors \
  --cache-dir /home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_zimage_512_FIXED \
  --steps 100 \
  --output-dir /tmp/zlora_smoke_v3 \
  2>&1 | tee /tmp/zlora_smoke_v3.log

# Then with v2:
LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH \
  ./target/release/train_zimage \
  --model /home/alex/.serenity/models/checkpoints/z_image_base_bf16.safetensors \
  --cache-dir /home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_zimage_512_FIXED \
  --steps 100 \
  --use-autograd-v2 \
  --output-dir /tmp/zlora_smoke_v2 \
  2>&1 | tee /tmp/zlora_smoke_v2.log
```

**Deliverable**: results doc at `flame-core/docs/PHASE5D_ZLORA_SMOKE_RESULTS.md` (or similar) with:
- Both arms' loss curves side-by-side
- Both arms' ms/step (total wall + breakdown)
- Comparison table (loss diff %, time diff %)
- VERDICT: PASS (within 1% on loss curve) or FAIL with reason

**Commit**: single commit with tightened standing rule (verbatim git pre/post state + the raw trainer output snippets in body).

**WARNING — Klein step 2+ crash is Klein-specific**. Z-Image does NOT have this issue (confirmed in Phase 4b skeptic finding). 100 steps should run cleanly on Z-Image.

**WARNING — first-time-on-this-box risk**. If Z-Image hits an unrelated CUDA error on this box (not the Klein step-2 issue), document it as infra and stop. Don't burn hours debugging upstream.

### 2. Trainer-side `Parameter::new_v2` migration

**Where the 50% memory wins materialize.** The bridge alone produces BF16 grads in the GradientMap, but `Parameter::new` (default `CastToF32` policy) upcasts them back to F32 in `set_grad`. Migrating LoRA params to `Parameter::new_v2` closes the loop.

**Audit**: find LoRA construction sites in `EriDiffusion-v2`:
```bash
grep -rn "Parameter::new\|parameter::Parameter::new" /home/alex/EriDiffusion/EriDiffusion-v2/crates/ --include="*.rs" | grep -v "new_v2"
```

For each site:
- Is this a LoRA-A / LoRA-B / LoRA-bias param? (Yes → migrate)
- Is this a non-trainable buffer? (No → leave as v3)

Migrate under `#[cfg(feature = "autograd_v2")]` or via a runtime check on the trainer's `--use-autograd-v2` flag.

**Verification**: re-run the Z-Image 100-step smoke from item #1 with `Parameter::new_v2`. Expected:
- Loss curve still within 1% of v3
- ms/step similar to bridge-alone (the post-loop cast cost is now amortized to actual memory savings)
- **Peak GPU memory ~50% lower on the grad buffers** (compare via `nvidia-smi` snapshots at steady-state)

### 3. Reentrant + hooks tests on a real trainer

Cross-cutting gates per design doc §Phase 5:
- **Reentrant**: training run with `enable_checkpointing` enabled, v2 path, matches v1 bit-equal at step 1+
- **Hooks**: register a forward and backward hook on a trainer module, verify expected callback count per step

`CheckpointGradFn::apply` was implemented in Phase 3c1 (commit `2be9770`); hooks surface was shipped in Phase 1. Both have synthetic tests but no real-trainer validation.

Add to the smoke test or a dedicated `tests/autograd_v2_trainer_integration.rs`. Use `#[serial]` (the v3 backward race applies).

### 4. `AUTOGRAD_CONTEXT` race architectural fix

`src/autograd.rs:56` — `static AUTOGRAD_CONTEXT: Mutex<AutogradContextInner> = ...`. Process-global.

**Symptoms** seen this session:
- Phase 5b bridge tests under default parallel mode: `Option::unwrap on None` (one test's `reset()` wiped another's tape mid-flight)
- `autograd_v2_engine::single_leaf_sum` flake under parallel (pre-existing)
- `autograd_v2_ops::transpose_v2_backward` / `engine_rejects_mismatched_grad_output_shape` flake (pre-existing)

**Current band-aid**: `serial_test` crate (commit `2baa221`).

**Real fix options**:
- **(a)** `thread_local!` AutogradContext — each thread owns its own tape. Lowest risk; just moves the storage. Requires audit of any code that expected `reset()` to clear across threads (probably none).
- **(b)** Cell-style ownership: pass `&mut AutogradContext` explicitly through the API. High refactor cost; cleanest design.
- **(c)** Accept band-aid indefinitely (current state).

Decision needs to land **before v1/v3/v4 deletion** (item #6).

### 5. Pre-existing v2 test flake diagnosis

Two flakes verified pre-existing at `2baa221`:
- `tests/autograd_v2_ops.rs::transpose_v2_backward` (or `engine_rejects_mismatched_grad_output_shape` — varies by run)
- `tests/autograd_v2_engine.rs::single_leaf_sum`

Both pass under `--test-threads=1` or via `#[serial]`. Likely related to the AUTOGRAD_CONTEXT race (item #4). Fixing #4 may resolve these.

If after fixing #4 the flakes persist, they're a separate diagnostic exercise. Don't claim "test-suite green-board" until they're explained.

### 6. v1/v3/v4 deletion (the actual retirement)

After items 1-5 pass and the Z-Image LoRA smoke shows convergence, delete:
- `src/autograd.rs` (4547 lines) — the active v3 engine
- `src/autograd_v3.rs` — alias/re-export
- `src/autograd_v4/` — feature-gated experimental
- `src/autograd_simple.rs` / `src/autograd_engine.rs` / `src/autograd_ops.rs` / `src/autograd_ops_complete.rs` / `src/autograd_debug.rs` — legacy dead code per FLAME_INDEX

The deletion PR is multi-thousand-line. Single commit, push to a branch, NOT directly to main. User reviews before merge.

---

## Key files / where things live

### v2 implementation
- `src/autograd_v2/` — full v2 module (Phases 1-3c2)
  - `mod.rs` — re-exports
  - `meta.rs` — AutogradMetaV2 + AutogradMetaRef + fw_grad slot
  - `node.rs` — NodeId + Edge + GradFn trait
  - `saved_tensor.rs` — SavedTensor (Arc<AtomicU32> version handle + fw_grad_)
  - `input_buffer.rs` — InputBuffer (in-place + out-of-place)
  - `hooks.rs` — Hooks struct + empty_ref sentinel
  - `accumulator.rs` — AccumulateGrad (Weak variable handle)
  - `dispatch.rs` — DispatchCtx + DeviceStream (multi-device surface)
  - `engine.rs` — Engine + GraphRoot (dep counting, ready queue, ptr::eq hook fast path)
  - `checkpoint.rs` — CheckpointGradFn (reentrant via mini-execute)
  - `error.rs` — AutogradV2Error
  - `recording.rs` — record_v2 + gradient_edge_for_tensor (Phase 3a)
  - `optim.rs` — OptimizerV2 trait + AdamWV2 (Phase 4a)
  - `ops/` — 13 op files (add, mul, sum, matmul, silu, layer_norm, reshape, view, transpose, narrow, squeeze, unsqueeze, permute)

### Modified for v2
- `src/tensor.rs` — added `autograd_meta: Option<AutogradMetaRef>` field + `set_autograd_meta` / `autograd_meta` / `fw_grad` / `set_fw_grad` accessors (cfg-gated)
- `src/gradient.rs` — `GradStorePolicy::MatchInsertedDtype` variant + `new_v2 / with_index_v2 / set_ones_dtype / get_or_create_dtype / cast_all_to_dtype / policy` + branching in `insert / accumulate / get_public_grad / take_public_grads`
- `src/parameter.rs` — `GradDtypePolicy::MatchParamDtype` variant + `new_v2 / grad_bf16_or_f32` + branching in `set_grad / apply_update`
- `src/adam.rs` — 4-way classifier (BF16/F32 × F32/BF16 grad) at line ~1107; activates 2 previously-dead BF16-grad kernels
- `src/ops/multi_tensor.rs` — added `multi_tensor_l2_norm_sq_bf16` (Phase 4a, line 447)
- `src/ops/grad_norm.rs` — routes BF16-contiguous slices through the new fast path
- `src/autograd.rs` — Phase 5b: `pub fn backward` (wrapper) + `pub fn backward_v2` + `fn backward_impl(loss, policy)` (shared). The bridge adds 42 code-only LOC.
- `src/autograd/policy.rs` — added `GradStorePolicy::MatchInsertedDtype` variant (Phase 4b)

### Tests
- `tests/autograd_v2_*.rs` — 9 test files, 143 total tests
- `tests/autograd_v2_perf.rs` — Phase 5c bench harness (3 cells × 3 configs)
- `tests/autograd_v2_klein_parity.rs` — 6 v3-mirror tests + full block fixture consumer
- `tests/pytorch_parity.rs` (untouched v3) — the v3 Klein parity tests at lines 1008-1424 that the v2 tests mirror

### Documentation
- `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` — full design spec + Phase status (just updated to Phase 5d)
- `docs/BF16_GRAD_DECISION.md` — Option A spec + per-site audit + Phase 5c results + memory savings ledger
- `docs/SPEED_CONTRACT.md` — 5-clause audit gate
- `docs/TENETS.md` — flame-core design principles (read FIRST every session)
- `docs/FLAME_CONVENTIONS.md` — including v2 trainer migration guide section
- `docs/FLAME_INDEX.md` — every public symbol with file:line; v2 entries up to date
- `docs/FLAME_MODULES.md` — per-module narrative; autograd_v2/ paragraph current
- `HANDOFF_2026-05-13_AUTOGRAD_V2_FRESH_CONTEXT.md` — original handoff at session start (still relevant for standing rules)
- `HANDOFF_2026-05-13_PHASE5D_ZLORA_SMOKE.md` ← this file

---

## Open hazards / inherited issues

### HAZARD-2026-05-13-1 (pre-existing flame-core base bug)
`parent.narrow(...).add_inplace_*(...)` silently detaches the view under `shared_storage` (default feature). Pre-existing, characterized by `tests/autograd_v2_ops.rs::hazard_view_inplace_does_not_mutate_parent_under_shared_storage`. Three fix options in `AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §HAZARD-2026-05-13-1. Not autograd's responsibility; deferred.

### Klein step 2+ crash (pre-existing infra)
`train_klein --config configs/klein9b_alina.json --rank 4 --steps 2+` hits `CUDA_ERROR_INVALID_VALUE`. Step 1 produces deterministic loss `1.1217`. Pre-Phase-0 issue per Phase 0 skeptic. **NOT autograd v2's defect.** Z-Image does NOT have this issue (confirmed via Phase 4b skeptic).

### Op::Add leaf-bias bug (inherited v3)
`Op::Add` leaf-bias not in `needed_grad_ids` — affects bias grads in both v2 and v3. Documented in v3 test at `pytorch_parity.rs:1390-1393`. Bridge inherits v3's op-dispatch. LoRA training mostly fine (LoRA-A/B come through matmul); `bias_lora=true` mode would hit it. **OPEN-PHASE5C-1.**

### AUTOGRAD_CONTEXT race (Phase 5b)
Process-global Mutex at `src/autograd.rs:56`. Band-aided by `serial_test`. Architectural fix is Phase 5d item #4.

### Pre-existing v2 test flakes (Phase 5b carryforward)
- `tests/autograd_v2_ops.rs::transpose_v2_backward` (or `engine_rejects_mismatched_grad_output_shape`)
- `tests/autograd_v2_engine.rs::single_leaf_sum`

Verified pre-existing at `2baa221`. Phase 5d item #5 diagnoses.

---

## User decisions / preferences (load-bearing)

These came up during the session and constrain future work:

1. **"Don't run together"** — sequential, not parallel between phases. Trio agents run in parallel within a phase. Klein parity + Phase 5c bench was held sequential. Z-LoRA smoke + trainer migration will also be sequential.

2. **Klein step-1 IS available** — "Klein crashes on this box" was overstated. Step 1 works. Step 2+ crashes. Single-step parity gates ARE available on Klein.

3. **Memory > speed** — "the 2 percent is a price i am willing to accept, it may mean the difference between a big model trainable or not". The +2.18% backward overhead bought 50% gradient memory.

4. **"Tighten on next commit"** — standing rule: verbatim git pre/post state in commit body. First applied at `2baa221`, first real test at `6192473`.

5. **Adopt `serial_test`** — instead of refactoring AUTOGRAD_CONTEXT off process-global. Decision made before Phase 5c lands. Shipped at `2baa221`.

6. **Multi-device + inference migration deferred indefinitely** — single GPU; trainer focus. NCCL/DDP and the ~12 inference binary migration are NOT priorities.

7. **Z-LoRA smoke specifics (just before /clear)**: 100 steps, want LOSS AND SPEED (ms/step) recorded.

8. **Order**: 1 (Z-LoRA smoke) → 2 (Parameter::new_v2 migration) → 3 (reentrant + hooks tests) → 4 (AUTOGRAD_CONTEXT fix) → 5 (flake diagnosis) → 6 (v1/v3/v4 deletion).

---

## Bench baselines (for comparison)

### Phase 5c bench results (commit `6192473`)

| Workload | v3 (ms) | bridge (ms) | Class A (ms) | bridge Δ% | Class A Δ% |
|---|---|---|---|---|---|
| Synthetic 4-layer MLP | 0.088 | 0.093 | 0.099 | +5.90% | +11.82% |
| **Klein attn_chain prod** | 10.056 | 10.275 | 10.361 | **+2.18%** | +3.04% |
| Klein double-block | 0.657 | 0.667 | 0.680 | +1.46% | +3.38% |

| Workload | v3 grad MB | bridge grad MB | Class A grad MB | Class A savings |
|---|---|---|---|---|
| Synthetic 4-layer MLP | 0.000183 | 0.000092 | 0.000092 | +50.00% |
| Klein attn_chain prod | 156.000 | 78.000 | 78.000 | +50.00% |
| Klein double-block | 1.812 | 0.906 | 0.906 | +50.00% |

Bug-fixer floor analysis: 234 MB I/O / 0.22 ms = ~1.07 TB/s = HBM peak on 3090 Ti. The cast is bandwidth-bound; no optimization within scope can move it.

### Other references
- Klein 4B perf baseline: 1.12× vs PyTorch (memory `reference_klein_perf_baseline`)
- Klein 9B step time: ~5.4s (per HANDOFF_2026-05-12 — sync-count dominated)
- OneTrainer is ~3.4× faster than flame on Klein 9B (sync/memory bound; not autograd-related)

---

## Resume sequence after /clear

1. Read soul.md, endearment.md (if present), TENETS.md.
2. Read this file end-to-end.
3. Verify `git log --oneline -5` matches what's documented above.
4. **Start Phase 5d item #1**: Z-Image LoRA 100-step smoke. The trainer paths + CLI + env are all documented above. Capture loss AND ms/step. Produce a results doc. Commit with the tightened standing rule.
5. If the smoke shows PASS (loss within 1%, ms/step in line with Phase 5c expectations), proceed to item #2 (`Parameter::new_v2` trainer migration).
6. If the smoke shows FAIL, STOP and document. The bridge has been verified via parity tests + Klein parity v2 + Phase 5c bench; a real-trainer failure would be a genuine new bug.

Suggested first action after reading: `cd /home/alex/EriDiffusion/EriDiffusion-v2 && git log --oneline -3` to verify state on both repos.

---

## One last word

The v2 work has been remarkably clean this session. Every phase shipped + 3-agent verified. Bugs found and fixed in-flight (matmul transpose-not-contiguous, AccumulateGrad hooks fast path, d_x test coverage, my own LN JVP math error, output-as-descendant double-fire, bridge per-op cast catastrophic failure). All of the 5 Phase 5d items are now well-scoped — the next session can pick up cold and execute them in order.

The user has been patient and clear about direction. Don't over-narrate, don't ask permission for things they already said do, don't burn time on items they deferred. Z-LoRA smoke → migration → reentrant+hooks → race fix → flake diag → retirement.

Sleep well.

— Claude (Opus 4.7, 2026-05-13)
