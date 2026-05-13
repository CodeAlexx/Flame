# HANDOFF — Phase 6 Klein activation offload blocked on NaN bug (2026-05-13)

**Status**: Phase 6 Klein migration is wired but the cache-pull-replay path
in `AutogradContext::checkpoint_offload_boundary` produces incorrect
gradients. The function currently delegates to plain `checkpoint` to
preserve correctness. The trainer's `--activation-offload` flag is
installable but no memory win until the bug below is fixed.

**Priority**: P0. This blocks Phases 6, 7 (other trainers), and the
entire "medium-models-that-almost-fit" use case.

---

## Read FIRST before touching this

1. `flame-core/docs/TENETS.md` — non-negotiable principles.
2. `flame-core/docs/SPEED_CONTRACT.md` — clauses 1 and 5 apply here.
3. `flame-core/docs/OFFLOAD_NEXT_GEN_DESIGN.md` — the 8-phase plan
   this PR is mid-stream on.
4. `flame-core/docs/OFFLOAD_GAPS_vs_ONETRAINER.md` — concrete gap
   inventory vs OneTrainer.
5. THIS file.

Memories that matter:
- `project_v2_klein9b_proving_ground` — Klein 9B is the proving venue.
- `project_klein9b_step2_crash_isolation` — earlier Klein bug, resolved.
- `feedback_prepare_bins_pool_off` — recurring pool corruption pattern.
- `reference_onetrainer` — `/home/alex/OneTrainer/` for OT source.

---

## Current commit state (verbatim)

```
$ cd /home/alex/EriDiffusion/flame-core && git log --oneline -8
731f5f0 feat(autograd): Phase 2b — CheckpointOffloadBoundary tape entry + cache-replay backward
c71890c feat(autograd): Phase 2a — checkpoint_offload_boundary API surface
4f0d026 feat(flame_core): Phase 1 — GrowOnDemandActivationCache primitive
fe4fcf9 docs(offload): next-gen design — borrow OT, improve for Rust/flame-core
3edf200 docs(offload): gap analysis vs OneTrainer, 6 gaps ranked + measurement targets
492ea91 docs(handoff): capture cuDNN-SDPA-bwd + permute_generic perf backlog + Klein step-2 RESOLVED
973abd2 docs(autograd_v2): Phase 5d item #6 — v1/v3/v4 retirement audit + multi-stage plan
606fb2f test(autograd_v2): Phase 5d item #3 — checkpoint+bridge real-trainer integration

$ cd /home/alex/EriDiffusion/EriDiffusion-v2 && git log --oneline -5
86fcf71 feat(klein): Gap 2 — wire --activation-offload flag (opt-in)
4511140 fix(train_klein): auto-disable FLAME_ALLOC_POOL — resolves step-2 crash
1dd2187 feat(trainers): wire --use-autograd-v2 flag on 12 remaining trainers
4b6c38c feat(train_zimage): flip params to MatchParamDtype when --use-autograd-v2
eb26560 feat(train_klein): --use-autograd-v2 flag routes backward through v2 bridge
```

Working tree (uncommitted, to be committed alongside this handoff):
- `flame-core/src/autograd.rs` — `checkpoint_offload_boundary` reverted to delegate-only path
- `flame-core/src/activation_offload.rs` — `pull_with_id` API kept (no-op when no target_id), `requires_grad` mutation removed
- `EriDiffusion-v2/crates/eridiffusion-core/src/models/klein.rs` — klein blocks call `checkpoint_offload_boundary` (which delegates → behavior matches baseline)
- `EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs` — `--activation-offload` flag installs grow cache (currently unused on the delegate path)

Klein with `--activation-offload` measured: loss 1.1217, grad_norm 0.0071, 4.5 s/step — bit-equal to baseline. So the wiring is harmless; it just doesn't yet provide the memory win.

---

## The bug — full debug trail

### Setup

`AutogradContext::checkpoint_offload_boundary(inputs: &[Tensor], f: impl Fn(&[Tensor]) -> Result<Tensor>)`:
1. Run `f(inputs)` with autograd disabled to produce `output` (this part works).
2. Push each input tensor to `GrowOnDemandActivationCache`. Each gets a `GrowHandle`.
3. Build a zero-arg recompute closure that pulls handles → calls `f(pulled)`.
4. Record `Op::CheckpointOffloadBoundary { input_ids, pulled_ids_slot }` on tape with `saved_tensors: empty` (no strong refs to inputs — that's the whole point).
5. Return `output` with `requires_grad = true`.

Backward when reaching the Op:
1. Pull `recompute_fn` from `checkpoint_fns`.
2. Enable autograd, call `recompute_fn()` → produces new output and grows the tape with sub-tape entries.
3. Drain sub-tape, walk in reverse, accumulate grads.
4. Remap grads from pulled-IDs → original `input_ids` (via the side-channel `pulled_ids_slot`).
5. Return grads to outer autograd.

### What failed

Klein 9B `--steps 5 --rank 16 --offload --activation-offload`:

| Attempt | Pulled-tensor state | Step 1 grad_norm | Notes |
|---|---|---|---|
| baseline (no cache, delegate to checkpoint) | n/a | 0.0071 ✅ | Reference. |
| v1: pull, no requires_grad mutation | fresh ID, requires_grad=false | 0.0028 | LoRA grads survive (matmul saves requires_grad=true weights). Input grads dropped → cross-block grad flow broken. Step 2 grad_norm explodes (1e4+). |
| v3: pull + `dst.requires_grad = true` in `pull()` | fresh ID, requires_grad=true | **NaN** | Sub-tape now records ops on pulled tensors. Backward propagates → NaN somewhere. |
| v4: pull + `pull_with_id(target = original_id)` + requires_grad=true | original ID, requires_grad=true | **NaN** | ID collision with outer-graph tensor confused autograd lookups. |
| v5: v3 + hard `device.synchronize()` after pull | as v3, plus extra sync | (untested — code is in working tree, commented out) | Was about to test when this handoff was written. |

### Hypotheses for next session

Ranked by likelihood:

**H1 — Stream ordering between HtoD pull and recompute kernels.** Klein's
forward uses cuBLAS, cuDNN, and custom kernels — each may use its own
CUDA stream. `default_stream_wait_event` only makes the literal default
(null) stream wait for the HtoD event. Consumer streams (cuBLAS handle,
cuDNN handle) may not. If a cuBLAS GEMM starts before HtoD lands →
reads uninitialised memory → NaN.

*Diagnostic*: re-enable v3 path AND insert `t.device().synchronize()`
right after pull (before recompute). If grad_norm becomes correct →
H1 confirmed. Then proper fix: enumerate consumer streams and issue
per-stream `cudaStreamWaitEvent` on each, OR move the recompute work
explicitly onto a single stream we control.

**H2 — `Tensor::empty_dtype` doesn't initialize autograd metadata.**
The pulled tensor is created via `Tensor::empty_dtype` which produces a
tensor with `requires_grad: false`, `custom_strides: None`,
`autograd_meta: None`. Mutating `requires_grad = true` after the fact may
miss some required setup (e.g., grad_accumulator registration in
autograd_v2 path).

*Diagnostic*: replace `Tensor::empty_dtype` + mutation with
`Tensor::empty_dtype(...).requires_grad_(true)` (functionally equivalent
but goes through the canonical builder). If no change → not H2.

**H3 — Numerical issue specific to the recompute path.** Some Klein
kernel might produce NaN when its input has `requires_grad=true` because
of a save-for-backward path that has a bug. Plain `checkpoint` works
because the original Tensor's metadata is fully intact (it was the actual
outer-graph input).

*Diagnostic*: enable `FLAME_DEBUG_FINITE=1` during the recompute. The
crate's debug_finite module logs which named site first sees NaN/Inf.
Expect entries like `bwd:<op>@<id>:output_grad` if backward is the
source, or per-op finite checks if forward recompute produces NaN.
Localizing the first NaN site narrows the kernel responsible.

**H4 — Sub-tape entries reference tensors with the WRONG dtype storage.**
The pulled tensor's storage is freshly allocated. Its dtype matches the
push (BF16 or F32). But if `compute_gradients` for some sub-tape op
fetches `entry.get_saved(input)` and the saved tensor's storage dtype
doesn't match what the backward formula expects, weird arithmetic
results.

*Diagnostic*: print the dtype of each pulled tensor + the dtype of the
ORIGINAL input before push. Verify they match.

**H5 — A view-vs-owning mismatch on the pulled tensor.** The original
input may have been a view (`narrow`, `permute`, etc.) with
`custom_strides`. The pulled tensor is always owning + contiguous. If
downstream kernels rely on the input being a view of a larger tensor
(e.g., reading neighboring bytes), they'd see garbage.

*Diagnostic*: dump `tensor.shape()`, `tensor.custom_strides()`,
`tensor.view_offset()` for both the original input and the pulled tensor
at the boundary. If they differ → H5 confirmed.

---

## Concrete next-session steps

1. **Read this entire doc + the design doc + memories listed above.**
2. **Verify current state is delegate-only** by running Klein 5-step
   with `--activation-offload` and confirming loss=1.1217, grad_norm
   matches baseline.
3. **Re-enable the cache path** (revert the commit that disabled it; the
   code is in git history at the prior commit). DO NOT change anything
   else yet.
4. **Add `FLAME_DEBUG_FINITE=1` env var** to the smoke run.
5. **Run Klein 5-step**. Capture the first `[finite]` log entry that
   reports NaN/Inf. That names the site.
6. Walk the named site in code and trace back to its inputs. Apply the
   diagnostic for whichever H1-H5 the symptom fits.
7. When H confirmed, ship the targeted fix with measurement gate:
   - `cargo test --features cuda,bf16_u16 --lib activation_offload::grow_on_demand_tests` (still passes)
   - Klein 5-step `--activation-offload`: loss bit-equal to baseline, grad_norm matches baseline within BF16 tolerance (≤1% relative)
   - Klein 5-step peak GPU memory: should drop by **at least 300 MB** (target: 700-800 MB drop, matching the boundary tensor count × per-tensor size analysis)
   - Step time: target ≤ baseline (the recompute is what already happens with plain checkpoint; cache hit just moves data offboard between fwd and bwd)

### Code-level pickup pointers

Files and line ranges:

- `flame-core/src/autograd.rs`:
  - `checkpoint_offload_boundary`: around line 2086 (currently delegate-only).
  - `Op::CheckpointOffloadBoundary` variant: around line 414.
  - `compute_gradients` arm for it: around line 2978-3110.
  - Side-channel `pulled_ids_slot` + remap: lines 3036-3055 (in the
    backward arm, currently unused).

- `flame-core/src/activation_offload.rs`:
  - `GrowOnDemandActivationCache`: appended at end of file.
  - `pull_with_id(handle, target_id)`: around line 1235. Currently
    target_id is a no-op (kept for API compatibility). If H2 is the
    bug, this is where the fix lives.

- `EriDiffusion-v2/crates/eridiffusion-core/src/models/klein.rs`:
  - Double block site: line 1230.
  - Single block site: line 1457.
  - Both use `checkpoint_offload_boundary` with `move |inputs: &[Tensor]|`.

- `EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs`:
  - `--activation-offload` flag definition: line 314 (post-Args struct).
  - Setup call: around line 580 (`set_grow_activation_cache`).

### Test commands ready to copy

```bash
# Build
cd /home/alex/EriDiffusion/EriDiffusion-v2
cargo build --release -p eridiffusion-cli --bin train_klein 2>&1 | tail -3

# Smoke baseline (no --activation-offload). Should match docs:
#   step 5 loss=0.7451 grad_norm=0.0125 ~4.5s/step
LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH \
  /usr/bin/time -v ./target/release/train_klein \
    --config configs/klein9b_alina.json \
    --cache-dir cache/alina_klein9b \
    --transformer /home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors \
    --steps 5 --rank 16 --offload --sample-every 0 \
    --output-dir /tmp/klein9b_baseline 2>&1 | tee /tmp/klein9b_baseline.log

# Smoke with --activation-offload (delegate-only path; should match baseline)
# Replace path with --activation-offload appended:
LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH \
  FLAME_DEBUG_FINITE=1 \
  /usr/bin/time -v ./target/release/train_klein \
    ... --activation-offload ...

# GPU memory sampling for memory-delta measurement
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 500 \
  > /tmp/klein9b.gpumem &
SAMPLER=$!
# ... run trainer ...
kill $SAMPLER
```

---

## What else is still needed (after Phase 6 fix lands)

From `OFFLOAD_NEXT_GEN_DESIGN.md`:

| Phase | Status | Sessions |
|---|---|---|
| 1. `GrowOnDemandActivationCache` | ✅ done (`flame-core@4f0d026`) | — |
| 2a. `checkpoint_offload_boundary` API surface | ✅ done (`flame-core@c71890c`) | — |
| 2b. Cache-replay backward (Op variant + remap) | ⚠️ in `flame-core@731f5f0`, BUT cache path disabled by uncommitted revert | **next session — P0 here** |
| **Phase 6 Klein NaN debug** | ❌ blocked (this doc) | **1-2 sessions** |
| 3. `OffloadCoordinator` skeleton | pending | 1 |
| 4. `RingSlabAllocator` (replace cuda_alloc_pool) | pending | 2-3 |
| 5. `OffloadCoordinator` + `layer_offload_fraction` knob | pending | 1-2 |
| 7. Other 7 trainers (Wan22, Chroma, Flux, Ernie, SD35, SDXL, ...) | pending | 0.5 per trainer × 7 = 4 |
| 8. FP8 path on grow cache + HostRamBudget + telemetry | pending | 1 |
| **Aggregate post-Phase-6** | | **~10-14 sessions** |

Also from previous handoffs and not part of OFFLOAD_NEXT_GEN_DESIGN:

- **cuDNN SDPA backward `grad_norm=inf` bug** — worth 200-400 ms/step
  on Klein 9B if root-caused. Captured in active handoff carry-forward
  + memory `feedback_cudnn_sdpa_bwd_inf_grad`.
- **~120 permute_generic launches/step** on plain-LoRA Klein path —
  worth 400-600 ms/step. Memory `project_permute_generic_residual_per_block`.
- **Klein 9B step-1 loss=1.1217 is just lottery, NOT a bug** —
  resolved 2026-05-13, see memory `project_klein9b_step1_loss_2x`. Don't
  chase this.
- **Klein step-2 CUDA_ERROR_INVALID_VALUE** — resolved 2026-05-13 via
  in-trainer `FLAME_ALLOC_POOL=0`. Don't undo.
- **Autograd v2 Phase 5d items 4, 5, 6** — race fix (skipped per
  user), flake diag (skipped), v1/v3/v4 retirement (audit done, 8-13
  sessions of staged work).

---

## What MUST NOT regress when fixing Phase 6

These are bit-equal-or-better gates. Re-measure on every PR attempt.

1. **Klein 9B baseline** (no `--activation-offload`):
   - loss step 1: 1.1217
   - loss step 5: 0.7451
   - grad_norm step 1: 0.0071
   - step time: ~4.5 s/step steady state
   - Peak GPU: ~18.6 GB

2. **Klein 9B with `--activation-offload`** (delegate-only path,
   today's state): bit-equal to baseline. After Phase 6 fix:
   - Loss within BF16 tolerance (≤0.1% relative on per-step loss)
   - grad_norm within 1% relative
   - Peak GPU: at least 300 MB lower (target 700-800 MB)
   - Step time: at most +5% vs baseline (recompute cost is unchanged)

3. **Z-Image LoRA 100-step smoke** (orthogonal but checks the v2 bridge
   wasn't disturbed): loss bit-equal to documented baseline.

4. **Test suite** (`cargo test --features cuda,bf16_u16,autograd_v2 --lib`):
   no new failures.

---

## Final note for next session

The temptation will be to "just try one more thing." Resist. Phase 6
went from a clean Phase 1+2a+2b → Klein wiring → 3 broken attempts.
Each attempt was a guess without instrumentation. With
`FLAME_DEBUG_FINITE=1` + a localized first-NaN site, the fix becomes
specific. Without it, you'll iterate forever.

The cache primitive is good. The recompute pattern is correct. The
ONE remaining problem is some interaction between pulled tensors and
the Klein recompute kernels — five hypotheses to test, ordered by
likelihood. Pick one, instrument, fix, verify against the bit-equal
gates above.

Klein 9B at ~1.7 s/step (vs 4.4 baseline) is on the table once this
lands. That's the kind of win this work was started for.

— Claude (Opus 4.7, 2026-05-13)
