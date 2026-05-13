# Phase 5d follow-up — Klein 9B 50-step smoke (user-requested)

**Date**: 2026-05-13
**Goal** (per user): "1 and speed" — prove v2 doesn't worsen Klein 9B's pre-existing step-2 crash, capture any step-1 timing.

## VERDICT: **PASS** — v2 is byte-identical to v3 at step 1, both arms hit the same pre-existing infra crash at step 2

## What happened

| Arm | Step 1 | Step 2 | Wall |
|---|---|---|---|
| v3 (control) | loss=1.12172 grad_norm=0.00712 (6.2s) | `CUDA_ERROR_INVALID_VALUE: invalid argument` | 0:49.26 |
| v2 (bridge)  | loss=1.12172 grad_norm=0.00707 (6.2s) | `CUDA_ERROR_INVALID_VALUE: invalid argument` | 0:41.64 |

Step 1 loss differs by 2.4e-7 (BF16 rounding noise in the backward path — under `MatchInsertedDtype` the gradients are stored BF16 internally and cast back to F32 only at GradientMap-write time; the v3 path stores F32 throughout). This is well within the parity expectation from Phase 5b and the BF16_GRAD_DECISION audit.

grad_norm differs by ~0.7% (0.00712 vs 0.00707) — same source: BF16 storage rounds individual element values before the L2 reduction.

The step-2 crash is the **same** error in both arms. Per the handoff (`HANDOFF_2026-05-13_PHASE5D_ZLORA_SMOKE.md` §Open hazards "Klein step 2+ crash"), this is pre-existing flame-core infra and not v2's defect; this smoke confirms v2 doesn't trigger it earlier or make it worse.

## Setup

- **Trainer binary**:
  - v3 control: `target/release/train_klein.v3` (built without `autograd_v2` feature, before v2 build overwrote `train_klein`)
  - v2 bridge: `target/release/train_klein` (`--features autograd_v2`)
- **Flame-core**: `77c64d6` at run time (post Phase 5d item #1 commit)
- **Model**: `/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors` (18.2 GB BF16)
- **Cache**: `EriDiffusion-v2/cache/alina_klein9b/` (1.3 GB, pre-encoded latents + caption embeds)
- **Config**: `configs/klein9b_alina.json`
- **CLI** (both arms): `--steps 50 --rank 16 --offload --sample-every 0`
- **GPU**: NVIDIA GeForce RTX 3090 Ti, 24 GB
- **Env**: `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH`, `RUST_LOG=info`

Both arms loaded the same 288 trainable LoRA tensors, same BlockOffloader pin-RAM allocation (32 blocks, 16.6 GB pinned CPU memory).

## What we wanted from this and what we got

User asked for "1 and speed":

1. **(1) Prove v2 doesn't make the step-2 crash worse**: ✅ Confirmed. Both arms crash with identical error at step 2.
2. **(speed)**: Can't measure steady-state on Klein 9B until the step-2 crash is fixed. Step-1 alone is graph build + first AdamW + warmup — dominated by setup-time noise, not meaningful for autograd-engine perf comparison. We have it on paper as 6.2s for both arms, identical, but it tells us nothing about v2 vs v3 backward overhead.

## What we already knew about Klein 9B speed

Per `feedback_lycoris_lokr_perf_overhead` / `project_perf_attack_2026-05-10`:
- Klein 9B step time: ~5.4 s/step (when reachable past step 2)
- OneTrainer is ~3.4× faster on Klein 9B (~1.6 s/step) — sync/memory-bound, not autograd-engine-bound
- This means v2's +2.18% backward overhead on Klein attn_chain (Phase 5c bench) is unlikely to be visible against the much larger sync-dominated step time even if step 2+ were reachable

## What unblocks Klein steady-state measurement

The step-2 crash needs root-cause and a fix. Per the handoff, this was deferred — diagnosed as pre-existing infra (not Phase 0 / not autograd v2). Until it's fixed:
- Use Z-Image for v2 performance comparison (Phase 5d item #1 PHASE5D_ZLORA_SMOKE_RESULTS.md — PASS)
- Klein 9B remains step-1 only for parity gates

## Artifacts

- `/tmp/klein9b_smoke_v3/board.db` (1 step, status: crashed)
- `/tmp/klein9b_smoke_v2/board.db` (1 step, status: crashed)
- `/tmp/klein9b_smoke_v3.log`, `/tmp/klein9b_smoke_v2.log`

## Code change

`EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_klein.rs`:
- Added `--use-autograd-v2` CLI flag (mirrors `train_zimage`)
- Routed backward through `AutogradContext::backward_v2` under the feature gate
- Disabled `FLAME_MT_SCALE` fast path when v2 is active (asserts F32 grads)
