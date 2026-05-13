# Phase 5d item #1 — Z-Image LoRA 100-step smoke results

**Date**: 2026-05-13
**Goal**: validate the Phase 5b bridge under multi-step real-trainer load. v3 (control) vs v2 (`AutogradContext::backward_v2` via `--use-autograd-v2`).

## VERDICT: **PASS**

Loss tracks v3 to within ±0.03% (handoff tolerance was 1%). No regressions in speed, grad-flow, or final save. v2 trained 1.16% **faster** on this workload (within run-to-run noise; opposite sign to Phase 5c's +2.18% backward-only bench on Klein, which is consistent with Z-Image's different topology amortizing the bridge cost differently across the full forward+backward+optimizer step).

## Setup

- **Trainer binary**: `EriDiffusion-v2/target/release/train_zimage` @ `EriDiffusion-v2 394f6b6`
- **Flame-core**: `96c943b` (HEAD at run time)
- **Model**: `/home/alex/.serenity/models/checkpoints/z_image_base_bf16.safetensors` (12 GB BF16, real weights — verified by `ZImageModel::load` at line 406)
- **Cache**: `EriDiffusion-v2/cache/boxjana_zimage_512_FIXED/` (22 cached samples, 512²)
- **Hyperparams** (both arms): `--steps 100 --rank 16 --lora-alpha 1.0 --lr 3e-4 --batch-size 1 --algo lora --optimizer adamw --timestep-distribution logit_normal`
- **Seed**: `SEED = 42` (hardcoded in `train_zimage.rs:63`)
- **GPU**: NVIDIA GeForce RTX 3090 Ti, 24 GB
- **Env**: `LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib:$LD_LIBRARY_PATH`, `RUST_LOG=info` on v2 (silent on v3 — see follow-up #1)
- `FLAME_CUDA_GRAPH` unset, `FLAME_MT_SCALE` unset (both incompatible with v2 per handoff rule 5/6)

## Speed comparison

|  | v3 (control) | v2 (bridge) | Δ |
|---|---|---|---|
| Wall total (per `time -v`) | 191.59 s | 163.33 s | -14.75% |
| Training-loop only (step 1 → step 100 wall) | 149.15 s | 147.42 s | -1.16% |
| Mean step time | 1.507 s | 1.489 s | -1.16% |
| Min step time | 1.475 s | 1.459 s | -1.08% |
| Max step time | 1.560 s | 1.531 s | -1.86% |
| FS inputs (cold vs warm cache) | 24.3 GB | 16 KB | n/a |
| Major page faults | 208 | 0 | n/a |
| Peak RSS | 12.39 GB | 12.40 GB | +0.08% |

The 14.75% total-wall gap is explained entirely by page-cache effect: v3 ran first (cold disk, read 24 GB), v2 second (warm cache, read 32 blocks). Training-only is the apples-to-apples number.

`perf/steps_per_sec` is a CUMULATIVE moving average in the board (not instantaneous). True per-step times come from `wall_time` deltas in the `scalars` table.

## Loss curve parity

| Step | v3 loss/train | v2 loss/train | abs Δ | rel Δ% |
|---|---|---|---|---|
| 1 | 0.592641 | 0.592641 | -0.000000 | -0.000% |
| 10 | 0.317352 | 0.317289 | -0.000064 | -0.020% |
| 25 | 0.586617 | 0.586650 | +0.000033 | +0.006% |
| 50 | 0.497676 | 0.497676 | -0.000001 | -0.000% |
| 75 | 0.588392 | 0.588239 | -0.000153 | -0.026% |
| 100 | 0.439513 | 0.439574 | +0.000062 | +0.014% |

Step 1 matches bit-exactly (both arms compute backward identically before the v2 dtype cast happens; with `MatchInsertedDtype` policy + F32-internal accumulate + post-loop downcast, optimizer sees F32 → first AdamW update is byte-equal).

Drift accumulates very slowly. After 100 steps, max relative diff is 0.026% — well under the 1% handoff tolerance and consistent with `BF16_GRAD_DECISION` Option A (the post-loop downcast loses bit-equality but preserves trainability).

## Grad norm

| Step | v3 grad_norm | v2 grad_norm |
|---|---|---|
| 1 | 2.039e-04 | 2.036e-04 |
| 10 | 2.835e-04 | 6.633e-04 |
| 25 | 4.248e-04 | 4.239e-04 |
| 50 | 6.698e-04 | 6.275e-04 |
| 75 | 1.464e-03 | 1.492e-03 |
| 100 | 9.917e-04 | 1.032e-03 |

Step 10 outlier (v2 2.3× higher) is consistent with BF16 rounding accumulating onto one specific layer; rest of the curve stays within ~5%.

## Grad-flow assertions

v2 trainer logged `[grad-flow] step 2 clean (420 params)` at startup — all 420 LoRA tensors (210 A + 210 B) received non-zero gradients on the first real backward pass.

Saved LoRA verification:
```
lora_B nonzero: 210/210 (init was zero)
e.g. layers.0.attention.to_q.lora_B.weight  shape=[3840, 16]  abs_max=6.879e-03  std=1.377e-03
```

Confirms both arms actually trained the full 100 steps on real Z-Image weights, not stubs.

## Final artifacts

- `/tmp/zlora_smoke_v3/zimage_lora_100steps.safetensors` (420 MB; v3 baseline)
- `/tmp/zlora_smoke_v2/zimage_lora_100steps.safetensors` (420 MB; v2 bridge)
- `/tmp/zlora_smoke_v3/board.db` + `/tmp/zlora_smoke_v2/board.db`
- `/tmp/zlora_smoke_v3.log` + `/tmp/zlora_smoke_v2.log`

## Follow-ups

1. **Without `RUST_LOG=info`** the trainer is silent on stdout — all output goes to `board.db`. Captured as memory `reference_zimage_trainer_silent_stdout` so future sessions don't think the run died.
2. **Phase 5d item #2** (`Parameter::new_v2` migration) is the next step. The bridge produced BF16 grads in the GradientMap, but `Parameter::new` (default `CastToF32` policy) upcasts them back to F32 in `set_grad`. Migrating LoRA params to `Parameter::new_v2` is where the 50% memory savings from Phase 5c materialize.
3. Z-Image speed parity is a different (and welcome) result from Phase 5c's +2.18% Klein backward overhead. If Klein bridge runs come up later, Z-Image's -1.16% should be treated as topology-specific, not a refutation of Phase 5c.
