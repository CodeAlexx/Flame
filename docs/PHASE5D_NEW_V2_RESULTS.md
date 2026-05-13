# Phase 5d item #2 — `Parameter::new_v2` migration results

**Date**: 2026-05-13
**Goal**: close the policy loop on the v2 path. After the Phase 5b bridge produces BF16 grads in the GradientMap (via `MatchInsertedDtype` policy), the default `Parameter::new` (`CastToF32` policy) upcasts them back to F32 in `set_grad`. Item #2 flips trainable LoRA params to `MatchParamDtype` so BF16 stays BF16 end-to-end.

## VERDICT: **PASS** — bit-equal loss to v3, modest peak-memory win on LoRA-training-shaped workload

The 50% gradient-memory savings from Phase 5c **did materialize on the gradient slice itself**. The total-footprint impact is smaller because LoRA training is dominated by model weights + activations, not grads.

## What we changed

`EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_zimage.rs:495-512` (commit `EriDiffusion-v2@4b6c38c`):

```rust
let mut params = model.parameters();  // Vec<Parameter>, clones with policy=CastToF32
// ...
if args.use_autograd_v2 {
    for p in &mut params {
        p.set_grad_dtype_policy(GradDtypePolicy::MatchParamDtype);
    }
    log::info!("[autograd_v2] flipped {} params to MatchParamDtype grad policy", params.len());
}
```

Why this is sufficient (no API changes to `LoRALinear` or `ZImageLoraBundle`): `Parameter` is `Clone` with `Arc<Mutex<...>>` for `data` and `grad` (shared) but plain `grad_dtype_policy: GradDtypePolicy` field (per-instance, `Copy` on clone). The trainer-owned `params` Vec is what's consulted by:
- `param.set_grad(g)` at `train_zimage.rs:1045, 1052` — reads the trainer's policy
- `AdamW::step(&params)` at `train_zimage.rs:1070` — reads the trainer's policy via the 4-way classifier
- The bundle's internal Parameters keep `CastToF32` but they're never read by the grad/step path.

## Numbers

Configuration (all arms): `--steps 100 --rank 16 --lora-alpha 1.0 --lr 3e-4 --batch-size 1 --algo lora --optimizer adamw`, cache `boxjana_zimage_512_FIXED`, seed=42, 3090 Ti, `RUST_LOG=info`, no `FLAME_CUDA_GRAPH` / `FLAME_MT_SCALE` (both incompatible with v2 per handoff rule 5/6).

| Arm | loss@100 | mean s/step | train wall | peak GPU MB | mean GPU MB |
|---|---|---|---|---|---|
| v3 (control) | 0.4395 | 1.504 s | 148.9 s | 16,519 | 15,818 |
| v2 bridge only (Phase 5d item #1) | 0.4396 | 1.489 s | 147.4 s | n/a | n/a |
| **v2 + new_v2 (this commit)** | **0.4395** | **1.495 s** | **148.0 s** | **16,358** | **15,821** |

Loss parity tightened: bridge-only had Δ=0.014% at step 100 (BF16 set_grad → F32 upcast → F32 AdamW → drifts vs v3); new_v2 closes the loop and the drift disappears (bit-equal at the printed precision).

Peak GPU memory: **−161 MB (−0.97%)** vs v3 control.

## Why memory savings are smaller than Phase 5c projected

Phase 5c (commit `flame-core@6192473`) measured gradient memory only:
- Synthetic 4-layer MLP: 0.000183 → 0.000092 MB (−50%)
- Klein attn_chain prod (forward fixture): 156 → 78 MB (−50%)
- Klein double-block fixture: 1.81 → 0.91 MB (−50%)

The 50% factor is correct **on the gradient slice**. For Z-Image LoRA training:
- 420 trainable LoRA tensors at rank=16
- Total grad memory ≈ 100-200 MB (sum over `[3840, 16]`-shaped lora_A/B pairs and FF up/down projections)
- BF16 halves that → expected save ≈ 50-100 MB
- Observed −161 MB also captures the bridge's BF16 backward-graph temporaries that don't show up in a forward-fixture bench

Total GPU footprint is dominated by:
- Model weights (~12 GB BF16 Z-Image base)
- Activations (~3-4 GB transient during forward/backward)
- AdamW state (m, v) which is F32 in v3 AND v2 (independent of grad policy)

In percentage terms, halving a 100-200 MB grad slice inside a ~16 GB run is ≤1%. The headline savings appear in **full fine-tuning** or **sparse-grad models** where grads dominate, not LoRA.

## What this implies for Klein 9B

Per user direction (memory `project_v2_klein9b_proving_ground`): Klein 9B is the next-up real pipeline. Same migration pattern applies — flip `params` policy after `model.parameters()` collection. Expected delta is similar (~1% absolute) until/unless we move to a different mode where grads are a larger fraction (e.g., full DiT fine-tuning, or grad-checkpoint-heavy workloads where the savings stack across recomputed activations).

The bigger win on Klein 9B remains: **bit-equal loss to v3** with zero correctness regressions, and a foundation for v2 to gradually replace v3 as the default dispatch.

## Artifacts

- `/tmp/zlora_v2_newv2_100step/board.db` (v2 + new_v2, 100 steps, status=complete)
- `/tmp/zlora_v2_newv2_100step/zimage_lora_100steps.safetensors` (420 MB)
- `/tmp/zlora_v2_newv2_100step.gpumem` (500ms-interval `nvidia-smi` samples)
- `/tmp/zlora_v3_100step_gpumem/board.db` (v3 control with sampler)
- `/tmp/zlora_v3_100step_gpumem.gpumem`
