# PLAN_STRIDE_REFACTOR — nsys profile baseline

**Captured:** 2026-04-23 ~10:25 | **Binary:** `klein9b_infer` (plain, NUM_STEPS=1) at flame-core b06a9b4 (Phase 11b) | **GPU:** RTX 3090 Ti | **Prompt:** default ("Beautiful young woman...")

Run spans full pipeline: model load + text encoder (Qwen 8B) + **1** denoise step + VAE decode. Multiply relevant counts by NUM_STEPS=50 to estimate production-run pressure.

## Headline numbers

| Metric | Count | Total |
|---|---:|---:|
| CUDA memcpy DtoD — all | 75,781 | 33,568 MB |
| CUDA memcpy DtoD — ≥1 MB (the target) | **1,573** | **33,569 MB** |
| CUDA memcpy DtoD — <1 KB (param/meta) | 73,831 | 4 MB |
| CUDA memcpy HtoD | 1,812 | 37,295 MB (model load) |
| CUDA kernels launched | 107 unique | — |

The **1,573 large DtoDs (≥1 MB)** are the stride-refactor target. Each averages 21 MB — these are full tensor-data copies from materializing views.

## Top kernel names preceding a large DtoD (the "producer" side)

Correlation: kernel ends within 20 µs before a ≥1 MB DtoD memcpy starts.

| Count | Kernel | Interpretation |
|---:|---|---|
| **421** | `rope_halfsplit_bf16_kernel` | Klein RoPE. Each RoPE output is about to be materialized — the downstream Q/K/V permute chain into SDPA forces `.contiguous()`. **Target: task #34 cuDNN SDPA stride-aware.** |
| **375** | `permute0213_kernel<bf16>` | Plain permute-materialize kernel. Every call IS a view-materialization. **Target: find callers, switch to stride-aware consumption.** |
| 218 | `bf16_to_f32` | dtype casts producing a new F32 buffer. Likely in softmax / norm paths. |
| 121 | `f32_to_bf16` | reverse casts. |
| 73 | `div_kernel` | |
| 72 | `sum_dim_keepdim_kernel` | softmax numerator? |
| 72 | `mul_scalar_kernel` | schedule sigmas. |
| 72 | `max_dim_keepdim_kernel` | softmax denom / row max. |
| 4 | `group_norm_forward_bf16_kernel` | VAE. |

## Top kernel names following a large DtoD (the "consumer" side)

Correlation: kernel starts within 10 µs after a ≥1 MB DtoD memcpy ends.

| Count | Kernel |
|---:|---|
| 216 | `f32_to_bf16` |
| 213 | `permute0213_kernel<bf16>` |
| 142 | `bf16_to_f32` |
| 71 | `mul_scalar_kernel` |
| 48 | cuBLASLt gemm |
| 15 | `rope_fused_bf16_kernel` |

The `f32_to_bf16 / bf16_to_f32` repetition on both sides tells us lots of ops follow the pattern: *"BF16 in, cast to F32, compute, cast back, materialize for next kernel."* Each cast-and-materialize is a new tensor alloc + DtoD.

## Targets in priority order

1. **cuDNN SDPA stride-aware shim** (task #34) — removes ~421 large DtoDs / step by letting Q, K, V pass strided views directly. Estimated savings: ~9 GB memory traffic / denoise step.

2. **`permute0213_kernel` callers** — find the Rust side of `permute_021` / `permute_0213` calls and:
   - If the caller is cuBLAS/cuBLASLt: pass strides + transpose flag, skip the kernel entirely.
   - If the caller is a fused kernel we own: evaluate stride-awareness in its functor.
   - ~375 large DtoDs / step. Estimated savings: ~8 GB memory traffic / step.

3. **`bf16_to_f32` / `f32_to_bf16` audit** — 218 + 216 + 121 + 142 = 697 cast events. Many are probably forced by ops that aren't BF16-native (softmax intermediate, div, mul_scalar). Each cast is a tensor-alloc + DtoD. Some were supposed to go away with the TensorIterator port — audit which remain.

4. **Cold-path kernels** (group_norm VAE, text encoder kernels): very few large DtoDs each, not worth guarding.

## Phase 2a kernel-wrapper guardrail audit scope

Per Alex's direction 2026-04-23 ~10:10: target guardrails only to kernels the profile flags. From the data above:

| Kernel | Current state | Action |
|---|---|---|
| cuDNN SDPA (shim) | partially stride-aware (e856a27) | complete via task #34 |
| `permute0213_kernel` | always materializes | find callers, route to stride-aware alternative |
| `rope_halfsplit_bf16_kernel` | assumes contig input | may need `.contiguous()` guard, THEN stride-aware Q/K/V sink unlocks full skip |
| `group_norm_forward_bf16_kernel` | NHWC, assumes contig | low priority, 4 calls/step |
| cuBLASLt gemms | already stride-aware (lda/ldb) | nothing to do; just pass strides at call site |
| `bf16_to_f32` / `f32_to_bf16` | always fresh-alloc | out of scope — these are legit cast ops |

Cold-path kernels (`mul_bc_kernel`, `gather_rows_bf16_kernel`, `sin_kernel`, `cos_kernel`, tiny misc): **skip guardrails**. <10 large DtoDs each.

## Raw artifacts

- `/tmp/klein_nsys/klein_1step.nsys-rep` (6 MB, Nsight Systems GUI-importable)
- `/tmp/klein_nsys/klein_1step.sqlite` (15 MB, queryable)

## Klein per-step cost (baseline for Phase 2 gate)

From 50-step production run at b06a9b4:
- **Total:** 204 s turbo / 186 s plain, seed 42
- **Denoise:** 2.51 s / step
- **VAE decode:** 6.5 s
- **Text encoder (Qwen):** ~few seconds

## Per-denoise-step deltas (1-step vs 2-step subtraction)

Captured 2026-04-23 ~10:35 by profiling 1-step and 2-step runs, subtracting
to isolate the pure denoise-per-step cost from model-load + text-encoder +
VAE-decode one-shots.

| Metric | Per step |
|---|---:|
| Total kernel time | 2,234 ms |
| **Total DtoD memcpy time** | **103 ms** |
| **DtoD as % of GPU time** | **4.6%** |
| Large DtoD (≥1 MB) count | 200 |
| Large DtoD memory moved | 8,649 MB |
| All DtoD count | 73,946 |

**Per-step kernel count deltas (what actually fires inside a denoise step):**

| Per step | Kernel | Nature |
|---:|---|---|
| **300** | `slice_copy_kernel` | narrow() BF16 materialization — task #33 |
| 160 | `rms_norm_kernel` | core compute (can't eliminate) |
| 128 | `rope_fused_bf16_kernel` | 2× per attn block (Q + K) |
| 114 | `modulate_pre_bf16_kernel` | core compute |
| 112 | `gate_residual_bf16_kernel` | core compute |
| 82/80/34 | cuBLAS gemms | core compute |
| 80 | `swiglu_fused_bf16_kernel` | core compute |
| 80 | `qkv_split_permute_bf16_kernel` | fused QKV split |
| 64 | `cudnn_generated_fort_native_sdpa...` | attention (stride-aware ✓) |
| **48** | `permute0213_kernel` | materialization — task #35 |
| 16 | `attn_split_txt_img_bf16_kernel` | core |
| 12 | `mul_scalar_kernel` | low — task #37 drops in priority |
| 10 | `bf16_to_f32` cast | |
| 7 | `f32_to_bf16` cast | |

## The stride refactor ceiling (important)

DtoD memcpy time = **103 ms out of 2,234 ms per step = 4.6%**. Eliminating
100% of DtoDs would top out at 4.6% speedup. Realistically eliminating
80% of the eliminable DtoDs gives **~3-4% total speedup**.

This contradicts the refactor plan's "2-20× slowdown vs PyTorch" framing.
The 371,580 DtoD per 5 steps (now re-confirmed at 73,946/step in this
profile) is **accurate as a count**, but the per-DtoD cost is tiny: each
averages ~4 µs. Per-step DtoD cost is 103 ms, **not seconds**.

## Where PyTorch's speed advantage actually comes from

Hypotheses to test (not in this profile):
- `torch.compile` kernel fusion (JIT-automated versions of our fused kernels)
- cuda graphs for the denoise loop (eliminates per-step kernel-launch overhead — 10-30% speedup on its own)
- More aggressive per-shape algorithm selection in cuBLAS/cuDNN
- Better memory pooling (fewer cudaMalloc cycles)
- Additional fused patterns (fused residual-add-layernorm, fused modulate+compute) beyond what we've implemented

## Revised recommendation

Given the 4.6% ceiling, the full Phase 2/2b/2c/3/4 stride refactor does
not justify ~30 hours of work.

**Cheaper wins, higher ROI:**
1. **cuda graphs for the denoise loop** — ~8 hours of work, estimated 10-30% speedup. Eliminates per-step kernel-launch overhead which is the real-world cost of the 44k-per-step kernel launches, not the memcpy bandwidth.
2. **Tasks #33 + #35 only** — lightweight narrow→view and permute0213-caller rerouting. Cumulative ~3% speedup, completes Phase 2a cleanly.
3. **PyTorch-style fused ops audit** — find which fused kernels PyTorch has that we lack (e.g., fused residual+layernorm). Each adds a kernel but eliminates 2-3.

**Deprioritize:** full Phase 2b/2c (WMMA→cuDNN training move, full stride-aware kernel sweep, PyTorch-parity sweep for cold-path kernels).
