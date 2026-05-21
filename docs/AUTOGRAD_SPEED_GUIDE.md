# Autograd Speed Guide

Current state, profiling methodology, and next steps for closing the backward pass speed gap with PyTorch.

Last updated: 2026-05-20

---

## Current Speed

| Resolution | flame-core | OneTrainer/PyTorch | Ratio |
|------------|-----------|-------------------|-------|
| 512x512    | 2.8 s/step | 1.63 s/step       | 1.72x |
| 1024x1024  | ~8.5 s/step | 5.36 s/step       | ~1.59x |

Forward ops are at PyTorch parity (14/17 ops within 1.5x, 10 ops faster). The gap is entirely in the backward pass.

---

## Where the Time Goes

Profile with `FLAME_PROFILE=1`:

```
Z-Image 512x512, 30 transformer blocks with gradient checkpointing:

Total backward:   1.94s (100% of backward time)
Breakdown:        30 checkpoint blocks × 64.5ms each
Per block:        108 sub-backward ops, 0.9ms recompute + 63ms sub-backward
Per op average:   0.58ms
PyTorch per op:   ~0.30ms (2x faster)

No single slow op — the overhead is per-op CPU dispatch:
  mutex lock on GradientMap, compute_gradients() call overhead,
  accumulate() with potential BF16→F32 cast, gradient filtering.
```

PyTorch achieves 0.30ms per op via CUDA graph replay for the backward pass — all kernel launches are recorded once and replayed as a single graph launch, eliminating CPU dispatch overhead.

---

## How to Profile

### Basic backward profiling
```bash
FLAME_PROFILE=1 cargo run --release --bin zimage_lora_train -- \
  --config config.toml --output-dir output --max-steps 3
```

Shows per-op-type timing, top 10 slowest nodes, kernel vs overhead breakdown.

### Checkpoint sub-backward detail
The `[checkpoint:ID]` lines show recompute vs sub_bwd time and BF16 cast count per block.

### Per-op backward trace
```bash
FLAME_BACKWARD_TRACE=1 cargo run ...
```
Prints every backward op with shapes. Warning: very verbose (~3200 lines per step).

### Forward op benchmark
```bash
cd /home/alex/EriDiffusion/flame-core
cargo run --release --bin op_bench_flame
```
17 ops with CUDA event timing, 100 warmup / 200 timed iterations.

---

## What Was Fixed (2026-04-12/13)

### Forward fixes
1. **Memory pool `TensorStorage::Drop` disabled** — `#[cfg(not(shared_storage))]` gate meant pool never recycled buffers with default features. Every op hit raw `cudaMalloc` (~800μs). This was the root cause of 8-24x forward slowness across all ops.
2. **12+ BF16 ops bypassed pool** — raw `device.alloc::<u16>` in `bf16_ops.rs`, `bf16_elementwise.rs`, `tensor.rs`. All replaced with `pool_alloc_u16`.
3. **Abs was `square().sqrt()`** — replaced with sign-bit-clear kernel (131μs → 7.2μs).
4. **Softmax kernel** — 3-pass shared-memory → 2-pass online softmax with warp shuffles (1504μs → 157μs).

### Backward correctness fixes
5. **`softmax_backward` swapped args** — `(output, grad_output)` were reversed. Wrong gradients in all trainers.
6. **`Op::Sum` backward F32 dtype** — produced F32 grads regardless of BF16 input, breaking downstream BF16 guard checks.
7. **`Op::MatMul` missing 3D×3D backward** — crashed with "Transpose requires 2D tensor" on batched matmul.
8. **3D×3D matmul backward used Tensor methods** — recorded to autograd tape during backward, causing 3s/iter hang.

### Backward speed fixes
9. **Removed `ensure_bf16()` round-trip** — was casting every gradient to BF16 then accumulate() cast back to F32. ~40 extra CUDA kernels per backward pass eliminated.
10. **All `.to_dtype()` → `.to_dtype_no_grad()`** in backward paths — avoids autograd check overhead.
11. **Deferred BF16→F32 cast in `accumulate()`** — first gradient stored as-is, cast only on actual accumulation. Saves 630 kernel launches per step.
12. **CompactIndex for checkpoint sub-backward** — O(1) Vec-based gradient lookup instead of HashMap.

---

## Flash Attention Backward Kernel

A complete wmma tensor core backward kernel exists at `src/cuda/flash_attention_bwd.cu`:

- 585 lines, HD=64/96/128 specializations
- FlashAttention-2 backward algorithm (KV-outer loop, 7 stages)
- Register accumulation for dK/dV (no staging buffers)
- Only dQ uses FP32 global staging (atomicAdd across blocks)
- Forward kernel updated to save LSE (`float* LSE` parameter)
- Training SDPA wired through `forward_train()` with LSE saving

**Currently disabled** — gated behind `FLAME_FUSED_ATTN_BWD=1`. At seq_len=1024 (Z-Image), the 7-stage pipeline with per-stage `__syncthreads` is slower than 12 separate fully-pipelined kernel launches (4.2s vs 2.8s/step). May win at larger sequence lengths (4096+).

### HiDream-O1 SDPA fast-path status (2026-05-21)

The production cuDNN SDPA backward fast path now covers unmasked BF16
4D attention with supported head dimensions even when sequence lengths are
not 64-aligned. `sdpa::forward_train()` pads Q/K/V to the next 64-token
boundary, records the real Q/KV lengths in `Op::FlashAttention`, and passes
those lengths into the cuDNN padding-mask graph for forward and backward.
The returned output is sliced back to the real Q length, so callers keep the
same tensor contract while avoiding the slow recompute path for padded
unmasked attention.

`try_cudnn_sdpa_backward()` still intentionally bails when `mask.is_some()`.
The generic masked path is a compatibility route for arbitrary binary masks
and attention biases. Do not route normal unmasked model attention through it
just because the raw token count is not divisible by 64.

HiDream-O1 uses a structured prefix-causal/full attention pattern: AR/text
prefix rows are causal, while image rows use full attention over all tokens.
The model now calls `attention::sdpa_prefix_causal_full` instead of
materializing the old `[B, 1, S, S]` mixed binary mask. Internally that
primitive records a single custom autograd op. Forward runs the smaller prefix
causal pass plus a suffix all-ones masked pass; backward recomputes once
against the exact prefix-causal/full mask. This is the current fastest path
proved by the O1 SDPA trap on 2026-05-21; the unmasked suffix/cuDNN attempts
failed this machine's cuDNN plan builder at O1's non-aligned suffix shape.
The in-tree FA2 forward supports head dimensions 64/96/128 with a runtime
causal flag; HD128 non-causal uses a 64x32 K tile and PyTorch-style
`UNFUSE_FMA` softmax scaling. Direct FA2-vs-FP32-reference tests pass, so the
remaining O1 failure is not a gross FA2 correctness bug.
Before structured attention, a 3-step EDV2 O1 probe produced:

```text
FLAME_LOG_SDPA_BWD=1 train_hidream_o1 --steps 3 ...
108 [sdpa-bwd] bail:mask-present
```

Causal cuDNN backward is available behind `FLAME_CUDNN_SDPA_BWD_CAUSAL=1`,
but keep it opt-in: O1's prefix shape has produced cuDNN plan failures and
slower timings in local probes. With `FLAME_LOG_SDPA_BWD=1`, the mixed O1
path should show `PrefixCausalFullAttention` / masked recompute behavior for
the custom op, not a generic full `[S,S]` mask in the model. The remaining O1
gate is a strict parity issue: the fixed production harness now returns FAIL
when forward/objective/per-layer metrics fail, and the current first failing
stage is `forward::layer00.attn_out`. `layer00.sdpa_out` is within threshold
(`max_abs=1.953125e-3` on the pinned Full-model dump), but the sparse attention
rounding differences spread through `o_proj` and later residuals. Do not run
the 1000-step `/eri2` training proof until this parity gate is green.

North-star reminder: structured SDPA only proves the old masked hot road is
gone. O1 is not fixed until a real trained LoRA renders clean trigger-bound
samples and step speed is beating, matching, or close to the ai-toolkit O1
trainer reference. Always record same-seed no-LoRA vs LoRA samples after the
run; image quality is the validity gate, not short-run loss alone.

Trainer parity note: EDV2's O1 default LoRA surface must stay aligned with
ai-toolkit/public O1 LoRAs: 252 Qwen language-layer adapters plus the five
pixel/timestep/final heads (`x_embedder`, `t_embedder1`, `final_layer2`).
The transformer-only route is an explicit ablation, not the default path.

---

## Next Steps: CUDA Graph for Checkpoint Sub-Backward

### The opportunity
108 ops × 30 blocks = 3240 kernel launches per step at 0.58ms dispatch each. CUDA graph replay would reduce this to 30 graph launches at ~0.1ms each.

Expected speedup: backward from 1.9s → ~1.0s, total step from 2.8s → ~1.9s (close to PyTorch 1.63s).

### The infrastructure
`cuda_graph.rs` has working capture/replay/instantiate. The main backward path already supports CUDA graphs (`FLAME_CUDA_GRAPH=1`).

### The challenge
Each of the 30 blocks has different weight pointers (different transformer block weights loaded via block-swap). CUDA graph replay uses the exact same memory addresses as capture.

### Approaches (in order of feasibility)

**Option A: Capture 30 separate graphs (one per block)**
- Step 0: warmup (normal backward, fills allocator pool)
- Step 1: capture each block's sub-backward as its own graph
- Step 2+: replay each block's graph
- Pros: simple, each graph uses correct weight pointers
- Cons: 30 graph instantiations (~30ms one-time cost), 30 graph launches per step
- Expected: 30 × ~0.1ms = 3ms vs current 30 × 63ms = 1890ms

**Option B: Single graph with pointer patching**
- Capture one block's sub-backward
- Use `cudaGraphExecUpdate` or `cudaGraphExecKernelNodeSetParams` to patch weight pointers
- Replay 30 times with different pointers
- Pros: one graph, one instantiation
- Cons: complex pointer patching, may not work if kernel args change

**Option C: Fixed staging buffer**
- Copy each block's weights to a fixed staging area before replay
- Graph always reads from the same staging area
- Pros: simple graph capture
- Cons: 30 weight copies per step (but weights are small vs activations)

### Implementation sketch (Option A)

In the checkpoint backward handler (`autograd.rs:1913`):

```rust
// Step 0: normal sub-backward (warmup)
// Step 1: begin_capture → sub-backward → end_capture → instantiate
// Step 2+: launch(graph_exec)

// Key: each block gets its own graph_exec in a Vec<CudaGraphExec>
// indexed by the block's position (0..29).
// Invalidate on tape structure change (different number of sub-entries).
```

### Key constraint
No `cudaMalloc` / `cudaFree` during capture. The caching allocator pool must be fully warmed up before capture. The warmup step (step 0) handles this naturally — all allocation sizes are seen during the first backward, populating the pool.

---

## Backward Path Conventions

Rules for code in `compute_gradients()` and checkpoint sub-backward:

1. **Never use `.to_dtype()` — always `.to_dtype_no_grad()`**. Even with autograd disabled, `to_dtype` has more overhead.
2. **Prefer `GpuOps::` over Tensor methods** for new backward ops. GpuOps are pure CUDA dispatchers with no autograd checks.
3. **Existing Tensor methods are safe** (autograd disabled via atomic flag) but add ~1μs per call for the check.
4. **Gradient accumulation**: `accumulate()` defers BF16→F32 cast until second gradient. Don't pre-cast.
5. **`ensure_bf16()` is now a no-op** — gradients stay in their computed dtype until accumulation.

---

## File Reference

| File | Role |
|------|------|
| `src/autograd.rs` | `compute_gradients()`, checkpoint backward (~line 1913), main backward loop |
| `src/cuda_graph.rs` | CUDA graph capture/replay infrastructure |
| `src/gradient.rs` | `GradientMap`, `accumulate()`, `CompactIndex` |
| `src/cuda/flash_attention_bwd.cu` | Fused wmma backward kernel (disabled) |
| `src/cuda/flash_attention_fwd.cu` | Forward with LSE output |
| `src/sdpa.rs` | Training SDPA routing (`forward_train()`) |
| `src/bin/op_bench_flame.rs` | Forward op benchmark |
| `benchmarks/op_bench_pytorch.py` | PyTorch reference benchmark |
| `docs/FLAME_AUTOGRAD_INTERNALS.md` | Complete autograd architecture audit |
| `docs/FLAME_KERNELS.md` | Kernel catalog with benchmark table |
