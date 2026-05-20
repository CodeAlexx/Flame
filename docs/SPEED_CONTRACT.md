# flame-core speed contract

The north star: **flame-core is one framework. Per-call inefficiency in any
primitive multiplies across every model that uses it. Fix the primitive once →
every model is fast.**

OneTrainer is fast *before* a new model is added. Models added inherit the
framework's speed because the primitives are right. flame-core's goal is the
same: any model — DiT, MMDiT, MoE, video DiT, autoregressive, audio — that
plugs into flame-core is fast out of the gate.

This document is the audit gate. Every PR touching a primitive, kernel,
launch wrapper, autograd op, or memory subsystem must point at the clauses
it satisfies (or explicitly request an exemption with a reason).

---

## The five clauses

### 1. Sync — primitives do not host-stall

**Rule.** No flame-core primitive used in a training step shall call
`cudaStreamSynchronize`, `cudaDeviceSynchronize`, `cudaFree`-of-stream-active
memory, `cudarc::*_sync_copy*`, or any other API that blocks the CPU thread
on GPU completion.

**Reason.** Host stalls multiply across ops. A primitive that syncs once per
call, called 300×/step, costs at least 300 host-stall windows even if each
sync is cheap. PyTorch eager's per-step sync count is ~8. flame-core's was
268 before Class E.

**Permitted exceptions.**

- **Step boundary reads.** `loss.to_vec()?[0]` and `total_norm.item()?` at
  end-of-step are explicit, single-call, named, batched. OK.
- **Init time.** Weight load, allocator initialization, one-time kernel
  precompile. Anything not on the per-step path.
- **Error checking after kernel launch.** `cudaGetLastError()` is non-blocking.
- **Stream-stream waits via CUDA events.** These are GPU-side; not host stalls.

**Fail mode at PR review.** Grep the diff for `cudaStreamSynchronize`,
`cudaDeviceSynchronize`, `synchronize()`, `*_sync_copy*`. Each hit must be
justified in the commit message.

**Reference implementations.**

- `flame-core/cuda/narrow_strided.cu` (commit `b552f61`) — Class E fix.
  Inline kernel-arg metadata replaces per-call `cudaMalloc + cudaMemcpyAsync
  + cudaStreamSynchronize + cudaFree`. Microbench: 1000 cudaStreamSynchronize
  → 0. Real-trainer (klein 9B): 268/step → 0/step.

**Two correct primitive patterns.**

- **Small/fixed metadata (≤ 4 KB total).** Inline as kernel-arg struct
  passed by value. See `NarrowMeta` in `narrow_strided.cu`. No allocation,
  no copy, no sync. Permitted when metadata fits and rank/size bounds are
  known.
- **Variable/large metadata or workspace.** Cached per-stream device buffer,
  grown on first use, reused across calls. See
  `flame-core/cuda/gemm_bf16_cublaslt.cu` and `gemm_bf16_fp32acc.cu` —
  cuBLASLt workspace cache. Lifecycle owned by the primitive, freed only at
  process exit.

**Forbidden pattern.** The "alloc → async copy → launch → sync → free"
sequence in any launch wrapper. This is the narrow pre-fix shape and the
current `sdpa_stream_bf16.cu` shape (11 mallocs/call).

---

### 2. Dtype — storage matches param dtype end-to-end

**Rule.** Storage dtype matches the parameter's declared dtype throughout
the autograd graph. F32 is `opmath_t` (compute precision) inside kernels
only, never as storage.

**Reason.** Class A in our profile: F32 grad storage policy forces
BF16→F32→BF16 casts at every grad accumulation. ~9,500 `bf16_to_f32` +
~9,700 `f32_to_bf16` launches per step on klein 9B alone. Same pattern
hits every BF16-param model.

**Specific bans.**

- `Parameter::set_grad` shall not cast incoming BF16 grads to F32.
- `GradientMap::insert` / `accumulate` shall not enforce F32.
- AccumulateGrad in autograd_v2 shall accumulate BF16-in-BF16-out, with
  F32 only as kernel-internal accumulation.
- `tensor_narrow.rs:169-183` style cast detours shall be removed
  (Class B handoff: `CLASS_B_NARROW_BACKWARD_HANDOFF.md`).

**Permitted exceptions.** F32 master weights in an optimizer (Kahan
summation pattern). Explicitly stored separately from the autograd grad
buffer.

**Fail mode at PR review.** Search the diff for `.to_dtype(DType::F32)`,
`ensure_f32`, F32 cast helpers. Each hit must be justified.

**Reference fix in progress.** Class B narrow_backward
(handed off, design in `CLASS_B_NARROW_BACKWARD_HANDOFF.md`).

**Bigger workstream.** The full Class A migration touches `GradientMap`,
`Parameter`, Adam, SGD, `ops::grad_norm`, optimizer parity tests. Should
land alongside the autograd_v2 rewrite per
`AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §7.

---

### 3. Autograd — one generation, per-op, Result-typed

**Rule.** flame-core has exactly one autograd engine. Each backward op is
a `GradFn` struct in its own file. `apply()` returns
`Result<Vec<Option<Tensor>>>`. Saved tensors hold version-counter handles
and release through `&self`.

**Reason.** Today's `autograd.rs` is 4547 lines of one `enum Op` plus
manual match arms across 3 stacked generations (`autograd.rs`,
`autograd_v3.rs`, `autograd_v4/`). Adding an op requires touching 5 sites.
Result: per-op cost includes giant-match dispatch overhead AND every site
must be visited for any cross-cutting fix (Classes A, B, E all
encountered this).

**The plan.** `autograd_v2/` per
`flame-core/docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`. Phases 0-5.

**Fail mode at PR review.** Adding a new `Op` enum variant to
`autograd.rs` is a contract violation. New ops go into `autograd_v2/`
once it lands; until then, every new manual-match-arm site is a tech
debt entry to pay back at v2 migration.

---

### 4. Kernel/launch — per-op kernel within 2× of PT ATen

**Rule.** Every per-op CUDA kernel in flame-core shall execute within
2× the wall time of PyTorch's `ATen/native/cuda` equivalent at matched
shape, dtype, contiguity. Anything beyond that is a kernel rewrite
candidate, not a "framework limitation."

**Reason.** OneTrainer / SimpleTuner / EDv2-reference all use PyTorch eager
(no fusion, no compile). They still win on wall-clock for many ops.
The gap is per-kernel quality, not fusion strategy. flame's existing
GEMM is faster per-call than PT's; flame's `sum_dim_keepdim_bf16` is
134× slower (Class C).

**Class C target.** `cuda_kernel_sources.rs:850`
`sum_dim_keepdim_bf16_kernel`. Geometry is "one thread per output,
scalar serial loop, no shared mem, no warp reduce." Should be a
block-per-output-row + warp shuffle, matching PT's
`aten/src/ATen/native/cuda/Reduce.cuh` pattern.

**Fail mode at PR review.** A new kernel without a benchmark vs PT
ATen at matched shapes is a contract violation. The benchmark goes
in `flame-core/benches/` or as a `#[test]` with a hard upper bound.

**No more fused kernels until the per-op baseline is right.**
flame-core already has 18+ fused C kernels and 13 fused Rust functions.
OT uses zero fused kernels and is faster. Fusion is not the missing
lever. Per-op kernel quality is.

---

### 5. Memory/IO — bandwidth-aware primitives

**Rule.** Framework primitives that move data between host and device
or stream weights from disk shall be bandwidth-bound by the underlying
hardware (PCIe, NVLink, SSD), not by suboptimal use of CUDA APIs.

**Reason.** Large models (klein 9B, ltx2, wan22 14B+14B, hidream 32B,
sensenova_u1, nucleus, flux 12B) all use `BlockOffloader` to fit on
commodity GPUs. If BlockOffloader has a per-call inefficiency, every
large-model trainer inherits it. ~3 s/step of klein 9B's 3.8 s/step is
non-kernel non-sync time — almost certainly memory/IO.

**Specific requirements.**

- **Pinned host memory** for all H2D weight transfers.
- **Async H2D on a dedicated transfer stream**, with GPU-side
  `cudaStreamWaitEvent` to chain dependencies — never host-side
  `cudaStreamSynchronize`.
- **Double-buffered prefetch:** while compute runs on slot N, H2D
  loads slot N+1. Prefetch issue must precede compute use by enough
  time to hide H2D latency.
- **No re-upload churn:** a tensor uploaded once should not be
  re-uploaded on subsequent steps unless it changed.
- **D2H reads minimized:** anything that goes to host per step
  (loss, grad_norm, status flags) is batched and named.

**Current state.** `BlockOffloader` (in `eridiffusion-core`) implements
most of these correctly. The "missing 3 s/step" investigation is open
work — likely candidates: overlap quality between transfer and
compute streams, per-step `cuMemcpyDtoHAsync` events (the original
profile showed 1.9/step at 220 ms each — what's going to host?),
activation offload churn.

**Fail mode at PR review.** Any new primitive that does H2D or D2H
in the per-step path without a stated bandwidth target.

**Open workstream.** Full BlockOffloader investigation deferred per
user; framework-wide because it touches 13+ model families across
DiT, MMDiT, MoE, video DiT, multimodal.

---

## How a PR satisfies the contract

A PR touching a flame-core primitive must:

1. **Name the clauses** it satisfies in the commit message body.
2. **Cite measurement.** Microbench or nsys profile showing the rule
   isn't violated. If adding a new op, include a comparison to PT
   ATen at one realistic shape.
3. **Reference any precedent.** If using the inline-kernel-arg pattern
   from Class E or the cached-workspace pattern from cuBLASLt, name it.

A PR that doesn't touch a primitive (e.g., new training feature, new
data loader, doc-only change) is exempt.

---

## Reference list

| Doc | What it covers |
|---|---|
| `SPEED_CONTRACT.md` (this) | The five clauses, audit gates, reference impls |
| `AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` | Clause 3 in detail. Phases 0-5, BF16-grad migration is Clause 2 dependency |
| `CLASS_B_NARROW_BACKWARD_HANDOFF.md` | Class B narrow_backward dtype detour — handed off to parallel session |
| commit `b552f61` (flame-core) | Class E narrow primitive — Reference Implementation #1 for Clause 1 |
| `tests/narrow_sync_microbench.rs` | Microbench gate for Clause 1 regression |

---

## Open audit items

These were identified during the audit that produced this contract:

1. **`sdpa_stream_bf16.cu`** — 11 cudaMalloc + 11 cudaFree per call.
   Same anti-pattern as narrow pre-fix, 11× concentration. Universal
   SDPA fallback path. Fix template = cached workspace per
   `(Q_t, K_b, d)` shape tuple.
2. **`cudnn_sdpa.cpp` / `cudnn_sdpa_bwd.cpp`** — small malloc counts
   (4 and 2 respectively). Need deeper read to confirm they're
   workspace caches.
3. **Dead `.cu` duplicates** — `flame-core/cuda/narrow_strided_backward.cu`,
   `flame-core/src/cuda/narrow_strided.cu`,
   `flame-core/src/cuda/narrow_strided_backward.cu`, and `.bak` files.
   All contain the pre-fix anti-pattern. Not compiled, but they look
   canonical and could mislead a future patch. Delete proposed,
   pending user approval.
4. **Class A migration** — F32 grad storage policy. Cross-cutting:
   GradientMap, Parameter, Adam, SGD, grad_norm. Best landed with
   autograd_v2 Phase 4.
5. **Class C kernel rewrites** — `sum_dim_keepdim_bf16` (134× PT),
   any other per-kernel hot path with similar gap. Independent
   workstream.
6. **BlockOffloader bandwidth investigation** — find the per-step
   memory cost driver across all 13+ trainer families. Open.
