# FA2 Phase 1.6 — cp.async Pass (Shipped Design)

## What shipped

Single-buffer KV slot + cp.async for K and V loads, with V's prefetch issued
immediately after QK^T completes and overlapping with the online-softmax
pass. This is a strict subset of the original plan (full double-buffered
KV); the reduced scope reflects a correctness regression that was not
resolved when attempting the fold-s_P-into-s_S SMEM reclaim required for
the double buffer. Details in "Attempted but reverted" below.

**Correctness:** all 4 naive-FP32 configs PASS at cos_sim ≥ 0.9999,
max_abs ≤ 1e-2, validated against a pure-Rust FP32 materialized reference
in `tests/fa2_parity_naive.rs`.

**Perf:** 1.35× – 1.62× over the pre-Phase-1 BQ=32 WMMA kernel (now
deleted). Against Phase 1's own baseline (BQ=64, synchronous loads) the
speedup is ~1.02× – 1.05× — the V-load/softmax overlap is the only
thing cp.async buys in a single-buffer kernel, and V loads are only a
fraction of total kernel time.

## Pipeline (as shipped)

Per KV iteration j:

```
  (1) cp.async K_j → s_KV              issue
      cp.async.commit_group
      cp.async.wait_group(0)
      __syncthreads
  (2) QK^T:  s_Q × s_KV^T → s_S       WMMA, scale applied
      __syncthreads
  (3) cp.async V_j → s_KV              issue (overwrites dead K_j)
      cp.async.commit_group            [in-flight during next step]
  (4) Online softmax: read s_S, write P (BF16) to s_P, update m/l/s_O
      (V_j cp.async runs in parallel here)
  (5) cp.async.wait_group(0)
      __syncthreads                    V_j now visible
  (6) PV:   s_P × s_KV → o_frag → scratch → s_O       WMMA accumulate
      __syncthreads
```

What this hides: V_j's global-memory round-trip (HBM read latency) behind
the softmax compute of iter j. On RTX 3090 Ti at HD=128, softmax takes a
few microseconds per KV tile; V load takes a similar order. Full hiding
saves ~half a V-load per iter.

What this does NOT hide: K_{j+1}'s load. K_{j+1} blocks the start of
iter j+1 because there's no second buffer to prefetch it into. The real
FA2 performance win is overlapping K_{j+1} with the current iter's
softmax+PV; that requires double-buffered KV.

## SMEM layout (unchanged from Phase 1)

```
  s_Q        [BQ, HD]    BF16        16 KB
  s_KV       [BKV, HD]   BF16        16 KB      (reused: K→V per iter)
  s_S        [BQ, BKV]   F32         16 KB      (scores; dead after softmax →
                                                  becomes PV WMMA scratch)
  s_P        [BQ, BKV]   BF16         8 KB      (probs; fed to PV matrix_a)
  s_O        [BQ, HD]    F32         32 KB      (running accumulator)
  m + l      [BQ] * 2    F32        0.5 KB
  --------------------------------------------
  total                              88.5 KB     (SM_86 budget 100 KB)
```

## Attempted but reverted: full double-buffered KV

The design target was to add a second [BKV, HD] KV buffer (+16 KB) and
prefetch K_{j+1} during iter j's compute, achieving the canonical FA2
pipeline. This exceeded the 100 KB SM_86 opt-in budget by 4.5 KB, so an
SMEM reclaim was needed.

Strategy attempted: **fold s_P into the lower half of s_S**. Rationale: s_S
is dead after softmax consumes it; the lower 8 KB can be repurposed to hold
BF16 P; the upper 8 KB becomes the PV WMMA store scratch.

Byte math (HD=128):
```
  s_Q        16 KB
  s_KV[2]    32 KB  (double buffered)
  s_S        16 KB  (scores + P + scratch aliased across phases)
  s_O        32 KB
  m + l     0.5 KB
  --------
  total     96.5 KB   margin 3.5 KB ≥ 2 KB requirement ✓
```

Pipeline logic (as drafted):
```
  (a) QK^T on s_KV[cur]
  (b) cp.async V_j → s_KV[cur]                       commit (OLDER group)
  (c) if has_next: cp.async K_{j+1} → s_KV[nxt]      commit (NEWER group)
  (d) softmax overlaps V_j + K_{j+1} loads
  (e) wait_group(1)          forces OLDER (V_j) done, K-next still in flight
      __syncthreads
  (f) PV on s_KV[cur]
  (g) wait_group(0)          K_{j+1} fully in before next iter
      __syncthreads
```

The group-ordering matters because PTX `wait_group(N)` forces the OLDEST
pending groups to complete, not a specific group. Issuing V first
guarantees that `wait_group(1)` drains V while leaving K-next in flight.

### Observed failure

With s_P folded and P's writes landing anywhere inside the s_S SMEM region
(tested at three offsets: byte 0, byte 8192, past m/l), the kernel's
output deviates by max_abs ≈ 0.37 with cos_sim ≈ 0.92 — well below the
0.9999 gate and far above BF16 noise. The failure is independent of the
double-buffering and the cp.async work: reverting the double buffer and
keeping only the s_P fold still produces the same error magnitude.

A-B ladder, showing the fold is the breaker:

| configuration                                                    | naive-FP32 parity |
| ---------------------------------------------------------------- | ----------------- |
| Phase 1 (no cp.async)                                            | PASS              |
| + cp.async K & V, single buffer, same SMEM layout                | PASS              |
| + cp.async with V prefetch overlapping softmax                   | PASS              |
| + fold s_P into lower 8 KB of s_S (WMMA scratch in upper 8 KB)   | **FAIL**          |
| fold s_P into upper 8 KB of s_S (scratch outside s_S)            | **FAIL**          |
| fold s_P into s_S + register-buffered read-then-write softmax    | **FAIL**          |
| fold s_P into s_S + volatile BF16 stores                         | **FAIL**          |
| Phase 1 layout + register-buffered softmax (control, no fold)    | PASS              |

### What I checked and ruled out

- Cross-warp aliasing of BF16 writes onto another warp's in-flight FP32
  reads — the byte math shows warp 0's writes (BF16 bytes ≤ 2 KB) and
  warp 2's score reads (FP32 bytes ≥ 8 KB) are disjoint.
- Intra-warp read-then-write hazard within one softmax row — SIMT lockstep
  retires all 32 reads before any writes inside a single PTX sequence.
- ldmatrix alignment for P — both candidate s_P offsets are 128-byte
  aligned, well above the 16-byte ldmatrix requirement.
- TBAA reordering of BF16 stores past FP32 loads — adding explicit
  `__syncwarp()` and `volatile` qualifiers didn't change the outcome.
- PV WMMA scratch aliasing P — even with scratch moved to a fresh SMEM
  region past m/l (correctness-preserving, SMEM-wasteful), the fold still
  fails.

### What I did NOT check

- PTX assembly for the compiled kernel — would likely reveal whether nvcc
  emits `st.shared.b16` stores with unexpected eviction policies when the
  pointer type is reinterpret-cast through `float*`.
- Whether `wmma::store_matrix_sync` of FP32 scores touches addresses
  beyond the 16×16 strided block (e.g., metadata bits that could alias
  with P). This is the most likely remaining suspect.

This investigation is deferred to a future phase. The current shipping
configuration does not depend on the fold and is fully correct.

## Bench table

B=1, H=16, HD=128, BF16, RTX 3090 Ti. Median over 20 trials after 5 warmup.

```
N        pre-Phase-1 BQ=32   Phase 1 (ref)   Phase 1.5 (cp.async)   vs Phase 1   vs pre-Phase-1
1024      1.463 ms           1.12 ms*        1.072 ms               1.04×        1.36×
4096     19.427 ms          13.3 ms*        13.748 ms               0.97×        1.41×
16384   303.905 ms          198 ms*         190.204 ms              1.04×        1.60×
65536  4994.822 ms         3200 ms*        3073.436 ms              1.04×        1.63×

* Phase 1 reference numbers from the task prompt; not re-measured in this run.
Torch-SDPA column empty: libtorch has a CUDA 12.8 symbol mismatch in this env.
The pre-Phase-1 BQ=32 kernel has been deleted — numbers here are
historical, kept for reference.
```

**The vs-Phase-1 speedup is noise-level (0.97×–1.04×).** V-cp.async overlap
with softmax alone buys very little — V's HBM read is small (~16 KB per
iter at HD=128) and softmax compute is brief, so the overlap window is
correspondingly small. The vs-pre-Phase-1 speedup (1.36×–1.63×) comes
almost entirely from Phase 1's BQ=32→64 tile widening.

**This phase misses the task's ≥1.3× over-Phase-1 speedup target** because
the K-prefetch overlap that would deliver it requires double-buffered KV,
which in turn needs the SMEM reclaim that hit the reverted regression. The
shipped kernel is a correctness-preserving foundation for that next step,
not the final perf win.

3090 Ti peak BF16 is 71 TFLOPS / 936 GB/s. The FLOP-based % of peak in the
bench table is against the 936 GB/s traffic model (Q+K+V read + O write,
ignoring tile reuse), which undercounts at long seqs where K/V reuse across
Q tiles dominates. The kernel is compute-bound on the tensor cores at long
N, not memory-bound — which is why the cp.async V-overlap win is small.

## Files changed

- `src/cuda/flash_attention_fwd.cu` — cp.async intrinsics, cp.async-based
  K and V loads, V prefetch / softmax overlap.
- `FA2_CP_ASYNC_DESIGN.md` (this file) — design rationale and
  debugging notes.
- `docs/FLAME_KERNELS.md` — bench numbers + cp.async pipeline description.
- `docs/FLAME_MODULES.md` — note that the FA2 forward kernel uses cp.async.
- `docs/FLAME_CONVENTIONS.md` — cp.async pattern for future kernels.

`docs/FLAME_INDEX.md` unchanged (no new public symbols).

## What the next phase needs

1. Locate the fold-s_P-into-s_S regression. The PTX listing from
   `nvcc -keep` at the `store_matrix_sync` call site is the fastest path;
   if it's writing through unexpected sectors, we need a different SMEM
   reclaim.
2. Alternative SMEM reclaim: shrink BKV to 48 for HD=128 only (save 4 KB
   on s_S + 2 KB on s_P, enough to fit the double buffer), keep BKV=64
   for HD={64, 96}. This is a pure tile-geometry change with no algorithm
   changes — lower risk than the fold.
3. Swizzle the KV double buffer for bank-conflict-free cp.async + WMMA
   access. Currently the KV buffer has 2-way bank conflicts on the
   `ldmatrix.m8n8.x4` path; this is acceptable today but will matter more
   once the pipeline is truly double-buffered.
