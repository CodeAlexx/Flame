# BF16 Gradient Storage Decision (autograd v2 Phase 0)

**Date**: 2026-05-13
**Scope**: flame-core autograd v2 cross-cutting policy for gradient dtype.
**Status**: Decision recorded. No behavior change in Phase 0 — full
implementation lands in Phase 4 (optimizer + trainer integration).

## Decision

**Option A — Class A path**: when a parameter is BF16, its gradient is
stored as **BF16 end-to-end** through autograd, the gradient map, the
parameter handle, and the optimizer. F32 is permitted **only as
`opmath_t` inside kernels** (accumulators, reductions, scratch).

This supersedes the v1 `InternalFP32_PublicBF16` policy, which
unconditionally upcasts every gradient to F32 in the gradient map,
parameter handle, and optimizer entry path. Under v1 a BF16-param /
BF16-activation training graph pays 2× gradient memory and an extra
F32↔BF16 cast on every backward op.

## Why Option A

- **Class A recovery (memory + perf)**: BF16 grads halve gradient
  memory, eliminate the per-op F32 cast around the gradient map, and
  unlock the BF16-grad Adam kernel path that is currently dead code
  (see `src/adam.rs` line 1273-1283 — `adam_fused_f32param_bf16grad_kernel`
  is implemented but unreachable because `Parameter::set_grad` casts
  away the BF16 dtype upstream).
- **Reference parity**: PyTorch matches gradient dtype to parameter
  dtype by default; flame-core's BF16 trainers (Klein, Z-Image, Chroma,
  Qwen, ERNIE) currently diverge here. Option A closes that gap.
- **Defer-cost is real**: keeping F32 internal grads "until later" was
  the v1 default and has accumulated a long tail of `to_dtype(F32)`
  detours that need rewriting eventually. Doing it once with v2 is
  cheaper than doing it twice.

## Scope of cross-cutting changes (Phase 4)

These are the F32-coercion sites that Phase 4 must rewrite to preserve
the param-dtype of incoming gradients. Recorded here as an inline TODO
audit; no source change in Phase 0.

### `GradientMap` (`src/gradient.rs`)

- **Line 99** — `set_ones(...)` hard-codes `Tensor::ones_dtype(..., DType::F32, ...)`.
  Under Option A: must allocate ones in the loss tensor's dtype (BF16
  for a BF16 loss).
- **Lines 145-153** — `get_public_grad(...)` does `g_fp32.to_dtype(DType::BF16)`.
  Under Option A: grad is already BF16 — no cast needed. The
  `GradStorePolicy::InternalFP32_PublicBF16` enum variant becomes a
  legacy v1-only path; v2 needs a new variant `MatchParamDtype` (or
  similar) that returns grads in their native storage dtype.
- **Lines 156-170** — `take_public_grads(...)` same pattern as above.
- **Lines 184-189** — `insert(...)` calls `grad.to_dtype(DType::F32)`
  unconditionally. Under Option A: preserve grad dtype.
- **Lines 209-213** — `ensure_f32` helper inside `accumulate()` upcasts
  any non-F32 existing entry. Under Option A: instead, accumulate in
  the target dtype with F32 only inside the kernel.
- **Lines 218-223** — `add_to_existing` (in `accumulate`) casts incoming
  grad to F32 when dtypes differ. Under Option A: enforce matching
  dtypes via the upstream contract; cast only as opmath inside the
  kernel.
- **Lines 245, 253** — `Tensor::zeros_dtype(shape, DType::F32, ...)`
  in `get_or_zeros`-style helpers. Under Option A: take the dtype from
  the inserted grad / parameter.

### `Parameter` (`src/parameter.rs`)

- **Lines 119-123** — `Parameter::set_grad(...)` casts every incoming
  grad to F32 unconditionally. Under Option A: preserve dtype; if a
  trainer needs F32 grads, it casts upstream.
- **Lines 189-197** — `Parameter::apply_update(...)` casts the update
  to F32, computes the sub in F32, then writes back. Under Option A:
  follow the param's dtype; F32 only as opmath inside the kernel.

### Adam optimizer (`src/adam.rs`)

- **Lines 1075-1082** (comment) — multi-tensor fast path assumes
  "BF16 params + F32 grads" because `Parameter::set_grad` casts. Under
  Option A: this assumption breaks — the classifier needs a 3rd case
  "(BF16 param, BF16 grad)" routed to
  `adam_fused_multi_bf16_bf16grad_kernel` (already in
  `MODULE_NAME`, currently unused).
- **Lines 1106, 1112** — classifier checks `g.dtype() == DType::F32`.
  Under Option A: allow BF16 grads too.
- **Lines 1273-1283** (comment + kernel name) —
  `adam_fused_f32param_bf16grad_kernel` is unreachable today; Option A
  makes it live. Same for the BF16-param BF16-grad kernel inside
  `fused::adam_fused_step`.

### SGD optimizer

There is no `src/sgd.rs` — SGD-style optimizers live in
`src/sgd.rs` (the module name in lib.rs) but checks need to confirm.
At Phase 4 audit time, sweep that module for the same `to_dtype(F32)`
patterns.

### Grad norm (`src/ops/grad_norm.rs`)

- **Line 12** (doc) — example shows `g.to_dtype(F32)?.square()?.mean()?`.
  Under Option A: the doc stays correct as opmath, but the public
  helper must accept BF16 grads.
- **Lines 90, 108, 111** — `multi_tensor_l2_norm_sq_f32` enforces
  "all F32" and the per-tensor fallback casts to F32. Under Option A:
  the fast path needs a BF16 sibling
  (`multi_tensor_l2_norm_sq_bf16`), and the per-tensor fallback keeps
  F32 only as opmath.

### Trainer-side callers (out of crate)

Trainers that explicitly read grads as F32 (e.g., Klein, Z-Image
checkpointing) need their grad-reading paths updated to accept the
param-dtype. The Phase 4 work in `EriDiffusion-v2` is a separate
audit; not in scope for this doc.

## Migration strategy (Phase 4)

The Phase 4 plan, gated by `#[cfg(feature = "autograd_v2")]`:

1. Add a new `GradStorePolicy::MatchParamDtype` variant; default v2
   `GradientMap::new()` uses it.
2. Rewrite `set_ones` / `set` / `insert` / `accumulate` /
   `get_public_grad` to honor the new policy.
3. Rewrite `Parameter::set_grad` and `apply_update` to preserve dtype
   under v2 (gate via the `autograd_v2` feature flag — v1 path stays
   unchanged for parity-gate runs).
4. Wire the BF16-grad Adam kernels (single + multi-tensor) into the
   classifier under v2.
5. Add `multi_tensor_l2_norm_sq_bf16` and route v2 grad-norm through
   it.
6. Add optimizer parity tests: same starting state under
   `(BF16 param, F32 grad)` v1 vs `(BF16 param, BF16 grad)` v2 must
   converge to within a documented tolerance, since the two are NOT
   bit-equal (BF16 grad accumulation truncates differently from
   F32 → BF16 cast at the end).

## What this is NOT

- **Not a Phase 0 code change**: nothing in this doc has been
  implemented. Phase 0 only adds the feature flag, the empty module,
  the in-place version-bump audit + tests, and this decision record.
- **Not a parity-gate metric**: bit-equal v1 vs v2 is impossible by
  construction once we change the gradient dtype contract. The Phase 5
  parity gate uses convergence-style metrics
  (loss-after-N-steps within tolerance) for trainers that flip to v2.
- **Not a Class B fix**: Class B (narrow backward F32 detour) is a
  separate workstream — see §10 of
  `AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`. Option A reduces gradient
  memory and removes the per-op cast, but doesn't touch
  `src/tensor_narrow.rs`'s F32-only backward kernel.
- **Not a Class E fix**: Class E (cudarc sync-copy sweep) is a
  separate workstream — same reference §10.

## References

- `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §7 — original BF16-grad
  storage discussion.
- `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §10 — Class A / B / E
  scope separation.
- `src/adam.rs` lines 1273-1283 — proof that the BF16-grad Adam path
  is implemented but unreachable today.
