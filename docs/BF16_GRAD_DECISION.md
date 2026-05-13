# BF16 Gradient Storage Decision (autograd v2 Phase 0)

**Date**: 2026-05-13 (Phase 0 decision); updated 2026-05-13 (Phase 4a partial).
**Scope**: flame-core autograd v2 cross-cutting policy for gradient dtype.
**Status**: Phase 4a partial. The `Parameter` + `Adam` + `grad_norm`
F32-coercion sites listed below have been rewritten or have the
dtype-preserving path wired (see "Phase 4a status" markers per site).
The `GradientMap` rewrite and trainer integration smoke are deferred to
Phase 4b. See `src/autograd_v2/optim.rs` for the v2-facing optimizer
surface added in Phase 4a.

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

**Phase 4a status: DEFERRED to Phase 4b.** v2 grads do not flow through
`GradientMap` on the recording path (they land in
`AutogradMetaV2::grad` via `AccumulateGrad`). The Phase 4b decision is
whether to add the `MatchParamDtype` variant here or route v2 grads
exclusively through `meta.grad` and leave GradientMap as a pure v3
artifact. Defer until trainer integration smoke shows what shape
trainers actually need.

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

**Phase 4a status: SHIPPED.** `Parameter` carries a `GradDtypePolicy`
field with variants `CastToF32` (v1/v3 default — unchanged) and
`MatchParamDtype` (new — Option A). `Parameter::new_v2(t)` constructs
with the v2 policy. v3 trainers continue to use `Parameter::new(t)`
and see no behavior change.

- **Lines 199-218** — `Parameter::set_grad(...)` honors policy: under
  `CastToF32` casts to F32 as before; under `MatchParamDtype`
  preserves the incoming grad's dtype.
- **Lines 303-340** — `Parameter::apply_update(...)` honors policy:
  under `CastToF32` computes in F32 as before; under
  `MatchParamDtype` brings the update to the param's native dtype and
  computes `data - update` at the param's dtype (F32 still permitted
  as opmath inside the kernel).
- **Line 243** — `Parameter::grad_bf16_or_f32()` new accessor —
  returns the native-dtype grad without casting.

### Adam optimizer (`src/adam.rs`)

**Phase 4a status: SHIPPED.** Classifier extended from a 2-tuple
optional to a 4-arm dispatch (`(BF16, F32)`, `(BF16, BF16)`,
`(F32, F32)`, fall-through for `(F32, BF16)` which has no multi-tensor
kernel but has a per-param `adam_fused_step_f32` arm with the
`adam_fused_f32param_bf16grad_kernel` already wired). The previously-
dead `adam_fused_multi_bf16_bf16grad_kernel` is now reachable.

- **Lines 1107-1144** — Phase 4a four-way classifier. Tests in
  `tests/autograd_v2_phase4a.rs`:
  - `adam_step_bf16_param_bf16_grad_no_panic`
  - `adam_step_bf16_param_bf16_grad_matches_f32_reference`
  - `adam_step_f32_param_bf16_grad_no_panic`
- **Lines 1167-1187** — grad-pointer pack region: BF16 branch added
  for `grad_is_bf16` case.
- **Line 1213** — `grad_is_bf16` flag now propagates to
  `fused::adam_fused_multi_tensor_step` (no longer hard-coded false).

### SGD optimizer

There is no `src/sgd.rs` — SGD-style optimizers live in
`src/sgd.rs` (the module name in lib.rs) but checks need to confirm.
At Phase 4 audit time, sweep that module for the same `to_dtype(F32)`
patterns.

### Grad norm (`src/ops/grad_norm.rs` + `src/ops/multi_tensor.rs`)

**Phase 4a status: SHIPPED.** `multi_tensor_l2_norm_sq_bf16` exists at
`src/ops/multi_tensor.rs:447` with its own stage-1 BF16 kernel (F32
opmath); the F32 stage-2 reducer is shared.
`global_l2_norm` in `src/ops/grad_norm.rs:99` routes all-BF16-contiguous
slices through it. The per-tensor F32 fallback below still casts BF16
to F32 — that's an opmath cast, not a grad-storage cast, so under
Option A it's the documented exception.

- **Line 12** (doc) — example shows `g.to_dtype(F32)?.square()?.mean()?`.
  Now correct as opmath; the public helper accepts BF16 grads through
  the fast path.
- **Lines 90, 108, 111** — `multi_tensor_l2_norm_sq_f32` is the F32
  fast path; `multi_tensor_l2_norm_sq_bf16` is its BF16 sibling. Tests
  in `tests/autograd_v2_phase4a.rs`:
  - `multi_tensor_l2_norm_sq_bf16_matches_f32_reference`
  - `global_l2_norm_routes_bf16_through_fast_path`

### Trainer-side callers (out of crate)

Trainers that explicitly read grads as F32 (e.g., Klein, Z-Image
checkpointing) need their grad-reading paths updated to accept the
param-dtype. The Phase 4 work in `EriDiffusion-v2` is a separate
audit; not in scope for this doc.

## Migration strategy (Phase 4)

The original Phase 4 plan, with Phase 4a status annotations
(`[4a-DONE]` = shipped; `[4b]` = deferred):

1. `[4b]` Add a new `GradStorePolicy::MatchParamDtype` variant; default
   v2 `GradientMap::new()` uses it.
2. `[4b]` Rewrite GradientMap `set_ones` / `set` / `insert` /
   `accumulate` / `get_public_grad` to honor the new policy.
3. `[4a-DONE]` Rewrite `Parameter::set_grad` and `apply_update` to
   preserve dtype under v2 — gated via `GradDtypePolicy` enum on the
   parameter (`Parameter::new_v2(t)` constructs with the new policy).
   The v1 path is unchanged for parity-gate runs.
4. `[4a-DONE]` Wire the BF16-grad Adam kernels (single + multi-tensor)
   into the classifier under v2.
5. `[4a-DONE]` Add `multi_tensor_l2_norm_sq_bf16` and route v2 grad-
   norm through it.
6. `[4a-DONE]` Add optimizer tolerance test: `(BF16 param, BF16 grad)`
   v2 vs `(F32 param, F32 grad)` reference must converge to within
   BF16 tolerance (5e-3 absolute on the post-step param at lr=1e-3,
   verified by
   `adam_step_bf16_param_bf16_grad_matches_f32_reference`).

The v2-facing entry point shipped in Phase 4a is
`flame_core::autograd_v2::AdamWV2` (a thin wrapper around `AdamW`).
Trainer integration smoke is Phase 4b.

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
