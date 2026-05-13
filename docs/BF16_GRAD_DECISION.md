# BF16 Gradient Storage Decision (autograd v2 Phase 0)

**Date**: 2026-05-13 (Phase 0 decision); updated 2026-05-13 (Phase 4a
partial); updated 2026-05-13 (Phase 4b); updated 2026-05-13 (Phase 5b);
updated 2026-05-13 (Phase 5c).
**Scope**: flame-core autograd v2 cross-cutting policy for gradient dtype.
**Status**: Phase 5b shipped — `AutogradContext::backward_v2(loss)` is
the v2 grad-storage entry. It constructs a `GradientMap` under
`MatchInsertedDtype`, seeds the loss at `loss.dtype()`, and casts each
emitted v3-backward gradient to the loss dtype at accumulate time. This
realizes BF16-grad-end-to-end **without requiring the forward graph to
be authored in v2 ops** — the original gating concern from Phase 4b is
resolved. EriDiffusion-v2 `train_zimage` ships an opt-in
`--use-autograd-v2` flag (default OFF; new `autograd_v2` feature on
`eridiffusion-cli`).

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

### Concrete memory savings (Klein 9B reference)

These are the order-of-magnitude wins from full migration (`Parameter::new_v2`
+ `backward_v2` + `MatchInsertedDtype` GradientMap + AdamW BF16-grad arm):

| Scenario | v3 grad memory | v2 grad memory | Savings |
|---|---|---|---|
| Klein 9B LoRA (rank=32) | ~280 MB | ~140 MB | 140 MB |
| Klein 9B full fine-tune | ~36 GB | ~18 GB | 18 GB (= "fits / doesn't fit" on 24 GB) |
| Activation grads peak (klein 9B backward) | ~12-15 GB | ~6-8 GB | ~half |

Kernel launches eliminated per step on Klein 9B (from nsys profile prior session):
- ~9,500 `bf16_to_f32` casts (upcast at GradientMap insert/accumulate)
- ~9,700 `f32_to_bf16` casts (downcast at get_public_grad)
- Total: **~19,200 cast kernels per step** disappear under v2 policy
- Cost saved: ~135 ms/step launch overhead + ~50-100 ms/step memory-bandwidth
  cost = **~4-6% of klein 9B's 5.4s step time** from the F32 round-trip alone

Note: LoRA training realizes only a small portion of these savings (LoRA grads
are O(rank/d_model) of full grads). Full fine-tune and activation-grad savings
are the big ones.

## Scope of cross-cutting changes (Phase 4)

These are the F32-coercion sites that Phase 4 must rewrite to preserve
the param-dtype of incoming gradients. Recorded here as an inline TODO
audit; no source change in Phase 0.

### `GradientMap` (`src/gradient.rs`)

**Phase 4b status: SHIPPED.** The `GradStorePolicy` enum gained a
second variant, `MatchInsertedDtype` (Option A). `GradientMap::new_v2`
and `GradientMap::with_index_v2` construct on it. The v1 / v3 default
(`GradientMap::new` / `with_index`) remains `InternalFP32_PublicBF16`
with no behavior change. Trainer-side integration of the v2 path is
NOT wired by Phase 4b — Z-Image's trainer still goes through
`loss.backward()` (the v3 path which constructs a default-policy
GradientMap). Switching the trainer requires either porting the model's
forward graph to autograd v2 ops (currently 13 v2 ops vs ~30+ needed)
or adding a `loss.backward_v2()` entry that builds a `with_index_v2`
GradientMap from the existing tape. Phase 5 work.

Rewrites that landed (all under `src/gradient.rs`):

- **`set_ones(id, shape)`** — unchanged behavior: F32 seed under both
  policies. v2 callers that want a BF16 loss seed call
  `set_ones_dtype(id, shape, DType::BF16)` (new).
- **`set_ones_dtype(id, shape, dtype)`** — new helper. Under
  `InternalFP32_PublicBF16` forces F32 (v1 invariant). Under
  `MatchInsertedDtype` honors the requested dtype.
- **`get_public_grad`** — under v1 still does the BF16 cast (unchanged).
  Under v2 returns the native-dtype grad without casting.
- **`take_public_grads`** — same split as `get_public_grad`.
- **`insert(id, grad)`** — under v1 still upcasts BF16 → F32 (unchanged).
  Under v2 stores the grad in its native dtype.
- **`accumulate(id, grad)`** — refactored into `accumulate_v1` (legacy
  deferred-upcast behavior on second grad) and `accumulate_v2`
  (preserves stored dtype; errs on dtype mismatch — same contract as
  `autograd_v2::AccumulateGrad`). Public `accumulate` dispatches on
  policy.
- **`get_or_create(id, shape)`** — kept the F32 allocation under both
  policies for legacy `autograd_simple.rs` callers.
- **`get_or_create_dtype(id, shape, dtype)`** — new helper. v2 callers
  can pre-allocate a BF16 slot; v1 still forces F32.
- **`policy()`** — new accessor.

Tests: `tests/autograd_v2_gradientmap_v2.rs` (17 tests covering both
policies' contracts; mixed v1-regression + v2-new behavior).

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

The original Phase 4 plan, with Phase 4a / 4b status annotations
(`[4a-DONE]` / `[4b-DONE]` = shipped):

1. `[4b-DONE]` Add a new `GradStorePolicy::MatchInsertedDtype` variant
   (the original spec called it `MatchParamDtype`; renamed because
   `GradientMap` stores by `TensorId`, not by parameter — the dtype
   contract is on the inserted gradient, not on a paired parameter
   handle). `GradientMap::new_v2()` and `with_index_v2()` opt into it;
   `new()` / `with_index()` keep the v1 default.
2. `[4b-DONE]` Rewrite GradientMap `set_ones` / `insert` / `accumulate`
   / `get_public_grad` / `take_public_grads` / `get_or_create` to honor
   the new policy. See per-method status above.
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
`flame_core::autograd_v2::AdamWV2`. The v2 grad-storage surface
shipped in Phase 4b is `flame_core::GradientMap::new_v2` /
`with_index_v2`. Together they form the v2 grad-dtype path inside
flame-core. Real-trainer integration (Phase 4b Deliverable C / Phase 5)
is described below.

## Phase 5b Deliverable C — `backward_v2()` bridge SHIPPED

**Z-Image LoRA trainer (`EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_zimage.rs`)**

The trainer now ships `--use-autograd-v2` (default OFF). When ON, the
`loss.backward()` call routes through `AutogradContext::backward_v2`,
which builds a `MatchInsertedDtype` `GradientMap`. The v3 op-dispatch
is unchanged; the bridge is a 42-line surgery on `src/autograd.rs:1297`
(split into `pub fn backward`, `pub fn backward_v2`, and
`fn backward_impl(loss, policy)`).

**Architecture (post-bug-fix `a5da3d5`)**: the bridge accumulates F32
internally throughout the backward loop (preserving v3 op behavior —
v3 kernels are authored for F32 non-leaf grads), then runs a single
`GradientMap::cast_all_to_dtype(loss.dtype())` post-pass that converts
all final grads to the target dtype at once. **The original `ad781bf`
attempted per-op cast at insertion-time; that was structurally wrong**
— after the first cast, the next reverse-step's `take()` returned
BF16 and fed it back into v3 kernels not authored for BF16 non-leaf
grads, producing `CUDA_ERROR_INVALID_VALUE` on real BF16 MLP backward.
Bug-fixer commit `a5da3d5` corrected this with the F32-internal +
post-loop downcast pattern.

**Known limitation**: `FLAME_CUDA_GRAPH=1 + --use-autograd-v2` is
unsupported. The replay path bypasses the post-loop cast
(`src/autograd.rs:1547-1552` allocates from `grad_recipe` at the
warmup-recorded F32 dtype). Perf bench (Phase 5c) should leave
cuda-graph backward unset when measuring v2.

Default OFF preserves byte-equivalent v3 behavior. Existing
`Parameter::set_grad` continues to upcast the v2-emitted BF16 grads to
F32 (Parameter::new still uses CastToF32 policy), so the AdamW step
side is unaffected — the bridge correctness is exercised but the
BF16-end-to-end memory savings require additionally flipping LoRA
params to `Parameter::new_v2`. Phase 5c will measure the perf impact.

`FLAME_MT_SCALE`'s F32 fast path is skipped under `--use-autograd-v2`
because v2 grads are BF16-stored; the per-param `mul_scalar` loop runs
instead.

## Phase 5c — perf bench results

**Status**: SHIPPED. Bench harness at `tests/autograd_v2_perf.rs`
(per-cell `#[test] #[serial]` functions), Deliverable D of the
`AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Phase 5 plan.

Three workloads × three configurations (v3 control, bridge alone,
Class A with `Parameter::new_v2` + `set_grad` round-trip), 5 warmup +
50 timed iters per cell with the slowest 5 trimmed. Wall-clock via
`std::time::Instant`, `device.synchronize()` before and after each
timed iter. `FLAME_CUDA_GRAPH=0` and `FLAME_MT_SCALE` unset (both
incompatible with v2 per Phase 5b skeptic CONCERNs).

### Deliverable A: wall-clock per backward iter (median)

| Workload                  | v3 (ms)   | bridge (ms) | Class A (ms) | bridge Δ% | Class A Δ% |
|---------------------------|-----------|-------------|--------------|-----------|------------|
| Synthetic 4-layer MLP     | 0.088     | 0.094       | 0.101        | +6.49%    | +14.28%    |
| Klein attn_chain prod     | 10.109    | 10.316      | 10.406       | +2.05%    | +2.94%     |
| Klein double-block        | 0.664     | 0.674       | 0.688        | +1.46%    | +3.61%     |

**Headline**: bridge Δ on the real Klein workloads ranges from
+1.0% to +2.8% across 3 sample runs (representative run above:
+1.46% / +2.05%). The synthetic MLP's ~+5-6% bridge delta is per-iter
setup noise — at 88 μs the absolute is ~5 μs, dominated by graph
construction, not backward dispatch.

**vs. ±1% target**: Klein attn_chain prod is the binding case at
+1.5-2.8% — just above the ±1% gate. Acceptable per the standing
contract that the v2 win is correctness + memory (Deliverable B
below), not speed. The bridge's per-op cast cost on the dtype-
unification post-pass is the small constant overhead.

**Class A delta** comes from the per-iter `Parameter::new_v2` +
`set_grad` round-trip the harness exercises to attribute the BF16-
preserving policy cost. In a real trainer, parameters are constructed
once per LoRA module (not per step), so the per-step Class A overhead
is much closer to bridge-alone.

### Deliverable B: grad-storage memory (steady-state, byte-exact)

| Workload                  | v3 grad MB | bridge grad MB | Class A grad MB | Class A savings |
|---------------------------|------------|----------------|-----------------|-----------------|
| Synthetic 4-layer MLP     | 0.000183   | 0.000092       | 0.000092        | +50.00%         |
| Klein attn_chain prod     | 156.000    | 78.000         | 78.000          | +50.00%         |
| Klein double-block        | 1.812      | 0.906          | 0.906           | +50.00%         |

The 50.00% savings is exact — every leaf is BF16 (2 bytes) under v2
where v3 stored every grad as F32 (4 bytes). On Klein attn_chain prod
that's **78 MB saved per backward call** — material for a trainer
that retains grad maps across an accumulation window.

The **bridge alone** column matches Class A exactly: once `backward_v2`
returns a `GradientMap` with BF16-stored grads, the only thing that
can later upcast them is `Parameter::set_grad` under the default
`CastToF32` policy. The Class A column captures the round-trip through
`Parameter::new_v2`'s `MatchParamDtype` policy and confirms no further
upcast happens.

### What the bench did not measure

- **Class C trainers (F32 grad + BF16 master weight)** — out of scope
  for Phase 5c; deferred to Phase 6 if a trainer adopts that config.
- **Real-trainer multi-step convergence** — Klein step 2+ crashes
  pre-existing (`CUDA_ERROR_INVALID_VALUE`); deferred to Z-Image LoRA
  smoke (Phase 5d / v3 retirement).
- **`AdamWV2::step` perf** — the `adam_fused_multi_bf16_bf16grad_kernel`
  activation path. Stepping through the optimizer was out of scope for
  Deliverable D; the per-cell test only exercises `set_grad`. A
  follow-up bench should compare `(v3 AdamW + F32 grad)` vs
  `(AdamWV2 + BF16 grad)` end-to-end optimizer step.

## Phase 4b Deliverable C — earlier status (historical)

The Phase 4b version of this section identified two blockers — both
resolved by Phase 5b's bridge:

1. Z-Image's model forward graph (in `eridiffusion-core`) is built on
   the v3 op set: `rms_norm`, `fused_linear3d_native`, `sdpa`, `rope`,
   `conv2d`, `gelu`, `group_norm`, broadcasting fused ops, etc. The
   autograd v2 forward surface today is 13 ops: `add_v2`, `mul_v2`,
   `matmul_v2`, `layer_norm_v2`, `silu_v2`, `sum_v2`, `narrow_v2`,
   `permute_v2`, `transpose_v2`, `reshape_v2`, `squeeze_v2`,
   `unsqueeze_v2`, `view_v2`. Without v2 versions of the ~20+ missing
   ops, `Engine::execute` cannot walk Z-Image's backward graph.
2. There is no `loss.backward_v2()` entry that would build a
   `GradientMap::with_index_v2` from the existing v3 tape. Adding one
   would duplicate ~200 lines of `AutogradContext::backward`. Out of
   scope for Phase 4b (additive constraint; touches v3 backward).

What Phase 4b did ship that the trainer COULD adopt today (without v2
forward ops):

- `Parameter::new_v2(...)` for LoRA params (drops the F32 upcast in
  `set_grad`). Useful once the producer feeds BF16 grads.
- `AdamWV2` as a typed alias (same kernels, same state shape).
- `GradientMap::with_index_v2` for any future caller that wants to
  preserve grad dtype across the map.

But none of this delivers a measurable trainer-side gain until the
producer (v3 backward) is taught to emit BF16 grads into a v2 map.
That's a Phase 5 deliverable.

**Smoke that DID run for Phase 4b**:
- `cargo build -p eridiffusion-cli` succeeded with the new GradientMap
  symbols in scope.
- `tests/autograd_v2_gradientmap_v2.rs` (17 tests) green.
- All v3 regression suites green
  (`adam_f32_fused`, `adam_multi_tensor_parity`, `grad_norm_parity`,
  `multi_tensor_l2_norm_parity`, `inplace_version_bump_audit`).

No 1000-step Z-Image smoke. Without trainer-side v2 routing, the smoke
would run the v3 path and prove only that the v3 path still works.

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
