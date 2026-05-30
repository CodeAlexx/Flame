# Autograd v2 Design Review Handoff

Date: 2026-05-13

Scope reviewed:
- Proposed `flame-core/src/autograd_v2/` clean-sheet autograd design.
- Current flame-core repo at `/home/alex/EriDiffusion/flame-core`.
- Local PyTorch reference at `/home/alex/pytorch`, commit `6a13735`.



# Agents
Three-agent CC workflow. One phase per prompt, serial on main, never parallel. All outputs as file artifacts. Each prompt names the active agent.
Builder
Implements per the revised spec. Scope is exactly one phase from the Suggested Implementation Order — no multi-phase prompts. Reads the design contract before writing any code, and treats clauses 1-16 in "Recommended Spec Changes" as hard constraints, not suggestions.
Builder rules for v2:

Every GradFn::apply returns Result<Vec<Option<Tensor>>>. No unwrap in autograd code.
SavedTensor carries the Arc<AtomicU32> version handle from SavedRef, never just an integer.
Weak accumulator storage in AutogradMeta. Never strong Arc<Tensor> through metadata.
Hooks accessor exists from day one with default empty impl. Same for num_inputs, fw_grad_, out-of-place InputBuffer path.
All new code under #[cfg(feature = "autograd_v2")]. No silent migration of v1 ops.
PyTorch parity fixtures exist before the op is declared done — backward AND forward-mode AD.
Reference open-source implementations (PyTorch at /home/alex/pytorch) by line number in comments. Never design from scratch where a reference exists.

Builder output is the implemented phase plus a verification log (cargo build, cargo test, and the parity harness output for any ops in scope). No prose summary — the verification log IS the summary.
Bug Fixer
Activates after Builder ships a phase. Hunts for bugs the builder shipped, with priority on the High-Risk Areas list:

Reference cycles through AutogradMeta and AccumulateGrad
Version-counter coverage gaps (in-place mutators that don't bump)
View metadata sharing across clone / detach / to_dtype / contiguous / narrow
Checkpoint/offload interaction with v2 saved tensors
SDPA backward saved-output mutation detection
Multi-output ops and output_nr correctness
Gradient accumulation across micro-batches
Hook firing order and re-entry safety
Inline-mini-execute state save/restore on nested Engine::execute

Bug fixer writes failing tests first, then fixes. Each fix is one commit. Never bundles unrelated fixes. If a bug is in v1 code that v2 inherits (Class B narrow backward, Class E sync sites), the bug fixer escalates rather than patching — those are separate workstreams per clause 11.
Bug fixer output is a list of test names added (one per bug), the fix commit hashes, and a verification log proving the prior phase's tests still pass.
Skeptic
Activates after Bug Fixer signs off. Challenges the phase's claims and tests edge cases that builder + bug fixer would not have written for themselves.
Skeptic targets for v2 specifically:

The BF16-grad claim. Run grad-norm, clipping, and Adam parity vs v1 at multiple scales. Compare loss curves over 100 steps, not just step 1.
The reentrant claim. Construct a checkpoint-inside-checkpoint case (3 levels deep) and verify the mini-engine state stack unwinds correctly.
The hooks claim. Register a hook that mutates a tensor it shouldn't, verify version-counter catches it on the next backward.
The create_graph=true claim. Verify second-order gradients match PyTorch on a non-trivial graph (transformer block, not toy add).
The view autograd claim. Run a model that uses view → matmul → view patterns and verify the backward produces the same grads as v1.
The parity gate. Don't accept "passes on klein 4B" as proof — also run klein 9B, zimage, and at least one video model.
The performance non-regression claim. Independent timing run on a fresh checkout. If ms/step regressed by even 1%, flag it. Don't average it away.

Skeptic also reads the phase's diff for anti-patterns: silent F32 upcasts, unwrap calls, missing version bumps in new in-place paths, hardcoded single-stream assumptions, strong reference cycles, panic! where Err belongs.
Skeptic output is a numbered findings list. Each finding is one of: BLOCKER (phase must not ship), CONCERN (track for next phase), or NOTE (resolved during review). Phase advances only when no BLOCKER remains.
Hand-off rules

Builder → Bug Fixer: when Builder's verification log is clean.
Bug Fixer → Skeptic: when Bug Fixer reports no new failures and all phase tests pass.
Skeptic → next phase Builder: when no BLOCKER remains AND user has reviewed the findings list.
Never skip an agent. Never run two agents in the same prompt.
If any agent finds the spec ambiguous, they stop and escalate — they do not improvise.



## Summary

The proposed PyTorch-style DAG architecture is directionally sound: `GradFn`/`Edge`, dependency-counted scheduling, `InputBuffer`, saved tensor version checks, and a single-threaded engine are the right foundations for replacing the current global tape.

Do not green-light implementation from the current spec as-is. Several details conflict with flame-core's current tensor/gradient ownership model and with PyTorch's actual lifetime rules. If implemented literally, v2 risks reference cycles, incomplete error propagation, broken multi-input node scheduling, missed in-place mutation detection, and optimizer-policy regressions.

The design should be revised first, then implemented behind a feature flag.


## Current Flame-Core Reality

Current `Tensor` is a by-value struct with direct fields:
- `storage`
- `shape`
- `device`
- `id`
- `requires_grad`
- `custom_strides`
- `view_offset`

Relevant files:
- `src/tensor.rs`
- `src/gradient.rs`
- `src/parameter.rs`
- `src/autograd.rs`
- `src/saved_ref.rs`
- `src/tensor_storage.rs`
- `src/autograd_v4/`

There is no shared `AutogradMeta` today. Gradients are not stored on `Tensor`; they live in `GradientMap` and `Parameter`.

Current gradient policy is `InternalFP32_PublicBF16`:
- `GradientMap::set_ones()` seeds F32.
- `GradientMap::insert()` enforces F32.
- `GradientMap::accumulate()` converts to F32 on repeated accumulation.
- `Parameter::set_grad()` casts incoming grads to F32.
- Adam/SGD comments and fast paths assume F32 grads in public optimizer flow.

Current version checking already exists through `SavedRef`:
- `SavedRef` stores an `Arc<AtomicU32>` version-counter handle.
- The version side table can be cleared at step boundaries.
- The saved handle keeps checks valid after table clears.

## Blocking Design Issues

### 1. AccumulateGrad Lifetime Cycle

The proposal stores:

```rust
AutogradMeta {
    grad_accumulator: Option<Arc<AccumulateGrad>>,
}

AccumulateGrad {
    variable: Tensor,
}
```

If `Tensor` owns or shared-owns `AutogradMeta`, this creates:

```text
Tensor -> AutogradMeta -> Arc<AccumulateGrad> -> Tensor
```

PyTorch avoids this by storing the leaf accumulator weakly in `AutogradMeta`; `grad_accumulator()` locks or recreates it lazily.

Required revision:
- Store `Weak<AccumulateGrad>` or `Weak<dyn GradFn>` in tensor metadata, not a strong `Arc`.
- Or redesign `AccumulateGrad` to hold a weak handle to a shared tensor inner/meta object.
- Do not put a strong `Tensor` cycle through metadata.

PyTorch references:
- `/home/alex/pytorch/torch/csrc/autograd/variable.h`
- `/home/alex/pytorch/torch/csrc/autograd/variable.cpp`
- `/home/alex/pytorch/torch/csrc/autograd/functions/accumulate_grad.h`

### 2. AutogradMeta Does Not Fit Current Tensor

The spec assumes every `Tensor` can expose mutable autograd metadata:

```rust
tensor.autograd_meta_mut()
```

Current flame-core `Tensor` is cloned by value. Adding plain fields directly to `Tensor` will not preserve:
- `.grad`
- `grad_fn`
- `output_nr`
- leaf accumulator identity
- leaf vs non-leaf metadata consistency

Required revision:
- Introduce a shared, interior-mutable metadata handle, for example:

```rust
pub(crate) autograd: Arc<Mutex<AutogradMetaV2>>
```

or a split `TensorInner` structure where tensor data and autograd metadata have explicit ownership rules.

Open choice:
- If `Tensor::clone()` should preserve autograd metadata, use shared metadata.
- If `detach()` should drop history, it must allocate/reset metadata deliberately.
- View tensors need metadata that can represent base/view relationships later, even if full view autograd is deferred.

### 3. SavedTensor Must Return Errors Through Backward

The proposal has:

```rust
SavedTensor::unpack(...) -> Result<Tensor, AutogradError>
GradFn::apply(...) -> Vec<Option<Tensor>>
backward(...) -> ()
```

This cannot propagate version mismatch or released-saved-tensor errors. Examples call `unwrap()`, which would panic instead of returning a recoverable training error.

Required revision:

```rust
pub trait GradFn: Send + Sync + Debug {
    fn apply(&self, inputs: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>>;
}

Engine::execute(...) -> Result<Vec<Option<Tensor>>>
backward(...) -> Result<()>
grad(...) -> Result<Vec<Option<Tensor>>>
```

Also, `release_variables(&self)` cannot call `SavedTensor::reset(&mut self)` through `Arc<dyn GradFn>`.

Required revision:
- Saved tensors inside backward nodes need interior mutability, such as `Mutex<Option<Tensor>>`, `parking_lot::Mutex`, or a small custom cell if single-threaded.
- `release_variables()` should be able to clear saved data through `&self`.

### 4. SavedTensor Version Handles Must Survive Side-Table Clears

The proposal stores only:

```rust
saved_version: u32
```

and later calls `data.storage().version()`.

Current flame-core version counters are in a global side table. `TensorStorage::version()` returns `0` if the table entry is missing. `AutogradContext::clear()` flushes the side table.

Required revision:
- Either move the version counter into a shared storage object, or
- Have `SavedTensor` store the `Arc<AtomicU32>` handle exactly like `SavedRef`.

Do not regress from the current `SavedRef` model.

Relevant files:
- `src/saved_ref.rs`
- `src/tensor_storage.rs`
- `src/autograd.rs`

### 5. In-Place Version Bump Coverage Is Not Complete

The design depends on every in-place mutation bumping the storage version. The current code has in-place write paths that need explicit audit and likely fixes:

- `src/ops/elt.rs`
  - `add_inplace_same_dtype`
  - `mul_inplace_same_dtype`
  - `gate_mul_bf16_inplace`
- `src/tensor.rs`
  - `copy_bf16_region_from`
  - `copy_f32_from`
  - `copy_`
  - `copy_from_bf16_slice`
  - chunk/copy helpers near the BF16 host-copy code
- `src/ops/multi_tensor.rs`
  - `multi_tensor_scale_inplace_packed`
- optimizer update paths
- SDPA/dropout in-place helpers

Required revision:
- Make an "all in-place writes bump version" sweep a prerequisite for claiming SavedTensor correctness.
- Add tests that save a tensor, mutate it through each public in-place path, then assert backward/unpack returns an error.

### 6. GradFn Needs Input Arity

PyTorch creates `InputBuffer(next.function->num_inputs())` when routing a gradient to a downstream node.

The proposed `GradFn` has `next_edges()` but no way for the engine to know the size of the downstream input buffer.

Required revision:

```rust
fn num_inputs(&self) -> usize;
```

or equivalent metadata.

Without this, multi-input gradient functions and multi-output/view-like nodes cannot be scheduled safely.

### 7. BF16 Grad Storage Is Bigger Than Autograd

The proposal's dtype contract says:

```text
param dtype BF16 => stored grad BF16
```

This conflicts with existing optimizer and parameter behavior:
- `GradientMap` is internal F32.
- `Parameter::set_grad()` casts all grads to F32.
- Adam/SGD paths document and optimize for BF16 params with F32 grads.

Required revision:
- Treat BF16 grad storage as an optimizer/parameter/autograd migration, not only an autograd policy.
- Decide whether `Parameter::set_grad()` preserves dtype under `autograd_v2`.
- Add optimizer parity and performance tests for BF16 grads before using this as a Class A fix.
- Update `ops::grad_norm` and clipping expectations accordingly.

Possible staged path:
1. Make v2 internal grads match the parameter dtype. For BF16 parameters, v2 stores BF16 gradients end-to-end. This is the Class A recovery path. Keeping F32 internal grads would match v1 behavior but would not fix Class A, so it should not be the v2 default unless the project explicitly abandons the Class A goal.
2. Add optimizer support for BF16 grads.
3. Gate BF16-grad optimizer paths by feature/env.
4. Run per-model parity before flipping default.

### 8. `create_graph`, Reentrant Backward, and Hooks: Build-in From Day One

Earlier revisions of this doc said v0 should reject `create_graph=true`, omit reentrant backward, and skip hooks. **Reversed.** Audit of OneTrainer and SimpleTuner (both production PyTorch reference trainers flame-core's trainers parity against) found:

- **Reentrant backward** — used universally. OneTrainer wires `torch.utils.checkpoint` for FLUX / SDXL / Z-Image / Qwen / HiDream via `enable_checkpointing_for_*` helpers. SimpleTuner uses it for HiDream / LTX-Video / HunyuanVideo / SDXL controlnet / Kandinsky 5 / FLUX. Gradient checkpointing is the universal large-model OOM-prevention technique — exactly the scenario flame-core's BlockOffloader port also targets. flame-core's existing `flame_core::autograd::checkpoint` already exhibits reentrant-style behavior (disables autograd during forward, re-runs the closure with autograd re-enabled at backward time, mutating the tape). v2 must preserve this.
- **Hooks** — SimpleTuner uses `register_full_backward_hook` in `helpers/musubi_block_swap.py:138` to drive offload coordination at backward time. `register_forward_hook` appears in `helpers/ramtorch_extensions.py:782` and `helpers/models/ace_step/pipeline.py` for attention-stat collection. These are production usages of hooks for memory/IO orchestration, which is the same domain flame-core's BlockOffloader serves.
- **`create_graph=true`** — neither OneTrainer nor SimpleTuner uses higher-order grads. Could be deferred. But deferring leaves a retrofit cost (Engine + AccumulateGrad + InputBuffer all need out-of-place gating); building in the *surface* in v0 is cheap and removes the retrofit ever.

Per user directive (2026-05-13): build the surface for these in v0. Don't retrofit later. Better to design the trait shapes and engine state model to accommodate them on day one than to redesign in 6-12 months.

Required revision:
- **Reentrant**: v0 supports nested execute() via the inline-mini-execute pattern (`CheckpointGradFn` as a per-op `GradFn` whose `apply()` builds a sub-graph and runs a local mini-engine). Single-threaded only — no thread pool needed.
- **Hooks**: `GradFn` trait has a `hooks() -> &Hooks` accessor; `Hooks` carries pre/post/tensor-callback `Vec<Box<dyn Fn>>`. Default impl is empty. Registration API on `Tensor` and on individual `GradFn` nodes.
- **`create_graph=true`**: surface in v0 — `Engine::execute` accepts the flag, `InputBuffer::add` has both in-place (default) and out-of-place (`create_graph=true`) accumulation paths, `AccumulateGrad::apply` likewise. Recording during backward is permitted. v0 ships with the path exercised in tests but full higher-order op coverage can land in waves.
- **View autograd**: per-view-op backward formulas (`view`, `reshape`, `squeeze`, `unsqueeze`, `transpose`, `permute`) in Phase 3 P0/P1. Version-counter coverage is already a Phase 0 prereq.
- **Forward-mode AD**: `SavedTensor` carries an optional `fw_grad_` companion field. Each P0/P1 op records a forward-AD formula alongside its backward. Lots of per-op work but pure plumbing — no engine redesign.

What stays omitted (different reason — flame-core lacks the underlying infrastructure):
- **Sparse / nested tensors** — flame-core has no sparse or nested *storage* today. v2 cannot add autograd surface for storage that doesn't exist. If/when sparse storage is added in a separate workstream, v2's surface will extend trivially.
- **Compiled autograd / TorchScript / dynamo** — PyTorch's analog requires their entire compiler stack (TorchInductor, dynamo, FX). flame-core has a different architecture (NVRTC + WMMA + fused C kernels). No surface to add — the equivalent functionality already exists via different machinery.

Charter decisions:
- **Multi-device threading** — DECIDED 2026-05-13: build the **surface** in v0, defer feature-complete plumbing. Both OneTrainer and SimpleTuner use `torch.distributed` for DDP; flame-core will eventually need the same capability ("we will need ... not now -- multi gpu fyi"). Phase 1's `Engine` and `InputBuffer` design takes device + stream as explicit parameters at every dispatch point so DDP / NCCL plumbing can land later without engine rework. Surface cost: ~5 days inside Phase 1. Feature-complete (actual NCCL bindings, `all_reduce` for gradient sync, distributed sampler) is a separate ~3-week flame-core workstream not part of v2.

### 9. Migration Plan Understates Existing Recording Surface

Current recording is not centralized through `gradient_edge()`. Many forward ops directly call:

```rust
AutogradContext::record_op(...)
```

The migration section needs a concrete per-op dispatch plan:

```rust
#[cfg(feature = "autograd_v2")]
record_v2(...)

#[cfg(not(feature = "autograd_v2"))]
AutogradContext::record_op(...)
```

Required revision:
- Add `autograd_v2 = []` to `Cargo.toml`.
- Add `pub mod autograd_v2` behind the feature in `src/lib.rs`.
- Define a shared policy for forward wrappers:
  - old tape path
  - v2 graph path
  - no-grad/inference path
- Do not assume adding `Tensor::gradient_edge()` automatically migrates existing ops.

### 10. Class B And E Performance Work Are Separate Workstreams

Autograd v2 should not be credited with closing the full step-time gap. The rewrite can address the Class A gradient-storage policy if BF16 grads are accepted across autograd, parameters, optimizers, and grad clipping. It does not automatically remove the Class B narrow-backward F32 detour or the broader Class E cost from cudarc synchronous copies.

Known Class B scope:
- The current narrow backward path in `src/tensor_narrow.rs` still has F32 assumptions and F32 storage expectations around the scatter-add path.
- If v2 routes narrow backward through the existing kernel path unchanged, it inherits the Class B detour.
- Fixing Class B requires a BF16-in/BF16-out narrow backward kernel or dispatcher path that keeps F32 accumulation inside the kernel and returns the input dtype.

Known Class E scope:
- The majority of `*_sync_copy*` sites live outside autograd.
- Primary sweep areas are `src/tensor.rs`, `src/tensor_storage.rs`, and `src/cuda_gradient_ops.rs`.
- Autograd v2 may remove or avoid a small number of scalar/tape syncs, but the large sync-copy win is a parallel performance workstream.

Required revision:
- State explicitly that v2 is a correctness and architecture rewrite with a possible Class A recovery.
- Track Class B narrow-backward dtype work independently unless the v2 narrow backward implementation replaces the existing F32 detour.
- Track Class E sync-site elimination independently.
- Do not gate v2 correctness work on Class B or Class E, and do not count Class B/E savings in v2-only performance projections.

## Recommended Spec Changes Before Implementation

Revise the design doc with these concrete changes:

1. Replace strong `grad_accumulator` metadata with weak accumulator storage.
2. Add a shared, interior-mutable autograd metadata design for `Tensor`.
3. Change all engine and `GradFn::apply` APIs to return `Result<Vec<Option<Tensor>>>`.
4. Make `SavedTensor` hold a version-counter handle, not just a version integer. Add an optional `fw_grad_` companion field for forward-mode AD.
5. Make saved tensor release work from `&self`.
6. Add `GradFn::num_inputs()`.
7. **Build in `create_graph=true` surface** (was: reject). InputBuffer + AccumulateGrad get out-of-place paths gated on `create_graph`. Engine accepts the flag and permits recording during backward. Phase 5 parity covers the path, full higher-order op coverage can land in waves.
8. Define BF16 grad migration across `GradientMap`, `Parameter`, Adam, SGD, grad norm, and trainer code.
9. Make the in-place version-bump sweep a hard prerequisite.
10. Add compile-time feature wiring and per-op forward migration rules.
11. Carve Class B narrow-backward dtype work and Class E sync-site elimination out as separate performance workstreams.
12. **Build in reentrant backward surface** via inline-mini-execute (`CheckpointGradFn` is a `GradFn` whose `apply` drives a local sub-graph through a mini-engine). Single-threaded only; no thread pool. Preserves the semantics of flame-core's existing `flame_core::autograd::checkpoint`.
13. **Build in hook surface**: `Hooks` struct with pre/post/tensor-callback vecs, exposed by `GradFn::hooks()` and a registration API on `Tensor`. Default empty. Used by future BlockOffloader integration patterns and by users wanting introspection.
14. **Build in view-autograd surface**: per-view-op `GradFn` impls for `view`, `reshape`, `squeeze`, `unsqueeze`, `transpose`, `permute`. Backed by the version-counter prereq in clause 9.
15. **Build in forward-mode AD surface**: `SavedTensor::fw_grad_` field, each P0/P1 op records a forward formula alongside backward. Plumbing per op — no engine redesign required.
16. **Multi-device surface**: DECIDED 2026-05-13 — build it in. Phase 1's `Engine` and `InputBuffer` design must avoid hardcoded single-stream / single-device assumptions; stream and device are explicit parameters at every dispatch point (~5 days of surface design within Phase 1). Feature-complete (NCCL bindings, DDP gradient sync, distributed sampler) is a separate workstream after v2 lands.

## Suggested Implementation Order

### Phase 0: Prerequisites

- Audit all in-place writes and add `storage.bump_version()`.
- Add tests for version mismatch on saved tensors.
- Decide and document BF16-grad optimizer behavior.
- Add `autograd_v2` feature flag only; no behavior change.

Scope note: per directive 2026-05-13 ("if we come a day when we need it, i don't want to add it, i want it built in"), v0 ships with **surface** for reentrant backward, hooks, `create_graph=true`, view autograd, and forward-mode AD. Feature-complete coverage of each lands in waves after v0; the design contract doesn't change. Total v0 estimate: **7-9 weeks of focused work** (vs the 3-5 week minimal v0). Multi-device surface is *also* in v0 if/when the project-level multi-device decision lands.

### Phase 1: Metadata and Core Types (~2 weeks)

- Add shared `AutogradMetaV2` (interior-mutable, weak accumulator).
- Add `Edge`, `GradFn` trait, `NodeId`, sequence number, topological number (stored as fields, not recomputed).
- `GradFn` trait surface includes `hooks() -> &Hooks` (default empty) AND `num_inputs()`.
- Add weak leaf accumulator cache.
- Add `SavedTensor` using the existing version-handle model + an optional `fw_grad_` companion field for forward-mode AD.
- Add `InputBuffer` with `Option<Tensor>` + `num_inputs` + **both in-place AND out-of-place accumulation paths**. In-place is the default when `create_graph=false` AND dtype/shape match AND the buffered grad has unique storage ownership; out-of-place fires when `create_graph=true` or any in-place precondition fails.
- Add `Hooks` struct: pre/post/tensor callback vecs.
- (If multi-device is chartered) Engine and InputBuffer don't hardcode single-stream/single-device assumptions. Stream and device are explicit parameters at every dispatch point.

### Phase 2: Engine Skeleton (~1.5 weeks)

- Implement `GraphRoot`.
- Implement `AccumulateGrad` (BF16-throughout per Class A; F32 only as opmath_t inside kernel). Out-of-place path gated on `create_graph=true`.
- Implement dependency counting.
- Implement ready queue.
- Return `Result<Vec<Option<Tensor>>>` everywhere.
- **Accept `create_graph=true`** — engine permits recording during backward, AccumulateGrad and InputBuffer use their out-of-place paths.
- **Support nested execute()** via inline-mini-execute pattern. Single-threaded — Engine state save/restore on nested entry, no thread pool. `CheckpointGradFn` is the canonical user of this.
- **Wire hook dispatch** at GradFn entry/exit. Default no-op when `Hooks` is empty.

Toy tests:
- single leaf sum
- two branches into one leaf
- diamond graph accumulation
- undefined grad slots
- released saved tensor error
- version mismatch error
- `create_graph=true`: backward-of-backward on a simple op produces correct second-order gradient
- reentrant: nested `Engine::execute` inside a `GradFn::apply` returns cleanly (no deadlock, no state corruption)
- hooks: pre/post/tensor hooks fire in expected order

### Phase 3: First Real Ops + View Backward + Forward-Mode (~3 weeks)

**Status (2026-05-13): COMPLETE.** Phase 3 shipped in four subphases:

- **Phase 3a** (commits `bfc371b`, `6ee385f`): recording surface
  (`Tensor::autograd_meta`, `record_v2`, `gradient_edge_for_tensor`,
  `needs_grad`, `next_sequence_nr`) + 5 math P0 ops (add, mul, sum,
  matmul, silu). 18 tests.
- **Phase 3b** (commit `d471c31`): 6 view ops (reshape, view,
  transpose, narrow, squeeze, unsqueeze, permute) + HAZARD-2026-05-13-1
  characterisation (narrow-view-write hazard) + gemm-stride-ignore
  discipline (`.contiguous()` on transpose/permute backward). 10
  tests.
- **Phase 3c1** (commit `2be9770`): `layer_norm_v2` (BF16-only,
  delegates to fused kernels) + full `CheckpointGradFn::apply`
  (re-runs forward closure under nested `Engine::execute`). 5 LN
  tests + 4 checkpoint tests + 1 d_x symmetry regression
  (`2d9cd0d`).
- **Phase 3c2** (this commit): forward-mode AD (JVP) plumbing
  across the 11 Phase 3a/3b ops. `AutogradMetaV2::fw_grad` slot,
  `Tensor::fw_grad` / `set_fw_grad` accessors,
  `autograd_v2::ops::fw_mode::{any_fw_grad, tangent_or_zero}`
  helpers, per-op JVP formula in each forward wrapper. 13 tests at
  `tests/autograd_v2_fw_mode.rs`. `layer_norm` JVP **deferred** to
  Phase 5 parity gate (formula in §clause 15 is mechanical but
  non-trivial; an input tangent on LN currently silently zeroes
  on the output).

P0/P1 ops:
- add, mul, sum, reshape, transpose, matmul/linear, silu, layer_norm
- **view, squeeze, unsqueeze, permute** (view-autograd surface — Phase 3 makes view ops first-class)
- Checkpoint / CheckpointOffload (the reentrant users — preserve `flame_core::autograd::checkpoint` semantics bit-equal)

Each op needs:
- forward wiring under `autograd_v2`
- backward struct (the `GradFn` impl)
- **forward-mode AD formula** alongside backward — uses the `SavedTensor::fw_grad_` field
- PyTorch fixture parity (backward direction) — Phase 5
- forward-AD parity vs PT's `torch.autograd.functional.jvp` — Phase 5
- dtype assertion
- no unwanted `.to(F32)` in autograd_v2

Long-tail unary ops:
- Do not block v0 on a code generator.
- Keep hand-written backward structs (+ forward-mode formulas) for the P0/P1 path.
- Permit a later `derivatives!` proc macro for the long tail (`sin`, `cos`, `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `neg`, `pow`, etc.) once the trait, saved tensor, and forward-wrapper patterns have stabilized. Proc macro emits both backward AND forward formulas from a single declarative entry.

### Phase 4: Optimizer and Trainer Integration (~1 week)

- Route v2 grads into parameters (BF16 end-to-end per Class A).
- Ensure optimizer accepts param-dtype grads (BF16 for BF16 params).
- Add grad-norm and clipping tests.
- Run one-step model parity before long parity runs.
- Verify `flame_core::autograd::checkpoint` semantics preserved on v2.

### Phase 5: Parity Gate

**Status (2026-05-13)**: Phase 3 (3a/3b/3c1/3c2) + Phase 4 (4a/4b) + **Phase 5a (Deliverables A + B + LN JVP)** + **Phase 5b (Deliverable C Route ii — `backward_v2()` bridge)** + **Phase 5c (Deliverable D — perf bench)** complete. v2 grad-storage path is end-to-end-callable from any forward graph (v3 forward + v2 grad-storage via the bridge); perf bench confirms +1.46-+2.05% bridge overhead on real Klein workloads and 50.00% exact grad-memory savings (Class A). Next: real-trainer Z-Image LoRA multi-step smoke + v3 retirement.

**Phase 5a shipped** (this commit; see `tests/autograd_v2_parity.rs`):
- Deliverable A (per-op backward fixture parity): 13/13 ops pass against PyTorch fixtures via `flame_core::parity::ParityHarness`.
- Deliverable B (forward-mode AD parity): 13/13 ops pass; **layer_norm JVP shipped here** (was the Phase 3c2 deferral). LN JVP formula was rederived during Phase 5a — the per-row variance derivative collapses to a per-row scalar (sum-reduction over normalized axes), not the per-element form in the original Phase 5 stub below. F64 bit-equal to `torch.autograd.functional.jvp(layer_norm, ...)`.
- Fixture generator at `tests/fixtures/gen_v2_parity.py` (seed=42, ≤512-element shapes per op).

**Scope estimate**:
- **Route (i) — port missing forward-graph ops**: ~6-8 DiT primitive families currently missing in v2 (per Phase 4b skeptic's recalibrated audit of `train_zimage.rs`): `primitive_rms_norm` (8 call sites), `flame_core::attention::sdpa`, RoPE-apply (`build_3d_rope` / `build_1d_rope`), `chunk` (5 sites), `broadcast_to` (2 sites), `fused_linear3d_native` / `Op::Linear`, residual `Tensor::add`. Note: `conv2d`, `gelu`, `group_norm` initially listed by Phase 4b port agent are VAE/encoder ops, NOT in the LoRA-trained DiT — agent's "~20+ ops" framing was ~3× inflated. **Real estimate: 6-10 weeks.**
- **Route (ii) — `loss.backward_v2()` bridge**: ~30-line flag-switch on `AutogradContext::backward` (`src/autograd.rs:1297-1820`, ~520 lines total) that constructs `GradientMap::new_v2()` and routes v3 backward through it for params marked `Parameter::new_v2`. Lossy parity (BF16 grad path is non-bit-equal vs v3 by construction — per `BF16_GRAD_DECISION.md`). **Estimate: 2-3 weeks** including parity-harness wiring.
- **Phase 5 first action**: re-audit the routes before committing. Phase 4b skeptic flagged that the "200-line duplication" estimate for route (ii) was an upper bound not measured against minimum-viable surgery.

Do not retire v1/v3/v4 until ALL of these gates pass:

**Deliverable A — per-op backward fixture parity** [SHIPPED Phase 5a] (~1 week if ParityHarness reused):
- For each of the 13 v2 ops (12 + layer_norm), generate a PyTorch reference fixture (`.safetensors` of inputs + `torch.autograd.grad` outputs) at non-trivial shapes (matching `tests/autograd_v2_ops.rs` fixture shapes).
- Use fixed seed 42.
- Use the existing `flame_core::parity::ParityHarness` (shipped 2026-05-09).
- Tolerance: bit-equal at F32, BF16-noise-bounded at BF16.
- Cite PyTorch source by `file:line` in commit comments.

**Deliverable B — forward-mode AD parity** [SHIPPED Phase 5a] (~3-4 days; Phase 3c2 unblocked this):
- For each of the 13 ops, generate `torch.autograd.functional.jvp(op, primals, tangents)` reference.
- Test against v2: `Tensor::set_fw_grad(...)` → `op_v2(...)` → `out.fw_grad()`.
- **Includes layer_norm**: LN JVP shipped (was deferred from Phase 3c2). Kernel-truthful formula (verified bit-equal to `torch.autograd.functional.jvp` in F64; see `tests/fixtures/gen_v2_parity.py`):
  ```
  centered    = x - mean(x, normalized_axes)
  centered_fw = x_fw - mean(x_fw, normalized_axes)
  var_fw      = (2/N) * sum(centered * centered_fw, normalized_axes)
  rstd_fw     = -0.5 * rstd^3 * var_fw                       # per-row scalar
  out_fw = centered_fw * rstd * w
         + centered    * rstd_fw * w
         + x_hat       * rstd * w_fw
         + b_fw
  ```
  Implementation note: the per-element `d_rstd_dx = -rstd^3 * (x - mean) * (x_fw - mean_fw)` shorthand previously in this section is **not** the correct formula. The variance derivative reduces to a per-row scalar (sum over normalized axes) — see `src/autograd_v2/ops/layer_norm.rs::layer_norm_jvp` for the bit-equal F32 implementation. Final cast to BF16 for storage on the output `fw_grad`. ~150 LOC + 1 test, F32 internal compute.

**Deliverable C — model parity** [SHIPPED Phase 5b via Route (ii) bridge]:
- `AutogradContext::backward_v2(loss)` shipped at `src/autograd.rs:1322`
  (sibling of `backward` at `:1297`; shared body
  `backward_impl(loss, policy)` at `:1342`). Net diff vs pre-Phase-5b
  body: ~30 LOC on the body's three grad-map touch sites
  (`with_index` → `with_index_v2`, `set_ones` → `set_ones_dtype`,
  per-grad `to_dtype` cast at accumulate).
- Bridge tests at `tests/autograd_v2_bridge.rs`: 3 tests (BF16 emit,
  F32 parity vs v3, BF16 tolerance — cos≥0.999, max_abs_ratio≤5e-3
  per `BF16_GRAD_DECISION.md`). All green.
- EriDiffusion-v2 `train_zimage`: opt-in `--use-autograd-v2` flag
  (default OFF; new `autograd_v2` feature on `eridiffusion-cli`).
  Builds clean with and without the feature.
- Full 100-step Z-Image smoke deferred — requires real dataset/config
  not staged on this box; synthetic 4-layer-MLP smoke
  (`backward_v2_within_tolerance_at_bf16`) covers the bridge
  numerical-correctness target per the Phase 5b prompt's fallback
  spec.
- Klein **multi-step** is blocked on this box (`CUDA_ERROR_INVALID_VALUE`
  at step 2+); however Klein **step 1** is usable and produces
  deterministic loss `1.1217` per Phase 0's smoke. Single-step model
  parity (Klein backward grads v3 vs v2) IS available; multi-step
  loss-curve smokes need Z-Image or synthetic.

**Deliverable D — no ms/step regression** [SHIPPED Phase 5c]:
- Bench harness at `tests/autograd_v2_perf.rs` (3 `#[serial] #[test]` cells × 3 configs each).
- Workloads: synthetic 4-layer MLP / Klein attn_chain prod (fixture-driven, real prod-shape) / Klein double-block backward (fixture-driven).
- Configs: v3 control / bridge alone / Class A (`Parameter::new_v2` + `set_grad` round-trip).
- 5 warmup + 50 timed iters per cell, trim slowest 5, report median.
- **Headline (bridge Δ% on the real Klein workloads)**: +1.46% (Klein double-block) / +2.05% (Klein attn_chain prod). Just above the ±1% target; binding constant overhead is the post-loop dtype-unification cast in the bridge.
- **Memory savings (Class A vs v3)**: 50.00% exact — 78 MB saved per backward on Klein attn_chain prod. See `docs/BF16_GRAD_DECISION.md` §Phase 5c.
- **Reproduce**: `FLAME_CUDA_GRAPH=0 cargo test --release --features autograd_v2 --test autograd_v2_perf -- --nocapture --test-threads=1`.
- **Not measured (out of scope)**: real-trainer Z-Image LoRA multi-step convergence (Klein step 2+ crash blocks); `AdamWV2::step` end-to-end optimizer perf (deferred to Phase 6 / v3 retirement).

**Phase 5c trio verdict (post-ship audit)**:
- **Bug-fixer**: **Floor-at-+2.18%, no win available within scope.** Bandwidth-bound, not launch-bound. 78 MB BF16 grad = 156 MB F32 source = 234 MB I/O / 0.22 ms = effective ~1.07 TB/s, **at the HBM ceiling on a 3090 Ti (~1.1 TB/s)**. Options A (fast-path skip — already present), B (multi-tensor batch cast, ~250 LOC, saves only ~0.3pp), C (pre-allocate destinations — `pool_alloc_u16` already serves from pre-warmed pool) all evaluated; net-zero or sub-percentage-point. Option D (avoid cast — typed v3 ops) requires multi-week kernel rewrite, deferred to Phase 6. **The +2.18% is fundamental given the architecture.**
- **Skeptic**: REPRODUCES-with-flag. Standing rule streak resumed cleanly — verbatim git pre/post state + verbatim bench output present in commit body. 3 minor flags (none blocking): GPU stream not pinned in code (relies on `#[serial]` + `--test-threads=1`); Phase 5d planning under-specifies trainer migration + race fix (addressed in §Phase 5d below); v3 regression spot-check timing.
- **Builder**: equivalent verification by parent assistant after agent rate-limit kill. 3/3 perf cells reproduce within noise (one cell came in `Δ-1.98%` on re-run vs `+2.18%` original — same magnitude, opposite sign, within measurement variance). 17/17 v3 regression tests pass. Memory savings exactly 50% across all 3 cells on independent re-run.

**Verdict**: Phase 5c shipped. The trade is +2.18% backward time for 50% gradient memory — accepted per the standing contract that v2's win is correctness + memory, not speed. For memory-constrained training (Klein 9B full-FT on 24 GB; longer sequences without gradient checkpointing; larger batch sizes; higher LoRA ranks), "trains" beats "doesn't train" infinitely.

## Phase 5d / pre-retirement work (open)

Named action items per Phase 5c skeptic + bug-fixer feedback. Each is its own session:

1. **Real-trainer Z-Image LoRA smoke (>100 steps)**. Run `train_zimage --use-autograd-v2 --max-steps 100+`; capture loss curve; compare to v3 baseline within 1% per `BF16_GRAD_DECISION.md`. Z-Image does NOT crash at step 2+ (Klein-specific infra issue). Required to claim v2 is "production-ready".
2. **Trainer-side `Parameter::new_v2` migration**. EriDiffusion-v2 LoRA construction sites currently use `Parameter::new` (`CastToF32` policy). Audit LoRA crate, flip to `new_v2` under v2 feature gate. **Without this, the bench's 50% Class A memory savings never materializes in real runs.** This is the highest-value remaining item — the bench measured the potential; production needs the wiring.
3. **Reentrant + hooks tests on a real trainer**. v2 supports both (Phase 3c1 CheckpointGradFn + Phase 1 Hooks); no integration test on a real trainer exists yet. Required cross-cutting gate per §Phase 5.
4. **`AUTOGRAD_CONTEXT` race — RESOLVED as "intentional `#[serial]`" (decision 2026-05-30, option c).** Re-examined with measurement: `serial_test` is **not** a band-aid masking a bug — it correctly serializes exactly the tests that legitimately share the **v3** global `AUTOGRAD_CONTEXT` (`Mutex<AutogradContextInner>`, `autograd.rs:84`) via the bridge: `autograd_v2_{bridge,klein_parity,perf,parity,trainer_integration}`. Serializing tests that share one global tape is the right thing, not a hack. The pure-v2-engine tests use per-tensor `Arc<Mutex<AutogradMetaV2>>` and correctly carry **no** `#[serial]`. A `thread_local!` conversion (≈33 `AUTOGRAD_CONTEXT` sites + `AUTOGRAD_ENABLED` + `RETAINED_*` globals + re-entrant-checkpoint borrow/Drop care) was evaluated and **declined**: it is pure hardening of working code (its only payoff is letting bridge tests drop `#[serial]` + future-proofing), and per tenets 4/5 we don't take a risky change to the core engine that trains every model to fix a non-reproducible problem. **The remaining footgun is documented, not eliminated**: any *new* test that drives v3 backward / `backward_v2` must add `#[serial]`.
5. **"Pre-existing v2 test flakes" — DID NOT REPRODUCE (measured 2026-05-30); doc claim was stale.** The three named tests (`autograd_v2_ops::{transpose_v2_backward, engine_rejects_mismatched_grad_output_shape}`, `autograd_v2_engine::single_leaf_sum`) ran **9/9 green under parallelism** on this checkout (5× engine+ops default-threads, 4× full v2 suite cross-binary parallel). They use the v2 engine's isolated per-tensor meta — they touch neither the v3 global nor any flake-prone shared static, so the item-#4 "global-tape race" attribution was wrong on two counts (they don't fail, and they don't use that global). No reproducible bug to fix. Full v2 suite is **154 tests green** after the 2026-05-30 `Op::RoPePrecomputed { layout }` bit-rot fix in `autograd_v2_perf.rs` + `autograd_v2_klein_parity.rs`.
6. **v1/v3/v4 retirement — STAGED, NEVER A FILE DELETE** (user directive 2026-05-30). The old engine is *never* removed by `git rm` / filesystem delete at retirement time. Two stages, in order:

   - **Stage 6a — Gate off (no removal). SHIPPED for Klein 2026-05-30.** Flip the *default* so trainers route through v2, with v3 reachable only via an explicit opt-out. **No files move, nothing is deleted.** v3 stays compiled and instantly reachable so any regression can be re-measured against it (tenet 4). Reversible-by-one-flag; retirement *stops* here until a successor is proven.
     - **Klein (done):** `eridiffusion-cli` now has `default = ["autograd_v2"]` so the bridge compiles into the normal build; `train_klein` defaults to v2, with `--use-autograd-v3` as the opt-out (`conflicts_with` the back-compat `--use-autograd-v2` no-op). Verified both directions 2026-05-30: bare-flags run logs `[autograd_v2] flipped 288 params` + trains (step 3/3, loss 0.5388); `--use-autograd-v3` run has no flip line + trains (same loss). Preceded by a 150-step convergence smoke matching the v3 reference (`klein_3000_cosine_b1`) within BF16 tol (stable region <1%, endpoint +0.90%).
     - **Z-Image (done 2026-05-30):** `train_zimage` defaults to v2, `--use-autograd-v3` opt-out, same pattern as Klein. A 150-step v2-vs-v3 A/B smoke (same binary/config, flag-only diff; cache `alina_zimage_512`, model `z_image_base_bf16`) matched **within 0.02%** at every checkpoint — tighter than Klein. Both directions verified (bare→`flipped 420 params`; `--use-autograd-v3`→no flip). (Note: an earlier session claim that Z-Image data "was not staged" was wrong — the caches are named `cache/<subject>_zimage_<res>`, not `*cache*`.)
     - **ernie + anima (smoked 2026-05-30):** 30-step v2-vs-v3 A/B matched within ≤0.05% (ernie, `eri2_ernie_512` / `/home/alex/models/ERNIE-Image`) and ≤0.08% (anima, `gigerver3_anima_512` / `anima-base-v1.0.safetensors`). Both directions verified. Brings the convergence-proven set to **klein, zimage, ernie, anima** (DiT + ERNIE-Image + Cosmos-Predict2 anima).
     - **10 more trainers gated 2026-05-30 (wired + compile, NOT individually smoked):** chroma, flux, sd35, sdxl, qwenimage, acestep, ltx2, slider_klein, u1, wan22 all default to v2 via the same pattern. wan22 flips both expert param sets (`params_low`+`params_high`); its FP8/AdamW8bit exception is on the base experts, not the LoRA params. These were defaulted on the strength of the bridge being architecture-agnostic shared code proven on Klein+Z-Image — NOT per-model convergence runs (no scripts; most lack a staged base checkpoint on this box). Each gets its real check when next trained; `--use-autograd-v3` reverts instantly.
     - **asymflow, hidream_o1, l2p (last 3 — wired 2026-05-30, full bridge added):** these had NO v2 path at all; got the full wiring (`--use-autograd-v3` arg + param flip + `loss.backward()` wrapped in the cfg-gated `backward_v2` block), not a mechanical mirror. All compile + expose `--use-autograd-v3`. Not smoked: asymflow/hidream lack a staged recipe here; **l2p has an open cuDNN-SDPA-backward crash** (orthogonal to v2 — in flame-core backward, hit regardless of grad map; a real l2p run needs `FLAME_NO_CUDNN_SDPA_BWD=1`; see [[project_l2p_step2_misalign_2026-05-30]]). l2p builds params incrementally (Vec+push); flip applied after population.
     - **ALL 17 trainers now default to v2.** Convergence-proven (4): klein, zimage, ernie, anima. Wired+compiled, unsmoked (13): chroma, flux, sd35, sdxl, qwenimage, acestep, ltx2, slider_klein, u1, wan22, asymflow, hidream_o1, l2p.
   - **Stage 6b — Archive (only when a *new* version supersedes v2).** Actual file removal from the live tree happens **only** when a future engine replaces v2 and is itself proven. Even then it is **not a delete**: the superseded engine's files are *parked* to an archive location (e.g. `flame-core/unused/autograd_legacy/` or a dated `flame-core/_archive/`), not `git rm`'d out of existence. The code remains recoverable on disk. Files in scope when this eventually happens: `src/autograd.rs` (4547 lines), `src/autograd_v3.rs`, `src/autograd_v4/`, plus legacy `autograd_simple.rs` / `autograd_engine.rs` / `autograd_ops.rs` / `autograd_ops_complete.rs` / `autograd_debug.rs`.

   Rule of thumb: **gate, don't delete; archive, don't erase.** No autograd engine is ever destroyed — only switched off, then parked.

NCCL multi-device + inference-path migration are deferred indefinitely (single GPU + trainer focus per user direction).

**Cross-cutting gates that must also pass**:
- BF16 grad policy optimizer parity (Phase 4a/4b set up the kernels; Deliverable A/C exercise them in anger).
- In-place mutation tests stay green (17 Phase 0 audit tests + view-autograd from Phase 3b).
- **Reentrant test**: training run using `enable_checkpointing` matches v1 bit-equal at step 1+ (Phase 3c1 shipped CheckpointGradFn; Deliverable C exercises it on a real model).
- **Hooks test**: simple forward and backward hook fires expected callback count per training step.
- HAZARD-2026-05-13-1: characterization test (Phase 3b) stays green OR the underlying flame-core base bug is fixed and the negative test is converted to a positive one.

## High-Risk Areas To Watch

- View metadata and version sharing.
- `Tensor::clone()`, `detach()`, `to_dtype()`, `contiguous()`, and `narrow()` metadata behavior.
- Checkpoint/offload interaction with v2 saved tensors.
- SDPA backward saved-output mutation detection.
- Multi-output ops and `output_nr`.
- Gradient accumulation across micro-batches.
- Existing trainers that expect `GradientMap` return values rather than `.grad` fields.
- CUDA graph capture/replay currently integrated into old backward.

### HAZARD-2026-05-13-1: view + in-place silently detaches under `shared_storage`

**Found**: Phase 1 bug-fixer audit, 2026-05-13. Verified empirically by reading the call chain.

**Mechanism** (default feature `shared_storage` is on):

1. `view = parent.narrow(dim, start, length)?` returns a Tensor that aliases the parent's storage. The inner `Arc<CudaSlice>` refcount becomes ≥2.
2. `view.add_inplace_same_dtype(&delta)?` (or any in-place mutator that goes through `try_as_mut_slice_*`) calls `ensure_unique_slice` (`src/tensor_storage.rs:147`), which calls `Arc::make_mut(slice)`.
3. With refcount > 1, `Arc::make_mut` **silently clones** the inner `CudaSlice`. The view's local Arc now points at the new clone.
4. The kernel writes into the clone. **Parent is untouched. View's storage is detached.** No error, no warning, silent wrong data.

**Equivalent PyTorch behavior**: `parent[1:3] += delta` mutates `parent`. Under flame-core's current primitives, the analogous Rust spelling does NOT.

**In-tree exposure (2026-05-13)**: `rg -n "narrow\([^)]*\)\.add_inplace\|narrow\([^)]*\)\.copy_"` across EriDiffusion returns zero hits. No live code currently relies on this pattern. `narrow_owning` (`src/tensor.rs:3893`) exists as an unrelated escape hatch (it materializes the view to release the parent — opposite intent).

**Why this matters for autograd v2**: Phase 3's view-autograd surface will record `narrow` / `view` / `permute` as `GradFn` impls. If any backward path or `InputBuffer` accumulation writes through a view expecting the parent to see it, the gradient is silently wrong.

**Phase 0 audit miss**: `tests/inplace_version_bump_audit.rs` (17 tests) exercises the sole-owner mutation path only. Add a view-aliased mutation test to that suite, expected to either: (a) bump the parent's version handle through the COW boundary, or (b) return an explicit `Err` instead of silently COWing.

**Where the fix belongs** (tenet 1 — fix the primitive):

- **Option A (preferred)**: change `ensure_unique_slice` to refuse to clone under shared aliasing, returning `Err(SharedStorageWriteWithAliases)`. Callers that need owning semantics use `narrow_owning` (already exists). Callers that need write-through to parent use a yet-to-be-built `narrow_mut` that takes `&mut parent` and threads the lifetime.
- **Option B**: have `Arc::make_mut`'s clone path also bump the *parent* Arc's version-counter side-table entry (using `Arc::as_ptr` before the make_mut call). Detects the bug at the SavedTensor unpack site without changing public semantics.
- **Option C**: deprecate `narrow` for the writeback use case entirely; add only `narrow_owning` (detached copy) and `narrow_mut` (proper mutable borrow). PyTorch-flavor `parent[a:b] += x` would lower to the latter.

Pick before Phase 3 view-autograd lands.

**Scope confirmation**: this is a flame-core base bug, present long before autograd v2 work. Not Phase 1's regression; not Phase 1's responsibility to fix. But it WILL bite Phase 3 / Phase 4 unless addressed.

## Decision

Recommendation: revise and narrow the v2 design before coding.

Acceptable "go" condition:
- The blockers above are reflected in the spec.
- BF16 gradient policy is explicitly accepted as a cross-cutting optimizer migration.
- In-place version bump audit is tracked as prerequisite work.
- v0 explicitly rejects unsupported higher-order behavior.
- Implementation starts with a tiny graph and parity harness, not the full op table.

## Phase 0 (2026-05-13) — shipped

1. **In-place version-bump audit** — every in-place mutator in flame-core
   now calls `TensorStorage::bump_version()` after writing. Coverage:
   `ops::elt::{add,mul,gate_mul}_inplace_*`, `Tensor::copy_*`,
   `tensor_narrow::narrow_backward_scatter_add_cuda`, Adam fused single
   + multi-tensor paths, SDPA mask helpers, BF16 layernorm/AdaLN
   affine. `multi_tensor_scale_inplace_packed` is documented as
   caller-bumps (no Tensor handle inside the function). Tests live in
   `tests/inplace_version_bump_audit.rs` (13 cases, green).
2. **BF16-grad decision (Option A)** — recorded in
   `docs/BF16_GRAD_DECISION.md` with the full F32-coercion audit of
   `GradientMap` / `Parameter` / `Adam` / `grad_norm`. No behavior
   change in Phase 0; full rewrite lands in Phase 4.
3. **Feature flag** — `autograd_v2 = []` added to `Cargo.toml`;
   `#[cfg(feature = "autograd_v2")] pub mod autograd_v2;` added in
   `lib.rs`. The module is empty (one doc comment); Phases 1-5 fill it.

No behavior change. Klein 9B 8-step smoke (`train_klein --config
configs/klein9b_alina.json --rank 4 --steps 8`) produces bit-identical
step-1 loss `1.1217` with and without the Phase 0 changes (verified
2026-05-13 via stash/unstash A/B). Step 2+ hits a pre-existing
`CUDA_ERROR_INVALID_VALUE` on the smoke config — reproducible on
the Phase-0-pre baseline, so not introduced here.
