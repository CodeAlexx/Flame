# flame-core tenets

The non-negotiable principles for working in this codebase. Everything in
flame-core — every primitive, every kernel, every API surface — answers to
these. If a change violates a tenet, the change is wrong, even if it works.

The audit gates in `docs/SPEED_CONTRACT.md` derive from these. The contract
tells you *what to check*. This document tells you *why*.

---

## 1. Fix the primitive, ship every model

**flame-core is one framework. Per-call inefficiency in any primitive
multiplies across every model that uses it. Fix the primitive once → every
model is fast.**

OneTrainer is fast *before* a new model is added. New models inherit the
framework's speed because the primitives are right, not because each model
got individually tuned. flame-core's goal is the same.

The corollary: **never optimize at the trainer or model level when the cost
lives in a primitive.** Every model-specific workaround is interest paid on
the wrong account. The primitive stays slow, the next model imports it slow,
and you accumulate a graveyard of fast-paths that don't compose.

### Examples of where the fix belongs

| Symptom seen at | Fix belongs in |
|---|---|
| Klein training loss reads stalling step | flame-core (loss-read primitive, sync contract) |
| Z-Image attention forward slow | flame-core SDPA primitive (not a Z-Image fast path) |
| Backward `d_o` BF16/F32 cast issue | flame-core autograd SDPA backward dispatch arm |
| Bias gradient reductions slow on every model | flame-core `cuda_ops.rs` `Tensor::sum` BF16 path |
| Narrow backward stalling | flame-core narrow primitive (commit `b552f61`) |
| Memory churn in attention fallback | flame-core SDPA streaming primitive |

A model file (`klein.rs`, `zimage.rs`, `chroma.rs`, etc.) should contain
*model architecture* — what layers, what shapes, what connectivity. It
should not contain workarounds for primitive inefficiencies. If a workaround
appears there, it's a bug to be migrated into flame-core.

A trainer (`train_klein.rs`, etc.) should contain *training loop policy* —
schedule, optimizer config, dataset wiring, logging cadence. Not primitive
fast paths.

---

## 2. Primitives' APIs make the right thing easy and the wrong thing hard

A primitive's API is part of its correctness budget. If a caller can
*accidentally* take a slow path because the slow path is the most natural
spelling, that's the primitive's bug, not the caller's.

### Examples

- `Tensor::add_bf16_iter` already does the right thing automatically.
  Callers reach for it and inherit BF16 throughput. ✅
- `Tensor::sum` currently casts BF16 → F32 → reduce → F32 → BF16 by default.
  There's no `sum_bf16_direct` for callers to reach for, so they accept the
  F32 round-trip without knowing they're paying for it. ⚠️
- `narrow_backward_scatter_add_cuda` used to require BF16 callers to cast
  to F32 and back manually (the detour at `tensor_narrow.rs:169-183`). The
  primitive forced the bad spelling. (Class B handoff fixes this.) ⚠️

The rule: **if there's only one way to call a primitive, that way must be
fast at the storage dtype of the inputs.** Don't accept "fast variants" the
caller has to know about — make the main entry point fast.

---

## 3. The dispatcher is where per-primitive policy lives

Stream assignment, fusion detection, allocation reuse, graph capture,
workspace caching, error handling — these are dispatcher concerns. Without
a dispatcher, every primitive makes those decisions independently, and they
drift. With one, you fix the policy once and every primitive inherits it.

flame-core has these concerns scattered today (autograd v1/v3/v4 stack,
multiple SDPA paths, multiple conv2d implementations). Consolidating into
clean dispatcher layers is the work that lets the speed contract stay
enforceable without per-PR vigilance.

`docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` is the autograd dispatcher
plan. The kernel/SDPA dispatcher is open work.

---

## 4. Measurement beats assertion

A claim about performance is not real until it's measured. A "this should
be faster" comment is a hypothesis. An nsys profile showing the actual
delta is evidence.

This is what bit us with the `autograd.rs:1493` "sync source" claim that
turned out to be dead code in a disabled cfg branch. Pre-condition for
landing any Class A/B/C/E fix: a measurement attributable to the named
site, then a re-measurement after the fix showing the named site is gone.

`tests/narrow_sync_microbench.rs` is the model — a regression gate that
fails if the primitive's sync contract is violated. Every primitive fix
should leave behind one of these.

---

## 5. Reject fixes that live in the wrong place

When reviewing a PR (or an agent's prompt) that touches a trainer or model
file: ask whether the same fix could live in flame-core. If yes, the PR is
in the wrong place.

This is the auto-reject rule. The principle is more important than the
local feature win. A model-level fix that ships in a trainer is a regression
against tenet 1 — even if the model gets faster, the framework didn't get
faster, and the next model imports the same slow primitive.

---

## How tenets relate to the speed contract

| Tenet | Speed contract clause |
|---|---|
| 1. Fix the primitive | Cuts across all 5 clauses |
| 2. API makes right easy | Especially clause 2 (dtype) — no F32-cast traps |
| 3. Dispatcher policy | Clause 3 (autograd), clause 4 (kernel), clause 5 (memory/IO) |
| 4. Measurement beats assertion | Every clause's "fail mode at PR review" |
| 5. Reject wrong-place fixes | Audit rule on every PR |

The tenets are *why*. The contract is *what to check*. Read both before
proposing a fix that touches a primitive.
