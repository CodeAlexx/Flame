# Phase 5d item #6 — v1/v3/v4 retirement audit + plan

**Date**: 2026-05-13
**Status**: AUDIT ONLY. No deletion performed this session.

## Why audit-first instead of execute

The original handoff `HANDOFF_2026-05-13_PHASE5D_ZLORA_SMOKE.md` described item #6 as "delete `src/autograd.rs` (4547 lines) + a few alias/dead files, single commit, push to a branch, user reviews". The audit below shows that's misleading by an order of magnitude — the deletion cannot be a single self-contained commit because:

1. The Phase 5b `backward_v2` bridge **lives inside `src/autograd.rs`**. Deleting the file removes the bridge that `--use-autograd-v2` trainers depend on.
2. **222 external call sites** in the wider workspace depend on `AutogradContext::*` symbols. These are not just trainers — they're every prepare/sample/train binary.
3. v2 today has no equivalents for `AutogradContext::no_grad`, `clear`, `is_recording`, `set_enabled`. Migrating consumers requires defining those v2-side first.

So item #6 is a **multi-stage migration**, not a delete. This doc maps the stages.

## Deletion targets (per handoff)

| File / dir | Lines | Role |
|---|---|---|
| `src/autograd.rs` | 4,665 | Active v3 engine + AutogradContext + `backward_v2` bridge (Phase 5b) |
| `src/autograd_v3.rs` | 533 | Alias / re-export layer for v3 |
| `src/autograd_v4/` (6 files) | ~392 | Feature-gated experimental v4 |
| `src/autograd_simple.rs` | 452 | Legacy dead code per FLAME_INDEX |
| `src/autograd_engine.rs` | 283 | Legacy dead code per FLAME_INDEX |
| `src/autograd_ops.rs` | 533 | Legacy dead code per FLAME_INDEX |
| `src/autograd_ops_complete.rs` | 898 | Legacy dead code per FLAME_INDEX |
| `src/autograd_debug.rs` | 171 | Legacy dead code per FLAME_INDEX |
| **TOTAL** | **~7,927** | |

Note: 4 of these (autograd_simple/engine/ops/ops_complete/debug) are flagged dead by `FLAME_INDEX.md` and may already be unreachable. **Verifying with a `cargo build` after removing them is the cheap first step** — if it builds, that's ~2,337 lines deleted with no migration work.

## External call-site histogram

Across `EriDiffusion-v2/crates/` + `inference-flame/src/`:

| Symbol | Call sites | Notes |
|---|---|---|
| `AutogradContext::no_grad()` | 126 | Pervasive — guard scopes around inference, sample, optimizer step |
| `AutogradContext::clear()` | 54 | End-of-step cleanup in trainers |
| `AutogradContext::checkpoint()` | 17 | Reentrant — universal trainer memory pattern |
| `AutogradContext::is_recording()` | 10 | Hot-path autograd gate |
| `AutogradContext::checkpoint_offload()` | 7 | Activation offload pool integration |
| `AutogradContext::backward_v2()` | 4 | Phase 5b bridge (trainer-side opt-in) |
| `AutogradContext::set_enabled()` | 2 | Klein parity tests + a few trainers |
| `AutogradContext::record_op()` | 1 | Single tape-record call site |
| `autograd::set_activation_offload_pool()` | 1 | Trainer setup |
| `autograd::clear_activation_offload_pool()` | 1 | Trainer setup |

**Total: 222 call sites** that need to either migrate to v2 equivalents or hit preserved wrappers.

`AutogradContext::backward()` (the v3 backward, 60+ sites) is not in this histogram because it's called as `loss.backward()` on the Tensor — same migration burden, different lookup pattern.

## What lives in `autograd.rs` that the v2 path still needs

Looking at what v2's bridge consumes from v3's `autograd.rs`:

| Symbol | Used by v2? | Notes |
|---|---|---|
| `AutogradContext::backward_v2()` | YES (it IS the bridge) | Must be relocated or preserved |
| `AutogradContext::backward()` shared impl `backward_impl()` | YES | Per Phase 5b `a5da3d5` the v2 path shares v3's op-dispatch loop |
| `AUTOGRAD_CONTEXT: Mutex<...>` static | YES (transitively, via bridge) | The bridge stores its tape here |
| `AutogradContext::clear() / set_enabled()` | YES (called by trainers around v2 backward) | Trainers reset state for the next step |
| Op enum (`Add`, `Mul`, `MatMul`, etc.) | YES | Bridge dispatches through v3's `compute_gradients` |
| `compute_gradients()` | YES | v2 bridge calls into v3's backward kernels |

**Verdict**: Item #6 cannot delete `src/autograd.rs` cleanly. It requires:

## Recommended retirement stages

### Stage A — Verify dead code is actually dead (cheap)

Delete the 4 files FLAME_INDEX flags as legacy:
- `src/autograd_simple.rs` (452 lines)
- `src/autograd_engine.rs` (283 lines)
- `src/autograd_ops.rs` (533 lines)
- `src/autograd_ops_complete.rs` (898 lines)
- `src/autograd_debug.rs` (171 lines, may have one debug-only consumer — check)

Run `cargo build --release --all-features` on flame-core and the workspace. If clean: commit + push. **~2,337 lines deleted, no migration risk.**

Estimated: 1 session.

### Stage B — Retire autograd_v4

`autograd_v4/` is feature-gated experimental code. Confirm nothing in the workspace enables the v4 feature flag:
```bash
grep -rn "feature.*autograd_v4\|features.*autograd_v4" --include='*.toml' --include='*.rs'
```

If not enabled by any consumer: delete `src/autograd_v4/` + remove the feature from `flame-core/Cargo.toml`. **~392 lines deleted.**

Estimated: 1 session.

### Stage C — Extract `backward_v2` from `autograd.rs`

Move the bridge to a v2-pure home:
- Create `src/autograd_v2/bridge.rs`
- Move `AutogradContext::backward_v2` and the shared `backward_impl(loss, policy)` there
- Keep `AUTOGRAD_CONTEXT` static where it is for now — the bridge still uses it transitively
- Update 4 trainer call sites (Z-Image, Klein, and any others added since) to call the new path

After this, `src/autograd.rs` no longer hosts v2 code. The bridge can survive eventual v3 deletion.

Estimated: 1-2 sessions. Risk: medium (touches the hottest paths of the engine).

### Stage D — Define v2 equivalents for `no_grad`/`clear`/`is_recording`/`set_enabled`

The v2 module has its own per-Tensor `AutogradMetaV2` and DispatchCtx, but lacks the AutogradContext-style global state ops. To unblock workspace migration:
- Define `flame_core::autograd_v2::context::{no_grad, clear, is_recording, set_enabled}` that operate on a v2-private global (or thread-local, if item #4 lands).
- These can initially be wrappers calling into AutogradContext — the goal is to give consumers a v2 import path so they can migrate without a flag-day.

Estimated: 1 session.

### Stage E — Workspace migration

Run a workspace-wide search/replace:
```
flame_core::autograd::AutogradContext   →   flame_core::autograd_v2::context
loss.backward()                          →   loss.backward_v2()
```

222 call sites. Each needs to be reviewed for whether it can migrate cleanly or needs a per-binary `--use-autograd-v2` flag like Z-Image/Klein have today.

Estimated: 3-5 sessions, spread across trainers. **Parity gates per binary** (the existing 100-step Z-Image smoke is the template).

### Stage F — Actually delete `src/autograd.rs` + `src/autograd_v3.rs`

After A-E land and all binaries pass under the v2 path, delete the v3 engine. **~5,200 lines deleted, single commit, branch (not main), user review.**

Estimated: 1 session for the delete + 1-2 sessions for CI/test cleanup after.

## Aggregate effort

A: 1, B: 1, C: 1-2, D: 1, E: 3-5, F: 1-3 → **~8-13 sessions** to complete item #6 honestly.

## What this session shipped

- This audit document
- No deletion

## What to do next session

Start with **Stage A** — it's a clean win, low risk, fast. Get the dead-code deletion in, then evaluate B-F based on appetite. Klein 9B step-2 crash root cause (separate, queued via memory `klein9b-step2-crash-bisect-status`) may take precedence depending on user direction.

The handoff envisioning item #6 as "delete the file in one commit" was off — the workspace dependency surface is wider than that. This audit replaces that single-line item with the multi-stage plan above.
