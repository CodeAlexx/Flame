# flame-core docs (for future Claude sessions)

**Read this first.** The reference set has two tiers:

**Tier 1 — principles and audit gates.** These are the "why" docs. Short
and load-bearing. Read before making non-trivial changes.

| File | When to read it |
|---|---|
| [`../TENETS.md`](../TENETS.md) | Non-negotiable principles. "flame-core is one framework. Fix the primitive once → every model is fast." If a change violates a tenet, it's wrong even if it works. |
| [`SPEED_CONTRACT.md`](./SPEED_CONTRACT.md) | The 5-clause audit gate that operationalizes the tenets. Every PR touching a primitive, kernel, launch wrapper, autograd op, or memory subsystem must satisfy (or request exemption from) the relevant clauses. The TENETS doc tells you *why*, this one tells you *what to check*. |

**Tier 2 — symbol / module / kernel / convention / diagnostics index.**
The grep-able reference files in this directory, curated for LLM agents:

| File | When to read it |
|---|---|
| [`FLAME_INDEX.md`](./FLAME_INDEX.md) | "Where is X defined?" Flat symbol → `file:line` + 1-line description, grouped by module. The first place to look. |
| [`FLAME_MODULES.md`](./FLAME_MODULES.md) | "What does this codebase contain?" One paragraph per public module — what lives here, what depends on it, key types/functions. Read at session start to orient. |
| [`FLAME_KERNELS.md`](./FLAME_KERNELS.md) | "What CUDA kernels exist? Where? What's their layout?" Catalog of every NVRTC + `.cu` kernel with file, perf notes, layout assumptions. Critical for any perf work. |
| [`FLAME_CONVENTIONS.md`](./FLAME_CONVENTIONS.md) | "What's the convention for X?" Naming, file layout, dispatch patterns, gotchas. The stuff that takes 3 grep rounds to figure out each session. |
| [`FLAME_DIAGNOSTICS.md`](./FLAME_DIAGNOSTICS.md) | "I have a symptom — which probe finds the bug?" Symptom-first index of autograd probes, SDPA flags, parity harness, allocator knobs, and the trap meta-pattern for localizing gradient-direction bugs. |

The tier-2 docs cross-reference the tier-1 docs. When you see a "Reference
impl for SPEED_CONTRACT clause N" annotation in FLAME_INDEX or
FLAME_KERNELS (e.g. `cuda/narrow_strided.cu`, `cuda/sdpa_stream_bf16.cu`),
follow the link back to `SPEED_CONTRACT.md` to learn the pattern, then
apply it to whatever primitive you're touching.

### In-flight handoff docs

Two design-review handoffs are currently open and worth knowing about
before any cross-cutting work:

| File | What it covers |
|---|---|
| [`AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`](./AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md) | Plan to consolidate the 3 stacked autograd generations into a single per-op `GradFn` engine. Class A (BF16 grad storage) is scheduled to land alongside. |
| [`CLASS_B_NARROW_BACKWARD_HANDOFF.md`](./CLASS_B_NARROW_BACKWARD_HANDOFF.md) | Removed the BF16→F32→BF16 cast detour in `narrow_backward_scatter_add_cuda` (commit `15d8ef8`, 2026-05-12). Done; kept as the template for the rest of the Class A / Class B migration. |

## Pointers for fast lookup

- **Liveness annotations**: ⭐ = used by `inference-flame` (live), ⚠️ = legacy
  or training-only, plain = utility / framework. Use these to know what's safe
  to modify vs touch carefully.
- **Multiple generations**: this codebase has duplicate paths for many
  features (autograd v3/v4, sdpa vs sdpa_legacy, conv vs cuda_conv2d_*,
  multiple cuda_kernels files). The CONVENTIONS doc lists which is canonical
  for each.
- **Two CUDA build pipelines**: NVRTC (runtime, in `.rs` files as string
  consts) vs `cc-rs/nvcc` (build-time, `.cu` files in `cuda/` and `src/cuda/`,
  driven by `build.rs`). The KERNELS doc separates them.
- **BF16 family vs F32 framework**: the `bf16_*` modules are the inference
  hot path. The non-BF16 `cuda_*` and `cuda_kernels*` modules are the
  training/F32 framework. CONVENTIONS doc explains.

## When these docs get stale

- A new pub fn / struct → add a line to `FLAME_INDEX.md`
- A new CUDA kernel → add to `FLAME_KERNELS.md` (with perf notes if known)
- A new convention or gotcha → add to `FLAME_CONVENTIONS.md`
- A new module → add a paragraph to `FLAME_MODULES.md`

These are curated, not generated. A 5-minute update beats a 30-minute
rediscovery in a future session.

## See also

The flame-core source root has a few existing handoff/perf notes worth
knowing about:

```
flame-core/docs/BF16_CUDA_OPS_TODO.md
flame-core/docs/BF16_KERNELS_IMPLEMENTATION_PROMPT.md
flame-core/docs/HANDOFF_BF16_KERNELS.md
```

Plus these PERF notes in the repo root (not in docs/):
- `PERF_PERMUTE_FALLBACK_FIX.md`
- `PERF_SDPA_FLASH_KERNEL.md`
- `PERF_SDPA_QTILE_ATTEMPT.md`
- `PERF_VAE_PERMUTE.md`

Read those when working on the specific area they cover.
