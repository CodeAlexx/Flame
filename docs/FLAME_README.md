# flame-core docs (for future Claude sessions)

**Read this first.** Four reference files in this directory. Each one is
short, grep-able, and curated for an LLM agent to look things up by name.

| File | When to read it |
|---|---|
| [`FLAME_INDEX.md`](./FLAME_INDEX.md) | "Where is X defined?" Flat symbol → `file:line` + 1-line description, grouped by module. The first place to look. |
| [`FLAME_MODULES.md`](./FLAME_MODULES.md) | "What does this codebase contain?" One paragraph per public module — what lives here, what depends on it, key types/functions. Read at session start to orient. |
| [`FLAME_KERNELS.md`](./FLAME_KERNELS.md) | "What CUDA kernels exist? Where? What's their layout?" Catalog of every NVRTC + `.cu` kernel with file, perf notes, layout assumptions. Critical for any perf work. |
| [`FLAME_CONVENTIONS.md`](./FLAME_CONVENTIONS.md) | "What's the convention for X?" Naming, file layout, dispatch patterns, gotchas. The stuff that takes 3 grep rounds to figure out each session. |

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
