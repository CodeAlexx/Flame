# flame-core (for Claude sessions)

Tensor library + CUDA kernels for diffusion model training and inference.
~100K lines, ~80 Rust files, ~50 CUDA kernels, multiple generations stacked
on top of each other.

## Read these first

The `docs/` directory has 4 reference files curated specifically for LLM
agents. **Read [`docs/FLAME_README.md`](./docs/FLAME_README.md) first** —
it's the index to the index. The 4 files:

- [`docs/FLAME_INDEX.md`](./docs/FLAME_INDEX.md) — flat symbol → `file:line`
  lookup, grouped by module. The first place to look when you need "where
  is X defined". ⭐ marks live (used by `inference-flame`), ⚠️ marks legacy.
- [`docs/FLAME_MODULES.md`](./docs/FLAME_MODULES.md) — one paragraph per
  public module. Read at session start to know where things live.
- [`docs/FLAME_KERNELS.md`](./docs/FLAME_KERNELS.md) — every CUDA kernel
  (NVRTC + build-time `.cu`) with launch configs and perf notes. Critical
  for any perf work.
- [`docs/FLAME_CONVENTIONS.md`](./docs/FLAME_CONVENTIONS.md) — naming, file
  layout, dispatch patterns, gotchas. The stuff that takes 3 grep rounds to
  figure out each session.

## Quick orientation

- **Live inference path**: `bf16_elementwise / bf16_ops / cuda_ops_bf16 /
  attention::sdpa / ops::fused_inference / serialization`. These are the
  modules `inference-flame` actually depends on.
- **Training path**: `cuda_kernels* / cuda_ops::GpuOps / autograd_v3 /
  optimizers / loss / regularization / pooling`. F32-heavy.
- **Multiple generations**: there are 3+ autograd engines, 2 SDPA
  dispatchers, 4 conv2d implementations, 2 versions of silu/gelu. The docs
  mark which is canonical for each.

## Two CUDA build pipelines

1. **NVRTC** (runtime-compiled, in `.rs` files as `const &str`) — preferred
   for new BF16 inference primitives. Files: `bf16_elementwise.rs`,
   `bf16_ops.rs`, `bf16_convert.rs`, `cuda_kernels*.rs`.
2. **Build-time** (`cc-rs/nvcc`, `.cu` files in `cuda/` and `src/cuda/`) —
   for cuBLASLt wrappers, flash attention, conv2d, anything that links
   against cuBLAS/cuDNN. `build.rs` lists every `.cu` source.

When in doubt: new BF16 inference op → NVRTC. cuBLASLt or cuDNN op → `.cu`.

## Hard rules

- **NEVER use the legacy paths in new code.** Specifically: `sdpa_legacy`,
  `cuda_tensor*`, `cuda_kernels::CudaKernels` (training F32), the standalone
  `attention/sdpa.rs::LayerNorm`. The CONVENTIONS doc lists the canonical
  alternative for each.
- **NEVER use F32 fallbacks** in inference code. If a kernel doesn't have
  a BF16 path, write one — don't silently cast.
- **NEVER add a new conv2d implementation.** Use `conv::Conv2d`.
- **NEVER pre-transpose weights at every call.** Use `fused_linear3d_native`
  which does the transpose inside the GEMM via cuBLASLt `TRANSA=T`.

## Known perf landmines (read CONVENTIONS for the full list)

- `lc(n)` vs `lc_pairs(n)` — using the wrong one halves vectorized kernel speed
- `<cfloat>` / `<float.h>` are not available in NVRTC — use literal constants
- `#pragma unroll` doesn't survive macro expansion — use `_Pragma("unroll")`
- `Tensor::softmax` has a BF16 last-dim fast path — for non-last-dim you'll
  hit the slow 5-step pipeline
- `cuda_ops_bf16::group_norm_bf16` is **NHWC**, not NCHW
- There are TWO silu/gelu implementations — `Tensor::silu` calls `bf16_ops::`,
  not `cuda/cuda_ops.cu::fc_silu_bf16`. Editing the wrong one is a no-op.

## When you change things

- New `pub fn` / `pub struct` → add a line to `docs/FLAME_INDEX.md`
- New CUDA kernel → add to `docs/FLAME_KERNELS.md`
- New convention or gotcha → add to `docs/FLAME_CONVENTIONS.md`
- New module → add a paragraph to `docs/FLAME_MODULES.md`

These are curated, not generated. A 5-minute update beats a 30-minute
rediscovery in a future session.
