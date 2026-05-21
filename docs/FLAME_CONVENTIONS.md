# flame-core conventions and gotchas

> The patterns, naming rules, and dispatch tricks that take 3 grep rounds to
> figure out each session. Read once, save hours later.

---

## File / module layout

### Three places kernels can live

1. **NVRTC inline string consts** — `src/bf16_*.rs`, `src/cuda_kernels*.rs`,
   `src/conv3d_bf16.rs`, etc. The kernel source is a `const &str`, compiled
   at runtime via `cudarc::nvrtc::compile_ptx_with_opts`. Each module has
   an `ensure(dev, name, code)` helper that compiles + caches via
   `dev.get_func(name, name).is_some()`. The launcher is a small `pub fn`
   in the same module.

2. **Build-time `.cu` files at `cuda/`** (repo root). Older surface. Symbols
   are `fc_*` returning `fc_status_t`. Compiled via `cc-rs/nvcc` from
   `build.rs`. The Rust FFI declarations live in `src/cuda_ops_ffi.rs`.

3. **Build-time `.cu` files at `src/cuda/`** (newer). Symbols are `flame_*`
   returning `int`. The Rust FFI declarations live in `src/cuda/ffi.rs`.

`build.rs` lists every `.cu` file with `cuda_sources.push(...)`. If you add
a new `.cu` file, you must add it there or it won't compile.

There's also `src/kernels/` (.cu files for SDPA / RoPE / GeGLU primitives)
and `kernels/` at the repo root (some duplicates of the `src/kernels/` files).
Always check `build.rs` to see which copy is actually built.

### Modules that look duplicated but aren't

- `attention/sdpa.rs` (the live public dispatcher) **vs** `sdpa.rs` (the
  lower-level dispatcher under it). Use `attention::sdpa` from model code.
- `attention/sdpa_legacy.rs` and `sdpa_legacy.rs` — both legacy, unrelated
  to the live path. Don't call.
- `layer_norm.rs::LayerNorm` (the live struct) **vs** `attention/sdpa.rs::LayerNorm`
  (a duplicate inside the attention module — don't use).
- `cuda_ops.rs::GpuOps` (training F32) **vs** `cuda_ops_bf16.rs` (live BF16).
  The live entry from `Tensor::permute` falls back to `GpuOps::permute_generic`
  for non-fast-path orders, but otherwise `cuda_ops.rs` is training-only.
- `cuda_kernels.rs` and `cuda_kernels_gpu.rs` — both training NVRTC F32
  kernel registries.
- Multiple conv2d implementations: `conv::Conv2d` (use this), `cuda_conv2d*.rs`
  (older direct CUDA), `ops/conv2d*.rs` (alternative entry points).
- Never add a second optimizer struct alongside the canonical one. New optimizer variants go into the canonical file as methods or as a new named type.

When you find yourself with two functions with the same name in different
files: the canonical one is whichever `inference-flame` actually calls.
Search `inference-flame/src` for `flame_core::module_name::fn_name` to find
out.

---

## Naming conventions

### Module / function naming

| Pattern | Meaning | Example |
|---|---|---|
| `*_bf16` | BF16 operand | `silu_bf16`, `rms_norm_bf16`, `add_bf16` |
| `*_f32` | F32 operand | `mse_loss_f32`, `silu_f32` |
| `*_into` | Output-into variant (writes to a passed-in `&mut Tensor`) | `gelu_bf16_into`, `layer_norm_into` |
| `*_with_stats` | Returns intermediate stats (mean/rstd) for backward | `layer_norm_bf16_with_stats` |
| `fused_*` | Single kernel covering multiple ops | `fused_rms_norm`, `fused_residual_gate` |
| `*_native` | Takes weight in standard PyTorch `[Cout, Cin]` layout (not pre-transposed) | `fused_linear3d_native` |
| `*_flat` | Fast path for contiguous same-shape inputs | `add_bf16_flat_kernel`, `mul_bf16_flat_kernel` |
| `*_kernel` | The kernel itself (vs the launcher) | `silu_bf16_kernel` |
| `flame_*` | C-side `extern "C"` symbol returning `int` | `flame_flash_attention_bf16` |
| `fc_*` | C-side `extern "C"` symbol returning `fc_status_t` | `fc_rms_norm_bf16` |

### Tensor parameter ordering

For elementwise: `op(input1, input2)`. For norms: `norm(x, weight, bias, eps)`
or `norm(x, weight, eps)` if no bias. For matmul: `matmul(x, weight)` (NOT
`matmul(weight, x)`). For attention: `sdpa(q, k, v, mask)`.

### Output dtype

By default, BF16 ops return BF16. F32 ops return F32. There is NO automatic
promotion — passing an F32 tensor to a BF16 op raises an error. Use
`.to_dtype(DType::BF16)` explicitly.

---

## The launch config family

### `lc(n)` vs `lc_pairs(n)` — vectorized vs scalar kernels

In `bf16_*.rs` and `bf16_convert.rs`:

```rust
#[inline]
fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

#[inline]
fn lc_pairs(n: usize) -> LaunchConfig {
    let pairs = (n + 1) / 2;
    LaunchConfig::for_num_elems(pairs as u32)
}
```

**Use `lc(n)` for 1-element-per-thread kernels** (the default).
**Use `lc_pairs(n)` for 2-element-per-thread vectorized kernels** that
process `__nv_bfloat162` pairs. If you launch a vectorized kernel with
`lc(n)`, you'll launch 2× as many threads as needed and they'll all check
`if (i2 < n2)` and exit, wasting half the work. The kernel will still
produce correct output but at half the speed. **This bit me twice.**

`build.rs` C++ code (in `cuda/` and `src/cuda/`) uses an inline
`launch_grid(n_pairs, &grid, &block)` helper instead.

### Block / grid sizing

For "1 row per block" kernels (RMSNorm, LayerNorm, softmax):
```rust
let block_size = 1usize;
while block_size * 2 <= cols && block_size * 2 <= 1024 { block_size *= 2; }
if block_size < 32 { block_size = 32; }
let grid = (rows as u32, 1, 1);
let block = (block_size as u32, 1, 1);
```

Power-of-two block size up to 1024, minimum 32 (one warp). Shared memory is
sized for the reduction.

---

## NVRTC pitfalls

### `<cfloat>` and `<float.h>` are NOT available

NVRTC only ships a minimal subset of the C++ standard library. If you need
`FLT_MAX`, define it as a literal in the kernel source:

```cuda
#define LOCAL_FLT_MAX 3.402823466e+38f
```

Don't `#include <float.h>` or `#include <cfloat>`. They will fail with
"cannot open source file" at runtime when the NVRTC compile happens — and
the failure only shows up the first time the kernel is called, not at
build time.

### `#pragma unroll` inside macro definitions

`#pragma` doesn't survive C preprocessor macro stringification. If you have
a macro that defines a kernel (like `flash_attention_fwd.cu`'s
`DEFINE_FLASH_ATTN_WMMA_KERNEL`), use `_Pragma("unroll")` instead:

```cuda
_Pragma("unroll")
for (int off = 16; off > 0; off >>= 1) { ... }
```

### Compiled-once kernel cache

`cudarc` caches NVRTC-compiled functions per `(device, name)`. If you change
the kernel source string in `bf16_ops.rs` and rebuild, the new kernel WILL
be picked up at runtime (because the device-side cache is per-process and a
fresh `cargo run` starts a new process). But if you forget to bump the
function name and the cache layer is shared somehow (e.g. across test runs),
you can get the old kernel. Easiest debug: rename the kernel temporarily to
force a recompile.

---

## BF16 vs F32 hot path — what to use when

| Situation | Use |
|---|---|
| Inference, BF16 pointwise binary (add/sub/mul/div/max/min, broadcast or strided) | `tensor_iterator::ops::binary::*_bf16_iter` (auto-routed by `Tensor::*`) |
| Inference, BF16 pointwise unary (silu/gelu/sqrt/exp/abs/neg/square/…) | `tensor_iterator::ops::{unary,transcendentals}::*_bf16_iter` (auto-routed by `Tensor::*`) |
| Inference, BF16 comparison (ge/gt/le/lt/eq/ne) | `tensor_iterator::ops::comparison::*_bf16_iter` (output is BF16 0.0/1.0) |
| Inference, BF16 matmul | `Tensor::matmul` (auto-routes) or `ops::fused_inference::fused_linear3d_native` for the cuBLASLt+bias path; `fused_linear3d_native_pytorch_parity` (2026-05-20) for bit-exact PyTorch parity at ~1% overhead |
| Inference, last-dim softmax | `Tensor::softmax` BF16 fast path → `bf16_elementwise::softmax_lastdim_bf16` |
| Inference, 2D transpose | `bf16_elementwise::transpose2d_bf16` |
| Inference, DiT patchify/unpatchify | `bf16_elementwise::patchify_bf16 / unpatchify_bf16` |
| Inference, RMSNorm/LayerNorm | `cuda_ops_bf16::rms_norm_bf16 / layer_norm_bf16` |
| Inference, RMSNorm `(weight+1)` formulation | `ops::fused_inference::fused_rms_norm` with weight pre-added 1.0 at load time (Gemma3, MagiHuman, Z-Image NextDiT) — see FLAME_INDEX.md RMSNorm section |
| Inference, attention | `attention::sdpa` |
| Inference, RoPE | `bf16_ops::rope_fused_bf16` (interleaved-pair) or `bf16_ops::rope_halfsplit_bf16` (Z-Image, MagiHuman halfsplit format). Both rotate the FULL last dim — wrap with split→rotate→cat for partial-rotation models like MagiHuman (head_dim=128, ROPE_DIM=96). |
| Inference, FFN gate-residual | `bf16_ops::gate_residual_fused_bf16` and `bf16_ops::swiglu_fused_bf16` |
| Training F32 tensor add | falls through to `cuda_ops::GpuOps::add` |
| Need autograd | use the `Tensor::*` methods (they record on the tape); the bare `*_iter` functions DO NOT record |

### The autograd recording trap

`tensor_iterator::ops::binary::add_bf16_iter(a, b)` does NOT record on the
autograd tape. `Tensor::add(&b)` DOES (when `requires_grad` is true).

So:
- Inference code can call the bare functions directly — slightly faster, no
  tape overhead.
- Training code must call `Tensor::add(&b)` (or `tensor_a + tensor_b` if
  the operator overload is in scope).

The fused inference primitives in `ops::fused_inference::*` also do NOT
record. They're designed for inference, not training.

#### `rope_fused_bf16` autograd — historical bug

The interleaved variant `bf16_ops::rope_fused_bf16` (used by Klein,
Z-Image, Chroma, Wan, FLUX) had `output.requires_grad: false` hardcoded
and never recorded `Op::RoPePrecomputed`. Only the halfsplit variant
`rope_fused_bf16_halfsplit` (used by LTX-2) recorded.

Symptom: in any LoRA trainer that runs Q/K through interleaved RoPE
(e.g. Z-Image trainer's Q/K post-LoRA-delta path), `lora_B` for Q and K
adapters stayed at zero-init across every training step while `lora_B`
for V (which skips RoPE) trained normally. The asymmetry was the
diagnostic: identical config, only Q/K stuck.

The fix at `bf16_ops.rs:735-757` propagates `requires_grad` and records
the op when the input requires grad. The `Op::RoPePrecomputed`
backward dispatcher was correct *for the models it had seen so far*
(broadcast cos `[1, N, half]` → interleaved, per-head cos
`[BH, N, half]` → halfsplit); the recording side was the missing link.

#### `Op::RoPePrecomputed` carries an explicit layout tag (2026-05-20)

The shape-sniffing dispatcher above had a hidden hazard: HiDream-O1's
MRoPE emits cos of shape `[1, S, half]` — rank-3 — but the forward is
`rope_halfsplit_bf16` (HuggingFace `rotate_half` convention used by the
Qwen3-VL backbone). The old shape-sniff classified rank-3 broadcast cos
as Interleaved, so backward applied the wrong rotation while forward
applied the right one. Symptom: Q/K LoRA-B grad cos ≈ 0.01-0.05 vs
PyTorch with magnitude in the right ballpark (orthogonal direction,
correct length — the trap meta-pattern catches this; magnitude-only
checks do not).

`Op::RoPePrecomputed` now carries an explicit
`autograd::RopeLayout` tag (`autograd.rs:177`, variants `Interleaved`
and `Halfsplit`). The three fused-RoPE forwards each pass the correct
tag at their `record_op` site:

- `bf16_ops::rope_fused_bf16` at `bf16_ops.rs:923` → `Interleaved`
- `bf16_ops::rope_fused_bf16_f32pe` at `bf16_ops.rs:1068` → `Interleaved`
- `bf16_ops::rope_halfsplit_bf16` at `bf16_ops.rs:1183` → `Halfsplit`

Backward dispatches by tag (`autograd.rs:4207-4230`), never by shape.
**When adding a new fused-RoPE primitive, pass the correct
`RopeLayout` at the record site. Don't sniff the cos tensor.** New
record_op callers outside `bf16_ops.rs` (e.g. trainer code in
`EriDiffusion-v2/.../chroma.rs:2332`) must also supply the tag.

**Pattern to remember:** any new BF16 inference primitive in
`bf16_ops.rs` / `ops/fused_inference.rs` that's reachable from a trainer's
forward must propagate `x.requires_grad` and record an autograd op,
OR fail loud when called inside `AutogradContext::is_recording()`.
Hardcoding `requires_grad: false` on a forward path that trainers can
hit is a silent gradient-killer.

**Diagnostic signal:** a specific LoRA target's `lora_B` is *exactly*
0.0 (not 1e-30, EXACTLY zero) across many training steps while sibling
adapters in the same block train normally → autograd chain cut, not a
learning rate or scaling issue. Inspect every op the affected target's
output flows through; one of them is inference-only.

#### SDXL trainer autograd sweep (2026-04-25)

When the SDXL trainer was first wired up, `loss.backward()` failed with
"backward called on tensor that doesn't require grad" — meaning the
forward had broken the chain somewhere upstream of the loss. Audit
turned up four flame-core primitives that ran but did not record:

1. **`ops::conv2d::conv2d_forward`** — added `Op::Conv2d` recording at
   the end of the closure. Wraps the closure result so that
   `requires_grad` propagates from input/weight/bias.
2. **`cuda_ops::GpuOps::permute_nchw_to_nhwc`** /
   **`permute_nhwc_to_nchw`** — added `Op::Permute` recording with the
   correct dim vectors `[0,2,3,1]` and `[0,3,1,2]`.
3. **`cuda_ops::GpuOps::upsample2d_nearest`** — added the new
   `Op::UpsampleNearest2D` variant + recording. The forward kernel was
   already in place; the backward was a stub that returned zeros.
4. **`sdxl-trainer::model::linear_fused`** — was routed through the
   inference-only `fused_linear3d_native`. Rerouted to
   `linear_autograd` (a tape-recording `matmul + add`).

Beyond recording, the **F32 training-path conv2d backward**
(`cuda_conv2d::CudaConv2d::conv2d_backward`) had two real bugs that
fired the moment the dispatcher started using it:

- It assumed F32 inputs and asserted on BF16 storage. The autograd
  dispatcher now casts to F32 at the call site (the kernel returns F32
  grads, which `accumulate_parameter_grads` handles).
- The grad-input matmul had an extra `.transpose()` on the weight
  reshape — `weight.reshape([oc, ic*kh*kw]).transpose() @ grad_col`
  produces a dim mismatch on any output where `oc != ic*kh*kw`. Removed
  the transpose: `grad_col [B*OH*OW, OC] @ weight_2d [OC, IC*KH*KW]` →
  `grad_input_col [B*OH*OW, IC*KH*KW]`.

And **`autograd_ops_complete::group_norm_backward`** ran its rank-checking
`guard_tensor` on `mean`/`var` (saved as F32 by design — see
`group_norm.rs:402-427`) **before** the BF16 fast path that ignores
them. Reordered: check the BF16 fast path first, only guard mean/var
on the F32 fallback.

**`upsample2d_nearest_backward`**: the F32 stub at
`cuda_kernels.rs::upsample2d_nearest_backward` returned a zeroed
gradient. Replaced with a real NVRTC kernel that mirrors the forward
mapping (`h_in_idx = h * h_in / h_out`) and uses `atomicAdd` to scatter
into the input grad. BF16 grad_outputs are cast to F32 inside the
backward (atomicAdd on BF16 is unsupported on older arches).

**Pattern:** any new training-path kernel that lands in flame-core must
be exercised by an actual `loss.backward()` call before being marked
done. Forward parity does not catch broken backward kernels — only
the smoke binary does (see `sdxl-trainer/src/bin/autograd_smoke.rs`
for the template: synthesize fake inputs, run forward + MSE +
backward, count params with grads + nonzero grads).

### `contiguous()` and friends on non-contig views

`Tensor::contiguous()` on a non-contig view (the typical post-`narrow` /
`permute` case) dispatches to `GpuOps::materialize_view` or one of the
permute fast-paths. Those kernels allocate the output via
`Tensor::empty_dtype`, which sets `requires_grad: false`. Without
`contiguous()`'s own autograd wiring, the result would not carry a grad,
and anything downstream (notably `to_dtype`, which re-routes non-contig
inputs through `self.contiguous()?.to_dtype(dtype)`) would silently drop
the chain.

`contiguous()` itself closes that gap — it propagates `requires_grad`
and records an identity-reshape (`Op::Reshape { new_shape = self.shape }`)
whose backward is a no-op shape match. Gradients from the contig output
route unchanged back to the strided source, which then carries them
through its own `Op::Slice` / `Op::Permute`.

If you add another path inside `contiguous()` (a new fast-path kernel,
say), route it through `finalize_contiguous_autograd(contig)` before
returning.

---

## Layout conventions

### NCHW vs NHWC

- `tensor::Tensor` has no inherent layout — it's just `[d0, d1, d2, ...]`.
- `conv::Conv2d::forward(input)` expects **NCHW**.
- `conv::Conv2d::forward_nhwc(input)` expects **NHWC**.
- `cuda_ops_bf16::group_norm_bf16(x, ...)` expects **NHWC** (this trips
  people up — the docstring says `[N, H, W, C]`).
- `group_norm::group_norm(...)` (the functional API) handles either layout
  by converting if needed.

If you're going to call `group_norm_bf16` directly, permute first:
```rust
let nhwc = nchw.permute(&[0, 2, 3, 1])?;
let out_nhwc = group_norm_bf16(&nhwc, ...)?;
let out = out_nhwc.permute(&[0, 3, 1, 2])?;
```

### `[B, N, C]` vs `[B, C, N]` for sequence-like tensors

Most flame-core inference functions use `[B, N, C]` (batch, sequence, channel)
— same as PyTorch transformer convention. Some legacy training code uses
`[B, C, N]`. Check the function docs.

### `fused_linear3d_native` is strictly 3D — wrap 2D inputs explicitly

Unlike PyTorch's `nn.Linear` (which broadcasts over arbitrary leading dims),
`ops::fused_inference::fused_linear3d_native` rejects 2D input with
`InvalidShape: input must be 3D [B,N,Cin]`. Same for the non-`_native`
variant. When porting code that does linear ops on `[N, D]` tensors
(timestep embedders, sinusoidal MLPs, conv-as-matmul collapses), reshape:

```rust
let n = x.shape().dims()[0];
let x3d = x.reshape(&[1, n, in_features])?;
let y = fused_linear3d_native(&x3d, w, bias)?; // [1, n, out_features]
let y2d = y.reshape(&[n, out_features])?;
```

Surfaces as a runtime panic deep in the forward pass on the first call site
that feeds a 2D tensor — easy to mistake for a layout bug. Encountered on
SenseNova-U1's `extract_feature_gen` and `time_or_scale_embed`.

### Attention `[B, H, N, D]`

`attention::sdpa(q, k, v, mask)` expects `[B, H, N, D]` where H is heads, N
is sequence length, D is head_dim. Reshape from `[B, N, H*D]` first via
`.reshape(&[B, N, H, D]).permute(&[0, 2, 1, 3])`.

### Weight layouts

Standard PyTorch `nn.Linear` saves weight as `[Cout, Cin]` row-major.
Most flame-core matmul functions want this layout — `Tensor::matmul`,
`ops::fused_inference::fused_linear3d_native`, etc.

The exception is `fused_linear3d` (without `_native`): it wants
**pre-transposed** `[Cin, Cout]` row-major. Klein and a few other models
pre-transpose all linear weights at load time. New code should use
`fused_linear3d_native` instead.

### Picking `fused_linear3d_native` vs `_pytorch_parity` (2026-05-20)

Both take PyTorch `[Cout, Cin]` weight layout and produce the same
mathematical result, but with 1-2 BF16 ULP differences from PyTorch's
shipping cuBLASLt invocation.

- **`fused_linear3d_native`** — default. Cached descriptor + cached
  algo. ~1% faster. Use for all established baselines (Klein, Z-Image,
  Chroma, FLUX, QwenImage, Wan, LTX-2) — they were validated against
  this path and changing the algo will regress their parity numbers.
- **`fused_linear3d_native_pytorch_parity`** — strict PyTorch
  bit-exact. Biased calls mirror PyTorch `gemm_and_bias`: workspace =
  1 MiB, BIAS_POINTER on descriptor before
  `cublasLtMatmulAlgoGetHeuristic`, per-call heuristic, no
  BIAS_DATA_TYPE / batch / alignment attrs. No-bias calls delegate to
  `fused_linear3d_native`, because PyTorch no-bias `F.linear` follows
  the matmul path and O1 Full `x_embedder.proj1` matches native
  byte-exactly. Use where ai-toolkit per-op parity is the contract:
  HiDream-O1 `TimestepEmbedder`, `BottleneckPatchEmbed` proj1/proj2
  (no-LoRA paths).

**Do not flip established models to the parity variant** to "improve
parity" — their reference points are against `fused_linear3d_native`
output, not against PyTorch's raw output. You'd make their parity
worse.

---

## Adding a new BF16 op — template

Pointwise (elementwise) BF16 ops go through the TensorIterator pipeline.
Fused / structured / reduction kernels stay under `bf16_ops.rs` with the
old NVRTC pattern. Pick the right lane.

### Pointwise unary, binary, transcendental, comparison

Three pieces: a `.cu` functor, a `declare_stub!`, and a Rust `_bf16_iter`
wrapper. All three are small — each existing op under
`src/tensor_iterator/ops/` is the working template. Copy the nearest
sibling (`silu` for unary, `add` for binary, `exp` for transcendental,
`ge` for comparison) and rename.

1. **`.cu` functor** — drop a new file under `src/cuda/{unary,binary,cmp}/`
   with a `__device__` functor struct and an `extern "C"` entry that calls
   `flame::iter::gpu_kernel<NARGS>(...)` with that functor. For
   transcendentals use the f32-opmath shim: convert BF16→f32, compute,
   `__float2bfloat16_rn` back. Add the file to the `cuda_sources` list in
   `build.rs`.

2. **`declare_stub!`** at the top of the matching `tensor_iterator/ops/*.rs`:
   register the FFI entry in the `DispatchStub` registry. See the 27
   existing invocations for the exact shape.

3. **Rust wrapper** in the same file:
   ```rust
   pub fn my_op_bf16_iter(x: &Tensor) -> Result<Tensor> {
       let mut iter = build_unary_op(x, /*promote=*/false)?;
       dispatch_my_op(&mut iter)?;   // from declare_stub!
       Ok(iter.take_output())
   }
   ```
   `build_unary_op` / `build_binary_op` / `build_comparison_op` (in
   `tensor_iterator/base.rs`) handle broadcast, coalescing, and output
   allocation. You do not write offset math.

4. **Autograd** — add an `Op::` variant and a backward arm in
   `autograd_ops_complete.rs` if the op ever runs under `requires_grad`.
   Inference-only ops skip this and call `_bf16_iter` directly.

5. **Wire into `Tensor::*`** — add a `pub fn my_op(&self)` method to
   `tensor.rs` (or the appropriate `tensor_ops_*.rs`) that routes BF16 to
   `my_op_bf16_iter` via `dispatch_unary_bf16` /
   `dispatch_binary_bf16` in `tensor_iterator/dispatch_helpers.rs`, and
   F32 through `GpuOps`.

6. **Parity test** — add `tests/tensor_iterator_my_op_parity.rs`. Compare
   contig output bit-exact to a BF16 oracle (write one inline if none
   exists; the three retained oracles `silu_bf16`/`gelu_bf16`/`square_bf16`
   in `bf16_ops.rs` are the pattern).

### Fused / structured / reduction (RoPE, swiglu, softmax, …)

These stay NVRTC with the old pattern. In `src/bf16_ops.rs`:

```rust
const CUDA_MY_FUSED: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void my_fused_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                           __nv_bfloat16* __restrict__ Y,
                           long n) {
    // ... your fused math ...
}
"#;

pub fn my_fused_bf16(x: &Tensor) -> Result<Tensor> {
    // ensure kernel compiled, alloc output, launch with lc(n) or lc_pairs(n).
    // See `rope_fused_bf16` or `swiglu_fused_bf16` for the full pattern.
}
```

Use `lc_pairs(n)` for a 2-element-per-thread vectorized kernel; `lc(n)`
for scalar. Same `ensure` + `get_func` + `f.launch` dance as the existing
fused kernels.

### Build-time `.cu` kernel (cuBLASLt wrapper, big kernel)

1. Create `src/cuda/my_kernel.cu` with an `extern "C" int flame_my_op_bf16(...)`
   entry point and any `__global__` kernels it dispatches to.
2. Add `cuda_sources.push("src/cuda/my_kernel.cu");` in `build.rs`.
3. Declare in `src/cuda/ffi.rs`:
   ```rust
   pub fn flame_my_op_bf16(
       handle: *mut core::ffi::c_void,
       /* ... */
       stream: *mut core::ffi::c_void,
   ) -> i32;
   ```
4. Write the Rust wrapper in `src/ops/fused_inference.rs`:
   ```rust
   #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
   pub fn fused_my_op(input: &Tensor, ...) -> Result<Tensor> {
       // alloc output, get stream, get cublasLt handle
       let device = input.device();
       let stream = device_lt::stream_ptr(device)?;
       let lt = device_lt::cublaslt_handle_ptr(device)?;
       let workspace_size: usize = 4 * 1024 * 1024;
       let workspace: cudarc::driver::CudaSlice<u8> = unsafe { device.alloc(workspace_size)? };
       // ... build the call ...
       let ret = unsafe {
           crate::cuda::ffi::flame_my_op_bf16(
               lt, input.as_device_ptr_bf16("fused_my_op:input")? as *const _,
               /* ... */
               stream,
           )
       };
       if ret != 0 {
           return Err(Error::Cuda(format!("fused_my_op error: {ret}")));
       }
       Ok(output)
   }
   ```

After editing the `.cu` file, `touch src/cuda/my_kernel.cu` and rebuild —
cargo's incremental build sometimes misses `.cu` mtime changes.

---

## Common gotchas

### Rust `f32::round` ≠ PyTorch `tensor.round()` (rounding-mode parity)

When porting quantization code from PyTorch:
- `f32::round()` (Rust default) is **round-half-away-from-zero**.
- `tensor.round()` (PyTorch) is **IEEE 754 round-half-to-even** (banker's
  rounding, default for IEEE float-to-int).

They diverge on exactly the .5 boundary, so on dense quantization
codes (e.g. int8 absmax-scaled rounds) you'll see ~0.02% of values
off by ±1 with the wrong choice — silent numerical drift that erodes
parity tests one row at a time. Use `f32::round_ties_even()`
(stabilized 1.77) for any PyTorch-parity quant path.

Caught while porting `int8_weight_only_qt_kernel.rs` from torchao
0.14.1; full receipt at that module's parity binary
(`bin/parity_int8_torchao_qt.rs`). Same gotcha applies to any future
GPU quant kernel that needs PyTorch-parity codes — use `rintf` (CUDA
banker's-round-to-nearest-int) NOT `roundf` (round-half-away-from-zero).

### View-autograd backwards under `shared_storage` (HAZARD-2026-05-13-1)

**TL;DR**: For view ops (`narrow`, `transpose`, `permute`, `view`,
`reshape`, `squeeze`, `unsqueeze`), the backward must NEVER write
through a view back into a parent tensor. Allocate a fresh zero
tensor and scatter-add via the dedicated `*_backward_scatter_add_*`
kernel (e.g. `narrow_backward_scatter_add_cuda` in
`tensor_narrow.rs:169`).

**Why**: Under the `shared_storage` feature (default-on),
`parent.narrow(...)` returns a Tensor whose inner `Arc<CudaSlice>`
aliases the parent's storage (refcount ≥ 2). Any in-place mutation
through `try_as_mut_slice_*` calls `ensure_unique_slice` →
`Arc::make_mut`, which SILENTLY clones the slice when the refcount
> 1. The view's local Arc points at the clone; the kernel writes
into the clone; the parent is untouched. No error, no warning, silent
wrong data. This is HAZARD-2026-05-13-1, characterised by the negative
test `tests/autograd_v2_ops.rs::hazard_view_inplace_does_not_mutate_parent_under_shared_storage`.

**autograd_v2 implementations** (`src/autograd_v2/ops/`):
- `NarrowGradFn::apply` allocates a fresh `Tensor::zeros_dtype(...)` and calls
  `Tensor::narrow_backward_scatter_add_cuda(&g, &mut grad_in, dim, start, length)`.
  That kernel writes through the `&mut grad_in` handle — there's no
  view/aliasing involved.
- `ReshapeGradFn` / `SqueezeGradFn` / `UnsqueezeGradFn` reshape grad
  via `Tensor::reshape` / `unsqueeze` / `squeeze` — those are also
  metadata-only views (no in-place writes happen inside them) and the
  resulting tensor is consumed downstream, not written into.

### View-op backwards that return strided views must call `.contiguous()`

Project-wide, `gemm`/`bmm` kernels read storage as if it were row-
major contiguous and IGNORE per-tensor strides. View backwards that
return strided views (`TransposeGradFn::apply`, `PermuteGradFn::apply`)
must call `.contiguous()` on the strided result before returning, so
downstream gemm/bmm consumers see correctly-laid-out memory. Phase 3a
matmul fix (commit `6ee385f`) added this same `.contiguous()` to
`MatMulGradFn`'s transpose paths; Phase 3b applies the discipline
to `TransposeGradFn` and `PermuteGradFn`. Annotate each `.contiguous()`
call with a `// HAZARD-2026-05-13-1 + gemm-stride-ignore` comment so
future maintainers know why it's there.

### `DType::I32` tensors hold f32 bytes, not real i32 bytes

The `TensorStorage::I32 { data: StorageSlice<f32>, .. }` variant uses an
**f32 buffer** as backing storage. `Tensor::to_dtype(DType::I32)` does not
convert values — it only relabels the storage type. Consumers that need
integer values cast back to F32 first via `to_dtype(DType::F32)`, which
goes through `storage.to_f32()` and preserves the f32 view.

Practical effect: if you have an F32 tensor with values `[1.0, 3.0, 6.0]`
and call `.to_dtype(DType::I32)`, the underlying GPU buffer still holds
the f32 bit patterns of `1.0`, `3.0`, `6.0` — i.e. the bytes
`0x3F800000`, `0x40400000`, `0x40C00000`. Reading those bytes as `int*`
yields `1065353216`, `1077936128`, `1086324736`, **not** `1`, `3`, `6`.

Most flame-core ops that take I32 indices (e.g. `index_select0`,
`gather_rows`) compensate by casting back to F32 internally before
doing the actual gather, which is why the existing convention works
end-to-end without anyone noticing the bit mismatch.

**This breaks any C/CUDA kernel that takes `const int*` directly.**
That includes the build-time MoE kernels in `src/cuda/grouped_mm.cu`
and `src/cuda/fused_gated_scatter_add.cu` — both expect real i32 bytes
for `offsets`/`indices`.

The wrappers in `ops/grouped_mm.rs` and `ops/fused_gated_scatter_add.rs`
work around this by accepting `&[i32]` host slices and HtoD-copying into
a temporary `cudarc::driver::CudaSlice<i32>` (which IS real i32 bytes)
before launching the kernel. For per-step inference where offsets/indices
are tiny (E≤1024 experts × per-token routing), the HtoD overhead is
microseconds and irrelevant against the millisecond-class GEMMs.

If a future caller ever produces i32 indices on the GPU directly (e.g.
from a GPU-side top-K routing kernel), the right fix is to add an
unsafe variant that accepts a raw `*const i32` device pointer rather
than going through a Tensor — adding a "real i32" dtype path is a
larger change than is warranted for the few sites that need it.

### `TensorStorage` has no `U8` arm — use raw `CudaSlice<u8>` for U8 device buffers

`DType::U8` is declared in `dtype.rs:10` but `TensorStorage` does not have a
matching variant — `tensor_storage.rs:418-444` returns
`Error::InvalidOperation` for `DType::U8` in `zeros` / `empty` / etc. You
cannot construct a `Tensor` with `DType::U8`; it will fail at storage
construction.

If you need a U8 device buffer (e.g. for 8-bit quant codes), allocate
`cudarc::driver::CudaSlice<u8>` directly via
`device.alloc_zeros::<u8>(n)?` and hold it outside the Tensor / autograd
graph. The canonical pattern is `adam8bit_kernel::alloc_state` (the bnb
8-bit AdamW optimizer holds the per-param U8 `m_codes` / `v_codes` as raw
slices on the optimizer struct, NOT as Tensors). The FFI launcher takes
the raw `CudaSlice<u8>` and dereferences `*device_ptr_mut()` for the
kernel arg list.

Plumbing U8 through `Tensor` + the alloc pool + autograd is a full-day
refactor with no current consumer — don't reach for it just because the
DType is declared. The escape hatch above is sufficient for kernel-local
state buffers that never enter the autograd graph.

### `Tensor::from_vec` is F32-only — bypass for non-F32 host uploads

`Tensor::from_vec` (`tensor.rs:1165`) accepts `Vec<f32>` only. If you need
to upload a small constant vector of a different dtype (e.g. a 256-entry
LUT, a per-block BF16 absmax table, a `Vec<u8>` codebook), do not try to
route through `Tensor` — use `device.htod_copy(vec)?` directly. It returns
a `CudaSlice<T>` for any `T: cudarc::driver::DeviceRepr`.

Canonical example: `adam8bit_kernel::upload_qmap` does
`device.htod_copy(qmap.to_vec())` to land a `[f32; 256]` LUT on the device
as a `CudaSlice<f32>`, then passes that slice directly to the NVRTC
launcher — no Tensor wrapper, no `Storage` allocation, no version
tracking. Same pattern works for `Vec<bf16>`, `Vec<u8>`, `Vec<i32>`. The
device slice is owned by the caller; allocate once, reuse across launches.

For F32 vectors that DO need to be a `Tensor` (because they will be
broadcast against tracked tensors or enter the autograd graph),
`Tensor::from_vec` is still correct — but the LUT / state-buffer case
above is none of those.

### cuBLAS leaves `cudaErrorInvalidValue` latched after first GEMM call

cuBLAS GEMM's first invocation per process probes device capabilities
internally. Some of those probes call CUDA APIs that fail with
`cudaErrorInvalidValue=2` and **latch it in the per-thread last-error
queue** even though GEMM itself returns success via `cublasStatus_t`.

The next kernel that calls `cudaGetLastError()` after its launch picks
up that sticky error and reports it as its own failure. Symptom:
`flame_<op>_bf16_kernel failed with code 2` on completely valid inputs
with canonical strides — and the same op against the same shapes works
fine in isolation (no upstream GEMM).

**Fix discipline (mirrors PyTorch's `TORCH_CUDA_CHECK_AFTER_LAUNCH`):**
every tensor_iterator-style FFI kernel drains the last-error queue
right before launch:

```cpp
extern "C" int flame_<op>_bf16_kernel(const flame::iter::IterMetadata* meta,
                                       void* stream_void) {
    // Drain any sticky error left by upstream (e.g. cuBLAS GEMM
    // capability probe) so cudaGetLastError below reflects only
    // this launch.
    (void)cudaGetLastError();
    flame::iter::launch_gpu_kernel<...>(*meta, ..., stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
```

All 26 kernels in `src/cuda/{binary,unary,cmp}/` carry this drain.
Skipping it is fine for in-isolation correctness but produces flaky
training failures whenever a matmul precedes the kernel — and FLUX 1
trainer hit it deterministically on every QKV LoRA delta.

**Bisection helpers** in `flame_core::device`:
- `cuda_peek_last_error() -> i32` — non-clearing peek.
- `cuda_probe(tag) -> i32` — synchronizes the device, then consumes +
  prints the per-thread last error (sync errors have already been
  consumed by `cudaDeviceSynchronize`'s return value, latched
  launch-validation errors are consumed by `cudaGetLastError`). Sprinkle
  between forward stages to localize which kernel left the sticky.

### "My .cu changes aren't taking effect"

`cargo build` uses cargo's incremental cache, which sometimes misses
`.cu` mtime updates inside `OUT_DIR`. Force a rebuild:
```bash
touch src/cuda/your_file.cu src/cuda/flash_attention_fwd.cu
cargo build --release ...
```

If that doesn't work, `cargo clean -p flame-core` and rebuild.

### `full_like` uses `default_dtype`, not source dtype

`Tensor::full_like(value)` goes through `Tensor::full`, which casts the
resulting tensor to `default_dtype()` (BF16 in production). It does **not**
inherit `self.dtype()`. This is the historical contract; changing it would
silently affect several regularization / dropout paths that rely on the
default cast.

If you're building a constant tensor that must match the source's dtype,
**do not use `full_like`**. Build the constant yourself:

```rust
let c = Tensor::from_vec(vec![value], Shape::from_dims(&[1]), device)?
    .to_dtype(source.dtype())?;
```

Then call `maximum` / `minimum` / whichever op needs it — those broadcast
scalar-shaped tensors fine.

**Gotcha fixed 2026-04-20**: `Tensor::clamp` used to call `full_like` for
its min/max constants. With `default_dtype = BF16`, clamping an F32 tensor
would build BF16 constants and `maximum`/`minimum` rejected the dtype
mismatch. `clamp` now builds constants in the source's dtype. Any future
op that needs a "constant like" tensor in the caller's dtype should follow
the same pattern.

### Three paths for silu/gelu (only one is live), plus a fourth for gelu_exact

Historical debris — there are THREE BF16 silu/gelu implementations:
1. `tensor_iterator::ops::unary::silu_bf16_iter / gelu_bf16_iter` (LIVE — the
   TensorIterator path, `.cu` functor in `src/cuda/unary/`). `Tensor::silu` and
   `Tensor::gelu` route here for BF16.
2. `bf16_ops::silu_bf16 / gelu_bf16 / square_bf16` (NVRTC, retained as parity
   oracles for the `_iter` functions — tests under
   `tests/tensor_iterator_{silu,gelu,square}_parity.rs` call these).
3. `cuda_ops_bf16::silu_bf16 / gelu_bf16` → `fc_silu_bf16 / fc_gelu_bf16`
   (build-time, in `cuda/cuda_ops.cu`). Reached only via `cuda_ops.rs`
   dispatch and `attention::sdpa`'s MLP call (`gelu_bf16_into`).

If you edit `cuda/cuda_ops.cu` kernels, `Tensor::silu / Tensor::gelu` will
NOT change. If you edit the NVRTC strings in `bf16_ops.rs`, parity tests
will shift but the live path will NOT change. To edit the live path, touch
the `.cu` functor under `src/cuda/unary/`.

This bit me during the elementwise perf work.

**Two GELU math forms, four paths total (added 2026-05-21).** The default
`Tensor::gelu()` is the **tanh approximation** —
`y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))`. This matches
PyTorch's `F.gelu(x, approximate='tanh')` and is used by Z-Image, Klein,
Chroma, and every other flame-core model trained or ported to date. Don't
change which form `Tensor::gelu` calls — trained LoRAs would silently
invalidate.

For PyTorch parity with bare `nn.GELU()` (default `approximate='none'`,
exact erf), use `Tensor::gelu_exact()` instead. Math:
`y = 0.5 * x * (1 + erff(x/√2))`. Added for Cosmos-Predict2.5
(`inference-flame/src/models/cosmos_predict25_dit.rs::mlp`); kernel in
`src/cuda/unary/gelu_exact.cu`. **Forward-only** — no autograd
registration; do not use in trainers. Fast-path: `bf16_ops::
gelu_exact_bf16_contig_direct`; iterator: `tensor_iterator::ops::unary::
gelu_exact_bf16_iter`. Parity test:
`tests/tensor_iterator_gelu_exact_parity.rs` against PyTorch GPU reference
at `tests/data/gelu_exact_ref.safetensors`. The reference was generated on
CUDA (see `gen_gelu_exact_ref.py`); NEVER regenerate from CPU per
CONTEXT.md.

### `tensor.to_dtype()` has BF16↔F32 fast paths (`!requires_grad` only)

As of 2026-05-12 (commit `1332019` + bugfix `ef0faff`),
`Tensor::to_dtype` has direct-call fast paths for the two hot cast
cases (both contiguous source, **and source does NOT require grad**):
- **BF16 → F32**: allocate F32 output, dispatch
  `bf16_convert::bf16_to_f32_u16` directly.
- **F32 → BF16**: allocate BF16 output, dispatch
  `bf16_convert::f32_to_bf16_u16` directly.

The fast paths skip the F32-staging round-trip the legacy code did
(`alloc_aligned_f32` + `storage.to_f32` + `dtod_copy` + optional
`f32_to_bf16` = 2–3 kernels + 2–3 allocs per cast). All other dtype
combinations (F32→F64, BF16↔F16, etc.) still go through the staging
path — those callers are cold enough that the redundancy is acceptable.

**Autograd guard:** the fast paths return a fresh tensor with
`requires_grad: false` hardcoded — they bypass `Op::Cast` recording.
Without the `!self.requires_grad` gate (the bugfix), any cast inside a
training forward erases the grad marker, the loss ends up
not-requires-grad, and `loss.backward()` fails. The current dispatch
checks `requires_grad` first: tensors that don't need grad get the
17–34× cast speedup; tensors that need grad fall through to the legacy
path (`Tensor::cast` records `Op::Cast` correctly).

If you're adding a new dtype to `DType`, mirror the BF16↔F32 pattern in
`tensor.rs:773` AND preserve the `!requires_grad` guard — the legacy
path is the autograd-aware path.

### RMSNorm has one canonical path — `norm::rms_norm`

Until 2026-05-12 there were two RMSNorm forward paths:
1. `norm::rms_norm` — training entry, `Op::RMSNorm` autograd record.
2. `cuda_ops_bf16::rms_norm_bf16` → `fc_rms_norm_bf16` — inference-only entry.

Commit `d729ede` collapsed them: `cuda_ops_bf16::rms_norm_bf16` is now a thin
wrapper that calls `norm::rms_norm`. Both paths share the same dispatch into
the new `rms_norm_forward_bf16_vec` kernel (13–16× faster than the legacy
scalar path on production shapes). The `fc_rms_norm_bf16` `.cu` kernel
remains as the fallback inside `norm::` when `norm_size % 4 != 0` — but if
you're optimizing the BF16 RMSNorm hot path, **edit `norm.rs`, not
`cuda/cuda_ops.cu`**. Inference and training now move together.

### Vec-kernel env overrides (A/B benchmarking the new vec paths)

The 2026-05-12 perf attack added vectorized BF16 kernels alongside the
legacy paths, with env switches to force the legacy for A/B comparisons.
Production trainers should NOT set these — the vec dispatch is on by
default. Use them only when benching or hunting a numerical regression:

| Env var | What it forces | Source |
|---|---|---|
| `FLAME_RMS_NORM_LEGACY=1` | Scalar 1-thread-per-row RMSNorm fwd + bwd | `src/norm.rs` |
| `FLAME_LAYER_NORM_FWD_LEGACY=1` | Smem-tree LayerNorm forward | `cuda/cuda_ops.cu` |
| `FLAME_LAYER_NORM_LEGACY=1` | Scalar LayerNorm backward (no cross-row kernel) | `cuda/src/flame_norm_bf16.cu` |
| `FLAME_PERMUTE_LEGACY=1` | Scalar grid-strided `permute0213` / `permute021` | `cuda/permute0213.cu` |
| `FLAME_PERMUTE_FASTPATH=0` | Bypass `permute_generic` fast-path dispatcher (rank-2 `[1,0]` + rank-4 `[0,1,3,2]` route to slow scatter kernel instead of `launch_permute10_*` / `launch_permute0132_*`) | `src/cuda_kernels.rs::CudaKernels::permute_fastpath` |
| `FLAME_SWIGLU_LEGACY=1` | Scalar `swiglu_fused_bf16_kernel` (no vec2 pair loads) | `src/bf16_ops.rs` |
| `FLAME_SLICE_COPY_LEGACY=1` | Generic strided slice kernel (no `cudaMemcpyAsync` fast path) | `cuda/bf16_slice_index.cu` |
| `FLAME_GROUP_NORM_STATS_LEGACY=1` | Smem-tree GroupNorm stats kernel | `cuda/cuda_ops.cu` |

All vec kernels require their dispatch precondition (typically a length
divisibility — `norm_size % 4 == 0`, `C % 4 == 0`, `spatial % 4 == 0`,
etc.); shapes that don't qualify fall through to the legacy kernel
automatically. The env vars are an additional manual override.

### SM_86 shared-memory budget — opt in to 100 KB

RTX 3090 / 3090 Ti are `sm_86`. The per-thread-block static shared memory
on sm_86 is 48 KB. To use up to **100 KB dynamic** shared memory per block
(which any nontrivial flash-attention tile layout needs) you must opt in:

```cpp
cudaError_t err = cudaFuncSetAttribute(
    my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    (int)requested_bytes          // must be <= 100 * 1024 on sm_86
);
if (err != cudaSuccess) return (int)err;
my_kernel<<<grid, block, requested_bytes, stream>>>(...);
```

Above 100 KB → `cudaFuncSetAttribute` returns `cudaErrorInvalidValue` and
the launch never happens. Under-budget launches are silently fine.

`src/cuda/flash_attention_fwd.cu` uses this to request 88.5 KB for HD=128.
`sm_89+` (Ada, Hopper) have larger per-block budgets (164 KB / 228 KB) —
if you write a kernel tuned for those, gate the larger layout behind
`__CUDA_ARCH__ >= 890`.

### `cp.async` pipelining pattern (SM_80+)

`src/cuda/flash_attention_fwd.cu` is the reference for the cp.async pattern
in flame-core. The idiom:

```cpp
// Outside extern "C" — templated wait_group cannot have C linkage.
__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* gmem) {
    unsigned smem_int = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N> __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
```

Rules of engagement:

1. **cp.async.cg needs 16-byte alignment** on both shared and global
   pointers. A uint4 vectorized layout enforces this for free.
2. **Always pair a group of issued loads with exactly one `cp_async_commit`**.
   The commit demarcates the group boundary for `wait_group`.
3. **`wait_group(N)` is per-thread**. After it returns, only *this thread*'s
   cp.async writes are visible to *this thread*. For cross-thread visibility,
   **always follow `wait_group` with `__syncthreads()`**.
4. **Group ordering**: PTX `wait_group(N)` forces the OLDEST pending groups
   to complete until ≤N remain. To guarantee a specific group is done,
   issue it FIRST so it's the oldest, then `wait_group(K-1)` where K is
   the total number of groups you've committed since that one. If you only
   care that a specific group is done AND don't mind waiting for everything
   newer, `wait_group(0)` is the simple path.
5. **No masked cp.async**. Out-of-bounds rows in a tile must be handled
   with a separate regular STS zero-store; mixing mask and cp.async in one
   loop is messy but correct.
6. **Overlap opportunities**: the canonical FA2 pattern is to prefetch
   the next KV tile during the current tile's compute. Even without a
   second buffer, you can overlap V's load with softmax because K is dead
   after QK^T and V can reuse the same SMEM slot; the `cp.async V →
   s_KV` issues right after the QK^T `__syncthreads` and the matching
   `wait_group(0) + __syncthreads` sits just before PV.

See `FA2_CP_ASYNC_DESIGN.md` at crate root for the full byte-math rationale
behind the FA2 forward kernel's cp.async pipeline, and the ladder of
attempted optimizations and what broke at each step. Future kernels that
want cp.async should follow the same commit/wait discipline — don't
invent new idioms.

### Shared-memory region reuse (s_K/s_V → s_KV)

When SMEM is tight, look for tiles that are **live in disjoint stages**.
In `flash_attention_fwd.cu` the K tile is only needed for QK^T (stage 1);
V is only needed for PV (stage 3); they never coexist. So instead of two
16 KB slots we have one 16 KB `s_KV` slot that's reloaded between stages.
Costs one extra global-mem barrier per KV iteration, saves 16 KB that
pays for the BQ=64 tile vs BQ=32. The "reusable SMEM region" pattern
comes up elsewhere too (backward kernels can reuse the stashed P/dP/dS
pad) — look for it before cutting tile sizes.

### Alias-casting shared memory: race if the strides differ

Aliasing a `float* s_S` region as a `__nv_bfloat16* s_P` reinterpret_cast
of the *same* bytes is only safe if you never interleave reads/writes
within a stage AND the row strides under the two views cover the same
addresses for each logical row. In `flash_attention_fwd.cu`'s first
attempt, `s_S[qi*BKV*4 bytes]` vs `s_P[qi*BKV*2 bytes]` rows landed at
*different* byte offsets — writes to row N of P corrupted row N/2 of S
in another warp. Always allocate `s_P` as a separate region or prove the
stride equivalence before aliasing.

### `Tensor::softmax` has a fast path

For `dtype == BF16` and `dim == last_dim` and `!requires_grad`, `Tensor::softmax`
dispatches to `bf16_elementwise::softmax_lastdim_bf16` (single fused kernel,
no scratch alloc). For everything else it falls back to the slow 5-step
pipeline `max_dim → sub → exp → sum → div`. If you need fast softmax for a
non-last dim, you'll need to permute first.

### Softmax tail masking: `-INFINITY`, not `0.0f`

When a kernel has to mask invalid tail positions before an online
softmax (ragged sequence lengths, KV-tile boundaries, padding), fill
with `-INFINITY` rather than `0.0f`. Reason: `exp(0 - new_max)` is
small but **nonzero**, so masked cells contribute spurious terms to
the softmax denominator. `exp(-∞ - m) == 0` exactly, and
`max(x, -∞) == x` so the running max stays correct.

The flash-attn forward (`src/cuda/flash_attention_fwd.cu`) got bitten
by this — 0.0f-masked invalid cells in the second K tile inflated
the denominator ~25-30% at Sk=72, under-scaling output ~32%. Fixed
2026-04-19 via `mask_tail_2d_float_neg_inf`; regression test
`tests/sdpa_ragged_sk.rs`. Any new online-softmax kernel should reach
for this helper (or write its own equivalent) — don't reuse
`zero_tail_2d_float`.

### Group norm layout trap

`cuda_ops_bf16::group_norm_bf16` takes **NHWC** (`[N, H, W, C]`), not NCHW.
The functional `group_norm::group_norm` handles either. Document at the call
site whichever you're using or you'll spend 30 minutes debugging "channels
look weird."

### Pre-transposing weights for fused_linear3d (the older one)

`ops::fused_inference::fused_linear3d` (without `_native`) wants
**pre-transposed** `[Cin, Cout]` weight. Klein loads weights and immediately
transposes them via `bf16_elementwise::transpose2d_bf16`. New code should
use `fused_linear3d_native` instead — it takes the standard PyTorch
`[Cout, Cin]` layout and uses cuBLASLt `TRANSA=T` to do the transpose
inside the GEMM (no extra pass over memory).

### Linear layers must not materialize transposed weights

`nn::Linear` stores weight as `[out_features, in_features]` (PyTorch
layout) and uses `ops::gemm_bf16::matmul_bf16_trans(input, weight, false,
true)` — the `trans_b=true` flag routes through cuBLASLt with TRANSB=T, so
the transpose lives inside the GEMM. Do NOT materialize transposed weights
via `transpose2d_bf16` on the forward hot path: that creates a separate
GPU buffer which grows stale after optimizer steps (the fused Adam kernel
writes `self.weight` in place through a raw pointer, bypassing any cache).
Backward follows the same rule — `Op::Linear` backward uses
`matmul_bf16_trans(grad, weight, false, false)` for `grad_input` and
`matmul_bf16_trans(grad, input, true, false)` for `grad_weight`, mirroring
the `Op::BatchMatMul` fast path.

### `.unsqueeze(1)` for broadcasting modulation

In every DiT block forward, the modulation params are `[B, dim]` and need
to broadcast over `[B, N, dim]`. The pattern is:
```rust
let modulated = norm.mul(&scale.add_scalar(1.0)?.unsqueeze(1)?)?
                    .add(&shift.unsqueeze(1)?)?;
```

`unsqueeze(1)` makes the `[B, dim]` into `[B, 1, dim]` which broadcasts over
the seq dim during `.mul` and `.add`.

### Storage match patterns

When you need raw access to the BF16 data inside a `Tensor`:
```rust
let xs = match &x.storage {
    TensorStorage::BF16 { data, .. } => data,
    _ => return Err(Error::InvalidOperation("expects BF16".into())),
};
```

For mutable access:
```rust
let ys = match &mut out.storage {
    TensorStorage::BF16 { data, .. } => data,
    _ => unreachable!(),
};
let ys = ensure_unique_slice(ys)?;
```

`ensure_unique_slice` makes the `Arc<CudaSlice<u16>>` unique (clones the
data if shared) so you can write to it without breaking aliasing. The
helpers `slice_ref(xs)` and `slice_ref(ys)` give you `&CudaSlice<u16>` for
launch params.

### Async vs sync in tests

Most flame-core ops dispatch to async streams. If you're testing and want to
ensure a kernel actually finished before reading data, call
`device.synchronize()` between the launch and the read-back. The
`Tensor::to_vec()` family does this internally.

### Strict mode

When `strict::is_enabled()` (env var `FLAME_STRICT_DTYPE=1` or similar),
flame-core will:
- Fail any op that needs an implicit F32 fallback
- Fail any silent clone (like `.contiguous()` on a non-contiguous tensor
  inside a kernel)
- Fail tape ops that try to record an unsupported op

For inference code, leave it off (the default). For training, turn it on
to catch silent precision drops.

### Training with flame-core autograd — the hard-won lessons

These were discovered during `klein-trainer` development (2026-04-09).

**1. Inference binaries MUST wrap in `AutogradContext::no_grad()`.**
The autograd tape is enabled by default. Any binary that only does forward
passes (e.g. `prepare_dataset`, VAE encode, text encode) will accumulate
saved tensors into the global tape forever and OOM after ~20 samples at
1024². Fix: `let _guard = AutogradContext::no_grad();` at the top of `run()`.

**2. `Tensor::narrow` BF16 fast path silently drops `requires_grad`.**
Fixed 2026-04-09 (flame-core commit `12d1433`). Before the fix, every
`narrow()` on a BF16 tensor went through `cuda_ops_bf16::slice_axis_bf16`
which returned a new tensor with `requires_grad=false` and no `Op::Slice`
recorded. This broke the autograd chain through any fused QKV split,
causing `pred.requires_grad=false` and silent zero gradients. **If your
model forward calls `.narrow()` on a tensor that should carry gradients,
verify this commit is applied.** Post-fix, the BF16 fast path records
`Op::Slice` and propagates `requires_grad`.

**3. `Tensor::bmm` dispatches on `self.dtype()` only — not `other`.**
If `self` is BF16, the BF16 kernel is called even if `other` is F32. The
kernel then fails with `"bmm_bf16_fp32acc_out: tensors must be BF16"`.
Keep Q/K/V in the SAME dtype throughout attention. Don't mix F32 cast
intermediates into a BF16 bmm.

**4. Use `Op::FlashAttention` instead of decomposed attention for training.**
A hand-rolled `bmm → mul_scalar → softmax → bmm` chain records ~12
tape entries per attention block and saves the N×N score matrix (1 GB at
1024², 113 MB at 512²). `Op::FlashAttention` records ONE entry and its
backward handler (`attention_backward_recompute`) recomputes scores from
Q/K/V, saving only the three small input tensors. For Klein 4B (25 blocks)
this reduces tape entries by ~275 and eliminates ~2.8 GB of saved scores.

To use: compute attention under `no_grad`, then record `Op::FlashAttention`
manually:
```rust
let output = {
    let _guard = AutogradContext::no_grad();
    q.bmm(&k.transpose_dims(1,2)?)?.mul_scalar(scale)?.softmax(-1)?.bmm(&v)?
}?;
if q.requires_grad() || k.requires_grad() || v.requires_grad() {
    let mut out = output.requires_grad_(true);
    AutogradContext::record_op(out.id(), Op::FlashAttention {
        query: q.id(), key: k.id(), value: v.id(),
        mask: None, scale, causal: false,
    }, vec![(q.id(), q.clone()), (k.id(), k.clone()), (v.id(), v.clone())]);
}
```

**5. BF16 `clone_result()` is a full GPU memcpy — not cheap.**
Unlike F32 storage (which is `Arc<CudaSlice<f32>>` and clones by ref-bump),
BF16 uses raw `CudaSlice<u16>`. `clone_result()` does
`device.alloc + dtod_copy`. Every `record_op` that saves BF16 tensors
this way allocates new GPU memory. ~300 saves per forward = ~8 GB overhead.
`Tensor::clone()` (the derived Clone) is equally expensive for BF16
because `CudaSlice<u16>::clone` is a deep copy.

**Future fix:** Arc-wrap BF16 storage (the `shared_storage` feature flag
exists but only covers `from_bf16_arena`). Until then, minimize saves:
use `Vec::new()` for saved_tensors in ops whose backward doesn't need them
(e.g. `Op::Slice` only uses `input_shape` from the Op variant, not saved
tensor data).

**6. `requires_grad` "infects" the residual stream.**
After the first LoRA delta is added via `base.add(&lora_delta)`, the
residual tensor has `requires_grad=true`. All subsequent ops on that tensor
record to the tape — even frozen base-weight matmuls. Klein 4B: block 0's
LoRA add infects img → 24 subsequent blocks record everything → ~2700
tape entries instead of ~60. This is standard autograd behavior but without
gradient checkpointing it makes per-step backward ~45 min for DiT models.
**Gradient checkpointing per block is the fix** — see
`gradient_checkpointing.rs` docs.

### Reductions: BF16 inputs use a dedicated single-kernel path

`Tensor::sum` and `Tensor::mean` on BF16 inputs dispatch to
`bf16_reduce::sum_bf16` / `mean_bf16` — a single kernel that reads
BF16, accumulates in F32 in-kernel, and writes BF16 (Foundation fix
#B, 2026-05-12). The legacy BF16→F32 cast + F32-reduce + F32→BF16
cast triple pass is preserved behind `FLAME_BF16_REDUCE_LEGACY=1`.

The mean path fuses the `1/n` divide into the F32→BF16 cast kernel
so the entire reduction stays on a single CUDA stream — do **not**
add a host-side `dtoh_sync_copy` to read the F32 scratch back to
multiply on the host. That serializes the training pipeline (~10%
slowdown observed during initial prototyping). Pass `1/n` as a kernel
arg to `f32_scalar_to_bf16_kernel` instead.

**Known landmine in the F32 sum kernel** (`cuda_kernel_sources.rs:260`):
`sum_kernel` launches with `grid_size = n.div_ceil(256).min(1024)`
and each thread reads exactly one element with no grid-stride loop.
For `n > 256 * 1024 = 262144`, elements past index 262143 are
**silently dropped**. The BF16-native replacement uses a grid-stride
loop so it's correct for any size. Don't trust the F32 sum kernel on
large tensors until it's rewritten.

---

## Build flags

| Feature | What it does | Default |
|---|---|---|
| `cuda` | Enable CUDA backend | on |
| `bf16_u16` | Enable BF16-as-u16 storage | on |
| `cudnn` | Enable cuDNN integration | depends |
| `flash_attn` | Enable external flash-attn FFI shim | off (in-tree wmma is the real path) |
| `autograd_v4` | Enable v4 autograd engine | off |
| `shared_storage` | Arc-wrap BF16 storage for cheap `clone()` (training perf) | off |
| `borrowed_weights` | Enable borrowed-weight tensor variant (BlockOffloader) | off |
| `python` | Build PyO3 bindings | off |
| `capi` | Build C API surface | off |
| `dtype_trace` | Compile in dtype trace prints (slow) | off |
| `legacy_cpu_autograd` | **EXPLICITLY BANNED** — `compile_error!` if set | n/a |

### Runtime env knobs (read at process start)

| Env var | What it does | Default |
|---|---|---|
| `FLAME_ALLOC_POOL` | `0` disables the CUDA alloc pool (production scripts use 0) | pool on |
| `FLAME_ASSERT_GRAD_FLOW` | `1` asserts all leaves received grad at backward (see TRAINER_DIAGNOSTICS) | off |
| `FLAME_LOG_SDPA_BWD` | `1` logs every `try_cudnn_sdpa_backward` bail-out reason | off |
| `FLAME_TRACE_PERMUTE_GENERIC` | `1` enables `permute_generic_trace` (counts `(shape, perm, dtype)` tuples per call) | off |
| `FLAME_MT_SCALE` | `1` enables Phase 2 multi-tensor scale kernel in clip-grad path | off (production grad_norms don't trip clip threshold) |
| `FLAME_TI_CACHE_DISABLE` | `1` disables Phase 1 TensorIterator geometry cache | cache on |
| `FLAME_AUTOGRAD_SAVED_LEGACY` | `1` routes `record_op` through pre-Phase-2 `Vec<(TensorId, Tensor)>` instead of `SmallVec<[SavedRef; 4]>` | SavedRef on |
| `FLAME_F32_ZERO_INIT` | `1` re-enables F32 zero-init on alloc-pool cache miss (Fix #A1 default: uninit, matches PT BFCAllocator) | uninit |
| `FLAME_BF16_REDUCE_LEGACY` | `1` routes `Tensor::sum`/`mean` BF16 through cast-then-F32-reduce-then-cast instead of native BF16-in / F32-accum / BF16-out (Fix #B) | native BF16 on |
| `FLAME_HANDLE_TLS_DISABLE` | `1` disables TLS Cell for cublasLt handle/stream lookups + per-shape GEMM cache in `gemm_bf16_fp32acc.cu` (Fix #C) | TLS on |
| `FLAME_GEMM_BF16_WORKSPACE_BYTES` | Bytes of persistent cuBLASLt workspace per device for BF16 GEMMs (Fix #C; `0` = no workspace) | `0` |
| `FLAME_HOT_FAST_PATH_DISABLE` | `1` disables direct-kernel fast path on `Tensor::silu/gelu/add/mul/mul_scalar` for BF16-contig (Fix #F); falls through to TensorIterator | fast path on |
| `FLAME_PERMUTE_FASTPATH` | `0` disables tuned rank-2 `[1,0]` and rank-4 `[0,1,3,2]` permute kernels (Fix #G); routes to generic scatter | fast path on |
| `FLAME_NO_CUDNN_SDPA_BWD` | `1` forces decomposed SDPA backward (skip `try_cudnn_sdpa_backward`); rollback for Stage 2 cuDNN bwd re-enable | cuDNN on for unmasked BF16 shapes, with auto-padding for non-64 token counts |
| `FLAME_CUDNN_SDPA_BWD_CAUSAL` | `1` opts causal SDPA backward into the cuDNN graph; keep off until per-model probes show it beats the fallback | off |

### Phase 2 `SavedRef` caveat — version counter is in a side table, not in `TensorStorage`

The naive PyTorch port would put `AtomicU32 version` inside the
`TensorStorage` enum variants (so it's shared across all `Tensor::clone()`
siblings via the inner `Arc`). flame-core does **not** do that, because
`TensorStorage::Drop` hand-rolls a `ptr::read` + `Arc::try_unwrap` to return
memory to the alloc pool — wrapping the variants in `Arc<StorageInner<_>>`
breaks that across all 6 variants.

Instead, version counters live in a process-global `RwLock<HashMap<usize,
Arc<AtomicU32>>>` keyed by inner Arc-pointer address (`tensor_storage.rs:1-256`).
Captured `SavedRef`s hold a private `Arc<AtomicU32>` clone so their version
check survives `AutogradContext::clear()`'s flush of the table.

**What this means for you**:
- Any new in-place mutator on a `TensorStorage` variant MUST call
  `storage.bump_version()` after the write. `add_inplace_same_dtype`,
  `evict_block`, and any `as_mut_device_ptr*` write path already do this.
- The table is flushed at `AutogradContext::clear()` — fine for normal
  trainer loops which call clear per step. A long-running test without
  clear could grow the table; bound it manually if needed.

---

## Launch-wrapper pattern (SPEED_CONTRACT clause 1)

Every C-side `flame_*_launch` wrapper that runs inside a training step MUST
avoid per-call `cudaMalloc` / `cudaFree` / `cudaStreamSynchronize` /
`cudaDeviceSynchronize`. See [`SPEED_CONTRACT.md`](./SPEED_CONTRACT.md)
clause 1 for the full rule.

Two correct shapes — pick the one that matches your metadata size:

### 1. Inline kernel-arg struct (small, fixed metadata ≤ 4 KB)

Pack shape / strides / small parameters into a `struct` and pass it **by
value** through CUDA's kernel-argument space. No allocation, no copy, no
sync. Use when rank / size bounds are known and the struct fits.

Reference: `cuda/narrow_strided.cu`.

```cuda
#define FLAME_NARROW_MAX_RANK 8

struct NarrowMeta {
    int64_t shape[FLAME_NARROW_MAX_RANK];
    int64_t strides[FLAME_NARROW_MAX_RANK];
};

extern "C" __global__
void narrow_strided_kernel(
    const uint8_t* src, uint8_t* dst,
    int rank, NarrowMeta meta,   // passed by value
    int dim, int64_t start, int64_t elem_size, int64_t n_elements) { ... }

extern "C" int flame_narrow_strided_launch(...) {
    NarrowMeta meta = {};
    for (int i = 0; i < rank; ++i) {
        meta.shape[i]   = out_shape_host[i];
        meta.strides[i] = src_strides_host[i];
    }
    narrow_strided_kernel<<<blocks, threads, 0, stream>>>(
        ..., rank, meta, ...);
    return (int)cudaGetLastError();
}
```

This pattern fixed Class E (narrow ops, klein 9B 268 `cudaStreamSynchronize`/step → 0).

### 2. Cached per-device workspace (variable / large metadata or scratch)

When you need a real device-side workspace (scratch buffers, multiple
intermediate tensors, anything > a few KB), allocate **once on first call**
into a per-device cache, grow only on shape change, never free on the hot
path. Lifecycle owned by the primitive; freed at process exit.

Reference: `cuda/sdpa_stream_bf16.cu` (11-buffer fused workspace), and
`cuda/gemm_bf16_cublaslt.cu` / `cuda/gemm_bf16_fp32acc.cu` (cuBLASLt
workspace cache).

```cuda
struct SdpaStreamWorkspace {
    int device = -1;
    int q_t = 0, k_b = 0, d = 0;
    void*  base        = nullptr;
    size_t total_bytes = 0;
    float* scores      = nullptr;      // offsets into `base`
    // ... 10 more sub-buffer pointers
};

static SdpaStreamWorkspace* sdpa_ws_acquire(int device, int q_t, int k_b, int d, ...) {
    static std::mutex ws_mutex;
    static std::vector<SdpaStreamWorkspace> ws_cache;
    std::lock_guard<std::mutex> lock(ws_mutex);
    // ... lookup / grow / fixup pointers from base + offsets
}
```

Same shape for the cached cuBLAS handle in `sdpa_ws_get_cublas_handle`
(`cublasCreate` is ~ms; do it once per device, reuse forever).

### Forbidden pattern

"alloc → async copy → launch → sync → free" inside the launch wrapper.
This was the narrow pre-fix shape, the current legacy paths in some `.cu`
files, and is the first thing to look for when investigating a per-step
`cudaStreamSynchronize` count > 8 (PyTorch eager's baseline).

When auditing a `.cu` file, grep the launch wrapper for `cudaMalloc`,
`cudaMemcpyAsync`, `cudaStreamSynchronize`, `cudaDeviceSynchronize`,
`cudaFree`. Hits must be justified per SPEED_CONTRACT clause 1.

---

## Block offloading conventions

FlameSwap is deleted. All block offloading uses `BlockOffloader` with
`prefetch_block`/`await_block` for transfer-compute overlap. No exceptions.

- **`BlockOffloader`** (`flame_core::offload`, moved into flame-core 2026-05-12
  commit `df00c5f`): sole mechanism for both training and inference.
  Double-buffered GPU slots, dedicated transfer stream, pinned CPU storage
  for all block weights. Was at `flame-diffusion/src/block_offload.rs`; now
  imported as `use flame_core::offload::{BlockOffloader, BlockFacilitator, BlockHandle}`.
- **klein-trainer**: `--block-swap` flag triggers BlockOffloader.
- **chroma-trainer**: `--block-swap` flag triggers BlockOffloader.
- **wan-trainer**: 14B+ (dim > 4096) automatically uses BlockOffloader +
  `Wan22Dit::load_shared_only`. 5B preloads all blocks resident.
- **Inference models**: each implements `BlockFacilitator` and creates a
  `BlockOffloader` at load time. Forward loops use prefetch/await pattern.
- **`await_block_handle` over `await_block`**: prefer the scoped-handle API
  on the hot path. `await_block` falls back to host-side
  `cudaDeviceSynchronize` for slot reuse; the handle's Drop records a
  `compute_done` event and the next prefetch does `cudaStreamWaitEvent`
  instead — GPU-side, no host stall.

### Ring allocator semantics (`flame_core::ring_alloc`, Gap 1 Phase 1, 2026-05-14)

- **`RingPtr` is a borrowed view, not RAII.** Dropping a `RingPtr` does
  nothing. The underlying bytes are owned by the `RingAllocator` and
  return to the pool only at `RingAllocator::reset()`. Match the ring's
  scope to a single training step: forward fills, backward drains,
  reset between steps. A `RingPtr` held past a reset is a
  use-after-free.
- **Slabs allocate lazily, never free.** First touch in a slab triggers
  `cudaMalloc`; subsequent steps reuse it. `cuda_malloc_count` equals
  the count of distinct slabs touched, not the count of allocations.
  Verified by `tests/ring_alloc_microbench.rs`.
- **Direction is type-level.** `forward_handle(idx)` returns
  `RingForwardHandle`; `backward_handle(idx)` returns
  `RingBackwardHandle`. There is no bool flag. The two handle types
  borrow the allocator mutably, so the compiler enforces that forward
  and backward allocations don't interleave within the same lexical
  scope (which would break the cursor invariant).
- **`alloc(0)` is an error**, as is `alloc(n)` with `n > slab_bytes`
  (a single allocation cannot span slabs). Forward/backward use
  `ceil_16` / `floor_16` for alignment.
- **Per-allocation reclaim is intentionally absent.** OT's
  `StaticLayerTensorAllocator.deallocate` is a no-op for retiring
  ranges; mirror that. Reclaiming a sub-range would break the
  "cursor walks linearly" invariant that makes overlap structurally
  impossible.

### Ring allocator as pool backend (`flame_core::ring_alloc::pool_adapter`, Gap 1 Phase 2a, 2026-05-14)

Opt-in routing of `cuda_alloc_pool` cache-MISSES through a
`RingAllocator` via the `PoolMissAllocator` trait. Conventions when
using this surface:

- **Install once at trainer init, BEFORE the first `pool_alloc_*` call.**
  Construct `RingAllocator::new(device, num_slabs, slab_bytes)`, wrap in
  `Arc::new(RingPoolAdapter::new(ring))`, then call
  `cuda_alloc_pool::install_miss_allocator(adapter.clone())`. Hold a
  long-lived `Arc<RingPoolAdapter>` for per-step `reset()` calls.
- **Per-step lifecycle is `clear_pool_cache()` THEN `adapter.reset()`,
  in that order.** Place the pair immediately after
  `AutogradContext::clear()` at the step boundary.
  - `clear_pool_cache` drains every bucket free-list entry. Entries
    pointing into ring slabs (`is_external: true`) are reconstructed-
    and-`mem::forget`-ed (no `cudaFree`); the slab `Arc` keeps memory
    alive across the reset.
  - `adapter.reset()` then snaps the ring cursors back to extremes.
  - Reversing the order is undefined: `reset()`-before-clear leaves
    bucket entries pointing at bytes the next forward allocation will
    re-issue.
- **Cross-step tensors MUST NOT pass through `pool_alloc_*` while a ring
  is installed.** Parameters, optimizer state, EMA weights live across
  steps; if they were ring-backed, `reset()` would re-issue their
  bytes. The pool's hot-path callers all allocate transient step-scope
  tensors — verify by grepping for `pool_alloc_*` in the trainer init
  path and ensuring it doesn't fire there.
- **The opt-in is process-global.** `install_miss_allocator` returns
  any previously installed allocator. Production usage = one allocator
  per trainer process. Multi-trainer tests must install/uninstall
  around test boundaries.
- **Failure mode is self-documenting.** If the ring exhausts mid-step
  (working set > `num_slabs * slab_bytes`), the allocator returns
  `Error::OutOfMemory("RingAllocator exhausted...")` which propagates
  out of `pool_alloc_*`. Bump ring size by env or commit constant.
- **Cache-HIT path is unchanged.** Only cache-MISS routes through the
  ring. Steady-state allocations are mostly hits; the ring sees
  per-step new-shape allocations and once they're returned to the
  pool the bucket free list serves the next request.
- **No autograd-direction propagation in 2a.** All ring routes go
  `alloc_forward`. The bidirectional invariant from Phase 1 is unused
  here. Phase 2b can wire `AutogradContext::is_backward()` into the
  adapter for direction-typed allocs.
- **f32 routing is DISABLED in 2a (post-Builder bug-fix, 2026-05-14).**
  The Builder's first commit (`f82ab9b`) routed f32 misses through the
  ring, made safe via a `cudarc-pinctx::install_external_ptr_hook` that
  consulted `external_ptrs` to skip `cudaFree` on ring pointers. The
  Klein 9B 5-step smoke FALSIFIED that approach: backward-graph
  teardown panicked with `CUDA_ERROR_INVALID_VALUE` inside
  `CudaSlice<f32>::drop` at `pool_return_f32`-driven drops
  (`/tmp/klein9b_phase2a_5step.log`). Root cause: derived
  `CudaSlice<f32>` values produced by flame-core's many F32
  intermediaries (autograd_v3 gradient bufs, fused_linear3d_native
  workspaces, broadcast metadata buffers, sliced views) escape the
  `external_ptrs` registration window — their `cu_device_ptr` is an
  offset into the same slab but is NOT in the set. Drop then calls
  `cudaFree` on a mid-slab offset. The fix
  (`pool_adapter.rs::alloc_f32` returns `Err`) realigns the code with
  the Builder's commit-message intent. u16 routing stays ON because
  all `pool_alloc_u16` callers wrap the slice in
  `TensorStorage::BF16 { data: Arc<CudaSlice<u16>> }` and
  `TensorStorage::Drop` routes through `pool_return_u16`. No derived
  `CudaSlice<u16>` path exists.

### External-ptr entries in `cuda_alloc_pool` (Phase 2 post-reboot, 2026-05-13)

`FreeEntry::is_external = true` indicates the entry's backing bytes are
owned by an external allocator (currently `RingAllocator` slabs handed
out by `BlockOffloader::alloc_bf16_via_ring`). The contract:

- **External entries NEVER enter the active free list.** Both
  `push_f32` and `push_u16` carry a guard immediately before
  `list.push(entry)`: if `entry.is_external`, call
  `reconstruct_and_forget::<T>(entry.ptr, entry.len, entry.device)`
  (mirror is destroyed without `cudaFree`; slab `Arc` retains
  ownership), then `unregister_external_ptr(entry.ptr)`, then `return`.
  Caching them would alias: the same bytes are handed out by the next
  `ring_alloc::forward_handle(0).alloc(...)`, so a subsequent
  `try_pop` would silently re-use live ring memory and corrupt
  training.
- **Mirror-layout invariant must hold across reconstruct + forget.**
  `reconstruct_and_forget::<T>` rebuilds a `CudaSlice<T>` via the
  pinned cudarc 0.11.x mirror layout (see `cuda_alloc_pool.rs` top
  comment). Same layout as `ring_alloc/pool_adapter::synth_slice`.
- **`unregister_external_ptr` MUST follow every `is_external`
  reconstruct.** Otherwise the `external_ptrs` refcount map grows
  unbounded across steps and tagged-pointer lookups slow down.
- **The bypass path also fires when the pool is inactive or the
  bucket cap is exceeded.** Pre-existing `is_external` branches at
  the top of `push_*` already use `reconstruct_and_forget`; the new
  Phase 2 guard is the *successful* path's variant of the same logic.
- **`install_external_ptr_hook` is wired by `BlockOffloader::ensure_ring`.**
  The hook (`ring_external_ptr_hook` in `src/offload/mod.rs`) routes
  `cudarc::driver::CudaSlice::drop` through `global_pool().is_external_ptr`
  so ring-slab offsets that escape into autograd and drop outside
  `pool_return_u16` skip `cudaFree` (which would otherwise panic with
  `CUDA_ERROR_INVALID_VALUE`). The hook is a function pointer atomically
  swapped — installing it from multiple `BlockOffloader` instances or in
  combination with `install_miss_allocator` is idempotent.
- **`external_ptrs` is a `HashMap<u64, u32>` refcount, not a `HashSet`**
  (2026-05-14 Phase 2 round-2 fix). The `RingAllocator` cyclically
  reuses slab offsets — when the forward cursor wraps, the same
  `device_ptr` may be handed out for a new allocation while a prior
  tensor with the same ptr is still alive. A `HashSet` made
  registration idempotent: the first drop unregistered the ptr and the
  second drop saw `is_external_ptr=false`, tagged its `FreeEntry`
  non-external, and `clear_cache` later called `free_async` on a
  ring-slab offset → `CUDA_ERROR_INVALID_VALUE` panic at step 0 of
  Klein 9B `--offload`. The refcount keeps the ptr marked external
  until ALL live tensors sharing it have been forgotten. Regression
  test: `cuda_alloc_pool::tests::test_external_ptr_refcount_under_ring_wrap`.
- **`FLAME_POOL_CLEAR_DEBUG=1`** routes `clear_cache` through a
  per-entry `eprintln!` + `catch_unwind` slow path that logs every
  drop's `(ptr,bucket,is_u16,tagged_ext,hook_ext)` and survives a
  panic in any single drop. Used to localize the round-2 ring-wrap
  bug. Leave off for production runs.

---

## Quick "where do I X" reference

| I want to... | Look at |
|---|---|
| Add a fused inference kernel | `src/cuda/fused_*.cu` + `src/cuda/ffi.rs` + `src/ops/fused_inference.rs` |
| Add a BF16 elementwise op | `src/bf16_elementwise.rs` (flat path) + `src/bf16_ops.rs` (single-arg) |
| Add a build-time `.cu` file | Create file → add to `build.rs` `cuda_sources.push(...)` → declare in `src/cuda/ffi.rs` or `src/cuda_ops_ffi.rs` |
| Change the SDPA kernel | `src/cuda/flash_attention_fwd.cu` (wmma path) |
| Change RMSNorm | `src/norm.rs` NVRTC kernels (`RMS_NORM_FWD_KERNEL_BF16_VEC` / `_BWD_KERNEL_BF16_VEC` / `RMS_NORM_GRAD_WEIGHT_KERNEL_BF16` — vec=4 path, both training + inference). `cuda/cuda_ops.cu::rms_norm_kernel` is now the legacy `norm_size % 4 != 0` fallback. |
| Change LayerNorm | Forward: `cuda/cuda_ops.cu::layer_norm_forward_bf16_vec_kernel` (legacy smem-tree fallback in same file). Backward: `cuda/src/flame_norm_bf16.cu::layer_norm_backward_bf16_vec_kernel` + `layer_norm_grad_weight_bias_bf16_vec_kernel`. |
| Change cuBLASLt linear | `src/cuda/fused_linear3d.cu` |
| Add a new diffusion model | `inference-flame/src/models/your_model.rs` — flame-core just provides primitives |
| Save/load safetensors | `serialization::load_file_filtered / save_file` |
| Get the global device | `flame_core::global_cuda_device()` |
| Disable autograd (inference) | `let _guard = AutogradContext::no_grad();` |
| Record a fused attention Op | See "Training with autograd" §4 above |
| Train a LoRA on a DiT | `klein-trainer/` — reference impl with all the gotchas applied |
| Get a cublasLt handle | `cuda::device_lt::cublaslt_handle_ptr(device)?` |
| Get a stream pointer | `cuda::device_lt::stream_ptr(device)?` |
| Force-rebuild after .cu edit | `touch the_file.cu` then `cargo build --release ...` |
| Run a test binary | `cargo run --bin minimal_test --release` |

---

## Bug fix: i32-to-f32 parameter passing to CUDA kernels

**Fixed 2026-04-12.** Files: `conv3d_simple.rs`, `cuda_kernels_gpu.rs`,
`cuda_kernels.rs`, `cuda/kernels.rs`, `cuda_tensor.rs`.

Several `alloc_from_pool_and_copy()` helpers passed `i32` dimension/shape
data to CUDA kernels by casting `x as f32` (numeric conversion). Kernels
that declared `int*` parameters then reinterpreted the IEEE 754 float bits
as integers — e.g. `3i32 → 3.0f32 (0x40400000) → int 1077936128`.

**Fix:** Use `f32::from_bits(x as u32)` to bit-preserve the integer value.
The CUDA kernel reads the correct int via `__float_as_int()` or direct
`int*` reinterpret.

**Two patterns exist** — know which your kernel uses before choosing:

1. Kernel declares `int* dims` (pointer reinterpret) → use `f32::from_bits(x as u32)`
   (bit-preserving). Example: `conv3d_simple.rs`.
2. Kernel declares `float* dims_f32` and casts `(int)dims_f32[i]` → use `x as f32`
   (numeric). Example: `cuda_kernels_gpu.rs`, `cuda_kernel_sources.rs`.

Using the wrong pattern: pattern 1 with numeric cast reads garbage ints;
pattern 2 with bit-cast reads denormalized floats that truncate to 0.

**Rule:** Always check the kernel source before choosing. Grep for
`int\*.*dims` vs `float\*.*dims` in the kernel declaration.

---

### QwenImage-specific conventions (2026-04-14 parity audit)

Three bugs found and fixed in `qwenimage-trainer/src/model.rs` during
parity testing against musubi-tuner. All three are "silent wrong answer"
failures — the code ran without errors but produced wrong results.

**7. Sinusoidal timestep embedding requires `scale=1000`.**
The QwenImage model's `QwenTimestepProjEmbeddings` uses
`Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)`.
The `scale=1000` multiplies the timestep BEFORE computing sin/cos. Without
it, the sinusoidal frequencies see values in [0,1] instead of [0,1000] — a
1000× error in the conditioning signal. Reference: `qwen_image_model.py:256`.
The inference-flame implementation (`qwenimage_dit.rs:242`) already had
this correct: `t_f32.mul_scalar(1000.0)`.

**8. Text RoPE offset with `scale_rope=True` must divide by 2.**
When `scale_rope=True` (QwenImage default), image positions use symmetric
centering `[-H/2, H/2)`. The text position offset must match:
`max_vid_index = max(height // 2, width // 2)` (Python `qwen_image_model.py:342`).
The Rust code was using `max(height, width)` without dividing by 2, placing
text tokens at 2× the correct offset. This shifts all text-image cross-
attention RoPE relationships.

**9. `AdaLayerNormContinuous` output chunk order is `(scale, shift)`.**
Python (`qwen_image_model.py:547`): `scale, shift = torch.chunk(emb, 2, dim=1)`.
Scale is first, shift is second. The Rust `norm_out` code had them swapped.
This caused cosine similarity to drop from 0.9999 to 0.62 on the final
prediction — catastrophic but only affecting the output layer. Per-block
modulation uses a *different* chunk order (`shift, scale, gate`) and was
already correct.

**Parity test location:** `qwenimage-trainer/src/bin/parity_test.rs` (forward)
and `qwenimage-trainer/src/bin/train_parity_test.rs` (20-step training).
Run with `python tools/dump_forward.py` first, then the Rust binary.

**Timestep sampling:** QwenImage uses `qwen_shift` mode:
`sigmoid(randn * sigmoid_scale)` pushed through the shift formula. This
is NOT uniform sampling. See `hv_train_network.py:1192-1201`.

**Loss precision:** Cast prediction and target to F32 BEFORE computing
squared difference. BF16 squared errors can overflow for large residuals.

## Optimizer step — fused kernels only

All optimizer steps must use a fused CUDA kernel. No scalar-ops fallbacks
in optimizer code — the previous `step_scalar_ops` path in `src/adam.rs`
allocated ~14 full-size tensors per param per step (mul_scalar, add,
mul_scalar, add, div_scalar, div_scalar, sqrt, add_scalar, div,
mul_scalar, and optional decoupled-wd mul/add), which is allocator churn
at full-tune or F32-embedding-heavy scale. `src/adam.rs` dispatches to
four NVRTC kernels covering `{BF16, F32}` params × `{BF16, F32}` grads.

Unsupported dtype combinations return `Err(Error::InvalidInput(...))` —
conversion is the trainer's responsibility. Never add a silent-cast
fallback to a scalar-op chain. If you need a new dtype (F16, I8, …),
add a dedicated fused kernel, not a fallback.

The `adam_fused` module must preserve the decoupled weight decay
(Loshchilov & Hutter 2017) bug-prevention receipt verbatim at the top of
each kernel source. The `m = β₁m + (1-β₁)g / v = β₂v + (1-β₂)g² / p -=
lr·m̂/(√v̂+ε) / p -= lr·wd·p` shape is load-bearing. Folding `wd` into
`grad` before the moment updates collapses the Adam step to
`~sign(param)` for freshly-initialized LoRA_A matrices (whose B partner
is zero) and unlearns them at uniform `lr·sign(p)` per step. That bug
destroyed Klein 4B LoRA_A training in April 2026 — do not reintroduce it.

### Optimizer kernel placement: `<optimizer>_kernel.rs`, not `bf16_*.rs`

NVRTC kernels for OPTIMIZERS go in their own top-level file,
e.g. `src/adam8bit_kernel.rs`, NOT in `bf16_ops.rs` / `bf16_elementwise.rs`
/ `bf16_reduce.rs`. The `bf16_*` naming convention is reserved for
forward-pass inference primitives — pointwise, fused, single dtype-class
in/out. Optimizer kernels:
- Touch multiple dtype classes per launch (F32 master param + F32 or BF16
  grad + U8 quant state + F32 absmax scales). The `bf16_*` files assume a
  BF16-dominant dtype contract.
- Mutate persistent optimizer-owned state (`m_codes`, `v_codes`, `absmax`
  buffers) that lives outside the autograd graph and outside the
  `cuda_alloc_pool` — not a fit for the inference-primitive contract.
- Are launched once per param per step from `Optimizer::step`, not
  per-forward — different cache / launch-storm profile.

Pattern: one file per optimizer kernel + its Rust launcher, named
`<optimizer>_kernel.rs`. Re-export the public surface (`pub use
cuda_impl::*` under the `cuda + bf16_u16` feature gate). Examples:
`src/adam.rs` (the multi-variant AdamW family — historical, slightly
larger surface) and `src/adam8bit_kernel.rs` (the bnb 8-bit AdamW kernel,
added 2026-05-17 — the canonical shape for any new optimizer kernel).

### Multi-tensor reductions: launch all, sync once

Trainers MUST NOT compute global L2 norm with the per-tensor
`.to_vec()?[0]` pattern — that forces N D2H syncs per step. Klein 9B LoRA
hits ~200 syncs each iteration that way. Use
`flame_core::ops::grad_norm::global_l2_norm` (or `_with_scale`) which
returns a 1-element FP32 device tensor; the caller chooses when (if ever)
to `.item()` it for host-side logging.

Pattern:

```rust
let grad_refs: Vec<&Tensor> = params
    .iter()
    .filter_map(|p| grads.get(p.id()))
    .collect();
let total_norm = flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?
    .item()? as f32;     // ← exactly one D2H sync per step
let scale = if total_norm > clip {clip / total_norm} else {1.0};
for p in &params { /* mul_scalar(scale) and set_grad */ }
```

The 8 EriDiffusion-v2 `train_*.rs` binaries all use this pattern as of
the Fusion Sprint Phase 5 migration. New trainers must adopt it from
inception.

**Phase 3 (2026-05-12, launch-storm refactor):** `global_l2_norm` now
has an Apex-style multi-tensor fast path. When every grad is F32 +
contiguous, it dispatches to
`ops::multi_tensor::multi_tensor_l2_norm_sq_f32` — two-stage reduction
kernel, 3 launches total instead of 2N+(N-1)+1. Verified on zimage
rank=8 LoRA: `sum_kernel` + `add_kernel` per step dropped from 561+561
to 1.3+1.3. Env override `FLAME_MT_L2NORM=0` falls back to the legacy
per-tensor fold. BF16 grads still go through the legacy path
(F32-only fast path; BF16 falls through with on-the-fly cast).

### Multi-tensor primitives: foreach-style is the right pattern

When a trainer or library function iterates "do op X on each of N
parameters / gradients", `flame-core` already has the infrastructure to
collapse N launches into 1. Two locations to check first:

- `adam::adam_fused_multi_tensor_step` (in `adam.rs:858`) — used by
  Adam optimizer. The new `param_is_bf16: bool` discriminator (Phase 1,
  2026-05-12) routes between BF16 and F32 multi-tensor kernels.
- `ops::multi_tensor` (new module, Phase 3, 2026-05-12) — generalized
  packed-buffer + kernel-loader pattern. Currently has L2 norm and
  in-place scale (Phase 2, default-off in callers — see below).

The packed-buffer pattern: build `Vec<u64>` of pointers + sizes (extract
each pointer via `g.as_mut_device_ptr_f32(...)` for F32 or
`g.as_mut_device_ptr_bf16(...)` for BF16 — the helper returns `u64`
directly so callers don't need to depend on cudarc just to call
`device_ptr_mut`). Drop the `&mut` borrow per iteration (the u64
doesn't carry a Rust lifetime), single H2D copy of the packed buffer,
one kernel launch with grid = n_tensors. See `adam.rs:1093-1230`,
`ops/multi_tensor.rs::multi_tensor_l2_norm_sq_f32`, and
`ops/multi_tensor.rs::multi_tensor_scale_inplace_packed` for canonical
implementations.

**Trainer-side wiring of `multi_tensor_scale_inplace_packed` (Phase 2,
2026-05-12, default-off):** the trainer's clip-grad path used to do
`for param in &params { g_scaled = g.mul_scalar(scale)?; param.set_grad(g_scaled); }`
— N launches per step when `total_norm > CLIP_GRAD_NORM`. The new path
extracts pointers via `g.as_mut_device_ptr_f32(...)` from
`grads.get_mut(param.id())` (which releases the `&mut Tensor` between
iterations because the u64 escapes the borrow), builds
`packed = [ptrs(n) | sizes(n)]`, and calls
`multi_tensor_scale_inplace_packed` once. Default off because zimage
and klein production grad-norms hover at 0.004–0.17 and never trip
the `> 1.0` clip; enable per-trainer via `FLAME_MT_SCALE=1` for
configs where the path fires. See
`EriDiffusion-v2/HANDOFF_2026-05-12_PHASE2_SCALE_FOLLOWUP.md`.

### cudarc launch path: context-pin skips `cuCtxSetCurrent` (Phase 4, 2026-05-12)

The workspace uses a vendored cudarc 0.11.9 (at `../cudarc-pinctx`,
wired via `[patch.crates-io] cudarc = { path = "../cudarc-pinctx" }`
in flame-core/Cargo.toml AND EriDiffusion-v2/Cargo.toml). The single
patch: `src/driver/safe/threading.rs` caches the per-thread bound CUDA
context in a thread-local. `bind_to_thread()` short-circuits when the
requested context is already current on this thread.

Upstream cudarc 0.11.9 calls `bind_to_thread()` from every safe launch
and every alloc/free. Each call is a `cuCtxSetCurrent` driver call
(~106 ns). On zimage training that was 91,637 redundant calls per step
(~9 ms/step of pure CPU stall). After the patch: 0 calls per step
(absent from top-15 driver APIs), driver_api/step dropped from 200,472
to 88,903 (−55.7%).

Implications for new code:

- Don't add `Device::bind_to_thread()` calls in flame-core — they were
  redundant before the patch and remain a no-op after. cudarc's safe
  wrappers already do this.
- If you add a new thread that uses CUDA (currently none in flame-core
  or EriDiffusion-v2), call `bind_to_thread()` once on the new thread.
  Subsequent calls hit the fast path.
- Multi-device usage still works correctly — the thread-local cache
  flips between contexts on each switch. Single-device single-thread
  training (the universal case) hits the fast path on every call after
  the first.

If you ever clone the workspace fresh, `../cudarc-pinctx` must exist as
a sibling of `flame-core/` and `EriDiffusion-v2/`. See
[CodeAlexx/cudarc-pinctx](https://github.com/CodeAlexx/cudarc-pinctx)
for the upstream parity diff and full setup.

---

## Conv3d — cuDNN-first dispatch, im2vol as fallback (2026-04)

`Conv3dBF16::forward()` tries `cudnn::cudnn_conv3d_bf16` first and falls
back to the legacy im2vol+GEMM path only when cuDNN refuses. That's
because the im2vol path materializes a `col_rows × col_cols` BF16
columns tensor explicitly, and at LTX-2.3 LatentUpsampler shapes
(`[1, 1024, 33, 14, 24]`, 3³ kernel) that's ~60 GB. It OOMs every GPU
short of an H100. cuDNN picks implicit-GEMM / Winograd and never
materializes the columns — the whole upsampler runs in ~1.6 s with
<100 MB of workspace.

### Env knobs
- `FLAME_CUDNN_CONV3D_WS_LIMIT_MB` — cuDNN workspace ceiling in MiB.
  Default 256. Raise if you hit algo-refused-by-workspace; lower if
  you're near the VRAM ceiling elsewhere.
- `FLAME_CUDNN_CONV3D_STRICT=1` — fail fast with the cuDNN error
  instead of falling back to im2vol. Useful during parity verification
  so you never silently slip onto the legacy path.
- `BF16_CONV_DEBUG=1` — prints to stderr when the cuDNN path bails,
  including the error and the shape that failed.

### Gotcha: grouped Conv3d only via cuDNN
The legacy im2vol+GEMM path has never supported `groups ≠ 1`. The new
cuDNN path does, and the dispatcher returns a clean `Unsupported`
error when grouped Conv3d is called without cuDNN available. If you
add grouped Conv3d to a model and disable cuDNN, you'll trip this.

### Gotcha: bias is not fused
Today we apply bias via a separate `cudnnAddTensor`. Fusing into
`cudnnConvolutionBiasActivationForward` is a follow-up — don't assume
bias is a free add.

---

## SDPA — auto-stream on large shapes (2026-04)

`sdpa::forward_bf16` now auto-routes to the streaming kernel
(`cuda_ops_bf16::sdpa_stream_bf16`, which uses online softmax + tiled
Q/K/V and never materializes scores) whenever
`B * H * Q * K > FLAME_SDPA_STREAM_THRESHOLD` (default `2_000_000_000`
elements). The materialized fallback still handles small shapes because
it's slightly faster when the peak is well under a GB.

### Why the threshold exists
The materialized path allocates an FP32 `[B, H, Q, K]` scores tensor plus
a BF16 copy plus temporary softmax state, peaking near 8 bytes/element.
For LTX-2.3 stage-2 self-attention at 768×448 (B=1, H=30, Q=K=11088)
that's 3.68 G elements ≈ 29 GB — OOMs every 24 GB card. Streaming kernel
peaks at the tile scratch (~hundreds of MB), not GB.

### Env knobs
- `FLAME_SDPA_STREAM_THRESHOLD=<elements>` — override the auto-stream
  threshold (`B*H*Q*K` element count). Set lower if you want streaming
  earlier, higher if your card has enough VRAM to eat the materialized
  peak at intermediate shapes.
- `FLAME_SDPA_FORCE_STREAM=1` — unconditional stream path for ANY
  shape. Pre-existing env; the auto-threshold is an additive layer.
- `FLAME_SDPA_CHUNK_MAX=<q_chunk>` — per-Q chunk size used by the
  stream kernel. Default 2048. Lower it if you're tight on workspace.

### Mask handling on the stream path
`sdpa_stream_bf16` already accepts `mask: Option<&Tensor>`. The
auto-stream path passes the mask through untouched — no caller change
required. Common causal / padding masks work; custom attention biases
that needed the materialized-scores path still fit because the stream
applies the mask tile-by-tile.

### What does NOT fix this
- Faster GEMM for QK^T alone (the scores tensor is the OOM).
- Pool trimming (only helps cached-free memory).
- Chunking the sampler step (unchanged token count per call).
The stream kernel is the fix; the threshold dispatcher routes to it
automatically.

---

## cuDNN SDPA backward — padding-mask alignment + saved-O lookup (2026-05-12, updated 2026-05-20)

cuDNN's flash-bwd graph still expects the physical Q/KV dimensions to be
64-aligned. Launching odd sequence lengths directly can return
`CUDA_ERROR_MISALIGNED_ADDRESS`. The training dispatcher handles that for
normal unmasked BF16 attention by padding Q/K/V to the next 64-token
boundary before cuDNN forward, saving Stats/O for the padded shape, and
passing the real Q/KV lengths to cuDNN forward/backward as padding-mask
sequence-length tensors. The public output is sliced back to the real Q
length.

This is the default for unmasked BF16 SDPA with head dim 64/96/128 and is
intended for every model, not only HiDream-O1. Arbitrary binary masks and
attention biases still route through the compatibility fallback because the
current cuDNN wrapper only wires causal and padding-mask semantics, not an
arbitrary additive mask tensor.

If a model's mask is actually structured, expose the structure instead of
materializing `[B, H, Q, K]`. HiDream-O1's mixed mask is represented by
`attention::sdpa_prefix_causal_full`: causal prefix rows plus full suffix rows.
The default implementation is an O1-safe single-op hybrid: forward uses the
smaller prefix-causal pass plus a suffix all-ones masked pass, and backward
recomputes once against the exact prefix-causal/full mask via
`Op::PrefixCausalFullAttention`. This avoids both the older full `[S,S]`
materialized mask on forward and the two-SDPA shared-K/V gradient collapse.
The fully unmasked suffix diagnostic path and the old structured full-then-
slice path are not production defaults; do not promote either without
`tests/sdpa_prefix_causal_full_grad.rs` coverage and finite end-to-end
training proof. Do not reintroduce a cuDNN additive-bias route for binary masks
without both SDPA parity tests and finite end-to-end training proof; a previous
attempt produced non-finite LoRA gradients.

Causal cuDNN forward/backward is guarded by `FLAME_CUDNN_SDPA_BWD_CAUSAL=1`.
It is correct enough to run the O1 probe without the old crash, but it was
slower than the fallback in a 10-step O1 measurement, so do not make it the
default until a model-specific speed probe proves it wins.

### ⚠️ Hidden bug class: saved-tensor id-exclusion is broken

`fetch_saved(entry, id)` at `autograd.rs:2090-2115` materializes non-contig
views via `.contiguous()`, which produces a **fresh `TensorId`**. So any
code that does:

```rust
saved_tensors.iter().find(|t| t.id != some_other_id)
```

is **always wrong** — the fresh-clone id will never match the recorded
saved-Q/K/V id, so the exclusion clause is a no-op and the first
shape-match wins. This is exactly how the cuDNN SDPA bwd `grad_norm=inf`
bug fired pre-Stage 2 (saved-O was always returning saved-Q because they
share `[B,H,Nq,D]` shape in self-attention).

**Always use the saved-by-id pattern**: at forward record time, store the
specific `TensorId` you'll need in the `Op` enum (e.g.
`Op::FlashAttention { output: Option<TensorId>, stats: Option<TensorId>, ..}`).
At backward, call `fetch_saved(entry, &recorded_id)` directly. Never
shape-find with id-exclusion.

The `Op::FlashAttention` struct (`autograd.rs:332-348`) is the canonical
example post-fix.

---

## Stride hazards in kernel paths (the Bug #4-6 family)

flame-core has THREE chokepoints for "raw-pointer kernel reads a
saved/passed tensor" — each must materialize non-contiguous views, or
the kernel reads the parent storage from offset 0 with the view's
smaller logical shape and produces wrong values.

### Chokepoint 1: backward saves — `fetch_saved` in `autograd.rs`

The closure `fetch_saved` defined inside `compute_gradients`
(`src/autograd.rs:1855-1894`) wraps `entry.get_saved` /
`CHECKPOINT_MANAGER.fetch_saved` and **materializes any non-contiguous
result**. Backward kernels (rms_norm_backward, sdpa_bwd, mul backward,
silu backward, etc.) read saved tensors via raw `tensor_raw_ptr` /
`storage_ref().try_as_slice_*` — those ignore custom_strides /
view_offset.

**Rule**: backward arms in `compute_gradients` use `fetch_saved` for
any saved tensor whose data is read by a kernel. **Anti-pattern**:
direct `entry.get_saved(...)` on a tensor-then-pass-to-kernel —
bypasses the chokepoint. Only use `entry.get_saved(...)` when you
need just `shape()` / `dims()` / `dtype()`, never the data.

The ops that USE direct `entry.get_saved` for shape-only purposes are
fine (Cat backward, Reshape backward, Permute backward — they only
read shape). Audit any new backward that reads via raw pointer.

### Chokepoint 2: forward `clone_result` — `tensor.rs`

`Tensor::clone_result()` (`src/tensor.rs:2913-2935`) is the canonical
fallible deep-clone. For non-contiguous inputs it materializes via
`.contiguous()` first, then duplicates the storage. The materialize
step uses `materialize_view` / `permute_generic` which walks the
strided source correctly.

The pre-fix bug: `clone_result` copied `*numel` (parent storage size)
elements via `dtod_copy(slice_ref(data), new_data)` and labeled the
output with `self.shape` (the view's smaller shape). For a transposed
[16,8] view, this stored parent's data in [16,8] row-major order but
labeled it as [8,16] — `out[i,j]` resolved to `parent[i*16+j]` rather
than the correct `parent[j*8+i]`.

**Rule**: this is now safe to call on any tensor including views.
Don't add new `dtod_copy(parent.data, ...)` paths that bypass the
contiguity check.

### Chokepoint 3: kernel-launch boundaries reading via `try_as_slice_*`

Functions that launch CUDA kernels with `tensor.storage.try_as_slice_f32()?`
or `try_as_slice_u16()?` read from the storage's offset 0 — ignoring
`view_offset`. They MUST contiguify non-contiguous inputs first.
Already done in:

- `src/cuda_kernels.rs:413-490` (`CudaKernels::add`)
- `src/cuda_kernels.rs:474-540` (`CudaKernels::mul`)
- `src/cuda_kernels.rs:2010-2030` (`CudaKernels::div`)
- `src/ops/elt.rs:84-137` (`add_same_dtype`)
- `src/ops/elt.rs:193-235` (`mul_same_dtype`)

Audit unaudited sites:

```bash
grep -rn "storage.try_as_slice\|storage_ref().try_as_slice" \
  /home/alex/EriDiffusion/flame-core/src/ | grep -v test
```

For each site, ask:
1. Does the caller guarantee contiguity? (e.g., it's just been
   produced by a kernel that always emits contig output, or it's a
   freshly-allocated `Tensor::empty_dtype`.)
2. If not, add `if !t.is_contiguous() { t = t.contiguous()? }` at
   the top of the function.

The conv2d / norm / sgd / activation_offload paths mostly fall under
case 1 (operating on freshly-emitted kernel outputs), but explicit
audit is cheap insurance.

### What `is_contiguous()` actually checks

```rust
pub fn is_contiguous(&self) -> bool {
    self.custom_strides.is_none() && self.view_offset == 0
}
```

So `narrow()` at start=0 STILL returns false (because narrow sets
custom_strides to the parent's strides). Any contiguity check must
honor this or it'll produce silent garbage on offset-0 narrows.


---

## PyTorch Parity Tests

**Source of truth**: PyTorch (BSD-3) is the oracle. Fixtures are produced
by `scripts/generate_pytorch_fixtures.py` and live under
`tests/pytorch_fixtures/<category>/<op>/<shape_name>.safetensors`.
Every fixture stores `input*` + `output` tensors generated with PyTorch's
op on the same device/dtype the Rust test will use.

**Smoke test entry point**: `tests/parity_smoke.rs` (Fusion Sprint Phase 0).
SHA256-pins the smoke fixture so silent regeneration / corruption is caught
at test load.

**Comparator**: `tests/parity_helpers/mod.rs::compare_tensor`
(promotes both tensors to FP32 host-side, then per-element atol/rtol).
On mismatch, prints the top-K largest absolute deltas with their flat
indices — never just "FAILED".

**Tolerance defaults** (from `parity_helpers`):

| Path | atol | rtol | Used for |
|-----:|:----:|:----:|----------|
| BF16 element-wise | 1e-2 | 1e-2 | unary/binary ops, fused norms, fused softmax, attention output |
| FP32 reductions   | 1e-5 | 1e-5 | norm stats, optimizer moments, grad clip, loss reductions |

**Helpers**:
- `parity_helpers::compare_bf16(got, expected)` — applies BF16 defaults.
- `parity_helpers::compare_fp32_reduction(got, expected)` — applies FP32 defaults.
- `parity_helpers::assert_parity_bf16(name, got, expected)` — assert + pretty-print on fail.
- `parity_helpers::sha256_file(path)` — pin a fixture's bytes.

**Rule**: every new fused kernel ships with a parity fixture. PRs that
add `flame-core/cuda/*.cu` MUST include a paired Python generator under
`scripts/generate_*_fixture.py` and a `tests/parity_*.rs` Rust test
that loads the fixture and calls `compare_bf16` (or `compare_fp32_reduction`
for reduction outputs). No exceptions.

**Anti-pattern**: comparing flame-core against itself. The comparator is
designed for `(got: flame_core, expected: pytorch_fixture)`. Never wire
it to a flame-core-vs-flame-core regression test — that's a tautology.

## Autograd v2 (Phase 3a) conventions

These came up while writing the recording surface and the 5 P0 ops.
They apply to every Phase 3+ op that lands afterwards.

### Phase 3a op forward wrappers record only when at least one input is tracked

Every Phase 3+ `*_v2` forward wrapper follows this skeleton:

```rust
pub fn add_v2(a: &Tensor, b: &Tensor, ctx: &DispatchCtx) -> Result<Tensor> {
    let out = a.add(b)?;                           // use the existing flame-core forward
    if needs_grad(&[a, b]) {                       // gating predicate from recording.rs
        let grad_fn = AddGradFn::new(a, b);
        let recorded = record_v2(grad_fn, vec![out], ctx);
        Ok(recorded.into_iter().next().unwrap())
    } else {
        Ok(out)                                    // inference path: zero overhead
    }
}
```

Rule: **never call `record_v2` unconditionally.** Inference traffic
(`requires_grad=false` everywhere) MUST pay zero v2 overhead. The
`needs_grad` check is the gate; it's cheap (one mutex lock per input
in the worst case, returning false early on the common no-meta
tensors).

### DO NOT override `GradFn::hooks()` unless your op carries hooks

`GradFn::hooks()` has a default impl returning `Hooks::empty_ref()` —
the engine's no-hook fast path is a `std::ptr::eq` comparison against
this sentinel that skips the empty for-loops entirely. If your op
overrides `hooks()` to return a non-sentinel reference (e.g. an empty
`Hooks` struct embedded on the op), every backward step pays the
empty-loop iteration overhead. Carry an `Option<Hooks>` slot on the
op and only override `hooks()` when populated, if you must support
per-op hook registration.

### `Tensor::clone()` preserves `autograd_meta`; `detach_v2()` drops it

Per PyTorch semantics. `clone()` (derive-generated) clones the
`Option<Arc<Mutex<AutogradMetaV2>>>` field — both handles share
metadata, gradients flow into the same slot. `detach_v2()` allocates
a fresh handle with `autograd_meta = None`. Do not write code that
expects `clone()` to drop history; use `detach_v2()` for that
explicitly.

### `Tensor::autograd_meta` field is `cfg(feature = "autograd_v2")`-gated

The default flame-core build (no `autograd_v2` feature) does NOT have
the field. Every internal Tensor constructor that uses the struct
literal syntax cites the field under
`#[cfg(feature = "autograd_v2")]`:

```rust
Tensor {
    storage, shape, device, id, requires_grad,
    custom_strides, view_offset,
    #[cfg(feature = "autograd_v2")]
    autograd_meta: None,
}
```

96 such call sites exist across `tensor.rs` and 19 other files (BF16
ops, conv2d, norm, etc.). A new Tensor constructor MUST add the
cfg-gated field, otherwise the `autograd_v2` feature build breaks.
The default build doesn't see the field at all and ignores the line.

### Save tensors via `SavedTensor::save_named(t, "OpName:slot")`

The named form attaches the op name + slot to the saved tensor so
that the version-mismatch error message tells you which op failed
and which input was mutated:

```
autograd_v2: saved tensor version mismatch in MatMulGradFn:b:
expected v0, got v1
```

Use the literal `"OpName:slot"` form (e.g. `"MulGradFn:a"`,
`"MatMulGradFn:b"`). Phase 3+ ops will pile up — readable error
messages save real debugging time.

## Autograd v2 (Phase 4a) — Parameter dtype policy

### `Parameter::new(t)` vs `Parameter::new_v2(t)`

Phase 4a added a `GradDtypePolicy` field on every `Parameter`:

- `Parameter::new(t)` → `CastToF32` (v1/v3 default). `set_grad` casts
  every incoming grad to F32, regardless of param dtype. This is the
  invariant existing trainers (Klein, Z-Image, Chroma today) rely on.
- `Parameter::new_v2(t)` → `MatchParamDtype` (autograd v2). `set_grad`
  preserves the incoming grad's dtype. BF16 params keep BF16 grads
  end-to-end through the optimizer, halving gradient memory and
  routing through the previously-dead BF16-grad Adam kernels.

The two policies are NOT interchangeable mid-run: optimizer state (m,
v) is keyed by `TensorId`, but the per-step kernel selected by the
classifier depends on `(param_dtype, grad_dtype)`. Switching policy
mid-training would mix two kernels' numerical conventions on the same
m/v slots. Set policy at construction time; don't flip later.

### `param.grad()` vs `param.grad_bf16_or_f32()`

`grad()` is the v1 accessor and remains intact for compatibility.
`grad_bf16_or_f32()` is the same function with a different name —
introduced so v2 callers explicitly signal "I'm prepared for a BF16
grad to come out". Both return the native-dtype grad under
`MatchParamDtype` and an F32 grad under `CastToF32`.

### Adam classifier dispatch matrix (Phase 4a)

`Adam::step` dispatches on `(param_dtype, grad_dtype)`:

| param dtype | grad dtype | multi-tensor kernel                            | per-param fallback                     |
|-------------|------------|------------------------------------------------|----------------------------------------|
| BF16        | F32        | `adam_fused_multi_bf16_f32grad_kernel`         | `adam_fused_step` (BF16-grad F32 path) |
| BF16        | BF16       | `adam_fused_multi_bf16_bf16grad_kernel` (4a)   | `adam_fused_step` (BF16-grad BF16 path)|
| F32         | F32        | `adam_fused_multi_f32param_f32grad_kernel`     | `adam_fused_step_f32` (F32 path)       |
| F32         | BF16       | (no kernel — falls through)                    | `adam_fused_step_f32` → `adam_fused_f32param_bf16grad_kernel` (4a) |

(F32, BF16) has no multi-tensor kernel today; trainers that need it
should keep their param packs homogeneous (all (BF16, BF16) or all
(F32, F32)) for the multi-tensor fast path.

### `multi_tensor_l2_norm_sq_bf16` vs `_f32`

Both live in `src/ops/multi_tensor.rs`. The BF16 variant has its own
stage-1 kernel (BF16 load → F32 opmath → F32 partial); both share the
F32 stage-2 reducer. `global_l2_norm` in `src/ops/grad_norm.rs`
auto-routes BF16-only-contiguous slices through the BF16 fast path
under the same `FLAME_MT_L2NORM=0` env gate as the F32 path.

Mixed-dtype slices (some BF16, some F32) fall through to the legacy
per-tensor loop. If you see this in a hot path, fix it upstream
(make the trainer pick one dtype) rather than adding a third kernel.

## Autograd v2 (Phase 3c2) — forward-mode AD conventions

### `Tensor::set_fw_grad` implicitly allocates `autograd_meta`

If you call `set_fw_grad` on a tensor that has no `autograd_meta`
(common — leaf tensors built by `Tensor::from_vec` start with
`autograd_meta = None`), Phase 3c2 silently allocates a fresh
`AutogradMetaV2::leaf_no_grad()`. PyTorch parity: setting `_fw_grad`
enables forward-mode-AD tracking on a previously-untracked tensor.

**What this does NOT do:** it does NOT enable backward-mode tracking.
`requires_grad` stays `false` on the auto-allocated meta. If you want
both gates, install the meta yourself first via
`new_meta_ref(AutogradMetaV2::leaf_requires_grad())` and then call
`set_fw_grad`.

### `fw_grad` and `requires_grad` are orthogonal

A tensor can have any combination of `{fw_grad set, requires_grad
true}`. The v2 op forward wrappers gate the two paths separately:

- `needs_grad(&[a, b, ...])` → controls backward-mode recording
  (`record_v2` install).
- `any_fw_grad(&[a, b, ...])` → controls forward-mode JVP install
  on the output (`Tensor::set_fw_grad`).

A `mul` whose only input with a tangent does NOT have requires_grad
still installs a JVP on the output. A `mul` whose inputs are
require_grad-true but have no `fw_grad` still records backward-mode
but does not install a JVP. Both flow paths are independent.

### Tangent zero-materialisation: `tangent_or_zero`

For multi-input ops (`mul`, `matmul`, `add`), the JVP convention is
"no tangent ⇒ zero tangent". The helper
`autograd_v2::ops::fw_mode::tangent_or_zero(t)` returns
`t.fw_grad().unwrap_or(zeros_like_with_dtype(t.dtype()))`. Downstream
formulas (e.g. `a_dot @ b + a @ b_dot`) can always operate on real
tensors. The zeros allocation is cheap on small shapes but a future
micro-opt can short-circuit the zero-side term — not done today.

### `transpose` / `permute` forward-mode tangents call `.contiguous()`

Same discipline as the backward formulas: a strided view returned by
`Tensor::transpose` / `Tensor::permute` would scramble any downstream
gemm/bmm consumer that reads via `try_as_slice_*` and ignores
strides. The Phase 3c2 forward-mode AD path materialises contiguous
before storing in `out.fw_grad`. See HAZARD-2026-05-13-1 +
gemm-stride-ignore conventions above for the underlying class.

### `layer_norm_v2` forward-mode AD SHIPPED (Phase 5a)

The LN JVP deferred from Phase 3c2 ships in Phase 5a. Calling
`layer_norm_v2` with a `fw_grad` on any of `x`/`weight`/`bias`
installs the per-row JVP on the output. Computed in F32 internally
through `mean` / `var` / `rstd` / `rstd_fw` (the BF16 stats kernel
returns only `out` and recomputes the intermediates per call).
Final cast back to the primal's BF16 dtype before storage.

Formula (verified bit-equal to `torch.autograd.functional.jvp` in F64;
see `tests/fixtures/gen_v2_parity.py` for the parity check):

```text
centered    = x - mean(x, normalized_axes)
centered_fw = x_fw - mean(x_fw, normalized_axes)
var_fw      = (2/N) * sum(centered * centered_fw, normalized_axes)
rstd_fw     = -0.5 * rstd^3 * var_fw
out_fw      = centered_fw * rstd * w
            + centered    * rstd_fw * w
            + x_hat       * rstd * w_fw
            + b_fw
```

The design-review handoff (`AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`
§Phase 5 Deliverable B) wrote a per-element `d_rstd_dx` shorthand. The
kernel-truthful form requires the per-row sum reduction shown above
(variance derivative collapses to a per-row scalar, not per-element).
This is documented in the source.

## Autograd v2 (Phase 5a) — PyTorch parity fixtures

These conventions apply to any new v2 op shipping after Phase 5a.

### Parity fixtures live under `tests/fixtures/`, regenerated by `gen_v2_parity.py`

For each new v2 op, add a generator function in
`tests/fixtures/gen_v2_parity.py` and a test in
`tests/autograd_v2_parity.rs`. Two fixture files per op:

- `<op>_backward.safetensors`: `input_0` … `input_N` (op inputs in
  argument order), `grad_output` (always all-ones), `grad_input_0` …
  `grad_input_N` (PyTorch reference grads).
- `<op>_jvp.safetensors`: `input_0` … `input_N`, `tangent_0` …
  `tangent_N`, `out_fw` (`torch.autograd.functional.jvp` reference).

`torch.manual_seed(42)` is set once at module load — do not re-seed
per op. Keep shapes small (≤512 elements) so each fixture file stays
under ~5 KB. Use `dtype=torch.bfloat16` only for ops whose v2 surface
is BF16-only (today: just `layer_norm`).

### Fixtures are version-controlled, NOT regenerated at test time

Commit the `.safetensors` outputs alongside the generator script. The
Rust tests run without Python; the fixtures are bytes-on-disk
ground-truth. Only re-run the generator when:

- a new op is added (and only that op's fixture lands in the commit),
- a shape changes (rare; bumps fixture size, document why),
- PyTorch upgrades to a major version that changes default numerics
  for the op.

### Tolerance discipline by op family

`tests/autograd_v2_parity.rs` defines three tolerance helpers:

- `tol_f32_tight` (`min_cos=0.99999`, `max_abs_ratio=1e-4`) — view
  ops + simple-arithmetic F32 (add, mul, sum-trivial, silu in some
  configs). PyTorch and flame-core should agree to ~rounding here.
- `tol_f32_loose` (`min_cos=0.9999`, `max_abs_ratio=5e-3`) — matmul
  + ops that go through cuBLAS strided-batch or accumulating
  reductions. cuBLAS uses different tile/split orderings than
  PyTorch's CPU reference; per-element error can hit 1e-3 ratio.
- `tol_bf16_ln` (`min_cos=0.999`, `max_abs_ratio=5e-2`) — layer_norm
  specifically. The BF16 stats kernel compounds noise through per-row
  mean / rstd / dweight row-sums.

When adding a new op, pick the band that matches the op's numerical
profile. **Don't widen `tol_bf16_ln` to make a new BF16 op pass** —
investigate. Same for `tol_f32_loose` on a non-matmul op: a divergence
above 5e-3 in F32 is almost certainly a real bug, not a tolerance
issue.

## Autograd v2 (Phase 5b) — trainer migration guide

The v2 stack is **strictly additive over v3**. Default behavior is unchanged.
Opt into v2 surfaces at the granularity you want.

### When to use which entry point

| Want | Use | Result |
|---|---|---|
| F32 grads everywhere (v3 status quo) | `Parameter::new(...)` + `loss.backward()` | Zero change from pre-v2 trainers |
| BF16 model, F32 grads (memory-safe mixed) | `Parameter::new(...)` with BF16 weights | `set_grad` upcasts to F32, Adam runs `(BF16, F32)` arm |
| F32 model, BF16 grads (rare; some mixed-precision setups) | `Parameter::new_v2(...)` + `loss.backward_v2()` | Routes through `adam_fused_f32param_bf16grad_kernel` |
| BF16 model, BF16 grads end-to-end (Class A) | `Parameter::new_v2(...)` + `loss.backward_v2()` + `AdamWV2` | `adam_fused_multi_bf16_bf16grad_kernel` fires |
| Per-tensor F32 compute when needed | `tensor.to_dtype(DType::F32)?` works in any path | One-shot upcast |
| F32 loss reduction on a BF16 model | `loss.to_dtype(DType::F32)?.backward()` | Upcasts loss before backward; grads are F32 |

The 4-way Adam classifier at `src/adam.rs:1107-1144` picks the right kernel
for **any** `(param_dtype, grad_dtype)` combination — no all-or-nothing decision.

### Per-parameter mix-and-match

```rust
let lora_a = Parameter::new_v2(weights_a);       // BF16 grads on this
let critical_param = Parameter::new(weights_b);  // F32 grads on this
```

The Adam classifier dispatches on each param's grad dtype independently.

### Orthogonal axes

Three independent dtype knobs:

| Axis | Default | v2 override | Set via |
|---|---|---|---|
| Parameter storage dtype | inferred from input | inferred from input | `Tensor::to_dtype` before `Parameter::new_v2` |
| Gradient storage dtype | F32 (v3) | matches param dtype | `Parameter::new` vs `Parameter::new_v2` |
| Optimizer moment dtype (m, v) | F32 regardless | F32 regardless (BF16 supported) | `flame_core::config::set_optimizer_moment_dtype` |
| GradientMap policy | `InternalFP32_PublicBF16` | `MatchInsertedDtype` | `GradientMap::new` vs `new_v2` |

### Phase 5b limitations (carried into Phase 5c)

1. **End-to-end BF16 needs BOTH the bridge AND `Parameter::new_v2`**.
   `loss.backward_v2()` alone produces BF16 grads in the GradientMap, but
   the trainer's call to `param.set_grad(g)` under `Parameter::new` (default
   `CastToF32` policy) upcasts the grad back to F32 before AdamW. Migrate
   LoRA params (or any params you want BF16 grads on) to `Parameter::new_v2`.

2. **`FLAME_CUDA_GRAPH=1 + --use-autograd-v2` is unsupported**. The replay
   path pre-allocates grad buffers at the warmup-recorded F32 dtype, bypassing
   the post-loop cast. Bench / training with both flags simultaneously =
   undefined behavior. Disable cuda-graph if using `--use-autograd-v2`.

3. **`FLAME_MT_SCALE` (multi-tensor scale fast path) asserts F32 grads**.
   The trainer's `--use-autograd-v2` flag disables this fast path automatically;
   if you build a custom trainer that uses both, gate `MT_SCALE` off when v2
   is on.

4. **`AUTOGRAD_CONTEXT` is process-global** (`src/autograd.rs:56`). Tests
   that touch v3 backward (including the bridge) cannot run in parallel
   with each other without state contamination. The bridge tests are
   consolidated into one driver as a band-aid; the architectural fix is
   to adopt `serial_test` or a global test-mutex for v3-backward-using tests.

### Migration recipe for a typical trainer

```rust
// Before (v3):
let mut params = vec![Parameter::new(weight_init)];
let mut opt = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
// ... training loop ...
let mut grads = loss.backward()?;
for p in &mut params { p.set_grad(grads.get(p.id())?.clone()); }
opt.step(&mut params);

// After (v2, full BF16 end-to-end):
#[cfg(feature = "autograd_v2")]
let mut params = vec![Parameter::new_v2(weight_init)];        // ← changed
let mut opt = AdamWV2::new(0.001, 0.9, 0.999, 1e-8, 0.01);    // ← changed
// ... training loop ...
let mut grads = AutogradContext::backward_v2(&loss)?;          // ← changed
for p in &mut params {
    p.set_grad(grads.get(p.id())?.clone());                    // ← unchanged
}
opt.step(&mut params);                                         // ← unchanged
```

Three call-sites changed. `set_grad` and `opt.step` interfaces are unchanged
(the 4-way Adam dispatch + the dtype-preserving `set_grad` handle the
difference internally).

### Validating the bridge against existing PyTorch fixtures

When migrating a trainer or audit, the cheapest sanity check is to adapt
its existing v3 backward-parity tests to also exercise `backward_v2()`
against the **same** PyTorch fixtures. `tests/autograd_v2_klein_parity.rs`
demonstrates the pattern: six Klein component scenarios (rms_norm,
apply_rope, attn_chain, …) reuse the v3 fixtures in
`tests/pytorch_fixtures/patterns/klein_ext_*.safetensors` and a 7th
reconstructs a small Klein double-block from `klein_block_backward.safetensors`
(qkv linear → narrow chunks → reshape/permute → SDPA → permute back →
out linear → gated residual → MLP (up linear → silu*chunk → down linear)
→ gated residual). All scenarios consolidate into a single `#[test]`
driver to dodge the `AUTOGRAD_CONTEXT` parallel-mode race (CONCERN #3).
Tolerance `min_cos=0.99 / max_abs_ratio=5e-2` matches the v3 Klein gate;
the bridge lands at max abs_ratio 0.0093 across the chain (well inside
the band). Use `flame_core::parity::ParityHarness` for the actual
compare so the `dx`/`dweight` keys match the fixture naming convention.

## `ring_alloc` — wrap is only legal when the opposite cursor is idle

`flame_core::ring_alloc::RingAllocator` is a bidirectional ring allocator
(Phase 1 — see `src/ring_alloc/mod.rs` and `docs/RING_ALLOC_DESIGN.md`).
Forward allocations grow `allocation_end`; backward allocations retreat
`allocation_start`. When either cursor walks off the end of the slab list,
the underlying OT algorithm cyclically wraps.

**The convention.** Cyclic wrap is permitted ONLY when the OPPOSITE
direction has not fired yet:

| Direction | Wrap permitted when |
|---|---|
| forward (slab N-1 → slab 0)  | `allocation_start == total_bytes` (no backward state yet) |
| backward (slab 0 → slab N-1) | `allocation_end == 0` (no forward state yet) |

If the opposite cursor is non-idle, a wrap attempt returns `Error::OutOfMemory`
with `"RingAllocator exhausted: <fwd|bwd> wrap with <bwd|fwd> active would
lap live allocations"`. This is stricter than OT (which trusts caller
sizing and would silently wrap). It catches the failure mode where forward
wraps post-backward-fire and lands at `(slab 0, intra 0)` — silently
overlapping a still-live early forward allocation, since the linear-regime
`new_end > allocation_start` check trivially passes when `new_end` is small.

**Why this matters.** Phase 2 callers (BlockOffloader) treat the ring as a
per-step working set. A silent wrap-lap corrupts step-N forward results
mid-backward — exactly the failure mode the ring exists to prevent. The
strict-wrap discipline makes "too small a ring" loud at the alloc site
instead of silent at use site. Test coverage:
- `forward_wrap_with_backward_active_does_not_lap_silently`
- `backward_wrap_with_forward_active_does_not_lap_silently`
- the wrap-when-only-forward-active legal path is exercised by
  `slab_boundary_stress_advances_slab_idx` (existing Test 4).

## `static_slab_v2` — every BF16 alloc must go through `pool_alloc_u16`

R1a–R2c shipped a transient-scope slab allocator
(`src/static_slab_v2.rs`) that dispatches under
`FLAME_USE_STATIC_SLAB=1` + active `StepSlabGuard`. The slab eliminates
the BF16 use-after-free bug class structurally: one `cudaMalloc` per
device lifetime, bump-cursor per-step, strict-reset at guard drop,
never `cudaFree` during training.

For this to be correct, **every BF16 allocation that produces a
training-step tensor must enter through `pool_alloc_u16`** — not
through raw `device.alloc::<u16>` / `device.alloc_zeros::<u16>`.
Raw cudart allocs bypass the slab dispatch AND the trap range table,
so any view-pointer validation against them sees a stale exact-pointer
`Freed` event from a prior lifetime and the trap fires false-positive
on `Tensor::cat`. This was the actual root cause of the first R2c
Klein 9B failure (RMSNorm bypass at `norm.rs::rms_norm_forward_bf16`).

| ✅ Correct | ❌ Wrong |
|---|---|
| `pool_alloc_u16(device, n)` | `unsafe { device.alloc::<u16>(n) }` |
| `Tensor::zeros_dtype(_, DType::BF16, _)` | `device.alloc_zeros::<u16>(n)` |

Exceptions: pre-step staging buffers (allocate, write, drop within one
op, never escape into autograd) and the slab's own zero-size short-
circuit (`alloc_u16(0)`).

**Sweep before merging a new BF16 kernel.** Grep
`device\.alloc::<u16>|device\.alloc_zeros::<u16>` in the file you
touched. If the alloc produces a tensor that flows into autograd, you
must route through the pool.

## `StepSlabGuard` — placement and lifetime

When wiring a trainer to the slab path, the guard MUST be the FIRST
allocation-creating local in the per-step body. Rust's reverse drop
order means the guard drops LAST, after step-local tensors. Drop-
order accidents = `live_count != 0` at guard drop = strict-reset
panic.

```rust
for step in start_step..args.steps {
    let _slab_step = StepSlabGuard::enter(device.clone())?;
    // batch load, forward, loss, backward, optimizer step,
    // AutogradContext::clear() — ALL transient allocations here.
}
// validation / sampling lives OUTSIDE the guard scope.
```

Two `r2b_wiring_lint` tests in `train_klein.rs` enforce the pattern
at build time. The same shape applies to any future trainer.

### Persistent state must pre-warm BEFORE the slab env-flag activates

Optimizer state (Adam `m`/`v`, custom optimizers' state HashMaps) is
allocated lazily on first `opt.step` call. If that allocation happens
INSIDE the slab guard scope, the persistent state survives the step
boundary in the slab's range → `live_count` permanently leaked → all
subsequent `reset()`s error.

Fix: pre-warm before activating the slab. Trainer-side:
```rust
// (a) Build params + load checkpoint state
opt.ensure_state_initialized(&params)?;
// (b) NOW activate slab dispatch
std::env::set_var("FLAME_USE_STATIC_SLAB", "1");
// (c) Run the step loop
```

Locked down by `r2b_bf1_persistent_state_in_guard_scope_panics`
(`#[should_panic]`) + `r2b_bf3_adam_prewarm_allows_step_inside_guard`
in `flame-core/tests/r2a_adversarial.rs`.

### Don't `Send` the guard across threads

`StepSlabGuard` has no `!Send` marker today (a follow-up). The thread-
local active flag is per-thread by construction. Moving the guard via
`thread::spawn(move || drop(guard))` strands the source thread's
active flag at `Some(..)` — every subsequent `enter()` on that thread
returns `Err("nested guards forbidden")`. Locked down by
`sk_send_across_threads_corrupts_enter_thread_local`.

### Drop-during-unwind is panic-suppressed

`StepSlabGuard::drop` calls `slab.reset()`. If `live_count != 0`, the
strict reset returns `Err` and the guard normally panics with the
count. BUT — if the thread is already unwinding (`thread::panicking()
== true`), the panic is suppressed and logged instead. This avoids
the double-panic-during-unwind abort that would mask the original
panic. So a panic mid-step propagates cleanly; the slab leaks are
the secondary signal.
