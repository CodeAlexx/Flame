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
| Inference, BF16 in/out, elementwise op on shape-equal tensors | `bf16_elementwise::*_bf16` (auto-routed by `Tensor::*` for matching shapes) |
| Inference, BF16 unary op (silu/gelu/sqrt/etc) | `bf16_ops::*_bf16` |
| Inference, BF16 matmul | `Tensor::matmul` (auto-routes) or `ops::fused_inference::fused_linear3d_native` for the cuBLASLt+bias path |
| Inference, RMSNorm/LayerNorm | `cuda_ops_bf16::rms_norm_bf16 / layer_norm_bf16` (the `Tensor::softmax` BF16 fast path is automatic) |
| Inference, attention | `attention::sdpa` |
| Inference, RoPE | `bf16_ops::rope_fused_bf16` (interleaved-pair) or `bf16_ops::rope_halfsplit_bf16` (Z-Image format) |
| Inference, FFN gate-residual | `bf16_ops::gate_residual_fused_bf16` and `bf16_ops::swiglu_fused_bf16` |
| Training F32 tensor add | falls through to `cuda_ops::GpuOps::add` |
| Need autograd | use the `Tensor::*` methods (they record on the tape); the bare BF16 functions DO NOT record |

### The autograd recording trap

`bf16_elementwise::add_bf16(a, b)` does NOT record on the autograd tape.
`Tensor::add(&b)` DOES (when `requires_grad` is true).

So:
- Inference code can call the bare functions directly — slightly faster, no
  tape overhead.
- Training code must call `Tensor::add(&b)` (or `tensor_a + tensor_b` if
  the operator overload is in scope).

The fused inference primitives in `ops::fused_inference::*` also do NOT
record. They're designed for inference, not training.

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

---

## Adding a new BF16 op — template

### Single-arg unary (like silu, gelu)

In `src/bf16_ops.rs`:

```rust
const CUDA_MY_OP: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void my_op_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                        __nv_bfloat16* __restrict__ Y,
                        long n) {
    long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n2 = n >> 1;
    if (i2 < n2) {
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X);
        __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(Y);
        float2 v = __bfloat1622float2(x2[i2]);
        // ... your math on v.x and v.y ...
        y2[i2] = __floats2bfloat162_rn(result.x, result.y);
    }
    if (i2 == n2 && (n & 1)) {
        long last = n - 1;
        // ... scalar path for the tail ...
    }
}
"#;

pub fn my_op_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let data = unsafe { x.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc my_op: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: data.into(), numel: n },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };
    ensure(&x.device, "my_op_bf16_kernel", CUDA_MY_OP)?;
    let f = x.device
        .get_func("my_op_bf16_kernel", "my_op_bf16_kernel")
        .ok_or_else(|| Error::Cuda("my_op_bf16_kernel missing".into()))?;
    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("my_op_bf16 expects BF16".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;
    unsafe {
        f.launch(lc_pairs(n), (slice_ref(xs), ys, n as i64))?;
    }
    Ok(out)
}
```

Use `lc_pairs(n)` for the vectorized kernel. Use `lc(n)` if you went with a
scalar 1-element-per-thread kernel.

### Two-input elementwise (like add, mul)

In `src/bf16_elementwise.rs`. Add a new flat-path kernel to `CUDA_ADD_MUL_BF16_FLAT`,
add a `pub fn` that uses `shapes_equal_no_broadcast` to dispatch:

```rust
pub fn my_binary_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if shapes_equal_no_broadcast(a, b) {
        return launch_bf16_flat(a, b, "my_binary_bf16_flat_kernel", "my_binary_bf16");
    }
    // fall back to the broadcast path
    launch_bf16_elementwise(a, b, "my_binary_bf16_kernel", "my_binary_bf16")
}
```

Add the broadcast-path kernel to `CUDA_ADD_MUL_BF16` (the slow generic 8-D
broadcast path).

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

### "My .cu changes aren't taking effect"

`cargo build` uses cargo's incremental cache, which sometimes misses
`.cu` mtime updates inside `OUT_DIR`. Force a rebuild:
```bash
touch src/cuda/your_file.cu src/cuda/flash_attention_fwd.cu
cargo build --release ...
```

If that doesn't work, `cargo clean -p flame-core` and rebuild.

### Two paths for silu/gelu

There are TWO BF16 silu/gelu implementations:
1. `bf16_ops::silu_bf16 / gelu_bf16` (NVRTC, in `src/bf16_ops.rs`)
2. `cuda_ops_bf16::silu_bf16 / gelu_bf16` → `fc_silu_bf16 / fc_gelu_bf16`
   (build-time, in `cuda/cuda_ops.cu`)

`Tensor::silu()` and `Tensor::gelu()` route to **#1** (the `bf16_ops` NVRTC
path). If you edit `cuda/cuda_ops.cu`'s silu/gelu kernels, NOTHING in
`Tensor::silu` will change. If you want to fix the live silu/gelu, edit
`bf16_ops.rs`.

This bit me during the elementwise perf work.

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

---

## Block offloading conventions

FlameSwap is deleted. All block offloading uses `BlockOffloader` with
`prefetch_block`/`await_block` for transfer-compute overlap. No exceptions.

- **`BlockOffloader`** (`flame-diffusion::block_offload`): sole mechanism for
  both training and inference. Double-buffered GPU slots, dedicated transfer
  stream, pinned CPU storage for all block weights.
- **klein-trainer**: `--block-swap` flag triggers BlockOffloader.
- **chroma-trainer**: `--block-swap` flag triggers BlockOffloader.
- **wan-trainer**: 14B+ (dim > 4096) automatically uses BlockOffloader +
  `Wan22Dit::load_shared_only`. 5B preloads all blocks resident.
- **Inference models**: each implements `BlockFacilitator` and creates a
  `BlockOffloader` at load time. Forward loops use prefetch/await pattern.

---

## Quick "where do I X" reference

| I want to... | Look at |
|---|---|
| Add a fused inference kernel | `src/cuda/fused_*.cu` + `src/cuda/ffi.rs` + `src/ops/fused_inference.rs` |
| Add a BF16 elementwise op | `src/bf16_elementwise.rs` (flat path) + `src/bf16_ops.rs` (single-arg) |
| Add a build-time `.cu` file | Create file → add to `build.rs` `cuda_sources.push(...)` → declare in `src/cuda/ffi.rs` or `src/cuda_ops_ffi.rs` |
| Change the SDPA kernel | `src/cuda/flash_attention_fwd.cu` (wmma path) |
| Change RMSNorm | `cuda/cuda_ops.cu::rms_norm_kernel` (the live one) |
| Change LayerNorm | `cuda/cuda_ops.cu::layer_norm_forward_bf16_kernel` |
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
