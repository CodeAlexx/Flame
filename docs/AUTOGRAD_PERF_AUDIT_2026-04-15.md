# Autograd Performance Audit — Can We Make It Faster?

**Date**: 2026-04-15
**Scope**: Read-only source audit of `src/autograd.rs` (3781 lines, canonical engine), `src/gradient.rs`, `src/autograd_v4/`, `src/autograd_v3.rs`
**Focus**: Backward pass throughput — kernel launch overhead, allocation waste, fusion opportunities
**Prior audits**: See `FLAME_READONLY_AUDIT_2026-04-13.md` (correctness + safety), `FLAME_OVERHEAD_AUDIT.md` (dispatch overhead)

---

## What's Already Fast (Don't Touch)

The autograd engine has significant optimizations already in place. These are working well:

| Optimization | Location | Impact |
|---|---|---|
| `SmallVec<[(TensorId, Tensor); 3]>` for saved tensors | `autograd.rs:36` | Eliminates Vec heap alloc per recorded op (~2660/step) |
| `AUTOGRAD_ENABLED` atomic fast-path | `autograd.rs:42,637` | Skips mutex lock when autograd disabled (checkpoint forward, inference) |
| Vec-based `GradientMap` with `CompactIndex` | `gradient.rs:17-46,79` | O(1) gradient lookup, no hashing during backward |
| In-place gradient accumulation | `gradient.rs:219` | `add_inplace_same_dtype` — no temporary tensor for accumulation |
| Lazy F32 upcast in accumulate | `gradient.rs:207-213` | First grad stored as-is, cast deferred until second arrives |
| `matmul_bf16_trans` for matmul backward | `autograd.rs:1658-1760` | cuBLASLt TRANSA/TRANSB flags — 0 transposes materialized |
| Fused SiLU backward kernel | `autograd.rs:1824-1878` | Single CUDA kernel via `flame_silu_backward_bf16` |
| Fused SwiGLU backward kernel | `autograd.rs:2191-2261` | Single CUDA kernel via `flame_swiglu_backward_bf16` |
| Fused RoPE backward | `autograd.rs:2263-2279` | Reuses `rope_halfsplit_bf16` with negated sin |
| Checkpoint forward disables autograd | `autograd.rs:1294-1310` | Avoids hundreds of wasted GPU memcpys per block |
| CUDA graph capture/replay for backward | `autograd.rs:978-1029` | Entire backward as single graph launch on repeat steps |
| `ensure_bf16` is now a no-op | `autograd.rs:3759` | Removed wasteful BF16↔F32 round-trip per gradient |
| Checkpoint offload (level 2) | `autograd.rs:1343-1492` | CPU offload of saved tensors, eliminates recompute overhead |
| Eager-free sub-backward in checkpoint | `autograd.rs:2062-2079` | Saved tensors freed per-entry, not held for full block |

---

## High Impact — Fuse These Backward Kernels

### 1. GELU Backward: 12 kernel launches → 1

**Location**: `autograd.rs:1794-1821`

**Current implementation** (decomposed into 12 `GpuOps` calls):
```
mul → mul → add → mul_scalar → tanh → add_scalar → mul → mul →
add_scalar → mul_scalar → mul → add → mul_scalar → mul
```

Each launch has ~5-10μs host overhead plus intermediate tensor allocations. For models using GELU (SDXL, some DiT variants), this fires for every GELU in every block.

**Proposed fix**: Write `flame_gelu_backward_bf16` / `_f32` fused CUDA kernels, matching the pattern of the existing SiLU backward (`autograd.rs:1832-1845`).

```cuda
// Single kernel, zero intermediates:
__global__ void gelu_backward_bf16(const void* dout, const void* x,
                                    void* dx, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = __bfloat162float(((const __nv_bfloat16*)x)[i]);
    float doi = __bfloat162float(((const __nv_bfloat16*)dout)[i]);
    float k = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
    float tanh_k = tanhf(k);
    float dk = 0.7978845608f * (1.0f + 3.0f * 0.044715f * xi * xi);
    float deriv = 0.5f * (1.0f + tanh_k) + 0.5f * xi * (1.0f - tanh_k * tanh_k) * dk;
    ((__nv_bfloat16*)dx)[i] = __float2bfloat16(doi * deriv);
}
```

**Expected savings**: ~100-200μs per GELU backward (×20+ blocks per step).

---

### 2. Tanh Backward: 6 kernel launches → 1

**Location**: `autograd.rs:1881-1891`

**Current**: `tanh` → `mul` → `ones` → `mul_scalar` → `add` → `mul` = 6 launches + 4 intermediate tensors.

**Proposed kernel**:
```cuda
// tanh_bwd: dx[i] = dout[i] * (1 - tanh(x[i])²)
```

**Expected savings**: ~50-80μs per Tanh backward.

---

### 3. Sigmoid Backward: 6 kernel launches → 1

**Location**: `autograd.rs:1893-1902`

**Current**: `sigmoid` → `ones` → `mul_scalar` → `add` → `mul` → `mul` = 6 launches + 4 intermediates.

**Proposed kernel**:
```cuda
// sigmoid_bwd: sig = 1/(1+exp(-x)); dx[i] = dout[i] * sig * (1 - sig)
```

**Expected savings**: ~50-80μs per Sigmoid backward.

---

### 4. ReLU Backward: Decomposed comparison → Fused mask-multiply

**Location**: `autograd.rs:3696-3718`

**Current**: Allocates a full-size F32 zeros tensor (`alloc_zeros_from_pool` + memset), calls `input.gt(&zero)` for the mask (comparison kernel), then `GpuOps::mul` (multiply kernel). That's 1 allocation + 1 memset + 2 kernel launches.

Note: `autograd_v3.rs:440-451` already has an NVRTC kernel for this, but the canonical `autograd.rs` uses the decomposed path.

**Proposed kernel**:
```cuda
// relu_bwd: dx[i] = (x[i] > 0) ? dout[i] : 0
```

**Expected savings**: ~30-50μs per ReLU backward.

---

## Medium Impact — Reduce Allocation Overhead

### 5. `compute_gradients` returns `Vec` — should be `SmallVec`

**Location**: `autograd.rs:1523-1527`

Every backward op allocates a `Vec<(TensorId, Tensor)>` for its return value. With ~2660 tape entries per Klein 4B step, that's **2660 heap allocations per backward pass** — one per op.

Most ops return 1-3 gradients (exactly matching the SmallVec capacity already used for `SavedTensors`).

**Fix**: Change return type from `Vec<(TensorId, Tensor)>` to `SmallVec<[(TensorId, Tensor); 3]>`. Update all `Ok(vec![...])` returns. The caller at line 1064 already just iterates the result, so the API change is transparent.

**Expected savings**: Eliminates ~2660 `malloc`/`free` pairs per backward pass. Wall-clock ~0.1-0.3ms depending on allocator pressure.

---

### 6. Compact Index Building Allocates Vecs Inside `flat_map`

**Location**: `autograd.rs:857-938`

The compact index builder runs under the global `AUTOGRAD_CONTEXT` mutex. Each tape entry's `flat_map` closure allocates a `Vec<TensorId>` to collect IDs from the Op enum, then the iterator consumes it. With ~2660 entries, that's ~2660 short-lived Vec allocations while holding the lock.

**Fix options** (pick one):
- Use `SmallVec<[TensorId; 8]>` instead of `Vec` in the closure
- Add `fn tensor_ids(&self, out: &mut SmallVec<[TensorId; 8]>)` to `Op` enum — avoids the allocation entirely and centralizes the ID extraction logic (currently duplicated between compact index building and the `needed_grad_ids` loop)
- Drain the tape first (line 969), then build the index outside the lock from the owned entries

---

### 7. `backward()` Takes the Global Mutex 3+ Times

**Location**: `autograd.rs:846-972, 1244-1250`

The backward path acquires `AUTOGRAD_CONTEXT` at least 3 times:
1. Lines 847-972: drain tape + build index
2. Lines 1244-1250: re-enable autograd
3. Checkpoint backward acquires it multiple times (lines 1969, 1992, 1998, 2009)

For the non-checkpoint path, this is fine (uncontended mutex ~25ns). But checkpoint backward grabs the lock repeatedly per block — up to 6 acquisitions per checkpoint entry.

**Not urgent** for single-stream training. Worth noting if multi-stream parallelism is planned.

---

## Low Impact / Niche

### 8. Softmax/LogSoftmax Backward Recomputes Forward

**Location**: `autograd.rs:3061-3091`

`input_tensor.softmax(*dim)` is called during backward to reconstruct the output, then passed to `softmax_backward`. The forward output could be saved as a saved tensor instead.

**Trade-off**: Memory vs compute. Only worth doing if softmax ops outside attention are frequent. Attention has its own backward path and doesn't hit this code.

### 9. `Slice` Backward Multi-Axis Has Duplicated Code

**Location**: `autograd.rs:2977-3012`

The `can_gpu_multi_axis` branch and the else-fallback branch (lines 2977-2993 vs 2996-3012) contain identical code. The `can_gpu_multi_axis` check is redundant since both branches do the same thing.

**Fix**: Remove the `can_gpu_multi_axis` check and the else branch — just use the single implementation for all multi-axis cases.

### 10. `Split` Backward Is Incorrect

**Location**: `autograd.rs:3015-3037`

The comment says "we don't track which split output this is" and the implementation just adds `output_grad` to a zeros tensor of the full input shape. This means the gradient is placed at position [0..output_size] regardless of which split chunk it came from.

**This is a correctness issue, not a perf issue**, but noting it here. Prior audit (`FLAME_READONLY_AUDIT_2026-04-13.md`) may have already flagged this.

---

## Summary — Ranked by ROI

| Priority | Change | Type | Est. Savings/step |
|----------|--------|------|-------------------|
| **1** | Fused GELU backward kernel | Kernel fusion | ~100-200μs × N_blocks |
| **2** | SmallVec return from `compute_gradients` | Alloc removal | ~2660 heap allocs eliminated |
| **3** | Fused Tanh backward kernel | Kernel fusion | ~50-80μs × N_ops |
| **4** | Fused Sigmoid backward kernel | Kernel fusion | ~50-80μs × N_ops |
| **5** | Fused ReLU backward kernel | Kernel fusion | ~30-50μs × N_ops |
| **6** | SmallVec in compact index `flat_map` | Alloc removal | ~2660 allocs under lock |
| **7** | Deduplicate Slice backward branches | Code cleanup | Zero perf, reduces confusion |

**Bottom line**: The engine is in good shape. The CUDA graph replay path means most of these savings only apply to the first 1-2 warmup steps (after that, the entire backward is a single graph launch). The kernel fusion wins matter most when CUDA graphs are disabled or when the graph needs recapture (shape changes, dynamic models).

For models that **don't use CUDA graphs** (e.g., variable-resolution training, models with dynamic shapes), the GELU fusion alone could save 2-4ms per step on a 20-block model.
