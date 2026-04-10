# FLAME Autograd Internals

Complete audit of the automatic differentiation system in flame-core.  
All file references are relative to `/home/alex/EriDiffusion/flame-core/src/`.

---

## 1. Graph Construction

### 1.1 The Node Struct: `TapeEntry`

The computation graph is a linear tape of `TapeEntry` nodes (`autograd.rs:282-309`):

```rust
struct TapeEntry {
    output_id: TensorId,                      // ID of the tensor this op produced
    op: Op,                                   // Which operation (enum variant)
    saved_tensors: Vec<(TensorId, Tensor)>,   // Inputs needed for backward (Vec, not HashMap)
}
```

- `output_id`: The `TensorId` of the output tensor. Used during backward to look up the gradient flowing into this node.
- `op`: An `Op` enum variant (47 variants, `autograd.rs:36-279`) describing the forward operation.
- `saved_tensors`: A `Vec<(TensorId, Tensor)>` of input tensors saved for backward. Looked up by linear scan (`get_saved`, line 297) which is fast because most ops save 1-3 tensors.

### 1.2 Where the Graph Lives: Global Mutex

The tape is stored in a **global, mutex-protected** singleton (`autograd.rs:29-31`):

```rust
lazy_static! {
    static ref AUTOGRAD_CONTEXT: Mutex<AutogradContextInner> = Mutex::new(AutogradContextInner::new());
}
```

`AutogradContextInner` (`autograd.rs:322-341`) contains:
- `tape: Vec<TapeEntry>` -- the linear computation tape
- `enabled: bool` -- whether recording is active
- `checkpoint_fns: HashMap<TensorId, Arc<dyn Fn() -> Result<Tensor>>>` -- recompute closures for activation checkpointing

There is **no per-tensor graph** -- a single global tape records all operations from all tensors.

### 1.3 Edge Representation

Edges are **implicit**, encoded within each `Op` variant's `TensorId` fields. For example:
- `Op::Add { lhs: TensorId, rhs: TensorId, ... }` -- edges from lhs and rhs to output
- `Op::MatMul { lhs: TensorId, rhs: TensorId }` -- edges from lhs and rhs to output

During backward, these IDs are used to look up saved tensors and to accumulate gradients into the correct slots in the `GradientMap`.

### 1.4 Tracing `a + b` (Exact Code Path)

When `a.add(&b)` is called (`tensor.rs:1678-1703`):

1. **Compute the result** -- dispatches to `bf16_elementwise::add_bf16()` or `GpuOps::add()` depending on dtype. This launches a CUDA kernel and returns a new `Tensor` with a fresh `TensorId` and `requires_grad = false`.

2. **Check autograd eligibility** -- `if self.requires_grad || other.requires_grad`

3. **Set output grad tracking** -- `output.requires_grad = true`

4. **Record the op** -- calls `AutogradContext::record_op()` (`autograd.rs:485-512`):
   - Acquires the global mutex lock on `AUTOGRAD_CONTEXT`
   - Checks `ctx.enabled` (returns immediately if false)
   - Optionally registers saved tensors with `CHECKPOINT_MANAGER` for CPU offload
   - Pushes a `TapeEntry { output_id, op, saved_tensors }` onto `ctx.tape`

For `Add`, saved_tensors is **empty** (`Vec::new()`) because add backward only needs the output gradient, not the inputs. For `Mul`, saved_tensors contains clones of both operands (`tensor.rs:1747`).

### 1.5 TensorId

`TensorId` is a monotonically increasing `usize` from a global atomic counter (`tensor.rs:86-96`):

```rust
static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
```

Every tensor gets a unique ID at creation. IDs are never reused.

---

## 2. Backward Pass

### 2.1 Entry Point

`Tensor::backward()` (`tensor.rs:1220-1222`) simply calls `AutogradContext::backward(self)`.

`AutogradContext::backward()` (`autograd.rs:646-1031`) is the main backward implementation.

### 2.2 Validation

Before processing (`autograd.rs:652-667`):
- Checks `loss.requires_grad` -- errors if false
- Checks `loss.shape.elem_count() == 1` -- requires scalar loss

### 2.3 Topological Order

**There is no explicit topological sort.** The tape is processed in **reverse insertion order** (`autograd.rs:879`):

```rust
for entry in ctx.tape.iter().rev() {
```

This works because the tape records operations in forward execution order, so reversing it gives a valid reverse topological order. This assumes no reordering or lazy evaluation -- which is true since every `Tensor` method immediately executes and records.

### 2.4 Compact Index Construction

Before the backward loop, a `CompactIndex` is built (`autograd.rs:683-759`). This maps every `TensorId` that appears in the tape (outputs, inputs, saved tensors) to a sequential `usize` index. The `GradientMap` then uses a flat `Vec<Option<Tensor>>` for O(1) gradient lookup instead of HashMap.

### 2.5 Gradient Initialization

```rust
let mut gradients = GradientMap::with_index(device.clone(), compact_index);
gradients.set_ones(loss.id, loss.shape.clone())?;
```

The loss gradient is initialized to a tensor of ones (FP32, `gradient.rs:97-101`).

### 2.6 Frozen-Weight Gradient Filtering

Before the backward loop, a `needed_grad_ids: HashSet<TensorId>` is built containing:
- All `output_id` values from the tape (intermediate chain nodes)
- All saved tensor IDs where `requires_grad() == true` (trainable parameters)

Frozen base-model weight IDs are excluded. This saves ~5 GB GPU memory for Klein 4B by not accumulating gradients for frozen weights that the optimizer never reads.

### 2.7 Per-Node Backward Sequence

For each tape entry in reverse order:

1. **Take output gradient** -- `gradients.take(entry.output_id)` removes the gradient from the map (avoids a clone)
2. **Compute input gradients** -- calls `compute_gradients(entry, &output_grad, &device)`
3. **Accumulate input gradients** -- for each `(tensor_id, grad)` returned, calls `gradients.accumulate(tensor_id, grad)` **only if** `needed_grad_ids.contains(&tensor_id)`. Gradients for frozen weights are silently dropped.

There is **no per-node locking** beyond the outer tape lock. The entire backward runs under a single `AUTOGRAD_CONTEXT.lock()` hold.

### 2.7 CUDA Graph Integration

When `FLAME_CUDA_GRAPH=1` (`autograd.rs:765-986`), the backward uses a warmup-capture-replay protocol:
- **Step 0 (warmup)**: normal backward, fills the caching allocator pool
- **Step 1 (capture)**: wraps backward in `cudaStreamBeginCapture`/`cudaStreamEndCapture`
- **Step 2+ (replay)**: launches the cached graph with a single `cudaGraphLaunch`

If the tape length changes, the graph is invalidated and re-captured.

### 2.8 Checkpoint Recomputation

When backward encounters `Op::Checkpoint` (`autograd.rs:1327-1397`):

1. **Retrieve recompute closure** from `ctx.checkpoint_fns`
2. **Record tape position** (`tape_start`)
3. **Re-enable autograd** (`ctx.enabled = true`) so the recomputed forward records new tape entries
4. **Execute the closure** -- this re-runs forward ops, appending entries to the tape
5. **Extract the sub-tape** -- `ctx.tape.drain(tape_start..)` removes recomputed entries
6. **Re-disable autograd** (`ctx.enabled = false`)
7. **Backward through sub-tape** -- creates a temporary `GradientMap`, seeds it with `output_grad` at the recomputed output's ID, then processes the sub-tape in reverse using `compute_gradients`
8. **Extract input gradients** -- collects gradients for the original input tensor IDs from `sub_grads`

---

## 3. CRITICAL: Does Backward Create MORE Autograd Nodes?

### 3.1 Answer: NO (in the main backward loop)

**Before the backward loop starts**, autograd is disabled at two levels:

1. **Mutex-guarded flag**: `ctx.enabled = false`
2. **Lock-free atomic flag**: `AUTOGRAD_ENABLED.store(false, Ordering::Relaxed)`

`AutogradContext::record_op()` checks the **atomic flag first** (before acquiring the mutex):

```rust
pub fn record_op(...) {
    // Fast path: skip lock entirely when autograd is disabled
    if !AUTOGRAD_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let mut ctx = AUTOGRAD_CONTEXT.lock()...;
    if !ctx.enabled { return; }  // double-check under lock
}
```

The atomic check is critical: without it, backward ops that call high-level Tensor methods (e.g., `input.sigmoid()` inside SiLU backward) would **deadlock** trying to re-acquire the already-held `AUTOGRAD_CONTEXT` mutex. The atomic provides a lock-free early exit that prevents this.

### 3.2 compute_gradients Uses a Mix of GpuOps and Tensor Methods

Inside `compute_gradients()`, backward computations use both:

**GpuOps static methods (preferred — NO autograd overhead):**
- `GpuOps::mul(output_grad, rhs_tensor)` -- raw CUDA dispatch, no `record_op` call
- `GpuOps::matmul(output_grad, &rhs_t)` -- calls `launch_gemm`, no `record_op`
- `GpuOps::mul_scalar(...)`, `GpuOps::transpose(...)`, `GpuOps::broadcast(...)`, `GpuOps::div(...)`, `GpuOps::sigmoid(...)`, `GpuOps::tanh(...)`

**Tensor methods (safe due to atomic early-out, but add minor overhead):**
- `output_grad.clone_result()` -- creates new tensor
- `output_grad.to_dtype(DType::BF16)` -- dtype conversion
- Various `.reshape()`, `.softmax()` etc. in specific backward handlers

**Activation backward ops (SiLU, GELU, Tanh, Sigmoid)** are implemented inline in `compute_gradients` using GpuOps directly. This avoids the overhead of Tensor method requires_grad checks and is the safest pattern for backward code.

The key architectural point: `GpuOps` methods are **pure CUDA dispatchers** -- they never touch the autograd tape. Tensor methods go through `GpuOps` internally but also check `requires_grad` and call `record_op` (which exits via the atomic check during backward).

### 3.3 Exception: Checkpoint Recomputation

During checkpoint backward (`autograd.rs:1351-1355`), autograd is **temporarily re-enabled**:

```rust
ctx.enabled = true;
// ... run recompute closure ...
ctx.enabled = false;
```

This means the recomputed forward DOES record new tape entries. But these are immediately drained from the main tape and processed in a local sub-backward loop, then discarded.

### 3.4 CPU Fallback in autograd_ops.rs

`BackwardOps` in `autograd_ops.rs` uses `Tensor::mul()`, `Tensor::mul_scalar()` etc. which DO go through autograd checks. However, since backward disables autograd globally, these calls don't record. **But**: several methods in `autograd_ops.rs` download data to CPU via `.to_vec()` (e.g., `relu_backward` line 87-88, `gelu_backward` lines 174-175, `silu_backward` lines 211-212, `tanh_backward` lines 237-238). These are the **legacy CPU fallback paths** and are NOT used by the main `compute_gradients()` dispatcher, which calls into `autograd_ops_complete.rs` GPU implementations or direct `GpuOps` calls instead.

---

## 4. Gradient Storage

### 4.1 Data Structure

`GradientMap` (`gradient.rs:54-63`) uses a **dual-mode** storage:

```rust
pub struct GradientMap {
    vec_store: Vec<Option<Tensor>>,       // Fast path: O(1) indexed access
    index: Option<CompactIndex>,          // Maps TensorId -> Vec index
    overflow: HashMap<TensorId, Tensor>,  // Fallback for IDs not in index
    device: Arc<CudaDevice>,
    policy: GradStorePolicy,
}
```

- **Fast path (backward)**: `CompactIndex` is built from the tape before backward. Maps TensorId -> sequential usize. Storage is a flat `Vec<Option<Tensor>>` pre-allocated to capacity. All lookups are O(1).
- **Fallback**: HashMap for IDs not in the compact index (typically not used during backward).

### 4.2 Accumulation

`accumulate()` (`gradient.rs:203-255`):
1. Upcast incoming gradient to FP32 if needed
2. If slot exists: `existing.add(&grad)` (GPU addition), ensure result is FP32
3. If slot empty: store gradient directly

### 4.3 Dtype Policy

`GradStorePolicy` (`autograd/policy.rs:1-11`) has a single variant:

```rust
InternalFP32_PublicBF16
```

- **Internally**: all gradients stored as FP32 for numerical stability
- **Publicly**: when gradients are exported (e.g., for optimizer), they're cast to BF16 via `get_public_grad()` (`gradient.rs:143-154`)

### 4.4 Every Output Gradient Gets `ensure_bf16`

At the end of `compute_gradients()` (`autograd.rs:2599-2602`):

```rust
grads.into_iter()
    .map(|(id, tensor)| ensure_bf16(tensor).map(|t| (id, t)))
    .collect()
```

Every gradient produced by compute_gradients is cast to BF16 before being returned. However, `accumulate()` immediately casts back to FP32 for storage. This means there's a BF16->FP32 round-trip on every accumulation, which loses precision but matches the mixed-precision policy.

---

## 5. Per-Op Backward Implementations

### 5.1 Complete Op Listing

| Op Variant | Backward Location | GPU or CPU | Notes |
|---|---|---|---|
| `Add` | `autograd.rs:1174-1188` | GPU (clone_result, reduce_grad_for_broadcast) | No saved tensors needed |
| `Sub` | `autograd.rs:1191-1194` | GPU (GpuOps::mul_scalar) | |
| `Mul` | `autograd.rs:1197-1234` | GPU (GpuOps::mul) | Saves both operands |
| `Div` | `autograd.rs` | GPU (GpuOps::broadcast, GpuOps::div, GpuOps::mul, GpuOps::mul_scalar, reduce_grad_for_broadcast) | Saves both operands; Op stores lhs_shape/rhs_shape for broadcast reduction |
| `MulScalar` | `autograd.rs:1237-1240` | GPU (Tensor::mul_scalar) | |
| `AddScalar` | `autograd.rs:1243-1246` | GPU (clone_result) | Identity gradient |
| `MatMul` | `autograd.rs:1248-1263` | GPU (GpuOps::transpose, GpuOps::matmul) | Saves both operands |
| `ReLU` | `autograd.rs:1266-1270` | GPU (custom relu_backward, line 2669) | Saves input |
| `GELU` | `autograd.rs` | GPU (GpuOps::mul, GpuOps::tanh, GpuOps::add_scalar, GpuOps::mul_scalar) | Inline impl using GpuOps; saves input |
| `SiLU` | `autograd.rs` | GPU (GpuOps::sigmoid, GpuOps::mul, GpuOps::add, GpuOps::mul_scalar) | Inline impl using GpuOps; saves input |
| `Tanh` | `autograd.rs` | GPU (GpuOps::tanh, GpuOps::mul, GpuOps::add) | Inline impl: 1-tanh²(x); saves input |
| `Sigmoid` | `autograd.rs` | GPU (GpuOps::sigmoid, GpuOps::mul, GpuOps::add) | Inline impl: sig*(1-sig); saves input |
| `Square` | `autograd.rs` | GPU (GpuOps::mul_scalar, GpuOps::mul) | Saves input |
| `Sqrt` | `autograd.rs` | GPU (GpuOps::sqrt, GpuOps::mul_scalar, GpuOps::div) | d/dx sqrt(x) = 0.5/sqrt(x); saves input |
| `Sum` | `autograd.rs:1313-1318` | GPU (GpuOps::broadcast) | |
| `Mean` | `autograd.rs:1399-1406` | GPU (GpuOps::mul_scalar, GpuOps::broadcast) | |
| `Transpose` | `autograd.rs:1409-1412` | GPU (GpuOps::transpose) | |
| `Conv2d` | `autograd.rs:1415-1458` | GPU (cuda_conv2d::conv2d_backward) | Saves input + weight |
| `Conv2dNHWC` | `autograd.rs:1460-1504` | GPU (cuda_conv2d + permute) | NHWC layout handling |
| `LayerNorm` | `autograd.rs:1507-1564` | GPU (autograd_ops_complete::layer_norm_backward, or cuda_ops_bf16) | Saves input |
| `RMSNorm` | `autograd.rs:1567-1602` | GPU (norm::rms_norm_backward) | Saves input + inv_rms |
| `Linear` | `autograd.rs:1605-1639` | GPU (Tensor::matmul, transpose) | Saves input + weight |
| `BatchMatMul` | `autograd.rs:1642-1657` | GPU (Tensor::batch_matmul, transpose_batch) | Saves both operands |
| `Reshape` | `autograd.rs:1660-1666` | GPU (reshape, zero-copy) | Saves input shape |
| `Permute` | `autograd.rs:1669-1673` | GPU (permute with inverse) | |
| `AddBias` | `autograd.rs:1676-1690` | GPU (clone_result, sum_dims) | |
| `SumDim` | `autograd.rs:1693-1702` | GPU (reshape + broadcast_to) | Saves input |
| `SumDimKeepdim` | `autograd.rs:1787-1796` | GPU (broadcast_to) | Saves input |
| `SumDims` | `autograd.rs:1799-1823` | GPU (reshape + broadcast_to) | Saves input |
| `Repeat` | `autograd.rs:1826-1896` | GPU (reshape + sum_dim_keepdim) | Saves input |
| `Clamp` | `autograd.rs:1705-1717` | GPU (autograd_ops_complete::clamp_backward) | Saves input |
| `MaxDim` | `autograd.rs:1751-1784` | GPU (max_dim + eq + mul) | Saves input |
| `Embedding` | `autograd.rs:1898-1928` | GPU (CudaKernels::scatter_add) | Saves indices + weight |
| `IndexSelect` | `autograd.rs:1931-1959` | GPU (cuda_kernels::scatter_add) | Saves input + indices |
| `Cat` | `autograd.rs:1962-1988` | GPU (slice per input) | Saves all inputs |
| `Split` | `autograd.rs:2070-2092` | GPU (add) | Saves input |
| `Slice` | `autograd.rs` | GPU (gpu_scatter_add_narrow via FFI) | Uses `tensor_raw_ptr`/`tensor_raw_ptr_mut` for dtype-aware device pointers (cuda_ptr returns null for BF16) |
| `Abs` | `autograd.rs:2094-2101` | GPU (sign + mul) | Saves input |
| `Log` | `autograd.rs:2104-2113` | GPU (ones/div/mul) | Saves input |
| `Softmax` | `autograd.rs:2116-2123` | GPU (autograd_ops_complete::softmax_backward) | Recomputes softmax |
| `LogSoftmax` | `autograd.rs:2126-2134` | GPU (autograd_ops_complete::log_softmax_backward) | Recomputes log_softmax |
| `Maximum` | `autograd.rs:2137-2154` | GPU (ge + mask + mul) | Saves both operands |
| `Minimum` | `autograd.rs:2157-2174` | GPU (le + mask + mul) | Saves both operands |
| `Where` | `autograd.rs:2177-2198` | GPU (mask + mul) | Saves cond, t, f |
| `MSELoss` | `autograd.rs:2200-2234` | GPU (sub + GpuOps::mul + GpuOps::broadcast) | Saves predictions + targets |
| `L1Loss` | `autograd.rs:2237-2259` | GPU (sub + sign + mul) | Saves predictions + targets |
| `HuberLoss` | `autograd.rs:2262-2303` | GPU (sign + mask + where_tensor) | Saves predictions + targets |
| `BCELoss` | `autograd.rs:2306-2333` | GPU (clamp + sub + div + mul) | Saves predictions + targets |
| `NLLLoss` | `autograd.rs:2336-2375` | GPU (CudaKernels::scatter_add) | Saves log_probs + targets |
| `GroupNorm` | `autograd.rs:2378-2429` | GPU (autograd_ops_complete::group_norm_backward) | Saves input + mean + var |
| `FlashAttention` | `autograd.rs:2432-2534` | GPU (flash_attention_backward or recompute) | Saves Q, K, V |
| `SageAttention` | `autograd.rs:2537-2588` | GPU (sage_attention_backward) | Saves Q, K, V + attn weights |
| `Cast` | `autograd.rs:1321-1325` | GPU (to_dtype) | Pass-through gradient |
| `Checkpoint` | `autograd.rs:1327-1397` | GPU (recompute forward + sub-backward) | Special handling |

### 5.2 CPU Fallback Flags

The following ops in `autograd_ops.rs` (the LEGACY module) download to CPU via `.to_vec()`:
- `relu_backward` (line 87-88) -- **NOT USED** by main backward; custom GPU impl at autograd.rs:2669
- `gelu_backward` (line 174-175)
- `silu_backward` (line 211-212)
- `tanh_backward` (line 237-238)

These are all in `BackwardOps` which is the OLD implementation. The main `compute_gradients()` dispatcher calls `autograd_ops_complete.rs` versions or direct GpuOps calls, which are all GPU.

**The `autograd_ops.rs` file header confirms this**: `#[allow(dead_code)]` and the comment says "Legacy autograd ops; keep compiled but unused until rewritten."

### 5.3 One CPU Download Remaining

The custom `relu_backward` at `autograd.rs:2669-2689` constructs a zero tensor and uses `input.gt(&zero)` which dispatches through GpuOps comparison -- this is GPU. No CPU download in the active code path.

---

## 6. Memory Lifecycle

### 6.1 Saved Tensor Lifetime

Saved tensors are `Tensor` values (which contain `Arc<CudaSlice<T>>` via `TensorStorage`). They are:

1. **Created** when `record_op` is called during forward -- typically via `.clone()` on the `Tensor` struct. Since `TensorStorage` uses shared-ownership via Arc (for `CudaSlice`), this clone is cheap (Arc increment + shape copy). The underlying GPU memory is shared.

2. **Held alive** by the `TapeEntry.saved_tensors` Vec, which lives inside the global `AUTOGRAD_CONTEXT.tape`.

3. **Freed** when `ctx.tape.clear()` is called at the end of backward (`autograd.rs:1025`). This drops all `TapeEntry` values, which drops their `saved_tensors` Vecs, which drops the `Tensor` values. If no other reference exists, the `TensorStorage`'s `CudaSlice` is freed (returned to the caching allocator pool).

### 6.2 Gradient Tensor Lifetime

Gradient tensors live in the `GradientMap`:

1. **Created** during backward when `compute_gradients` produces them
2. **Stored** in `GradientMap.vec_store` (FP32)
3. **Consumed** by `gradients.take()` when processed (moved out, not cloned)
4. **Returned** to the caller as the `GradientMap` result of `backward()`
5. **Eventually freed** when the `GradientMap` is dropped by the training loop after `accumulate_parameter_grads` copies them into `Parameter` grad slots

### 6.3 What Keeps Tensors Alive

- `Tensor` struct has `Clone` derived (line 136 `#[derive(Clone)]`)
- `TensorStorage` variants hold `Arc<CudaSlice<T>>` or `CudaSlice<T>` directly depending on feature flags
- The tape holds `Tensor` clones in `saved_tensors` -- this increments Arc refcounts
- When checkpoint discards intermediate tape entries (`ctx.tape.truncate(tape_start)` at autograd.rs:1073), the saved tensors within those entries are dropped, freeing GPU memory for the intermediates (this is the memory saving)

### 6.4 Caching Allocator

`cuda_alloc_pool.rs` implements a caching allocator:
- **Exact-size match** -- free lists keyed by exact element count (not power-of-2 buckets despite the comment)
- **Max 2 GiB** per allocation in the pool
- **Max 32 entries** per size class
- **90%+ reuse** in backward (same shapes repeat each step)
- When a `CudaSlice<f32>` is dropped, it's returned to the pool via `pool_return_f32` instead of calling `cudaFree`

---

## 7. Kernel Dispatch Path

### 7.1 Trace: backward calls GpuOps::matmul

```
compute_gradients(entry, output_grad, device)        [autograd.rs:1132]
  -> GpuOps::matmul(output_grad, &rhs_t)             [cuda_ops.rs:885]
    -> crate::ops::gemm::launch_gemm(a, b)            [ops/gemm.rs]
      -> cudarc CudaDevice::gemm() or cublasLt GEMM   [cudarc FFI]
        -> CUDA kernel launch (async)                  [GPU]
```

### 7.2 Function Call Count

From `compute_gradients` entry to CUDA kernel launch for a matmul backward:
1. `compute_gradients` -- match on Op, extract saved tensors
2. `GpuOps::transpose` -- creates transposed view (may be zero-copy or launch a kernel)
3. `GpuOps::matmul` -- calls `launch_gemm`
4. `launch_gemm` -- sets up cuBLAS parameters
5. cuBLAS API call -- enqueues kernel on stream

Total: **~5 Rust function calls** from backward match arm to kernel enqueue.

### 7.3 Synchronization Points

- **No explicit sync** in the hot path. All kernel launches are asynchronous on the CUDA stream.
- The `AUTOGRAD_CONTEXT` mutex is held for the entire backward duration (acquired once at line 672, released at line 1028).
- `CHECKPOINT_MANAGER` mutex is checked per-op but only if `CHECKPOINT_HAS_ENTRIES` atomic is true (`autograd.rs:1158`). Without checkpoints, this is a single `Relaxed` atomic load (~1ns).
- `GpuOps::get_kernels()` hits a `Mutex<HashMap>` cache but only on first use per device (`cuda_ops.rs:89-101`).

---

## 8. Klein 4B Estimates

### 8.1 Klein 4B Architecture (from klein-trainer)

- 19 double blocks, 38 single blocks
- inner_dim = 3072, head_dim = 128, num_heads = 24, mlp_hidden = 9216
- LoRA adapters on: img_qkv, img_out, txt_qkv, txt_out (double), qkv, out (single)

### 8.2 Ops per Double Block Forward

Counting ops recorded to the tape in `double_block_forward` (`model.rs:935-1089`):

| Sub-function | Approximate tape entries |
|---|---|
| `modulate_pre` (img) | ~15 ops (to_dtype, sum_dim_keepdim, div_scalar, sub, square, sum_dim_keepdim, div_scalar, add_scalar, sqrt, div, mul, add_scalar, broadcast_to, mul, add, to_dtype) |
| `modulate_pre` (txt) | ~15 ops |
| `linear3d` (img_qkv) | ~3 ops (reshape, matmul, reshape) |
| LoRA forward_delta (img_qkv) | ~3 ops (matmul down, matmul up, mul_scalar) |
| add (img_qkv + delta) | 1 op |
| `linear3d` (txt_qkv) | ~3 ops |
| LoRA forward_delta (txt_qkv) | ~3 ops |
| add (txt_qkv + delta) | 1 op |
| split_qkv (img): 3x(narrow, reshape, permute) | ~9 ops |
| split_qkv (txt) | ~9 ops |
| head_rms_norm x4 | ~40 ops (each: to_dtype, square, sum_dim_keepdim, div_scalar, add_scalar, sqrt, reshape, div, mul, to_dtype) |
| cat (q,k,v) x3 | 3 ops |
| apply_rope x2 | ~24 ops (each: reshape, narrow x2, squeeze x2, mul x2, sub, add, unsqueeze x2, cat, reshape) |
| sdpa_train | 1 op (FlashAttention, fused) |
| narrow + permute + reshape x2 (img, txt out) | ~6 ops |
| linear3d (img_out) + LoRA + add | ~7 ops |
| linear3d (txt_out) + LoRA + add | ~7 ops |
| gate_residual x2 (attn) | ~8 ops (each: to_dtype, unsqueeze, broadcast_to, mul, add) |
| modulate_pre x2 (mlp) | ~30 ops |
| swiglu x2 | ~16 ops (each: linear3d, narrow x2, silu, mul, linear3d) |
| gate_residual x2 (mlp) | ~8 ops |

**Per double block: ~210 tape entries** (without checkpointing)

With checkpointing: only **1 Checkpoint entry** per block on the main tape (the ~210 ops are truncated).

### 8.3 Ops per Single Block Forward

From `single_block_forward` (`model.rs:1092-1175`):

| Sub-function | Approximate tape entries |
|---|---|
| modulate_pre | ~15 ops |
| linear3d (qkv_mlp) | ~3 ops |
| narrow (qkv) + LoRA + add | ~5 ops |
| narrow (gate_up) | ~1 op |
| split q,k,v: 3x(narrow, reshape, permute) | ~9 ops |
| head_rms_norm x2 | ~20 ops |
| apply_rope x2 | ~24 ops |
| sdpa_train | 1 op |
| permute + reshape | ~2 ops |
| gate_up narrow x2 | ~2 ops |
| silu + mul | ~2 ops |
| cat | 1 op |
| linear3d (out) + LoRA + add | ~7 ops |
| gate_residual | ~4 ops |

**Per single block: ~96 tape entries** (without checkpointing)

With checkpointing: only **1 Checkpoint entry** per block.

### 8.4 Total Forward Tape (with checkpointing)

| Phase | Entries |
|---|---|
| img_in, txt_in linear3d | ~6 |
| timestep_embedding | ~12 |
| time_in (2x linear + silu) | ~8 |
| build_rope_2d | ~30 |
| vec.silu() | 1 |
| shared_modulation x3 | ~15 |
| 19 double blocks (checkpointed) | 19 + 19 (cat/narrow packing) = ~57 |
| cat(txt, img) | 1 |
| 38 single blocks (checkpointed) | 38 |
| final layer (narrow, modulation, linear) | ~10 |
| MSE loss | 1 |
| Total (main tape) | **~180 entries** |

### 8.5 Backward Node Count

During backward with checkpointing:
- Each checkpoint entry triggers recomputation of ~210 (double) or ~96 (single) ops
- These are processed in temporary sub-tapes, not the main tape
- Total kernel launches during backward:

| Phase | Kernel launches (approximate) |
|---|---|
| ~180 main tape entries backward | ~360 (each entry produces 1-4 gradient ops) |
| 19 double block recompute + sub-backward | 19 x (210 forward + 420 backward) = ~12,000 |
| 38 single block recompute + sub-backward | 38 x (96 forward + 192 backward) = ~10,900 |
| **Total per step** | **~23,300 kernel launches** |

### 8.6 Memory Impact of Checkpointing

Without checkpointing: all ~6,600 tape entries with saved tensors kept alive  
With checkpointing: only ~180 tape entries on main tape; intermediates freed after each block

This is the core trade-off: ~2x compute (forward recomputation during backward) for O(blocks) memory instead of O(total_ops) memory.

---

## Appendix: File Index

| File | Role |
|---|---|
| `autograd.rs` | Core engine: Op enum, TapeEntry, AutogradContextInner, backward(), compute_gradients(), checkpoint() |
| `autograd/policy.rs` | GradStorePolicy enum (single variant: InternalFP32_PublicBF16) |
| `gradient.rs` | GradientMap with CompactIndex Vec-based storage, accumulation, FP32/BF16 policy |
| `tensor.rs` | Tensor struct, TensorId, requires_grad, record_op calls in arithmetic methods |
| `autograd_ops.rs` | LEGACY BackwardOps (CPU fallback, NOT used by main backward) |
| `autograd_ops_complete.rs` | GPU backward implementations: layer_norm, batch_norm, softmax, gelu, silu, tanh, sigmoid, dropout, clamp, group_norm |
| `cuda_ops.rs` | GpuOps static methods -- pure CUDA dispatchers, NO autograd recording |
| `gradient_checkpointing.rs` | CheckpointManager, CheckpointPolicy, CPU offload / recompute support |
| `cuda_alloc_pool.rs` | Caching allocator -- exact-size free lists, eliminates cudaMalloc/cudaFree during backward |
| `cuda_graph.rs` | CUDA Graph capture/replay for backward (FLAME_CUDA_GRAPH=1) |
| `adam.rs` | Adam/AdamW optimizer -- FP32 state, reads public BF16 gradients from Parameters |
