# FLAME Gradient Mutability Audit

**Date**: 2026-02-05
**Codebase**: flame-core (EriDiffusion)
**Scope**: Core tensor system, autograd engine, gradient storage, parameter mutability, optimizer feasibility, training readiness
**Comparison baseline**: huggingface/candle audit (`/tmp/candle-audit/GRADIENT_MUTABILITY_AUDIT.md`)

---

## 1. Core Type Architecture

### 1.1 Tensor Internals

The foundational type is `Tensor`, a plain struct with NO interior mutability:

```rust
// flame-core/src/tensor.rs:97-112
#[derive(Clone)]
pub struct Tensor {
    pub(crate) storage: TensorStorage,
    pub(crate) shape: Shape,
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) id: TensorId,
    pub(crate) requires_grad: bool,
}
```

**Critical contrast with candle**: Candle wraps storage in `Arc<RwLock<Storage>>`, giving every Tensor interior mutability at the storage level. FLAME's Tensor has NO `RwLock`, NO `Mutex`, NO `RefCell` — storage is owned directly. The Tensor itself is value-typed (Clone copies it).

This is a deliberate design choice: FLAME's Tensor is an immutable value object. Mutability lives elsewhere — in `Parameter`.

### 1.2 TensorId — Identity Tracking

```rust
// flame-core/src/tensor.rs:46-56
static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

impl TensorId {
    pub fn new() -> Self {
        TensorId(TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}
```

Every tensor gets a unique monotonically increasing ID. This is the key used for gradient lookup in `GradientMap`. Identical to candle's approach, but with `pub` visibility on the inner `usize` — you can construct and inspect TensorIds freely.

### 1.3 Storage Type

```rust
// flame-core/src/tensor_storage.rs:54-93
pub enum TensorStorage {
    F32 { data: StorageSlice<f32>, numel: usize },
    F16 { data: StorageSlice<f32>, numel: usize, scale: f32 },
    BF16 { data: StorageSlice<u16>, numel: usize },       // with bf16_u16 feature
    BF16Arena { ptr: NonNull<u16>, numel: usize, device: Arc<CudaDevice>, lease: ArenaLease },
    I8 { data: StorageSlice<i8>, numel: usize },
    I32 { data: StorageSlice<f32>, numel: usize },
    Bool { data: StorageSlice<f32>, numel: usize },
}
```

**GPU-only**: FLAME has no CPU backend. Every `CudaSlice<T>` is a GPU allocation. This simplifies the storage model compared to candle's `enum Storage { Cpu, Cuda, Metal }` dispatch.

**Feature-gated storage wrapping**:

```rust
// flame-core/src/tensor_storage.rs:13-16
#[cfg(feature = "shared_storage")]
pub(crate) type StorageSlice<T> = Arc<CudaSlice<T>>;
#[cfg(not(feature = "shared_storage"))]
pub(crate) type StorageSlice<T> = CudaSlice<T>;
```

With `shared_storage` enabled, `StorageSlice` is `Arc<CudaSlice<T>>` — reference-counted GPU memory with copy-on-write via `Arc::make_mut()`:

```rust
// flame-core/src/tensor_storage.rs:41-45
#[cfg(feature = "shared_storage")]
pub(crate) fn ensure_unique_slice<T: DeviceRepr + Clone>(
    slice: &mut StorageSlice<T>,
) -> Result<&mut CudaSlice<T>> {
    Ok(Arc::make_mut(slice))
}
```

Without `shared_storage`, `StorageSlice` is bare `CudaSlice<T>` — exclusive ownership, no reference counting overhead. This is the training-oriented path: no COW overhead, direct mutable access to GPU memory.

### 1.4 BF16 Native Storage

FLAME has a special `BF16Arena` variant using a GPU memory arena with `NonNull<u16>` raw pointers and `ArenaLease` tracking. This is a memory-pool-backed path for BF16 tensors, avoiding per-tensor cudaMalloc overhead. This is a training-specific optimization not found in candle.

### 1.5 Clone Semantics

`Tensor` derives `Clone`, and `TensorStorage` derives `Clone`. What does cloning a tensor mean?

- Without `shared_storage`: Clone copies the GPU memory (`CudaSlice::clone()` allocates new device memory and copies). Every clone is a full deep copy.
- With `shared_storage`: Clone bumps the `Arc` refcount. The actual GPU memory is shared until `ensure_unique_slice()` triggers COW.

Additionally, `clone_result()` always performs a device-to-device GPU copy with a new `TensorId`. This is the "clone for computation" path — the resulting tensor has a fresh identity in the autograd tape.

---

## 2. Gradient Flow Architecture

### 2.1 Tape-Based Autograd

FLAME uses a **tape-based** (linear trace) system, not a DAG-based system like candle:

```rust
// flame-core/src/autograd.rs:29-32
lazy_static::lazy_static! {
    static ref AUTOGRAD_CONTEXT: Mutex<AutogradContextInner> =
        Mutex::new(AutogradContextInner::new());
}
```

```rust
// flame-core/src/autograd.rs:284-299
struct AutogradContextInner {
    tape: Vec<TapeEntry>,
    enabled: bool,
}
```

```rust
// flame-core/src/autograd.rs:272-282
struct TapeEntry {
    output_id: TensorId,
    op: Op,
    saved_tensors: HashMap<TensorId, Tensor>,
}
```

**Global mutable state**: The autograd context is a global `Mutex<Vec<TapeEntry>>`. Every forward operation appends to the tape. Backward processes the tape in reverse.

**Contrast with candle**: Candle embeds graph structure in the tensor itself via `BackpropOp` (each tensor knows its parent ops). FLAME externalizes the graph into a global tape. This means:
- FLAME tensors are simpler (no `op` field)
- FLAME cannot do selective backward on arbitrary subgraphs (must process entire tape)
- FLAME's tape is naturally ordered — no topological sort needed during backward

### 2.2 Operation Recording

```rust
// flame-core/src/autograd.rs:441-473
impl AutogradContext {
    pub fn record_op(output_id: TensorId, op: Op, saved_tensors: Vec<(TensorId, Tensor)>) {
        let mut ctx = match AUTOGRAD_CONTEXT.lock() {
            Ok(guard) => guard,
            Err(_) => return,  // SILENT FAILURE on poisoned mutex
        };

        if !ctx.enabled {
            return;
        }

        // Apply checkpointing policy to saved tensors
        {
            if let Ok(mut mgr) = CHECKPOINT_MANAGER.lock() {
                for (id, tensor) in &saved_tensors {
                    let _ = mgr.checkpoint_saved_tensor(*id, tensor);
                }
            } else {
                return;  // SILENT FAILURE
            }
        }

        // ...insert into tape
    }
}
```

**Checkpointing integration**: Every `record_op` call passes saved tensors through the `CHECKPOINT_MANAGER`. If the checkpoint policy says "offload to CPU" or "mark for recompute", it happens at record time. This is deeply integrated — not bolted on as an afterthought.

**Silent failure pattern**: Multiple `return` statements on mutex poisoning with no error propagation. This is a correctness concern — a poisoned mutex during training means the tape silently stops recording, and backward will produce wrong gradients with no error.

### 2.3 Op Enum (50+ Operations)

```rust
// flame-core/src/autograd.rs:36-270
pub enum Op {
    Add { lhs, rhs, lhs_shape, rhs_shape },
    Sub { lhs, rhs },
    Mul { lhs, rhs },
    Div { lhs, rhs },
    MatMul { lhs, rhs },
    Conv2d { input, weight, stride, padding },
    Linear { input, weight, bias },
    LayerNorm { input, normalized_shape },
    RMSNorm { input, weight, eps, inv_rms, normalized_shape },
    BatchMatMul { lhs, rhs },
    GroupNorm { input, num_groups, weight, bias },
    FlashAttention { query, key, value, mask, scale, causal },
    SageAttention { query_id, key_id, value_id, scale, causal, quantized },
    Conv2dNHWC { input, weight, stride, padding },
    Embedding { weight, indices },
    Softmax { input, dim },
    // ...and ~35 more
}
```

**Diffusion-relevant ops present**: `Conv2d`, `Conv2dNHWC`, `GroupNorm`, `FlashAttention`, `SageAttention`, `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`, `GELU`, `SiLU`. These cover the core operations in Flux/SD3/SDXL architectures.

**Ops NOT present**: No `DepthwiseConv2d`, no `Upsample`/`Interpolate` (though `upsampling.rs` exists as a module), no `ScaledDotProductAttention` as a single op (decomposed into `FlashAttention` or `SageAttention`), no `AdaLayerNorm` / `AdaGroupNorm` (modulated norms used in diffusion). These would need backward rules added.

### 2.4 Backward Pass

```rust
// flame-core/src/autograd.rs:607-662
pub fn backward(loss: &Tensor) -> Result<GradientMap> {
    if !loss.requires_grad {
        return Err(Error::InvalidOperation(
            "backward() called on tensor that doesn't require grad".into(),
        ));
    }

    if loss.shape.elem_count() != 1 {
        return Err(Error::InvalidOperation(
            "backward() requires scalar loss tensor".into(),
        ));
    }

    let device = loss.device.clone();
    let mut gradients = GradientMap::new(device.clone());
    gradients.set_ones(loss.id, loss.shape.clone())?;

    {
        let mut ctx = AUTOGRAD_CONTEXT.lock()
            .map_err(|_| Error::Training("autograd context mutex poisoned".into()))?;

        // Disable autograd during backward pass
        let prev_enabled = ctx.enabled;
        ctx.enabled = false;

        // Process tape in reverse
        for entry in ctx.tape.iter().rev() {
            if let Some(output_grad) = gradients.get(entry.output_id) {
                let output_grad = output_grad.clone_result()?;
                let input_grads = compute_gradients(entry, &output_grad, &device)?;
                for (tensor_id, grad) in input_grads {
                    gradients.accumulate(tensor_id, grad)?;
                }
            }
        }

        ctx.enabled = prev_enabled;
        ctx.tape.clear();  // Tape is consumed after backward
    }

    Ok(gradients)
}
```

**Key observations**:

1. **Autograd disabled during backward**: `ctx.enabled = false` prevents recording backprop-of-backprop. No second-order derivatives possible. Same limitation as candle, but explicit and intentional.

2. **Tape consumed**: `ctx.tape.clear()` after backward. You get ONE backward pass per forward pass. `retain_graph` parameter exists on the free function but is ignored.

3. **Gradient accumulation built in**: `gradients.accumulate()` handles the case where a tensor is used multiple times in the forward pass (fan-out nodes). Candle also does this, but FLAME's implementation explicitly upcasts to FP32 before accumulation — a training-quality detail.

4. **Holds global lock during entire backward**: The `AUTOGRAD_CONTEXT` mutex is held for the entire backward pass. No concurrent forward passes possible during backward. For single-model training this is fine; for pipeline parallelism it would need redesign.

### 2.5 NoGradGuard

```rust
// flame-core/src/autograd.rs:666-688
pub struct NoGradGuard {
    prev_state: bool,
}

impl NoGradGuard {
    fn new() -> Self {
        if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
            let prev = ctx.enabled;
            ctx.enabled = false;
            Self { prev_state: prev }
        } else {
            Self { prev_state: true }
        }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
            ctx.enabled = self.prev_state;
        }
    }
}
```

RAII pattern: `let _guard = AutogradContext::no_grad();` disables tape recording in scope. Essential for inference, validation, and EMA updates. Works correctly with nested guards (saves/restores previous state).

---

## 3. Gradient Storage

### 3.1 GradientMap

```rust
// flame-core/src/gradient.rs:14-18
pub struct GradientMap {
    gradients: HashMap<TensorId, Tensor>,
    device: Arc<CudaDevice>,
    policy: GradStorePolicy,
}
```

**Contrast with candle**: Candle's `GradStore(HashMap<TensorId, Tensor>)` is a thin newtype with no `new()` constructor — you can't construct one externally. FLAME's `GradientMap` has full public API: `new()`, `get()`, `get_mut()`, `take()`, `insert()`, `accumulate()`, `iter()`, `clear()`, `len()`.

**Full public mutability**: You can read, write, accumulate, and take gradients from any `GradientMap`. This is a critical difference — candle locks you out of gradient manipulation; FLAME gives you full control.

### 3.2 FP32 Gradient Policy

```rust
// flame-core/src/gradient.rs:53-64
pub fn get_public_grad(&self, id: TensorId) -> Result<Tensor> {
    match self.policy {
        GradStorePolicy::InternalFP32_PublicBF16 => {
            let g_fp32 = self.get_fp32(id)?;
            let grad = g_fp32.to_dtype(DType::BF16)?;
            Ok(grad)
        }
    }
}
```

**Dual-precision gradient design**: Gradients are stored internally as FP32 for numerical accuracy. When retrieved via `get_public_grad()`, they're downcast to BF16 for communication. The raw FP32 gradients are accessible via `get()` / `get_mut()` for optimizers that need full precision (Adam moment updates).

This is a production-quality mixed-precision design. Candle has no equivalent.

### 3.3 Gradient Accumulation

```rust
// flame-core/src/gradient.rs:105-133
pub fn accumulate(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
    let grad = if grad.dtype() != DType::F32 {
        grad.to_dtype(DType::F32)?
    } else {
        grad
    };
    match self.gradients.get_mut(&id) {
        Some(existing) => {
            if existing.dtype() != DType::F32 {
                let up = existing.to_dtype(DType::F32)?;
                *existing = up;
            }
            let sum = existing.add(&grad)?;
            let sum = if sum.dtype() != DType::F32 {
                sum.to_dtype(DType::F32)?
            } else {
                sum
            };
            *existing = sum;
        }
        None => {
            self.gradients.insert(id, grad);
        }
    }
    Ok(())
}
```

**FP32-enforced accumulation**: Every incoming gradient is upcast to FP32 before accumulation. This prevents the catastrophic precision loss that happens when accumulating many small BF16 gradients. Production training requires this; candle doesn't do it.

**Not truly in-place**: `existing.add(&grad)` allocates a new tensor, then `*existing = sum` replaces the old one. Two GPU allocations per accumulate. An in-place `addmm_` or `axpy` kernel would be more efficient, but the current approach is correct.

### 3.4 TensorGradExt Trait

```rust
// flame-core/src/gradient.rs:174-186
pub trait TensorGradExt {
    fn grad<'a>(&self, gradients: &'a GradientMap) -> Option<&'a Tensor>;
    fn grad_mut<'a>(&self, gradients: &'a mut GradientMap) -> Option<&'a mut Tensor>;
    fn take_grad(&self, gradients: &mut GradientMap) -> Option<Tensor>;
    fn has_grad(&self, gradients: &GradientMap) -> bool;
}
```

Extension methods on `Tensor` for ergonomic gradient access. `tensor.grad(&grads)` instead of `grads.get(tensor.id)`.

---

## 4. The Parameter System — FLAME's Mutability Solution

### 4.1 Parameter Type

This is where FLAME fundamentally diverges from candle:

```rust
// flame-core/src/parameter.rs:10-19
pub struct Parameter {
    data: Arc<Mutex<Tensor>>,
    grad: Arc<Mutex<Option<Tensor>>>,
    requires_grad: bool,
    id: TensorId,
}
```

**Interior mutability by design**: `Parameter` wraps a `Tensor` in `Arc<Mutex<>>`, providing:
- Thread-safe mutable access to weight data
- Separate gradient storage per parameter
- In-place weight updates without consuming the parameter

**Aliased as `Var`**: `pub use parameter::Parameter as Var;` — FLAME's `Var` is NOT candle's `Var`. Candle's `Var` is a newtype around `Tensor` with `is_variable == true`. FLAME's `Parameter`/`Var` is a completely separate type with its own storage semantics.

### 4.2 The Tensor/Parameter Split

This is the architectural decision that matters most:

| Aspect | FLAME Tensor | FLAME Parameter |
|--------|-------------|-----------------|
| Mutability | Immutable value object | `Arc<Mutex<Tensor>>` interior mutability |
| Gradient | Via external `GradientMap` | Via internal `Arc<Mutex<Option<Tensor>>>` |
| Identity | `TensorId` (monotonic counter) | Copies `TensorId` from wrapped tensor |
| Cloning | Deep copy (or Arc refcount bump) | Arc clone (shared mutable state) |
| Use case | Activations, intermediates | Trainable weights |

**This is the correct architecture for training**. Activations are ephemeral — they flow through the graph, get used once, and die. They don't need mutability. Weights are persistent — they get updated every step and must be modifiable in place. Giving every tensor interior mutability (like candle does with `Arc<RwLock<Storage>>`) is paying mutability overhead for activations that never need it.

### 4.3 Parameter API — Weight Updates

```rust
// flame-core/src/parameter.rs:75-82
pub fn set_data(&self, tensor: Tensor) -> Result<()> {
    let mut data_lock = self.data.lock()
        .map_err(|_| Error::Training("parameter data mutex poisoned".into()))?;
    *data_lock = tensor;
    Ok(())
}
```

```rust
// flame-core/src/parameter.rs:139-154
pub fn apply_update(&self, update: &Tensor) -> Result<()> {
    let mut data_lock = self.data.lock()
        .map_err(|_| Error::Training("parameter data mutex poisoned".into()))?;

    let compute = ComputeF32::for_input(&data_lock)?;
    let update_f32 = if update.dtype() == DType::F32 {
        update.clone_result()?
    } else {
        update.to_dtype(DType::F32)?
    };
    let new_f32 = compute.tensor().sub(&update_f32)?;
    *data_lock = compute.into_output(new_f32)?;
    Ok(())
}
```

**FP32 compute path**: `apply_update()` uses `ComputeF32` to upcast the parameter to FP32, apply the update in FP32, then downcast back to the parameter's native dtype. This is essential for BF16 training — accumulating small learning-rate-scaled updates directly in BF16 causes weight stagnation.

**Not truly in-place on GPU**: Despite the `Mutex`-guarded mutation, the actual GPU operation is `new_f32 = data.sub(&update)` which allocates a new GPU tensor. The "in-place" mutation is at the Rust level (replacing the `Tensor` inside the `Mutex`), not at the CUDA level. The old GPU allocation is freed when the replaced `Tensor` is dropped.

### 4.4 Gradient Management on Parameter

```rust
// flame-core/src/parameter.rs:85-113
pub fn set_grad(&self, grad: Tensor) -> Result<()> {
    let mut grad_lock = self.grad.lock()
        .map_err(|_| Error::Training("parameter grad mutex poisoned".into()))?;
    let grad = if grad.dtype() == DType::F32 { grad } else { grad.to_dtype(DType::F32)? };
    *grad_lock = Some(grad);
    Ok(())
}

pub fn grad(&self) -> Option<Tensor> {
    if let Ok(grad_lock) = self.grad.lock() {
        grad_lock.as_ref().map(|g| g.clone())
    } else {
        None
    }
}

pub fn zero_grad(&self) {
    if let Ok(mut grad_lock) = self.grad.lock() {
        *grad_lock = None;
    }
}
```

**Gradient lifecycle**: After `backward()` returns a `GradientMap`, user code must manually transfer gradients to Parameters via `param.set_grad(grad_map.get(param.id()).clone())`. The optimizer then reads `param.grad()` and calls `param.apply_update()`.

**Gap**: There is no automatic `GradientMap → Parameter` gradient transfer. The user must write the bridge. This is a usability concern but not an architectural blocker.

---

## 5. Optimizer Implementations

### 5.1 Adam / AdamW

```rust
// flame-core/src/adam.rs:7-24
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: u32,
    m: HashMap<TensorId, Tensor>,    // First moment estimates
    v: HashMap<TensorId, Tensor>,    // Second moment estimates
    weight_decay: f32,
}
```

```rust
// flame-core/src/adam.rs:42-120
pub fn step(&mut self, parameters: &[Parameter]) -> Result<()> {
    self.t += 1;
    let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
    let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

    for param in parameters {
        if let Some(mut grad) = param.grad() {
            // ...dtype coercion to state dtype...
            // ...AdamW weight decay on grad...

            // m = β1 * m + (1 - β1) * g
            *m = m.mul_scalar(self.beta1)?.add(&grad.mul_scalar(1.0 - self.beta1)?)?;
            // v = β2 * v + (1 - β2) * g²
            *v = v.mul_scalar(self.beta2)?.add(&grad_sq.mul_scalar(1.0 - self.beta2)?)?;

            // Bias-corrected update
            let m_hat = m.div_scalar(bias_correction1)?;
            let v_hat = v.div_scalar(bias_correction2)?;
            let update = m_hat.div(&v_hat.sqrt()?.add_scalar(self.eps)?)?.mul_scalar(self.lr)?;

            param.apply_update(&update)?;
        }
    }
    Ok(())
}
```

**Comparison with candle's Adam**:

| Aspect | Candle | FLAME |
|--------|--------|-------|
| State storage | HashMap with `Var::set()` | HashMap with `*m = new_tensor` |
| Weight update | `var.set(&var.sub(&update)?)` | `param.apply_update(&update)` |
| FP32 compute | No (uses parameter dtype) | Yes (`ComputeF32` in `apply_update()`) |
| Gradient clip | No | Available separately |
| Weight decay | L2 penalty on grad | L2 penalty on grad (AdamW style) |
| State dtype | Same as param | Configurable via `config::select_optimizer_state_dtype()` |
| Temporaries per param | ~8-10 GPU allocations | ~8-10 GPU allocations |

**Temporary allocation count**: Both frameworks allocate multiple intermediate GPU tensors per parameter per step. FLAME's Adam: `grad.mul_scalar()`, `m.mul_scalar()`, `.add()`, `grad.mul(&grad)`, `v.mul_scalar()`, `.add()`, `m.div_scalar()`, `v.div_scalar()`, `v_hat.sqrt()`, `.add_scalar()`, `m_hat.div()`, `.mul_scalar()` — approximately 12 temporaries per parameter per step.

This is the single biggest performance concern for both frameworks. A fused CUDA kernel for Adam would reduce this to 1-2 kernel launches per parameter.

### 5.2 SGD with Custom CUDA Kernels

```rust
// flame-core/src/sgd/mod.rs:11-28
const CUDA_SRC: &str = r#"
extern "C" __global__
void sgd_f32(float* __restrict__ p, const float* __restrict__ g, size_t n, float lr){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){ p[i] -= lr * g[i]; }
}
extern "C" __global__
void sgd_bf16(__nv_bfloat16* __restrict__ p, const float* __restrict__ g, size_t n, float lr){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        float pi = __bfloat162float(p[i]);
        pi -= lr * g[i];
        p[i] = __float2bfloat16_rn(pi);
    }
}
"#;
```

**True in-place GPU update**: SGD uses custom CUDA kernels that modify parameter memory directly on the GPU. `p[i] -= lr * g[i]` — no temporaries, no allocations, single kernel launch. The BF16 variant upcasts to F32 for the arithmetic, then rounds back.

```rust
// flame-core/src/sgd/mod.rs:57-100
pub fn step_inplace(param: &mut Tensor, grad: &Tensor, lr: f32) -> Result<()> {
    // ...validation...
    unsafe {
        match param.dtype() {
            DType::F32 => {
                let p_slice = param.storage_mut().try_as_mut_slice_f32()?;
                let g_slice = grad.storage_ref().try_as_slice_f32()?;
                let func = param_dev.get_func("flame_sgd", "sgd_f32").ok_or(...)?;
                func.launch(cfg, (p_slice, g_slice, n as u64, lr))?;
            }
            DType::BF16 => {
                let p_slice = param.storage_mut().try_as_mut_slice_u16()?;
                // ...
            }
        }
    }
}
```

**`storage_mut()` exists on Tensor**: Unlike candle where `storage_mut()` is `pub(crate)`, FLAME's SGD calls `param.storage_mut()` to get a mutable reference to the underlying `CudaSlice`. This is the path for true zero-copy weight updates.

**This is what a production optimizer looks like**: One kernel launch, zero temporaries, direct GPU memory mutation. The Adam optimizer should eventually get the same treatment (fused Adam CUDA kernel), but the SGD implementation proves the storage layer supports it.

---

## 6. Training Infrastructure Assessment

### 6.1 Mixed Precision (AMP)

```rust
// flame-core/src/mixed_precision.rs:21-40
pub struct AMPContext {
    pub enabled: bool,
    pub compute_dtype: DType,
    pub loss_scale: f32,
    pub dynamic_scaling: bool,
    pub growth_factor: f32,       // default: 2.0
    pub backoff_factor: f32,      // default: 0.5
    pub growth_interval: usize,   // default: 2000
    current_step: usize,
    steps_since_update: usize,
}
```

Full dynamic loss scaling with:
- `scale_loss()` — scale loss before backward
- `unscale_grads()` — unscale gradients after backward
- `update_scale()` — adjust scale based on inf/nan detection
- Growth/backoff with configurable intervals

Candle has **none of this**. For BF16 diffusion training, dynamic loss scaling is mandatory to prevent underflow in early training and overflow during fine-tuning.

### 6.2 Gradient Clipping

```rust
// flame-core/src/gradient_clip.rs:9-16
pub enum GradientClipStrategy {
    ClipByNorm { max_norm: f32 },
    ClipByValue { min_value: f32, max_value: f32 },
    AdaptiveClip { clip_factor: f32 },
}
```

Three strategies implemented with GPU-accelerated norm computation. `ClipByNorm` is the standard approach for diffusion training (typically `max_norm = 1.0`).

**Not compiled into lib.rs**: `gradient_clip.rs` is NOT declared as a `pub mod` in `lib.rs`. It exists as source code but may not be part of the compiled crate. This needs verification.

### 6.3 Gradient Checkpointing

```rust
// flame-core/src/gradient_checkpointing.rs:25-41
pub struct CheckpointManager {
    saved_activations: HashMap<TensorId, CheckpointedTensor>,
    policy: CheckpointPolicy,
    device: Option<Arc<CudaDevice>>,
    memory_saved: usize,
    recompute_count: usize,
    tensor_registry: HashMap<TensorId, Weak<Tensor>>,
}

enum CheckpointedTensor {
    OnDevice(Tensor),
    Deleted {
        compute_fn: Box<dyn Fn() -> Result<Tensor> + Send + Sync>,
        shape: Shape,
        dtype: DType,
    },
}
```

Three policies: `CPUOffload`, `Recompute`, `Adaptive { memory_threshold, prefer_recompute }`.

**Deeply integrated with autograd**: The `CHECKPOINT_MANAGER` is called from within `AutogradContext::record_op()`. When a tensor is saved for backward, the checkpoint manager decides whether to keep it on GPU, offload to CPU, or mark for recomputation. This is not a bolt-on — it's part of the recording path.

**Legacy status**: The file header says `"Legacy checkpointing utilities; modern API to follow in Phase 3."` This is scaffolding, not a finished system.

### 6.4 LoRA

```rust
// flame-core/src/lora.rs:32-43
pub struct LoRALayer {
    pub lora_down: Tensor,     // [rank, in_features]
    pub lora_up: Tensor,       // [out_features, rank]
    pub config: LoRAConfig,
    pub scale: f32,            // alpha / rank
    device: Arc<CudaDevice>,
}
```

Standard LoRA with configurable rank, alpha, dropout. `merge_weights()` for inference-time weight merging.

**Legacy status**: Header says `"Legacy LoRA module kept for reference; on hold until Phase 3."` The LoRA layers use bare `Tensor` (not `Parameter`), so they'd need to be wrapped in `Parameter` for training.

### 6.5 Serialization — SafeTensors Compatibility

```rust
// flame-core/src/serialization.rs:10-16
pub enum SerializationFormat {
    Binary,       // Native FLAME format ("FLMT"/"FLMM")
    SafeTensors,  // HuggingFace compatible
}
```

**Custom SafeTensors implementation**: FLAME implements its own SafeTensors reader/writer rather than using the `safetensors` crate. This means:
- Full control over the loading path
- Tensors load directly as FLAME `Tensor` objects
- No intermediate conversion through a third-party type
- But: must be validated against the actual safetensors spec (endianness, alignment, metadata format)

**Loading produces `Tensor`, not `Parameter`**: `load_tensors()` returns `HashMap<String, Tensor>`. To use loaded weights for training, you must wrap each in `Parameter::new(tensor)`. Unlike candle's `Var::from_tensor()` which copies the entire storage, FLAME's `Parameter::new()` wraps the tensor in `Arc<Mutex<>>` — **zero-copy wrapping** (the Tensor is moved, not cloned).

This is a significant advantage over candle for loading large diffusion models.

### 6.6 Loss Functions

`loss.rs` provides: MSE, L1, Huber, BCE, CrossEntropy, NLL. The `mse_loss_bf16()` variant upcasts to FP32 for numerical stability. These are the basics — diffusion training primarily uses MSE loss (noise prediction) or flow matching loss (velocity prediction).

### 6.7 Regularization

Dropout, Dropout2d, L1 regularization, L2 regularization, Spectral norm. Standard toolkit.

---

## 7. Where Mutability Blocks or Enables Training

### 7.1 What Works

| Capability | Status | Notes |
|-----------|--------|-------|
| Forward pass with autograd | **Works** | Tape records ops, backward traverses |
| `backward()` produces gradients | **Works** | Returns `GradientMap` with FP32 gradients |
| Parameter in-place updates | **Works** | `Arc<Mutex<Tensor>>` allows mutation |
| Adam/AdamW optimizer | **Works** | Takes `&[Parameter]`, reads grads, applies updates |
| SGD with CUDA kernels | **Works** | True in-place GPU mutation |
| Mixed precision (AMP) | **Works** | Dynamic loss scaling, FP32 master weights |
| BF16 compute with FP32 gradients | **Works** | `InternalFP32_PublicBF16` policy |
| SafeTensors loading | **Works** | Loads diffusers weights directly |
| Weight → Parameter conversion | **Works** | Zero-copy `Parameter::new(tensor)` |
| NoGrad inference mode | **Works** | RAII guard pattern |
| Gradient accumulation | **Works** | Built into `GradientMap::accumulate()` |

### 7.2 What's Missing or Broken

| Capability | Status | Severity | Notes |
|-----------|--------|----------|-------|
| GradientMap → Parameter bridge | **Missing** | Medium | Manual loop required after backward() |
| Gradient clipping integration | **Missing** | Medium | `gradient_clip.rs` not in `pub mod` list |
| Fused Adam CUDA kernel | **Missing** | High (perf) | 12 temporaries per param per step |
| `retain_graph` for backward | **Missing** | Low | Tape always consumed |
| Second-order derivatives | **Missing** | Low | Autograd disabled during backward |
| LR scheduling | **Missing** | Medium | No scheduler abstraction |
| EMA (Exponential Moving Average) | **Missing** | High | Required for diffusion model inference quality |
| Gradient scaling integration | **Partial** | Medium | AMPContext exists but not wired into backward() |
| Multiple autograd versions | **Confusing** | Low | `autograd.rs`, `autograd_v3.rs`, `autograd_v4/` — which is canonical? |
| Silent failures on mutex poison | **Bug** | High | Tape silently stops recording, wrong gradients |
| Massive `#![allow(dead_code)]` | **Smell** | Low | Indicates transitional state |
| In-place ops for Tensor | **Limited** | Medium | Only `copy_()`, `reshape_inplace()`, `copy_from_bf16_slice()` — no `add_()`, `mul_()` |

### 7.3 Detailed Analysis of Missing Pieces

**GradientMap → Parameter bridge** (estimated: ~30 lines):
```rust
// What's needed:
fn apply_gradients(params: &[Parameter], grads: &GradientMap) -> Result<()> {
    for param in params {
        if let Some(grad) = grads.get(param.id()) {
            param.set_grad(grad.clone())?;
        }
    }
    Ok(())
}
```

**EMA** (estimated: ~100 lines):
Diffusion models use EMA weights for inference. Needs: `EMAModel { shadow_params: Vec<Parameter>, decay: f32 }` with `update()` that computes `shadow = decay * shadow + (1 - decay) * param` for each parameter.

**LR Scheduling** (estimated: ~80 lines):
At minimum: warmup + cosine annealing. Common in diffusion training.

**Fused Adam kernel** (estimated: ~60 lines CUDA + 100 lines Rust):
The SGD kernel proves the architecture supports this. Would reduce Adam from ~12 kernel launches to 1 per parameter.

---

## 8. Comparison Verdict: FLAME vs. Candle for Diffusion Training

### 8.1 Architecture Comparison

| Dimension | Candle | FLAME | Winner |
|-----------|--------|-------|--------|
| Tensor mutability | `Arc<RwLock<Storage>>` on every tensor | Immutable Tensor + mutable Parameter | **FLAME** (correct separation) |
| Gradient storage | `GradStore` — no public constructor | `GradientMap` — full public API | **FLAME** |
| Gradient accumulation | None built in | FP32-enforced accumulation | **FLAME** |
| Gradient precision | Same as tensor dtype | Internal FP32, public BF16 | **FLAME** |
| Optimizer architecture | `Var::set()` copies storage | `Parameter::apply_update()` via Mutex | **FLAME** |
| In-place CUDA updates | Not possible (no mutable storage access) | SGD has custom kernels | **FLAME** |
| Mixed precision | None | Full AMP with dynamic loss scaling | **FLAME** |
| Gradient clipping | None | Three strategies (needs compilation fix) | **FLAME** |
| Gradient checkpointing | None | Scaffold with CPUOffload/Recompute | **FLAME** |
| SafeTensors loading | Via `safetensors` crate → copy to `Var` | Direct load → zero-copy to `Parameter` | **FLAME** |
| Backend support | CPU + CUDA + Metal | CUDA only | **Candle** |
| Model zoo | Large (Llama, Mistral, Whisper, etc.) | None (framework only) | **Candle** |
| Maturity | Production-used by HuggingFace | In development | **Candle** |
| Autograd approach | DAG embedded in tensor | Global tape (linear trace) | **Draw** (trade-offs) |
| Code health | Clean, minimal warnings | `#![allow(dead_code, ...)]` 30 suppressions | **Candle** |

### 8.2 For Your Specific Use Case: Diffusion Model Training with Diffusers/SafeTensors Weights

Candle's advantages (model zoo, multi-backend) are irrelevant — you're loading external weights and training on CUDA only.

FLAME was **designed for this use case**:
1. Load safetensors weights as `Tensor`
2. Wrap in `Parameter` (zero-copy)
3. Forward pass records to tape
4. `backward()` produces FP32 `GradientMap`
5. Transfer gradients to Parameters
6. Adam/AdamW applies updates with FP32 compute
7. AMP handles loss scaling
8. Zero gradients, repeat

The full training loop is architecturally complete. What's missing is plumbing (the gradient transfer step, LR scheduling, EMA) and performance optimization (fused Adam kernel). These are engineering tasks, not architectural redesigns.

### 8.3 Risk Assessment

**Low risk**:
- Core tensor/parameter/gradient architecture is sound
- Autograd tape approach is well-suited for diffusion training (sequential forward pass, single backward)
- SafeTensors loading path is direct
- BF16 support is deep (native storage, FP32 compute paths, proper rounding)

**Medium risk**:
- Multiple autograd versions (`autograd.rs` vs `autograd_v3.rs` vs `autograd_v4/`) suggest the autograd system is still stabilizing
- `gradient_clip.rs` not compiled into crate — integration gaps
- Silent mutex failure pattern could produce wrong gradients with no error
- `#![allow(dead_code)]` suppressing ~30 warning categories hides issues

**High risk**:
- No evidence of end-to-end training test (forward → backward → optimizer step → loss decreases)
- Temporary allocation count in Adam is a VRAM pressure concern for large models
- Tape holds entire computation graph in memory until backward completes — for diffusion models with 100+ ops per timestep, this is substantial
- `autograd_ops_complete.rs` compiled but relationship to main `autograd.rs` unclear — are backward rules complete for all 50+ ops?

### 8.4 Final Verdict

**FLAME is architecturally correct for diffusion model training.** The Tensor/Parameter separation, FP32 gradient policy, AMP infrastructure, and custom CUDA kernel path demonstrate that someone who understands production training built this.

**What candle would require to reach parity**:
- ~335 lines of core changes (from candle audit)
- Public `GradStore` constructor
- Gradient accumulation
- Mixed precision support
- Gradient clipping
- Custom CUDA optimizer kernels
- Estimated: 1000-1500 lines of new code + fork maintenance burden

**What FLAME requires to be training-ready**:
- ~30 lines: GradientMap → Parameter bridge helper
- ~100 lines: EMA implementation
- ~80 lines: LR scheduler
- ~50 lines: Fix gradient_clip compilation + wire into training loop
- ~160 lines: Fused Adam CUDA kernel (performance, not correctness)
- Fix silent mutex failures (change `return` to `Err()`)
- Verify backward rules are complete for all ops used in target diffusion architectures
- End-to-end training smoke test
- Estimated: ~500 lines of new code, no architectural changes

**The answer is FLAME. Build on what you have, not on what you'd have to fork.**

---

## Appendix A: Data Flow Diagram

```
                    FLAME Training Loop
                    ====================

    ┌─────────────────────────────────────────────────┐
    │               SafeTensors File                   │
    │  (Flux, SD3, etc. from HuggingFace)             │
    └────────────────────┬────────────────────────────┘
                         │ load_tensors()
                         ▼
    ┌─────────────────────────────────────────────────┐
    │         HashMap<String, Tensor>                  │
    │  (weights as immutable Tensor objects)           │
    └────────────────────┬────────────────────────────┘
                         │ Parameter::new(tensor)  ← ZERO COPY
                         ▼
    ┌─────────────────────────────────────────────────┐
    │         Vec<Parameter>                           │
    │  (each wraps Tensor in Arc<Mutex<>>)            │
    └────────────────────┬────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────────────┐
    │                    │   TRAINING LOOP             │
    │                    ▼                             │
    │  ┌──────────────────────┐                       │
    │  │   Forward Pass       │                       │
    │  │   (tape records ops) │                       │
    │  └──────────┬───────────┘                       │
    │             │                                    │
    │             ▼                                    │
    │  ┌──────────────────────┐                       │
    │  │   Loss Computation   │                       │
    │  │   (MSE / flow match) │                       │
    │  └──────────┬───────────┘                       │
    │             │ AMP: scale_loss()                  │
    │             ▼                                    │
    │  ┌──────────────────────┐                       │
    │  │   backward()         │                       │
    │  │   (tape in reverse)  │──→ GradientMap (FP32) │
    │  └──────────┬───────────┘                       │
    │             │ AMP: unscale_grads()              │
    │             │ Clip: clip_grads()                │
    │             ▼                                    │
    │  ┌──────────────────────┐                       │
    │  │   param.set_grad()   │  ← bridge needed     │
    │  └──────────┬───────────┘                       │
    │             │                                    │
    │             ▼                                    │
    │  ┌──────────────────────┐                       │
    │  │   optimizer.step()   │                       │
    │  │   (Adam: FP32 math)  │                       │
    │  │   apply_update()     │                       │
    │  └──────────┬───────────┘                       │
    │             │                                    │
    │             ▼                                    │
    │  ┌──────────────────────┐                       │
    │  │   zero_grad()        │                       │
    │  │   (clear param grads)│                       │
    │  └──────────┬───────────┘                       │
    │             │                                    │
    │             └──────→ repeat                      │
    └─────────────────────────────────────────────────┘
```

## Appendix B: Module Compilation Status

```
lib.rs module declarations vs. actual usage:

COMPILED AND ACTIVE:
  autograd          ✅  Main autograd engine
  parameter         ✅  Parameter type (aliased as Var)
  gradient          ✅  GradientMap
  tensor            ✅  Core Tensor type
  tensor_storage    ✅  TensorStorage
  adam              ✅  Adam/AdamW optimizers
  sgd               ✅  SGD with CUDA kernels
  mixed_precision   ✅  AMPContext
  serialization     ✅  SafeTensors support
  loss              ✅  Loss functions
  lora              ✅  LoRA (legacy, on hold)
  gradient_checkpointing  ✅  CheckpointManager
  regularization    ✅  Dropout, L1, L2

COMPILED BUT STATUS UNCLEAR:
  autograd_v3       ⚠️  Comment says "Primary" but autograd.rs is used
  autograd_v4       ⚠️  Feature-gated (autograd_v4 feature)
  autograd_ops_complete  ⚠️  Compiled, relationship to autograd.rs unknown

NOT COMPILED:
  gradient_clip     ❌  Not declared as pub mod in lib.rs
  autograd_ops      ❌  Commented out
  autograd_engine   ❌  Commented out
```
