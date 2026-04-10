# FLAME Operation Dispatch Overhead Audit

**Date**: 2026-04-09
**Scope**: Full call-stack trace from user-facing API to CUDA kernel for 8 core operations
**Codebase**: `/home/alex/EriDiffusion/flame-core/src/`
**Feature set assumed**: `cuda` + `bf16_u16` (the production inference/training config)

---

## Table of Contents

1. [Tensor Struct Layout](#tensor-struct-layout)
2. [Operation 1: BF16 Addition (`a.add(b)`)](#1-bf16-addition)
3. [Operation 2: Matrix Multiply (`a.matmul(b)`)](#2-matrix-multiply)
4. [Operation 3: RMS Norm](#3-rms-norm)
5. [Operation 4: Linear Forward](#4-linear-forward)
6. [Operation 5: SDPA (Scaled Dot-Product Attention)](#5-sdpa)
7. [Operation 6: Backward of MatMul](#6-backward-of-matmul)
8. [Operation 7: Gradient Accumulation](#7-gradient-accumulation)
9. [Operation 8: Adam Optimizer Step](#8-adam-optimizer-step)
10. [Autograd Cost Analysis](#autograd-cost-analysis)
11. [No-Grad Path Analysis](#no-grad-path-analysis)
12. [Summary Table](#summary-table)
13. [Ranked Fixes](#ranked-fixes)

---

## Tensor Struct Layout

```rust
// tensor.rs:137-152
pub struct Tensor {
    pub(crate) storage: TensorStorage,  // enum, ~24-40 bytes depending on variant
    pub(crate) shape: Shape,            // Vec<usize> + cached strides = ~48 bytes
    pub(crate) device: Arc<CudaDevice>, // 8 bytes (pointer)
    pub(crate) id: TensorId,            // usize = 8 bytes
    pub(crate) requires_grad: bool,     // 1 byte + padding
}
// Total: ~80-100 bytes per Tensor (not counting Arc overhead)
```

```rust
// tensor_storage.rs:54-100
pub enum TensorStorage {
    // BF16 variant (bf16_u16 feature):
    BF16 { data: CudaSlice<u16>, numel: usize },  // ~24 bytes
    // Also: BF16Arena, BF16View, F32, F16, I8, I32, Bool variants
}
```

```rust
// autograd.rs:282-292
struct TapeEntry {
    output_id: TensorId,                       // 8 bytes
    op: Op,                                    // enum, varies 16-80 bytes
    saved_tensors: Vec<(TensorId, Tensor)>,    // 24 bytes header + N * ~108 bytes per entry
}
// Minimum TapeEntry size: ~48 bytes (Op::Add with 0 saved tensors)
// Typical TapeEntry for MatMul: ~48 + 2 * 108 = ~264 bytes (saves both inputs)
```

---

## 1. BF16 Addition

**User code**: `let c = a.add(&b)?;`

### Call Stack (fast path: same shape, no broadcast)

```
tensor.rs:1678  Tensor::add(&self, other)
                  ├─ dtype check: self.dtype() == BF16 && other.dtype() == BF16  [2 match stmts on TensorStorage]
                  ├─ CALL bf16_elementwise::add_bf16(self, other)
                  │     bf16_elementwise.rs:532
                  │     ├─ shapes_equal_no_broadcast(a, b)                       [3 comparisons]
                  │     │    ├─ a.dtype() == BF16                                [1 match on TensorStorage]
                  │     │    ├─ b.dtype() == BF16                                [1 match on TensorStorage]
                  │     │    ├─ a.shape().dims() == b.shape().dims()             [slice comparison]
                  │     │    └─ a.shape().elem_count() == b.shape().elem_count() [1 comparison]
                  │     ├─ CALL launch_bf16_flat(a, b, "add_bf16_flat_kernel", "add_bf16")
                  │     │     bf16_elementwise.rs:559
                  │     │     ├─ 2x debug_assert_eq!(dtype, BF16)                [debug only]
                  │     │     ├─ n = a.shape().elem_count()                      [1 field read]
                  │     │     ├─ ALLOC: device.alloc::<u16>(n)                   [1 CUDA alloc via pool]
                  │     │     ├─ CONSTRUCT Tensor { storage, shape, device, id, requires_grad }
                  │     │     │    ├─ TensorStorage::BF16 { data: data.into(), numel: n }
                  │     │     │    ├─ shape.clone()                              [1 Vec clone]
                  │     │     │    ├─ device.clone()                             [1 Arc clone]
                  │     │     │    └─ TensorId::new()                            [1 atomic fetch_add]
                  │     │     ├─ CALL ensure(&device, "add_bf16_flat_kernel", CUDA_ADD_MUL_BF16_FLAT)
                  │     │     │     bf16_elementwise.rs:323
                  │     │     │     ├─ device.get_func(nm, nm)                   [HashMap lookup in cudarc]
                  │     │     │     └─ RETURN Ok(()) if found (hot path)
                  │     │     ├─ device.get_func(kernel_name, kernel_name)       [2nd HashMap lookup, REDUNDANT]
                  │     │     ├─ a.as_device_ptr_bf16("add_bf16:a")              [1 match + format! for tag]
                  │     │     ├─ b.as_device_ptr_bf16("add_bf16:b")              [1 match + format! for tag]
                  │     │     ├─ out.as_mut_device_ptr_bf16("add_bf16:out")      [1 match + format! for tag]
                  │     │     ├─ LaunchConfig (grid/block calculation)            [arithmetic]
                  │     │     └─ f.launch(cfg, args)                             [cuLaunchKernel FFI]
                  │     └─ RETURN Tensor
                  ├─ CHECK requires_grad: self.requires_grad || other.requires_grad
                  ├─ IF requires_grad:
                  │     ├─ output.requires_grad = true
                  │     ├─ CALL AutogradContext::record_op(...)                  [SEE AUTOGRAD SECTION]
                  │     │     ├─ MUTEX LOCK: AUTOGRAD_CONTEXT                    [1 mutex acquire]
                  │     │     ├─ CHECK ctx.enabled                               [1 bool check]
                  │     │     ├─ MUTEX LOCK: CHECKPOINT_MANAGER                  [1 mutex acquire]
                  │     │     ├─ Vec::new() for saved_tensors                    [HEAP ALLOC - empty Vec]
                  │     │     ├─ Shape::clone() x2 for lhs_shape, rhs_shape      [2 Vec clones]
                  │     │     └─ tape.push(TapeEntry)                            [possible Vec realloc]
                  └─ RETURN Ok(output)
```

### Cost Breakdown (inference, no grad)

| Cost Type | Count | Notes |
|-----------|-------|-------|
| Function calls | 7 | add → add_bf16 → shapes_equal → launch_bf16_flat → ensure → get_func → launch |
| Mutex/lock acquisitions | 0 | No autograd when requires_grad=false |
| Heap allocations | 2 | 1 CUDA alloc (pooled), 1 Shape::clone (Vec) |
| Shape/dtype checks | 6 | 2 dtype checks in add(), 4 in shapes_equal_no_broadcast() |
| Match statements | 4 | 2 in dtype(), 2 in as_device_ptr_bf16() |
| Arc clones | 1 | device.clone() for output Tensor |
| HashMap lookups | 2 | ensure() + get_func() - **REDUNDANT** |
| String formatting | 3 | as_device_ptr_bf16 tag strings - **even on success path** |
| Atomic ops | 1 | TensorId::new() |

### Cost Breakdown (training, requires_grad=true)

Add to above:
| Cost Type | Count | Notes |
|-----------|-------|-------|
| Mutex acquisitions | 2 | AUTOGRAD_CONTEXT + CHECKPOINT_MANAGER |
| Heap allocations | 3 | Vec::new() for saved_tensors, 2x Shape::clone |
| Vec push | 1 | tape.push() |

**Minimal calls needed**: 3 (dtype check → alloc output → launch kernel)
**Actual calls (inference)**: 7
**Overhead factor**: 2.3x

---

## 2. Matrix Multiply

**User code**: `let c = a.matmul(&b)?;`

### Call Stack (BF16 path)

```
tensor.rs:1257  Tensor::matmul(&self, other)
                  ├─ trace_dtype_enabled()                               [1 OnceLock atomic load]
                  ├─ dtype mismatch check: self.dtype() != other.dtype()  [2 match stmts]
                  │    └─ format! string on error path only               [OK, lazy]
                  ├─ CALL ops::gemm::launch_gemm(self, other)
                  │     ops/gemm.rs:65
                  │     ├─ dtype mismatch check AGAIN                     [2 match stmts, REDUNDANT]
                  │     │    └─ format! on error path                      [OK]
                  │     ├─ compute_for_storage(lhs.dtype())               [1 match on DType → BF16AccF32]
                  │     ├─ MATCH mode → gemm_bf16_acc_f32(lhs, rhs)
                  │     │     ops/gemm.rs:183
                  │     │     ├─ shape checks: a_shape.len() != 2, etc.   [4 checks]
                  │     │     ├─ dimension mismatch check (k != k_rhs)    [1 check + format! on error]
                  │     │     ├─ zero-size check (m==0 || n==0)           [1 check]
                  │     │     └─ CALL cuda_ops_bf16::gemm_bf16(lhs, rhs, None)
                  │     │           cuda_ops_bf16.rs:1019
                  │     │           ├─ CALL strict::scope("cuda_ops.gemm_bf16", ...)
                  │     │           │     strict.rs:338 (non-strict build): body()  [no-op wrapper]
                  │     │           ├─ CALL validate_gemm_bf16_inputs(x, w, bias)
                  │     │           │     cuda_ops_bf16.rs:855
                  │     │           │     ├─ ensure_bf16(x, "gemm_bf16:x")         [1 dtype + 1 storage_dtype check]
                  │     │           │     ├─ ensure_bf16(w, "gemm_bf16:w")         [1 dtype + 1 storage_dtype check]
                  │     │           │     ├─ Arc::ptr_eq(x.device(), w.device())    [1 pointer compare]
                  │     │           │     ├─ shape rank checks (x_dims.len() != 2)  [2 checks + format!]
                  │     │           │     └─ K dimension match                      [1 check + format!]
                  │     │           ├─ strict_block_lt_fallback(m, n)               [no-op in non-strict]
                  │     │           ├─ Tensor::empty_dtype(shape, BF16, device)     [1 CUDA alloc via pool]
                  │     │           │     ├─ alloc_log check                        [1 OnceLock atomic]
                  │     │           │     ├─ TensorStorage::empty()                 [1 match + pool_alloc_u16]
                  │     │           │     └─ TensorId::new()                        [1 atomic]
                  │     │           ├─ CALL gemm_bf16_into_impl(&mut out, x, w, None, m, k, n)
                  │     │           │     cuda_ops_bf16.rs:930
                  │     │           │     ├─ ensure_bf16(out, "gemm_bf16:out")      [REDUNDANT - just created as BF16]
                  │     │           │     ├─ Arc::ptr_eq check                      [REDUNDANT - same device]
                  │     │           │     ├─ output shape validation                [REDUNDANT - just created with correct shape]
                  │     │           │     ├─ strict_block_lt_fallback(m, n)         [REDUNDANT - called 3 lines ago]
                  │     │           │     ├─ default_stream(x)                      [device.cuda_stream_raw_ptr()]
                  │     │           │     ├─ tensor_as_view_bf16(x, "gemm_bf16:x") [1 dtype check + shape iteration]
                  │     │           │     │     cuda_ops_ffi.rs:412
                  │     │           │     │     ├─ dtype check: tensor.dtype() != BF16  [REDUNDANT, 4th time]
                  │     │           │     │     ├─ shape loop to fill dims[8], strides[8]
                  │     │           │     │     └─ as_device_ptr_bf16(tag)              [1 match + format! tag]
                  │     │           │     ├─ tensor_as_view_bf16(w, "gemm_bf16:w") [same overhead]
                  │     │           │     ├─ tensor_as_view_bf16_mut(out, "gemm_bf16:out") [same overhead]
                  │     │           │     └─ UNSAFE fc_gemm_bf16(&vx, &vw, null, &mut vy, stream)
                  │     │           │           [C FFI → cuBLASLt GEMM]
                  │     │           └─ status check + possible fallback logging
                  ├─ CHECK requires_grad
                  ├─ IF requires_grad:
                  │     ├─ CALL AutogradContext::record_op(...)
                  │     └─ saved_tensors: vec![(self.id, self.clone()), (other.id, other.clone())]
                  │           ├─ 2x Tensor::clone()                     [2 GPU d2d copies! EXPENSIVE]
                  │           └─ 2x Arc::clone for device               [2 atomic increments]
                  └─ RETURN Ok(output)
```

### Cost Breakdown (inference, no grad)

| Cost Type | Count | Notes |
|-----------|-------|-------|
| Function calls | 10 | matmul → launch_gemm → gemm_bf16_acc_f32 → gemm_bf16 → scope → validate → empty_dtype → gemm_bf16_into_impl → 3x tensor_as_view_bf16 → fc_gemm_bf16 |
| Mutex/lock acquisitions | 0 | |
| Heap allocations | 2 | 1 CUDA alloc, 1 Shape |
| Shape/dtype checks | 17 | dtype mismatch (2) + dtype mismatch AGAIN (2) + compute_for_storage (1) + validate (5) + into_impl (3+1 redundant) + 3x view_bf16 (3) |
| Match statements | 10 | 4 dtype(), 1 compute_for_storage, 1 mode dispatch, 4 as_device_ptr_bf16 |
| Arc clones | 1 | device for output |
| HashMap lookups | 0 | cuBLASLt handle is OnceLock-cached |
| String formatting | 6 | 3 view_bf16 tags + 3 as_device_ptr_bf16 tags |
| Atomic ops | 2 | TensorId + trace flag |
| Redundant checks | 8 | dtype checked 4x, device ptr_eq 2x, strict_block 2x, output shape 1x |

### Cost Breakdown (training)

Add: 2 mutex locks, 2 Tensor clones (GPU d2d copy!), Vec alloc for saved_tensors.

**Minimal calls needed**: 4 (dtype check → validate shapes → alloc output → cuBLASLt call)
**Actual calls (inference)**: 10+
**Overhead factor**: 2.5x

---

## 3. RMS Norm

**User code**: `let out = rms_norm.forward(&input)?;`

### Call Stack (BF16 path)

```
norm.rs:1034   RMSNorm::forward(&self, input)
                 ├─ rank check + assert_nhwc_public (if rank==4)      [1 rank check]
                 ├─ trap_is_bf16("RMSNorm::forward in", input)        [1 dtype + 1 storage_dtype check]
                 ├─ normalized_shape validation loop                   [N iterations for N-dim norm]
                 ├─ CALL rms_norm_forward(input, &self.normalized_shape, weight, eps)
                 │     norm.rs:686
                 │     ├─ dtype check: input.dtype() != BF16           [REDUNDANT, just checked above]
                 │     ├─ storage dtype check: storage.dtype() != BF16 [REDUNDANT]
                 │     ├─ norm_size product                            [iter().product()]
                 │     ├─ divisibility check                           [1 modulo]
                 │     └─ CALL rms_norm_forward_bf16(input, weight, batch_size, norm_size, eps)
                 │           norm.rs:716
                 │           ├─ storage dtype check AGAIN               [REDUNDANT, 3rd time]
                 │           ├─ CudaKernels::ensure_kernel(...)         [HashMap-based kernel cache]
                 │           ├─ device.get_func(...)                    [HashMap lookup]
                 │           ├─ ALLOC: device.alloc::<u16>(numel)       [CUDA alloc]
                 │           ├─ alloc_zeros_from_pool(device, batch)    [F32 alloc for inv_rms]
                 │           ├─ Weight branch: weight dtype check       [REDUNDANT, 4th/5th check]
                 │           │    ├─ as_device_ptr_bf16("input")        [1 match + format! tag]
                 │           │    ├─ as_device_ptr_bf16("weight")       [1 match + format! tag]
                 │           │    └─ launch_kernel!(f, cfg, ...)        [cuLaunchKernel]
                 │           └─ CONSTRUCT output Tensor
                 ├─ dtype check on output (output.dtype() != BF16)     [REDUNDANT, just created as BF16]
                 ├─ trap_is_bf16("RMSNorm::forward out", &output)      [REDUNDANT]
                 ├─ IF needs_grad:
                 │     ├─ input.clone_result()                          [GPU d2d copy]
                 │     ├─ weight clone                                  [GPU d2d copy]
                 │     ├─ Construct inv_rms Tensor wrapper              [Tensor construction]
                 │     └─ AutogradContext::record_op(...)               [2 mutex locks]
                 └─ RETURN Ok(output)
```

### Cost Breakdown (inference)

| Cost Type | Count | Notes |
|-----------|-------|-------|
| Function calls | 6 | forward → rms_norm_forward → rms_norm_forward_bf16 → ensure_kernel → get_func → launch |
| Mutex/lock acquisitions | 0 | |
| Heap allocations | 3 | 2 CUDA allocs (output + inv_rms), 1 Shape |
| Dtype/storage checks | 8 | 3 redundant checks of BF16 dtype/storage |
| Match statements | 4 | dtype() x2 + as_device_ptr_bf16 x2 |
| HashMap lookups | 2 | ensure_kernel + get_func |
| String formatting | 2 | as_device_ptr_bf16 tags |
| Redundant checks | 5 | dtype checked 3 extra times, output dtype checked after creation |

**Minimal calls needed**: 4 (validate → alloc output → alloc inv_rms → launch kernel)
**Actual calls**: 6
**Overhead factor**: 1.5x

---

## 4. Linear Forward

**User code**: `let out = linear.forward(&input)?;`

### Call Stack (BF16 non-cuDNN, non-arena path)

```
linear.rs:177  Linear::forward(&self, input)
                 ├─ strict::scope("linear.forward", ...)               [closure wrapper, no-op non-strict]
                 ├─ rank check (if rank==4: assert_nhwc_public)        [1 check]
                 ├─ dtype + storage_dtype logging if not BF16           [log::error! even if BF16 - checked every call]
                 ├─ trap_is_bf16("Linear::forward in", input)          [1 dtype + 1 storage_dtype]
                 ├─ in_features dimension check                        [1 comparison]
                 ├─ try_forward_arena_fast_path(input, &input_shape)   [arena check, returns None if no arena]
                 │     ├─ ArenaScratch::from_tensor_with_align(...)    [constructor]
                 │     └─ forward_arena_with_scratch(...)               [more checks, returns None]
                 ├─ cuDNN check (if feature cudnn)                     [conditional compilation]
                 ├─ batch_size = product of leading dims               [iter().product()]
                 ├─ env check: FLAME_LINEAR_CLEAR_POOL                 [std::env::var SYSCALL!]
                 ├─ input.reshape(&[batch_size, in_features])          [Shape allocation + clone]
                 ├─ trap_is_bf16("Linear::forward weight", &self.weight) [REDUNDANT]
                 ├─ weight transpose: weight_t_cache or transpose2d_bf16()
                 │     ├─ If cached: cached.clone()                    [Tensor clone = GPU d2d copy if shared_storage]
                 │     └─ If not: reshape + transpose2d_bf16           [CUDA kernel launch]
                 ├─ CALL input_2d.matmul(&weight_t)                    [ENTIRE MATMUL STACK FROM #2]
                 ├─ IF bias:
                 │     ├─ bias dtype check                              [2 checks]
                 │     ├─ bias.reshape(&[1, out_features])             [Shape alloc]
                 │     └─ output.add(&bias_view)                       [ENTIRE ADD STACK FROM #1]
                 ├─ output.reshape(&output_shape)                      [Shape alloc]
                 ├─ output dtype check (if not BF16: to_dtype)         [REDUNDANT]
                 ├─ IF requires_grad:
                 │     ├─ 2-3x clone_result()                          [2-3 GPU d2d copies]
                 │     └─ AutogradContext::record_op(...)               [2 mutex locks]
                 ├─ trap_is_bf16("Linear::forward out", &output)       [REDUNDANT]
                 └─ RETURN Ok(output)
```

### Cost Breakdown (inference, no bias, cached weight_t)

| Cost Type | Count | Notes |
|-----------|-------|-------|
| Function calls | ~15 | forward → scope → trap x3 → arena_fast_path → reshape → matmul (10 internal) → reshape |
| Mutex/lock acquisitions | 0 | |
| Heap allocations | ~5 | 2-3 Shape allocs (reshape), 1 CUDA alloc (matmul output), weight_t clone |
| Dtype/storage checks | ~12 | trap x3 = 6 checks + matmul internal 6 |
| Syscalls | 1 | `std::env::var("FLAME_LINEAR_CLEAR_POOL")` - **UNCACHED SYSCALL ON EVERY LINEAR CALL** |
| Redundant checks | ~8 | trap_is_bf16 on output (known BF16), dtype checks in matmul |
| String formatting | ~8 | trap tags, matmul view tags |

**Critical finding**: `std::env::var("FLAME_LINEAR_CLEAR_POOL")` at `linear.rs:266` is an **uncached syscall** executed on every `Linear::forward()` call. A transformer with 100+ linear layers calls this 100+ times per forward pass.

**Minimal calls needed**: 4 (validate dims → transpose weight → cuBLASLt GEMM → add bias)
**Actual calls (inference, no bias)**: ~15
**Overhead factor**: 3.75x

---

## 5. SDPA (Scaled Dot-Product Attention)

**User code**: `let out = sdpa::forward(&q, &k, &v, None)?;`

### Call Stack (BF16 flash attention path, D=128, no mask)

```
sdpa.rs:94     sdpa::forward(q, k, v, mask)
                 ├─ strict::scope("sdpa.forward", ...)                 [closure wrapper]
                 ├─ CALL forward_inner(q, k, v, mask)
                 │     sdpa.rs:223
                 │     ├─ shape4(q), shape4(k), shape4(v)              [3x shape extraction, each 4 dims]
                 │     ├─ batch/head/dim validation                    [6 comparisons]
                 │     ├─ seq mismatch check                           [1 comparison]
                 │     ├─ mask shape validation (if mask)               [skipped]
                 │     ├─ trap_is_bf16(q), trap_is_bf16(k), trap_is_bf16(v)  [6 dtype checks]
                 │     ├─ dtype == BF16 check x3                       [3 match stmts, REDUNDANT after trap]
                 │     └─ CALL forward_bf16(q, k, v, mask, b, h, q_len, k_len, d_q)
                 │           sdpa.rs:376
                 │           ├─ scale = 1/sqrt(d_q)                    [arithmetic]
                 │           ├─ (d_q == 64||96||128) && mask.is_none() && use_flash_attn()
                 │           │     use_flash_attn(): OnceLock atomic load
                 │           ├─ CALL forward_flash_bf16(q, k, v, b, h, q_len, k_len, d_q)
                 │           │     sdpa.rs:460
                 │           │     ├─ bh = (b*h) as i32
                 │           │     ├─ device_lt::stream_ptr(device)     [raw stream extraction]
                 │           │     ├─ Tensor::empty_dtype(q.shape.clone(), BF16, device.clone())
                 │           │     │     ├─ Shape::clone()              [Vec clone]
                 │           │     │     ├─ Arc::clone()                [atomic]
                 │           │     │     ├─ pool_alloc_u16              [CUDA alloc]
                 │           │     │     └─ TensorId::new()             [atomic]
                 │           │     ├─ q.as_device_ptr_bf16("flash_attn:q")  [match + format!]
                 │           │     ├─ k.as_device_ptr_bf16("flash_attn:k")  [match + format!]
                 │           │     ├─ v.as_device_ptr_bf16("flash_attn:v")  [match + format!]
                 │           │     ├─ output.as_device_ptr_bf16("flash_attn:o") [match + format!]
                 │           │     └─ flame_flash_attention_bf16(...)   [C FFI → CUDA kernel]
                 ├─ trap_is_bf16("sdpa.forward out", &output)          [REDUNDANT]
                 └─ RETURN Ok(output)
```

### Cost Breakdown (flash path, inference)

| Cost Type | Count | Notes |
|-----------|-------|-------|
| Function calls | 6 | forward → forward_inner → forward_bf16 → forward_flash_bf16 → empty_dtype → kernel |
| Heap allocations | 2 | 1 CUDA alloc, 1 Shape clone |
| Dtype checks | 12 | 6 trap_is_bf16 + 3 dtype==BF16 + 3 REDUNDANT final trap |
| Match statements | 8 | 4 as_device_ptr_bf16, 4 dtype() |
| String formatting | 4 | Flash attention pointer tags |
| OnceLock reads | 1 | use_flash_attn() |
| Redundant checks | 6 | 3 dtype==BF16 after trap already confirmed; output trap |

### SDPA Materialized Fallback Path (much heavier)

When flash is not available (mask present, or d_q not 64/96/128), the fallback does:
- 2x reshape (Q,K,V to 3D)
- 1x transpose (K)
- 2x batched GEMM (QK^T, attn*V) each with full `bmm_bf16_fp32acc_out` overhead
- 1x to_dtype(F32) for softmax staging
- 1x mul_scalar (scale)
- 1x softmax
- 1x to_dtype(BF16) back
- Total: ~40+ function calls, 8+ allocations

**Minimal calls needed**: 3 (validate shapes → alloc output → launch kernel)
**Actual calls (flash path)**: 6
**Overhead factor (flash)**: 2.0x
**Overhead factor (materialized)**: ~10x+

---

## 6. Backward of MatMul

**User code**: Triggered by `AutogradContext::backward(&loss)`

### Call Stack

```
autograd.rs:1248  compute_gradients for Op::MatMul { lhs, rhs }
                    ├─ fetch_saved(lhs) → lookup in TapeEntry.saved_tensors
                    │     ├─ linear scan of Vec<(TensorId, Tensor)>    [O(n), n=2 typically]
                    ├─ fetch_saved(rhs)                                [same]
                    ├─ GpuOps::transpose(rhs_tensor)                   [FULL TRANSPOSE: alloc + kernel]
                    │     cuda_ops.rs:428
                    │     ├─ get_kernels() → MUTEX LOCK on KERNELS_CACHE  [HashMap lookup]
                    │     └─ kernels.transpose(tensor)                 [CUDA kernel launch]
                    ├─ GpuOps::matmul(output_grad, &rhs_t)            [FULL MATMUL from #2]
                    │     [10+ function calls, see matmul section]
                    ├─ GpuOps::transpose(lhs_tensor)                   [ANOTHER TRANSPOSE]
                    ├─ GpuOps::matmul(&lhs_t, output_grad)            [ANOTHER FULL MATMUL]
                    └─ RETURN vec![(*lhs, grad_lhs), (*rhs, grad_rhs)]
                         [Vec ALLOC with 2 entries]
```

### Cost Per MatMul Backward

| Cost Type | Count | Notes |
|-----------|-------|-------|
| Function calls | ~25 | 2x full matmul (10 each) + 2x transpose (~2 each) + bookkeeping |
| Mutex acquisitions | 2 | 2x KERNELS_CACHE for transpose |
| Heap allocations | ~8 | 2x transpose output, 2x matmul output, Shapes, result Vec |
| GPU d2d copies | 0 | Uses saved tensors directly |

**Minimal calls needed**: 4 (transpose B → grad_lhs GEMM → transpose A → grad_rhs GEMM)
**Actual calls**: ~25
**Overhead factor**: 6.25x

---

## 7. Gradient Accumulation

**User code**: Called during backward for each op.

### Call Stack

```
gradient.rs:203  GradientMap::accumulate(&mut self, id, grad)
                   ├─ IF grad.dtype() != F32:
                   │     grad.to_dtype(F32)                            [BF16→F32 conversion kernel]
                   ├─ self.resolve(id)                                 [Option<&CompactIndex>.and_then(idx.get)]
                   │     ├─ index.as_ref()                             [Option deref]
                   │     └─ idx.get(id) → HashMap<TensorId, usize> lookup  [HASH + compare]
                   ├─ IF vec_store[idx] exists:
                   │     ├─ existing.dtype() check                     [1 match]
                   │     ├─ existing.add(&grad)?                       [FULL ADD from #1]
                   │     │     [7 function calls including CUDA kernel launch]
                   │     └─ *existing = result                         [tensor replacement]
                   └─ ELSE:
                        └─ self.overflow.insert(id, grad)              [HashMap insert]
```

### Cost Per Accumulation

| Cost Type | Count | Notes |
|-----------|-------|-------|
| Function calls | ~9 | accumulate → to_dtype → resolve → HashMap::get → add (7 internal) |
| Heap allocations | ~3 | to_dtype output, add output, Shape |
| HashMap lookups | 1 | CompactIndex lookup |
| CUDA kernels | 2 | to_dtype conversion + add |

Note: The `to_dtype(F32)` conversion is called on EVERY accumulation when gradients come in as BF16. This allocates a new F32 tensor every time.

**Minimal calls needed**: 2 (resolve → in-place add)
**Actual calls**: ~9
**Overhead factor**: 4.5x

---

## 8. Adam Optimizer Step

**User code**: `adam.step(&parameters)?;`

### Call Stack (per parameter)

```
adam.rs:47     Adam::step(&mut self, parameters)
                ├─ t += 1                                              [1 increment]
                ├─ bias_correction1, bias_correction2                   [2 powi() calls]
                ├─ FOR EACH parameter with gradient:
                │     ├─ param.grad()                                   [Option<Tensor>]
                │     ├─ param.id()                                     [field access]
                │     ├─ config::select_optimizer_state_dtype()         [function call]
                │     ├─ IF grad.dtype() != state_dtype:
                │     │     grad.to_dtype(state_dtype)                  [CONVERSION KERNEL]
                │     ├─ IF weight_decay > 0:
                │     │     ├─ param.tensor()                           [field access]
                │     │     ├─ param_tensor.to_dtype(state_dtype)       [CONVERSION if needed]
                │     │     ├─ param_adjust.mul_scalar(wd)              [KERNEL: scalar mul]
                │     │     └─ grad = grad.add(&...)?                   [KERNEL: add]
                │     ├─ HashMap::entry(param_id) for m                 [HASH + lookup]
                │     │     └─ if Vacant: zeros_like_with_dtype()       [CUDA alloc, first step only]
                │     ├─ HashMap::entry(param_id) for v                 [HASH + lookup]
                │     │     └─ if Vacant: zeros_like_with_dtype()       [CUDA alloc, first step only]
                │     ├─ HashMap::get_mut for m                         [HASH + lookup, REDUNDANT after entry]
                │     ├─ HashMap::get_mut for v                         [HASH + lookup, REDUNDANT after entry]
                │     │
                │     │  — UPDATE m: m = m * beta1 + grad * (1-beta1) —
                │     ├─ m.mul_scalar(beta1)                            [KERNEL]
                │     ├─ grad.mul_scalar(1.0 - beta1)                   [KERNEL]
                │     ├─ m_scaled.add(&grad_scaled)                     [KERNEL]
                │     │     Total: 3 CUDA kernel launches + 3 allocations for m update
                │     │
                │     │  — UPDATE v: v = v * beta2 + grad² * (1-beta2) —
                │     ├─ grad.mul(&grad)                                [KERNEL]
                │     ├─ v.mul_scalar(beta2)                            [KERNEL]
                │     ├─ grad_sq.mul_scalar(1.0 - beta2)                [KERNEL]
                │     ├─ v_scaled.add(&grad_sq_scaled)                  [KERNEL]
                │     │     Total: 4 CUDA kernel launches + 4 allocations for v update
                │     │
                │     │  — COMPUTE update —
                │     ├─ m.div_scalar(bias_correction1)                 [KERNEL]
                │     ├─ v.div_scalar(bias_correction2)                 [KERNEL]
                │     ├─ v_hat.sqrt()                                   [KERNEL]
                │     ├─ v_sqrt.add_scalar(eps)                         [KERNEL]
                │     ├─ m_hat.div(&denominator)                        [KERNEL]
                │     ├─ ratio.mul_scalar(lr)                           [KERNEL]
                │     │     Total: 6 CUDA kernel launches + 6 allocations
                │     │
                │     └─ param.apply_update(&update)                    [KERNEL: param -= update]
```

### Cost Per Parameter Per Step

| Cost Type | Count | Notes |
|-----------|-------|-------|
| CUDA kernel launches | 13-15 | mul_scalar x4, add x2, mul x1, div_scalar x2, div x1, sqrt x1, add_scalar x1, apply_update x1 |
| Heap allocations | 13-15 | Each kernel creates a NEW tensor |
| HashMap lookups | 4 | 2x entry + 2x get_mut (REDUNDANT) |
| Function calls | ~100+ | 13 ops x ~7 calls each |

**Critical finding**: Every intermediate tensor is allocated, used once, then dropped. Zero in-place operations. A fused Adam CUDA kernel would replace 13-15 separate kernel launches with 1.

**Minimal calls needed**: 1 (single fused Adam kernel: read param, m, v, grad → write param, m, v)
**Actual calls**: ~100+
**Overhead factor**: 100x

---

## Autograd Cost Analysis

### `record_op` per-call cost (autograd.rs:485-512)

```
AutogradContext::record_op(output_id, op, saved_tensors):
  1. MUTEX LOCK: AUTOGRAD_CONTEXT.lock()              [Mutex<AutogradContextInner>]
  2. CHECK: ctx.enabled                                [bool read]
  3. MUTEX LOCK: CHECKPOINT_MANAGER.lock()             [2nd global mutex]
  4. FOR EACH saved tensor: mgr.checkpoint_saved_tensor(id, tensor)
       └─ match policy → Recompute: no-op             [1 match]
  5. ctx.record(TapeEntry { output_id, op, saved_tensors })
       └─ tape.push(entry)                            [Vec push, possible realloc]
  6. DROP both mutex guards
```

**Per-call cost**:
- 2 mutex lock/unlock cycles
- 1 enabled check
- N checkpoint policy matches (N = saved tensor count)
- 1 Vec push
- Saved tensors Vec itself was heap-allocated at call site

**TapeEntry size estimate**:
- `output_id: TensorId` = 8 bytes
- `op: Op` = varies, 16-80 bytes (Op::Add = ~40 bytes with 2 TensorId + 2 Shape)
- `saved_tensors: Vec<(TensorId, Tensor)>` = 24 bytes header
  - Each entry: 8 (TensorId) + ~100 (Tensor) = ~108 bytes
  - MatMul saves 2 = 216 bytes of tensor data (pointers, not GPU data)
- **Typical TapeEntry**: 48 + 216 = ~264 bytes for MatMul

### Does inference pay the autograd tax?

**Partially, but the hot path is cheap**:

```rust
// tensor.rs:1280
if self.requires_grad || other.requires_grad {
    // ... record_op only reached if requires_grad is true
}
```

For inference tensors where `requires_grad = false`:
- The `if` check is 2 bool reads (cheap)
- `record_op` is never called
- The `Vec::new()` for `saved_tensors` in the `Op::Add` variant IS still constructed at the call site... wait, no:

Looking more carefully at `tensor.rs:1690-1698`:
```rust
AutogradContext::record_op(
    output.id,
    Op::Add { lhs: self.id, rhs: other.id, lhs_shape: self.shape.clone(), rhs_shape: other.shape.clone() },
    Vec::new(),
);
```

This code is inside the `if self.requires_grad || other.requires_grad` block, so the `Vec::new()` and `Shape::clone()` calls only happen when gradients are needed. **Good.**

**However**: The `AUTOGRAD_CONTEXT` mutex and `CHECKPOINT_MANAGER` mutex are ALWAYS created as global statics (lazy_static). Their existence doesn't cost anything at runtime unless locked.

---

## No-Grad Path Analysis

**User code**: `let _guard = AutogradContext::no_grad();`

### NoGradGuard creation (autograd.rs:1112-1121)

```
NoGradGuard::new():
  1. AUTOGRAD_CONTEXT.lock()      [MUTEX LOCK]
  2. prev = ctx.enabled           [bool read]
  3. ctx.enabled = false           [bool write]
  4. DROP mutex guard
  RETURN NoGradGuard { prev_state: prev }
```

### NoGradGuard drop (autograd.rs:1124-1129)

```
Drop::drop(&mut self):
  1. AUTOGRAD_CONTEXT.lock()      [MUTEX LOCK]
  2. ctx.enabled = self.prev_state [bool write]
  3. DROP mutex guard
```

**Total cost**: 2 mutex lock/unlock cycles per no_grad scope.

### What changes in no_grad mode?

In `record_op`:
```rust
if !ctx.enabled {
    return;  // Early exit after 1 mutex lock
}
```

So with `no_grad()`, `record_op` still:
1. Acquires the AUTOGRAD_CONTEXT mutex
2. Checks `ctx.enabled`
3. Returns immediately

**The mutex lock still happens on every op that checks requires_grad and calls record_op.** But since inference tensors have `requires_grad = false`, `record_op` is never even called. The `no_grad()` guard is only useful when you have tensors with `requires_grad = true` but don't want to record during certain operations.

**Conclusion**: For pure inference (no tensor has `requires_grad = true`), the remaining overhead per op is just the `if self.requires_grad || other.requires_grad` boolean check. **No mutex cost.**

---

## Summary Table

| Operation | Actual calls | Minimal calls | Overhead factor | Biggest waste |
|-----------|-------------|---------------|-----------------|---------------|
| BF16 Add (inference) | 7 | 3 | 2.3x | Redundant HashMap lookup in ensure+get_func; string formatting for tags |
| MatMul (inference) | 10+ | 4 | 2.5x | dtype checked 4x; device ptr_eq checked 2x; strict_block 2x |
| RMS Norm (inference) | 6 | 4 | 1.5x | dtype re-checked 3x after initial trap |
| Linear Forward (inference) | ~15 | 4 | 3.75x | **Uncached env::var syscall**; matmul overhead; redundant traps |
| SDPA Flash (inference) | 6 | 3 | 2.0x | 6 redundant dtype checks after trap |
| SDPA Materialized (inference) | ~40+ | ~8 | 5.0x | Multiple alloc/free cycles for intermediate tensors |
| MatMul Backward | ~25 | 4 | 6.25x | 2x full matmul overhead + 2x transpose |
| Gradient Accumulation | ~9 | 2 | 4.5x | BF16→F32 conversion on every call; add creates new tensor |
| Adam Step (per param) | ~100+ | 1 | 100x | 13-15 separate kernel launches instead of 1 fused kernel |

---

## Ranked Fixes (Highest Impact First)

### 1. Fused Adam CUDA Kernel
**Impact**: ~100x overhead reduction per parameter per step
**Effort**: Medium (1-2 days)
**Location**: `adam.rs:47-124`

Currently launches 13-15 separate CUDA kernels per parameter, each allocating a new tensor. A single fused kernel reading `(param, m, v, grad)` and writing `(param, m, v)` in one pass would:
- Eliminate ~14 kernel launches → 1
- Eliminate ~14 temporary tensor allocations → 0
- Reduce Adam from ~100 function calls to ~5
- Save ~28 GPU memory allocator round-trips per parameter per step

### 2. Cache `FLAME_LINEAR_CLEAR_POOL` env var
**Impact**: Eliminates 1 syscall per Linear::forward() call
**Effort**: Trivial (5 minutes)
**Location**: `linear.rs:266`

```rust
// CURRENT (syscall every call):
if std::env::var("FLAME_LINEAR_CLEAR_POOL").ok().as_deref() == Some("1") {

// FIX:
static CLEAR_POOL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
if *CLEAR_POOL.get_or_init(|| std::env::var("FLAME_LINEAR_CLEAR_POOL").ok().as_deref() == Some("1")) {
```

For a model with 100 linear layers and 20 steps, this saves 2000 syscalls per forward pass.

### 3. Eliminate redundant dtype checks in gemm_bf16 pipeline
**Impact**: Removes ~8 redundant checks per matmul
**Effort**: Low (30 minutes)
**Location**: `cuda_ops_bf16.rs:930-961` (`gemm_bf16_into_impl`)

`gemm_bf16_into_impl` re-validates the output tensor it JUST created:
- `ensure_bf16(out)` — output was created 2 lines above as BF16
- `Arc::ptr_eq` — output uses the same device
- Output shape check — shape was set at creation
- `strict_block_lt_fallback` — called 3 lines above

Create an `_unchecked` internal variant or use `debug_assert!` for these.

### 4. Eliminate string formatting in `as_device_ptr_bf16` tags
**Impact**: Removes 2-6 `format!` calls per operation
**Effort**: Low (1 hour)
**Location**: `tensor.rs:511-527`, `cuda_ops_ffi.rs:412-432`

The `tag: &str` parameter in `as_device_ptr_bf16` is used only in error messages, but string literals are passed (not formatted). The issue is that callers like `bf16_elementwise.rs:378` use:
```rust
a.as_device_ptr_bf16(&format!("{}:a", op_name))?
```
This `format!` runs unconditionally even on success. Change to static `&str` tags or use a closure-based error message.

### 5. Merge `ensure` + `get_func` in elementwise kernels
**Impact**: Eliminates 1 redundant HashMap lookup per elementwise op
**Effort**: Trivial (15 minutes)
**Location**: `bf16_elementwise.rs:323-337, 371-375`

`ensure()` calls `device.get_func()` to check if kernel exists, then the caller immediately calls `device.get_func()` again. Combine into a single function that returns the `CudaFunction`.

### 6. In-place gradient accumulation
**Impact**: Eliminates 1 allocation + 1 kernel launch per accumulation
**Effort**: Medium (2-3 hours)
**Location**: `gradient.rs:203`

Currently `accumulate` calls `existing.add(&grad)` which allocates a NEW tensor, then replaces the old one. A fused in-place `axpy` kernel (existing += grad) would halve memory traffic and eliminate the allocation.

The infrastructure already exists: `cuda_ops_bf16::axpby_bf16` at `cuda_ops_bf16.rs:258` does `y = a*x + b*y` in-place. Wire gradient accumulation through this.

### 7. Skip BF16→F32 conversion in gradient accumulation
**Impact**: Eliminates 1 conversion kernel per accumulation when grad is BF16
**Effort**: Low (30 minutes)
**Location**: `gradient.rs:205-208`

Currently forces every incoming gradient to F32. If the accumulator is already F32 and the grad is BF16, could use a fused BF16→F32 accumulate kernel instead of separate convert + add.

### 8. Remove `trap_is_bf16` on outputs known to be BF16
**Impact**: Eliminates 2-4 dtype checks per norm/linear/sdpa call
**Effort**: Trivial (15 minutes)
**Location**: `norm.rs:1074`, `linear.rs:344`, `sdpa.rs:97`

When a function creates an output tensor as `empty_dtype(..., BF16, ...)` or `zeros_dtype(..., BF16, ...)`, the immediately-following `trap_is_bf16("..out", &output)` is provably redundant. Replace with `debug_assert!` or remove entirely.

### 9. Remove duplicate CHECKPOINT_MANAGER lock from record_op
**Impact**: Eliminates 1 mutex lock/unlock per recorded op
**Effort**: Low (15 minutes)
**Location**: `autograd.rs:497-504`

`record_op` acquires `CHECKPOINT_MANAGER` on every call, but `CheckpointPolicy::Recompute` (the default) is a no-op. Add a fast-path check:
```rust
if CHECKPOINT_HAS_ENTRIES.load(Ordering::Relaxed) {
    // Only then lock the manager
}
```
Wait — `CHECKPOINT_HAS_ENTRIES` is only set to true when `CPUOffload` policy is active and an activation is saved. But `checkpoint_saved_tensor` always runs the match and returns. The fast path should skip the entire block when policy is `Recompute`:

```rust
// Skip checkpoint manager entirely when no checkpoint policy is active.
// CHECKPOINT_HAS_ENTRIES is false initially and only set by CPUOffload.
if !saved_tensors.is_empty() && CHECKPOINT_HAS_ENTRIES.load(Ordering::Relaxed) {
    if let Ok(mut mgr) = CHECKPOINT_MANAGER.lock() { ... }
}
```

Actually, looking more carefully: `CHECKPOINT_HAS_ENTRIES` starts as `false` and the lock acquires CHECKPOINT_MANAGER regardless. The fix is to gate on the atomic flag first. But currently the flag is only set INSIDE `checkpoint_saved_tensor` (after locking), so for the initial lock we need a different guard — e.g., a separate `CHECKPOINT_POLICY_ACTIVE` atomic set when policy changes from `Recompute`.

### 10. Pre-transpose weight in matmul backward
**Impact**: Saves 1 transpose + kernel per backward pass when weights are reused
**Effort**: Medium (1-2 hours)
**Location**: `autograd.rs:1248-1264`

MatMul backward transposes the saved tensors on every backward call. For parameters (weights), the transpose could be cached alongside the saved tensor in the TapeEntry.
