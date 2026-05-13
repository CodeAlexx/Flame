# Autograd v2 Design Review Handoff

Date: 2026-05-13

Scope reviewed:
- Proposed `flame-core/src/autograd_v2/` clean-sheet autograd design.
- Current flame-core repo at `/home/alex/EriDiffusion/flame-core`.
- Local PyTorch reference at `/home/alex/pytorch`, commit `6a13735`.

## Summary

The proposed PyTorch-style DAG architecture is directionally sound: `GradFn`/`Edge`, dependency-counted scheduling, `InputBuffer`, saved tensor version checks, and a single-threaded engine are the right foundations for replacing the current global tape.

Do not green-light implementation from the current spec as-is. Several details conflict with flame-core's current tensor/gradient ownership model and with PyTorch's actual lifetime rules. If implemented literally, v2 risks reference cycles, incomplete error propagation, broken multi-input node scheduling, missed in-place mutation detection, and optimizer-policy regressions.

The design should be revised first, then implemented behind a feature flag.

## Current Flame-Core Reality

Current `Tensor` is a by-value struct with direct fields:
- `storage`
- `shape`
- `device`
- `id`
- `requires_grad`
- `custom_strides`
- `view_offset`

Relevant files:
- `src/tensor.rs`
- `src/gradient.rs`
- `src/parameter.rs`
- `src/autograd.rs`
- `src/saved_ref.rs`
- `src/tensor_storage.rs`
- `src/autograd_v4/`

There is no shared `AutogradMeta` today. Gradients are not stored on `Tensor`; they live in `GradientMap` and `Parameter`.

Current gradient policy is `InternalFP32_PublicBF16`:
- `GradientMap::set_ones()` seeds F32.
- `GradientMap::insert()` enforces F32.
- `GradientMap::accumulate()` converts to F32 on repeated accumulation.
- `Parameter::set_grad()` casts incoming grads to F32.
- Adam/SGD comments and fast paths assume F32 grads in public optimizer flow.

Current version checking already exists through `SavedRef`:
- `SavedRef` stores an `Arc<AtomicU32>` version-counter handle.
- The version side table can be cleared at step boundaries.
- The saved handle keeps checks valid after table clears.

## Blocking Design Issues

### 1. AccumulateGrad Lifetime Cycle

The proposal stores:

```rust
AutogradMeta {
    grad_accumulator: Option<Arc<AccumulateGrad>>,
}

AccumulateGrad {
    variable: Tensor,
}
```

If `Tensor` owns or shared-owns `AutogradMeta`, this creates:

```text
Tensor -> AutogradMeta -> Arc<AccumulateGrad> -> Tensor
```

PyTorch avoids this by storing the leaf accumulator weakly in `AutogradMeta`; `grad_accumulator()` locks or recreates it lazily.

Required revision:
- Store `Weak<AccumulateGrad>` or `Weak<dyn GradFn>` in tensor metadata, not a strong `Arc`.
- Or redesign `AccumulateGrad` to hold a weak handle to a shared tensor inner/meta object.
- Do not put a strong `Tensor` cycle through metadata.

PyTorch references:
- `/home/alex/pytorch/torch/csrc/autograd/variable.h`
- `/home/alex/pytorch/torch/csrc/autograd/variable.cpp`
- `/home/alex/pytorch/torch/csrc/autograd/functions/accumulate_grad.h`

### 2. AutogradMeta Does Not Fit Current Tensor

The spec assumes every `Tensor` can expose mutable autograd metadata:

```rust
tensor.autograd_meta_mut()
```

Current flame-core `Tensor` is cloned by value. Adding plain fields directly to `Tensor` will not preserve:
- `.grad`
- `grad_fn`
- `output_nr`
- leaf accumulator identity
- leaf vs non-leaf metadata consistency

Required revision:
- Introduce a shared, interior-mutable metadata handle, for example:

```rust
pub(crate) autograd: Arc<Mutex<AutogradMetaV2>>
```

or a split `TensorInner` structure where tensor data and autograd metadata have explicit ownership rules.

Open choice:
- If `Tensor::clone()` should preserve autograd metadata, use shared metadata.
- If `detach()` should drop history, it must allocate/reset metadata deliberately.
- View tensors need metadata that can represent base/view relationships later, even if full view autograd is deferred.

### 3. SavedTensor Must Return Errors Through Backward

The proposal has:

```rust
SavedTensor::unpack(...) -> Result<Tensor, AutogradError>
GradFn::apply(...) -> Vec<Option<Tensor>>
backward(...) -> ()
```

This cannot propagate version mismatch or released-saved-tensor errors. Examples call `unwrap()`, which would panic instead of returning a recoverable training error.

Required revision:

```rust
pub trait GradFn: Send + Sync + Debug {
    fn apply(&self, inputs: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>>;
}

Engine::execute(...) -> Result<Vec<Option<Tensor>>>
backward(...) -> Result<()>
grad(...) -> Result<Vec<Option<Tensor>>>
```

Also, `release_variables(&self)` cannot call `SavedTensor::reset(&mut self)` through `Arc<dyn GradFn>`.

Required revision:
- Saved tensors inside backward nodes need interior mutability, such as `Mutex<Option<Tensor>>`, `parking_lot::Mutex`, or a small custom cell if single-threaded.
- `release_variables()` should be able to clear saved data through `&self`.

### 4. SavedTensor Version Handles Must Survive Side-Table Clears

The proposal stores only:

```rust
saved_version: u32
```

and later calls `data.storage().version()`.

Current flame-core version counters are in a global side table. `TensorStorage::version()` returns `0` if the table entry is missing. `AutogradContext::clear()` flushes the side table.

Required revision:
- Either move the version counter into a shared storage object, or
- Have `SavedTensor` store the `Arc<AtomicU32>` handle exactly like `SavedRef`.

Do not regress from the current `SavedRef` model.

Relevant files:
- `src/saved_ref.rs`
- `src/tensor_storage.rs`
- `src/autograd.rs`

### 5. In-Place Version Bump Coverage Is Not Complete

The design depends on every in-place mutation bumping the storage version. The current code has in-place write paths that need explicit audit and likely fixes:

- `src/ops/elt.rs`
  - `add_inplace_same_dtype`
  - `mul_inplace_same_dtype`
  - `gate_mul_bf16_inplace`
- `src/tensor.rs`
  - `copy_bf16_region_from`
  - `copy_f32_from`
  - `copy_`
  - `copy_from_bf16_slice`
  - chunk/copy helpers near the BF16 host-copy code
- `src/ops/multi_tensor.rs`
  - `multi_tensor_scale_inplace_packed`
- optimizer update paths
- SDPA/dropout in-place helpers

Required revision:
- Make an "all in-place writes bump version" sweep a prerequisite for claiming SavedTensor correctness.
- Add tests that save a tensor, mutate it through each public in-place path, then assert backward/unpack returns an error.

### 6. GradFn Needs Input Arity

PyTorch creates `InputBuffer(next.function->num_inputs())` when routing a gradient to a downstream node.

The proposed `GradFn` has `next_edges()` but no way for the engine to know the size of the downstream input buffer.

Required revision:

```rust
fn num_inputs(&self) -> usize;
```

or equivalent metadata.

Without this, multi-input gradient functions and multi-output/view-like nodes cannot be scheduled safely.

### 7. BF16 Grad Storage Is Bigger Than Autograd

The proposal's dtype contract says:

```text
param dtype BF16 => stored grad BF16
```

This conflicts with existing optimizer and parameter behavior:
- `GradientMap` is internal F32.
- `Parameter::set_grad()` casts all grads to F32.
- Adam/SGD paths document and optimize for BF16 params with F32 grads.

Required revision:
- Treat BF16 grad storage as an optimizer/parameter/autograd migration, not only an autograd policy.
- Decide whether `Parameter::set_grad()` preserves dtype under `autograd_v2`.
- Add optimizer parity and performance tests for BF16 grads before using this as a Class A fix.
- Update `ops::grad_norm` and clipping expectations accordingly.

Possible staged path:
1. Make v2 internal grads match the parameter dtype. For BF16 parameters, v2 stores BF16 gradients end-to-end. This is the Class A recovery path. Keeping F32 internal grads would match v1 behavior but would not fix Class A, so it should not be the v2 default unless the project explicitly abandons the Class A goal.
2. Add optimizer support for BF16 grads.
3. Gate BF16-grad optimizer paths by feature/env.
4. Run per-model parity before flipping default.

### 8. `create_graph` Is Over-Promised

The spec exposes `create_graph`, and `InputBuffer` has special handling for it, but the proposed v0 explicitly omits reentrant backward and higher-order complexity.

Required revision:
- For v0, return `Unsupported` if `create_graph=true`.
- Remove create-graph-specific behavior from the initial engine until there is a complete higher-order design.

### 9. Migration Plan Understates Existing Recording Surface

Current recording is not centralized through `gradient_edge()`. Many forward ops directly call:

```rust
AutogradContext::record_op(...)
```

The migration section needs a concrete per-op dispatch plan:

```rust
#[cfg(feature = "autograd_v2")]
record_v2(...)

#[cfg(not(feature = "autograd_v2"))]
AutogradContext::record_op(...)
```

Required revision:
- Add `autograd_v2 = []` to `Cargo.toml`.
- Add `pub mod autograd_v2` behind the feature in `src/lib.rs`.
- Define a shared policy for forward wrappers:
  - old tape path
  - v2 graph path
  - no-grad/inference path
- Do not assume adding `Tensor::gradient_edge()` automatically migrates existing ops.

### 10. Class B And E Performance Work Are Separate Workstreams

Autograd v2 should not be credited with closing the full step-time gap. The rewrite can address the Class A gradient-storage policy if BF16 grads are accepted across autograd, parameters, optimizers, and grad clipping. It does not automatically remove the Class B narrow-backward F32 detour or the broader Class E cost from cudarc synchronous copies.

Known Class B scope:
- The current narrow backward path in `src/tensor_narrow.rs` still has F32 assumptions and F32 storage expectations around the scatter-add path.
- If v2 routes narrow backward through the existing kernel path unchanged, it inherits the Class B detour.
- Fixing Class B requires a BF16-in/BF16-out narrow backward kernel or dispatcher path that keeps F32 accumulation inside the kernel and returns the input dtype.

Known Class E scope:
- The majority of `*_sync_copy*` sites live outside autograd.
- Primary sweep areas are `src/tensor.rs`, `src/tensor_storage.rs`, and `src/cuda_gradient_ops.rs`.
- Autograd v2 may remove or avoid a small number of scalar/tape syncs, but the large sync-copy win is a parallel performance workstream.

Required revision:
- State explicitly that v2 is a correctness and architecture rewrite with a possible Class A recovery.
- Track Class B narrow-backward dtype work independently unless the v2 narrow backward implementation replaces the existing F32 detour.
- Track Class E sync-site elimination independently.
- Do not gate v2 correctness work on Class B or Class E, and do not count Class B/E savings in v2-only performance projections.

## Recommended Spec Changes Before Implementation

Revise the design doc with these concrete changes:

1. Replace strong `grad_accumulator` metadata with weak accumulator storage.
2. Add a shared, interior-mutable autograd metadata design for `Tensor`.
3. Change all engine and `GradFn::apply` APIs to return `Result`.
4. Make `SavedTensor` hold a version-counter handle, not just a version integer.
5. Make saved tensor release work from `&self`.
6. Add `GradFn::num_inputs()`.
7. Remove or reject `create_graph=true` in v0.
8. Define BF16 grad migration across `GradientMap`, `Parameter`, Adam, SGD, grad norm, and trainer code.
9. Make the in-place version-bump sweep a hard prerequisite.
10. Add compile-time feature wiring and per-op forward migration rules.
11. Carve Class B narrow-backward dtype work and Class E sync-site elimination out as separate performance workstreams.

## Suggested Implementation Order

### Phase 0: Prerequisites

- Audit all in-place writes and add `storage.bump_version()`.
- Add tests for version mismatch on saved tensors.
- Decide and document BF16-grad optimizer behavior.
- Add `autograd_v2` feature flag only; no behavior change.

### Phase 1: Metadata and Core Types

- Add shared `AutogradMetaV2`.
- Add `Edge`, `GradFn`, `NodeId`, sequence number, topological number.
- Add weak leaf accumulator cache.
- Add `SavedTensor` using the existing version-handle model.
- Add `InputBuffer` with `Option<Tensor>` and `num_inputs`.
- `InputBuffer::add` must keep PyTorch's in-place accumulation fast path: if `create_graph=false`, dtype/shape match, and the existing buffered grad has unique storage ownership, accumulate in-place. Otherwise accumulate out-of-place. Since v0 should reject `create_graph=true`, the first implementation can still carry the flag but should test the `false` fast path explicitly.

### Phase 2: Engine Skeleton

- Implement `GraphRoot`.
- Implement `AccumulateGrad`.
- Implement dependency counting.
- Implement ready queue.
- Return `Result` everywhere.
- Reject `create_graph=true` with a clear error.

Toy tests:
- single leaf sum
- two branches into one leaf
- diamond graph accumulation
- undefined grad slots
- released saved tensor error
- version mismatch error

### Phase 3: First Real Ops

Start with:
- add
- mul
- sum
- reshape
- transpose
- matmul/linear
- silu
- layer_norm

Each op needs:
- forward wiring under `autograd_v2`
- backward struct
- PyTorch fixture parity
- dtype assertion
- no unwanted `.to(F32)` in autograd_v2

Long-tail unary ops:
- Do not block v0 on a code generator.
- Keep hand-written backward structs for the P0/P1 path.
- Permit a later `derivatives!` proc macro for the long tail (`sin`, `cos`, `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `neg`, `pow`, etc.) once the trait, saved tensor, and forward-wrapper patterns have stabilized.

### Phase 4: Optimizer and Trainer Integration

- Route v2 grads into parameters.
- Ensure optimizer accepts the selected grad dtype.
- Add grad-norm and clipping tests.
- Run one-step model parity before long parity runs.

### Phase 5: Parity Gate

Do not retire v1 until:
- per-op fixture parity passes for all P0/P1 ops
- model parity passes for the target model table
- no ms/step regression on klein 4B
- BF16 grad policy has optimizer parity
- in-place mutation tests are green

## High-Risk Areas To Watch

- View metadata and version sharing.
- `Tensor::clone()`, `detach()`, `to_dtype()`, `contiguous()`, and `narrow()` metadata behavior.
- Checkpoint/offload interaction with v2 saved tensors.
- SDPA backward saved-output mutation detection.
- Multi-output ops and `output_nr`.
- Gradient accumulation across micro-batches.
- Existing trainers that expect `GradientMap` return values rather than `.grad` fields.
- CUDA graph capture/replay currently integrated into old backward.

## Decision

Recommendation: revise and narrow the v2 design before coding.

Acceptable "go" condition:
- The blockers above are reflected in the spec.
- BF16 gradient policy is explicitly accepted as a cross-cutting optimizer migration.
- In-place version bump audit is tracked as prerequisite work.
- v0 explicitly rejects unsupported higher-order behavior.
- Implementation starts with a tiny graph and parity harness, not the full op table.
