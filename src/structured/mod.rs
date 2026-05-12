//! Phase 4 — structured-kernel pattern (PyTorch's `meta + impl_` split).
//!
//! PyTorch references:
//!   - `aten/src/ATen/native/cuda/UnaryGpuKernels.cu` (silu impl)
//!   - `aten/src/ATen/native/Activation.cpp` (silu meta)
//!
//! Plain summary of the pattern:
//!   1. `meta(input)` runs once: validates input, computes output shape/dtype,
//!      **pre-allocates the output Tensor** (so the impl doesn't reallocate
//!      every call).
//!   2. `impl_(input, output)` runs the kernel writing directly into the
//!      pre-allocated output.
//!   3. The autograd wrapper around the pair records the op for backward;
//!      if `requires_grad=false` it skips the recording entirely.
//!
//! The advantage over the current `Tensor::silu` → `dispatch_unary_bf16` →
//! `silu_bf16_iter` → `TensorIteratorBase::build_unary_op(None, x)` chain is
//! that allocation is hoisted out of the iterator's hot path and given a
//! single explicit seam (`StructuredKernel::meta`). Future phases can swap
//! the alloc strategy (e.g. pre-allocated reusable scratch, caller-supplied
//! output) by replacing `meta` without touching `impl_`.
//!
//! Phase 4 ships **silu only** as exemplar. All other ops keep going through
//! their existing `Tensor::*` methods unchanged.

pub mod kernel;
pub mod silu;

pub use kernel::StructuredKernel;
pub use silu::SiluStructured;
