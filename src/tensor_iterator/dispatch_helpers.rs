//! Phase 10 dispatch helpers for `Tensor::<op>` method bodies.
//!
//! After Phases 4–9 migrated 27 elementwise ops to the TensorIterator
//! pipeline, each `Tensor::<op>` method still carried a ~15–25-line
//! dtype-switch dispatch block. The shape was uniform: BF16+BF16 routed
//! through `crate::ops::<op>_iter`, everything else fell back to
//! `GpuOps::<op>`, all wrapped in `#[cfg(feature = "cuda")]` gating.
//!
//! These helpers collapse that boilerplate to a single call per method.
//! They do **not** touch autograd — tape bookkeeping stays at the method
//! site, byte-for-byte identical to the pre-Phase-10 version. The plan's
//! R4 gate (autograd saves must not move) is enforced by that split:
//! helpers own dispatch, methods own the tape.
//!
//! Four helper shapes, one per operand pattern found in the 27-op set:
//!   * `dispatch_binary_bf16`      — add, sub, mul, div
//!   * `dispatch_unary_bf16`       — silu, gelu, square, abs, relu, neg,
//!                                   sigmoid, tanh, exp, log, sqrt, rsqrt,
//!                                   recip
//!   * `dispatch_scalar_bf16`      — mul_scalar, add_scalar
//!   * `dispatch_comparison_bf16`  — ge, gt, le, lt, eq, ne, and the
//!                                   binary maximum/minimum which share
//!                                   the two-tensor signature
//!
//! Each variant dispatches to the iter path when both operands (or the
//! single operand for unary/scalar) are BF16, and to the caller-supplied
//! fallback otherwise. The fallback is typically a `GpuOps::<op>`
//! associated fn — GpuOps itself may further route BF16 to the same iter
//! function (Phases 6–9 already did that at that layer), but in the
//! `Tensor::*` methods the explicit BF16 short-circuit preserves the
//! pre-Phase-10 contract: BF16+BF16 never leaves the TensorIterator
//! pipeline on its way through a Tensor method.
//!
//! Non-cuda builds still reach these helpers; the iter functions
//! themselves return an `Error::InvalidOperation` on the non-cuda cfg
//! branch (see e.g. `ops::add_iter::add_bf16_kernel`). That preserves
//! pre-Phase-10 behavior where `Tensor::add(BF16, BF16)` errored out
//! without cuda.

use crate::{DType, Result, Tensor};

/// Route a binary op through the TensorIterator pipeline for BF16+BF16,
/// else fall back. `bf16_fn` is typically one of the `*_bf16_iter`
/// functions in `crate::ops::*_iter`; `fallback_fn` is typically a
/// `GpuOps::<op>` associated fn for F32 / mixed-dtype inputs.
#[inline]
pub fn dispatch_binary_bf16<F1, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    bf16_fn: F1,
    fallback_fn: F2,
) -> Result<Tensor>
where
    F1: FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
    F2: FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
{
    if lhs.dtype() == DType::BF16 && rhs.dtype() == DType::BF16 {
        bf16_fn(lhs, rhs)
    } else {
        fallback_fn(lhs, rhs)
    }
}

/// Route a unary op through the TensorIterator pipeline for BF16,
/// else fall back. Used by silu/gelu/square/abs/relu/neg/sigmoid/tanh/
/// exp/log/sqrt/rsqrt/recip.
#[inline]
pub fn dispatch_unary_bf16<F1, F2>(
    x: &Tensor,
    bf16_fn: F1,
    fallback_fn: F2,
) -> Result<Tensor>
where
    F1: FnOnce(&Tensor) -> Result<Tensor>,
    F2: FnOnce(&Tensor) -> Result<Tensor>,
{
    if x.dtype() == DType::BF16 {
        bf16_fn(x)
    } else {
        fallback_fn(x)
    }
}

/// Route a scalar op (one tensor + one f32) through the TensorIterator
/// pipeline for BF16, else fall back. Used by mul_scalar and add_scalar
/// (both PyTorch's `opmath_gpu_kernel_with_scalars` shape).
#[inline]
pub fn dispatch_scalar_bf16<F1, F2>(
    x: &Tensor,
    scalar: f32,
    bf16_fn: F1,
    fallback_fn: F2,
) -> Result<Tensor>
where
    F1: FnOnce(&Tensor, f32) -> Result<Tensor>,
    F2: FnOnce(&Tensor, f32) -> Result<Tensor>,
{
    if x.dtype() == DType::BF16 {
        bf16_fn(x, scalar)
    } else {
        fallback_fn(x, scalar)
    }
}

/// Route a comparison op through the TensorIterator pipeline for
/// BF16+BF16, else fall back. The iter path writes BF16 0.0/1.0
/// sentinels (bit-exact with PyTorch's `opmath_t=float` semantics); the
/// fallback `GpuOps::cmp_*` goes through the F32 round-trip.
#[inline]
pub fn dispatch_comparison_bf16<F1, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    bf16_fn: F1,
    fallback_fn: F2,
) -> Result<Tensor>
where
    F1: FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
    F2: FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
{
    // Structurally the same as dispatch_binary_bf16, but kept as a
    // distinct function so that a future change to comparison dispatch
    // (e.g. adding type promotion for BF16-vs-F32 compares) doesn't
    // silently alter the binary-arith path.
    if lhs.dtype() == DType::BF16 && rhs.dtype() == DType::BF16 {
        bf16_fn(lhs, rhs)
    } else {
        fallback_fn(lhs, rhs)
    }
}
