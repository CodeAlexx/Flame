//! Comparison BF16 ops: ge, gt, le, lt, eq, ne.
//!
//! All route through `build_comparison_op` + `launch_gpu_kernel<2, ...>`.
//!
//! Output dtype divergence from PyTorch: flame-core writes BF16 0.0/1.0
//! sentinels instead of `kBool` bytes. See the long comment on
//! `TensorIteratorBase::build_comparison_op` in
//! `src/tensor_iterator/base.rs` for the rationale.
//!
//! NaN semantics: IEEE 754 (eq(NaN, NaN) = false, ne(NaN, NaN) = true —
//! matches PyTorch).
//!
//! Content relocated verbatim from
//! `src/ops/{ge,gt,le,lt,eq,ne}_iter.rs` in Phase 11.
//!
//! PyTorch reference: `aten/src/ATen/native/cuda/CompareKernels.cu`.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

// =============================================================================
// ge (greater-or-equal)
// =============================================================================

crate::declare_stub!(pub GE_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn ge_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_ge_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_ge_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn ge_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "ge_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn ge_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "ge_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "ge_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_comparison_op(None, a, b)?;
    ge_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_ge_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// gt (greater-than)
// =============================================================================

crate::declare_stub!(pub GT_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn gt_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_gt_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_gt_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn gt_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "gt_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn gt_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "gt_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "gt_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_comparison_op(None, a, b)?;
    gt_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_gt_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// le (less-or-equal)
// =============================================================================

crate::declare_stub!(pub LE_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn le_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_le_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_le_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn le_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "le_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn le_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "le_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "le_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_comparison_op(None, a, b)?;
    le_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_le_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// lt (less-than)
// =============================================================================

crate::declare_stub!(pub LT_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn lt_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_lt_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_lt_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn lt_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "lt_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn lt_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "lt_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "lt_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_comparison_op(None, a, b)?;
    lt_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_lt_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// eq (equal) — eq(NaN, NaN) = false (IEEE 754, matches PyTorch).
// =============================================================================

crate::declare_stub!(pub EQ_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn eq_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_eq_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_eq_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn eq_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "eq_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn eq_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "eq_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "eq_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_comparison_op(None, a, b)?;
    eq_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_eq_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// ne (not-equal) — ne(NaN, NaN) = true (IEEE 754, matches PyTorch).
// =============================================================================

crate::declare_stub!(pub NE_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn ne_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_ne_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_ne_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn ne_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "ne_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn ne_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "ne_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "ne_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_comparison_op(None, a, b)?;
    ne_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_ne_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}
