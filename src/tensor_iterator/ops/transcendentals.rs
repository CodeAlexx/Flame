//! Transcendental unary BF16 ops: exp, log, sqrt, rsqrt, recip.
//!
//! All use f32 opmath inside the functor (bf16→f32 on load, f32 intrinsic,
//! `__float2bfloat16_rn` on store) — BF16 has no `__hexp`/`__hlog` etc.
//! Matches PyTorch's `exp_kernel_cuda` using `std::exp((float)x)` inside
//! `gpu_kernel` lambda.
//!
//! Content relocated verbatim from `src/ops/{exp,log,sqrt,rsqrt,recip}_iter.rs`
//! in Phase 11.
//!
//! PyTorch references:
//!   - exp / log / sqrt: `aten/src/ATen/native/cuda/UnaryOpsKernel.cu`
//!   - rsqrt: `aten/src/ATen/native/cuda/UnaryOpsKernel.cu`
//!   - recip (`reciprocal`): `aten/src/ATen/native/cuda/UnaryOpsKernel.cu`

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

// =============================================================================
// exp: Y = exp(X). BF16 storage, f32 opmath.
// =============================================================================

crate::declare_stub!(pub EXP_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn exp_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_exp_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_exp_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn exp_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "exp_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn exp_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "exp_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    exp_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_exp_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// log: Y = log(X). BF16 storage, f32 opmath.
// =============================================================================

crate::declare_stub!(pub LOG_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn log_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_log_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_log_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn log_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "log_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn log_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "log_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    log_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_log_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// sqrt: Y = sqrt(X). BF16 storage, f32 opmath.
// =============================================================================

crate::declare_stub!(pub SQRT_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn sqrt_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_sqrt_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_sqrt_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn sqrt_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "sqrt_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn sqrt_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "sqrt_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    sqrt_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_sqrt_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// rsqrt: Y = 1/sqrt(X). BF16 storage, f32 opmath.
// =============================================================================

crate::declare_stub!(pub RSQRT_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn rsqrt_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_rsqrt_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_rsqrt_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn rsqrt_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "rsqrt_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn rsqrt_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "rsqrt_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    rsqrt_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_rsqrt_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// recip: Y = 1/X. BF16 storage, f32 opmath.
// =============================================================================

crate::declare_stub!(pub RECIP_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn recip_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_recip_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_recip_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn recip_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "recip_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn recip_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "recip_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    recip_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_recip_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}
