//! Unary BF16 elementwise ops: silu, gelu, square, abs, relu, sigmoid, tanh, neg.
//!
//! Each op is one `declare_stub!`, one Rust kernel wrapper (calls FFI),
//! one public `_bf16_iter` entry point (builds iterator + dispatches),
//! one `extern "C"` FFI decl into the corresponding `src/cuda/unary/<op>.cu`.
//!
//! Content relocated verbatim from `src/ops/{silu,gelu,square,abs,relu,sigmoid,tanh,neg}_iter.rs`
//! in Phase 11. See `tensor_iterator::ops` module docs for the grouping rationale.
//!
//! PyTorch references:
//!   - silu: `aten/src/ATen/native/cuda/ActivationSiluKernel.cu`
//!   - gelu: `aten/src/ATen/native/cuda/ActivationGeluKernel.cu`
//!   - square: derived (`x * x`)
//!   - abs: `aten/src/ATen/native/cuda/AbsKernel.cu`
//!   - relu: `clamp_min(x, 0)`
//!   - sigmoid: `aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu`
//!   - tanh: `aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu`
//!   - neg: `aten/src/ATen/native/cuda/UnarySignKernels.cu`

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

// =============================================================================
// silu
// =============================================================================

crate::declare_stub!(pub SILU_STUB);

/// Dispatch target registered on `SILU_STUB`. Builds an `IterMetadata` POD
/// from the iterator and invokes the CUDA kernel.
///
/// The iterator must have been built with a pending output (`add_output(None)`)
/// and a single BF16 input; the caller owns the iterator and is expected to
/// `take_output(0)` after this returns Ok.
#[cfg(feature = "cuda")]
pub(crate) fn silu_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_silu_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_silu_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn silu_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "silu_bf16_kernel requires the cuda feature".into(),
    ))
}

/// Public entry. Builds the iterator, dispatches, returns the freshly-
/// allocated contig BF16 output.
pub fn silu_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "silu_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    silu_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    /// See src/cuda/unary/silu.cu. Returns 0 on success, nonzero CUDA error
    /// code otherwise.
    fn flame_silu_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// gelu
// =============================================================================

crate::declare_stub!(pub GELU_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn gelu_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_gelu_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_gelu_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn gelu_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "gelu_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn gelu_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "gelu_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    gelu_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_gelu_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// square: Y = X * X (fp32 round-trip).
// =============================================================================

crate::declare_stub!(pub SQUARE_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn square_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_square_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_square_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn square_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "square_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn square_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "square_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    square_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_square_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// abs: Y = sign-bit-clear(X). Native BF16 — no fp round-trip.
// =============================================================================

crate::declare_stub!(pub ABS_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn abs_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_abs_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_abs_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn abs_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "abs_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn abs_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "abs_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    abs_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_abs_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// relu: Y = max(X, 0). Native BF16 comparison (no fp round-trip).
// =============================================================================

crate::declare_stub!(pub RELU_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn relu_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_relu_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_relu_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn relu_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "relu_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn relu_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "relu_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    relu_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_relu_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// sigmoid: Y = 1/(1 + exp(-X)). BF16 storage, f32 opmath (matching PyTorch).
// =============================================================================

crate::declare_stub!(pub SIGMOID_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn sigmoid_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_sigmoid_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_sigmoid_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn sigmoid_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "sigmoid_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn sigmoid_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "sigmoid_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    sigmoid_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_sigmoid_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// tanh: Y = tanh(X). BF16 storage, f32 opmath (matching PyTorch).
// =============================================================================

crate::declare_stub!(pub TANH_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn tanh_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_tanh_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_tanh_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn tanh_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "tanh_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn tanh_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "tanh_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    tanh_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_tanh_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// neg: Y = -X. Native BF16 sign-bit flip.
// =============================================================================

crate::declare_stub!(pub NEG_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn neg_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_neg_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_neg_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn neg_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "neg_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn neg_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), crate::DType::BF16, "neg_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
    neg_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_neg_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}
