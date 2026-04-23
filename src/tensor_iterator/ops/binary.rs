//! Binary BF16 ops: add, sub, mul, div, maximum, minimum, mul_scalar, add_scalar.
//!
//! Same-shape contig+contig, same-shape ≥1 strided, and broadcast all route
//! through `build_binary_op` — no short-circuit to legacy flat paths.
//!
//! Scalar ops (mul_scalar, add_scalar) bypass the `DispatchStub` registry
//! because `BF16ElementwiseKernel`'s fn-pointer signature can't carry the
//! f32 scalar argument. PyTorch mirror: `opmath_gpu_kernel_with_scalars`
//! captures the scalar inside the lambda at the call site, not via
//! REGISTER_DISPATCH.
//!
//! Content relocated verbatim from
//! `src/ops/{add,sub,mul,div,maximum,minimum,mul_scalar,add_scalar}_iter.rs`
//! in Phase 11.
//!
//! PyTorch references:
//!   - add: `aten/src/ATen/native/cuda/BinaryAddSubKernel.cu`
//!   - sub: `aten/src/ATen/native/cuda/BinarySubKernel.cu`
//!   - mul: `aten/src/ATen/native/cuda/BinaryMulKernel.cu`
//!   - div: `aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu`
//!   - max/min: `aten/src/ATen/native/cuda/MaxMinElementwiseKernel.cu`
//!   - scalar ops: `Loops.cuh::opmath_gpu_kernel_with_scalars`

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

// =============================================================================
// add
// =============================================================================

crate::declare_stub!(pub ADD_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn add_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_add_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_add_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn add_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "add_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn add_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "add_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "add_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_binary_op(None, a, b)?;
    add_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_add_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// sub
// =============================================================================

crate::declare_stub!(pub SUB_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn sub_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_sub_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_sub_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn sub_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "sub_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn sub_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "sub_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "sub_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_binary_op(None, a, b)?;
    sub_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_sub_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// mul
// =============================================================================

crate::declare_stub!(pub MUL_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn mul_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_mul_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_mul_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn mul_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "mul_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn mul_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "mul_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "mul_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_binary_op(None, a, b)?;
    mul_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_mul_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// div
// =============================================================================

crate::declare_stub!(pub DIV_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn div_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_div_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_div_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn div_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "div_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn div_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "div_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "div_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_binary_op(None, a, b)?;
    div_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_div_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// maximum
// =============================================================================

crate::declare_stub!(pub MAXIMUM_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn maximum_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_maximum_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_maximum_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn maximum_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "maximum_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn maximum_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "maximum_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "maximum_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_binary_op(None, a, b)?;
    maximum_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_maximum_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// minimum
// =============================================================================

crate::declare_stub!(pub MINIMUM_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn minimum_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_minimum_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_minimum_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn minimum_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "minimum_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn minimum_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "minimum_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "minimum_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_binary_op(None, a, b)?;
    minimum_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_minimum_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// mul_scalar: NARGS=1 + captured f32 scalar. Bypasses the registry.
// =============================================================================

pub fn mul_scalar_bf16_iter(x: &Tensor, scalar: f32) -> Result<Tensor> {
    debug_assert_eq!(
        x.dtype(),
        crate::DType::BF16,
        "mul_scalar_bf16_iter expects BF16"
    );
    #[cfg(feature = "cuda")]
    {
        let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
        let meta = iter.build_iter_metadata()?;
        let stream = iter.stream()?;
        let status = unsafe { flame_mul_scalar_bf16_kernel(&meta, scalar, stream) };
        if status != 0 {
            return Err(Error::Cuda(format!(
                "flame_mul_scalar_bf16_kernel failed with code {}",
                status
            )));
        }
        iter.take_output(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, scalar);
        Err(Error::InvalidOperation(
            "mul_scalar_bf16_iter requires the cuda feature".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_mul_scalar_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        scalar: f32,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

// =============================================================================
// add_scalar: NARGS=1 + captured f32 scalar. Bypasses the registry.
// =============================================================================

pub fn add_scalar_bf16_iter(x: &Tensor, scalar: f32) -> Result<Tensor> {
    debug_assert_eq!(
        x.dtype(),
        crate::DType::BF16,
        "add_scalar_bf16_iter expects BF16"
    );
    #[cfg(feature = "cuda")]
    {
        let mut iter = TensorIteratorBase::build_unary_op(None, x)?;
        let meta = iter.build_iter_metadata()?;
        let stream = iter.stream()?;
        let status = unsafe { flame_add_scalar_bf16_kernel(&meta, scalar, stream) };
        if status != 0 {
            return Err(Error::Cuda(format!(
                "flame_add_scalar_bf16_kernel failed with code {}",
                status
            )));
        }
        iter.take_output(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, scalar);
        Err(Error::InvalidOperation(
            "add_scalar_bf16_iter requires the cuda feature".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_add_scalar_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        scalar: f32,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}
