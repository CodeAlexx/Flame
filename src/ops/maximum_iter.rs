//! Phase 5b maximum on the TensorIterator pipeline. Mirrors `ops::add_iter`.
//!
//! Replaces `bf16_elementwise::max_bf16` (broadcast path only — no flat
//! fast-path existed in the legacy kernel).
//!
//! PyTorch reference: `aten/src/ATen/native/cuda/MaxMinElementwiseKernel.cu`.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

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
