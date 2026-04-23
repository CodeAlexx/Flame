//! Phase 9 eq (equal) on the TensorIterator pipeline.
//! See `ops::ge_iter` for the output-dtype and naming discussion.
//! NaN semantics: IEEE 754 — eq(NaN, NaN) = false (matches PyTorch).

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

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
