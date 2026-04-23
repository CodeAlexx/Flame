//! Phase 7 sqrt on the TensorIterator pipeline. Mirror of `sigmoid_iter.rs`.
//!
//! Y = sqrt(X). BF16 storage, f32 opmath (matching PyTorch).
//!
//! PyTorch reference: `at::native::sqrt_kernel_cuda` in
//! aten/src/ATen/native/cuda/UnaryOpsKernel.cu.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

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
