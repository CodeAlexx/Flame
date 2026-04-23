//! Phase 6 ReLU on the TensorIterator pipeline. Mirror of `silu_iter.rs`.
//!
//! Y = max(X, 0). Native BF16 comparison (no fp round-trip).
//!
//! PyTorch reference: relu lowers via `clamp_min(x, 0)`; the CUDA functor
//! reduces to `a > 0 ? a : 0`. We register directly on a `RELU_STUB`.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

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
