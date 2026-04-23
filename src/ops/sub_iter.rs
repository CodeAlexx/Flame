//! Phase 5b sub on the TensorIterator pipeline. Mirrors `ops::add_iter`.
//!
//! Replaces the former route through `bf16_elementwise::sub_bf16`
//! (which composed sub as `a + (-1 * b)` on the slow broadcast path and
//! used `__hsub2` on the flat path). Every BF16 sub now goes through
//! `build_binary_op` + `launch_gpu_kernel<2, SubBF16Op>` — one dispatch
//! path, stride-respecting, fp32-round-trip math.
//!
//! PyTorch reference: `aten/src/ATen/native/cuda/BinarySubKernel.cu`.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

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
