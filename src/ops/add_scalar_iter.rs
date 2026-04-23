//! Phase 5b add_scalar on the TensorIterator pipeline. NARGS=1 + captured
//! scalar. See `ops::mul_scalar_iter` for why this bypasses the
//! DispatchStub registry.
//!
//! Replaces the BF16 branch of `GpuOps::add_scalar` (which dispatched to
//! `ops::elt::add_scalar_same_dtype` → `launch_add_scalar_bf16` in
//! `cuda/add_inplace.cu`). Same fp32-add-then-round math.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

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
