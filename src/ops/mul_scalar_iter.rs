//! Phase 5b mul_scalar on the TensorIterator pipeline. NARGS=1 (one tensor
//! input) + a captured scalar.
//!
//! Design note — scalar ops bypass the DispatchStub registry. The
//! `BF16ElementwiseKernel` fn-pointer type in
//! `tensor_iterator::dispatch` has signature `fn(&mut TensorIteratorBase)`,
//! which can't carry the extra `scalar: f32` argument. PyTorch handles
//! this via `opmath_gpu_kernel_with_scalars` (Loops.cuh:200), where the
//! scalar is bound into the lambda's captures at the call site — no
//! separate registration. flame-core mirrors that shape: the wrapper
//! builds the iterator, calls the extern "C" directly with the scalar
//! as an argument, and returns the output. No `declare_stub!` /
//! `register_stub!` for scalar ops.
//!
//! Replaces the BF16 branch of `GpuOps::mul_scalar` (which dispatched to
//! `ops::elt::mul_scalar_same_dtype` → `launch_mul_scalar_bf16` in
//! `cuda/add_inplace.cu`). Same fp32-multiply-then-round math.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

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
