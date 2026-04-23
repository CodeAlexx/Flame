//! Phase 9 ge (greater-or-equal) on the TensorIterator pipeline.
//!
//! Replaces the legacy `bf16_elementwise::ge_bf16` (dead path: never wired
//! into `Tensor::ge`, which went through `GpuOps::compare_binary` — an F32
//! round-trip via `CudaKernels::compare_ge`). Every BF16 ge now goes
//! through `build_comparison_op` + `launch_gpu_kernel<2, GeBF16Op>`.
//!
//! Output dtype divergence from PyTorch: see the long comment in
//! `TensorIteratorBase::build_comparison_op` (src/tensor_iterator/base.rs).
//! flame-core writes BF16 0.0/1.0 sentinels instead of kBool bytes.
//!
//! PyTorch reference: `aten/src/ATen/native/cuda/CompareKernels.cu`
//! (`ge_kernel_cuda` → `compare_kernel_impl<scalar_t>` → `gpu_kernel(iter,
//! CompareFunctor<scalar_t>{OpType::GE})`).

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

crate::declare_stub!(pub GE_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn ge_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_ge_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_ge_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn ge_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "ge_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn ge_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "ge_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "ge_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_comparison_op(None, a, b)?;
    ge_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_ge_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}
