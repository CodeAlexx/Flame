//! Phase 4 SiLU on the TensorIterator pipeline.
//!
//! `silu_bf16_iter` is the public Rust entry point called by
//! `Tensor::silu()` for BF16 inputs. Shape after Phase 4:
//!
//!   silu_bf16_iter(x)          -- builds the iterator
//!       ↓
//!   build_unary_op(None, x)    -- shape / stride / alloc / reorder / coalesce
//!       ↓
//!   silu_bf16_kernel(iter)     -- dispatcher target (registered on SILU_STUB)
//!       ↓
//!   flame_silu_bf16_kernel     -- FFI into src/cuda/unary/silu.cu
//!       ↓
//!   launch_gpu_kernel<1, SiluBF16Op>(meta, SiluBF16Op{}, stream)
//!
//! PyTorch reference: `at::native::silu_kernel` in
//! aten/src/ATen/native/cuda/ActivationSiluKernel.cu — a `silu_kernel`
//! function taking `TensorIteratorBase&` and calling `gpu_kernel(iter, λ)`,
//! registered via `REGISTER_DISPATCH(silu_stub, &silu_kernel)`.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

// Declare the stub backing `Tensor::silu()`. `register_all_bf16_kernels`
// (in `tensor_iterator::dispatch`) registers `silu_bf16_kernel` against it.
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

/// No-CUDA stub: compiled when the `cuda` feature is off. Phase 4 is
/// CUDA-only; the no-cuda path never reaches here (`Tensor::silu()` has a
/// cfg-gated else branch).
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
