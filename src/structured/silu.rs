//! Structured-kernel exemplar: `silu(x)` for BF16 tensors.
//!
//! Mirrors PyTorch's `structured_silu_native::meta` (Activation.cpp) +
//! `structured_silu_native::impl_` (UnaryGpuKernels.cu) split.
//!
//! Reuses the existing TensorIterator `silu_bf16_kernel` for the GPU work
//! — this phase is about dispatch shape, not kernel code.

use super::kernel::StructuredKernel;
use crate::autograd::{AutogradContext, Op};
use crate::tensor_iterator::TensorIteratorBase;
use crate::{DType, Error, Result, Tensor};

/// Structured `silu(x)` for BF16. Exemplar of the
/// [`StructuredKernel`](super::StructuredKernel) pattern.
///
/// Today this is BF16-only — the structured pattern's value is the
/// `meta`/`impl_` split, not dtype dispatch. The fallback (non-BF16) path
/// still belongs in `Tensor::silu`.
pub struct SiluStructured;

impl StructuredKernel for SiluStructured {
    type Input<'a> = &'a Tensor;

    /// Validate + pre-allocate the output.
    ///
    /// PyTorch's silu meta is just `build_borrowing_unary_op(maybe_get_output(0), self)`
    /// — same shape, same dtype as input, contig row-major.
    fn meta(x: &Tensor) -> Result<Tensor> {
        if x.dtype() != DType::BF16 {
            return Err(Error::InvalidOperation(format!(
                "SiluStructured::meta: BF16 only (got {:?})",
                x.dtype()
            )));
        }
        // Same shape, same dtype, same device. `empty_dtype` is the
        // canonical alloc path; it does not memset.
        Tensor::empty_dtype(x.shape().clone(), x.dtype(), x.device().clone())
    }

    /// Run the kernel into the pre-allocated `out`.
    ///
    /// Builds the TensorIterator with `add_output(Some(&out))` so
    /// `allocate_or_resize_outputs` sees an existing tensor and skips the
    /// allocation path entirely — the kernel writes through the borrowed
    /// output pointer.
    fn impl_(x: &Tensor, out: Tensor) -> Result<Tensor> {
        // Borrow `out` for the iterator's lifetime. The iterator
        // captures `&out` as `OperandSrc::Borrowed`; after the kernel
        // returns we drop the iterator and `out` is the same tensor
        // (its storage was mutated in-place via the borrowed pointer).
        {
            let mut iter = TensorIteratorBase::build_unary_op(Some(&out), x)?;
            crate::tensor_iterator::ops::unary::silu_bf16_kernel(&mut iter)?;
        }
        Ok(out)
    }

    /// `meta` → autograd record → `impl_`.
    ///
    /// Matches the current `Tensor::silu` recording shape: records
    /// `Op::SiLU { input: x.id }` against the output, saving the input
    /// tensor (clone is cheap — it's the storage `Arc` + metadata).
    fn dispatch(x: &Tensor) -> Result<Tensor> {
        let out = Self::meta(x)?;
        let mut out = Self::impl_(x, out)?;
        // Match `Tensor::silu`: set requires_grad + record AFTER the
        // kernel runs. The saved input (`x.clone()`) keeps the input's
        // storage alive for backward; the output id is what the engine
        // looks up.
        if x.requires_grad {
            out.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    out.id,
                    Op::SiLU { input: x.id },
                    vec![(x.id, x.clone())],
                );
            }
        }
        Ok(out)
    }
}
