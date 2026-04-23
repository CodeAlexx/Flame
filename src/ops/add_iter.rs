//! Phase 4 add on the TensorIterator pipeline — first binary op on the new
//! path. Covers:
//!   - same-shape contig+contig
//!   - same-shape ≥1 strided
//!   - broadcast (compute_shape handles the shape math; operand stride=0
//!     on broadcasted dims, as in PyTorch)
//!
//! Unlike the Phase 3 router (`add_iter.rs` pre-Phase-4), there is no
//! short-circuit to the legacy `bf16_elementwise::add_bf16` flat path.
//! Every BF16 add call routes through `build_binary_op` + the new kernel.
//! This is the point of the plan (one dispatch path; fast-path gates
//! deleted, not patched).
//!
//! Klein byte-equal gate: on contig inputs the new functor's fp32
//! round-trip add (`__float2bfloat16_rn(va + vb)`) may diverge bit-for-bit
//! from the pre-Phase-4 `__hadd2` flat kernel at rounding-tie boundaries.
//! That drift, if it shows up in Klein, surfaces as a Phase 4 blocker;
//! see the Phase 4 section of plan-this-and-fix-encapsulated-hennessy.md
//! and R5 in TENSORITERATOR_PORT_REFERENCE.md §9 for the escalation path.

use crate::tensor_iterator::TensorIteratorBase;
use crate::{Error, Result, Tensor};

crate::declare_stub!(pub ADD_STUB);

#[cfg(feature = "cuda")]
pub(crate) fn add_bf16_kernel(iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    let meta = iter.build_iter_metadata()?;
    let stream = iter.stream()?;
    let status = unsafe { flame_add_bf16_kernel(&meta, stream) };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_add_bf16_kernel failed with code {}",
            status
        )));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn add_bf16_kernel(_iter: &mut TensorIteratorBase<'_>) -> Result<()> {
    Err(Error::InvalidOperation(
        "add_bf16_kernel requires the cuda feature".into(),
    ))
}

pub fn add_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), crate::DType::BF16, "add_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), crate::DType::BF16, "add_bf16_iter expects BF16");
    let mut iter = TensorIteratorBase::build_binary_op(None, a, b)?;
    add_bf16_kernel(&mut iter)?;
    iter.take_output(0)
}

#[cfg(feature = "cuda")]
extern "C" {
    fn flame_add_bf16_kernel(
        meta: *const crate::tensor_iterator::IterMetadata,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}
