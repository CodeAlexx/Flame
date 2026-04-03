use crate::{tensor_ext::to_owning_fp32_strong, DType, Result, Tensor};

#[inline]
pub fn as_owned_f32(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 {
        to_owning_fp32_strong(t)
    } else {
        t.to_dtype(DType::F32)
    }
}
