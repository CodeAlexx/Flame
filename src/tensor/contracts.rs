use crate::{telemetry, DType, FlameError, Result, Tensor};

// NOTE: Apply assert_nhwc_bf16_public only on public tensors leaving op/engine boundaries;
// never on internal scratch buffers or FP32 GradientMap entries.
#[inline]
pub fn assert_nhwc_public(op: &str, tensor: &Tensor) -> Result<()> {
    if tensor.rank() != 4 {
        return Err(FlameError::InvalidInput(format!(
            "{op}: expected rank=4, got {}",
            tensor.rank()
        )));
    }
    if !tensor.is_nhwc() {
        return Err(FlameError::InvalidInput(format!(
            "{op}: expected NHWC layout"
        )));
    }
    Ok(())
}

#[inline]
pub fn trap_is_bf16(op: &str, tensor: &Tensor) -> Result<()> {
    if tensor.dtype() != DType::BF16 || tensor.storage_dtype() != DType::BF16 {
        telemetry::record_dtype_trap(op, tensor.dtype(), tensor.storage_dtype());
        return Err(FlameError::InvalidInput(format!(
            "{op}: expected logical/storage BF16, got logical={:?} storage={:?}",
            tensor.dtype(),
            tensor.storage_dtype()
        )));
    }
    Ok(())
}

#[inline]
pub fn assert_nhwc_bf16_public(op: &str, tensor: &Tensor) -> Result<()> {
    assert_nhwc_public(op, tensor)?;
    trap_is_bf16(op, tensor)
}

#[inline]
pub fn scalar_for_dtype(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => half::bf16::from_f32(x).to_f32(),
        DType::F32 => x,
        _ => x,
    }
}
