//! GPU-only dtype casts used by matmul/conv backends.
use crate::{DType, Error, Result, Tensor};

#[cfg(feature = "bf16_u16")]
use std::sync::Arc;

#[cfg(feature = "bf16_u16")]
use crate::bf16_convert;

/// Cast a BF16 tensor to F32 without recording autograd.
pub fn cast_bf16_to_f32(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput("expected BF16 tensor".into()));
    }
    #[cfg(feature = "bf16_u16")]
    {
        let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::F32, Arc::clone(x.device()))?;
        // Use as_device_ptr_bf16 to support both owning and arena storage
        let src_ptr = x.as_device_ptr_bf16("cast_bf16_to_f32")? as u64;
        let dst = out.storage_mut().try_as_mut_slice_f32()?;
        bf16_convert::bf16_u16_to_f32(
            Arc::clone(x.device()),
            src_ptr,
            dst,
            x.shape().elem_count(),
        )?;
        Ok(out)
    }
    #[cfg(not(feature = "bf16_u16"))]
    {
        Err(Error::Unsupported(
            "BF16 requires the bf16_u16 feature".into(),
        ))
    }
}

/// Cast an F32 tensor to BF16 without recording autograd.
pub fn cast_f32_to_bf16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        return Err(Error::InvalidInput("expected F32 tensor".into()));
    }
    #[cfg(feature = "bf16_u16")]
    {
        let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, Arc::clone(x.device()))?;
        let src = x.storage_ref().try_as_slice_f32()?;
        // Use as_mut_device_ptr_bf16 to support both owning and arena storage
        let dst_ptr = out.as_mut_device_ptr_bf16("cast_f32_to_bf16")? as u64;
        bf16_convert::f32_to_bf16_u16(
            Arc::clone(x.device()),
            src,
            dst_ptr,
            x.shape().elem_count(),
        )?;
        Ok(out)
    }
    #[cfg(not(feature = "bf16_u16"))]
    {
        Err(Error::Unsupported(
            "BF16 requires the bf16_u16 feature".into(),
        ))
    }
}
