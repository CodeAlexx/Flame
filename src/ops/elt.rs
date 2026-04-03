#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::DType;
use crate::{strict::allow_clone, Error, Result, Tensor};

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::device::CudaStreamRawPtrExt;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::{DevicePtr, DevicePtrMut};

/// In-place addition: `dst += src`. Tensors must share shape, dtype, and device.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn add_inplace_same_dtype(dst: &mut Tensor, src: &Tensor) -> Result<()> {
    if dst.dtype() != src.dtype() {
        return Err(Error::InvalidInput(
            "add_inplace_same_dtype: dtype mismatch".into(),
        ));
    }
    if dst.shape() != src.shape() {
        return Err(Error::InvalidInput(
            "add_inplace_same_dtype: shape mismatch".into(),
        ));
    }
    if dst.device().ordinal() != src.device().ordinal() {
        return Err(Error::InvalidInput(
            "add_inplace_same_dtype: tensors on different devices".into(),
        ));
    }

    let elems = dst.shape().elem_count() as i64;
    let stream = dst.device().cuda_stream_raw_ptr();

    unsafe {
        match dst.dtype() {
            DType::BF16 => {
                let dst_ptr = match dst.storage_mut().try_as_mut_slice_u16() {
                    Ok(slice) => *slice.device_ptr_mut() as *mut core::ffi::c_void,
                    Err(_) => dst.as_mut_device_ptr_bf16("add_inplace_same_dtype:dst")?
                        as *mut core::ffi::c_void,
                };
                let src_ptr = match src.storage_ref().try_as_slice_u16() {
                    Ok(slice) => *slice.device_ptr() as *const core::ffi::c_void,
                    Err(_) => src.as_device_ptr_bf16("add_inplace_same_dtype:src")?
                        as *const core::ffi::c_void,
                };
                crate::cuda::ffi::launch_add_inplace_bf16(dst_ptr, src_ptr, elems, stream);
            }
            DType::F32 => {
                let dst_ptr = {
                    let slice = dst.storage_mut().try_as_mut_slice_f32().map_err(|_| {
                        Error::InvalidOperation(
                            "add_inplace_same_dtype: expected F32 storage".into(),
                        )
                    })?;
                    *slice.device_ptr_mut() as *mut f32
                };
                let src_ptr = {
                    let slice = src.storage_ref().try_as_slice_f32().map_err(|_| {
                        Error::InvalidOperation(
                            "add_inplace_same_dtype: expected F32 storage".into(),
                        )
                    })?;
                    *slice.device_ptr() as *const f32
                };
                crate::cuda::ffi::launch_add_inplace_f32(dst_ptr, src_ptr, elems, stream);
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "add_inplace_same_dtype: dtype {:?} not supported",
                    other
                )));
            }
        }
    }
    Ok(())
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn add_inplace_same_dtype(_dst: &mut Tensor, _src: &Tensor) -> Result<()> {
    Err(Error::Unsupported(
        "add_inplace_same_dtype requires the `cuda` and `bf16_u16` features".into(),
    ))
}

/// Allocate a new tensor equal to `lhs + rhs` in the operands' dtype.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn add_same_dtype(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::InvalidInput(
            "add_same_dtype: operands must share dtype".into(),
        ));
    }
    let _guard = allow_clone();
    let mut out = lhs.clone_result()?;
    add_inplace_same_dtype(&mut out, rhs)?;
    Ok(out)
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn add_same_dtype(_lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
    Err(Error::Unsupported(
        "add_same_dtype requires the `cuda` and `bf16_u16` features".into(),
    ))
}

/// Legacy helper: replace `out` with `lhs + rhs`.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn add_out(lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<()> {
    *out = add_same_dtype(lhs, rhs)?;
    Ok(())
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn add_out(_lhs: &Tensor, _rhs: &Tensor, _out: &mut Tensor) -> Result<()> {
    Err(Error::Unsupported(
        "add_out requires the `cuda` and `bf16_u16` features".into(),
    ))
}

/// In-place multiplication: `dst *= src`. Tensors must share shape, dtype, and device.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn mul_inplace_same_dtype(dst: &mut Tensor, src: &Tensor) -> Result<()> {
    if dst.dtype() != src.dtype() {
        return Err(Error::InvalidInput(
            "mul_inplace_same_dtype: dtype mismatch".into(),
        ));
    }
    if dst.shape() != src.shape() {
        return Err(Error::InvalidInput(
            "mul_inplace_same_dtype: shape mismatch".into(),
        ));
    }
    if dst.device().ordinal() != src.device().ordinal() {
        return Err(Error::InvalidInput(
            "mul_inplace_same_dtype: tensors on different devices".into(),
        ));
    }

    let elems = dst.shape().elem_count() as i64;
    let stream = dst.device().cuda_stream_raw_ptr();

    unsafe {
        match dst.dtype() {
            DType::BF16 => {
                let dst_ptr = match dst.storage_mut().try_as_mut_slice_u16() {
                    Ok(slice) => *slice.device_ptr_mut() as *mut core::ffi::c_void,
                    Err(_) => dst.as_mut_device_ptr_bf16("mul_inplace_same_dtype:dst")?
                        as *mut core::ffi::c_void,
                };
                let src_ptr = match src.storage_ref().try_as_slice_u16() {
                    Ok(slice) => *slice.device_ptr() as *const core::ffi::c_void,
                    Err(_) => src.as_device_ptr_bf16("mul_inplace_same_dtype:src")?
                        as *const core::ffi::c_void,
                };
                crate::cuda::ffi::launch_mul_inplace_bf16(dst_ptr, src_ptr, elems, stream);
            }
            DType::F32 => {
                let dst_ptr = {
                    let slice = dst.storage_mut().try_as_mut_slice_f32().map_err(|_| {
                        Error::InvalidOperation(
                            "mul_inplace_same_dtype: expected F32 storage".into(),
                        )
                    })?;
                    *slice.device_ptr_mut() as *mut f32
                };
                let src_ptr = {
                    let slice = src.storage_ref().try_as_slice_f32().map_err(|_| {
                        Error::InvalidOperation(
                            "mul_inplace_same_dtype: expected F32 storage".into(),
                        )
                    })?;
                    *slice.device_ptr() as *const f32
                };
                crate::cuda::ffi::launch_mul_inplace_f32(dst_ptr, src_ptr, elems, stream);
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "mul_inplace_same_dtype: dtype {:?} not supported",
                    other
                )));
            }
        }
    }
    Ok(())
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn mul_inplace_same_dtype(_dst: &mut Tensor, _src: &Tensor) -> Result<()> {
    Err(Error::Unsupported(
        "mul_inplace_same_dtype requires the `cuda` and `bf16_u16` features".into(),
    ))
}

/// Allocate a new tensor equal to `lhs * rhs` in the operands' dtype.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn mul_same_dtype(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::InvalidInput(
            "mul_same_dtype: operands must share dtype".into(),
        ));
    }
    let _guard = allow_clone();
    let mut out = lhs.clone_result()?;
    mul_inplace_same_dtype(&mut out, rhs)?;
    Ok(out)
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn mul_same_dtype(_lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
    Err(Error::Unsupported(
        "mul_same_dtype requires the `cuda` and `bf16_u16` features".into(),
    ))
}

/// Multiply a `[B,T,H]` tensor by a `[B,H]` gate in-place (BF16 only).
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn gate_mul_bf16_inplace(dst: &mut Tensor, gate: &Tensor) -> Result<()> {
    if dst.dtype() != DType::BF16 || dst.storage_dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "gate_mul_bf16_inplace: destination must be BF16".into(),
        ));
    }
    let dims = dst.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "gate_mul_bf16_inplace: expected [B,T,H] dst, got {:?}",
            dims
        )));
    }
    let batch = dims[0];
    let tokens = dims[1];
    let hidden = dims[2];

    let mut gate_view = match gate.shape().dims() {
        [b, 1, h] if *b == batch && *h == hidden => gate.reshape(&[batch, hidden])?,
        [b, h] if *b == batch && *h == hidden => gate.clone_result()?,
        other => {
            return Err(Error::InvalidShape(format!(
                "gate_mul_bf16_inplace: expected gate [B,1,H] or [B,H], got {:?}",
                other
            )))
        }
    };

    if gate_view.dtype() != DType::BF16 || gate_view.storage_dtype() != DType::BF16 {
        gate_view = gate_view.to_dtype(DType::BF16)?;
    }

    unsafe {
        crate::cuda::ffi::launch_gate_mul_bf16(
            dst.as_mut_device_ptr_bf16("gate_mul_bf16_inplace:dst")? as *mut _,
            gate_view.as_device_ptr_bf16("gate_mul_bf16_inplace:gate")? as *const _,
            batch as i32,
            tokens as i32,
            hidden as i32,
            dst.device().cuda_stream_raw_ptr(),
        );
    }
    Ok(())
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn gate_mul_bf16_inplace(_dst: &mut Tensor, _gate: &Tensor) -> Result<()> {
    Err(Error::Unsupported(
        "gate_mul_bf16_inplace requires the `cuda` and `bf16_u16` features".into(),
    ))
}

/// Multiply by scalar and return tensor in original dtype.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn mul_scalar_same_dtype(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
    let _guard = allow_clone();
    let mut out = tensor.clone_result()?;
    mul_scalar_assign_same_dtype(&mut out, tensor, scalar)?;
    Ok(out)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn mul_scalar_assign_same_dtype(dst: &mut Tensor, src: &Tensor, scalar: f32) -> Result<()> {
    if dst.dtype() != src.dtype() {
        return Err(Error::InvalidInput(
            "mul_scalar_same_dtype: dtype mismatch".into(),
        ));
    }
    if dst.shape() != src.shape() {
        return Err(Error::InvalidInput(
            "mul_scalar_same_dtype: shape mismatch".into(),
        ));
    }
    if dst.device().ordinal() != src.device().ordinal() {
        return Err(Error::InvalidInput(
            "mul_scalar_same_dtype: tensors on different devices".into(),
        ));
    }

    let elems = dst.shape().elem_count() as i64;
    let stream = dst.device().cuda_stream_raw_ptr();

    unsafe {
        match dst.dtype() {
            DType::BF16 => {
                let dst_ptr = dst.as_mut_device_ptr_bf16("mul_scalar_same_dtype:dst")?
                    as *mut core::ffi::c_void;
                let src_ptr = src.as_device_ptr_bf16("mul_scalar_same_dtype:src")?
                    as *const core::ffi::c_void;
                crate::cuda::ffi::launch_mul_scalar_bf16(dst_ptr, src_ptr, scalar, elems, stream);
            }
            DType::F32 => {
                let dst_ptr = {
                    let slice = dst.storage_mut().try_as_mut_slice_f32().map_err(|_| {
                        Error::InvalidOperation(
                            "mul_scalar_same_dtype: expected F32 storage".into(),
                        )
                    })?;
                    *slice.device_ptr_mut() as *mut f32
                };
                let src_ptr = {
                    let slice = src.storage_ref().try_as_slice_f32().map_err(|_| {
                        Error::InvalidOperation(
                            "mul_scalar_same_dtype: expected F32 storage".into(),
                        )
                    })?;
                    *slice.device_ptr() as *const f32
                };
                crate::cuda::ffi::launch_mul_scalar_f32(dst_ptr, src_ptr, scalar, elems, stream);
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "mul_scalar_same_dtype: dtype {:?} not supported",
                    other
                )));
            }
        }
    }
    Ok(())
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn mul_scalar_same_dtype(_tensor: &Tensor, _scalar: f32) -> Result<Tensor> {
    Err(Error::Unsupported(
        "mul_scalar_same_dtype requires the `cuda` and `bf16_u16` features".into(),
    ))
}

/// Add scalar and return tensor in original dtype.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn add_scalar_same_dtype(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
    let _guard = allow_clone();
    let mut out = tensor.clone_result()?;
    add_scalar_assign_same_dtype(&mut out, tensor, scalar)?;
    Ok(out)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn add_scalar_assign_same_dtype(dst: &mut Tensor, src: &Tensor, scalar: f32) -> Result<()> {
    if dst.dtype() != src.dtype() {
        return Err(Error::InvalidInput(
            "add_scalar_same_dtype: dtype mismatch".into(),
        ));
    }
    if dst.shape() != src.shape() {
        return Err(Error::InvalidInput(
            "add_scalar_same_dtype: shape mismatch".into(),
        ));
    }
    if dst.device().ordinal() != src.device().ordinal() {
        return Err(Error::InvalidInput(
            "add_scalar_same_dtype: tensors on different devices".into(),
        ));
    }

    let elems = dst.shape().elem_count() as i64;
    let stream = dst.device().cuda_stream_raw_ptr();

    unsafe {
        match dst.dtype() {
            DType::BF16 => {
                let dst_ptr = dst.as_mut_device_ptr_bf16("add_scalar_same_dtype:dst")?
                    as *mut core::ffi::c_void;
                let src_ptr = src.as_device_ptr_bf16("add_scalar_same_dtype:src")?
                    as *const core::ffi::c_void;
                crate::cuda::ffi::launch_add_scalar_bf16(dst_ptr, src_ptr, scalar, elems, stream);
            }
            DType::F32 => {
                let dst_ptr = {
                    let slice = dst.storage_mut().try_as_mut_slice_f32().map_err(|_| {
                        Error::InvalidOperation(
                            "add_scalar_same_dtype: expected F32 storage".into(),
                        )
                    })?;
                    *slice.device_ptr_mut() as *mut f32
                };
                let src_ptr = {
                    let slice = src.storage_ref().try_as_slice_f32().map_err(|_| {
                        Error::InvalidOperation(
                            "add_scalar_same_dtype: expected F32 storage".into(),
                        )
                    })?;
                    *slice.device_ptr() as *const f32
                };
                crate::cuda::ffi::launch_add_scalar_f32(dst_ptr, src_ptr, scalar, elems, stream);
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "add_scalar_same_dtype: dtype {:?} not supported",
                    other
                )));
            }
        }
    }
    Ok(())
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn add_scalar_same_dtype(_tensor: &Tensor, _scalar: f32) -> Result<Tensor> {
    Err(Error::Unsupported(
        "add_scalar_same_dtype requires the `cuda` and `bf16_u16` features".into(),
    ))
}
