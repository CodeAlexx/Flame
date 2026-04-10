#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::{DevicePtr, DevicePtrMut};

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda::device_lt;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::autograd::{AutogradContext, Op};
use crate::{DType, Error, Result, Tensor};

/// Sum over the last dimension, keeping the dimension (size 1) and returning
/// the result in `out_dtype`. FP32 accumulation happens inside the CUDA kernel.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn sum_dim_keepdim_as(x: &Tensor, dim: usize, out_dtype: DType) -> Result<Tensor> {
    if out_dtype != DType::BF16 && out_dtype != DType::F32 {
        return Err(Error::Unsupported(
            "sum_dim_keepdim_as: only BF16 or F32 outputs supported".into(),
        ));
    }

    // Only support summing the last dimension for now
    let rank = x.shape().rank();
    if dim != rank - 1 {
        return Err(Error::Unsupported(
            "sum_dim_keepdim_as: only last-dimension reduction supported".into(),
        ));
    }

    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(
            "sum_dim_keepdim_as: expect 3D tensor [B,M,K]".into(),
        ));
    }
    let batch = dims[0] as i32;
    let rows = dims[1] as i32;
    let cols = dims[2] as i32;

    // Output shape mirrors input with last dim = 1
    let mut out_shape = dims.to_vec();
    out_shape[rank - 1] = 1;
    if x.dtype() != DType::BF16 {
        eprintln!(
            "[bf16][error] sum_dim_keepdim_as expected BF16, got {:?} shape {:?}",
            x.dtype(),
            x.shape().dims()
        );
        let msg = format!(
            "sum_dim_keepdim_as: expected BF16 input, got {:?} with shape {:?}",
            x.dtype(),
            x.shape().dims()
        );
        return Err(Error::Unsupported(msg.into()));
    }

    let device = x.device();
    let mut out = Tensor::zeros_dtype(
        crate::Shape::from_dims(&out_shape),
        out_dtype,
        device.clone(),
    )?;

    let x_ptr = x.as_device_ptr_bf16("sum_dim_keepdim_as:x")? as *const core::ffi::c_void;
    let out_ptr = match out_dtype {
        DType::BF16 => {
            out.as_mut_device_ptr_bf16("sum_dim_keepdim_as:out")? as *mut core::ffi::c_void
        }
        DType::F32 => {
            let slice = out.storage_mut().try_as_mut_slice_f32().map_err(|_| {
                Error::InvalidOperation("sum_dim_keepdim_as: expected F32 storage".into())
            })?;
            *slice.device_ptr_mut() as *mut core::ffi::c_void
        }
        _ => unreachable!(),
    };
    let stream = device_lt::stream_ptr(device)?;

    unsafe {
        match out_dtype {
            DType::BF16 => crate::cuda::ffi::launch_sum_last_keepdim_bf16(
                x_ptr, out_ptr, batch, rows, cols, stream,
            ),
            DType::F32 => crate::cuda::ffi::launch_sum_last_keepdim_bf16_to_f32(
                x_ptr, out_ptr, batch, rows, cols, stream,
            ),
            _ => unreachable!(),
        }
    }

    if x.requires_grad() {
        out.requires_grad = true;
        if AutogradContext::is_recording() {
            AutogradContext::record_op(
                out.id,
                Op::SumDimKeepdim { input: x.id(), dim },
                vec![(x.id(), x.clone_result()?)],
            );
        }
    }

    Ok(out)
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn sum_dim_keepdim_as(_x: &Tensor, _dim: usize, _out_dtype: DType) -> Result<Tensor> {
    Err(Error::Unsupported(
        "sum_dim_keepdim_as requires the `cuda` and `bf16_u16` features".into(),
    ))
}
