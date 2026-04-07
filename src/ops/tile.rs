use crate::{ops::utils::as_owned_f32, DType, Error, Result, Tensor};

#[cfg(feature = "cuda")]
use crate::cuda::ffi::launch_tile_bc_to_bhwc_f32;

#[cfg(feature = "cuda")]
use crate::device::CudaStreamRawPtrExt;
#[cfg(feature = "cuda")]
use cudarc::driver::DevicePtr;

/// Tile a [B, C] tensor into [B, H, W, C] with owning FP32 storage on the GPU.
/// This avoids the broadcast path so AdaLayerNorm can operate with true FP32 buffers.
#[cfg(feature = "cuda")]
pub fn tile_bc_to_bhwc_f32(
    in_bc: &Tensor,
    b: usize,
    h: usize,
    w: usize,
    c: usize,
) -> Result<Tensor> {
    if in_bc.dtype() != DType::F32 {
        return Err(Error::InvalidInput(
            "tile_bc_to_bhwc_f32 expects F32 input; cast to F32 first".into(),
        ));
    }
    let owned = as_owned_f32(in_bc)?;
    if crate::env_flags::sdxl_debug_shapes_enabled() {
        eprintln!(
            "[tile_bc_to_bhwc_f32] src dtype {:?} storage {:?} shape {:?}",
            owned.dtype(),
            owned.storage_dtype(),
            owned.shape().dims()
        );
    }
    let device = owned.device();
    let out_shape = crate::Shape::from_dims(&[b, h, w, c]);
    let mut out = Tensor::zeros_dtype(out_shape, DType::F32, device.clone())?;

    let src_slice = owned
        .storage_ref()
        .try_as_slice_f32()
        .map_err(|_| Error::InvalidOperation("tile_bc_to_bhwc_f32: expected F32 storage".into()))?;
    let dst_slice = out
        .storage_mut()
        .try_as_mut_slice_f32()
        .map_err(|_| Error::InvalidOperation("tile_bc_to_bhwc_f32: expected F32 storage".into()))?;

    let stream = device.cuda_stream_raw_ptr();
    unsafe {
        launch_tile_bc_to_bhwc_f32(
            *src_slice.device_ptr() as *const f32,
            *dst_slice.device_ptr() as *mut f32,
            b as i32,
            h as i32,
            w as i32,
            c as i32,
            stream,
        );
    }
    if crate::env_flags::sdxl_debug_shapes_enabled() {
        eprintln!(
            "[tile_bc_to_bhwc_f32] out dtype {:?} storage {:?} shape {:?}",
            out.dtype(),
            out.storage_dtype(),
            out.shape().dims()
        );
    }
    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn tile_bc_to_bhwc_f32(
    in_bc: &Tensor,
    b: usize,
    h: usize,
    w: usize,
    c: usize,
) -> Result<Tensor> {
    let owned = as_owned_f32(in_bc)?;
    let mut data = owned.to_vec()?;
    let mut out = Vec::with_capacity(b * h * w * c);
    for batch in 0..b {
        for _yy in 0..h {
            for _xx in 0..w {
                let offset = batch * c;
                out.extend_from_slice(&data[offset..offset + c]);
            }
        }
    }
    Tensor::from_vec(
        out,
        crate::Shape::from_dims(&[b, h, w, c]),
        owned.device.clone(),
    )
}
