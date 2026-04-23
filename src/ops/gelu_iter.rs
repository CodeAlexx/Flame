//! Stride-aware BF16 GELU dispatcher — second kernel on the TensorIterator
//! port (see `ops::silu_iter` for session 1). Short-circuits contiguous
//! inputs to the existing `bf16_ops::gelu_bf16` vectorized NVRTC kernel;
//! strided inputs drop through to `flame_gelu_bf16_strided` backed by
//! `src/cuda/activation_gelu_iter.cu`.

#[cfg(feature = "cuda")]
use crate::cuda::ffi;
#[cfg(feature = "cuda")]
use crate::device::CudaStreamRawPtrExt;
#[cfg(feature = "cuda")]
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Tensor, TensorId};
#[cfg(feature = "cuda")]
use std::ffi::c_void;

/// Stride-aware BF16 GELU (tanh approximation). Output is a freshly
/// allocated contiguous row-major BF16 tensor with `x.shape()`.
///
/// * Contiguous `x`: delegates to `bf16_ops::gelu_bf16` unchanged.
/// * Strided `x`: walks the input via `flame_gelu_bf16_strided` and writes
///   into a contig output.
pub fn gelu_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16, "gelu_bf16_iter expects BF16");

    if x.is_contiguous() {
        return crate::bf16_ops::gelu_bf16(x);
    }

    #[cfg(feature = "cuda")]
    {
        strided_impl(x)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Error::InvalidOperation(
            "gelu_bf16_iter strided path requires the cuda feature".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
fn strided_impl(x: &Tensor) -> Result<Tensor> {
    const FLAME_MAX_DIMS: usize = 6;

    let shape = x.shape().clone();
    let rank = shape.rank();
    if rank > FLAME_MAX_DIMS {
        return Err(Error::InvalidOperation(format!(
            "gelu_bf16_iter: rank {} exceeds FLAME_MAX_DIMS ({})",
            rank, FLAME_MAX_DIMS
        )));
    }
    let n = shape.elem_count();

    let mut sizes_i64: [i64; FLAME_MAX_DIMS] = [1; FLAME_MAX_DIMS];
    let mut strides_i64: [i64; FLAME_MAX_DIMS] = [0; FLAME_MAX_DIMS];
    {
        let dims = shape.dims();
        let strides = x.strides();
        for i in 0..rank {
            sizes_i64[i] = dims[i] as i64;
            strides_i64[i] = strides[i] as i64;
        }
    }
    let x_offset_elems = x.offset() as i64;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape,
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,
    };

    let x_ptr = x.as_device_ptr_bf16("gelu_bf16_iter:x")? as *const c_void;
    let y_ptr = out.as_mut_device_ptr_bf16("gelu_bf16_iter:y")? as *mut c_void;
    let stream: *mut c_void = x.device.cuda_stream_raw_ptr();

    let status = unsafe {
        ffi::flame_gelu_bf16_strided(
            x_ptr,
            x_offset_elems,
            y_ptr,
            rank as i32,
            sizes_i64.as_ptr(),
            strides_i64.as_ptr(),
            n as i64,
            stream,
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_gelu_bf16_strided failed with code {}",
            status
        )));
    }
    Ok(out)
}
