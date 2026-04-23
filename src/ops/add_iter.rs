//! Stride-aware BF16 elementwise add dispatcher — first BINARY kernel on
//! the TensorIterator port (session 4, 2026-04-22).
//!
//! Routes:
//!   * Different-shape inputs (broadcast)       → `bf16_elementwise::add_bf16`
//!                                                 (broadcast path, unchanged).
//!   * Same-shape + both contig                 → `bf16_elementwise::add_bf16`
//!                                                 (fast `__hadd2` path,
//!                                                 unchanged).
//!   * Same-shape + ≥1 strided (permute / narrow)
//!                                              → `flame_add_bf16_strided`
//!                                                 (new iterator path).
//!
//! The existing `shapes_equal_no_broadcast` fast path in
//! `bf16_elementwise::add_bf16` does NOT inspect strides or `view_offset`
//! — it reads storage-base-linear, which produces wrong bytes for strided
//! inputs. The iterator path fixes that latent issue for real callers
//! whenever narrow eventually flips to view-return.

#[cfg(feature = "cuda")]
use crate::cuda::ffi;
#[cfg(feature = "cuda")]
use crate::device::CudaStreamRawPtrExt;
#[cfg(feature = "cuda")]
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Tensor, TensorId};
#[cfg(feature = "cuda")]
use std::ffi::c_void;

pub fn add_bf16_iter(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), DType::BF16, "add_bf16_iter expects BF16");
    debug_assert_eq!(b.dtype(), DType::BF16, "add_bf16_iter expects BF16");

    // Different-shape → broadcast path (unchanged).
    if a.shape().dims() != b.shape().dims() {
        return crate::bf16_elementwise::add_bf16(a, b);
    }

    // Same-shape + both contig → fast path (unchanged).
    if a.is_contiguous() && b.is_contiguous() {
        return crate::bf16_elementwise::add_bf16(a, b);
    }

    // Same-shape + at least one strided → new iterator path.
    #[cfg(feature = "cuda")]
    {
        strided_impl(a, b)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Error::InvalidOperation(
            "add_bf16_iter strided path requires the cuda feature".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
fn strided_impl(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    const FLAME_MAX_DIMS: usize = 6;

    let shape = a.shape().clone();
    let rank = shape.rank();
    if rank > FLAME_MAX_DIMS {
        return Err(Error::InvalidOperation(format!(
            "add_bf16_iter: rank {} exceeds FLAME_MAX_DIMS ({})",
            rank, FLAME_MAX_DIMS
        )));
    }
    let n = shape.elem_count();

    let mut sizes_i64: [i64; FLAME_MAX_DIMS] = [1; FLAME_MAX_DIMS];
    let mut a_strides_i64: [i64; FLAME_MAX_DIMS] = [0; FLAME_MAX_DIMS];
    let mut b_strides_i64: [i64; FLAME_MAX_DIMS] = [0; FLAME_MAX_DIMS];
    {
        let dims = shape.dims();
        let a_strides = a.strides();
        let b_strides = b.strides();
        for i in 0..rank {
            sizes_i64[i] = dims[i] as i64;
            a_strides_i64[i] = a_strides[i] as i64;
            b_strides_i64[i] = b_strides[i] as i64;
        }
    }
    let a_offset = a.offset() as i64;
    let b_offset = b.offset() as i64;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&a.device, n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape,
        device: a.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,
    };

    let a_ptr = a.as_device_ptr_bf16("add_bf16_iter:a")? as *const c_void;
    let b_ptr = b.as_device_ptr_bf16("add_bf16_iter:b")? as *const c_void;
    let y_ptr = out.as_mut_device_ptr_bf16("add_bf16_iter:y")? as *mut c_void;
    let stream: *mut c_void = a.device.cuda_stream_raw_ptr();

    let status = unsafe {
        ffi::flame_add_bf16_strided(
            a_ptr,
            a_offset,
            a_strides_i64.as_ptr(),
            b_ptr,
            b_offset,
            b_strides_i64.as_ptr(),
            y_ptr,
            rank as i32,
            sizes_i64.as_ptr(),
            n as i64,
            stream,
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_add_bf16_strided failed with code {}",
            status
        )));
    }
    Ok(out)
}
