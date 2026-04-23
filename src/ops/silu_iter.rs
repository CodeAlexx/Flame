//! Stride-aware BF16 SiLU dispatcher — entry point for the TensorIterator
//! migration (HANDOFF_2026-04-22_TENSORITERATOR_PORT).
//!
//! `silu_bf16_iter` short-circuits contiguous inputs straight to the existing
//! `bf16_ops::silu_bf16` vectorized NVRTC kernel — the fast path every
//! current caller takes today pays zero iterator overhead. Only strided
//! inputs (permute views, narrow views, `as_strided`) hit the new
//! `flame_silu_bf16_strided` FFI path backed by
//! `src/cuda/activation_silu_iter.cu`.
//!
//! This mirrors PyTorch's `gpu_kernel_impl_nocast` decision in
//! `aten/src/ATen/native/cuda/CUDALoops.cuh`: the `iter.is_contiguous()`
//! branch picks the vectorized kernel, the `else` branch builds an
//! `OffsetCalculator` and launches the strided kernel.

#[cfg(feature = "cuda")]
use crate::cuda::ffi;
#[cfg(feature = "cuda")]
use crate::device::CudaStreamRawPtrExt;
#[cfg(feature = "cuda")]
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Tensor, TensorId};
#[cfg(feature = "cuda")]
use std::ffi::c_void;

/// Stride-aware BF16 SiLU. Output is always a freshly allocated contiguous
/// row-major BF16 tensor with `x.shape()`.
///
/// * Contiguous `x`: delegates to `bf16_ops::silu_bf16` unchanged (bit-exact
///   with the pre-migration path).
/// * Strided `x` (custom strides and/or non-zero view offset): walks the
///   input via `flame_silu_bf16_strided` and writes into a contig output.
///
/// Rank must be ≤ `flame-core`'s `Strides` capacity (6). Returns an error
/// for higher ranks; no real DL tensor in the codebase exceeds 5.
pub fn silu_bf16_iter(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16, "silu_bf16_iter expects BF16");

    // Fast path: contig input → existing vectorized NVRTC kernel. Every
    // caller before this session hits this branch. Same SASS, same timing,
    // same output bits.
    if x.is_contiguous() {
        return crate::bf16_ops::silu_bf16(x);
    }

    // Slow path: strided input → new OffsetCalculator-driven kernel.
    #[cfg(feature = "cuda")]
    {
        strided_impl(x)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Error::InvalidOperation(
            "silu_bf16_iter strided path requires the cuda feature".into(),
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
            "silu_bf16_iter: rank {} exceeds FLAME_MAX_DIMS ({})",
            rank, FLAME_MAX_DIMS
        )));
    }
    let n = shape.elem_count();

    // Host-side metadata captured before we move `shape` into the output
    // Tensor. Buffers are stack-local; their lifetime extends past the FFI
    // call because `extern "C" int flame_silu_bf16_strided` copies the
    // contents into a local `StridedOffsetCalc` before the kernel launch —
    // no device-side reference after the call.
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

    // Fresh contig BF16 output — matches `bf16_ops::silu_bf16`'s allocation
    // pattern (pool-backed, skips the zero-fill that `Tensor::zeros_dtype`
    // would pay for).
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

    // Device pointers (to storage base, pre-offset).
    let x_ptr = x.as_device_ptr_bf16("silu_bf16_iter:x")? as *const c_void;
    let y_ptr = out.as_mut_device_ptr_bf16("silu_bf16_iter:y")? as *mut c_void;
    let stream: *mut c_void = x.device.cuda_stream_raw_ptr();

    let status = unsafe {
        ffi::flame_silu_bf16_strided(
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
            "flame_silu_bf16_strided failed with code {}",
            status
        )));
    }
    Ok(out)
}
