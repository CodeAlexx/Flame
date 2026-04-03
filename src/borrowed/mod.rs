#![allow(dead_code)]

//! Borrowed-weight primitives for BF16 matmuls.
//!
//! These helpers let callers pass non-owning BF16 buffers (typically carved
//! from a staging arena) directly into the cuBLASLt GEMM path without first
//! materializing a `Tensor`. The API intentionally stays lean so the higher
//! layers (SD3.5 T5 stager) can compose their own pinned-host/device allocators.

use crate::cuda_ops_bf16::{
    FC_ERR_INVALID_ARGUMENT, FC_ERR_LAUNCH, FC_ERR_OOM, FC_ERR_UNSUPPORTED, FC_OK,
    FC_STATUS_LT_FALLBACK,
};
use crate::cuda_ops_ffi::{
    fc_gemm_bf16, tensor_as_view_bf16, tensor_as_view_bf16_mut, CudaStream, FcTensorView,
};
use crate::{strict, DType, Error, Result, Shape, Tensor};
use core::ffi::c_void;
use cudarc::driver::CudaDevice;
use log::warn;
use std::cmp;
use std::ptr;
use std::sync::Arc;

/// Guard that runs a release callback when dropped.
pub struct BorrowGuard {
    release: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl BorrowGuard {
    /// Create a guard from a release closure.
    pub fn new<F>(release: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            release: Some(Box::new(release)),
        }
    }

    /// Create a guard that does nothing (useful when callers don't need cleanup).
    pub fn noop() -> Self {
        Self { release: None }
    }

    /// Consume the guard without running the release closure.
    pub fn into_inner(mut self) -> Option<Box<dyn FnOnce() + Send + 'static>> {
        self.release.take()
    }
}

impl Drop for BorrowGuard {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

/// Non-owning BF16 view backed by device memory.
pub struct DeviceBf16View {
    ffi: FcTensorView,
    dims: Vec<usize>,
    strides: Vec<usize>,
    device: Arc<CudaDevice>,
}

impl DeviceBf16View {
    /// Build a view from explicit dimensions and strides.
    pub fn with_strides(
        ptr: *const u16,
        dims: &[usize],
        strides: &[usize],
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        if ptr.is_null() {
            return Err(Error::InvalidInput(
                "DeviceBf16View::with_strides: data pointer cannot be null".into(),
            ));
        }
        if dims.len() != strides.len() {
            return Err(Error::InvalidInput(
                "DeviceBf16View::with_strides: dims/strides rank mismatch".into(),
            ));
        }
        if dims.len() > FcTensorView::MAX_RANK {
            return Err(Error::InvalidInput(format!(
                "DeviceBf16View::with_strides: rank {} exceeds FcTensorView::MAX_RANK ({})",
                dims.len(),
                FcTensorView::MAX_RANK
            )));
        }

        let mut ffi_dims = [0i64; FcTensorView::MAX_RANK];
        let mut ffi_strides = [0i64; FcTensorView::MAX_RANK];
        for (i, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
            ffi_dims[i] = dim as i64;
            ffi_strides[i] = stride as i64;
        }

        Ok(Self {
            ffi: FcTensorView {
                data: ptr as *mut c_void,
                dims: ffi_dims,
                strides: ffi_strides,
                rank: dims.len() as i32,
            },
            dims: dims.to_vec(),
            strides: strides.to_vec(),
            device,
        })
    }

    /// Build a contiguous row-major view.
    pub fn contiguous(ptr: *const u16, dims: &[usize], device: Arc<CudaDevice>) -> Result<Self> {
        let strides = contiguous_strides(dims);
        Self::with_strides(ptr, dims, &strides, device)
    }

    /// Borrow the underlying device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Dimensionality metadata.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Tensor rank.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Strides metadata.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Return the FFI-friendly tensor view (by value).
    pub fn as_ffi_view(&self) -> FcTensorView {
        self.ffi
    }
}

/// Lt matmul context for borrowed GEMMs.
pub struct LtMatmulCtx {
    stream: CudaStream,
}

impl LtMatmulCtx {
    /// Create a context from an explicit CUDA stream.
    pub fn new(stream: CudaStream) -> Self {
        Self { stream }
    }

    /// Create a context using the default stream of `tensor`.
    pub fn for_tensor(tensor: &Tensor) -> Self {
        Self::new(CudaStream::from_raw(tensor.device().cuda_stream_raw_ptr()))
    }

    /// Access the CUDA stream.
    pub fn stream(&self) -> CudaStream {
        self.stream
    }

    /// Execute BF16 GEMM where `w`/`bias` are borrowed views.
    pub fn gemm_into_borrowed(
        &self,
        out: &mut Tensor,
        input: &Tensor,
        weight: &DeviceBf16View,
        bias: Option<&DeviceBf16View>,
    ) -> Result<()> {
        ensure_bf16_tensor(input, "borrowed_gemm:input")?;
        ensure_bf16_tensor(out, "borrowed_gemm:out")?;
        if !Arc::ptr_eq(input.device(), weight.device()) {
            return Err(Error::InvalidInput(
                "borrowed_gemm: weight device mismatch".into(),
            ));
        }
        if let Some(b) = bias {
            if !Arc::ptr_eq(input.device(), b.device()) {
                return Err(Error::InvalidInput(
                    "borrowed_gemm: bias device mismatch".into(),
                ));
            }
        }

        let (m, k) = matrix_dims(input.shape().dims(), "borrowed_gemm:input")?;
        let (k_w, n) = matrix_dims(weight.dims(), "borrowed_gemm:weight")?;
        if k_w != k {
            return Err(Error::InvalidInput(format!(
                "borrowed_gemm: weight K mismatch (expected {k}, got {k_w})"
            )));
        }

        let out_dims = out.shape().dims();
        if out_dims.len() != 2 || out_dims[0] != m || out_dims[1] != n {
            return Err(Error::InvalidInput(format!(
                "borrowed_gemm: output shape {:?} incompatible with [{m}, {n}]",
                out_dims
            )));
        }

        if let Some(b) = bias {
            let dims = b.dims();
            let ok = match dims {
                [cols] => *cols == n,
                [rows, cols] => *rows == 1 && *cols == n,
                _ => false,
            };
            if !ok {
                return Err(Error::InvalidInput(format!(
                    "borrowed_gemm: bias shape {:?} incompatible with cols {n}",
                    dims
                )));
            }
        }

        let vx = tensor_as_view_bf16(input, "borrowed_gemm:input")?;
        let mut vy = tensor_as_view_bf16_mut(out, "borrowed_gemm:out")?;
        let vw = weight.as_ffi_view();
        let bias_view = bias.map(|b| b.as_ffi_view());
        let bias_ptr = bias_view
            .as_ref()
            .map(|view| view as *const _)
            .unwrap_or(ptr::null());

        let status = unsafe { fc_gemm_bf16(&vx, &vw, bias_ptr, &mut vy, self.stream.as_raw()) };
        process_gemm_status(status, m, n, k)
    }
}

fn contiguous_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut stride = 1usize;
    for idx in (0..dims.len()).rev() {
        strides[idx] = stride;
        stride *= cmp::max(dims[idx], 1);
    }
    strides
}

fn ensure_bf16_tensor(t: &Tensor, tag: &str) -> Result<()> {
    if t.dtype() != DType::BF16 || t.storage_dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "{tag}: expected BF16 storage (logical {:?}, storage {:?})",
            t.dtype(),
            t.storage_dtype()
        )));
    }
    Ok(())
}

fn matrix_dims(dims: &[usize], tag: &str) -> Result<(usize, usize)> {
    if dims.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "{tag}: expected rank-2 tensor, got {:?}",
            dims
        )));
    }
    Ok((dims[0], dims[1]))
}

fn process_gemm_status(status: i32, m: usize, n: usize, k: usize) -> Result<()> {
    match status {
        FC_OK => Ok(()),
        FC_STATUS_LT_FALLBACK => {
            strict::record_layout_fix("cuda_ops.gemm_bf16.lt_fallback", &Shape::from_dims(&[m, n]));
            warn!(
                "borrowed_gemm: cuBLASLt fallback fired for (m={}, n={}, k={})",
                m, n, k
            );
            Ok(())
        }
        FC_ERR_INVALID_ARGUMENT => Err(Error::InvalidInput(
            "borrowed_gemm: invalid argument".into(),
        )),
        FC_ERR_LAUNCH => Err(Error::Cuda("borrowed_gemm: kernel launch failed".into())),
        FC_ERR_OOM => Err(Error::Cuda("borrowed_gemm: cudaMalloc failed (OOM)".into())),
        FC_ERR_UNSUPPORTED => {
            strict::record_layout_fix("cuda_ops.gemm_bf16.unsupported", &Shape::from_dims(&[m, n]));
            Err(Error::Unsupported(
                "borrowed_gemm: unsupported layout".into(),
            ))
        }
        other => Err(Error::Cuda(format!(
            "borrowed_gemm: unexpected status {}",
            other
        ))),
    }
}

impl FcTensorView {
    const MAX_RANK: usize = 8;
}
