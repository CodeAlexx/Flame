//! CUDA-only narrow implementation using existing kernels

use crate::{Tensor, Result, FlameError, Shape, DType};
use crate::cuda::ffi;
use crate::cuda::dtype_tag::dtype_to_tag;
use std::ffi::c_void;
use crate::device::CudaStreamRawPtrExt;
use crate::tensor_storage::TensorStorage;
use crate::tensor::{alloc_zeros_from_pool, TensorId};
use cudarc::driver::{LaunchConfig, LaunchAsync};
use std::sync::Arc;

impl Tensor {
    /// General narrow along `dim` using CUDA gather with strided input.
    pub fn narrow_general_cuda(&self, dim: usize, start: usize, length: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        let rank = dims.len();
        if dim >= rank {
            return Err(FlameError::InvalidOperation(format!(
                "narrow: dim {} out of range for rank {}",
                dim, rank
            )));
        }
        if start + length > dims[dim] {
            return Err(FlameError::InvalidOperation(format!(
                "narrow: range [{}..{}) exceeds dim {} (size {})",
                start, start + length, dim, dims[dim]
            )));
        }

        // Output shape and strides (row-major)
        let mut out_dims = dims.to_vec();
        out_dims[dim] = length;
        let mut out_strides_elems = vec![0i64; rank];
        if rank > 0 {
            out_strides_elems[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                out_strides_elems[i] = out_strides_elems[i + 1] * (out_dims[i + 1] as i64);
            }
        }
        let out_shape = Shape::from_dims(&out_dims);
        let n_elements: i64 = out_dims.iter().fold(1i64, |acc, &d| acc * d as i64);

        // Element size from logical dtype
        let elem_size: i64 = self.dtype().size_in_bytes() as i64;

        // Allocate destination with logical dtype
        let dtype = self.dtype();
        let mut dst = Tensor::zeros_dtype(out_shape, dtype, self.device.clone())?;

        // Build contiguous input strides (elements)
        let in_strides_elems: Vec<i64> = self.shape.strides().into_iter().map(|x| x as i64).collect();
        let out_shape_i64: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();

        // Raw pointers
        let src_ptr = self.cuda_ptr() as *const c_void;
        let dst_ptr = dst.cuda_ptr_mut() as *mut c_void;

        // Device stream (default stream if not configured)
        let stream: *mut c_void = self.device().cuda_stream_raw_ptr();

        let code = unsafe {
            ffi::flame_narrow_strided_launch(
                src_ptr,
                dst_ptr,
                rank as i32,
                out_shape_i64.as_ptr(),
                in_strides_elems.as_ptr(),
                out_strides_elems.as_ptr(),
                dim as i32,
                start as i64,
                elem_size,
                n_elements,
                stream,
            )
        };
        if code != 0 {
            return Err(FlameError::CudaError(format!(
                "narrow_strided_launch failed with code {}",
                code
            )));
        }
        Ok(dst)
    }

    /// Narrow (slice) a tensor along a dimension - CUDA only, no CPU fallback
    pub fn narrow_cuda(&self, dim: usize, start: usize, length: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of range for tensor with {} dimensions", dim, dims.len())
            ));
        }
        
        if start + length > dims[dim] {
            return Err(FlameError::InvalidOperation(
                format!("Slice [{}, {}) out of range for dimension {} of size {}", 
                    start, start + length, dim, dims[dim])
            ));
        }
        
        // For batch dimension (dim 0) with single item, we can use existing slice operation
        if dim == 0 && length == 1 {
            let rank = dims.len();
            // Use general slice helper ranges API
            let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(rank);
            for (i, &d) in dims.iter().enumerate() {
                if i == dim { ranges.push((start, start + length)); }
                else { ranges.push((0, d)); }
            }
            return self.slice(&ranges);
        }
        
        // For other cases, we'll need to implement a proper narrow
        // For now, return an error to avoid CPU fallback
        Err(FlameError::InvalidOperation(
            "narrow operation only supported for batch dimension (dim=0) currently".to_string()
        ))
    }

    /// Scatter-add backward for narrow: accumulates grad_out into grad_in slice
    pub fn narrow_backward_scatter_add_cuda(
        grad_out: &Tensor,
        grad_in: &mut Tensor,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<()> {
        // Validate shapes
        let in_shape = grad_in.shape();
        let out_shape = grad_out.shape();
        let rank = in_shape.rank();
        if rank != out_shape.rank() {
            return Err(FlameError::InvalidOperation("narrow backward: rank mismatch".into()));
        }
        if dim >= rank {
            return Err(FlameError::InvalidOperation(format!(
                "narrow backward: dim {} out of range",
                dim
            )));
        }
        let in_dims = in_shape.dims();
        let out_dims = out_shape.dims();
        for i in 0..rank {
            if i == dim {
                if out_dims[i] != length {
                    return Err(FlameError::InvalidOperation(format!(
                        "narrow backward: out length {} != {} at axis {}",
                        out_dims[i], length, i
                    )));
                }
                if start + length > in_dims[i] {
                    return Err(FlameError::InvalidOperation(format!(
                        "narrow backward: range [{}..{}) exceeds input dim {} (size {})",
                        start, start + length, i, in_dims[i]
                    )));
                }
            } else if out_dims[i] != in_dims[i] {
                return Err(FlameError::InvalidOperation(format!(
                    "narrow backward: shape mismatch at axis {} (out {} vs in {})",
                    i, out_dims[i], in_dims[i]
                )));
            }
        }

        // DType tag mapping
        let dtype_tag: i32 = dtype_to_tag(grad_in.dtype());

        // Strides and shapes
        let in_strides_elems: Vec<i64> = in_shape.strides().into_iter().map(|x| x as i64).collect();
        let mut out_strides_elems = vec![0i64; rank];
        if rank > 0 {
            out_strides_elems[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                out_strides_elems[i] = out_strides_elems[i + 1] * (out_dims[i + 1] as i64);
            }
        }
        let out_shape_i64: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
        let n_elements: i64 = out_shape_i64.iter().product();
        let elem_size: i64 = grad_in.dtype().size_in_bytes() as i64;

        // Raw pointers
        let go_ptr = grad_out.cuda_ptr() as *const c_void;
        let gi_ptr = grad_in.cuda_ptr_mut() as *mut c_void;
        let stream: *mut c_void = grad_in.device().cuda_stream_raw_ptr();

        let code = unsafe {
            ffi::flame_narrow_backward_scatter_add_launch(
                go_ptr,
                gi_ptr,
                rank as i32,
                out_shape_i64.as_ptr(),
                in_strides_elems.as_ptr(),
                out_strides_elems.as_ptr(),
                dim as i32,
                start as i64,
                elem_size,
                n_elements,
                dtype_tag,
                stream,
            )
        };
        if code != 0 {
            return Err(FlameError::CudaError(format!(
                "narrow_backward_scatter_add_launch failed with code {}",
                code
            )));
        }
        Ok(())
    }
}
