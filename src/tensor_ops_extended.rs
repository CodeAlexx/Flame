//! Extended tensor operations for FLAME
//! Adds operations needed for diffusion model training

use crate::autograd::{AutogradContext, Op};
use crate::config::default_dtype;
use crate::cuda_ops::GpuOps;
use crate::DType;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::{
    cuda_ops_bf16,
    cuda_ops_ffi::CudaStream,
    device::CudaStreamRawPtrExt,
    staging::{bf16_copy_async_tagged, ArenaScratch},
};
use crate::{Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use std::ffi::c_void;
use std::sync::Arc;

impl Tensor {
    // NOTE [Factories & BF16 policy]
    // These high-level factory helpers currently build host-side FP32 vectors then upload.
    // Do NOT use them on hot training paths. Prefer BF16-specialized GPU kernels that:
    //   - generate FP32 in registers per element; and
    //   - immediately convert and store as BF16 (no full FP32 buffers).
    // See: bf16_ops + bf16_convert modules and bf16_u16 feature.
    /// Random uniform tensor
    pub fn rand(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let data: Vec<f32> = (0..size).map(|_| rng.gen::<f32>()).collect();

        Self::from_vec(data, shape, device)
    }

    /// Create a tensor with uniform distribution
    pub fn uniform(shape: Shape, low: f32, high: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let data: Vec<f32> = (0..size).map(|_| rng.gen_range(low..high)).collect();

        Self::from_vec(data, shape, device)
    }

    /// Create a tensor with normal distribution
    pub fn normal(shape: Shape, mean: f32, std: f32, device: Arc<CudaDevice>) -> Result<Self> {
        Self::randn(shape, mean, std, device)
    }

    /// Permute/transpose dimensions
    pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim0 >= dims.len() || dim1 >= dims.len() {
            return Err(Error::InvalidOperation(format!(
                "Transpose dimensions out of bounds: {} and {} for tensor with {} dims",
                dim0,
                dim1,
                dims.len()
            )));
        }

        // Create permutation
        let mut perm: Vec<usize> = (0..dims.len()).collect();
        perm.swap(dim0, dim1);

        self.permute(&perm)
    }

    /// Squeeze dimensions of size 1
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Tensor> {
        let dims = self.shape.dims();

        let new_dims: Vec<usize> = if let Some(d) = dim {
            if d >= dims.len() {
                return Err(Error::InvalidOperation(format!(
                    "Dimension {} out of range",
                    d
                )));
            }
            if dims[d] != 1 {
                return self.clone_result();
            }
            dims.iter()
                .enumerate()
                .filter(|(i, _)| *i != d)
                .map(|(_, &size)| size)
                .collect()
        } else {
            dims.iter().copied().filter(|&size| size != 1).collect()
        };

        self.reshape(&new_dims)
    }

    /// Unsqueeze: Add a dimension of size 1
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim > dims.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of range for unsqueeze",
                dim
            )));
        }

        let mut new_dims = dims.to_vec();
        new_dims.insert(dim, 1);

        self.reshape(&new_dims)
    }

    /// Greater than comparison (broadcast-safe).
    /// Phase 9/10: BF16+BF16 → TensorIterator (`ops::gt_iter::gt_bf16_iter`),
    /// which writes BF16 0.0/1.0 sentinels bit-exactly matching PyTorch's
    /// `opmath_t=float` semantics. Other dtype combinations stay on
    /// `GpuOps::cmp_gt` (F32 round-trip). Comparisons have no autograd
    /// tape — `.requires_grad` doesn't propagate through a boolean mask.
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        crate::tensor_iterator::dispatch_comparison_bf16(
            self,
            other,
            crate::tensor_iterator::ops::comparison::gt_bf16_iter,
            GpuOps::cmp_gt,
        )
    }

    /// Greater than or equal comparison. See `gt` for dispatch notes.
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        crate::tensor_iterator::dispatch_comparison_bf16(
            self,
            other,
            crate::tensor_iterator::ops::comparison::ge_bf16_iter,
            GpuOps::cmp_ge,
        )
    }

    /// Less than comparison. See `gt` for dispatch notes.
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        crate::tensor_iterator::dispatch_comparison_bf16(
            self,
            other,
            crate::tensor_iterator::ops::comparison::lt_bf16_iter,
            GpuOps::cmp_lt,
        )
    }

    /// Not equal comparison. See `gt` for dispatch notes. IEEE:
    /// ne(NaN, NaN) = true.
    pub fn ne(&self, other: &Tensor) -> Result<Tensor> {
        crate::tensor_iterator::dispatch_comparison_bf16(
            self,
            other,
            crate::tensor_iterator::ops::comparison::ne_bf16_iter,
            GpuOps::cmp_ne,
        )
    }

    /// Conditional where operation
    pub fn where_op(&self, condition: &Tensor, other: &Tensor) -> Result<Tensor> {
        // self is the true value, other is the false value, condition is the mask
        condition.where_tensor(self, other)
    }

    /// Compute variance along dimensions
    pub fn var(&self, dims: &[usize], unbiased: bool, keepdim: bool) -> Result<Tensor> {
        // Compute mean
        let mean = self.mean_along_dims(dims, keepdim)?;

        // Compute (x - mean)^2
        let diff = self.sub(&mean)?;
        let sq_diff = diff.square()?;

        // Compute mean of squared differences
        let var = sq_diff.mean_along_dims(dims, keepdim)?;

        // If unbiased, apply Bessel's correction
        if unbiased {
            let n = dims
                .iter()
                .map(|&d| self.shape.dims()[d])
                .product::<usize>() as f32;
            if n > 1.0 {
                var.mul_scalar(n / (n - 1.0))
            } else {
                Ok(var)
            }
        } else {
            Ok(var)
        }
    }

    /// Compute mean along dimensions
    pub fn mean_along_dims(&self, dims: &[usize], _keepdim: bool) -> Result<Tensor> {
        // Sum along dimensions
        let mut result = self.clone_result()?;
        for &dim in dims {
            result = result.sum_dim(dim)?;
        }

        // Compute divisor
        let divisor = dims
            .iter()
            .map(|&d| self.shape.dims()[d])
            .product::<usize>() as f32;

        // Divide by count
        result.mul_scalar(1.0 / divisor)
    }

    /// Create a tensor like another tensor
    pub fn full_like(&self, value: f32) -> Result<Tensor> {
        Self::full(self.shape.clone(), value, self.device.clone())
    }

    /// Create zeros like another tensor
    pub fn zeros_like(&self) -> Result<Tensor> {
        Self::zeros(self.shape.clone(), self.device.clone())
    }

    /// Create ones like another tensor  
    pub fn ones_like(&self) -> Result<Tensor> {
        Self::ones(self.shape.clone(), self.device.clone())
    }

    /// Apply affine transformation: a * x + b
    pub fn affine(&self, a: f32, b: f32) -> Result<Tensor> {
        self.mul_scalar(a)?.add_scalar(b)
    }

    /// Sum along dimension with option to keep dimension (GPU-only)
    pub fn sum_keepdim(&self, dim: isize) -> Result<Tensor> {
        let ndim = self.shape.dims().len() as isize;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        if dim >= self.shape.dims().len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of bounds",
                dim
            )));
        }
        // Delegate to GPU axis-reduction that keeps dimension (FP32 compute)
        GpuOps::sum_dim_keepdim(self, dim)
    }

    /// Permute tensor dimensions (already implemented above)
    /// Chunk tensor into n chunks along specified dimension
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Tensor>> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            )));
        }

        let dim_size = shape[dim];
        if dim_size % chunks != 0 {
            return Err(Error::InvalidOperation(format!(
                "Cannot evenly chunk dimension {} of size {} into {} chunks",
                dim, dim_size, chunks
            )));
        }

        let chunk_size = dim_size / chunks;
        let mut result = Vec::new();

        for i in 0..chunks {
            let start = i * chunk_size;
            let end = (i + 1) * chunk_size;

            // Create slice indices
            let mut slice_ranges: Vec<(usize, usize)> = Vec::new();
            for (d, &size) in shape.iter().enumerate() {
                if d == dim {
                    slice_ranges.push((start, end));
                } else {
                    slice_ranges.push((0, size));
                }
            }

            // Slice the tensor
            let chunk = self.slice(&slice_ranges)?;
            result.push(chunk);
        }

        Ok(result)
    }

    /// Slice tensor along multiple dimensions
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<Tensor> {
        let mut output = GpuOps::slice(self, ranges)?;
        // Record autograd slice op
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Slice {
                        input: self.id,
                        ranges: ranges.to_vec(),
                        input_shape: self.shape.clone(),
                    },
                    vec![(self.id, self.clone())],
                );
            }
        }
        Ok(output)
    }

    /// Concatenate tensors along a dimension
    pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(Error::InvalidOperation(
                "Cannot concatenate empty tensor list".into(),
            ));
        }

        // Stride refactor Phase 2a safety net: cat copies via linear DMA /
        // slice-slice assuming row-major contiguous storage. Materialize any
        // strided views first.
        let materialized: Vec<Tensor> = tensors
            .iter()
            .map(|t| {
                if t.is_contiguous() {
                    Ok((*t).clone())
                } else {
                    t.contiguous()
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let tensors_vec: Vec<&Tensor> = materialized.iter().collect();
        let tensors: &[&Tensor] = &tensors_vec;

        // Check all tensors have same shape except for concat dimension
        let first_shape = tensors[0].shape().dims();
        let device = tensors[0].device().clone();

        for tensor in tensors.iter().skip(1) {
            let shape = tensor.shape().dims();
            if shape.len() != first_shape.len() {
                return Err(Error::InvalidOperation(
                    "All tensors must have same number of dimensions".into(),
                ));
            }

            for (i, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i != dim && s1 != s2 {
                    return Err(Error::InvalidOperation(format!(
                        "Dimension {} must match for concatenation (got {} and {})",
                        i, s1, s2
                    )));
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        output_shape[dim] = tensors.iter().map(|t| t.shape().dims()[dim]).sum();

        let dtype = tensors[0].dtype();
        for tensor in tensors.iter() {
            if tensor.dtype() != dtype {
                return Err(Error::InvalidOperation(
                    "Concatenation requires matching dtypes".into(),
                ));
            }
            if tensor.device().ordinal() != device.ordinal() {
                return Err(Error::InvalidOperation(
                    "Tensors must reside on the same device".into(),
                ));
            }
        }

        let mut output =
            Tensor::zeros_dtype(Shape::from_dims(&output_shape), dtype, device.clone())?;

        let rank = output_shape.len();
        let outer: usize = if dim == 0 {
            1
        } else {
            output_shape[..dim].iter().product()
        };
        let row_elems: usize = if dim + 1 >= rank {
            1
        } else {
            output_shape[dim + 1..].iter().product()
        };
        let rows_per_outer = output_shape[dim];
        let total_rows_per_outer = rows_per_outer * row_elems;

        #[derive(Clone, Copy)]
        struct TensorSliceInfo {
            rows: usize,
        }

        let infos: Vec<TensorSliceInfo> = tensors
            .iter()
            .map(|t| TensorSliceInfo {
                rows: t.shape().dims()[dim],
            })
            .collect();

        match dtype {
            DType::F32 => {
                let out_slice = output
                    .storage_mut()
                    .try_as_mut_slice_f32()
                    .map_err(|_| Error::InvalidOperation("cat: expected F32 storage".into()))?;
                let mut prefix_rows = 0usize;
                for (tensor, info) in tensors.iter().zip(infos.iter()) {
                    let src_slice = tensor
                        .storage_ref()
                        .try_as_slice_f32()
                        .map_err(|_| Error::InvalidOperation("cat: expected F32 storage".into()))?;
                    let len_per_outer = info.rows * row_elems;
                    for o in 0..outer {
                        let src_start = o * len_per_outer;
                        let src_end = src_start + len_per_outer;
                        let dst_outer_base = o * total_rows_per_outer;
                        let dst_start = dst_outer_base + prefix_rows * row_elems;
                        let dst_end = dst_start + len_per_outer;
                        let src_view = src_slice.slice(src_start..src_end);
                        let mut dst_view = out_slice.slice_mut(dst_start..dst_end);
                        tensor
                            .device()
                            .dtod_copy(&src_view, &mut dst_view)
                            .map_err(|e| Error::Cuda(format!("cat F32 copy failed: {e:?}")))?;
                    }
                    prefix_rows += info.rows;
                }
            }
            DType::BF16 => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    return Err(Error::Unsupported(
                        "cat BF16 requires bf16_u16 feature".into(),
                    ));
                }
                #[cfg(feature = "bf16_u16")]
                {
                    // One `cuMemcpy2DAsync_v2` per input tensor on the null stream:
                    // the CUDA DMA engine strides across the `outer` dimension in a
                    // single call instead of one `flame_k_copy_bf16` launch per
                    // (tensor, outer) pair. For joint-attention QKV cats on
                    // [B=1, H=12, N, D=128] this replaces 24 kernel launches with a
                    // single DMA op — on motif, ~33 000 → ~1 400 total launches per
                    // forward, dropping the `flame_k_copy_bf16` total from 31 % of
                    // GPU time to well under 1 %.
                    use cudarc::driver::sys::{
                        CUdeviceptr, CUmemorytype_enum, CUresult, CUstream, CUDA_MEMCPY2D,
                    };
                    let device = output.device().clone();
                    let stream_ptr: CUstream = core::ptr::null_mut();
                    let dst_base = output.as_mut_device_ptr_bf16("cat:dst")? as *mut u16;
                    let bf16_size = std::mem::size_of::<u16>();
                    // total_rows_per_outer already = sum_input_rows * row_elems
                    // (elements per outer slice of output).
                    let dst_pitch = total_rows_per_outer * bf16_size;
                    let mut prefix_rows = 0usize;
                    for (tensor, info) in tensors.iter().zip(infos.iter()) {
                        let src_base = tensor.as_device_ptr_bf16("cat:src")? as *const u16;
                        let len_per_outer = info.rows * row_elems;
                        let width_bytes = len_per_outer * bf16_size;
                        if outer == 0 || len_per_outer == 0 {
                            prefix_rows += info.rows;
                            continue;
                        }
                        if outer == 1 {
                            // Single outer slice: plain 1-D async D2D copy, skips
                            // the 2D descriptor construction.
                            let dst_start = prefix_rows * row_elems;
                            let dst_ptr = unsafe { dst_base.add(dst_start) } as *mut c_void;
                            let src_ptr = src_base as *const c_void;
                            let stream = CudaStream::from_raw(device.cuda_stream_raw_ptr());
                            bf16_copy_async_tagged(
                                dst_ptr,
                                src_ptr,
                                len_per_outer,
                                &stream,
                                "cat:outer1",
                            )?;
                        } else {
                            let dst_start_elems = prefix_rows * row_elems;
                            let dst_start_ptr = unsafe { dst_base.add(dst_start_elems) };
                            // Manual construction — CUDA_MEMCPY2D_st's CUmemorytype
                            // field has no 0 variant, so `mem::zeroed()` panics on
                            // newer rustc.
                            let params = CUDA_MEMCPY2D {
                                srcXInBytes: 0,
                                srcY: 0,
                                srcMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
                                srcHost: std::ptr::null(),
                                srcDevice: src_base as CUdeviceptr,
                                srcArray: std::ptr::null_mut(),
                                srcPitch: width_bytes,
                                dstXInBytes: 0,
                                dstY: 0,
                                dstMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
                                dstHost: std::ptr::null_mut(),
                                dstDevice: dst_start_ptr as CUdeviceptr,
                                dstArray: std::ptr::null_mut(),
                                dstPitch: dst_pitch,
                                WidthInBytes: width_bytes,
                                Height: outer,
                            };
                            // 2026-05-15 trap: validate src/dst ptrs against
                            // the BF16 pool's live-ptr tracker. When
                            // FLAME_POOL_TRAP_BF16=1 and either ptr's last
                            // recorded state is not Live, panic with
                            // provenance instead of getting cuMemcpy2D's
                            // opaque INVALID_VALUE. This is the soul.md-
                            // pattern trap for the Klein 9B step-13 crash.
                            crate::cuda_alloc_pool::trap_validate_bf16_ptr(
                                params.srcDevice as u64,
                                "Tensor::cat/cuMemcpy2D src",
                            );
                            crate::cuda_alloc_pool::trap_validate_bf16_ptr(
                                params.dstDevice as u64,
                                "Tensor::cat/cuMemcpy2D dst",
                            );
                            let rc = unsafe {
                                cudarc::driver::sys::lib().cuMemcpy2DAsync_v2(&params, stream_ptr)
                            };
                            if rc != CUresult::CUDA_SUCCESS {
                                return Err(Error::Cuda(format!(
                                    "cuMemcpy2DAsync_v2 (cat) failed: {rc:?} \
                                     src={:#x} dst={:#x} srcPitch={} dstPitch={} \
                                     width={} height={}",
                                    params.srcDevice,
                                    params.dstDevice,
                                    params.srcPitch,
                                    params.dstPitch,
                                    params.WidthInBytes,
                                    params.Height,
                                )));
                            }
                        }
                        prefix_rows += info.rows;
                    }
                    // Copies are enqueued on the null stream. Subsequent null-stream
                    // consumers (cuBLASLt GEMMs, elementwise kernels) sync via
                    // legacy-default-stream semantics.
                }
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "cat: dtype {:?} not supported",
                    other
                )));
            }
        }

        // AUTOGRAD: Record operation if needed
        let requires_grad = tensors.iter().any(|t| t.requires_grad);
        if requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                let mut saved_tensors = Vec::new();
                let mut input_ids = Vec::new();

                for tensor in tensors {
                    let id = tensor.id;
                    saved_tensors.push((id, (*tensor).alias()));
                    input_ids.push(id);
                }

                AutogradContext::record_op(
                    output.id,
                    Op::Cat {
                        inputs: input_ids,
                        dim,
                    },
                    saved_tensors,
                );
            }
        }

        // Contract: `cat` ALWAYS returns a row-major contiguous tensor. Output
        // is built from `Tensor::zeros_dtype` (fresh contiguous allocation)
        // and the per-dtype branches above write into that buffer via DMA;
        // no path produces a strided view. Callers may rely on this contract
        // and avoid defensive `.contiguous()` calls. Release-time `assert!`
        // (not `debug_assert!`) — production builds also depend on this
        // guarantee (e.g. inference-flame turbo_vaed reshape-after-cat).
        // Cost is one stride-pattern check, sub-microsecond.
        assert!(
            output.is_contiguous(),
            "cat output not contiguous (contract violation): \
             shape={:?} custom_strides={:?} view_offset={}",
            output.shape().dims(),
            output.custom_strides,
            output.view_offset,
        );

        Ok(output)
    }

    /// Index select along a dimension
    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            )));
        }

        // Indices should be 1D
        let indices_shape = indices.shape().dims();
        if indices_shape.len() != 1 {
            return Err(Error::InvalidOperation(format!(
                "Indices must be 1D, got shape {:?}",
                indices_shape
            )));
        }

        let num_indices = indices.shape().elem_count();
        let mut out_dims = shape.to_vec();
        out_dims[dim] = num_indices;
        let out_shape = Shape::from_dims(&out_dims);

        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if self.dtype() == DType::BF16 && self.storage_dtype() == DType::BF16 {
            let indices_owned;
            let indices_i32 =
                if indices.dtype() == DType::I32 && indices.storage_dtype() == DType::I32 {
                    indices
                } else {
                    indices_owned = indices.to_dtype(DType::I32)?;
                    &indices_owned
                };
            if !Arc::ptr_eq(self.device(), indices_i32.device()) {
                return Err(Error::InvalidOperation(
                    "index_select requires indices to reside on the same device".into(),
                ));
            }

            let scratch = ArenaScratch::from_tensor_with_align(self, ArenaScratch::DEFAULT_ALIGN);
            let mut output = scratch.borrow_shape(out_shape.clone())?;
            cuda_ops_bf16::index_select_bf16_into(self, dim, indices_i32, &mut output)?;

            if self.requires_grad {
                output.requires_grad = true;
                if AutogradContext::is_recording() {
                    AutogradContext::record_op(
                        output.id,
                        Op::IndexSelect {
                            input: self.id,
                            indices: indices.id(),
                            dim,
                        },
                        vec![(self.id, self.alias()), (indices.id(), indices.alias())],
                    );
                }
            }
            return Ok(output);
        }

        let mut output = GpuOps::index_select(self, dim, indices)?;
        // Record autograd op for backward via scatter_add
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::IndexSelect {
                        input: self.id,
                        indices: indices.id(),
                        dim,
                    },
                    vec![(self.id, self.alias()), (indices.id(), indices.alias())],
                );
            }
        }
        Ok(output)
    }

    /// Replace slices along `dim` of `self` at `indices` with the
    /// corresponding slices from `values`. Returns a NEW tensor with the
    /// same shape as `self`. Non-indexed positions are copied from `self`,
    /// indexed positions are copied from `values`.
    ///
    /// Constraints:
    /// - `indices` must be 1-D with `len(indices) == values.shape[dim]`.
    /// - `values.shape` must match `self.shape` everywhere except `dim`,
    ///   where it equals `len(indices)`.
    /// - `self.dtype()` must equal `values.dtype()` (BF16 or F32).
    ///
    /// Autograd: gradient w.r.t. `self` masks out the indexed rows of
    /// upstream (they were overwritten); gradient w.r.t. `values` is
    /// `index_select(upstream, dim, indices)`.
    ///
    /// Used by TREAD's scatter-back step (`features::tread::TreadStep`).
    pub fn index_assign(&self, dim: usize, indices: &Tensor, values: &Tensor) -> Result<Tensor> {
        let mut output = self.index_assign_no_grad(dim, indices, values)?;

        if self.requires_grad || values.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::IndexAssign {
                        input: self.id,
                        indices: indices.id(),
                        values: values.id(),
                        dim,
                    },
                    vec![
                        (self.id, self.alias()),
                        (indices.id(), indices.alias()),
                        (values.id(), values.alias()),
                    ],
                );
            }
        }
        Ok(output)
    }

    /// Forward-only variant of [`Tensor::index_assign`]. Used internally by
    /// the backward pass to construct gradients without re-recording.
    pub fn index_assign_no_grad(
        &self,
        dim: usize,
        indices: &Tensor,
        values: &Tensor,
    ) -> Result<Tensor> {
        let self_dims = self.shape().dims();
        if dim >= self_dims.len() {
            return Err(Error::InvalidOperation(format!(
                "index_assign: dim {} out of bounds for tensor with {} dims",
                dim,
                self_dims.len()
            )));
        }
        let idx_shape = indices.shape().dims();
        if idx_shape.len() != 1 {
            return Err(Error::InvalidOperation(format!(
                "index_assign: indices must be 1-D, got shape {:?}",
                idx_shape
            )));
        }
        let n_idx = indices.shape().elem_count();

        let val_dims = values.shape().dims();
        if val_dims.len() != self_dims.len() {
            return Err(Error::InvalidOperation(format!(
                "index_assign: rank mismatch self={:?} values={:?}",
                self_dims, val_dims
            )));
        }
        for (d, (a, b)) in self_dims.iter().zip(val_dims.iter()).enumerate() {
            let expected = if d == dim { n_idx } else { *a };
            if *b != expected {
                return Err(Error::InvalidOperation(format!(
                    "index_assign: values shape {:?} incompatible with self {:?} along dim {} (expected {} got {})",
                    val_dims, self_dims, d, expected, b
                )));
            }
        }
        if self.dtype() != values.dtype() {
            return Err(Error::InvalidOperation(format!(
                "index_assign: dtype mismatch self={:?} values={:?}",
                self.dtype(),
                values.dtype()
            )));
        }
        if !Arc::ptr_eq(self.device(), values.device())
            || !Arc::ptr_eq(self.device(), indices.device())
        {
            return Err(Error::InvalidOperation(
                "index_assign: tensors must reside on the same device".into(),
            ));
        }

        // Build `inverse_mask` on host: for each k in [0, dim_size), record
        // the position in `indices` (or -1 if k is not in `indices`). This
        // lets the kernel decide src per element with O(1) lookup.
        let dim_size = self_dims[dim];
        let indices_f32 = indices.to_dtype(DType::F32)?;
        let host_idx: Vec<f32> = indices_f32.to_vec()?;
        let mut inverse_mask = vec![-1_i32; dim_size];
        for (pos, &fi) in host_idx.iter().enumerate() {
            let i = fi as i64;
            if i < 0 || i >= dim_size as i64 {
                return Err(Error::InvalidOperation(format!(
                    "index_assign: index {} out of bounds for dim_size {}",
                    i, dim_size
                )));
            }
            inverse_mask[i as usize] = pos as i32;
        }

        // Compute strides for self (output) and values.
        let ndim = self_dims.len();
        let mut self_strides = vec![1_i32; ndim];
        for i in (0..ndim - 1).rev() {
            self_strides[i] = self_strides[i + 1] * self_dims[i + 1] as i32;
        }
        let mut val_strides = vec![1_i32; ndim];
        for i in (0..ndim - 1).rev() {
            val_strides[i] = val_strides[i + 1] * val_dims[i + 1] as i32;
        }

        let self_dims_i32: Vec<i32> = self_dims.iter().map(|&x| x as i32).collect();

        let total_elem = self.shape().elem_count();
        let total_i32: i32 = total_elem
            .try_into()
            .map_err(|_| Error::InvalidOperation("index_assign: tensor too large".into()))?;

        let device = self.device().clone();

        let dtype = self.dtype();
        let output = match dtype {
            DType::F32 => index_assign_f32(
                self,
                values,
                &inverse_mask,
                &self_dims_i32,
                &self_strides,
                &val_strides,
                dim as i32,
                total_i32,
                ndim as i32,
                &device,
            )?,
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            DType::BF16 => index_assign_bf16(
                self,
                values,
                &inverse_mask,
                &self_dims_i32,
                &self_strides,
                &val_strides,
                dim as i32,
                total_i32,
                ndim as i32,
                &device,
            )?,
            other => {
                return Err(Error::Unsupported(format!(
                    "index_assign: dtype {:?} not yet supported (F32, BF16 only)",
                    other
                )));
            }
        };

        Ok(output)
    }

    /// Expand tensor to a new shape (broadcasting)
    pub fn expand(&self, new_shape: &[usize]) -> Result<Tensor> {
        let shape = self.shape().dims();

        // Validate expansion
        if new_shape.len() < shape.len() {
            return Err(Error::InvalidOperation(
                "Cannot expand to fewer dimensions".into(),
            ));
        }

        // Check compatibility
        let offset = new_shape.len() - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let new_dim = new_shape[i + offset];
            if dim != new_dim && dim != 1 {
                return Err(Error::InvalidOperation(format!(
                    "Cannot expand dimension {} from {} to {}",
                    i, dim, new_dim
                )));
            }
        }

        // For now, implement as broadcast_to
        self.broadcast_to(&Shape::from_dims(new_shape))
    }

    /// Compute natural logarithm (GPU). Phase 10: BF16 → TensorIterator,
    /// else → GpuOps::log.
    pub fn log(&self) -> Result<Tensor> {
        crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::transcendentals::log_bf16_iter,
            GpuOps::log,
        )
    }

    /// Compute reciprocal square root. Phase 10: BF16 → native
    /// TensorIterator rsqrt (single `__frsqrt_rn`); else the composite
    /// `sqrt().reciprocal()` fallback (two F32 round-trips).
    pub fn rsqrt(&self) -> Result<Tensor> {
        crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::transcendentals::rsqrt_bf16_iter,
            |x| x.sqrt()?.reciprocal(),
        )
    }

    /// Negate tensor. Phase 10: BF16 → TensorIterator (native sign-bit
    /// flip); else the `mul_scalar(-1.0)` composition (bit-exact to the
    /// iter path on finite values).
    pub fn neg(&self) -> Result<Tensor> {
        crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::unary::neg_bf16_iter,
            |x| x.mul_scalar(-1.0),
        )
    }

    /// Compute absolute value. Phase 10: BF16 → TensorIterator + explicit
    /// `Op::Abs` autograd record; non-BF16 falls back to the composite
    /// `square().sqrt()`, whose own autograd tape (Square then Sqrt) is
    /// the pre-Phase-10 behavior for non-BF16 inputs. Keeping the Abs
    /// tape record gated on BF16 preserves that split byte-for-byte.
    pub fn abs(&self) -> Result<Tensor> {
        if self.dtype() == DType::BF16 {
            let mut output = crate::tensor_iterator::ops::unary::abs_bf16_iter(self)?;
            if self.requires_grad {
                output.requires_grad = true;
                if crate::AutogradContext::is_recording() {
                    crate::AutogradContext::record_op(
                        output.id,
                        crate::autograd::Op::Abs { input: self.id },
                        vec![(self.id, self.clone())],
                    );
                }
            }
            return Ok(output);
        }
        self.square()?.sqrt()
    }

    /// Clamp values between min and max.
    ///
    /// The min/max constant tensors are built in the source tensor's dtype
    /// (not the workspace default dtype). This matters when the workspace
    /// default is BF16 and the caller is clamping an F32 tensor: without
    /// this, `full_like` would return BF16 constants and `maximum` / `minimum`
    /// would panic on the dtype mismatch.
    pub fn clamp(&self, min: f32, max: f32) -> Result<Tensor> {
        if min > max {
            return Err(Error::InvalidInput(
                format!("clamp: min ({}) greater than max ({})", min, max).into(),
            ));
        }

        let dtype = self.dtype();
        let lower = Tensor::from_vec(vec![min], Shape::from_dims(&[1]), self.device.clone())?
            .to_dtype(dtype)?;
        let upper = Tensor::from_vec(vec![max], Shape::from_dims(&[1]), self.device.clone())?
            .to_dtype(dtype)?;
        // `maximum` / `minimum` broadcast scalar-shaped tensors internally.
        let clipped = self.maximum(&lower)?;
        clipped.minimum(&upper)
    }

    /// Compute element-wise maximum with another tensor (GPU). Phase 10:
    /// BF16+BF16 → TensorIterator (handles broadcast via stride=0); other
    /// dtypes → broadcast-then-`GpuOps::max_elemwise` (F32 path).
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        let mut out = crate::tensor_iterator::dispatch_binary_bf16(
            self,
            other,
            crate::tensor_iterator::ops::binary::maximum_bf16_iter,
            |a, b| {
                let bshape = broadcast_shapes(a.shape().dims(), b.shape().dims())?;
                let a_bc = if a.shape().dims() != bshape {
                    a.broadcast_to(&Shape::from_dims(&bshape))?
                } else {
                    a.clone_result()?
                };
                let b_bc = if b.shape().dims() != bshape {
                    b.broadcast_to(&Shape::from_dims(&bshape))?
                } else {
                    b.clone_result()?
                };
                crate::cuda_ops::GpuOps::max_elemwise(&a_bc, &b_bc)
            },
        )?;
        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    out.id,
                    Op::Maximum {
                        a: self.id,
                        b: other.id,
                    },
                    vec![(self.id, self.alias()), (other.id, other.alias())],
                );
            }
        }
        Ok(out)
    }

    /// Compute element-wise minimum with another tensor. Phase 10:
    /// BF16+BF16 → TensorIterator (native min); other dtypes preserve the
    /// composite `-max(-a, -b)` fallback (unchanged from Phase 5b).
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        let mut out = crate::tensor_iterator::dispatch_binary_bf16(
            self,
            other,
            crate::tensor_iterator::ops::binary::minimum_bf16_iter,
            |a, b| {
                let bshape = broadcast_shapes(a.shape().dims(), b.shape().dims())?;
                let a_bc = if a.shape().dims() != bshape {
                    a.broadcast_to(&Shape::from_dims(&bshape))?
                } else {
                    a.clone_result()?
                };
                let b_bc = if b.shape().dims() != bshape {
                    b.broadcast_to(&Shape::from_dims(&bshape))?
                } else {
                    b.clone_result()?
                };
                let neg_a = a_bc.neg()?;
                let neg_b = b_bc.neg()?;
                let neg_max = neg_a.maximum(&neg_b)?;
                neg_max.neg()
            },
        )?;
        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    out.id,
                    Op::Minimum {
                        a: self.id,
                        b: other.id,
                    },
                    vec![(self.id, self.alias()), (other.id, other.alias())],
                );
            }
        }
        Ok(out)
    }

    /// Get maximum value along a dimension
    pub fn max_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        GpuOps::max_dim(self, dim, keepdim)
    }

    /// Sum along dimension keeping dimension (GPU-only)
    pub fn sum_dim_keepdim(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of bounds",
                dim
            )));
        }
        // Delegate to GPU axis-reduction that keeps dimension (FP32 compute)
        let mut output = GpuOps::sum_dim_keepdim(self, dim)?;

        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::SumDimKeepdim {
                        input: self.id,
                        dim,
                    },
                    vec![(self.id, self.alias())],
                );
            }
        }

        Ok(output)
    }

    /// Divide by another tensor. Phase 10: BF16+BF16 → TensorIterator
    /// (handles broadcast internally); other dtypes → explicit broadcast
    /// + `GpuOps::div` F32 path. Autograd records with ORIGINAL tensor
    /// IDs so gradients flow to the real inputs, not broadcast views.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let mut output = crate::tensor_iterator::dispatch_binary_bf16(
            self,
            other,
            crate::tensor_iterator::ops::binary::div_bf16_iter,
            |a, b| {
                let (lhs, rhs) = if a.shape == b.shape {
                    (a.clone_result()?, b.clone_result()?)
                } else {
                    let bshape = broadcast_shapes(a.shape().dims(), b.shape().dims())?;
                    let lhs = if a.shape().dims() == bshape {
                        a.clone_result()?
                    } else {
                        a.broadcast_to(&Shape::from_dims(&bshape))?
                    };
                    let rhs = if b.shape().dims() == bshape {
                        b.clone_result()?
                    } else {
                        b.broadcast_to(&Shape::from_dims(&bshape))?
                    };
                    (lhs, rhs)
                };
                GpuOps::div(&lhs, &rhs)
            },
        )?;
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Div {
                        lhs: self.id,
                        rhs: other.id,
                        lhs_shape: self.shape.clone(),
                        rhs_shape: other.shape.clone(),
                    },
                    vec![(self.id, self.clone()), (other.id, other.clone())],
                );
            }
        }
        Ok(output)
    }

    /// Element-wise equality comparison. Phase 10: BF16+BF16 → iter,
    /// else → `GpuOps::cmp_eq`. IEEE: eq(NaN, NaN) = false. The explicit
    /// shape-equality check stays — `eq` is the only comparison that
    /// requires matching shapes (pre-broadcast) on the GpuOps path.
    pub fn eq(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(Error::InvalidOperation(
                "Equality comparison requires tensors with identical shapes".into(),
            ));
        }
        crate::tensor_iterator::dispatch_comparison_bf16(
            self,
            other,
            crate::tensor_iterator::ops::comparison::eq_bf16_iter,
            GpuOps::cmp_eq,
        )
    }

    /// Create tensor filled with single value
    pub fn full(shape: Shape, value: f32, device: Arc<CudaDevice>) -> Result<Tensor> {
        let size = shape.elem_count();
        let data = vec![value; size];
        let t = Tensor::from_slice(&data, shape, device)?;
        let dd = default_dtype();
        if dd != DType::F32 {
            t.to_dtype(dd)
        } else {
            Ok(t)
        }
    }

    /// Create tensor filled with ones (static method)
    pub fn ones_like_shape(shape: Shape, device: Arc<CudaDevice>) -> Result<Tensor> {
        Self::full(shape, 1.0, device)
    }

    /// Create identity matrix (F32).
    ///
    /// For arbitrary dtype (BF16/F16/F32), use [`Tensor::eye_dtype`].
    pub fn eye(n: usize, device: Arc<CudaDevice>) -> Result<Tensor> {
        Self::eye_dtype(n, DType::F32, device)
    }

    /// Create identity matrix with explicit dtype.
    ///
    /// Used by OFT-Neumann (LyCORIS): `R = I + 2Q + 2Q^2 + ...` where the
    /// identity must match the parameter dtype (BF16 trainers).
    pub fn eye_dtype(n: usize, dtype: DType, device: Arc<CudaDevice>) -> Result<Tensor> {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_slice_dtype(&data, Shape::from_dims(&[n, n]), device, dtype)
    }

    /// Create range tensor
    pub fn arange(start: f32, end: f32, step: f32, device: Arc<CudaDevice>) -> Result<Tensor> {
        let n = ((end - start) / step).ceil() as usize;
        let data: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
        Tensor::from_slice(&data, Shape::from_dims(&[n]), device)
    }

    /// Compute sign of elements (-1, 0, or 1)
    pub fn sign(&self) -> Result<Tensor> {
        let zero = self.full_like(0.0)?;
        let dtype = self.dtype();

        let positive = self.gt(&zero)?.to_dtype(dtype)?;
        let negative = self.lt(&zero)?.to_dtype(dtype)?;

        let neg_scaled = negative.mul_scalar(-1.0)?;
        positive.add(&neg_scaled)
    }

    /// Element-wise floor operation
    pub fn floor(&self) -> Result<Tensor> {
        GpuOps::floor(self)
    }

    /// Element-wise ceil operation
    pub fn ceil(&self) -> Result<Tensor> {
        GpuOps::ceil(self)
    }

    /// Element-wise round operation
    pub fn round(&self) -> Result<Tensor> {
        GpuOps::round(self)
    }

    /// Element-wise less than or equal comparison. Phase 10: BF16+BF16
    /// → iter, else → `GpuOps::cmp_le`.
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        crate::tensor_iterator::dispatch_comparison_bf16(
            self,
            other,
            crate::tensor_iterator::ops::comparison::le_bf16_iter,
            GpuOps::cmp_le,
        )
    }

    /// Subtract a scalar from all elements
    pub fn sub_scalar(&self, scalar: f32) -> Result<Tensor> {
        let neg_scalar = -scalar;
        self.add_scalar(neg_scalar)
    }

    /// Find the maximum value in the tensor
    pub fn max_all(&self) -> Result<f32> {
        if self.shape.elem_count() == 0 {
            return Err(Error::InvalidOperation(
                "Cannot find max of empty tensor".into(),
            ));
        }
        let max_tensor = GpuOps::reduce_max(self)?;
        max_tensor.to_scalar::<f32>()
    }

    /// Find the minimum value in the tensor
    pub fn min_all(&self) -> Result<f32> {
        if self.shape.elem_count() == 0 {
            return Err(Error::InvalidOperation(
                "Cannot find min of empty tensor".into(),
            ));
        }
        let min_tensor = GpuOps::reduce_min(self)?;
        min_tensor.to_scalar::<f32>()
    }

    /// Find the sum of all elements in the tensor
    pub fn sum_all(&self) -> Result<Tensor> {
        let kernels = crate::cuda_kernels_gpu::CudaKernels::new(self.device.clone())?;
        let sum = kernels.sum_all(self)?;
        if self.dtype() == DType::F32 {
            Ok(sum)
        } else {
            sum.to_dtype(self.dtype())
        }
    }

    /// Flip tensor along specified dimensions
    pub fn flip(&self, dims: &[usize]) -> Result<Tensor> {
        if dims.is_empty() {
            return self.clone_result();
        }

        let rank = self.shape.dims().len();
        if dims.len() != 1 || dims[0] != rank.saturating_sub(1) {
            return Err(Error::InvalidOperation(
                "Flip only supports flipping along the last dimension currently".into(),
            ));
        }

        GpuOps::flip_last_dim(self)
    }

    /// Create upper triangular matrix (ones above diagonal, zeros below)
    pub fn triu(&self, diagonal: i32) -> Result<Tensor> {
        let shape_dims = self.shape.dims();
        let rank = shape_dims.len();
        if rank < 2 {
            return Err(Error::InvalidOperation(
                "triu requires at least 2D tensor".into(),
            ));
        }

        let rows = shape_dims[rank - 2];
        let cols = shape_dims[rank - 1];
        let device = self.device.clone();

        let row_idx = Tensor::arange(0.0, rows as f32, 1.0, device.clone())?.reshape(&[rows, 1])?;
        let col_idx = Tensor::arange(0.0, cols as f32, 1.0, device.clone())?.reshape(&[1, cols])?;

        let threshold = row_idx.add_scalar(diagonal as f32)?;
        let mut mask = col_idx.ge(&threshold)?;

        if rank > 2 {
            let mut reshape_dims = vec![1; rank];
            reshape_dims[rank - 2] = rows;
            reshape_dims[rank - 1] = cols;
            mask = mask.reshape(&reshape_dims)?;
            mask = mask.broadcast_to(self.shape())?;
        }

        let mask = if mask.dtype() == self.dtype() {
            mask
        } else {
            mask.to_dtype(self.dtype())?
        };

        self.mul(&mask)
    }

    /// Conditional selection based on mask
    pub fn where_tensor(&self, true_tensor: &Tensor, false_tensor: &Tensor) -> Result<Tensor> {
        if self.shape != true_tensor.shape || self.shape != false_tensor.shape {
            return Err(Error::InvalidOperation(
                "All tensors must have the same shape for where operation".into(),
            ));
        }

        let zero = self.full_like(0.0)?;
        let mask = self.ne(&zero)?;
        let mask_cast = if mask.dtype() == true_tensor.dtype() {
            mask.clone_result()?
        } else {
            mask.to_dtype(true_tensor.dtype())?
        };

        let mask_for_true = mask_cast.clone_result()?;
        let inv_mask = mask_cast.neg()?.add_scalar(1.0)?;

        let true_part = mask_for_true.mul(true_tensor)?;
        let false_part = inv_mask.mul(false_tensor)?;
        true_part.add(&false_part)
    }

    /// Log softmax along a dimension
    pub fn log_softmax(&self, dim: isize) -> Result<Tensor> {
        // More numerically stable implementation
        let shape = self.shape().dims();
        let ndim = shape.len() as isize;

        // Handle negative dimension
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        // Compute max for numerical stability
        let max_vals = self.max_dim(dim, true)?;
        let shifted = self.sub(&max_vals)?;

        // Compute log(sum(exp(x - max)))
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum_dim_keepdim(dim)?;
        let log_sum_exp = sum_exp.log()?;

        // log_softmax = x - max - log(sum(exp(x - max)))
        let mut output = shifted.sub(&log_sum_exp)?;

        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::LogSoftmax {
                        input: self.id,
                        dim: dim as isize,
                    },
                    vec![(self.id, self.alias())],
                );
            }
        }

        match crate::config::default_dtype() {
            DType::F32 => Ok(output),
            dt => output.to_dtype(dt),
        }
    }

    /// Get data as 1D vector of i64
    pub fn to_vec1<T: From<f32>>(&self) -> Result<Vec<T>> {
        let data = self.to_vec()?;
        Ok(data.into_iter().map(|x| T::from(x)).collect())
    }

    /// Get data as 2D vector
    pub fn to_vec2<T: From<f32>>(&self) -> Result<Vec<Vec<T>>> {
        if self.shape.dims().len() != 2 {
            return Err(Error::InvalidOperation(format!(
                "to_vec2 requires 2D tensor, got {:?}",
                self.shape.dims()
            )));
        }

        let data = self.to_vec()?;
        let rows = self.shape.dims()[0];
        let cols = self.shape.dims()[1];

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let row: Vec<T> = data[i * cols..(i + 1) * cols]
                .iter()
                .map(|&x| T::from(x))
                .collect();
            result.push(row);
        }

        Ok(result)
    }

    /// Create tensor from 2D vector
    pub fn from_vec2<T: Into<f32> + Copy>(
        data: Vec<Vec<T>>,
        device: Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let rows = data.len();
        if rows == 0 {
            return Err(Error::InvalidOperation("Empty 2D vector".into()));
        }

        let cols = data[0].len();
        let mut flat_data = Vec::with_capacity(rows * cols);

        for row in data {
            if row.len() != cols {
                return Err(Error::InvalidOperation("Inconsistent row lengths".into()));
            }
            for val in row {
                flat_data.push(val.into());
            }
        }

        Tensor::from_vec(flat_data, Shape::from_dims(&[rows, cols]), device)
    }

    /// Create scalar tensor
    pub fn from_scalar(value: f32, device: Arc<CudaDevice>) -> Result<Tensor> {
        Tensor::from_vec(vec![value], Shape::from_dims(&[1]), device)
    }

    /// Get scalar value from tensor
    pub fn to_scalar<T: From<f32>>(&self) -> Result<T> {
        if self.shape.elem_count() != 1 {
            return Err(Error::InvalidOperation(format!(
                "to_scalar requires scalar tensor, got shape {:?}",
                self.shape.dims()
            )));
        }

        let data = self.to_vec()?;
        Ok(T::from(data[0]))
    }

    /// Split tensor into multiple tensors along a dimension
    pub fn split(&self, sizes: &[usize], dim: usize) -> Result<Vec<Tensor>> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            )));
        }

        // Check sizes sum to dimension size
        let total_size: usize = sizes.iter().sum();
        if total_size != shape[dim] {
            return Err(Error::InvalidOperation(format!(
                "Split sizes {:?} don't sum to dimension size {}",
                sizes, shape[dim]
            )));
        }

        let mut result = Vec::new();
        let mut offset = 0;

        for &size in sizes {
            // Create slice ranges
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            for (d, &dim_size) in shape.iter().enumerate() {
                if d == dim {
                    ranges.push((offset, offset + size));
                } else {
                    ranges.push((0, dim_size));
                }
            }

            let chunk = self.slice(&ranges)?;
            result.push(chunk);
            offset += size;
        }

        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            for tensor in result.iter_mut() {
                tensor.requires_grad = true;
            }
            if AutogradContext::is_recording() {
                for tensor in result.iter_mut() {
                    // Record split operation for each output
                    AutogradContext::record_op(
                        tensor.id,
                        Op::Split {
                            input: self.id,
                            sizes: sizes.to_vec(),
                            dim,
                        },
                        vec![(self.id, self.alias())],
                    );
                }
            }
        }

        Ok(result)
    }
}

/// Helper function to broadcast shapes
fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
    let max_len = shape1.len().max(shape2.len());
    let mut result = vec![1; max_len];

    // Right-align shapes
    let offset1 = max_len - shape1.len();
    let offset2 = max_len - shape2.len();

    for i in 0..max_len {
        let dim1 = if i >= offset1 { shape1[i - offset1] } else { 1 };
        let dim2 = if i >= offset2 { shape2[i - offset2] } else { 1 };

        if dim1 == dim2 {
            result[i] = dim1;
        } else if dim1 == 1 {
            result[i] = dim2;
        } else if dim2 == 1 {
            result[i] = dim1;
        } else {
            return Err(Error::InvalidOperation(format!(
                "Cannot broadcast dimensions {} and {}",
                dim1, dim2
            )));
        }
    }

    Ok(result)
}

// Note: softmax, unsqueeze, full_like, and gelu are already implemented in tensor.rs

// ─── index_assign: NVRTC kernel + dispatcher ───────────────────────────────
//
// Replaces slices of `self` along `dim` at the positions given by `indices`
// with the corresponding slices from `values`. Backs `Tensor::index_assign`
// (forward + autograd-internal `*_no_grad`). Used by TREAD's scatter step.
//
// Strategy: precompute on the host an `inverse_mask[k] = pos in indices`
// (or -1) of length `dim_size`; the kernel then does a single O(1) lookup
// per output element to decide whether to copy from `self` or `values`.
//
// Two NVRTC modules: `index_assign_f32_kernel` (training path) and
// `index_assign_bf16_kernel` (inference / mixed-precision path).
const INDEX_ASSIGN_F32_KERNEL_NAME: &str = "index_assign_f32_kernel";
const INDEX_ASSIGN_BF16_KERNEL_NAME: &str = "index_assign_bf16_kernel";

const INDEX_ASSIGN_F32_KERNEL_SRC: &str = r#"
extern "C" __global__ void index_assign_f32_kernel(
    const float* __restrict__ self_in,
    const float* __restrict__ values_in,
    const int* __restrict__ inverse_mask,
    const int* __restrict__ self_dims,
    const int* __restrict__ self_strides,
    const int* __restrict__ val_strides,
    float* __restrict__ out,
    int ndim,
    int dim,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose `idx` into per-axis coords (output is dense / contiguous,
    // matching self_strides since output shape == self shape).
    int rem = idx;
    int coord_dim = 0;
    int val_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        int s = self_strides[d];
        int c;
        if (s > 0) {
            c = rem / s;
            rem = rem - c * s;
        } else {
            c = 0;
        }
        if (d == dim) {
            coord_dim = c;
        } else {
            val_offset += c * val_strides[d];
        }
    }

    int pos = inverse_mask[coord_dim];
    if (pos >= 0) {
        // Indexed → copy from values, with values' axis-`dim` coord = pos.
        val_offset += pos * val_strides[dim];
        out[idx] = values_in[val_offset];
    } else {
        // Non-indexed → copy from self.
        out[idx] = self_in[idx];
    }
}
"#;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const INDEX_ASSIGN_BF16_KERNEL_SRC: &str = r#"
extern "C" __global__ void index_assign_bf16_kernel(
    const unsigned short* __restrict__ self_in,
    const unsigned short* __restrict__ values_in,
    const int* __restrict__ inverse_mask,
    const int* __restrict__ self_dims,
    const int* __restrict__ self_strides,
    const int* __restrict__ val_strides,
    unsigned short* __restrict__ out,
    int ndim,
    int dim,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int rem = idx;
    int coord_dim = 0;
    int val_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        int s = self_strides[d];
        int c;
        if (s > 0) {
            c = rem / s;
            rem = rem - c * s;
        } else {
            c = 0;
        }
        if (d == dim) {
            coord_dim = c;
        } else {
            val_offset += c * val_strides[d];
        }
    }

    int pos = inverse_mask[coord_dim];
    if (pos >= 0) {
        val_offset += pos * val_strides[dim];
        out[idx] = values_in[val_offset];
    } else {
        out[idx] = self_in[idx];
    }
}
"#;

#[allow(clippy::too_many_arguments)]
fn index_assign_f32(
    self_t: &Tensor,
    values: &Tensor,
    inverse_mask: &[i32],
    self_dims_i32: &[i32],
    self_strides: &[i32],
    val_strides: &[i32],
    dim: i32,
    total: i32,
    ndim: i32,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    use cudarc::driver::{LaunchAsync, LaunchConfig};

    // Materialize non-contiguous inputs.
    let self_owned;
    let self_t = if self_t.is_contiguous() {
        self_t
    } else {
        self_owned = self_t.contiguous()?;
        &self_owned
    };
    let val_owned;
    let values = if values.is_contiguous() {
        values
    } else {
        val_owned = values.contiguous()?;
        &val_owned
    };

    crate::cuda_kernels::CudaKernels::ensure_kernel(
        device,
        INDEX_ASSIGN_F32_KERNEL_NAME,
        INDEX_ASSIGN_F32_KERNEL_SRC,
    )?;
    let func = device
        .get_func(INDEX_ASSIGN_F32_KERNEL_NAME, INDEX_ASSIGN_F32_KERNEL_NAME)
        .ok_or_else(|| Error::Cuda(format!("Failed to load {}", INDEX_ASSIGN_F32_KERNEL_NAME)))?;

    let d_inv = device
        .htod_copy(inverse_mask.to_vec())
        .map_err(|e| Error::Cuda(format!("htod inverse_mask: {:?}", e)))?;
    let d_dims = device
        .htod_copy(self_dims_i32.to_vec())
        .map_err(|e| Error::Cuda(format!("htod self_dims: {:?}", e)))?;
    let d_self_strides = device
        .htod_copy(self_strides.to_vec())
        .map_err(|e| Error::Cuda(format!("htod self_strides: {:?}", e)))?;
    let d_val_strides = device
        .htod_copy(val_strides.to_vec())
        .map_err(|e| Error::Cuda(format!("htod val_strides: {:?}", e)))?;

    let mut output = Tensor::empty_dtype(self_t.shape().clone(), DType::F32, device.clone())?;

    let threads: u32 = 256;
    let grid: u32 = ((total as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                self_t.storage.try_as_slice_f32()?,
                values.storage.try_as_slice_f32()?,
                &d_inv,
                &d_dims,
                &d_self_strides,
                &d_val_strides,
                output.storage_mut().try_as_mut_slice_f32()?,
                ndim,
                dim,
                total,
            ),
        )
        .map_err(|e| Error::Cuda(format!("launch index_assign_f32: {:?}", e)))?;
    }

    Ok(output)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
#[allow(clippy::too_many_arguments)]
fn index_assign_bf16(
    self_t: &Tensor,
    values: &Tensor,
    inverse_mask: &[i32],
    self_dims_i32: &[i32],
    self_strides: &[i32],
    val_strides: &[i32],
    dim: i32,
    total: i32,
    ndim: i32,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    use cudarc::driver::{LaunchAsync, LaunchConfig};

    // BF16 path needs OWNING (not arena/view) storage to bind a u16 slice
    // for the kernel. clone_result materializes arena/view → owning AND
    // resolves non-contiguous strides via contiguous() in one shot.
    let self_owned = self_t.clone_result()?;
    let self_t = &self_owned;
    let val_owned = values.clone_result()?;
    let values = &val_owned;

    crate::cuda_kernels::CudaKernels::ensure_kernel(
        device,
        INDEX_ASSIGN_BF16_KERNEL_NAME,
        INDEX_ASSIGN_BF16_KERNEL_SRC,
    )?;
    let func = device
        .get_func(INDEX_ASSIGN_BF16_KERNEL_NAME, INDEX_ASSIGN_BF16_KERNEL_NAME)
        .ok_or_else(|| Error::Cuda(format!("Failed to load {}", INDEX_ASSIGN_BF16_KERNEL_NAME)))?;

    let d_inv = device
        .htod_copy(inverse_mask.to_vec())
        .map_err(|e| Error::Cuda(format!("htod inverse_mask: {:?}", e)))?;
    let d_dims = device
        .htod_copy(self_dims_i32.to_vec())
        .map_err(|e| Error::Cuda(format!("htod self_dims: {:?}", e)))?;
    let d_self_strides = device
        .htod_copy(self_strides.to_vec())
        .map_err(|e| Error::Cuda(format!("htod self_strides: {:?}", e)))?;
    let d_val_strides = device
        .htod_copy(val_strides.to_vec())
        .map_err(|e| Error::Cuda(format!("htod val_strides: {:?}", e)))?;

    let mut output = Tensor::empty_dtype(self_t.shape().clone(), DType::BF16, device.clone())?;

    let threads: u32 = 256;
    let grid: u32 = ((total as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                self_t.storage.try_as_slice_u16()?,
                values.storage.try_as_slice_u16()?,
                &d_inv,
                &d_dims,
                &d_self_strides,
                &d_val_strides,
                output.storage_mut().try_as_mut_slice_u16()?,
                ndim,
                dim,
                total,
            ),
        )
        .map_err(|e| Error::Cuda(format!("launch index_assign_bf16: {:?}", e)))?;
    }

    Ok(output)
}

#[cfg(test)]
mod index_assign_tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    fn dev() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("device 0")
    }

    #[test]
    fn index_assign_identity_recovers_self() {
        let device = dev();
        // [B=2, T=4, D=3]
        let total = 2 * 4 * 3;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let self_t =
            Tensor::from_vec(data.clone(), Shape::from_dims(&[2, 4, 3]), device.clone()).unwrap();
        // Indices = all positions; values = self → output should equal self.
        let idx_data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
        let idx_f32 = Tensor::from_vec(idx_data, Shape::from_dims(&[4]), device.clone()).unwrap();
        let idx = idx_f32.to_dtype(DType::I32).unwrap();
        let out = self_t.index_assign(1, &idx, &self_t).unwrap();
        let out_v = out.to_vec().unwrap();
        assert_eq!(out_v, data);
    }

    #[test]
    fn index_assign_round_trip_via_index_select() {
        let device = dev();
        let total = 2 * 4 * 3;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let self_t =
            Tensor::from_vec(data.clone(), Shape::from_dims(&[2, 4, 3]), device.clone()).unwrap();
        // Pick indices = [0, 2]
        let idx_data: Vec<f32> = vec![0.0, 2.0];
        let idx_f32 = Tensor::from_vec(idx_data, Shape::from_dims(&[2]), device.clone()).unwrap();
        let idx = idx_f32.to_dtype(DType::I32).unwrap();

        let gathered = self_t.index_select(1, &idx).unwrap();
        let scattered = self_t.index_assign(1, &idx, &gathered).unwrap();
        // Should equal self_t exactly: we put back what we took.
        let s_v = scattered.to_vec().unwrap();
        assert_eq!(s_v, data);
    }

    #[test]
    fn index_assign_writes_only_indexed_rows() {
        let device = dev();
        // [T=4, D=2]
        let self_data: Vec<f32> = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0];
        let self_t =
            Tensor::from_vec(self_data, Shape::from_dims(&[4, 2]), device.clone()).unwrap();
        // values shape [2, 2] → put rows at indices 1, 3
        let val_data: Vec<f32> = vec![99.0, 98.0, 97.0, 96.0];
        let val_t = Tensor::from_vec(val_data, Shape::from_dims(&[2, 2]), device.clone()).unwrap();
        let idx_f32 =
            Tensor::from_vec(vec![1.0, 3.0], Shape::from_dims(&[2]), device.clone()).unwrap();
        let idx = idx_f32.to_dtype(DType::I32).unwrap();
        let out = self_t.index_assign(0, &idx, &val_t).unwrap();
        let v = out.to_vec().unwrap();
        // Row 0: 10, 11 (untouched). Row 1: 99, 98. Row 2: 30, 31 (untouched). Row 3: 97, 96.
        assert_eq!(v, vec![10.0, 11.0, 99.0, 98.0, 30.0, 31.0, 97.0, 96.0]);
        // Shape preserved.
        assert_eq!(out.shape().dims(), &[4, 2]);
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    #[test]
    fn index_assign_bf16_round_trip() {
        let device = dev();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let self_t = Tensor::from_vec(data.clone(), Shape::from_dims(&[2, 4, 3]), device.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let idx_f32 =
            Tensor::from_vec(vec![1.0, 3.0], Shape::from_dims(&[2]), device.clone()).unwrap();
        let idx = idx_f32.to_dtype(DType::I32).unwrap();
        let gathered = self_t.index_select(1, &idx).unwrap();
        let scattered = self_t.index_assign(1, &idx, &gathered).unwrap();
        // Compare in F32 (BF16 → F32 → vec).
        let s_v = scattered.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        // BF16 round-trip is lossy on arbitrary floats, but our values are
        // small ints (0..24) which are exactly representable in BF16.
        assert_eq!(s_v, data);
    }
}
