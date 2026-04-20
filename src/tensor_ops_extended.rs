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

    /// Greater than comparison (broadcast-safe)
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        GpuOps::cmp_gt(self, other)
    }

    /// Greater than or equal comparison (broadcast-safe)
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        GpuOps::cmp_ge(self, other)
    }

    /// Less than comparison (broadcast-safe)
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        GpuOps::cmp_lt(self, other)
    }

    /// Not equal comparison (broadcast-safe)
    pub fn ne(&self, other: &Tensor) -> Result<Tensor> {
        GpuOps::cmp_ne(self, other)
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
                            let rc = unsafe {
                                cudarc::driver::sys::lib()
                                    .cuMemcpy2DAsync_v2(&params, stream_ptr)
                            };
                            if rc != CUresult::CUDA_SUCCESS {
                                return Err(Error::Cuda(format!(
                                    "cuMemcpy2DAsync_v2 (cat) failed: {rc:?} \
                                     src={:#x} dst={:#x} srcPitch={} dstPitch={} \
                                     width={} height={}",
                                    params.srcDevice, params.dstDevice,
                                    params.srcPitch, params.dstPitch,
                                    params.WidthInBytes, params.Height,
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
                        vec![
                            (self.id, self.alias()),
                            (indices.id(), indices.alias()),
                        ],
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
                    vec![
                        (self.id, self.alias()),
                        (indices.id(), indices.alias()),
                    ],
                );
            }
        }
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

    /// Compute natural logarithm (GPU)
    pub fn log(&self) -> Result<Tensor> {
        GpuOps::log(self)
    }

    /// Compute reciprocal square root
    pub fn rsqrt(&self) -> Result<Tensor> {
        self.sqrt()?.reciprocal()
    }

    /// Negate tensor
    pub fn neg(&self) -> Result<Tensor> {
        self.mul_scalar(-1.0)
    }

    /// Compute absolute value
    pub fn abs(&self) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if self.dtype() == DType::BF16 {
            let mut output = crate::bf16_elementwise::abs_bf16(self)?;
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
        // Fallback for non-BF16
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
        let lower = Tensor::from_vec(
            vec![min],
            Shape::from_dims(&[1]),
            self.device.clone(),
        )?
        .to_dtype(dtype)?;
        let upper = Tensor::from_vec(
            vec![max],
            Shape::from_dims(&[1]),
            self.device.clone(),
        )?
        .to_dtype(dtype)?;
        // `maximum` / `minimum` broadcast scalar-shaped tensors internally.
        let clipped = self.maximum(&lower)?;
        clipped.minimum(&upper)
    }

    /// Compute element-wise maximum with another tensor (GPU)
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        // Check shapes are compatible for broadcasting
        let broadcast_shape = broadcast_shapes(self.shape().dims(), other.shape().dims())?;

        // Broadcast to common shape
        let a = if self.shape().dims() != broadcast_shape {
            self.broadcast_to(&Shape::from_dims(&broadcast_shape))?
        } else {
            self.clone_result()?
        };
        let b = if other.shape().dims() != broadcast_shape {
            other.broadcast_to(&Shape::from_dims(&broadcast_shape))?
        } else {
            other.clone_result()?
        };

        // GPU elementwise max
        let mut out = crate::cuda_ops::GpuOps::max_elemwise(&a, &b)?;

        // Autograd record
        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    out.id,
                    Op::Maximum {
                        a: self.id,
                        b: other.id,
                    },
                    vec![
                        (self.id, self.alias()),
                        (other.id, other.alias()),
                    ],
                );
            }
        }
        Ok(out)
    }

    /// Compute element-wise minimum with another tensor
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        // Check shapes are compatible for broadcasting
        let broadcast_shape = broadcast_shapes(self.shape().dims(), other.shape().dims())?;

        // Broadcast both tensors if needed
        let a = if self.shape().dims() != broadcast_shape {
            self.broadcast_to(&Shape::from_dims(&broadcast_shape))?
        } else {
            self.clone_result()?
        };

        let b = if other.shape().dims() != broadcast_shape {
            other.broadcast_to(&Shape::from_dims(&broadcast_shape))?
        } else {
            other.clone_result()?
        };

        // Compute minimum via -max(-a, -b)
        let neg_a = a.neg()?;
        let neg_b = b.neg()?;
        let neg_max = neg_a.maximum(&neg_b)?;
        let mut out = neg_max.neg()?;

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    out.id,
                    Op::Minimum {
                        a: self.id,
                        b: other.id,
                    },
                    vec![
                        (self.id, self.alias()),
                        (other.id, other.alias()),
                    ],
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
                    Op::SumDimKeepdim { input: self.id, dim },
                    vec![(self.id, self.alias())],
                );
            }
        }

        Ok(output)
    }

    /// Divide by another tensor
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        // Broadcast both tensors to common shape if needed
        let mut lhs = if self.shape == other.shape {
            self.clone_result()?
        } else {
            let broadcast_shape = broadcast_shapes(self.shape().dims(), other.shape().dims())?;
            if self.shape().dims() == broadcast_shape {
                self.clone_result()?
            } else {
                self.broadcast_to(&Shape::from_dims(&broadcast_shape))?
            }
        };
        let rhs = if self.shape == other.shape {
            other.clone_result()?
        } else {
            let broadcast_shape = broadcast_shapes(self.shape().dims(), other.shape().dims())?;
            if other.shape().dims() == broadcast_shape {
                other.clone_result()?
            } else {
                other.broadcast_to(&Shape::from_dims(&broadcast_shape))?
            }
        };

        let mut output = GpuOps::div(&lhs, &rhs)?;

        // AUTOGRAD: Record with ORIGINAL tensor IDs so gradients flow
        // back to the actual inputs, not broadcast intermediates.
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
                    vec![
                        (self.id, self.clone()),
                        (other.id, other.clone()),
                    ],
                );
            }
        }

        Ok(output)
    }

    /// Element-wise equality comparison
    pub fn eq(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(Error::InvalidOperation(
                "Equality comparison requires tensors with identical shapes".into(),
            ));
        }
        GpuOps::cmp_eq(self, other)
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

    /// Create identity matrix
    pub fn eye(n: usize, device: Arc<CudaDevice>) -> Result<Tensor> {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_slice(&data, Shape::from_dims(&[n, n]), device)
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

    /// Element-wise less than or equal comparison
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        GpuOps::cmp_le(self, other)
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
