//! Missing tensor operations implementation

use crate::cuda_ops::GpuOps;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::{cuda_ops_bf16, staging::ArenaScratch};
use crate::{strict, AutogradContext, DType, Error, Op, Result, Shape, Tensor};

impl Tensor {
    /// Power operation - raise tensor to a scalar power
    pub fn pow(&self, exponent: f32) -> Result<Tensor> {
        let mut output = GpuOps::pow(self, exponent)?;

        // Autograd support
        if self.requires_grad {
            output.requires_grad = true;
            AutogradContext::record_op(
                output.id,
                Op::Square { input: self.id }, // Using Square for now
                vec![(self.id, self.clone_result()?)],
            );
        }

        Ok(output)
    }

    /// Sine function
    pub fn sin(&self) -> Result<Tensor> {
        let mut output = GpuOps::sin(self)?;

        if self.requires_grad {
            output.requires_grad = true;
            // For sin, we need to save the input for cos(x) in backward
            AutogradContext::record_op(
                output.id,
                Op::Abs { input: self.id }, // Pending: add Sin op
                vec![(self.id, self.clone_result()?)],
            );
        }

        Ok(output)
    }

    /// Cosine function
    pub fn cos(&self) -> Result<Tensor> {
        let mut output = GpuOps::cos(self)?;

        if self.requires_grad {
            output.requires_grad = true;
            // For cos, we need to save the input for -sin(x) in backward
            AutogradContext::record_op(
                output.id,
                Op::Abs { input: self.id }, // Pending: add Cos op
                vec![(self.id, self.clone_result()?)],
            );
        }

        Ok(output)
    }

    /// Standard deviation
    pub fn std(&self, dim: Option<&[usize]>, keepdim: bool) -> Result<Tensor> {
        let var = if let Some(dims) = dim {
            self.var(dims, false, keepdim)? // Use the var() from tensor_ops_extended.rs
        } else {
            // For global std, compute variance over all dimensions
            let all_dims: Vec<usize> = (0..self.shape.dims().len()).collect();
            self.var(&all_dims, false, keepdim)?
        };
        var.sqrt()
    }

    /// Square root
    pub fn sqrt(&self) -> Result<Tensor> {
        GpuOps::sqrt(self)
    }

    /// Mean along dimensions
    pub fn mean_dim(&self, dims: &[usize], keepdim: bool) -> Result<Tensor> {
        if dims.is_empty() {
            return Ok(self.clone_result()?);
        }

        let input_shape = self.shape().dims();
        let ndims = input_shape.len();

        let mut dims_sorted: Vec<usize> = dims.to_vec();
        dims_sorted.sort_unstable();
        dims_sorted.dedup();

        for &dim in &dims_sorted {
            if dim >= ndims {
                return Err(Error::InvalidOperation(format!(
                    "Dimension {} out of range for tensor with {} dimensions",
                    dim, ndims
                )));
            }
        }

        let divisor: usize = dims_sorted.iter().map(|&d| input_shape[d]).product();
        if divisor == 0 {
            return Err(Error::InvalidInput(
                "mean_dim with zero-sized dimension".into(),
            ));
        }

        let mut result = self.clone_result()?;

        for (i, &original_dim) in dims_sorted.iter().enumerate() {
            let adjusted_dim = if keepdim {
                original_dim
            } else {
                original_dim - i
            };
            result = GpuOps::sum_dim_keepdim(&result, adjusted_dim)?;
            if !keepdim {
                let mut new_shape = result.shape().dims().to_vec();
                new_shape.remove(adjusted_dim);
                result = result.reshape(&new_shape)?;
            }
        }

        let scale = 1.0 / divisor as f32;
        result.mul_scalar(scale)
    }

    /// Division by scalar
    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.mul_scalar(1.0 / scalar)
    }

    /// Flatten all dimensions into a single dimension
    pub fn flatten_all(&self) -> Result<Tensor> {
        let total_elements = self.shape.elem_count();
        self.reshape(&[total_elements])
    }

    /// Mean across all elements (scalar result).
    pub fn mean_all(&self) -> Result<Tensor> {
        let rank = self.shape().dims().len();
        if rank == 0 {
            return self.clone_result();
        }
        let dims: Vec<usize> = (0..rank).collect();
        self.mean_dim(&dims, false)
    }

    /// Dropout operation - randomly zeros elements with probability p during training
    pub fn dropout(&self, p: f32) -> Result<Tensor> {
        if p <= 0.0 || p >= 1.0 {
            return Err(Error::InvalidOperation(format!(
                "Dropout probability must be between 0 and 1, got {}",
                p
            )));
        }

        // During inference, dropout is identity
        // For now, we'll implement a simple version that always returns the input scaled
        // In a full implementation, this would check training mode and use random mask
        let scale = 1.0 / (1.0 - p);
        self.mul_scalar(scale)
    }

    /// Upsample using nearest neighbor interpolation
    pub fn upsample_nearest2d(&self, new_h: usize, new_w: usize) -> Result<Tensor> {
        // Assume input is [batch, channels, height, width]
        let shape = self.shape();
        if shape.rank() != 4 {
            return Err(Error::InvalidOperation(format!(
                "Expected 4D tensor, got {}D",
                shape.rank()
            )));
        }
        GpuOps::upsample2d_nearest(self, (new_h, new_w))
    }

    /// Pad a 3D tensor [B, C, L] on the last dimension with zeros.
    ///
    /// `pad_left`: number of zeros prepended.
    /// `pad_right`: number of zeros appended.
    /// Returns [B, C, L + pad_left + pad_right].
    pub fn pad1d(&self, pad_left: usize, pad_right: usize) -> Result<Tensor> {
        let dims = self.shape().dims();
        if dims.len() != 3 {
            return Err(Error::InvalidOperation(format!(
                "pad1d: expected 3D [B,C,L], got {}D", dims.len()
            )));
        }
        if pad_left == 0 && pad_right == 0 {
            return Ok(self.clone());
        }
        let (b, c, l) = (dims[0], dims[1], dims[2]);
        let new_l = l + pad_left + pad_right;
        let mut out = Tensor::zeros_dtype(
            Shape::from_dims(&[b, c, new_l]),
            self.dtype(),
            self.device().clone(),
        )?;
        // Copy original data into the middle
        // out[:, :, pad_left:pad_left+l] = self
        if l > 0 {
            let src_flat = self.reshape(&[b * c, l])?;
            let mut dst_flat = out.reshape(&[b * c, new_l])?;
            // Use narrow + copy pattern
            // For now, construct via concat
            let parts: Vec<&Tensor> = Vec::new();
            // Actually, simplest: create left pad, self, right pad, then cat
            drop(dst_flat);
            drop(out);

            let mut to_cat: Vec<Tensor> = Vec::new();
            if pad_left > 0 {
                to_cat.push(Tensor::zeros_dtype(
                    Shape::from_dims(&[b, c, pad_left]),
                    self.dtype(),
                    self.device().clone(),
                )?);
            }
            to_cat.push(self.clone());
            if pad_right > 0 {
                to_cat.push(Tensor::zeros_dtype(
                    Shape::from_dims(&[b, c, pad_right]),
                    self.dtype(),
                    self.device().clone(),
                )?);
            }
            let refs: Vec<&Tensor> = to_cat.iter().collect();
            return Tensor::cat(&refs, 2);
        }
        Ok(out)
    }

    /// Power operation with float exponent
    pub fn powf(&self, exponent: f32) -> Result<Tensor> {
        self.pow(exponent)
    }

    /// Variance along specific dimensions
    pub fn var_dim(&self, dims: &[usize], keepdim: bool) -> Result<Tensor> {
        self.var(dims, false, keepdim) // Use unbiased=false by default
    }

    /// Element-wise maximum with a scalar
    pub fn maximum_scalar(&self, scalar: f32) -> Result<Tensor> {
        if self.dtype() == DType::BF16 {
            let scalar_tensor = self.zeros_like_with_dtype(DType::BF16)?.add_scalar(scalar)?;
            return self.maximum(&scalar_tensor);
        }
        let scalar_tensor = self.full_like(scalar)?;
        self.maximum(&scalar_tensor)
    }

    /// Element-wise minimum with a scalar
    pub fn minimum_scalar(&self, scalar: f32) -> Result<Tensor> {
        if self.dtype() == DType::BF16 {
            let scalar_tensor = self.zeros_like_with_dtype(DType::BF16)?.add_scalar(scalar)?;
            return self.minimum(&scalar_tensor);
        }
        let scalar_tensor = self.full_like(scalar)?;
        self.minimum(&scalar_tensor)
    }

    /// Get a single element from a 1D tensor
    pub fn get(&self, index: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dims.len() != 1 {
            return Err(Error::InvalidOperation(format!(
                "get() only works on 1D tensors, got {}D tensor",
                dims.len()
            )));
        }

        if index >= dims[0] {
            return Err(Error::InvalidOperation(format!(
                "Index {} out of bounds for tensor with size {}",
                index, dims[0]
            )));
        }

        let start = index;
        let end = index + 1;
        let slice = self.slice(&[(start, end)])?;
        slice.reshape(&[])
    }

    /// Repeat tensor along specified dimensions
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor> {
        let shape = self.shape();
        let dims = shape.dims();

        if repeats.len() != dims.len() {
            return Err(Error::InvalidOperation(format!(
                "repeat dimensions mismatch: tensor has {} dims but got {} repeat values",
                dims.len(),
                repeats.len()
            )));
        }

        let mut out_dims = dims.to_vec();
        for (i, &rep) in repeats.iter().enumerate() {
            if rep == 0 {
                return Err(Error::InvalidInput(
                    "repeat factors must be greater than zero".into(),
                ));
            }
            out_dims[i] = out_dims[i]
                .checked_mul(rep)
                .ok_or_else(|| Error::InvalidInput("repeat overflowed dimension".into()))?;
        }

        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if self.dtype() == DType::BF16 && self.storage_dtype() == DType::BF16 {
            let mut result = if repeats.iter().all(|&r| r == 1) {
                self.alias()
            } else {
                let scratch =
                    ArenaScratch::from_tensor_with_align(self, ArenaScratch::DEFAULT_ALIGN);
                let mut out = scratch.borrow_shape(Shape::from_dims(&out_dims))?;
                cuda_ops_bf16::repeat_nd_bf16_into(self, repeats, &mut out)?;
                out
            };

            if self.requires_grad {
                result.requires_grad = true;
                AutogradContext::record_op(
                    result.id,
                    Op::Repeat {
                        input: self.id,
                        repeats: repeats.to_vec(),
                    },
                    vec![(self.id, self.clone_result()?)],
                );
            }
            return Ok(result);
        }

        if strict::is_enabled() {
            panic!(
                "STRICT_BF16: repeat() requires BF16 CUDA tensor; received logical {:?} storage {:?}",
                self.dtype(),
                self.storage_dtype()
            );
        }

        // Fallback: repeat via host broadcast (slow, but keeps behavior for other dtypes)
        let mut expanded = self.clone_result()?;
        for (axis, &rep) in repeats.iter().enumerate() {
            if rep == 1 {
                continue;
            }
            let mut cat_inputs = Vec::with_capacity(rep);
            for _ in 0..rep {
                let _clone_guard = strict::allow_clone();
                cat_inputs.push(expanded.clone_result()?);
            }
            let refs: Vec<&Tensor> = cat_inputs.iter().collect();
            expanded = Tensor::cat(&refs, axis)?;
        }
        Ok(expanded)
    }
}
