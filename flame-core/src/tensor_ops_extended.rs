//! Extended tensor operations for FLAME
//! Adds operations needed for diffusion model training

use crate::{Tensor, Shape, Result, FlameError};
use crate::autograd::{AutogradContext, Op};
use crate::cuda_ops::GpuOps;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

impl Tensor {
    /// Random uniform tensor
    pub fn rand(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let data: Vec<f32> = (0..size)
            .map(|_| rng.gen::<f32>())
            .collect();
            
        Self::from_vec(data, shape, device)
    }
    
    /// Create a tensor with uniform distribution
    pub fn uniform(shape: Shape, low: f32, high: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let data: Vec<f32> = (0..size)
            .map(|_| rng.gen_range(low..high))
            .collect();
            
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
            return Err(FlameError::InvalidOperation(
                format!("Transpose dimensions out of bounds: {} and {} for tensor with {} dims", 
                    dim0, dim1, dims.len())
            ));
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
                return Err(FlameError::InvalidOperation(
                    format!("Dimension {} out of range", d)
                ));
            }
            if dims[d] != 1 {
                return Ok(self.clone()?);
            }
            dims.iter().enumerate()
                .filter(|(i, _)| *i != d)
                .map(|(_, &size)| size)
                .collect()
        } else {
            dims.iter().copied()
                .filter(|&size| size != 1)
                .collect()
        };
        
        self.reshape(&new_dims)
    }
    
    /// Unsqueeze: Add a dimension of size 1
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim > dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of range for unsqueeze", dim)
            ));
        }
        
        let mut new_dims = dims.to_vec();
        new_dims.insert(dim, 1);
        
        self.reshape(&new_dims)
    }
    
    /// Greater than comparison
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        
        let result: Vec<f32> = self_data.iter()
            .zip(other_data.iter())
            .map(|(a, b)| if a > b { 1.0 } else { 0.0 })
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Greater than or equal comparison
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        
        let result: Vec<f32> = self_data.iter()
            .zip(other_data.iter())
            .map(|(a, b)| if a >= b { 1.0 } else { 0.0 })
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Less than comparison
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        
        let result: Vec<f32> = self_data.iter()
            .zip(other_data.iter())
            .map(|(a, b)| if a < b { 1.0 } else { 0.0 })
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Not equal comparison
    pub fn ne(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        
        let result: Vec<f32> = self_data.iter()
            .zip(other_data.iter())
            .map(|(a, b)| if a != b { 1.0 } else { 0.0 })
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
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
            let n = dims.iter().map(|&d| self.shape.dims()[d]).product::<usize>() as f32;
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
    pub fn mean_along_dims(&self, dims: &[usize], keepdim: bool) -> Result<Tensor> {
        // Sum along dimensions
        let mut result = self.clone()?;
        for &dim in dims {
            result = result.sum_dim(dim)?;
        }
        
        // Compute divisor
        let divisor = dims.iter()
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
    
    /// Sum along dimension with option to keep dimension
    pub fn sum_keepdim(&self, dim: isize) -> Result<Tensor> {
        let ndim = self.shape.dims().len() as isize;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        
        if dim >= self.shape.dims().len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of bounds", dim)
            ));
        }
        
        self.sum_dim_keepdim(dim)
    }
    
    /// Permute tensor dimensions (already implemented above)
    /// Chunk tensor into n chunks along specified dimension
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Tensor>> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of bounds for tensor with {} dimensions", dim, shape.len())
            ));
        }
        
        let dim_size = shape[dim];
        if dim_size % chunks != 0 {
            return Err(FlameError::InvalidOperation(
                format!("Cannot evenly chunk dimension {} of size {} into {} chunks", dim, dim_size, chunks)
            ));
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
        // Validate ranges
        let shape = self.shape().dims();
        if ranges.len() != shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Number of ranges {} doesn't match tensor dimensions {}", ranges.len(), shape.len())
            ));
        }
        
        // Calculate new shape
        let mut new_shape = Vec::new();
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start >= end || end > shape[i] {
                return Err(FlameError::InvalidOperation(
                    format!("Invalid range [{}, {}) for dimension {} of size {}", start, end, i, shape[i])
                ));
            }
            new_shape.push(end - start);
        }
        
        // For now, implement as a copy operation
        // In production, this would be a view into the original data
        let new_size = new_shape.iter().product();
        let mut output_data = vec![0.0f32; new_size];
        
        // Copy data (simplified - in production would use CUDA kernels)
        let src_data = self.to_vec()?;
        
        // Calculate strides
        let mut src_strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            src_strides[i] = src_strides[i + 1] * shape[i + 1];
        }
        
        let mut dst_strides = vec![1; new_shape.len()];
        for i in (0..new_shape.len() - 1).rev() {
            dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
        }
        
        // Copy elements
        fn copy_recursive(
            src: &[f32],
            dst: &mut [f32],
            ranges: &[(usize, usize)],
            src_shape: &[usize],
            src_strides: &[usize],
            dst_strides: &[usize],
            dim: usize,
            src_offset: usize,
            dst_offset: usize,
        ) {
            if dim == ranges.len() {
                dst[dst_offset] = src[src_offset];
                return;
            }
            
            let (start, end) = ranges[dim];
            for i in start..end {
                let new_src_offset = src_offset + i * src_strides[dim];
                let new_dst_offset = dst_offset + (i - start) * dst_strides[dim];
                copy_recursive(
                    src, dst, ranges, src_shape, src_strides, dst_strides,
                    dim + 1, new_src_offset, new_dst_offset
                );
            }
        }
        
        copy_recursive(
            &src_data, &mut output_data, ranges, shape, &src_strides, &dst_strides,
            0, 0, 0
        );
        
        // Create output tensor
        Tensor::from_slice(&output_data, Shape::from_dims(&new_shape), self.device.clone())
    }
    
    /// Concatenate tensors along a dimension
    pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(FlameError::InvalidOperation("Cannot concatenate empty tensor list".into()));
        }
        
        // Check all tensors have same shape except for concat dimension
        let first_shape = tensors[0].shape().dims();
        let device = tensors[0].device().clone();
        
        for tensor in tensors.iter().skip(1) {
            let shape = tensor.shape().dims();
            if shape.len() != first_shape.len() {
                return Err(FlameError::InvalidOperation(
                    "All tensors must have same number of dimensions".into()
                ));
            }
            
            for (i, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i != dim && s1 != s2 {
                    return Err(FlameError::InvalidOperation(
                        format!("Dimension {} must match for concatenation (got {} and {})", i, s1, s2)
                    ));
                }
            }
        }
        
        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        output_shape[dim] = tensors.iter().map(|t| t.shape().dims()[dim]).sum();
        
        // Concatenate data
        let output_size = output_shape.iter().product();
        let mut output_data = vec![0.0f32; output_size];
        
        // Calculate strides
        let mut strides = vec![1; output_shape.len()];
        for i in (0..output_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * output_shape[i + 1];
        }
        
        let mut offset_along_dim = 0;
        for tensor in tensors {
            let tensor_data = tensor.to_vec()?;
            let tensor_shape = tensor.shape().dims();
            
            // Copy tensor data to output
            fn copy_tensor(
                src: &[f32],
                dst: &mut [f32],
                shape: &[usize],
                output_shape: &[usize],
                strides: &[usize],
                concat_dim: usize,
                dim_offset: usize,
                dim: usize,
                src_idx: usize,
                dst_idx: usize,
            ) {
                if dim == shape.len() {
                    dst[dst_idx] = src[src_idx];
                    return;
                }
                
                let size = shape[dim];
                for i in 0..size {
                    let src_stride = if dim + 1 < shape.len() {
                        shape[dim + 1..].iter().product::<usize>()
                    } else {
                        1
                    };
                    
                    let dst_offset = if dim == concat_dim {
                        (i + dim_offset) * strides[dim]
                    } else {
                        i * strides[dim]
                    };
                    
                    copy_tensor(
                        src, dst, shape, output_shape, strides, concat_dim, dim_offset,
                        dim + 1,
                        src_idx + i * src_stride,
                        dst_idx + dst_offset
                    );
                }
            }
            
            copy_tensor(
                &tensor_data, &mut output_data,
                tensor_shape, &output_shape, &strides,
                dim, offset_along_dim,
                0, 0, 0
            );
            
            offset_along_dim += tensor_shape[dim];
        }
        
        let mut output = Tensor::from_slice(&output_data, Shape::from_dims(&output_shape), device)?;
        
        // AUTOGRAD: Record operation if needed
        let requires_grad = tensors.iter().any(|t| t.requires_grad);
        if requires_grad {
            output.requires_grad = true;
            
            let mut saved_tensors = Vec::new();
            let mut input_ids = Vec::new();
            
            for tensor in tensors {
                let id = tensor.id;
                saved_tensors.push((id, (*tensor).clone()?));
                input_ids.push(id);
            }
            
            AutogradContext::record_op(
                output.id,
                Op::Cat { 
                    inputs: input_ids,
                    dim
                },
                saved_tensors
            );
        }
        
        Ok(output)
    }
    
    /// Index select along a dimension
    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of bounds for tensor with {} dimensions", dim, shape.len())
            ));
        }
        
        // Indices should be 1D
        let indices_shape = indices.shape().dims();
        if indices_shape.len() != 1 {
            return Err(FlameError::InvalidOperation(
                format!("Indices must be 1D, got shape {:?}", indices_shape)
            ));
        }
        
        // Get indices as vec
        let indices_data = indices.to_vec()?;
        let num_indices = indices_data.len();
        
        // Calculate output shape
        let mut output_shape = shape.to_vec();
        output_shape[dim] = num_indices;
        
        // Gather data
        let output_size: usize = output_shape.iter().product();
        let mut output_data = vec![0.0f32; output_size];
        let src_data = self.to_vec()?;
        
        // Calculate strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        // Copy selected indices
        fn copy_indexed(
            src: &[f32],
            dst: &mut [f32],
            shape: &[usize],
            output_shape: &[usize],
            strides: &[usize],
            indices: &[f32],
            select_dim: usize,
            dim: usize,
            src_base: usize,
            dst_idx: usize,
        ) {
            if dim == shape.len() {
                // We've reached a single element
                return;
            }
            
            if dim == select_dim {
                // Use indices for this dimension
                for (out_i, &idx) in indices.iter().enumerate() {
                    let src_idx = src_base + (idx as usize) * strides[dim];
                    let dst_offset = out_i * strides[dim] * output_shape[dim] / shape[dim];
                    
                    if dim + 1 == shape.len() {
                        dst[dst_idx + dst_offset] = src[src_idx];
                    } else {
                        copy_indexed(
                            src, dst, shape, output_shape, strides, indices, select_dim,
                            dim + 1, src_idx, dst_idx + dst_offset
                        );
                    }
                }
            } else {
                // Normal iteration for other dimensions
                for i in 0..shape[dim] {
                    let src_offset = i * strides[dim];
                    let dst_offset = i * strides[dim] * output_shape[dim] / shape[dim];
                    
                    if dim + 1 == shape.len() {
                        dst[dst_idx + dst_offset] = src[src_base + src_offset];
                    } else {
                        copy_indexed(
                            src, dst, shape, output_shape, strides, indices, select_dim,
                            dim + 1, src_base + src_offset, dst_idx + dst_offset
                        );
                    }
                }
            }
        }
        
        copy_indexed(
            &src_data, &mut output_data,
            shape, &output_shape, &strides, &indices_data,
            dim, 0, 0, 0
        );
        
        Tensor::from_slice(&output_data, Shape::from_dims(&output_shape), self.device.clone())
    }
    
    /// Expand tensor to a new shape (broadcasting)
    pub fn expand(&self, new_shape: &[usize]) -> Result<Tensor> {
        let shape = self.shape().dims();
        
        // Validate expansion
        if new_shape.len() < shape.len() {
            return Err(FlameError::InvalidOperation(
                "Cannot expand to fewer dimensions".into()
            ));
        }
        
        // Check compatibility
        let offset = new_shape.len() - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let new_dim = new_shape[i + offset];
            if dim != new_dim && dim != 1 {
                return Err(FlameError::InvalidOperation(
                    format!("Cannot expand dimension {} from {} to {}", i, dim, new_dim)
                ));
            }
        }
        
        // For now, implement as broadcast_to
        self.broadcast_to(&Shape::from_dims(new_shape))
    }
    
    
    /// Compute natural logarithm
    pub fn log(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let output: Vec<f32> = data.iter().map(|x| x.ln()).collect();
        Tensor::from_slice(&output, self.shape.clone(), self.device.clone())
    }
    
    
    /// Compute reciprocal square root
    pub fn rsqrt(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let output: Vec<f32> = data.iter().map(|x| 1.0 / x.sqrt()).collect();
        Tensor::from_slice(&output, self.shape.clone(), self.device.clone())
    }
    
    /// Negate tensor
    pub fn neg(&self) -> Result<Tensor> {
        self.mul_scalar(-1.0)
    }
    
    /// Compute absolute value
    pub fn abs(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let output: Vec<f32> = data.iter().map(|x| x.abs()).collect();
        Tensor::from_slice(&output, self.shape.clone(), self.device.clone())
    }
    
    /// Clamp values between min and max
    pub fn clamp(&self, min: f32, max: f32) -> Result<Tensor> {
        let data = self.to_vec()?;
        let output: Vec<f32> = data.iter().map(|x| x.clamp(min, max)).collect();
        Tensor::from_slice(&output, self.shape.clone(), self.device.clone())
    }
    
    /// Compute element-wise maximum with another tensor
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        // Check shapes are compatible for broadcasting
        let broadcast_shape = broadcast_shapes(self.shape().dims(), other.shape().dims())?;
        
        // Broadcast both tensors if needed
        let a = if self.shape().dims() != &broadcast_shape {
            self.broadcast_to(&Shape::from_dims(&broadcast_shape))?
        } else {
            self.clone()?
        };
        
        let b = if other.shape().dims() != &broadcast_shape {
            other.broadcast_to(&Shape::from_dims(&broadcast_shape))?
        } else {
            other.clone()?
        };
        
        // Compute maximum
        let a_data = a.to_vec()?;
        let b_data = b.to_vec()?;
        let output: Vec<f32> = a_data.iter().zip(b_data.iter())
            .map(|(a, b)| a.max(*b))
            .collect();
        
        Tensor::from_slice(&output, Shape::from_dims(&broadcast_shape), self.device.clone())
    }
    
    /// Compute element-wise minimum with another tensor
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        // Check shapes are compatible for broadcasting
        let broadcast_shape = broadcast_shapes(self.shape().dims(), other.shape().dims())?;
        
        // Broadcast both tensors if needed
        let a = if self.shape().dims() != &broadcast_shape {
            self.broadcast_to(&Shape::from_dims(&broadcast_shape))?
        } else {
            self.clone()?
        };
        
        let b = if other.shape().dims() != &broadcast_shape {
            other.broadcast_to(&Shape::from_dims(&broadcast_shape))?
        } else {
            other.clone()?
        };
        
        // Compute minimum
        let a_data = a.to_vec()?;
        let b_data = b.to_vec()?;
        let output: Vec<f32> = a_data.iter().zip(b_data.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        
        Tensor::from_slice(&output, Shape::from_dims(&broadcast_shape), self.device.clone())
    }
    
    
    /// Get maximum value along a dimension
    pub fn max_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of bounds", dim)
            ));
        }
        
        // Calculate output shape
        let mut output_shape = shape.to_vec();
        if keepdim {
            output_shape[dim] = 1;
        } else {
            output_shape.remove(dim);
        }
        
        // For now, implement with to_vec (in production would use CUDA kernels)
        let data = self.to_vec()?;
        
        // Calculate strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        let output_size: usize = output_shape.iter().product();
        let mut output = vec![f32::NEG_INFINITY; output_size];
        
        // Compute max
        for idx in 0..data.len() {
            // Decompose linear index
            let mut remaining = idx;
            let mut indices = vec![0; shape.len()];
            for i in 0..shape.len() {
                indices[i] = remaining / strides[i];
                remaining %= strides[i];
            }
            
            // Calculate output index
            let mut out_idx = 0;
            let mut out_stride = 1;
            for i in (0..shape.len()).rev() {
                if i != dim {
                    let dim_idx = if i > dim && !keepdim { i - 1 } else { i };
                    if dim_idx < output_shape.len() {
                        out_idx += indices[i] * out_stride;
                        out_stride *= output_shape[dim_idx];
                    }
                }
            }
            
            output[out_idx] = output[out_idx].max(data[idx]);
        }
        
        Tensor::from_slice(&output, Shape::from_dims(&output_shape), self.device.clone())
    }
    
    /// Sum along dimension keeping dimension
    pub fn sum_dim_keepdim(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of bounds", dim)
            ));
        }
        
        // Calculate output shape (keep dimension with size 1)
        let mut output_shape = shape.to_vec();
        output_shape[dim] = 1;
        
        // For now, implement with to_vec
        let data = self.to_vec()?;
        
        // Calculate strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        let output_size: usize = output_shape.iter().product();
        let mut output = vec![0.0f32; output_size];
        
        // Sum along dimension
        for idx in 0..data.len() {
            // Decompose linear index
            let mut remaining = idx;
            let mut indices = vec![0; shape.len()];
            for i in 0..shape.len() {
                indices[i] = remaining / strides[i];
                remaining %= strides[i];
            }
            
            // Calculate output index (set dim index to 0)
            indices[dim] = 0;
            let mut out_idx = 0;
            for i in 0..shape.len() {
                out_idx += indices[i] * strides[i];
            }
            
            output[out_idx] += data[idx];
        }
        
        Tensor::from_slice(&output, Shape::from_dims(&output_shape), self.device.clone())
    }
    
    /// Divide by another tensor
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        // Check if shapes match exactly
        if self.shape == other.shape {
            // Use CUDA kernel for GPU-accelerated division
            let mut output = GpuOps::div(self, other)?;
            
            // AUTOGRAD: Record operation if needed
            if self.requires_grad || other.requires_grad {
                output.requires_grad = true;
                
                AutogradContext::record_op(
                    output.id,
                    Op::Div { 
                        lhs: self.id,
                        rhs: other.id
                    },
                    vec![
                        (self.id, self.clone()?),
                        (other.id, other.clone()?)
                    ]
                );
            }
            
            Ok(output)
        } else {
            // Broadcasting case
            let broadcast_shape = broadcast_shapes(self.shape().dims(), other.shape().dims())?;
            
            // Broadcast both tensors to the target shape
            let self_broadcast = if self.shape().dims() == &broadcast_shape {
                self.clone()?
            } else {
                self.broadcast_to(&Shape::from_dims(&broadcast_shape))?
            };
            
            let other_broadcast = if other.shape().dims() == &broadcast_shape {
                other.clone()?
            } else {
                other.broadcast_to(&Shape::from_dims(&broadcast_shape))?
            };
            
            // Now divide with matching shapes
            self_broadcast.div(&other_broadcast)
        }
    }
    
    
    /// Element-wise equality comparison
    pub fn eq(&self, other: &Tensor) -> Result<Tensor> {
        // Check shapes are compatible
        if self.shape().dims() != other.shape().dims() {
            return Err(FlameError::InvalidOperation(
                "Tensors must have same shape for equality comparison".into()
            ));
        }
        
        let a_data = self.to_vec()?;
        let b_data = other.to_vec()?;
        let output: Vec<f32> = a_data.iter().zip(b_data.iter())
            .map(|(a, b)| if a == b { 1.0 } else { 0.0 })
            .collect();
        
        Tensor::from_slice(&output, self.shape.clone(), self.device.clone())
    }
    
    /// Create tensor filled with single value
    pub fn full(shape: Shape, value: f32, device: Arc<CudaDevice>) -> Result<Tensor> {
        let size = shape.elem_count();
        let data = vec![value; size];
        Tensor::from_slice(&data, shape, device)
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
        let data = self.to_vec()?;
        let sign_data: Vec<f32> = data.iter()
            .map(|&x| {
                if x > 0.0 { 1.0 }
                else if x < 0.0 { -1.0 }
                else { 0.0 }
            })
            .collect();
        
        Tensor::from_vec(sign_data, self.shape.clone(), self.device.clone())
    }
    
    /// Element-wise floor operation
    pub fn floor(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let floor_data: Vec<f32> = data.iter()
            .map(|&x| x.floor())
            .collect();
        
        Tensor::from_vec(floor_data, self.shape.clone(), self.device.clone())
    }
    
    /// Element-wise ceil operation
    pub fn ceil(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let ceil_data: Vec<f32> = data.iter()
            .map(|&x| x.ceil())
            .collect();
        
        Tensor::from_vec(ceil_data, self.shape.clone(), self.device.clone())
    }
    
    /// Element-wise round operation
    pub fn round(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let round_data: Vec<f32> = data.iter()
            .map(|&x| x.round())
            .collect();
        
        Tensor::from_vec(round_data, self.shape.clone(), self.device.clone())
    }
    
    /// Element-wise less than or equal comparison
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        
        let result: Vec<f32> = self_data.iter()
            .zip(other_data.iter())
            .map(|(a, b)| if a <= b { 1.0 } else { 0.0 })
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Subtract a scalar from all elements
    pub fn sub_scalar(&self, scalar: f32) -> Result<Tensor> {
        let neg_scalar = -scalar;
        self.add_scalar(neg_scalar)
    }
    
    /// Find the maximum value in the tensor
    pub fn max_all(&self) -> Result<f32> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Err(FlameError::InvalidOperation("Cannot find max of empty tensor".into()));
        }
        Ok(data.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
    }
    
    /// Find the sum of all elements in the tensor
    pub fn sum_all(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let sum: f32 = data.iter().sum();
        Tensor::from_slice(&[sum], Shape::from_dims(&[1]), self.device.clone())
    }
    
    /// Flip tensor along specified dimensions
    pub fn flip(&self, dims: &[usize]) -> Result<Tensor> {
        let mut data = self.to_vec()?;
        let shape_dims = self.shape.dims();
        
        // For simplicity, only implement flipping along the last dimension for now
        if dims.len() == 1 && dims[0] == shape_dims.len() - 1 {
            let last_dim_size = shape_dims[dims[0]];
            let stride = data.len() / last_dim_size;
            
            for i in 0..stride {
                let start = i * last_dim_size;
                let end = start + last_dim_size;
                data[start..end].reverse();
            }
        } else {
            return Err(FlameError::InvalidOperation("Flip only supports flipping along the last dimension currently".into()));
        }
        
        Tensor::from_vec(data, self.shape.clone(), self.device.clone())
    }
    
    /// Create upper triangular matrix (ones above diagonal, zeros below)
    pub fn triu(&self, diagonal: i32) -> Result<Tensor> {
        let shape_dims = self.shape.dims();
        if shape_dims.len() < 2 {
            return Err(FlameError::InvalidOperation("triu requires at least 2D tensor".into()));
        }
        
        let data = self.to_vec()?;
        let mut result = vec![0.0f32; data.len()];
        
        let rows = shape_dims[shape_dims.len() - 2];
        let cols = shape_dims[shape_dims.len() - 1];
        let matrix_size = rows * cols;
        let num_matrices = data.len() / matrix_size;
        
        for m in 0..num_matrices {
            let offset = m * matrix_size;
            for i in 0..rows {
                for j in 0..cols {
                    if j as i32 >= i as i32 + diagonal {
                        let idx = offset + i * cols + j;
                        result[idx] = data[idx];
                    }
                }
            }
        }
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Conditional selection based on mask
    pub fn where_tensor(&self, true_tensor: &Tensor, false_tensor: &Tensor) -> Result<Tensor> {
        if self.shape != true_tensor.shape || self.shape != false_tensor.shape {
            return Err(FlameError::InvalidOperation(
                "All tensors must have the same shape for where operation".into()
            ));
        }
        
        let mask_data = self.to_vec()?;
        let true_data = true_tensor.to_vec()?;
        let false_data = false_tensor.to_vec()?;
        
        let result: Vec<f32> = mask_data.iter()
            .zip(true_data.iter())
            .zip(false_data.iter())
            .map(|((&m, &t), &f)| if m > 0.0 { t } else { f })
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
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
            
            AutogradContext::record_op(
                output.id,
                Op::LogSoftmax { 
                    input: self.id,
                    dim: dim as isize
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// Get data as 1D vector of i64
    pub fn to_vec1<T: From<f32>>(&self) -> Result<Vec<T>> {
        let data = self.to_vec()?;
        Ok(data.into_iter().map(|x| T::from(x)).collect())
    }
    
    /// Get data as 2D vector
    pub fn to_vec2<T: From<f32>>(&self) -> Result<Vec<Vec<T>>> {
        if self.shape.dims().len() != 2 {
            return Err(FlameError::InvalidOperation(
                format!("to_vec2 requires 2D tensor, got {:?}", self.shape.dims())
            ));
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
    pub fn from_vec2<T: Into<f32> + Copy>(data: Vec<Vec<T>>, device: Arc<CudaDevice>) -> Result<Tensor> {
        let rows = data.len();
        if rows == 0 {
            return Err(FlameError::InvalidOperation("Empty 2D vector".into()));
        }
        
        let cols = data[0].len();
        let mut flat_data = Vec::with_capacity(rows * cols);
        
        for row in data {
            if row.len() != cols {
                return Err(FlameError::InvalidOperation("Inconsistent row lengths".into()));
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
            return Err(FlameError::InvalidOperation(
                format!("to_scalar requires scalar tensor, got shape {:?}", self.shape.dims())
            ));
        }
        
        let data = self.to_vec()?;
        Ok(T::from(data[0]))
    }
    
    /// Split tensor into multiple tensors along a dimension
    pub fn split(&self, sizes: &[usize], dim: usize) -> Result<Vec<Tensor>> {
        let shape = self.shape().dims();
        if dim >= shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of bounds for tensor with {} dimensions", dim, shape.len())
            ));
        }
        
        // Check sizes sum to dimension size
        let total_size: usize = sizes.iter().sum();
        if total_size != shape[dim] {
            return Err(FlameError::InvalidOperation(
                format!("Split sizes {:?} don't sum to dimension size {}", sizes, shape[dim])
            ));
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
            for (i, tensor) in result.iter_mut().enumerate() {
                tensor.requires_grad = true;
                
                // Record split operation for each output
                AutogradContext::record_op(
                    tensor.id,
                    Op::Split { 
                        input: self.id,
                        sizes: sizes.to_vec(),
                        dim
                    },
                    vec![(self.id, self.clone()?)]
                );
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
            return Err(FlameError::InvalidOperation(
                format!("Cannot broadcast dimensions {} and {}", dim1, dim2)
            ));
        }
    }
    
    Ok(result)
}

// Note: softmax, unsqueeze, full_like, and gelu are already implemented in tensor.rs
