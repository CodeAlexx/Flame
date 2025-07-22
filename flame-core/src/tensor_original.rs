use crate::{Shape, Result, FlameError};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
use cudarc::cublas::CudaBlas;
use std::sync::Arc;
use crate::cuda_kernels::CudaKernels;

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// Helper function to broadcast two shape arrays
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

/// The core tensor type with GPU-accelerated operations
pub struct Tensor {
    pub(crate) data: CudaSlice<f32>,
    pub(crate) shape: Shape,
    pub(crate) device: Arc<CudaDevice>,
    
    // For autograd
    pub(crate) requires_grad: bool,
    pub(crate) grad: Option<Box<Tensor>>,
    pub(crate) graph_id: Option<crate::autograd::TensorId>,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        // Copy the data on the GPU
        let mut data = self.device.alloc_zeros::<f32>(self.shape.elem_count())
            .expect("Failed to allocate GPU memory for clone");
        
        // Copy data from source to destination
        self.device.dtod_copy(&self.data, &mut data)
            .expect("Failed to copy GPU data");
        
        // Clone the gradient if it exists
        let grad = self.grad.as_ref().map(|g| g.clone());
        
        Self {
            data,
            shape: self.shape.clone(),
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            grad,
            graph_id: None,
        }
    }
}

impl Tensor {
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let data = device.alloc_zeros::<f32>(size)
            .map_err(|_| FlameError::CudaDriver)?;
        Ok(Self { 
            data, 
            shape, 
            device,
            requires_grad: false,
            grad: None,
            graph_id: None,
        })
    }
    
    /// Create a new tensor filled with ones
    pub fn ones(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let ones_vec = vec![1.0f32; size];
        Self::from_vec(ones_vec, shape, device)
    }

    /// Create a new tensor from a Vec
    pub fn from_vec(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        let data = device.htod_sync_copy(&data)
            .map_err(|_| FlameError::CudaDriver)?;
        Ok(Self { 
            data, 
            shape, 
            device,
            requires_grad: false,
            grad: None,
            graph_id: None,
        })
    }

    /// Create random tensor
    pub fn randn(shape: Shape, mean: f32, std: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std).unwrap();
        
        let cpu_data: Vec<f32> = (0..size)
            .map(|_| normal.sample(&mut rng))
            .collect();
            
        Self::from_vec(cpu_data, shape, device)
    }

    /// Enable gradient computation
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// THE KEY OPERATION - Mutable weight update
    pub fn update_weights(&mut self, gradient: &Tensor, lr: f32) -> Result<()> {
        if self.shape != gradient.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: gradient.shape.clone(),
            });
        }

        // Use CUDA kernel for in-place weight update
        CudaKernels::update_weights(self, gradient, lr)?;
        Ok(())
    }

    /// Matrix multiplication using cuBLAS
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let (m, k) = match self.shape.dims() {
            [m, k] => (*m, *k),
            _ => return Err(FlameError::InvalidOperation("matmul requires 2D tensors".into())),
        };
        
        let (k2, n) = match other.shape.dims() {
            [k2, n] => (*k2, *n),
            _ => return Err(FlameError::InvalidOperation("matmul requires 2D tensors".into())),
        };
        
        if k != k2 {
            return Err(FlameError::ShapeMismatch {
                expected: Shape::from_dims(&[k, n]),
                got: other.shape.clone(),
            });
        }

        let out_shape = Shape::from_dims(&[m, n]);
        let mut output = Tensor::zeros(out_shape, self.device.clone())?;
        
        // Use cuBLAS gemm
        let blas = CudaBlas::new(self.device.clone())
            .map_err(|_| FlameError::CuBlas)?;
            
        use cudarc::cublas::{GemmConfig, Gemm};
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };
        
        unsafe {
            blas.gemm(cfg, &other.data, &self.data, &mut output.data)
                .map_err(|_| FlameError::CuBlas)?;
        }
        
        Ok(output)
    }
    
    /// Batch matrix multiplication
    /// For tensors with 3+ dimensions, applies matmul to the last 2 dimensions
    /// Broadcasting is supported for batch dimensions
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor> {
        let self_shape = self.shape.dims();
        let other_shape = other.shape.dims();
        
        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(FlameError::InvalidOperation(
                "bmm requires tensors with at least 2 dimensions".into()
            ));
        }
        
        // Extract batch dimensions and matrix dimensions
        let self_batch = &self_shape[..self_shape.len() - 2];
        let other_batch = &other_shape[..other_shape.len() - 2];
        
        let (m, k1) = (self_shape[self_shape.len() - 2], self_shape[self_shape.len() - 1]);
        let (k2, n) = (other_shape[other_shape.len() - 2], other_shape[other_shape.len() - 1]);
        
        if k1 != k2 {
            return Err(FlameError::InvalidOperation(
                format!("bmm: incompatible matrix dimensions {} vs {}", k1, k2)
            ));
        }
        
        // Broadcast batch dimensions
        let broadcast_batch = broadcast_shapes(self_batch, other_batch)?;
        
        // Calculate total batch size
        let batch_size: usize = broadcast_batch.iter().product();
        
        // Reshape to 3D: [batch_size, m, k] and [batch_size, k, n]
        let self_3d = self.reshape_for_bmm(&broadcast_batch, m, k1)?;
        let other_3d = other.reshape_for_bmm(&broadcast_batch, k1, n)?;
        
        // Prepare output shape
        let mut output_shape = broadcast_batch.to_vec();
        output_shape.push(m);
        output_shape.push(n);
        
        let mut output = Tensor::zeros(Shape::from_dims(&output_shape), self.device.clone())?;
        
        // Use cuBLAS
        let blas = CudaBlas::new(self.device.clone())
            .map_err(|_| FlameError::CuBlas)?;
        
        // For now, implement as loop over batches
        // TODO: Use cuBLAS batched GEMM for better performance
        for b in 0..batch_size {
            let self_offset = b * m * k1;
            let other_offset = b * k1 * n;
            let output_offset = b * m * n;
            
            // Create views for this batch
            let self_batch = self_3d.slice(self_offset, m * k1)?;
            let other_batch = other_3d.slice(other_offset, k1 * n)?;
            
            // Perform matrix multiplication for this batch
            use cudarc::cublas::{GemmConfig, Gemm};
            let cfg = GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k1 as i32,
                alpha: 1.0,
                lda: n as i32,
                ldb: k1 as i32,
                beta: 0.0,
                ldc: n as i32,
            };
            
            unsafe {
                // Create a temporary buffer for this batch
                let mut batch_output = self.device.alloc_zeros::<f32>(m * n)
                    .map_err(|_| FlameError::CudaDriver)?;
                    
                blas.gemm(cfg, &other_batch.data, &self_batch.data, &mut batch_output)
                    .map_err(|_| FlameError::CuBlas)?;
                    
                // Now actually tries GPU-to-GPU memory copy
                let kernel_code = r#"
extern "C" __global__ void copy_at_offset(
    float* output,
    const float* input,
    int offset,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[offset + idx] = input[idx];
    }
}
"#;
                
                CudaKernels::ensure_kernel(&self.device, "copy_at_offset", kernel_code)?;
                
                let f = self.device.get_func("copy_at_offset", "copy_at_offset")
                    .ok_or_else(|| FlameError::Cuda("Failed to get copy_at_offset kernel".into()))?;
                
                let cfg = cudarc::driver::LaunchConfig::for_num_elems((m * n) as u32);
                
                launch_kernel!(f, cfg,
                    &output.data,
                    &batch_output,
                    output_offset as i32,
                    (m * n) as i32
                )?;
            }
        }
        
        Ok(output)
    }
    
    /// Helper to reshape tensor for batch matrix multiplication
    fn reshape_for_bmm(&self, target_batch: &[usize], m: usize, n: usize) -> Result<Tensor> {
        let self_shape = self.shape.dims();
        let self_batch = &self_shape[..self_shape.len() - 2];
        
        // If shapes match, just flatten batch dimensions
        if self_batch == target_batch {
            let batch_size: usize = target_batch.iter().product();
            return self.reshape(&[batch_size, m, n]);
        }
        
        // Otherwise, need to broadcast
        // This is a simplified version - full broadcasting would require more work
        Err(FlameError::InvalidOperation(
            "Broadcasting for bmm not fully implemented yet".into()
        ))
    }
    
    /// Create a slice view of the tensor
    fn slice(&self, start: usize, len: usize) -> Result<Tensor> {
        // This is a simplified slice that assumes contiguous memory
        // Full implementation would handle strides
        if start + len > self.shape.elem_count() {
            return Err(FlameError::InvalidOperation(
                "Slice out of bounds".into()
            ));
        }
        
        // Now actually tries GPU-to-GPU slice copy
        let kernel_code = r#"
extern "C" __global__ void slice_kernel(
    float* output,
    const float* input,
    int start,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = input[start + idx];
    }
}
"#;
        
        CudaKernels::ensure_kernel(&self.device, "slice_kernel", kernel_code)?;
        
        let f = self.device.get_func("slice_kernel", "slice_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get slice_kernel".into()))?;
        
        let slice_data = self.device.alloc_zeros::<f32>(len)?;
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(len as u32);
        
        launch_kernel!(f, cfg,
            &slice_data,
            &self.data,
            start as i32,
            len as i32
        )?;
        
        Ok(Tensor {
            data: slice_data,
            shape: Shape::from_dims(&[len]),
            device: self.device.clone(),
            requires_grad: false,
            grad: None,
            graph_id: None,
        })
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated addition
        CudaKernels::add(self, other)
    }

    /// Subtract another tensor
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        // Implement as self + (-1 * other)
        let neg_other = other.mul_scalar(-1.0)?;
        self.add(&neg_other)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated multiplication
        CudaKernels::mul(self, other)
    }

    /// Scalar multiplication 
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated scalar multiplication
        CudaKernels::mul_scalar(self, scalar)
    }
    
    /// Scale tensor by a scalar (alias for mul_scalar)
    pub fn scale(&self, scalar: f32) -> Result<Tensor> {
        self.mul_scalar(scalar)
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated ReLU
        CudaKernels::relu(self)
    }
    
    /// GELU activation
    pub fn gelu(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated GELU
        CudaKernels::gelu(self)
    }
    
    /// SiLU (Swish) activation
    pub fn silu(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated SiLU
        CudaKernels::silu(self)
    }
    
    /// Tanh activation
    pub fn tanh(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated Tanh
        CudaKernels::tanh(self)
    }

    /// Square all elements
    pub fn square(&self) -> Result<Tensor> {
        self.mul(self)
    }

    /// Mean reduction
    pub fn mean(&self) -> Result<Tensor> {
        let sum = self.sum()?;
        let count = self.shape.elem_count() as f32;
        sum.mul_scalar(1.0 / count)
    }

    /// Sum reduction 
    pub fn sum(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated sum reduction
        CudaKernels::sum(self)
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Copy to CPU
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.device.dtoh_sync_copy(&self.data)
            .map_err(|_| FlameError::CudaDriver)?)
    }

    /// Get single value
    pub fn item(&self) -> Result<f32> {
        if self.shape.elem_count() != 1 {
            return Err(FlameError::InvalidOperation("item() requires tensor with single element".into()));
        }
        Ok(self.to_vec()?[0])
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated transpose
        CudaKernels::transpose(self)
    }
    
    /// Transpose two dimensions of a tensor
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
    
    /// Get raw data reference (for internal use by CUDA kernels)
    pub fn data(&self) -> &CudaSlice<f32> {
        &self.data
    }
    
    /// Get mutable raw data reference (for internal use by CUDA kernels)
    pub fn data_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.data
    }
    
    /// Clone the tensor (creates a new tensor with copied data)
    pub fn clone(&self) -> Result<Tensor> {
        let mut data = unsafe { self.device.alloc::<f32>(self.shape.elem_count()) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        self.device.dtod_copy(&self.data, &mut data)
            .map_err(|_| FlameError::CudaDriver)?;
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            grad: None,
            graph_id: None,
        })
    }
    
    /// Detach from computation graph
    pub fn detach(&self) -> Result<Tensor> {
        let mut data = unsafe { self.device.alloc::<f32>(self.shape.elem_count()) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        self.device.dtod_copy(&self.data, &mut data)
            .map_err(|_| FlameError::CudaDriver)?;
            
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            device: self.device.clone(),
            requires_grad: false,
            grad: None,
            graph_id: None,
        })
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
    
    /// Accumulate gradients
    pub fn accumulate_grad(&mut self, grad: Tensor) -> Result<()> {
        match &mut self.grad {
            Some(existing) => {
                // Add to existing gradient
                **existing = existing.add(&grad)?;
            }
            None => {
                self.grad = Some(Box::new(grad));
            }
        }
        Ok(())
    }
    
    /// Get gradient
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_deref()
    }
    
    /// Perform backward pass to compute gradients
    pub fn backward(&self) -> Result<()> {
        if !self.requires_grad {
            return Err(FlameError::InvalidOperation(
                "backward() called on tensor that doesn't require grad".into()
            ));
        }
        
        if self.shape.elem_count() != 1 {
            return Err(FlameError::InvalidOperation(
                "backward() can only be called on scalar tensors".into()
            ));
        }
        
        // Use the autograd engine to compute gradients
        crate::autograd_engine::ENGINE.with(|engine| {
            let engine = engine.lock().unwrap();
            engine.backward(self)
        })
    }
    
    /// Reshape tensor to new dimensions
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.shape.elem_count() {
            return Err(FlameError::InvalidOperation(
                format!("Cannot reshape from {:?} to {:?}", self.shape.dims(), new_shape)
            ));
        }
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::from_dims(new_shape),
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            grad: None,
            graph_id: None,
        })
    }
    
    /// Create a view of the tensor with new shape (shares data)
    pub fn view(&self, new_shape: &[usize]) -> Result<Tensor> {
        // For now, view is the same as reshape since we clone the data pointer
        // In a full implementation, view would share the underlying data
        self.reshape(new_shape)
    }
    
    /// Flatten tensor to 2D: [batch_size, -1]
    pub fn flatten(&self, start_dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if start_dim >= dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("start_dim {} out of range for tensor with {} dims", start_dim, dims.len())
            ));
        }
        
        let batch_size: usize = dims[..start_dim].iter().product();
        let feature_size: usize = dims[start_dim..].iter().product();
        
        self.reshape(&[batch_size, feature_size])
    }
    
    /// Squeeze: Remove dimensions of size 1
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Tensor> {
        let dims = self.shape.dims();
        
        let new_dims: Vec<usize> = if let Some(d) = dim {
            if d >= dims.len() {
                return Err(FlameError::InvalidOperation(
                    format!("Dimension {} out of range", d)
                ));
            }
            if dims[d] != 1 {
                return self.clone();
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
    
    /// Permute/transpose dimensions
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        let shape = self.shape.dims();
        if dims.len() != shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Permute dims {:?} doesn't match tensor dims {:?}", dims, shape)
            ));
        }
        
        // Check for valid permutation
        let mut seen = vec![false; dims.len()];
        for &d in dims {
            if d >= dims.len() {
                return Err(FlameError::InvalidOperation(
                    format!("Invalid permutation dimension: {}", d)
                ));
            }
            if seen[d] {
                return Err(FlameError::InvalidOperation(
                    format!("Duplicate dimension in permutation: {}", d)
                ));
            }
            seen[d] = true;
        }
        
        // For now, implement specific cases we need
        if shape.len() == 2 && dims == &[1, 0] {
            // Simple 2D transpose
            return self.transpose();
        } else if shape.len() == 4 && dims == &[0, 2, 1, 3] {
            // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
            return self.permute_0213();
        } else if shape.len() == 4 && dims == &[0, 1, 3, 2] {
            // [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
            return self.permute_0132();
        }
        
        // General permutation not yet implemented
        Err(FlameError::InvalidOperation(
            format!("General permutation {:?} not yet implemented", dims)
        ))
    }
    
    /// Specific permutation: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    fn permute_0213(&self) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                "permute_0213 requires 4D tensor".into()
            ));
        }
        
        let (d0, d1, d2, d3) = (dims[0], dims[1], dims[2], dims[3]);
        let data = self.to_vec()?;
        let mut result = vec![0.0f32; data.len()];
        
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                for i2 in 0..d2 {
                    for i3 in 0..d3 {
                        let src_idx = i0 * d1 * d2 * d3 + i1 * d2 * d3 + i2 * d3 + i3;
                        let dst_idx = i0 * d2 * d1 * d3 + i2 * d1 * d3 + i1 * d3 + i3;
                        result[dst_idx] = data[src_idx];
                    }
                }
            }
        }
        
        Tensor::from_vec(result, Shape::from_dims(&[d0, d2, d1, d3]), self.device.clone())
    }
    
    /// Specific permutation: [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
    fn permute_0132(&self) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                "permute_0132 requires 4D tensor".into()
            ));
        }
        
        let (d0, d1, d2, d3) = (dims[0], dims[1], dims[2], dims[3]);
        let data = self.to_vec()?;
        let mut result = vec![0.0f32; data.len()];
        
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                for i2 in 0..d2 {
                    for i3 in 0..d3 {
                        let src_idx = i0 * d1 * d2 * d3 + i1 * d2 * d3 + i2 * d3 + i3;
                        let dst_idx = i0 * d1 * d3 * d2 + i1 * d3 * d2 + i3 * d2 + i2;
                        result[dst_idx] = data[src_idx];
                    }
                }
            }
        }
        
        Tensor::from_vec(result, Shape::from_dims(&[d0, d1, d3, d2]), self.device.clone())
    }
    
    /// Add bias (broadcasting over batch dimensions)
    pub fn add_bias(&self, bias: &Tensor) -> Result<Tensor> {
        let shape = self.shape.dims();
        let bias_shape = bias.shape.dims();
        
        if bias_shape.len() != 1 || bias_shape[0] != shape[shape.len() - 1] {
            return Err(FlameError::InvalidOperation(
                format!("Bias shape {:?} incompatible with tensor shape {:?}", bias_shape, shape)
            ));
        }
        
        let data = self.to_vec()?;
        let bias_data = bias.to_vec()?;
        let mut result = vec![0.0f32; data.len()];
        
        let features = shape[shape.len() - 1];
        for i in 0..data.len() {
            let feature_idx = i % features;
            result[i] = data[i] + bias_data[feature_idx];
        }
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Flatten tensor from a given dimension
    pub fn flatten_from(&self, from_dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if from_dim >= dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("flatten_from: dimension {} out of range for tensor with {} dimensions", from_dim, dims.len())
            ));
        }
        
        // Calculate new shape
        let mut new_dims = dims[..from_dim].to_vec();
        let flattened_size: usize = dims[from_dim..].iter().product();
        new_dims.push(flattened_size);
        
        // Data remains the same, just reshape
        self.reshape(&new_dims)
    }
    
    /// Get a CUDA slice reference to the tensor data
    pub fn to_cuda_slice(&self) -> Result<&CudaSlice<f32>> {
        Ok(&self.data)
    }
    
    /// Get a mutable CUDA slice reference to the tensor data
    pub fn to_cuda_slice_mut(&mut self) -> Result<&mut CudaSlice<f32>> {
        Ok(&mut self.data)
    }
}