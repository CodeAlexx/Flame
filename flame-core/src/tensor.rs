use crate::{Shape, Result, FlameError};
use crate::autograd::{AutogradContext, Op};
use crate::gradient::{GradientMap, TensorGradExt};
use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::cublas::CudaBlas;
use std::sync::Arc;
use crate::cuda_ops::GpuOps;
use crate::cuda_kernels::CudaKernels;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global tensor ID counter
static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Unique tensor identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

impl TensorId {
    pub fn new() -> Self {
        TensorId(TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
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
    /// GPU memory (wrapped in Arc for cheap cloning)
    pub(crate) data: Arc<CudaSlice<f32>>,
    
    /// Shape of this tensor
    pub(crate) shape: Shape,
    
    /// Device this tensor lives on
    pub(crate) device: Arc<CudaDevice>,
    
    /// Unique identifier for gradient tracking
    pub(crate) id: TensorId,
    
    /// Whether gradients should be computed
    pub(crate) requires_grad: bool,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

// Note: We don't implement the standard Clone trait because we have a custom clone() method
// that returns Result<Tensor> for consistency with other operations

impl Tensor {
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let data = device.alloc_zeros::<f32>(size)
            .map_err(|_| FlameError::CudaDriver)?;
        Ok(Self { 
            data: Arc::new(data), 
            shape, 
            device,
            id: TensorId::new(),
            requires_grad: false,
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
            data: Arc::new(data), 
            shape, 
            device,
            id: TensorId::new(),
            requires_grad: false,
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
    
    /// Create tensor from slice
    pub fn from_slice(data: &[f32], shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        Self::from_vec(data.to_vec(), shape, device)
    }
    
    /// Compute gradients via automatic differentiation
    pub fn backward(&self) -> Result<GradientMap> {
        AutogradContext::backward(self)
    }
    
    /// Set data from slice (useful for initialization)
    /// This creates a new tensor with the provided data
    pub fn set_data(&self, data: &[f32]) -> Result<Tensor> {
        if data.len() != self.shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        // Create new tensor with the provided data
        Self::from_slice(data, self.shape.clone(), self.device.clone())
    }

    /// Functional weight update - returns new tensor
    pub fn update_weights(&self, gradient: &Tensor, lr: f32) -> Result<Tensor> {
        if self.shape != gradient.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: gradient.shape.clone(),
            });
        }

        // w = w - lr * grad (functional style)
        let grad_scaled = gradient.mul_scalar(lr)?;
        self.sub(&grad_scaled)
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
        
        // Allocate output data
        let mut output_data = unsafe { self.device.alloc::<f32>(m * n) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        unsafe {
            blas.gemm(cfg, &*other.data, &*self.data, &mut output_data)
                .map_err(|_| FlameError::CuBlas)?;
        }
        
        // Create output tensor
        let mut output = Tensor {
            data: Arc::new(output_data),
            shape: out_shape,
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::MatMul { 
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
            let self_batch = self_3d.slice_internal(self_offset, m * k1)?;
            let other_batch = other_3d.slice_internal(other_offset, k1 * n)?;
            
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
                    
                blas.gemm(cfg, &*other_batch.data, &*self_batch.data, &mut batch_output)
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
                
                unsafe {
                    use cudarc::driver::LaunchAsync;
                    f.launch(cfg, (
                        &*output.data,
                        &batch_output,
                        output_offset as i32,
                        (m * n) as i32,
                    ))?;
                }
            }
        }
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::BatchMatMul { 
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
    
    /// Create a slice view of the tensor (internal use)
    fn slice_internal(&self, start: usize, len: usize) -> Result<Tensor> {
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
        
        unsafe {
            use cudarc::driver::LaunchAsync;
            f.launch(cfg, (
                &slice_data,
                &*self.data,
                start as i32,
                len as i32,
            ))?;
        }
        
        Ok(Tensor {
            data: Arc::new(slice_data),
            shape: Shape::from_dims(&[len]),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        })
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated addition
        let mut output = GpuOps::add(self, other)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Add { 
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
        let mut output = self.add(&neg_other)?;
        
        // Record subtraction operation if needed
        if self.requires_grad || other.requires_grad {
            
            // Record as subtraction
            AutogradContext::record_op(
                output.id,
                Op::Sub { 
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
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated multiplication
        let mut output = GpuOps::mul(self, other)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Mul { 
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
    }

    /// Scalar multiplication 
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated scalar multiplication
        let mut output = GpuOps::mul_scalar(self, scalar)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::MulScalar { 
                    input: self.id,
                    scalar
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// Scale tensor by a scalar (alias for mul_scalar)
    pub fn scale(&self, scalar: f32) -> Result<Tensor> {
        self.mul_scalar(scalar)
    }
    
    /// Add scalar to all elements
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated scalar addition
        let mut output = GpuOps::add_scalar(self, scalar)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::AddScalar { 
                    input: self.id,
                    scalar
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated ReLU
        let mut output = GpuOps::relu(self)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::ReLU { 
                    input: self.id
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// GELU activation
    pub fn gelu(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated GELU
        let mut output = GpuOps::gelu(self)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::GELU { 
                    input: self.id
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// SiLU (Swish) activation
    pub fn silu(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated SiLU
        let mut output = GpuOps::silu(self)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::SiLU { 
                    input: self.id
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// Tanh activation
    pub fn tanh(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated Tanh
        let mut output = GpuOps::tanh(self)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Tanh { 
                    input: self.id
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// Sigmoid activation
    pub fn sigmoid(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated sigmoid
        let mut output = GpuOps::sigmoid(self)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Sigmoid { 
                    input: self.id
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// Error function (erf) - needed for GELU
    pub fn erf(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter()
            .map(|&x| {
                // Approximation of erf using a series expansion
                // This is good for |x| < 3.7
                let a1 =  0.254829592;
                let a2 = -0.284496736;
                let a3 =  1.421413741;
                let a4 = -1.453152027;
                let a5 =  1.061405429;
                let p  =  0.3275911;
                
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                let x = x.abs();
                
                let t = 1.0 / (1.0 + p * x);
                let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
                
                sign * y
            })
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Exponential function
    pub fn exp(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter().map(|&x| x.exp()).collect();
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    

    /// Square all elements
    pub fn square(&self) -> Result<Tensor> {
        let mut output = self.mul(self)?;
        
        // Record square operation if needed
        if self.requires_grad {
            // Record as square
            AutogradContext::record_op(
                output.id,
                Op::Square { 
                    input: self.id
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }

    /// Mean reduction
    pub fn mean(&self) -> Result<Tensor> {
        let sum = self.sum()?;
        let count = self.shape.elem_count() as f32;
        let mut output = sum.mul_scalar(1.0 / count)?;
        
        // Record mean operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Mean { 
                    input: self.id,
                    input_shape: self.shape.clone()
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }

    /// Sum reduction 
    pub fn sum(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated sum reduction
        let mut output = GpuOps::sum(self)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Sum { 
                    input: self.id,
                    input_shape: self.shape.clone()
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// Sum along specific dimensions
    pub fn sum_dims(&self, dims: &[usize]) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated sum reduction along dims
        let mut output = GpuOps::sum_dims(self, dims)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::SumDim { 
                    input: self.id,
                    dim: dims[0] // For now, handle single dim - TODO: extend for multiple dims
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// Sum along a specific dimension
    pub fn sum_dim(&self, dim: usize) -> Result<Tensor> {
        // For now, implement a simple version for the bias gradient case
        // This should sum along dimension 0 (batch dimension)
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of range for tensor with {} dimensions", dim, dims.len())
            ));
        }
        
        // For now, only implement dim=0 case which is needed for bias gradients
        if dim != 0 {
            return Err(FlameError::InvalidOperation(
                "sum_dim currently only supports dim=0".into()
            ));
        }
        
        // Sum along batch dimension
        let batch_size = dims[0];
        let remaining_dims: Vec<usize> = dims[1..].to_vec();
        let remaining_size: usize = remaining_dims.iter().product();
        
        let data = self.to_vec()?;
        let mut result = vec![0.0f32; remaining_size];
        
        for b in 0..batch_size {
            let offset = b * remaining_size;
            for i in 0..remaining_size {
                result[i] += data[offset + i];
            }
        }
        
        let mut output = Tensor::from_vec(result, Shape::from_dims(&remaining_dims), self.device.clone())?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::SumDim { 
                    input: self.id,
                    dim
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
    
    /// Get tensor ID for gradient tracking
    pub fn id(&self) -> TensorId {
        self.id
    }
    
    /// Check if this tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Copy to CPU
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.device.dtoh_sync_copy(&*self.data)
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
        let mut output = GpuOps::transpose(self)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Transpose { 
                    input: self.id
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
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
    
    /// Broadcast tensor to a new shape
    pub fn broadcast_to(&self, target_shape: &Shape) -> Result<Tensor> {
        let src_shape = self.shape.dims();
        let dst_shape = target_shape.dims();
        
        // Check if broadcast is valid
        if src_shape.len() > dst_shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Cannot broadcast from {:?} to {:?}: source has more dimensions", 
                    src_shape, dst_shape)
            ));
        }
        
        // Pad source shape with ones on the left
        let mut padded_src: Vec<usize> = vec![1; dst_shape.len() - src_shape.len()];
        padded_src.extend_from_slice(src_shape);
        
        // Check compatibility
        for (i, (&src_dim, &dst_dim)) in padded_src.iter().zip(dst_shape.iter()).enumerate() {
            if src_dim != dst_dim && src_dim != 1 {
                return Err(FlameError::InvalidOperation(
                    format!("Cannot broadcast dimension {} from {} to {}", i, src_dim, dst_dim)
                ));
            }
        }
        
        // If shapes are equal, just clone
        if &padded_src[..] == dst_shape {
            return self.clone();
        }
        
        // Use CUDA kernel for efficient broadcasting
        GpuOps::broadcast(self, target_shape)
    }
    
    /// Get raw data reference (for internal use by CUDA kernels)
    pub fn data(&self) -> &CudaSlice<f32> {
        &*self.data
    }
    
    /// Get the strides of this tensor
    /// Assumes C-contiguous (row-major) layout
    pub fn stride(&self) -> Vec<usize> {
        let dims = self.shape.dims();
        let ndim = dims.len();
        if ndim == 0 {
            return vec![];
        }
        
        let mut strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }
    
    
    /// Convert tensor to a Vec<f32> on CPU
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let cpu_data = self.device.dtoh_sync_copy(&*self.data)
            .map_err(|_| FlameError::CudaDriver)?;
        
        Ok(cpu_data)
    }
    
    /// Clone the tensor (creates a new tensor with copied data)
    pub fn clone(&self) -> Result<Tensor> {
        let mut data = unsafe { self.device.alloc::<f32>(self.shape.elem_count()) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        self.device.dtod_copy(&*self.data, &mut data)
            .map_err(|_| FlameError::CudaDriver)?;
        
        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
        })
    }
    
    /// Detach from computation graph
    pub fn detach(&self) -> Result<Tensor> {
        let mut data = unsafe { self.device.alloc::<f32>(self.shape.elem_count()) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        self.device.dtod_copy(&*self.data, &mut data)
            .map_err(|_| FlameError::CudaDriver)?;
            
        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        })
    }
    
    /// 2D Convolution
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor> {
        crate::cuda_conv2d::conv2d(self, weight, bias, stride, padding)
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
            id: TensorId::new(),
            requires_grad: self.requires_grad,
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
        
        let mut output = if shape.len() == 2 && dims == &[1, 0] {
            // Simple 2D transpose
            self.transpose()?
        } else if shape.len() == 4 && dims == &[0, 2, 1, 3] {
            // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
            self.permute_0213()?
        } else if shape.len() == 4 && dims == &[0, 1, 3, 2] {
            // [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
            self.permute_0132()?
        } else {
            // General permutation not yet implemented
            return Err(FlameError::InvalidOperation(
                format!("General permutation {:?} not yet implemented", dims)
            ));
        };
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Permute { 
                    input: self.id,
                    dims: dims.to_vec()
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
    }
    
    /// Compute softmax along a dimension
    pub fn softmax(&self, dim: isize) -> Result<Tensor> {
        let shape = self.shape().dims();
        let ndim = shape.len() as isize;
        
        // Handle negative dimension
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        
        if dim >= shape.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of bounds for tensor with {} dimensions", dim, shape.len())
            ));
        }
        
        // Compute max along dimension for numerical stability
        let max_vals = self.max_dim(dim, true)?;
        let shifted = self.sub(&max_vals)?;
        
        // Compute exp
        let exp_vals = shifted.exp()?;
        
        // Sum along dimension
        let sum_exp = exp_vals.sum_dim_keepdim(dim)?;
        
        // Divide by sum
        let mut output = exp_vals.div(&sum_exp)?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::Softmax { 
                    input: self.id,
                    dim: dim as isize
                },
                vec![(self.id, self.clone()?)]
            );
        }
        
        Ok(output)
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
        
        let mut output = Tensor::from_vec(result, Shape::from_dims(&[d0, d2, d1, d3]), self.device.clone())?;
        
        // Inherit gradient tracking
        if self.requires_grad {
            output.requires_grad = true;
        }
        
        Ok(output)
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
        
        let mut output = Tensor::from_vec(result, Shape::from_dims(&[d0, d1, d3, d2]), self.device.clone())?;
        
        // Inherit gradient tracking
        if self.requires_grad {
            output.requires_grad = true;
        }
        
        Ok(output)
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
        
        let mut output = Tensor::from_vec(result, self.shape.clone(), self.device.clone())?;
        
        // AUTOGRAD: Record operation if needed
        if self.requires_grad || bias.requires_grad {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id,
                Op::AddBias { 
                    input: self.id,
                    bias: bias.id
                },
                vec![
                    (self.id, self.clone()?),
                    (bias.id, bias.clone()?)
                ]
            );
        }
        
        Ok(output)
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
        Ok(&*self.data)
    }
    
    /// Transpose the last two dimensions (for batch operations)
    pub fn transpose_batch(&self) -> Result<Tensor> {
        let ndim = self.shape.dims().len();
        if ndim < 2 {
            return Err(FlameError::InvalidOperation(
                "Transpose batch requires at least 2 dimensions".into()
            ));
        }
        
        // Swap last two dimensions
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm[ndim - 2] = ndim - 1;
        perm[ndim - 1] = ndim - 2;
        
        self.permute(&perm)
    }
    
    /// Batch matrix multiplication
    pub fn batch_matmul(&self, other: &Tensor) -> Result<Tensor> {
        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();
        
        if self_dims.len() < 2 || other_dims.len() < 2 {
            return Err(FlameError::InvalidOperation(
                "Batch matmul requires at least 2D tensors".into()
            ));
        }
        
        // Check batch dimensions match
        if self_dims[..self_dims.len()-2] != other_dims[..other_dims.len()-2] {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        // Check matrix dimensions are compatible
        let m = self_dims[self_dims.len() - 2];
        let k1 = self_dims[self_dims.len() - 1];
        let k2 = other_dims[other_dims.len() - 2];
        let n = other_dims[other_dims.len() - 1];
        
        if k1 != k2 {
            return Err(FlameError::InvalidOperation(
                format!("Matrix dimensions incompatible for matmul: ({}, {}) @ ({}, {})", 
                    m, k1, k2, n)
            ));
        }
        
        // For now, implement using regular matmul on flattened batches
        // TODO: Optimize with batched GEMM
        let batch_size: usize = self_dims[..self_dims.len()-2].iter().product();
        
        // Reshape to [batch_size, m, k] and [batch_size, k, n]
        let self_3d = self.reshape(&[batch_size, m, k1])?;
        let other_3d = other.reshape(&[batch_size, k2, n])?;
        
        // Perform matmul for each batch element
        let mut results = Vec::new();
        for b in 0..batch_size {
            // Get slice for this batch
            let self_2d = self_3d.slice_1d(b * m * k1, (b + 1) * m * k1)?
                .reshape(&[m, k1])?;
            let other_2d = other_3d.slice_1d(b * k2 * n, (b + 1) * k2 * n)?
                .reshape(&[k2, n])?;
            
            let result = self_2d.matmul(&other_2d)?;
            results.push(result);
        }
        
        // Stack results
        let stacked = Self::stack(&results, 0)?;
        
        // Reshape back to original batch dimensions
        let mut output_shape = self_dims[..self_dims.len()-2].to_vec();
        output_shape.push(m);
        output_shape.push(n);
        
        stacked.reshape(&output_shape)
    }
    
    /// Stack tensors along a new dimension
    pub fn stack(tensors: &[Tensor], axis: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(FlameError::InvalidOperation("Cannot stack empty tensor list".into()));
        }
        
        // Verify all tensors have the same shape
        let first_shape = &tensors[0].shape;
        for t in &tensors[1..] {
            if t.shape != *first_shape {
                return Err(FlameError::ShapeMismatch {
                    expected: first_shape.clone(),
                    got: t.shape.clone(),
                });
            }
        }
        
        // Create new shape with added dimension
        let mut new_dims = first_shape.dims().to_vec();
        new_dims.insert(axis, tensors.len());
        
        // Concatenate data
        let mut data = Vec::new();
        for t in tensors {
            data.extend(t.to_vec()?);
        }
        
        Tensor::from_vec(data, Shape::from_dims(&new_dims), tensors[0].device.clone())
    }
    
    /// Slice tensor data (renamed to avoid conflict)
    pub fn slice_1d(&self, start: usize, end: usize) -> Result<Tensor> {
        if end > self.shape.elem_count() || start > end {
            return Err(FlameError::InvalidOperation(
                format!("Invalid slice range {}..{} for tensor with {} elements", 
                    start, end, self.shape.elem_count())
            ));
        }
        
        let data = self.to_vec()?;
        let slice_data = data[start..end].to_vec();
        
        // Calculate shape of slice
        let slice_len = end - start;
        Tensor::from_vec(slice_data, Shape::from_dims(&[slice_len]), self.device.clone())
    }
    
    /// Squeeze dimension (renamed to avoid conflict)
    pub fn squeeze_dim(&self, dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of range for tensor with {} dimensions", dim, dims.len())
            ));
        }
        
        if dims[dim] != 1 {
            return Err(FlameError::InvalidOperation(
                format!("Cannot squeeze dimension {} with size {}", dim, dims[dim])
            ));
        }
        
        let mut new_dims = dims.to_vec();
        new_dims.remove(dim);
        
        self.reshape(&new_dims)
    }
    
}

/// Implementation of gradient access trait
impl TensorGradExt for Tensor {
    fn grad<'a>(&self, gradients: &'a GradientMap) -> Option<&'a Tensor> {
        gradients.get(self.id)
    }
    
    fn grad_mut<'a>(&self, gradients: &'a mut GradientMap) -> Option<&'a mut Tensor> {
        gradients.get_mut(self.id)
    }
    
    fn take_grad(&self, gradients: &mut GradientMap) -> Option<Tensor> {
        gradients.take(self.id)
    }
    
    fn has_grad(&self, gradients: &GradientMap) -> bool {
        gradients.contains(self.id)
    }
}