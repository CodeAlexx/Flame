use crate::{Shape, Result, FlameError, DType};
use crate::autograd::{AutogradContext, Op};
use crate::gradient::{GradientMap, TensorGradExt};
use crate::tensor_storage::TensorStorage;
use crate::cuda_memory_alignment::alloc_aligned_f32;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchConfig, LaunchAsync, DeviceSlice, CudaFunction};
use cudarc::cublas::CudaBlas;
use std::sync::Arc;
use crate::cuda_ops::GpuOps;
use crate::cuda_kernels::CudaKernels;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::fmt;

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

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
#[derive(Clone)]
pub struct Tensor {
    /// GPU memory storage with dtype support
    pub(crate) storage: TensorStorage,
    
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

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {{ shape: {:?}, dtype: {:?}, device: cuda:{}, id: {} }}", 
               self.shape, self.storage.dtype(), self.device.ordinal(), self.id.0)
    }
}

// Note: We don't implement the standard Clone trait because we have a custom clone() method
// that returns Result<Tensor> for consistency with other operations

impl Tensor {
    /// Create a causal mask tensor
    pub fn causal_mask(seq_len: usize, device: &Arc<CudaDevice>) -> Result<Self> {
        // Create a lower triangular mask
        let mut mask_data = vec![0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }
        
        // Convert to tensor
        let shape = Shape::from_dims(&[seq_len, seq_len]);
        Self::from_vec(mask_data, shape, device.clone())
    }
    
    /// Apply a mask to tensor (set masked positions to value)
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Self> {
        if self.shape != mask.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: mask.shape.clone(),
            });
        }
        
        // Use CUDA kernel for masked fill
        let kernel_code = r#"
extern "C" __global__ void masked_fill_kernel(
    const float* input,
    const float* mask,
    float* output,
    float value,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = mask[idx] > 0.5f ? value : input[idx];
    }
}
"#;
        
        crate::cuda_kernels::CudaKernels::ensure_kernel(&self.device, "masked_fill_kernel", kernel_code)?;
        
        let f = self.device.get_func("masked_fill_kernel", "masked_fill_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get masked_fill_kernel".into()))?;
        
        let n = self.shape.elem_count();
        let output_data = alloc_zeros_from_pool(&self.device, n)?;
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
        
        crate::launch_kernel!(f, cfg,
            self.storage.as_slice(),
            mask.storage.as_slice(),
            &output_data,
            value,
            n as i32
        );
        
        Ok(Self {
            storage: TensorStorage::F32 { data: output_data, numel: n },
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        })
    }
    
    /// Create a new tensor filled with zeros (defaults to F32)
    pub fn zeros(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        Self::zeros_dtype(shape, DType::F32, device)
    }
    
    /// Create tensor with specific dtype
    pub fn zeros_dtype(shape: Shape, dtype: DType, device: Arc<CudaDevice>) -> Result<Self> {
        let storage = TensorStorage::zeros(&shape, dtype, &device)?;
        Ok(Self {
            storage,
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
        })
    }
    
    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }
    
    /// Get data as F32 (converting if necessary)
    /// This is for backward compatibility - prefer using storage directly
    pub fn data(&self) -> Result<Arc<CudaSlice<f32>>> {
        match &self.storage {
            TensorStorage::F32 { data, .. } => Ok(Arc::new(data.clone())),
            _ => {
                let f32_data = self.storage.to_f32(&self.device)?;
                Ok(Arc::new(f32_data))
            }
        }
    }
    
    /// Get raw CUDA pointer for cuDNN operations (read-only)
    pub fn cuda_ptr(&self) -> *const f32 {
        use cudarc::driver::DevicePtr;
        use std::ffi::c_void;
        let ptr_addr = *self.storage.as_slice().device_ptr();
        // Cast u64 GPU address to pointer
        ptr_addr as *const c_void as *const f32
    }
    
    /// Get mutable raw CUDA pointer for cuDNN operations
    pub fn cuda_ptr_mut(&mut self) -> *mut f32 {
        use cudarc::driver::DevicePtr;
        use std::ffi::c_void;
        // We need to get a mutable reference to the storage
        // This is safe because we're the only owner of this tensor
        let slice = match &mut self.storage {
            TensorStorage::F32 { data, .. } |
            TensorStorage::F16 { data, .. } |
            TensorStorage::BF16 { data, .. } => data,
            TensorStorage::I8 { .. } => panic!("Cannot get f32 pointer from I8 storage"),
        };
        let ptr_addr = *slice.device_ptr();
        // Cast u64 GPU address to pointer
        ptr_addr as *mut c_void as *mut f32
    }
    
    /// Cast to different dtype
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if self.dtype() == dtype {
            return self.clone();
        }
        
        // For now, we store everything as F32 but track the dtype
        // Use aligned allocation for the new tensor to avoid CUDA issues
        let numel = self.shape.elem_count();
        
        // Debug large allocations
        if numel > 100000 {
            eprintln!("to_dtype: converting {} elements from {:?} to {:?}", numel, self.dtype(), dtype);
        }
        
        // Use aligned allocation
        let mut f32_data = alloc_aligned_f32(&self.device, numel)?;
        
        // Copy data from source storage
        match &self.storage {
            TensorStorage::F32 { data, .. } |
            TensorStorage::F16 { data, .. } |
            TensorStorage::BF16 { data, .. } => {
                // For now, just use the allocation as-is
                // The extra padding shouldn't cause issues since we track numel separately
                self.device.dtod_copy(data, &mut f32_data)?;
            },
            TensorStorage::I8 { .. } => {
                return Err(FlameError::InvalidOperation("I8 to dtype conversion not implemented".into()));
            }
        }
        
        let storage = match dtype {
            DType::F32 => TensorStorage::F32 { data: f32_data, numel },
            DType::F16 => TensorStorage::F16 { data: f32_data, numel, scale: 1.0 }, // Stored as F32 but marked as F16
            DType::BF16 => TensorStorage::BF16 { data: f32_data, numel }, // Stored as F32 but marked as BF16
            _ => return Err(FlameError::InvalidOperation(
                format!("Unsupported dtype: {:?}", dtype)
            )),
        };
        
        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        })
    }
    
    /// Create a new tensor filled with ones (defaults to F32)
    pub fn ones(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let ones_vec = vec![1.0f32; size];
        Self::from_vec(ones_vec, shape, device)
    }
    
    /// Create a new tensor filled with ones with specific dtype
    pub fn ones_dtype(shape: Shape, dtype: DType, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let ones_vec = vec![1.0f32; size];
        Self::from_vec_dtype(ones_vec, shape, device, dtype)
    }

    /// Create a new tensor from a Vec
    pub fn from_vec(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        // Allocate from memory pool
        let numel = data.len();
        let mut cuda_data = alloc_from_pool(&device, numel)?;
        
        // If the allocated size is larger than our data, we need to handle it carefully
        if cuda_data.len() > numel {
            // Pad the data to match the allocated size
            let mut padded_data = data;
            padded_data.resize(cuda_data.len(), 0.0);
            device.htod_copy_into(padded_data, &mut cuda_data)
                .map_err(|_| FlameError::CudaDriver)?;
        } else {
            // Normal case - sizes match
            device.htod_copy_into(data, &mut cuda_data)
                .map_err(|_| FlameError::CudaDriver)?;
        }
        Ok(Self { 
            storage: TensorStorage::F32 { data: cuda_data, numel }, 
            shape, 
            device,
            id: TensorId::new(),
            requires_grad: false,
        })
    }
    
    /// Create a new tensor from a Vec with specific dtype
    pub fn from_vec_dtype(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>, dtype: DType) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        // Allocate from memory pool
        let numel = data.len();
        let mut cuda_data = alloc_from_pool(&device, numel)?;
        
        // If the allocated size is larger than our data, we need to handle it carefully
        if cuda_data.len() > numel {
            // Pad the data to match the allocated size
            let mut padded_data = data;
            padded_data.resize(cuda_data.len(), 0.0);
            device.htod_copy_into(padded_data, &mut cuda_data)
                .map_err(|_| FlameError::CudaDriver)?;
        } else {
            // Normal case - sizes match
            device.htod_copy_into(data, &mut cuda_data)
                .map_err(|_| FlameError::CudaDriver)?;
        }
        
        // Create storage with specified dtype
        let storage = match dtype {
            DType::F32 => TensorStorage::F32 { data: cuda_data, numel },
            DType::F16 => TensorStorage::F16 { data: cuda_data, numel, scale: 1.0 },
            DType::BF16 => {
                // BF16 stored as F32 for now (cudarc limitation)
                TensorStorage::BF16 { data: cuda_data, numel }
            }
            _ => return Err(FlameError::InvalidOperation(
                format!("Unsupported dtype for from_vec_dtype: {:?}", dtype)
            )),
        };
        
        Ok(Self { 
            storage, 
            shape, 
            device,
            id: TensorId::new(),
            requires_grad: false,
        })
    }

    /// Create tensor from raw GPU data
    pub fn from_raw(
        data: Arc<CudaSlice<f32>>, 
        shape: Shape, 
        device: Arc<CudaDevice>,
        requires_grad: bool
    ) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        Ok(Self { 
            storage: TensorStorage::F32 { data: (*data).clone(), numel: shape.elem_count() }, 
            shape, 
            device,
            id: TensorId::new(),
            requires_grad,
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
    
    /// Create a tensor with random values like another tensor
    pub fn rand_like(tensor: &Tensor) -> Result<Self> {
        Self::randn(tensor.shape.clone(), 0.0, 1.0, tensor.device.clone())
    }
    
    /// Create a BF16 tensor from F32 data (stored as F32 internally)
    pub fn from_bf16_slice(data: CudaSlice<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let numel = shape.elem_count();
        if data.len() != numel {
            return Err(FlameError::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        
        Ok(Self {
            storage: TensorStorage::BF16 { data, numel },
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
        })
    }
    
    /// Create a BF16 tensor from F32 data
    pub fn from_f32_to_bf16(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        use crate::bf16_support::BF16Ops;
        BF16Ops::from_f32(data, shape, device)
    }
    
    /// Get F32 slice for BF16 tensor (stored as F32 internally)
    pub fn as_bf16_slice(&self) -> Result<&CudaSlice<f32>> {
        if self.dtype() != DType::BF16 {
            return Err(FlameError::InvalidOperation("Not a BF16 tensor".to_string()));
        }
        Ok(self.storage.as_slice())
    }
    
    /// Convert tensor to BF16
    pub fn to_bf16(&self) -> Result<Self> {
        if self.dtype() == DType::BF16 {
            return Ok(self.clone()?);
        }
        
        // Convert to F32 first if needed
        let f32_data = self.to_vec()?;
        Self::from_f32_to_bf16(f32_data, self.shape.clone(), self.device.clone())
    }
    
    /// Create a BF16 tensor from CUDA slice (stored as F32)
    pub fn from_cuda_bf16(data: CudaSlice<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        Self::from_bf16_slice(data, shape, device)
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

    pub fn from_slice_dtype(data: &[f32], shape: Shape, device: Arc<CudaDevice>, dtype: DType) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        Self::from_vec_dtype(data.to_vec(), shape, device, dtype)
    }
    
    /// Compute gradients via automatic differentiation
    pub fn backward(&self) -> Result<GradientMap> {
        AutogradContext::backward(self)
    }
    
    /// Compute gradients with debug information
    pub fn backward_debug(&self) -> Result<GradientMap> {
        AutogradContext::backward_debug(self)
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
        
        // Allocate output data using aligned allocation
        let mut output_data = alloc_aligned_f32(&self.device, m * n)?;
        
        let self_data = self.storage.as_slice();
        let other_data = other.storage.as_slice();
        
        unsafe {
            blas.gemm(cfg, &*other_data, &*self_data, &mut output_data)
                .map_err(|_| FlameError::CuBlas)?;
        }
        
        // Create output tensor
        let mut output = Tensor {
            storage: TensorStorage::F32 { data: output_data, numel: m * n },
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
        
        // Support 3D and 4D tensors
        if self_shape.len() == 3 && other_shape.len() == 3 {
            let (batch, m, k1) = (self_shape[0], self_shape[1], self_shape[2]);
            let (batch2, k2, n) = (other_shape[0], other_shape[1], other_shape[2]);
            
            if batch != batch2 {
                return Err(FlameError::InvalidOperation(
                    format!("bmm: batch size mismatch {} vs {}", batch, batch2)
                ));
            }
            
            if k1 != k2 {
                return Err(FlameError::InvalidOperation(
                    format!("bmm: incompatible matrix dimensions {} vs {}", k1, k2)
                ));
            }
            
            return self.bmm_3d(batch, m, k1, n, other);
        } else if self_shape.len() == 4 && other_shape.len() == 4 {
            // 4D case: [batch, heads, seq, dim] @ [batch, heads, dim, seq2]
            let (batch, heads, m, k1) = (self_shape[0], self_shape[1], self_shape[2], self_shape[3]);
            let (batch2, heads2, k2, n) = (other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
            
            if batch != batch2 || heads != heads2 {
                return Err(FlameError::InvalidOperation(
                    format!("bmm: batch/heads mismatch: [{}, {}] vs [{}, {}]", batch, heads, batch2, heads2)
                ));
            }
            
            if k1 != k2 {
                return Err(FlameError::InvalidOperation(
                    format!("bmm: incompatible matrix dimensions {} vs {}", k1, k2)
                ));
            }
            
            // Reshape to 3D for computation
            let total_batch = batch * heads;
            let self_3d = self.reshape(&[total_batch, m, k1])?;
            let other_3d = other.reshape(&[total_batch, k2, n])?;
            
            // Do 3D BMM
            let result_3d = self_3d.bmm_3d(total_batch, m, k1, n, &other_3d)?;
            
            // Reshape back to 4D
            return result_3d.reshape(&[batch, heads, m, n]);
        } else {
            return Err(FlameError::InvalidOperation(
                format!("bmm: unsupported tensor shapes {:?} @ {:?}", self_shape, other_shape)
            ));
        }
        
    }
    
    /// Helper for 3D batch matrix multiplication
    fn bmm_3d(&self, batch: usize, m: usize, k: usize, n: usize, other: &Tensor) -> Result<Tensor> {
        // Prepare output shape [batch, m, n]
        let output_shape = vec![batch, m, n];
        let mut output = Tensor::zeros(Shape::from_dims(&output_shape), self.device.clone())?;
        
        // Use cuBLAS
        let blas = CudaBlas::new(self.device.clone())
            .map_err(|_| FlameError::CuBlas)?;
        
        // Simple loop-based implementation for now
        for b in 0..batch {
            let self_offset = b * m * k;
            let other_offset = b * k * n;
            let output_offset = b * m * n;
            
            // Create temporary tensors for this batch's slice
            // This is inefficient but works for now
            let self_data = self.to_vec()?;
            let other_data = other.to_vec()?;
            
            let mut self_batch_data = vec![0.0f32; m * k];
            let mut other_batch_data = vec![0.0f32; k * n];
            
            for i in 0..(m * k) {
                self_batch_data[i] = self_data[self_offset + i];
            }
            for i in 0..(k * n) {
                other_batch_data[i] = other_data[other_offset + i];
            }
            
            let self_batch = Tensor::from_vec(
                self_batch_data,
                Shape::from_dims(&[m, k]),
                self.device.clone()
            )?;
            
            let other_batch = Tensor::from_vec(
                other_batch_data,
                Shape::from_dims(&[k, n]),
                self.device.clone()
            )?;
            
            // Use regular matmul for this batch
            let batch_result = self_batch.matmul(&other_batch)?;
            
            // Copy result back to output
            let batch_result_data = batch_result.to_vec()?;
            let mut output_data = output.to_vec()?;
            
            for i in 0..(m * n) {
                output_data[output_offset + i] = batch_result_data[i];
            }
            
            output = Tensor::from_vec(output_data, output.shape.clone(), output.device.clone())?;
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
        // Check if we can broadcast
        let self_batch_prod: usize = self_batch.iter().product();
        let target_batch_prod: usize = target_batch.iter().product();
        
        if self_batch_prod == 1 {
            // Broadcast single batch to target batch size
            let expanded = self.broadcast_to(&Shape::from_dims(&[target_batch_prod, m, n]))?;
            Ok(expanded)
        } else if target_batch_prod % self_batch_prod == 0 {
            // Can broadcast if target is multiple of self
            let repeat_factor = target_batch_prod / self_batch_prod;
            let self_flat = self.reshape(&[self_batch_prod, m, n])?;
            
            // Repeat self_flat to match target batch size
            let mut repeated_data = Vec::new();
            for _ in 0..repeat_factor {
                let self_data = self_flat.to_vec()?;
                repeated_data.extend_from_slice(&self_data);
            }
            
            Tensor::from_vec(
                repeated_data,
                Shape::from_dims(&[target_batch_prod, m, n]),
                self.device.clone()
            )
        } else {
            Err(FlameError::InvalidOperation(
                format!("Cannot broadcast batch dimensions {:?} to {:?}", self_batch, target_batch)
            ))
        }
    }
    
    /// Create a slice view of the tensor (internal use)
    fn slice_internal(&self, start: usize, len: usize) -> Result<Tensor> {
        // Slice implementation for contiguous memory
        // Non-contiguous tensors would require stride handling
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
        
        let slice_data = alloc_zeros_from_pool(&self.device, len)?;
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(len as u32);
        
        launch_kernel!(f, cfg,
            &slice_data,
            self.storage.as_slice(),
            start as i32,
            len as i32
        )?;
        
        Ok(Tensor {
            storage: TensorStorage::F32 { data: slice_data, numel: len },
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
        // Implement as self + (-1 * other)
        // This automatically handles broadcasting through the add operation
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
        // Use GpuOps directly to avoid recording during backward
        let mut output = GpuOps::mul(self, self)?;
        
        // Set requires_grad if input requires grad
        if self.requires_grad {
            output.requires_grad = true;
            
            // Record as square operation
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
                    dim: dims[0] // Currently handling single dimension, multi-dim reduction handled iteratively
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
    
    /// Reshape tensor to new shape
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let new_shape = Shape::from_dims(shape);
        if self.shape.elem_count() != new_shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: new_shape,
                got: self.shape.clone(),
            });
        }
        
        Ok(Tensor {
            id: TensorId::new(),
            storage: self.storage.clone(),
            shape: new_shape,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
        })
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
        Ok(self.device.dtoh_sync_copy(self.storage.as_slice())
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
        let cpu_data = self.device.dtoh_sync_copy(self.storage.as_slice())
            .map_err(|_| FlameError::CudaDriver)?;
        
        Ok(cpu_data)
    }
    
    /// Clone the tensor (creates a new tensor with copied data)
    pub fn clone(&self) -> Result<Tensor> {
        // Commented out for performance
        // println!("CLONE DEBUG: Cloning tensor with shape {:?}, dtype {:?}, elem_count: {}", 
        //          self.shape, self.dtype(), self.shape.elem_count());
        
        // Clone the storage while preserving dtype
        let storage = match &self.storage {
            TensorStorage::F32 { data, numel } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device.dtod_copy(data, &mut new_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                TensorStorage::F32 { data: new_data, numel: *numel }
            }
            TensorStorage::F16 { data, numel, scale } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device.dtod_copy(data, &mut new_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                TensorStorage::F16 { data: new_data, numel: *numel, scale: *scale }
            }
            TensorStorage::BF16 { data, numel } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device.dtod_copy(data, &mut new_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                TensorStorage::BF16 { data: new_data, numel: *numel }
            }
            TensorStorage::I8 { data, numel } => {
                let mut new_data = unsafe { self.device.alloc::<i8>(*numel) }
                    .map_err(|_| FlameError::CudaDriver)?;
                self.device.dtod_copy(data, &mut new_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                TensorStorage::I8 { data: new_data, numel: *numel }
            }
        };
        
        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
        })
    }
    
    /// Detach from computation graph
    pub fn detach(&self) -> Result<Tensor> {
        // Clone the storage while preserving dtype
        let storage = match &self.storage {
            TensorStorage::F32 { data, numel } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device.dtod_copy(data, &mut new_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                TensorStorage::F32 { data: new_data, numel: *numel }
            }
            TensorStorage::F16 { data, numel, scale } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device.dtod_copy(data, &mut new_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                TensorStorage::F16 { data: new_data, numel: *numel, scale: *scale }
            }
            TensorStorage::BF16 { data, numel } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device.dtod_copy(data, &mut new_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                TensorStorage::BF16 { data: new_data, numel: *numel }
            }
            TensorStorage::I8 { data, numel } => {
                let mut new_data = unsafe { self.device.alloc::<i8>(*numel) }
                    .map_err(|_| FlameError::CudaDriver)?;
                self.device.dtod_copy(data, &mut new_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                TensorStorage::I8 { data: new_data, numel: *numel }
            }
        };
            
        Ok(Tensor {
            storage,
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
        let max_vals = GpuOps::max_dim(self, dim, true)?;
        let shifted = self.sub(&max_vals)?;
        
        // Compute exp
        let exp_vals = shifted.exp()?;
        
        // Sum along dimension
        let sum_exp = GpuOps::sum_dim_keepdim(&exp_vals, dim)?;
        
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
        Ok(self.storage.as_slice())
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
        
        // Implement using regular matmul on flattened batches
        // Future optimization: Use cuBLAS batched GEMM when available
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
    
    /// Narrow (slice) a tensor along a dimension - CUDA only implementation
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor> {
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
        
        // Create output shape
        let mut output_dims = dims.to_vec();
        output_dims[dim] = length;
        let output_shape = Shape::from_dims(&output_dims);
        
        // For now, only support up to 4D tensors
        if dims.len() > 4 {
            return Err(FlameError::InvalidOperation(
                format!("narrow operation currently supports up to 4D tensors, got {}D", dims.len())
            ));
        }
        
        // Pad dimensions to 4D for kernel
        let mut input_size = [1, 1, 1, 1];
        let mut output_size = [1, 1, 1, 1];
        for (i, &d) in dims.iter().enumerate() {
            input_size[i] = d;
        }
        for (i, &d) in output_dims.iter().enumerate() {
            output_size[i] = d;
        }
        
        // Get CUDA kernels instance
        use crate::cuda_kernels::CudaKernels;
        let cuda_kernels = CudaKernels::new(self.device.clone())?;
        
        // Allocate output tensor
        let mut output = Tensor::zeros(output_shape, self.device.clone())?;
        
        // Get the narrow kernel
        let kernel = cuda_kernels.kernels.get("narrow_kernel")
            .ok_or_else(|| FlameError::KernelError("narrow_kernel not found".into()))?
            .clone();
        
        // Launch kernel
        let n = output.shape.elem_count();
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (
                self.storage.as_slice(),
                output.storage.as_slice(),
                input_size[0] as i32, input_size[1] as i32, input_size[2] as i32, input_size[3] as i32,
                output_size[0] as i32, output_size[1] as i32, output_size[2] as i32, output_size[3] as i32,
                dim as i32, start as i32
            ))?;
        }
        
        self.device.synchronize()?;
        
        Ok(output)
    }
    
    /// Copy data from another tensor
    pub fn copy_(&mut self, other: &Tensor) -> Result<()> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        self.storage = other.storage.clone();
        Ok(())
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

/// Allocate memory from the pool for internal use
pub(crate) fn alloc_from_pool(device: &Arc<CudaDevice>, size: usize) -> Result<CudaSlice<f32>> {
    // Use aligned allocation to avoid CUDA alignment issues
    alloc_aligned_f32(device, size)
}

/// Allocate zeroed memory from the pool for internal use
pub(crate) fn alloc_zeros_from_pool(device: &Arc<CudaDevice>, size: usize) -> Result<CudaSlice<f32>> {
    let mut data = alloc_from_pool(device, size)?;
    device.memset_zeros(&mut data)?;
    Ok(data)
}

/// Drop implementation to return memory to pool
impl Drop for Tensor {
    fn drop(&mut self) {
        // Return memory to the pool when tensor is dropped
        if let Ok(pool) = crate::memory_pool::MEMORY_POOL.get_pool(&self.device) {
            if let Ok(mut pool_guard) = pool.lock() {
                match &self.storage {
                    TensorStorage::F32 { data, .. } => pool_guard.deallocate(data.clone()),
                    TensorStorage::F16 { data, .. } => pool_guard.deallocate(data.clone()),
                    TensorStorage::BF16 { data, .. } => pool_guard.deallocate(data.clone()),
                    TensorStorage::I8 { .. } => {
                        // I8 storage is not pooled (for now)
                        // It will be cleaned up automatically by cudarc
                    }
                }
            }
        }
    }
}