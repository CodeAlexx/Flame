use crate::{Shape, DType, Result, FlameError};
use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::cublas::CudaBlas;
use std::sync::Arc;

// Helper function for allocating and copying to GPU via memory pool
fn alloc_and_copy_to_pool<T: AsRef<[f32]>>(device: &Arc<CudaDevice>, data: T) -> Result<CudaSlice<f32>> {
    let slice = data.as_ref();
    let mut cuda_data = crate::tensor::alloc_from_pool(device, slice.len())?;
    device.htod_copy_into(slice, &mut cuda_data).map_err(|_| FlameError::CudaDriver)?;
    Ok(cuda_data)
}


/// The core tensor type - owns CUDA memory and supports mutable operations
pub struct CudaTensor {
    pub(crate) data: CudaSlice<f32>,
    pub(crate) shape: Shape,
    pub(crate) device: Arc<CudaDevice>,
}

impl CudaTensor {
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let data = crate::tensor::alloc_zeros_from_pool(&device, size)
            .map_err(|_| FlameError::CudaDriver)?;
        Ok(Self { data, shape, device })
    }

    /// Create a new tensor from a Vec
    pub fn from_vec(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(FlameError::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        let data = alloc_and_copy_to_pool(&device, &data)
            .map_err(|_| FlameError::CudaDriver)?;
        Ok(Self { data, shape, device })
    }

    /// Create random tensor (using cuRAND later, CPU random for now)
    pub fn randn(shape: Shape, mean: f32, std: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        use rand::Rng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std)
            .map_err(|e| FlameError::InvalidOperation(format!("invalid normal params: {e}")))?;
        
        let cpu_data: Vec<f32> = (0..size)
            .map(|_| normal.sample(&mut rng))
            .collect();
            
        Self::from_vec(cpu_data, shape, device)
    }

    /// THE KEY OPERATION - Mutable weight update
    pub fn update_weights(&mut self, gradient: &CudaTensor, lr: f32) -> Result<()> {
        if self.shape != gradient.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: gradient.shape.clone(),
            });
        }

        // For now, use cuBLAS axpy: self = self + (-lr) * gradient
        // Later we'll write a custom kernel
        let blas = CudaBlas::new(self.device.clone())
            .map_err(|_| FlameError::CuBlas)?;
        unsafe {
            // y = alpha * x + y where y is self.data(), x is gradient.data()
            cudarc::cublas::sys::cublasSaxpy_v2(
                blas.handle().cast(),
                self.shape.elem_count() as i32,
                &(-lr) as *const f32,
                gradient.data().as_ptr(),
                1,
                self.data().as_mut_ptr(),
                1,
            );
        }
        Ok(())
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &CudaTensor) -> Result<CudaTensor> {
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
        let mut output = CudaTensor::zeros(out_shape, self.device.clone())?;
        
        // Use cuBLAS for matrix multiplication
        let blas = CudaBlas::new(self.device.clone())
            .map_err(|_| FlameError::CuBlas)?;
        unsafe {
            blas.gemm(
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &1.0f32,
                other.data().as_ptr(),
                n as i32,
                self.data().as_ptr(),
                k as i32,
                &0.0f32,
                output.data().as_mut_ptr(),
                n as i32,
            ).map_err(|_| FlameError::CuBlas)?;
        }
        
        Ok(output)
    }

    /// Add another tensor (creates new tensor for now)
    pub fn add(&self, other: &CudaTensor) -> Result<CudaTensor> {
        let shape = self.shape.broadcast_shape_binary_op(&other.shape)?;
        let mut output = CudaTensor::zeros(shape.clone(), self.device.clone())?;
        
        // For now, simple element-wise add (no broadcasting yet)
        if self.shape == other.shape {
            let size = self.shape.elem_count();
            let blas = CudaBlas::new(self.device.clone())
                .map_err(|_| FlameError::CuBlas)?;
            
            // Copy self to output
            self.device.dtod_copy(&self.data(), &mut output.data())
                .map_err(|_| FlameError::CudaDriver)?;
            
            // Add other to output
            unsafe {
                cudarc::cublas::sys::cublasSaxpy_v2(
                    blas.handle().cast(),
                    size as i32,
                    &1.0f32 as *const f32,
                    other.data().as_ptr(),
                    1,
                    output.data().as_mut_ptr(),
                    1,
                );
            }
        } else {
            return Err(FlameError::InvalidOperation("Broadcasting not implemented yet".into()));
        }
        
        Ok(output)
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Copy to CPU for inspection
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.device.dtoh_sync_copy(&self.data())
            .map_err(|_| FlameError::CudaDriver)?)
    }

    /// Get a single value (for loss printing)
    pub fn item(&self) -> Result<f32> {
        if self.shape.elem_count() != 1 {
            return Err(FlameError::InvalidOperation("item() requires tensor with single element".into()));
        }
        Ok(self.to_vec()?[0])
    }
}

// Later we'll add:
// - Custom CUDA kernels for weight update
// - ReLU, other activations
// - Gradients storage
// - Autograd
