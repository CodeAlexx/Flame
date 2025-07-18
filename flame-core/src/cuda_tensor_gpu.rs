use crate::{Shape, Result, FlameError};
use crate::cuda::CudaKernels;
use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::cublas::CudaBlas;
use std::sync::Arc;

/// The core tensor type with GPU-accelerated operations
pub struct CudaTensor {
    pub(crate) data: CudaSlice<f32>,
    pub(crate) shape: Shape,
    pub(crate) device: Arc<CudaDevice>,
}

impl CudaTensor {
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let data = device.alloc_zeros::<f32>(size)
            .map_err(|_| FlameError::CudaDriver)?;
        Ok(Self { data, shape, device })
    }

    /// Create a new tensor filled with ones
    pub fn ones(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let mut data = device.alloc::<f32>(size)
            .map_err(|_| FlameError::CudaDriver)?;
        CudaKernels::fill(&device, &mut data, 1.0)?;
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
        let data = device.htod_sync_copy(&data)
            .map_err(|_| FlameError::CudaDriver)?;
        Ok(Self { data, shape, device })
    }

    /// Create random tensor (CPU random for now, will add cuRAND later)
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

    /// THE KEY OPERATION - GPU-accelerated weight update
    pub fn update_weights(&mut self, gradient: &CudaTensor, lr: f32) -> Result<()> {
        if self.shape != gradient.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: gradient.shape.clone(),
            });
        }

        CudaKernels::update_weights(&self.device, &mut self.data, &gradient.data, lr)?;
        Ok(())
    }

    /// Matrix multiplication using cuBLAS
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
                other.data.as_ptr(),
                n as i32,
                self.data.as_ptr(),
                k as i32,
                &0.0f32,
                output.data.as_mut_ptr(),
                n as i32,
            ).map_err(|_| FlameError::CuBlas)?;
        }
        
        Ok(output)
    }

    /// Element-wise addition
    pub fn add(&self, other: &CudaTensor) -> Result<CudaTensor> {
        let shape = self.shape.broadcast_shape_binary_op(&other.shape)?;
        let mut output = CudaTensor::zeros(shape.clone(), self.device.clone())?;
        
        // For now, simple element-wise add (no broadcasting yet)
        if self.shape == other.shape {
            CudaKernels::add(&self.device, &self.data, &other.data, &mut output.data)?;
        } else {
            return Err(FlameError::InvalidOperation("Broadcasting not implemented yet".into()));
        }
        
        Ok(output)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &CudaTensor) -> Result<CudaTensor> {
        if self.shape != other.shape {
            return Err(FlameError::InvalidOperation("Shapes must match for multiplication".into()));
        }
        
        let mut output = CudaTensor::zeros(self.shape.clone(), self.device.clone())?;
        CudaKernels::mul(&self.device, &self.data, &other.data, &mut output.data)?;
        Ok(output)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Result<CudaTensor> {
        let mut output = CudaTensor::zeros(self.shape.clone(), self.device.clone())?;
        CudaKernels::mul_scalar(&self.device, &self.data, scalar, &mut output.data)?;
        Ok(output)
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<CudaTensor> {
        let mut output = CudaTensor::zeros(self.shape.clone(), self.device.clone())?;
        CudaKernels::relu(&self.device, &self.data, &mut output.data)?;
        Ok(output)
    }

    /// Subtract another tensor
    pub fn sub(&self, other: &CudaTensor) -> Result<CudaTensor> {
        // a - b = a + (-1 * b)
        let neg_other = other.mul_scalar(-1.0)?;
        self.add(&neg_other)
    }

    /// Square all elements
    pub fn square(&self) -> Result<CudaTensor> {
        self.mul(self)
    }

    /// Mean reduction
    pub fn mean(&self) -> Result<CudaTensor> {
        // For now, simple implementation - sum and divide
        let sum = self.sum()?;
        let count = self.shape.elem_count() as f32;
        sum.mul_scalar(1.0 / count)
    }

    /// Sum reduction (temporary CPU implementation)
    pub fn sum(&self) -> Result<CudaTensor> {
        // TODO: Implement GPU reduction kernel
        let cpu_data = self.to_vec()?;
        let sum: f32 = cpu_data.iter().sum();
        CudaTensor::from_vec(vec![sum], Shape::from_dims(&[1]), self.device.clone())
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Copy to CPU for inspection
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.device.dtoh_sync_copy(&self.data)
            .map_err(|_| FlameError::CudaDriver)?)
    }

    /// Get a single value (for loss printing)
    pub fn item(&self) -> Result<f32> {
        if self.shape.elem_count() != 1 {
            return Err(FlameError::InvalidOperation("item() requires tensor with single element".into()));
        }
        Ok(self.to_vec()?[0])
    }

    /// Detach from computation graph (for autograd later)
    pub fn detach(&self) -> Result<CudaTensor> {
        let mut output = CudaTensor::zeros(self.shape.clone(), self.device.clone())?;
        self.device.dtod_copy(&self.data, &mut output.data)
            .map_err(|_| FlameError::CudaDriver)?;
        Ok(output)
    }
}