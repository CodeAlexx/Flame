use crate::{Shape, Result, FlameError, Tensor};
use crate::cuda_kernels::CudaKernels;
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
        // For now, use device method to fill with ones
        let ones = vec![1.0f32; size];
        let data = device.htod_sync_copy(&ones[..])
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

        // Use the kernel that updates weights and returns new tensor
        let kernels = CudaKernels::new(self.device.clone())?;
        let gradient_tensor = Tensor {
            storage: crate::tensor_storage::TensorStorage::F32 { 
                data: gradient.data.clone(), 
                numel: gradient.shape.elem_count() 
            },
            shape: gradient.shape.clone(),
            device: gradient.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        let self_tensor = Tensor {
            storage: crate::tensor_storage::TensorStorage::F32 { 
                data: self.data.clone(), 
                numel: self.shape.elem_count() 
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        let updated = kernels.update_weights(&self_tensor, &gradient_tensor, lr)?;
        // Extract the data from the Arc - we need to clone it
        // Extract data from the updated tensor's storage
        if let crate::tensor_storage::TensorStorage::F32 { data, .. } = &updated.storage {
            self.data = data.clone();
        } else {
            return Err(FlameError::InvalidOperation("Expected F32 storage".into()));
        }
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
            blas.gemm(cfg, &other.data, &self.data, &mut output.data)
                .map_err(|_| FlameError::CuBlas)?;
        }
        
        Ok(output)
    }

    /// Element-wise addition
    pub fn add(&self, other: &CudaTensor) -> Result<CudaTensor> {
        let shape = self.shape.broadcast_shape_binary_op(&other.shape)?;
        let mut output = CudaTensor::zeros(shape.clone(), self.device.clone())?;
        
        // For now, simple element-wise add (no broadcasting yet)
        if self.shape == other.shape {
            let kernels = CudaKernels::new(self.device.clone())?;
            let self_tensor = Tensor {
                storage: crate::tensor_storage::TensorStorage::F32 { 
                    data: self.data.clone(), 
                    numel: self.shape.elem_count() 
                },
                shape: self.shape.clone(),
                device: self.device.clone(),
                id: crate::tensor::TensorId::new(),
                requires_grad: false,
            };
            let other_tensor = Tensor {
                storage: crate::tensor_storage::TensorStorage::F32 { 
                    data: other.data.clone(), 
                    numel: other.shape.elem_count() 
                },
                shape: other.shape.clone(),
                device: other.device.clone(),
                id: crate::tensor::TensorId::new(),
                requires_grad: false,
            };
            let result = kernels.add(&self_tensor, &other_tensor)?;
            // Extract data from result's storage
            if let crate::tensor_storage::TensorStorage::F32 { data, .. } = &result.storage {
                output.data = data.clone();
            } else {
                return Err(FlameError::InvalidOperation("Expected F32 storage".into()));
            }
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
        let kernels = CudaKernels::new(self.device.clone())?;
        let self_tensor = Tensor {
            storage: crate::tensor_storage::TensorStorage::F32 { 
                data: self.data.clone(), 
                numel: self.shape.elem_count() 
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        let other_tensor = Tensor {
            storage: crate::tensor_storage::TensorStorage::F32 { 
                data: other.data.clone(), 
                numel: other.shape.elem_count() 
            },
            shape: other.shape.clone(),
            device: other.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        let result = kernels.mul(&self_tensor, &other_tensor)?;
        // Extract data from result's storage
        if let crate::tensor_storage::TensorStorage::F32 { data, .. } = &result.storage {
            output.data = data.clone();
        } else {
            return Err(FlameError::InvalidOperation("Expected F32 storage".into()));
        }
        Ok(output)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Result<CudaTensor> {
        let mut output = CudaTensor::zeros(self.shape.clone(), self.device.clone())?;
        let kernels = CudaKernels::new(self.device.clone())?;
        let self_tensor = Tensor {
            storage: crate::tensor_storage::TensorStorage::F32 { 
                data: self.data.clone(), 
                numel: self.shape.elem_count() 
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        let result = kernels.mul_scalar(&self_tensor, scalar)?;
        // Extract data from result's storage
        if let crate::tensor_storage::TensorStorage::F32 { data, .. } = &result.storage {
            output.data = data.clone();
        } else {
            return Err(FlameError::InvalidOperation("Expected F32 storage".into()));
        }
        Ok(output)
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<CudaTensor> {
        let mut output = CudaTensor::zeros(self.shape.clone(), self.device.clone())?;
        let kernels = CudaKernels::new(self.device.clone())?;
        let self_tensor = Tensor {
            storage: crate::tensor_storage::TensorStorage::F32 { 
                data: self.data.clone(), 
                numel: self.shape.elem_count() 
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        let result = kernels.relu(&self_tensor)?;
        // Extract data from result's storage
        if let crate::tensor_storage::TensorStorage::F32 { data, .. } = &result.storage {
            output.data = data.clone();
        } else {
            return Err(FlameError::InvalidOperation("Expected F32 storage".into()));
        }
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

    /// Sum reduction using GPU kernel
    pub fn sum(&self) -> Result<CudaTensor> {
        // Use CUB device reduction for efficient GPU sum
        let total_elements = self.shape.elem_count();
        
        // Allocate output buffer for single element
        let sum_buf = unsafe {
            self.device.alloc::<f32>(1)?
        };
        
        // Launch custom reduction kernel
        let kernels = CudaKernels::new(self.device.clone())?;
        let self_tensor = Tensor {
            storage: crate::tensor_storage::TensorStorage::F32 { 
                data: self.data.clone(), 
                numel: self.shape.elem_count() 
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        };
        let sum_result = kernels.sum_kernel(&self_tensor)?;
        
        // Convert back to CudaTensor
        // Note: sum_result is a Tensor with storage field, not data
        if let crate::tensor_storage::TensorStorage::F32 { data, .. } = &sum_result.storage {
            Ok(CudaTensor {
                data: data.clone(),
                shape: sum_result.shape.clone(),
                device: sum_result.device.clone(),
            })
        } else {
            Err(FlameError::InvalidOperation("Expected F32 storage".into()))
        }
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