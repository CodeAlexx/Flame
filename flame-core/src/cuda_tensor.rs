use crate::{Shape, Result, FlameError};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

// Helper to allocate from pool and copy data
fn alloc_from_pool_and_copy(device: &Arc<CudaDevice>, data: &[i32]) -> Result<CudaSlice<f32>> {
    let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let mut cuda_data = crate::tensor::alloc_from_pool(device, f32_data.len())?;
    device.htod_copy_into(&f32_data, &mut cuda_data).map_err(|_| FlameError::CudaDriver)?;
    Ok(cuda_data)
}


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

    /// Create random tensor (CPU random for now)
    pub fn randn(shape: Shape, mean: f32, std: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
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
    /// For now, we'll do it on CPU and copy back (inefficient but works)
    pub fn update_weights(&mut self, gradient: &CudaTensor, lr: f32) -> Result<()> {
        if self.shape != gradient.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: gradient.shape.clone(),
            });
        }

        // Download to CPU
        let mut weight_data = self.device.dtoh_sync_copy(&self.data())
            .map_err(|_| FlameError::CudaDriver)?;
        let grad_data = gradient.device.dtoh_sync_copy(&gradient.data())
            .map_err(|_| FlameError::CudaDriver)?;

        // Update on CPU
        for i in 0..weight_data.len() {
            weight_data[i] -= lr * grad_data[i];
        }

        // Upload back to GPU - create new slice
        self.data = alloc_from_pool_and_copy(&self.device, &weight_data)
            .map_err(|_| FlameError::CudaDriver)?;

        Ok(())
    }

    /// Matrix multiplication (using cuBLAS later, CPU for now)
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

        // Download to host for matmul (debug path; not used in active GPU ops)
        let a_data = self.device.dtoh_sync_copy(&self.data())
            .map_err(|_| FlameError::CudaDriver)?;
        let b_data = other.device.dtoh_sync_copy(&other.data())
            .map_err(|_| FlameError::CudaDriver)?;

        // CPU matmul
        let mut c_data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }

        // Upload result
        CudaTensor::from_vec(c_data, Shape::from_dims(&[m, n]), self.device.clone())
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
