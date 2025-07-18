use crate::{Result, FlameError};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::sync::Arc;

// External CUDA kernel declarations
extern "C" {
    fn update_weights_f32(
        weights: *mut f32,
        gradients: *const f32,
        learning_rate: f32,
        num_elements: i32,
    );
    
    fn add_f32(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        num_elements: i32,
    );
    
    fn mul_f32(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        num_elements: i32,
    );
    
    fn mul_scalar_f32(
        input: *const f32,
        scalar: f32,
        out: *mut f32,
        num_elements: i32,
    );
    
    fn relu_f32(
        input: *const f32,
        out: *mut f32,
        num_elements: i32,
    );
    
    fn relu_backward_f32(
        grad_output: *const f32,
        input: *const f32,
        grad_input: *mut f32,
        num_elements: i32,
    );
    
    fn fill_f32(
        tensor: *mut f32,
        value: f32,
        num_elements: i32,
    );
    
    fn copy_f32(
        src: *const f32,
        dst: *mut f32,
        num_elements: i32,
    );
}

// Rust-safe wrappers using cudarc's launch mechanism
pub struct CudaKernels;

impl CudaKernels {
    const BLOCK_SIZE: u32 = 256;
    
    fn launch_config(num_elements: usize) -> LaunchConfig {
        let block_size = Self::BLOCK_SIZE;
        let grid_size = (num_elements as u32 + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }
    
    /// Update weights in-place on GPU
    pub fn update_weights(
        device: &Arc<CudaDevice>,
        weights: &mut CudaSlice<f32>,
        gradients: &CudaSlice<f32>,
        learning_rate: f32,
    ) -> Result<()> {
        let num_elements = weights.len();
        if num_elements != gradients.len() {
            return Err(FlameError::InvalidOperation(
                "Weight and gradient tensors must have same size".into()
            ));
        }
        
        let config = Self::launch_config(num_elements);
        
        unsafe {
            device.launch_kernel(
                update_weights_f32 as *const (),
                config,
                (
                    weights.as_mut_ptr(),
                    gradients.as_ptr(),
                    learning_rate,
                    num_elements as i32,
                ),
            ).map_err(|_| FlameError::CudaDriver)?;
        }
        
        Ok(())
    }
    
    /// Element-wise addition
    pub fn add(
        device: &Arc<CudaDevice>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let num_elements = a.len();
        if num_elements != b.len() || num_elements != out.len() {
            return Err(FlameError::InvalidOperation(
                "Tensors must have same size for addition".into()
            ));
        }
        
        let config = Self::launch_config(num_elements);
        
        unsafe {
            device.launch_kernel(
                add_f32 as *const (),
                config,
                (
                    a.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    num_elements as i32,
                ),
            ).map_err(|_| FlameError::CudaDriver)?;
        }
        
        Ok(())
    }
    
    /// Element-wise multiplication
    pub fn mul(
        device: &Arc<CudaDevice>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let num_elements = a.len();
        if num_elements != b.len() || num_elements != out.len() {
            return Err(FlameError::InvalidOperation(
                "Tensors must have same size for multiplication".into()
            ));
        }
        
        let config = Self::launch_config(num_elements);
        
        unsafe {
            device.launch_kernel(
                mul_f32 as *const (),
                config,
                (
                    a.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    num_elements as i32,
                ),
            ).map_err(|_| FlameError::CudaDriver)?;
        }
        
        Ok(())
    }
    
    /// Scalar multiplication
    pub fn mul_scalar(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<f32>,
        scalar: f32,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let num_elements = input.len();
        if num_elements != out.len() {
            return Err(FlameError::InvalidOperation(
                "Input and output must have same size".into()
            ));
        }
        
        let config = Self::launch_config(num_elements);
        
        unsafe {
            device.launch_kernel(
                mul_scalar_f32 as *const (),
                config,
                (
                    input.as_ptr(),
                    scalar,
                    out.as_mut_ptr(),
                    num_elements as i32,
                ),
            ).map_err(|_| FlameError::CudaDriver)?;
        }
        
        Ok(())
    }
    
    /// ReLU activation
    pub fn relu(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let num_elements = input.len();
        if num_elements != out.len() {
            return Err(FlameError::InvalidOperation(
                "Input and output must have same size".into()
            ));
        }
        
        let config = Self::launch_config(num_elements);
        
        unsafe {
            device.launch_kernel(
                relu_f32 as *const (),
                config,
                (
                    input.as_ptr(),
                    out.as_mut_ptr(),
                    num_elements as i32,
                ),
            ).map_err(|_| FlameError::CudaDriver)?;
        }
        
        Ok(())
    }
    
    /// Fill tensor with value
    pub fn fill(
        device: &Arc<CudaDevice>,
        tensor: &mut CudaSlice<f32>,
        value: f32,
    ) -> Result<()> {
        let num_elements = tensor.len();
        let config = Self::launch_config(num_elements);
        
        unsafe {
            device.launch_kernel(
                fill_f32 as *const (),
                config,
                (
                    tensor.as_mut_ptr(),
                    value,
                    num_elements as i32,
                ),
            ).map_err(|_| FlameError::CudaDriver)?;
        }
        
        Ok(())
    }
}