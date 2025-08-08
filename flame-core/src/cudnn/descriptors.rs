// cuDNN Descriptor Management
// Handles tensor, filter, and convolution descriptors for cuDNN operations

use std::os::raw::{c_void, c_int};
use std::ptr;
use crate::{Result, FlameError, DType};

// cuDNN constants
pub const CUDNN_TENSOR_NCHW: c_int = 0;
pub const CUDNN_CROSS_CORRELATION: c_int = 1;
pub const CUDNN_CONVOLUTION: c_int = 0;

// Data type mapping
pub const CUDNN_DATA_FLOAT: c_int = 0;
pub const CUDNN_DATA_DOUBLE: c_int = 1;
pub const CUDNN_DATA_HALF: c_int = 2;
pub const CUDNN_DATA_INT8: c_int = 3;
pub const CUDNN_DATA_INT32: c_int = 4;
pub const CUDNN_DATA_INT8x4: c_int = 5;
pub const CUDNN_DATA_UINT8: c_int = 6;
pub const CUDNN_DATA_UINT8x4: c_int = 7;
pub const CUDNN_DATA_INT8x32: c_int = 8;
pub const CUDNN_DATA_BFLOAT16: c_int = 9;

// FFI bindings
#[link(name = "cudnn")]
extern "C" {
    // Tensor descriptors
    fn cudnnCreateTensorDescriptor(desc: *mut *mut c_void) -> c_int;
    fn cudnnDestroyTensorDescriptor(desc: *mut c_void) -> c_int;
    fn cudnnSetTensor4dDescriptor(
        desc: *mut c_void,
        format: c_int,
        datatype: c_int,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int
    ) -> c_int;
    
    fn cudnnSetTensorNdDescriptor(
        desc: *mut c_void,
        datatype: c_int,
        nbDims: c_int,
        dimA: *const c_int,
        strideA: *const c_int
    ) -> c_int;
    
    // Filter descriptors
    fn cudnnCreateFilterDescriptor(desc: *mut *mut c_void) -> c_int;
    fn cudnnDestroyFilterDescriptor(desc: *mut c_void) -> c_int;
    fn cudnnSetFilter4dDescriptor(
        desc: *mut c_void,
        datatype: c_int,
        format: c_int,
        k: c_int,
        c: c_int,
        h: c_int,
        w: c_int
    ) -> c_int;
    
    // Convolution descriptors
    fn cudnnCreateConvolutionDescriptor(desc: *mut *mut c_void) -> c_int;
    fn cudnnDestroyConvolutionDescriptor(desc: *mut c_void) -> c_int;
    fn cudnnSetConvolution2dDescriptor(
        desc: *mut c_void,
        pad_h: c_int,
        pad_w: c_int,
        stride_h: c_int,
        stride_w: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        mode: c_int,
        compute_type: c_int
    ) -> c_int;
}

/// Convert FLAME DType to cuDNN data type
pub fn dtype_to_cudnn(dtype: DType) -> c_int {
    match dtype {
        DType::F32 => CUDNN_DATA_FLOAT,
        DType::F64 => CUDNN_DATA_DOUBLE,
        DType::F16 => CUDNN_DATA_HALF,
        DType::BF16 => CUDNN_DATA_BFLOAT16,
        DType::I32 => CUDNN_DATA_INT32,
        _ => CUDNN_DATA_FLOAT, // Default to float
    }
}

/// Tensor descriptor for cuDNN operations
pub struct TensorDescriptor {
    desc: *mut c_void,
}

impl TensorDescriptor {
    pub fn new(shape: &[usize], dtype: DType) -> Result<Self> {
        let mut desc: *mut c_void = ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut desc) };
        if status != 0 {
            return Err(FlameError::CudaError(format!("Failed to create tensor descriptor: {}", status)));
        }
        
        let tensor_desc = TensorDescriptor { desc };
        
        // Set the descriptor based on dimensionality
        match shape.len() {
            2 => {
                // For 2D tensors (M x N), treat as 4D with batch=1, channels=1
                tensor_desc.set_4d(dtype, 1, 1, shape[0], shape[1])?;
            }
            3 => {
                // For 3D tensors (B x M x N), treat as 4D with channels=1
                tensor_desc.set_4d(dtype, shape[0], 1, shape[1], shape[2])?;
            }
            4 => {
                // Native 4D tensor
                tensor_desc.set_4d(dtype, shape[0], shape[1], shape[2], shape[3])?;
            }
            _ => {
                // For other dimensions, use Nd descriptor
                tensor_desc.set_nd(dtype, shape)?;
            }
        }
        
        Ok(tensor_desc)
    }
    
    pub fn set_4d(&self, dtype: DType, n: usize, c: usize, h: usize, w: usize) -> Result<()> {
        let status = unsafe {
            cudnnSetTensor4dDescriptor(
                self.desc,
                CUDNN_TENSOR_NCHW,
                dtype_to_cudnn(dtype),
                n as c_int,
                c as c_int,
                h as c_int,
                w as c_int
            )
        };
        if status != 0 {
            return Err(FlameError::CudaError(format!("Failed to set tensor descriptor: {}", status)));
        }
        Ok(())
    }
    
    pub fn set_nd(&self, dtype: DType, shape: &[usize]) -> Result<()> {
        let dims: Vec<c_int> = shape.iter().map(|&d| d as c_int).collect();
        
        // Calculate strides (assuming C-contiguous layout)
        let mut strides = vec![1i32; shape.len()];
        for i in (0..shape.len()-1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as c_int;
        }
        
        let status = unsafe {
            cudnnSetTensorNdDescriptor(
                self.desc,
                dtype_to_cudnn(dtype),
                shape.len() as c_int,
                dims.as_ptr(),
                strides.as_ptr()
            )
        };
        if status != 0 {
            return Err(FlameError::CudaError(format!("Failed to set Nd tensor descriptor: {}", status)));
        }
        Ok(())
    }
    
    pub fn as_ptr(&self) -> *mut c_void {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { cudnnDestroyTensorDescriptor(self.desc); }
        }
    }
}

/// Filter descriptor for cuDNN convolution operations
pub struct FilterDescriptor {
    desc: *mut c_void,
}

impl FilterDescriptor {
    pub fn new() -> Result<Self> {
        let mut desc: *mut c_void = ptr::null_mut();
        let status = unsafe { cudnnCreateFilterDescriptor(&mut desc) };
        if status != 0 {
            return Err(FlameError::CudaError(format!("Failed to create filter descriptor: {}", status)));
        }
        Ok(FilterDescriptor { desc })
    }
    
    pub fn set_4d(&self, dtype: DType, k: usize, c: usize, h: usize, w: usize) -> Result<()> {
        let status = unsafe {
            cudnnSetFilter4dDescriptor(
                self.desc,
                dtype_to_cudnn(dtype),
                CUDNN_TENSOR_NCHW,
                k as c_int,
                c as c_int,
                h as c_int,
                w as c_int
            )
        };
        if status != 0 {
            return Err(FlameError::CudaError(format!("Failed to set filter descriptor: {}", status)));
        }
        Ok(())
    }
    
    pub fn as_ptr(&self) -> *mut c_void {
        self.desc
    }
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { cudnnDestroyFilterDescriptor(self.desc); }
        }
    }
}

/// Convolution descriptor for cuDNN operations
pub struct ConvolutionDescriptor {
    desc: *mut c_void,
}

impl ConvolutionDescriptor {
    pub fn new() -> Result<Self> {
        let mut desc: *mut c_void = ptr::null_mut();
        let status = unsafe { cudnnCreateConvolutionDescriptor(&mut desc) };
        if status != 0 {
            return Err(FlameError::CudaError(format!("Failed to create convolution descriptor: {}", status)));
        }
        Ok(ConvolutionDescriptor { desc })
    }
    
    pub fn set_2d(&self, padding: usize, stride: usize, dilation: usize, dtype: DType) -> Result<()> {
        let status = unsafe {
            cudnnSetConvolution2dDescriptor(
                self.desc,
                padding as c_int,
                padding as c_int,
                stride as c_int,
                stride as c_int,
                dilation as c_int,
                dilation as c_int,
                CUDNN_CROSS_CORRELATION,
                dtype_to_cudnn(dtype)
            )
        };
        if status != 0 {
            return Err(FlameError::CudaError(format!("Failed to set convolution descriptor: {}", status)));
        }
        Ok(())
    }
    
    pub fn as_ptr(&self) -> *mut c_void {
        self.desc
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { cudnnDestroyConvolutionDescriptor(self.desc); }
        }
    }
}