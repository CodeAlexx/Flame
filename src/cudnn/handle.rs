// cuDNN Handle Management
// Provides thread-safe handle management for cuDNN operations

use crate::{Error, Result};
use std::os::raw::{c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

// Global cuDNN handle with lazy initialization
static CUDNN_HANDLE: OnceLock<Arc<Mutex<CudnnHandle>>> = OnceLock::new();

// FFI bindings for handle management
#[link(name = "cudnn")]
extern "C" {
    fn cudnnCreate(handle: *mut *mut c_void) -> c_int;
    fn cudnnDestroy(handle: *mut c_void) -> c_int;
    fn cudnnSetStream(handle: *mut c_void, stream: *mut c_void) -> c_int;
}

pub struct CudnnHandle {
    handle: *mut c_void,
}

impl CudnnHandle {
    /// Create a new cuDNN handle
    pub fn new() -> Result<Self> {
        let mut handle: *mut c_void = ptr::null_mut();
        let status = unsafe { cudnnCreate(&mut handle) };

        if status != 0 {
            return Err(Error::CudaError(format!(
                "Failed to create cuDNN handle: {}",
                status
            )));
        }

        Ok(CudnnHandle { handle })
    }

    /// Get the raw handle pointer
    pub fn as_ptr(&self) -> *mut c_void {
        self.handle
    }

    /// Set CUDA stream for this handle
    pub fn set_stream(&self, stream: *mut c_void) -> Result<()> {
        let status = unsafe { cudnnSetStream(self.handle, stream) };
        if status != 0 {
            return Err(Error::CudaError(format!(
                "Failed to set cuDNN stream: {}",
                status
            )));
        }
        Ok(())
    }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                cudnnDestroy(self.handle);
            }
        }
    }
}

unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}

/// Get or create the global cuDNN handle
pub fn get_cudnn_handle() -> Result<Arc<Mutex<CudnnHandle>>> {
    if let Some(h) = CUDNN_HANDLE.get() {
        return Ok(h.clone());
    }
    let handle = CudnnHandle::new()?;
    let arc = Arc::new(Mutex::new(handle));
    let _ = CUDNN_HANDLE.set(arc.clone());
    Ok(arc)
}
