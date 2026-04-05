use crate::{Error, Result};
use core::ffi::c_void;
use cudarc::driver::CudaDevice;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

struct LtContext {
    stream: *mut c_void,
    handle: *mut c_void,
}

unsafe impl Send for LtContext {}
unsafe impl Sync for LtContext {}

static CONTEXTS: Lazy<Mutex<HashMap<usize, LtContext>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn init_context(_device: &Arc<CudaDevice>) -> Result<LtContext> {
    // Use the default CUDA stream (null) so cublasLt GEMMs pipeline with
    // elementwise kernels without implicit sync barriers between streams.
    let stream: *mut c_void = core::ptr::null_mut();

    let mut handle: *mut c_void = core::ptr::null_mut();
    let handle_status = unsafe { crate::cuda::ffi::cublasLtCreate(&mut handle as *mut _) };
    if handle_status != 0 {
        return Err(Error::Cuda(format!(
            "cublasLtCreate failed: {}",
            handle_status
        )));
    }

    Ok(LtContext { stream, handle })
}

pub fn stream_ptr(device: &Arc<CudaDevice>) -> Result<*mut c_void> {
    let key = device.ordinal();
    {
        let map = CONTEXTS.lock().unwrap();
        if let Some(ctx) = map.get(&key) {
            return Ok(ctx.stream);
        }
    }

    let ctx = init_context(device)?;
    let stream = ctx.stream;
    let mut map = CONTEXTS.lock().unwrap();
    map.entry(key).or_insert(ctx);
    Ok(stream)
}

pub fn cublaslt_handle_ptr(device: &Arc<CudaDevice>) -> Result<*mut c_void> {
    let key = device.ordinal();
    {
        let map = CONTEXTS.lock().unwrap();
        if let Some(ctx) = map.get(&key) {
            return Ok(ctx.handle);
        }
    }

    let ctx = init_context(device)?;
    let handle = ctx.handle;
    let mut map = CONTEXTS.lock().unwrap();
    map.entry(key).or_insert(ctx);
    Ok(handle)
}
