use cuda_runtime_sys::{cudaError, cudaMemGetInfo};

/// Query global device memory usage via the CUDA runtime.
///
/// Returns `(free_bytes, total_bytes)` when successful, or the CUDA error code on failure.
pub fn cuda_mem_get_info() -> Result<(usize, usize), cudaError> {
    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    let status = unsafe {
        cudaMemGetInfo(
            &mut free_bytes as *mut usize,
            &mut total_bytes as *mut usize,
        )
    };
    if status == cudaError::cudaSuccess {
        Ok((free_bytes, total_bytes))
    } else {
        Err(status)
    }
}

/// Convenience helper returning the free memory in MiB when available.
pub fn cuda_mem_get_free_mb() -> Option<usize> {
    match cuda_mem_get_info() {
        Ok((free_bytes, _)) => Some(free_bytes / (1024 * 1024)),
        Err(_) => None,
    }
}
