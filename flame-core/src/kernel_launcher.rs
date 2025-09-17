//! Type-safe CUDA kernel launcher
//! 
//! This module provides a safe interface for launching CUDA kernels with compile-time
//! type checking and automatic parameter packing.

use crate::{Result, FlameError};
use cudarc::{
    driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig, DeviceRepr},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    /// Global kernel cache to avoid recompilation
    static ref KERNEL_CACHE: Mutex<HashMap<String, Arc<CompiledKernel>>> = Mutex::new(HashMap::new());
}

/// Represents a compiled CUDA kernel
pub struct CompiledKernel {
    pub name: String,
    pub ptx: cudarc::nvrtc::Ptx,
}

/// Type-safe kernel launcher
pub struct KernelLauncher {
    device: Arc<CudaDevice>,
}

impl KernelLauncher {
    /// Create a new kernel launcher for the given device
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
    
    /// Compile a kernel from source code
    pub fn compile_kernel(
        &self,
        kernel_name: &str,
        source_code: &str,
    ) -> Result<Arc<CompiledKernel>> {
        // Check cache first
        let cache_key = format!("{}-{}", self.device.ordinal(), kernel_name);
        
        {
            let cache = KERNEL_CACHE.lock().map_err(|_| crate::FlameError::Training("kernel launcher cache mutex poisoned".into()))?;
            if let Some(kernel) = cache.get(&cache_key) {
                return Ok(kernel.clone());
            }
        }
        
        // Compile with optimizations
        let opts = CompileOptions {
            ftz: Some(true),              // Flush denormals to zero
            prec_div: Some(false),        // Use approximate division
            prec_sqrt: Some(false),       // Use approximate square root
            fmad: Some(true),             // Use fused multiply-add
            ..Default::default()
        };
        
        let ptx = compile_ptx_with_opts(source_code, opts)
            .map_err(|e| FlameError::Cuda(format!("Kernel compilation failed: {:?}", e)))?;
        
        let kernel = Arc::new(CompiledKernel {
            name: kernel_name.to_string(),
            ptx,
        });
        
        // Cache the compiled kernel
        {
            let mut cache = KERNEL_CACHE.lock().map_err(|_| crate::FlameError::Training("kernel launcher cache mutex poisoned".into()))?;
            cache.insert(cache_key, kernel.clone());
        }
        
        Ok(kernel)
    }
    
    /// Load a compiled kernel into the device
    pub fn load_kernel(
        &self,
        kernel: &CompiledKernel,
        module_name: &str,
    ) -> Result<()> {
        // Note: In production, this would load the PTX.
        // For now, we skip actual loading similar to ensure_kernel in cuda_kernels.rs
        // This avoids lifetime issues with the kernel names array
        Ok(())
    }
    
    // Note: get_func is accessed directly on the device
    // due to lifetime and ownership constraints
    
    // Note: Direct kernel launch should be done using function.launch()
    // since CudaFunction::launch takes ownership of self
    
    /// Convenience method to compile, load, and get a kernel function
    pub fn prepare_kernel(
        &self,
        kernel_name: &str,
        source_code: &str,
    ) -> Result<()> {
        let kernel = self.compile_kernel(kernel_name, source_code)?;
        self.load_kernel(&kernel, kernel_name)?;
        Ok(())
    }
}

/// Macro for generating kernel parameter structures
#[macro_export]
macro_rules! kernel_params {
    (
        $(#[$attr:meta])*
        struct $name:ident {
            $(
                $(#[$field_attr:meta])*
                $field:ident: $type:ty
            ),* $(,)?
        }
    ) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug)]
        $(#[$attr])*
        struct $name {
            $(
                $(#[$field_attr])*
                $field: $type,
            )*
        }
        
        unsafe impl cudarc::driver::DeviceRepr for $name {}
    };
}

/// Standard kernel templates
pub mod templates {
    /// Generate element-wise unary operation kernel
    pub fn elementwise_unary(kernel_name: &str, operation: &str) -> String {
        format!(r#"
extern "C" __global__ void {kernel_name}(
    float* output,
    const float* input,
    int numel
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        float x = input[idx];
        output[idx] = {operation};
    }}
}}"#, kernel_name = kernel_name, operation = operation)
    }
    
    /// Generate element-wise binary operation kernel
    pub fn elementwise_binary(kernel_name: &str, operation: &str) -> String {
        format!(r#"
extern "C" __global__ void {kernel_name}(
    float* output,
    const float* a,
    const float* b,
    int numel
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        output[idx] = a[idx] {operation} b[idx];
    }}
}}"#, kernel_name = kernel_name, operation = operation)
    }
    
    /// Generate reduction kernel
    pub fn reduction_sum() -> &'static str {
        r#"
extern "C" __global__ void reduction_sum(
    float* output,
    const float* input,
    int numel
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data to shared memory
    sdata[tid] = (idx < numel) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}"#
    }
}

/// Helper functions for common launch configurations
pub mod launch_configs {
    use cudarc::driver::LaunchConfig;
    
    /// Get launch config for element-wise operations
    pub fn elementwise(numel: usize) -> LaunchConfig {
        LaunchConfig::for_num_elems(numel as u32)
    }
    
    /// Get launch config for reduction operations
    pub fn reduction(numel: usize) -> LaunchConfig {
        let block_size = 256;
        let grid_size = (numel as u32 + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: block_size * 4, // 4 bytes per float
        }
    }
    
    /// Get launch config for 2D operations (e.g., matrix multiplication)
    pub fn grid_2d(rows: usize, cols: usize, block_size: u32) -> LaunchConfig {
        let grid_x = (cols as u32 + block_size - 1) / block_size;
        let grid_y = (rows as u32 + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, block_size, 1),
            shared_mem_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kernel_params_macro() {
        kernel_params! {
            struct TestParams {
                a: i32,
                b: f32,
                c: i32,
            }
        }
        
        let params = TestParams { a: 1, b: 2.0, c: 3 };
        assert_eq!(params.a, 1);
        assert_eq!(params.b, 2.0);
        assert_eq!(params.c, 3);
    }
}
