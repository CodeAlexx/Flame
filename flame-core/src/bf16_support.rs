use crate::{Result, FlameError, Shape, DType, Tensor};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{Ptx, compile_ptx_with_opts};
use std::sync::Arc;
use half::bf16;

/// BF16 conversion kernels
const BF16_KERNELS: &str = r#"
extern "C" __global__ void bf16_to_f32(const __nv_bfloat16* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

extern "C" __global__ void f32_to_bf16(const float* input, __nv_bfloat16* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

extern "C" __global__ void bf16_add(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

extern "C" __global__ void bf16_mul(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmul(a[idx], b[idx]);
    }
}

extern "C" __global__ void bf16_scale(const __nv_bfloat16* input, float scale, __nv_bfloat16* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(val * scale);
    }
}
"#;

/// Compile BF16 kernels
pub fn compile_bf16_kernels(device: &Arc<CudaDevice>) -> Result<Ptx> {
    let opts = vec![
        "-arch=compute_80".to_string(), // BF16 requires compute capability 8.0+
        "-use_fast_math".to_string(),
        "-std=c++14".to_string(),
    ];
    
    compile_ptx_with_opts(BF16_KERNELS, &opts)
        .map_err(|e| FlameError::InvalidOperation(format!("Failed to compile BF16 kernels: {}", e)))
}

/// Convert F32 slice to BF16
pub fn f32_to_bf16(
    device: &Arc<CudaDevice>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    let n = input.len();
    if n != output.len() {
        return Err(FlameError::InvalidOperation(
            format!("Size mismatch: input {} vs output {}", n, output.len())
        ));
    }
    
    let ptx = compile_bf16_kernels(device)?;
    device.load_ptx(ptx, "bf16_module", &["f32_to_bf16"])
        .map_err(|e| FlameError::InvalidOperation(format!("Failed to load PTX: {}", e)))?;
    
    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    
    let func = device.get_func("bf16_module", "f32_to_bf16")
        .map_err(|e| FlameError::InvalidOperation(format!("Failed to get function: {}", e)))?;
    
    unsafe {
        func.launch(config, (input, output, n as i32))
            .map_err(|e| FlameError::InvalidOperation(format!("Kernel launch failed: {}", e)))?;
    }
    
    Ok(())
}

/// Convert BF16 slice to F32
pub fn bf16_to_f32(
    device: &Arc<CudaDevice>,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<f32>,
) -> Result<()> {
    let n = input.len();
    if n != output.len() {
        return Err(FlameError::InvalidOperation(
            format!("Size mismatch: input {} vs output {}", n, output.len())
        ));
    }
    
    let ptx = compile_bf16_kernels(device)?;
    device.load_ptx(ptx, "bf16_module", &["bf16_to_f32"])
        .map_err(|e| FlameError::InvalidOperation(format!("Failed to load PTX: {}", e)))?;
    
    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    
    let func = device.get_func("bf16_module", "bf16_to_f32")
        .map_err(|e| FlameError::InvalidOperation(format!("Failed to get function: {}", e)))?;
    
    unsafe {
        func.launch(config, (input, output, n as i32))
            .map_err(|e| FlameError::InvalidOperation(format!("Kernel launch failed: {}", e)))?;
    }
    
    Ok(())
}

/// BF16 tensor operations wrapper
pub struct BF16Ops;

impl BF16Ops {
    /// Create BF16 tensor from F32 data
    pub fn from_f32(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Tensor> {
        let numel = shape.elem_count();
        
        // Allocate BF16 storage
        let mut bf16_data = device.alloc::<bf16>(numel)
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to allocate BF16: {}", e)))?;
        
        // Upload F32 data to GPU
        let f32_data = device.htod_sync_copy(&data)
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to copy to device: {}", e)))?;
        
        // Convert F32 to BF16
        f32_to_bf16(&device, &f32_data, &mut bf16_data)?;
        
        // Create tensor with BF16 storage
        Tensor::from_bf16_slice(bf16_data, shape, device)
    }
    
    /// Convert BF16 tensor to F32
    pub fn to_f32(tensor: &Tensor) -> Result<Tensor> {
        if tensor.dtype() != DType::BF16 {
            return Err(FlameError::InvalidOperation("Tensor is not BF16".to_string()));
        }
        
        let shape = tensor.shape().clone();
        let device = tensor.device().clone();
        let numel = shape.elem_count();
        
        // Get BF16 data
        let bf16_data = tensor.as_bf16_slice()?;
        
        // Allocate F32 storage
        let mut f32_data = device.alloc::<f32>(numel)
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to allocate F32: {}", e)))?;
        
        // Convert BF16 to F32
        bf16_to_f32(&device, bf16_data, &mut f32_data)?;
        
        // Create F32 tensor
        Tensor::from_cuda_slice(f32_data, shape, device)
    }
    
    /// Add two BF16 tensors
    pub fn add_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
            return Err(FlameError::InvalidOperation("Both tensors must be BF16".to_string()));
        }
        
        let shape = a.shape().clone();
        let device = a.device().clone();
        let numel = shape.elem_count();
        
        // Get BF16 data
        let a_data = a.as_bf16_slice()?;
        let b_data = b.as_bf16_slice()?;
        
        // Allocate output
        let mut out_data = device.alloc::<bf16>(numel)
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to allocate output: {}", e)))?;
        
        // Compile and run kernel
        let ptx = compile_bf16_kernels(&device)?;
        device.load_ptx(ptx, "bf16_module", &["bf16_add"])
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to load PTX: {}", e)))?;
        
        let block_size = 256;
        let grid_size = (numel + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let func = device.get_func("bf16_module", "bf16_add")
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to get function: {}", e)))?;
        
        unsafe {
            func.launch(config, (a_data, b_data, &mut out_data, numel as i32))
                .map_err(|e| FlameError::InvalidOperation(format!("Kernel launch failed: {}", e)))?;
        }
        
        // Create output tensor
        Tensor::from_bf16_slice(out_data, shape, device)
    }
    
    /// Multiply two BF16 tensors
    pub fn mul_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
            return Err(FlameError::InvalidOperation("Both tensors must be BF16".to_string()));
        }
        
        let shape = a.shape().clone();
        let device = a.device().clone();
        let numel = shape.elem_count();
        
        // Get BF16 data
        let a_data = a.as_bf16_slice()?;
        let b_data = b.as_bf16_slice()?;
        
        // Allocate output
        let mut out_data = device.alloc::<bf16>(numel)
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to allocate output: {}", e)))?;
        
        // Compile and run kernel
        let ptx = compile_bf16_kernels(&device)?;
        device.load_ptx(ptx, "bf16_module", &["bf16_mul"])
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to load PTX: {}", e)))?;
        
        let block_size = 256;
        let grid_size = (numel + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let func = device.get_func("bf16_module", "bf16_mul")
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to get function: {}", e)))?;
        
        unsafe {
            func.launch(config, (a_data, b_data, &mut out_data, numel as i32))
                .map_err(|e| FlameError::InvalidOperation(format!("Kernel launch failed: {}", e)))?;
        }
        
        // Create output tensor
        Tensor::from_bf16_slice(out_data, shape, device)
    }
}