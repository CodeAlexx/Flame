//! GPU-based gradient operations for efficient gradient modifications
//! Avoids CPU-GPU transfers by performing all operations on GPU

use crate::{Tensor, Result, FlameError, Shape, cuda_kernel_compiler::compile_cuda_kernel};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// GPU kernels for gradient operations
pub const GRADIENT_KERNELS: &str = r#"
extern "C" __global__ void clip_gradient_kernel(
    float* grad,
    const float clip_value,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        grad[idx] = fminf(fmaxf(g, -clip_value), clip_value);
    }
}

extern "C" __global__ void normalize_gradient_kernel(
    float* grad,
    const float* grad_norm_inv,
    const float max_norm,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // grad_norm_inv contains 1/norm if norm > max_norm, else 1.0
        grad[idx] *= grad_norm_inv[0] * max_norm;
    }
}

extern "C" __global__ void compute_l2_norm_kernel(
    const float* grad,
    float* partial_sums,
    const int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and square
    float sum = 0.0f;
    if (idx < n) {
        float g = grad[idx];
        sum = g * g;
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void finalize_l2_norm_kernel(
    const float* partial_sums,
    float* norm,
    const int num_blocks
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    
    // Load partial sums
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Final reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Compute sqrt and write result
    if (tid == 0) {
        norm[0] = sqrtf(sdata[0]);
    }
}

extern "C" __global__ void add_gradient_noise_kernel(
    float* grad,
    const float* noise,
    const float noise_scale,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] += noise[idx] * noise_scale;
    }
}

extern "C" __global__ void scale_gradient_kernel(
    float* grad,
    const float scale,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] *= scale;
    }
}

extern "C" __global__ void adam_update_kernel(
    float* param,
    float* m,
    float* v,
    const float* grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1,
    const float bias_correction2,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        
        // Update biased first moment
        float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
        m[idx] = m_new;
        
        // Update biased second moment
        float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
        v[idx] = v_new;
        
        // Compute bias-corrected moments
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;
        
        // Update parameter
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" __global__ void clamp_tensor_kernel(
    float* data,
    const float min_val,
    const float max_val,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fminf(fmaxf(data[idx], min_val), max_val);
    }
}
"#;

/// GPU-accelerated gradient operations
pub struct CudaGradientOps {
    device: Arc<CudaDevice>,
    kernels_loaded: bool,
}

impl CudaGradientOps {
    /// Create new CUDA gradient operations handler
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let mut ops = Self {
            device,
            kernels_loaded: false,
        };
        ops.ensure_kernels()?;
        Ok(ops)
    }
    
    /// Ensure kernels are loaded
    fn ensure_kernels(&mut self) -> Result<()> {
        if !self.kernels_loaded {
            // Compile CUDA C source to PTX
            let ptx = compile_cuda_kernel(GRADIENT_KERNELS, "gradient_ops")?;
            
            // Load the compiled PTX
            self.device
                .load_ptx(ptx, "gradient_ops", &[
                    "clip_gradient_kernel",
                    "normalize_gradient_kernel",
                    "compute_l2_norm_kernel",
                    "finalize_l2_norm_kernel",
                    "add_gradient_noise_kernel",
                    "scale_gradient_kernel",
                    "adam_update_kernel",
                    "clamp_tensor_kernel",
                ])
                .map_err(|e| FlameError::Cuda(format!("Failed to load gradient kernels: {}", e)))?;
            self.kernels_loaded = true;
        }
        Ok(())
    }
    
    /// Clip gradient values on GPU
    pub fn clip_gradient(&self, grad: &mut Tensor, clip_value: f32) -> Result<()> {
        let n = grad.shape().elem_count();
        let f = self.device
            .get_func("gradient_ops", "clip_gradient_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get clip_gradient kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(f, cfg,
            grad.storage.try_as_slice_f32()?,
            clip_value,
            n as i32
        )?;
        
        self.device.synchronize()?;
        Ok(())
    }
    
    /// Normalize gradient by L2 norm on GPU
    pub fn normalize_gradient(&self, grad: &mut Tensor, max_norm: f32) -> Result<()> {
        let n = grad.shape().elem_count();
        
        // Compute L2 norm
        let norm = self.compute_l2_norm(grad)?;
        
        if norm > max_norm {
            // Normalize gradient
            let scale = max_norm / norm;
            self.scale_gradient(grad, scale)?;
        }
        
        Ok(())
    }
    
    /// Compute L2 norm of tensor on GPU
    pub fn compute_l2_norm(&self, tensor: &Tensor) -> Result<f32> {
        let n = tensor.shape().elem_count();
        let block_size = 256;
        let num_blocks = (n + block_size - 1) / block_size;
        
        // Allocate partial sums
        let partial_sums = crate::tensor::alloc_zeros_from_pool(&self.device, num_blocks)?;
        
        // First reduction pass
        let f1 = self.device
            .get_func("gradient_ops", "compute_l2_norm_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get compute_l2_norm kernel".into()))?;
        
        let cfg1 = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
        };
        
        launch_kernel!(f1, cfg1,
            tensor.storage.try_as_slice_f32()?,
            &partial_sums,
            n as i32
        )?;
        
        // Final reduction
        let norm_result = crate::tensor::alloc_zeros_from_pool(&self.device, 1)?;
        let f2 = self.device
            .get_func("gradient_ops", "finalize_l2_norm_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get finalize_l2_norm kernel".into()))?;
        
        let cfg2 = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (256 * std::mem::size_of::<f32>()) as u32,
        };
        
        launch_kernel!(f2, cfg2,
            &partial_sums,
            &norm_result,
            num_blocks as i32
        )?;
        
        self.device.synchronize()?;
        
        // Copy result back
        let norm_host: Vec<f32> = self.device.dtoh_sync_copy(&norm_result)?;
        
        Ok(norm_host[0])
    }
    
    /// Add noise to gradient on GPU
    pub fn add_gradient_noise(&self, grad: &mut Tensor, noise_scale: f32) -> Result<()> {
        let n = grad.shape().elem_count();
        
        // Generate noise on GPU
        let noise = Tensor::randn(grad.shape().clone(), 0.0, 1.0, self.device.clone())?;
        
        let f = self.device
            .get_func("gradient_ops", "add_gradient_noise_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get add_gradient_noise kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(f, cfg,
            grad.storage.try_as_slice_f32()?,
            noise.storage.try_as_slice_f32()?,
            noise_scale,
            n as i32
        )?;
        
        self.device.synchronize()?;
        Ok(())
    }
    
    /// Scale gradient on GPU
    pub fn scale_gradient(&self, grad: &mut Tensor, scale: f32) -> Result<()> {
        let n = grad.shape().elem_count();
        let f = self.device
            .get_func("gradient_ops", "scale_gradient_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get scale_gradient kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(f, cfg,
            grad.storage.try_as_slice_f32()?,
            scale,
            n as i32
        )?;
        
        self.device.synchronize()?;
        Ok(())
    }
    
    /// Clamp tensor values on GPU
    pub fn clamp_tensor(&self, tensor: &mut Tensor, min_val: f32, max_val: f32) -> Result<()> {
        let n = tensor.shape().elem_count();
        let f = self.device
            .get_func("gradient_ops", "clamp_tensor_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get clamp_tensor kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(f, cfg,
            tensor.storage.try_as_slice_f32()?,
            min_val,
            max_val,
            n as i32
        )?;
        
        self.device.synchronize()?;
        Ok(())
    }
    
    /// Adam optimizer update on GPU
    pub fn adam_update(
        &self,
        param: &mut Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        grad: &Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: usize,
    ) -> Result<()> {
        let n = param.shape().elem_count();
        
        // Compute bias correction
        let bias_correction1 = 1.0 - beta1.powi(step as i32);
        let bias_correction2 = 1.0 - beta2.powi(step as i32);
        
        let f = self.device
            .get_func("gradient_ops", "adam_update_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get adam_update kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(f, cfg,
            param.storage.try_as_slice_f32()?,
            m.storage.try_as_slice_f32()?,
            v.storage.try_as_slice_f32()?,
            grad.storage.try_as_slice_f32()?,
            lr,
            beta1,
            beta2,
            eps,
            bias_correction1,
            bias_correction2,
            n as i32
        )?;
        
        self.device.synchronize()?;
        Ok(())
    }
}

/// Extension methods for Tensor to use GPU operations
impl Tensor {
    /// Clip tensor values on GPU
    pub fn clamp_gpu(&mut self, min: f32, max: f32) -> Result<()> {
        let ops = CudaGradientOps::new(self.device.clone())?;
        ops.clamp_tensor(self, min, max)
    }
    
    /// Normalize by L2 norm on GPU
    pub fn normalize_l2_gpu(&mut self, max_norm: f32) -> Result<()> {
        let ops = CudaGradientOps::new(self.device.clone())?;
        ops.normalize_gradient(self, max_norm)
    }
}
