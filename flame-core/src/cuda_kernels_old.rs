use crate::{Result, FlameError, Tensor, Shape};
use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig, CudaFunction}, 
    nvrtc::{compile_ptx_with_opts, CompileOptions}
};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref KERNEL_CACHE: Mutex<HashMap<String, ()>> = Mutex::new(HashMap::new());
}

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

// Kernel names as static strings
const ADD_KERNEL: &str = "add_kernel";
const MUL_KERNEL: &str = "mul_kernel";
const MUL_SCALAR_KERNEL: &str = "mul_scalar_kernel";
const RELU_KERNEL: &str = "relu_kernel";
const UPDATE_WEIGHTS_KERNEL: &str = "update_weights_kernel";
const SUM_KERNEL: &str = "sum_kernel";
const TRANSPOSE_KERNEL: &str = "transpose_kernel";
const IM2COL_KERNEL: &str = "im2col_kernel";
const COL2IM_KERNEL: &str = "col2im_kernel";

/// CUDA kernel implementations
pub struct CudaKernels;

impl CudaKernels {
    /// Get or compile a kernel
    fn ensure_kernel(device: &Arc<CudaDevice>, name: &'static str, code: &str) -> Result<()> {
        let mut cache = KERNEL_CACHE.lock().unwrap();
        let key = format!("{}-{}", device.ordinal(), name);
        
        if !cache.contains_key(&key) {
            // Compile the kernel
            let opts = CompileOptions {
                ftz: Some(true),
                prec_div: Some(false),
                prec_sqrt: Some(false),
                fmad: Some(true),
                ..Default::default()
            };
            
            let ptx = compile_ptx_with_opts(code, opts)
                .map_err(|e| FlameError::Cuda(format!("CUDA compilation failed: {:?}", e)))?;
            
            device.load_ptx(ptx, name, &[name])
                .map_err(|e| FlameError::Cuda(format!("PTX loading failed: {:?}", e)))?;
            
            cache.insert(key, ());
        }
        
        Ok(())
    }
    
    /// Element-wise addition kernel
    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(FlameError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        
        let kernel_code = r#"
extern "C" __global__ void add_kernel(float *out, const float *a, const float *b, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = a[idx] + b[idx];
    }
}"#;
        
        Self::ensure_kernel(&a.device, ADD_KERNEL, kernel_code)?;
        
        let f = a.device.get_func(ADD_KERNEL, ADD_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get add_kernel".into()))?;
        
        let mut output = Tensor::zeros(a.shape.clone(), a.device.clone())?;
        let numel = a.shape.elem_count() as i32;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, output.data()?, a.storage.as_slice(), b.storage.as_slice(), numel)?;
        
        Ok(output)
    }
    
    /// Element-wise multiplication kernel
    pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(FlameError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        
        let kernel_code = r#"
extern "C" __global__ void mul_kernel(float *out, const float *a, const float *b, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = a[idx] * b[idx];
    }
}"#;
        
        Self::ensure_kernel(&a.device, MUL_KERNEL, kernel_code)?;
        
        let f = a.device.get_func(MUL_KERNEL, MUL_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get mul_kernel".into()))?;
        
        let mut output = Tensor::zeros(a.shape.clone(), a.device.clone())?;
        let numel = a.shape.elem_count() as i32;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, output.data()?, a.storage.as_slice(), b.storage.as_slice(), numel)?;
        
        Ok(output)
    }
    
    /// Scalar multiplication kernel
    pub fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void mul_scalar_kernel(float *out, const float *input, float scalar, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = input[idx] * scalar;
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, MUL_SCALAR_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(MUL_SCALAR_KERNEL, MUL_SCALAR_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get mul_scalar_kernel".into()))?;
        
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let numel = tensor.shape.elem_count() as i32;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, output.data()?, tensor.storage.as_slice(), scalar, numel)?;
        
        Ok(output)
    }
    
    /// ReLU activation kernel
    pub fn relu(tensor: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void relu_kernel(float *out, const float *input, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = fmaxf(0.0f, input[idx]);
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, RELU_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(RELU_KERNEL, RELU_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get relu_kernel".into()))?;
        
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let numel = tensor.shape.elem_count() as i32;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, output.data()?, tensor.storage.as_slice(), numel)?;
        
        Ok(output)
    }
    
    /// In-place weight update kernel
    pub fn update_weights(weights: &mut Tensor, gradients: &Tensor, lr: f32) -> Result<()> {
        if weights.shape != gradients.shape {
            return Err(FlameError::ShapeMismatch {
                expected: weights.shape.clone(),
                got: gradients.shape.clone(),
            });
        }
        
        let kernel_code = r#"
extern "C" __global__ void update_weights_kernel(float *weights, const float *gradients, float lr, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        weights[idx] -= lr * gradients[idx];
    }
}"#;
        
        Self::ensure_kernel(&weights.device, UPDATE_WEIGHTS_KERNEL, kernel_code)?;
        
        let f = weights.device.get_func(UPDATE_WEIGHTS_KERNEL, UPDATE_WEIGHTS_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get update_weights_kernel".into()))?;
        
        let numel = weights.shape.elem_count() as i32;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, weights.data()?, gradients.storage.as_slice(), lr, numel)?;
        
        Ok(())
    }
    
    /// Sum reduction kernel (simple version)
    pub fn sum(tensor: &Tensor) -> Result<Tensor> {
        // For simplicity, we'll use a two-pass approach
        // First pass: partial sums
        // Second pass: final sum
        
        let kernel_code = r#"
extern "C" __global__ void sum_kernel(float *out, const float *input, int numel) {
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
        atomicAdd(out, sdata[0]);
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, SUM_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(SUM_KERNEL, SUM_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get sum_kernel".into()))?;
        
        let mut output = Tensor::zeros(Shape::from_dims(&[1]), tensor.device.clone())?;
        let numel = tensor.shape.elem_count() as i32;
        
        let block_size = 256;
        let grid_size = (numel as u32 + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: block_size * 4, // 4 bytes per float
        };
        
        launch_kernel!(f, cfg, output.data()?, tensor.storage.as_slice(), numel)?;
        
        Ok(output)
    }
    
    /// Transpose kernel for 2D matrices
    pub fn transpose(tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Err(FlameError::InvalidOperation(
                format!("Transpose requires 2D tensor, got {:?}", dims)
            ));
        }
        
        let (rows, cols) = (dims[0], dims[1]);
        
        let kernel_code = r#"
extern "C" __global__ void transpose_kernel(
    float *out, const float *input, 
    int rows, int cols
) {
    // Use shared memory for coalesced access
    extern __shared__ float tile[];
    
    int tile_size = 32;
    int x = blockIdx.x * tile_size + threadIdx.x;
    int y = blockIdx.y * tile_size + threadIdx.y;
    
    int tid_in = threadIdx.y * (tile_size + 1) + threadIdx.x;
    int tid_out = threadIdx.x * (tile_size + 1) + threadIdx.y;
    
    // Load tile to shared memory
    if (x < cols && y < rows) {
        tile[tid_in] = input[y * cols + x];
    }
    __syncthreads();
    
    // Write transposed tile
    x = blockIdx.y * tile_size + threadIdx.x;
    y = blockIdx.x * tile_size + threadIdx.y;
    
    if (x < rows && y < cols) {
        out[y * rows + x] = tile[tid_out];
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, TRANSPOSE_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(TRANSPOSE_KERNEL, TRANSPOSE_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get transpose_kernel".into()))?;
        
        let transposed_shape = Shape::from_dims(&[cols, rows]);
        let mut output = Tensor::zeros(transposed_shape, tensor.device.clone())?;
        
        let tile_size = 32;
        let grid_x = (cols + tile_size - 1) / tile_size;
        let grid_y = (rows + tile_size - 1) / tile_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (tile_size as u32, tile_size as u32, 1),
            shared_mem_bytes: (tile_size * (tile_size + 1) * 4) as u32, // Extra column to avoid bank conflicts
        };
        
        launch_kernel!(f, cfg, output.data()?, tensor.storage.as_slice(), rows as i32, cols as i32)?;
        
        Ok(output)
    }
}
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("Im2col requires 4D tensor [N,C,H,W], got {:?}", dims)
            ));
        }
        
        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        
        let out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        let out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        
        let kernel_code = r#"
struct Im2ColParams {
    int channels;
    int height;
    int width;
    int kernel_h;
    int kernel_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int height_col;
    int width_col;
};

extern "C" __global__ void im2col_kernel(
    const float* data_im,
    float* data_col,
    Im2ColParams params
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = params.channels * params.height_col * params.width_col;
    
    if (index >= total) return;
    
    const int w_col = index % params.width_col;
    const int h_col = (index / params.width_col) % params.height_col;
    const int c_im = index / (params.width_col * params.height_col);
    const int c_col = c_im * params.kernel_h * params.kernel_w;
    
    const int h_offset = h_col * params.stride_h - params.pad_h;
    const int w_offset = w_col * params.stride_w - params.pad_w;
    
    float* data_col_ptr = data_col + (c_col * params.height_col + h_col) * params.width_col + w_col;
    const float* data_im_ptr = data_im + c_im * params.height * params.width;
    
    for (int i = 0; i < params.kernel_h; ++i) {
        for (int j = 0; j < params.kernel_w; ++j) {
            int h_im = h_offset + i * params.dilation_h;
            int w_im = w_offset + j * params.dilation_w;
            
            *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < params.height && w_im < params.width) ?
                data_im_ptr[h_im * params.width + w_im] : 0.0f;
            
            data_col_ptr += params.height_col * params.width_col;
        }
    }
}"#;
        
        Self::ensure_kernel(&input.device, IM2COL_KERNEL, kernel_code)?;
        
        let f = input.device.get_func(IM2COL_KERNEL, IM2COL_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get im2col_kernel".into()))?;
        
        // Output shape: [batch, channels * kernel_h * kernel_w, out_h * out_w]
        let col_shape = Shape::from_dims(&[
            batch,
            channels * kernel_h * kernel_w,
            out_h * out_w
        ]);
        
        let mut output = Tensor::zeros(col_shape, input.device.clone())?;
        
        // Process each batch separately
        for b in 0..batch {
            let batch_offset = b * channels * height * width;
            let col_offset = b * channels * kernel_h * kernel_w * out_h * out_w;
            
            let total_threads = channels * out_h * out_w;
            let block_size = 256;
            let grid_size = (total_threads as u32 + block_size - 1) / block_size;
            
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            
            unsafe {
                let input_slice = input.storage.as_slice().slice(batch_offset..);
                let output_slice = output.data()?.slice_mut(col_offset..);
                
                launch_kernel!(f, cfg,
                    &input_slice,
                    channels as i32,
                    height as i32,
                    width as i32,
                    kernel_h as i32,
                    kernel_w as i32,
                    pad_h as i32,
                    pad_w as i32,
                    stride_h as i32,
                    stride_w as i32,
                    dilation_h as i32,
                    dilation_w as i32,
                    out_h as i32,
                    out_w as i32,
                    &output_slice
                )?;
            }
        }
        
        Ok(output)
    }
}