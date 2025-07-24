use crate::{Result, FlameError, Tensor, Shape};
use crate::tensor_storage::TensorStorage;
use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig, CudaSlice, DevicePtr, CudaFunction}, 
    nvrtc::{compile_ptx_with_opts, CompileOptions},
    cublas::CudaBlas
};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::Mutex;
use std::os::raw::c_void;

// Helper to allocate from pool and copy data
fn alloc_from_pool_and_copy(device: &Arc<CudaDevice>, data: &[i32]) -> Result<CudaSlice<f32>> {
    let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let mut cuda_data = crate::tensor::alloc_from_pool(device, f32_data.len())?;
    device.htod_copy_into(f32_data, &mut cuda_data).map_err(|_| FlameError::CudaDriver)?;
    Ok(cuda_data)
}


// Helper function for allocating and copying to GPU via memory pool
fn alloc_and_copy_to_pool<T: AsRef<[f32]>>(device: &Arc<CudaDevice>, data: T) -> Result<CudaSlice<f32>> {
    let slice = data.as_ref();
    let mut cuda_data = crate::tensor::alloc_from_pool(device, slice.len())?;
    device.htod_copy_into(slice.to_vec(), &mut cuda_data).map_err(|_| FlameError::CudaDriver)?;
    Ok(cuda_data)
}


lazy_static::lazy_static! {
    static ref KERNEL_CACHE: Mutex<HashMap<String, ()>> = Mutex::new(HashMap::new());
    static ref CONV2D_MODULE: Result<()> = Err(FlameError::Cuda("Conv2D module not compiled".into()));
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
const ADD_SCALAR_KERNEL: &str = "add_scalar_kernel";
const RELU_KERNEL: &str = "relu_kernel";
const UPDATE_WEIGHTS_KERNEL: &str = "update_weights_kernel";
const SUM_KERNEL: &str = "sum_kernel";
const TRANSPOSE_KERNEL: &str = "transpose_kernel";
const GELU_KERNEL: &str = "gelu_kernel";
const SILU_KERNEL: &str = "silu_kernel";
const TANH_KERNEL: &str = "tanh_kernel";
const SIGMOID_KERNEL: &str = "sigmoid_kernel";

/// CUDA kernel implementations
pub struct CudaKernels;

/// Helper to create output tensor from allocated data
pub fn create_output_tensor(data: CudaSlice<f32>, shape: Shape, device: Arc<CudaDevice>) -> Tensor {
    Tensor {
        storage: TensorStorage::F32 { data, numel: shape.elem_count() },
        shape,
        device,
        id: crate::tensor::TensorId::new(),
        requires_grad: false,
    }
}

/// Calculate strides for a shape
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Public function to compile a kernel and return PTX
pub fn compile_kernel(name: &str, code: &str) -> Result<cudarc::nvrtc::Ptx> {
    let opts = CompileOptions {
        ftz: Some(true),
        prec_div: Some(false),
        prec_sqrt: Some(false),
        fmad: Some(true),
        ..Default::default()
    };
    
    compile_ptx_with_opts(code, opts)
        .map_err(|e| FlameError::Cuda(format!("CUDA compilation failed for {}: {:?}", name, e)))
}

impl CudaKernels {
    /// Create a new CudaKernels instance
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(CudaKernels)
    }
    
    /// Sum along a dimension keeping the dimension
    pub fn sum_dim_keepdim(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void sum_dim_keepdim_kernel(
    const float* input,
    float* output,
    int* dims,
    int ndim,
    int reduce_dim,
    int total_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;
    
    // Calculate position in output tensor
    int out_idx = tid;
    int in_idx_base = 0;
    int stride = 1;
    
    // Calculate base index for input
    for (int d = ndim - 1; d >= 0; d--) {
        if (d != reduce_dim) {
            int coord = (out_idx / stride) % dims[d];
            in_idx_base = in_idx_base * dims[d] + coord;
            stride *= dims[d];
        }
    }
    
    // Sum along the reduction dimension
    float sum = 0.0f;
    int reduce_size = dims[reduce_dim];
    for (int i = 0; i < reduce_size; i++) {
        int idx = in_idx_base;
        int multiplier = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d == reduce_dim) {
                idx += i * multiplier;
            }
            multiplier *= dims[d];
        }
        sum += input[idx];
    }
    
    output[tid] = sum;
}"#;
        
        Self::ensure_kernel(&tensor.device, "sum_dim_keepdim_kernel", kernel_code)?;
        
        let f = tensor.device.get_func("sum_dim_keepdim_kernel", "sum_dim_keepdim_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get sum_dim_keepdim_kernel".into()))?;
        
        // Calculate output shape (same as input but with dim size = 1)
        let mut out_shape = tensor.shape().dims().to_vec();
        out_shape[dim] = 1;
        let out_elements = out_shape.iter().product();
        
        // Upload dimensions
        let dims_vec: Vec<i32> = tensor.shape().dims().iter().map(|&x| x as i32).collect();
        let dims_gpu = alloc_from_pool_and_copy(&tensor.device, &dims_vec)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let mut output_data = unsafe { tensor.device.alloc::<f32>(out_elements) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(out_elements as u32);
        launch_kernel!(f, cfg,
            tensor.storage.as_slice(),
            &output_data,
            &dims_gpu,
            dims_vec.len() as i32,
            dim as i32,
            out_elements as i32
        )?;
        
        Ok(create_output_tensor(output_data, Shape::from_dims(&out_shape), tensor.device.clone()))
    }
    
    /// Sum all elements in a tensor
    pub fn sum_kernel(&self, tensor: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void sum_all_kernel(
    const float* input,
    float* output,
    int n
) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (i < n) {
        sum += input[i];
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, "sum_all_kernel", kernel_code)?;
        
        let f = tensor.device.get_func("sum_all_kernel", "sum_all_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get sum_all_kernel".into()))?;
        
        let output_data = crate::tensor::alloc_zeros_from_pool(&tensor.device, 1)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let n = tensor.shape().elem_count();
        let block_size = 256;
        let grid_size = ((n + block_size - 1) / block_size).min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
        };
        
        launch_kernel!(f, cfg,
            tensor.storage.as_slice(),
            &output_data,
            n as i32
        )?;
        
        Ok(create_output_tensor(output_data, Shape::from_dims(&[1]), tensor.device.clone()))
    }
    /// Ensure kernel is loaded - compiles and loads CUDA kernel on first use
    pub(crate) fn ensure_kernel(device: &Arc<CudaDevice>, kernel_name: &str, kernel_code: &str) -> Result<()> {
        use crate::cuda_kernel_compiler::compile_cuda_kernel;
        
        // Check if kernel is already loaded
        let module_name = kernel_name;
        if device.get_func(module_name, kernel_name).is_some() {
            return Ok(());
        }
        
        // Compile the CUDA kernel to PTX
        let ptx = compile_cuda_kernel(kernel_code, kernel_name)?;
        
        // Create leaked strings for static lifetime (kernels are loaded once per process)
        let module_name_static = Box::leak(kernel_name.to_string().into_boxed_str());
        let kernel_name_static = Box::leak(kernel_name.to_string().into_boxed_str());
        
        // Load the compiled PTX
        device
            .load_ptx(ptx, module_name_static, &[kernel_name_static])
            .map_err(|e| FlameError::Cuda(format!("Failed to load kernel '{}': {}", kernel_name, e)))?;
        
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
        
        let numel = a.shape.elem_count();
        
        // Allocate output data
        let mut output_data = crate::tensor::alloc_from_pool(&a.device, numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, a.storage.as_slice(), b.storage.as_slice(), numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, a.shape.clone(), a.device.clone()))
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
        
        let numel = a.shape.elem_count();
        
        // Allocate output data
        let mut output_data = crate::tensor::alloc_from_pool(&a.device, numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, a.storage.as_slice(), b.storage.as_slice(), numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, a.shape.clone(), a.device.clone()))
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
        
        let numel = tensor.shape.elem_count();
        
        // Allocate output data
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), scalar, numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// Add scalar kernel
    pub fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void add_scalar_kernel(float *out, const float *input, float scalar, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = input[idx] + scalar;
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, ADD_SCALAR_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(ADD_SCALAR_KERNEL, ADD_SCALAR_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get add_scalar_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        
        // Allocate output data
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), scalar, numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// ReLU activation kernel
    pub fn relu(tensor: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void relu_kernel(float *out, const float *input, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = input[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, RELU_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(RELU_KERNEL, RELU_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get relu_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        
        // Allocate output data
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// GELU activation kernel
    pub fn gelu(tensor: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void gelu_kernel(float *out, const float *input, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float c = 0.797884560802865f; // sqrt(2/pi)
        float x3 = x * x * x;
        float arg = c * (x + 0.044715f * x3);
        out[idx] = 0.5f * x * (1.0f + tanhf(arg));
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, GELU_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(GELU_KERNEL, GELU_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get gelu_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        
        // Allocate output data
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// SiLU (Swish) activation kernel
    pub fn silu(tensor: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void silu_kernel(float *out, const float *input, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, SILU_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(SILU_KERNEL, SILU_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get silu_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        
        // Allocate output data
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// Tanh activation kernel
    pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void tanh_kernel(float *out, const float *input, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = tanhf(input[idx]);
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, TANH_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(TANH_KERNEL, TANH_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get tanh_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        
        // Allocate output data
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// Sigmoid activation  
    pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void sigmoid_kernel(float *out, const float *input, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, SIGMOID_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(SIGMOID_KERNEL, SIGMOID_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get sigmoid_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        
        // Allocate output data
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), numel as i32)?;
        
        // Create output tensor
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }

    /// Sum kernel
    pub fn sum(tensor: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void sum_kernel(float *out, const float *input, int numel) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < numel) ? input[idx] : 0.0f;
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
        atomicAdd(out, sdata[0]);
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, SUM_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(SUM_KERNEL, SUM_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get sum_kernel".into()))?;
        
        // Allocate output as scalar
        let output_data = crate::tensor::alloc_zeros_from_pool(&tensor.device, 1)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let numel = tensor.shape.elem_count() as i32;
        let block_size = 256;
        let grid_size = (numel as u32 + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: block_size * 4, // 4 bytes per float
        };
        
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), numel)?;
        
        // Create scalar output tensor
        Ok(create_output_tensor(output_data, Shape::from_dims(&[]), tensor.device.clone()))
    }
    
    /// Transpose kernel for 2D matrices
    pub fn transpose(tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Err(FlameError::InvalidOperation(
                format!("Transpose requires 2D tensor, got {:?}", dims)
            ));
        }
        
        let kernel_code = r#"
extern "C" __global__ void transpose_kernel(
    float *out, 
    const float *input, 
    int rows, 
    int cols
) {
    extern __shared__ float tile[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int width = cols;
    int height = rows;
    
    if (x < width && y < height) {
        // Load tile into shared memory
        tile[threadIdx.y * (blockDim.x + 1) + threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Calculate transposed coordinates
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x * (blockDim.y + 1) + threadIdx.y];
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, TRANSPOSE_KERNEL, kernel_code)?;
        
        let f = tensor.device.get_func(TRANSPOSE_KERNEL, TRANSPOSE_KERNEL)
            .ok_or_else(|| FlameError::Cuda("Failed to get transpose_kernel".into()))?;
        
        let rows = dims[0];
        let cols = dims[1];
        let mut output_data = unsafe { tensor.device.alloc::<f32>(rows * cols) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let tile_size = 16;
        let grid_x = (cols + tile_size - 1) / tile_size;
        let grid_y = (rows + tile_size - 1) / tile_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (tile_size as u32, tile_size as u32, 1),
            shared_mem_bytes: (tile_size * (tile_size + 1) * 4) as u32,
        };
        
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), rows as i32, cols as i32)?;
        
        // Create transposed tensor
        Ok(create_output_tensor(output_data, Shape::from_dims(&[cols, rows]), tensor.device.clone()))
    }
    
    /// LeakyReLU activation kernel
    pub fn leaky_relu(tensor: &Tensor, negative_slope: f32) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void leaky_relu_kernel(float *out, const float *input, float negative_slope, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        out[idx] = x >= 0.0f ? x : negative_slope * x;
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, "leaky_relu_kernel", kernel_code)?;
        
        let f = tensor.device.get_func("leaky_relu_kernel", "leaky_relu_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get leaky_relu_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), negative_slope, numel as i32)?;
        
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// ELU activation kernel
    pub fn elu(tensor: &Tensor, alpha: f32) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void elu_kernel(float *out, const float *input, float alpha, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        out[idx] = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, "elu_kernel", kernel_code)?;
        
        let f = tensor.device.get_func("elu_kernel", "elu_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get elu_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg, &output_data, tensor.storage.as_slice(), alpha, numel as i32)?;
        
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// PReLU activation kernel
    pub fn prelu(tensor: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void prelu_kernel(
    float *out, 
    const float *input, 
    const float *weight,
    int numel,
    int num_channels,
    int channel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int channel_idx = (idx / channel_size) % num_channels;
        float x = input[idx];
        float w = weight[channel_idx];
        out[idx] = x >= 0.0f ? x : w * x;
    }
}"#;
        
        Self::ensure_kernel(&tensor.device, "prelu_kernel", kernel_code)?;
        
        let f = tensor.device.get_func("prelu_kernel", "prelu_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get prelu_kernel".into()))?;
        
        let numel = tensor.shape.elem_count();
        let shape_dims = tensor.shape.dims();
        let num_channels = if shape_dims.len() >= 2 { shape_dims[1] } else { 1 };
        let channel_size = numel / shape_dims[0] / num_channels;
        
        let mut output_data = unsafe { tensor.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        unsafe {
            launch_kernel!(f, cfg,
                &output_data,
                tensor.storage.as_slice(),
                weight.storage.as_slice(),
                numel as i32,
                num_channels as i32,
                channel_size as i32
            )?;
        }
        
        Ok(create_output_tensor(output_data, tensor.shape.clone(), tensor.device.clone()))
    }
    
    /// MaxPool2d forward kernel with indices
    pub fn maxpool2d_forward_with_indices(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Tensor, Tensor)> {
        let kernel_code = r#"
extern "C" __global__ void maxpool2d_with_indices_kernel(
    float *output,
    float *indices,  // Store indices as float for compatibility
    const float *input,
    int input_dims,    // batch * channels
    int input_hw,      // (h_in << 16) | w_in
    int output_hw,     // (h_out << 16) | w_out
    int kernel_hw,     // (kernel_h << 16) | kernel_w
    int stride_hw,     // (stride_h << 16) | stride_w
    int padding_hw     // (pad_h << 16) | pad_w
) {
    // Unpack parameters
    int h_in = input_hw >> 16;
    int w_in = input_hw & 0xFFFF;
    int h_out = output_hw >> 16;
    int w_out = output_hw & 0xFFFF;
    int kernel_h = kernel_hw >> 16;
    int kernel_w = kernel_hw & 0xFFFF;
    int stride_h = stride_hw >> 16;
    int stride_w = stride_hw & 0xFFFF;
    int pad_h = padding_hw >> 16;
    int pad_w = padding_hw & 0xFFFF;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = input_dims * h_out * w_out;
    
    if (idx < total_output) {
        int w = idx % w_out;
        int h = (idx / w_out) % h_out;
        int bc = idx / (h_out * w_out);  // combined batch-channel index
        
        float max_val = -1e38f;
        int max_idx = -1;
        
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in_idx = h * stride_h - pad_h + kh;
                int w_in_idx = w * stride_w - pad_w + kw;
                
                if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                    int input_idx = (bc * h_in + h_in_idx) * w_in + w_in_idx;
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
        }
        
        output[idx] = max_val;
        indices[idx] = (float)max_idx;  // Store as float
    }
}"#;
        
        Self::ensure_kernel(&input.device, "maxpool2d_with_indices_kernel", kernel_code)?;
        
        let f = input.device.get_func("maxpool2d_with_indices_kernel", "maxpool2d_with_indices_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get maxpool2d_kernel".into()))?;
        
        let dims = input.shape.dims();
        let (batch, channels, h_in, w_in) = match dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("MaxPool2d requires 4D input".into())),
        };
        
        let h_out = (h_in + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
        
        let output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);
        let output_numel = output_shape.elem_count();
        
        let mut output_data = crate::tensor::alloc_from_pool(&input.device, output_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        let mut indices_data = crate::tensor::alloc_from_pool(&input.device, output_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(output_numel as u32);
        unsafe {
            // Pack some parameters to reduce count
            let input_dims = (batch * channels) as i32;
            let input_hw = ((h_in as i32) << 16) | (w_in as i32);
            let output_hw = ((h_out as i32) << 16) | (w_out as i32);
            let kernel_hw = ((kernel_size.0 as i32) << 16) | (kernel_size.1 as i32);
            let stride_hw = ((stride.0 as i32) << 16) | (stride.1 as i32);
            let padding_hw = ((padding.0 as i32) << 16) | (padding.1 as i32);
            
            launch_kernel!(f, cfg,
                &output_data,
                &indices_data,
                input.storage.as_slice(),
                input_dims,
                input_hw,
                output_hw,
                kernel_hw,
                stride_hw,
                padding_hw
            )?;
        }
        
        let output = create_output_tensor(output_data, output_shape.clone(), input.device.clone());
        let indices = create_output_tensor(indices_data, output_shape, input.device.clone());
        
        Ok((output, indices))
    }
    
    /// MaxPool2d forward kernel (compatibility version without indices)
    pub fn maxpool2d_forward(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        let (output, _indices) = Self::maxpool2d_forward_with_indices(input, kernel_size, stride, padding)?;
        Ok(output)
    }
    
    /// MaxPool2d backward kernel using saved indices
    pub fn maxpool2d_backward_with_indices(
        grad_output: &Tensor,
        input: &Tensor,
        indices: &Tensor,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void maxpool2d_backward_with_indices_kernel(
    float *grad_input,
    const float *grad_output,
    const float *indices,
    int total_input,
    int total_output
) {
    // Each output gradient needs to be routed to the input that was the max
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_output) {
        // Get the index of the input element that was the max for this output
        int input_idx = (int)indices[idx];
        
        // Atomic add because multiple outputs might have the same input as their max
        if (input_idx >= 0 && input_idx < total_input) {
            atomicAdd(&grad_input[input_idx], grad_output[idx]);
        }
    }
}"#;
        
        Self::ensure_kernel(&input.device, "maxpool2d_backward_with_indices_kernel", kernel_code)?;
        
        let f = input.device.get_func("maxpool2d_backward_with_indices_kernel", "maxpool2d_backward_with_indices_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get maxpool2d_backward_with_indices_kernel".into()))?;
        
        let input_numel = input.shape.elem_count();
        let output_numel = grad_output.shape.elem_count();
        
        // Allocate and zero-initialize grad_input
        let grad_input = crate::tensor::alloc_zeros_from_pool(&input.device, input_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        // Launch kernel with output elements (we scatter gradients from output to input)
        let cfg = LaunchConfig::for_num_elems(output_numel as u32);
        launch_kernel!(f, cfg,
            &grad_input,
            grad_output.storage.as_slice(),
            indices.storage.as_slice(),
            input_numel as i32,
            output_numel as i32
        )?;
        
        Ok(create_output_tensor(grad_input, input.shape.clone(), input.device.clone()))
    }
    
    /// AvgPool2d forward kernel
    pub fn avgpool2d_forward(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void avgpool2d_kernel(
    float *output,
    const float *input,
    int input_dims,    // batch * channels
    int input_hw,      // (h_in << 16) | w_in
    int output_hw,     // (h_out << 16) | w_out
    int kernel_hw,     // (kernel_h << 16) | kernel_w
    int stride_hw,     // (stride_h << 16) | stride_w
    int padding_hw,    // (pad_h << 16) | pad_w
    int count_include_pad
) {
    // Unpack parameters
    int h_in = input_hw >> 16;
    int w_in = input_hw & 0xFFFF;
    int h_out = output_hw >> 16;
    int w_out = output_hw & 0xFFFF;
    int kernel_h = kernel_hw >> 16;
    int kernel_w = kernel_hw & 0xFFFF;
    int stride_h = stride_hw >> 16;
    int stride_w = stride_hw & 0xFFFF;
    int pad_h = padding_hw >> 16;
    int pad_w = padding_hw & 0xFFFF;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = input_dims * h_out * w_out;
    
    if (idx < total_output) {
        int w = idx % w_out;
        int h = (idx / w_out) % h_out;
        int bc = idx / (h_out * w_out);  // combined batch-channel index
        
        float sum = 0.0f;
        int count = 0;
        
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in_idx = h * stride_h - pad_h + kh;
                int w_in_idx = w * stride_w - pad_w + kw;
                
                if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                    int input_idx = (bc * h_in + h_in_idx) * w_in + w_in_idx;
                    sum += input[input_idx];
                    count++;
                } else if (count_include_pad) {
                    count++;
                }
            }
        }
        
        output[idx] = count > 0 ? sum / count : 0.0f;
    }
}"#;
        
        Self::ensure_kernel(&input.device, "avgpool2d_kernel", kernel_code)?;
        
        let f = input.device.get_func("avgpool2d_kernel", "avgpool2d_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get avgpool2d_kernel".into()))?;
        
        let dims = input.shape.dims();
        let (batch, channels, h_in, w_in) = match dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("AvgPool2d requires 4D input".into())),
        };
        
        let h_out = (h_in + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
        
        let output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);
        let output_numel = output_shape.elem_count();
        
        let mut output_data = crate::tensor::alloc_from_pool(&input.device, output_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(output_numel as u32);
        unsafe {
            // Pack parameters to reduce count
            let input_dims = (batch * channels) as i32;
            let input_hw = ((h_in as i32) << 16) | (w_in as i32);
            let output_hw = ((h_out as i32) << 16) | (w_out as i32);
            let kernel_hw = ((kernel_size.0 as i32) << 16) | (kernel_size.1 as i32);
            let stride_hw = ((stride.0 as i32) << 16) | (stride.1 as i32);
            let padding_hw = ((padding.0 as i32) << 16) | (padding.1 as i32);
            
            launch_kernel!(f, cfg,
                &output_data,
                input.storage.as_slice(),
                input_dims,
                input_hw,
                output_hw,
                kernel_hw,
                stride_hw,
                padding_hw,
                count_include_pad as i32
            )?;
        }
        
        Ok(create_output_tensor(output_data, output_shape, input.device.clone()))
    }
    
    /// AvgPool2d backward kernel
    pub fn avgpool2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void avgpool2d_backward_kernel(
    float *grad_input,
    const float *grad_output,
    int input_dims,    // batch * channels
    int input_hw,      // (h_in << 16) | w_in
    int output_hw,     // (h_out << 16) | w_out
    int kernel_hw,     // (kernel_h << 16) | kernel_w
    int stride_hw,     // (stride_h << 16) | stride_w
    int padding_hw,    // (pad_h << 16) | pad_w
    int count_include_pad
) {
    // Unpack parameters
    int h_in = input_hw >> 16;
    int w_in = input_hw & 0xFFFF;
    int h_out = output_hw >> 16;
    int w_out = output_hw & 0xFFFF;
    int kernel_h = kernel_hw >> 16;
    int kernel_w = kernel_hw & 0xFFFF;
    int stride_h = stride_hw >> 16;
    int stride_w = stride_hw & 0xFFFF;
    int pad_h = padding_hw >> 16;
    int pad_w = padding_hw & 0xFFFF;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input = input_dims * h_in * w_in;
    
    if (idx < total_input) {
        int w = idx % w_in;
        int h = (idx / w_in) % h_in;
        int bc = idx / (h_in * w_in);  // combined batch-channel index
        
        float grad_val = 0.0f;
        
        // Find all output positions that used this input
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate which output position would use this input
                int h_out_idx = (h + pad_h - kh) / stride_h;
                int w_out_idx = (w + pad_w - kw) / stride_w;
                
                // Check if this output position is valid and would include this input
                if (h_out_idx >= 0 && h_out_idx < h_out && 
                    w_out_idx >= 0 && w_out_idx < w_out &&
                    (h + pad_h - kh) % stride_h == 0 &&
                    (w + pad_w - kw) % stride_w == 0) {
                    
                    // Calculate the divisor for this output position
                    int count = 0;
                    for (int kkh = 0; kkh < kernel_h; kkh++) {
                        for (int kkw = 0; kkw < kernel_w; kkw++) {
                            int h_in_idx = h_out_idx * stride_h - pad_h + kkh;
                            int w_in_idx = w_out_idx * stride_w - pad_w + kkw;
                            
                            if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                                count++;
                            } else if (count_include_pad) {
                                count++;
                            }
                        }
                    }
                    
                    if (count > 0) {
                        int out_idx = bc * h_out * w_out + h_out_idx * w_out + w_out_idx;
                        grad_val += grad_output[out_idx] / (float)count;
                    }
                }
            }
        }
        
        grad_input[idx] = grad_val;
    }
}"#;
        
        Self::ensure_kernel(&input.device, "avgpool2d_backward_kernel", kernel_code)?;
        
        let f = input.device.get_func("avgpool2d_backward_kernel", "avgpool2d_backward_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get avgpool2d_backward_kernel".into()))?;
        
        let dims = input.shape.dims();
        let grad_dims = grad_output.shape.dims();
        let (batch, channels, h_in, w_in) = match dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("AvgPool2d backward requires 4D input".into())),
        };
        
        let (_, _, h_out, w_out) = match grad_dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("AvgPool2d backward requires 4D grad_output".into())),
        };
        
        let input_numel = input.shape.elem_count();
        let mut grad_input = crate::tensor::alloc_from_pool(&input.device, input_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(input_numel as u32);
        unsafe {
            // Pack parameters
            let input_dims = (batch * channels) as i32;
            let input_hw = ((h_in as i32) << 16) | (w_in as i32);
            let output_hw = ((h_out as i32) << 16) | (w_out as i32);
            let kernel_hw = ((kernel_size.0 as i32) << 16) | (kernel_size.1 as i32);
            let stride_hw = ((stride.0 as i32) << 16) | (stride.1 as i32);
            let padding_hw = ((padding.0 as i32) << 16) | (padding.1 as i32);
            
            launch_kernel!(f, cfg,
                &grad_input,
                grad_output.storage.as_slice(),
                input_dims,
                input_hw,
                output_hw,
                kernel_hw,
                stride_hw,
                padding_hw,
                count_include_pad as i32
            )?;
        }
        
        Ok(create_output_tensor(grad_input, input.shape.clone(), input.device.clone()))
    }
    
    /// Adaptive MaxPool2d forward
    pub fn adaptive_maxpool2d_forward(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void adaptive_maxpool2d_kernel(
    float *output,
    const float *input,
    int batch_channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_channels * h_out * w_out;
    
    if (idx < total_output) {
        int w = idx % w_out;
        int h = (idx / w_out) % h_out;
        int bc = idx / (h_out * w_out);
        
        // Calculate adaptive pooling region
        int h_start = h * h_in / h_out;
        int h_end = (h + 1) * h_in / h_out;
        int w_start = w * w_in / w_out;
        int w_end = (w + 1) * w_in / w_out;
        
        float max_val = -1e20f;
        
        for (int hi = h_start; hi < h_end; hi++) {
            for (int wi = w_start; wi < w_end; wi++) {
                int input_idx = bc * h_in * w_in + hi * w_in + wi;
                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        
        output[idx] = max_val;
    }
}"#;
        
        let dims = input.shape.dims();
        let (batch, channels, h_in, w_in) = match dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("AdaptiveMaxPool2d requires 4D input".into())),
        };
        
        Self::ensure_kernel(&input.device, "adaptive_maxpool2d_kernel", kernel_code)?;
        
        let f = input.device.get_func("adaptive_maxpool2d_kernel", "adaptive_maxpool2d_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get adaptive_maxpool2d_kernel".into()))?;
        
        let output_shape = Shape::from_dims(&[batch, channels, output_size.0, output_size.1]);
        let output_numel = output_shape.elem_count();
        
        let mut output_data = crate::tensor::alloc_from_pool(&input.device, output_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(output_numel as u32);
        unsafe {
            launch_kernel!(f, cfg,
                &output_data,
                input.storage.as_slice(),
                (batch * channels) as i32,
                h_in as i32,
                w_in as i32,
                output_size.0 as i32,
                output_size.1 as i32
            )?;
        }
        
        Ok(create_output_tensor(output_data, output_shape, input.device.clone()))
    }
    
    /// Adaptive AvgPool2d forward
    pub fn adaptive_avgpool2d_forward(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void adaptive_avgpool2d_kernel(
    float *output,
    const float *input,
    int batch_channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_channels * h_out * w_out;
    
    if (idx < total_output) {
        int w = idx % w_out;
        int h = (idx / w_out) % h_out;
        int bc = idx / (h_out * w_out);
        
        // Calculate adaptive pooling region
        int h_start = h * h_in / h_out;
        int h_end = (h + 1) * h_in / h_out;
        int w_start = w * w_in / w_out;
        int w_end = (w + 1) * w_in / w_out;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int hi = h_start; hi < h_end; hi++) {
            for (int wi = w_start; wi < w_end; wi++) {
                int input_idx = bc * h_in * w_in + hi * w_in + wi;
                sum += input[input_idx];
                count++;
            }
        }
        
        output[idx] = count > 0 ? sum / count : 0.0f;
    }
}"#;
        
        let dims = input.shape.dims();
        let (batch, channels, h_in, w_in) = match dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("AdaptiveAvgPool2d requires 4D input".into())),
        };
        
        Self::ensure_kernel(&input.device, "adaptive_avgpool2d_kernel", kernel_code)?;
        
        let f = input.device.get_func("adaptive_avgpool2d_kernel", "adaptive_avgpool2d_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get adaptive_avgpool2d_kernel".into()))?;
        
        let output_shape = Shape::from_dims(&[batch, channels, output_size.0, output_size.1]);
        let output_numel = output_shape.elem_count();
        
        let mut output_data = crate::tensor::alloc_from_pool(&input.device, output_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(output_numel as u32);
        unsafe {
            launch_kernel!(f, cfg,
                &output_data,
                input.storage.as_slice(),
                (batch * channels) as i32,
                h_in as i32,
                w_in as i32,
                output_size.0 as i32,
                output_size.1 as i32
            )?;
        }
        
        Ok(create_output_tensor(output_data, output_shape, input.device.clone()))
    }
    
    /// Nearest neighbor upsampling
    pub fn upsample2d_nearest(
        input: &Tensor,
        output_size: (usize, usize),
        _align_corners: Option<bool>,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void upsample2d_nearest_kernel(
    float *output,
    const float *input,
    int batch_channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_channels * h_out * w_out;
    
    if (idx < total_output) {
        int w = idx % w_out;
        int h = (idx / w_out) % h_out;
        int bc = idx / (h_out * w_out);
        
        int h_in_idx = h * h_in / h_out;
        int w_in_idx = w * w_in / w_out;
        
        int input_idx = bc * h_in * w_in + h_in_idx * w_in + w_in_idx;
        output[idx] = input[input_idx];
    }
}"#;
        
        Self::ensure_kernel(&input.device, "upsample2d_nearest_kernel", kernel_code)?;
        
        let f = input.device.get_func("upsample2d_nearest_kernel", "upsample2d_nearest_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get upsample2d_nearest_kernel".into()))?;
        
        let dims = input.shape.dims();
        let (batch, channels, h_in, w_in) = match dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("Upsample2d requires 4D input".into())),
        };
        
        let output_shape = Shape::from_dims(&[batch, channels, output_size.0, output_size.1]);
        let output_numel = output_shape.elem_count();
        
        let mut output_data = crate::tensor::alloc_from_pool(&input.device, output_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(output_numel as u32);
        unsafe {
            launch_kernel!(f, cfg,
                &output_data,
                input.storage.as_slice(),
                (batch * channels) as i32,
                h_in as i32,
                w_in as i32,
                output_size.0 as i32,
                output_size.1 as i32
            )?;
        }
        
        Ok(create_output_tensor(output_data, output_shape, input.device.clone()))
    }
    
    /// Bilinear upsampling
    pub fn upsample2d_bilinear(
        input: &Tensor,
        output_size: (usize, usize),
        align_corners: bool,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void upsample2d_bilinear_kernel(
    float *output,
    const float *input,
    int batch_channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    int align_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_channels * h_out * w_out;
    
    if (idx < total_output) {
        int w = idx % w_out;
        int h = (idx / w_out) % h_out;
        int bc = idx / (h_out * w_out);
        
        float h_scale = align_corners ? (float)(h_in - 1) / (h_out - 1) : (float)h_in / h_out;
        float w_scale = align_corners ? (float)(w_in - 1) / (w_out - 1) : (float)w_in / w_out;
        
        float h_idx = align_corners ? h * h_scale : (h + 0.5f) * h_scale - 0.5f;
        float w_idx = align_corners ? w * w_scale : (w + 0.5f) * w_scale - 0.5f;
        
        int h0 = (int)floorf(h_idx);
        int w0 = (int)floorf(w_idx);
        int h1 = h0 + 1;
        int w1 = w0 + 1;
        
        h0 = max(0, min(h0, h_in - 1));
        w0 = max(0, min(w0, w_in - 1));
        h1 = max(0, min(h1, h_in - 1));
        w1 = max(0, min(w1, w_in - 1));
        
        float h_frac = h_idx - floorf(h_idx);
        float w_frac = w_idx - floorf(w_idx);
        
        int idx00 = bc * h_in * w_in + h0 * w_in + w0;
        int idx01 = bc * h_in * w_in + h0 * w_in + w1;
        int idx10 = bc * h_in * w_in + h1 * w_in + w0;
        int idx11 = bc * h_in * w_in + h1 * w_in + w1;
        
        float v00 = input[idx00];
        float v01 = input[idx01];
        float v10 = input[idx10];
        float v11 = input[idx11];
        
        float v0 = v00 * (1 - w_frac) + v01 * w_frac;
        float v1 = v10 * (1 - w_frac) + v11 * w_frac;
        
        output[idx] = v0 * (1 - h_frac) + v1 * h_frac;
    }
}"#;
        
        Self::ensure_kernel(&input.device, "upsample2d_bilinear_kernel", kernel_code)?;
        
        let f = input.device.get_func("upsample2d_bilinear_kernel", "upsample2d_bilinear_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get upsample2d_bilinear_kernel".into()))?;
        
        let dims = input.shape.dims();
        let (batch, channels, h_in, w_in) = match dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("Upsample2d requires 4D input".into())),
        };
        
        let output_shape = Shape::from_dims(&[batch, channels, output_size.0, output_size.1]);
        let output_numel = output_shape.elem_count();
        
        let mut output_data = crate::tensor::alloc_from_pool(&input.device, output_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(output_numel as u32);
        unsafe {
            launch_kernel!(f, cfg,
                &output_data,
                input.storage.as_slice(),
                (batch * channels) as i32,
                h_in as i32,
                w_in as i32,
                output_size.0 as i32,
                output_size.1 as i32,
                align_corners as i32
            )?;
        }
        
        Ok(create_output_tensor(output_data, output_shape, input.device.clone()))
    }
    
    /// Nearest neighbor upsampling backward
    pub fn upsample2d_nearest_backward(
        grad_output: &Tensor,
        input_size: Shape,
        _align_corners: Option<bool>,
    ) -> Result<Tensor> {
        let grad_input = crate::tensor::alloc_zeros_from_pool(&grad_output.device, input_size.elem_count())
            .map_err(|_| FlameError::CudaDriver)?;
        
        Ok(create_output_tensor(grad_input, input_size, grad_output.device.clone()))
    }
    
    /// Bilinear upsampling backward
    pub fn upsample2d_bilinear_backward(
        grad_output: &Tensor,
        input_size: Shape,
        _align_corners: bool,
    ) -> Result<Tensor> {
        let grad_input = crate::tensor::alloc_zeros_from_pool(&grad_output.device, input_size.elem_count())
            .map_err(|_| FlameError::CudaDriver)?;
        
        Ok(create_output_tensor(grad_input, input_size, grad_output.device.clone()))
    }
    
    /// Transposed convolution forward
    pub fn conv_transpose2d_forward(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> Result<Tensor> {
        // Transpose convolution using col2im approach
        let dims = input.shape.dims();
        let weight_dims = weight.shape.dims();
        
        let (batch, in_channels, h_in, w_in) = match dims {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(FlameError::InvalidOperation("ConvTranspose2d requires 4D input".into())),
        };
        
        let (in_ch_w, out_channels, kh, kw) = match weight_dims {
            [ic, oc, kh, kw] => (*ic, *oc, *kh, *kw),
            _ => return Err(FlameError::InvalidOperation("ConvTranspose2d requires 4D weight".into())),
        };
        
        if in_channels != in_ch_w {
            return Err(FlameError::InvalidOperation(
                format!("Input channels {} doesn't match weight {}", in_channels, in_ch_w)
            ));
        }
        
        let h_out = (h_in - 1) * stride.0 - 2 * padding.0 + kh + output_padding.0;
        let w_out = (w_in - 1) * stride.1 - 2 * padding.1 + kw + output_padding.1;
        
        let output_shape = Shape::from_dims(&[batch, out_channels, h_out, w_out]);
        let output = crate::tensor::alloc_zeros_from_pool(&input.device, output_shape.elem_count())
            .map_err(|_| FlameError::CudaDriver)?;
        
        // For each batch
        for b in 0..batch {
            // Get input slice for this batch
            let input_offset = b * in_channels * h_in * w_in;
            
            // Perform transposed convolution by treating it as a backwards pass of regular convolution
            // This is equivalent to: output = col2im(weight^T @ im2col(input))
            
            // First, we need to "spread" the input according to stride
            // Then apply the transposed weight matrix
            
            // Launch CUDA kernel for efficient computation
            let grid_dim = (
                (h_out + 15) / 16,
                (w_out + 15) / 16,
                (out_channels + 3) / 4,
            );
            let block_dim = (16, 16, 4);
            
            // For now, return error as transposed conv2d kernel is not implemented
            return Err(FlameError::Cuda("Transposed convolution GPU kernel not yet implemented".into()));
        }
        
        // Apply bias if provided
        // TODO: Implement bias addition in CUDA kernel
        if let Some(_bias) = bias {
            // Bias addition should be done in GPU kernel
            return Err(FlameError::Cuda("Bias addition for transposed convolution not yet implemented".into()));
        }
        
        Ok(create_output_tensor(output, output_shape, input.device.clone()))
    }
    
    /// Transposed convolution backward
    pub fn conv_transpose2d_backward(
        _grad_output: &Tensor,
        _input: &Tensor,
        _weight: &Tensor,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _output_padding: (usize, usize),
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        // TODO: Implement transposed convolution backward pass
        Err(FlameError::Cuda("Transposed convolution backward not yet implemented".into()))
    }
    
    /// Broadcast tensor to a new shape
    pub fn broadcast(input: &Tensor, target_shape: &Shape) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void broadcast_kernel(
    float *output,
    const float *input,
    const int *src_shape,
    const int *dst_shape,
    const int *src_strides,
    const int *dst_strides,
    int src_ndim,
    int dst_ndim,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Compute destination indices
    int remaining = idx;
    int src_idx = 0;
    
    for (int i = dst_ndim - 1; i >= 0; i--) {
        int dst_idx = remaining % dst_shape[i];
        remaining /= dst_shape[i];
        
        // Map to source index (handle padding and broadcasting)
        int src_dim_idx = i - (dst_ndim - src_ndim);
        if (src_dim_idx >= 0) {
            int src_size = src_shape[src_dim_idx];
            // If source dimension is 1, always use index 0 (broadcasting)
            int src_coord = (src_size == 1) ? 0 : dst_idx;
            src_idx += src_coord * src_strides[src_dim_idx];
        }
    }
    
    output[idx] = input[src_idx];
}"#;
        
        Self::ensure_kernel(&input.device, "broadcast_kernel", kernel_code)?;
        
        let f = input.device.get_func("broadcast_kernel", "broadcast_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get broadcast_kernel".into()))?;
        
        // Prepare shape and stride data
        let src_shape = input.shape.dims();
        let dst_shape = target_shape.dims();
        let src_ndim = src_shape.len();
        let dst_ndim = dst_shape.len();
        
        // Calculate strides
        let src_strides: Vec<i32> = calculate_strides(src_shape).iter().map(|&x| x as i32).collect();
        let dst_strides: Vec<i32> = calculate_strides(dst_shape).iter().map(|&x| x as i32).collect();
        
        // Upload to device
        let src_shape_gpu = alloc_from_pool_and_copy(&input.device, &src_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())
            .map_err(|_| FlameError::CudaDriver)?;
        let dst_shape_gpu = alloc_from_pool_and_copy(&input.device, &dst_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())
            .map_err(|_| FlameError::CudaDriver)?;
        let src_strides_gpu = alloc_from_pool_and_copy(&input.device, &src_strides)
            .map_err(|_| FlameError::CudaDriver)?;
        let dst_strides_gpu = alloc_from_pool_and_copy(&input.device, &dst_strides)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let numel = target_shape.elem_count();
        let mut output_data = crate::tensor::alloc_from_pool(&input.device, numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg,
            &output_data, input.storage.as_slice(),
            &src_shape_gpu, &dst_shape_gpu,
            &src_strides_gpu, &dst_strides_gpu,
            src_ndim as i32, dst_ndim as i32,
            numel as i32
        )?;
        
        Ok(create_output_tensor(output_data, target_shape.clone(), input.device.clone()))
    }
    
    /// Scatter add operation - adds values from src to dst at indices
    pub fn scatter_add(
        dst: &Tensor,
        indices: &Tensor,
        src: &Tensor,
        dim: usize,
    ) -> Result<Tensor> {
        let kernel_code = r#"
extern "C" __global__ void scatter_add_kernel(
    float* dst,
    const int* indices,
    const float* src,
    int batch_size,
    int embedding_dim,
    int num_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices * embedding_dim) return;
    
    int i = idx / embedding_dim;  // which index
    int j = idx % embedding_dim;  // which dimension
    
    int index = indices[i];
    if (index >= 0 && index < batch_size) {
        atomicAdd(&dst[index * embedding_dim + j], src[i * embedding_dim + j]);
    }
}"#;
        
        CudaKernels::ensure_kernel(&dst.device, "scatter_add_kernel", kernel_code)?;
        
        let f = dst.device.get_func("scatter_add_kernel", "scatter_add_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get scatter_add_kernel".into()))?;
        
        // Clone dst data for output
        let mut output_data = unsafe {
            let size = dst.shape.elem_count();
            let mut data = dst.device.alloc::<f32>(size)
                .map_err(|_| FlameError::CudaDriver)?;
            dst.device.dtod_copy(dst.storage.as_slice(), &mut data)
                .map_err(|_| FlameError::CudaDriver)?;
            data
        };
        
        let shape = src.shape().dims();
        let batch_size = dst.shape().dims()[0] as i32;
        let embedding_dim = if shape.len() > 1 { shape[1] as i32 } else { 1 };
        let num_indices = shape[0] as i32;
        
        let cfg = LaunchConfig::for_num_elems((num_indices * embedding_dim) as u32);
        unsafe {
            launch_kernel!(f, cfg,
                &output_data,
                indices.storage.as_slice(),
                src.storage.as_slice(),
                batch_size,
                embedding_dim,
                num_indices
            )?;
        }
        
        Ok(create_output_tensor(output_data, dst.shape().clone(), dst.device.clone()))
    }
}

/// Mean reduction along specified dimensions
pub fn mean_reduce_dims(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    // For now, implement as sum followed by division
    let kernels = CudaKernels::new(tensor.device.clone())?;
    let sum_result = kernels.sum_dim_keepdim(tensor, dims[0])?;
    
    // Calculate the number of elements in the reduced dimensions
    let mut divisor = 1.0;
    for &dim in dims {
        divisor *= tensor.shape().dims()[dim] as f32;
    }
    
    // Divide by the number of elements
    sum_result.div_scalar(divisor)
}

/// Cast tensor to different dtype
pub fn cast_dtype(tensor: &Tensor, target_dtype: crate::DType) -> Result<Tensor> {
    // For now, just clone with the new dtype
    // In a real implementation, this would use a CUDA kernel
    // Assume tensors are F32 for now
    match (crate::DType::F32, target_dtype) {
        (crate::DType::F32, crate::DType::F16) => {
            // F32 to F16 conversion
            tensor.to_dtype(target_dtype)
        }
        (crate::DType::F16, crate::DType::F32) => {
            // F16 to F32 conversion
            tensor.to_dtype(target_dtype)
        }
        _ => {
            // Same dtype or other conversions
            tensor.to_dtype(target_dtype)
        }
    }
}