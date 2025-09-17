use crate::{Tensor, Shape, Result, FlameError};
use crate::tensor_storage::TensorStorage;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, CudaSlice, CudaFunction};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use std::collections::HashMap;

// Import CUDA C kernel sources
use crate::cuda_kernel_sources::*;

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// Helper to create output tensor from allocated data
pub fn create_output_tensor(data: CudaSlice<f32>, shape: Shape, device: Arc<CudaDevice>) -> Tensor {
    let numel = shape.elem_count();
    Tensor {
        storage: TensorStorage::F32 { data, numel },
        shape,
        device,
        id: crate::tensor::TensorId::new(),
        requires_grad: false,
    }
}

/// GPU-only CUDA kernels using NVRTC runtime compilation
pub struct CudaKernels {
    pub device: Arc<CudaDevice>,
    pub kernels: HashMap<String, CudaFunction>,
}


impl CudaKernels {
    /// Create new GPU-only kernel instance with compiled kernels
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let mut kernels = HashMap::new();
        
        // Helper function to compile and load a kernel
        fn compile_and_load_kernel(device: &Arc<CudaDevice>, source: &str, kernel_name: &'static str) -> Result<CudaFunction> {
            // Compile the kernel
            let ptx = compile_ptx(source)
                .map_err(|e| FlameError::Cuda(format!("Failed to compile {}: {:?}", kernel_name, e)))?;
            
            // Load the PTX module
            device.load_ptx(ptx, kernel_name, &[kernel_name])?;
            
            // Get the function
            device.get_func(kernel_name, kernel_name)
                .ok_or_else(|| FlameError::Cuda(format!("Failed to get function {}", kernel_name)))
        }
        
        // Compile all kernels
        kernels.insert("add_kernel".to_string(), compile_and_load_kernel(&device, ADD_KERNEL, "add_kernel")?);
        kernels.insert("mul_kernel".to_string(), compile_and_load_kernel(&device, MUL_KERNEL, "mul_kernel")?);
        kernels.insert("mul_scalar_kernel".to_string(), compile_and_load_kernel(&device, MUL_SCALAR_KERNEL, "mul_scalar_kernel")?);
        kernels.insert("div_kernel".to_string(), compile_and_load_kernel(&device, DIV_KERNEL, "div_kernel")?);
        kernels.insert("add_scalar_kernel".to_string(), compile_and_load_kernel(&device, ADD_SCALAR_KERNEL, "add_scalar_kernel")?);
        kernels.insert("relu_kernel".to_string(), compile_and_load_kernel(&device, RELU_KERNEL, "relu_kernel")?);
        kernels.insert("sigmoid_kernel".to_string(), compile_and_load_kernel(&device, SIGMOID_KERNEL, "sigmoid_kernel")?);
        kernels.insert("gelu_kernel".to_string(), compile_and_load_kernel(&device, GELU_KERNEL, "gelu_kernel")?);
        kernels.insert("silu_kernel".to_string(), compile_and_load_kernel(&device, SILU_KERNEL, "silu_kernel")?);
        kernels.insert("tanh_kernel".to_string(), compile_and_load_kernel(&device, TANH_KERNEL, "tanh_kernel")?);
        kernels.insert("sum_kernel".to_string(), compile_and_load_kernel(&device, SUM_KERNEL, "sum_kernel")?);
        kernels.insert("transpose_kernel".to_string(), compile_and_load_kernel(&device, TRANSPOSE_KERNEL, "transpose_kernel")?);
        kernels.insert("update_weights_kernel".to_string(), compile_and_load_kernel(&device, UPDATE_WEIGHTS_KERNEL, "update_weights_kernel")?);
        kernels.insert("leaky_relu_kernel".to_string(), compile_and_load_kernel(&device, LEAKY_RELU_KERNEL, "leaky_relu_kernel")?);
        kernels.insert("elu_kernel".to_string(), compile_and_load_kernel(&device, ELU_KERNEL, "elu_kernel")?);
        kernels.insert("pow_kernel".to_string(), compile_and_load_kernel(&device, POW_KERNEL, "pow_kernel")?);
        kernels.insert("sin_kernel".to_string(), compile_and_load_kernel(&device, SIN_KERNEL, "sin_kernel")?);
        kernels.insert("cos_kernel".to_string(), compile_and_load_kernel(&device, COS_KERNEL, "cos_kernel")?);
        kernels.insert("sqrt_kernel".to_string(), compile_and_load_kernel(&device, SQRT_KERNEL, "sqrt_kernel")?);
        kernels.insert("narrow_kernel".to_string(), compile_and_load_kernel(&device, NARROW_KERNEL, "narrow_kernel")?);
        kernels.insert("permute_nhwc_to_nchw_kernel".to_string(), compile_and_load_kernel(&device, PERMUTE_NHWC_TO_NCHW_KERNEL, "permute_nhwc_to_nchw_kernel")?);
        kernels.insert("permute_nchw_to_nhwc_kernel".to_string(), compile_and_load_kernel(&device, PERMUTE_NCHW_TO_NHWC_KERNEL, "permute_nchw_to_nhwc_kernel")?);
        kernels.insert("index_select_kernel".to_string(), compile_and_load_kernel(&device, INDEX_SELECT_KERNEL, "index_select_kernel")?);
        kernels.insert("slice_kernel".to_string(), compile_and_load_kernel(&device, SLICE_KERNEL, "slice_kernel")?);
        // Newly added elementwise math and reductions
        kernels.insert("exp_kernel".to_string(), compile_and_load_kernel(&device, EXP_KERNEL, "exp_kernel")?);
        kernels.insert("log_kernel".to_string(), compile_and_load_kernel(&device, LOG_KERNEL, "log_kernel")?);
        kernels.insert("sum_dim_keepdim_kernel".to_string(), compile_and_load_kernel(&device, SUM_DIM_KEEPDIM_KERNEL, "sum_dim_keepdim_kernel")?);
        kernels.insert("max_elemwise_kernel".to_string(), compile_and_load_kernel(&device, MAX_ELEMWISE_KERNEL, "max_elemwise_kernel")?);
        kernels.insert("resize_bilinear_nhwc_kernel".to_string(), compile_and_load_kernel(&device, RESIZE_BILINEAR_NHWC_KERNEL, "resize_bilinear_nhwc_kernel")?);
        kernels.insert("center_crop_nhwc_kernel".to_string(), compile_and_load_kernel(&device, CENTER_CROP_NHWC_KERNEL, "center_crop_nhwc_kernel")?);
        kernels.insert("normalize_nhwc_kernel".to_string(), compile_and_load_kernel(&device, NORMALIZE_NHWC_KERNEL, "normalize_nhwc_kernel")?);
        kernels.insert("permute_w_khwkicoc_to_ocickhkw".to_string(), compile_and_load_kernel(&device, PERMUTE_W_KH_KW_IC_OC_TO_OC_IC_KH_KW, "permute_w_khwkicoc_to_ocickhkw")?);
        kernels.insert("permute_w_ocickhkw_to_khwkicoc".to_string(), compile_and_load_kernel(&device, PERMUTE_W_OC_IC_KH_KW_TO_KH_KW_IC_OC, "permute_w_ocickhkw_to_khwkicoc")?);
        
        Ok(Self { device, kernels })
    }

    /// Elementwise add with stride-based broadcasting (delegates to GPU helper)
    pub fn add_bc(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        crate::cuda_kernels_gpu::CudaKernels::add_bc(a, b)
    }

    /// Elementwise mul with stride-based broadcasting (delegates to GPU helper)
    pub fn mul_bc(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        crate::cuda_kernels_gpu::CudaKernels::mul_bc(a, b)
    }
    
    /// Element-wise addition kernel (rank-safe broadcasting)
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Normalize via broadcast if shapes differ (NumPy semantics)
        let (a_t, b_t) = if a.shape != b.shape {
            let target = a.shape.broadcast_shape_binary_op(&b.shape)?;
            // Use CPU-safe broadcast to handle rank differences robustly
            let a_bc = if &a.shape != &target { crate::cuda_kernels::CudaKernels::broadcast(a, &target)? } else { a.clone_result()? };
            let b_bc = if &b.shape != &target { crate::cuda_kernels::CudaKernels::broadcast(b, &target)? } else { b.clone_result()? };
            (a_bc, b_bc)
        } else {
            (a.clone_result()?, b.clone_result()?)
        };
        
        let mut output = Tensor::zeros(a_t.shape.clone(), a_t.device.clone())?;
        let n = a_t.shape.elem_count();
        
        let kernel = self.kernels.get("add_kernel")
            .ok_or_else(|| FlameError::Cuda("add_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, a_t.storage.try_as_slice_f32()?, b_t.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Element-wise multiplication kernel (rank-safe broadcasting)
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Normalize via broadcast if shapes differ (NumPy semantics)
        let (a_t, b_t) = if a.shape != b.shape {
            let target = a.shape.broadcast_shape_binary_op(&b.shape)?;
            let a_bc = if &a.shape != &target { crate::cuda_kernels::CudaKernels::broadcast(a, &target)? } else { a.clone_result()? };
            let b_bc = if &b.shape != &target { crate::cuda_kernels::CudaKernels::broadcast(b, &target)? } else { b.clone_result()? };
            (a_bc, b_bc)
        } else {
            (a.clone_result()?, b.clone_result()?)
        };
        
        let mut output = Tensor::zeros(a_t.shape.clone(), a_t.device.clone())?;
        let n = a_t.shape.elem_count();
        
        let kernel = self.kernels.get("mul_kernel")
            .ok_or_else(|| FlameError::Cuda("mul_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, a_t.storage.try_as_slice_f32()?, b_t.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Scalar multiplication kernel
    pub fn mul_scalar(&self, tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("mul_scalar_kernel")
            .ok_or_else(|| FlameError::Cuda("mul_scalar_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, scalar, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Add scalar kernel
    pub fn add_scalar(&self, tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("add_scalar_kernel")
            .ok_or_else(|| FlameError::Cuda("add_scalar_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, scalar, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// ReLU activation kernel
    pub fn relu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("relu_kernel")
            .ok_or_else(|| FlameError::Cuda("relu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Sigmoid activation kernel
    pub fn sigmoid(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("sigmoid_kernel")
            .ok_or_else(|| FlameError::Cuda("sigmoid_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// GELU activation kernel
    pub fn gelu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("gelu_kernel")
            .ok_or_else(|| FlameError::Cuda("gelu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// SiLU (Swish) activation kernel
    pub fn silu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("silu_kernel")
            .ok_or_else(|| FlameError::Cuda("silu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Tanh activation kernel
    pub fn tanh(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("tanh_kernel")
            .ok_or_else(|| FlameError::Cuda("tanh_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Sum kernel - reduction operation
    pub fn sum(&self, tensor: &Tensor) -> Result<Tensor> {
        let n = tensor.shape.elem_count();
        
        // Allocate output directly as zeros
        let output_data = crate::tensor::alloc_zeros_from_pool(&self.device, 1)?;
        
        let kernel = self.kernels.get("sum_kernel")
            .ok_or_else(|| FlameError::Cuda("sum_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = ((n + block_size - 1) / block_size).min(1024);
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, &output_data, n as u32)?;
        
        self.device.synchronize()?;
        
        Ok(Tensor {
            storage: TensorStorage::F32 { data: output_data, numel: 1 },
            shape: Shape::from_dims(&[]),
            device: tensor.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
        })
    }
    
    /// Sum reduction along specific dimensions
    pub fn sum_dims(&self, tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
        // Calculate output shape
        let mut output_shape = tensor.shape().dims().to_vec();
        for &dim in dims {
            output_shape[dim] = 1;
        }
        
        // For now, implement a simple version that handles the common case
        // of summing along batch dimensions for bias gradients
        if dims.is_empty() {
            return self.sum(tensor);
        }
        
        // Calculate strides and sizes
        let input_shape = tensor.shape().dims();
        let ndims = input_shape.len();
        let mut strides = vec![1; ndims];
        for i in (0..ndims-1).rev() {
            strides[i] = strides[i+1] * input_shape[i+1];
        }
        
        // Total output elements
        let output_elems: usize = output_shape.iter().product();
        let mut output = Tensor::zeros(Shape::from_dims(&output_shape), tensor.device.clone())?;
        
        // Implement GPU kernel for multi-dimensional reduction
        if true {  // FLAME is GPU-only
            return crate::cuda_kernels_gpu::mean_reduce_dims(tensor, dims);
        }
        
        // For CPU tensors, use existing implementation
        let input_data = tensor.to_vec()?;
        let mut output_data = vec![0.0f32; output_elems];
        
        // Iterate over output positions
        for out_idx in 0..output_elems {
            // Convert output index to multi-dimensional position
            let mut out_pos = vec![0; ndims];
            let mut idx = out_idx;
            for i in (0..ndims).rev() {
                out_pos[i] = idx % output_shape[i];
                idx /= output_shape[i];
            }
            
            // Sum over all positions that map to this output position
            let mut sum = 0.0f32;
            let mut count = 0;
            
            // Create iterator over dimensions to sum
            let mut pos = vec![0; ndims];
            loop {
                // Copy fixed dimensions from output position
                for i in 0..ndims {
                    if !dims.contains(&i) {
                        pos[i] = out_pos[i];
                    }
                }
                
                // Calculate linear index
                let lin_idx: usize = pos.iter().zip(&strides).map(|(p, s)| p * s).sum();
                sum += input_data[lin_idx];
                count += 1;
                
                // Increment position over dimensions we're summing
                let mut carry = true;
                for &dim in dims {
                    if carry {
                        pos[dim] += 1;
                        if pos[dim] < input_shape[dim] {
                            carry = false;
                        } else {
                            pos[dim] = 0;
                        }
                    }
                }
                
                if carry {
                    break;
                }
            }
            
            output_data[out_idx] = sum;
        }
        
        // Copy result back to GPU
        output.set_data(&output_data)?;
        Ok(output)
    }
    
    /// Transpose kernel for 2D matrices  
    pub fn transpose(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Err(FlameError::InvalidOperation(
                format!("Transpose requires 2D tensor, got {:?}", dims)
            ));
        }
        
        let rows = dims[0];
        let cols = dims[1];
        let mut output = Tensor::zeros(Shape::from_dims(&[cols, rows]), tensor.device.clone())?;
        
        let kernel = self.kernels.get("transpose_kernel")
            .ok_or_else(|| FlameError::Cuda("transpose_kernel not found".into()))?
            .clone();
        
        let block_size = 16;
        let grid_x = (cols + block_size - 1) / block_size;
        let grid_y = (rows + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, rows as u32, cols as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Weight update kernel (for SGD)
    pub fn update_weights(&self, weights: &Tensor, gradients: &Tensor, lr: f32) -> Result<Tensor> {
        if weights.shape != gradients.shape {
            return Err(FlameError::ShapeMismatch {
                expected: weights.shape.clone(),
                got: gradients.shape.clone(),
            });
        }
        
        let output = weights.clone_result()?;
        let n = weights.shape.elem_count();
        
        let kernel = self.kernels.get("update_weights_kernel")
            .ok_or_else(|| FlameError::Cuda("update_weights_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, output.storage.try_as_slice_f32()?, gradients.storage.try_as_slice_f32()?, lr, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    // Additional activation functions
    pub fn leaky_relu(&self, tensor: &Tensor, negative_slope: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("leaky_relu_kernel")
            .ok_or_else(|| FlameError::Cuda("leaky_relu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, negative_slope, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    pub fn elu(&self, tensor: &Tensor, alpha: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("elu_kernel")
            .ok_or_else(|| FlameError::Cuda("elu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, alpha, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    pub fn prelu(&self, _tensor: &Tensor, _weight: &Tensor) -> Result<Tensor> {
        // PReLU requires channel-wise parameters
        Err(FlameError::InvalidOperation("PReLU GPU kernel not yet implemented".into()))
    }
    
    pub fn pow(&self, tensor: &Tensor, exponent: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("pow_kernel")
            .ok_or_else(|| FlameError::Cuda("pow_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, exponent, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    pub fn sin(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("sin_kernel")
            .ok_or_else(|| FlameError::Cuda("sin_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    pub fn cos(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("cos_kernel")
            .ok_or_else(|| FlameError::Cuda("cos_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    pub fn sqrt(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("sqrt_kernel")
            .ok_or_else(|| FlameError::Cuda("sqrt_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(kernel, config, tensor.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    // Transposed convolution operations
    pub fn conv_transpose2d_forward(
        &self,
        _input: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _output_padding: (usize, usize),
        _groups: usize,
        _dilation: (usize, usize),
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("ConvTranspose2d forward GPU kernel not yet implemented".into()))
    }
    
    pub fn conv_transpose2d_backward(
        &self,
        _grad_output: &Tensor,
        _input: &Tensor,
        _weight: &Tensor,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _output_padding: (usize, usize),
        _groups: usize,
        _dilation: (usize, usize),
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        Err(FlameError::InvalidOperation("ConvTranspose2d backward GPU kernel not yet implemented".into()))
    }
    
    // Pooling operations - using CPU implementations for now
    pub fn maxpool2d_forward(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        // Use CPU implementation
        let (output, _indices) = crate::pooling_impl::maxpool2d_forward(input, kernel_size, stride, padding)?;
        Ok(output)
    }
    
    pub fn maxpool2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        // First do forward pass to get indices
        let (_output, indices) = crate::pooling_impl::maxpool2d_forward(input, kernel_size, stride, padding)?;
        // Then use indices for backward pass
        crate::pooling_impl::maxpool2d_backward(grad_output, input.shape.clone(), &indices)
    }
    
    pub fn avgpool2d_forward(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Result<Tensor> {
        // Use CPU implementation
        crate::pooling_impl::avgpool2d_forward(input, kernel_size, stride, padding, count_include_pad)
    }
    
    pub fn avgpool2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Result<Tensor> {
        // Use CPU implementation
        crate::pooling_impl::avgpool2d_backward(grad_output, input.shape.clone(), kernel_size, stride, padding, count_include_pad)
    }
    
    pub fn adaptive_maxpool2d_forward(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
        // Calculate adaptive kernel size and stride
        let shape = input.shape().dims();
        let (_, _, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = output_size;
        
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;
        let kernel_h = h_in - (h_out - 1) * stride_h;
        let kernel_w = w_in - (w_out - 1) * stride_w;
        
        // Use regular maxpool with calculated parameters
        let (output, _indices) = crate::pooling_impl::maxpool2d_forward(
            input, 
            (kernel_h, kernel_w), 
            (stride_h, stride_w), 
            (0, 0)
        )?;
        Ok(output)
    }
    
    pub fn adaptive_avgpool2d_forward(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
        // Calculate adaptive kernel size and stride
        let shape = input.shape().dims();
        let (_, _, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = output_size;
        
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;
        let kernel_h = h_in - (h_out - 1) * stride_h;
        let kernel_w = w_in - (w_out - 1) * stride_w;
        
        // Use regular avgpool with calculated parameters
        crate::pooling_impl::avgpool2d_forward(
            input, 
            (kernel_h, kernel_w), 
            (stride_h, stride_w), 
            (0, 0),
            false
        )
    }
    
    // Upsampling operations
    pub fn upsample2d_nearest(
        &self,
        _input: &Tensor,
        _output_size: (usize, usize),
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("Upsample2d nearest GPU kernel not yet implemented".into()))
    }
    
    pub fn upsample2d_bilinear(
        &self,
        _input: &Tensor,
        _output_size: (usize, usize),
        _align_corners: bool,
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("Upsample2d bilinear GPU kernel not yet implemented".into()))
    }
    
    pub fn upsample2d_nearest_backward(
        &self,
        _grad_output: &Tensor,
        _input_size: (usize, usize),
        _output_size: (usize, usize),
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("Upsample2d nearest backward GPU kernel not yet implemented".into()))
    }
    
    pub fn upsample2d_bilinear_backward(
        &self,
        _grad_output: &Tensor,
        _input_size: (usize, usize),
        _output_size: (usize, usize),
        _align_corners: bool,
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("Upsample2d bilinear backward GPU kernel not yet implemented".into()))
    }
    
    /// Broadcast tensor to a new shape
    pub fn broadcast(input: &Tensor, target_shape: &Shape) -> Result<Tensor> {
        if input.shape == *target_shape {
            return Ok(input.clone_result()?);
        }
        
        let input_dims = input.shape.dims();
        let target_dims = target_shape.dims();
        
        // CPU implementation for now until we have a proper CUDA kernel
        let input_data = input.to_vec()?;
        let mut output_data = vec![0.0f32; target_shape.elem_count()];
        
        // Calculate strides for both shapes
        let input_strides = input.shape.strides();
        let target_strides = target_shape.strides();
        
        // Align dimensions from the right
        let ndim = target_dims.len();
        let offset = ndim - input_dims.len();
        
        // For each element in the output
        for i in 0..target_shape.elem_count() {
            // Calculate indices for each dimension
            let mut target_idx = i;
            let mut input_idx = 0;
            
            for d in 0..ndim {
                let dim_idx = target_idx / target_strides[d];
                target_idx %= target_strides[d];
                
                // Map to input dimension
                if d >= offset {
                    let input_d = d - offset;
                    let input_dim_size = input_dims[input_d];
                    if input_dim_size > 1 {
                        input_idx += dim_idx * input_strides[input_d];
                    }
                    // If input_dim_size == 1, we broadcast (don't add to index)
                }
            }
            
            output_data[i] = input_data[input_idx];
        }
        
        // Create output tensor
        let output = Tensor::from_vec(output_data, target_shape.clone(), input.device.clone())?;
        Ok(output)
    }
    
    // Ensure a kernel is compiled and loaded
    pub fn ensure_kernel(device: &Arc<CudaDevice>, kernel_name: &str, kernel_code: &str) -> Result<()> {
        // Check if kernel is already loaded
        if device.get_func(kernel_name, kernel_name).is_some() {
            return Ok(());
        }
        
        // Compile the kernel
        let ptx = compile_ptx(kernel_code)
            .map_err(|e| FlameError::Cuda(format!("Failed to compile {}: {:?}", kernel_name, e)))?;
        
        // Use Box::leak to get 'static lifetime for kernel names
        let kernel_name_static = Box::leak(kernel_name.to_string().into_boxed_str());
        
        // Load the PTX module
        device.load_ptx(ptx, kernel_name_static, &[kernel_name_static])
            .map_err(|e| FlameError::Cuda(format!("Failed to load kernel {}: {:?}", kernel_name, e)))?;
        
        Ok(())
    }
    
    /// Element-wise division
    pub fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(FlameError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        let mut output = Tensor::zeros(a.shape.clone(), a.device.clone())?;
        let n = a.shape.elem_count();
        let kernel = self.kernels.get("div_kernel")
            .ok_or_else(|| FlameError::Cuda("div_kernel not found".into()))?
            .clone();
        let block_size = 256usize;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = LaunchConfig { grid_dim: (grid_size as u32,1,1), block_dim: (block_size as u32,1,1), shared_mem_bytes: 0 };
        launch_kernel!(kernel, cfg, a.storage.try_as_slice_f32()?, b.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as u32)?;
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Max reduction along dimension (GPU, with optional keepdim)
    pub fn max_dim(&self, tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dim >= dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of range for tensor with {} dimensions", dim, dims.len())
            ));
        }

        // Kernel computes keepdim=true; we reshape if keepdim=false
        let mut out_shape_keep = dims.to_vec();
        out_shape_keep[dim] = 1;
        let out_elems: usize = out_shape_keep.iter().product();

        // Upload dims as f32
        let dims_f32: Vec<f32> = dims.iter().map(|&x| x as f32).collect();
        let mut dims_gpu = unsafe { self.device.alloc::<f32>(dims_f32.len()) }
            .map_err(|_| FlameError::CudaDriver)?;
        self.device.htod_copy_into(dims_f32, &mut dims_gpu)
            .map_err(|_| FlameError::CudaDriver)?;

        // Allocate output keepdim
        let mut out = Tensor::zeros(Shape::from_dims(&out_shape_keep), tensor.device.clone())?;

        // Inline kernel: max along reduce_dim, keeping dimension size 1
        let kernel_code = r#"
extern "C" __global__ void max_dim_keepdim_kernel(
    const float* input,
    float* output,
    const float* dims_f32,
    int ndim,
    int reduce_dim,
    int out_elems
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_elems) return;

    int dims[8];
    for (int i = 0; i < ndim && i < 8; ++i) dims[i] = (int)dims_f32[i];

    int strides[8];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    int rem = tid;
    int out_coords[8];
    for (int i = 0; i < ndim; ++i) {
        int size = (i == reduce_dim) ? 1 : dims[i];
        int stride = 1;
        for (int j = i + 1; j < ndim; ++j) {
            stride *= (j == reduce_dim) ? 1 : dims[j];
        }
        out_coords[i] = (size == 0) ? 0 : (rem / stride) % size;
    }

    int base_idx = 0;
    for (int i = 0; i < ndim; ++i) {
        int coord = (i == reduce_dim) ? 0 : out_coords[i];
        base_idx += coord * strides[i];
    }

    float maxv = -3.402823e38f; // -FLT_MAX
    for (int d = 0; d < dims[reduce_dim]; ++d) {
        int idx = base_idx + d * strides[reduce_dim];
        float v = input[idx];
        if (v > maxv) maxv = v;
    }
    output[tid] = maxv;
}
"#;

        // Compile/load once per process
        Self::ensure_kernel(&self.device, "max_dim_keepdim_kernel", kernel_code)?;
        let f = self.device.get_func("max_dim_keepdim_kernel", "max_dim_keepdim_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get max_dim_keepdim_kernel".into()))?;

        let cfg = LaunchConfig::for_num_elems(out_elems as u32);
        launch_kernel!(f, cfg,
            tensor.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            &dims_gpu,
            dims.len() as i32,
            dim as i32,
            out_elems as i32
        )?;
        self.device.synchronize()?;

        if keepdim { Ok(out) } else {
            // Squeeze the reduced dimension
            let mut squeezed = Vec::with_capacity(dims.len() - 1);
            for (i, &d) in dims.iter().enumerate() {
                if i != dim { squeezed.push(d); }
            }
            out.reshape(&squeezed)
        }
    }
    
    /// Sum along dimension with keepdim (GPU kernel)
    pub fn sum_dim_keepdim(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dim >= dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of range for tensor with {} dimensions", dim, dims.len())
            ));
        }
        // Build output shape keeping reduced dim = 1
        let mut out_shape = dims.to_vec();
        out_shape[dim] = 1;
        let out_elems: usize = out_shape.iter().product();

        // Upload dims to GPU as i32
        let dims_i32: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
        let mut dims_gpu = unsafe { self.device.alloc::<f32>(dims_i32.len()) }
            .map_err(|_| FlameError::CudaDriver)?;
        self.device.htod_copy_into(dims_i32.iter().map(|&x| x as f32).collect::<Vec<_>>(), &mut dims_gpu)
            .map_err(|_| FlameError::CudaDriver)?;

        // Allocate output
        let mut out = Tensor::zeros(Shape::from_dims(&out_shape), tensor.device.clone())?;

        let kernel = self.kernels.get("sum_dim_keepdim_kernel")
            .ok_or_else(|| FlameError::Cuda("sum_dim_keepdim_kernel not found".into()))?
            .clone();

        let cfg = LaunchConfig::for_num_elems(out_elems as u32);
        launch_kernel!(kernel, cfg,
            tensor.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            &dims_gpu,
            dims.len() as i32,
            dim as i32,
            out_elems as i32
        )?;
        self.device.synchronize()?;
        Ok(out)
    }
    
    /// Sum all elements in a tensor
    pub fn sum_kernel(&self, tensor: &Tensor) -> Result<Tensor> {
        // Simple implementation - sum to scalar
        let data = tensor.to_vec()?;
        let sum: f32 = data.iter().sum();
        Tensor::from_vec(vec![sum], Shape::from_dims(&[1]), tensor.device.clone())
    }

    /// Elementwise max kernel
    pub fn max_elemwise(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(FlameError::ShapeMismatch { expected: a.shape.clone(), got: b.shape.clone() });
        }
        let mut output = Tensor::zeros(a.shape.clone(), a.device.clone())?;
        let n = a.shape.elem_count();
        let kernel = self.kernels.get("max_elemwise_kernel")
            .ok_or_else(|| FlameError::Cuda("max_elemwise_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(kernel, cfg, a.storage.try_as_slice_f32()?, b.storage.try_as_slice_f32()?, output.storage.try_as_slice_f32()?, n as i32)?;
        self.device.synchronize()?;
        Ok(output)
    }

    /// Resize NHWC via bilinear interpolation
    pub fn resize_bilinear_nhwc(&self, input: &Tensor, out_h: usize, out_w: usize, align_corners: bool) -> Result<Tensor> {
        let dims = input.shape.dims();
        let (n, h, w, c) = match dims { [n,h,w,c] => (*n,*h,*w,*c), _ => return Err(FlameError::InvalidOperation("resize_bilinear_nhwc expects NHWC".into())) };
        let output_shape = Shape::from_dims(&[n, out_h, out_w, c]);
        let total = n * out_h * out_w * c;
        let mut out = Tensor::zeros(output_shape, input.device.clone())?;
        let k = self.kernels.get("resize_bilinear_nhwc_kernel").ok_or_else(|| FlameError::Cuda("resize_bilinear_nhwc_kernel not found".into()))?.clone();
        let cfg = LaunchConfig::for_num_elems(total as u32);
        launch_kernel!(k, cfg, input.storage.try_as_slice_f32()?, out.storage.try_as_slice_f32()?, n as i32, h as i32, w as i32, c as i32, out_h as i32, out_w as i32, if align_corners {1} else {0})?;
        self.device.synchronize()?;
        Ok(out)
    }

    /// Center crop NHWC
    pub fn center_crop_nhwc(&self, input: &Tensor, tgt_h: usize, tgt_w: usize) -> Result<Tensor> {
        let dims = input.shape.dims();
        let (n, h, w, c) = match dims { [n,h,w,c] => (*n,*h,*w,*c), _ => return Err(FlameError::InvalidOperation("center_crop_nhwc expects NHWC".into())) };
        if tgt_h > h || tgt_w > w { return Err(FlameError::InvalidOperation("center crop size exceeds input".into())) }
        let y0 = ((h - tgt_h) / 2) as i32;
        let x0 = ((w - tgt_w) / 2) as i32;
        let output_shape = Shape::from_dims(&[n, tgt_h, tgt_w, c]);
        let total = n * tgt_h * tgt_w * c;
        let mut out = Tensor::zeros(output_shape, input.device.clone())?;
        let k = self.kernels.get("center_crop_nhwc_kernel").ok_or_else(|| FlameError::Cuda("center_crop_nhwc_kernel not found".into()))?.clone();
        let cfg = LaunchConfig::for_num_elems(total as u32);
        launch_kernel!(k, cfg, input.storage.try_as_slice_f32()?, out.storage.try_as_slice_f32()?, n as i32, h as i32, w as i32, c as i32, y0, x0, tgt_h as i32, tgt_w as i32)?;
        self.device.synchronize()?;
        Ok(out)
    }

    /// Normalize NHWC per channel
    pub fn normalize_nhwc(&self, input: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
        let dims = input.shape.dims();
        let (n, h, w, c) = match dims { [n,h,w,c] => (*n,*h,*w,*c), _ => return Err(FlameError::InvalidOperation("normalize_nhwc expects NHWC".into())) };
        if mean.len() != c || std.len() != c { return Err(FlameError::InvalidOperation("mean/std size must match channels".into())) }
        let inv_std: Vec<f32> = std.iter().map(|&v| 1.0f32 / v).collect();
        let mut mean_gpu = crate::tensor::alloc_from_pool(&input.device, c).map_err(|_| FlameError::CudaDriver)?;
        let mut inv_gpu = crate::tensor::alloc_from_pool(&input.device, c).map_err(|_| FlameError::CudaDriver)?;
        input.device.htod_copy_into(mean.to_vec(), &mut mean_gpu).map_err(|_| FlameError::CudaDriver)?;
        input.device.htod_copy_into(inv_std, &mut inv_gpu).map_err(|_| FlameError::CudaDriver)?;
        let output_shape = input.shape.clone();
        let total = n * h * w * c;
        let mut out = Tensor::zeros(output_shape, input.device.clone())?;
        let k = self.kernels.get("normalize_nhwc_kernel").ok_or_else(|| FlameError::Cuda("normalize_nhwc_kernel not found".into()))?.clone();
        let cfg = LaunchConfig::for_num_elems(total as u32);
        launch_kernel!(k, cfg, input.storage.try_as_slice_f32()?, out.storage.try_as_slice_f32()?, &mean_gpu, &inv_gpu, n as i32, h as i32, w as i32, c as i32)?;
        self.device.synchronize()?;
        Ok(out)
    }
    
    /// Permute tensor from NHWC to NCHW format on GPU
    pub fn permute_nhwc_to_nchw(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("Permute NHWC to NCHW requires 4D tensor, got {:?}", dims)
            ));
        }
        
        let batch = dims[0];
        let height = dims[1];
        let width = dims[2];
        let channels = dims[3];
        
        // Create output tensor with NCHW shape
        let output_shape = Shape::from_dims(&[batch, channels, height, width]);
        let mut output = Tensor::zeros(output_shape, tensor.device.clone())?;
        
        let kernel = self.kernels.get("permute_nhwc_to_nchw_kernel")
            .ok_or_else(|| FlameError::Cuda("permute_nhwc_to_nchw_kernel not found".into()))?
            .clone();
        
        let total_elements = tensor.shape.elem_count();
        let block_size = 256;
        let grid_size = (total_elements + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launch_kernel!(
            kernel, 
            config, 
            tensor.storage.try_as_slice_f32()?, 
            output.storage.try_as_slice_f32()?, 
            batch as u32,
            height as u32,
            width as u32,
            channels as u32
        )?;
        
        self.device.synchronize()?;
        Ok(output)
    }

    /// Permute tensor from NCHW to NHWC format on GPU
    pub fn permute_nchw_to_nhwc(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("Permute NCHW to NHWC requires 4D tensor, got {:?}", dims)
            ));
        }
        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let output_shape = Shape::from_dims(&[batch, height, width, channels]);
        let mut output = Tensor::zeros(output_shape, tensor.device.clone())?;
        let kernel = self.kernels.get("permute_nchw_to_nhwc_kernel")
            .ok_or_else(|| FlameError::Cuda("permute_nchw_to_nhwc_kernel not found".into()))?
            .clone();
        let total_elements = tensor.shape.elem_count();
        let block_size = 256;
        let grid_size = (total_elements + block_size - 1) / block_size;
        let config = LaunchConfig { grid_dim: (grid_size as u32,1,1), block_dim: (block_size as u32,1,1), shared_mem_bytes: 0 };
        launch_kernel!(
            kernel, 
            config, 
            tensor.storage.try_as_slice_f32()?, 
            output.storage.try_as_slice_f32()?, 
            batch as u32,
            channels as u32,
            height as u32,
            width as u32
        )?;
        self.device.synchronize()?;
        Ok(output)
    }

    /// Permute weights [KH,KW,IC,OC] -> [OC,IC,KH,KW]
    pub fn weight_khwkicoc_to_ocickhkw(&self, w: &Tensor) -> Result<Tensor> {
        let d = w.shape.dims();
        if d.len() != 4 { return Err(FlameError::InvalidOperation("weight permute expects 4D".into())); }
        let (kh, kw, ic, oc) = (d[0], d[1], d[2], d[3]);
        let mut out = Tensor::zeros(Shape::from_dims(&[oc, ic, kh, kw]), w.device.clone())?;
        let k = self.kernels.get("permute_w_khwkicoc_to_ocickhkw")
            .ok_or_else(|| FlameError::Cuda("permute_w_khwkicoc_to_ocickhkw not found".into()))?
            .clone();
        let total = (kh * kw * ic * oc) as u32;
        let cfg = LaunchConfig::for_num_elems(total);
        launch_kernel!(k, cfg, w.storage.try_as_slice_f32()?, out.storage.try_as_slice_f32()?, kh as i32, kw as i32, ic as i32, oc as i32)?;
        self.device.synchronize()?;
        Ok(out)
    }

    /// Permute weights [OC,IC,KH,KW] -> [KH,KW,IC,OC]
    pub fn weight_ocickhkw_to_khwkicoc(&self, w: &Tensor) -> Result<Tensor> {
        let d = w.shape.dims();
        if d.len() != 4 { return Err(FlameError::InvalidOperation("weight permute expects 4D".into())); }
        let (oc, ic, kh, kw) = (d[0], d[1], d[2], d[3]);
        let mut out = Tensor::zeros(Shape::from_dims(&[kh, kw, ic, oc]), w.device.clone())?;
        let k = self.kernels.get("permute_w_ocickhkw_to_khwkicoc")
            .ok_or_else(|| FlameError::Cuda("permute_w_ocickhkw_to_khwkicoc not found".into()))?
            .clone();
        let total = (kh * kw * ic * oc) as u32;
        let cfg = LaunchConfig::for_num_elems(total);
        launch_kernel!(k, cfg, w.storage.try_as_slice_f32()?, out.storage.try_as_slice_f32()?, oc as i32, ic as i32, kh as i32, kw as i32)?;
        self.device.synchronize()?;
        Ok(out)
    }

    /// Elementwise exponential
    pub fn exp(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut out = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        let kernel = self.kernels.get("exp_kernel")
            .ok_or_else(|| FlameError::Cuda("exp_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(kernel, cfg, tensor.storage.try_as_slice_f32()?, out.storage.try_as_slice_f32()?, n as i32)?;
        self.device.synchronize()?;
        Ok(out)
    }

    /// Elementwise natural logarithm
    pub fn log(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut out = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        let kernel = self.kernels.get("log_kernel")
            .ok_or_else(|| FlameError::Cuda("log_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(kernel, cfg, tensor.storage.try_as_slice_f32()?, out.storage.try_as_slice_f32()?, n as i32)?;
        self.device.synchronize()?;
        Ok(out)
    }

    /// Index select along a dimension (GPU kernel)
    pub fn index_select(&self, tensor: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape().dims();
        let idx_shape = indices.shape().dims();
        if dim >= shape.len() { return Err(FlameError::InvalidOperation(format!("Dimension {} out of bounds", dim))); }
        if idx_shape.len() != 1 { return Err(FlameError::InvalidOperation(format!("Indices must be 1D, got {:?}", idx_shape))); }
        let num_indices = indices.shape().elem_count();
        let mut out_shape = shape.to_vec();
        out_shape[dim] = num_indices;
        let out_shape_s = Shape::from_dims(&out_shape);
        let mut output = Tensor::zeros(out_shape_s.clone(), tensor.device.clone())?;

        // Prepare strides and dims arrays
        let ndim = shape.len() as i32;
        let in_dims: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let out_dims: Vec<i32> = out_shape.iter().map(|&x| x as i32).collect();
        let mut in_strides = vec![1i32; shape.len()];
        for i in (0..shape.len()-1).rev() { in_strides[i] = in_strides[i+1] * shape[i+1] as i32; }
        let mut out_strides = vec![1i32; out_shape.len()];
        for i in (0..out_shape.len()-1).rev() { out_strides[i] = out_strides[i+1] * out_shape[i+1] as i32; }

        // Copy arrays to device
        let d_in_dims = unsafe { self.device.alloc::<i32>(in_dims.len()) }?; self.device.htod_copy_into(in_dims.clone(), &mut {let mut x = d_in_dims.clone(); x})?;
        let d_out_dims = unsafe { self.device.alloc::<i32>(out_dims.len()) }?; self.device.htod_copy_into(out_dims.clone(), &mut {let mut x = d_out_dims.clone(); x})?;
        let d_in_strides = unsafe { self.device.alloc::<i32>(in_strides.len()) }?; self.device.htod_copy_into(in_strides.clone(), &mut {let mut x = d_in_strides.clone(); x})?;
        let d_out_strides = unsafe { self.device.alloc::<i32>(out_strides.len()) }?; self.device.htod_copy_into(out_strides.clone(), &mut {let mut x = d_out_strides.clone(); x})?;

        let kernel = self.kernels.get("index_select_kernel")
            .ok_or_else(|| FlameError::Cuda("index_select_kernel not found".into()))?
            .clone();

        let numel = out_shape_s.elem_count();
        let block = 256usize;
        let grid = (numel + block - 1) / block;
        let cfg = LaunchConfig { grid_dim: (grid as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: 0 };

        launch_kernel!(
            kernel,
            cfg,
            tensor.storage.try_as_slice_f32()?,
            indices.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            ndim,
            &d_in_dims,
            &d_out_dims,
            &d_in_strides,
            &d_out_strides,
            dim as i32,
            numel as i32
        )?;

        self.device.synchronize()?;
        Ok(output)
    }

    /// General slice across multiple dimensions (GPU kernel)
    pub fn slice(&self, tensor: &Tensor, ranges: &[(usize, usize)]) -> Result<Tensor> {
        let shape = tensor.shape().dims();
        if ranges.len() != shape.len() { return Err(FlameError::InvalidOperation(format!("ranges/dims mismatch"))); }
        let mut out_shape = Vec::with_capacity(ranges.len());
        let mut starts: Vec<i32> = Vec::with_capacity(ranges.len());
        for (i, &(s,e)) in ranges.iter().enumerate() {
            if s>=e || e>shape[i] { return Err(FlameError::InvalidOperation(format!("invalid range {}-{} for dim {}", s,e,i))); }
            out_shape.push(e - s);
            starts.push(s as i32);
        }
        let out_s = Shape::from_dims(&out_shape);
        let mut output = Tensor::zeros(out_s.clone(), tensor.device.clone())?;

        // Strides
        let mut in_strides = vec![1i32; shape.len()];
        for i in (0..shape.len()-1).rev() { in_strides[i] = in_strides[i+1] * shape[i+1] as i32; }
        let mut out_strides = vec![1i32; out_shape.len()];
        for i in (0..out_shape.len()-1).rev() { out_strides[i] = out_strides[i+1] * out_shape[i+1] as i32; }

        let d_in_strides = unsafe { self.device.alloc::<i32>(in_strides.len()) }?; self.device.htod_copy_into(in_strides.clone(), &mut {let mut x = d_in_strides.clone(); x})?;
        let d_out_strides = unsafe { self.device.alloc::<i32>(out_strides.len()) }?; self.device.htod_copy_into(out_strides.clone(), &mut {let mut x = d_out_strides.clone(); x})?;
        let d_starts = unsafe { self.device.alloc::<i32>(starts.len()) }?; self.device.htod_copy_into(starts.clone(), &mut {let mut x = d_starts.clone(); x})?;

        let kernel = self.kernels.get("slice_kernel")
            .ok_or_else(|| FlameError::Cuda("slice_kernel not found".into()))?
            .clone();
        let numel = out_s.elem_count();
        let block = 256usize; let grid = (numel + block - 1)/block;
        let cfg = LaunchConfig{ grid_dim:(grid as u32,1,1), block_dim:(block as u32,1,1), shared_mem_bytes:0};

        launch_kernel!(
            kernel,
            cfg,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            shape.len() as i32,
            &d_in_strides,
            &d_out_strides,
            &d_starts,
            numel as i32
        )?;

        self.device.synchronize()?;
        Ok(output)
    }
}

/// Scatter add operation for efficient gradient accumulation
/// This is a generic tensor operation, not model-specific
pub fn scatter_add(
    input_shape: &[usize],
    grad_output: &Tensor,
    indices: &Tensor,
    dim: usize,
) -> Result<Tensor> {
    let device = grad_output.device();
    
    if true {  // FLAME is GPU-only
        // For GPU, we would use a CUDA kernel
        // For now, fallback to CPU implementation
        let mut input_grad = Tensor::zeros(Shape::from_dims(input_shape), device.clone())?;
        let grad_data = grad_output.to_vec()?;
        let indices_data = indices.to_vec()?;
        let mut input_grad_data = vec![0.0f32; input_grad.shape().elem_count()];
        
        let grad_shape = grad_output.shape().dims();
        let input_shape = input_grad.shape().dims();
        
        // Calculate position in flattened array
        for i in 0..grad_data.len() {
            let mut grad_idx = i;
            let mut pos = vec![0; grad_shape.len()];
            
            for d in (0..grad_shape.len()).rev() {
                pos[d] = grad_idx % grad_shape[d];
                grad_idx /= grad_shape[d];
            }
            
            let idx = indices_data[pos[dim]] as i64 as usize;
            pos[dim] = idx;
            
            let mut in_idx = 0;
            let mut stride = 1;
            for d in (0..input_shape.len()).rev() {
                in_idx += pos[d] * stride;
                stride *= input_shape[d];
            }
            
            input_grad_data[in_idx] += grad_data[i];
        }
        
        Tensor::from_vec(input_grad_data, Shape::from_dims(input_shape), device.clone())
    } else {
        // CPU implementation
        let mut input_grad = Tensor::zeros(Shape::from_dims(input_shape), device.clone())?;
        let grad_data = grad_output.to_vec()?;
        let indices_data = indices.to_vec()?;
        let mut input_grad_data = vec![0.0f32; input_grad.shape().elem_count()];
        
        let grad_shape = grad_output.shape().dims();
        let input_shape = input_grad.shape().dims();
        
        // Calculate position in flattened array
        for i in 0..grad_data.len() {
            let mut grad_idx = i;
            let mut pos = vec![0; grad_shape.len()];
            
            for d in (0..grad_shape.len()).rev() {
                pos[d] = grad_idx % grad_shape[d];
                grad_idx /= grad_shape[d];
            }
            
            let idx = indices_data[pos[dim]] as i64 as usize;
            pos[dim] = idx;
            
            let mut in_idx = 0;
            let mut stride = 1;
            for d in (0..input_shape.len()).rev() {
                in_idx += pos[d] * stride;
                stride *= input_shape[d];
            }
            
            input_grad_data[in_idx] += grad_data[i];
        }
        
        Tensor::from_vec(input_grad_data, Shape::from_dims(input_shape), device.clone())
    }
}

/// Compile CUDA kernel from source to PTX
/// This is used by other modules that generating custom kernels
pub fn compile_kernel(kernel_name: &str, kernel_code: &str) -> Result<Vec<u8>> {
    let ptx = compile_ptx(kernel_code)
        .map_err(|e| FlameError::Cuda(format!("Failed to compile {}: {:?}", kernel_name, e)))?;
    // PTX is already a compiled binary, we can't extract bytes from it
    // For now, return an error as this function isn't used
    Err(FlameError::InvalidOperation("Cannot extract bytes from compiled PTX".into()))
}
