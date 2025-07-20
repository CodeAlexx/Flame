use crate::{Tensor, Shape, Result, FlameError};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, CudaSlice, CudaFunction};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use std::collections::HashMap;

// Import CUDA C kernel sources
use crate::cuda_kernel_sources::*;

/// Helper to create output tensor from allocated data
pub fn create_output_tensor(data: CudaSlice<f32>, shape: Shape, device: Arc<CudaDevice>) -> Tensor {
    Tensor {
        data: Arc::new(data),
        shape,
        device,
        id: crate::tensor::TensorId::new(),
        requires_grad: false,
    }
}

/// GPU-only CUDA kernels using NVRTC runtime compilation
pub struct CudaKernels {
    device: Arc<CudaDevice>,
    kernels: HashMap<String, CudaFunction>,
}


impl CudaKernels {
    /// Create new GPU-only kernel instance with compiled kernels
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let mut kernels = HashMap::new();
        
        // Helper function to compile and load a kernel
        fn compile_and_load_kernel(device: &Arc<CudaDevice>, source: &str, kernel_name: &'static str) -> Result<CudaFunction> {
            // Compile the kernel
            let ptx = compile_ptx(source)
                .map_err(|e| FlameError::KernelError(format!("Failed to compile {}: {:?}", kernel_name, e)))?;
            
            // Load the PTX module
            device.load_ptx(ptx, kernel_name, &[kernel_name])?;
            
            // Get the function
            device.get_func(kernel_name, kernel_name)
                .ok_or_else(|| FlameError::KernelError(format!("Failed to get function {}", kernel_name)))
        }
        
        // Compile all kernels
        kernels.insert("add_kernel".to_string(), compile_and_load_kernel(&device, ADD_KERNEL, "add_kernel")?);
        kernels.insert("mul_kernel".to_string(), compile_and_load_kernel(&device, MUL_KERNEL, "mul_kernel")?);
        kernels.insert("mul_scalar_kernel".to_string(), compile_and_load_kernel(&device, MUL_SCALAR_KERNEL, "mul_scalar_kernel")?);
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
        
        Ok(Self { device, kernels })
    }
    
    /// Element-wise addition kernel
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(FlameError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        
        let mut output = Tensor::zeros(a.shape.clone(), a.device.clone())?;
        let n = a.shape.elem_count();
        
        let kernel = self.kernels.get("add_kernel")
            .ok_or_else(|| FlameError::KernelError("add_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (a.data.as_ref(), b.data.as_ref(), &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Element-wise multiplication kernel
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(FlameError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        
        let mut output = Tensor::zeros(a.shape.clone(), a.device.clone())?;
        let n = a.shape.elem_count();
        
        let kernel = self.kernels.get("mul_kernel")
            .ok_or_else(|| FlameError::KernelError("mul_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (a.data.as_ref(), b.data.as_ref(), &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Scalar multiplication kernel
    pub fn mul_scalar(&self, tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("mul_scalar_kernel")
            .ok_or_else(|| FlameError::KernelError("mul_scalar_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), scalar, &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Add scalar kernel
    pub fn add_scalar(&self, tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("add_scalar_kernel")
            .ok_or_else(|| FlameError::KernelError("add_scalar_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), scalar, &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// ReLU activation kernel
    pub fn relu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("relu_kernel")
            .ok_or_else(|| FlameError::KernelError("relu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Sigmoid activation kernel
    pub fn sigmoid(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("sigmoid_kernel")
            .ok_or_else(|| FlameError::KernelError("sigmoid_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// GELU activation kernel
    pub fn gelu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("gelu_kernel")
            .ok_or_else(|| FlameError::KernelError("gelu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// SiLU (Swish) activation kernel
    pub fn silu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("silu_kernel")
            .ok_or_else(|| FlameError::KernelError("silu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Tanh activation kernel
    pub fn tanh(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("tanh_kernel")
            .ok_or_else(|| FlameError::KernelError("tanh_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &*output.data, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    /// Sum kernel - reduction operation
    pub fn sum(&self, tensor: &Tensor) -> Result<Tensor> {
        let n = tensor.shape.elem_count();
        
        // Allocate output directly as zeros
        let output_data = self.device.alloc_zeros::<f32>(1)?;
        
        let kernel = self.kernels.get("sum_kernel")
            .ok_or_else(|| FlameError::KernelError("sum_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = ((n + block_size - 1) / block_size).min(1024);
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &output_data, n as u32))?;
        }
        
        self.device.synchronize()?;
        
        Ok(Tensor {
            data: Arc::new(output_data),
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
        
        // For now, use a simple CPU implementation and copy to GPU
        // TODO: Implement proper GPU kernel for multi-dimensional reduction
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
            .ok_or_else(|| FlameError::KernelError("transpose_kernel not found".into()))?
            .clone();
        
        let block_size = 16;
        let grid_x = (cols + block_size - 1) / block_size;
        let grid_y = (rows + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &*output.data, rows as u32, cols as u32))?;
        }
        
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
        
        let output = weights.clone()?;
        let n = weights.shape.elem_count();
        
        let kernel = self.kernels.get("update_weights_kernel")
            .ok_or_else(|| FlameError::KernelError("update_weights_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (output.data.as_ref(), gradients.data.as_ref(), lr, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    // Additional activation functions
    pub fn leaky_relu(&self, tensor: &Tensor, negative_slope: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("leaky_relu_kernel")
            .ok_or_else(|| FlameError::KernelError("leaky_relu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &*output.data, negative_slope, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    pub fn elu(&self, tensor: &Tensor, alpha: f32) -> Result<Tensor> {
        let mut output = Tensor::zeros(tensor.shape.clone(), tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        
        let kernel = self.kernels.get("elu_kernel")
            .ok_or_else(|| FlameError::KernelError("elu_kernel not found".into()))?
            .clone();
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(config, (tensor.data.as_ref(), &*output.data, alpha, n as u32))?;
        }
        
        self.device.synchronize()?;
        Ok(output)
    }
    
    pub fn prelu(&self, _tensor: &Tensor, _weight: &Tensor) -> Result<Tensor> {
        // PReLU requires channel-wise parameters - simplified version
        Err(FlameError::InvalidOperation("PReLU GPU kernel not yet implemented".into()))
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
    
    // Pooling operations - these need actual GPU implementations
    pub fn maxpool2d_forward(
        _input: &Tensor,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("MaxPool2d GPU kernel not yet implemented".into()))
    }
    
    pub fn maxpool2d_backward(
        _grad_output: &Tensor,
        _input: &Tensor,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("MaxPool2d backward GPU kernel not yet implemented".into()))
    }
    
    pub fn avgpool2d_forward(
        _input: &Tensor,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _count_include_pad: bool,
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("AvgPool2d GPU kernel not yet implemented".into()))
    }
    
    pub fn avgpool2d_backward(
        _grad_output: &Tensor,
        _input: &Tensor,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _count_include_pad: bool,
    ) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("AvgPool2d backward GPU kernel not yet implemented".into()))
    }
    
    pub fn adaptive_maxpool2d_forward(_input: &Tensor, _output_size: (usize, usize)) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("AdaptiveMaxPool2d GPU kernel not yet implemented".into()))
    }
    
    pub fn adaptive_avgpool2d_forward(_input: &Tensor, _output_size: (usize, usize)) -> Result<Tensor> {
        Err(FlameError::InvalidOperation("AdaptiveAvgPool2d GPU kernel not yet implemented".into()))
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
        // For now, simplified broadcast that requires compatible shapes
        // Full broadcasting with PTX would be complex
        if input.shape == *target_shape {
            return Ok(input.clone()?);
        }
        
        // Check if it's a simple broadcast case (e.g., [1, C] -> [B, C])
        let input_dims = input.shape.dims();
        let target_dims = target_shape.dims();
        
        // Simple case: adding dimensions at the beginning
        if input_dims.len() < target_dims.len() {
            let diff = target_dims.len() - input_dims.len();
            let mut compatible = true;
            
            for i in 0..input_dims.len() {
                if input_dims[i] != target_dims[i + diff] && input_dims[i] != 1 {
                    compatible = false;
                    break;
                }
            }
            
            if compatible {
                // For now, just return a tensor with the right shape
                // Real implementation would use a proper broadcast kernel
                let output = Tensor::zeros(target_shape.clone(), input.device.clone())?;
                
                // This is a placeholder - real implementation needs a proper PTX kernel
                return Ok(output);
            }
        }
        
        Err(FlameError::InvalidOperation(format!(
            "Broadcast from {:?} to {:?} not yet implemented",
            input.shape, target_shape
        )))
    }
    
    // Stub for ensure_kernel - not needed with pre-compiled kernels
    pub fn ensure_kernel(_device: &CudaDevice, _kernel_name: &str, _kernel_code: &str) -> Result<()> {
        Ok(())
    }
}

/// Compile CUDA kernel from source to PTX
/// This is used by other modules that generate custom kernels
pub fn compile_kernel(kernel_name: &str, kernel_code: &str) -> Result<Vec<u8>> {
    let ptx = compile_ptx(kernel_code)
        .map_err(|e| FlameError::KernelError(format!("Failed to compile {}: {:?}", kernel_name, e)))?;
    // PTX is already a compiled binary, we can't extract bytes from it
    // For now, return an error as this function isn't used
    Err(FlameError::InvalidOperation("Cannot extract bytes from compiled PTX".into()))
}