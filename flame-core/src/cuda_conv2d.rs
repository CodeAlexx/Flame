//! CUDA kernels for 2D convolution operations

use crate::{Tensor, Shape, Result, FlameError};
use crate::autograd::{AutogradContext, Op};
use crate::tensor::TensorId;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// CUDA kernel source for im2col transformation
pub const IM2COL_KERNEL: &str = r#"
extern "C" __global__ void im2col_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w
) {
    // For now, use fixed padding and stride
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * kernel_h * kernel_w * out_height * out_width;
    
    if (index < total_elements) {
        int w_out = index % out_width;
        int h_out = (index / out_width) % out_height;
        int kw = (index / (out_width * out_height)) % kernel_w;
        int kh = (index / (out_width * out_height * kernel_w)) % kernel_h;
        int c = (index / (out_width * out_height * kernel_w * kernel_h)) % channels;
        int batch = index / (out_width * out_height * kernel_w * kernel_h * channels);
        
        int h_in = h_out * stride_h - pad_h + kh;
        int w_in = w_out * stride_w - pad_w + kw;
        
        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            int input_idx = batch * (channels * height * width) + 
                           c * (height * width) + 
                           h_in * width + 
                           w_in;
            output[index] = input[input_idx];
        } else {
            output[index] = 0.0f;
        }
    }
}
"#;

/// CUDA kernel for col2im (backward pass)
pub const COL2IM_KERNEL: &str = r#"
extern "C" __global__ void col2im_kernel(
    const float* col,
    float* im,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w
) {
    // For now, use fixed padding and stride
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (index < total_elements) {
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / (width * height)) % channels;
        int batch = index / (width * height * channels);
        
        float val = 0.0f;
        
        // Find all positions in col that map to this pixel
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_out_start = (h + pad_h - kh + stride_h - 1) / stride_h;
                int w_out_start = (w + pad_w - kw + stride_w - 1) / stride_w;
                
                if (h_out_start < out_height && w_out_start < out_width &&
                    (h + pad_h - kh) % stride_h == 0 && 
                    (w + pad_w - kw) % stride_w == 0) {
                    
                    int col_idx = batch * (channels * kernel_h * kernel_w * out_height * out_width) +
                                 (c * kernel_h * kernel_w + kh * kernel_w + kw) * out_height * out_width +
                                 h_out_start * out_width + w_out_start;
                    val += col[col_idx];
                }
            }
        }
        
        im[index] = val;
    }
}
"#;

/// CUDA kernel for gradient w.r.t. bias
pub const BIAS_GRAD_KERNEL: &str = r#"
extern "C" __global__ void bias_grad_kernel(
    const float* grad_output,
    float* grad_bias,
    int batch_size,
    int channels,
    int spatial_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    
    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < spatial_size; i++) {
            int idx = b * channels * spatial_size + c * spatial_size + i;
            sum += grad_output[idx];
        }
    }
    
    grad_bias[c] = sum;
}
"#;

/// CUDA kernel for bias addition
pub const ADD_BIAS_KERNEL: &str = r#"
extern "C" __global__ void add_bias_kernel(
    float* output,
    const float* bias,
    int batch_size,
    int channels,
    int spatial_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_size;
    
    if (index < total_elements) {
        int c = (index / spatial_size) % channels;
        output[index] += bias[c];
    }
}
"#;

/// GPU-accelerated 2D convolution
pub struct CudaConv2d;

impl CudaConv2d {
    /// Ensure kernels are loaded
    fn ensure_kernels(device: &Arc<CudaDevice>) -> Result<()> {
        use crate::cuda_kernels::CudaKernels;
        
        CudaKernels::ensure_kernel(device, "im2col_kernel", IM2COL_KERNEL)?;
        CudaKernels::ensure_kernel(device, "col2im_kernel", COL2IM_KERNEL)?;
        CudaKernels::ensure_kernel(device, "add_bias_kernel", ADD_BIAS_KERNEL)?;
        CudaKernels::ensure_kernel(device, "bias_grad_kernel", BIAS_GRAD_KERNEL)?;
        
        Ok(())
    }
    
    /// Forward convolution using im2col + matmul
    pub fn conv2d_forward(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Tensor> {
        if groups != 1 {
            return Err(FlameError::InvalidOperation(
                "Grouped convolution not yet implemented".into()
            ));
        }
        
        let device = input.device();
        Self::ensure_kernels(device)?;
        
        // Get dimensions
        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();
        
        let batch_size = input_dims[0];
        let in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];
        
        let out_channels = weight_dims[0];
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];
        
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
        
        // Allocate col buffer for im2col
        let col_size = batch_size * in_channels * kernel_h * kernel_w * out_height * out_width;
        let col_buffer = device.alloc_zeros::<f32>(col_size)?;
        
        // Perform im2col
        let f = device.get_func("im2col_kernel", "im2col_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get im2col kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems(col_size as u32);
        // Use proper device pointer types
        unsafe {
            f.launch(cfg, (
                input.data(),
                &col_buffer,
                batch_size as i32,
                in_channels as i32,
                in_height as i32,
                in_width as i32,
                kernel_h as i32,
                kernel_w as i32,
            ))?;
        }
        
        // Reshape for matrix multiplication
        let col_tensor = Tensor {
            data: Arc::new(col_buffer),
            shape: Shape::from_dims(&[batch_size * out_height * out_width, in_channels * kernel_h * kernel_w]),
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        // Reshape weight: [out_channels, in_channels, kh, kw] -> [out_channels, in_channels * kh * kw]
        let weight_2d = weight.reshape(&[out_channels, in_channels * kernel_h * kernel_w])?;
        
        // Perform matrix multiplication: [out_channels, in_c*kh*kw] @ [b*oh*ow, in_c*kh*kw]^T
        let weight_t = weight_2d.transpose()?;
        let output_2d = col_tensor.matmul(&weight_t)?;
        
        // Reshape output: [b*oh*ow, out_channels] -> [b, out_channels, oh, ow]
        let mut output = output_2d.reshape(&[batch_size, out_height, out_width, out_channels])?
            .permute(&[0, 3, 1, 2])?;
        
        // Add bias if provided
        if let Some(b) = bias {
            output = Self::add_bias(&output, b)?;
        }
        
        // Record operation for autograd
        if input.requires_grad || weight.requires_grad || (bias.is_some() && bias.unwrap().requires_grad) {
            output.requires_grad = true;
            
            let mut saved_tensors = vec![
                (input.id, input.clone()?),
                (weight.id, weight.clone()?),
            ];
            
            let bias_id = if let Some(b) = bias {
                saved_tensors.push((b.id, b.clone()?));
                Some(b.id)
            } else {
                None
            };
            
            AutogradContext::record_op(
                output.id,
                Op::Conv2d {
                    input: input.id,
                    weight: weight.id,
                    stride: stride_h,
                    padding: pad_h,
                },
                saved_tensors,
            );
        }
        
        Ok(output)
    }
    
    /// Add bias using CUDA kernel
    fn add_bias(output: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let device = output.device();
        let output_dims = output.shape().dims();
        
        let batch_size = output_dims[0];
        let channels = output_dims[1];
        let spatial_size = output_dims[2] * output_dims[3];
        
        let mut result = output.clone()?;
        
        let f = device.get_func("add_bias_kernel", "add_bias_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get add_bias kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems((batch_size * channels * spatial_size) as u32);
        
        unsafe {
            f.launch(cfg, (
                result.data(),
                bias.data(),
                batch_size as i32,
                channels as i32,
                spatial_size as i32,
            ))?;
        }
        
        Ok(result)
    }
    
    /// Backward pass for convolution (used in autograd)
    pub fn conv2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let device = input.device();
        Self::ensure_kernels(device)?;
        
        // Get dimensions
        let grad_dims = grad_output.shape().dims();
        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();
        
        let batch_size = grad_dims[0];
        let out_channels = grad_dims[1];
        let out_height = grad_dims[2];
        let out_width = grad_dims[3];
        
        let in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];
        
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];
        
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        
        // Gradient w.r.t. input using transposed convolution
        // First, perform im2col on grad_output
        let col_size = batch_size * out_channels * out_height * out_width;
        let grad_col = grad_output.reshape(&[batch_size * out_height * out_width, out_channels])?;
        
        // Weight gradient: grad_output @ input^T (after im2col)
        // First, im2col on input
        let input_col_size = batch_size * in_channels * kernel_h * kernel_w * out_height * out_width;
        let input_col_buffer = device.alloc_zeros::<f32>(input_col_size)?;
        
        let f_im2col = device.get_func("im2col_kernel", "im2col_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get im2col kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems(input_col_size as u32);
        unsafe {
            f_im2col.launch(cfg, (
                input.data(),
                &input_col_buffer,
                batch_size as i32,
                in_channels as i32,
                in_height as i32,
                in_width as i32,
                kernel_h as i32,
                kernel_w as i32,
            ))?;
        }
        
        // Compute weight gradient
        let input_col_tensor = Tensor {
            data: Arc::new(input_col_buffer),
            shape: Shape::from_dims(&[batch_size * out_height * out_width, in_channels * kernel_h * kernel_w]),
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        // grad_weight = grad_output^T @ input_col
        let grad_col_t = grad_col.transpose()?;
        let grad_weight_2d = grad_col_t.matmul(&input_col_tensor)?;
        let grad_weight = grad_weight_2d.reshape(&[out_channels, in_channels, kernel_h, kernel_w])?;
        
        // Gradient w.r.t. input using col2im
        // First compute weight^T @ grad_output
        let weight_t = weight.reshape(&[out_channels, in_channels * kernel_h * kernel_w])?.transpose()?;
        let grad_input_col = grad_col.matmul(&weight_t)?;
        
        // Now use col2im to get grad_input
        let grad_input_data = device.alloc_zeros::<f32>(input.shape().elem_count())?;
        
        let f_col2im = device.get_func("col2im_kernel", "col2im_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get col2im kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems(input.shape().elem_count() as u32);
        unsafe {
            f_col2im.launch(cfg, (
                grad_input_col.data(),
                &grad_input_data,
                batch_size as i32,
                in_channels as i32,
                in_height as i32,
                in_width as i32,
                kernel_h as i32,
                kernel_w as i32,
            ))?;
        }
        
        let grad_input = Tensor {
            data: Arc::new(grad_input_data),
            shape: input.shape().clone(),
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        // Bias gradient is sum over batch and spatial dimensions
        let grad_bias = if grad_output.shape().dims().len() == 4 {
            let grad_bias_data = device.alloc_zeros::<f32>(out_channels)?;
            
            let f_bias_grad = device.get_func("bias_grad_kernel", "bias_grad_kernel")
                .ok_or_else(|| FlameError::Cuda("Failed to get bias_grad kernel".into()))?;
            
            let cfg = LaunchConfig::for_num_elems(out_channels as u32);
            unsafe {
                f_bias_grad.launch(cfg, (
                    grad_output.data(),
                    &grad_bias_data,
                    batch_size as i32,
                    out_channels as i32,
                    (out_height * out_width) as i32,
                ))?;
            }
            
            Some(Tensor {
                data: Arc::new(grad_bias_data),
                shape: Shape::from_dims(&[out_channels]),
                device: device.clone(),
                id: TensorId::new(),
                requires_grad: false,
            })
        } else {
            None
        };
        
        Ok((grad_input, grad_weight, grad_bias))
    }
}

/// Convenience function for Conv2D forward
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    CudaConv2d::conv2d_forward(
        input,
        weight,
        bias,
        (stride, stride),
        (padding, padding),
        1,
    )
}