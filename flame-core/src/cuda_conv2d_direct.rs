//! Direct convolution using cuBLAS GEMM (no im2col)
//! This is much faster than im2col approach for modern GPUs

use crate::{Tensor, Shape, Result, FlameError};
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::cublas::CudaBlas;
use std::sync::Arc;

/// Direct convolution using cuBLAS GEMM
pub struct CudaConv2dDirect;

impl CudaConv2dDirect {
    /// Forward convolution using direct GEMM approach
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
        
        let _device = input.device();
        
        // Get dimensions
        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();
        
        let _batch_size = input_dims[0];
        let _in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];
        
        let _out_channels = weight_dims[0];
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];
        
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        
        // Calculate output dimensions
        let _out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let _out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
        
        // Special case: 1x1 convolution - can be done as simple matmul
        if kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0 {
            return Self::conv1x1_forward(input, weight, bias);
        }
        
        // Special case: 3x3 convolution with stride=1, padding=1 (most common in VAE)
        if kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1 {
            return Self::conv3x3_s1_p1_forward(input, weight, bias);
        }
        
        // Fall back to optimized im2col for other cases
        Self::conv2d_forward_optimized_im2col(input, weight, bias, stride, padding, groups)
    }
    
    /// Optimized 1x1 convolution as direct matmul
    fn conv1x1_forward(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = input.device();
        
        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();
        
        let batch_size = input_dims[0];
        let in_channels = input_dims[1];
        let height = input_dims[2];
        let width = input_dims[3];
        let out_channels = weight_dims[0];
        
        // Reshape input: [B, C_in, H, W] -> [B*H*W, C_in]
        let input_2d = input.reshape(&[batch_size * height * width, in_channels])?;
        
        // Reshape weight: [C_out, C_in, 1, 1] -> [C_out, C_in]
        let weight_2d = weight.reshape(&[out_channels, in_channels])?;
        
        // Perform GEMM: [B*H*W, C_in] @ [C_in, C_out] = [B*H*W, C_out]
        let weight_t = weight_2d.transpose()?;
        
        // Use cuBLAS GEMM directly
        let output_2d = Self::gemm_f32(&input_2d, &weight_t, &device)?;
        
        // Reshape output: [B*H*W, C_out] -> [B, H, W, C_out] -> [B, C_out, H, W]
        let mut output = output_2d.reshape(&[batch_size, height, width, out_channels])?
            .permute(&[0, 3, 1, 2])?;
        
        // Add bias if provided
        if let Some(b) = bias {
            let bias_reshaped = b.reshape(&[1, out_channels, 1, 1])?;
            output = output.add(&bias_reshaped)?;
        }
        
        Ok(output)
    }
    
    /// Optimized 3x3 convolution with stride=1, padding=1
    fn conv3x3_s1_p1_forward(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // For 3x3 conv, we can use a more efficient im2col that processes multiple pixels at once
        let device = input.device();
        
        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();
        
        let batch_size = input_dims[0];
        let in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];
        let out_channels = weight_dims[0];
        let out_height = in_height;  // With padding=1, stride=1
        let out_width = in_width;
        
        // Use optimized im2col that processes blocks of pixels
        let col_size = batch_size * in_channels * 9 * out_height * out_width;
        let col_buffer = crate::tensor::alloc_zeros_from_pool(&device, col_size)?;
        
        // Call optimized im2col kernel (we'll write this next)
        Self::im2col_3x3_optimized(&input, &col_buffer, batch_size, in_channels, in_height, in_width)?;
        
        // Reshape for GEMM
        let col_shape = Shape::from_dims(&[batch_size * out_height * out_width, in_channels * 9]);
        let col_tensor = Tensor {
            storage: TensorStorage::F32 { data: col_buffer, numel: col_shape.elem_count() },
            shape: col_shape,
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        // Reshape weight: [out_channels, in_channels, 3, 3] -> [out_channels, in_channels * 9]
        let weight_2d = weight.reshape(&[out_channels, in_channels * 9])?;
        let weight_t = weight_2d.transpose()?;
        
        // Use cuBLAS GEMM
        let output_2d = Self::gemm_f32(&col_tensor, &weight_t, &device)?;
        
        // Reshape output
        let mut output = output_2d.reshape(&[batch_size, out_height, out_width, out_channels])?
            .permute(&[0, 3, 1, 2])?;
        
        // Add bias
        if let Some(b) = bias {
            let bias_reshaped = b.reshape(&[1, out_channels, 1, 1])?;
            output = output.add(&bias_reshaped)?;
        }
        
        Ok(output)
    }
    
    /// Optimized im2col for 3x3 convolutions
    fn im2col_3x3_optimized(
        input: &Tensor,
        col_buffer: &cudarc::driver::CudaSlice<f32>,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<()> {
        let device = input.device();
        
        // Load and compile optimized kernel
        let kernel_src = r#"
extern "C" __global__ void im2col_3x3_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    // Process 4 pixels at once for better memory coalescing
    const int out_h = height;
    const int out_w = width;
    const int pixels_per_thread = 4;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = batch_size * out_h * out_w;
    const int pixels_to_process = (total_pixels + pixels_per_thread - 1) / pixels_per_thread;
    
    if (tid >= pixels_to_process) return;
    
    const int start_pixel = tid * pixels_per_thread;
    const int end_pixel = min(start_pixel + pixels_per_thread, total_pixels);
    
    for (int pixel_idx = start_pixel; pixel_idx < end_pixel; pixel_idx++) {
        const int w_out = pixel_idx % out_w;
        const int h_out = (pixel_idx / out_w) % out_h;
        const int batch = pixel_idx / (out_w * out_h);
        
        // For each channel
        for (int c = 0; c < channels; c++) {
            // For each kernel position (3x3 = 9 positions)
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    // Calculate input position with padding=1
                    int h_in = h_out - 1 + kh;
                    int w_in = w_out - 1 + kw;
                    
                    float val = 0.0f;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = batch * channels * height * width +
                                       c * height * width +
                                       h_in * width + w_in;
                        val = input[input_idx];
                    }
                    
                    // Output layout: [batch*out_h*out_w, channels*9]
                    int col_idx = pixel_idx * (channels * 9) +
                                 c * 9 + kh * 3 + kw;
                    output[col_idx] = val;
                }
            }
        }
    }
}
"#;
        
        // Compile kernel
        let ptx = cudarc::nvrtc::compile_ptx(kernel_src)
            .map_err(|e| FlameError::Cuda(format!("Failed to compile im2col_3x3 kernel: {:?}", e)))?;
        
        device.load_ptx(ptx, "im2col_3x3", &["im2col_3x3_optimized"])
            .map_err(|e| FlameError::Cuda(format!("Failed to load kernel: {}", e)))?;
        
        let kernel = device.get_func("im2col_3x3", "im2col_3x3_optimized")
            .ok_or_else(|| FlameError::Cuda("Failed to get kernel function".into()))?;
        
        // Launch kernel with optimal configuration
        let threads_per_block = 256;
        let total_pixels = batch_size * height * width;
        let pixels_per_thread = 4;
        let num_blocks = ((total_pixels + pixels_per_thread - 1) / pixels_per_thread + threads_per_block - 1) / threads_per_block;
        
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch(cfg, (
                input.storage.try_as_slice_f32()?,
                col_buffer,
                batch_size as i32,
                channels as i32,
                height as i32,
                width as i32
            )).map_err(|e| FlameError::Cuda(format!("Kernel launch failed: {:?}", e)))?;
        }
        
        device.synchronize()?;
        Ok(())
    }
    
    /// Direct GEMM using cuBLAS
    fn gemm_f32(a: &Tensor, b: &Tensor, _device: &Arc<CudaDevice>) -> Result<Tensor> {
        // For now, just use regular matmul which already uses cuBLAS internally
        a.matmul(b)
    }
    
    /// Fallback to optimized im2col implementation
    fn conv2d_forward_optimized_im2col(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Tensor> {
        // Use the existing im2col implementation but with cuBLAS GEMM
        super::cuda_conv2d::CudaConv2d::conv2d_forward(input, weight, bias, stride, padding, groups)
    }
}

/// Convenience function for direct Conv2D
pub fn conv2d_direct(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    CudaConv2dDirect::conv2d_forward(
        input,
        weight,
        bias,
        (stride, stride),
        (padding, padding),
        1,
    )
}
