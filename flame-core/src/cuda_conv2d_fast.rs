//! Fast direct convolution using optimized CUDA kernels (no im2col)
//! Based on the production-ready kernels from cuda_txt/forward.txt

use crate::{Tensor, Shape, Result, FlameError};
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Parameters for convolution kernel
#[repr(C)]
#[derive(Clone, Copy)]
struct Conv2dParams {
    batch_size: i32,
    in_channels: i32,
    out_channels: i32,
    in_height: i32,
    in_width: i32,
    kernel_h: i32,
    kernel_w: i32,
    out_height: i32,
    out_width: i32,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
}

// Safety: Conv2dParams is a POD type with repr(C)
unsafe impl cudarc::driver::DeviceRepr for Conv2dParams {}

use std::sync::Mutex;
use std::collections::HashMap;
use lazy_static::lazy_static;

lazy_static! {
    static ref KERNEL_CACHE: Mutex<HashMap<String, bool>> = Mutex::new(HashMap::new());
}

/// Fast direct convolution without im2col
pub struct CudaConv2dFast;

impl CudaConv2dFast {
    /// Forward convolution using direct CUDA kernel
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
                "Grouped convolution not yet implemented in fast path".into()
            ));
        }
        
        let device = input.device();
        
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
        
        // Allocate output
        let output_size = batch_size * out_channels * out_height * out_width;
        let output_data = crate::tensor::alloc_zeros_from_pool(device, output_size)?;
        
        // Compile and launch the optimized kernel
        let kernel_code = r###"
struct Conv2dParams {
    int batch_size;
    int in_channels;
    int out_channels;
    int in_height;
    int in_width;
    int kernel_h;
    int kernel_w;
    int out_height;
    int out_width;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
};

extern "C" __global__ void conv2d_direct(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const Conv2dParams* __restrict__ params_ptr
) {{
    const Conv2dParams params = *params_ptr;
    const int batch_size = params.batch_size;
    const int in_channels = params.in_channels;
    const int out_channels = params.out_channels;
    const int in_height = params.in_height;
    const int in_width = params.in_width;
    const int kernel_h = params.kernel_h;
    const int kernel_w = params.kernel_w;
    const int out_height = params.out_height;
    const int out_width = params.out_width;
    const int stride_h = params.stride_h;
    const int stride_w = params.stride_w;
    const int pad_h = params.pad_h;
    const int pad_w = params.pad_w;
    // Calculate global position
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_output = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_output) return;
    
    // Decompose tid into output coordinates
    const int n = tid / (out_channels * out_height * out_width);
    const int remainder = tid - n * (out_channels * out_height * out_width);
    const int c_out = remainder / (out_height * out_width);
    const int spatial = remainder - c_out * (out_height * out_width);
    const int out_h = spatial / out_width;
    const int out_w = spatial - out_h * out_width;
    
    // Initialize with bias if present
    float sum = bias ? bias[c_out] : 0.0f;
    
    // Perform direct convolution (no im2col)
    for (int c_in = 0; c_in < in_channels; c_in++) {{
        for (int kh = 0; kh < kernel_h; kh++) {{
            for (int kw = 0; kw < kernel_w; kw++) {{
                const int in_h = out_h * stride_h - pad_h + kh;
                const int in_w = out_w * stride_w - pad_w + kw;
                
                // Check bounds
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {{
                    // Calculate input index
                    const int input_idx = n * in_channels * in_height * in_width +
                                         c_in * in_height * in_width +
                                         in_h * in_width + in_w;
                    
                    // Calculate weight index
                    const int weight_idx = c_out * in_channels * kernel_h * kernel_w +
                                          c_in * kernel_h * kernel_w +
                                          kh * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }}
            }}
        }}
    }}
    
    // Write output
    output[tid] = sum;
}}

// Optimized version for 3x3 convolutions with stride=1, padding=1
extern "C" __global__ void conv2d_3x3_s1_p1(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const Conv2dParams* __restrict__ params_ptr
) {{
    const Conv2dParams params = *params_ptr;
    const int batch_size = params.batch_size;
    const int in_channels = params.in_channels;
    const int out_channels = params.out_channels;
    const int height = params.in_height;
    const int width = params.in_width;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_output = batch_size * out_channels * height * width;
    
    if (tid >= total_output) return;
    
    // Decompose tid
    const int n = tid / (out_channels * height * width);
    const int remainder = tid - n * (out_channels * height * width);
    const int c_out = remainder / (height * width);
    const int spatial = remainder - c_out * (height * width);
    const int h = spatial / width;
    const int w = spatial - h * width;
    
    float sum = bias ? bias[c_out] : 0.0f;
    
    // Unrolled 3x3 convolution with padding=1
    for (int c_in = 0; c_in < in_channels; c_in++) {{
        const int base_idx = n * in_channels * height * width + c_in * height * width;
        const int weight_base = c_out * in_channels * 9 + c_in * 9;
        
        // Process 3x3 kernel
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {{
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {{
                const int in_h = h - 1 + kh;
                const int in_w = w - 1 + kw;
                
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {{
                    sum += input[base_idx + in_h * width + in_w] * 
                           weight[weight_base + kh * 3 + kw];
                }}
            }}
        }}
    }}
    
    output[tid] = sum;
}}
"###;
        
        // Compile kernel
        let ptx = cudarc::nvrtc::compile_ptx(&kernel_code)
            .map_err(|e| FlameError::Cuda(format!("Failed to compile conv2d kernel: {:?}", e)))?;
        
        // Choose kernel based on configuration
        let kernel_name = if kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1 {
            "conv2d_3x3_s1_p1"
        } else {
            "conv2d_direct"
        };
        
        device.load_ptx(ptx, "conv2d_fast", &[kernel_name])
            .map_err(|e| FlameError::Cuda(format!("Failed to load kernel: {}", e)))?;
        
        // Launch configuration
        let threads_per_block = 256;
        let total_threads = output_size;
        let num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
        
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Launch kernel
        // Create a dummy zero bias if not provided
        let zero_bias = if bias.is_none() {
            let zero_data = crate::tensor::alloc_zeros_from_pool(device, out_channels)?;
            Some(zero_data)
        } else {
            None
        };
        
        let bias_slice = if let Some(b) = bias {
            b.storage.as_slice()
        } else {
            zero_bias.as_ref().unwrap()
        };
        
        // Create parameter struct
        let params = Conv2dParams {
            batch_size: batch_size as i32,
            in_channels: in_channels as i32,
            out_channels: out_channels as i32,
            in_height: in_height as i32,
            in_width: in_width as i32,
            kernel_h: kernel_h as i32,
            kernel_w: kernel_w as i32,
            out_height: out_height as i32,
            out_width: out_width as i32,
            stride_h: stride_h as i32,
            stride_w: stride_w as i32,
            pad_h: pad_h as i32,
            pad_w: pad_w as i32,
        };
        
        // Allocate params buffer on device
        let params_slice = device.htod_copy(vec![params])?;
        
        // Get the kernel function
        let kernel_func = device.get_func("conv2d_fast", kernel_name)
            .ok_or_else(|| FlameError::Cuda("Failed to get kernel function".into()))?;
        
        // Launch kernel with packed parameters
        unsafe {
            kernel_func.launch(cfg, (
                input.storage.as_slice(),
                weight.storage.as_slice(),
                bias_slice,
                &output_data,
                &params_slice,
            )).map_err(|e| FlameError::Cuda(format!("Kernel launch failed: {:?}", e)))?;
        }
        
        device.synchronize()?;
        
        // Create output tensor
        Ok(Tensor {
            storage: TensorStorage::F32 { 
                data: output_data, 
                numel: output_size 
            },
            shape: Shape::from_dims(&[batch_size, out_channels, out_height, out_width]),
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        })
    }
}

/// Convenience function for fast Conv2D
pub fn conv2d_fast(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    CudaConv2dFast::conv2d_forward(
        input,
        weight,
        bias,
        (stride, stride),
        (padding, padding),
        1,
    )
}