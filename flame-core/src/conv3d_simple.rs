//! Simple 3D Convolution and normalization layers for video processing
//! 
//! This module provides 3D convolution operations and batch normalization
//! for processing video data with temporal dimensions.
//! Contains CPU-oriented 3D conv helpers; CUDA kernels are used elsewhere in active paths.

use crate::{Tensor, Shape, Result, FlameError, CudaDevice};
use std::sync::Arc;
use cudarc::driver::{LaunchAsync, LaunchConfig, CudaSlice};

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


// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// 3D Convolution layer
pub struct Conv3d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize, usize),  // (depth, height, width)
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub groups: usize,
    pub bias: bool,
    
    // Parameters
    pub weight: Tensor,
    pub bias_tensor: Option<Tensor>,
    
    // Device
    pub device: Arc<CudaDevice>,
}

impl Conv3d {
    /// Create a new Conv3d layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: Option<(usize, usize, usize)>,
        dilation: Option<(usize, usize, usize)>,
        groups: Option<usize>,
        bias: bool,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let stride = stride.unwrap_or((1, 1, 1));
        let padding = padding.unwrap_or((0, 0, 0));
        let dilation = dilation.unwrap_or((1, 1, 1));
        let groups = groups.unwrap_or(1);
        
        // Validate parameters
        if in_channels % groups != 0 {
            return Err(FlameError::InvalidOperation(
                format!("in_channels {} must be divisible by groups {}", in_channels, groups)
            ));
        }
        
        if out_channels % groups != 0 {
            return Err(FlameError::InvalidOperation(
                format!("out_channels {} must be divisible by groups {}", out_channels, groups)
            ));
        }
        
        // Initialize weight: [out_channels, in_channels/groups, kd, kh, kw]
        let weight_shape = Shape::from_dims(&[
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
        ]);
        
        let fan_in = (in_channels / groups) * kernel_size.0 * kernel_size.1 * kernel_size.2;
        let scale = (2.0 / fan_in as f32).sqrt();
        let weight = Tensor::randn(weight_shape, 0.0, scale, device.clone())?;
        
        // Initialize bias if needed
        let bias_tensor = if bias {
            Some(Tensor::zeros(Shape::from_dims(&[out_channels]), device.clone())?)
        } else {
            None
        };
        
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight,
            bias_tensor,
            device,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Validate input shape: [N, C, D, H, W]
        let input_shape = input.shape().dims();
        if input_shape.len() != 5 {
            return Err(FlameError::InvalidOperation(
                format!("Conv3d expects 5D input [N,C,D,H,W], got {:?}", input_shape)
            ));
        }
        
        let (batch_size, in_channels, d_in, h_in, w_in) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        );
        
        if in_channels != self.in_channels {
            return Err(FlameError::InvalidOperation(
                format!("Expected {} input channels, got {}", self.in_channels, in_channels)
            ));
        }
        
        // Calculate output dimensions
        let d_out = (d_in + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1) / self.stride.0 + 1;
        let h_out = (h_in + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1) / self.stride.1 + 1;
        let w_out = (w_in + 2 * self.padding.2 - self.dilation.2 * (self.kernel_size.2 - 1) - 1) / self.stride.2 + 1;
        
        let output_shape = Shape::from_dims(&[batch_size, self.out_channels, d_out, h_out, w_out]);
        
        // Perform convolution
        let mut output = self.conv3d_cpu(input, &output_shape)?;
        
        // Add bias if present
        if let Some(bias) = &self.bias_tensor {
            self.add_bias_3d(&mut output, bias)?;
        }
        
        Ok(output)
    }
    
    /// CPU implementation of 3D convolution
    fn conv3d_cpu(&self, input: &Tensor, output_shape: &Shape) -> Result<Tensor> {
        // Extract dimensions from shapes
        let input_shape = input.shape().dims();
        let batch = input_shape[0];
        let d_in = input_shape[2];
        let h_in = input_shape[3];
        let w_in = input_shape[4];
        
        let output_dims = output_shape.dims();
        let d_out = output_dims[2];
        let h_out = output_dims[3];
        let w_out = output_dims[4];
        
        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        
        // Simple CPU implementation
        // Now actually tries to perform 3D convolution
        let kernel_code = r#"
extern "C" __global__ void conv3d_forward(
    float *output,
    const float *input,
    const float *weight,
    const int *input_dims,    // [batch, in_channels, d_in, h_in, w_in]
    const int *output_dims,   // [out_channels, d_out, h_out, w_out]
    const int *kernel_dims,   // [kernel_d, kernel_h, kernel_w]
    const int *conv_params    // [stride_d, stride_h, stride_w, pad_d, pad_h, pad_w]
) {
    // Unpack dimensions from arrays
    int batch = input_dims[0];
    int in_channels = input_dims[1];
    int d_in = input_dims[2];
    int h_in = input_dims[3];
    int w_in = input_dims[4];
    
    int out_channels = output_dims[0];
    int d_out = output_dims[1];
    int h_out = output_dims[2];
    int w_out = output_dims[3];
    
    int kernel_d = kernel_dims[0];
    int kernel_h = kernel_dims[1];
    int kernel_w = kernel_dims[2];
    
    int stride_d = conv_params[0];
    int stride_h = conv_params[1];
    int stride_w = conv_params[2];
    int pad_d = conv_params[3];
    int pad_h = conv_params[4];
    int pad_w = conv_params[5];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * d_out * h_out * w_out;
    
    if (idx < total) {
        int w = idx % w_out;
        int h = (idx / w_out) % h_out;
        int d = (idx / (w_out * h_out)) % d_out;
        int oc = (idx / (w_out * h_out * d_out)) % out_channels;
        int b = idx / (out_channels * d_out * h_out * w_out);
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_d; kd++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int d_in_idx = d * stride_d - pad_d + kd;
                        int h_in_idx = h * stride_h - pad_h + kh;
                        int w_in_idx = w * stride_w - pad_w + kw;
                        
                        if (d_in_idx >= 0 && d_in_idx < d_in &&
                            h_in_idx >= 0 && h_in_idx < h_in &&
                            w_in_idx >= 0 && w_in_idx < w_in) {
                            
                            int in_idx = ((b * in_channels + ic) * d_in + d_in_idx) * h_in * w_in + h_in_idx * w_in + w_in_idx;
                            int w_idx = ((oc * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                            
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}"#;
        
        // Launch the kernel
        crate::cuda_kernels_gpu::CudaKernels::ensure_kernel(&self.device, "conv3d_forward", kernel_code)?;
        
        let f = self.device.get_func("conv3d_forward", "conv3d_forward")
            .ok_or_else(|| crate::FlameError::Cuda("Failed to get conv3d kernel".into()))?;
        
        let output_numel = output_shape.elem_count();
        let mut output_data = crate::tensor::alloc_from_pool(&self.device, output_numel)
            .map_err(|_| crate::FlameError::CudaDriver)?;
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(output_numel as u32);
        // Pack dimensions into arrays to reduce parameter count
        let input_dims = vec![batch as i32, self.in_channels as i32, d_in as i32, h_in as i32, w_in as i32];
        let output_dims = vec![self.out_channels as i32, d_out as i32, h_out as i32, w_out as i32];
        let kernel_dims = vec![kd as i32, kh as i32, kw as i32];
        let conv_params = vec![sd as i32, sh as i32, sw as i32, pd as i32, ph as i32, pw as i32];
        
        let input_dims_gpu = alloc_from_pool_and_copy(&self.device, &input_dims)?;
        let output_dims_gpu = alloc_from_pool_and_copy(&self.device, &output_dims)?;
        let kernel_dims_gpu = alloc_from_pool_and_copy(&self.device, &kernel_dims)?;
        let conv_params_gpu = alloc_from_pool_and_copy(&self.device, &conv_params)?;
        
        launch_kernel!(f, cfg,
            &output_data,
            input.storage.try_as_slice_f32()?,
            self.weight.storage.try_as_slice_f32()?,
            &input_dims_gpu,
            &output_dims_gpu,
            &kernel_dims_gpu,
            &conv_params_gpu
        )?;
        
        Ok(crate::cuda_kernels::create_output_tensor(output_data, output_shape.clone(), self.device.clone()))
    }
    
    /// Add bias to output tensor
    fn add_bias_3d(&self, output: &mut Tensor, bias: &Tensor) -> Result<()> {
        // Always use CUDA since we're in a CUDA-only context
        // Use CUDA kernel for efficient bias addition
        let dims = output.shape().dims();
        let out_channels = dims[1];
        
        // Reshape bias for broadcasting and add
        let bias_shape = Shape::from_dims(&[1, out_channels, 1, 1, 1]);
        let bias_reshaped = bias.reshape(bias_shape.dims())?;
        
        // Broadcast and add using CUDA operations
        *output = output.add(&bias_reshaped)?;
        Ok(())
    }
}

/// Batch Normalization 3D layer
pub struct BatchNorm3d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub affine: bool,
    pub track_running_stats: bool,
    
    // Learnable parameters
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    
    // Running statistics
    pub running_mean: Option<Tensor>,
    pub running_var: Option<Tensor>,
    pub num_batches_tracked: usize,
    
    // Device
    pub device: Arc<CudaDevice>,
}

impl BatchNorm3d {
    /// Create a new BatchNorm3d layer
    pub fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);
        
        let (weight, bias) = if affine {
            let weight = Tensor::from_vec(
                vec![1.0f32; num_features],
                Shape::from_dims(&[num_features]),
                device.clone(),
            )?;
            let bias = Tensor::zeros(Shape::from_dims(&[num_features]), device.clone())?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        
        let (running_mean, running_var) = if track_running_stats {
            let running_mean = Tensor::zeros(Shape::from_dims(&[num_features]), device.clone())?;
            let running_var = Tensor::from_vec(
                vec![1.0f32; num_features],
                Shape::from_dims(&[num_features]),
                device.clone(),
            )?;
            (Some(running_mean), Some(running_var))
        } else {
            (None, None)
        };
        
        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
            num_batches_tracked: 0,
            device,
        })
    }
    
    /// Forward pass
    pub fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor> {
        // Validate input shape: [N, C, D, H, W]
        let dims = input.shape().dims();
        if dims.len() != 5 {
            return Err(FlameError::InvalidOperation(
                format!("BatchNorm3d expects 5D input [N,C,D,H,W], got {:?}", dims)
            ));
        }
        
        let num_channels = dims[1];
        if num_channels != self.num_features {
            return Err(FlameError::InvalidOperation(
                format!("Expected {} channels, got {}", self.num_features, num_channels)
            ));
        }
        
        // Now actually tries to perform batch normalization
        let kernel_code = r#"
extern "C" __global__ void batchnorm3d_forward(
    float *output,
    const float *input,
    const float *mean,
    const float *var,
    const float *weight,
    const float *bias,
    float eps,
    int batch,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * spatial_size;
    
    if (idx < total) {
        int c = (idx / spatial_size) % channels;
        
        float val = input[idx];
        float m = mean[c];
        float v = var[c];
        float w = weight ? weight[c] : 1.0f;
        float b = bias ? bias[c] : 0.0f;
        
        // Normalize
        float normalized = (val - m) / sqrtf(v + eps);
        
        // Scale and shift
        output[idx] = normalized * w + b;
    }
}"#;
        
        crate::cuda_kernels_gpu::CudaKernels::ensure_kernel(&self.device, "batchnorm3d_forward", kernel_code)?;
        
        let f = self.device.get_func("batchnorm3d_forward", "batchnorm3d_forward")
            .ok_or_else(|| FlameError::Cuda("Failed to get batchnorm3d kernel".into()))?;
        
        let spatial_size = dims[2] * dims[3] * dims[4];
        let numel = input.shape().elem_count();
        let mut output_data = crate::tensor::alloc_from_pool(&self.device, numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        // Use running stats if available, otherwise compute batch stats
        let (batch_mean, batch_var);
        let (mean, var) = if let (Some(rm), Some(rv)) = (&self.running_mean, &self.running_var) {
            (rm.storage.try_as_slice_f32()?, rv.storage.try_as_slice_f32()?)
        } else {
            // Compute batch statistics
            batch_mean = self.compute_batch_mean(input)?;
            batch_var = self.compute_batch_var(input, &batch_mean)?;
            (batch_mean.storage.try_as_slice_f32()?, batch_var.storage.try_as_slice_f32()?)
        };
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(numel as u32);
        launch_kernel!(f, cfg,
            &output_data,
            input.storage.try_as_slice_f32()?,
            mean,
            var,
            self.weight.as_ref().map(|w| w.storage.try_as_slice_f32()).transpose()?.unwrap_or(mean), // Use mean as dummy for null pointer
            self.bias.as_ref().map(|b| b.storage.try_as_slice_f32()).transpose()?.unwrap_or(mean),   // Use mean as dummy for null pointer
            self.eps,
            dims[0] as i32,
            self.num_features as i32,
            spatial_size as i32
        )?;
        
        Ok(Tensor::from_raw(
            Arc::new(output_data),
            input.shape().clone(),
            self.device.clone(),
            input.requires_grad
        )?)
    }
    
    /// Compute batch mean for each channel
    fn compute_batch_mean(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        let batch = dims[0];
        let spatial_size = dims[2] * dims[3] * dims[4];
        let n = (batch * spatial_size) as f32;
        
        // Sum across batch and spatial dimensions for each channel
        let mut channel_sums = vec![0.0f32; self.num_features];
        let data = input.to_vec()?;
        
        for b in 0..batch {
            for c in 0..self.num_features {
                for i in 0..spatial_size {
                    let idx = b * self.num_features * spatial_size + c * spatial_size + i;
                    channel_sums[c] += data[idx];
                }
            }
        }
        
        // Divide by n to get mean
        for c in 0..self.num_features {
            channel_sums[c] /= n;
        }
        
        Tensor::from_vec(channel_sums, Shape::from_dims(&[self.num_features]), self.device.clone())
    }
    
    /// Compute batch variance for each channel
    fn compute_batch_var(&self, input: &Tensor, mean: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        let batch = dims[0];
        let spatial_size = dims[2] * dims[3] * dims[4];
        let n = (batch * spatial_size) as f32;
        
        let mean_data = mean.to_vec()?;
        let input_data = input.to_vec()?;
        
        // Sum squared differences
        let mut channel_vars = vec![0.0f32; self.num_features];
        
        for b in 0..batch {
            for c in 0..self.num_features {
                for i in 0..spatial_size {
                    let idx = b * self.num_features * spatial_size + c * spatial_size + i;
                    let diff = input_data[idx] - mean_data[c];
                    channel_vars[c] += diff * diff;
                }
            }
        }
        
        // Divide by n to get variance
        for c in 0..self.num_features {
            channel_vars[c] /= n;
        }
        
        Tensor::from_vec(channel_vars, Shape::from_dims(&[self.num_features]), self.device.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conv3d() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        let conv = Conv3d::new(
            3,  // in_channels
            16, // out_channels
            (3, 3, 3), // kernel_size
            Some((1, 1, 1)), // stride
            Some((1, 1, 1)), // padding
            None, // dilation
            None, // groups
            true, // bias
            device.clone()
        )?;
        
        // Test input: [batch=2, channels=3, depth=8, height=32, width=32]
        let input = Tensor::randn(
            Shape::from_dims(&[2, 3, 8, 32, 32]),
            0.0,
            0.1,
            device
        )?;
        
        let output = conv.forward(&input)?;
        
        // Check output shape: [2, 16, 8, 32, 32] with padding
        assert_eq!(output.shape().dims(), &[2, 16, 8, 32, 32]);
        
        Ok(())
    }
    
    #[test]
    fn test_batch_norm_3d() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        let mut bn = BatchNorm3d::new(
            16, // num_features
            None,
            None,
            None,
            None,
            device.clone()
        )?;
        
        // Test input: [batch=2, channels=16, depth=4, height=8, width=8]
        let input = Tensor::randn(
            Shape::from_dims(&[2, 16, 4, 8, 8]),
            0.0,
            1.0,
            device
        )?;
        
        let output = bn.forward(&input, true)?;
        
        // Check output shape matches input
        assert_eq!(output.shape().dims(), input.shape().dims());
        
        Ok(())
    }
}
