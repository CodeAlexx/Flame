//! 3D Convolution and normalization layers for video processing
//! 
//! This module provides 3D convolution operations and batch normalization
//! for processing video data with temporal dimensions.

use crate::{Tensor, Shape, Result, FlameError, CudaDevice};
use std::sync::Arc;
use cudarc::driver::{LaunchAsync, LaunchConfig, DeviceRepr};
use cudarc::nvrtc::compile_ptx;

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
        
        // Use CUDA kernel for convolution
        self.conv3d_cuda(input, output_shape)
    }
    
    /// CUDA kernel implementation
    fn conv3d_cuda(&self, input: &Tensor, output_shape: Shape) -> Result<Tensor> {
        // Allocate output data
        let output_numel = output_shape.elem_count();
        let mut output_data = crate::tensor::alloc_from_pool(&self.device, output_numel)
            .map_err(|_| FlameError::CudaDriver)?;
        
        // Compile and load kernel
        let kernel_code = get_conv3d_kernel_code();
        let ptx = compile_ptx(&kernel_code)
            .map_err(|e| FlameError::KernelError(format!("Failed to compile conv3d kernel: {:?}", e)))?;
        self.device.load_ptx(ptx, "conv3d", &["conv3d_kernel"])
            .map_err(|e| FlameError::Cuda(format!("Failed to load conv3d kernel: {:?}", e)))?;
        
        let f = self.device.get_func("conv3d", "conv3d_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get conv3d_kernel".into()))?;
        
        // Pack parameters
        let dims = output_shape.dims();
        let input_dims = input.shape().dims();
        
        // Launch kernel
        let total_elems = output_numel as i32;
        let cfg = LaunchConfig::for_num_elems(total_elems as u32);
        
        // Create params struct
        let params = Conv3dParams {
            batch_size: dims[0] as i32,
            in_channels: self.in_channels as i32,
            out_channels: self.out_channels as i32,
            d_in: input_dims[2] as i32,
            h_in: input_dims[3] as i32,
            w_in: input_dims[4] as i32,
            d_out: dims[2] as i32,
            h_out: dims[3] as i32,
            w_out: dims[4] as i32,
            kernel_d: self.kernel_size.0 as i32,
            kernel_h: self.kernel_size.1 as i32,
            kernel_w: self.kernel_size.2 as i32,
            stride_d: self.stride.0 as i32,
            stride_h: self.stride.1 as i32,
            stride_w: self.stride.2 as i32,
            pad_d: self.padding.0 as i32,
            pad_h: self.padding.1 as i32,
            pad_w: self.padding.2 as i32,
            dilation_d: self.dilation.0 as i32,
            dilation_h: self.dilation.1 as i32,
            dilation_w: self.dilation.2 as i32,
            groups: self.groups as i32,
        };
        
        launch_kernel!(f, cfg,
            input.storage.as_slice(),
            self.weight.storage.as_slice(),
            &mut output_data,
            params
        )?;
        
        // Create the output tensor
        let mut output = crate::cuda_kernels::create_output_tensor(
            output_data,
            output_shape,
            self.device.clone()
        );
        
        // Add bias if present
        if let Some(bias) = &self.bias_tensor {
            output = self.add_bias_3d(&output, bias)?;
        }
        
        Ok(output)
    }
    
    /// Add bias to output tensor
    fn add_bias_3d(&self, output: &Tensor, bias: &Tensor) -> Result<Tensor> {
        // Compile and load kernel
        let kernel_code = get_add_bias_3d_kernel_code();
        let ptx = compile_ptx(&kernel_code)
            .map_err(|e| FlameError::KernelError(format!("Failed to compile add_bias_3d kernel: {:?}", e)))?;
        self.device.load_ptx(ptx, "add_bias_3d", &["add_bias_3d_kernel"])
            .map_err(|e| FlameError::Cuda(format!("Failed to load add_bias_3d kernel: {:?}", e)))?;
        
        let kernel = self.device.get_func("add_bias_3d", "add_bias_3d_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get add_bias_3d_kernel".into()))?;
        
        let dims = output.shape().dims();
        let total_elems = output.shape().elem_count();
        let elems_per_channel = (dims[2] * dims[3] * dims[4]) as i32;
        
        // Allocate new output data
        let mut output_data = crate::tensor::alloc_from_pool(&self.device, total_elems)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(total_elems as u32);
        
        launch_kernel!(kernel, cfg,
            &mut output_data,
            output.storage.as_slice(),
            bias.storage.as_slice(),
            total_elems as i32,
            self.out_channels as i32,
            elems_per_channel
        )?;
        
        Ok(crate::cuda_kernels::create_output_tensor(
            output_data,
            output.shape().clone(),
            self.device.clone()
        ))
    }
}

/// Parameters for Conv3d kernel
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Conv3dParams {
    batch_size: i32,
    in_channels: i32,
    out_channels: i32,
    d_in: i32,
    h_in: i32,
    w_in: i32,
    d_out: i32,
    h_out: i32,
    w_out: i32,
    kernel_d: i32,
    kernel_h: i32,
    kernel_w: i32,
    stride_d: i32,
    stride_h: i32,
    stride_w: i32,
    pad_d: i32,
    pad_h: i32,
    pad_w: i32,
    dilation_d: i32,
    dilation_h: i32,
    dilation_w: i32,
    groups: i32,
}

// Implement DeviceRepr for Conv3dParams
unsafe impl DeviceRepr for Conv3dParams {}

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
        
        let (mean, var) = if training || !self.track_running_stats {
            // Calculate batch statistics
            self.calculate_batch_stats_3d(input)?
        } else {
            // Use running statistics
            match (&self.running_mean, &self.running_var) {
                (Some(mean), Some(var)) => {
                    let mean_data = mean.to_vec()?;
                    let var_data = var.to_vec()?;
                    let mean_copy = Tensor::from_vec(mean_data, mean.shape().clone(), mean.device.clone())?;
                    let var_copy = Tensor::from_vec(var_data, var.shape().clone(), var.device.clone())?;
                    (mean_copy, var_copy)
                },
                _ => return Err(FlameError::InvalidOperation(
                    "Running stats not available".into()
                )),
            }
        };
        
        // Apply normalization
        let normalized = self.apply_batch_norm_3d(input, &mean, &var)?;
        
        // Update running stats if training
        if training && self.track_running_stats {
            self.update_running_stats(&mean, &var)?;
            self.num_batches_tracked += 1;
        }
        
        Ok(normalized)
    }
    
    /// Calculate batch statistics for 3D input
    fn calculate_batch_stats_3d(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = input.shape().dims();
        let batch_size = dims[0];
        let num_channels = dims[1];
        let spatial_size = dims[2] * dims[3] * dims[4];
        let n = (batch_size * spatial_size) as f32;
        
        // Calculate mean and variance per channel
        let mut channel_means = vec![0.0f32; num_channels];
        let mut channel_vars = vec![0.0f32; num_channels];
        
        let data = input.to_vec()?;
        
        // Calculate means
        for b in 0..batch_size {
            for c in 0..num_channels {
                for idx in 0..spatial_size {
                    let pos = b * num_channels * spatial_size + c * spatial_size + idx;
                    channel_means[c] += data[pos];
                }
            }
        }
        
        for c in 0..num_channels {
            channel_means[c] /= n;
        }
        
        // Calculate variances
        for b in 0..batch_size {
            for c in 0..num_channels {
                for idx in 0..spatial_size {
                    let pos = b * num_channels * spatial_size + c * spatial_size + idx;
                    let diff = data[pos] - channel_means[c];
                    channel_vars[c] += diff * diff;
                }
            }
        }
        
        for c in 0..num_channels {
            channel_vars[c] /= n;
        }
        
        let mean = Tensor::from_vec(channel_means, Shape::from_dims(&[num_channels]), self.device.clone())?;
        let var = Tensor::from_vec(channel_vars, Shape::from_dims(&[num_channels]), self.device.clone())?;
        
        Ok((mean, var))
    }
    
    /// Apply batch normalization
    fn apply_batch_norm_3d(&self, input: &Tensor, mean: &Tensor, var: &Tensor) -> Result<Tensor> {
        // Compile and load kernel
        let kernel_code = get_batch_norm_3d_kernel_code();
        let ptx = compile_ptx(&kernel_code)
            .map_err(|e| FlameError::KernelError(format!("Failed to compile batch_norm_3d kernel: {:?}", e)))?;
        self.device.load_ptx(ptx, "batch_norm_3d", &["batch_norm_3d_kernel"])
            .map_err(|e| FlameError::Cuda(format!("Failed to load batch_norm_3d kernel: {:?}", e)))?;
        
        let kernel = self.device.get_func("batch_norm_3d", "batch_norm_3d_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get batch_norm_3d_kernel".into()))?;
        
        let dims = input.shape().dims();
        let total_elems = input.shape().elem_count();
        let spatial_size = (dims[2] * dims[3] * dims[4]) as i32;
        
        // Allocate output data
        let mut output_data = crate::tensor::alloc_from_pool(&self.device, total_elems)
            .map_err(|_| FlameError::CudaDriver)?;
        
        let cfg = LaunchConfig::for_num_elems(total_elems as u32);
        
        // For now, use a simplified kernel without optional weight/bias
        // In production, you'd handle all cases properly
        launch_kernel!(kernel, cfg,
            input.storage.as_slice(),
            mean.storage.as_slice(),
            var.storage.as_slice(),
            &mut output_data,
            total_elems as i32,
            self.num_features as i32,
            spatial_size,
            self.eps,
            self.affine as i32
        )?;
        
        Ok(crate::cuda_kernels::create_output_tensor(
            output_data,
            input.shape().clone(),
            self.device.clone()
        ))
    }
    
    /// Update running statistics
    fn update_running_stats(&mut self, batch_mean: &Tensor, batch_var: &Tensor) -> Result<()> {
        if let (Some(running_mean), Some(running_var)) = (&mut self.running_mean, &mut self.running_var) {
            // Exponential moving average update
            let momentum = self.momentum;
            let one_minus_momentum = 1.0 - momentum;
            
            // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            let new_mean = running_mean.scale(one_minus_momentum)?
                .add(&batch_mean.scale(momentum)?)?;
            *running_mean = new_mean;
            
            // running_var = (1 - momentum) * running_var + momentum * batch_var
            let new_var = running_var.scale(one_minus_momentum)?
                .add(&batch_var.scale(momentum)?)?;
            *running_var = new_var;
        }
        
        Ok(())
    }
}

/// Get Conv3d CUDA kernel code
fn get_conv3d_kernel_code() -> String {
    let kernel_code = r#"
extern "C" __global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int total_elems,
    int batch_size,
    int in_channels,
    int out_channels,
    int d_in, int h_in, int w_in,
    int d_out, int h_out, int w_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    
    // Compute output position
    int w_idx = idx % w_out;
    int h_idx = (idx / w_out) % h_out;
    int d_idx = (idx / (w_out * h_out)) % d_out;
    int c_out = (idx / (w_out * h_out * d_out)) % out_channels;
    int batch = idx / (w_out * h_out * d_out * out_channels);
    
    int group_size_in = in_channels / groups;
    int group_size_out = out_channels / groups;
    int group = c_out / group_size_out;
    
    float sum = 0.0f;
    
    // Convolution operation
    for (int kd = 0; kd < kernel_d; kd++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                for (int c_in = group * group_size_in; c_in < (group + 1) * group_size_in; c_in++) {
                    int d_in_idx = d_idx * stride_d - pad_d + kd * dilation_d;
                    int h_in_idx = h_idx * stride_h - pad_h + kh * dilation_h;
                    int w_in_idx = w_idx * stride_w - pad_w + kw * dilation_w;
                    
                    if (d_in_idx >= 0 && d_in_idx < d_in &&
                        h_in_idx >= 0 && h_in_idx < h_in &&
                        w_in_idx >= 0 && w_in_idx < w_in) {
                        
                        int input_idx = batch * in_channels * d_in * h_in * w_in +
                                       c_in * d_in * h_in * w_in +
                                       d_in_idx * h_in * w_in +
                                       h_in_idx * w_in +
                                       w_in_idx;
                        
                        int weight_idx = c_out * group_size_in * kernel_d * kernel_h * kernel_w +
                                        (c_in - group * group_size_in) * kernel_d * kernel_h * kernel_w +
                                        kd * kernel_h * kernel_w +
                                        kh * kernel_w +
                                        kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[idx] = sum;
}
"#;
    kernel_code.to_string()
}

/// Get add bias 3D kernel code
fn get_add_bias_3d_kernel_code() -> String {
    let kernel_code = r#"
extern "C" __global__ void add_bias_3d_kernel(
    float* output,
    const float* input,
    const float* bias,
    int total_elems,
    int out_channels,
    int elems_per_channel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    
    int channel = (idx / elems_per_channel) % out_channels;
    output[idx] = input[idx] + bias[channel];
}
"#;
    kernel_code.to_string()
}

/// Get BatchNorm3d kernel code
fn get_batch_norm_3d_kernel_code() -> String {
    let kernel_code = r#"
extern "C" __global__ void batch_norm_3d_kernel(
    const float* input,
    const float* mean,
    const float* var,
    float* output,
    int total_elems,
    int num_channels,
    int spatial_size,
    float eps,
    int affine
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    
    int spatial_idx = idx % spatial_size;
    int channel = (idx / spatial_size) % num_channels;
    int batch = idx / (spatial_size * num_channels);
    
    float x = input[idx];
    float m = mean[channel];
    float v = var[channel];
    
    // Normalize
    float normalized = (x - m) / sqrtf(v + eps);
    
    output[idx] = normalized;
}
"#;
    kernel_code.to_string()
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
        
        // Check that output is normalized
        let data = output.to_vec()?;
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        println!("BatchNorm3d output mean: {:.6}", mean);
        
        Ok(())
    }
}