//! Simple 3D Convolution and normalization layers for video processing
//! 
//! This module provides 3D convolution operations and batch normalization
//! for processing video data with temporal dimensions.
//! Currently implements CPU versions with TODO for CUDA kernels.

use crate::{Tensor, Shape, Result, FlameError, CudaDevice};
use std::sync::Arc;

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
        // Simple CPU implementation
        // Now actually tries to perform 3D convolution
        let kernel_code = r#"
extern "C" __global__ void conv3d_forward(
    float *output,
    const float *input,
    const float *weight,
    int batch, int in_channels, int out_channels,
    int d_in, int h_in, int w_in,
    int d_out, int h_out, int w_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
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
        crate::cuda_kernels::CudaKernels::ensure_kernel(&self.device, "conv3d_forward", kernel_code)?;
        
        let f = self.device.get_func("conv3d_forward", "conv3d_forward")
            .ok_or_else(|| crate::FlameError::Cuda("Failed to get conv3d kernel".into()))?;
        
        let output_numel = output_shape.elem_count();
        let mut output_data = unsafe { self.device.alloc::<f32>(output_numel) }
            .map_err(|_| crate::FlameError::CudaDriver)?;
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(output_numel as u32);
        unsafe {
            f.launch(cfg, (
                &mut output_data,
                &*input.data(),
                &*self.weight.data(),
                batch as i32, self.in_channels as i32, self.out_channels as i32,
                d_in as i32, h_in as i32, w_in as i32,
                d_out as i32, h_out as i32, w_out as i32,
                kd as i32, kh as i32, kw as i32,
                sd as i32, sh as i32, sw as i32,
                pd as i32, ph as i32, pw as i32,
            )).map_err(|_| crate::FlameError::Cuda("Failed to launch conv3d kernel".into()))?;
        }
        
        crate::cuda_kernels::create_output_tensor(output_data, output_shape, self.device.clone())
    }
    
    /// Add bias to output tensor
    fn add_bias_3d(&self, output: &mut Tensor, bias: &Tensor) -> Result<()> {
        // Simple CPU implementation
        // TODO: Replace with CUDA kernel when available
        
        let output_data = output.to_vec()?;
        let bias_data = bias.to_vec()?;
        let mut result = output_data;
        
        let dims = output.shape().dims();
        let spatial_size = dims[2] * dims[3] * dims[4];
        
        for idx in 0..result.len() {
            let channel = (idx / spatial_size) % self.out_channels;
            result[idx] += bias_data[channel];
        }
        
        *output = Tensor::from_vec(result, output.shape().clone(), output.device.clone())?;
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
        
        crate::cuda_kernels::CudaKernels::ensure_kernel(&self.device, "batchnorm3d_forward", kernel_code)?;
        
        let f = self.device.get_func("batchnorm3d_forward", "batchnorm3d_forward")
            .ok_or_else(|| FlameError::Cuda("Failed to get batchnorm3d kernel".into()))?;
        
        let spatial_size = dims[2] * dims[3] * dims[4];
        let numel = input.shape().elem_count();
        let mut output_data = unsafe { self.device.alloc::<f32>(numel) }
            .map_err(|_| FlameError::CudaDriver)?;
        
        // Use running stats if available, otherwise compute batch stats
        let (mean, var) = if let (Some(rm), Some(rv)) = (&self.running_mean, &self.running_var) {
            (rm.data(), rv.data())
        } else {
            // Compute batch statistics
            let batch_mean = self.compute_batch_mean(input)?;
            let batch_var = self.compute_batch_var(input, &batch_mean)?;
            (batch_mean.data(), batch_var.data())
        };
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(numel as u32);
        unsafe {
            f.launch(cfg, (
                &mut output_data,
                &*input.data(),
                &*mean,
                &*var,
                self.weight.as_ref().map(|w| &**w.data()).unwrap_or(&cudarc::driver::CudaSlice::from_raw_parts(std::ptr::null_mut(), 0)),
                self.bias.as_ref().map(|b| &**b.data()).unwrap_or(&cudarc::driver::CudaSlice::from_raw_parts(std::ptr::null_mut(), 0)),
                self.eps,
                dims[0] as i32,
                self.num_features as i32,
                spatial_size as i32,
            )).map_err(|_| FlameError::Cuda("Failed to launch batchnorm3d kernel".into()))?;
        }
        
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