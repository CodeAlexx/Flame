use crate::{Result, FlameError, Tensor, Shape};

/// Configuration for 2D max pooling
#[derive(Debug, Clone)]
pub struct MaxPool2dConfig {
    pub kernel_size: (usize, usize),
    pub stride: Option<(usize, usize)>,
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub return_indices: bool,
}

impl MaxPool2dConfig {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: None, // Default to kernel_size
            padding: (0, 0),
            dilation: (1, 1),
            return_indices: false,
        }
    }
}

/// 2D Max Pooling layer
pub struct MaxPool2d {
    pub config: MaxPool2dConfig,
}

impl MaxPool2d {
    pub fn new(config: MaxPool2dConfig) -> Self {
        Self { config }
    }
    
    /// Forward pass for max pooling
    /// Input shape: [batch, channels, height, width]
    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("MaxPool2d expects 4D input, got {:?}", shape)
            ));
        }
        
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.config.kernel_size;
        let (sh, sw) = self.config.stride.unwrap_or(self.config.kernel_size);
        let (ph, pw) = self.config.padding;
        let (dh, dw) = self.config.dilation;
        
        // Calculate output dimensions
        let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        
        let output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);
        
        // Use CUDA kernel for the actual computation
        let output = crate::cuda_kernels::CudaKernels::maxpool2d_forward(
            input,
            self.config.kernel_size,
            self.config.stride.unwrap_or(self.config.kernel_size),
            self.config.padding,
        )?;
        
        // Return tensor and optional indices (None for now as indices not implemented yet)
        Ok((output, None))
    }
    
    /// Backward pass for max pooling
    pub fn backward(
        &self,
        grad_output: &Tensor,
        input: &Tensor,
        indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        crate::cuda_kernels::CudaKernels::maxpool2d_backward(
            grad_output,
            input,
            self.config.kernel_size,
            self.config.stride.unwrap_or(self.config.kernel_size),
            self.config.padding,
        )
    }
}

/// Configuration for 2D average pooling
#[derive(Debug, Clone)]
pub struct AvgPool2dConfig {
    pub kernel_size: (usize, usize),
    pub stride: Option<(usize, usize)>,
    pub padding: (usize, usize),
    pub count_include_pad: bool,
    pub divisor_override: Option<usize>,
}

impl AvgPool2dConfig {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: None,
            padding: (0, 0),
            count_include_pad: true,
            divisor_override: None,
        }
    }
}

/// 2D Average Pooling layer
pub struct AvgPool2d {
    pub config: AvgPool2dConfig,
}

impl AvgPool2d {
    pub fn new(config: AvgPool2dConfig) -> Self {
        Self { config }
    }
    
    /// Forward pass for average pooling
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("AvgPool2d expects 4D input, got {:?}", shape)
            ));
        }
        
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.config.kernel_size;
        let (sh, sw) = self.config.stride.unwrap_or(self.config.kernel_size);
        let (ph, pw) = self.config.padding;
        
        // Calculate output dimensions
        let h_out = (h_in + 2 * ph - kh) / sh + 1;
        let w_out = (w_in + 2 * pw - kw) / sw + 1;
        
        let output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);
        
        crate::cuda_kernels::CudaKernels::avgpool2d_forward(
            input,
            self.config.kernel_size,
            self.config.stride.unwrap_or(self.config.kernel_size),
            self.config.padding,
            self.config.count_include_pad,
        )
    }
    
    /// Backward pass for average pooling
    pub fn backward(&self, grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        crate::cuda_kernels::CudaKernels::avgpool2d_backward(
            grad_output,
            input,
            self.config.kernel_size,
            self.config.stride.unwrap_or(self.config.kernel_size),
            self.config.padding,
            self.config.count_include_pad,
        )
    }
}

/// Adaptive Average Pooling 2D
pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
    
    /// Forward pass - adapts kernel size to achieve desired output
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("AdaptiveAvgPool2d expects 4D input, got {:?}", shape)
            ));
        }
        
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = self.output_size;
        
        // Calculate adaptive kernel size and stride
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;
        let kernel_h = h_in - (h_out - 1) * stride_h;
        let kernel_w = w_in - (w_out - 1) * stride_w;
        
        let output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);
        
        crate::cuda_kernels::CudaKernels::adaptive_avgpool2d_forward(
            input,
            self.output_size,
        )
    }
}

/// Global Average Pooling 2D - pools each channel to a single value
pub struct GlobalAvgPool2d;

impl GlobalAvgPool2d {
    pub fn new() -> Self {
        Self
    }
    
    /// Forward pass - averages across entire spatial dimensions
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("GlobalAvgPool2d expects 4D input, got {:?}", shape)
            ));
        }
        
        let (batch, channels, _, _) = (shape[0], shape[1], shape[2], shape[3]);
        
        // Global pooling is equivalent to adaptive pooling with output size (1, 1)
        let adaptive_pool = AdaptiveAvgPool2d::new((1, 1));
        let pooled = adaptive_pool.forward(input)?;
        
        // Reshape to remove spatial dimensions
        pooled.reshape(&[batch, channels])
    }
}

/// Adaptive Max Pooling 2D
pub struct AdaptiveMaxPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveMaxPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
    
    /// Forward pass - adapts kernel size to achieve desired output
    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("AdaptiveMaxPool2d expects 4D input, got {:?}", shape)
            ));
        }
        
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = self.output_size;
        
        // Calculate adaptive kernel size and stride
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;
        let kernel_h = h_in - (h_out - 1) * stride_h;
        let kernel_w = w_in - (w_out - 1) * stride_w;
        
        let output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);
        
        let output = crate::cuda_kernels::CudaKernels::adaptive_maxpool2d_forward(
            input,
            self.output_size,
        )?;
        
        // Return tensor and optional indices (None for now as indices not implemented yet)
        Ok((output, None))
    }
}

/// Global Max Pooling 2D - pools each channel to a single value
pub struct GlobalMaxPool2d;

impl GlobalMaxPool2d {
    pub fn new() -> Self {
        Self
    }
    
    /// Forward pass - takes maximum across entire spatial dimensions
    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("GlobalMaxPool2d expects 4D input, got {:?}", shape)
            ));
        }
        
        let (batch, channels, _, _) = (shape[0], shape[1], shape[2], shape[3]);
        
        // Global pooling is equivalent to adaptive pooling with output size (1, 1)
        let adaptive_pool = AdaptiveMaxPool2d::new((1, 1));
        let (pooled, indices) = adaptive_pool.forward(input)?;
        
        // Reshape to remove spatial dimensions
        Ok((pooled.reshape(&[batch, channels])?, indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use cudarc::driver::CudaDevice;
    
    #[test]
    fn test_maxpool2d_output_shape() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[2, 3, 8, 8]), 0.0, 1.0, device)?;
        
        let config = MaxPool2dConfig::new((2, 2));
        let pool = MaxPool2d::new(config);
        
        let (output, _) = pool.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 3, 4, 4]);
        
        Ok(())
    }
    
    #[test]
    fn test_avgpool2d_output_shape() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[1, 16, 32, 32]), 0.0, 1.0, device)?;
        
        let mut config = AvgPool2dConfig::new((3, 3));
        config.stride = Some((2, 2));
        config.padding = (1, 1);
        
        let pool = AvgPool2d::new(config);
        let output = pool.forward(&input)?;
        
        // Output: (32 + 2*1 - 3) / 2 + 1 = 16
        assert_eq!(output.shape().dims(), &[1, 16, 16, 16]);
        
        Ok(())
    }
    
    #[test]
    fn test_global_pooling() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[2, 64, 7, 7]), 0.0, 1.0, device)?;
        
        let global_avg = GlobalAvgPool2d::new();
        let output = global_avg.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 64]);
        
        let global_max = GlobalMaxPool2d::new();
        let (output, _) = global_max.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 64]);
        
        Ok(())
    }
}