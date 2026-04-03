use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::{DType, Result, Shape, Tensor};

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
        assert_nhwc_bf16_public("MaxPool2d::forward in", input)?;

        let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        let input_nchw_f32 = input_nchw.to_dtype(DType::F32)?;
        debug_assert_eq!(input_nchw_f32.dtype(), DType::F32);
        debug_assert_eq!(input_nchw_f32.storage_dtype(), DType::F32);
        debug_assert_eq!(input_nchw_f32.storage_dtype(), DType::F32);

        let shape = input_nchw_f32.shape().dims();
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.config.kernel_size;
        let (sh, sw) = self.config.stride.unwrap_or(self.config.kernel_size);
        let (ph, pw) = self.config.padding;
        let (dh, dw) = self.config.dilation;

        // Calculate output dimensions
        let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

        let _output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);

        let output_nchw = crate::cuda_kernels::CudaKernels::maxpool2d_forward(
            &input_nchw_f32,
            self.config.kernel_size,
            self.config.stride.unwrap_or(self.config.kernel_size),
            self.config.padding,
        )?;

        let mut output = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&output_nchw)?;
        if output.dtype() != DType::BF16 {
            output = output.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("MaxPool2d::forward out", &output)?;

        // Return tensor and optional indices (None for now as indices not implemented yet)
        Ok((output, None))
    }

    /// Backward pass for max pooling
    pub fn backward(
        &self,
        grad_output: &Tensor,
        input: &Tensor,
        _indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        assert_nhwc_bf16_public("MaxPool2d::backward grad", grad_output)?;
        assert_nhwc_bf16_public("MaxPool2d::backward in", input)?;

        let grad_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(grad_output)?;
        let grad_nchw_f32 = grad_nchw.to_dtype(DType::F32)?;
        debug_assert_eq!(grad_nchw_f32.dtype(), DType::F32);
        debug_assert_eq!(grad_nchw_f32.storage_dtype(), DType::F32);

        let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        let input_nchw_f32 = input_nchw.to_dtype(DType::F32)?;
        debug_assert_eq!(input_nchw_f32.dtype(), DType::F32);
        debug_assert_eq!(input_nchw_f32.storage_dtype(), DType::F32);

        let grad_input_nchw = crate::cuda_kernels::CudaKernels::maxpool2d_backward(
            &grad_nchw_f32,
            &input_nchw_f32,
            self.config.kernel_size,
            self.config.stride.unwrap_or(self.config.kernel_size),
            self.config.padding,
        )?;

        let mut grad_input = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&grad_input_nchw)?;
        if grad_input.dtype() != DType::BF16 {
            grad_input = grad_input.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("MaxPool2d::backward out", &grad_input)?;
        Ok(grad_input)
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
        assert_nhwc_bf16_public("AvgPool2d::forward in", input)?;

        let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        let input_nchw_f32 = input_nchw.to_dtype(DType::F32)?;
        debug_assert_eq!(input_nchw_f32.dtype(), DType::F32);

        let shape = input_nchw_f32.shape().dims();
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.config.kernel_size;
        let (sh, sw) = self.config.stride.unwrap_or(self.config.kernel_size);
        let (ph, pw) = self.config.padding;

        // Calculate output dimensions
        let h_out = (h_in + 2 * ph - kh) / sh + 1;
        let w_out = (w_in + 2 * pw - kw) / sw + 1;

        let _output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);

        let output_nchw = crate::cuda_kernels::CudaKernels::avgpool2d_forward(
            &input_nchw_f32,
            self.config.kernel_size,
            self.config.stride.unwrap_or(self.config.kernel_size),
            self.config.padding,
            self.config.count_include_pad,
        )?;

        let mut output = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&output_nchw)?;
        if output.dtype() != DType::BF16 {
            output = output.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("AvgPool2d::forward out", &output)?;
        Ok(output)
    }

    /// Backward pass for average pooling
    pub fn backward(&self, grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("AvgPool2d::backward grad", grad_output)?;
        assert_nhwc_bf16_public("AvgPool2d::backward in", input)?;

        let grad_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(grad_output)?;
        let grad_nchw_f32 = grad_nchw.to_dtype(DType::F32)?;
        debug_assert_eq!(grad_nchw_f32.dtype(), DType::F32);
        debug_assert_eq!(grad_nchw_f32.storage_dtype(), DType::F32);

        let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        let input_nchw_f32 = input_nchw.to_dtype(DType::F32)?;
        debug_assert_eq!(input_nchw_f32.dtype(), DType::F32);
        debug_assert_eq!(input_nchw_f32.storage_dtype(), DType::F32);

        let grad_input_nchw = crate::cuda_kernels::CudaKernels::avgpool2d_backward(
            &grad_nchw_f32,
            &input_nchw_f32,
            self.config.kernel_size,
            self.config.stride.unwrap_or(self.config.kernel_size),
            self.config.padding,
            self.config.count_include_pad,
        )?;

        let mut grad_input = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&grad_input_nchw)?;
        if grad_input.dtype() != DType::BF16 {
            grad_input = grad_input.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("AvgPool2d::backward out", &grad_input)?;
        Ok(grad_input)
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
        assert_nhwc_bf16_public("AdaptiveAvgPool2d::forward in", input)?;

        let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        let input_nchw_f32 = input_nchw.to_dtype(DType::F32)?;
        debug_assert_eq!(input_nchw_f32.dtype(), DType::F32);
        debug_assert_eq!(input_nchw_f32.storage_dtype(), DType::F32);

        let shape = input_nchw_f32.shape().dims();
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = self.output_size;

        // Calculate adaptive kernel size and stride
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;
        let _kernel_h = h_in - (h_out - 1) * stride_h;
        let _kernel_w = w_in - (w_out - 1) * stride_w;

        let _output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);

        let output_nchw = crate::cuda_kernels::CudaKernels::adaptive_avgpool2d_forward(
            &input_nchw_f32,
            self.output_size,
        )?;

        let mut output = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&output_nchw)?;
        if output.dtype() != DType::BF16 {
            output = output.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("AdaptiveAvgPool2d::forward out", &output)?;
        Ok(output)
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
        assert_nhwc_bf16_public("GlobalAvgPool2d::forward in", input)?;

        let (batch, _, _, channels) = (
            input.shape().dims()[0],
            input.shape().dims()[1],
            input.shape().dims()[2],
            input.shape().dims()[3],
        );

        // Global pooling is equivalent to adaptive pooling with output size (1, 1)
        let adaptive_pool = AdaptiveAvgPool2d::new((1, 1));
        let pooled = adaptive_pool.forward(input)?;

        let pooled_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(&pooled)?;
        let mut flattened = pooled_nchw.reshape(&[batch, channels])?;
        if flattened.dtype() != DType::BF16 {
            flattened = flattened.to_dtype(DType::BF16)?;
        }
        // 2D op: rank guard skipped; BF16 storage enforced on outputs.
        Ok(flattened)
    }
}

impl Default for GlobalAvgPool2d {
    fn default() -> Self {
        Self::new()
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
        assert_nhwc_bf16_public("AdaptiveMaxPool2d::forward in", input)?;

        let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        let input_nchw_f32 = input_nchw.to_dtype(DType::F32)?;
        debug_assert_eq!(input_nchw_f32.dtype(), DType::F32);

        let shape = input_nchw_f32.shape().dims();
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = self.output_size;

        // Calculate adaptive kernel size and stride
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;
        let _kernel_h = h_in - (h_out - 1) * stride_h;
        let _kernel_w = w_in - (w_out - 1) * stride_w;

        let _output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);

        let output_nchw = crate::cuda_kernels::CudaKernels::adaptive_maxpool2d_forward(
            &input_nchw_f32,
            self.output_size,
        )?;

        let mut output = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&output_nchw)?;
        if output.dtype() != DType::BF16 {
            output = output.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("AdaptiveMaxPool2d::forward out", &output)?;

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
        assert_nhwc_bf16_public("GlobalMaxPool2d::forward in", input)?;

        let (batch, _, _, channels) = (
            input.shape().dims()[0],
            input.shape().dims()[1],
            input.shape().dims()[2],
            input.shape().dims()[3],
        );

        // Global pooling is equivalent to adaptive pooling with output size (1, 1)
        let adaptive_pool = AdaptiveMaxPool2d::new((1, 1));
        let (pooled, indices) = adaptive_pool.forward(input)?;

        let pooled_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(&pooled)?;
        let mut flattened = pooled_nchw.reshape(&[batch, channels])?;
        if flattened.dtype() != DType::BF16 {
            flattened = flattened.to_dtype(DType::BF16)?;
        }
        // 2D op: rank guard skipped; BF16 storage enforced on outputs.
        Ok((flattened, indices))
    }
}

impl Default for GlobalMaxPool2d {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    #[test]
    fn test_maxpool2d_output_shape() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[2, 8, 8, 3]), 0.0, 1.0, device)?
            .to_dtype(DType::BF16)?;

        let config = MaxPool2dConfig::new((2, 2));
        let pool = MaxPool2d::new(config);

        let (output, _) = pool.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 4, 4, 3]);

        Ok(())
    }

    #[test]
    fn test_avgpool2d_output_shape() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[1, 32, 32, 16]), 0.0, 1.0, device)?
            .to_dtype(DType::BF16)?;

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
        let input = Tensor::randn(Shape::from_dims(&[2, 7, 7, 64]), 0.0, 1.0, device)?
            .to_dtype(DType::BF16)?;

        let global_avg = GlobalAvgPool2d::new();
        let output = global_avg.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 64]);

        let global_max = GlobalMaxPool2d::new();
        let (output, _) = global_max.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 64]);

        Ok(())
    }
}
