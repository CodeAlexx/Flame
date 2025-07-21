use crate::{Result, FlameError, Tensor, Shape};

/// Upsampling modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpsampleMode {
    Nearest,
    Linear,
    Bilinear,
    Bicubic,
    Trilinear,
}

/// Configuration for 2D upsampling
#[derive(Debug, Clone)]
pub struct Upsample2dConfig {
    pub size: Option<(usize, usize)>,
    pub scale_factor: Option<(f32, f32)>,
    pub mode: UpsampleMode,
    pub align_corners: Option<bool>,
    pub recompute_scale_factor: Option<bool>,
}

impl Upsample2dConfig {
    pub fn new(mode: UpsampleMode) -> Self {
        Self {
            size: None,
            scale_factor: None,
            mode,
            align_corners: None,
            recompute_scale_factor: None,
        }
    }
    
    pub fn with_size(mut self, size: (usize, usize)) -> Self {
        self.size = Some(size);
        self.scale_factor = None;
        self
    }
    
    pub fn with_scale_factor(mut self, scale_factor: (f32, f32)) -> Self {
        self.scale_factor = Some(scale_factor);
        self.size = None;
        self
    }
}

/// 2D Upsampling layer
pub struct Upsample2d {
    pub config: Upsample2dConfig,
}

impl Upsample2d {
    pub fn new(config: Upsample2dConfig) -> Self {
        Self { config }
    }
    
    /// Forward pass for upsampling
    /// Input shape: [batch, channels, height, width]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("Upsample2d expects 4D input, got {:?}", shape)
            ));
        }
        
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        
        // Determine output size
        let (h_out, w_out) = if let Some(size) = self.config.size {
            size
        } else if let Some(scale) = self.config.scale_factor {
            (
                (h_in as f32 * scale.0) as usize,
                (w_in as f32 * scale.1) as usize,
            )
        } else {
            return Err(FlameError::InvalidOperation(
                "Upsample2d requires either size or scale_factor".into()
            ));
        };
        
        let output_shape = Shape::from_dims(&[batch, channels, h_out, w_out]);
        
        // Use CUDA kernel for the actual computation
        match self.config.mode {
            UpsampleMode::Nearest => {
                crate::cuda_ops::GpuOps::upsample2d_nearest(
                    input,
                    (h_out, w_out),
                )
            }
            UpsampleMode::Linear | UpsampleMode::Bilinear => {
                crate::cuda_ops::GpuOps::upsample2d_bilinear(
                    input,
                    (h_out, w_out),
                    self.config.align_corners.unwrap_or(false),
                )
            }
            _ => {
                Err(FlameError::InvalidOperation(
                    format!("Upsampling mode {:?} not yet implemented", self.config.mode)
                ))
            }
        }
    }
    
    /// Backward pass for upsampling
    pub fn backward(&self, grad_output: &Tensor, input_shape: &[usize]) -> Result<Tensor> {
        match self.config.mode {
            UpsampleMode::Nearest => {
                let (h_in, w_in) = (input_shape[2], input_shape[3]);
                let (h_out, w_out) = (grad_output.shape().dims()[2], grad_output.shape().dims()[3]);
                crate::cuda_ops::GpuOps::upsample2d_nearest_backward(
                    grad_output,
                    (h_in, w_in),
                    (h_out, w_out),
                )
            }
            UpsampleMode::Linear | UpsampleMode::Bilinear => {
                let (h_in, w_in) = (input_shape[2], input_shape[3]);
                let (h_out, w_out) = (grad_output.shape().dims()[2], grad_output.shape().dims()[3]);
                crate::cuda_ops::GpuOps::upsample2d_bilinear_backward(
                    grad_output,
                    (h_in, w_in),
                    (h_out, w_out),
                    self.config.align_corners.unwrap_or(false),
                )
            }
            _ => {
                Err(FlameError::InvalidOperation(
                    format!("Upsampling mode {:?} backward not yet implemented", self.config.mode)
                ))
            }
        }
    }
}

/// Configuration for 2D transposed convolution (deconvolution)
#[derive(Debug, Clone)]
pub struct ConvTranspose2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub output_padding: (usize, usize),
    pub groups: usize,
    pub bias: bool,
    pub dilation: (usize, usize),
}

impl ConvTranspose2dConfig {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            groups: 1,
            bias: true,
            dilation: (1, 1),
        }
    }
}

/// 2D Transposed Convolution layer (Deconvolution)
pub struct ConvTranspose2d {
    pub config: ConvTranspose2dConfig,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl ConvTranspose2d {
    pub fn new(config: ConvTranspose2dConfig, device: std::sync::Arc<crate::CudaDevice>) -> Result<Self> {
        // Initialize weights
        // Weight shape for ConvTranspose2d: [in_channels, out_channels/groups, kh, kw]
        let weight_shape = Shape::from_dims(&[
            config.in_channels,
            config.out_channels / config.groups,
            config.kernel_size.0,
            config.kernel_size.1,
        ]);
        
        // Xavier initialization
        let fan_in = config.in_channels * config.kernel_size.0 * config.kernel_size.1;
        let fan_out = config.out_channels * config.kernel_size.0 * config.kernel_size.1;
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        
        let weight = Tensor::randn(weight_shape, 0.0, std, device.clone())?;
        
        let bias = if config.bias {
            Some(Tensor::zeros(
                Shape::from_dims(&[config.out_channels]),
                device,
            )?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            weight,
            bias,
        })
    }
    
    /// Forward pass for transposed convolution
    /// Input shape: [batch, in_channels, height, width]
    /// Output shape: [batch, out_channels, height_out, width_out]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("ConvTranspose2d expects 4D input, got {:?}", shape)
            ));
        }
        
        // Use CUDA kernel for the actual computation
        crate::cuda_ops::GpuOps::conv_transpose2d_forward(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.config.stride,
            self.config.padding,
            self.config.output_padding,
            self.config.groups,
            self.config.dilation,
        )
    }
    
    /// Backward pass for transposed convolution
    pub fn backward(
        &self,
        grad_output: &Tensor,
        input: &Tensor,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        crate::cuda_ops::GpuOps::conv_transpose2d_backward(
            grad_output,
            input,
            &self.weight,
            self.config.stride,
            self.config.padding,
            self.config.output_padding,
            self.config.groups,
            self.config.dilation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use cudarc::driver::CudaDevice;
    
    #[test]
    fn test_upsample2d_output_shape() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[1, 3, 4, 4]), 0.0, 1.0, device)?;
        
        // Test with size
        let config = Upsample2dConfig::new(UpsampleMode::Nearest)
            .with_size((8, 8));
        let upsample = Upsample2d::new(config);
        let output = upsample.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 3, 8, 8]);
        
        // Test with scale factor
        let config2 = Upsample2dConfig::new(UpsampleMode::Bilinear)
            .with_scale_factor((2.0, 2.0));
        let upsample2 = Upsample2d::new(config2);
        let output2 = upsample2.forward(&input)?;
        assert_eq!(output2.shape().dims(), &[1, 3, 8, 8]);
        
        Ok(())
    }
    
    #[test]
    fn test_conv_transpose2d_output_shape() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[1, 16, 4, 4]), 0.0, 1.0, device.clone())?;
        
        // Test basic transposed convolution
        let config = ConvTranspose2dConfig::new(16, 32, (3, 3));
        let conv_transpose = ConvTranspose2d::new(config, device)?;
        let output = conv_transpose.forward(&input)?;
        
        // Output size: (input - 1) * stride - 2 * padding + kernel_size + output_padding
        // (4 - 1) * 1 - 2 * 0 + 3 + 0 = 6
        assert_eq!(output.shape().dims(), &[1, 32, 6, 6]);
        
        Ok(())
    }
}