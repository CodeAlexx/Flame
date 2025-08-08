use crate::{Tensor, Shape, Result, FlameError};
use std::sync::Arc;

/// 2D Convolution parameters
pub struct Conv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            groups: 1,
        }
    }
}

/// 2D Convolution layer
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub config: Conv2dConfig,
}

impl Conv2d {
    /// Create a new Conv2d layer with simple parameters
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let config = Conv2dConfig {
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (padding, padding),
            groups: 1,
        };
        Self::from_config(config, device)
    }
    
    /// Create a new Conv2d layer with bias option
    pub fn new_with_bias(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        device: Arc<cudarc::driver::CudaDevice>,
        bias: bool,
    ) -> Result<Self> {
        let config = Conv2dConfig {
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (padding, padding),
            groups: 1,
        };
        Self::from_config_with_bias(config, device, bias)
    }
    
    /// Create a new Conv2d layer from config
    pub fn from_config(config: Conv2dConfig, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let (kh, kw) = config.kernel_size;
        let weight_shape = Shape::from_dims(&[
            config.out_channels,
            config.in_channels / config.groups,
            kh,
            kw,
        ]);
        
        // Initialize weights with Xavier/He initialization
        let fan_in = (config.in_channels / config.groups) * kh * kw;
        let fan_out = (config.out_channels / config.groups) * kh * kw;
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        
        let weight = Tensor::randn(weight_shape, 0.0, std, device.clone())?.requires_grad_(true);
        
        Ok(Self {
            weight,
            bias: None,
            config,
        })
    }
    
    /// Create a new Conv2d layer from config with bias option
    pub fn from_config_with_bias(config: Conv2dConfig, device: Arc<cudarc::driver::CudaDevice>, bias: bool) -> Result<Self> {
        let (kh, kw) = config.kernel_size;
        let weight_shape = Shape::from_dims(&[
            config.out_channels,
            config.in_channels / config.groups,
            kh,
            kw,
        ]);
        
        // Initialize weights with Xavier/He initialization
        let fan_in = (config.in_channels / config.groups) * kh * kw;
        let fan_out = (config.out_channels / config.groups) * kh * kw;
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        
        let weight = Tensor::randn(weight_shape, 0.0, std, device.clone())?.requires_grad_(true);
        
        let bias = if bias {
            Some(Tensor::zeros(Shape::from_dims(&[config.out_channels]), device)?.requires_grad_(true))
        } else {
            None
        };
        
        Ok(Self {
            weight,
            bias,
            config,
        })
    }
    
    /// Add bias to the convolution layer
    pub fn with_bias(mut self, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let bias_shape = Shape::from_dims(&[self.config.out_channels]);
        self.bias = Some(Tensor::zeros(bias_shape, device)?);
        Ok(self)
    }
    
    /// Forward pass of Conv2d
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Validate input shape: [batch, in_channels, height, width]
        let input_dims = input.shape().dims();
        if input_dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("Conv2d expects 4D input [N,C,H,W], got {:?}", input_dims)
            ));
        }
        
        let batch_size = input_dims[0];
        let in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];
        
        if in_channels != self.config.in_channels {
            return Err(FlameError::InvalidOperation(
                format!("Expected {} input channels, got {}", self.config.in_channels, in_channels)
            ));
        }
        
        // Calculate output dimensions
        let (kh, kw) = self.config.kernel_size;
        let (sh, sw) = self.config.stride;
        let (ph, pw) = self.config.padding;
        
        let out_height = (in_height + 2 * ph - kh) / sh + 1;
        let out_width = (in_width + 2 * pw - kw) / sw + 1;
        
        let _output_shape = Shape::from_dims(&[
            batch_size,
            self.config.out_channels,
            out_height,
            out_width,
        ]);
        
        // Always use GPU convolution since FLAME only supports CUDA
        let output = input.conv2d(
            &self.weight,
            self.bias.as_ref(),
            self.config.stride.0,
            self.config.padding.0,
        )?;
        
        // Add bias if present
        let mut output = if let Some(bias) = &self.bias {
            self.add_bias(&output, bias)?
        } else {
            output
        };
        
        // Record operation for autograd if needed
        if input.requires_grad() || self.weight.requires_grad() {
            use crate::autograd::{AutogradContext, Op};
            
            output.requires_grad = true;
            
            let mut saved = vec![
                (input.id(), input.clone()?),
                (self.weight.id(), self.weight.clone()?),
            ];
            
            // Save bias if it exists and requires grad
            let _bias_id = if let Some(bias) = &self.bias {
                if bias.requires_grad() {
                    saved.push((bias.id(), bias.clone()?));
                }
                Some(bias.id())
            } else {
                None
            };
            
            AutogradContext::record_op(
                output.id(),
                Op::Conv2d {
                    input: input.id(),
                    weight: self.weight.id(),
                    stride: self.config.stride.0,
                    padding: self.config.padding.0,
                },
                saved,
            );
        }
        
        Ok(output)
    }
    
    /// Convolution using im2col algorithm
    fn conv2d_im2col(&self, input: &Tensor, output_shape: Shape) -> Result<Tensor> {
        let input_dims = input.shape().dims();
        let batch_size = input_dims[0];
        let in_height = input_dims[2];
        let in_width = input_dims[3];
        
        let (kh, kw) = self.config.kernel_size;
        let (_sh, _sw) = self.config.stride;
        let (_ph, _pw) = self.config.padding;
        
        let out_height = output_shape.dims()[2];
        let out_width = output_shape.dims()[3];
        
        // Im2col: transform input patches to columns
        let col_shape = Shape::from_dims(&[
            batch_size,
            self.config.in_channels * kh * kw,
            out_height * out_width,
        ]);
        
        let col = self.im2col(input, &col_shape, in_height, in_width, out_height, out_width)?;
        
        // Reshape weight for matrix multiplication
        let weight_reshape = Shape::from_dims(&[
            self.config.out_channels,
            self.config.in_channels * kh * kw,
        ]);
        
        // Get weight data and reshape
        let weight_data = self.weight.to_vec()?;
        let weight_2d = Tensor::from_vec(
            weight_data,
            weight_reshape,
            self.weight.device().clone()
        )?;
        
        // Perform batched matrix multiplication
        let mut output_data = Vec::with_capacity(batch_size * self.config.out_channels * out_height * out_width);
        
        for b in 0..batch_size {
            // Extract batch from col
            let batch_start = b * col_shape.dims()[1] * col_shape.dims()[2];
            let batch_end = (b + 1) * col_shape.dims()[1] * col_shape.dims()[2];
            let col_data = col.to_vec()?;
            let batch_col_data = col_data[batch_start..batch_end].to_vec();
            
            let batch_col = Tensor::from_vec(
                batch_col_data,
                Shape::from_dims(&[col_shape.dims()[1], col_shape.dims()[2]]),
                col.device.clone()
            )?;
            
            // weight_2d: [out_channels, in_channels * kh * kw]
            // batch_col: [in_channels * kh * kw, out_h * out_w]
            // result: [out_channels, out_h * out_w]
            let result = weight_2d.matmul(&batch_col)?;
            output_data.extend(result.to_vec()?);
        }
        
        // Reshape to final output shape
        Tensor::from_vec(output_data, output_shape, input.device().clone())
    }
    
    /// Im2col transformation
    fn im2col(
        &self,
        input: &Tensor,
        col_shape: &Shape,
        in_height: usize,
        in_width: usize,
        out_height: usize,
        out_width: usize,
    ) -> Result<Tensor> {
        let input_data = input.to_vec()?;
        let batch_size = input.shape().dims()[0];
        let in_channels = self.config.in_channels;
        let (kh, kw) = self.config.kernel_size;
        let (sh, sw) = self.config.stride;
        let (ph, pw) = self.config.padding;
        
        let mut col_data = vec![0.0f32; col_shape.elem_count()];
        
        for b in 0..batch_size {
            for c in 0..in_channels {
                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        for out_h in 0..out_height {
                            for out_w in 0..out_width {
                                let in_h = out_h * sh + kh_idx;
                                let in_w = out_w * sw + kw_idx;
                                
                                let col_idx = b * (in_channels * kh * kw * out_height * out_width)
                                    + (c * kh * kw + kh_idx * kw + kw_idx) * out_height * out_width
                                    + out_h * out_width
                                    + out_w;
                                
                                if in_h >= ph && in_h < in_height + ph && in_w >= pw && in_w < in_width + pw {
                                    let input_idx = b * (in_channels * in_height * in_width)
                                        + c * (in_height * in_width)
                                        + (in_h - ph) * in_width
                                        + (in_w - pw);
                                    col_data[col_idx] = input_data[input_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Tensor::from_vec(col_data, col_shape.clone(), input.device().clone())
    }
    
    /// Add bias to output
    fn add_bias(&self, output: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let output_dims = output.shape().dims();
        let batch_size = output_dims[0];
        let out_channels = output_dims[1];
        let out_height = output_dims[2];
        let out_width = output_dims[3];
        
        let mut output_data = output.to_vec()?;
        let bias_data = bias.to_vec()?;
        
        for b in 0..batch_size {
            for c in 0..out_channels {
                let bias_val = bias_data[c];
                for h in 0..out_height {
                    for w in 0..out_width {
                        let idx = b * (out_channels * out_height * out_width)
                            + c * (out_height * out_width)
                            + h * out_width
                            + w;
                        output_data[idx] += bias_val;
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, output.shape().clone(), output.device().clone())
    }
}