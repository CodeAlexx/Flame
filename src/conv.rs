use crate::cuda_ops_bf16::{self, ConvActivation};
use crate::{DType, Error, Result, Shape, Tensor};
use std::sync::Arc;

// Cached env flags for hot-path Conv2d::forward — used to be a syscall
// on every call (up to 3 per call).
#[inline]
fn no_cudnn_conv() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_NO_CUDNN_CONV").is_ok())
}

#[inline]
fn force_f32_conv() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FORCE_F32_CONV").is_ok())
}

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
    weight_hwio_bf16: Tensor,
    pub bias: Option<Tensor>,
    bias_bf16: Option<Tensor>,
    pub config: Conv2dConfig,
}

enum ConvInit {
    Random,
    Zeroed,
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

    /// Create a new Conv2d layer with zeroed BF16 parameters (for checkpoint loading).
    pub fn new_zeroed(
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
        Self::from_config_with_init(config, device, false, ConvInit::Zeroed)
    }

    /// Create a new Conv2d layer with bias and zeroed BF16 parameters (for checkpoint loading).
    pub fn new_with_bias_zeroed(
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
        Self::from_config_with_init(config, device, bias, ConvInit::Zeroed)
    }

    /// Create a new Conv2d layer from config
    pub fn from_config(
        config: Conv2dConfig,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        Self::from_config_with_init(config, device, false, ConvInit::Random)
    }

    fn convert_param(reference: &Tensor, source: &Tensor, name: &str) -> Result<Tensor> {
        if reference.shape() != source.shape() {
            return Err(Error::ShapeMismatch {
                expected: reference.shape().clone(),
                got: source.shape().clone(),
            });
        }

        let mut tensor = if source.dtype() != reference.dtype() {
            source.to_dtype(reference.dtype())?
        } else if source.storage_dtype() != reference.storage_dtype() {
            source.to_dtype(reference.storage_dtype())?
        } else {
            source.clone_result()?
        };

        if !Arc::ptr_eq(tensor.device(), reference.device()) {
            return Err(Error::InvalidInput(format!(
                "{name} expects tensor on the same device as the destination"
            )));
        }

        Ok(tensor)
    }

    /// Copy convolution weights from external tensor (shape/dtype checked).
    pub fn copy_weight_from(&mut self, source: &Tensor) -> Result<()> {
        let requires_grad = self.weight.requires_grad();
        let tensor = Self::convert_param(&self.weight, source, "Conv2d::copy_weight_from")?;
        self.weight = tensor.requires_grad_(requires_grad);
        self.refresh_weight_cache()?;
        Ok(())
    }

    /// Copy convolution bias from external tensor (shape/dtype checked).
    pub fn copy_bias_from(&mut self, source: &Tensor) -> Result<()> {
        let bias = self
            .bias
            .as_mut()
            .ok_or_else(|| Error::InvalidOperation("Conv2d has no bias parameter".into()))?;
        let requires_grad = bias.requires_grad();
        let tensor = Self::convert_param(bias, source, "Conv2d::copy_bias_from")?;
        *bias = tensor.requires_grad_(requires_grad);
        self.refresh_bias_cache()?;
        Ok(())
    }

    /// Create a new Conv2d layer from config with bias option
    pub fn from_config_with_bias(
        config: Conv2dConfig,
        device: Arc<cudarc::driver::CudaDevice>,
        bias: bool,
    ) -> Result<Self> {
        Self::from_config_with_init(config, device, bias, ConvInit::Random)
    }

    fn from_config_with_init(
        config: Conv2dConfig,
        device: Arc<cudarc::driver::CudaDevice>,
        bias: bool,
        init: ConvInit,
    ) -> Result<Self> {
        let (kh, kw) = config.kernel_size;
        let weight_shape = Shape::from_dims(&[
            config.out_channels,
            config.in_channels / config.groups,
            kh,
            kw,
        ]);

        let weight = match init {
            ConvInit::Random => {
                let fan_in = (config.in_channels / config.groups) * kh * kw;
                let fan_out = (config.out_channels / config.groups) * kh * kw;
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                let t = Tensor::randn(weight_shape, 0.0, std, device.clone())?;
                #[cfg(feature = "bf16_u16")]
                let t = t.to_dtype(DType::BF16)?;
                t.requires_grad_(true)
            }
            ConvInit::Zeroed => {
                Tensor::zeros_dtype(weight_shape, DType::BF16, device.clone())?.requires_grad_(true)
            }
        };

        let bias_tensor = if bias {
            Some(
                Tensor::zeros_dtype(
                    Shape::from_dims(&[config.out_channels]),
                    DType::BF16,
                    device.clone(),
                )?
                .requires_grad_(true),
            )
        } else {
            None
        };

        let hwio_shape = Shape::from_dims(&[
            kh,
            kw,
            config.in_channels / config.groups,
            config.out_channels,
        ]);
        let weight_hwio_bf16 = Tensor::zeros_dtype(hwio_shape, DType::BF16, device.clone())?;

        let mut conv = Self {
            weight,
            weight_hwio_bf16,
            bias: bias_tensor,
            bias_bf16: None,
            config,
        };
        conv.refresh_weight_cache()?;
        conv.refresh_bias_cache()?;
        Ok(conv)
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.shape().dims() != self.weight.shape().dims() {
            return Err(Error::InvalidShape(format!(
                "Conv2d::set_weight: expected {:?}, got {:?}",
                self.weight.shape().dims(),
                weight.shape().dims()
            )));
        }
        self.weight = weight;
        self.refresh_weight_cache()
    }

    pub fn set_bias(&mut self, bias: Option<Tensor>) -> Result<()> {
        self.bias = bias;
        self.refresh_bias_cache()
    }

    fn refresh_weight_cache(&mut self) -> Result<()> {
        let mut perm = self.weight.permute(&[2, 3, 1, 0])?;
        if perm.dtype() != DType::BF16 || perm.storage_dtype() != DType::BF16 {
            perm = perm.to_dtype(DType::BF16)?;
        } else {
            // Ensure contiguous memory layout for the kernel
            perm = perm.clone_result()?;
        }
        self.weight_hwio_bf16 = perm.requires_grad_(false);
        Ok(())
    }

    fn refresh_bias_cache(&mut self) -> Result<()> {
        self.bias_bf16 = if let Some(bias) = &self.bias {
            let mut cached = bias.clone_result()?;
            if cached.dtype() != DType::BF16 || cached.storage_dtype() != DType::BF16 {
                cached = cached.to_dtype(DType::BF16)?;
            }
            Some(cached.requires_grad_(false))
        } else {
            None
        };
        Ok(())
    }

    /// Add bias to the convolution layer
    pub fn with_bias(mut self, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let bias_shape = Shape::from_dims(&[self.config.out_channels]);
        self.bias = Some(Tensor::zeros(bias_shape, device)?);
        self.refresh_bias_cache()?;
        Ok(self)
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias_bf16.as_ref()
    }

    /// Forward pass of Conv2d
    ///
    /// Input: NCHW BF16. Output: NCHW BF16.
    /// Tries cuDNN first (if feature enabled), falls back to custom CUDA kernel.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(
                "Conv2d::forward expects 4D input tensor".into(),
            ));
        }
        if input.dtype() != DType::BF16 || input.storage_dtype() != DType::BF16 {
            return Err(Error::InvalidInput(
                "Conv2d::forward expects BF16 inputs".into(),
            ));
        }

        // Try cuDNN first — native BF16, no format conversion needed
        #[cfg(feature = "cudnn")]
        {
            if !no_cudnn_conv() {
                match crate::cudnn::cudnn_conv2d_bf16(
                    input,
                    &self.weight, // OIHW format, cuDNN native
                    self.bias_bf16.as_ref(),
                    (self.config.stride.0, self.config.stride.1),
                    (self.config.padding.0, self.config.padding.1),
                    (1, 1), // Conv2dConfig has no dilation field — always 1
                    self.config.groups,
                ) {
                    Ok(output) => return Ok(output),
                    Err(e) => {
                        log::warn!("cuDNN conv2d failed, falling back to custom kernel: {:?}", e);
                    }
                }
            }
        }

        // Fallback: custom NHWC BF16 kernel (requires format permutes)
        let x_nhwc = input.permute(&[0, 2, 3, 1])?;

        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            if x_nhwc.dtype() == DType::BF16
                && self.weight.dtype() == DType::BF16
                && !force_f32_conv()
            {
                match crate::cuda_ops_bf16::conv2d_bf16(
                    &x_nhwc,
                    &self.weight_hwio_bf16,
                    self.bias_bf16.as_ref(),
                    (self.config.stride.0 as i32, self.config.stride.1 as i32),
                    (self.config.padding.0 as i32, self.config.padding.1 as i32),
                    (1, 1),
                    self.config.groups as i32,
                    crate::cuda_ops_bf16::ConvActivation::None,
                ) {
                    Ok(y_nhwc) => {
                        return Ok(y_nhwc.permute(&[0, 3, 1, 2])?);
                    }
                    Err(err) => {
                        return Err(Error::Cuda(format!(
                            "conv2d_bf16 failed: {err:?} shape={:?} weight={:?}",
                            x_nhwc.shape().dims(),
                            self.weight.shape().dims(),
                        )));
                    }
                }
            }
            // Fallback to F32
            self.forward_fallback(&x_nhwc)
        }
    }

    /// Forward pass for NHWC input, returning NHWC output (no format permutes).
    /// Use this when the caller already has NHWC data to avoid 2 permute ops per conv.
    pub fn forward_nhwc(&self, input: &Tensor) -> Result<Tensor> {
        if input.dtype() != DType::BF16 || input.storage_dtype() != DType::BF16 {
            return Err(Error::InvalidInput(
                "Conv2d::forward_nhwc expects BF16 inputs".into(),
            ));
        }

        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            if !force_f32_conv() {
                return crate::cuda_ops_bf16::conv2d_bf16(
                    input,
                    &self.weight_hwio_bf16,
                    self.bias_bf16.as_ref(),
                    (self.config.stride.0 as i32, self.config.stride.1 as i32),
                    (self.config.padding.0 as i32, self.config.padding.1 as i32),
                    (1, 1),
                    self.config.groups as i32,
                    crate::cuda_ops_bf16::ConvActivation::None,
                );
            }
        }
        // Fallback: use the NCHW path
        let nchw = input.permute(&[0, 3, 1, 2])?;
        self.forward(&nchw)?.permute(&[0, 2, 3, 1])
    }

    fn forward_fallback(&self, x_nhwc: &Tensor) -> Result<Tensor> {
        let mut out = crate::cuda_conv2d::CudaConv2d::conv2d_forward_nhwc(
            x_nhwc,
            &self.weight_hwio_bf16,
            self.bias_bf16.as_ref(),
            self.config.stride,
            self.config.padding,
        )?;
        if out.dtype() != DType::BF16 {
            out = out.to_dtype(DType::BF16)?;
        }
        out.permute(&[0, 3, 1, 2])
    }

    /// Convolution using im2col algorithm
    #[allow(dead_code)]
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

        let col = self.im2col(
            input, &col_shape, in_height, in_width, out_height, out_width,
        )?;

        // Reshape weight for matrix multiplication
        let weight_reshape =
            Shape::from_dims(&[self.config.out_channels, self.config.in_channels * kh * kw]);

        // Get weight data and reshape
        let weight_data = self.weight.to_vec()?;
        let weight_2d =
            Tensor::from_vec(weight_data, weight_reshape, self.weight.device().clone())?;

        // Perform batched matrix multiplication
        let mut output_data =
            Vec::with_capacity(batch_size * self.config.out_channels * out_height * out_width);

        for b in 0..batch_size {
            // Extract batch from col
            let batch_start = b * col_shape.dims()[1] * col_shape.dims()[2];
            let batch_end = (b + 1) * col_shape.dims()[1] * col_shape.dims()[2];
            let col_data = col.to_vec()?;
            let batch_col_data = col_data[batch_start..batch_end].to_vec();

            let batch_col = Tensor::from_vec(
                batch_col_data,
                Shape::from_dims(&[col_shape.dims()[1], col_shape.dims()[2]]),
                col.device.clone(),
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
    #[allow(dead_code)]
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

                                if in_h >= ph
                                    && in_h < in_height + ph
                                    && in_w >= pw
                                    && in_w < in_width + pw
                                {
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
    #[allow(dead_code)]
    fn add_bias(&self, output: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let output_dims = output.shape().dims();
        let batch_size = output_dims[0];
        let out_channels = output_dims[1];
        let out_height = output_dims[2];
        let out_width = output_dims[3];

        let mut output_data = output.to_vec()?;
        let bias_data = bias.to_vec()?;

        for b in 0..batch_size {
            for (c, &bias_val) in bias_data.iter().enumerate().take(out_channels) {
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
