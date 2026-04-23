//! Layer normalization for FLAME

use crate::autograd::{AutogradContext, Op};
#[cfg(feature = "bf16_u16")]
use crate::cuda_ops_bf16;
use crate::tensor::contracts::{assert_nhwc_bf16_public, trap_is_bf16};
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// Layer normalization configuration
pub struct LayerNormConfig {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            normalized_shape: vec![],
            eps: 1e-5,
            elementwise_affine: true,
        }
    }
}

/// Layer normalization layer
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    /// Create new layer normalization
    pub fn new(normalized_shape: Vec<usize>, eps: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let param_shape = Shape::from_dims(&normalized_shape);

        // Initialize weight and bias in BF16 so downstream copy preserves dtype.
        let weight = Tensor::ones_dtype(param_shape.clone(), DType::BF16, device.clone())?
            .requires_grad_(true);
        let bias =
            Tensor::zeros_dtype(param_shape, DType::BF16, device.clone())?.requires_grad_(true);

        Ok(Self {
            normalized_shape,
            eps,
            weight: Some(weight),
            bias: Some(bias),
        })
    }

    /// Create layer norm without affine parameters
    pub fn new_no_affine(normalized_shape: Vec<usize>, eps: f32) -> Self {
        Self {
            normalized_shape,
            eps,
            weight: None,
            bias: None,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.rank() == 4 {
            assert_nhwc_bf16_public("LayerNorm::forward in", input)?;
        } else if input.dtype() != DType::BF16 {
            // Non-4D input: rank guard skipped; BF16 storage enforced on outputs.
            return Err(Error::InvalidInput(
                "LayerNorm::forward expects BF16 storage at the public boundary".into(),
            ));
        }

        // Check if we can use cuDNN for this operation
        #[cfg(feature = "cudnn")]
        if crate::cudnn::is_cudnn_norm_compatible(input, "layer") {
            let mut output = crate::cudnn::cudnn_layer_norm(
                input,
                &self.normalized_shape,
                self.weight.as_ref(),
                self.bias.as_ref(),
                self.eps as f64,
            )?;

            if output.dtype() != DType::BF16 {
                output = output.to_dtype(DType::BF16)?;
            }
            if output.rank() == 4 {
                assert_nhwc_bf16_public("LayerNorm::forward out", &output)?;
            }

            return Ok(output);
        }

        // Fallback to regular implementation
        let output = layer_norm(
            input,
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )?;

        if output.rank() == 4 {
            assert_nhwc_bf16_public("LayerNorm::forward out", &output)?;
        }

        Ok(output)
    }

    /// Forward pass into a provided BF16 tensor.
    pub fn forward_into(&self, input: &Tensor, output: &mut Tensor) -> Result<()> {
        if input.rank() == 4 {
            assert_nhwc_bf16_public("LayerNorm::forward_into in", input)?;
        } else if input.dtype() != DType::BF16 {
            return Err(Error::InvalidInput(
                "LayerNorm::forward_into expects BF16 storage at the public boundary".into(),
            ));
        }

        let weight = self.weight.as_ref();
        let bias = self.bias.as_ref();
        layer_norm_into(
            input,
            &self.normalized_shape,
            weight,
            bias,
            self.eps,
            output,
        )
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
            source.clone()
        };

        if !Arc::ptr_eq(tensor.device(), reference.device()) {
            return Err(Error::InvalidInput(format!(
                "{name} expects tensor on the same device as the destination"
            )));
        }

        Ok(tensor)
    }

    /// Copy the affine weight parameter from an external tensor.
    pub fn copy_weight_from(&mut self, source: &Tensor) -> Result<()> {
        let weight = self
            .weight
            .as_mut()
            .ok_or_else(|| Error::InvalidOperation("LayerNorm has no affine weight".into()))?;
        let requires_grad = weight.requires_grad();
        let tensor = Self::convert_param(weight, source, "LayerNorm::copy_weight_from")?;
        *weight = tensor.requires_grad_(requires_grad);
        Ok(())
    }

    /// Copy the affine bias parameter from an external tensor.
    pub fn copy_bias_from(&mut self, source: &Tensor) -> Result<()> {
        let bias = self
            .bias
            .as_mut()
            .ok_or_else(|| Error::InvalidOperation("LayerNorm has no affine bias".into()))?;
        let requires_grad = bias.requires_grad();
        let tensor = Self::convert_param(bias, source, "LayerNorm::copy_bias_from")?;
        *bias = tensor.requires_grad_(requires_grad);
        Ok(())
    }
}

/// CUDA kernel for layer normalization forward pass
pub const LAYER_NORM_FWD_KERNEL: &str = r#"
extern "C" __global__ void layer_norm_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float* mean_out,
    float* rstd_out,
    int batch_size,
    int norm_size,
    float eps
) {
    int idx = blockIdx.x;
    if (idx >= batch_size) return;
    
    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < norm_size; i++) {
        sum += input[idx * norm_size + i];
    }
    float mean = sum / norm_size;
    mean_out[idx] = mean;
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int i = 0; i < norm_size; i++) {
        float diff = input[idx * norm_size + i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / norm_size;
    float rstd = rsqrtf(var + eps);
    rstd_out[idx] = rstd;
    
    // Normalize and apply affine transform
    for (int i = 0; i < norm_size; i++) {
        int global_idx = idx * norm_size + i;
        float normalized = (input[global_idx] - mean) * rstd;
        
        if (weight != nullptr) {
            normalized = normalized * weight[i];
        }
        if (bias != nullptr) {
            normalized = normalized + bias[i];
        }
        
        output[global_idx] = normalized;
    }
}
"#;

/// CUDA kernel for layer normalization backward pass
pub const LAYER_NORM_BWD_KERNEL: &str = r#"
extern "C" __global__ void layer_norm_backward(
    const float* grad_output,
    const float* input,
    const float* mean,
    const float* rstd,
    const float* weight,
    float* grad_input,
    float* grad_weight,
    float* grad_bias,
    int batch_size,
    int norm_size
) {
    int idx = blockIdx.x;
    if (idx >= batch_size) return;
    
    float mean_val = mean[idx];
    float rstd_val = rstd[idx];
    
    // Calculate gradients for this batch element
    float grad_mean = 0.0f;
    float grad_var = 0.0f;
    
    // First pass: calculate grad_mean and grad_var
    for (int i = 0; i < norm_size; i++) {
        int global_idx = idx * norm_size + i;
        float grad_out = grad_output[global_idx];
        
        if (weight != nullptr) {
            grad_out = grad_out * weight[i];
        }
        
        grad_mean += grad_out;
        grad_var += grad_out * (input[global_idx] - mean_val);
    }
    
    grad_mean = grad_mean * (-rstd_val) / norm_size;
    grad_var = grad_var * (-0.5f) * rstd_val * rstd_val * rstd_val / norm_size;
    
    // Second pass: calculate grad_input
    for (int i = 0; i < norm_size; i++) {
        int global_idx = idx * norm_size + i;
        float grad_out = grad_output[global_idx];
        
        if (weight != nullptr) {
            grad_out = grad_out * weight[i];
        }
        
        float x_centered = input[global_idx] - mean_val;
        grad_input[global_idx] = rstd_val * grad_out + grad_mean + 2.0f * grad_var * x_centered / norm_size;
        
        // Accumulate weight and bias gradients
        if (idx == 0 && grad_weight != nullptr) {
            atomicAdd(&grad_weight[i], grad_output[global_idx] * (input[global_idx] - mean_val) * rstd_val);
        }
        if (idx == 0 && grad_bias != nullptr) {
            atomicAdd(&grad_bias[i], grad_output[global_idx]);
        }
    }
}
"#;

/// Functional layer normalization
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    if input.rank() == 4 {
        assert_nhwc_bf16_public("layer_norm::in", input)?;
    } else if input.dtype() != DType::BF16 {
        // Non-4D input: rank guard skipped; BF16 storage enforced on outputs.
        return Err(Error::InvalidInput(
            "layer_norm expects BF16 storage at the public boundary".into(),
        ));
    }

    let device = input.device();

    // Validate input shape
    let input_dims = input.shape().dims();
    if input_dims.len() < normalized_shape.len() {
        return Err(Error::InvalidOperation(
            "Input dimensions less than normalized_shape".into(),
        ));
    }

    // Check that normalized_shape matches the last dimensions
    let offset = input_dims.len() - normalized_shape.len();
    for (i, &dim) in normalized_shape.iter().enumerate() {
        if input_dims[offset + i] != dim {
            return Err(Error::InvalidOperation(format!(
                "Shape mismatch at dim {}: expected {}, got {}",
                offset + i,
                dim,
                input_dims[offset + i]
            )));
        }
    }

    let batch_size: usize = input_dims[..offset].iter().product();
    let norm_size: usize = normalized_shape.iter().product();

    let artifacts = if input.dtype() == DType::BF16 && input.storage.dtype() == DType::BF16 {
        layer_norm_forward_bf16(input, weight, bias, batch_size, norm_size, eps)?
    } else {
        layer_norm_forward_f32(input, weight, bias, batch_size, norm_size, eps)?
    };

    let LayerNormForwardArtifacts {
        mut output,
        mean_data,
        rstd_data,
    } = artifacts;

    if output.dtype() != DType::BF16 {
        output = output.to_dtype(DType::BF16)?;
    }

    let needs_grad = input.requires_grad
        || weight.map(|w| w.requires_grad).unwrap_or(false)
        || bias.map(|b| b.requires_grad).unwrap_or(false);

    if needs_grad {
        output.requires_grad = true;
        if AutogradContext::is_recording() {
            let mut saved_tensors = vec![(input.id, input.clone_result()?)];
            if let Some(w) = weight {
                saved_tensors.push((w.id, w.clone_result()?));
            }
            if let Some(b) = bias {
                saved_tensors.push((b.id, b.clone_result()?));
            }

            let mean_tensor = Tensor {
                storage: TensorStorage::F32 {
                    data: mean_data.into(),
                    numel: batch_size,
                },
                shape: Shape::from_dims(&[batch_size]),
                device: device.clone(),
                id: crate::tensor::TensorId::new(),
                requires_grad: false,
                custom_strides: None,
                view_offset: 0,

            };
            let rstd_tensor = Tensor {
                storage: TensorStorage::F32 {
                    data: rstd_data.into(),
                    numel: batch_size,
                },
                shape: Shape::from_dims(&[batch_size]),
                device: device.clone(),
                id: crate::tensor::TensorId::new(),
                requires_grad: false,
                custom_strides: None,
                view_offset: 0,

            };

            saved_tensors.push((mean_tensor.id, mean_tensor));
            saved_tensors.push((rstd_tensor.id, rstd_tensor));

            AutogradContext::record_op(
                output.id,
                Op::LayerNorm {
                    input: input.id,
                    normalized_shape: normalized_shape.to_vec(),
                },
                saved_tensors,
            );
        } else {
            drop(mean_data);
            drop(rstd_data);
        }
    } else {
        drop(mean_data);
        drop(rstd_data);
    }

    if output.rank() == 4 {
        assert_nhwc_bf16_public("layer_norm::out", &output)?;
    }

    Ok(output)
}

/// Functional layer normalization that writes into a provided tensor.
pub fn layer_norm_into(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
    output: &mut Tensor,
) -> Result<()> {
    if input.rank() == 4 {
        assert_nhwc_bf16_public("layer_norm_into::in", input)?;
    } else {
        trap_is_bf16("layer_norm_into::in", input)?;
    }
    trap_is_bf16("layer_norm_into::out", output)?;

    if !Arc::ptr_eq(input.device(), output.device()) {
        return Err(Error::InvalidInput(
            "layer_norm_into expects input/output tensors on the same device".into(),
        ));
    }
    if input.shape() != output.shape() {
        return Err(Error::ShapeMismatch {
            expected: input.shape().clone(),
            got: output.shape().clone(),
        });
    }

    let input_dims = input.shape().dims();
    if input_dims.len() < normalized_shape.len() {
        return Err(Error::InvalidOperation(
            "layer_norm_into: input dimensions less than normalized_shape".into(),
        ));
    }
    let offset = input_dims.len() - normalized_shape.len();
    for (i, &dim) in normalized_shape.iter().enumerate() {
        if input_dims[offset + i] != dim {
            return Err(Error::InvalidOperation(format!(
                "layer_norm_into: shape mismatch at dim {}: expected {}, got {}",
                offset + i,
                dim,
                input_dims[offset + i]
            )));
        }
    }

    if input.requires_grad()
        || weight.map(|w| w.requires_grad).unwrap_or(false)
        || bias.map(|b| b.requires_grad).unwrap_or(false)
        || output.requires_grad()
    {
        return Err(Error::InvalidOperation(
            "layer_norm_into does not support autograd-enabled tensors".into(),
        ));
    }

    if let Some(w) = weight {
        trap_is_bf16("layer_norm_into::weight", w)?;
    }
    if let Some(b) = bias {
        trap_is_bf16("layer_norm_into::bias", b)?;
    }

    let batch_size: usize = input_dims[..offset].iter().product();
    let norm_size: usize = normalized_shape.iter().product();

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        if input.storage_dtype() != DType::BF16 {
            return Err(Error::InvalidOperation(
                "layer_norm_into expects BF16 storage on input".into(),
            ));
        }
        let device = input.device();
        let mut mean_buf = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;
        let mut rstd_buf = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;
        crate::cuda_ops_bf16::layer_norm_bf16_into_with_stats(
            output,
            input,
            weight,
            bias,
            norm_size,
            eps,
            &mut mean_buf,
            &mut rstd_buf,
        )?;
    }
    #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
    {
        let _ = (input, normalized_shape, weight, bias, eps, output);
        return Err(Error::Unsupported(
            "layer_norm_into requires cuda + bf16_u16 features".into(),
        ));
    }

    if output.rank() == 4 {
        assert_nhwc_bf16_public("layer_norm_into::out", output)?;
    } else {
        trap_is_bf16("layer_norm_into::out", output)?;
    }

    Ok(())
}

/// Add layer norm method to Tensor
impl Tensor {
    /// Apply layer normalization
    pub fn layer_norm(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        layer_norm(self, normalized_shape, weight, bias, eps)
    }

    /// Apply layer normalization into an existing tensor.
    pub fn layer_norm_into(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
        output: &mut Tensor,
    ) -> Result<()> {
        layer_norm_into(self, normalized_shape, weight, bias, eps, output)
    }
}

struct LayerNormForwardArtifacts {
    output: Tensor,
    mean_data: CudaSlice<f32>,
    rstd_data: CudaSlice<f32>,
}

fn layer_norm_forward_f32(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    batch_size: usize,
    norm_size: usize,
    eps: f32,
) -> Result<LayerNormForwardArtifacts> {
    use crate::cuda_kernels::CudaKernels;

    let device = input.device();
    CudaKernels::ensure_kernel(device, "layer_norm_forward", LAYER_NORM_FWD_KERNEL)?;

    let output_data = crate::tensor::alloc_zeros_from_pool(device, input.shape().elem_count())?;
    let mean_data = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;
    let rstd_data = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;

    let f = device
        .get_func("layer_norm_forward", "layer_norm_forward")
        .ok_or_else(|| Error::Cuda("Failed to get layer_norm kernel".into()))?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    match (weight, bias) {
        (Some(w), Some(b)) => {
            launch_kernel!(
                f,
                cfg,
                input.storage.try_as_slice_f32()?,
                w.storage.try_as_slice_f32()?,
                b.storage.try_as_slice_f32()?,
                &output_data,
                &mean_data,
                &rstd_data,
                batch_size as i32,
                norm_size as i32,
                eps
            )?;
        }
        (Some(w), None) => {
            launch_kernel!(
                f,
                cfg,
                input.storage.try_as_slice_f32()?,
                w.storage.try_as_slice_f32()?,
                0usize,
                &output_data,
                &mean_data,
                &rstd_data,
                batch_size as i32,
                norm_size as i32,
                eps
            )?;
        }
        (None, Some(b)) => {
            launch_kernel!(
                f,
                cfg,
                input.storage.try_as_slice_f32()?,
                0usize,
                b.storage.try_as_slice_f32()?,
                &output_data,
                &mean_data,
                &rstd_data,
                batch_size as i32,
                norm_size as i32,
                eps
            )?;
        }
        (None, None) => {
            launch_kernel!(
                f,
                cfg,
                input.storage.try_as_slice_f32()?,
                0usize,
                0usize,
                &output_data,
                &mean_data,
                &rstd_data,
                batch_size as i32,
                norm_size as i32,
                eps
            )?;
        }
    }

    let output = Tensor {
        storage: TensorStorage::F32 {
            data: output_data.into(),
            numel: input.shape().elem_count(),
        },
        shape: input.shape().clone(),
        device: device.clone(),
        id: crate::tensor::TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    Ok(LayerNormForwardArtifacts {
        output,
        mean_data,
        rstd_data,
    })
}

fn layer_norm_forward_bf16(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    batch_size: usize,
    norm_size: usize,
    eps: f32,
) -> Result<LayerNormForwardArtifacts> {
    #[cfg(feature = "bf16_u16")]
    {
        if input.storage.dtype() != DType::BF16 {
            return Err(Error::InvalidOperation(
                "layer_norm: expected BF16 storage".into(),
            ));
        }

        let device = input.device();
        let mut mean_data = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;
        let mut rstd_data = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;

        let output = cuda_ops_bf16::layer_norm_bf16_with_stats(
            input,
            weight,
            bias,
            norm_size,
            eps,
            &mut mean_data,
            &mut rstd_data,
        )?;

        return Ok(LayerNormForwardArtifacts {
            output,
            mean_data,
            rstd_data,
        });
    }
    #[cfg(not(feature = "bf16_u16"))]
    {
        let _ = (input, weight, bias, batch_size, norm_size, eps);
        Err(Error::Unsupported(
            "layer_norm BF16 path requires bf16_u16 feature".into(),
        ))
    }
}
