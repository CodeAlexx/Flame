use crate::group_norm::group_norm;
use crate::tensor::contracts::{assert_nhwc_bf16_public, assert_nhwc_public, trap_is_bf16};
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::{AutogradContext, DType, Error, Op, Result, Shape, Tensor};
use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
#[allow(dead_code)]
mod fused_bf16 {
    use super::*;
    use std::ffi::c_void;

    #[allow(improper_ctypes)]
    extern "C" {
        fn layernorm_affine_bf16_nhwc_forward(
            x: *const c_void,
            y: *mut c_void,
            gamma: *const c_void,
            beta: *const c_void,
            B: i32,
            H: i32,
            W: i32,
            C: i32,
            eps: f32,
            stream: *mut c_void,
        );

        fn adaln_modulate_bf16_nhwc_forward(
            x: *const c_void,
            y: *mut c_void,
            gamma: *const c_void,
            beta: *const c_void,
            mod_s: *const c_void,
            mod_b: *const c_void,
            B: i32,
            H: i32,
            W: i32,
            C: i32,
            eps: f32,
            stream: *mut c_void,
        );
    }

    #[inline]
    fn as_bf16_ptr(tensor: &Tensor, tag: &str) -> Result<*const c_void> {
        tensor
            .as_device_ptr_bf16(tag)
            .map(|ptr| ptr as *const c_void)
    }

    #[inline]
    fn as_bf16_mut_ptr(tensor: &mut Tensor, tag: &str) -> Result<*mut c_void> {
        tensor
            .as_mut_device_ptr_bf16(tag)
            .map(|ptr| ptr as *mut c_void)
    }

    pub(super) fn layernorm_affine_bf16_inplace(
        x: &mut Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        b: i32,
        h: i32,
        w: i32,
        c: i32,
        eps: f32,
    ) -> Result<()> {
        let stream = *x.device().cu_stream() as *mut c_void;
        let x_ptr = as_bf16_mut_ptr(x, "layernorm_affine_bf16_inplace.x")?;
        let g_ptr = gamma
            .map(|t| as_bf16_ptr(t, "layernorm_affine_bf16_inplace.gamma"))
            .transpose()?;
        let b_ptr = beta
            .map(|t| as_bf16_ptr(t, "layernorm_affine_bf16_inplace.beta"))
            .transpose()?;
        unsafe {
            layernorm_affine_bf16_nhwc_forward(
                x_ptr,
                x_ptr,
                g_ptr.unwrap_or(std::ptr::null()),
                b_ptr.unwrap_or(std::ptr::null()),
                b,
                h,
                w,
                c,
                eps,
                stream,
            );
        }
        Ok(())
    }

    pub(super) fn adaln_modulate_bf16_inplace(
        x: &mut Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        mod_scale: Option<&Tensor>,
        mod_shift: Option<&Tensor>,
        b: i32,
        h: i32,
        w: i32,
        c: i32,
        eps: f32,
    ) -> Result<()> {
        let stream = *x.device().cu_stream() as *mut c_void;
        let x_ptr = as_bf16_mut_ptr(x, "adaln_modulate_bf16_inplace.x")?;
        let g_ptr = gamma
            .map(|t| as_bf16_ptr(t, "adaln_modulate_bf16_inplace.gamma"))
            .transpose()?;
        let b_ptr = beta
            .map(|t| as_bf16_ptr(t, "adaln_modulate_bf16_inplace.beta"))
            .transpose()?;
        let ms_ptr = mod_scale
            .map(|t| as_bf16_ptr(t, "adaln_modulate_bf16_inplace.mod_scale"))
            .transpose()?;
        let mb_ptr = mod_shift
            .map(|t| as_bf16_ptr(t, "adaln_modulate_bf16_inplace.mod_shift"))
            .transpose()?;
        unsafe {
            adaln_modulate_bf16_nhwc_forward(
                x_ptr,
                x_ptr,
                g_ptr.unwrap_or(std::ptr::null()),
                b_ptr.unwrap_or(std::ptr::null()),
                ms_ptr.unwrap_or(std::ptr::null()),
                mb_ptr.unwrap_or(std::ptr::null()),
                b,
                h,
                w,
                c,
                eps,
                stream,
            );
        }
        Ok(())
    }
}

/// Batch Normalization layer
pub struct BatchNorm2d {
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
}

impl BatchNorm2d {
    /// Create a new BatchNorm2d layer
    pub fn new(
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
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
        })
    }

    /// Forward pass for BatchNorm2d
    pub fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor> {
        assert_nhwc_bf16_public("BatchNorm2d::forward in", input)?;

        let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        let input_stats = if input_nchw.dtype() == DType::BF16 {
            input_nchw.to_dtype(DType::F32)?
        } else {
            input_nchw.clone_result()?
        };

        let dims = input_stats.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidOperation(
                "BatchNorm2d expects 4D input [N,C,H,W] after internal permutation".into(),
            ));
        }

        let num_channels = dims[1];
        if num_channels != self.num_features {
            return Err(Error::InvalidOperation(format!(
                "Expected {} channels, got {}",
                self.num_features, num_channels
            )));
        }

        let (mean, var) = if training || !self.track_running_stats {
            self.calculate_batch_stats(&input_nchw)?
        } else {
            let running_mean = self.running_mean.as_ref().ok_or_else(|| {
                Error::InvalidOperation("Running mean not available for evaluation".into())
            })?;
            let running_var = self.running_var.as_ref().ok_or_else(|| {
                Error::InvalidOperation("Running var not available for evaluation".into())
            })?;

            let mean_bc = running_mean.reshape(&[1, self.num_features, 1, 1])?;
            let var_bc = running_var.reshape(&[1, self.num_features, 1, 1])?;
            (mean_bc, var_bc)
        };

        if training && self.track_running_stats {
            self.update_running_stats(&mean, &var)?;
        }

        let normalized = self.normalize(&input_stats, &mean, &var)?;

        // Apply affine transformation if enabled
        let normalized = if self.affine {
            self.apply_affine(&normalized)?
        } else {
            normalized
        };

        let normalized_bf16 = if normalized.dtype() == DType::BF16 {
            normalized
        } else {
            normalized.to_dtype(DType::BF16)?
        };

        let output = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&normalized_bf16)?;
        assert_nhwc_bf16_public("BatchNorm2d::forward out", &output)?;
        Ok(output)
    }

    /// Calculate batch mean and variance
    fn calculate_batch_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let reduction_dims = [0, 2, 3];
        let mean = input.mean_dim(&reduction_dims, true)?;
        let centered = input.sub(&mean)?;
        let var = centered.square()?.mean_dim(&reduction_dims, true)?;
        Ok((mean, var))
    }

    /// Update running statistics
    fn update_running_stats(&mut self, batch_mean: &Tensor, batch_var: &Tensor) -> Result<()> {
        if let (Some(running_mean), Some(running_var)) =
            (&mut self.running_mean, &mut self.running_var)
        {
            let momentum = self.momentum;
            let one_minus_momentum = 1.0 - momentum;

            let mean_flat = batch_mean.reshape(&[self.num_features])?.detach()?;
            let var_flat = batch_var.reshape(&[self.num_features])?.detach()?;

            let decayed_mean = running_mean.mul_scalar(one_minus_momentum)?;
            let mean_update = mean_flat.mul_scalar(momentum)?;
            *running_mean = decayed_mean.add(&mean_update)?;

            let decayed_var = running_var.mul_scalar(one_minus_momentum)?;
            let var_update = var_flat.mul_scalar(momentum)?;
            *running_var = decayed_var.add(&var_update)?;

            self.num_batches_tracked += 1;
        }

        Ok(())
    }

    /// Normalize input using mean and variance
    fn normalize(&self, input: &Tensor, mean: &Tensor, var: &Tensor) -> Result<Tensor> {
        let centered = input.sub(mean)?;
        let inv_std = var.add_scalar(self.eps)?.rsqrt()?;
        centered.mul(&inv_std)
    }

    /// Apply affine transformation
    fn apply_affine(&self, normalized: &Tensor) -> Result<Tensor> {
        match (self.weight.as_ref(), self.bias.as_ref()) {
            (Some(weight), Some(bias)) => {
                let weight_view = weight.reshape(&[1, self.num_features, 1, 1])?;
                let bias_view = bias.reshape(&[1, self.num_features, 1, 1])?;
                normalized.mul(&weight_view)?.add(&bias_view)
            }
            (Some(weight), None) => {
                let weight_view = weight.reshape(&[1, self.num_features, 1, 1])?;
                normalized.mul(&weight_view)
            }
            (None, Some(bias)) => {
                let bias_view = bias.reshape(&[1, self.num_features, 1, 1])?;
                normalized.add(&bias_view)
            }
            _ => normalized.clone_result(),
        }
    }
}

/// Layer Normalization
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,

    // Learnable parameters
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    /// Create a new LayerNorm layer with default elementwise_affine=true
    pub fn new(
        normalized_shape: usize,
        eps: f32,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        Self::new_with_affine(vec![normalized_shape], eps, true, device)
    }

    /// Create a new LayerNorm layer with explicit parameters
    pub fn new_with_affine(
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let num_elements: usize = normalized_shape.iter().product();

        let (weight, bias) = if elementwise_affine {
            let weight = Tensor::from_vec_dtype(
                vec![1.0f32; num_elements],
                Shape::from_dims(&normalized_shape),
                device.clone(),
                DType::BF16,
            )?;
            let bias =
                Tensor::zeros_dtype(Shape::from_dims(&normalized_shape), DType::BF16, device)?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
            bias,
        })
    }

    /// Forward pass for LayerNorm
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.rank() == 4 {
            assert_nhwc_public("LayerNorm::forward in", input)?;
        }
        trap_is_bf16("LayerNorm::forward in", input)?;

        let weight = if self.elementwise_affine {
            self.weight.as_ref()
        } else {
            None
        };
        let bias = if self.elementwise_affine {
            self.bias.as_ref()
        } else {
            None
        };

        let mut output =
            crate::layer_norm::layer_norm(input, &self.normalized_shape, weight, bias, self.eps)?;

        if output.dtype() != DType::BF16 {
            output = output.to_dtype(DType::BF16)?;
        }
        debug_assert_eq!(output.dtype(), DType::BF16);
        if output.rank() == 4 {
            assert_nhwc_public("LayerNorm::forward out", &output)?;
        }
        Ok(output)
    }

    pub fn forward_into(&self, input: &Tensor, output: &mut Tensor) -> Result<()> {
        if input.rank() == 4 {
            assert_nhwc_public("LayerNorm::forward_into in", input)?;
        }
        trap_is_bf16("LayerNorm::forward_into in", input)?;

        let weight = if self.elementwise_affine {
            self.weight.as_ref()
        } else {
            None
        };
        let bias = if self.elementwise_affine {
            self.bias.as_ref()
        } else {
            None
        };

        crate::layer_norm::layer_norm_into(
            input,
            &self.normalized_shape,
            weight,
            bias,
            self.eps,
            output,
        )
    }
}

/// Group Normalization
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
    pub affine: bool,

    // Learnable parameters
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl GroupNorm {
    /// Create a new GroupNorm layer with default affine=true
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        Self::new_with_affine(num_groups, num_channels, eps, true, device)
    }

    /// Create a new GroupNorm layer with explicit affine parameter
    pub fn new_with_affine(
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        affine: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(Error::InvalidOperation(format!(
                "num_channels {} must be divisible by num_groups {}",
                num_channels, num_groups
            )));
        }

        let (weight, bias) = if affine {
            let weight = Tensor::from_vec_dtype(
                vec![1.0f32; num_channels],
                Shape::from_dims(&[num_channels]),
                device.clone(),
                DType::BF16,
            )?;
            let bias = Tensor::zeros_dtype(Shape::from_dims(&[num_channels]), DType::BF16, device)?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        Ok(Self {
            num_groups,
            num_channels,
            eps,
            affine,
            weight,
            bias,
        })
    }

    /// Forward pass for GroupNorm
    /// Input shape: [N, C, H, W]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = if self.affine {
            self.weight.as_ref()
        } else {
            None
        };
        let bias = if self.affine {
            self.bias.as_ref()
        } else {
            None
        };

        group_norm(input, self.num_groups, weight, bias, self.eps)
    }

    /// Apply affine transformation
    fn apply_affine(&self, normalized: &Tensor) -> Result<Tensor> {
        match (self.weight.as_ref(), self.bias.as_ref()) {
            (Some(weight), Some(bias)) => {
                let weight_view = weight.reshape(&[1, self.num_channels, 1, 1])?;
                let bias_view = bias.reshape(&[1, self.num_channels, 1, 1])?;
                normalized.mul(&weight_view)?.add(&bias_view)
            }
            (Some(weight), None) => {
                let weight_view = weight.reshape(&[1, self.num_channels, 1, 1])?;
                normalized.mul(&weight_view)
            }
            (None, Some(bias)) => {
                let bias_view = bias.reshape(&[1, self.num_channels, 1, 1])?;
                normalized.add(&bias_view)
            }
            _ => normalized.clone_result(),
        }
    }
}

/// Instance Normalization
pub struct InstanceNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub affine: bool,
    pub track_running_stats: bool,

    // Learnable parameters
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,

    // Running statistics (usually not used in InstanceNorm)
    pub running_mean: Option<Tensor>,
    pub running_var: Option<Tensor>,
    pub num_batches_tracked: usize,
}

impl InstanceNorm2d {
    /// Create a new InstanceNorm2d layer
    pub fn new(
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
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
        })
    }

    /// Forward pass for InstanceNorm2d
    /// Input shape: [N, C, H, W]
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("InstanceNorm2d::forward in", input)?;

        let input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        let input_stats = if input_nchw.dtype() == DType::BF16 {
            input_nchw.to_dtype(DType::F32)?
        } else {
            input_nchw.clone_result()?
        };

        let dims = input_stats.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidOperation(
                "InstanceNorm2d expects 4D input [N,C,H,W] after internal permutation".into(),
            ));
        }

        if dims[1] != self.num_features {
            return Err(Error::InvalidOperation(format!(
                "Expected {} channels, got {}",
                self.num_features, dims[1]
            )));
        }

        let mean = input_stats.mean_dim(&[2, 3], true)?;
        let centered = input_stats.sub(&mean)?;
        let var = centered.square()?.mean_dim(&[2, 3], true)?;
        let inv_std = var.add_scalar(self.eps)?.rsqrt()?;
        let normalized = centered.mul(&inv_std)?;

        let normalized = if self.affine {
            self.apply_affine(&normalized)?
        } else {
            normalized
        };

        let normalized_bf16 = if normalized.dtype() == DType::BF16 {
            normalized
        } else {
            normalized.to_dtype(DType::BF16)?
        };

        let output = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&normalized_bf16)?;
        assert_nhwc_bf16_public("InstanceNorm2d::forward out", &output)?;
        Ok(output)
    }

    /// Apply affine transformation (same as BatchNorm2d)
    fn apply_affine(&self, normalized: &Tensor) -> Result<Tensor> {
        match (self.weight.as_ref(), self.bias.as_ref()) {
            (Some(weight), Some(bias)) => {
                let weight_view = weight.reshape(&[1, self.num_features, 1, 1])?;
                let bias_view = bias.reshape(&[1, self.num_features, 1, 1])?;
                normalized.mul(&weight_view)?.add(&bias_view)
            }
            (Some(weight), None) => {
                let weight_view = weight.reshape(&[1, self.num_features, 1, 1])?;
                normalized.mul(&weight_view)
            }
            (None, Some(bias)) => {
                let bias_view = bias.reshape(&[1, self.num_features, 1, 1])?;
                normalized.add(&bias_view)
            }
            _ => normalized.clone_result(),
        }
    }
}

struct RmsNormForwardArtifacts {
    output: Tensor,
    inv_rms: CudaSlice<f32>,
}

fn rms_norm_forward(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    eps: f32,
) -> Result<RmsNormForwardArtifacts> {
    if input.dtype() != DType::BF16 || input.storage.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "RMSNorm expects BF16 input storage".into(),
        ));
    }

    let norm_size: usize = normalized_shape.iter().product();
    if norm_size == 0 {
        return Err(Error::InvalidOperation(
            "RMSNorm normalized_shape must be non-empty".into(),
        ));
    }

    let total_elems = input.shape().elem_count();
    if total_elems % norm_size != 0 {
        return Err(Error::InvalidOperation(
            "RMSNorm input size is not divisible by normalized_shape".into(),
        ));
    }

    let batch_size = total_elems / norm_size;
    rms_norm_forward_bf16(input, weight, batch_size, norm_size, eps)
}

fn rms_norm_forward_bf16(
    input: &Tensor,
    weight: Option<&Tensor>,
    batch_size: usize,
    norm_size: usize,
    eps: f32,
) -> Result<RmsNormForwardArtifacts> {
    use crate::cuda_kernels::CudaKernels;

    // Caller (rms_norm_forward) already validated BF16 dtype
    debug_assert_eq!(input.storage.dtype(), DType::BF16);

    let device = input.device();
    CudaKernels::ensure_kernel(device, "rms_norm_forward_bf16", RMS_NORM_FWD_KERNEL_BF16)?;
    let f = device
        .get_func("rms_norm_forward_bf16", "rms_norm_forward_bf16")
        .ok_or_else(|| Error::Cuda("Failed to get rms_norm_forward_bf16 kernel".into()))?;

    let output_data = unsafe { device.alloc::<u16>(input.shape().elem_count()) }
        .map_err(|e| Error::Cuda(format!("rms_norm forward alloc failed: {e:?}")))?;
    let inv_rms_data = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;

    use cudarc::driver::DevicePtr;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    match weight {
        Some(w) => {
            if w.dtype() != DType::BF16 || w.storage.dtype() != DType::BF16 {
                return Err(Error::InvalidInput(
                    "RMSNorm expects BF16 weight storage".into(),
                ));
            }
            let input_ptr = input.as_device_ptr_bf16("rms_norm_forward_bf16:input")? as u64;
            let output_ptr = *output_data.device_ptr();
            let weight_ptr = w.as_device_ptr_bf16("rms_norm_forward_bf16:weight")? as u64;

            launch_kernel!(
                f,
                cfg,
                input_ptr,
                output_ptr,
                weight_ptr,
                &inv_rms_data,
                batch_size as i32,
                norm_size as i32,
                eps,
                1i32
            );
        }
        None => {
            let input_ptr = input.as_device_ptr_bf16("rms_norm_forward_bf16:input")? as u64;
            let output_ptr = *output_data.device_ptr();
            let null_w = device.null::<u16>()?;

            launch_kernel!(
                f,
                cfg,
                input_ptr,
                output_ptr,
                &null_w,
                &inv_rms_data,
                batch_size as i32,
                norm_size as i32,
                eps,
                0i32
            );
        }
    }

    let output = Tensor {
        storage: TensorStorage::BF16 {
            data: output_data.into(),
            numel: input.shape().elem_count(),
        },
        shape: input.shape().clone(),
        device: device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    Ok(RmsNormForwardArtifacts {
        output,
        inv_rms: inv_rms_data,
    })
}

pub(crate) fn rms_norm_backward(
    grad_out: &Tensor,
    input: &Tensor,
    weight: Option<&Tensor>,
    inv_rms: &Tensor,
    normalized_shape: &[usize],
) -> Result<(Tensor, Option<Tensor>)> {
    if grad_out.dtype() != DType::BF16 || grad_out.storage.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "RMSNorm backward expects BF16 grad_out storage".into(),
        ));
    }
    if input.dtype() != DType::BF16 || input.storage.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "RMSNorm backward expects BF16 input storage".into(),
        ));
    }
    if inv_rms.storage.dtype() != DType::F32 {
        return Err(Error::InvalidInput(
            "RMSNorm backward expects inv_rms stored as F32".into(),
        ));
    }

    let total_elems = input.shape().elem_count();
    let batch_size = inv_rms.shape().elem_count();
    if batch_size == 0 || total_elems % batch_size != 0 {
        return Err(Error::InvalidOperation(
            "RMSNorm backward: invalid inv_rms shape".into(),
        ));
    }
    let norm_size = total_elems / batch_size;
    let expected_norm: usize = normalized_shape.iter().product();
    if expected_norm != norm_size {
        return Err(Error::InvalidOperation(
            "RMSNorm backward: normalized_shape mismatch".into(),
        ));
    }

    let (grad_input, grad_weight_f32) =
        rms_norm_backward_bf16(grad_out, input, weight, inv_rms, batch_size, norm_size)?;

    let grad_weight_tensor = if let Some(data) = grad_weight_f32 {
        let device = input.device.clone();
        let grad_weight_f32_tensor = Tensor {
            storage: TensorStorage::F32 {
                data: data.into(),
                numel: norm_size,
            },
            shape: Shape::from_dims(normalized_shape),
            device,
            id: TensorId::new(),
            requires_grad: false,
        };
        Some(grad_weight_f32_tensor.to_dtype(DType::BF16)?)
    } else {
        None
    };

    Ok((grad_input, grad_weight_tensor))
}

fn rms_norm_backward_bf16(
    grad_out: &Tensor,
    input: &Tensor,
    weight: Option<&Tensor>,
    inv_rms: &Tensor,
    batch_size: usize,
    norm_size: usize,
) -> Result<(Tensor, Option<CudaSlice<f32>>)> {
    use crate::cuda_kernels::CudaKernels;

    let device = input.device();
    CudaKernels::ensure_kernel(device, "rms_norm_backward_bf16", RMS_NORM_BWD_KERNEL_BF16)?;
    let f = device
        .get_func("rms_norm_backward_bf16", "rms_norm_backward_bf16")
        .ok_or_else(|| Error::Cuda("Failed to get rms_norm_backward_bf16 kernel".into()))?;

    let grad_input_data = unsafe { device.alloc::<u16>(input.shape().elem_count()) }
        .map_err(|e| Error::Cuda(format!("rms_norm backward alloc failed: {e:?}")))?;
    let mut grad_weight_data = if weight.is_some() {
        Some(crate::tensor::alloc_zeros_from_pool(device, norm_size)?)
    } else {
        None
    };

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    match (weight, grad_weight_data.as_mut()) {
        (Some(w), Some(gw)) => {
            if w.dtype() != DType::BF16 || w.storage.dtype() != DType::BF16 {
                return Err(Error::InvalidInput(
                    "RMSNorm backward expects BF16 weight storage".into(),
                ));
            }
            launch_kernel!(
                f,
                cfg,
                grad_out.storage.try_as_slice_u16()?,
                input.storage.try_as_slice_u16()?,
                w.storage.try_as_slice_u16()?,
                &grad_input_data,
                gw,
                inv_rms.storage.try_as_slice_f32()?,
                batch_size as i32,
                norm_size as i32,
                1i32
            );
        }
        _ => {
            let null_weight = device.null::<u16>()?;
            let mut null_grad_weight = device.null::<f32>()?;
            launch_kernel!(
                f,
                cfg,
                grad_out.storage.try_as_slice_u16()?,
                input.storage.try_as_slice_u16()?,
                &null_weight,
                &grad_input_data,
                &mut null_grad_weight,
                inv_rms.storage.try_as_slice_f32()?,
                batch_size as i32,
                norm_size as i32,
                0i32
            );
        }
    }

    let grad_input = Tensor {
        storage: TensorStorage::BF16 {
            data: grad_input_data.into(),
            numel: input.shape().elem_count(),
        },
        shape: input.shape().clone(),
        device: device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    Ok((grad_input, grad_weight_data))
}

/// RMS Normalization (Root Mean Square Layer Normalization)
/// Used in many modern transformer models like LLaMA, Mistral, etc.
pub struct RMSNorm {
    pub eps: f32,
    pub elementwise_affine: bool,
    pub normalized_shape: Vec<usize>,

    // Learnable parameters
    pub weight: Option<Tensor>,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let num_elements: usize = normalized_shape.iter().product();

        let weight = if elementwise_affine {
            Some(Tensor::from_vec_dtype(
                vec![1.0f32; num_elements],
                Shape::from_dims(&normalized_shape),
                device,
                DType::BF16,
            )?)
        } else {
            None
        };

        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
        })
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
            .ok_or_else(|| Error::InvalidOperation("RMSNorm has no affine weight".into()))?;
        let requires_grad = weight.requires_grad();
        let tensor = Self::convert_param(weight, source, "RMSNorm::copy_weight_from")?;
        *weight = tensor.requires_grad_(requires_grad);
        Ok(())
    }

    /// Forward pass for RMSNorm
    /// RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.rank() == 4 {
            assert_nhwc_public("RMSNorm::forward in", input)?;
        }
        trap_is_bf16("RMSNorm::forward in", input)?;

        let input_dims = input.shape().dims();
        let input_shape_len = input_dims.len();
        let normalized_shape_len = self.normalized_shape.len();

        // Validate that normalized_shape matches the last dimensions of input
        if normalized_shape_len > input_shape_len {
            return Err(Error::InvalidOperation(
                "Normalized shape is larger than input shape".into(),
            ));
        }

        let start_idx = input_shape_len - normalized_shape_len;
        for i in 0..normalized_shape_len {
            if input_dims[start_idx + i] != self.normalized_shape[i] {
                return Err(Error::InvalidOperation(format!(
                    "Shape mismatch at dimension {}: expected {}, got {}",
                    i,
                    self.normalized_shape[i],
                    input_dims[start_idx + i]
                )));
            }
        }

        let artifacts = rms_norm_forward(
            input,
            &self.normalized_shape,
            self.weight.as_ref(),
            self.eps,
        )?;

        let mut output = artifacts.output;
        if output.dtype() != DType::BF16 {
            output = output.to_dtype(DType::BF16)?;
        }
        // Output is created as BF16 by rms_norm_forward_bf16 — skip redundant check
        debug_assert_eq!(output.dtype(), DType::BF16);
        if output.rank() == 4 {
            assert_nhwc_public("RMSNorm::forward out", &output)?;
        }

        let needs_grad = input.requires_grad
            || self
                .weight
                .as_ref()
                .map(|w| w.requires_grad)
                .unwrap_or(false);

        if needs_grad {
            output.requires_grad = true;

            let mut saved_tensors = vec![(input.id, input.clone_result()?)];
            if let Some(w) = &self.weight {
                saved_tensors.push((w.id, w.clone_result()?));
            }

            let batch_size = artifacts.inv_rms.len();
            let inv_rms_tensor = Tensor {
                storage: TensorStorage::F32 {
                    data: artifacts.inv_rms.into(),
                    numel: batch_size,
                },
                shape: Shape::from_dims(&[batch_size]),
                device: input.device.clone(),
                id: TensorId::new(),
                requires_grad: false,
            };
            let inv_rms_id = inv_rms_tensor.id;
            saved_tensors.push((inv_rms_id, inv_rms_tensor));

            AutogradContext::record_op(
                output.id,
                Op::RMSNorm {
                    input: input.id,
                    weight: self.weight.as_ref().map(|w| w.id),
                    eps: self.eps,
                    inv_rms: inv_rms_id,
                    normalized_shape: self.normalized_shape.clone(),
                },
                saved_tensors,
            );
        }

        Ok(output)
    }
}

pub const RMS_NORM_FWD_KERNEL_BF16: &str = r#"
#include <cuda_bf16.h>

__device__ inline float rms_bf16_load(const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

__device__ inline void rms_bf16_store(__nv_bfloat16* ptr, int idx, float value) {
    ptr[idx] = __float2bfloat16_rn(value);
}

extern "C" __global__ void rms_norm_forward_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const __nv_bfloat16* weight,
    float* inv_rms_out,
    int batch_size,
    int norm_size,
    float eps,
    int has_weight
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    int base = row * norm_size;
    float sum_sq = 0.0f;
    for (int i = 0; i < norm_size; ++i) {
        float v = rms_bf16_load(input, base + i);
        sum_sq += v * v;
    }

    float mean_sq = sum_sq / norm_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    inv_rms_out[row] = inv_rms;

    for (int i = 0; i < norm_size; ++i) {
        float val = rms_bf16_load(input, base + i) * inv_rms;
        if (has_weight && weight != nullptr) {
            val *= rms_bf16_load(weight, i);
        }
        rms_bf16_store(output, base + i, val);
    }
}
"#;

pub const RMS_NORM_BWD_KERNEL_BF16: &str = r#"
#include <cuda_bf16.h>

__device__ inline float rms_bwd_load(const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

__device__ inline void rms_bwd_store(__nv_bfloat16* ptr, int idx, float value) {
    ptr[idx] = __float2bfloat16_rn(value);
}

extern "C" __global__ void rms_norm_backward_bf16(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* grad_input,
    float* grad_weight,
    const float* inv_rms,
    int batch_size,
    int norm_size,
    int has_weight
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    int base = row * norm_size;
    float inv = inv_rms[row];
    float inv_cubed = inv * inv * inv;

    float dot = 0.0f;
    for (int i = 0; i < norm_size; ++i) {
        float go = rms_bwd_load(grad_out, base + i);
        float x = rms_bwd_load(input, base + i);
        float w = (has_weight && weight != nullptr) ? rms_bwd_load(weight, i) : 1.0f;
        dot += (go * w) * x;
    }

    float coeff = inv_cubed * dot / norm_size;

    for (int i = 0; i < norm_size; ++i) {
        float go = rms_bwd_load(grad_out, base + i);
        float x = rms_bwd_load(input, base + i);
        float w = (has_weight && weight != nullptr) ? rms_bwd_load(weight, i) : 1.0f;
        float scaled = go * w;
        float grad_x = inv * scaled - x * coeff;
        rms_bwd_store(grad_input, base + i, grad_x);

        if (has_weight && grad_weight != nullptr) {
            float contrib = go * x * inv;
            atomicAdd(&grad_weight[i], contrib);
        }
    }
}
"#;

/// RMSNorm specifically for 1D inputs (common in transformers)
pub struct RMSNorm1d {
    pub normalized_shape: usize,
    pub eps: f32,
    pub weight: Option<Tensor>,
}

impl RMSNorm1d {
    /// Create a new RMSNorm1d layer
    pub fn new(
        normalized_shape: usize,
        eps: f32,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let weight = Some(Tensor::from_vec_dtype(
            vec![1.0f32; normalized_shape],
            Shape::from_dims(&[normalized_shape]),
            device,
            DType::BF16,
        )?);

        Ok(Self {
            normalized_shape,
            eps,
            weight,
        })
    }

    /// Forward pass for RMSNorm1d
    /// Input shape: [..., normalized_shape]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        let last_dim = dims[dims.len() - 1];

        if last_dim != self.normalized_shape {
            return Err(Error::InvalidOperation(format!(
                "Expected last dimension {}, got {}",
                self.normalized_shape, last_dim
            )));
        }

        // Use the general RMSNorm with the last dimension
        let rms_norm = RMSNorm {
            normalized_shape: vec![self.normalized_shape],
            eps: self.eps,
            elementwise_affine: self.weight.is_some(),
            weight: match &self.weight {
                Some(w) => Some(w.clone_result()?),
                None => None,
            },
        };

        rms_norm.forward(input)
    }
}
