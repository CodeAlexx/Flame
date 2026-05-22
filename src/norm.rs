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
        // Autograd v2 prereq: bump version on in-place mutation.
        x.storage_ref().bump_version();
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
        // Autograd v2 prereq: bump version on in-place mutation.
        x.storage_ref().bump_version();
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

    // The `rms_norm_forward_bf16` kernel assumes row-major contiguous layout
    // (reads `row * norm_size + i` directly from the raw pointer, ignoring
    // strides). If the caller hands us a permuted view — e.g. Klein's
    // `split_qkv` produces `[B, H, L, D]` via `.permute([0, 2, 1, 3])` on a
    // `[B, L, H, D]` contiguous buffer — the kernel reads the wrong element
    // ordering, silently shipping a tensor whose logical [b, h, l, d] index
    // returns the value that was physically at (b, l, h, d). The output has
    // correct per-row RMS, so it's invisible to unit tests that only check
    // magnitude; but downstream ops that rely on the logical axis labels
    // (cat on dim=2, RoPE keyed on `l`) see scrambled data.
    // Bit-exact by construction once `contiguous()` is enforced.
    let input_owned;
    let input = if input.is_contiguous() {
        input
    } else {
        input_owned = input.contiguous()?;
        &input_owned
    };

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

    // 2026-05-12 perf: dispatch to the vectorized (4-wide, 256-thread) kernel
    // when norm_size is divisible by 4. Production shapes (Z-Image hidden=2560,
    // head_dim=128, klein=4096, chroma=3072) all qualify. Fallback to the
    // legacy single-thread-per-row scalar kernel only for odd geometries.
    // Env override `FLAME_RMS_NORM_LEGACY=1` forces the scalar path for A/B
    // benchmarking; keep this gate until the vec path has been validated in
    // training, then drop the gate.
    let use_vec = norm_size % 4 == 0
        && std::env::var("FLAME_RMS_NORM_LEGACY")
            .map(|v| v == "0")
            .unwrap_or(true);

    let use_hidden4096_pytorch = use_vec
        && norm_size == 4096
        && weight.is_none()
        && std::env::var("FLAME_RMS_NORM_PYTORCH_HIDDEN4096")
            .map(|v| v != "0")
            .unwrap_or(true);
    let use_head128 = use_vec && norm_size == 128;
    let kernel_src = if use_hidden4096_pytorch {
        RMS_NORM_FWD_KERNEL_BF16_HIDDEN4096_PYTORCH
    } else if use_head128 {
        RMS_NORM_FWD_KERNEL_BF16_HEAD128
    } else if use_vec {
        RMS_NORM_FWD_KERNEL_BF16_VEC
    } else {
        RMS_NORM_FWD_KERNEL_BF16
    };
    let kernel_name = if use_hidden4096_pytorch {
        "rms_norm_forward_bf16_hidden4096_pytorch"
    } else if use_head128 {
        "rms_norm_forward_bf16_head128"
    } else if use_vec {
        "rms_norm_forward_bf16_vec"
    } else {
        "rms_norm_forward_bf16"
    };
    CudaKernels::ensure_kernel(device, kernel_name, kernel_src)?;
    let f = device
        .get_func(kernel_name, kernel_name)
        .ok_or_else(|| Error::Cuda(format!("Failed to get {kernel_name} kernel")))?;

    let output_data = crate::cuda_alloc_pool::pool_alloc_u16(device, input.shape().elem_count())
        .map_err(|e| Error::Cuda(format!("rms_norm forward alloc failed: {e:?}")))?;
    let inv_rms_data = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;

    use cudarc::driver::DevicePtr;

    let cfg = if use_hidden4096_pytorch {
        LaunchConfig {
            grid_dim: (((batch_size as u32) + 15) / 16, 1, 1),
            block_dim: (32, 16, 1),
            shared_mem_bytes: 0,
        }
    } else if use_head128 {
        LaunchConfig {
            grid_dim: (((batch_size as u32) + 15) / 16, 1, 1),
            block_dim: (32, 16, 1),
            shared_mem_bytes: 0,
        }
    } else if use_vec {
        // Match PyTorch's vectorized RMSNorm launch:
        // dim3 threads(warp_size, num_threads() / warp_size) == (32, 8).
        // Shared memory holds `threads.y * 3 / 2` floats for the same
        // inter-warp combine tree used by layer_norm_kernel.cu.
        LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 12 * std::mem::size_of::<f32>() as u32,
        }
    } else {
        LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        }
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
        custom_strides: None,
        view_offset: 0,
        #[cfg(feature = "autograd_v2")]
        autograd_meta: None,
    };

    Ok(RmsNormForwardArtifacts {
        output,
        inv_rms: inv_rms_data,
    })
}

/// Bench-only escape hatch — call `rms_norm_backward` directly without
/// going through the autograd machinery. **Do not use in production code.**
/// Used by `benches/rms_norm_vec.rs` to time the backward kernel in isolation.
#[doc(hidden)]
pub fn rms_norm_backward_for_bench(
    grad_out: &Tensor,
    input: &Tensor,
    weight: Option<&Tensor>,
    inv_rms: &Tensor,
    _batch_size: usize,
    norm_size: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    rms_norm_backward(grad_out, input, weight, inv_rms, &[norm_size], weight.is_some())
}

pub(crate) fn rms_norm_backward(
    grad_out: &Tensor,
    input: &Tensor,
    weight: Option<&Tensor>,
    inv_rms: &Tensor,
    normalized_shape: &[usize],
    compute_grad_weight: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let grad_out_bf16_owned;
    let grad_out_bf16: &Tensor =
        if grad_out.dtype() == DType::BF16 && grad_out.storage.dtype() == DType::BF16 {
            grad_out
        } else {
            grad_out_bf16_owned = grad_out.to_dtype(DType::BF16)?;
            &grad_out_bf16_owned
        };
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

    // Same contiguity fix as the forward: the backward kernel reads the
    // device pointer as a contiguous `[batch * norm_size]` buffer. Klein's
    // `split_qkv → permute` saves a strided view into the autograd tape;
    // without this, the backward computes gradients in physical-memory
    // order but labels them with logical shape → scrambled grads on the
    // same class of inputs as the forward bug. Observed as step-1
    // grad_norm ~5e10 on LoRA B parameters that flow through QK RMSNorm.
    let input_owned;
    let input = if input.is_contiguous() {
        input
    } else {
        input_owned = input.contiguous()?;
        &input_owned
    };
    let grad_out_owned;
    let grad_out_bf16 = if grad_out_bf16.is_contiguous() {
        grad_out_bf16
    } else {
        grad_out_owned = grad_out_bf16.contiguous()?;
        &grad_out_owned
    };

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
        rms_norm_backward_bf16(
            grad_out_bf16,
            input,
            weight,
            inv_rms,
            batch_size,
            norm_size,
            compute_grad_weight,
        )?;

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
            custom_strides: None,
            view_offset: 0,
            #[cfg(feature = "autograd_v2")]
            autograd_meta: None,
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
    compute_grad_weight: bool,
) -> Result<(Tensor, Option<CudaSlice<f32>>)> {
    use crate::cuda_kernels::CudaKernels;
    use cudarc::driver::DevicePtr;

    let device = input.device();

    // 2026-05-12 perf: dispatch to the vectorized backward when norm_size is
    // divisible by 4 (all production shapes qualify). Env override
    // `FLAME_RMS_NORM_LEGACY=1` forces the scalar path (mirrors forward dispatch).
    let use_vec = norm_size % 4 == 0
        && std::env::var("FLAME_RMS_NORM_LEGACY")
            .map(|v| v == "0")
            .unwrap_or(true);

    let kernel_src = if use_vec {
        RMS_NORM_BWD_KERNEL_BF16_VEC
    } else {
        RMS_NORM_BWD_KERNEL_BF16
    };
    let kernel_name = if use_vec {
        "rms_norm_backward_bf16_vec"
    } else {
        "rms_norm_backward_bf16"
    };
    CudaKernels::ensure_kernel(device, kernel_name, kernel_src)?;
    let f = device
        .get_func(kernel_name, kernel_name)
        .ok_or_else(|| Error::Cuda(format!("Failed to get {kernel_name} kernel")))?;

    let grad_input_data =
        crate::cuda_alloc_pool::pool_alloc_u16(device, input.shape().elem_count())
            .map_err(|e| Error::Cuda(format!("rms_norm backward alloc failed: {e:?}")))?;
    let mut grad_weight_data = if compute_grad_weight && weight.is_some() {
        Some(crate::tensor::alloc_zeros_from_pool(device, norm_size)?)
    } else {
        None
    };

    let cfg = if use_vec {
        LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8 * std::mem::size_of::<f32>() as u32,
        }
    } else {
        LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        }
    };

    let grad_out_ptr = grad_out.as_device_ptr_bf16("rms_norm_backward_bf16:grad_out")? as u64;
    let input_ptr = input.as_device_ptr_bf16("rms_norm_backward_bf16:input")? as u64;
    let grad_input_ptr = *grad_input_data.device_ptr();

    match weight {
        Some(w) => {
            if w.dtype() != DType::BF16 || w.storage.dtype() != DType::BF16 {
                return Err(Error::InvalidInput(
                    "RMSNorm backward expects BF16 weight storage".into(),
                ));
            }
            let weight_ptr = w.as_device_ptr_bf16("rms_norm_backward_bf16:weight")? as u64;
            let gw_ptr: u64 = grad_weight_data
                .as_mut()
                .map(|gw| *gw.device_ptr())
                .unwrap_or(0u64);
            launch_kernel!(
                f,
                cfg,
                grad_out_ptr,
                input_ptr,
                weight_ptr,
                grad_input_ptr,
                gw_ptr,
                inv_rms.storage.try_as_slice_f32()?,
                batch_size as i32,
                norm_size as i32,
                1i32
            );

            // 2026-05-12 perf: second kernel for grad_weight cross-row reduction.
            // The vec backward above writes only `grad_input` (the inline
            // atomicAdd path was removed). This kernel does the
            // batch_size-row reduction with ~500× fewer atomicAdds via tiling.
            // Only fires in the vec path; the legacy scalar kernel still
            // accumulates grad_weight inline.
            if use_vec && grad_weight_data.is_some() {
                CudaKernels::ensure_kernel(
                    device,
                    "rms_norm_grad_weight_bf16_vec",
                    RMS_NORM_GRAD_WEIGHT_KERNEL_BF16,
                )?;
                let gw_func = device
                    .get_func(
                        "rms_norm_grad_weight_bf16_vec",
                        "rms_norm_grad_weight_bf16_vec",
                    )
                    .ok_or_else(|| {
                        Error::Cuda("Failed to get rms_norm_grad_weight_bf16_vec kernel".into())
                    })?;

                const COLS_PER_BLOCK: u32 = 64;
                const ROWS_PER_BLOCK: u32 = 512;
                let cols_blocks = ((norm_size as u32) + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
                let rows_blocks = ((batch_size as u32) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
                let gw_cfg = LaunchConfig {
                    grid_dim: (cols_blocks, rows_blocks, 1),
                    block_dim: (COLS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                launch_kernel!(
                    gw_func,
                    gw_cfg,
                    grad_out_ptr,
                    input_ptr,
                    inv_rms.storage.try_as_slice_f32()?,
                    gw_ptr,
                    batch_size as i32,
                    norm_size as i32
                );
            }
        }
        _ => {
            let null_weight = device.null::<u16>()?;
            let mut null_grad_weight = device.null::<f32>()?;
            launch_kernel!(
                f,
                cfg,
                grad_out_ptr,
                input_ptr,
                *null_weight.device_ptr(),
                grad_input_ptr,
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
        custom_strides: None,
        view_offset: 0,
        #[cfg(feature = "autograd_v2")]
        autograd_meta: None,
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
        rms_norm(
            input,
            &self.normalized_shape,
            self.weight.as_ref(),
            self.eps,
        )
    }
}

/// Functional RMSNorm.
///
/// Applies RMSNorm over the last `normalized_shape.len()` dimensions and uses
/// the fused BF16 kernel + `Op::RMSNorm` autograd path when available.
pub fn rms_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    if input.rank() == 4 {
        assert_nhwc_public("rms_norm::in", input)?;
    }
    // Auto-cast non-BF16 inputs. Use autograd-aware cast when the input
    // requires grad so the gradient chain isn't silently broken.
    let input = if input.dtype() != DType::BF16 {
        if input.requires_grad {
            &input.to_dtype(DType::BF16)?
        } else {
            &input.to_dtype_no_grad(DType::BF16)?
        }
    } else {
        input
    };
    trap_is_bf16("rms_norm::in", input)?;

    let input_dims = input.shape().dims();
    let input_shape_len = input_dims.len();
    let normalized_shape_len = normalized_shape.len();

    // Validate that normalized_shape matches the last dimensions of input
    if normalized_shape_len > input_shape_len {
        return Err(Error::InvalidOperation(
            "Normalized shape is larger than input shape".into(),
        ));
    }

    let start_idx = input_shape_len - normalized_shape_len;
    for i in 0..normalized_shape_len {
        if input_dims[start_idx + i] != normalized_shape[i] {
            return Err(Error::InvalidOperation(format!(
                "Shape mismatch at dimension {}: expected {}, got {}",
                i,
                normalized_shape[i],
                input_dims[start_idx + i]
            )));
        }
    }

    let artifacts = rms_norm_forward(input, normalized_shape, weight, eps)?;

    let mut output = artifacts.output;
    if output.dtype() != DType::BF16 {
        output = output.to_dtype(DType::BF16)?;
    }
    // Output is created as BF16 by rms_norm_forward_bf16 — skip redundant check
    debug_assert_eq!(output.dtype(), DType::BF16);
    if output.rank() == 4 {
        assert_nhwc_public("rms_norm::out", &output)?;
    }

    let needs_grad = input.requires_grad || weight.map(|w| w.requires_grad).unwrap_or(false);

    if needs_grad {
        output.requires_grad = true;
        if AutogradContext::is_recording() {
            let mut saved_tensors = vec![(input.id, input.alias())];
            if let Some(w) = weight {
                saved_tensors.push((w.id, w.alias()));
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
                custom_strides: None,
                view_offset: 0,
                #[cfg(feature = "autograd_v2")]
                autograd_meta: None,
            };
            let inv_rms_id = inv_rms_tensor.id;
            saved_tensors.push((inv_rms_id, inv_rms_tensor));

            AutogradContext::record_op(
                output.id,
                Op::RMSNorm {
                    input: input.id,
                    weight: weight.map(|w| w.id),
                    eps,
                    inv_rms: inv_rms_id,
                    normalized_shape: normalized_shape.to_vec(),
                },
                saved_tensors,
            );
        }
    }

    Ok(output)
}

#[doc(hidden)]
pub fn rms_norm_inv_rms(
    input: &Tensor,
    normalized_shape: &[usize],
    eps: f32,
) -> Result<Tensor> {
    let input = if input.dtype() != DType::BF16 {
        input.to_dtype_no_grad(DType::BF16)?
    } else {
        input.alias()
    };
    let norm_size: usize = normalized_shape.iter().product();
    if norm_size == 0 {
        return Err(Error::InvalidOperation(
            "RMSNorm normalized_shape must be non-empty".into(),
        ));
    }
    let batch_size = input.shape().elem_count() / norm_size;
    let artifacts = rms_norm_forward(&input, normalized_shape, None, eps)?;
    Ok(Tensor {
        storage: TensorStorage::F32 {
            data: artifacts.inv_rms.into(),
            numel: batch_size,
        },
        shape: Shape::from_dims(&[batch_size]),
        device: input.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,
        #[cfg(feature = "autograd_v2")]
        autograd_meta: None,
    })
}

#[doc(hidden)]
pub fn rms_norm_head128_mean_sq(input: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || input.storage.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "RMSNorm head128 mean_sq expects BF16 input storage".into(),
        ));
    }
    let norm_size = *input
        .shape()
        .dims()
        .last()
        .ok_or_else(|| Error::InvalidInput("RMSNorm head128 mean_sq needs rank >= 1".into()))?;
    if norm_size != 128 {
        return Err(Error::InvalidInput(format!(
            "RMSNorm head128 mean_sq expects last dim 128, got {norm_size}",
        )));
    }
    let input_owned;
    let input = if input.is_contiguous() {
        input
    } else {
        input_owned = input.contiguous()?;
        &input_owned
    };

    use crate::cuda_kernels::CudaKernels;
    use cudarc::driver::DevicePtr;

    let device = input.device();
    let batch_size = input.shape().elem_count() / norm_size;
    CudaKernels::ensure_kernel(
        device,
        "rms_norm_mean_sq_bf16_head128",
        RMS_NORM_MEAN_SQ_KERNEL_BF16_HEAD128,
    )?;
    let f = device
        .get_func("rms_norm_mean_sq_bf16_head128", "rms_norm_mean_sq_bf16_head128")
        .ok_or_else(|| Error::Cuda("Failed to get rms_norm_mean_sq_bf16_head128 kernel".into()))?;
    let mean_sq_data = crate::tensor::alloc_zeros_from_pool(device, batch_size)?;
    let input_ptr = input.as_device_ptr_bf16("rms_norm_mean_sq_bf16_head128:input")? as u64;
    let cfg = LaunchConfig {
        grid_dim: (((batch_size as u32) + 15) / 16, 1, 1),
        block_dim: (32, 16, 1),
        shared_mem_bytes: 0,
    };
    launch_kernel!(
        f,
        cfg,
        input_ptr,
        &mean_sq_data,
        batch_size as i32,
        norm_size as i32
    );

    Ok(Tensor {
        storage: TensorStorage::F32 {
            data: mean_sq_data.into(),
            numel: batch_size,
        },
        shape: Shape::from_dims(&[batch_size]),
        device: device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,
        #[cfg(feature = "autograd_v2")]
        autograd_meta: None,
    })
}

impl Tensor {
    /// Functional RMSNorm convenience wrapper.
    pub fn rms_norm(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        rms_norm(self, normalized_shape, weight, eps)
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
        float sq = __fmul_rn(v, v);
        sum_sq = __fadd_rn(sum_sq, sq);
    }

    float mean_sq = __fmul_rn(sum_sq, 1.0f / (float)norm_size);
    float denom = __fadd_rn(mean_sq, eps);
    float inv_rms = __frsqrt_rn(denom);
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

/// Vectorized RMSNorm forward (BF16) — parallel reduction within a block.
///
/// 2026-05-12 perf: replaces the legacy single-thread-per-row scalar loop.
/// Uses PyTorch's `(warp_size, num_threads/warp_size) == (32, 8)` launch,
/// vec_size=4 BF16 loads (8-byte aligned), per-thread F32 partial sum_sq,
/// `shfl_down` intra-warp reduction, and the same shared-memory inter-warp
/// combine tree. Mirrors the design of
/// PyTorch `aten/src/ATen/native/cuda/layer_norm_kernel.cu::vectorized_layer_norm_kernel_impl`
/// stripped to RMSNorm (no mean, no beta).
///
/// Caller MUST verify `norm_size % 4 == 0` before launching (production
/// shapes 2560 and 128 are both multiples of 4); otherwise dispatch to the
/// legacy scalar kernel above.
pub const RMS_NORM_FWD_KERNEL_BF16_VEC: &str = r#"
#include <cuda_bf16.h>

struct __align__(8) bf16x4 {
    __nv_bfloat16 v[4];
};

struct WelfordDataLN {
    float mean;
    float sigma2;
    float count;
};

__device__ __forceinline__ WelfordDataLN welford_online_sum_rms(float val, WelfordDataLN curr) {
    float sq = __fmul_rn(val, val);
    return WelfordDataLN{0.0f, __fadd_rn(curr.sigma2, sq), 0.0f};
}

__device__ __forceinline__ WelfordDataLN welford_combine_rms(WelfordDataLN data_b, WelfordDataLN data_a) {
    return WelfordDataLN{0.0f, __fadd_rn(data_b.sigma2, data_a.sigma2), 0.0f};
}

__device__ __forceinline__ WelfordDataLN compute_rms_stats_pytorch(
    const __nv_bfloat16* __restrict__ row_input,
    int norm_size,
    float* smem
) {
    const int VEC = 4;
    const int n_vec = norm_size / VEC;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const bf16x4* X = reinterpret_cast<const bf16x4*>(row_input);

    WelfordDataLN wd{0.0f, 0.0f, 0.0f};
    for (int i = thrx; i < n_vec; i += numx) {
        bf16x4 data = X[i];
        _Pragma("unroll")
        for (int k = 0; k < VEC; ++k) {
            wd = welford_online_sum_rms(__bfloat162float(data.v[k]), wd);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordDataLN wd_b{
            __shfl_down_sync(0xffffffff, wd.mean, offset),
            __shfl_down_sync(0xffffffff, wd.sigma2, offset),
            __shfl_down_sync(0xffffffff, wd.count, offset)
        };
        wd = welford_combine_rms(wd, wd_b);
    }

    if (blockDim.y > 1) {
        float* meansigmabuf = smem;
        float* countbuf = smem + blockDim.y;
        for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
            if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int wrt_y = threadIdx.y - offset;
                meansigmabuf[2 * wrt_y] = wd.mean;
                meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
                countbuf[wrt_y] = wd.count;
            }
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y < offset) {
                WelfordDataLN wd_b{
                    meansigmabuf[2 * threadIdx.y],
                    meansigmabuf[2 * threadIdx.y + 1],
                    countbuf[threadIdx.y]
                };
                wd = welford_combine_rms(wd, wd_b);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            meansigmabuf[0] = wd.mean;
            meansigmabuf[1] = __fmul_rn(wd.sigma2, 1.0f / (float)norm_size);
        }
        __syncthreads();
        return WelfordDataLN{meansigmabuf[0], meansigmabuf[1], 0.0f};
    }

    return WelfordDataLN{
        __shfl_sync(0xffffffff, wd.mean, 0),
        __fmul_rn(__shfl_sync(0xffffffff, wd.sigma2, 0), 1.0f / (float)norm_size),
        0.0f
    };
}

extern "C" __global__ void rms_norm_forward_bf16_vec(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ inv_rms_out,
    int batch_size,
    int norm_size,
    float eps,
    int has_weight
) {
    const int VEC = 4;
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int n_vec = norm_size / VEC;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int numx = blockDim.x * blockDim.y;

    const __nv_bfloat16* row_input = input + row * norm_size;
    const bf16x4* X = reinterpret_cast<const bf16x4*>(row_input);
    bf16x4* Y = reinterpret_cast<bf16x4*>(output + row * norm_size);
    const bf16x4* W = (has_weight && weight != nullptr)
        ? reinterpret_cast<const bf16x4*>(weight) : nullptr;

    extern __shared__ float smem[];
    WelfordDataLN wd = compute_rms_stats_pytorch(row_input, norm_size, smem);
    const float denom = __fadd_rn(wd.sigma2, eps);
    const float inv_rms = rsqrtf(denom);
    if (thrx == 0) inv_rms_out[row] = inv_rms;

    // Pass 2: vectorized normalize + write.
    for (int i = thrx; i < n_vec; i += numx) {
        bf16x4 d = X[i];
        bf16x4 w_val;
        if (W != nullptr) w_val = W[i];
        bf16x4 out;
        _Pragma("unroll")
        for (int k = 0; k < VEC; ++k) {
            float v = __bfloat162float(d.v[k]) * inv_rms;
            if (W != nullptr) v *= __bfloat162float(w_val.v[k]);
            out.v[k] = __float2bfloat16_rn(v);
        }
        Y[i] = out;
    }
}
"#;

/// PyTorch Python-op RMSNorm forward for hidden_size=4096, no weight.
///
/// HiDream-O1's RMSNorm module is not native layer_norm; it executes
/// `x.float().pow(2).mean(-1)`, `torch.rsqrt`, then casts the unit-normalized
/// result back to BF16 before the caller multiplies by weight. PyTorch's F32
/// mean kernel uses one warp per row with vec4 loads for this shape. The
/// layer-norm-style 8-warp reduction above is faster but differs by a few ULPs
/// on high-magnitude rows, enough to flip BF16 ties.
pub const RMS_NORM_FWD_KERNEL_BF16_HIDDEN4096_PYTORCH: &str = r#"
#include <cuda_bf16.h>

struct __align__(8) f32x4 {
    float v[4];
};

extern "C" __global__ void rms_norm_forward_bf16_hidden4096_pytorch(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ inv_rms_out,
    int batch_size,
    int norm_size,
    float eps,
    int has_weight
) {
    if (norm_size != 4096 || has_weight != 0 || weight != nullptr) return;

    const int lane = threadIdx.x;
    const int warp_row = threadIdx.y;
    const int row = blockIdx.x * blockDim.y + warp_row;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_input = input + (long long)row * 4096;
    __nv_bfloat16* row_output = output + (long long)row * 4096;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // Mirrors PyTorch's generic F32 mean reduction for a contiguous
    // [rows,4096] tensor after the separate pow(2) kernel: vt0=4,
    // input_vec_size=4, block=(32,16), one warp per row.
    for (int idx = lane; idx < 1024; idx += 32) {
        const int base = idx * 4;
        float v0 = __bfloat162float(row_input[base + 0]);
        float v1 = __bfloat162float(row_input[base + 1]);
        float v2 = __bfloat162float(row_input[base + 2]);
        float v3 = __bfloat162float(row_input[base + 3]);
        acc0 = acc0 + (v0 * v0);
        acc1 = acc1 + (v1 * v1);
        acc2 = acc2 + (v2 * v2);
        acc3 = acc3 + (v3 * v3);
    }

    float sum = acc0;
    sum = sum + acc1;
    sum = sum + acc2;
    sum = sum + acc3;

    for (int offset = 1; offset < 32; offset <<= 1) {
        sum = sum + __shfl_down_sync(0xffffffff, sum, offset);
    }

    const float mean_sq = __shfl_sync(0xffffffff, sum, 0) * (1.0f / 4096.0f);
    const float inv_rms = rsqrtf(mean_sq + eps);
    if (lane == 0) inv_rms_out[row] = inv_rms;

    for (int idx = lane; idx < 4096; idx += 32) {
        float v = __bfloat162float(row_input[idx]) * inv_rms;
        row_output[idx] = __float2bfloat16_rn(v);
    }
}
"#;

/// PyTorch generic-mean RMSNorm forward for head_dim=128.
///
/// Mirrors PyTorch 2.7 `Reduce.cuh` for `pow(2).mean(-1)` on a contiguous
/// FP32 tensor of width 128:
/// block=(32,16), one warp per output row, four strided values per lane
/// (`lane + {0,32,64,96}`), lane-local four-accumulator combine, then
/// ascending-offset warp reduce. The `dim0 > 128` vectorization gate in
/// PyTorch 2.7 is intentionally strict; width 128 uses this non-vectorized
/// path.
/// This path is both faster for 128-wide per-head RMSNorm and closer to
/// HiDream/Qwen3-VL's Python RMSNorm than the layer-norm-style 8-warp row
/// reduction above.
pub const RMS_NORM_FWD_KERNEL_BF16_HEAD128: &str = r#"
#include <cuda_bf16.h>

struct __align__(8) bf16x4 {
    __nv_bfloat16 v[4];
};

extern "C" __global__ void rms_norm_forward_bf16_head128(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ inv_rms_out,
    int batch_size,
    int norm_size,
    float eps,
    int has_weight
) {
    if (norm_size != 128) return;

    const int lane = threadIdx.x;
    const int warp_row = threadIdx.y;
    const int row = blockIdx.x * blockDim.y + warp_row;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_input = input + (long long)row * 128;
    __nv_bfloat16* row_output = output + (long long)row * 128;
    const bf16x4* X = reinterpret_cast<const bf16x4*>(row_input);
    bf16x4* Y = reinterpret_cast<bf16x4*>(row_output);
    const bf16x4* W = (has_weight && weight != nullptr)
        ? reinterpret_cast<const bf16x4*>(weight) : nullptr;

    float v0 = __bfloat162float(row_input[lane]);
    float v1 = __bfloat162float(row_input[lane + 32]);
    float v2 = __bfloat162float(row_input[lane + 64]);
    float v3 = __bfloat162float(row_input[lane + 96]);
    float acc0 = v0 * v0;
    float acc1 = v1 * v1;
    float acc2 = v2 * v2;
    float acc3 = v3 * v3;

    float sum = acc0 + acc1;
    sum = sum + acc2;
    sum = sum + acc3;

    for (int offset = 1; offset < 32; offset <<= 1) {
        sum = sum + __shfl_down_sync(0xffffffff, sum, offset);
    }

    const float mean_sq = __shfl_sync(0xffffffff, sum, 0) * (1.0f / 128.0f);
    const float denom = mean_sq + eps;
    const float inv_rms = rsqrtf(denom);
    if (lane == 0) inv_rms_out[row] = inv_rms;

    bf16x4 out;
    bf16x4 w_val;
    bf16x4 data = X[lane];
    if (W != nullptr) w_val = W[lane];
    _Pragma("unroll")
    for (int k = 0; k < 4; ++k) {
        float v = __bfloat162float(data.v[k]) * inv_rms;
        if (W != nullptr) {
            v = v * __bfloat162float(w_val.v[k]);
        }
        out.v[k] = __float2bfloat16_rn(v);
    }
    Y[lane] = out;
}
"#;

pub const RMS_NORM_MEAN_SQ_KERNEL_BF16_HEAD128: &str = r#"
#include <cuda_bf16.h>

struct __align__(8) bf16x4 {
    __nv_bfloat16 v[4];
};

extern "C" __global__ void rms_norm_mean_sq_bf16_head128(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ mean_sq_out,
    int batch_size,
    int norm_size
) {
    if (norm_size != 128) return;

    const int lane = threadIdx.x;
    const int warp_row = threadIdx.y;
    const int row = blockIdx.x * blockDim.y + warp_row;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_input = input + (long long)row * 128;
    float v0 = __bfloat162float(row_input[lane]);
    float v1 = __bfloat162float(row_input[lane + 32]);
    float v2 = __bfloat162float(row_input[lane + 64]);
    float v3 = __bfloat162float(row_input[lane + 96]);
    float sum = (v0 * v0) + (v1 * v1);
    sum = sum + (v2 * v2);
    sum = sum + (v3 * v3);

    for (int offset = 1; offset < 32; offset <<= 1) {
        sum = sum + __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        mean_sq_out[row] = __shfl_sync(0xffffffff, sum, 0) * (1.0f / 128.0f);
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

/// Vectorized RMSNorm backward (BF16) — parallel reduction within a block.
///
/// 2026-05-12 perf: replaces the legacy single-thread-per-row scalar loop.
/// Same parallel pattern as the vec forward (256 threads/block, vec_size=4,
/// warp-shuffle + smem inter-warp reduction for the `dot` partial). The
/// `grad_weight` atomicAdds remain — different blocks (rows) update the same
/// element, so cross-block accumulation requires either atomics or a
/// separate cross-row reduction kernel. Within a block, threads stride by
/// `n_threads` so two threads in the same block never touch the same
/// `grad_weight[idx]` (no intra-block atomic contention).
///
/// Caller MUST verify `norm_size % 4 == 0` before launching.
pub const RMS_NORM_BWD_KERNEL_BF16_VEC: &str = r#"
#include <cuda_bf16.h>

struct __align__(8) bf16x4 {
    __nv_bfloat16 v[4];
};

extern "C" __global__ void rms_norm_backward_bf16_vec(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const float* __restrict__ inv_rms,
    int batch_size,
    int norm_size,
    int has_weight
) {
    const int VEC = 4;
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int n_vec = norm_size / VEC;
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    const bf16x4* GO = reinterpret_cast<const bf16x4*>(grad_out + row * norm_size);
    const bf16x4* X  = reinterpret_cast<const bf16x4*>(input    + row * norm_size);
    bf16x4*       GI = reinterpret_cast<bf16x4*>(grad_input + row * norm_size);
    const bf16x4* W  = (has_weight && weight != nullptr)
        ? reinterpret_cast<const bf16x4*>(weight) : nullptr;

    const float inv = inv_rms[row];
    const float inv_cubed = inv * inv * inv;

    // Pass 1: per-thread partial dot = sum_i (go[i] * w[i] * x[i]).
    float dot = 0.0f;
    for (int i = tid; i < n_vec; i += n_threads) {
        bf16x4 g = GO[i];
        bf16x4 d = X[i];
        bf16x4 wv;
        if (W != nullptr) wv = W[i];
        _Pragma("unroll")
        for (int k = 0; k < VEC; ++k) {
            float go_v = __bfloat162float(g.v[k]);
            float x_v  = __bfloat162float(d.v[k]);
            float w_v  = (W != nullptr) ? __bfloat162float(wv.v[k]) : 1.0f;
            dot += (go_v * w_v) * x_v;
        }
    }

    // Intra-warp reduction (no smem).
    for (int off = 16; off > 0; off >>= 1) {
        dot += __shfl_xor_sync(0xffffffff, dot, off);
    }

    // Inter-warp reduction via shared memory.
    extern __shared__ float smem[];  // sized to (n_warps) floats from launch
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = (n_threads + 31) >> 5;

    if (lane == 0) smem[warp_id] = dot;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < n_warps) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, off);
        }
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();

    const float total_dot = smem[0];
    const float coeff = inv_cubed * total_dot / (float)norm_size;

    // Pass 2: vectorized grad_input write only. The `grad_weight` accumulation
    // is moved to a separate cross-row reduction kernel (`rms_norm_grad_weight_bf16_vec`)
    // — accumulating it here would require batch_size × norm_size atomicAdds
    // (10M+ on production shapes), brutally cross-block-serialized. The
    // separate kernel reduces atomic volume by ~500× by tiling rows per block
    // and emitting one atomicAdd per column per row-tile.
    for (int i = tid; i < n_vec; i += n_threads) {
        bf16x4 g = GO[i];
        bf16x4 d = X[i];
        bf16x4 wv;
        if (W != nullptr) wv = W[i];
        bf16x4 out;
        _Pragma("unroll")
        for (int k = 0; k < VEC; ++k) {
            float go_v = __bfloat162float(g.v[k]);
            float x_v  = __bfloat162float(d.v[k]);
            float w_v  = (W != nullptr) ? __bfloat162float(wv.v[k]) : 1.0f;
            float scaled = go_v * w_v;
            float grad_x = inv * scaled - x_v * coeff;
            out.v[k] = __float2bfloat16_rn(grad_x);
        }
        GI[i] = out;
    }
}
"#;

/// Cross-row reduction kernel for `grad_weight` (BF16 RMSNorm backward).
///
/// 2026-05-12 perf: separates `grad_weight` accumulation from the per-row
/// backward (`rms_norm_backward_bf16_vec`). Mathematics:
///   grad_weight[j] = sum_r (grad_out[r,j] * input[r,j] * inv_rms[r])
///
/// Tiling: each block handles `COLS_PER_BLOCK=64` consecutive columns and a
/// `ROWS_PER_BLOCK=512` row chunk. Each thread owns ONE column and loops
/// over the chunk's rows accumulating in F32. At end, one atomicAdd per
/// column per row-chunk (vs 1 atomicAdd per element per row in the legacy
/// inline path — ~500× fewer atomic ops on production shapes).
///
/// Coalescing: consecutive threadIdx.x access consecutive `[*, col]`
/// addresses across rows; row stride = `norm_size`, so within one row the
/// 64 threads of a block read 64 consecutive BF16 = 128 bytes (one cache
/// line). Reads are fully coalesced.
pub const RMS_NORM_GRAD_WEIGHT_KERNEL_BF16: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void rms_norm_grad_weight_bf16_vec(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ input,
    const float*         __restrict__ inv_rms,
    float*               __restrict__ grad_weight,
    int batch_size,
    int norm_size
) {
    const int COLS_PER_BLOCK = 64;
    const int ROWS_PER_BLOCK = 512;

    const int col = blockIdx.x * COLS_PER_BLOCK + threadIdx.x;
    if (col >= norm_size) return;

    const int row_start = blockIdx.y * ROWS_PER_BLOCK;
    int row_end = row_start + ROWS_PER_BLOCK;
    if (row_end > batch_size) row_end = batch_size;

    float acc = 0.0f;
    for (int r = row_start; r < row_end; ++r) {
        const int idx = r * norm_size + col;
        float go  = __bfloat162float(grad_out[idx]);
        float x   = __bfloat162float(input[idx]);
        float inv = inv_rms[r];
        acc += go * x * inv;
    }
    atomicAdd(&grad_weight[col], acc);
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
