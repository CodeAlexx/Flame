//! CUDA kernels for 2D convolution operations

use crate::cuda_memory_alignment::alloc_aligned_f32;
use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

// Import kernel source from dedicated module
use crate::cuda_conv2d_kernels::CONV2D_KERNELS;

// Helper to copy i32 array to GPU as f32
fn copy_i32_to_gpu(device: &Arc<CudaDevice>, data: &[i32]) -> Result<CudaSlice<f32>> {
    let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let mut gpu_data =
        unsafe { device.alloc::<f32>(f32_data.len()) }.map_err(|_| Error::CudaDriver)?;
    device
        .htod_copy_into(f32_data, &mut gpu_data)
        .map_err(|_| Error::CudaDriver)?;
    Ok(gpu_data)
}

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        let result = unsafe { $func.launch($cfg, ($($args,)*)) };
        result.map_err(|e| crate::Error::Cuda(format!("Kernel launch failed: {:?}", e)))
    }};
}

/// GPU-accelerated 2D convolution
pub struct CudaConv2d;

impl CudaConv2d {
    /// Ensure kernels are loaded
    fn ensure_kernels(device: &Arc<CudaDevice>) -> Result<()> {
        // Set CUDA_HOME if not already set
        if std::env::var("CUDA_HOME").is_err() {
            std::env::set_var("CUDA_HOME", "/usr/local/cuda-12.4");
        }

        // Compile CUDA kernels first
        let ptx = cudarc::nvrtc::compile_ptx(CONV2D_KERNELS)
            .map_err(|e| Error::Cuda(format!("Failed to compile Conv2D kernels: {:?}", e)))?;

        // Synchronize after compilation to prevent race conditions
        device
            .synchronize()
            .map_err(|_| Error::Cuda("Failed to synchronize after kernel compilation".into()))?;

        device
            .load_ptx(
                ptx,
                "conv2d_ops",
                &[
                    "im2col_kernel_simple",
                    "im2col_kernel",
                    "im2col_kernel_v2",
                    "col2im_kernel_simple",
                    "col2im_kernel",
                    "col2im_kernel_v2",
                    "add_bias_nhwc_kernel",
                    "add_bias_nchw_kernel",
                    "bias_grad_kernel",
                    "check_conv_dimensions_kernel",
                    "im2col_optimized_kernel",
                ],
            )
            .map_err(|e| Error::Cuda(format!("Failed to load Conv2D kernels: {:?}", e)))?;
        Ok(())
    }

    /// Forward convolution routed through the shared ops backend.
    pub fn conv2d_forward(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Tensor> {
        if groups != 1 {
            return Err(Error::InvalidOperation(
                "Grouped convolution not yet implemented".into(),
            ));
        }
        if input.dtype() != DType::F32 || weight.dtype() != DType::F32 {
            return Err(Error::InvalidInput(
                "conv2d_forward expects F32 tensors".into(),
            ));
        }

        let device = input.device();
        Self::ensure_kernels(device)?;

        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();

        if input_dims.len() != 4 || weight_dims.len() != 4 {
            return Err(Error::InvalidInput(
                "conv2d expects 4D input and weight".into(),
            ));
        }

        let batch = input_dims[0];
        let in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];

        let out_channels = weight_dims[0];
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        let out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

        let col_elems = batch * in_channels * kernel_h * kernel_w * out_height * out_width;
        let col_buffer = alloc_aligned_f32(device, col_elems)?;

        let dims = vec![
            batch as i32,
            in_channels as i32,
            in_height as i32,
            in_width as i32,
            kernel_h as i32,
            kernel_w as i32,
        ];
        let conv_params = vec![
            pad_h as i32,
            pad_w as i32,
            stride_h as i32,
            stride_w as i32,
            out_height as i32,
            out_width as i32,
        ];

        let dims_gpu = copy_i32_to_gpu(device, &dims)?;
        let params_gpu = copy_i32_to_gpu(device, &conv_params)?;

        let func = device
            .get_func("conv2d_ops", "im2col_kernel_v2")
            .ok_or_else(|| Error::Cuda("Failed to get im2col kernel v2".into()))?;

        let cfg = LaunchConfig::for_num_elems(col_elems as u32);
        launch_kernel!(
            func,
            cfg,
            input.storage.try_as_slice_f32()?,
            &col_buffer,
            &dims_gpu,
            &params_gpu
        )?;

        let col_shape = Shape::from_dims(&[
            batch * out_height * out_width,
            in_channels * kernel_h * kernel_w,
        ]);
        let col_tensor = Tensor {
            storage: TensorStorage::F32 {
                data: col_buffer.into(),
                numel: col_shape.elem_count(),
            },
            shape: col_shape,
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };

        let weight_matrix = weight
            .reshape(&[out_channels, in_channels * kernel_h * kernel_w])?
            .transpose()?;

        let y_2d = col_tensor.matmul(&weight_matrix)?;
        let mut y = y_2d
            .reshape(&[batch, out_height, out_width, out_channels])?
            .permute(&[0, 3, 1, 2])?;

        if let Some(b) = bias {
            let bias_view = b.reshape(&[1, out_channels, 1, 1])?;
            y = y.add(&bias_view)?;
        }

        Ok(y)
    }

    /// NHWC adapter: x [N,H,W,C], w [KH,KW,IC,OC] -> y [N,H_out,W_out,OC]
    pub fn conv2d_forward_nhwc(
        input_nhwc: &Tensor,
        weight_khwkicoc: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        // Validate shapes
        let id = input_nhwc.shape().dims();
        let wd = weight_khwkicoc.shape().dims();
        if id.len() != 4 || wd.len() != 4 {
            return Err(Error::InvalidOperation(
                "conv2d_nhwc expects 4D input and 4D weight".into(),
            ));
        }
        // NHWC: [N,H,W,C], Weight: [KH,KW,IC,OC]
        let (_n, _h, _w, c) = (id[0], id[1], id[2], id[3]);
        let (_kh, _kw, ic, _oc) = (wd[0], wd[1], wd[2], wd[3]);
        if ic != c {
            println!(
                "conv2d_forward_nhwc mismatch: input {:?}, weight {:?}",
                id, wd
            );
            return Err(Error::InvalidOperation(
                "weight IC must match input C".into(),
            ));
        }

        assert_nhwc_bf16_public("cuda_conv2d::forward_nhwc in", input_nhwc)?;
        if weight_khwkicoc.dtype() != DType::BF16 {
            return Err(Error::InvalidInput(
                "cuda_conv2d::conv2d_forward_nhwc expects BF16 weights".into(),
            ));
        }
        if let Some(b) = bias {
            if b.dtype() != DType::BF16 {
                return Err(Error::InvalidInput(
                    "cuda_conv2d::conv2d_forward_nhwc expects BF16 bias".into(),
                ));
            }
        }

        // Convert to kernel layouts
        let mut x_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input_nhwc)?;
        if x_nchw.dtype() != DType::F32 {
            x_nchw = x_nchw.to_dtype(DType::F32)?;
        }
        let weight_f32 = weight_khwkicoc.to_dtype(DType::F32)?;
        let mut w_ocic = crate::cuda_ops::GpuOps::weight_khwkicoc_to_ocickhkw(&weight_f32)?;
        let bias_f32 = if let Some(b) = bias {
            Some(b.to_dtype(DType::F32)?)
        } else {
            None
        };
        // Call NCHW forward through shared ops backend (groups=1)
        let y_nchw = crate::ops::conv2d::conv2d_forward(
            &x_nchw,
            &w_ocic,
            bias_f32.as_ref(),
            (stride.0, stride.1),
            (padding.0, padding.1),
            1,
        )?;
        // Convert back to NHWC
        let mut y_nhwc = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&y_nchw)?;
        if y_nhwc.dtype() != DType::BF16 {
            y_nhwc = y_nhwc.to_dtype(DType::BF16)?;
        }

        // Record NHWC op for autograd so backward converts appropriately
        if input_nhwc.requires_grad
            || weight_khwkicoc.requires_grad
            || bias.map(|b| b.requires_grad).unwrap_or(false)
        {
            let mut out = y_nhwc.clone_result()?;
            out.requires_grad = true;
            if crate::autograd::AutogradContext::is_recording() {
                crate::autograd::AutogradContext::record_op(
                    out.id(),
                    crate::autograd::Op::Conv2dNHWC {
                        input: input_nhwc.id(),
                        weight: weight_khwkicoc.id(),
                        stride: stride.0,
                        padding: padding.0,
                    },
                    vec![(input_nhwc.id(), x_nchw), (weight_khwkicoc.id(), w_ocic)],
                );
            }
            return Ok(out);
        }
        assert_nhwc_bf16_public("cuda_conv2d::forward_nhwc out", &y_nhwc)?;
        Ok(y_nhwc)
    }

    /// Add bias using CUDA kernel
    #[allow(dead_code, unused_mut)]
    fn add_bias(output: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let device = output.device();
        let output_dims = output.shape().dims();

        let batch_size = output_dims[0];
        let channels = output_dims[1];
        let spatial_size = output_dims[2] * output_dims[3];

        let mut result = output.clone_result()?;

        let f = device
            .get_func("conv2d_ops", "add_bias_nchw_kernel")
            .ok_or_else(|| Error::Cuda("Failed to get add_bias kernel".into()))?;

        let cfg = LaunchConfig::for_num_elems((batch_size * channels * spatial_size) as u32);

        launch_kernel!(
            f,
            cfg,
            result.storage.try_as_slice_f32()?,
            bias.storage.try_as_slice_f32()?,
            batch_size as i32,
            channels as i32,
            spatial_size as i32
        )?;

        device.synchronize()?;
        Ok(result)
    }

    /// Backward pass for convolution (used in autograd)
    pub fn conv2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let device = input.device();
        Self::ensure_kernels(device)?;

        // Get dimensions
        let grad_dims = grad_output.shape().dims();
        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();

        let batch_size = grad_dims[0];
        let out_channels = grad_dims[1];
        let out_height = grad_dims[2];
        let out_width = grad_dims[3];

        let in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];

        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        // Gradient w.r.t. input using transposed convolution
        // First, perform im2col on grad_output
        let _col_size = batch_size * out_channels * out_height * out_width;
        let grad_col = grad_output.reshape(&[batch_size * out_height * out_width, out_channels])?;

        // Weight gradient: grad_output @ input^T (after im2col)
        // First, im2col on input
        let input_col_size =
            batch_size * in_channels * kernel_h * kernel_w * out_height * out_width;
        let input_col_buffer = crate::tensor::alloc_zeros_from_pool(device, input_col_size)?;

        let cfg = LaunchConfig::for_num_elems(input_col_size as u32);

        if stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1 {
            let f_im2col = device
                .get_func("conv2d_ops", "im2col_kernel_simple")
                .ok_or_else(|| Error::Cuda("Failed to get im2col_kernel_simple".into()))?;

            launch_kernel!(
                f_im2col,
                cfg,
                input.storage.try_as_slice_f32()?,
                &input_col_buffer,
                batch_size as i32,
                in_channels as i32,
                in_height as i32,
                in_width as i32,
                kernel_h as i32,
                kernel_w as i32,
                out_height as i32,
                out_width as i32
            )?;
        } else {
            // Use v2 kernel with arrays to avoid parameter limit
            let dims = vec![
                batch_size as i32,
                in_channels as i32,
                in_height as i32,
                in_width as i32,
                kernel_h as i32,
                kernel_w as i32,
            ];
            let conv_params = vec![
                pad_h as i32,
                pad_w as i32,
                stride_h as i32,
                stride_w as i32,
                out_height as i32,
                out_width as i32,
            ];

            let dims_gpu = copy_i32_to_gpu(device, &dims)?;
            let params_gpu = copy_i32_to_gpu(device, &conv_params)?;

            let f_im2col = device
                .get_func("conv2d_ops", "im2col_kernel_v2")
                .ok_or_else(|| Error::Cuda("Failed to get im2col kernel v2".into()))?;

            launch_kernel!(
                f_im2col,
                cfg,
                input.storage.try_as_slice_f32()?,
                &input_col_buffer,
                &dims_gpu,
                &params_gpu
            )?;
        }

        device.synchronize()?;

        // Compute weight gradient
        let input_col_shape = Shape::from_dims(&[
            batch_size * out_height * out_width,
            in_channels * kernel_h * kernel_w,
        ]);
        let input_col_tensor = Tensor {
            storage: TensorStorage::F32 {
                data: input_col_buffer.into(),
                numel: input_col_shape.elem_count(),
            },
            shape: input_col_shape,
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };

        // grad_weight = grad_output^T @ input_col
        let grad_col_t = grad_col.transpose()?;
        let grad_weight_2d = grad_col_t.matmul(&input_col_tensor)?;
        let grad_weight =
            grad_weight_2d.reshape(&[out_channels, in_channels, kernel_h, kernel_w])?;

        // Gradient w.r.t. input using col2im
        // First compute weight^T @ grad_output
        let weight_t = weight
            .reshape(&[out_channels, in_channels * kernel_h * kernel_w])?
            .transpose()?;
        let grad_input_col = grad_col.matmul(&weight_t)?;

        // Now use col2im to get grad_input
        let grad_input_data =
            crate::tensor::alloc_zeros_from_pool(device, input.shape().elem_count())?;

        let cfg = LaunchConfig::for_num_elems(input.shape().elem_count() as u32);

        {
            // Use v2 kernel with arrays to avoid parameter limit
            let dims = vec![
                batch_size as i32,
                in_channels as i32,
                in_height as i32,
                in_width as i32,
                kernel_h as i32,
                kernel_w as i32,
            ];
            let conv_params = vec![
                pad_h as i32,
                pad_w as i32,
                stride_h as i32,
                stride_w as i32,
                out_height as i32,
                out_width as i32,
            ];

            let dims_gpu = copy_i32_to_gpu(device, &dims)?;
            let params_gpu = copy_i32_to_gpu(device, &conv_params)?;

            let f_col2im = device
                .get_func("conv2d_ops", "col2im_kernel_v2")
                .ok_or_else(|| Error::Cuda("Failed to get col2im kernel v2".into()))?;

            launch_kernel!(
                f_col2im,
                cfg,
                grad_input_col.storage.try_as_slice_f32()?,
                &grad_input_data,
                &dims_gpu,
                &params_gpu
            )?;
        }

        device.synchronize()?;

        let grad_input = Tensor {
            storage: TensorStorage::F32 {
                data: grad_input_data.into(),
                numel: input.shape().elem_count(),
            },
            shape: input.shape().clone(),
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };

        // Bias gradient is sum over batch and spatial dimensions
        let grad_bias = if grad_output.shape().dims().len() == 4 {
            let grad_bias_data = crate::tensor::alloc_zeros_from_pool(device, out_channels)?;

            let f_bias_grad = device
                .get_func("conv2d_ops", "bias_grad_kernel")
                .ok_or_else(|| Error::Cuda("Failed to get bias_grad kernel".into()))?;

            let cfg = LaunchConfig::for_num_elems(out_channels as u32);
            launch_kernel!(
                f_bias_grad,
                cfg,
                grad_output.storage.try_as_slice_f32()?,
                &grad_bias_data,
                batch_size as i32,
                out_channels as i32,
                (out_height * out_width) as i32
            )?;

            device.synchronize()?;

            Some(Tensor {
                storage: TensorStorage::F32 {
                    data: grad_bias_data.into(),
                    numel: out_channels,
                },
                shape: Shape::from_dims(&[out_channels]),
                device: device.clone(),
                id: TensorId::new(),
                requires_grad: false,
            })
        } else {
            None
        };

        Ok((grad_input, grad_weight, grad_bias))
    }
}

/// Convenience function for Conv2D forward
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    crate::ops::conv2d::conv2d_forward(input, weight, bias, (stride, stride), (padding, padding), 1)
}
