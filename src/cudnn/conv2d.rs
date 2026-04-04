// cuDNN Conv2D — native BF16 on Ampere+ (no FP32 conversion)
//
// Uses NCHW format with BF16 tensors and FP32 compute (tensor cores).
// cuDNN 9.x handles all workspace, algorithm selection, and format conversion internally.

use crate::cudnn::{
    algorithms::AlgorithmSelector,
    descriptors::{
        ConvolutionDescriptor, FilterDescriptor, TensorDescriptor,
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
    },
    handle::get_cudnn_handle,
};
use crate::{DType, Error, Result, Shape, Tensor};
use std::os::raw::{c_int, c_void};

#[link(name = "cudnn")]
extern "C" {
    fn cudnnConvolutionForward(
        handle: *mut c_void,
        alpha: *const c_void,
        x_desc: *mut c_void,
        x: *const c_void,
        w_desc: *mut c_void,
        w: *const c_void,
        conv_desc: *mut c_void,
        algo: c_int,
        workspace: *mut c_void,
        workspace_size: usize,
        beta: *const c_void,
        y_desc: *mut c_void,
        y: *mut c_void,
    ) -> c_int;

    fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: *mut c_void,
        x_desc: *mut c_void,
        w_desc: *mut c_void,
        conv_desc: *mut c_void,
        y_desc: *mut c_void,
        algo: c_int,
        size: *mut usize,
    ) -> c_int;

    fn cudnnAddTensor(
        handle: *mut c_void,
        alpha: *const c_void,
        bias_desc: *mut c_void,
        bias_data: *const c_void,
        beta: *const c_void,
        y_desc: *mut c_void,
        y: *mut c_void,
    ) -> c_int;
}

/// Perform Conv2D using cuDNN with native BF16 tensors.
///
/// Input: NCHW BF16, Weight: OIHW (out_ch, in_ch/groups, kH, kW) BF16
/// Output: NCHW BF16
///
/// Uses FP32 compute with tensor core math (ALLOW_CONVERSION) for accuracy.
pub fn cudnn_conv2d_bf16(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> Result<Tensor> {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    if input_shape.dims().len() != 4 {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv2d_bf16: input must be 4D NCHW, got {:?}",
            input_shape
        )));
    }
    if weight_shape.dims().len() != 4 {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv2d_bf16: weight must be 4D OIHW, got {:?}",
            weight_shape
        )));
    }

    let batch_size = input_shape.dims()[0];
    let in_channels = input_shape.dims()[1];
    let in_height = input_shape.dims()[2];
    let in_width = input_shape.dims()[3];

    let out_channels = weight_shape.dims()[0];
    let kernel_channels = weight_shape.dims()[1];
    let kernel_h = weight_shape.dims()[2];
    let kernel_w = weight_shape.dims()[3];

    if in_channels / groups != kernel_channels {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv2d_bf16: in_channels/groups ({}/{}) != kernel_channels ({})",
            in_channels, groups, kernel_channels
        )));
    }

    let out_height = (in_height + 2 * padding.0 - kernel_h) / stride.0 + 1;
    let out_width = (in_width + 2 * padding.1 - kernel_w) / stride.1 + 1;

    // Get cuDNN handle
    let handle = get_cudnn_handle()?;
    let handle_guard = handle
        .lock()
        .map_err(|_| Error::Training("cudnn handle mutex poisoned".into()))?;

    // All descriptors use BF16
    let data_type = DType::BF16;
    // Compute in FP32 for accuracy
    let compute_type = DType::F32;

    // Input descriptor: NCHW BF16
    let x_desc = TensorDescriptor::new(input_shape.dims(), data_type)?;

    // Filter descriptor: OIHW BF16
    let w_desc = FilterDescriptor::new()?;
    w_desc.set_4d(data_type, out_channels, kernel_channels, kernel_h, kernel_w)?;

    // Output descriptor: NCHW BF16
    let y_desc = TensorDescriptor::new(
        &[batch_size, out_channels, out_height, out_width],
        data_type,
    )?;

    // Convolution descriptor: FP32 compute, with tensor core math
    let conv_desc = ConvolutionDescriptor::new()?;
    conv_desc.set_2d_asymmetric(padding, stride, (1, 1), compute_type)?;
    conv_desc.set_math_type(CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)?;
    if groups > 1 {
        conv_desc.set_group_count(groups)?;
    }

    // Algorithm selection
    let mut algo = AlgorithmSelector::select_forward_algorithm(
        kernel_h, kernel_w, batch_size, in_channels, in_height, in_width,
    );

    // Get workspace size
    let mut workspace_size: usize = 0;
    let mut status = unsafe {
        cudnnGetConvolutionForwardWorkspaceSize(
            handle_guard.as_ptr(),
            x_desc.as_ptr(),
            w_desc.as_ptr(),
            conv_desc.as_ptr(),
            y_desc.as_ptr(),
            algo,
            &mut workspace_size,
        )
    };

    if status != 0 {
        // Fallback algorithm
        algo = AlgorithmSelector::get_fallback_algorithm(algo);
        status = unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                handle_guard.as_ptr(),
                x_desc.as_ptr(),
                w_desc.as_ptr(),
                conv_desc.as_ptr(),
                y_desc.as_ptr(),
                algo,
                &mut workspace_size,
            )
        };
        if status != 0 {
            return Err(Error::CudaError(format!(
                "cudnn_conv2d_bf16: workspace size query failed: {}",
                status
            )));
        }
    }

    // Allocate output as BF16
    let output_shape = Shape::from_dims(&[batch_size, out_channels, out_height, out_width]);
    let mut output = Tensor::zeros_dtype(output_shape, DType::BF16, input.device.clone())?;

    // Allocate workspace if needed
    let mut workspace_alloc: Option<cudarc::driver::CudaSlice<u8>> = None;
    let workspace_ptr = if workspace_size > 0 {
        let alloc = unsafe { input.device.alloc::<u8>(workspace_size) }
            .map_err(|e| Error::Cuda(format!("cudnn workspace alloc: {:?}", e)))?;
        use cudarc::driver::DevicePtr;
        let ptr = *alloc.device_ptr() as *mut c_void;
        workspace_alloc = Some(alloc);
        ptr
    } else {
        std::ptr::null_mut()
    };

    // Alpha/beta as FP32 (cuDNN uses compute type for scaling)
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // Get BF16 device pointers
    let x_ptr = input.as_device_ptr_bf16("cudnn_conv2d_bf16:input")? as *const c_void;
    let w_ptr = weight.as_device_ptr_bf16("cudnn_conv2d_bf16:weight")? as *const c_void;
    let y_ptr = output.as_mut_device_ptr_bf16("cudnn_conv2d_bf16:output")? as *mut c_void;

    // Run convolution
    status = unsafe {
        cudnnConvolutionForward(
            handle_guard.as_ptr(),
            &alpha as *const f32 as *const c_void,
            x_desc.as_ptr(),
            x_ptr,
            w_desc.as_ptr(),
            w_ptr,
            conv_desc.as_ptr(),
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const f32 as *const c_void,
            y_desc.as_ptr(),
            y_ptr,
        )
    };

    if status != 0 {
        // Try IMPLICIT_GEMM as last resort
        algo = crate::cudnn::algorithms::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        status = unsafe {
            cudnnConvolutionForward(
                handle_guard.as_ptr(),
                &alpha as *const f32 as *const c_void,
                x_desc.as_ptr(),
                x_ptr,
                w_desc.as_ptr(),
                w_ptr,
                conv_desc.as_ptr(),
                algo,
                workspace_ptr,
                workspace_size,
                &beta as *const f32 as *const c_void,
                y_desc.as_ptr(),
                y_ptr,
            )
        };
        if status != 0 {
            return Err(Error::CudaError(format!(
                "cudnn_conv2d_bf16: all algorithms failed, status={}",
                status
            )));
        }
    }

    // Add bias if provided: bias is [out_channels], broadcast to [1, out_channels, 1, 1]
    if let Some(bias) = bias {
        let bias_desc = TensorDescriptor::new(&[1, out_channels, 1, 1], data_type)?;
        let alpha_bias: f32 = 1.0;
        let beta_bias: f32 = 1.0; // ADD to existing output

        let bias_ptr = bias.as_device_ptr_bf16("cudnn_conv2d_bf16:bias")? as *const c_void;

        let bias_status = unsafe {
            cudnnAddTensor(
                handle_guard.as_ptr(),
                &alpha_bias as *const f32 as *const c_void,
                bias_desc.as_ptr(),
                bias_ptr,
                &beta_bias as *const f32 as *const c_void,
                y_desc.as_ptr(),
                y_ptr,
            )
        };
        if bias_status != 0 {
            log::warn!("cudnn_conv2d_bf16: bias addition failed (status {}), continuing without", bias_status);
        }
    }

    // workspace_alloc dropped here — async-freed on stream, safe
    drop(workspace_alloc);

    Ok(output)
}
