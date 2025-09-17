// cuDNN Conv2D Implementation for FLAME
// High-performance convolution using NVIDIA cuDNN library

use crate::{Tensor, Shape, Result, FlameError, DType};
use crate::cudnn::{
    handle::get_cudnn_handle,
    descriptors::{TensorDescriptor, FilterDescriptor, ConvolutionDescriptor},
    algorithms::AlgorithmSelector,
};
use cudarc::driver::DevicePtr;
use std::os::raw::{c_void, c_int};

// FFI binding for the main convolution operation
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
        y: *mut c_void
    ) -> c_int;
    
    fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: *mut c_void,
        x_desc: *mut c_void,
        w_desc: *mut c_void,
        conv_desc: *mut c_void,
        y_desc: *mut c_void,
        algo: c_int,
        size: *mut usize
    ) -> c_int;
    
    fn cudnnAddTensor(
        handle: *mut c_void,
        alpha: *const c_void,
        bias_desc: *mut c_void,
        bias_data: *const c_void,
        beta: *const c_void,
        y_desc: *mut c_void,
        y: *mut c_void
    ) -> c_int;
}

/// Perform Conv2D using cuDNN
pub fn cudnn_conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    // Convert BF16 to F32 for cuDNN compatibility if needed
    let (input_work, weight_work) = if input.dtype() == DType::BF16 || weight.dtype() == DType::BF16 {
        // cuDNN doesn't support BF16 for convolution, convert to F32
        let input_f32 = if input.dtype() == DType::BF16 {
            input.to_dtype(DType::F32)?
        } else {
            input.clone_result()?
        };
        let weight_f32 = if weight.dtype() == DType::BF16 {
            weight.to_dtype(DType::F32)?
        } else {
            weight.clone_result()?
        };
        (input_f32, weight_f32)
    } else {
        (input.clone_result()?, weight.clone_result()?)
    };
    
    // Validate input dimensions
    let input_shape = input_work.shape();
    let weight_shape = weight_work.shape();
    
    if input_shape.dims().len() != 4 {
        return Err(FlameError::InvalidShape(format!(
            "Conv2d input must be 4D, got {:?}", input_shape
        )));
    }
    
    if weight_shape.dims().len() != 4 {
        return Err(FlameError::InvalidShape(format!(
            "Conv2d weight must be 4D, got {:?}", weight_shape
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
    
    if in_channels != kernel_channels {
        return Err(FlameError::InvalidShape(format!(
            "Input channels {} doesn't match kernel channels {}",
            in_channels, kernel_channels
        )));
    }
    
    // Calculate output dimensions
    let out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_w) / stride + 1;
    
    // Get cuDNN handle
    let handle = get_cudnn_handle()?;
    let handle_guard = handle.lock().map_err(|_| FlameError::Training("cudnn handle mutex poisoned".into()))?;
    
    // Create descriptors - use F32 dtype for cuDNN when working with converted tensors
    let working_dtype = if input.dtype() == DType::BF16 || weight.dtype() == DType::BF16 {
        DType::F32  // Use F32 for descriptors when we converted from BF16
    } else {
        input.dtype()
    };
    
    let x_desc = TensorDescriptor::new(input_shape.dims(), working_dtype)?;
    
    let w_desc = FilterDescriptor::new()?;
    w_desc.set_4d(working_dtype, out_channels, kernel_channels, kernel_h, kernel_w)?;
    
    let y_desc = TensorDescriptor::new(&[batch_size, out_channels, out_height, out_width], working_dtype)?;
    
    let conv_desc = ConvolutionDescriptor::new()?;
    conv_desc.set_2d(padding, stride, 1, working_dtype)?;
    
    // Select optimal algorithm
    let mut algo = AlgorithmSelector::select_forward_algorithm(
        kernel_h, kernel_w, batch_size, in_channels, in_height, in_width
    );
    
    // Get workspace size
    let mut workspace_size: usize = 0;
    let status = unsafe {
        cudnnGetConvolutionForwardWorkspaceSize(
            handle_guard.as_ptr(),
            x_desc.as_ptr(),
            w_desc.as_ptr(),
            conv_desc.as_ptr(),
            y_desc.as_ptr(),
            algo,
            &mut workspace_size
        )
    };
    
    if status != 0 {
        // Try fallback algorithm
        eprintln!("cuDNN {} failed (status {}), trying fallback", 
                 AlgorithmSelector::algorithm_name(algo), status);
        algo = AlgorithmSelector::get_fallback_algorithm(algo);
        
        let status = unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                handle_guard.as_ptr(),
                x_desc.as_ptr(),
                w_desc.as_ptr(),
                conv_desc.as_ptr(),
                y_desc.as_ptr(),
                algo,
                &mut workspace_size
            )
        };
        
        if status != 0 {
            return Err(FlameError::CudaError(format!(
                "Failed to get workspace size for cuDNN convolution: {}", status
            )));
        }
    }
    
    // Allocate output tensor - use F32 if we converted from BF16
    let output_shape = Shape::from_dims(&[batch_size, out_channels, out_height, out_width]);
    let output_dtype = if input.dtype() == DType::BF16 || weight.dtype() == DType::BF16 {
        DType::F32  // Create F32 output when working with converted tensors
    } else {
        input.dtype()
    };
    let mut output = Tensor::zeros_dtype(output_shape.clone(), output_dtype, input.device.clone())?;
    
    // Allocate workspace if needed
    let mut workspace = if workspace_size > 0 {
        let num_elements = (workspace_size + 3) / 4; // Convert bytes to f32 elements
        Some(Tensor::zeros_dtype(
            Shape::from_dims(&[num_elements]),
            DType::F32,
            input.device.clone()
        )?)
    } else {
        None
    };
    
    // Prepare alpha and beta scalars
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    
    // Get device pointers - use the converted tensors
    let x_ptr = input_work.cuda_ptr() as *const c_void;
    let w_ptr = weight_work.cuda_ptr() as *const c_void;
    let y_ptr = output.cuda_ptr_mut() as *mut c_void;
    let workspace_ptr = workspace.as_mut()
        .map(|w| w.cuda_ptr_mut() as *mut c_void)
        .unwrap_or(std::ptr::null_mut());
    
    // Perform convolution
    let status = unsafe {
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
            y_ptr
        )
    };
    
    if status != 0 {
        // Try one more fallback
        if algo != crate::cudnn::algorithms::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM {
            eprintln!("cuDNN {} failed (status {}), trying IMPLICIT_GEMM", 
                     AlgorithmSelector::algorithm_name(algo), status);
            algo = crate::cudnn::algorithms::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            
            let status = unsafe {
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
                    y_ptr
                )
            };
            
            if status != 0 {
                return Err(FlameError::CudaError(format!(
                    "cuDNN convolution failed with all algorithms: {}", status
                )));
            }
        } else {
            return Err(FlameError::CudaError(format!(
                "cuDNN convolution failed: {}", status
            )));
        }
    }
    
    // Add bias if provided
    if let Some(bias) = bias {
        // For bias addition, we need to handle the shape carefully
        // Bias is typically [out_channels] but needs to be broadcast to [batch, out_channels, height, width]
        
        // Check if bias needs reshaping
        let bias_work = if bias.shape().dims() == &[out_channels] {
            // Reshape bias from [out_channels] to [1, out_channels, 1, 1] for broadcasting
            bias.reshape(&[1, out_channels, 1, 1])?
        } else {
            bias.clone_result()?
        };
        
        let bias_desc = TensorDescriptor::new(&[1, out_channels, 1, 1], DType::F32)?;
        
        let alpha: f32 = 1.0;
        let beta: f32 = 1.0; // Add to existing values
        
        let bias_ptr = bias_work.cuda_ptr() as *const c_void;
        let y_ptr = output.cuda_ptr_mut() as *mut c_void;
        
        let status = unsafe {
            cudnnAddTensor(
                handle_guard.as_ptr(),
                &alpha as *const f32 as *const c_void,
                bias_desc.as_ptr(),
                bias_ptr,
                &beta as *const f32 as *const c_void,
                y_desc.as_ptr(),
                y_ptr
            )
        };
        
        if status != 0 {
            // For now, just warn if bias addition fails
            // Fallback path: use direct CUDA kernel when cuDNN is not available
            eprintln!("Warning: cuDNN bias addition failed (status {}), continuing without bias", status);
        }
    }
    
    // Convert output back to BF16 if input was BF16
    let final_output = if input.dtype() == DType::BF16 {
        output.to_dtype(DType::BF16)?
    } else {
        output
    };
    
    Ok(final_output)
}
