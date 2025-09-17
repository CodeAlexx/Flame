//! CUDA kernels for 2D convolution operations

use crate::{Tensor, Shape, Result, FlameError};
use crate::autograd::{AutogradContext, Op};
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, CudaSlice};
use std::sync::Arc;

// Import kernel source from dedicated module
use crate::cuda_conv2d_kernels::CONV2D_KERNELS;

// Helper to copy i32 array to GPU as f32
fn copy_i32_to_gpu(device: &Arc<CudaDevice>, data: &[i32]) -> Result<CudaSlice<f32>> {
    let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let mut gpu_data = unsafe { device.alloc::<f32>(f32_data.len()) }
        .map_err(|_| FlameError::CudaDriver)?;
    device.htod_copy_into(f32_data, &mut gpu_data)
        .map_err(|_| FlameError::CudaDriver)?;
    Ok(gpu_data)
}



// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        let result = unsafe { $func.launch($cfg, ($($args,)*)) };
        result.map_err(|e| crate::FlameError::Cuda(format!("Kernel launch failed: {:?}", e)))
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
            .map_err(|e| FlameError::Cuda(format!("Failed to compile Conv2D kernels: {:?}", e)))?;
        
        // Synchronize after compilation to prevent race conditions
        device.synchronize()
            .map_err(|_| FlameError::Cuda("Failed to synchronize after kernel compilation".into()))?;
        
        device
            .load_ptx(ptx, "conv2d_ops", &[
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
            ])
            .map_err(|e| FlameError::Cuda(format!("Failed to load Conv2D kernels: {}", e)))?;
        Ok(())
    }
    
    /// Forward convolution using im2col + matmul
    pub fn conv2d_forward(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Tensor> {
        // This NCHW kernel path remains available directly.
        if groups != 1 {
            return Err(FlameError::InvalidOperation(
                "Grouped convolution not yet implemented".into()
            ));
        }
        
        let device = input.device();
        Self::ensure_kernels(device)?;
        
        // Check if we have custom kernels available
        let has_custom_kernels = device.get_func("conv2d_ops", "check_conv_dimensions_kernel").is_some();
        
        if !has_custom_kernels {
            return Err(FlameError::Cuda("Conv2D kernels not loaded. Please ensure CUDA is properly installed.".into()));
        }
        
        // Get dimensions
        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();
        
        let batch_size = input_dims[0];
        let in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];
        
        let out_channels = weight_dims[0];
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];
        
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
        
        // Allocate col buffer for im2col
        let col_size = batch_size * in_channels * kernel_h * kernel_w * out_height * out_width;
        
        // Special check for the problematic VAE case
        if col_size == 1207959552 {
            eprintln!("\n!!! FOUND THE PROBLEMATIC CONVOLUTION !!!");
            eprintln!("Input dimensions: {}x{}x{}x{}", batch_size, in_channels, in_height, in_width);
            eprintln!("Kernel: {}x{}", kernel_h, kernel_w);
            eprintln!("Stride: {}x{}", stride_h, stride_w);
            eprintln!("Padding: {}x{}", pad_h, pad_w);
            eprintln!("Output dimensions: {}x{}x{}x{}", batch_size, out_channels, out_height, out_width);
            eprintln!("This is trying to allocate for 1536x1536x512 = {}", 1536*1536*512);
            eprintln!("Actual col_size calculation: {} * {} * {} * {} * {} * {} = {}", 
                     batch_size, in_channels, kernel_h, kernel_w, out_height, out_width, col_size);
            
            // Try to understand what's happening
            if out_height == 1536 && out_width == 1536 {
                eprintln!("ERROR: Output is 1536x1536 but input is {}x{}", in_height, in_width);
                eprintln!("This means the padding formula is wrong!");
                let expected_out_h = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
                let expected_out_w = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
                eprintln!("Expected output size: {}x{}", expected_out_h, expected_out_w);
            }
        }
        
        // Debug large allocations
        if col_size > 100_000_000 {
            eprintln!("WARNING: Large conv2d allocation detected!");
            eprintln!("  Input: {}x{}x{}x{}", batch_size, in_channels, in_height, in_width);
            eprintln!("  Output: {}x{}x{}x{}", batch_size, out_channels, out_height, out_width);
            eprintln!("  Kernel: {}x{}, stride: {}x{}, padding: {}x{}", 
                     kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
            eprintln!("  Col buffer size: {} elements ({:.1} MB)", 
                     col_size, (col_size * 4) as f64 / 1024.0 / 1024.0);
            
            // Check if this is the problematic 1536x1536 case
            if col_size == 1207959552 {
                eprintln!("  ERROR: This is the 1536x1536x512 allocation!");
                eprintln!("  This suggests the VAE is configured for 1536x1536 but receiving 1024x1024");
                eprintln!("  The VAE may be padding the input internally");
            }
        }
        // println!("Allocating col buffer: size={} ({:.2} MB)", col_size, (col_size * 4) as f64 / 1024.0 / 1024.0);
        let col_buffer = crate::tensor::alloc_zeros_from_pool(&device, col_size)?;
        // println!("Col buffer allocated successfully");
        
        // Check dimensions first
        let error_flag = device.alloc_zeros::<i32>(1)?;
        let check_f = device.get_func("conv2d_ops", "check_conv_dimensions_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get check_conv_dimensions kernel".into()))?;
        
        launch_kernel!(check_f, LaunchConfig::for_num_elems(1),
            batch_size as i32,
            in_channels as i32,
            out_channels as i32,
            in_height as i32,
            in_width as i32,
            kernel_h as i32,
            kernel_w as i32,
            stride_h as i32,
            stride_w as i32,
            pad_h as i32,
            pad_w as i32,
            &error_flag
        )?;
        
        // Check error flag
        device.synchronize()?;
        let error_val: Vec<i32> = device.dtoh_sync_copy(&error_flag)?;
        
        match error_val[0] {
            1 => return Err(FlameError::InvalidOperation("Conv2D: Invalid output dimensions".into())),
            2 => return Err(FlameError::InvalidOperation("Conv2D: Kernel larger than padded input".into())),
            _ => {}
        }
        
        // Perform im2col using appropriate kernel
        // println!("Conv2d im2col setup:");
        // println!("  Input shape: {}x{}x{}x{}", batch_size, in_channels, in_height, in_width);
        // println!("  Output shape: {}x{}x{}x{}", batch_size, out_channels, out_height, out_width);
        // println!("  Kernel: {}x{}, stride: {}x{}, padding: {}x{}", 
        //          kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        // println!("  Col buffer size: {}", col_size);
        
        // Check if we can use the simple kernel (stride=1, padding=1)
        if stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1 {
        // println!("Using simple im2col kernel for stride=1, padding=1");
            
            // Launch enough threads to cover all elements
            let threads_per_block = 256u32;
            // Don't artificially limit blocks - we need to process all elements efficiently
            let max_blocks = 65535u32; // CUDA max grid size in x dimension
            let total_threads = col_size as u32;
            let num_blocks = ((total_threads + threads_per_block - 1) / threads_per_block).min(max_blocks);
            
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };
            
        // println!("Launch config: grid=({},1,1), block=({},1,1)", num_blocks, threads_per_block);
        // println!("Total work items: {}, threads launched: {}", col_size, num_blocks * threads_per_block);
            
            if num_blocks == max_blocks {
        // println!("WARNING: Grid size limited. May need multiple kernel launches.");
            }
            
            let f = device.get_func("conv2d_ops", "im2col_kernel_simple")
                .ok_or_else(|| FlameError::Cuda("Failed to get im2col_kernel_simple".into()))?;
            
        // println!("Launching im2col kernel...");
            
            // Launch with explicit synchronization
            unsafe {
                f.launch(cfg, (
                    input.storage.try_as_slice_f32()?,
                    &col_buffer,
                    batch_size as i32,
                    in_channels as i32,
                    in_height as i32,
                    in_width as i32,
                    kernel_h as i32,
                    kernel_w as i32,
                    out_height as i32,
                    out_width as i32
                )).map_err(|e| FlameError::Cuda(format!("Kernel launch failed: {:?}", e)))?;
            }
            
        // println!("Kernel launched, synchronizing...");
            
            // Explicit synchronization
            device.synchronize()
                .map_err(|e| FlameError::Cuda(format!("Synchronization failed: {:?}", e)))?;
            
        // println!("Synchronization complete!");
        } else {
        // println!("Using full im2col kernel with stride={}, padding={}", stride_h, pad_h);
            
            // Launch enough threads to cover all elements
            let threads_per_block = 256u32;
            let max_blocks = 65535u32; // CUDA max grid size in x dimension
            let total_threads = col_size as u32;
            let num_blocks = ((total_threads + threads_per_block - 1) / threads_per_block).min(max_blocks);
            
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };
            
        // println!("Launch config: grid=({},1,1), block=({},1,1)", num_blocks, threads_per_block);
        // println!("Total work items: {}, threads launched: {}", col_size, num_blocks * threads_per_block);
            
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
            
            let dims_gpu = copy_i32_to_gpu(&device, &dims)?;
            let params_gpu = copy_i32_to_gpu(&device, &conv_params)?;
            
            let f = device.get_func("conv2d_ops", "im2col_kernel_v2")
                .ok_or_else(|| FlameError::Cuda("Failed to get im2col kernel v2".into()))?;
            
            launch_kernel!(f, cfg,
                input.storage.try_as_slice_f32()?,
                &col_buffer,
                &dims_gpu,
                &params_gpu
            )?;
        }
        
        // Synchronize after im2col
        device.synchronize()
            .map_err(|_| FlameError::Cuda("Failed to synchronize after im2col kernel".into()))?;
        
        // Reshape for matrix multiplication
        let col_shape = Shape::from_dims(&[batch_size * out_height * out_width, in_channels * kernel_h * kernel_w]);
        let col_tensor = Tensor {
            storage: TensorStorage::F32 { data: col_buffer, numel: col_shape.elem_count() },
            shape: col_shape,
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        // Reshape weight: [out_channels, in_channels, kh, kw] -> [out_channels, in_channels * kh * kw]
        let weight_2d = weight.reshape(&[out_channels, in_channels * kernel_h * kernel_w])?;
        
        // Perform matrix multiplication: [out_channels, in_c*kh*kw] @ [b*oh*ow, in_c*kh*kw]^T
        let weight_t = weight_2d.transpose()?;
        let output_2d = col_tensor.matmul(&weight_t)?;
        
        // Reshape output: [b*oh*ow, out_channels] -> [b, out_channels, oh, ow]
        let mut output = output_2d.reshape(&[batch_size, out_height, out_width, out_channels])?
            .permute(&[0, 3, 1, 2])?;
        
        // Add bias if provided
        if let Some(b) = bias {
            output = Self::add_bias(&output, b)?;
        }
        
        // Record operation for autograd
        if input.requires_grad || weight.requires_grad || bias.map(|b| b.requires_grad).unwrap_or(false) {
            output.requires_grad = true;
            
            let mut saved_tensors = vec![
                (input.id, input.clone_result()?),
                (weight.id, weight.clone_result()?),
            ];
            
            let _bias_id = if let Some(b) = bias {
                saved_tensors.push((b.id, b.clone_result()?));
                Some(b.id)
            } else {
                None
            };
            
            AutogradContext::record_op(
                output.id,
                Op::Conv2d {
                    input: input.id,
                    weight: weight.id,
                    stride: stride_h,
                    padding: pad_h,
                },
                saved_tensors,
            );
        }
        
        Ok(output)
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
            return Err(FlameError::InvalidOperation("conv2d_nhwc expects 4D input and 4D weight".into()));
        }
        // NHWC: [N,H,W,C], Weight: [KH,KW,IC,OC]
        let (n, h, w, c) = (id[0], id[1], id[2], id[3]);
        let (kh, kw, ic, oc) = (wd[0], wd[1], wd[2], wd[3]);
        if ic != c { return Err(FlameError::InvalidOperation("weight IC must match input C".into())); }

        // Convert to kernel layouts
        let x_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input_nhwc)?;
        let w_ocic = crate::cuda_ops::GpuOps::weight_khwkicoc_to_ocickhkw(weight_khwkicoc)?;
        // Call NCHW forward (groups=1)
        let y_nchw = Self::conv2d_forward(&x_nchw, &w_ocic, bias, (stride.0, stride.1), (padding.0, padding.1), 1)?;
        // Convert back to NHWC
        let y_nhwc = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&y_nchw)?;

        // Record NHWC op for autograd so backward converts appropriately
        if input_nhwc.requires_grad || weight_khwkicoc.requires_grad || bias.map(|b| b.requires_grad).unwrap_or(false) {
            let mut out = y_nhwc.clone_result()?;
            out.requires_grad = true;
            crate::autograd::AutogradContext::record_op(
                out.id(),
                crate::autograd::Op::Conv2dNHWC { input: input_nhwc.id(), weight: weight_khwkicoc.id(), stride: stride.0, padding: padding.0 },
                vec![
                    (input_nhwc.id(), x_nchw),
                    (weight_khwkicoc.id(), w_ocic),
                ],
            );
            return Ok(out);
        }
        Ok(y_nhwc)
    }
    
    /// Add bias using CUDA kernel
    fn add_bias(output: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let device = output.device();
        let output_dims = output.shape().dims();
        
        let batch_size = output_dims[0];
        let channels = output_dims[1];
        let spatial_size = output_dims[2] * output_dims[3];
        
        let mut result = output.clone_result()?;
        
        let f = device.get_func("conv2d_ops", "add_bias_nchw_kernel")
            .ok_or_else(|| FlameError::Cuda("Failed to get add_bias kernel".into()))?;
        
        let cfg = LaunchConfig::for_num_elems((batch_size * channels * spatial_size) as u32);
        
        launch_kernel!(f, cfg,
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
        let input_col_size = batch_size * in_channels * kernel_h * kernel_w * out_height * out_width;
        let input_col_buffer = crate::tensor::alloc_zeros_from_pool(&device, input_col_size)?;
        
        let cfg = LaunchConfig::for_num_elems(input_col_size as u32);
        
        if stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1 {
            let f_im2col = device.get_func("conv2d_ops", "im2col_kernel_simple")
                .ok_or_else(|| FlameError::Cuda("Failed to get im2col_kernel_simple".into()))?;
            
            launch_kernel!(f_im2col, cfg,
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
            
            let dims_gpu = copy_i32_to_gpu(&device, &dims)?;
            let params_gpu = copy_i32_to_gpu(&device, &conv_params)?;
            
            let f_im2col = device.get_func("conv2d_ops", "im2col_kernel_v2")
                .ok_or_else(|| FlameError::Cuda("Failed to get im2col kernel v2".into()))?;
            
            launch_kernel!(f_im2col, cfg,
                input.storage.try_as_slice_f32()?,
                &input_col_buffer,
                &dims_gpu,
                &params_gpu
            )?;
        }
        
        device.synchronize()?;
        
        // Compute weight gradient
        let input_col_shape = Shape::from_dims(&[batch_size * out_height * out_width, in_channels * kernel_h * kernel_w]);
        let input_col_tensor = Tensor {
            storage: TensorStorage::F32 { data: input_col_buffer, numel: input_col_shape.elem_count() },
            shape: input_col_shape,
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        // grad_weight = grad_output^T @ input_col
        let grad_col_t = grad_col.transpose()?;
        let grad_weight_2d = grad_col_t.matmul(&input_col_tensor)?;
        let grad_weight = grad_weight_2d.reshape(&[out_channels, in_channels, kernel_h, kernel_w])?;
        
        // Gradient w.r.t. input using col2im
        // First compute weight^T @ grad_output
        let weight_t = weight.reshape(&[out_channels, in_channels * kernel_h * kernel_w])?.transpose()?;
        let grad_input_col = grad_col.matmul(&weight_t)?;
        
        // Now use col2im to get grad_input
        let grad_input_data = crate::tensor::alloc_zeros_from_pool(&device, input.shape().elem_count())?;
        
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
            
            let dims_gpu = copy_i32_to_gpu(&device, &dims)?;
            let params_gpu = copy_i32_to_gpu(&device, &conv_params)?;
            
            let f_col2im = device.get_func("conv2d_ops", "col2im_kernel_v2")
                .ok_or_else(|| FlameError::Cuda("Failed to get col2im kernel v2".into()))?;
            
            launch_kernel!(f_col2im, cfg,
                grad_input_col.storage.try_as_slice_f32()?,
                &grad_input_data,
                &dims_gpu,
                &params_gpu
            )?;
        }
        
        device.synchronize()?;
        
        let grad_input = Tensor {
            storage: TensorStorage::F32 { data: grad_input_data, numel: input.shape().elem_count() },
            shape: input.shape().clone(),
            device: device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        };
        
        // Bias gradient is sum over batch and spatial dimensions
        let grad_bias = if grad_output.shape().dims().len() == 4 {
            let grad_bias_data = crate::tensor::alloc_zeros_from_pool(&device, out_channels)?;
            
            let f_bias_grad = device.get_func("conv2d_ops", "bias_grad_kernel")
                .ok_or_else(|| FlameError::Cuda("Failed to get bias_grad kernel".into()))?;
            
            let cfg = LaunchConfig::for_num_elems(out_channels as u32);
            launch_kernel!(f_bias_grad, cfg,
                grad_output.storage.try_as_slice_f32()?,
                &grad_bias_data,
                batch_size as i32,
                out_channels as i32,
                (out_height * out_width) as i32
            )?;
            
            device.synchronize()?;
            
            Some(Tensor {
                storage: TensorStorage::F32 { data: grad_bias_data, numel: out_channels },
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
    
    /// Fallback convolution implementation using im2col + matmul
    pub fn conv2d_forward_fallback(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Tensor> {
        // Get dimensions
        let input_dims = input.shape().dims();
        let weight_dims = weight.shape().dims();
        
        let batch_size = input_dims[0];
        let in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];
        
        let out_channels = weight_dims[0];
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];
        
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
        
        // CPU-based im2col approach for actual convolution computation
        // This is slower than CUDA but produces correct results
        
        // Step 1: Pad input if necessary
        let _padded_input = if pad_h > 0 || pad_w > 0 {
            // Create padded tensor
            let padded_h = in_height + 2 * pad_h;
            let padded_w = in_width + 2 * pad_w;
            let _padded = Tensor::zeros(
                Shape::from_dims(&[batch_size, in_channels, padded_h, padded_w]),
                input.device.clone()
            )?;
            
            // Copy input to padded tensor (currently uses input as-is)
            // In a real implementation, we'd copy the input into the center of padded
            // For now, we'll use a simpler approach
            input.clone_result()?
        } else {
            input.clone_result()?
        };
        
        // Step 2: Perform convolution using matrix multiplication
        // Reshape weight to [out_channels, in_channels * kernel_h * kernel_w]
        let _weight_2d = weight.reshape(&[out_channels, in_channels * kernel_h * kernel_w])?;
        
        // For each output position, extract the corresponding input patch and multiply
        // Direct computation path; im2col is used elsewhere for performance
        
        // Create output tensor
        let mut output_data = vec![0.0f32; batch_size * out_channels * out_height * out_width];
        
        // Get input and weight data
        let input_data = input.to_vec_f32()?;
        let weight_data = weight.to_vec_f32()?;
        
        // Perform direct convolution
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0f32;
                        
                        // Apply kernel
                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;
                                    
                                    // Check bounds (handles padding)
                                    if ih >= pad_h && ih < in_height + pad_h && 
                                       iw >= pad_w && iw < in_width + pad_w {
                                        let ih_actual = ih - pad_h;
                                        let iw_actual = iw - pad_w;
                                        
                                        if ih_actual < in_height && iw_actual < in_width {
                                            let input_idx = b * in_channels * in_height * in_width +
                                                          ic * in_height * in_width +
                                                          ih_actual * in_width +
                                                          iw_actual;
                                            let weight_idx = oc * in_channels * kernel_h * kernel_w +
                                                           ic * kernel_h * kernel_w +
                                                           kh * kernel_w +
                                                           kw;
                                            
                                            sum += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                        
                        let output_idx = b * out_channels * out_height * out_width +
                                       oc * out_height * out_width +
                                       oh * out_width +
                                       ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
        
        // Create output tensor from computed data
        let output = Tensor::from_vec(
            output_data,
            Shape::from_dims(&[batch_size, out_channels, out_height, out_width]),
            input.device.clone()
        )?;
        
        // Add bias if provided
        if let Some(b) = bias {
            let bias_reshaped = b.reshape(&[1, out_channels, 1, 1])?;
            return output.add(&bias_reshaped);
        }
        
        Ok(output)
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
    CudaConv2d::conv2d_forward(
        input,
        weight,
        bias,
        (stride, stride),
        (padding, padding),
        1,
    )
}
