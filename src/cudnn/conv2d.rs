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
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::os::raw::{c_int, c_void};
use std::sync::Mutex;

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

    fn cudnnGetConvolutionForwardAlgorithm_v7(
        handle: *mut c_void,
        x_desc: *mut c_void,
        w_desc: *mut c_void,
        conv_desc: *mut c_void,
        y_desc: *mut c_void,
        requested_algo_count: c_int,
        returned_algo_count: *mut c_int,
        perf_results: *mut CudnnConvolutionFwdAlgoPerf2D,
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

// 2026-05-12 perf: per-shape algorithm + workspace cache.
//
// The legacy path called `AlgorithmSelector::select_forward_algorithm` — a
// pure static heuristic (no benchmark). For shapes outside its rules it
// returned IMPLICIT_PRECOMP_GEMM, which is often 2-3x slower than the
// optimal algorithm cuDNN's v7 heuristic would pick. PyTorch uses v7 +
// cache by default, which is why our VAE conv2d was 3-4x behind morning's
// inference-flame bench.
//
// Same pattern as src/cudnn/conv3d.rs ALGO_CACHE — mutex-guarded HashMap
// keyed on (input dims, weight dims, output dims, stride, padding,
// dilation, groups). First call benchmarks via v7; subsequent calls
// reuse the cached (algo, workspace_size). Cache is process-global.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct CudnnConvolutionFwdAlgoPerf2D {
    algo: c_int,
    status: c_int,
    time: f32,
    memory: usize,
    determinism: c_int,
    math_type: c_int,
    reserved: [c_int; 3],
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct AlgoKey2D {
    x_dims: [i32; 4],
    w_dims: [i32; 4],
    y_dims: [i32; 4],
    pad: [i32; 2],
    stride: [i32; 2],
    dilation: [i32; 2],
    groups: i32,
}

static ALGO_CACHE_2D: Lazy<Mutex<HashMap<AlgoKey2D, (c_int, usize)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn workspace_limit_2d_bytes() -> usize {
    // Default 1 GB; override via FLAME_CUDNN_WORKSPACE_LIMIT_MB.
    std::env::var("FLAME_CUDNN_WORKSPACE_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(1usize * 1024 * 1024 * 1024)
}

/// Select algo via cuDNN v7 heuristic + cache result. Falls back to the
/// static heuristic if v7 fails. Returns (algo, workspace_size).
fn select_conv2d_algo(
    handle: *mut c_void,
    x_desc: &TensorDescriptor,
    w_desc: &FilterDescriptor,
    conv_desc: &ConvolutionDescriptor,
    y_desc: &TensorDescriptor,
    key: &AlgoKey2D,
    fallback_hint: c_int,
) -> Result<(c_int, usize)> {
    let limit = workspace_limit_2d_bytes();

    if let Some(cached) = ALGO_CACHE_2D
        .lock()
        .map_err(|_| Error::CudaError("conv2d algo cache mutex poisoned".into()))?
        .get(key)
        .copied()
    {
        if cached.1 <= limit {
            return Ok(cached);
        }
    }

    let mut returned_count: c_int = 0;
    let mut perf_results = [CudnnConvolutionFwdAlgoPerf2D::default(); 8];
    let mut best: Option<(c_int, usize)> = None;

    let status = unsafe {
        cudnnGetConvolutionForwardAlgorithm_v7(
            handle,
            x_desc.as_ptr(),
            w_desc.as_ptr(),
            conv_desc.as_ptr(),
            y_desc.as_ptr(),
            perf_results.len() as c_int,
            &mut returned_count,
            perf_results.as_mut_ptr(),
        )
    };

    if status == 0 {
        for perf in perf_results.iter().take(returned_count.max(0) as usize) {
            if perf.status == 0 && perf.memory <= limit {
                best = Some((perf.algo, perf.memory));
                break;
            }
        }
    }

    // Fallback: try the static heuristic + its fallback chain.
    if best.is_none() {
        let mut candidates = [fallback_hint, 0, 0];
        candidates[1] = AlgorithmSelector::get_fallback_algorithm(candidates[0]);
        candidates[2] = AlgorithmSelector::get_fallback_algorithm(candidates[1]);
        for idx in 0..candidates.len() {
            let algo = candidates[idx];
            if idx > 0 && algo == candidates[idx - 1] {
                continue;
            }
            let mut ws = 0usize;
            let ws_status = unsafe {
                cudnnGetConvolutionForwardWorkspaceSize(
                    handle,
                    x_desc.as_ptr(),
                    w_desc.as_ptr(),
                    conv_desc.as_ptr(),
                    y_desc.as_ptr(),
                    algo,
                    &mut ws,
                )
            };
            if ws_status == 0 && ws <= limit {
                best = Some((algo, ws));
                break;
            }
        }
    }

    let pick = best.ok_or_else(|| {
        Error::CudaError(format!(
            "cudnn_conv2d_bf16: no usable forward algorithm under workspace limit {} MB",
            limit / (1024 * 1024)
        ))
    })?;

    ALGO_CACHE_2D
        .lock()
        .map_err(|_| Error::CudaError("conv2d algo cache mutex poisoned".into()))?
        .insert(key.clone(), pick);

    Ok(pick)
}

/// Perform Conv2D using cuDNN with native BF16 tensors.
///
/// Input: NCHW BF16, Weight: OIHW (out_ch, in_ch/groups, kH, kW) BF16
/// Output: NCHW BF16
///
/// Uses FP32 compute with tensor core math (ALLOW_CONVERSION) for accuracy.
///
/// `dilation` is `(1, 1)` for a standard conv. Output size follows
/// `out = (in + 2*pad - dilation*(kernel - 1) - 1) / stride + 1`.
pub fn cudnn_conv2d_bf16(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
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

    // Output shape with dilation: out = (in + 2*pad - dilation*(kernel - 1) - 1) / stride + 1.
    // For dilation=1 this reduces to the standard formula.
    let eff_kernel_h = dilation.0 * (kernel_h - 1) + 1;
    let eff_kernel_w = dilation.1 * (kernel_w - 1) + 1;
    let out_height = (in_height + 2 * padding.0 - eff_kernel_h) / stride.0 + 1;
    let out_width = (in_width + 2 * padding.1 - eff_kernel_w) / stride.1 + 1;

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
    conv_desc.set_2d_asymmetric(padding, stride, dilation, compute_type)?;
    conv_desc.set_math_type(CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)?;
    if groups > 1 {
        conv_desc.set_group_count(groups)?;
    }

    // 2026-05-12 perf: algorithm selection via cuDNN v7 heuristic + per-shape
    // cache (mirrors conv3d's pattern). Replaces the static heuristic in
    // AlgorithmSelector::select_forward_algorithm, which was picking
    // suboptimal algorithms (often IMPLICIT_PRECOMP_GEMM) for VAE shapes
    // where Winograd-NonFused or FFT-Tiling are better. The static
    // heuristic survives as a fallback if v7 fails.
    let fallback_hint = AlgorithmSelector::select_forward_algorithm(
        kernel_h,
        kernel_w,
        batch_size,
        in_channels,
        in_height,
        in_width,
    );
    let key = AlgoKey2D {
        x_dims: [
            batch_size as i32,
            in_channels as i32,
            in_height as i32,
            in_width as i32,
        ],
        w_dims: [
            out_channels as i32,
            kernel_channels as i32,
            kernel_h as i32,
            kernel_w as i32,
        ],
        y_dims: [
            batch_size as i32,
            out_channels as i32,
            out_height as i32,
            out_width as i32,
        ],
        pad: [padding.0 as i32, padding.1 as i32],
        stride: [stride.0 as i32, stride.1 as i32],
        dilation: [dilation.0 as i32, dilation.1 as i32],
        groups: groups as i32,
    };
    let (mut algo, mut workspace_size) = select_conv2d_algo(
        handle_guard.as_ptr(),
        &x_desc,
        &w_desc,
        &conv_desc,
        &y_desc,
        &key,
        fallback_hint,
    )?;
    let mut status: c_int = 0;

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
            log::warn!(
                "cudnn_conv2d_bf16: bias addition failed (status {}), continuing without",
                bias_status
            );
        }
    }

    // workspace_alloc dropped here — async-freed on stream, safe
    drop(workspace_alloc);

    Ok(output)
}
