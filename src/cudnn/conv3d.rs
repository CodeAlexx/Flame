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

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct AlgoKey {
    x_dims: [i32; 5],
    w_dims: [i32; 5],
    y_dims: [i32; 5],
    pad: [i32; 3],
    stride: [i32; 3],
    dilation: [i32; 3],
    groups: i32,
}

static ALGO_CACHE: Lazy<Mutex<HashMap<AlgoKey, (c_int, usize)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct CudnnConvolutionFwdAlgoPerf {
    algo: c_int,
    status: c_int,
    time: f32,
    memory: usize,
    determinism: c_int,
    math_type: c_int,
    reserved: [c_int; 3],
}

#[link(name = "cudnn")]
extern "C" {
    fn cudnnGetConvolutionForwardAlgorithm_v7(
        handle: *mut c_void,
        x_desc: *mut c_void,
        w_desc: *mut c_void,
        conv_desc: *mut c_void,
        y_desc: *mut c_void,
        requested_algo_count: c_int,
        returned_algo_count: *mut c_int,
        perf_results: *mut CudnnConvolutionFwdAlgoPerf,
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

fn workspace_limit_bytes() -> usize {
    const DEFAULT_LIMIT_MB: usize = 256;
    std::env::var("FLAME_CUDNN_CONV3D_WS_LIMIT_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_LIMIT_MB)
        * 1024
        * 1024
}

fn output_dim(in_size: usize, pad: usize, dilation: usize, kernel: usize, stride: usize) -> Result<usize> {
    let effective_kernel = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| Error::InvalidShape("cudnn_conv3d_bf16: effective kernel overflow".into()))?;

    let padded = in_size
        .checked_add(pad.saturating_mul(2))
        .ok_or_else(|| Error::InvalidShape("cudnn_conv3d_bf16: padded input overflow".into()))?;

    if padded < effective_kernel {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv3d_bf16: padded dim {} smaller than effective kernel {}",
            padded, effective_kernel
        )));
    }
    Ok((padded - effective_kernel) / stride + 1)
}

fn algo_name(algo: c_int) -> &'static str {
    AlgorithmSelector::algorithm_name(algo)
}

fn select_algo(
    handle: *mut c_void,
    x_desc: &TensorDescriptor,
    w_desc: &FilterDescriptor,
    conv_desc: &ConvolutionDescriptor,
    y_desc: &TensorDescriptor,
    key: &AlgoKey,
    fallback_hint: c_int,
) -> Result<(c_int, usize, bool)> {
    let limit = workspace_limit_bytes();

    if let Some(cached) = ALGO_CACHE.lock().map_err(|_| Error::CudaError("conv3d algo cache mutex poisoned".into()))?.get(key).copied() {
        if cached.1 <= limit {
            return Ok((cached.0, cached.1, true));
        }
    }

    let mut returned_count: c_int = 0;
    let mut perf_results = [CudnnConvolutionFwdAlgoPerf::default(); 8];
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

    if best.is_none() {
        let mut candidates = [fallback_hint, 0, 0];
        candidates[1] = AlgorithmSelector::get_fallback_algorithm(candidates[0]);
        candidates[2] = AlgorithmSelector::get_fallback_algorithm(candidates[1]);

        for idx in 0..candidates.len() {
            let algo = candidates[idx];
            if idx > 0 && algo == candidates[idx - 1] {
                continue;
            }
            let mut workspace_size = 0usize;
            let ws_status = unsafe {
                cudnnGetConvolutionForwardWorkspaceSize(
                    handle,
                    x_desc.as_ptr(),
                    w_desc.as_ptr(),
                    conv_desc.as_ptr(),
                    y_desc.as_ptr(),
                    algo,
                    &mut workspace_size,
                )
            };
            if ws_status == 0 && workspace_size <= limit {
                best = Some((algo, workspace_size));
                break;
            }
        }
    }

    let (algo, workspace_size) = best.ok_or_else(|| {
        Error::CudaError(format!(
            "cudnn_conv3d_bf16: no usable forward algorithm under workspace limit {} MB",
            limit / (1024 * 1024)
        ))
    })?;

    ALGO_CACHE
        .lock()
        .map_err(|_| Error::CudaError("conv3d algo cache mutex poisoned".into()))?
        .insert(key.clone(), (algo, workspace_size));

    Ok((algo, workspace_size, false))
}

pub fn cudnn_conv3d_bf16(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
) -> Result<Tensor> {
    let x_dims = input.shape().dims();
    let w_dims = weight.shape().dims();

    if input.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "cudnn_conv3d_bf16: expected BF16 input, got {:?}",
            input.dtype()
        )));
    }
    if weight.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "cudnn_conv3d_bf16: expected BF16 weight, got {:?}",
            weight.dtype()
        )));
    }
    if x_dims.len() != 5 {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv3d_bf16: input must be 5D NCDHW, got {:?}",
            x_dims
        )));
    }
    if w_dims.len() != 5 {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv3d_bf16: weight must be 5D OIDHW, got {:?}",
            w_dims
        )));
    }
    if groups == 0 {
        return Err(Error::InvalidInput("cudnn_conv3d_bf16: groups must be >= 1".into()));
    }
    if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
        return Err(Error::InvalidInput("cudnn_conv3d_bf16: stride must be >= 1".into()));
    }
    if dilation.0 == 0 || dilation.1 == 0 || dilation.2 == 0 {
        return Err(Error::InvalidInput("cudnn_conv3d_bf16: dilation must be >= 1".into()));
    }

    let (n, c_in, d_in, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4]);
    let (c_out, c_per_group, k_d, k_h, k_w) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3], w_dims[4]);

    if c_in % groups != 0 {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv3d_bf16: input channels {} not divisible by groups {}",
            c_in, groups
        )));
    }
    if c_out % groups != 0 {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv3d_bf16: output channels {} not divisible by groups {}",
            c_out, groups
        )));
    }
    if c_in / groups != c_per_group {
        return Err(Error::InvalidShape(format!(
            "cudnn_conv3d_bf16: weight channels {} do not match input/groups {}",
            c_per_group,
            c_in / groups
        )));
    }

    if let Some(bias) = bias {
        let b_dims = bias.shape().dims();
        if bias.dtype() != DType::BF16 {
            return Err(Error::InvalidInput(format!(
                "cudnn_conv3d_bf16: expected BF16 bias, got {:?}",
                bias.dtype()
            )));
        }
        if b_dims != [c_out] {
            return Err(Error::InvalidShape(format!(
                "cudnn_conv3d_bf16: bias must be [C_out], got {:?}",
                b_dims
            )));
        }
    }

    let d_out = output_dim(d_in, padding.0, dilation.0, k_d, stride.0)?;
    let h_out = output_dim(h_in, padding.1, dilation.1, k_h, stride.1)?;
    let w_out = output_dim(w_in, padding.2, dilation.2, k_w, stride.2)?;

    let x_desc = TensorDescriptor::new(&[n, c_in, d_in, h_in, w_in], DType::BF16)?;
    let w_desc = FilterDescriptor::new()?;
    w_desc.set_nd(DType::BF16, &[c_out, c_per_group, k_d, k_h, k_w])?;
    let y_desc = TensorDescriptor::new(&[n, c_out, d_out, h_out, w_out], DType::BF16)?;

    let conv_desc = ConvolutionDescriptor::new()?;
    conv_desc.set_nd(
        &[padding.0, padding.1, padding.2],
        &[stride.0, stride.1, stride.2],
        &[dilation.0, dilation.1, dilation.2],
        DType::F32,
    )?;
    conv_desc.set_math_type(CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)?;
    if groups > 1 {
        conv_desc.set_group_count(groups)?;
    }

    let output_shape = Shape::from_dims(&[n, c_out, d_out, h_out, w_out]);
    let mut output = Tensor::zeros_dtype(output_shape, DType::BF16, input.device.clone())?;

    let key = AlgoKey {
        x_dims: [n as i32, c_in as i32, d_in as i32, h_in as i32, w_in as i32],
        w_dims: [c_out as i32, c_per_group as i32, k_d as i32, k_h as i32, k_w as i32],
        y_dims: [n as i32, c_out as i32, d_out as i32, h_out as i32, w_out as i32],
        pad: [padding.0 as i32, padding.1 as i32, padding.2 as i32],
        stride: [stride.0 as i32, stride.1 as i32, stride.2 as i32],
        dilation: [dilation.0 as i32, dilation.1 as i32, dilation.2 as i32],
        groups: groups as i32,
    };

    let handle_arc = get_cudnn_handle()?;
    let handle = handle_arc
        .lock()
        .map_err(|_| Error::CudaError("cuDNN handle mutex poisoned".into()))?;

    let fallback_hint = AlgorithmSelector::select_forward_algorithm(k_h, k_w, n, c_in, h_in, w_in);
    let (algo, workspace_size, cache_hit) = select_algo(
        handle.as_ptr(),
        &x_desc,
        &w_desc,
        &conv_desc,
        &y_desc,
        &key,
        fallback_hint,
    )?;

    if std::env::var_os("BF16_CONV_DEBUG").is_some() {
        eprintln!(
            "[conv3d_cudnn] algo={} ({}) cache_hit={} workspace={} bytes x={:?} w={:?} y={:?} stride={:?} pad={:?} dil={:?} groups={}",
            algo,
            algo_name(algo),
            cache_hit,
            workspace_size,
            x_dims,
            w_dims,
            [n, c_out, d_out, h_out, w_out],
            stride,
            padding,
            dilation,
            groups,
        );
    }

    let mut workspace_alloc: Option<cudarc::driver::CudaSlice<u8>> = None;
    let workspace_ptr = if workspace_size > 0 {
        let alloc = unsafe { input.device.alloc::<u8>(workspace_size) }
            .map_err(|e| Error::Cuda(format!("cudnn conv3d workspace alloc: {e:?}")))?;
        use cudarc::driver::DevicePtr;
        let ptr = *alloc.device_ptr() as *mut c_void;
        workspace_alloc = Some(alloc);
        ptr
    } else {
        std::ptr::null_mut()
    };

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let x_ptr = input.as_device_ptr_bf16("cudnn_conv3d_bf16:input")? as *const c_void;
    let w_ptr = weight.as_device_ptr_bf16("cudnn_conv3d_bf16:weight")? as *const c_void;
    let y_ptr = output.as_mut_device_ptr_bf16("cudnn_conv3d_bf16:output")? as *mut c_void;

    let status = unsafe {
        cudnnConvolutionForward(
            handle.as_ptr(),
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
            "cudnn_conv3d_bf16: cudnnConvolutionForward failed: {}",
            status
        )));
    }

    if let Some(bias) = bias {
        let bias_desc = TensorDescriptor::new(&[1, c_out, 1, 1, 1], DType::BF16)?;
        let bias_ptr = bias.as_device_ptr_bf16("cudnn_conv3d_bf16:bias")? as *const c_void;
        let bias_status = unsafe {
            cudnnAddTensor(
                handle.as_ptr(),
                &alpha as *const f32 as *const c_void,
                bias_desc.as_ptr(),
                bias_ptr,
                &alpha as *const f32 as *const c_void,
                y_desc.as_ptr(),
                y_ptr,
            )
        };
        if bias_status != 0 {
            return Err(Error::CudaError(format!(
                "cudnn_conv3d_bf16: cudnnAddTensor failed: {}",
                bias_status
            )));
        }
    }

    drop(workspace_alloc);
    Ok(output)
}
