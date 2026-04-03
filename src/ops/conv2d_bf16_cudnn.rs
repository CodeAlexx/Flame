#![cfg(feature = "bf16_conv")]

use super::conv2d_bf16::Conv2dBF16Cfg;
use crate::cudnn::{
    algorithms::AlgorithmSelector, descriptors::*, handle::get_cudnn_handle,
    status::CUDNN_STATUS_SUCCESS,
};
use crate::memory_pool::Workspace;
use crate::{DType, Error, Result, Tensor};
use cudarc::driver::DevicePtr;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::os::raw::{c_int, c_void};
use std::ptr;
use std::sync::Mutex;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct AlgoKey {
    n: i32,
    c_in: i32,
    h: i32,
    w: i32,
    oc: i32,
    kh: i32,
    kw: i32,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
    dil_h: i32,
    dil_w: i32,
    groups: i32,
}

static ALGO_CACHE: Lazy<Mutex<HashMap<AlgoKey, c_int>>> = Lazy::new(|| Mutex::new(HashMap::new()));

pub fn run(
    workspace: &mut Workspace,
    x: &Tensor,
    w_oihw: &Tensor,
    y: &mut Tensor,
    cfg: Conv2dBF16Cfg,
) -> Result<()> {
    let handle_arc = get_cudnn_handle()?;
    let mut handle = handle_arc
        .lock()
        .map_err(|_| Error::CudaError("cuDNN handle mutex poisoned".into()))?;
    unsafe {
        launch(&mut handle, workspace, x, w_oihw, y, cfg)?;
    }
    Ok(())
}

unsafe fn launch(
    handle: &mut CudnnHandle,
    workspace: &mut Workspace,
    x: &Tensor,
    w_oihw: &Tensor,
    y: &mut Tensor,
    cfg: Conv2dBF16Cfg,
) -> Result<()> {
    let x_dims = x.shape().dims();
    let y_dims = y.shape().dims();
    let w_dims = w_oihw.shape().dims();

    let (n, c_in, h, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let (n_out, oc, ho, wo) = (y_dims[0], y_dims[1], y_dims[2], y_dims[3]);
    let (oc_check, ic_check, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);

    if n_out != n || oc_check != oc || ic_check != c_in {
        return Err(Error::InvalidInput(
            "Conv2dBF16 weight dims mismatch".into(),
        ));
    }

    let x_desc = TensorDescriptor::new(&[n, c_in, h, w_in], DType::BF16)?;
    let y_desc = TensorDescriptor::new(&[n, oc, ho, wo], DType::BF16)?;

    let f_desc = {
        let desc = FilterDescriptor::new()?;
        desc.set_4d(DType::BF16, oc, c_in, kh, kw)?;
        desc
    };

    let conv_desc = {
        let desc = ConvolutionDescriptor::new()?;
        desc.set_2d(
            cfg.pad.0 as usize,
            cfg.stride.0 as usize,
            cfg.dil.0 as usize,
            DType::F32,
        )?;
        desc.set_math_type(CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)?;
        desc
    };

    let key = AlgoKey {
        n: n as i32,
        c_in: c_in as i32,
        h: h as i32,
        w: w_in as i32,
        oc: oc as i32,
        kh: kh as i32,
        kw: kw as i32,
        stride_h: cfg.stride.0,
        stride_w: cfg.stride.1,
        pad_h: cfg.pad.0,
        pad_w: cfg.pad.1,
        dil_h: cfg.dil.0,
        dil_w: cfg.dil.1,
        groups: cfg.groups,
    };

    let mut chosen_algo = None;
    let mut workspace_size: usize = 0;
    let mut last_status = CUDNN_STATUS_SUCCESS;
    let mut cache_hit = false;
    {
        let mut cache = ALGO_CACHE.lock().unwrap();
        if let Some(&cached_algo) = cache.get(&key) {
            last_status = cudnnGetConvolutionForwardWorkspaceSize(
                handle.as_ptr(),
                x_desc.as_ptr(),
                f_desc.as_ptr(),
                conv_desc.as_ptr(),
                y_desc.as_ptr(),
                cached_algo,
                &mut workspace_size,
            );
            if last_status == CUDNN_STATUS_SUCCESS {
                chosen_algo = Some(cached_algo);
                cache_hit = true;
            } else {
                cache.remove(&key);
            }
        }
    }

    if chosen_algo.is_none() {
        let mut algo_candidates = [
            AlgorithmSelector::select_forward_algorithm(kh, kw, n, c_in, h, w_in),
            0,
            0,
        ];
        algo_candidates[1] = AlgorithmSelector::get_fallback_algorithm(algo_candidates[0]);
        algo_candidates[2] = AlgorithmSelector::get_fallback_algorithm(algo_candidates[1]);

        for idx in 0..algo_candidates.len() {
            let candidate = algo_candidates[idx];
            if idx > 0 && candidate == algo_candidates[idx - 1] {
                continue;
            }

            last_status = cudnnGetConvolutionForwardWorkspaceSize(
                handle.as_ptr(),
                x_desc.as_ptr(),
                f_desc.as_ptr(),
                conv_desc.as_ptr(),
                y_desc.as_ptr(),
                candidate,
                &mut workspace_size,
            );

            if last_status == CUDNN_STATUS_SUCCESS {
                chosen_algo = Some(candidate);
                break;
            }
        }
    }

    let algo = chosen_algo.ok_or_else(|| {
        Error::CudaError(format!("cuDNN workspace query failed: {}", last_status))
    })?;

    if !cache_hit {
        ALGO_CACHE.lock().unwrap().insert(key.clone(), algo);
    }

    if std::env::var_os("BF16_CONV_DEBUG").is_some() {
        eprintln!(
            "[bf16_conv] algo={} cache_hit={} workspace={} bytes dims=({}, {}, {}, {}) kernel=({}, {}, {}) stride={:?} pad={:?} dil={:?}",
            algo,
            cache_hit,
            workspace_size,
            n,
            c_in,
            h,
            w_in,
            oc,
            kh,
            kw,
            cfg.stride,
            cfg.pad,
            cfg.dil
        );
    }

    let workspace_buffer = if workspace_size > 0 {
        let elems = (workspace_size + 3) / 4; // in f32 elements
        Some(workspace.get_buffer(elems)?)
    } else {
        None
    };

    #[cfg(not(feature = "bf16_u16"))]
    compile_error!("bf16_conv requires bf16_u16 feature for BF16 storage");

    #[cfg(feature = "bf16_u16")]
    let x_ptr = x.as_device_ptr_bf16("conv2d_bf16:x")? as *const c_void;
    #[cfg(feature = "bf16_u16")]
    let w_ptr = w_oihw.as_device_ptr_bf16("conv2d_bf16:w")? as *const c_void;
    #[cfg(feature = "bf16_u16")]
    let y_ptr = y.as_mut_device_ptr_bf16("conv2d_bf16:y")? as *mut c_void;

    let workspace_ptr = workspace_buffer
        .map(|buf| *buf.device_ptr() as *mut c_void)
        .unwrap_or(ptr::null_mut());

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let status = cudnnConvolutionForward(
        handle.as_ptr(),
        &alpha as *const f32 as *const c_void,
        x_desc.as_ptr(),
        x_ptr,
        f_desc.as_ptr(),
        w_ptr,
        conv_desc.as_ptr(),
        algo,
        workspace_ptr,
        workspace_size,
        &beta as *const f32 as *const c_void,
        y_desc.as_ptr(),
        y_ptr,
    );

    if status != CUDNN_STATUS_SUCCESS {
        return Err(Error::CudaError(format!(
            "cuDNN convolution failed: {}",
            status
        )));
    }

    Ok(())
}

#[link(name = "cudnn")]
extern "C" {
    fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: *mut c_void,
        x_desc: *mut c_void,
        w_desc: *mut c_void,
        conv_desc: *mut c_void,
        y_desc: *mut c_void,
        algo: c_int,
        size_in_bytes: *mut usize,
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
        work_space: *mut c_void,
        work_space_size_in_bytes: usize,
        beta: *const c_void,
        y_desc: *mut c_void,
        y: *mut c_void,
    ) -> c_int;
}

use crate::cudnn::handle::CudnnHandle;
