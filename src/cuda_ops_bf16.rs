#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use core::ffi::c_void;
use std::os::raw::c_char;
use std::{
    collections::HashMap,
    convert::TryFrom,
    ffi::CStr,
    ptr,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use cuda_runtime_sys::{cudaError, cudaError_t, cudaGetErrorString, cudaGetLastError};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice};

use crate::perf_telemetry;
use crate::strict::{self, GuardMode};
use crate::{
    cuda_ops_ffi::{
        fc_axpby_bf16, fc_bf16_broadcast, fc_bf16_index_select, fc_bf16_repeat_axis,
        fc_bf16_repeat_nd, fc_bf16_slice, fc_gelu_bf16, fc_gemm_bf16, fc_group_norm_backward_bf16,
        fc_group_norm_bf16, fc_layer_norm_backward_bf16, fc_layer_norm_bf16, fc_relu_bf16,
        fc_rms_norm_bf16, fc_rms_norm_bf16_to_f32, fc_silu_bf16, flame_conv2d_nhwc_bf16,
        flame_sdpa_chunked_bf16,
        flame_status_to_result, sdpa_stream_bf16_launch, tensor_as_view_bf16,
        tensor_as_view_bf16_mut, tensor_as_view_f32_mut, CudaStream,
        FLAME_CUDA_ERR_UNSUPPORTED, FLAME_CUDA_OK,
    },
    staging::{
        arena_alloc, arena_record_and_release, conv2d_autotune_stats as staging_conv2d_stats,
        flush_sdpa_autotune_cache as staging_flush_sdpa_cache,
        reset_conv2d_autotune_stats as staging_reset_conv2d_stats,
        reset_sdpa_autotune_stats as staging_reset_sdpa_stats,
        sdpa_autotune_stats as staging_sdpa_stats,
    },
    tensor::Tensor,
    DType, Error, Result, Shape,
};

pub(crate) const FC_OK: i32 = 0;
pub(crate) const FC_ERR_INVALID_ARGUMENT: i32 = 1;
pub(crate) const FC_ERR_LAUNCH: i32 = 2;
pub(crate) const FC_ERR_OOM: i32 = 3;
pub(crate) const FC_ERR_UNSUPPORTED: i32 = 4;
pub(crate) const FC_STATUS_LT_FALLBACK: i32 = 5;

#[derive(Clone, Copy, Debug)]
pub struct SdpaWorkspace {
    pub ptr: *mut c_void,
    pub bytes: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConvActivation {
    None,
    Relu,
    Silu,
    Gelu,
}

impl ConvActivation {
    fn as_i32(self) -> i32 {
        match self {
            ConvActivation::None => 0,
            ConvActivation::Relu => 1,
            ConvActivation::Silu => 2,
            ConvActivation::Gelu => 3,
        }
    }
}

static CONV2D_STUB_WARNED: OnceLock<()> = OnceLock::new();

pub type Conv2dAutotuneStats = crate::staging::Conv2dAutotuneStats;
pub type SdpaAutotuneStats = crate::staging::SdpaAutotuneStats;

fn warn_stub_once(cell: &OnceLock<()>, message: &str) {
    if cell.set(()).is_ok() {
        log::warn!("{message}");
    }
}

#[derive(Debug)]
struct SdpaWorkspaceEntry {
    buffer: CudaSlice<u8>,
    bytes: u64,
}

fn sdpa_workspace_registry() -> &'static Mutex<HashMap<(i32, usize), SdpaWorkspaceEntry>> {
    static REGISTRY: OnceLock<Mutex<HashMap<(i32, usize), SdpaWorkspaceEntry>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn workspace_poison<T>(_: std::sync::PoisonError<T>) -> Error {
    Error::Cuda("sdpa_workspace_registry mutex poisoned".into())
}

fn acquire_sdpa_workspace(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
    required: u64,
) -> Result<SdpaWorkspace> {
    if required == 0 {
        return Ok(SdpaWorkspace {
            ptr: ptr::null_mut(),
            bytes: 0,
        });
    }

    let key = (device.ordinal() as i32, stream.as_raw() as usize);

    {
        let registry = sdpa_workspace_registry();
        let guard = registry.lock().map_err(workspace_poison)?;
        if let Some(entry) = guard.get(&key) {
            if entry.bytes >= required {
                let ptr = *entry.buffer.device_ptr() as *mut c_void;
                return Ok(SdpaWorkspace {
                    ptr,
                    bytes: entry.bytes,
                });
            }
        }
    }

    // allocate outside the lock
    let len = usize::try_from(required)
        .map_err(|_| Error::InvalidInput("sdpa workspace exceeds usize".into()))?;
    let buffer = device
        .alloc_zeros::<u8>(len)
        .map_err(|_| Error::CudaDriver)?;

    let mut registry = sdpa_workspace_registry().lock().map_err(workspace_poison)?;
    registry.insert(
        key,
        SdpaWorkspaceEntry {
            buffer,
            bytes: required,
        },
    );
    let entry = registry
        .get(&key)
        .expect("workspace entry must exist after insert");
    let ptr = *entry.buffer.device_ptr() as *mut c_void;
    Ok(SdpaWorkspace {
        ptr,
        bytes: entry.bytes,
    })
}

fn status_to_result(status: i32, op: &str) -> Result<()> {
    match status {
        FC_OK => Ok(()),
        FC_ERR_INVALID_ARGUMENT => Err(Error::InvalidInput(format!("{op}: invalid argument"))),
        FC_ERR_LAUNCH => Err(Error::Cuda(format!("{op}: kernel launch failed"))),
        FC_ERR_OOM => Err(Error::Cuda(format!("{op}: cudaMalloc failed (OOM)"))),
        FC_ERR_UNSUPPORTED => Err(Error::Unsupported(format!("{op}: unsupported"))),
        FC_STATUS_LT_FALLBACK => Ok(()),
        other => Err(Error::Cuda(format!("{op}: unexpected status {other}"))),
    }
}

fn ensure_bf16(t: &Tensor, name: &str) -> Result<()> {
    if t.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "{name}: expected BF16 tensor, got logical {:?}",
            t.dtype()
        )));
    }
    if t.storage_dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "{name}: expected BF16 storage, got {:?}",
            t.storage_dtype()
        )));
    }
    Ok(())
}

fn default_stream(t: &Tensor) -> CudaStream {
    use crate::device::CudaStreamRawPtrExt;
    let raw = t.device().cuda_stream_raw_ptr();
    CudaStream::from_raw(raw)
}

pub fn relu_bf16(x: &Tensor) -> Result<Tensor> {
    ensure_bf16(x, "relu_bf16:x")?;
    let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    ensure_bf16(&out, "relu_bf16:out")?;

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "relu_bf16:x")?;
    let mut vy = tensor_as_view_bf16_mut(&mut out, "relu_bf16:out")?;

    let status = unsafe { fc_relu_bf16(&vx, &mut vy, stream.as_raw()) };
    status_to_result(status, "fc_relu_bf16")?;
    Ok(out)
}

pub fn gelu_bf16(x: &Tensor) -> Result<Tensor> {
    ensure_bf16(x, "gelu_bf16:x")?;
    let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    ensure_bf16(&out, "gelu_bf16:out")?;

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "gelu_bf16:x")?;
    let mut vy = tensor_as_view_bf16_mut(&mut out, "gelu_bf16:out")?;

    let status = unsafe { fc_gelu_bf16(&vx, &mut vy, stream.as_raw()) };
    status_to_result(status, "fc_gelu_bf16")?;
    Ok(out)
}

pub fn gelu_bf16_into(x: &Tensor, out: &mut Tensor) -> Result<()> {
    ensure_bf16(x, "gelu_bf16_into:x")?;
    ensure_bf16(out, "gelu_bf16_into:out")?;
    if !Arc::ptr_eq(x.device(), out.device()) {
        return Err(Error::InvalidInput(
            "gelu_bf16_into: input and output tensors must share a device".into(),
        ));
    }
    if x.shape() != out.shape() {
        return Err(Error::ShapeMismatch {
            expected: x.shape().clone(),
            got: out.shape().clone(),
        });
    }

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "gelu_bf16_into:x")?;
    let mut vy = tensor_as_view_bf16_mut(out, "gelu_bf16_into:out")?;

    let status = unsafe { fc_gelu_bf16(&vx, &mut vy, stream.as_raw()) };
    status_to_result(status, "fc_gelu_bf16")
}

pub fn silu_bf16(x: &Tensor) -> Result<Tensor> {
    ensure_bf16(x, "silu_bf16:x")?;
    let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    ensure_bf16(&out, "silu_bf16:out")?;

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "silu_bf16:x")?;
    let mut vy = tensor_as_view_bf16_mut(&mut out, "silu_bf16:out")?;

    let status = unsafe { fc_silu_bf16(&vx, &mut vy, stream.as_raw()) };
    status_to_result(status, "fc_silu_bf16")?;
    Ok(out)
}

pub fn axpby_inplace_bf16(x: &Tensor, a: f32, y: &mut Tensor, b: f32) -> Result<()> {
    ensure_bf16(x, "axpby_bf16:x")?;
    ensure_bf16(y, "axpby_bf16:y")?;

    let stream = default_stream(y);
    let vx = tensor_as_view_bf16(x, "axpby_bf16:x")?;
    let mut vy = tensor_as_view_bf16_mut(y, "axpby_bf16:y")?;

    let status = unsafe { fc_axpby_bf16(&vx, a, &mut vy, b, stream.as_raw()) };
    status_to_result(status, "fc_axpby_bf16")
}

pub fn rms_norm_bf16(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    ensure_bf16(x, "rms_norm_bf16:x")?;
    if let Some(w) = weight {
        ensure_bf16(w, "rms_norm_bf16:weight")?;
    }
    let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    ensure_bf16(&out, "rms_norm_bf16:out")?;

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "rms_norm_bf16:x")?;
    let vw = if let Some(w) = weight {
        Some(tensor_as_view_bf16(w, "rms_norm_bf16:weight")?)
    } else {
        None
    };
    let mut vy = tensor_as_view_bf16_mut(&mut out, "rms_norm_bf16:out")?;

    let status = unsafe {
        fc_rms_norm_bf16(
            &vx,
            vw.as_ref().map(|v| v as *const _).unwrap_or(ptr::null()),
            eps,
            &mut vy,
            stream.as_raw(),
        )
    };
    status_to_result(status, "fc_rms_norm_bf16")?;
    Ok(out)
}

/// RMSNorm with BF16 input → F32 output (no weight).
/// Used for Gemma3-style `(1+weight)` formulation where the multiply
/// must happen in F32 precision to match PyTorch's `norm(x.float())`.
pub fn rms_norm_bf16_to_f32(x: &Tensor, eps: f32) -> Result<Tensor> {
    ensure_bf16(x, "rms_norm_bf16_to_f32:x")?;
    let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::F32, x.device().clone())?;

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "rms_norm_bf16_to_f32:x")?;
    let mut vy = tensor_as_view_f32_mut(&mut out, "rms_norm_bf16_to_f32:out")?;

    let status = unsafe {
        fc_rms_norm_bf16_to_f32(
            &vx,
            eps,
            &mut vy,
            stream.as_raw(),
        )
    };
    status_to_result(status, "fc_rms_norm_bf16_to_f32")?;
    Ok(out)
}

pub fn layer_norm_bf16(
    x: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    ensure_bf16(x, "layer_norm_bf16:x")?;
    if let Some(g) = gamma {
        ensure_bf16(g, "layer_norm_bf16:gamma")?;
    }
    if let Some(b) = beta {
        ensure_bf16(b, "layer_norm_bf16:beta")?;
    }

    let total_elems = x.shape().elem_count();
    if total_elems == 0 {
        return Err(Error::InvalidInput(
            "layer_norm_bf16: input tensor is empty".into(),
        ));
    }

    // Default to normalizing the last dimension when no gamma is provided.
    let norm_size = gamma
        .map(|g| g.shape().elem_count())
        .or_else(|| beta.map(|b| b.shape().elem_count()))
        .unwrap_or_else(|| x.shape().dims().last().copied().unwrap_or(1));

    let outer = total_elems
        .checked_div(norm_size)
        .ok_or_else(|| Error::InvalidInput("layer_norm_bf16: invalid norm size".into()))?;
    if outer == 0 {
        return Err(Error::InvalidInput(
            "layer_norm_bf16: invalid normalization shape".into(),
        ));
    }

    let mut mean_buf = crate::tensor::alloc_zeros_from_pool(x.device(), outer)?;
    let mut rstd_buf = crate::tensor::alloc_zeros_from_pool(x.device(), outer)?;
    let out =
        layer_norm_bf16_with_stats(x, gamma, beta, norm_size, eps, &mut mean_buf, &mut rstd_buf)?;
    drop(mean_buf);
    drop(rstd_buf);
    Ok(out)
}

fn layer_norm_bf16_with_stats_impl(
    out: &mut Tensor,
    x: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    norm_size: usize,
    eps: f32,
    mean_out: &mut cudarc::driver::CudaSlice<f32>,
    rstd_out: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<()> {
    ensure_bf16(x, "layer_norm_bf16:x")?;
    if let Some(g) = gamma {
        ensure_bf16(g, "layer_norm_bf16:gamma")?;
    }
    if let Some(b) = beta {
        ensure_bf16(b, "layer_norm_bf16:beta")?;
    }
    if norm_size == 0 {
        return Err(Error::InvalidInput(
            "layer_norm_bf16: norm_size must be > 0".into(),
        ));
    }

    let total_elems = x.shape().elem_count();
    if total_elems % norm_size != 0 {
        return Err(Error::InvalidInput(
            "layer_norm_bf16: norm_size must divide tensor elements".into(),
        ));
    }
    let outer = total_elems / norm_size;
    if mean_out.len() != outer || rstd_out.len() != outer {
        return Err(Error::InvalidInput(
            "layer_norm_bf16: mean/rstd buffers have incorrect length".into(),
        ));
    }

    ensure_bf16(out, "layer_norm_bf16:out")?;
    if !Arc::ptr_eq(out.device(), x.device()) {
        return Err(Error::InvalidInput(
            "layer_norm_bf16: output tensor must share device with input".into(),
        ));
    }
    if out.shape() != x.shape() {
        return Err(Error::ShapeMismatch {
            expected: x.shape().clone(),
            got: out.shape().clone(),
        });
    }

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "layer_norm_bf16:x")?;
    let vg = gamma
        .map(|g| tensor_as_view_bf16(g, "layer_norm_bf16:gamma"))
        .transpose()?;
    let vb = beta
        .map(|b| tensor_as_view_bf16(b, "layer_norm_bf16:beta"))
        .transpose()?;
    let mut vy = tensor_as_view_bf16_mut(out, "layer_norm_bf16:out")?;

    let norm_size_i64 = i64::try_from(norm_size)
        .map_err(|_| Error::InvalidInput("layer_norm_bf16: norm_size exceeds i64".into()))?;

    let mean_ptr = *mean_out.device_ptr_mut();
    let rstd_ptr = *rstd_out.device_ptr_mut();

    let status = unsafe {
        fc_layer_norm_bf16(
            &vx,
            vg.as_ref().map(|v| v as *const _).unwrap_or(ptr::null()),
            vb.as_ref().map(|v| v as *const _).unwrap_or(ptr::null()),
            norm_size_i64,
            eps,
            &mut vy,
            mean_ptr as *mut f32,
            rstd_ptr as *mut f32,
            stream.as_raw(),
        )
    };
    status_to_result(status, "fc_layer_norm_bf16")
}

pub fn layer_norm_bf16_into_with_stats(
    out: &mut Tensor,
    x: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    norm_size: usize,
    eps: f32,
    mean_out: &mut cudarc::driver::CudaSlice<f32>,
    rstd_out: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<()> {
    layer_norm_bf16_with_stats_impl(out, x, gamma, beta, norm_size, eps, mean_out, rstd_out)
}

pub fn layer_norm_bf16_with_stats(
    x: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    norm_size: usize,
    eps: f32,
    mean_out: &mut cudarc::driver::CudaSlice<f32>,
    rstd_out: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<Tensor> {
    let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    layer_norm_bf16_with_stats_impl(&mut out, x, gamma, beta, norm_size, eps, mean_out, rstd_out)?;
    Ok(out)
}

pub fn layer_norm_backward_bf16(
    x: &Tensor,
    dy: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    normalized_shape: &[usize],
    eps: f32,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    ensure_bf16(x, "layer_norm_backward_bf16:x")?;
    ensure_bf16(dy, "layer_norm_backward_bf16:dy")?;
    if !Arc::ptr_eq(x.device(), dy.device()) {
        return Err(Error::InvalidInput(
            "layer_norm_backward_bf16: x and dy must share a device".into(),
        ));
    }
    if x.shape() != dy.shape() {
        return Err(Error::ShapeMismatch {
            expected: x.shape().clone(),
            got: dy.shape().clone(),
        });
    }

    let dims = x.shape().dims();
    if dims.len() < normalized_shape.len() {
        return Err(Error::InvalidInput(
            "layer_norm_backward_bf16: normalized shape rank mismatch".into(),
        ));
    }
    let offset = dims.len() - normalized_shape.len();
    for (i, &dim) in normalized_shape.iter().enumerate() {
        if dims[offset + i] != dim {
            return Err(Error::InvalidInput(format!(
                "layer_norm_backward_bf16: expected dim {} == {}, got {}",
                offset + i,
                dim,
                dims[offset + i]
            )));
        }
    }

    if let Some(g) = gamma {
        ensure_bf16(g, "layer_norm_backward_bf16:gamma")?;
        if g.shape().dims() != normalized_shape {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(normalized_shape),
                got: g.shape().clone(),
            });
        }
        if !Arc::ptr_eq(g.device(), x.device()) {
            return Err(Error::InvalidInput(
                "layer_norm_backward_bf16: gamma must share device with input".into(),
            ));
        }
    }
    if let Some(b) = beta {
        ensure_bf16(b, "layer_norm_backward_bf16:beta")?;
        if b.shape().dims() != normalized_shape {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(normalized_shape),
                got: b.shape().clone(),
            });
        }
        if !Arc::ptr_eq(b.device(), x.device()) {
            return Err(Error::InvalidInput(
                "layer_norm_backward_bf16: beta must share device with input".into(),
            ));
        }
    }

    let norm_size = normalized_shape.iter().product::<usize>();
    if norm_size == 0 {
        return Err(Error::InvalidInput(
            "layer_norm_backward_bf16: normalized_shape must contain positive dims".into(),
        ));
    }
    let outer = x.shape().elem_count() / norm_size;

    let mut dx = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    dx.requires_grad = false;

    let gamma_ptr = if let Some(g) = gamma {
        g.as_device_ptr_bf16("layer_norm_backward_bf16:gamma")? as *const u16
    } else {
        ptr::null()
    };

    let mut dgamma_tensor = if gamma.is_some() {
        Some(Tensor::zeros_dtype(
            Shape::from_dims(normalized_shape),
            DType::F32,
            x.device().clone(),
        )?)
    } else {
        None
    };

    let mut dbeta_tensor = if beta.is_some() {
        Some(Tensor::zeros_dtype(
            Shape::from_dims(normalized_shape),
            DType::F32,
            x.device().clone(),
        )?)
    } else {
        None
    };

    let dgamma_ptr = if let Some(t) = dgamma_tensor.as_mut() {
        let slice = t.storage_mut().try_as_mut_slice_f32().map_err(|_| {
            Error::InvalidInput("layer_norm_backward_bf16: expected F32 grad gamma".into())
        })?;
        (*slice.device_ptr()) as u64 as *mut f32
    } else {
        ptr::null_mut()
    };

    let dbeta_ptr = if let Some(t) = dbeta_tensor.as_mut() {
        let slice = t.storage_mut().try_as_mut_slice_f32().map_err(|_| {
            Error::InvalidInput("layer_norm_backward_bf16: expected F32 grad beta".into())
        })?;
        (*slice.device_ptr()) as u64 as *mut f32
    } else {
        ptr::null_mut()
    };

    let status = unsafe {
        fc_layer_norm_backward_bf16(
            x.as_device_ptr_bf16("layer_norm_backward_bf16:x")?,
            dy.as_device_ptr_bf16("layer_norm_backward_bf16:dy")?,
            gamma_ptr,
            outer as i64,
            norm_size as i64,
            eps,
            dx.as_mut_device_ptr_bf16("layer_norm_backward_bf16:dx")?,
            dgamma_ptr,
            dbeta_ptr,
            default_stream(x).as_raw(),
        )
    };
    status_to_result(status, "fc_layer_norm_backward_bf16")?;

    if let Some(t) = dgamma_tensor.as_mut() {
        t.requires_grad = false;
    }
    if let Some(t) = dbeta_tensor.as_mut() {
        t.requires_grad = false;
    }

    Ok((dx, dgamma_tensor, dbeta_tensor))
}

pub fn group_norm_bf16(
    x: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    groups: i32,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(
            "group_norm_bf16 expects 4D input [N,H,W,C]".into(),
        ));
    }
    if groups <= 0 {
        return Err(Error::InvalidInput(
            "group_norm_bf16 requires groups > 0".into(),
        ));
    }
    let batch = dims[0];
    let groups_usize = usize::try_from(groups)
        .map_err(|_| Error::InvalidInput("group_norm_bf16: groups must fit in usize".into()))?;
    let expected_len = batch
        .checked_mul(groups_usize)
        .ok_or_else(|| Error::InvalidInput("group_norm_bf16: stats buffer overflow".into()))?;

    let mut mean_buf = crate::tensor::alloc_zeros_from_pool(x.device(), expected_len)?;
    let mut var_buf = crate::tensor::alloc_zeros_from_pool(x.device(), expected_len)?;
    let out = group_norm_bf16_with_stats(x, gamma, beta, groups, eps, &mut mean_buf, &mut var_buf)?;
    drop(mean_buf);
    drop(var_buf);
    Ok(out)
}

pub fn group_norm_bf16_with_stats(
    x: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    groups: i32,
    eps: f32,
    mean_out: &mut cudarc::driver::CudaSlice<f32>,
    var_out: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<Tensor> {
    ensure_bf16(x, "group_norm_bf16:x")?;
    if let Some(g) = gamma {
        ensure_bf16(g, "group_norm_bf16:gamma")?;
    }
    if let Some(b) = beta {
        ensure_bf16(b, "group_norm_bf16:beta")?;
    }

    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(
            "group_norm_bf16 expects 4D input [N,H,W,C]".into(),
        ));
    }
    if groups <= 0 {
        return Err(Error::InvalidInput(
            "group_norm_bf16 requires groups > 0".into(),
        ));
    }
    let batch = dims[0];
    let channels = dims[1];
    if channels % (groups as usize) != 0 {
        return Err(Error::InvalidInput(
            "group_norm_bf16: channels must be divisible by groups".into(),
        ));
    }
    let expected_len = batch
        .checked_mul(groups as usize)
        .ok_or_else(|| Error::InvalidInput("group_norm_bf16: stats buffer overflow".into()))?;
    if mean_out.len() != expected_len || var_out.len() != expected_len {
        return Err(Error::InvalidInput(
            "group_norm_bf16: mean/var buffers have incorrect length".into(),
        ));
    }

    let mut out = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    ensure_bf16(&out, "group_norm_bf16:out")?;

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "group_norm_bf16:x")?;
    let vg = gamma
        .map(|g| tensor_as_view_bf16(g, "group_norm_bf16:gamma"))
        .transpose()?;
    let vb = beta
        .map(|b| tensor_as_view_bf16(b, "group_norm_bf16:beta"))
        .transpose()?;
    let mut vy = tensor_as_view_bf16_mut(&mut out, "group_norm_bf16:out")?;

    let mean_ptr = *mean_out.device_ptr_mut();
    let var_ptr = *var_out.device_ptr_mut();

    let status = unsafe {
        fc_group_norm_bf16(
            &vx,
            vg.as_ref().map(|v| v as *const _).unwrap_or(ptr::null()),
            vb.as_ref().map(|v| v as *const _).unwrap_or(ptr::null()),
            groups,
            eps,
            &mut vy,
            mean_ptr as *mut f32,
            var_ptr as *mut f32,
            stream.as_raw(),
        )
    };

    status_to_result(status, "fc_group_norm_bf16")?;
    Ok(out)
}

pub fn group_norm_backward_bf16(
    x: &Tensor,
    dy: &Tensor,
    gamma: Option<&Tensor>,
    num_groups: usize,
    eps: f32,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    ensure_bf16(x, "group_norm_backward_bf16:x")?;
    ensure_bf16(dy, "group_norm_backward_bf16:dy")?;
    if !Arc::ptr_eq(x.device(), dy.device()) {
        return Err(Error::InvalidInput(
            "group_norm_backward_bf16: x and dy must share a device".into(),
        ));
    }
    if x.shape() != dy.shape() {
        return Err(Error::ShapeMismatch {
            expected: x.shape().clone(),
            got: dy.shape().clone(),
        });
    }

    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(
            "group_norm_backward_bf16 expects 4D input [N,C,H,W]".into(),
        ));
    }
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];
    let spatial = height * width;

    if num_groups == 0 || channels % num_groups != 0 {
        return Err(Error::InvalidInput(
            "group_norm_backward_bf16: channels must be divisible by groups".into(),
        ));
    }

    if let Some(g) = gamma {
        ensure_bf16(g, "group_norm_backward_bf16:gamma")?;
        if g.shape().dims() != [channels] {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(&[channels]),
                got: g.shape().clone(),
            });
        }
        if !Arc::ptr_eq(g.device(), x.device()) {
            return Err(Error::InvalidInput(
                "group_norm_backward_bf16: gamma must share device with input".into(),
            ));
        }
    }

    let mut dx = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    dx.requires_grad = false;

    let gamma_ptr = if let Some(g) = gamma {
        g.as_device_ptr_bf16("group_norm_backward_bf16:gamma")? as *const u16
    } else {
        ptr::null()
    };

    let mut dgamma_tensor = if gamma.is_some() {
        Some(Tensor::zeros_dtype(
            Shape::from_dims(&[channels]),
            DType::F32,
            x.device().clone(),
        )?)
    } else {
        None
    };

    let mut dbeta_tensor = Some(Tensor::zeros_dtype(
        Shape::from_dims(&[channels]),
        DType::F32,
        x.device().clone(),
    )?);
    if let Some(t) = dbeta_tensor.as_mut() {
        t.requires_grad = false;
    }

    let dgamma_ptr = if let Some(t) = dgamma_tensor.as_mut() {
        let slice = t.storage_mut().try_as_mut_slice_f32().map_err(|_| {
            Error::InvalidInput("group_norm_backward_bf16: expected F32 grad gamma".into())
        })?;
        (*slice.device_ptr()) as u64 as *mut f32
    } else {
        ptr::null_mut()
    };

    let dbeta_ptr = if let Some(t) = dbeta_tensor.as_mut() {
        let slice = t.storage_mut().try_as_mut_slice_f32().map_err(|_| {
            Error::InvalidInput("group_norm_backward_bf16: expected F32 grad beta".into())
        })?;
        (*slice.device_ptr()) as u64 as *mut f32
    } else {
        ptr::null_mut()
    };

    let status = unsafe {
        fc_group_norm_backward_bf16(
            x.as_device_ptr_bf16("group_norm_backward_bf16:x")?,
            dy.as_device_ptr_bf16("group_norm_backward_bf16:dy")?,
            gamma_ptr,
            batch_size as i64,
            channels as i64,
            spatial as i64,
            num_groups as i32,
            eps,
            dx.as_mut_device_ptr_bf16("group_norm_backward_bf16:dx")?,
            dgamma_ptr,
            dbeta_ptr,
            default_stream(x).as_raw(),
        )
    };
    status_to_result(status, "fc_group_norm_backward_bf16")?;

    if let Some(t) = dgamma_tensor.as_mut() {
        t.requires_grad = false;
    }

    Ok((dx, dgamma_tensor, dbeta_tensor))
}

fn validate_gemm_bf16_inputs(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
) -> Result<(usize, usize, usize)> {
    ensure_bf16(x, "gemm_bf16:x")?;
    ensure_bf16(w, "gemm_bf16:w")?;
    if !Arc::ptr_eq(x.device(), w.device()) {
        return Err(Error::InvalidInput(
            "gemm_bf16: input and weight tensors must share a device".into(),
        ));
    }

    let x_dims = x.shape().dims();
    let w_dims = w.shape().dims();
    if x_dims.len() != 2 || w_dims.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "gemm_bf16: expected [M,K] x [K,N] inputs, got {:?} and {:?}",
            x_dims, w_dims
        )));
    }
    let m = x_dims[0];
    let k = x_dims[1];
    if w_dims[0] != k {
        return Err(Error::InvalidInput(format!(
            "gemm_bf16: weight K mismatch (expected {k}, got {})",
            w_dims[0]
        )));
    }
    let n = w_dims[1];

    if let Some(b) = bias {
        ensure_bf16(b, "gemm_bf16:bias")?;
        if !Arc::ptr_eq(x.device(), b.device()) {
            return Err(Error::InvalidInput(
                "gemm_bf16: bias tensor must share device with inputs".into(),
            ));
        }
        let bdims = b.shape().dims();
        let valid = match bdims {
            [cols] => *cols == n,
            [rows, cols] => *rows == 1 && *cols == n,
            _ => false,
        };
        if !valid {
            return Err(Error::InvalidInput(format!(
                "gemm_bf16: bias shape {:?} incompatible with output cols {n}",
                bdims
            )));
        }
    }

    Ok((m, k, n))
}

fn strict_block_lt_fallback(m: usize, n: usize) -> Result<()> {
    #[cfg(feature = "strict_bf16")]
    {
        if strict::is_enabled() {
            let forced = std::env::var("FLAME_CUBLASLT_FORCE_FALLBACK")
                .ok()
                .map(|v| v.to_ascii_lowercase())
                .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "on"));
            if forced {
                let out_shape = Shape::from_dims(&[m, n]);
                strict::record_layout_fix("cuda_ops.gemm_bf16.lt_fallback", &out_shape);
                return Err(Error::InvalidInput(
                    "gemm_bf16: forced cuBLASLt fallback blocked under STRICT_BF16".into(),
                ));
            }
        }
    }
    Ok(())
}

fn gemm_bf16_into_impl(
    out: &mut Tensor,
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    ensure_bf16(out, "gemm_bf16:out")?;
    if !Arc::ptr_eq(out.device(), x.device()) {
        return Err(Error::InvalidInput(
            "gemm_bf16: output tensor must share device with inputs".into(),
        ));
    }
    let out_dims = out.shape().dims();
    if out_dims.len() != 2 || out_dims[0] != m || out_dims[1] != n {
        return Err(Error::InvalidInput(format!(
            "gemm_bf16: output shape {:?} incompatible with [{m}, {n}]",
            out_dims
        )));
    }

    strict_block_lt_fallback(m, n)?;

    let stream = default_stream(x);
    let vx = tensor_as_view_bf16(x, "gemm_bf16:x")?;
    let vw = tensor_as_view_bf16(w, "gemm_bf16:w")?;
    let vb = bias
        .map(|b| tensor_as_view_bf16(b, "gemm_bf16:bias"))
        .transpose()?;
    let mut vy = tensor_as_view_bf16_mut(out, "gemm_bf16:out")?;

    let status = unsafe {
        fc_gemm_bf16(
            &vx,
            &vw,
            vb.as_ref().map(|v| v as *const _).unwrap_or(ptr::null()),
            &mut vy,
            stream.as_raw(),
        )
    };

    if status == FC_STATUS_LT_FALLBACK {
        let out_shape = Shape::from_dims(&[m, n]);
        #[cfg(feature = "strict_bf16")]
        {
            if strict::is_enabled() {
                let _ = x.device().synchronize();
            }
        }
        strict::record_layout_fix("cuda_ops.gemm_bf16.lt_fallback", &out_shape);
        log::warn!(
            "gemm_bf16: cuBLASLt fell back to strided BF16 helper (m={}, n={}, k={})",
            m,
            n,
            k
        );
        if (m == 128 || m == 4096) && n == 1536 && k == 1536 {
            log::error!(
                "gemm_bf16 streaming_linear_tripwire: helper fallback fired for (m={}, n={}, k={})",
                m,
                n,
                k
            );
        }
        return Ok(());
    }

    if status == FC_ERR_UNSUPPORTED {
        let out_shape = Shape::from_dims(&[m, n]);
        strict::record_layout_fix("cuda_ops.gemm_bf16.unsupported", &out_shape);
    }
    status_to_result(status, "fc_gemm_bf16")
}

pub fn gemm_bf16_into(
    out: &mut Tensor,
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
) -> Result<()> {
    strict::scope("cuda_ops.gemm_bf16.into", GuardMode::env_default(), || {
        let (m, k, n) = validate_gemm_bf16_inputs(x, w, bias)?;
        strict_block_lt_fallback(m, n)?;
        gemm_bf16_into_impl(out, x, w, bias, m, k, n)
    })
}

pub fn gemm_bf16(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    strict::scope("cuda_ops.gemm_bf16", GuardMode::env_default(), || {
        let (m, k, n) = validate_gemm_bf16_inputs(x, w, bias)?;
        strict_block_lt_fallback(m, n)?;
        let out_shape = Shape::from_dims(&[m, n]);
        let mut out = Tensor::zeros_dtype(out_shape, DType::BF16, x.device().clone())?;
        gemm_bf16_into_impl(&mut out, x, w, bias, m, k, n)?;
        Ok(out)
    })
}

pub fn slice_axis_bf16(t: &Tensor, axis: usize, start: usize, len: usize) -> Result<Tensor> {
    ensure_bf16(t, "slice_axis_bf16:t")?;
    let dims = t.shape().dims();
    if axis >= dims.len() {
        return Err(Error::InvalidInput(format!(
            "slice_axis_bf16: axis {axis} out of range for shape {:?}",
            dims
        )));
    }
    if start > dims[axis] {
        return Err(Error::InvalidInput(format!(
            "slice_axis_bf16: start {} exceeds dimension {}",
            start, dims[axis]
        )));
    }
    if len > dims[axis] - start {
        return Err(Error::InvalidInput(format!(
            "slice_axis_bf16: slice {}..{} out of bounds for axis len {}",
            start,
            start + len,
            dims[axis]
        )));
    }

    let mut out_dims = dims.to_vec();
    out_dims[axis] = len;
    let mut out =
        Tensor::zeros_dtype(Shape::from_dims(&out_dims), DType::BF16, t.device().clone())?;

    if len == 0 {
        return Ok(out);
    }

    let stream = default_stream(t);
    let vx = tensor_as_view_bf16(t, "slice_axis_bf16:in")?;
    let mut vy = tensor_as_view_bf16_mut(&mut out, "slice_axis_bf16:out")?;
    let status = unsafe {
        fc_bf16_slice(
            &vx,
            axis as i32,
            start as i64,
            len as i64,
            &mut vy,
            stream.as_raw(),
        )
    };
    status_to_result(status, "fc_bf16_slice")?;
    Ok(out)
}

pub fn broadcast_to_bf16(t: &Tensor, out_shape: &[usize]) -> Result<Tensor> {
    ensure_bf16(t, "broadcast_bf16:t")?;
    if out_shape.len() != t.shape().rank() {
        return Err(Error::InvalidInput(format!(
            "broadcast_bf16: rank mismatch (input {} vs output {})",
            t.shape().rank(),
            out_shape.len()
        )));
    }

    let mut out =
        Tensor::zeros_dtype(Shape::from_dims(out_shape), DType::BF16, t.device().clone())?;

    let dims_i64: Vec<i64> = out_shape.iter().map(|&d| d as i64).collect();
    let strides = Shape::from_dims(out_shape).strides();
    let strides_i64: Vec<i64> = strides.iter().map(|&s| s as i64).collect();

    let stream = default_stream(t);
    let vx = tensor_as_view_bf16(t, "broadcast_bf16:in")?;
    let mut vy = tensor_as_view_bf16_mut(&mut out, "broadcast_bf16:out")?;

    let status = unsafe {
        fc_bf16_broadcast(
            &vx,
            dims_i64.as_ptr(),
            strides_i64.as_ptr(),
            out_shape.len() as i32,
            &mut vy,
            stream.as_raw(),
        )
    };
    status_to_result(status, "fc_bf16_broadcast")?;
    Ok(out)
}

pub fn broadcast_to_bf16_into(t: &Tensor, out: &mut Tensor) -> Result<()> {
    ensure_bf16(t, "broadcast_bf16_into:in")?;
    ensure_bf16(out, "broadcast_bf16_into:out")?;
    if t.shape().rank() != out.shape().rank() {
        return Err(Error::InvalidInput(format!(
            "broadcast_bf16_into: rank mismatch (input {} vs output {})",
            t.shape().rank(),
            out.shape().rank()
        )));
    }
    if !Arc::ptr_eq(t.device(), out.device()) {
        return Err(Error::InvalidInput(
            "broadcast_bf16_into: tensors must share a device".into(),
        ));
    }

    let out_shape: Vec<usize> = out.shape().dims().to_vec();
    let dims_i64: Vec<i64> = out_shape.iter().map(|&d| d as i64).collect();
    let strides = out.shape().strides();
    let strides_i64: Vec<i64> = strides.iter().map(|&s| s as i64).collect();

    let stream = default_stream(t);
    let vx = tensor_as_view_bf16(t, "broadcast_bf16_into:in")?;
    let mut vy = tensor_as_view_bf16_mut(out, "broadcast_bf16_into:out")?;

    let status = unsafe {
        fc_bf16_broadcast(
            &vx,
            dims_i64.as_ptr(),
            strides_i64.as_ptr(),
            dims_i64.len() as i32,
            &mut vy,
            stream.as_raw(),
        )
    };
    status_to_result(status, "fc_bf16_broadcast")?;
    Ok(())
}

pub fn repeat_axis_bf16(t: &Tensor, axis: usize, repeats: usize) -> Result<Tensor> {
    ensure_bf16(t, "repeat_axis_bf16:t")?;
    if repeats == 0 {
        return Err(Error::InvalidInput(
            "repeat_axis_bf16: repeats must be > 0".into(),
        ));
    }
    let dims = t.shape().dims();
    if axis >= dims.len() {
        return Err(Error::InvalidInput(format!(
            "repeat_axis_bf16: axis {axis} out of range for shape {:?}",
            dims
        )));
    }

    let mut out_dims = dims.to_vec();
    out_dims[axis] = out_dims[axis] * repeats;
    let mut out =
        Tensor::zeros_dtype(Shape::from_dims(&out_dims), DType::BF16, t.device().clone())?;

    let stream = default_stream(t);
    let vx = tensor_as_view_bf16(t, "repeat_axis_bf16:in")?;
    let mut vy = tensor_as_view_bf16_mut(&mut out, "repeat_axis_bf16:out")?;

    let status =
        unsafe { fc_bf16_repeat_axis(&vx, axis as i32, repeats as i64, &mut vy, stream.as_raw()) };
    status_to_result(status, "fc_bf16_repeat_axis")?;
    Ok(out)
}

pub fn repeat_nd_bf16_into(input: &Tensor, repeats: &[usize], output: &mut Tensor) -> Result<()> {
    ensure_bf16(input, "repeat_nd_bf16_into:input")?;
    ensure_bf16(output, "repeat_nd_bf16_into:output")?;
    if !Arc::ptr_eq(input.device(), output.device()) {
        return Err(Error::InvalidInput(
            "repeat_nd_bf16_into: input and output must share a device".into(),
        ));
    }

    let dims = input.shape().dims();
    if repeats.len() != dims.len() {
        return Err(Error::InvalidInput(format!(
            "repeat_nd_bf16_into: repeats {:?} must match input rank {}",
            repeats,
            dims.len()
        )));
    }

    let mut expected = Vec::with_capacity(dims.len());
    for (dim, repeat) in dims.iter().zip(repeats.iter()) {
        if *repeat == 0 {
            return Err(Error::InvalidInput(
                "repeat_nd_bf16_into: repeats must be > 0".into(),
            ));
        }
        expected.push(dim.checked_mul(*repeat).ok_or_else(|| {
            Error::InvalidInput("repeat_nd_bf16_into: dimension overflow".into())
        })?);
    }
    if output.shape().dims() != expected.as_slice() {
        return Err(Error::InvalidInput(format!(
            "repeat_nd_bf16_into: output shape {:?} does not match expected {:?}",
            output.shape().dims(),
            expected
        )));
    }

    let mut repeats_i64 = Vec::with_capacity(repeats.len());
    for &repeat in repeats {
        let val = i64::try_from(repeat)
            .map_err(|_| Error::InvalidInput("repeat_nd_bf16_into: repeat exceeds i64".into()))?;
        repeats_i64.push(val);
    }

    let stream = default_stream(input);
    let vx = tensor_as_view_bf16(input, "repeat_nd_bf16_into:input")?;
    let mut vy = tensor_as_view_bf16_mut(output, "repeat_nd_bf16_into:output")?;

    let status = unsafe {
        fc_bf16_repeat_nd(
            &vx,
            repeats_i64.as_ptr(),
            repeats_i64.len() as i32,
            &mut vy,
            stream.as_raw(),
        )
    };
    status_to_result(status, "fc_bf16_repeat_nd")
}

pub fn index_select_bf16_into(
    input: &Tensor,
    dim: usize,
    indices: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    ensure_bf16(input, "index_select_bf16_into:input")?;
    ensure_bf16(output, "index_select_bf16_into:output")?;
    if !Arc::ptr_eq(input.device(), output.device()) {
        return Err(Error::InvalidInput(
            "index_select_bf16_into: tensors must share a device".into(),
        ));
    }
    if indices.dtype() != DType::I32 || indices.storage_dtype() != DType::I32 {
        return Err(Error::InvalidInput(
            "index_select_bf16_into: indices must be I32 tensors".into(),
        ));
    }

    let dims = input.shape().dims();
    if dim >= dims.len() {
        return Err(Error::InvalidInput(format!(
            "index_select_bf16_into: dim {} out of range for rank {}",
            dim,
            dims.len()
        )));
    }

    let expected = {
        let mut out_dims = dims.to_vec();
        out_dims[dim] = indices.shape().elem_count();
        out_dims
    };
    if output.shape().dims() != expected.as_slice() {
        return Err(Error::InvalidInput(format!(
            "index_select_bf16_into: output shape {:?} does not match expected {:?}",
            output.shape().dims(),
            expected
        )));
    }

    let indices_slice = indices.storage.try_as_slice_f32().map_err(|_| {
        Error::InvalidInput("index_select_bf16_into: indices storage must be I32".into())
    })?;
    let indices_ptr = *indices_slice.device_ptr() as *const f32;
    let nidx = indices.shape().elem_count();

    let stream = default_stream(input);
    let vx = tensor_as_view_bf16(input, "index_select_bf16_into:input")?;
    let mut vy = tensor_as_view_bf16_mut(output, "index_select_bf16_into:output")?;

    let status = unsafe {
        fc_bf16_index_select(
            &vx,
            dim as i32,
            indices_ptr,
            nidx as i64,
            &mut vy,
            stream.as_raw(),
        )
    };
    status_to_result(status, "fc_bf16_index_select")
}

pub fn conv2d_bf16(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: (i32, i32),
    pad: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
    activation: ConvActivation,
) -> Result<Tensor> {
    ensure_bf16(x, "conv2d_bf16:x")?;
    ensure_bf16(w, "conv2d_bf16:w")?;
    if !Arc::ptr_eq(x.device(), w.device()) {
        return Err(Error::InvalidInput(
            "conv2d_bf16: input and weight tensors must share a device".into(),
        ));
    }

    let x_dims = x.shape().dims();
    let w_dims = w.shape().dims();
    if x_dims.len() != 4 || w_dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "conv2d_bf16: expected NHWC input and KH-KW-IC-OC weights, got {:?} and {:?}",
            x_dims, w_dims
        )));
    }

    let n = x_dims[0];
    let h = x_dims[1];
    let w_in = x_dims[2];
    let c = x_dims[3];
    let kh = w_dims[0];
    let kw = w_dims[1];
    let oc = w_dims[3];

    if groups <= 0 {
        return Err(Error::InvalidInput(
            "conv2d_bf16: groups must be positive".into(),
        ));
    }
    let groups = groups as usize;
    if groups > c || groups > oc {
        return Err(Error::InvalidInput(
            "conv2d_bf16: groups exceeds channel count".into(),
        ));
    }
    if c % groups != 0 {
        return Err(Error::InvalidInput(format!(
            "conv2d_bf16: input channels {} must be divisible by groups {}",
            c, groups
        )));
    }
    if oc % groups != 0 {
        return Err(Error::InvalidInput(format!(
            "conv2d_bf16: output channels {} must be divisible by groups {}",
            oc, groups
        )));
    }
    let cin_per_group = c / groups;
    let cout_per_group = oc / groups;
    let depthwise = groups == c && oc == c && cout_per_group == 1 && cin_per_group == 1;

    if w_dims[2] != cin_per_group {
        return Err(Error::InvalidInput(format!(
            "conv2d_bf16: expected weight Cin/groups {}, got {}",
            cin_per_group, w_dims[2]
        )));
    }

    if w_dims[3] != oc {
        return Err(Error::InvalidInput(format!(
            "conv2d_bf16: expected weight Cout {}, got {}",
            oc, w_dims[3]
        )));
    }

    if kh == 0 || kw == 0 {
        return Err(Error::InvalidInput(
            "conv2d_bf16: kernel dimensions must be positive".into(),
        ));
    }

    if let Some(b) = bias {
        ensure_bf16(b, "conv2d_bf16:bias")?;
        if !Arc::ptr_eq(x.device(), b.device()) {
            return Err(Error::InvalidInput(
                "conv2d_bf16: bias tensor must share device with inputs".into(),
            ));
        }
        let bdims = b.shape().dims();
        if bdims.len() != 1 || bdims[0] != oc {
            return Err(Error::InvalidInput(format!(
                "conv2d_bf16: bias shape {:?} incompatible with OC {}",
                bdims, oc
            )));
        }
    }

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = pad;
    let (dil_h, dil_w) = dilation;
    if stride_h <= 0 || stride_w <= 0 || dil_h <= 0 || dil_w <= 0 {
        return Err(Error::InvalidInput(
            "conv2d_bf16: stride and dilation must be positive".into(),
        ));
    }

    let effective_kh = (kh as i64 - 1) * dil_h as i64 + 1;
    let effective_kw = (kw as i64 - 1) * dil_w as i64 + 1;
    let ho_numer = h as i64 + 2 * pad_h as i64 - effective_kh;
    let wo_numer = w_in as i64 + 2 * pad_w as i64 - effective_kw;
    let ho = if ho_numer < 0 {
        0
    } else {
        ho_numer / stride_h as i64 + 1
    };
    let wo = if wo_numer < 0 {
        0
    } else {
        wo_numer / stride_w as i64 + 1
    };
    if ho < 0 || wo < 0 {
        return Err(Error::InvalidInput(
            "conv2d_bf16: computed negative output dimensions".into(),
        ));
    }

    let mut out = Tensor::zeros_dtype(
        Shape::from_dims(&[n, ho as usize, wo as usize, oc]),
        DType::BF16,
        x.device().clone(),
    )?;

    for (name, value) in [
        ("N", n),
        ("H", h),
        ("W", w_in),
        ("Cin", c),
        ("Kh", kh),
        ("Kw", kw),
        ("Cout", oc),
        ("Hout", ho as usize),
        ("Wout", wo as usize),
    ] {
        if value > i32::MAX as usize {
            return Err(Error::InvalidInput(format!(
                "conv2d_bf16: dimension {name}={value} exceeds i32 range"
            )));
        }
    }

    let stream = default_stream(x);
    let device_idx = x.device().ordinal() as i32;
    let x_ptr = x.as_device_ptr_bf16("conv2d_bf16:x")? as *const c_void;
    let w_ptr = w.as_device_ptr_bf16("conv2d_bf16:w")? as *const c_void;
    let bias_ptr = if let Some(b) = bias {
        b.as_device_ptr_bf16("conv2d_bf16:bias")? as *const c_void
    } else {
        ptr::null()
    };
    let y_ptr = out.as_mut_device_ptr_bf16("conv2d_bf16:out")? as *mut c_void;

    let mut workspace_ptr: *mut c_void = ptr::null_mut();
    let mut workspace_bytes: u64 = 0;
    let use_workspace = !depthwise && n > 0 && ho as usize > 0 && wo as usize > 0;
    let mut fallback_alloc: Option<cudarc::driver::CudaSlice<u8>> = None;
    if use_workspace {
        let workspace_elems = (n as u64)
            .checked_mul(ho as u64)
            .and_then(|v| v.checked_mul(wo as u64))
            .and_then(|v| v.checked_mul(kh as u64))
            .and_then(|v| v.checked_mul(kw as u64))
            .and_then(|v| v.checked_mul(c as u64))
            .ok_or_else(|| {
                Error::InvalidInput("conv2d_bf16: workspace size overflowed u64".into())
            })?;
        workspace_bytes = workspace_elems
            .checked_mul(std::mem::size_of::<u16>() as u64)
            .ok_or_else(|| {
                Error::InvalidInput("conv2d_bf16: workspace byte size overflowed u64".into())
            })?;
        if workspace_bytes > 0 {
            if workspace_bytes > usize::MAX as u64 {
                return Err(Error::InvalidInput(
                    "conv2d_bf16: workspace requirement exceeds platform limits".into(),
                ));
            }
            match arena_alloc(device_idx, &stream, workspace_bytes as usize, 64) {
                Ok(ptr) => {
                    workspace_ptr = ptr as *mut c_void;
                }
                Err(e) => {
                    log::debug!(
                        "conv2d_bf16: arena_alloc failed ({:?}), falling back to device alloc",
                        e
                    );
                    let alloc = unsafe { x.device().alloc(workspace_bytes as usize) }
                        .map_err(|e| Error::Cuda(format!("Fallback alloc failed: {:?}", e)))?;
                    workspace_ptr = *alloc.device_ptr() as *mut c_void;
                    fallback_alloc = Some(alloc);
                }
            }
        }
    }

    let status = unsafe {
        flame_conv2d_nhwc_bf16(
            x_ptr,
            w_ptr,
            bias_ptr,
            n as i32,
            h as i32,
            w_in as i32,
            c as i32,
            kh as i32,
            kw as i32,
            stride_h as i32,
            stride_w as i32,
            pad_h as i32,
            pad_w as i32,
            dil_h as i32,
            dil_w as i32,
            oc as i32,
            activation as i32,
            groups as i32,
            y_ptr,
            workspace_ptr,
            workspace_bytes,
            stream.as_raw(),
        )
    };

    // NOTE: fallback_alloc is dropped at end of scope, but that's safe because
    // all subsequent ops on `out` run on the same CUDA stream and are ordered
    // after the conv kernel. No synchronize needed — CUDA stream ordering
    // guarantees the kernel completes before any op that reads `out`.
    if status != FLAME_CUDA_OK {
        let cuda_err: cudaError_t = unsafe { cudaGetLastError() };
        if cuda_err != cudaError::cudaSuccess {
            unsafe {
                let msg_ptr = cudaGetErrorString(cuda_err);
                if !msg_ptr.is_null() {
                    if let Ok(msg) = CStr::from_ptr(msg_ptr).to_str() {
                        log::error!(
                            "conv2d_bf16 cudaGetLastError={:?} ({msg}) dims={:?} weight={:?} stride={:?} pad={:?} groups={}",
                            cuda_err,
                            x.shape().dims(),
                            w.shape().dims(),
                            (stride_h, stride_w),
                            (pad_h, pad_w),
                            groups
                        );
                    } else {
                        log::error!(
                            "conv2d_bf16 cudaGetLastError={:?} (unprintable) dims={:?} weight={:?}",
                            cuda_err,
                            x.shape().dims(),
                            w.shape().dims()
                        );
                    }
                }
            }
        } else {
            log::error!(
                "conv2d_bf16 received status {:?} with cudaSuccess; dims={:?} weight={:?} stride={:?} pad={:?} groups={}",
                status,
                x.shape().dims(),
                w.shape().dims(),
                (stride_h, stride_w),
                (pad_h, pad_w),
                groups
            );
        }
    }
    if workspace_bytes > 0 {
        arena_record_and_release(device_idx, &stream)?;
    }

    if status == FLAME_CUDA_ERR_UNSUPPORTED {
        warn_stub_once(
            &CONV2D_STUB_WARNED,
            "conv2d_bf16: CUDA kernel reported unsupported configuration",
        );
    }
    flame_status_to_result(status, "flame_conv2d_nhwc_bf16")?;

    Ok(out)
}

pub fn sdpa_stream_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    chunk: usize,
    causal: bool,
    scale: Option<f32>,
) -> Result<Tensor> {
    sdpa_stream_bf16_with_workspace(q, k, v, mask, chunk, causal, scale, None)
}

pub fn sdpa_stream_bf16_with_workspace(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    chunk: usize,
    causal: bool,
    scale: Option<f32>,
    workspace: Option<SdpaWorkspace>,
) -> Result<Tensor> {
    let disable_stream = matches!(
        std::env::var("FLAME_SDPA_DISABLE_STREAM")
            .ok()
            .map(|v| v.to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    );
    struct SdpaLogData {
        b: usize,
        h: usize,
        q_len: usize,
        k_len: usize,
        dh: usize,
        dv: usize,
        mask_heads: usize,
        has_mask: bool,
        workspace_bytes: u64,
        stream_id: u64,
        scale: f32,
    }

    let mut log_data: Option<SdpaLogData> = None;
    let start = Instant::now();

    let result = strict::scope("sdpa_stream_bf16", GuardMode::env_default(), || {
        ensure_bf16(q, "sdpa_stream_bf16:Q")?;
        ensure_bf16(k, "sdpa_stream_bf16:K")?;
        ensure_bf16(v, "sdpa_stream_bf16:V")?;
        if !Arc::ptr_eq(q.device(), k.device()) || !Arc::ptr_eq(q.device(), v.device()) {
            return Err(Error::InvalidInput(
                "sdpa_stream_bf16: Q, K, V tensors must share a device".into(),
            ));
        }
        if chunk == 0 {
            return Err(Error::InvalidInput(
                "sdpa_stream_bf16: chunk size must be > 0".into(),
            ));
        }

        let q_dims = q.shape().dims();
        let k_dims = k.shape().dims();
        let v_dims = v.shape().dims();
        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "sdpa_stream_bf16: expected [B,H,Q,Dh] / [B,H,K,Dh] tensors, got {:?} {:?} {:?}",
                q_dims, k_dims, v_dims
            )));
        }
        let b = q_dims[0];
        let h = q_dims[1];
        let q_len = q_dims[2];
        let dh = q_dims[3];
        if disable_stream {
            return Err(Error::Unsupported(
                "sdpa_stream_bf16 disabled via FLAME_SDPA_DISABLE_STREAM".into(),
            ));
        }
        if k_dims[0] != b || k_dims[1] != h || k_dims[3] != dh {
            return Err(Error::InvalidInput(
                "sdpa_stream_bf16: K tensor shape mismatch".into(),
            ));
        }
        if v_dims[0] != b || v_dims[1] != h || v_dims[2] != k_dims[2] {
            return Err(Error::InvalidInput(
                "sdpa_stream_bf16: V tensor shape mismatch (B/H/K dimensions must match inputs)"
                    .into(),
            ));
        }

        let scale_val = scale.unwrap_or_else(|| 1.0f32 / (dh as f32).sqrt());
        if h > i32::MAX as usize || dh > i32::MAX as usize || chunk > i32::MAX as usize {
            return Err(Error::InvalidInput(
                "sdpa_stream_bf16: configuration exceeds supported range".into(),
            ));
        }

        if let Some(m) = mask {
            if !Arc::ptr_eq(q.device(), m.device()) {
                return Err(Error::InvalidInput(
                    "sdpa_stream_bf16: mask tensor must share device with inputs".into(),
                ));
            }
            let dims = m.shape().dims();
            if dims.len() != 4 {
                return Err(Error::InvalidInput(format!(
                    "sdpa_stream_bf16: mask must be 4D [B, H, Q, K], got {:?}",
                    dims
                )));
            }
        }

        let dv = v_dims[3];
        let out_shape = Shape::from_dims(&[b, h, q_len, dv]);
        let mut out = Tensor::zeros_dtype(out_shape, DType::BF16, q.device().clone())?;

        let stream = default_stream(q);
        let k_len = k_dims[2];
        for (name, value) in [
            ("B", b),
            ("H", h),
            ("Q", q_len),
            ("K", k_len),
            ("Dh", dh),
            ("Dv", dv),
        ] {
            if value > i32::MAX as usize {
                return Err(Error::InvalidInput(format!(
                    "sdpa_stream_bf16: dimension {name}={value} exceeds i32 range"
                )));
            }
        }

        let q_ptr = q.as_device_ptr_bf16("sdpa_stream_bf16:Q")? as *const c_void;
        let k_ptr = k.as_device_ptr_bf16("sdpa_stream_bf16:K")? as *const c_void;
        let v_ptr = v.as_device_ptr_bf16("sdpa_stream_bf16:V")? as *const c_void;
        let out_ptr = out.as_mut_device_ptr_bf16("sdpa_stream_bf16:O")? as *mut c_void;

        let mut _mask_buffer: Option<Tensor> = None;
        let mut mask_heads = 0usize;
        let mut mask_dims: Option<[usize; 4]> = None;
        let mask_ptr = if let Some(m) = mask {
            let converted = if m.dtype() == DType::BF16 {
                m.clone_result()?
            } else if m.dtype() == DType::Bool || m.dtype() == DType::F32 {
                m.to_dtype(DType::BF16)?
            } else {
                return Err(Error::InvalidInput(format!(
                    "sdpa_stream_bf16: unsupported mask dtype {:?} (expected Bool/BF16/F32)",
                    m.dtype()
                )));
            };
            _mask_buffer = Some(converted);
            let dims = _mask_buffer.as_ref().unwrap().shape().dims();
            if dims[1] != 1 && dims[1] != h {
                return Err(Error::InvalidInput(format!(
                    "sdpa_stream_bf16: mask head dimension {} incompatible with H {}",
                    dims[1], h
                )));
            }
            mask_heads = dims[1];
            mask_dims = Some([dims[0], dims[1], dims[2], dims[3]]);
            _mask_buffer
                .as_ref()
                .unwrap()
                .as_device_ptr_bf16("sdpa_stream_bf16:mask")? as *const c_void
        } else {
            ptr::null::<c_void>()
        };

        let (mask_ptr_bf16, mask_stride_ek, mask_stride_eq, mask_stride_eh, mask_stride_eb) =
            if mask_heads == 0 {
                (ptr::null::<u16>(), 0i64, 0i64, 0i64, 0i64)
            } else {
                let dims = mask_dims.expect("mask dimensions must be present when mask_heads > 0");
                let stride_ek = 1i64;
                let stride_eq = dims[3] as i64;
                let stride_eh = if dims[1] == 1 {
                    0
                } else {
                    (dims[2] * dims[3]) as i64
                };
                let stride_eb = if dims[0] == 1 {
                    0
                } else {
                    (dims[1] * dims[2] * dims[3]) as i64
                };
                (
                    mask_ptr as *const u16,
                    stride_ek,
                    stride_eq,
                    stride_eh,
                    stride_eb,
                )
            };

        let parse_env_i32 = |name: &str| -> Option<i32> {
            std::env::var(name).ok().and_then(|v| v.parse::<i32>().ok())
        };
        let mut head_tile = parse_env_i32("FLAME_SDPA_HEAD_TILE").unwrap_or(12);
        if head_tile <= 0 {
            head_tile = 12;
        }
        if h > 0 {
            head_tile = head_tile.max(1).min(h as i32);
        } else {
            head_tile = 1;
        }

        let mut q_tile = parse_env_i32("STREAMING_SDPA_CHUNK_MAX").unwrap_or(96);
        if q_tile <= 0 {
            q_tile = 96;
        }
        let mut max_q_tile = parse_env_i32("FLAME_SDPA_MAX_Q_TILE").unwrap_or(q_tile);
        if max_q_tile <= 0 {
            max_q_tile = q_tile;
        }
        let mut min_q_tile = parse_env_i32("FLAME_SDPA_MIN_Q_TILE").unwrap_or(1);
        if min_q_tile <= 0 {
            min_q_tile = 1;
        }
        min_q_tile = min_q_tile.max(1);
        q_tile = q_tile.max(min_q_tile);
        max_q_tile = max_q_tile.max(min_q_tile);
        let q_len_i32 = q_len as i32;
        q_tile = q_tile.max(1);
        max_q_tile = max_q_tile.max(1);
        if q_len_i32 > 0 {
            q_tile = q_tile.min(q_len_i32);
            max_q_tile = max_q_tile.min(q_len_i32);
        }
        max_q_tile = max_q_tile.max(q_tile).max(1);

        let mut reason_buf = [0 as c_char; 256];
        let launch_ok = unsafe {
            sdpa_stream_bf16_launch(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                b as i32,
                h as i32,
                q_len as i32,
                k_len as i32,
                dh as i32,
                scale_val,
                mask_ptr_bf16,
                mask_stride_ek,
                mask_stride_eq,
                mask_stride_eh,
                mask_stride_eb,
                head_tile,
                q_tile,
                max_q_tile,
                stream.as_raw(),
                reason_buf.as_mut_ptr(),
                reason_buf.len() as i32,
            )
        };

        if launch_ok {
            log_data = Some(SdpaLogData {
                b,
                h,
                q_len,
                k_len,
                dh,
                dv,
                mask_heads,
                has_mask: mask.is_some(),
                workspace_bytes: 0,
                stream_id: stream.as_raw() as usize as u64,
                scale: scale_val,
            });
            return Ok(out);
        }

        let reason = if reason_buf[0] != 0 {
            unsafe { CStr::from_ptr(reason_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        } else {
            String::new()
        };
        if !reason.is_empty() {
            log::warn!(
                "sdpa_stream_bf16_launch unsupported ({}); falling back to chunked kernel (B={} H={} Q={} K={} Dh={} q_tile={} head_tile={})",
                reason,
                b,
                h,
                q_len,
                k_len,
                dh,
                q_tile,
                head_tile
            );
        } else {
            log::warn!(
                "sdpa_stream_bf16_launch unsupported; falling back to chunked kernel (B={} H={} Q={} K={} Dh={} q_tile={} head_tile={})",
                b,
                h,
                q_len,
                k_len,
                dh,
                q_tile,
                head_tile
            );
        }

        let mut workspace_ptr: *mut c_void = ptr::null_mut();
        let mut workspace_bytes: u64 = 0;
        let chunk_rows = chunk.min(q_len);
        if chunk_rows > 0 && k_len > 0 {
            let floats_per_chunk = (chunk_rows as u64)
                .checked_mul(k_len as u64)
                .and_then(|v| v.checked_add((chunk_rows as u64) * (dv as u64 + 2)))
                .ok_or_else(|| {
                    Error::InvalidInput("sdpa_stream_bf16: workspace size overflowed u64".into())
                })?;
            let required_bytes = floats_per_chunk
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| {
                    Error::InvalidInput(
                        "sdpa_stream_bf16: workspace byte size overflowed u64".into(),
                    )
                })?;
            if required_bytes > 0 {
                if let Some(ws) = workspace {
                    if ws.ptr.is_null() {
                        return Err(Error::InvalidInput(
                            "sdpa_stream_bf16: workspace pointer must be non-null".into(),
                        ));
                    }
                    if ws.bytes < required_bytes {
                        return Err(Error::InvalidInput(format!(
                            "sdpa_stream_bf16: provided workspace ({:.2} MiB) smaller than required ({:.2} MiB)",
                            ws.bytes as f64 / (1024.0 * 1024.0),
                            required_bytes as f64 / (1024.0 * 1024.0)
                        )));
                    }
                    workspace_ptr = ws.ptr;
                    workspace_bytes = required_bytes;
                } else {
                    let acquired = acquire_sdpa_workspace(q.device(), &stream, required_bytes)?;
                    workspace_ptr = acquired.ptr;
                    workspace_bytes = acquired.bytes;
                }
            }
        }

        if workspace_bytes == 0 {
            workspace_ptr = ptr::null_mut();
        }

        let status = unsafe {
            flame_sdpa_chunked_bf16(
                q_ptr,
                k_ptr,
                v_ptr,
                b as i32,
                h as i32,
                q_dims[2] as i32,
                k_dims[2] as i32,
                dh as i32,
                v_dims[3] as i32,
                scale_val,
                chunk as i32,
                if causal { 1 } else { 0 },
                mask_heads as i32,
                mask_ptr,
                out_ptr,
                workspace_ptr,
                workspace_bytes,
                stream.as_raw(),
            )
        };
        if status == FLAME_CUDA_ERR_UNSUPPORTED {
            return Err(Error::Unsupported(
                "sdpa_stream_bf16: configuration not supported on this backend".into(),
            ));
        }
        flame_status_to_result(status, "flame_sdpa_chunked_bf16")?;

        log_data = Some(SdpaLogData {
            b,
            h,
            q_len,
            k_len,
            dh,
            dv,
            mask_heads,
            has_mask: mask.is_some(),
            workspace_bytes,
            stream_id: stream.as_raw() as usize as u64,
            scale: scale_val,
        });

        Ok(out)
    });

    if perf_telemetry::telemetry_enabled() {
        if let Ok(ref _tensor) = result {
            if let Some(meta) = log_data {
                if let Ok(stats) = sdpa_autotune_stats() {
                    let elapsed = start.elapsed();
                    let elapsed_secs = elapsed.as_secs_f64();
                    let ops_attn = 2.0
                        * (meta.b as f64)
                        * (meta.h as f64)
                        * (meta.q_len as f64)
                        * (meta.k_len as f64)
                        * (meta.dh as f64);
                    let ops_value = (meta.b as f64)
                        * (meta.h as f64)
                        * (meta.q_len as f64)
                        * (meta.k_len as f64)
                        * (meta.dv as f64);
                    let ops_total = ops_attn + ops_value;
                    let tflops = if elapsed_secs > 0.0 {
                        (ops_total / elapsed_secs) / 1.0e12
                    } else {
                        0.0
                    };
                    perf_telemetry::log_sdpa_event(perf_telemetry::SdpaTelemetryRecord {
                        batch: meta.b,
                        heads: meta.h,
                        q_len: meta.q_len,
                        k_len: meta.k_len,
                        dh: meta.dh,
                        dv: meta.dv,
                        chunk_requested: chunk,
                        tuned_q_chunk: stats.last_q_chunk,
                        tuned_k_chunk: stats.last_k_chunk,
                        plan_source: stats.last_plan_source,
                        mask_heads: meta.mask_heads,
                        has_mask: meta.has_mask,
                        causal,
                        workspace_bytes: meta.workspace_bytes,
                        stream_id: meta.stream_id,
                        elapsed_ms: elapsed_secs * 1.0e3,
                        tflops,
                        scale: meta.scale,
                        candidate_count: stats.last_candidate_count,
                        best_time_ns: stats.last_best_time_ns,
                        autotune_env_forced: stats.env_forced,
                        autotune_clamped: stats.clamped,
                        autotune_skipped: stats.skipped,
                        autotune_fallback: stats.fallback,
                        autotune_errors: stats.errors,
                        autotune_tuned: stats.tuned,
                        cache_hits: stats.cache_hits,
                        cache_misses: stats.cache_misses,
                        cache_saved: stats.cache_saved,
                        cache_loads: stats.cache_loads,
                        cache_load_errors: stats.cache_load_errors,
                        cache_entries: stats.cache_entries,
                    });
                }
            }
        }
    }

    result
}

pub fn conv2d_autotune_stats() -> Result<Conv2dAutotuneStats> {
    staging_conv2d_stats()
}

pub fn reset_conv2d_autotune_stats() -> Result<()> {
    staging_reset_conv2d_stats()
}

pub fn sdpa_autotune_stats() -> Result<SdpaAutotuneStats> {
    staging_sdpa_stats()
}

pub fn reset_sdpa_autotune_stats() -> Result<()> {
    staging_reset_sdpa_stats()
}

pub fn flush_sdpa_autotune_cache() -> Result<()> {
    staging_flush_sdpa_cache()
}
