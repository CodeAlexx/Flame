#![allow(dead_code)]

use core::ffi::c_void;
use std::os::raw::{c_char, c_int, c_longlong};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FcTensorView {
    pub data: *mut c_void,
    pub dims: [i64; 8],
    pub strides: [i64; 8],
    pub rank: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FcWorkspace {
    pub ptr: *mut c_void,
    pub bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FcSdpaCfg {
    pub heads: i32,
    pub head_dim: i32,
    pub chunk: i32,
    pub scale: f32,
    pub causal: i32,
}

#[repr(C)]
#[derive(Debug)]
pub struct FlameStreamArenaOpaque {
    _private: [u8; 0],
}

pub type FlameStreamArenaHandle = *mut FlameStreamArenaOpaque;

pub const FLAME_CUDA_OK: i32 = 0;
pub const FLAME_CUDA_ERR_INVALID: i32 = 1;
pub const FLAME_CUDA_ERR_UNSUPPORTED: i32 = 2;
pub const FLAME_CUDA_ERR_CUDA: i32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FlameConv2dAutotuneStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tuned: u64,
    pub weak: u64,
    pub fallbacks: u64,
    pub workspace_skips: u64,
    pub errors: u64,
    pub reprobes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FlameSdpaAutotuneStats {
    pub env_forced: u64,
    pub clamped: u64,
    pub skipped: u64,
    pub fallback: u64,
    pub errors: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tuned: u64,
    pub last_q_chunk: u64,
    pub last_k_chunk: u64,
    pub cache_saved: u64,
    pub cache_loads: u64,
    pub cache_load_errors: u64,
    pub cache_entries: u64,
    pub last_candidate_count: u64,
    pub last_best_time_ns: u64,
    pub last_plan_source: u64,
    pub last_shape_b: u64,
    pub last_shape_h: u64,
    pub last_shape_q: u64,
    pub last_shape_k: u64,
    pub last_shape_dh: u64,
    pub last_shape_dv: u64,
    pub last_shape_mask_heads: u64,
    pub last_shape_causal: u64,
}

extern "C" {
    pub fn fc_ws_ensure_capacity(ws: *mut FcWorkspace, bytes: usize, stream: *mut c_void) -> i32;
    pub fn fc_relu_bf16(x: *const FcTensorView, y: *mut FcTensorView, stream: *mut c_void) -> i32;
    pub fn fc_gelu_bf16(x: *const FcTensorView, y: *mut FcTensorView, stream: *mut c_void) -> i32;
    pub fn fc_silu_bf16(x: *const FcTensorView, y: *mut FcTensorView, stream: *mut c_void) -> i32;
    pub fn fc_axpby_bf16(
        x: *const FcTensorView,
        a: f32,
        y: *mut FcTensorView,
        b: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_layer_norm_bf16(
        x: *const FcTensorView,
        gamma: *const FcTensorView,
        beta: *const FcTensorView,
        norm_size: c_longlong,
        eps: f32,
        y: *mut FcTensorView,
        mean: *mut f32,
        rstd: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_layer_norm_backward_bf16(
        x: *const u16,
        dy: *const u16,
        gamma: *const u16,
        outer_size: c_longlong,
        norm_size: c_longlong,
        eps: f32,
        dx: *mut u16,
        dgamma: *mut f32,
        dbeta: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_group_norm_bf16(
        x: *const FcTensorView,
        gamma: *const FcTensorView,
        beta: *const FcTensorView,
        groups: i32,
        eps: f32,
        y: *mut FcTensorView,
        mean: *mut f32,
        var: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_group_norm_backward_bf16(
        x: *const u16,
        dy: *const u16,
        gamma: *const u16,
        batch_size: c_longlong,
        channels: c_longlong,
        spatial_size: c_longlong,
        group_count: c_int,
        eps: f32,
        dx: *mut u16,
        dgamma: *mut f32,
        dbeta: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_rms_norm_bf16(
        x: *const FcTensorView,
        weight: *const FcTensorView,
        eps: f32,
        y: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_gemm_bf16(
        x: *const FcTensorView,
        w: *const FcTensorView,
        bias: *const FcTensorView,
        y: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_batched_gemm_bf16(
        a: *const FcTensorView,
        b: *const FcTensorView,
        bias: *const FcTensorView,
        c: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_conv2d_bf16(
        x: *const FcTensorView,
        w: *const FcTensorView,
        bias: *const FcTensorView,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        dil_h: i32,
        dil_w: i32,
        y: *mut FcTensorView,
        ws: *mut FcWorkspace,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_sdpa_stream_bf16(
        q: *const FcTensorView,
        k: *const FcTensorView,
        v: *const FcTensorView,
        mask: *const FcTensorView,
        cfg: *const FcSdpaCfg,
        ws: *mut FcWorkspace,
        o: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn sdpa_stream_bf16_launch(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut c_void,
        batch: c_int,
        heads: c_int,
        q_len: c_int,
        k_len: c_int,
        head_dim: c_int,
        scale: f32,
        mask: *const u16,
        mask_stride_ek: i64,
        mask_stride_eq: i64,
        mask_stride_eh: i64,
        mask_stride_eb: i64,
        head_tile: c_int,
        q_tile: c_int,
        max_q_tile: c_int,
        stream: *mut c_void,
        unsupported_reason: *mut c_char,
        reason_buflen: c_int,
    ) -> bool;
    pub fn fc_bf16_slice(
        x: *const FcTensorView,
        axis: i32,
        start: i64,
        len: i64,
        y: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_bf16_index_select(
        x: *const FcTensorView,
        axis: i32,
        indices: *const f32,
        nidx: i64,
        y: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_bf16_broadcast(
        x: *const FcTensorView,
        out_dims: *const i64,
        out_strides: *const i64,
        rank: i32,
        y: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_bf16_repeat_axis(
        x: *const FcTensorView,
        axis: i32,
        repeats: i64,
        y: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_bf16_repeat_nd(
        x: *const FcTensorView,
        repeats: *const i64,
        rank: i32,
        y: *mut FcTensorView,
        stream: *mut c_void,
    ) -> i32;
    pub fn fc_bf16_memcpy_async(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_arena_create(
        device: i32,
        stream: *mut c_void,
        capacity_bytes: u64,
        out: *mut FlameStreamArenaHandle,
    ) -> i32;
    pub fn flame_arena_reset(arena: FlameStreamArenaHandle) -> i32;
    pub fn flame_arena_alloc(
        arena: FlameStreamArenaHandle,
        bytes: u64,
        align: u64,
        out_ptr: *mut *mut c_void,
    ) -> i32;
    pub fn flame_arena_record_and_release(arena: FlameStreamArenaHandle) -> i32;
    pub fn flame_arena_destroy(arena: FlameStreamArenaHandle) -> i32;
    pub fn flame_h2d_async(
        dst_device: *mut c_void,
        src_host: *const c_void,
        bytes: u64,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_d2h_async(
        dst_host: *mut c_void,
        src_device: *const c_void,
        bytes: u64,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_d2d_async(
        dst_device: *mut c_void,
        src_device: *const c_void,
        bytes: u64,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_bf16_zero_async(dst: *mut c_void, elems: u64, stream: *mut c_void) -> i32;
    pub fn flame_bf16_copy_async(
        dst: *mut c_void,
        src: *const c_void,
        elems: u64,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_conv2d_autotune_get_stats(out: *mut FlameConv2dAutotuneStats) -> i32;
    pub fn flame_conv2d_autotune_reset_stats() -> i32;
    pub fn flame_sdpa_autotune_get_stats(out: *mut FlameSdpaAutotuneStats) -> i32;
    pub fn flame_sdpa_autotune_reset_stats() -> i32;
    pub fn flame_sdpa_autotune_flush_cache() -> i32;
    pub fn flame_sdpa_chunked_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        b: i32,
        h: i32,
        q_len: i32,
        k_len: i32,
        dh: i32,
        dv: i32,
        scale: f32,
        chunk: i32,
        causal: i32,
        mask_heads: i32,
        mask: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_nhwc_to_nchw_f32(
        input: *const f32,
        output: *mut f32,
        N: i32,
        H: i32,
        W: i32,
        C: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_nchw_to_nhwc_f32(
        input: *const f32,
        output: *mut f32,
        N: i32,
        C: i32,
        H: i32,
        W: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_nhwc_to_nchw_bf16(
        input: *const c_void,
        output: *mut c_void,
        N: i32,
        H: i32,
        W: i32,
        C: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_nchw_to_nhwc_bf16(
        input: *const c_void,
        output: *mut c_void,
        N: i32,
        C: i32,
        H: i32,
        W: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn flame_conv2d_nhwc_bf16(
        x: *const c_void,
        w: *const c_void,
        bias: *const c_void,
        N: i32,
        H: i32,
        W: i32,
        Cin: i32,
        Kh: i32,
        Kw: i32,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        dil_h: i32,
        dil_w: i32,
        Cout: i32,
        activation: i32,
        groups: i32,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> i32;
}

/// Placeholder stream wrapper until the high-level API wires this in.
#[derive(Clone, Copy, Debug)]
pub struct CudaStream {
    raw: *mut c_void,
}

impl CudaStream {
    pub const fn from_raw(raw: *mut c_void) -> Self {
        Self { raw }
    }

    pub const fn as_raw(&self) -> *mut c_void {
        self.raw
    }
}

use crate::{DType, Error, Result, Tensor};

/// Convert a BF16 tensor into the FFI view structure.
pub fn tensor_as_view_bf16(tensor: &Tensor, tag: &str) -> Result<FcTensorView> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "tensor_as_view_bf16: expected BF16 storage".into(),
        ));
    }
    let shape = tensor.shape();
    let rank = shape.rank() as i32;
    let mut dims = [0i64; 8];
    let mut strides = [0i64; 8];
    for i in 0..rank as usize {
        dims[i] = shape.dims()[i] as i64;
        strides[i] = shape.strides()[i] as i64;
    }
    let ptr = tensor.as_device_ptr_bf16(tag)? as *mut c_void;
    Ok(FcTensorView {
        data: ptr,
        dims,
        strides,
        rank,
    })
}

pub fn tensor_as_view_bf16_mut(tensor: &mut Tensor, tag: &str) -> Result<FcTensorView> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "tensor_as_view_bf16_mut: expected BF16 storage".into(),
        ));
    }
    let shape = tensor.shape();
    let rank = shape.rank() as i32;
    let mut dims = [0i64; 8];
    let mut strides = [0i64; 8];
    for i in 0..rank as usize {
        dims[i] = shape.dims()[i] as i64;
        strides[i] = shape.strides()[i] as i64;
    }
    let ptr = tensor.as_mut_device_ptr_bf16(tag)? as *mut c_void;
    Ok(FcTensorView {
        data: ptr,
        dims,
        strides,
        rank,
    })
}

pub fn tensor_as_flat_view_bf16(
    tensor: &Tensor,
    leading: usize,
    trailing: usize,
    tag: &str,
) -> Result<FcTensorView> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "tensor_as_flat_view_bf16: expected BF16 storage".into(),
        ));
    }
    if tensor.shape().elem_count() != leading * trailing {
        return Err(Error::InvalidInput(format!(
            "tensor_as_flat_view_bf16: element count mismatch ({} != {} * {})",
            tensor.shape().elem_count(),
            leading,
            trailing
        )));
    }
    let ptr = tensor.as_device_ptr_bf16(tag)? as *mut c_void;
    let mut dims = [0i64; 8];
    let mut strides = [0i64; 8];
    dims[0] = leading as i64;
    dims[1] = trailing as i64;

    // Stride for the trailing dimension is always 1 for contiguous BF16 tensors.
    strides[1] = 1;
    strides[0] = if tensor.shape().rank() >= 2 {
        tensor.shape().strides()[tensor.shape().rank() - 2] as i64
    } else {
        trailing as i64
    };

    Ok(FcTensorView {
        data: ptr,
        dims,
        strides,
        rank: 2,
    })
}

pub fn tensor_as_flat_view_bf16_mut(
    tensor: &mut Tensor,
    leading: usize,
    trailing: usize,
    tag: &str,
) -> Result<FcTensorView> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "tensor_as_flat_view_bf16_mut: expected BF16 storage".into(),
        ));
    }
    if tensor.shape().elem_count() != leading * trailing {
        return Err(Error::InvalidInput(format!(
            "tensor_as_flat_view_bf16_mut: element count mismatch ({} != {} * {})",
            tensor.shape().elem_count(),
            leading,
            trailing
        )));
    }
    let ptr = tensor.as_mut_device_ptr_bf16(tag)? as *mut c_void;
    let mut dims = [0i64; 8];
    let mut strides = [0i64; 8];
    dims[0] = leading as i64;
    dims[1] = trailing as i64;
    strides[1] = 1;
    strides[0] = if tensor.shape().rank() >= 2 {
        tensor.shape().strides()[tensor.shape().rank() - 2] as i64
    } else {
        trailing as i64
    };

    Ok(FcTensorView {
        data: ptr,
        dims,
        strides,
        rank: 2,
    })
}

pub fn flame_status_to_result(status: i32, op: &str) -> Result<()> {
    match status {
        FLAME_CUDA_OK => Ok(()),
        FLAME_CUDA_ERR_INVALID => Err(Error::InvalidInput(format!("{op}: invalid argument"))),
        FLAME_CUDA_ERR_UNSUPPORTED => Err(Error::Unsupported(format!("{op}: unsupported"))),
        FLAME_CUDA_ERR_CUDA => Err(Error::Cuda(format!("{op}: CUDA failure (status={status})"))),
        other => Err(Error::Cuda(format!("{op}: unexpected status {other}"))),
    }
}

/// Workspace arena wrapper; grows once and reuses allocation.
#[derive(Debug)]
pub struct WorkspaceArena {
    raw: FcWorkspace,
}

impl WorkspaceArena {
    pub const fn new() -> Self {
        Self {
            raw: FcWorkspace {
                ptr: core::ptr::null_mut(),
                bytes: 0,
            },
        }
    }

    pub fn ensure(&mut self, bytes: usize, stream: &CudaStream) -> Result<()> {
        let rc = unsafe { fc_ws_ensure_capacity(&mut self.raw, bytes, stream.as_raw()) };
        if rc != 0 {
            return Err(Error::Cuda(format!("fc_ws_ensure_capacity rc={rc}")));
        }
        Ok(())
    }

    pub fn as_mut_ptr(&mut self) -> *mut FcWorkspace {
        &mut self.raw
    }

    pub fn device_ptr(&self) -> *mut c_void {
        self.raw.ptr
    }

    pub fn bytes(&self) -> usize {
        self.raw.bytes
    }
}
