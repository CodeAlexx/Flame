//! Fused CUDA kernels for inference: RMS norm, modulation, linear3d.
//! Each replaces multiple kernel launches with one.

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda::device_lt;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::DType;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::DevicePtr;
use crate::{Error, Result, Shape, Tensor};

/// Persistent per-device cuBLASLt workspace for `fused_linear3d*`.
///
/// Both `fused_linear3d` and `fused_linear3d_native` used to
/// `device.alloc::<u8>(4 MiB)` on every call. Klein 9B issues hundreds of
/// linear calls per step, so that's hundreds of cudaMalloc/cudaFree
/// cycles per step. This cache allocates once per device (on first use)
/// and hands out the cached pointer on every call. The C-side shim
/// (fused_linear3d.cu) explicitly says workspace ownership is the
/// caller's concern — this is that cache on the caller side.
///
/// Thread-safety: the lock is held for the duration of the downstream
/// FFI call to cublasLtMatmul. That's microseconds, and flame-core is
/// single-device-per-process for inference; if multi-threaded training
/// ever needs to call these concurrently, we'd want per-stream
/// workspaces anyway (not per-device), so this design doesn't preclude
/// that future extension.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
mod linear_workspace {
    use super::*;
    use cudarc::driver::DeviceSlice;
    use std::sync::{Arc, Mutex, OnceLock};

    struct Entry {
        device: Arc<cudarc::driver::CudaDevice>,
        slice: cudarc::driver::CudaSlice<u8>,
    }

    static CACHE: OnceLock<Mutex<Option<Entry>>> = OnceLock::new();

    /// Returns a guard holding the workspace pointer + size. The mutex
    /// stays locked for the guard's lifetime; callers should drop the
    /// guard immediately after the FFI call completes.
    pub(super) fn acquire(
        device: &Arc<cudarc::driver::CudaDevice>,
        min_bytes: usize,
    ) -> Result<Guard> {
        let mutex = CACHE.get_or_init(|| Mutex::new(None));
        let mut guard = mutex
            .lock()
            .map_err(|_| Error::InvalidOperation("linear workspace mutex poisoned".into()))?;

        let needs_alloc = match guard.as_ref() {
            None => true,
            Some(entry) => {
                !Arc::ptr_eq(&entry.device, device) || entry.slice.len() < min_bytes
            }
        };
        if needs_alloc {
            let new_slice: cudarc::driver::CudaSlice<u8> =
                unsafe { device.alloc(min_bytes) }
                    .map_err(|e| Error::Cuda(format!("linear workspace alloc failed: {e:?}")))?;
            *guard = Some(Entry { device: device.clone(), slice: new_slice });
        }
        // SAFETY: guard now holds a Some(Entry) that satisfies the size.
        // We store the raw pointer + size; the guard keeps the mutex held
        // so the CudaSlice isn't replaced while the caller uses the pointer.
        let entry = guard.as_ref().unwrap();
        let ptr = *entry.slice.device_ptr() as *mut u8;
        let size = entry.slice.len();
        Ok(Guard { _guard: guard, ptr, size })
    }

    pub(super) struct Guard<'a> {
        _guard: std::sync::MutexGuard<'a, Option<Entry>>,
        ptr: *mut u8,
        size: usize,
    }

    impl<'a> Guard<'a> {
        pub(super) fn ptr(&self) -> *mut u8 { self.ptr }
        pub(super) fn size(&self) -> usize { self.size }
    }
}

/// GPU-side FP8 E4M3 → BF16 dequantization.
/// Input: raw FP8 bytes on GPU (CudaSlice<u8>), scale, shape.
/// Output: new BF16 Tensor.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn dequant_fp8_to_bf16(
    fp8_data: &cudarc::driver::CudaSlice<u8>,
    scale: f32,
    shape: Shape,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let numel = shape.elem_count();
    let bf16_out: cudarc::driver::CudaSlice<u16> = unsafe { device.alloc(numel)? };
    let stream = device_lt::stream_ptr(device)?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fp8_to_bf16(
            *fp8_data.device_ptr() as *const _,
            *bf16_out.device_ptr() as *mut _,
            scale,
            numel,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!("fp8_to_bf16 CUDA error: {ret}")));
    }

    Ok(Tensor::from_bf16_slice_gpu(bf16_out, shape, std::sync::Arc::clone(device)))
}

/// GPU-side FP8 E4M3 → BF16 dequantization INTO an existing Tensor.
/// Zero allocation — writes directly into the output tensor's GPU memory.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn dequant_fp8_to_bf16_into(
    fp8_data: &cudarc::driver::CudaSlice<u8>,
    scale: f32,
    output: &Tensor,
) -> Result<()> {
    let numel = output.shape().elem_count();
    let stream = device_lt::stream_ptr(output.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fp8_to_bf16(
            *fp8_data.device_ptr() as *const _,
            output.as_device_ptr_bf16("dequant_into:output")? as *mut _,
            scale,
            numel,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!("fp8_to_bf16_into CUDA error: {ret}")));
    }
    Ok(())
}

/// Fused FP8 E4M3 dequant + transpose into a pre-allocated BF16 tensor.
/// Reads [M, N] row-major FP8 data, writes [N, M] row-major BF16.
/// One kernel launch, zero allocation.
///
/// - `fp8_data`: FP8 bytes on GPU, length = M * N
/// - `scale`: dequant scale factor
/// - `output`: pre-allocated BF16 tensor with shape [N, M]
/// - `m`: rows of the FP8 input (out_features)
/// - `n`: cols of the FP8 input (in_features)
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn dequant_fp8_transpose_into(
    fp8_data: &cudarc::driver::CudaSlice<u8>,
    scale: f32,
    output: &Tensor,
    m: usize,
    n: usize,
) -> Result<()> {
    let expected = m * n;
    let out_elems = output.shape().elem_count();
    if out_elems != expected {
        return Err(Error::InvalidShape(format!(
            "dequant_fp8_transpose_into: output has {out_elems} elements, expected {expected} (N={n} x M={m})"
        )));
    }

    let stream = device_lt::stream_ptr(output.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_dequant_transpose_bf16(
            *fp8_data.device_ptr() as *const _,
            output.as_device_ptr_bf16("dequant_transpose:output")? as *mut _,
            scale,
            m as i32,
            n as i32,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!(
            "flame_fused_dequant_transpose_bf16 CUDA error: {ret}"
        )));
    }
    Ok(())
}

/// Fused RMS normalization: BF16 → BF16 with weight multiply.
/// Replaces 6 kernel launches (cast + sq + mean + rsqrt + mul + mul_weight) with 1.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_rms_norm: input and weight must be BF16".into(),
        ));
    }

    let dims = input.shape().dims();
    let cols = *dims.last().ok_or_else(|| {
        Error::InvalidInput("fused_rms_norm: empty shape".into())
    })?;
    let rows = input.shape().elem_count() / cols;

    let output = Tensor::empty_dtype(input.shape().clone(), DType::BF16, input.device().clone())?;

    let stream = device_lt::stream_ptr(input.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_rms_norm_bf16(
            input.as_device_ptr_bf16("fused_rms_norm:input")? as *const _,
            weight.as_device_ptr_bf16("fused_rms_norm:weight")? as *const _,
            output.as_device_ptr_bf16("fused_rms_norm:output")? as *mut _,
            rows as i32,
            cols as i32,
            eps,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_rms_norm CUDA error: {ret}")));
    }

    Ok(output)
}

/// Fused modulation: out = x * (1 + scale) + shift. All BF16.
/// Replaces 4 kernel launches (add_scalar + cast + mul + add) with 1.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::BF16 || scale.dtype() != DType::BF16 || shift.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_modulate: all inputs must be BF16".into(),
        ));
    }

    let n = x.shape().elem_count();
    let output = Tensor::empty_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;

    let stream = device_lt::stream_ptr(x.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_modulate_bf16(
            x.as_device_ptr_bf16("fused_modulate:x")? as *const _,
            scale.as_device_ptr_bf16("fused_modulate:scale")? as *const _,
            shift.as_device_ptr_bf16("fused_modulate:shift")? as *const _,
            output.as_device_ptr_bf16("fused_modulate:output")? as *mut _,
            n,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_modulate CUDA error: {ret}")));
    }

    Ok(output)
}

/// Fused 3D linear: [B, N, Cin] @ [Cin, Cout] + bias = [B, N, Cout].
/// No reshape kernels. Bias fused into cublasLt GEMM epilogue.
/// Weight must be PRE-TRANSPOSED to [Cin, Cout] (same as existing linear3d).
/// Replaces 4 launches (reshape + gemm + reshape + bias_add) with 1 cublasLt call.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_linear3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_linear3d: input and weight must be BF16".into(),
        ));
    }

    let in_shape = input.shape().dims();
    if in_shape.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d: input must be 3D [B,N,Cin], got {:?}",
            in_shape
        )));
    }
    let batch_size = in_shape[0];
    let seq_len = in_shape[1];
    let in_features = in_shape[2];

    let w_shape = weight.shape().dims();
    if w_shape.len() != 2 || w_shape[0] != in_features {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d: weight must be [Cin={in_features},Cout] (pre-transposed), got {:?}",
            w_shape
        )));
    }
    let out_features = w_shape[1];

    let out_shape = Shape::from_dims(&[batch_size, seq_len, out_features]);
    let output = Tensor::empty_dtype(out_shape, DType::BF16, input.device().clone())?;

    let device = input.device();
    let stream = device_lt::stream_ptr(device)?;
    let lt = device_lt::cublaslt_handle_ptr(device)?;

    let bias_ptr = if let Some(b) = bias {
        b.as_device_ptr_bf16("fused_linear3d:bias")? as *const _
    } else {
        std::ptr::null()
    };

    let workspace_size: usize = 4 * 1024 * 1024;
    let ws = linear_workspace::acquire(device, workspace_size)?;

    let ret = unsafe {
        crate::cuda::ffi::flame_linear3d_bf16(
            lt,
            input.as_device_ptr_bf16("fused_linear3d:input")? as *const _,
            weight.as_device_ptr_bf16("fused_linear3d:weight")? as *const _,
            bias_ptr,
            output.as_device_ptr_bf16("fused_linear3d:output")? as *mut _,
            batch_size as i32,
            seq_len as i32,
            in_features as i32,
            out_features as i32,
            ws.ptr() as *mut _,
            ws.size(),
            stream,
        )
    };
    drop(ws);

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_linear3d cublasLt error: {ret}")));
    }

    Ok(output)
}

/// Fused 3D linear with bias epilogue, accepting the weight in **standard
/// PyTorch `[Cout, Cin]` row-major layout** (no pre-transpose required).
///
/// Internally uses cuBLASLt with TRANSA=T so the transpose happens inside the
/// GEMM, eliminating the per-call `transpose2d_bf16` pass that the FLUX
/// blocks were paying. For `single_blocks.linear1` (3072 → 21504) the
/// transpose alone was ~10–15 ms — this function gets it back.
///
/// `weight` shape: `[out_features, in_features]` BF16 (PyTorch nn.Linear default).
/// `bias` shape:   `[out_features]` BF16, optional.
/// `input` shape:  `[B, N, in_features]` BF16.
/// Returns `[B, N, out_features]`.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_linear3d_native(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_linear3d_native: input and weight must be BF16".into(),
        ));
    }

    let in_shape = input.shape().dims();
    if in_shape.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d_native: input must be 3D [B,N,Cin], got {:?}",
            in_shape
        )));
    }
    let batch_size = in_shape[0];
    let seq_len = in_shape[1];
    let in_features = in_shape[2];

    let w_shape = weight.shape().dims();
    if w_shape.len() != 2 || w_shape[1] != in_features {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d_native: weight must be [Cout, Cin={in_features}] (PyTorch layout), got {:?}",
            w_shape
        )));
    }
    let out_features = w_shape[0];

    let out_shape = Shape::from_dims(&[batch_size, seq_len, out_features]);
    let output = Tensor::empty_dtype(out_shape, DType::BF16, input.device().clone())?;

    let device = input.device();
    let stream = device_lt::stream_ptr(device)?;
    let lt = device_lt::cublaslt_handle_ptr(device)?;

    let bias_ptr = if let Some(b) = bias {
        b.as_device_ptr_bf16("fused_linear3d_native:bias")? as *const _
    } else {
        std::ptr::null()
    };

    let workspace_size: usize = 4 * 1024 * 1024;
    let ws = linear_workspace::acquire(device, workspace_size)?;

    let ret = unsafe {
        crate::cuda::ffi::flame_linear3d_bf16_native(
            lt,
            input.as_device_ptr_bf16("fused_linear3d_native:input")? as *const _,
            weight.as_device_ptr_bf16("fused_linear3d_native:weight")? as *const _,
            bias_ptr,
            output.as_device_ptr_bf16("fused_linear3d_native:output")? as *mut _,
            batch_size as i32,
            seq_len as i32,
            in_features as i32,
            out_features as i32,
            ws.ptr() as *mut _,
            ws.size(),
            stream,
        )
    };
    drop(ws);

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_linear3d_native cublasLt error: {ret}")));
    }

    Ok(output)
}

/// Fused RMS norm + modulation in one kernel.
/// out = rms_norm(x, weight) * (1 + scale) + shift
/// Replaces fused_rms_norm + fused_modulate (2 launches → 1).
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_rms_norm_modulate(
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput("fused_rms_norm_modulate: inputs must be BF16".into()));
    }

    let dims = x.shape().dims();
    let cols = *dims.last().ok_or_else(|| Error::InvalidInput("empty shape".into()))?;
    let rows = x.shape().elem_count() / cols;

    let output = Tensor::empty_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    let stream = device_lt::stream_ptr(x.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_rms_norm_modulate_bf16(
            x.as_device_ptr_bf16("fused_norm_mod:x")? as *const _,
            weight.as_device_ptr_bf16("fused_norm_mod:w")? as *const _,
            scale.as_device_ptr_bf16("fused_norm_mod:scale")? as *const _,
            shift.as_device_ptr_bf16("fused_norm_mod:shift")? as *const _,
            output.as_device_ptr_bf16("fused_norm_mod:out")? as *mut _,
            rows as i32, cols as i32, eps, stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_rms_norm_modulate CUDA error: {ret}")));
    }
    Ok(output)
}

/// Fused residual + gating: out = x + gate * attn_out.
/// Replaces mul + add (2 launches → 1).
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_residual_gate(
    x: &Tensor,
    attn_out: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput("fused_residual_gate: inputs must be BF16".into()));
    }

    let n = x.shape().elem_count();
    let output = Tensor::empty_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    let stream = device_lt::stream_ptr(x.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_residual_gate_bf16(
            x.as_device_ptr_bf16("fused_res_gate:x")? as *const _,
            attn_out.as_device_ptr_bf16("fused_res_gate:attn")? as *const _,
            gate.as_device_ptr_bf16("fused_res_gate:gate")? as *const _,
            output.as_device_ptr_bf16("fused_res_gate:out")? as *mut _,
            n, stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_residual_gate CUDA error: {ret}")));
    }
    Ok(output)
}
