//! Fused CUDA kernels for inference: RMS norm, modulation, linear3d.
//! Each replaces multiple kernel launches with one.

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda::device_lt;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::DType;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::DevicePtr;
use crate::{Error, Result, Shape, Tensor};

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

    let output = Tensor::zeros_dtype(input.shape().clone(), DType::BF16, input.device().clone())?;

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
    let output = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;

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
    let output = Tensor::zeros_dtype(out_shape, DType::BF16, input.device().clone())?;

    let device = input.device();
    let stream = device_lt::stream_ptr(device)?;
    let lt = device_lt::cublaslt_handle_ptr(device)?;

    // cublasLt workspace
    let workspace_size: usize = 4 * 1024 * 1024; // 4MB
    let workspace: cudarc::driver::CudaSlice<u8> = unsafe { device.alloc(workspace_size)? };

    let bias_ptr = if let Some(b) = bias {
        b.as_device_ptr_bf16("fused_linear3d:bias")? as *const _
    } else {
        std::ptr::null()
    };

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
            *workspace.device_ptr() as *mut _,
            workspace_size,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_linear3d cublasLt error: {ret}")));
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

    let output = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
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
    let output = Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
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
