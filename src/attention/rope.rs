use crate::cuda::ffi;
use crate::device::CudaStreamRawPtrExt;
use crate::{DType, Error, Result, Tensor};

fn ensure_bf16(tensor: &Tensor, op: &str) -> Result<()> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            format!("{op} expects BF16 storage, got {:?}", tensor.dtype()).into(),
        ));
    }
    Ok(())
}

pub fn apply_rope(
    tensor: &Tensor,
    rope_dim: usize,
    base_theta: f32,
    pos_offset: i32,
) -> Result<Tensor> {
    ensure_bf16(tensor, "apply_rope")?;

    if rope_dim == 0 {
        return tensor.clone_result();
    }

    let dims = tensor.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(
            format!("RoPE expects [B,H,S,Dh] tensor, got {:?}", dims).into(),
        ));
    }

    let b = dims[0];
    let h = dims[1];
    let s = dims[2];
    let dh = dims[3];

    if rope_dim > dh || rope_dim % 2 != 0 {
        return Err(Error::InvalidInput(
            format!(
                "Invalid rope_dim {} for head dim {} (must be <= Dh and even)",
                rope_dim, dh
            )
            .into(),
        ));
    }

    let mut output =
        Tensor::zeros_dtype(tensor.shape().clone(), DType::BF16, tensor.device().clone())?;

    let stream = tensor.device().cuda_stream_raw_ptr();
    let status = unsafe {
        ffi::flame_rope_apply_bf16_fp32(
            tensor.as_device_ptr_bf16("rope_input")? as *const _,
            output.as_mut_device_ptr_bf16("rope_output")? as *mut _,
            b as i32,
            h as i32,
            s as i32,
            dh as i32,
            rope_dim as i32,
            base_theta,
            pos_offset,
            stream,
        )
    };

    if status != 0 {
        return Err(Error::Cuda("flame_rope_apply_bf16_fp32 failed".into()));
    }

    Ok(output)
}

/// Fused RoPE with precomputed cos/sin buffers.
///
/// x:   [B, H, S, D] BF16 — the tensor to rotate
/// cos: [1, H, S, D/2] BF16 — precomputed cosines (pre-expanded to H heads)
/// sin: [1, H, S, D/2] BF16 — precomputed sines (pre-expanded to H heads)
///
/// Returns [B, H, S, D] BF16 where:
///   out[..., :D/2] = x[..., :D/2]*cos - x[..., D/2:]*sin
///   out[..., D/2:] = x[..., :D/2]*sin + x[..., D/2:]*cos
///
/// Single kernel launch replaces 9 separate ops (2 narrow+clone, 4 mul, 1 sub, 1 add, 1 cat).
pub fn apply_rope_precomputed(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    ensure_bf16(x, "apply_rope_precomputed x")?;
    ensure_bf16(cos, "apply_rope_precomputed cos")?;
    ensure_bf16(sin, "apply_rope_precomputed sin")?;

    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(
            format!("apply_rope_precomputed expects [B,H,S,D], got {:?}", dims).into(),
        ));
    }
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    if d % 2 != 0 {
        return Err(Error::InvalidInput("RoPE head dim must be even".into()));
    }

    let mut output =
        Tensor::zeros_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;

    let stream = x.device().cuda_stream_raw_ptr();
    let status = unsafe {
        ffi::flame_rope_precomputed_bf16(
            x.as_device_ptr_bf16("rope_x")? as *const _,
            cos.as_device_ptr_bf16("rope_cos")? as *const _,
            sin.as_device_ptr_bf16("rope_sin")? as *const _,
            output.as_mut_device_ptr_bf16("rope_out")? as *mut _,
            b as i32,
            h as i32,
            s as i32,
            d as i32,
            stream,
        )
    };

    if status != 0 {
        return Err(Error::Cuda("flame_rope_precomputed_bf16 failed".into()));
    }

    Ok(output)
}
