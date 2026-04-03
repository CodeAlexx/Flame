//! FlashAttention-backed scaled dot-product attention.
//!
//! The BF16 contract is enforced at the call boundary: tensors passed in and
//! returned remain BF16 while the FlashAttention runtime performs FP32
//! accumulation internally.

#[cfg(not(feature = "bf16_u16"))]
use crate::{Error, Result, Tensor};

#[cfg(not(feature = "bf16_u16"))]
pub fn attention_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    _mask: Option<&Tensor>,
    _causal: bool,
    _scale: Option<f32>,
) -> Result<Tensor> {
    let _ = (q, k, v);
    Err(Error::Unsupported(
        "flash_attn requires the `bf16_u16` feature to be enabled".into(),
    ))
}

#[cfg(feature = "bf16_u16")]
mod bf16 {
    use cudarc::driver::DevicePtr;

    use crate::attention::flash_ffi;
    use crate::cuda::device_lt;
    use crate::{DType, Error, Result, Tensor};

    fn ensure_bf16(tensor: &Tensor, name: &str) -> Result<()> {
        if tensor.dtype() != DType::BF16 {
            return Err(Error::InvalidInput(format!(
                "{name} must be BF16 (got {:?})",
                tensor.dtype()
            )));
        }
        Ok(())
    }

    fn tensor_device_ptr(t: &Tensor, name: &str) -> Result<*const core::ffi::c_void> {
        let slice = t
            .storage_ref()
            .try_as_slice_u16()
            .map_err(|_| Error::InvalidInput(format!("{name}: expected BF16 storage")))?;
        Ok(*slice.device_ptr() as *const core::ffi::c_void)
    }

    fn tensor_device_ptr_mut(t: &mut Tensor, name: &str) -> Result<*mut core::ffi::c_void> {
        let slice = t
            .storage_mut()
            .try_as_mut_slice_u16()
            .map_err(|_| Error::InvalidInput(format!("{name}: expected BF16 storage")))?;
        Ok(*slice.device_ptr() as *mut core::ffi::c_void)
    }

    pub fn attention_impl(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
        scale: Option<f32>,
    ) -> Result<Tensor> {
        // FlashAttention v2 does not currently accept arbitrary masks; fall back to
        // SDPA when one is provided.
        if mask.is_some() {
            return super::super::sdpa::attention_impl(q, k, v, mask, causal, scale);
        }

        ensure_bf16(q, "q")?;
        ensure_bf16(k, "k")?;
        ensure_bf16(v, "v")?;

        let q_shape = q.shape().dims();
        if q_shape.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "FlashAttention expects [B,H,T,D] layout, got {:?}",
                q_shape
            )));
        }
        let (batch, heads, q_tokens, dim) = (
            q_shape[0] as i32,
            q_shape[1] as i32,
            q_shape[2] as i32,
            q_shape[3] as i32,
        );

        let k_shape = k.shape().dims();
        let v_shape = v.shape().dims();
        if k_shape != [q_shape[0], q_shape[1], k_shape[2], q_shape[3]]
            || v_shape != [q_shape[0], q_shape[1], k_shape[2], q_shape[3]]
        {
            return Err(Error::InvalidInput(
                "k/v shapes must match q on batch/head/embed".into(),
            ));
        }
        let kv_tokens = k_shape[2] as i32;

        let device = q.device().clone();
        let stream = device_lt::stream_ptr(&device)
            .map_err(|err| Error::Cuda(format!("stream_ptr failed: {err}")))?;

        let mut output = Tensor::zeros_dtype(q.shape().clone(), DType::BF16, device.clone())
            .map_err(|err| Error::Cuda(format!("alloc output: {err}")))?;

        match flash_ffi::get_flash() {
            Ok(lib) => {
                let scale = scale.unwrap_or_else(|| 1.0f32 / (dim as f32).sqrt());
                let q_ptr = tensor_device_ptr(q, "q")?;
                let k_ptr = tensor_device_ptr(k, "k")?;
                let v_ptr = tensor_device_ptr(v, "v")?;
                let out_ptr = tensor_device_ptr_mut(&mut output, "out")?;

                let status = unsafe {
                    (lib.fa_bf16_forward)(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        out_ptr,
                        batch,
                        heads,
                        q_tokens,
                        kv_tokens,
                        dim,
                        scale,
                        if causal { 1 } else { 0 },
                        stream,
                    )
                };

                if status != 0 {
                    return Err(Error::Cuda(format!(
                        "flash_attn bf16 forward status={status}"
                    )));
                }

                Ok(output)
            }
            Err(err) => Err(Error::Unsupported(format!("flash_attn unavailable: {err}"))),
        }
    }
}

#[cfg(feature = "bf16_u16")]
pub use bf16::attention_impl;
