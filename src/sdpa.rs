#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda_ops_bf16;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda_ops_ffi::CudaStream;
use crate::device::CudaStreamRawPtrExt;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::ops::gemm_bf16::bmm_bf16_fp32acc_out;
use crate::staging::{borrow_bf16_arena_tensor, ArenaScratch};
use crate::tensor::contracts::trap_is_bf16;
use crate::{
    ops_ext::{full_like, shape4, transpose_last2},
    strict::{record_layout_fix, scope, GuardMode},
    DType, Error, Shape, Tensor,
};
type SdpaResult<T> = crate::Result<T>;

const NEG_INF: f32 = -1.0e9;

fn parse_env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().and_then(|value| {
        let value = value.to_ascii_lowercase();
        match value.as_str() {
            "1" | "true" | "on" | "yes" => Some(true),
            "0" | "false" | "off" | "no" => Some(false),
            _ => None,
        }
    })
}

fn chunk_limit_from_env() -> Option<usize> {
    std::env::var("FLAME_SDPA_CHUNK_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn allow_sdpa_f32_fallback() -> bool {
    if crate::strict::is_enabled() {
        return false;
    }
    match parse_env_flag("SDPA_NO_F32_FALLBACK") {
        Some(true) => false,
        Some(false) | None => true,
    }
}

pub fn forward(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> SdpaResult<Tensor> {
    scope("sdpa.forward", GuardMode::env_default(), || {
        let output = forward_inner(q, k, v, mask)?;
        trap_is_bf16("sdpa.forward out", &output)?;
        Ok(output)
    })
}

fn forward_inner(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> SdpaResult<Tensor> {
    let (bq, hq, q_len, d_q) = shape4(q)?;
    let (bk, hk, k_len, d_k) = shape4(k)?;
    let (bv, hv, v_len, d_v) = shape4(v)?;

    if !(bq == bk && bq == bv && hq == hk && hq == hv) {
        return Err(Error::InvalidInput(format!(
            "batch/head mismatch: q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if !(d_q == d_k && d_q == d_v) {
        return Err(Error::InvalidInput(format!(
            "embed mismatch: q_dim={} k_dim={} v_dim={}",
            d_q, d_k, d_v
        )));
    }
    if k_len != v_len {
        return Err(Error::InvalidInput(format!(
            "sequence mismatch: k_len={} v_len={}",
            k_len, v_len
        )));
    }

    if let Some(m) = mask {
        let dims = m.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "mask must be 4D [B,H,Q,K], got shape {:?}",
                dims
            )));
        }
        if !(dims[0] == bq || dims[0] == 1)
            || !(dims[1] == hq || dims[1] == 1)
            || dims[2] != q_len
            || dims[3] != k_len
        {
            return Err(Error::InvalidInput(format!(
                "mask dims {:?} not broadcastable to [B,H,Q,K] = [{},{},{},{}]",
                dims, bq, hq, q_len, k_len
            )));
        }
    }

    trap_is_bf16("sdpa.forward q", q)?;
    trap_is_bf16("sdpa.forward k", k)?;
    trap_is_bf16("sdpa.forward v", v)?;

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        if q.dtype() == DType::BF16 && k.dtype() == DType::BF16 && v.dtype() == DType::BF16 {
            return forward_bf16(q, k, v, mask, bq, hq, q_len, k_len, d_q);
        }
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    if !allow_sdpa_f32_fallback() {
        return Err(Error::Unsupported(
            "sdpa.forward: FP32 fallback disabled; inputs must remain BF16 on CUDA".into(),
        ));
    }

    forward_f32(q, k, v, mask, d_q)
}

#[cfg(feature = "autograd_v4")]
pub fn forward_v4(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: Option<f32>,
    chunk: Option<(usize, usize)>,
) -> SdpaResult<Tensor> {
    scope("sdpa.forward_v4", GuardMode::env_default(), || {
        let output = forward_v4_inner(q, k, v, mask, causal, scale, chunk)?;
        trap_is_bf16("sdpa.forward_v4 out", &output)?;
        Ok(output)
    })
}

#[cfg(feature = "autograd_v4")]
fn forward_v4_inner(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: Option<f32>,
    chunk: Option<(usize, usize)>,
) -> SdpaResult<Tensor> {
    trap_is_bf16("sdpa.forward_v4 q", q)?;
    trap_is_bf16("sdpa.forward_v4 k", k)?;
    trap_is_bf16("sdpa.forward_v4 v", v)?;
    use crate::autograd_v4::{sdpa_forward, SdpaConfig, SdpaSave};

    let mut cfg = SdpaConfig::default();
    cfg.save = SdpaSave::SaveLSE;
    cfg.causal = causal;
    cfg.scale = scale;
    if let Some((cq, ck)) = chunk {
        cfg.chunk_q = Some(cq);
        cfg.chunk_kv = Some(ck);
    }

    let (output, _ctx) = sdpa_forward(q, k, v, mask, &cfg)?;
    Ok(output)
}

fn forward_f32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    d_q: usize,
) -> SdpaResult<Tensor> {
    let q32 = q.to_dtype(DType::F32)?;
    let k32 = k.to_dtype(DType::F32)?;
    let v32 = v.to_dtype(DType::F32)?;

    let k_t = transpose_last2(&k32)?;
    let mut scores = q32.bmm(&k_t)?;
    let scale = 1.0 / (d_q as f32).sqrt();
    scores = scores.mul_scalar(scale)?;

    if let Some(mask_raw) = mask {
        let target_dims = scores.shape().dims().to_vec();
        let expanded = if mask_raw.shape().dims() != target_dims.as_slice() {
            mask_raw.broadcast_to(&Shape::from_dims(&target_dims))?
        } else {
            mask_raw.reshape(&target_dims)?
        };
        let mask_f32 = expanded.to_dtype(DType::F32)?;
        let ones = full_like(&mask_f32, 1.0)?;
        let complement = ones.sub(&mask_f32)?;
        let penalty = complement.mul_scalar(NEG_INF)?;
        scores = scores.add(&penalty)?;
    }

    let attn = scores.softmax(-1)?;
    let output32 = attn.bmm(&v32)?;

    if q.dtype() == DType::F32 {
        Ok(output32)
    } else {
        output32.to_dtype(q.dtype())
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn forward_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d_q: usize,
) -> SdpaResult<Tensor> {
    let scale = 1.0 / (d_q as f32).sqrt();

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        let limit = chunk_limit_from_env().unwrap_or(2048);
        let chunk = std::cmp::max(1usize, q_len.min(limit));
        log::trace!(
            "sdpa_stream_bf16 launch: B={} H={} Q={} K={} Dh={} chunk={} (limit {})",
            b,
            h,
            q_len,
            k_len,
            d_q,
            chunk,
            limit
        );
        match cuda_ops_bf16::sdpa_stream_bf16(q, k, v, mask, chunk, false, Some(scale)) {
            Ok(out) => {
                let dims = out.shape().dims();
                if dims.len() != 4 || dims[0] != b || dims[1] != h || dims[2] != q_len {
                    return Err(Error::InvalidOperation(format!(
                        "sdpa_stream_bf16 produced unexpected shape {:?}, expected [{},{},{},{}]",
                        dims, b, h, q_len, d_q
                    )));
                }
                if dims[3] != d_q {
                    return Err(Error::InvalidOperation(format!(
                        "sdpa_stream_bf16 produced value dim {}, expected {}",
                        dims[3], d_q
                    )));
                }
                return Ok(out);
            }
            Err(Error::Unsupported(reason)) => {
                return Err(Error::Unsupported(format!(
                    "sdpa_stream_bf16 unsupported (no fallback): {}",
                    reason
                )))
            }
            Err(err) => return Err(err),
        }
    }

    Err(Error::Unsupported(
        "sdpa_stream_bf16 unavailable (no fallback)".into(),
    ))
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn forward_bf16_fallback(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d_q: usize,
    scale: f32,
) -> SdpaResult<Tensor> {
    let device = q.device().clone();
    let bh = b * h;
    let q_flat = q.reshape(&[bh, q_len, d_q])?;
    let k_flat = k.reshape(&[bh, k_len, d_q])?;
    let v_flat = v.reshape(&[bh, k_len, d_q])?;
    let k_t = k_flat.transpose_dims(1, 2)?;

    let stream = CudaStream::from_raw(q.device().cuda_stream_raw_ptr());
    let logits_shape = Shape::from_dims(&[bh, q_len, k_len]);
    let mut logits_bf16 = borrow_bf16_arena_tensor(
        device.clone(),
        &stream,
        logits_shape.clone(),
        ArenaScratch::DEFAULT_ALIGN,
    )
    .or_else(|_| Tensor::zeros_dtype(logits_shape.clone(), DType::BF16, device.clone()))?;
    bmm_bf16_fp32acc_out(&q_flat, &k_t, &mut logits_bf16, false, false)?;

    let mut scores = logits_bf16.mul_scalar(scale)?;
    drop(logits_bf16);

    if let Some(mask_raw) = mask {
        scores = {
            let mut converted: Option<Tensor> = None;
            let mask_bf16 =
                if mask_raw.dtype() == DType::BF16 && mask_raw.storage_dtype() == DType::BF16 {
                    mask_raw
                } else {
                    converted = Some(mask_raw.to_dtype(DType::BF16)?);
                    converted.as_ref().unwrap()
                };

            let target_dims = [b, h, q_len, k_len];
            let mut broadcasted: Option<Tensor> = None;
            let mask_prepared = if mask_bf16.shape().dims() == target_dims {
                mask_bf16
            } else {
                record_layout_fix("sdpa.mask_broadcast", mask_bf16.shape());
                broadcasted = Some(mask_bf16.broadcast_to(&Shape::from_dims(&target_dims))?);
                broadcasted.as_ref().unwrap()
            };

            let mask_view = mask_prepared.reshape(&[bh, q_len, k_len])?;
            let penalty = mask_view.neg()?.add_scalar(1.0)?.mul_scalar(NEG_INF)?;
            let updated = scores.add(&penalty)?;
            updated
        };
    }

    let attn = scores.softmax(-1)?;
    drop(scores);
    let attn_bf16 = if attn.dtype() == DType::BF16 {
        attn
    } else {
        attn.to_dtype(DType::BF16)?
    };

    let out_shape = Shape::from_dims(&[bh, q_len, d_q]);
    enum OutputBuffer {
        Arena(Tensor),
        Direct(Tensor),
    }

    let output = match borrow_bf16_arena_tensor(
        device.clone(),
        &stream,
        out_shape.clone(),
        ArenaScratch::DEFAULT_ALIGN,
    ) {
        Ok(tensor) => OutputBuffer::Arena(tensor),
        Err(err) => {
            log::warn!(
                "sdpa_fallback: arena allocation for output {:?} failed: {} -- switching to chunked accumulation",
                out_shape.dims(),
                err
            );
            OutputBuffer::Direct(Tensor::zeros_dtype(
                out_shape.clone(),
                DType::BF16,
                device.clone(),
            )?)
        }
    };

    let projected = match output {
        OutputBuffer::Arena(mut out_bf16) => {
            bmm_bf16_fp32acc_out(&attn_bf16, &v_flat, &mut out_bf16, false, false)?;
            out_bf16
        }
        OutputBuffer::Direct(mut out_direct) => {
            let chunk_limit = chunk_limit_from_env().unwrap_or(512).max(1);
            let mut start = 0usize;
            let elems_per_q = d_q;
            let elems_per_head = q_len * elems_per_q;

            while start < q_len {
                let len = std::cmp::min(chunk_limit, q_len - start);
                let attn_chunk = attn_bf16.narrow(1, start, len)?;
                let chunk_shape = Shape::from_dims(&[bh, len, d_q]);
                let mut chunk_out =
                    Tensor::zeros_dtype(chunk_shape.clone(), DType::BF16, device.clone())?;
                bmm_bf16_fp32acc_out(&attn_chunk, &v_flat, &mut chunk_out, false, false)?;

                let chunk_elems = len * elems_per_q;
                for bh_idx in 0..bh {
                    let dst_offset = bh_idx * elems_per_head + start * elems_per_q;
                    let src_offset = bh_idx * chunk_elems;
                    out_direct.copy_bf16_region_from(
                        dst_offset,
                        &chunk_out,
                        src_offset,
                        chunk_elems,
                    )?;
                }
                drop(chunk_out);
                start += len;
            }
            out_direct
        }
    };

    drop(attn_bf16);

    let projected = if q.dtype() == DType::BF16 {
        projected
    } else {
        projected.to_dtype(q.dtype())?
    };

    projected.reshape(&[b, h, q_len, d_q])
}
