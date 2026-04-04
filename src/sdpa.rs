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

    // Use the materialized GEMM path (batched cuBLASLt) when the attention matrix
    // fits in VRAM. BH*Q*K*2 bytes BF16:
    //   Klein 4B 512×512:  24*1536*1536*2 = 113MB — fine
    //   Klein 4B 1024×1024: 24*4608*4608*2 = 970MB — fine on 24GB
    //   Klein 9B 1024×1024: 32*4608*4608*2 = 1.36GB — fits on 24GB
    //   LTX-2.3:            32*768*768*2   = 36MB — fine
    // Only use the streaming kernel for truly huge sequences (video, >8K tokens).
    let attn_elems = b * h * q_len * k_len;
    let force_stream = parse_env_flag("FLAME_SDPA_FORCE_STREAM").unwrap_or(false);
    // ~1.5GB BF16 threshold (768M elements). Safe on 24GB+ GPUs.
    let use_materialized = !force_stream && attn_elems <= 768_000_000;

    if use_materialized {
        log::trace!(
            "sdpa forward_bf16_fallback (materialized): B={} H={} Q={} K={} Dh={} ({}M elems)",
            b, h, q_len, k_len, d_q, attn_elems / 1_000_000
        );
        return forward_bf16_fallback(q, k, v, mask, b, h, q_len, k_len, d_q, scale);
    }

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
    // Materialized SDPA: two batched BF16 GEMMs (FP32 acc) + FP32 softmax.
    // QK^T logits are upcast to FP32 for softmax to match Flash Attention precision
    // (avoids BF16 quantization of attention scores → spiky weights → outliers).
    // VRAM: BH*Q*K*4 bytes for the FP32 logits (temporary).
    //   SDXL level2: 20*1024*1024*4 = 80MB, level1: 10*4096*4096*4 = 672MB
    //   LTX-2.3:     32*768*768*4   = 72MB
    let device = q.device().clone();
    let bh = b * h;
    let t0 = std::time::Instant::now();

    // Flatten [B, H, seq, d] → [BH, seq, d] for batched GEMM
    let q_flat = q.reshape(&[bh, q_len, d_q])?;
    let k_flat = k.reshape(&[bh, k_len, d_q])?;
    let v_flat = v.reshape(&[bh, k_len, d_q])?;
    let t_reshape = t0.elapsed().as_millis();

    let k_t = k_flat.transpose_dims(1, 2)?; // [BH, d, K]
    let t_transpose = t0.elapsed().as_millis() - t_reshape;

    // QK^T → [BH, Q, K] via batched cuBLASLt (FP32 accumulation, tensor cores)
    // Output is BF16 (cublasLt constraint), but we upcast to FP32 immediately
    // for scaling + softmax to avoid BF16 quantization of attention logits.
    // Flash Attention keeps scores in FP32 SRAM; we emulate this by upcasting.
    let logits_shape = Shape::from_dims(&[bh, q_len, k_len]);
    let mut logits_bf16 = Tensor::zeros_dtype(logits_shape, DType::BF16, device.clone())?;
    bmm_bf16_fp32acc_out(&q_flat, &k_t, &mut logits_bf16, false, false)?;
    let t_qk = t0.elapsed().as_millis() - t_reshape - t_transpose;
    drop(k_t);

    // Upcast to FP32 for scale + softmax (matches Flash Attention precision)
    let logits_f32 = logits_bf16.to_dtype(DType::F32)?;
    drop(logits_bf16);

    // Scale in FP32
    let mut scores = logits_f32.mul_scalar(scale)?;
    let t_scale = t0.elapsed().as_millis() - t_reshape - t_transpose - t_qk;

    // Mask (if present) — apply in FP32
    if let Some(mask_raw) = mask {
        let target_dims = [b, h, q_len, k_len];
        let mask_f32 = if mask_raw.dtype() == DType::F32 {
            mask_raw.clone_result()?
        } else {
            mask_raw.to_dtype(DType::F32)?
        };
        let mask_prepared = if mask_f32.shape().dims() == target_dims {
            mask_f32
        } else {
            record_layout_fix("sdpa.mask_broadcast", mask_f32.shape());
            mask_f32.broadcast_to(&Shape::from_dims(&target_dims))?
        };
        let mask_view = mask_prepared.reshape(&[bh, q_len, k_len])?;
        let ones = full_like(&mask_view, 1.0)?;
        let complement = ones.sub(&mask_view)?;
        let penalty = complement.mul_scalar(NEG_INF)?;
        scores = scores.add(&penalty)?;
    }

    let t_pre_softmax = t0.elapsed().as_millis();

    // Softmax in FP32, then convert to BF16 for V multiplication.
    // scores is F32 here, so softmax computes entirely in F32.
    let attn = scores.softmax(-1)?;
    drop(scores);
    let attn_bf16 = if attn.dtype() != DType::BF16 {
        attn.to_dtype(DType::BF16)?
    } else {
        attn
    };
    let t_softmax = t0.elapsed().as_millis() - t_pre_softmax;

    let t_pre_pv = t0.elapsed().as_millis();

    // attn × V → [BH, Q, d] via batched cuBLASLt
    let out_shape = Shape::from_dims(&[bh, q_len, d_q]);
    let mut projected = Tensor::zeros_dtype(out_shape, DType::BF16, device)?;
    bmm_bf16_fp32acc_out(&attn_bf16, &v_flat, &mut projected, false, false)?;
    let t_pv = t0.elapsed().as_millis() - t_pre_pv;
    drop(attn_bf16);

    let total = t0.elapsed().as_millis();
    log::info!(
        "[SDPA] reshape={}ms transpose={}ms QK={}ms scale={}ms softmax={}ms PV={}ms total={}ms (BH={} Q={} K={} d={})",
        t_reshape, t_transpose, t_qk, t_scale, t_softmax, t_pv, total,
        bh, q_len, k_len, d_q
    );

    projected.reshape(&[b, h, q_len, d_q])
}
