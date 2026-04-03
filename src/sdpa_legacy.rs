#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::ops::gemm_bf16::bmm_bf16_fp32acc_out;
use crate::{
    ops_ext::{full_like, shape4, transpose_last2},
    DType, Error, Shape, Tensor,
};
use std::result::Result as StdResult;

type SdpaResult<T> = StdResult<T, Error>;

const NEG_INF: f32 = -1.0e9;

pub fn forward(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> SdpaResult<Tensor> {
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

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        if q.dtype() == DType::BF16 && k.dtype() == DType::BF16 && v.dtype() == DType::BF16 {
            return forward_bf16(q, k, v, mask, bq, hq, q_len, k_len, d_q);
        }
    }

    forward_f32(q, k, v, mask, d_q)
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
            mask_raw.clone_result()?
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
    let device = q.device().clone();
    let bh = b * h;
    let q_flat = q.reshape(&[bh, q_len, d_q])?.clone_result()?;
    let k_flat = k.reshape(&[bh, k_len, d_q])?.clone_result()?;
    let v_flat = v.reshape(&[bh, k_len, d_q])?.clone_result()?;
    let k_t = k_flat.transpose_dims(1, 2)?.clone_result()?;

    let mut logits_bf16 = Tensor::zeros_dtype(
        Shape::from_dims(&[bh, q_len, k_len]),
        DType::BF16,
        device.clone(),
    )?;
    bmm_bf16_fp32acc_out(&q_flat, &k_t, &mut logits_bf16, false, false)?;

    let mut scores = logits_bf16.clone_result()?;
    let scale = 1.0 / (d_q as f32).sqrt();
    scores = scores.mul_scalar(scale)?;

    if let Some(mask_raw) = mask {
        let target_dims = vec![b, h, q_len, k_len];
        let expanded = if mask_raw.shape().dims() != target_dims.as_slice() {
            mask_raw.broadcast_to(&Shape::from_dims(&target_dims))?
        } else {
            mask_raw.clone_result()?
        };
        let mask_view = expanded.reshape(&[bh, q_len, k_len])?;
        let mask_bf16 = if mask_view.dtype() == DType::BF16 {
            mask_view
        } else {
            mask_view.to_dtype(DType::BF16)?
        };
        let ones = full_like(&mask_bf16, 1.0)?;
        let complement = ones.sub(&mask_bf16)?;
        let penalty = complement.mul_scalar(NEG_INF)?;
        scores = scores.add(&penalty)?;
    }

    let attn = scores.softmax(-1)?;
    let attn_bf16 = if attn.dtype() == DType::BF16 {
        attn
    } else {
        attn.to_dtype(DType::BF16)?
    };

    let mut out_bf16 = Tensor::zeros_dtype(
        Shape::from_dims(&[bh, q_len, d_q]),
        DType::BF16,
        device.clone(),
    )?;
    bmm_bf16_fp32acc_out(&attn_bf16, &v_flat, &mut out_bf16, false, false)?;

    let out = out_bf16.to_dtype(q.dtype())?;
    let reshaped = out.reshape(&[b, h, q_len, d_q])?;
    Ok(reshaped)
}
