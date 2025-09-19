use crate::{ops_ext::{full_like, shape4, transpose_last2}, Tensor, DType, Error, Result, Shape};

const NEG_INF: f32 = -1.0e9;

pub fn forward(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
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

    let q32 = q.to_dtype(DType::F32)?;
    let k32 = k.to_dtype(DType::F32)?;
    let v32 = v.to_dtype(DType::F32)?;

    let k_t = transpose_last2(&k32)?;
    let mut scores = q32.bmm(&k_t)?;
    let scale = 1.0 / (d_q as f32).sqrt();
    scores = scores.mul_scalar(scale)?;

    if let Some(mask_raw) = mask {
        let target_dims = scores.shape().dims().clone();
        let expanded = if mask_raw.shape().dims() != &target_dims {
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
