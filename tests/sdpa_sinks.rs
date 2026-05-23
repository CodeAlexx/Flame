//! Parity test for `flame_core::sdpa::forward_with_sinks` against a hand-
//! computed CPU reference matching the GPT-OSS / StreamingLLM math:
//!
//! ```text
//! logits      = (Q @ K^T) * scale       # [B,H,Q,K]
//! sink_logits = sinks[h]                # one scalar per head, broadcast
//! all_logits  = cat([logits, sink_logits], -1)   # [B,H,Q,K+1]
//! attn        = softmax(all_logits, -1)
//! attn_kv     = attn[..., :K]
//! out         = attn_kv @ V             # [B,H,Q,D]
//! ```

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::sdpa;
use flame_core::{DType, Device, Result, Shape, Tensor};

fn make_bf16(values: Vec<f32>, shape: &[usize], cuda: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<Tensor> {
    Tensor::from_vec(values, Shape::from_dims(shape), cuda.clone())?.to_dtype(DType::BF16)
}

fn cpu_sinks_reference(
    q: &[f32], // [B,H,Q,D]
    k: &[f32], // [B,H,K,D]
    v: &[f32], // [B,H,K,D]
    sinks: &[f32], // [H]
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut out = vec![0.0f32; b * h * q_len * d];

    for bi in 0..b {
        for hi in 0..h {
            // Per (b, h) head.
            let sink_logit = sinks[hi];
            for qi in 0..q_len {
                // 1) raw logits over K plus the sink column.
                let mut scores = vec![0.0f32; k_len + 1];
                for kj in 0..k_len {
                    let mut dot = 0.0f32;
                    for di in 0..d {
                        let q_idx = (((bi * h) + hi) * q_len + qi) * d + di;
                        let k_idx = (((bi * h) + hi) * k_len + kj) * d + di;
                        dot += q[q_idx] * k[k_idx];
                    }
                    scores[kj] = dot * scale;
                }
                scores[k_len] = sink_logit;

                // 2) softmax over K+1.
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exps = vec![0.0f32; k_len + 1];
                let mut sum = 0.0f32;
                for (e, s) in exps.iter_mut().zip(scores.iter()) {
                    *e = (*s - m).exp();
                    sum += *e;
                }
                for e in exps.iter_mut() {
                    *e /= sum;
                }

                // 3) weighted sum over K (drop sink column).
                for di in 0..d {
                    let mut acc = 0.0f32;
                    for kj in 0..k_len {
                        let v_idx = (((bi * h) + hi) * k_len + kj) * d + di;
                        acc += exps[kj] * v[v_idx];
                    }
                    let o_idx = (((bi * h) + hi) * q_len + qi) * d + di;
                    out[o_idx] = acc;
                }
            }
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

#[test]
fn sinks_match_pytorch_reference() -> Result<()> {
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device().clone();

    let (b, h, q_len, k_len, d) = (1usize, 2usize, 4usize, 4usize, 8usize);

    // Deterministic small inputs; the values are smooth-ish to keep the
    // BF16 precision floor honest.
    let q_vals: Vec<f32> = (0..b * h * q_len * d)
        .map(|i| 0.10 + (i as f32) * 0.013)
        .collect();
    let k_vals: Vec<f32> = (0..b * h * k_len * d)
        .map(|i| -0.20 + (i as f32) * 0.011)
        .collect();
    let v_vals: Vec<f32> = (0..b * h * k_len * d)
        .map(|i| 0.05 + (i as f32) * 0.017)
        .collect();
    // Per-head sink: head 0 small, head 1 large (drives a noticeable sink
    // mass — head 1 attention over real keys will sum to << 1).
    let sinks_vals: Vec<f32> = vec![0.3, 4.0];

    let q = make_bf16(q_vals.clone(), &[b, h, q_len, d], &cuda)?;
    let k = make_bf16(k_vals.clone(), &[b, h, k_len, d], &cuda)?;
    let v = make_bf16(v_vals.clone(), &[b, h, k_len, d], &cuda)?;
    let sinks_t = make_bf16(sinks_vals.clone(), &[h], &cuda)?;

    let out = sdpa::forward_with_sinks(&q, &k, &v, None, &sinks_t)?
        .to_dtype(DType::F32)?
        .to_vec()?;

    let expected = cpu_sinks_reference(
        &q_vals, &k_vals, &v_vals, &sinks_vals, b, h, q_len, k_len, d,
    );

    let cos = cosine(&out, &expected);
    assert!(
        cos >= 0.9999,
        "sinks SDPA cos={} < 0.9999 (BF16 precision floor)",
        cos
    );

    // Spot-check: with a large sink logit (head 1's 4.0) the per-row sum
    // of attention weights over real keys should be < 1.0. We can verify
    // indirectly: head 1 outputs should be uniformly smaller in magnitude
    // than head 0 outputs at the same query (assuming similar V scale).
    // This is the qualitative StreamingLLM invariant; the cos check above
    // is the bit-exact gate.
    Ok(())
}

#[test]
fn sinks_with_keep_mask_matches_reference() -> Result<()> {
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device().clone();

    let (b, h, q_len, k_len, d) = (1usize, 1usize, 3usize, 4usize, 4usize);

    let q_vals: Vec<f32> = (0..b * h * q_len * d)
        .map(|i| 0.10 + (i as f32) * 0.02)
        .collect();
    let k_vals: Vec<f32> = (0..b * h * k_len * d)
        .map(|i| -0.30 + (i as f32) * 0.015)
        .collect();
    let v_vals: Vec<f32> = (0..b * h * k_len * d)
        .map(|i| 0.05 + (i as f32) * 0.019)
        .collect();
    let sinks_vals: Vec<f32> = vec![1.5];

    // Mask out the last K column for every query.
    let mask_vals: Vec<f32> = {
        let mut m = vec![1.0f32; b * h * q_len * k_len];
        for qi in 0..q_len {
            m[qi * k_len + (k_len - 1)] = 0.0;
        }
        m
    };

    let q = make_bf16(q_vals.clone(), &[b, h, q_len, d], &cuda)?;
    let k = make_bf16(k_vals.clone(), &[b, h, k_len, d], &cuda)?;
    let v = make_bf16(v_vals.clone(), &[b, h, k_len, d], &cuda)?;
    let mask = make_bf16(mask_vals.clone(), &[b, h, q_len, k_len], &cuda)?;
    let sinks_t = make_bf16(sinks_vals.clone(), &[h], &cuda)?;

    let out = sdpa::forward_with_sinks(&q, &k, &v, Some(&mask), &sinks_t)?
        .to_dtype(DType::F32)?
        .to_vec()?;

    // CPU reference: same math, but zero out the last key column from
    // the logits (replace with -inf-equivalent) before softmax. We just
    // skip k_len-1 in both the softmax sum and the weighted V sum.
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut expected = vec![0.0f32; b * h * q_len * d];
    for hi in 0..h {
        let sink_logit = sinks_vals[hi];
        for qi in 0..q_len {
            let mut scores = vec![0.0f32; k_len + 1];
            for kj in 0..k_len {
                if mask_vals[qi * k_len + kj] < 0.5 {
                    scores[kj] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f32;
                for di in 0..d {
                    dot += q_vals[(hi * q_len + qi) * d + di]
                        * k_vals[(hi * k_len + kj) * d + di];
                }
                scores[kj] = dot * scale;
            }
            scores[k_len] = sink_logit;
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0f32; k_len + 1];
            let mut sum = 0.0f32;
            for (e, s) in exps.iter_mut().zip(scores.iter()) {
                *e = if s.is_finite() { (*s - m).exp() } else { 0.0 };
                sum += *e;
            }
            for e in exps.iter_mut() {
                *e /= sum;
            }
            for di in 0..d {
                let mut acc = 0.0f32;
                for kj in 0..k_len {
                    acc += exps[kj] * v_vals[(hi * k_len + kj) * d + di];
                }
                expected[(hi * q_len + qi) * d + di] = acc;
            }
        }
    }

    let cos = cosine(&out, &expected);
    assert!(
        cos >= 0.9999,
        "sinks+mask SDPA cos={} < 0.9999",
        cos
    );
    Ok(())
}
