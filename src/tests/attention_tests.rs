#[cfg(test)]
mod tests {
    use crate::attention::rope::apply_rope;
    use crate::attention::{attention_impl, GeGLU};
    use crate::{DType, Device, Result, Shape, Tensor};
    use std::sync::Arc;

    fn rope_reference(
        data: &[f32],
        batch: usize,
        heads: usize,
        seq: usize,
        head_dim: usize,
        base_theta: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    for pair in 0..(head_dim / 2) {
                        let offset = ((((b * heads + h) * seq) + s) * head_dim + 2 * pair) as usize;
                        let x0 = data[offset];
                        let x1 = data[offset + 1];
                        let freq = 1.0f32 / base_theta.powf(2.0 * pair as f32 / head_dim as f32);
                        let angle = s as f32 * freq;
                        let (sin_angle, cos_angle) = angle.sin_cos();
                        out[offset] = x0 * cos_angle - x1 * sin_angle;
                        out[offset + 1] = x0 * sin_angle + x1 * cos_angle;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn rope_gpu_matches_reference() -> Result<()> {
        let dev = Device::cuda(0)?;
        let cuda = dev.cuda_device().clone();
        let (batch, heads, seq, head_dim) = (1, 2, 3, 4);
        let base_theta = 10_000.0f32;
        let values: Vec<f32> = (0..batch * heads * seq * head_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        let tensor = Tensor::from_vec(
            values.clone(),
            Shape::from_dims(&[batch, heads, seq, head_dim]),
            cuda.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let rotated = apply_rope(&tensor, head_dim, base_theta, 0)?
            .to_dtype(DType::F32)?
            .to_vec()?;
        let reference = rope_reference(&values, batch, heads, seq, head_dim, base_theta);
        for (a, b) in rotated.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1.5e-3, "mismatch: gpu={} ref={}", a, b);
        }
        Ok(())
    }

    fn sdpa_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: &[f32],
        q_len: usize,
        k_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; q_len * head_dim];
        for qi in 0..q_len {
            let mut scores = vec![0.0f32; k_len];
            for kj in 0..k_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = qi * head_dim + d;
                    let k_idx = kj * head_dim + d;
                    dot += q[q_idx] * k[k_idx];
                }
                let mask_idx = qi * k_len + kj;
                let masked = if mask[mask_idx] < 0.5 {
                    -1.0e9
                } else {
                    dot * scale
                };
                scores[kj] = masked;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores = vec![0.0f32; k_len];
            let mut sum = 0.0f32;
            for (dst, src) in exp_scores.iter_mut().zip(scores.iter()) {
                *dst = (*src - max_score).exp();
                sum += *dst;
            }
            if sum == 0.0 {
                continue;
            }
            for kj in 0..k_len {
                let weight = exp_scores[kj] / sum;
                for d in 0..head_dim {
                    let v_idx = kj * head_dim + d;
                    output[qi * head_dim + d] += weight * v[v_idx];
                }
            }
        }
        output
    }

    #[test]
    fn sdpa_matches_reference_masked() -> Result<()> {
        let dev = Device::cuda(0)?;
        let cuda = dev.cuda_device().clone();
        let (batch, heads, q_len, k_len, head_dim) = (1, 1, 2, 2, 2);
        let q_vals = vec![0.2f32, -0.4, 0.5, 0.1];
        let k_vals = vec![0.3f32, 0.7, -0.2, 0.4];
        let v_vals = vec![0.1f32, -0.3, 0.6, 0.2];
        let mask_vals = vec![1.0f32, 0.0, 1.0, 1.0];

        let q = Tensor::from_vec(
            q_vals.clone(),
            Shape::from_dims(&[batch, heads, q_len, head_dim]),
            cuda.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let k = Tensor::from_vec(
            k_vals.clone(),
            Shape::from_dims(&[batch, heads, k_len, head_dim]),
            cuda.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let v = Tensor::from_vec(
            v_vals.clone(),
            Shape::from_dims(&[batch, heads, k_len, head_dim]),
            cuda.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let mask = Tensor::from_vec(
            mask_vals.clone(),
            Shape::from_dims(&[batch, heads, q_len, k_len]),
            cuda.clone(),
        )?;

        let out = attention_impl(&q, &k, &v, Some(&mask), false, None)?
            .to_dtype(DType::F32)?
            .to_vec()?;

        let reference = sdpa_reference(
            &q_vals, &k_vals, &v_vals, &mask_vals, q_len, k_len, head_dim,
        );
        for (a, b) in out.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 2e-3, "mismatch: gpu={} ref={}", a, b);
        }
        Ok(())
    }

    #[test]
    fn geglu_matches_cpu_reference() -> Result<()> {
        let dev = Device::cuda(0)?;
        let cuda: Arc<_> = dev.cuda_device().clone();
        let (dim_in, dim_out) = (3, 2);
        let mut geglu = GeGLU::new(dim_in, dim_out, cuda.clone())?;

        let weight_vals: Vec<f32> = (0..(dim_out * 2 * dim_in))
            .map(|i| (i as f32) * 0.05)
            .collect();
        geglu.proj.weight = Tensor::from_vec(
            weight_vals.clone(),
            Shape::from_dims(&[dim_out * 2, dim_in]),
            cuda.clone(),
        )?
        .to_dtype(DType::BF16)?;
        geglu.proj.bias = Some(
            Tensor::zeros(Shape::from_dims(&[dim_out * 2]), cuda.clone())?.to_dtype(DType::BF16)?,
        );

        let input_vals = vec![0.8f32, -0.2, 0.5];
        let input = Tensor::from_vec(
            input_vals.clone(),
            Shape::from_dims(&[1, dim_in]),
            cuda.clone(),
        )?
        .to_dtype(DType::BF16)?;

        let projected = geglu.proj.forward(&input)?;
        let projected_vec = projected.to_vec()?;

        let mut expected = Vec::with_capacity(dim_out);
        let half = dim_out;
        for chunk in projected_vec.chunks(half * 2) {
            let (value, gate) = chunk.split_at(half);
            let gated: Vec<f32> = gate
                .iter()
                .map(|&g| {
                    0.5 * g
                        * (1.0
                            + ((2.0 / std::f32::consts::PI).sqrt() * (g + 0.044715 * g.powi(3)))
                                .tanh())
                })
                .collect();
            for (v, g) in value.iter().zip(gated.iter()) {
                expected.push(v * g);
            }
        }

        let actual = geglu.forward(&input)?.to_vec()?;
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: gpu={} ref={}", a, b);
        }
        Ok(())
    }
}
