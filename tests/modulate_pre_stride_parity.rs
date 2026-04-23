//! Phase 2b-2 regression: `modulate_pre_fused_bf16` produces identical
//! output for (A) strided narrow-view shift/scale inputs and (B) the same
//! inputs after .contiguous(). Mirrors the swiglu Phase 2b-1 test.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn hash_fill(shape: &[usize]) -> Vec<f32> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 512.0 - 1.0)
        .collect()
}

#[test]
fn modulate_pre_contig_matches_strided_on_shared_mod_narrow() {
    // Klein DiT modulation pattern (matches `klein.rs::shared_modulation_from_silu`):
    //   raw = linear3d(vec_silu, weight) → 2D [B, mod_dim*dim] contig
    //   chunks[j] = raw.narrow(1, j*dim, dim) → 2D [B, dim] strided
    //              (stride[0] = mod_dim*dim, stride[1] = 1, offset = j*dim)
    //
    // B=2: with B=1 the batch stride on shift/scale is never dereferenced
    // (batch_idx always 0), so strided and contig produce identical output
    // for any stride[0] — the test couldn't witness a stride-handling bug.
    let (b, n, dim, mod_dim) = (2, 64, 128, 6);
    let device = global_cuda_device();

    // x: [B, N, dim] contig activations.
    let x_data = hash_fill(&[b, n, dim]);
    let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, n, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Shared modulation: 2D [B, mod_dim*dim] contig. MSA pair at chunks (0,1).
    let shift_idx = 0;
    let mod_data = hash_fill(&[b, mod_dim * dim]);
    let shared_mod = Tensor::from_vec(mod_data, Shape::from_dims(&[b, mod_dim * dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // A: strided path — direct narrow on dim 1. Produces [B, dim] with
    // stride[0] = mod_dim*dim (the un-narrowed inner stride).
    let shift_strided = shared_mod.narrow(1, shift_idx * dim, dim).unwrap();
    let scale_strided = shared_mod.narrow(1, (shift_idx + 1) * dim, dim).unwrap();
    assert_eq!(shift_strided.shape().dims(), &[b, dim]);
    assert_eq!(scale_strided.shape().dims(), &[b, dim]);
    assert!(!shift_strided.is_contiguous(), "shift-view should be strided");
    assert!(!scale_strided.is_contiguous(), "scale-view should be strided");

    let out_strided = flame_core::bf16_ops::modulate_pre_fused_bf16(
        &x, &shift_strided, &scale_strided, 1e-5,
    )
    .unwrap();

    // B: contig path — materialize the views first.
    let shift_contig = shift_strided.contiguous().unwrap();
    let scale_contig = scale_strided.contiguous().unwrap();
    assert!(shift_contig.is_contiguous());
    assert!(scale_contig.is_contiguous());

    let out_contig = flame_core::bf16_ops::modulate_pre_fused_bf16(
        &x, &shift_contig, &scale_contig, 1e-5,
    )
    .unwrap();

    let va = out_strided.to_vec().unwrap();
    let vb = out_contig.to_vec().unwrap();
    assert_eq!(va.len(), vb.len());
    for (i, (a, b)) in va.iter().zip(vb.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "modulate_pre strided vs contig diverges at idx {}: strided={} contig={}",
            i, a, b
        );
    }
}

#[test]
fn modulate_pre_strided_math_on_klein_like_shape() {
    // Klein's actual DiT dim=4096 and mod_dim=6 (shift/scale/gate × MSA/MLP);
    // using a smaller proxy shape that exercises the same narrow pattern.
    let (b, n, dim, mod_dim) = (1, 16, 64, 6);
    let shift_idx = 3; // MLP pair: shift at 3, scale at 4
    let eps = 1e-5_f32;
    let device = global_cuda_device();

    let x_data = hash_fill(&[b, n, dim]);
    let x = Tensor::from_vec(x_data.clone(), Shape::from_dims(&[b, n, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let mod_data = hash_fill(&[b, mod_dim, dim]);
    let shared_mod = Tensor::from_vec(mod_data.clone(), Shape::from_dims(&[b, mod_dim, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let shift = shared_mod.narrow(1, shift_idx, 1).unwrap().squeeze(Some(1)).unwrap();
    let scale = shared_mod.narrow(1, shift_idx + 1, 1).unwrap().squeeze(Some(1)).unwrap();

    let out = flame_core::bf16_ops::modulate_pre_fused_bf16(&x, &shift, &scale, eps).unwrap();
    let out_vec = out.to_vec().unwrap();

    // Reference math: for each row (bi, ni), LN x then (1+scale)*normed + shift.
    for bi in 0..b {
        for ni in 0..n {
            // Compute mean/variance over dim from the BF16-rounded x values.
            let mut x_row = vec![0.0f32; dim];
            for d in 0..dim {
                let raw = x_data[bi * n * dim + ni * dim + d];
                let rounded = bf16_roundtrip(raw);
                x_row[d] = rounded;
            }
            let mean: f32 = x_row.iter().sum::<f32>() / dim as f32;
            let var: f32 = x_row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for d in 0..dim {
                let sh = bf16_roundtrip(mod_data[bi * mod_dim * dim + shift_idx * dim + d]);
                let sc = bf16_roundtrip(mod_data[bi * mod_dim * dim + (shift_idx + 1) * dim + d]);
                let expected = (1.0 + sc) * (x_row[d] - mean) * inv_std + sh;
                let got = out_vec[bi * n * dim + ni * dim + d];
                let tol = 0.02 * expected.abs().max(1.0);
                let diff = (got - expected).abs();
                assert!(
                    diff < tol,
                    "modulate_pre math mismatch at (b={},n={},d={}): got={} expected={} mean={} inv_std={} sh={} sc={}",
                    bi, ni, d, got, expected, mean, inv_std, sh, sc
                );
            }
        }
    }
}

fn bf16_roundtrip(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
    f32::from_bits((rounded as u32) << 16)
}
