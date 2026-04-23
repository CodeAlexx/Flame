//! Phase 2b-3 regression: `modulate_pre_split_apply_bf16` produces
//! identical output whether the 3D modulation tensor is contig or a
//! permute view.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn hash_fill(shape: &[usize]) -> Vec<f32> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 512.0 - 1.0)
        .collect()
}

#[test]
fn modulate_pre_split_apply_contig_matches_strided_permute_view() {
    // B=2 so the batch stride actually gets dereferenced (with B=1 strided
    // and contig paths produce identical output regardless of stride[0]).
    let (b, n, dim, mod_dim) = (2, 16, 64, 6);
    let shift_idx = 3_usize; // MLP pair
    let device = global_cuda_device();

    let x_data = hash_fill(&[b, n, dim]);
    let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, n, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Construct `mod_strided` as a permute-view: start contig [B, dim, mod_dim],
    // permute (1,2) → [B, mod_dim, dim] with strides (mod_dim*dim, 1, mod_dim).
    // Then contig_reference = mod_strided.contiguous().
    let packed_data = hash_fill(&[b, dim, mod_dim]);
    let packed = Tensor::from_vec(packed_data, Shape::from_dims(&[b, dim, mod_dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let mod_strided = packed.transpose_dims(1, 2).unwrap();
    assert_eq!(mod_strided.shape().dims(), &[b, mod_dim, dim]);
    assert!(!mod_strided.is_contiguous(), "permute view should not be contig");

    let out_strided = flame_core::bf16_ops::modulate_pre_split_apply_bf16(
        &x, &mod_strided, shift_idx, 1e-5,
    )
    .unwrap();

    let mod_contig = mod_strided.contiguous().unwrap();
    assert!(mod_contig.is_contiguous());
    let out_contig = flame_core::bf16_ops::modulate_pre_split_apply_bf16(
        &x, &mod_contig, shift_idx, 1e-5,
    )
    .unwrap();

    let va = out_strided.to_vec().unwrap();
    let vb = out_contig.to_vec().unwrap();
    assert_eq!(va.len(), vb.len());
    for (i, (a, b)) in va.iter().zip(vb.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "split_apply strided vs contig diverges at idx {}: strided={} contig={}",
            i, a, b
        );
    }
}

#[test]
fn modulate_pre_split_apply_strided_math_on_klein_like_shape() {
    // Verify strided output matches hand-computed LN + modulate.
    let (b, n, dim, mod_dim) = (2, 8, 32, 6);
    let shift_idx = 0_usize; // MSA pair
    let eps = 1e-5_f32;
    let device = global_cuda_device();

    let x_data = hash_fill(&[b, n, dim]);
    let x = Tensor::from_vec(x_data.clone(), Shape::from_dims(&[b, n, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Permute view modulation.
    let packed_data = hash_fill(&[b, dim, mod_dim]);
    let packed = Tensor::from_vec(packed_data.clone(), Shape::from_dims(&[b, dim, mod_dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let mod_view = packed.transpose_dims(1, 2).unwrap();

    let out = flame_core::bf16_ops::modulate_pre_split_apply_bf16(&x, &mod_view, shift_idx, eps).unwrap();
    let out_vec = out.to_vec().unwrap();

    // Reference: for each (b, n), LN over dim then (1+sc)*normed + sh.
    // mod_view[b, k, d] == packed[b, d, k] — that's the semantic transpose.
    for bi in 0..b {
        for ni in 0..n {
            let mut x_row = vec![0.0f32; dim];
            for d in 0..dim {
                x_row[d] = bf16_roundtrip(x_data[bi * n * dim + ni * dim + d]);
            }
            let mean: f32 = x_row.iter().sum::<f32>() / dim as f32;
            let var: f32 = x_row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for d in 0..dim {
                // mod_view[bi, shift_idx, d]     == packed[bi, d, shift_idx]
                // mod_view[bi, shift_idx + 1, d] == packed[bi, d, shift_idx + 1]
                let sh = bf16_roundtrip(packed_data[bi * dim * mod_dim + d * mod_dim + shift_idx]);
                let sc = bf16_roundtrip(packed_data[bi * dim * mod_dim + d * mod_dim + shift_idx + 1]);
                let expected = (1.0 + sc) * (x_row[d] - mean) * inv_std + sh;
                let got = out_vec[bi * n * dim + ni * dim + d];
                let tol = 0.02 * expected.abs().max(1.0);
                let diff = (got - expected).abs();
                assert!(
                    diff < tol,
                    "split_apply math mismatch at (b={},n={},d={}): got={} expected={} sh={} sc={}",
                    bi, ni, d, got, expected, sh, sc
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
