//! Phase 2b-4 regression: `gate_residual_fused_bf16` produces identical
//! output for (A) narrow-view gate inputs and (B) contiguous gate inputs.
//! Same test pattern as `modulate_pre_stride_parity.rs`.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn hash_fill(shape: &[usize]) -> Vec<f32> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 512.0 - 1.0)
        .collect()
}

#[test]
fn gate_residual_contig_matches_strided_on_shared_mod_narrow() {
    // Klein pattern: gate = shared_mod.narrow(1, j*dim, dim) → 2D [B, dim]
    // with stride[0] = mod_dim*dim. B=2 to make the batch stride matter.
    let (b, n, dim, mod_dim) = (2, 64, 128, 6);
    let gate_idx = 2_usize; // e.g. gate_msa
    let device = global_cuda_device();

    let r_data = hash_fill(&[b, n, dim]);
    let residual = Tensor::from_vec(r_data, Shape::from_dims(&[b, n, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let x_data = hash_fill(&[b, n, dim]);
    let x = Tensor::from_vec(x_data, Shape::from_dims(&[b, n, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let mod_data = hash_fill(&[b, mod_dim * dim]);
    let shared_mod = Tensor::from_vec(mod_data, Shape::from_dims(&[b, mod_dim * dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let gate_strided = shared_mod.narrow(1, gate_idx * dim, dim).unwrap();
    assert_eq!(gate_strided.shape().dims(), &[b, dim]);
    assert!(!gate_strided.is_contiguous(), "gate-view should be strided");

    let out_strided = flame_core::bf16_ops::gate_residual_fused_bf16(
        &residual, &gate_strided, &x,
    )
    .unwrap();

    let gate_contig = gate_strided.contiguous().unwrap();
    assert!(gate_contig.is_contiguous());
    let out_contig = flame_core::bf16_ops::gate_residual_fused_bf16(
        &residual, &gate_contig, &x,
    )
    .unwrap();

    let va = out_strided.to_vec().unwrap();
    let vb = out_contig.to_vec().unwrap();
    assert_eq!(va.len(), vb.len());
    for (i, (a, b)) in va.iter().zip(vb.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "gate_residual strided vs contig diverges at idx {}: strided={} contig={}",
            i, a, b
        );
    }
}

#[test]
fn gate_residual_strided_math_check() {
    // Hand-verify the math: out[b,n,d] = residual[b,n,d] + gate[b,d] * x[b,n,d].
    let (b, n, dim, mod_dim) = (2, 8, 32, 3);
    let gate_idx = 0_usize;
    let device = global_cuda_device();

    let r_data = hash_fill(&[b, n, dim]);
    let residual = Tensor::from_vec(r_data.clone(), Shape::from_dims(&[b, n, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let x_data = hash_fill(&[b, n, dim]);
    let x = Tensor::from_vec(x_data.clone(), Shape::from_dims(&[b, n, dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let mod_data = hash_fill(&[b, mod_dim * dim]);
    let shared_mod = Tensor::from_vec(mod_data.clone(), Shape::from_dims(&[b, mod_dim * dim]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let gate = shared_mod.narrow(1, gate_idx * dim, dim).unwrap();
    let out = flame_core::bf16_ops::gate_residual_fused_bf16(&residual, &gate, &x).unwrap();
    let out_vec = out.to_vec().unwrap();

    for bi in 0..b {
        for ni in 0..n {
            for di in 0..dim {
                let r_val = bf16_roundtrip(r_data[bi * n * dim + ni * dim + di]);
                let x_val = bf16_roundtrip(x_data[bi * n * dim + ni * dim + di]);
                let g_val = bf16_roundtrip(mod_data[bi * mod_dim * dim + gate_idx * dim + di]);
                let expected = r_val + g_val * x_val;
                let got = out_vec[bi * n * dim + ni * dim + di];
                let tol = 0.02 * expected.abs().max(1.0);
                let diff = (got - expected).abs();
                assert!(
                    diff < tol,
                    "gate_residual math mismatch at (b={},n={},d={}): got={} expected={}",
                    bi, ni, di, got, expected
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
