//! Phase 2b regression: `swiglu_fused_bf16` produces identical output
//! for (A) contig inputs and (B) narrow-view inputs of the same logical
//! content. Rules out the stride-path kernel misreading strided storage.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn hash_fill(shape: &[usize]) -> Vec<f32> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 100.0)
        .collect()
}

#[test]
fn swiglu_contig_matches_strided_on_split_gate_up() {
    // Klein's MLP shape: gate_up = [B, N, 2*mlp_hidden], split into halves.
    let (b, n, full) = (1, 64, 384);
    let half = full / 2;
    let device = global_cuda_device();

    let data = hash_fill(&[b, n, full]);
    let gate_up = Tensor::from_vec(data, Shape::from_dims(&[b, n, full]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // A: strided path — narrow produces views, swiglu now accepts strided.
    let gate_view = gate_up.narrow(2, 0, half).unwrap();
    let up_view = gate_up.narrow(2, half, half).unwrap();
    assert!(!gate_view.is_contiguous(), "narrow should return a view");
    assert!(!up_view.is_contiguous(), "narrow should return a view");
    let out_strided = flame_core::bf16_ops::swiglu_fused_bf16(&gate_view, &up_view).unwrap();

    // B: contig path — materialize the views first, then call swiglu.
    let gate_contig = gate_view.contiguous().unwrap();
    let up_contig = up_view.contiguous().unwrap();
    assert!(gate_contig.is_contiguous());
    assert!(up_contig.is_contiguous());
    let out_contig = flame_core::bf16_ops::swiglu_fused_bf16(&gate_contig, &up_contig).unwrap();

    let va = out_strided.to_vec().unwrap();
    let vb = out_contig.to_vec().unwrap();
    assert_eq!(va.len(), vb.len());
    for (i, (a, b)) in va.iter().zip(vb.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "swiglu strided vs contig diverges at idx {}: strided={} contig={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn swiglu_strided_on_klein_mlp_shape() {
    // Actual Klein shape: B=1, N=4608, mlp_hidden=15360 (full gate_up = 30720).
    // Smaller proxy: B=1, N=64, full=256.
    let (b, n, full) = (1, 64, 256);
    let half = full / 2;
    let device = global_cuda_device();

    let data = hash_fill(&[b, n, full]);
    let gate_up = Tensor::from_vec(data, Shape::from_dims(&[b, n, full]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let gate = gate_up.narrow(2, 0, half).unwrap();
    let up = gate_up.narrow(2, half, half).unwrap();
    let out = flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up).unwrap();
    assert_eq!(out.shape().dims(), &[b, n, half]);

    // Reference math: PyTorch BF16 `F.silu(g) * u` where the SiLU result is
    // itself BF16 before the multiply.
    let data_ref = hash_fill(&[b, n, full]);
    let out_vec = out.to_vec().unwrap();
    for i in 0..b * n * half {
        let bi = i / (n * half);
        let ni = (i / half) % n;
        let di = i % half;
        let g = data_ref[bi * n * full + ni * full + 0 + di];
        let u = data_ref[bi * n * full + ni * full + half + di];
        // Both g and u pass through a BF16 round-trip on load.
        let g_bf16 = half_from_f32(g);
        let u_bf16 = half_from_f32(u);
        let g_back = f32_from_bf16_bits(g_bf16);
        let u_back = f32_from_bf16_bits(u_bf16);
        let silu_g = g_back / (1.0 + (-g_back).exp());
        let silu_bf16 = f32_from_bf16_bits(half_from_f32(silu_g));
        let expected_bf16 = half_from_f32(silu_bf16 * u_back);
        let expected = f32_from_bf16_bits(expected_bf16);
        let got = out_vec[i];
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "swiglu mismatch at idx {}: got={} expected={} (g={}, u={})",
            i,
            got,
            expected,
            g_back,
            u_back
        );
    }
}

#[test]
fn swiglu_split_lastdim_matches_view_path() {
    let (b, n, full) = (1, 64, 256);
    let half = full / 2;
    let device = global_cuda_device();

    let data = hash_fill(&[b, n, full]);
    let gate_up = Tensor::from_vec(data, Shape::from_dims(&[b, n, full]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let gate = gate_up.narrow(2, 0, half).unwrap();
    let up = gate_up.narrow(2, half, half).unwrap();
    let old_path = flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up).unwrap();
    let packed_path = flame_core::bf16_ops::swiglu_split_lastdim_bf16(&gate_up).unwrap();

    let va = old_path.to_vec().unwrap();
    let vb = packed_path.to_vec().unwrap();
    assert_eq!(va.len(), vb.len());
    for (i, (a, b)) in va.iter().zip(vb.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "packed swiglu forward diverges at idx {}: old={} packed={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn swiglu_split_lastdim_backward_matches_view_path() {
    fn run(use_packed: bool) -> flame_core::Result<Vec<f32>> {
        flame_core::autograd::AutogradContext::clear();
        let (b, n, full) = (1, 8, 64);
        let half = full / 2;
        let device = global_cuda_device();
        let data = hash_fill(&[b, n, full]);
        let leaf = Tensor::from_vec(data, Shape::from_dims(&[b, n, full]), device.clone())?
            .to_dtype(DType::BF16)?
            .requires_grad_(true);
        let gate_up = leaf.reshape(&[b, n, full])?;

        let out = if use_packed {
            flame_core::bf16_ops::swiglu_split_lastdim_bf16(&gate_up)?
        } else {
            let gate = gate_up.narrow(2, 0, half)?;
            let up = gate_up.narrow(2, half, half)?;
            flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up)?
        };
        let loss = out.to_dtype(DType::F32)?.sum()?;
        let grads = flame_core::autograd::backward(&loss, false)?;
        let grad = grads.get(leaf.id()).expect("leaf grad");
        grad.to_vec()
    }

    let old_grad = run(false).unwrap();
    let packed_grad = run(true).unwrap();
    assert_eq!(old_grad.len(), packed_grad.len());
    for (i, (a, b)) in old_grad.iter().zip(packed_grad.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "packed swiglu backward diverges at idx {}: old={} packed={}",
            i,
            a,
            b
        );
    }
    flame_core::autograd::AutogradContext::clear();
}

fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn f32_from_bf16_bits(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
