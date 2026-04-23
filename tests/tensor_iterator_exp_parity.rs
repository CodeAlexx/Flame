#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `ops::exp_iter::exp_bf16_iter` (Phase 7).
//!
//! Exp uses f32 opmath inside the functor (__expf). The reference is
//! computed on the host in f32 (std `exp`) then rounded to BF16. The
//! fast-intrinsic `__expf` delta from libm `exp` is small enough that
//! cos_sim ≥ 0.9999 holds on random BF16 ranges — matches silu/sigmoid
//! threshold which also use `__expf`.

use flame_core::{ops::exp_iter::exp_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_exp_parity")
}

fn cos_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let av = av as f64;
        let bv = bv as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn rne_to_bf16_as_f32(v: f32) -> f32 {
    if v.is_nan() {
        return v;
    }
    let bits = v.to_bits();
    let bias = 0x0000_7FFFu32 + ((bits >> 16) & 1);
    let bf16_bits = (bits.wrapping_add(bias)) >> 16;
    f32::from_bits(bf16_bits << 16)
}

fn exp_cpu_reference(x_bf16_as_f32: &[f32]) -> Vec<f32> {
    x_bf16_as_f32
        .iter()
        .map(|&v| rne_to_bf16_as_f32(v.exp()))
        .collect()
}

fn make_bf16_moderate_tensor(dev: Arc<CudaDevice>, dims: &[usize], seed: u64) -> Result<Tensor> {
    let shape = Shape::from_dims(dims);
    let n = shape.elem_count();
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        // exp saturates beyond ~±88; keep in [-8, 8] to avoid inf/zero domination.
        data.push((u - 0.5) * 16.0);
    }
    let t_f32 = Tensor::from_vec(data, shape, dev)?;
    t_f32.to_dtype(DType::BF16)
}

#[test]
fn exp_iter_contig_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_moderate_tensor(dev, &[4096, 1024], 0xC0FFEE_20)?;
    assert!(x.is_contiguous());

    let x_f32 = x.to_vec_f32()?;
    let ref_f32 = exp_cpu_reference(&x_f32);

    let new_out = exp_bf16_iter(&x)?;
    let new_f32 = new_out.to_vec_f32()?;

    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(cs >= 0.9999, "exp contig cos_sim {cs} below threshold 0.9999");
    Ok(())
}

#[test]
fn exp_iter_permuted_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    let x_contig = make_bf16_moderate_tensor(dev, &[h, w], 0xABCD_20)?;

    let permuted = x_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!permuted.is_contiguous());

    let permuted_mat = permuted.contiguous()?;
    let ref_input_f32 = permuted_mat.to_vec_f32()?;
    let ref_f32 = exp_cpu_reference(&ref_input_f32);

    let new_out = exp_bf16_iter(&permuted)?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(cs >= 0.9999, "exp permuted view cos_sim {cs} below threshold 0.9999");
    Ok(())
}

#[test]
fn exp_iter_narrow_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let src = make_bf16_moderate_tensor(dev, &[2, 32, 128], 0xDEAD_20)?;

    let src_strides = src.strides();
    let offset = 4usize * src_strides[1];
    let narrow_view = src.as_strided(&[2, 16, 128], &src_strides, offset)?;
    assert!(!narrow_view.is_contiguous());

    let narrow_mat = narrow_view.contiguous()?;
    let ref_input_f32 = narrow_mat.to_vec_f32()?;
    let ref_f32 = exp_cpu_reference(&ref_input_f32);

    let new_out = exp_bf16_iter(&narrow_view)?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(cs >= 0.9999, "exp narrow view cos_sim {cs} below threshold 0.9999");
    Ok(())
}

#[test]
fn exp_iter_edge_values() -> Result<()> {
    let dev = cuda_device();
    // Landmarks: 0→1, ln(2)≈0.6931→2, -inf→0, large negative should not NaN.
    let ln2 = std::f32::consts::LN_2; // ≈ 0.6931472
    let values: Vec<f32> = vec![
        0.0, -0.0, ln2, 1.0, -1.0, 10.0, -10.0, f32::NEG_INFINITY, -50.0,
    ];
    let t_f32 = Tensor::from_vec(values.clone(), Shape::from_dims(&[values.len()]), dev)?;
    let x = t_f32.to_dtype(DType::BF16)?;

    let y = exp_bf16_iter(&x)?;
    let y_f32 = y.to_vec_f32()?;

    // Every output must be finite-or-zero (>=0), no NaN.
    for (i, &yi) in y_f32.iter().enumerate() {
        assert!(
            !yi.is_nan(),
            "exp edge element {i}: input={} output=NaN",
            values[i]
        );
        assert!(
            yi >= 0.0 || yi.is_infinite(),
            "exp edge element {i}: input={} output={yi} negative",
            values[i]
        );
    }
    assert_eq!(y_f32[0], 1.0, "exp(0) expected 1.0, got {}", y_f32[0]);
    // exp(ln(2)) should round to 2 in BF16 (ln(2) as BF16 ≈ 0.6953 due to BF16
    // precision loss; exp of that may not be exactly 2.0 after rounding).
    // Use a looser assertion: within BF16 precision.
    let exp_ln2 = y_f32[2];
    assert!(
        (exp_ln2 - 2.0).abs() <= 0.05,
        "exp(ln(2)) expected ~2.0, got {}",
        exp_ln2
    );
    // exp(-inf) → 0.
    assert_eq!(y_f32[7], 0.0, "exp(-inf) expected 0, got {}", y_f32[7]);
    // exp(-50) is a very small BF16 value; must be finite, non-NaN, and
    // <= the BF16-rounded expected value. Not required to be exactly 0:
    // BF16 has exponent range down to ~1.18e-38, so __expf(-50) ≈ 1.9e-22
    // is a valid BF16 subnormal-range value. The requirement is "not NaN".
    assert!(
        !y_f32[8].is_nan() && y_f32[8].is_finite() && y_f32[8] >= 0.0,
        "exp(-50) expected small non-NaN non-inf BF16 value, got {}",
        y_f32[8]
    );
    Ok(())
}
