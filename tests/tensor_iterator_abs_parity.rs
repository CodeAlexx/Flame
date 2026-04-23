#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `ops::abs_iter::abs_bf16_iter` (Phase 6).
//!
//! Abs is native BF16 — the functor clears bit 15 via a reinterpret cast.
//! Reference is computed on the host by loading the BF16 bytes, clearing
//! bit 15, and comparing against the GPU output byte-for-byte. For every
//! finite BF16 (including ±0, ±∞, subnormals): `abs(x) = x & 0x7FFF` is
//! deterministic, so the test is strictly bit-exact.
//!
//! Four cases:
//!   1. contig bit-exact vs CPU reference.
//!   2. permuted view bit-exact.
//!   3. narrow view bit-exact.
//!   4. edge-value test (±0, ±∞, BF16::MAX/MIN, NaN-free).

use flame_core::{ops::abs_iter::abs_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_abs_parity")
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

/// Host-side abs reference: clear the BF16 sign bit.
fn abs_cpu_reference(x_bf16_as_f32: &[f32]) -> Vec<f32> {
    // `to_vec_f32()` returns BF16 values expanded into f32 (low 16 bits zero).
    // The sign-bit flip on the BF16 is equivalent to clearing bit 31 on the
    // f32 representation whose low 16 bits are zero (standard BF16 → f32 layout).
    x_bf16_as_f32
        .iter()
        .map(|&v| {
            let bits = v.to_bits() & 0x7FFF_0000u32;
            f32::from_bits(bits)
        })
        .collect()
}

fn make_bf16_tensor(dev: Arc<CudaDevice>, dims: &[usize], seed: u64) -> Result<Tensor> {
    let shape = Shape::from_dims(dims);
    let n = shape.elem_count();
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        data.push((u - 0.5) * 8.0);
    }
    let t_f32 = Tensor::from_vec(data, shape, dev)?;
    t_f32.to_dtype(DType::BF16)
}

fn assert_bit_exact_vs_ref(new_out: &Tensor, ref_f32: &[f32], tag: &str) -> Result<()> {
    let new_f32 = new_out.to_vec_f32()?;
    assert_eq!(new_f32.len(), ref_f32.len(), "{tag}: length mismatch");
    for (i, (av, bv)) in new_f32.iter().zip(ref_f32.iter()).enumerate() {
        assert_eq!(
            av.to_bits(),
            bv.to_bits(),
            "{tag}: byte mismatch at element {i}: gpu={av} ref={bv}"
        );
    }
    Ok(())
}

#[test]
fn abs_iter_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[4096, 1024], 0xA65_42)?;
    assert!(x.is_contiguous());

    let x_f32 = x.to_vec_f32()?;
    let ref_f32 = abs_cpu_reference(&x_f32);

    let new_out = abs_bf16_iter(&x)?;
    assert_bit_exact_vs_ref(&new_out, &ref_f32, "abs_iter contig")?;
    Ok(())
}

#[test]
fn abs_iter_permuted_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    let x_contig = make_bf16_tensor(dev, &[h, w], 0xABCD_A65)?;

    let permuted = x_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!permuted.is_contiguous());

    let permuted_mat = permuted.contiguous()?;
    let mat_f32 = permuted_mat.to_vec_f32()?;
    let ref_f32 = abs_cpu_reference(&mat_f32);

    let new_out = abs_bf16_iter(&permuted)?;

    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "permuted view cos_sim {cs} below threshold 0.9999"
    );
    // Abs is bit-exact even via the strided path.
    assert_bit_exact_vs_ref(&new_out, &ref_f32, "abs_iter permuted")?;
    Ok(())
}

#[test]
fn abs_iter_narrow_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let src = make_bf16_tensor(dev, &[2, 32, 128], 0xDEAD_A65)?;

    let src_strides = src.strides();
    let offset = 4usize * src_strides[1];
    let narrow_view = src.as_strided(&[2, 16, 128], &src_strides, offset)?;
    assert!(!narrow_view.is_contiguous());
    assert_eq!(narrow_view.offset(), offset);

    let narrow_mat = narrow_view.contiguous()?;
    let mat_f32 = narrow_mat.to_vec_f32()?;
    let ref_f32 = abs_cpu_reference(&mat_f32);

    let new_out = abs_bf16_iter(&narrow_view)?;

    let new_f32 = new_out.to_vec_f32()?;
    assert_eq!(ref_f32.len(), new_f32.len());
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "narrow view cos_sim {cs} below threshold 0.9999"
    );
    assert_bit_exact_vs_ref(&new_out, &ref_f32, "abs_iter narrow")?;
    Ok(())
}

#[test]
fn abs_iter_edge_values() -> Result<()> {
    let dev = cuda_device();
    // BF16-representable edge values. Avoid NaN (bit pattern may differ
    // between paths, and abs of NaN is implementation-defined).
    let values: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 2.0, -2.0,
        f32::INFINITY, f32::NEG_INFINITY,
        // BF16::MAX ≈ 3.389e38 (0x7F7F), BF16::MIN ≈ -3.389e38 (0xFF7F).
        3.389e38, -3.389e38,
        // Smallest positive normal BF16: 2^-126 ≈ 1.175e-38.
        1.175e-38, -1.175e-38,
    ];
    let t_f32 = Tensor::from_vec(values.clone(), Shape::from_dims(&[values.len()]), dev)?;
    let x = t_f32.to_dtype(DType::BF16)?;

    let y = abs_bf16_iter(&x)?;
    let y_f32 = y.to_vec_f32()?;
    let x_f32 = x.to_vec_f32()?;

    for (i, (&xi, &yi)) in x_f32.iter().zip(y_f32.iter()).enumerate() {
        // abs should produce the positive magnitude. For +∞: +∞. For -∞: +∞.
        // For ±0: +0 (bit-exact: 0x0000).
        let expected = if xi == 0.0 { 0.0f32 } else { xi.abs() };
        assert_eq!(
            yi.to_bits(),
            expected.to_bits(),
            "abs edge element {i}: x={xi} y={yi} expected {expected}"
        );
    }
    Ok(())
}
