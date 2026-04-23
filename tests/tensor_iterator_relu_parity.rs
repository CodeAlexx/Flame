#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `tensor_iterator::ops::unary::relu_bf16_iter` (Phase 6).
//!
//! Relu is native BF16 via `__hgt(x, 0) ? x : 0`. Reference is computed on
//! the host: for each BF16 input value, if it is > 0 pass it through; else
//! return 0. Finite deterministic math; bit-exact is the appropriate gate.
//!
//! Four cases:
//!   1. contig bit-exact vs CPU reference.
//!   2. permuted view bit-exact.
//!   3. narrow view bit-exact.
//!   4. edge values (±0, ±∞, BF16::MAX/MIN).

use flame_core::{tensor_iterator::ops::unary::relu_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_relu_parity")
}

/// Host-side relu reference: `max(x, 0)` on the BF16 values.
fn relu_cpu_reference(x_bf16_as_f32: &[f32]) -> Vec<f32> {
    x_bf16_as_f32
        .iter()
        .map(|&v| if v > 0.0 { v } else { 0.0 })
        .collect()
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

#[test]
fn relu_iter_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[4096, 1024], 0xC0FFEE_65)?;
    assert!(x.is_contiguous());

    let x_f32 = x.to_vec_f32()?;
    let ref_f32 = relu_cpu_reference(&x_f32);

    let new_out = relu_bf16_iter(&x)?;
    assert_bit_exact_vs_ref(&new_out, &ref_f32, "relu_iter contig")?;
    Ok(())
}

#[test]
fn relu_iter_permuted_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    let x_contig = make_bf16_tensor(dev, &[h, w], 0xABCD_65)?;

    let permuted = x_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!permuted.is_contiguous());

    let permuted_mat = permuted.contiguous()?;
    let mat_f32 = permuted_mat.to_vec_f32()?;
    let ref_f32 = relu_cpu_reference(&mat_f32);

    let new_out = relu_bf16_iter(&permuted)?;

    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "permuted view cos_sim {cs} below threshold 0.9999"
    );
    assert_bit_exact_vs_ref(&new_out, &ref_f32, "relu_iter permuted")?;
    Ok(())
}

#[test]
fn relu_iter_narrow_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let src = make_bf16_tensor(dev, &[2, 32, 128], 0xDEAD_65)?;

    let src_strides = src.strides();
    let offset = 4usize * src_strides[1];
    let narrow_view = src.as_strided(&[2, 16, 128], &src_strides, offset)?;
    assert!(!narrow_view.is_contiguous());
    assert_eq!(narrow_view.offset(), offset);

    let narrow_mat = narrow_view.contiguous()?;
    let mat_f32 = narrow_mat.to_vec_f32()?;
    let ref_f32 = relu_cpu_reference(&mat_f32);

    let new_out = relu_bf16_iter(&narrow_view)?;

    let new_f32 = new_out.to_vec_f32()?;
    assert_eq!(ref_f32.len(), new_f32.len());
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "narrow view cos_sim {cs} below threshold 0.9999"
    );
    assert_bit_exact_vs_ref(&new_out, &ref_f32, "relu_iter narrow")?;
    Ok(())
}

#[test]
fn relu_iter_edge_values() -> Result<()> {
    let dev = cuda_device();
    // Avoid NaN — relu of NaN is implementation-defined.
    let values: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 2.0, -2.0,
        f32::INFINITY, f32::NEG_INFINITY,
        3.389e38, -3.389e38,
        1.175e-38, -1.175e-38,
    ];
    let t_f32 = Tensor::from_vec(values.clone(), Shape::from_dims(&[values.len()]), dev)?;
    let x = t_f32.to_dtype(DType::BF16)?;

    let y = relu_bf16_iter(&x)?;
    let y_f32 = y.to_vec_f32()?;
    let x_f32 = x.to_vec_f32()?;

    for (i, (&xi, &yi)) in x_f32.iter().zip(y_f32.iter()).enumerate() {
        let expected = if xi > 0.0 { xi } else { 0.0 };
        assert_eq!(
            yi.to_bits(),
            expected.to_bits(),
            "relu edge element {i}: x={xi} y={yi} expected {expected}"
        );
    }
    Ok(())
}
