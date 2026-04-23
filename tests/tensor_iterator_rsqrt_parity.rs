#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `tensor_iterator::ops::transcendentals::rsqrt_bf16_iter` (Phase 7).
//!
//! Rsqrt uses f32 opmath inside the functor (__frsqrt_rn). The reference
//! is computed on the host in f32 (`1.0 / v.sqrt()`) then rounded to BF16.

use flame_core::{tensor_iterator::ops::transcendentals::rsqrt_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_rsqrt_parity")
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

fn rsqrt_cpu_reference(x_bf16_as_f32: &[f32]) -> Vec<f32> {
    x_bf16_as_f32
        .iter()
        .map(|&v| rne_to_bf16_as_f32(1.0f32 / v.sqrt()))
        .collect()
}

fn make_bf16_positive_tensor(dev: Arc<CudaDevice>, dims: &[usize], seed: u64) -> Result<Tensor> {
    let shape = Shape::from_dims(dims);
    let n = shape.elem_count();
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        // Rsqrt domain: [0.1, 100.1) — keep bounded away from 0 for cos_sim.
        data.push(u * 100.0 + 0.1);
    }
    let t_f32 = Tensor::from_vec(data, shape, dev)?;
    t_f32.to_dtype(DType::BF16)
}

#[test]
fn rsqrt_iter_contig_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_positive_tensor(dev, &[4096, 1024], 0xC0FFEE_19)?;
    assert!(x.is_contiguous());

    let x_f32 = x.to_vec_f32()?;
    let ref_f32 = rsqrt_cpu_reference(&x_f32);

    let new_out = rsqrt_bf16_iter(&x)?;
    let new_f32 = new_out.to_vec_f32()?;

    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(cs >= 0.9999, "rsqrt contig cos_sim {cs} below threshold 0.9999");
    Ok(())
}

#[test]
fn rsqrt_iter_permuted_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    let x_contig = make_bf16_positive_tensor(dev, &[h, w], 0xABCD_19)?;

    let permuted = x_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!permuted.is_contiguous());

    let permuted_mat = permuted.contiguous()?;
    let ref_input_f32 = permuted_mat.to_vec_f32()?;
    let ref_f32 = rsqrt_cpu_reference(&ref_input_f32);

    let new_out = rsqrt_bf16_iter(&permuted)?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(cs >= 0.9999, "rsqrt permuted view cos_sim {cs} below threshold 0.9999");
    Ok(())
}

#[test]
fn rsqrt_iter_narrow_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let src = make_bf16_positive_tensor(dev, &[2, 32, 128], 0xDEAD_19)?;

    let src_strides = src.strides();
    let offset = 4usize * src_strides[1];
    let narrow_view = src.as_strided(&[2, 16, 128], &src_strides, offset)?;
    assert!(!narrow_view.is_contiguous());

    let narrow_mat = narrow_view.contiguous()?;
    let ref_input_f32 = narrow_mat.to_vec_f32()?;
    let ref_f32 = rsqrt_cpu_reference(&ref_input_f32);

    let new_out = rsqrt_bf16_iter(&narrow_view)?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(cs >= 0.9999, "rsqrt narrow view cos_sim {cs} below threshold 0.9999");
    Ok(())
}

#[test]
fn rsqrt_iter_edge_values() -> Result<()> {
    let dev = cuda_device();
    // Landmarks: 1→1, 4→0.5, 0.25→2, 0→+∞, -1→NaN.
    let values: Vec<f32> = vec![1.0, 4.0, 0.25, 0.0, -1.0];
    let t_f32 = Tensor::from_vec(values.clone(), Shape::from_dims(&[values.len()]), dev)?;
    let x = t_f32.to_dtype(DType::BF16)?;

    let y = rsqrt_bf16_iter(&x)?;
    let y_f32 = y.to_vec_f32()?;

    assert_eq!(y_f32[0], 1.0, "rsqrt(1) expected 1, got {}", y_f32[0]);
    assert_eq!(y_f32[1], 0.5, "rsqrt(4) expected 0.5, got {}", y_f32[1]);
    assert_eq!(y_f32[2], 2.0, "rsqrt(0.25) expected 2, got {}", y_f32[2]);
    assert!(y_f32[3].is_infinite() && y_f32[3] > 0.0, "rsqrt(0) expected +inf, got {}", y_f32[3]);
    assert!(y_f32[4].is_nan(), "rsqrt(-1) expected NaN, got {}", y_f32[4]);
    Ok(())
}
