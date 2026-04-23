#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `ops::add_iter::add_bf16_iter` — first BINARY kernel
//! on the TensorIterator port (session 4, 2026-04-22).
//!
//! Four cases:
//!
//! 1. Contig+contig same-shape: the dispatcher must short-circuit to the
//!    existing `bf16_elementwise::add_bf16` fast path, bit-exact.
//! 2. One-strided + one-contig same-shape: cos_sim ≥ 0.9999 against the
//!    materialized contiguous reference. Exercises the iterator on one
//!    strided input with the other trivially contig.
//! 3. Both-strided same-shape (both permuted views with matching logical
//!    shape): cos_sim ≥ 0.9999.
//! 4. Narrow-view + contig same-shape: cos_sim ≥ 0.9999. Exercises the
//!    `base_offset` field on the narrow side.

use flame_core::{ops::add_iter::add_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_add_parity")
}

fn assert_bit_exact_bf16(a: &Tensor, b: &Tensor, tag: &str) -> Result<()> {
    let a_host = a.to_vec_f32()?;
    let b_host = b.to_vec_f32()?;
    assert_eq!(a_host.len(), b_host.len(), "{tag}: length mismatch");
    for (i, (av, bv)) in a_host.iter().zip(b_host.iter()).enumerate() {
        assert_eq!(
            av.to_bits(),
            bv.to_bits(),
            "{tag}: byte mismatch at element {i}: a={av} b={bv}"
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
fn add_iter_contig_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[4096, 512], 0xADD_0001)?;
    let b = make_bf16_tensor(dev, &[4096, 512], 0xADD_0002)?;
    assert!(a.is_contiguous() && b.is_contiguous());

    let ref_out = flame_core::bf16_elementwise::add_bf16(&a, &b)?;
    let new_out = add_bf16_iter(&a, &b)?;

    assert_bit_exact_bf16(&ref_out, &new_out, "add_iter contig+contig")?;
    Ok(())
}

#[test]
fn add_iter_permuted_plus_contig_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    // Build `a` as an H×W contig, then view it as W×H via as_strided.
    let a_contig = make_bf16_tensor(dev.clone(), &[h, w], 0xADD_1111)?;
    let a_view = a_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!a_view.is_contiguous());
    // `b` is a contig tensor already laid out as W×H (matching the view's
    // logical shape).
    let b_contig = make_bf16_tensor(dev, &[w, h], 0xADD_2222)?;
    assert!(b_contig.is_contiguous());

    // Reference: materialize the view then add via the fast path.
    let a_mat = a_view.contiguous()?;
    let ref_out = flame_core::bf16_elementwise::add_bf16(&a_mat, &b_contig)?;

    let new_out = add_bf16_iter(&a_view, &b_contig)?;

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "permuted+contig cos_sim {cs} below threshold 0.9999"
    );
    Ok(())
}

#[test]
fn add_iter_both_permuted_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    // Two independent H×W contig tensors, both permuted to W×H.
    let a_contig = make_bf16_tensor(dev.clone(), &[h, w], 0xADD_3333)?;
    let b_contig = make_bf16_tensor(dev, &[h, w], 0xADD_4444)?;
    let a_view = a_contig.as_strided(&[w, h], &[1, w], 0)?;
    let b_view = b_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!a_view.is_contiguous() && !b_view.is_contiguous());

    let a_mat = a_view.contiguous()?;
    let b_mat = b_view.contiguous()?;
    let ref_out = flame_core::bf16_elementwise::add_bf16(&a_mat, &b_mat)?;

    let new_out = add_bf16_iter(&a_view, &b_view)?;

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "both-permuted cos_sim {cs} below threshold 0.9999"
    );
    Ok(())
}

#[test]
fn add_iter_narrow_view_plus_contig_cos_sim() -> Result<()> {
    let dev = cuda_device();
    // Source [2, 32, 128], narrow dim=1 start=4 length=16 → view [2, 16, 128]
    let src = make_bf16_tensor(dev.clone(), &[2, 32, 128], 0xADD_5555)?;
    let src_strides = src.strides();
    let offset = 4usize * src_strides[1];
    let narrow_view = src.as_strided(&[2, 16, 128], &src_strides, offset)?;
    assert!(!narrow_view.is_contiguous());

    let b_contig = make_bf16_tensor(dev, &[2, 16, 128], 0xADD_6666)?;
    assert!(b_contig.is_contiguous());

    let narrow_mat = narrow_view.contiguous()?;
    let ref_out = flame_core::bf16_elementwise::add_bf16(&narrow_mat, &b_contig)?;

    let new_out = add_bf16_iter(&narrow_view, &b_contig)?;

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "narrow+contig cos_sim {cs} below threshold 0.9999"
    );
    Ok(())
}
