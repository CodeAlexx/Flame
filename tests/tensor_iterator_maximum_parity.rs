#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `ops::maximum_iter::maximum_bf16_iter` (Phase 5b).
//! Reference: CPU fp32 fmaxf + rne BF16 rounding.

use flame_core::{ops::maximum_iter::maximum_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_maximum_parity")
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

fn assert_bit_exact_f32_slices(a: &[f32], b: &[f32], tag: &str) {
    assert_eq!(a.len(), b.len(), "{tag}: length mismatch");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            av.to_bits(),
            bv.to_bits(),
            "{tag}: byte mismatch at element {i}: a={av} b={bv}"
        );
    }
}

fn assert_bit_exact_bf16_tensors(a: &Tensor, b: &Tensor, tag: &str) -> Result<()> {
    assert_bit_exact_f32_slices(&a.to_vec_f32()?, &b.to_vec_f32()?, tag);
    Ok(())
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
fn maximum_iter_contig_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[4096, 512], 0x5B70_0001)?;
    let b = make_bf16_tensor(dev, &[4096, 512], 0x5B70_0002)?;

    let a_f = a.to_vec_f32()?;
    let b_f = b.to_vec_f32()?;
    // fmaxf-equivalent: pick the larger finite, matching the functor.
    let ref_f: Vec<f32> = a_f
        .iter()
        .zip(b_f.iter())
        .map(|(x, y)| rne_to_bf16_as_f32(x.max(*y)))
        .collect();

    let new_out = maximum_bf16_iter(&a, &b)?;
    assert_bit_exact_f32_slices(
        &ref_f,
        &new_out.to_vec_f32()?,
        "maximum_iter contig+contig",
    );
    Ok(())
}

#[test]
fn maximum_iter_permuted_plus_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    let a_contig = make_bf16_tensor(dev.clone(), &[h, w], 0x5B70_1111)?;
    let a_view = a_contig.as_strided(&[w, h], &[1, w], 0)?;
    let b_contig = make_bf16_tensor(dev, &[w, h], 0x5B70_2222)?;

    let a_mat = a_view.contiguous()?;
    let ref_out = maximum_bf16_iter(&a_mat, &b_contig)?;
    let new_out = maximum_bf16_iter(&a_view, &b_contig)?;
    assert_bit_exact_bf16_tensors(&ref_out, &new_out, "maximum_iter permuted+contig")?;
    Ok(())
}

#[test]
fn maximum_iter_both_permuted_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    let a_contig = make_bf16_tensor(dev.clone(), &[h, w], 0x5B70_3333)?;
    let b_contig = make_bf16_tensor(dev, &[h, w], 0x5B70_4444)?;
    let a_view = a_contig.as_strided(&[w, h], &[1, w], 0)?;
    let b_view = b_contig.as_strided(&[w, h], &[1, w], 0)?;

    let a_mat = a_view.contiguous()?;
    let b_mat = b_view.contiguous()?;
    let ref_out = maximum_bf16_iter(&a_mat, &b_mat)?;
    let new_out = maximum_bf16_iter(&a_view, &b_view)?;
    assert_bit_exact_bf16_tensors(&ref_out, &new_out, "maximum_iter both-permuted")?;
    Ok(())
}

#[test]
fn maximum_iter_narrow_view_plus_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let src = make_bf16_tensor(dev.clone(), &[2, 32, 128], 0x5B70_5555)?;
    let src_strides = src.strides();
    let offset = 4usize * src_strides[1];
    let narrow_view = src.as_strided(&[2, 16, 128], &src_strides, offset)?;

    let b_contig = make_bf16_tensor(dev, &[2, 16, 128], 0x5B70_6666)?;

    let narrow_mat = narrow_view.contiguous()?;
    let ref_out = maximum_bf16_iter(&narrow_mat, &b_contig)?;
    let new_out = maximum_bf16_iter(&narrow_view, &b_contig)?;
    assert_bit_exact_bf16_tensors(&ref_out, &new_out, "maximum_iter narrow+contig")?;
    Ok(())
}
