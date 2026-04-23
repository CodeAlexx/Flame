#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `tensor_iterator::ops::unary::square_bf16_iter` — third kernel
//! on the TensorIterator port (session 3, 2026-04-22). Same 3 cases as
//! silu/gelu sessions.
//!
//! Extra check vs the other sessions: the pre-migration `Tensor::square`
//! for BF16 used `GpuOps::mul(self, self)`. The migration short-circuits
//! contig BF16 to `bf16_ops::square_bf16` instead. Both compute
//! `bf16(fp32(x) * fp32(x))` and should be bit-equivalent, but we
//! include a test that compares `Tensor::square(x)` output (via the new
//! dispatcher) to `x.mul(&x)` (the explicit binary-mul path) on a
//! contig BF16 tensor to prove the dispatch flip is a no-op.

use flame_core::{tensor_iterator::ops::unary::square_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_square_parity")
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
fn square_iter_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[4096, 1024], 0xC0FFEE_44)?;
    assert!(x.is_contiguous());

    let ref_out = flame_core::bf16_ops::square_bf16(&x)?;
    let new_out = square_bf16_iter(&x)?;

    assert_bit_exact_bf16(&ref_out, &new_out, "square_iter contig")?;
    Ok(())
}

/// Session-3-specific: the pre-migration `Tensor::square` path for BF16
/// was `GpuOps::mul(self, self)`. Verify the new dispatcher's output
/// is bit-identical to the explicit `x.mul(&x)` path, so the dispatch
/// flip of `Tensor::square` is guaranteed to be a no-op on every BF16
/// caller.
#[test]
fn square_iter_matches_mul_self_self_contig() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[2048, 512], 0xFEED_1234)?;
    assert!(x.is_contiguous());

    let mul_out = x.mul(&x)?;
    let iter_out = square_bf16_iter(&x)?;

    assert_bit_exact_bf16(&mul_out, &iter_out, "square_iter vs mul(self,self)")?;
    Ok(())
}

#[test]
fn square_iter_permuted_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    let x_contig = make_bf16_tensor(dev, &[h, w], 0xABCD_BBBB)?;

    let permuted = x_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!permuted.is_contiguous());

    let permuted_mat = permuted.contiguous()?;
    let ref_out = flame_core::bf16_ops::square_bf16(&permuted_mat)?;
    let new_out = square_bf16_iter(&permuted)?;

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "permuted view cos_sim {cs} below threshold 0.9999"
    );
    Ok(())
}

#[test]
fn square_iter_narrow_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let src = make_bf16_tensor(dev, &[2, 32, 128], 0xDEAD_CCCC)?;

    let src_strides = src.strides();
    let offset = 4usize * src_strides[1];
    let narrow_view = src.as_strided(&[2, 16, 128], &src_strides, offset)?;
    assert!(!narrow_view.is_contiguous());
    assert_eq!(narrow_view.offset(), offset);

    let narrow_mat = narrow_view.contiguous()?;
    let ref_out = flame_core::bf16_ops::square_bf16(&narrow_mat)?;
    let new_out = square_bf16_iter(&narrow_view)?;

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    assert_eq!(ref_f32.len(), new_f32.len());
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "narrow view cos_sim {cs} below threshold 0.9999"
    );
    Ok(())
}
