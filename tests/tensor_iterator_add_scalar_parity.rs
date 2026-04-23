#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `tensor_iterator::ops::binary::add_scalar_bf16_iter` (Phase 5b).
//!
//! Reference: legacy `ops::elt::add_scalar_same_dtype` (GpuOps::add_scalar
//! BF16 branch). Same fp32 add → rne BF16.

use flame_core::{tensor_iterator::ops::binary::add_scalar_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_add_scalar_parity")
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
fn add_scalar_iter_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[4096, 512], 0x5D_0001)?;
    assert!(x.is_contiguous());

    let ref_out = flame_core::ops::elt::add_scalar_same_dtype(&x, 0.375)?;
    let new_out = add_scalar_bf16_iter(&x, 0.375)?;

    assert_bit_exact_bf16(&ref_out, &new_out, "add_scalar_iter contig")?;
    Ok(())
}

#[test]
fn add_scalar_iter_permuted_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 256usize;
    let w = 384usize;
    let x_contig = make_bf16_tensor(dev, &[h, w], 0x5D_1111)?;
    let x_view = x_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!x_view.is_contiguous());

    let x_mat = x_view.contiguous()?;
    let ref_out = flame_core::ops::elt::add_scalar_same_dtype(&x_mat, -1.5)?;

    let new_out = add_scalar_bf16_iter(&x_view, -1.5)?;

    let cs = cos_sim_f32(&ref_out.to_vec_f32()?, &new_out.to_vec_f32()?);
    assert!(cs >= 0.9999, "permuted cos_sim {cs} below 0.9999");
    Ok(())
}

#[test]
fn add_scalar_iter_edge_scalars() -> Result<()> {
    // scalar=0 is identity (bit-exact). scalar=1 shifts every element by 1.
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[128, 64], 0x5DEDE000)?;

    let out_zero = add_scalar_bf16_iter(&x, 0.0)?;
    assert_bit_exact_bf16(&x, &out_zero, "add by 0 == identity")?;

    let out_one = add_scalar_bf16_iter(&x, 1.0)?;
    let x_host = x.to_vec_f32()?;
    let o_host = out_one.to_vec_f32()?;
    for (xi, oi) in x_host.iter().zip(o_host.iter()) {
        // bf16(x + 1) matches a round-trip fp32 add + round.
        let expected = {
            // Simulate: fp32 + 1, round to BF16.
            let v = xi + 1.0f32;
            let bf16_bits = {
                let bits = v.to_bits();
                let rounding_bias = 0x0000_7FFFu32 + ((bits >> 16) & 1);
                ((bits.wrapping_add(rounding_bias)) >> 16) as u16
            };
            // Convert back to f32: lower bits zero, upper bits from bf16.
            f32::from_bits((bf16_bits as u32) << 16)
        };
        assert_eq!(
            oi.to_bits(),
            expected.to_bits(),
            "add_scalar by 1: input {xi} -> got {oi}, expected {expected}"
        );
    }
    Ok(())
}
