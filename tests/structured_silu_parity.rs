#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for the Phase 4 structured-kernel exemplar
//! (`Tensor::silu_structured` / `structured::SiluStructured`).
//!
//! Two gates:
//!
//! 1. **Forward bit-exact.** `Tensor::silu_structured()` must produce
//!    byte-identical output to `Tensor::silu()` on contiguous BF16 input.
//!    Same kernel, same data ptr, same iter — only the dispatch shape
//!    differs.
//!
//! 2. **Backward match.** Running each path under autograd through a
//!    `sum().backward()` chain must produce identical input gradients.
//!    Validates that `SiluStructured::dispatch` records autograd
//!    correctly (same `Op::SiLU` shape as `Tensor::silu`).

use flame_core::autograd::AutogradContext;
use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for structured_silu_parity")
}

fn make_bf16_tensor(dev: Arc<CudaDevice>, dims: &[usize], seed: u64) -> Result<Tensor> {
    let shape = Shape::from_dims(dims);
    let n = shape.elem_count();
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        data.push((u - 0.5) * 8.0);
    }
    let t_f32 = Tensor::from_vec(data, shape, dev)?;
    t_f32.to_dtype(DType::BF16)
}

/// Compare two BF16 tensors as raw f32-roundtripped bytes (BF16 → f32 is
/// lossless, so f32-bit comparison is sufficient for bit-exact).
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

#[test]
fn structured_silu_forward_matches_silu() -> Result<()> {
    let dev = cuda_device();
    // Same shape used by tensor_iterator_silu_parity: typical DiT MLP
    // activation, big enough to shake out lane errors.
    let x = make_bf16_tensor(dev, &[4096, 1024], 0xC0FFEE_42)?;
    assert!(x.is_contiguous());

    let ref_out = x.silu()?;
    let new_out = x.silu_structured()?;

    assert_bit_exact_bf16(&ref_out, &new_out, "structured_silu forward")?;
    Ok(())
}

#[test]
fn structured_silu_backward_matches_silu() -> Result<()> {
    let dev = cuda_device();
    // Smaller shape — backward is the gate here, not throughput. Still
    // big enough to cover multiple kernel blocks.
    let x_base = make_bf16_tensor(dev, &[128, 256], 0xDEAD_BEEF_42)?;

    // Path A: existing Tensor::silu
    let grad_a = {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let x = x_base.clone().requires_grad_(true);
        let xid = x.id();
        let y = x.silu()?;
        let loss = y.sum()?;
        let grads = AutogradContext::backward(&loss)?;
        let g = grads
            .get(xid)
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation("no grad for x in silu path".into())
            })?
            .clone();
        AutogradContext::set_enabled(false);
        g
    };

    // Path B: Tensor::silu_structured
    let grad_b = {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let x = x_base.clone().requires_grad_(true);
        let xid = x.id();
        let y = x.silu_structured()?;
        let loss = y.sum()?;
        let grads = AutogradContext::backward(&loss)?;
        let g = grads
            .get(xid)
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation("no grad for x in silu_structured path".into())
            })?
            .clone();
        AutogradContext::set_enabled(false);
        g
    };

    // Same op (Op::SiLU) → same backward kernel → bit-exact gradients.
    assert_bit_exact_bf16(&grad_a, &grad_b, "structured_silu backward")?;
    Ok(())
}

#[test]
fn structured_silu_no_grad_skips_recording() -> Result<()> {
    // requires_grad=false → silu_structured runs without touching the
    // autograd tape. Sanity: still produces the right answer.
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[64, 64], 0xBABE_4321)?;
    assert!(!x.requires_grad());

    let ref_out = x.silu()?;
    let new_out = x.silu_structured()?;
    assert!(!new_out.requires_grad());
    assert_bit_exact_bf16(&ref_out, &new_out, "structured_silu no-grad")?;
    Ok(())
}
