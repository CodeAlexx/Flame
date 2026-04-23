#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `tensor_iterator::ops::unary::silu_bf16_iter` — the first kernel on
//! flame-core's TensorIterator port
//! (HANDOFF_2026-04-22_TENSORITERATOR_PORT).
//!
//! Three cases:
//!
//! 1. **Contig bit-exact**. Contiguous BF16 input must hit the short-circuit
//!    branch in `silu_bf16_iter` and return literally the same bytes as the
//!    existing `bf16_ops::silu_bf16` vectorized NVRTC kernel — not
//!    `cos_sim ≥ 0.9999`, but every byte identical. This gates the perf
//!    promise (the fast path is unchanged).
//!
//! 2. **Permuted view**. A `[H, W]` tensor viewed as `[W, H]` via
//!    `as_strided` has non-trivial strides but `view_offset == 0`. The
//!    strided kernel's output must match `bf16_ops::silu_bf16` applied to
//!    the materialized (contiguous) copy at `cos_sim ≥ 0.9999`. This is the
//!    proof the `OffsetCalculator` actually computes the right offsets.
//!
//! 3. **Narrow view** (non-zero `view_offset`). A hand-built narrow view
//!    along a middle dimension (mirrors what `Tensor::narrow` will eventually
//!    return once it flips to view-return — many sessions away). Tests the
//!    `base_offset` field of `StridedOffsetCalc`.

use flame_core::{tensor_iterator::ops::unary::silu_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_silu_parity")
}

/// Compare two BF16 tensors as raw u16 byte slices. Fails the test on any
/// difference. Used for the contig bit-exact gate.
fn assert_bit_exact_bf16(a: &Tensor, b: &Tensor, tag: &str) -> Result<()> {
    let a_host = a.to_vec_f32()?;
    let b_host = b.to_vec_f32()?;
    assert_eq!(a_host.len(), b_host.len(), "{tag}: length mismatch");
    for (i, (av, bv)) in a_host.iter().zip(b_host.iter()).enumerate() {
        // BF16 → f32 is lossless; comparing as f32 bits is sufficient.
        assert_eq!(
            av.to_bits(),
            bv.to_bits(),
            "{tag}: byte mismatch at element {i}: a={av} b={bv}"
        );
    }
    Ok(())
}

/// Cosine similarity (f32) between two equal-length buffers. Returns 1.0 for
/// exact match, scales with noise proportionally. Used for the strided gate
/// where small rounding differences between the vectorized reference and
/// the scalar strided kernel are expected.
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
    // Deterministic, moderate-magnitude samples. Silu is well-conditioned
    // over [-4, 4] so any reasonable range suffices.
    let shape = Shape::from_dims(dims);
    let n = shape.elem_count();
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map top 24 bits to [-4.0, 4.0).
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        data.push((u - 0.5) * 8.0);
    }
    let t_f32 = Tensor::from_vec(data, shape, dev)?;
    t_f32.to_dtype(DType::BF16)
}

#[test]
fn silu_iter_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    // A shape typical of DiT MLP activations: [batch*seq, hidden].
    // 4096 × 1024 = ~8 MB, plenty of elements to shake out lane errors.
    let x = make_bf16_tensor(dev, &[4096, 1024], 0xC0FFEE_42)?;
    assert!(x.is_contiguous());

    let ref_out = flame_core::bf16_ops::silu_bf16(&x)?;
    let new_out = silu_bf16_iter(&x)?;

    assert_bit_exact_bf16(&ref_out, &new_out, "silu_iter contig")?;
    Ok(())
}

#[test]
fn silu_iter_permuted_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    // Build a [H=256, W=384] contig tensor. Permute to [W, H] via
    // as_strided: shape=[W,H], strides=[1, W].
    let h = 256usize;
    let w = 384usize;
    let x_contig = make_bf16_tensor(dev, &[h, w], 0xABCD_9999)?;

    let permuted = x_contig.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!permuted.is_contiguous());

    // Reference: materialize the view, run the existing vectorized kernel.
    let permuted_mat = permuted.contiguous()?;
    let ref_out = flame_core::bf16_ops::silu_bf16(&permuted_mat)?;

    // Under test: stride-aware silu on the view directly.
    let new_out = silu_bf16_iter(&permuted)?;

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
fn silu_iter_narrow_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    // Source: [B=2, T=32, D=128]. Narrow dim=1 start=4 length=16 → view
    // shape [2, 16, 128], inherits source strides [32*128, 128, 1],
    // view_offset = 4 * 128 = 512.
    let src = make_bf16_tensor(dev, &[2, 32, 128], 0xDEAD_F00D)?;

    let src_strides = src.strides();
    let offset = 4usize * src_strides[1];
    let narrow_view = src.as_strided(&[2, 16, 128], &src_strides, offset)?;
    // Narrow views retain source strides but have non-zero view_offset.
    assert!(!narrow_view.is_contiguous());
    assert_eq!(narrow_view.offset(), offset);

    // Reference: materialize via `contiguous()` (which dispatches to
    // `materialize_view` for non-zero offset), then the vectorized silu.
    let narrow_mat = narrow_view.contiguous()?;
    let ref_out = flame_core::bf16_ops::silu_bf16(&narrow_mat)?;

    let new_out = silu_bf16_iter(&narrow_view)?;

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
