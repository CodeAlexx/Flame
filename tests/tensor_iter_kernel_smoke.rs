#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Phase 2 smoke test — exercises the new
//! `flame::iter::launch_gpu_kernel<NARGS_IN, func_t>` launcher
//! (`src/cuda/tensor_iterator.cuh`) before any real op lands on it. Covers
//! ranks the session-1 silu parity test doesn't: rank 1, rank 5, rank 6, and
//! a rank-3 permuted-view case (same-stride permutation that forces
//! `OffsetCalculator<1>` to hit non-identity strides on every dim).
//!
//! We piggy-back on `silu_bf16_iter` rather than adding a bespoke copy
//! kernel: the functor is irrelevant to what the test checks (the offset
//! machinery, not the math), and adding a new `.cu` + `extern "C"` pair for
//! the test would duplicate the retarget work. Silu is the simplest
//! already-retargeted kernel.
//!
//! Oracles:
//! * Contig cases → `assert_bit_exact_bf16` against `bf16_ops::silu_bf16` on
//!   the same input. (If the new launcher's offset math is identity-equivalent
//!   on a contig input, every byte of output matches the vectorized
//!   reference.)
//! * Permuted-view case → `cos_sim ≥ 0.9999` against `bf16_ops::silu_bf16`
//!   applied to the materialized (contiguous) permutation. The strided kernel
//!   does per-element scalar math vs the vectorized `__nv_bfloat162` pair
//!   math, so last-ULP rounding differs.
//!
//! Rank 0 (scalar) is intentionally skipped — PyTorch handles degenerate
//! elementwise specially too (`numel == 0` early-return at Loops.cuh:92).

use flame_core::{ops::silu_iter::silu_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iter_kernel_smoke")
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

// ---------------------------------------------------------------------------
// The contig cases route through `silu_bf16_iter`'s `is_contiguous()`
// short-circuit back to `bf16_ops::silu_bf16` — they never exercise the new
// launcher. To actually hit the new launcher on these shapes we as_strided
// them with their own identity strides, which is a no-op mathematically but
// marks the resulting tensor as non-contiguous (`custom_strides: Some(...)`).
// That forces `silu_bf16_iter` down the strided impl.
// ---------------------------------------------------------------------------

fn to_strided_identity(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape().clone();
    let dims = shape.dims();
    let strides: Vec<usize> = x.strides().to_vec();
    x.as_strided(dims, &strides, 0)
}

#[test]
fn launcher_rank1_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    // Rank 1: the simplest shape the new launcher sees. 4096 elems is
    // enough for >1 block at (num_threads * thread_work_size) = 512
    // elements/block.
    let x = make_bf16_tensor(dev, &[4096], 0x1111_1111)?;
    let x_strided = to_strided_identity(&x)?;
    assert!(!x_strided.is_contiguous());

    let ref_out = flame_core::bf16_ops::silu_bf16(&x)?;
    let new_out = silu_bf16_iter(&x_strided)?;

    // Identity strides → per-element offset equals linear index → every byte
    // of output matches the vectorized reference.
    assert_bit_exact_bf16(&ref_out, &new_out, "launcher rank1 identity-strided")?;
    Ok(())
}

#[test]
fn launcher_rank5_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    // Rank 5 exercises more dims of the OffsetCalculator's unroll loop.
    let dims = [2, 3, 4, 5, 6]; // numel = 720
    let x = make_bf16_tensor(dev, &dims, 0x2222_2222)?;
    let x_strided = to_strided_identity(&x)?;
    assert!(!x_strided.is_contiguous());

    let ref_out = flame_core::bf16_ops::silu_bf16(&x)?;
    let new_out = silu_bf16_iter(&x_strided)?;

    assert_bit_exact_bf16(&ref_out, &new_out, "launcher rank5 identity-strided")?;
    Ok(())
}

#[test]
fn launcher_rank6_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    // Rank 6: flame-core's FLAME_MAX_DIMS. Any regression in the
    // OffsetCalculator's dim loop shows up here.
    let dims = [2, 2, 3, 3, 2, 2]; // numel = 144
    let x = make_bf16_tensor(dev, &dims, 0x3333_3333)?;
    let x_strided = to_strided_identity(&x)?;
    assert!(!x_strided.is_contiguous());

    let ref_out = flame_core::bf16_ops::silu_bf16(&x)?;
    let new_out = silu_bf16_iter(&x_strided)?;

    assert_bit_exact_bf16(&ref_out, &new_out, "launcher rank6 identity-strided")?;
    Ok(())
}

#[test]
fn launcher_rank3_permuted_view_cos_sim() -> Result<()> {
    let dev = cuda_device();
    // Build [A=8, B=12, C=16] contig, view as [B, A, C] via as_strided so
    // every dim has a non-trivial stride. Sanity check: each OffsetCalculator
    // dim contributes a non-identity term.
    let a = 8usize;
    let b = 12usize;
    let c = 16usize;
    let x_contig = make_bf16_tensor(dev, &[a, b, c], 0x4444_4444)?;

    // strides of the source are [b*c, c, 1]; the [B, A, C] permute takes dim 1
    // of the source to dim 0 of the view, dim 0 to dim 1, dim 2 to dim 2.
    // New strides: [c, b*c, 1].
    let permuted = x_contig.as_strided(&[b, a, c], &[c, b * c, 1], 0)?;
    assert!(!permuted.is_contiguous());

    let permuted_mat = permuted.contiguous()?;
    let ref_out = flame_core::bf16_ops::silu_bf16(&permuted_mat)?;
    let new_out = silu_bf16_iter(&permuted)?;

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    assert_eq!(ref_f32.len(), new_f32.len());
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "rank-3 permuted view cos_sim {cs} below threshold 0.9999"
    );
    Ok(())
}
