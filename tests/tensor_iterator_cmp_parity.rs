#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Phase 9 parity tests for the 6 comparison ops (ge, gt, le, lt, eq, ne)
//! on the TensorIterator pipeline.
//!
//! Output-dtype note: flame-core's compare output is BF16 0.0/1.0 (not
//! PyTorch's kBool). See `TensorIteratorBase::build_comparison_op` in
//! src/tensor_iterator/base.rs for why. The sentinel values are exact
//! under BF16 rounding (0.0f and 1.0f both have representable BF16
//! bitpatterns), so the CPU reference can assert bit-equality via
//! `to_vec_f32()`.
//!
//! NaN-handling test: IEEE 754 requires `x == x` → false when x is NaN
//! and `x != x` → true. PyTorch matches this. Our functor compares in
//! fp32 (`__bfloat162float`) which preserves BF16 NaNs as NaNs, so the
//! IEEE semantics survive through `__float2bfloat16_rn(1.0 or 0.0)`.

use flame_core::{
    ops::{
        eq_iter::eq_bf16_iter, ge_iter::ge_bf16_iter, gt_iter::gt_bf16_iter,
        le_iter::le_bf16_iter, lt_iter::lt_bf16_iter, ne_iter::ne_bf16_iter,
    },
    DType, Result, Shape, Tensor,
};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_cmp_parity")
}

/// Deterministic BF16 test data in [-4, 4].
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

/// Two BF16 tensors that share several coincident values (for exercising
/// eq/ne). Values are drawn from a small lattice so ~12% of pairs collide.
fn make_bf16_tensor_collide(dev: Arc<CudaDevice>, dims: &[usize], seed: u64) -> Result<Tensor> {
    let shape = Shape::from_dims(dims);
    let n = shape.elem_count();
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // 8 discrete values: {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0}
        let k = ((s >> 40) & 0x7) as u32;
        data.push(-1.5f32 + k as f32 * 0.5);
    }
    let t_f32 = Tensor::from_vec(data, shape, dev)?;
    t_f32.to_dtype(DType::BF16)
}

/// Reference: on BF16 data already materialized via `to_vec_f32()` (where
/// BF16→f32 is exact), the reference is a direct elementwise fp32 compare.
/// This matches the kernel's own math (`__bfloat162float` then IEEE compare).
fn cpu_ref(a_f: &[f32], b_f: &[f32], op: fn(f32, f32) -> bool) -> Vec<f32> {
    a_f.iter()
        .zip(b_f.iter())
        .map(|(x, y)| if op(*x, *y) { 1.0 } else { 0.0 })
        .collect()
}

fn assert_bit_eq_f32(got: &[f32], expected: &[f32], tag: &str) {
    assert_eq!(got.len(), expected.len(), "{tag}: len mismatch");
    let mut fails = 0;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        if g.to_bits() != e.to_bits() {
            fails += 1;
            if fails < 5 {
                eprintln!("{tag}: idx {i}: got {g} ({:#010x}) expected {e} ({:#010x})",
                         g.to_bits(), e.to_bits());
            }
        }
    }
    assert_eq!(fails, 0, "{tag}: {fails} bit mismatches");
}

// -- Per-op contig bit-exact tests ------------------------------------

#[test]
fn cmp_iter_ge_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[128, 96], 0x9_0001)?;
    let b = make_bf16_tensor(dev, &[128, 96], 0x9_0002)?;
    let a_f = a.to_vec_f32()?;
    let b_f = b.to_vec_f32()?;
    let expected = cpu_ref(&a_f, &b_f, |x, y| x >= y);
    let got = ge_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_bit_eq_f32(&got, &expected, "ge contig");
    Ok(())
}

#[test]
fn cmp_iter_gt_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[128, 96], 0x9_0003)?;
    let b = make_bf16_tensor(dev, &[128, 96], 0x9_0004)?;
    let a_f = a.to_vec_f32()?;
    let b_f = b.to_vec_f32()?;
    let expected = cpu_ref(&a_f, &b_f, |x, y| x > y);
    let got = gt_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_bit_eq_f32(&got, &expected, "gt contig");
    Ok(())
}

#[test]
fn cmp_iter_le_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[128, 96], 0x9_0005)?;
    let b = make_bf16_tensor(dev, &[128, 96], 0x9_0006)?;
    let a_f = a.to_vec_f32()?;
    let b_f = b.to_vec_f32()?;
    let expected = cpu_ref(&a_f, &b_f, |x, y| x <= y);
    let got = le_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_bit_eq_f32(&got, &expected, "le contig");
    Ok(())
}

#[test]
fn cmp_iter_lt_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[128, 96], 0x9_0007)?;
    let b = make_bf16_tensor(dev, &[128, 96], 0x9_0008)?;
    let a_f = a.to_vec_f32()?;
    let b_f = b.to_vec_f32()?;
    let expected = cpu_ref(&a_f, &b_f, |x, y| x < y);
    let got = lt_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_bit_eq_f32(&got, &expected, "lt contig");
    Ok(())
}

#[test]
fn cmp_iter_eq_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    // Use collision-friendly generator so eq actually has some 1.0s to find.
    let a = make_bf16_tensor_collide(dev.clone(), &[128, 96], 0x9_0009)?;
    let b = make_bf16_tensor_collide(dev, &[128, 96], 0x9_000A)?;
    let a_f = a.to_vec_f32()?;
    let b_f = b.to_vec_f32()?;
    let expected = cpu_ref(&a_f, &b_f, |x, y| x == y);
    let got = eq_bf16_iter(&a, &b)?.to_vec_f32()?;
    // Sanity: at least some matches so we're really testing.
    let num_true = expected.iter().filter(|&&v| v == 1.0).count();
    assert!(num_true > 100,
        "eq test fixture should produce >100 matches, got {num_true}");
    assert_bit_eq_f32(&got, &expected, "eq contig");
    Ok(())
}

#[test]
fn cmp_iter_ne_contig_bit_exact() -> Result<()> {
    let dev = cuda_device();
    let a = make_bf16_tensor_collide(dev.clone(), &[128, 96], 0x9_000B)?;
    let b = make_bf16_tensor_collide(dev, &[128, 96], 0x9_000C)?;
    let a_f = a.to_vec_f32()?;
    let b_f = b.to_vec_f32()?;
    let expected = cpu_ref(&a_f, &b_f, |x, y| x != y);
    let got = ne_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_bit_eq_f32(&got, &expected, "ne contig");
    Ok(())
}

// -- Output-dtype pin --------------------------------------------------

#[test]
fn cmp_iter_output_dtype_is_bf16() -> Result<()> {
    // Flame-core divergence from PyTorch: compare outputs are BF16 0.0/1.0,
    // not kBool 1-byte values. See `TensorIteratorBase::build_comparison_op`
    // for the rationale. All 6 ops produce BF16 outputs.
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[4, 8], 0x9_00D1)?;
    let b = make_bf16_tensor(dev, &[4, 8], 0x9_00D2)?;
    assert_eq!(ge_bf16_iter(&a, &b)?.dtype(), DType::BF16);
    assert_eq!(gt_bf16_iter(&a, &b)?.dtype(), DType::BF16);
    assert_eq!(le_bf16_iter(&a, &b)?.dtype(), DType::BF16);
    assert_eq!(lt_bf16_iter(&a, &b)?.dtype(), DType::BF16);
    assert_eq!(eq_bf16_iter(&a, &b)?.dtype(), DType::BF16);
    assert_eq!(ne_bf16_iter(&a, &b)?.dtype(), DType::BF16);
    Ok(())
}

// -- Broadcast --------------------------------------------------------

#[test]
fn cmp_iter_broadcast_ge() -> Result<()> {
    // [4, 3, 1] vs [1, 3, 2] → broadcasts to [4, 3, 2].
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[4, 3, 1], 0x9_0101)?;
    let b = make_bf16_tensor(dev, &[1, 3, 2], 0x9_0102)?;
    let got = ge_bf16_iter(&a, &b)?;
    assert_eq!(got.shape().dims(), &[4, 3, 2]);

    // CPU reference: expand both to [4, 3, 2] via to_vec_f32 + manual index math.
    let a_f = a.to_vec_f32()?; // 12 elements in [4,3,1] order
    let b_f = b.to_vec_f32()?; // 6  elements in [1,3,2] order
    let mut expected = vec![0.0f32; 24];
    for i in 0..4 {
        for j in 0..3 {
            for k in 0..2 {
                let a_val = a_f[i * 3 + j]; // [i,j,0] in [4,3,1]
                let b_val = b_f[j * 2 + k]; // [0,j,k] in [1,3,2]
                expected[i * 6 + j * 2 + k] = if a_val >= b_val { 1.0 } else { 0.0 };
            }
        }
    }
    assert_bit_eq_f32(&got.to_vec_f32()?, &expected, "ge broadcast [4,3,1] vs [1,3,2]");
    Ok(())
}

// -- Permuted / strided view -----------------------------------------

#[test]
fn cmp_iter_permuted_view_ge() -> Result<()> {
    // Transpose-as-stride: make a [W, H] view of [H, W] data, compare to a
    // fresh [W, H] tensor, and verify match count vs a pre-contiguous copy.
    let dev = cuda_device();
    let h = 64usize;
    let w = 48usize;
    let a_contig = make_bf16_tensor(dev.clone(), &[h, w], 0x9_0201)?;
    let a_view = a_contig.as_strided(&[w, h], &[1, w], 0)?;
    let b = make_bf16_tensor(dev, &[w, h], 0x9_0202)?;

    // Reference: materialize view first, then compare.
    let ref_out = ge_bf16_iter(&a_view.contiguous()?, &b)?;
    // Direct: let the iterator handle the permuted strides.
    let got_out = ge_bf16_iter(&a_view, &b)?;

    let ref_f = ref_out.to_vec_f32()?;
    let got_f = got_out.to_vec_f32()?;
    assert_bit_eq_f32(&got_f, &ref_f, "ge permuted view");
    // Also check the count — cos_sim isn't meaningful for 0/1 masks, but
    // total-true count IS, and should match exactly.
    let ref_true: f32 = ref_f.iter().sum();
    let got_true: f32 = got_f.iter().sum();
    assert_eq!(ref_true as i32, got_true as i32, "true-count drift");
    Ok(())
}

// -- NaN edge values --------------------------------------------------

#[test]
fn cmp_iter_nan_edge_values() -> Result<()> {
    // IEEE 754: NaN compares false against everything under <, <=, ==, >=, >;
    // NaN != anything (including itself) compares true.
    let dev = cuda_device();
    let nan = f32::NAN;
    let one = 1.0f32;

    // Build BF16 NaN by explicitly constructing a NaN pattern (simpler than
    // relying on f32→bf16 preserving NaN, which it does in `__float2bfloat16_rn`).
    let a_data = vec![nan, nan, one, one, nan, one];
    let b_data = vec![nan, one, nan, one, nan, nan];
    let a = Tensor::from_vec(a_data.clone(), Shape::from_dims(&[6]), dev.clone())?
        .to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(b_data.clone(), Shape::from_dims(&[6]), dev)?
        .to_dtype(DType::BF16)?;

    // Readback & verify NaNs survived the BF16 round-trip; if not, the rest
    // of the test isn't meaningful.
    let a_rt = a.to_vec_f32()?;
    assert!(a_rt[0].is_nan(), "BF16 round-trip lost NaN at index 0");

    // eq: only (1,1) is equal → expected [0,0,0,1,0,0]
    let eq_out = eq_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_eq!(eq_out, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "eq NaN");

    // ne: every NaN-involving pair is ne-true, only (1,1) is false
    let ne_out = ne_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_eq!(ne_out, vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0], "ne NaN");

    // lt/le/gt/ge: any NaN participant → false
    let ge_out = ge_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_eq!(ge_out, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "ge NaN");
    let gt_out = gt_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_eq!(gt_out, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "gt NaN");
    let le_out = le_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_eq!(le_out, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "le NaN");
    let lt_out = lt_bf16_iter(&a, &b)?.to_vec_f32()?;
    assert_eq!(lt_out, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "lt NaN");
    Ok(())
}

// -- Tensor::ge/gt/... dispatch smoke --------------------------------

#[test]
fn cmp_iter_tensor_method_dispatches() -> Result<()> {
    // Smoke-test that Tensor::ge/gt/le/lt/eq/ne route to the new iter
    // path on BF16 inputs and produce the same output as the direct
    // *_bf16_iter call.
    let dev = cuda_device();
    let a = make_bf16_tensor_collide(dev.clone(), &[16, 16], 0x9_0301)?;
    let b = make_bf16_tensor_collide(dev, &[16, 16], 0x9_0302)?;

    assert_bit_eq_f32(
        &a.ge(&b)?.to_vec_f32()?,
        &ge_bf16_iter(&a, &b)?.to_vec_f32()?,
        "Tensor::ge dispatches to iter path",
    );
    assert_bit_eq_f32(
        &a.gt(&b)?.to_vec_f32()?,
        &gt_bf16_iter(&a, &b)?.to_vec_f32()?,
        "Tensor::gt dispatches to iter path",
    );
    assert_bit_eq_f32(
        &a.le(&b)?.to_vec_f32()?,
        &le_bf16_iter(&a, &b)?.to_vec_f32()?,
        "Tensor::le dispatches to iter path",
    );
    assert_bit_eq_f32(
        &a.lt(&b)?.to_vec_f32()?,
        &lt_bf16_iter(&a, &b)?.to_vec_f32()?,
        "Tensor::lt dispatches to iter path",
    );
    assert_bit_eq_f32(
        &a.eq(&b)?.to_vec_f32()?,
        &eq_bf16_iter(&a, &b)?.to_vec_f32()?,
        "Tensor::eq dispatches to iter path",
    );
    assert_bit_eq_f32(
        &a.ne(&b)?.to_vec_f32()?,
        &ne_bf16_iter(&a, &b)?.to_vec_f32()?,
        "Tensor::ne dispatches to iter path",
    );
    Ok(())
}
