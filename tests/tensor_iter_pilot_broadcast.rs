#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Phase 4 broadcast coverage for `tensor_iterator::ops::binary::add_bf16_iter`.
//! Supplements `tensor_iterator_add_parity` which covers the
//! same-shape contig/permuted/narrow cases. The four cases here exercise
//! the BROADCAST path (operand shapes differ, compute_shape infers the
//! union, stride=0 on the broadcasted dim) on the Phase-4 pipeline.
//!
//! Reference: PyTorch TensorIterator port plan §5.2 (binary shape),
//!            plan-this-and-fix-encapsulated-hennessy.md §Phase 4 exit,
//!            Phase 4 Builder brief "tests/tensor_iter_pilot_broadcast.rs".
//!
//! Cases:
//!   1. Same-shape no-broadcast `[4,3,2] + [4,3,2]`: regression-gate the
//!      iterator's happy path against a direct BF16-math reference
//!      (bit-exact — this is identical to `tensor_iterator_add_parity`'s
//!      contig+contig case, but using the broadcast-smaller shape).
//!   2. Broadcast `[4,3,1] + [4,1,2] = [4,3,2]`: cos_sim ≥ 0.9999 against
//!      a materialized reference (the broadcast is stride-0 on a length-1
//!      dim for each operand).
//!   3. Scalar broadcast `[1,1] + [4,3] = [4,3]`: cos_sim ≥ 0.9999.
//!      Exercises the degenerate case where one operand is rank-preserving
//!      but fully broadcast.
//!   4. Permuted view + matching shape `as_strided([w,h], [1,w], 0) + [w,h]`:
//!      cos_sim ≥ 0.9999. Exercises strided input on the new pipeline (a
//!      sibling of `add_iter_permuted_plus_contig_cos_sim` that already
//!      passes, kept here for a per-phase smoke gate).

use flame_core::{tensor_iterator::ops::binary::add_bf16_iter, DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iter_pilot_broadcast")
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

// ---------- 1. Same-shape no-broadcast [4,3,2] + [4,3,2] ----------

#[test]
fn add_iter_broadcast_same_shape_bit_exact() -> Result<()> {
    // Regression gate: the new pipeline's same-shape contig output must
    // match a direct CPU fp32-round-trip-with-rne-rounding reference
    // element-for-element. Phase 5b rewrite: the previous reference
    // (`bf16_elementwise::add_bf16`) was deleted; CPU reference is the
    // equivalent math performed by `AddBF16Op` in src/cuda/binary/add.cu.
    let dev = cuda_device();
    let a = make_bf16_tensor(dev.clone(), &[4, 3, 2], 0xBCA51_0001u64)?;
    let b = make_bf16_tensor(dev, &[4, 3, 2], 0xBCA51_0002u64)?;

    let a_f = a.to_vec_f32()?;
    let b_f = b.to_vec_f32()?;
    let rne_to_bf16_as_f32 = |v: f32| -> f32 {
        if v.is_nan() {
            return v;
        }
        let bits = v.to_bits();
        let bias = 0x0000_7FFFu32 + ((bits >> 16) & 1);
        let bf16_bits = (bits.wrapping_add(bias)) >> 16;
        f32::from_bits(bf16_bits << 16)
    };
    let ref_f: Vec<f32> = a_f
        .iter()
        .zip(b_f.iter())
        .map(|(x, y)| rne_to_bf16_as_f32(x + y))
        .collect();

    let new_out = add_bf16_iter(&a, &b)?;
    let new_f = new_out.to_vec_f32()?;
    assert_eq!(ref_f.len(), new_f.len());
    for (i, (r, n)) in ref_f.iter().zip(new_f.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            n.to_bits(),
            "add_iter same-shape [4,3,2]: byte mismatch at {i}: ref={r} new={n}"
        );
    }
    Ok(())
}

// ---------- 2. Broadcast [4,3,1] + [4,1,2] = [4,3,2] ----------

#[test]
fn add_iter_broadcast_43x1_plus_4x1x2_cos_sim() -> Result<()> {
    let dev = cuda_device();
    // Build `[4,3,1]` and `[4,1,2]` — compute_shape broadcasts to [4,3,2].
    let a = make_bf16_tensor(dev.clone(), &[4, 3, 1], 0xBCA51_1111u64)?;
    let b = make_bf16_tensor(dev.clone(), &[4, 1, 2], 0xBCA51_2222u64)?;

    // Reference: materialize both sides and add via the same iterator
    // (same-shape post-expand), then compare against the broadcast-input
    // output of the iterator. Using add_bf16_iter on both sides means
    // any systematic drift in the new pipeline would show up identically
    // on both sides, so a residual nonzero abs-diff here specifically
    // measures the broadcast path's stride-0 correctness.
    let a_expanded = a.broadcast_to(&Shape::from_dims(&[4, 3, 2]))?;
    let b_expanded = b.broadcast_to(&Shape::from_dims(&[4, 3, 2]))?;
    let ref_out = add_bf16_iter(&a_expanded, &b_expanded)?;

    let new_out = add_bf16_iter(&a, &b)?;
    assert_eq!(new_out.shape().dims(), &[4, 3, 2]);

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "broadcast [4,3,1]+[4,1,2] cos_sim {cs} below 0.9999"
    );
    Ok(())
}

// ---------- 3. Scalar broadcast [1,1] + [4,3] = [4,3] ----------

#[test]
fn add_iter_broadcast_scalar_plus_shape_cos_sim() -> Result<()> {
    let dev = cuda_device();
    // [1,1] is a rank-2 scalar that broadcasts over both dims of [4,3].
    let a_scalar = make_bf16_tensor(dev.clone(), &[1, 1], 0xBCA51_3333u64)?;
    let b = make_bf16_tensor(dev.clone(), &[4, 3], 0xBCA51_4444u64)?;

    let a_expanded = a_scalar.broadcast_to(&Shape::from_dims(&[4, 3]))?;
    let ref_out = add_bf16_iter(&a_expanded, &b)?;

    let new_out = add_bf16_iter(&a_scalar, &b)?;
    assert_eq!(new_out.shape().dims(), &[4, 3]);

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "scalar-broadcast [1,1]+[4,3] cos_sim {cs} below 0.9999"
    );
    Ok(())
}

// ---------- 4. Permuted view + matching shape ----------

#[test]
fn add_iter_broadcast_permuted_view_plus_matching_cos_sim() -> Result<()> {
    let dev = cuda_device();
    let h = 128usize;
    let w = 96usize;
    // Contig `a_src` [h,w], viewed as [w,h] with strides [1,w] (permuted).
    let a_src = make_bf16_tensor(dev.clone(), &[h, w], 0xBCA51_5555u64)?;
    let a_view = a_src.as_strided(&[w, h], &[1, w], 0)?;
    assert!(!a_view.is_contiguous());

    // Contig `b` already in [w,h] frame.
    let b = make_bf16_tensor(dev, &[w, h], 0xBCA51_6666u64)?;
    assert!(b.is_contiguous());

    // Reference: materialize the permuted view, then add via the same
    // pipeline.
    let a_mat = a_view.contiguous()?;
    let ref_out = add_bf16_iter(&a_mat, &b)?;

    let new_out = add_bf16_iter(&a_view, &b)?;
    assert_eq!(new_out.shape().dims(), &[w, h]);

    let ref_f32 = ref_out.to_vec_f32()?;
    let new_f32 = new_out.to_vec_f32()?;
    let cs = cos_sim_f32(&ref_f32, &new_f32);
    assert!(
        cs >= 0.9999,
        "permuted-view + matching shape cos_sim {cs} below 0.9999"
    );
    Ok(())
}
