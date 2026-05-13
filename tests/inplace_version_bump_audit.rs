//! Phase 0 autograd v2 prereq — in-place version-bump audit tests.
//!
//! Every in-place mutation site in flame-core must call
//! `TensorStorage::bump_version()` after writing so that
//! `SavedTensor::unpack` (Phase 1+ of autograd v2) can detect mutation
//! through saved tensor references at unpack time.
//!
//! Each test in this file follows the same pattern:
//!   1. Construct an input tensor.
//!   2. Record the storage version (via `Tensor::storage_version()`).
//!   3. Run an in-place mutator.
//!   4. Assert the version is now strictly greater.
//!
//! See `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §5 for the
//! authoritative list and rationale.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::adam::{adam_fused_step, adam_fused_step_f32};
use flame_core::ops::elt::{add_inplace_same_dtype, gate_mul_bf16_inplace, mul_inplace_same_dtype};
use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn make_f32(values: Vec<f32>, dims: &[usize]) -> Tensor {
    let device = global_cuda_device().clone();
    Tensor::from_vec(values, Shape::from_dims(dims), device).expect("from_vec")
}

fn make_bf16(values: Vec<f32>, dims: &[usize]) -> Tensor {
    make_f32(values, dims)
        .to_dtype(DType::BF16)
        .expect("to_dtype BF16")
}

// -----------------------------------------------------------------------
// Elementwise in-place (F32 + BF16)
// -----------------------------------------------------------------------

#[test]
fn add_inplace_same_dtype_bumps_version_f32() {
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_f32(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let v0 = a.storage_version();
    add_inplace_same_dtype(&mut a, &b).expect("add_inplace");
    let v1 = a.storage_version();
    assert!(
        v1 > v0,
        "add_inplace_same_dtype (F32) did not bump version: v0={v0}, v1={v1}"
    );
}

#[test]
fn add_inplace_same_dtype_bumps_version_bf16() {
    let mut a = make_bf16(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_bf16(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let v0 = a.storage_version();
    add_inplace_same_dtype(&mut a, &b).expect("add_inplace");
    let v1 = a.storage_version();
    assert!(
        v1 > v0,
        "add_inplace_same_dtype (BF16) did not bump version: v0={v0}, v1={v1}"
    );
}

#[test]
fn mul_inplace_same_dtype_bumps_version_f32() {
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_f32(vec![2.0, 2.0, 2.0, 2.0], &[2, 2]);
    let v0 = a.storage_version();
    mul_inplace_same_dtype(&mut a, &b).expect("mul_inplace");
    let v1 = a.storage_version();
    assert!(
        v1 > v0,
        "mul_inplace_same_dtype (F32) did not bump version: v0={v0}, v1={v1}"
    );
}

#[test]
fn mul_inplace_same_dtype_bumps_version_bf16() {
    let mut a = make_bf16(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_bf16(vec![2.0, 2.0, 2.0, 2.0], &[2, 2]);
    let v0 = a.storage_version();
    mul_inplace_same_dtype(&mut a, &b).expect("mul_inplace");
    let v1 = a.storage_version();
    assert!(
        v1 > v0,
        "mul_inplace_same_dtype (BF16) did not bump version: v0={v0}, v1={v1}"
    );
}

#[test]
fn gate_mul_bf16_inplace_bumps_version() {
    // dst: [B=1, T=2, H=4]; gate: [B=1, H=4]
    let mut dst = make_bf16(vec![1.0; 8], &[1, 2, 4]);
    let gate = make_bf16(vec![2.0, 2.0, 2.0, 2.0], &[1, 4]);
    let v0 = dst.storage_version();
    gate_mul_bf16_inplace(&mut dst, &gate).expect("gate_mul");
    let v1 = dst.storage_version();
    assert!(
        v1 > v0,
        "gate_mul_bf16_inplace did not bump version: v0={v0}, v1={v1}"
    );
}

// -----------------------------------------------------------------------
// Tensor::copy_*
// -----------------------------------------------------------------------

#[test]
fn copy_f32_from_bumps_version() {
    let mut dst = make_f32(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]);
    let src = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let v0 = dst.storage_version();
    dst.copy_f32_from(&src).expect("copy_f32_from");
    let v1 = dst.storage_version();
    assert!(v1 > v0, "copy_f32_from did not bump version");
}

#[test]
fn copy_bf16_region_from_bumps_version() {
    let mut dst = make_bf16(vec![0.0; 8], &[2, 4]);
    let src = make_bf16(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let v0 = dst.storage_version();
    dst.copy_bf16_region_from(0, &src, 0, 8)
        .expect("copy_bf16_region_from");
    let v1 = dst.storage_version();
    assert!(v1 > v0, "copy_bf16_region_from did not bump version");
}

#[test]
fn copy_underscore_bumps_version() {
    let mut dst = make_f32(vec![0.0; 4], &[2, 2]);
    let src = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let v0 = dst.storage_version();
    dst.copy_(&src).expect("copy_");
    let v1 = dst.storage_version();
    // copy_ replaces storage via Arc clone — version of the NEW storage is
    // bumped relative to what it would otherwise be (0 from registration).
    assert!(v1 > v0, "copy_ did not bump version: v0={v0}, v1={v1}");
}

#[test]
fn copy_from_bf16_slice_bumps_version() {
    let mut dst = make_bf16(vec![0.0; 4], &[2, 2]);
    // BF16(1.0) = 0x3F80
    let bits = vec![0x3F80u16; 4];
    let v0 = dst.storage_version();
    dst.copy_from_bf16_slice(&bits)
        .expect("copy_from_bf16_slice");
    let v1 = dst.storage_version();
    assert!(v1 > v0, "copy_from_bf16_slice did not bump version");
}

// -----------------------------------------------------------------------
// Adam optimizer fused steps
// -----------------------------------------------------------------------

#[test]
fn adam_fused_step_bumps_versions() {
    // BF16 param, F32 grad/m/v
    let mut param = make_bf16(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]);
    let grad = make_f32(vec![0.01, 0.01, 0.01, 0.01], &[2, 2]);
    let mut m = make_f32(vec![0.0; 4], &[2, 2]);
    let mut v = make_f32(vec![0.0; 4], &[2, 2]);

    let p_v0 = param.storage_version();
    let m_v0 = m.storage_version();
    let v_v0 = v.storage_version();

    adam_fused_step(
        &mut param, &grad, &mut m, &mut v, 1e-3, 0.9, 0.999, 1e-8, 0.0, 1.0 - 0.9_f32.powi(1),
        1.0 - 0.999_f32.powi(1), None,
    )
    .expect("adam_fused_step");

    assert!(
        param.storage_version() > p_v0,
        "adam_fused_step did not bump param version"
    );
    assert!(
        m.storage_version() > m_v0,
        "adam_fused_step did not bump m version"
    );
    assert!(
        v.storage_version() > v_v0,
        "adam_fused_step did not bump v version"
    );
}

#[test]
fn adam_fused_step_f32_bumps_versions() {
    let mut param = make_f32(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]);
    let grad = make_f32(vec![0.01, 0.01, 0.01, 0.01], &[2, 2]);
    let mut m = make_f32(vec![0.0; 4], &[2, 2]);
    let mut v = make_f32(vec![0.0; 4], &[2, 2]);

    let p_v0 = param.storage_version();
    let m_v0 = m.storage_version();
    let v_v0 = v.storage_version();

    adam_fused_step_f32(
        &mut param, &grad, &mut m, &mut v, 1e-3, 0.9, 0.999, 1e-8, 0.0, 1.0 - 0.9_f32.powi(1),
        1.0 - 0.999_f32.powi(1),
    )
    .expect("adam_fused_step_f32");

    assert!(
        param.storage_version() > p_v0,
        "adam_fused_step_f32 did not bump param version"
    );
    assert!(
        m.storage_version() > m_v0,
        "adam_fused_step_f32 did not bump m version"
    );
    assert!(
        v.storage_version() > v_v0,
        "adam_fused_step_f32 did not bump v version"
    );
}

// -----------------------------------------------------------------------
// Bonus: SavedTensor-like check pattern — illustrate how a hypothetical
// `SavedTensor::unpack` will use the version-counter to detect mutation.
// -----------------------------------------------------------------------

#[test]
fn saved_version_pattern_detects_mutation() {
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Capture a "saved" snapshot of the version, as a future SavedTensor
    // would do at save time.
    let saved_version = a.storage_version();

    // Some other path mutates `a` in place.
    let b = make_f32(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    add_inplace_same_dtype(&mut a, &b).unwrap();

    // At unpack time, SavedTensor compares saved_version to current.
    let now = a.storage_version();
    assert!(
        now > saved_version,
        "version check would fail to detect mutation: saved={saved_version}, now={now}"
    );
}

#[test]
fn no_mutation_no_bump() {
    // Sanity check: reading a tensor does not bump version.
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let v0 = a.storage_version();
    let _vec = a.to_vec_f32().expect("to_vec_f32");
    let v1 = a.storage_version();
    assert_eq!(v0, v1, "read-only access should not bump version");
}
