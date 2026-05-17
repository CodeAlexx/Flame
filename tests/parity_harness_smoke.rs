#![cfg(feature = "cuda")]
//! Smoke test for `flame_core::parity::ParityHarness`.
//!
//! Builds a synthetic 3-tensor reference dump in a temp file, runs the
//! harness against three Rust-side tensors (one identical, one slightly
//! perturbed, one shape-mismatched) and verifies the result classification
//! and report rendering.

use flame_core::device::Device;
use flame_core::parity::{ParityHarness, ParityTolerance};
use flame_core::serialization::{save_tensors, SerializationFormat};
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

fn cuda_device() -> Device {
    Device::cuda(0)
        .expect("CUDA GPU required for parity smoke (skip with --features '' if unavailable)")
}

#[test]
fn parity_harness_detects_match_drift_and_shape_mismatch() -> Result<()> {
    let device = cuda_device().cuda_device().clone();
    let _ = flame_core::rng::set_seed(7);

    // Build a reference dump.
    let mut dump = HashMap::new();
    let a = Tensor::randn(Shape::from_dims(&[16, 32]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::F32)?;
    let b =
        Tensor::randn(Shape::from_dims(&[8, 8]), 0.0, 0.5, device.clone())?.to_dtype(DType::F32)?;
    let c =
        Tensor::randn(Shape::from_dims(&[4, 4]), 0.0, 1.0, device.clone())?.to_dtype(DType::F32)?;
    dump.insert("layer.exact".to_string(), a.clone());
    dump.insert("layer.drifted".to_string(), b.clone());
    dump.insert("layer.shape_ref".to_string(), c.clone());

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("flame-parity-smoke-{nanos}.safetensors"));
    save_tensors(&dump, &path, SerializationFormat::SafeTensors)?;

    // Load via the harness with a strict tolerance.
    let mut h = ParityHarness::load(&path, device.clone())?.with_tolerance(ParityTolerance {
        min_cos: 0.9999,
        max_abs_ratio: 0.01,
    });

    // Case 1: identical tensor -> passes.
    let r_exact = h.compare("layer.exact", &a)?;
    assert!(r_exact.passed, "exact-match should pass; got {:?}", r_exact);
    assert!(
        r_exact.cos > 0.999_99,
        "cos should be ~1; got {}",
        r_exact.cos
    );

    // Case 2: slight perturbation that exceeds 1% tolerance -> fails on
    // ratio (and likely on cos too at this magnitude).
    let noise = Tensor::randn(b.shape().clone(), 0.0, 0.5, device.clone())?.to_dtype(DType::F32)?;
    let drifted = b.add(&noise)?;
    let r_drift = h.compare("layer.drifted", &drifted)?;
    assert!(!r_drift.passed, "drifted should fail; got {:?}", r_drift);
    assert!(r_drift.note.is_none(), "drift is numeric, not a note");

    // Case 3: shape mismatch -> fails with a note, no metric NaN panic.
    let wrong_shape =
        Tensor::zeros(Shape::from_dims(&[3, 3]), device.clone())?.to_dtype(DType::F32)?;
    let r_shape = h.compare("layer.shape_ref", &wrong_shape)?;
    assert!(!r_shape.passed);
    assert!(
        r_shape
            .note
            .as_deref()
            .unwrap_or("")
            .contains("shape mismatch"),
        "expected shape-mismatch note; got {:?}",
        r_shape.note
    );

    // Case 4: missing key -> fails with a note.
    let r_missing = h.compare("layer.does_not_exist", &a)?;
    assert!(!r_missing.passed);
    assert!(
        r_missing.note.as_deref().unwrap_or("").contains("no key"),
        "expected missing-key note; got {:?}",
        r_missing.note
    );

    // Report shape sanity.
    let report = h.report();
    assert!(report.contains("layer.exact"));
    assert!(report.contains("FAIL"));
    assert!(report.contains("OK"));
    assert!(!h.is_clean());
    let _ = std::fs::remove_file(&path);
    Ok(())
}
