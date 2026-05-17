//! Autograd v2 Phase 4b — `GradientMap` `MatchInsertedDtype` policy.
//!
//! Coverage (all gated on `autograd_v2` feature so the default v3 build
//! is unaffected):
//!
//! - `GradientMap::new_v2()` / `with_index_v2()` construct with the
//!   `MatchInsertedDtype` policy; legacy `new()` / `with_index()`
//!   construct with the v1/v3 default `InternalFP32_PublicBF16`.
//! - `insert` under v2 preserves the inserted grad's dtype; under
//!   v1/v3 it upcasts BF16 → F32 (regression-protect the v3 invariant).
//! - `accumulate` under v2 keeps the stored dtype; mismatched-dtype
//!   accumulation returns Err (Phase 4b contract).
//! - `get_public_grad` under v2 returns the native-dtype grad without
//!   casting; under v1/v3 it returns BF16-cast output (unchanged).
//! - `set_ones_dtype` under v2 seeds at the requested dtype; under
//!   v1/v3 it forces F32 regardless (unchanged behavior).
//! - `get_or_create_dtype` under v2 honors the dtype argument; under
//!   v1/v3 it always allocates F32 (unchanged behavior).

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use flame_core::autograd::policy::GradStorePolicy;
use flame_core::gradient::{CompactIndex, GradientMap};
use flame_core::tensor::TensorId;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn dev() -> Arc<CudaDevice> {
    global_cuda_device()
}

// ---------------------------------------------------------------------
// Constructor policy contracts
// ---------------------------------------------------------------------

#[test]
fn gradient_map_new_uses_internal_fp32_policy() {
    let g = GradientMap::new(dev());
    assert_eq!(
        g.policy(),
        GradStorePolicy::InternalFP32_PublicBF16,
        "GradientMap::new must keep the v1/v3 default policy"
    );
}

#[test]
fn gradient_map_new_v2_uses_match_inserted_dtype_policy() {
    let g = GradientMap::new_v2(dev());
    assert_eq!(
        g.policy(),
        GradStorePolicy::MatchInsertedDtype,
        "GradientMap::new_v2 must opt into the v2 dtype-preserving policy"
    );
}

#[test]
fn gradient_map_with_index_uses_internal_fp32_policy() {
    let idx = CompactIndex::from_tensor_ids(std::iter::empty());
    let g = GradientMap::with_index(dev(), idx);
    assert_eq!(g.policy(), GradStorePolicy::InternalFP32_PublicBF16);
}

#[test]
fn gradient_map_with_index_v2_uses_match_inserted_dtype_policy() {
    let idx = CompactIndex::from_tensor_ids(std::iter::empty());
    let g = GradientMap::with_index_v2(dev(), idx);
    assert_eq!(g.policy(), GradStorePolicy::MatchInsertedDtype);
}

// ---------------------------------------------------------------------
// insert: v1 upcasts, v2 preserves
// ---------------------------------------------------------------------

#[test]
fn gradient_map_v1_insert_upcasts_bf16_to_f32() {
    let mut g = GradientMap::new(dev());
    let bf16_grad = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    let id = TensorId::new();
    g.insert(id, bf16_grad).unwrap();
    let stored = g.get(id).expect("insert must store the grad");
    assert_eq!(
        stored.dtype(),
        DType::F32,
        "v1/v3 policy must upcast BF16 grad to F32"
    );
}

#[test]
fn gradient_map_v2_insert_preserves_bf16_dtype() {
    let mut g = GradientMap::new_v2(dev());
    let bf16_grad = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    let id = TensorId::new();
    g.insert(id, bf16_grad).unwrap();
    let stored = g.get(id).expect("insert must store the grad");
    assert_eq!(
        stored.dtype(),
        DType::BF16,
        "MatchInsertedDtype must preserve BF16 grad dtype"
    );
}

#[test]
fn gradient_map_v2_insert_preserves_f32_dtype() {
    let mut g = GradientMap::new_v2(dev());
    let f32_grad = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::F32, dev()).unwrap();
    let id = TensorId::new();
    g.insert(id, f32_grad).unwrap();
    let stored = g.get(id).expect("insert must store the grad");
    assert_eq!(stored.dtype(), DType::F32);
}

// ---------------------------------------------------------------------
// accumulate: v1 upcasts on 2nd grad, v2 preserves and enforces match
// ---------------------------------------------------------------------

#[test]
fn gradient_map_v2_accumulate_preserves_bf16_dtype() {
    let mut g = GradientMap::new_v2(dev());
    let id = TensorId::new();

    let g1 = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    g.accumulate(id, g1).unwrap();

    let g2 = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    g.accumulate(id, g2).unwrap();

    let stored = g.get(id).expect("accumulate must keep the entry");
    assert_eq!(
        stored.dtype(),
        DType::BF16,
        "MatchInsertedDtype must keep BF16 after 2-grad accumulation"
    );
    // Sanity: 1 + 1 = 2 element-wise.
    let host = stored.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    for v in host {
        assert!((v - 2.0).abs() < 1e-3, "accumulated BF16 sum != 2.0: {v}");
    }
}

#[test]
fn gradient_map_v1_accumulate_upcasts_on_second_grad() {
    let mut g = GradientMap::new(dev());
    let id = TensorId::new();

    let g1 = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    g.accumulate(id, g1).unwrap();
    // v1 deferred-upcast: first grad stored as-is (BF16).
    assert_eq!(g.get(id).unwrap().dtype(), DType::BF16);

    let g2 = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    g.accumulate(id, g2).unwrap();
    // v1 upcasts existing on second-grad arrival.
    assert_eq!(
        g.get(id).unwrap().dtype(),
        DType::F32,
        "v1 must upcast existing grad to F32 on second accumulation"
    );
}

#[test]
fn gradient_map_v2_accumulate_errors_on_dtype_mismatch() {
    let mut g = GradientMap::new_v2(dev());
    let id = TensorId::new();

    let bf16 = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    g.accumulate(id, bf16).unwrap();

    let f32 = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::F32, dev()).unwrap();
    let err = g
        .accumulate(id, f32)
        .expect_err("MatchInsertedDtype must error on dtype-mismatched accumulation");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("MatchInsertedDtype") || msg.contains("does not match"),
        "expected dtype-mismatch error, got: {msg}"
    );
}

// ---------------------------------------------------------------------
// get_public_grad / take_public_grads
// ---------------------------------------------------------------------

#[test]
fn gradient_map_v2_get_public_grad_preserves_bf16() {
    let mut g = GradientMap::new_v2(dev());
    let id = TensorId::new();
    let bf16 = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    g.insert(id, bf16).unwrap();
    let public = g.get_public_grad(id).unwrap();
    assert_eq!(
        public.dtype(),
        DType::BF16,
        "v2 public grad must be native dtype (BF16), not cast"
    );
}

#[test]
fn gradient_map_v1_get_public_grad_returns_bf16_cast() {
    // v1 invariant: internally F32, publicly cast to BF16.
    let mut g = GradientMap::new(dev());
    let id = TensorId::new();
    let f32 = Tensor::ones_dtype(Shape::from_dims(&[4]), DType::F32, dev()).unwrap();
    g.insert(id, f32).unwrap();
    let public = g.get_public_grad(id).unwrap();
    assert_eq!(
        public.dtype(),
        DType::BF16,
        "v1 public grad must be BF16-cast (unchanged from pre-Phase-4b)"
    );
}

// ---------------------------------------------------------------------
// set_ones_dtype
// ---------------------------------------------------------------------

#[test]
fn gradient_map_v2_set_ones_dtype_seeds_bf16() {
    let mut g = GradientMap::new_v2(dev());
    let id = TensorId::new();
    g.set_ones_dtype(id, Shape::from_dims(&[4]), DType::BF16)
        .unwrap();
    let stored = g.get(id).unwrap();
    assert_eq!(stored.dtype(), DType::BF16);
}

#[test]
fn gradient_map_v1_set_ones_dtype_still_forces_f32() {
    // v1 invariant: storage is always F32 regardless of requested dtype.
    let mut g = GradientMap::new(dev());
    let id = TensorId::new();
    g.set_ones_dtype(id, Shape::from_dims(&[4]), DType::BF16)
        .unwrap();
    let stored = g.get(id).unwrap();
    assert_eq!(
        stored.dtype(),
        DType::F32,
        "v1 must keep F32 storage even if BF16 dtype was requested"
    );
}

#[test]
fn gradient_map_v1_set_ones_unchanged_behavior() {
    // Pre-Phase-4b regression: `set_ones` legacy signature still
    // produces F32-ones at the given shape on the v1 path.
    let mut g = GradientMap::new(dev());
    let id = TensorId::new();
    g.set_ones(id, Shape::from_dims(&[8])).unwrap();
    let stored = g.get(id).expect("set_ones must populate the grad slot");
    assert_eq!(stored.dtype(), DType::F32);
    assert_eq!(stored.shape().dims(), &[8]);
}

// ---------------------------------------------------------------------
// get_or_create_dtype
// ---------------------------------------------------------------------

#[test]
fn gradient_map_v2_get_or_create_dtype_honors_bf16() {
    let mut g = GradientMap::new_v2(dev());
    let id = TensorId::new();
    {
        let slot = g
            .get_or_create_dtype(id, Shape::from_dims(&[4]), DType::BF16)
            .unwrap();
        assert_eq!(slot.dtype(), DType::BF16);
    }
}

#[test]
fn gradient_map_v1_get_or_create_dtype_keeps_f32_invariant() {
    let mut g = GradientMap::new(dev());
    let id = TensorId::new();
    {
        // v1 ignores requested dtype and allocates F32 (the v3 invariant).
        let slot = g
            .get_or_create_dtype(id, Shape::from_dims(&[4]), DType::BF16)
            .unwrap();
        assert_eq!(
            slot.dtype(),
            DType::F32,
            "v1 must keep F32 storage even if BF16 dtype was requested"
        );
    }
}
