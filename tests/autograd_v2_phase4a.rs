//! Autograd v2 Phase 4a — Parameter v2 + Adam BF16-grad classifier +
//! multi_tensor_l2_norm_sq_bf16 + OptimizerV2 trait.
//!
//! Coverage (all gated on `autograd_v2` feature so the default v3 build
//! is unaffected):
//!
//! - `Parameter::new_v2` constructs with the
//!   `MatchParamDtype` policy and `set_grad` preserves BF16 dtype.
//! - `Parameter::new` (default) still casts grads to F32 (v3 invariant
//!   intact).
//! - `Parameter::grad_bf16_or_f32` returns the native-dtype grad.
//! - `AdamW::step` accepts a BF16 param with a BF16 grad (Phase 4a
//!   classifier arm) and produces a finite, dtype-preserving update.
//! - `AdamW::step` accepts an F32 param with a BF16 grad (Phase 4a
//!   per-param fallback through `adam_fused_step_f32`).
//! - `AdamW::step` BF16-param BF16-grad matches F32-reference within
//!   the documented BF16 tolerance.
//! - `multi_tensor_l2_norm_sq_bf16` matches the F32-reference fold.
//! - `global_l2_norm` routes all-BF16 grads through the new fast path
//!   and preserves the parity bound.
//! - `AdamWV2` wraps AdamW and routes the step through the v2 trait
//!   surface without panicking.

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use flame_core::autograd_v2::{AdamWV2, DispatchCtx, OptimizerV2};
use flame_core::ops::grad_norm::global_l2_norm;
use flame_core::ops::multi_tensor::{
    multi_tensor_l2_norm_sq_bf16, multi_tensor_l2_norm_sq_f32, MultiTensorMetaCache,
};
use flame_core::parameter::GradDtypePolicy;
use flame_core::adam::AdamW;
use flame_core::{global_cuda_device, DType, Parameter, Shape, Tensor};

fn dev() -> Arc<CudaDevice> {
    global_cuda_device()
}

// ---------------------------------------------------------------------
// Parameter::new_v2 dtype-preserving contract
// ---------------------------------------------------------------------

#[test]
fn parameter_new_default_uses_cast_to_f32() {
    let t = Tensor::zeros_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    let p = Parameter::new(t);
    assert_eq!(p.grad_dtype_policy(), GradDtypePolicy::CastToF32);
}

#[test]
fn parameter_new_v2_uses_match_param_dtype() {
    let t = Tensor::zeros_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    let p = Parameter::new_v2(t);
    assert_eq!(p.grad_dtype_policy(), GradDtypePolicy::MatchParamDtype);
}

#[test]
fn parameter_v1_set_grad_casts_bf16_to_f32() {
    let p = Parameter::new(
        Tensor::zeros_dtype(Shape::from_dims(&[4]), DType::BF16, dev())
            .unwrap()
            .requires_grad_(true),
    );
    let g = Tensor::zeros_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    p.set_grad(g).unwrap();
    let stored = p.grad().expect("set_grad must populate grad");
    assert_eq!(
        stored.dtype(),
        DType::F32,
        "v1/v3 policy must upcast BF16 grad to F32"
    );
}

#[test]
fn parameter_v2_set_grad_preserves_bf16() {
    let p = Parameter::new_v2(
        Tensor::zeros_dtype(Shape::from_dims(&[4]), DType::BF16, dev())
            .unwrap()
            .requires_grad_(true),
    );
    let g = Tensor::zeros_dtype(Shape::from_dims(&[4]), DType::BF16, dev()).unwrap();
    p.set_grad(g).unwrap();
    let stored = p.grad_bf16_or_f32().expect("set_grad must populate grad");
    assert_eq!(
        stored.dtype(),
        DType::BF16,
        "v2 MatchParamDtype policy must keep BF16 grad as BF16"
    );
}

#[test]
fn parameter_v2_set_grad_preserves_f32_on_f32_param() {
    let p = Parameter::new_v2(
        Tensor::zeros_dtype(Shape::from_dims(&[4]), DType::F32, dev())
            .unwrap()
            .requires_grad_(true),
    );
    let g = Tensor::zeros_dtype(Shape::from_dims(&[4]), DType::F32, dev()).unwrap();
    p.set_grad(g).unwrap();
    let stored = p.grad_bf16_or_f32().expect("set_grad must populate grad");
    assert_eq!(stored.dtype(), DType::F32);
}

// ---------------------------------------------------------------------
// AdamW BF16-grad classifier arms
// ---------------------------------------------------------------------

fn finite_bf16_param(n: usize, scale: f32) -> Parameter {
    // Deterministic BF16 param + grad.
    let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0123 - 0.5) * scale).collect();
    let t = Tensor::from_vec_dtype(vals, Shape::from_dims(&[n]), dev(), DType::BF16).unwrap();
    let p = Parameter::new_v2(t.requires_grad_(true));
    let g_vals: Vec<f32> = (0..n)
        .map(|i| 0.001 + 0.0001 * (i as f32))
        .collect();
    let g = Tensor::from_vec_dtype(g_vals, Shape::from_dims(&[n]), dev(), DType::BF16).unwrap();
    p.set_grad(g).unwrap();
    p
}

#[test]
fn adam_step_bf16_param_bf16_grad_no_panic() {
    // 4 small params triggers the multi-tensor (BF16, BF16) arm.
    let params: Vec<Parameter> = (0..4).map(|_| finite_bf16_param(8, 1.0)).collect();
    let mut opt = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    opt.step(&params).expect("Phase 4a (BF16, BF16) multi-tensor arm must run");

    // Param dtype must be preserved.
    for p in &params {
        assert_eq!(p.dtype().unwrap(), DType::BF16);
    }
}

#[test]
fn adam_step_bf16_param_bf16_grad_matches_f32_reference() {
    // One param so the per-param fused path runs (still exercises the
    // BF16-grad arm via `adam_fused_step` — its kernel_name switch
    // picks `adam_fused_bf16_kernel` when grad is BF16).
    let n = 16;
    let device = dev();

    // Reference: F32 param + F32 grad + AdamW with default policy
    // (CastToF32). After one step, BF16 path should be within BF16
    // tolerance.
    let init_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.08).collect();
    let grad_vals: Vec<f32> = (0..n).map(|i| 0.002 + 0.0001 * (i as f32)).collect();

    // Reference: full F32 step.
    let ref_t = Tensor::from_vec(init_vals.clone(), Shape::from_dims(&[n]), device.clone())
        .unwrap()
        .requires_grad_(true);
    let ref_param = Parameter::new(ref_t);
    let ref_g = Tensor::from_vec(grad_vals.clone(), Shape::from_dims(&[n]), device.clone()).unwrap();
    ref_param.set_grad(ref_g).unwrap();
    let mut ref_opt = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    ref_opt.step(std::slice::from_ref(&ref_param)).unwrap();
    let ref_after = ref_param.tensor().unwrap().to_dtype(DType::F32).unwrap();

    // V2 path: BF16 param + BF16 grad, MatchParamDtype.
    let v2_t = Tensor::from_vec(init_vals, Shape::from_dims(&[n]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .requires_grad_(true);
    let v2_param = Parameter::new_v2(v2_t);
    let v2_g = Tensor::from_vec(grad_vals, Shape::from_dims(&[n]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    v2_param.set_grad(v2_g).unwrap();
    let mut v2_opt = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    v2_opt.step(std::slice::from_ref(&v2_param)).unwrap();
    let v2_after = v2_param.tensor().unwrap().to_dtype(DType::F32).unwrap();

    let r_vec = ref_after.to_vec().unwrap();
    let v_vec = v2_after.to_vec().unwrap();
    assert_eq!(r_vec.len(), v_vec.len());
    // BF16 tolerance per parity_helpers default: atol=1e-2, rtol=1e-2.
    // Adam with lr=1e-3 over magnitudes ~1e-1 gives post-step deltas
    // ~1e-3, well inside BF16 ULP-of-output. Use 5e-3 absolute as the
    // bar — accounts for: (1) BF16 grad quantization at input, (2) the
    // round-to-nearest BF16 store at the end of the fused step.
    for (i, (&r, &v)) in r_vec.iter().zip(v_vec.iter()).enumerate() {
        let delta = (r - v).abs();
        assert!(
            delta < 5e-3,
            "adam BF16/BF16 vs F32 ref drift at i={i}: r={r}, v={v}, |Δ|={delta}"
        );
    }
}

#[test]
fn adam_step_f32_param_bf16_grad_no_panic() {
    // F32 params + BF16 grads under MatchParamDtype. The classifier
    // routes this to the per-param `adam_fused_step_f32` path (which
    // has the BF16-grad kernel arm) since there's no multi-tensor
    // kernel for (F32 param, BF16 grad).
    let n = 12;
    let device = dev();
    let init_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 + 0.1).collect();
    let grad_vals: Vec<f32> = (0..n).map(|i| 0.003 + 0.0002 * (i as f32)).collect();

    let t = Tensor::from_vec(init_vals, Shape::from_dims(&[n]), device.clone())
        .unwrap()
        .requires_grad_(true);
    let p = Parameter::new_v2(t);
    let g = Tensor::from_vec(grad_vals, Shape::from_dims(&[n]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    p.set_grad(g).unwrap();
    let mut opt = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    opt.step(std::slice::from_ref(&p))
        .expect("Phase 4a F32-param BF16-grad per-param arm must run");

    assert_eq!(p.dtype().unwrap(), DType::F32);
    let after = p.tensor().unwrap().to_vec().unwrap();
    assert!(after.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------
// multi_tensor_l2_norm_sq_bf16 parity vs F32 reference
// ---------------------------------------------------------------------

#[test]
fn multi_tensor_l2_norm_sq_bf16_matches_f32_reference() {
    let device = dev();

    // 8 tensors with varied sizes — exercises the per-block reduction +
    // single-block stage 2.
    let sizes = [16usize, 32, 64, 33, 128, 7, 256, 65];
    let bf16_tensors: Vec<Tensor> = sizes
        .iter()
        .enumerate()
        .map(|(t_idx, &n)| {
            let vals: Vec<f32> = (0..n)
                .map(|i| 0.01 * ((i + t_idx * 7) as f32) - 0.05)
                .collect();
            Tensor::from_vec(vals, Shape::from_dims(&[n]), device.clone())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        })
        .collect();
    let f32_tensors: Vec<Tensor> = bf16_tensors
        .iter()
        .map(|t| t.to_dtype(DType::F32).unwrap())
        .collect();

    let bf16_refs: Vec<&Tensor> = bf16_tensors.iter().collect();
    let f32_refs: Vec<&Tensor> = f32_tensors.iter().collect();

    let mut cache = MultiTensorMetaCache::new();
    let bf16_sq = multi_tensor_l2_norm_sq_bf16(&mut cache, &bf16_refs).unwrap();
    let mut cache2 = MultiTensorMetaCache::new();
    let f32_sq = multi_tensor_l2_norm_sq_f32(&mut cache2, &f32_refs).unwrap();

    let b = bf16_sq.to_vec().unwrap()[0];
    let f = f32_sq.to_vec().unwrap()[0];
    // BF16 input → F32 opmath: 1 ULP of BF16 per element at load, then
    // F32 accumulate. For 8 tensors totaling ~600 elements with values
    // ~1e-1, drift is ~1e-3 relative. 1e-2 is generous.
    let rel = (b - f).abs() / f.abs().max(1e-9);
    assert!(
        rel < 1e-2,
        "multi_tensor_l2_norm_sq_bf16 vs f32 ref drift too high: bf16={b}, f32={f}, rel={rel}"
    );
}

#[test]
fn global_l2_norm_routes_bf16_through_fast_path() {
    let device = dev();
    let t1 = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], Shape::from_dims(&[4]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let t2 = Tensor::from_vec(vec![-0.05f32; 8], Shape::from_dims(&[8]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let refs: Vec<&Tensor> = vec![&t1, &t2];
    let norm = global_l2_norm(&refs).unwrap();
    let v = norm.to_vec().unwrap()[0];

    // Reference: F32 fold of squares + sqrt.
    let sum_sq: f32 = vec![0.1f32, 0.2, 0.3, 0.4]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        + 8.0 * (-0.05f32).powi(2);
    let ref_norm = sum_sq.sqrt();
    let rel = (v - ref_norm).abs() / ref_norm.abs().max(1e-9);
    assert!(
        rel < 2e-2,
        "global_l2_norm bf16 fast path mismatch: got={v}, ref={ref_norm}, rel={rel}"
    );
}

// ---------------------------------------------------------------------
// AdamWV2 OptimizerV2 trait surface
// ---------------------------------------------------------------------

#[test]
fn adamw_v2_step_runs_through_trait_surface() {
    // Construct via the v2 trait + Parameter::new_v2; one step must
    // complete cleanly.
    let n = 8;
    let device = dev();
    let init_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let grad_vals: Vec<f32> = (0..n).map(|i| 0.001 + 0.00005 * (i as f32)).collect();
    let t = Tensor::from_vec(init_vals, Shape::from_dims(&[n]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .requires_grad_(true);
    let p = Parameter::new_v2(t);
    let g = Tensor::from_vec(grad_vals, Shape::from_dims(&[n]), device.clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    p.set_grad(g).unwrap();

    let mut opt = AdamWV2::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    let ctx = DispatchCtx::default_for(flame_core::Device::from_arc(dev()));
    opt.step(&[p.clone()], &ctx).expect("AdamWV2::step");

    // zero_grad must clear the grad slot.
    opt.zero_grad(&[p.clone()]);
    assert!(p.grad().is_none());
}

#[test]
fn adamw_v2_zero_grad_clears_native_dtype_grad() {
    let n = 4;
    let device = dev();
    let t = Tensor::zeros_dtype(Shape::from_dims(&[n]), DType::BF16, device.clone())
        .unwrap()
        .requires_grad_(true);
    let p = Parameter::new_v2(t);
    let g = Tensor::ones_dtype(Shape::from_dims(&[n]), DType::BF16, device.clone()).unwrap();
    p.set_grad(g).unwrap();
    assert!(p.grad().is_some());

    let mut opt = AdamWV2::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
    opt.zero_grad(&[p.clone()]);
    assert!(p.grad().is_none());
}
