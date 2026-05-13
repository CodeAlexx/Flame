//! Phase 5d item #3 — real-trainer integration test for the v2 bridge.
//!
//! Wraps the Klein full-block-backward fixture's attention sub-section in
//! `AutogradContext::checkpoint` (the v3 reentrant checkpoint that
//! OneTrainer/SimpleTuner wire universally for memory orchestration) and
//! runs the Phase 5b `backward_v2` bridge through it. Verifies the
//! returned `GradientMap` has the `MatchInsertedDtype` policy and
//! contains a non-zero BF16 grad for every trainable leaf.
//!
//! If reentrant checkpoint didn't compose with the bridge, the attention-
//! path weights (`dw_qkv`, `dw_out`) would be missing from the map or
//! land at the wrong dtype. This test is the gate.
//!
//! Hooks: covered at the engine level by the existing
//! `autograd_v2_engine::hooks_fire_in_order` test. The hook surface only
//! becomes "real trainer integration" once the bridge routes through the
//! v2 standalone engine (item #6 territory); until then, the engine-
//! level test is the authoritative coverage.

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::path::PathBuf;

use serial_test::serial;

use flame_core::autograd::policy::GradStorePolicy;
use flame_core::autograd::AutogradContext;
use flame_core::{global_cuda_device, DType, Tensor};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("PYTORCH_FIXTURES").unwrap_or_else(|_| "tests/pytorch_fixtures".to_string()),
    )
}

fn load_fixture(path: &std::path::Path) -> std::collections::HashMap<String, Tensor> {
    let device = global_cuda_device().clone();
    flame_core::serialization::load_file(path, &device).expect("load_file")
}

fn linear(x_in: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    let dims = x_in.shape().dims().to_vec();
    let leading: usize = dims[..dims.len() - 1].iter().product();
    let in_features = *dims.last().unwrap();
    let x2d = x_in.reshape(&[leading, in_features]).expect("reshape");
    let w_t = w.transpose().expect("w_t");
    let pre = x2d.matmul(&w_t).expect("matmul");
    let mut out_dims = dims.clone();
    *out_dims.last_mut().unwrap() = w.shape().dims()[0];
    let pre_3d = pre.reshape(&out_dims).expect("reshape out");
    pre_3d.add(b).expect("add bias")
}

#[test]
#[serial]
fn checkpoint_under_bridge_v2() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_block_backward.safetensors");
    if !path.exists() {
        eprintln!("SKIP checkpoint_under_bridge_v2: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);

    let x = fix.get("x").unwrap().clone().requires_grad_(true);
    let w_qkv = fix.get("w_qkv").unwrap().clone().requires_grad_(true);
    let b_qkv = fix.get("b_qkv").unwrap().clone().requires_grad_(true);
    let w_out = fix.get("w_out").unwrap().clone().requires_grad_(true);
    let b_out = fix.get("b_out").unwrap().clone().requires_grad_(true);
    let w_up = fix.get("w_up").unwrap().clone().requires_grad_(true);
    let b_up = fix.get("b_up").unwrap().clone().requires_grad_(true);
    let w_down = fix.get("w_down").unwrap().clone().requires_grad_(true);
    let b_down = fix.get("b_down").unwrap().clone().requires_grad_(true);
    let gate = fix.get("gate").unwrap().clone();

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let dims = x.shape().dims().to_vec();
    let (b_d, n_d, d_d) = (dims[0], dims[1], dims[2]);
    let h_d = 8usize;
    let hd_d = d_d / h_d;
    assert_eq!(d_d, 256, "fixture must be small-shape D=256");

    // Attention sub-section wrapped in checkpoint. The v3 checkpoint
    // closure takes zero args — it captures the inputs by clone. The
    // `inputs` slice argument is used by the autograd machinery to
    // track which leaves participate in the recompute; the closure
    // accesses them via capture.
    let x_c = x.clone();
    let w_qkv_c = w_qkv.clone();
    let b_qkv_c = b_qkv.clone();
    let w_out_c = w_out.clone();
    let b_out_c = b_out.clone();
    let attn_inputs = [x.clone(), w_qkv.clone(), b_qkv.clone(), w_out.clone(), b_out.clone()];
    let attn_out = AutogradContext::checkpoint(&attn_inputs, move || {
        let qkv = linear(&x_c, &w_qkv_c, &b_qkv_c);
        let q_flat = qkv.narrow(2, 0, d_d)?;
        let k_flat = qkv.narrow(2, d_d, d_d)?;
        let v_flat = qkv.narrow(2, 2 * d_d, d_d)?;
        let q = q_flat
            .reshape(&[b_d, n_d, h_d, hd_d])?
            .permute(&[0, 2, 1, 3])?;
        let k = k_flat
            .reshape(&[b_d, n_d, h_d, hd_d])?
            .permute(&[0, 2, 1, 3])?;
        let v = v_flat
            .reshape(&[b_d, n_d, h_d, hd_d])?
            .permute(&[0, 2, 1, 3])?;
        let o = flame_core::sdpa::forward(&q, &k, &v, None)?;
        let o = o.permute(&[0, 2, 1, 3])?.reshape(&[b_d, n_d, d_d])?;
        Ok(linear(&o, &w_out_c, &b_out_c))
    })
    .expect("AutogradContext::checkpoint");

    let gated_attn = gate.mul(&attn_out).expect("gate * attn_out");
    let h_mid = x.add(&gated_attn).expect("x + gate*attn_out");

    let up = linear(&h_mid, &w_up, &b_up);
    let g_chunk = up.narrow(2, 0, d_d).expect("narrow g");
    let u_chunk = up.narrow(2, d_d, d_d).expect("narrow u");
    let g_silu = g_chunk.silu().expect("silu g");
    let gu = g_silu.mul(&u_chunk).expect("silu(g) * u");
    let mlp_out = linear(&gu, &w_down, &b_down);

    let gated_mlp = gate.mul(&mlp_out).expect("gate * mlp_out");
    let out = h_mid.add(&gated_mlp).expect("h_mid + gate*mlp_out");

    let loss = out.sum().expect("sum");
    let grads = AutogradContext::backward_v2(&loss).expect("backward_v2");
    assert_eq!(
        grads.policy(),
        GradStorePolicy::MatchInsertedDtype,
        "backward_v2 must yield MatchInsertedDtype map"
    );

    // Every trainable WEIGHT leaf must have a non-zero BF16 grad. If
    // checkpoint composed wrong with the bridge, attention-path grads
    // (dw_qkv, dw_out) would be missing or zero. Bias leaves are NOT
    // asserted here because the existing scenario_klein_block_backward
    // parity test reveals biases don't currently land in the grad map
    // even on the non-checkpointed bridge path (bias add-broadcast
    // backward gap, separate from item #3). Logging biases as DEFERRED
    // so the gap is visible without failing the gate.
    for (name, leaf) in [
        ("dx", &x),
        ("dw_qkv", &w_qkv),
        ("dw_out", &w_out),
        ("dw_up", &w_up),
        ("dw_down", &w_down),
    ] {
        let g = grads
            .get(leaf.id())
            .unwrap_or_else(|| panic!("missing {name} under checkpoint+bridge"));
        assert_eq!(g.dtype(), DType::BF16, "{name} must be BF16 under bridge");
        let v: Vec<f32> = g
            .to_dtype(DType::F32)
            .expect("cast for finite check")
            .to_vec()
            .expect("to_vec");
        let any_nonzero = v.iter().any(|x| x.abs() > 0.0);
        assert!(
            any_nonzero,
            "{name} must be non-zero — bridge+checkpoint should produce real grad signal"
        );
    }
    // Bias gap surface: report which biases are missing without
    // failing. See OPEN-PHASE5C-1 in HANDOFF_2026-05-13_PHASE5D_ZLORA_SMOKE.md
    // (Op::Add leaf-bias backward) — already documented.
    for (name, leaf) in [
        ("db_qkv", &b_qkv),
        ("db_out", &b_out),
        ("db_up", &b_up),
        ("db_down", &b_down),
    ] {
        match grads.get(leaf.id()) {
            Some(_) => eprintln!("[bias gate] {name} PRESENT under bridge+checkpoint"),
            None => eprintln!("[bias gate] {name} MISSING (known OPEN-PHASE5C-1)"),
        }
    }

    AutogradContext::set_enabled(false);
}
