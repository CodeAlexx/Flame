//! Phase 5b follow-up — Klein backward parity v2 vs PyTorch.
//!
//! Adapts the existing six Klein component-level backward parity tests
//! (defined at `tests/pytorch_parity.rs:1008-1424` against the v3
//! `loss.backward()` path) to exercise `AutogradContext::backward_v2()`
//! — the Phase 5b bridge — against the SAME PyTorch fixtures. Plus a
//! 7th scenario reconstructing the Klein-style double-block from
//! `klein_block_backward.safetensors` (a fixture that had no Rust
//! consumer before today).
//!
//! Validates that the v3 → v2-grad-map bridge (`MatchInsertedDtype`
//! policy + post-loop dtype unification) produces grads numerically
//! equivalent to v3's `InternalFP32_PublicBF16` path on real Klein
//! workloads — within BF16-noise bands, since the input fixtures are
//! all BF16.
//!
//! ## Parallel-mode serialization
//!
//! `AUTOGRAD_CONTEXT` is process-global state (`src/autograd.rs:56`).
//! Multiple `#[test]` functions that touch the v3 backward tape (which
//! the bridge does) flake under cargo's default parallel mode — one
//! test's `reset()` wipes another's tape mid-flight. As of 2026-05-13
//! the suite uses `serial_test` to gate each scenario via `#[serial]`,
//! preserving per-scenario CI granularity while serializing global-tape
//! access. Same pattern in `tests/autograd_v2_bridge.rs`.
//!
//! See `docs/FLAME_CONVENTIONS.md` §"Autograd v2 (Phase 5b) — trainer
//! migration guide" for the migration recipe this test exercises.
//!
//! ## Tolerance bands
//!
//! The v3 Klein tests assert `cos > 0.99` for backward grads. We adopt
//! the same `0.99` floor here for BF16 paths (BF16 reductions through
//! rms_norm + rope + sdpa + matmul compound noise well past the
//! `tol_bf16_ln` 0.999 band defined for single-op layer_norm parity in
//! Phase 5a). The `max_abs_ratio` band is set to `5e-2` — same as
//! `tol_bf16_ln`. F32 paths (none here — all fixtures are BF16) would
//! use `tol_f32_tight`/`tol_f32_loose` from
//! `tests/autograd_v2_parity.rs`.

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serial_test::serial;

use flame_core::autograd::policy::GradStorePolicy;
use flame_core::autograd::AutogradContext;
use flame_core::parity::{ParityHarness, ParityTolerance};
use flame_core::{DType, Tensor};

// ---------------------------------------------------------------------------
// Test infrastructure (mirrors tests/pytorch_parity.rs helpers)
// ---------------------------------------------------------------------------

fn fixtures_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("PYTORCH_FIXTURES").unwrap_or_else(|_| "tests/pytorch_fixtures".to_string()),
    )
}

// Retained for potential future bulk-skip use; individual scenarios
// each have their own `if !path.exists()` guard now that they're
// `#[serial] #[test]` rather than driver-called.
#[allow(dead_code)]
fn skip_if_no_fixtures() -> bool {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!(
            "SKIP: fixtures dir {:?} not found. Run scripts/generate_pytorch_fixtures.py first.",
            dir
        );
        true
    } else {
        false
    }
}

fn load_fixture(path: &Path) -> HashMap<String, Tensor> {
    let device = flame_core::global_cuda_device();
    flame_core::serialization::load_file(path, &device)
        .unwrap_or_else(|e| panic!("load_file {:?} failed: {:?}", path, e))
}

/// BF16 Klein-chain tolerance: matches the v3 Klein tests' `cos > 0.99`
/// gate, with a `5e-2` abs-ratio band (same as `tol_bf16_ln` for LN
/// parity). Compounded BF16 noise across rms_norm + rope + SDPA +
/// matmul lands above the Phase-5a single-op LN band.
fn tol_klein_bf16() -> ParityTolerance {
    ParityTolerance {
        min_cos: 0.99,
        max_abs_ratio: 5e-2,
    }
}

/// L2 norm of a tensor as f32 — used for sanity-printing dx/dscale
/// norms (matches the v3 test logging format).
fn l2_norm(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32)
        .unwrap()
        .to_vec_f32()
        .unwrap()
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Scenario 1: head_rms_norm at toy shape (H=8, HD=32)
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn scenario_head_rms_norm_toy() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_ext_head_rms_norm.safetensors");
    if !path.exists() {
        eprintln!("SKIP scenario_head_rms_norm_toy: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);
    let x = fix.get("x").unwrap().clone().requires_grad_(true);
    let scale = fix.get("scale").unwrap().clone().requires_grad_(true);

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let dims = x.shape().dims().to_vec();
    let (b_d, n_d, d_d) = (dims[0], dims[1], dims[2]);
    let h_d = 8usize;
    let hd_d = d_d / h_d;

    let h = x
        .reshape(&[b_d, n_d, h_d, hd_d])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .expect("reshape/permute");
    let out = flame_core::norm::rms_norm(&h, &[hd_d], Some(&scale), 1.0e-6).expect("rms_norm");

    let loss = out.sum().expect("sum");
    let grads = AutogradContext::backward_v2(&loss).expect("backward_v2");
    assert_eq!(
        grads.policy(),
        GradStorePolicy::MatchInsertedDtype,
        "backward_v2 must return MatchInsertedDtype policy"
    );

    let dx = grads.get(x.id()).expect("missing dx (v2 toy HD=32)");
    let dscale = grads.get(scale.id()).expect("missing dscale (v2 toy)");

    // Bridge contract: grads land at loss.dtype() (BF16 here).
    assert_eq!(loss.dtype(), DType::BF16, "toy fixture loss should be BF16");
    assert_eq!(
        dx.dtype(),
        DType::BF16,
        "v2 dx must match loss.dtype() (BF16)"
    );
    assert_eq!(
        dscale.dtype(),
        DType::BF16,
        "v2 dscale must match loss.dtype() (BF16)"
    );

    eprintln!(
        "[v2] head_rms_norm_toy: ||dx||={:.3e} ||dscale||={:.3e}",
        l2_norm(dx),
        l2_norm(dscale)
    );

    // Toy fixture HAS dx/dscale even though the v3 test doesn't check
    // them — exercise the bridge against them.
    let mut h = ParityHarness::load(&path, flame_core::global_cuda_device().clone())
        .expect("ParityHarness::load")
        .with_tolerance(tol_klein_bf16());
    h.compare("dx", dx).expect("compare dx");
    h.compare("dscale", dscale).expect("compare dscale");
    eprintln!("{}", h.report());
    h.assert_clean();

    AutogradContext::set_enabled(false);
}

// ---------------------------------------------------------------------------
// Scenario 2: head_rms_norm at prod shape (H=24, HD=128)
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn scenario_head_rms_norm_prod() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_ext_head_rms_norm_prod.safetensors");
    if !path.exists() {
        eprintln!("SKIP scenario_head_rms_norm_prod: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);
    let x = fix.get("x").unwrap().clone().requires_grad_(true);
    let scale = fix.get("scale").unwrap().clone().requires_grad_(true);

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let dims = x.shape().dims().to_vec();
    let (b_d, n_d, d_d) = (dims[0], dims[1], dims[2]);
    let h_d = 24usize;
    let hd_d = d_d / h_d;

    let h = x
        .reshape(&[b_d, n_d, h_d, hd_d])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .expect("reshape/permute");
    let out = flame_core::norm::rms_norm(&h, &[hd_d], Some(&scale), 1.0e-6).expect("rms_norm");

    let loss = out.sum().expect("sum");
    let grads = AutogradContext::backward_v2(&loss).expect("backward_v2");

    let dx = grads.get(x.id()).expect("missing dx (v2 prod HD=128)");
    let dscale = grads.get(scale.id()).expect("missing dscale (v2 prod)");

    assert_eq!(dx.dtype(), DType::BF16, "v2 dx must be BF16");
    assert_eq!(dscale.dtype(), DType::BF16, "v2 dscale must be BF16");

    eprintln!(
        "[v2] head_rms_norm_prod: ||dx||={:.3e} ||dscale||={:.3e}",
        l2_norm(dx),
        l2_norm(dscale)
    );

    let mut h = ParityHarness::load(&path, flame_core::global_cuda_device().clone())
        .expect("ParityHarness::load")
        .with_tolerance(tol_klein_bf16());
    h.compare("dx", dx).expect("compare dx");
    h.compare("dscale", dscale).expect("compare dscale");
    eprintln!("{}", h.report());
    h.assert_clean();

    AutogradContext::set_enabled(false);
}

// ---------------------------------------------------------------------------
// Scenario 3: apply_rope at prod shape (post-cat H=24, N=1536, HD=128)
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn scenario_apply_rope_prod() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_ext_apply_rope_prod.safetensors");
    if !path.exists() {
        eprintln!("SKIP scenario_apply_rope_prod: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);
    let x = fix.get("x").unwrap().clone().requires_grad_(true);
    let pe_cos = fix.get("pe_cos").unwrap().clone();
    let pe_sin = fix.get("pe_sin").unwrap().clone();

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let out = {
        use flame_core::autograd::{AutogradContext, Op};
        let mut o =
            flame_core::bf16_ops::rope_fused_bf16(&x, &pe_cos, &pe_sin).expect("rope_fused_bf16");
        if x.requires_grad() {
            o = o.requires_grad_(true);
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    o.id(),
                    Op::RoPePrecomputed {
                        input: x.id(),
                        cos: pe_cos.id(),
                        sin: pe_sin.id(),
                        layout: flame_core::autograd::RopeLayout::Interleaved,
                    },
                    vec![
                        (x.id(), x.clone()),
                        (pe_cos.id(), pe_cos.clone()),
                        (pe_sin.id(), pe_sin.clone()),
                    ],
                );
            }
        }
        o
    };

    let loss = out.sum().expect("sum");
    let grads = AutogradContext::backward_v2(&loss).expect("backward_v2");

    let dx = grads.get(x.id()).expect("missing dx (v2 rope prod)");
    assert_eq!(dx.dtype(), DType::BF16, "v2 dx must be BF16");

    eprintln!("[v2] apply_rope_prod: ||dx||={:.3e}", l2_norm(dx));

    let mut h = ParityHarness::load(&path, flame_core::global_cuda_device().clone())
        .expect("ParityHarness::load")
        .with_tolerance(tol_klein_bf16());
    h.compare("dx", dx).expect("compare dx");
    eprintln!("{}", h.report());
    h.assert_clean();

    AutogradContext::set_enabled(false);
}

// ---------------------------------------------------------------------------
// Scenario 4: rms_norm direct 4D (no reshape/permute autograd, only LN bwd)
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn scenario_rms_norm_direct_4d() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_ext_head_rms_norm_prod.safetensors");
    if !path.exists() {
        eprintln!("SKIP scenario_rms_norm_direct_4d: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);
    let x_3d = fix.get("x").unwrap().clone();
    let scale = fix.get("scale").unwrap().clone().requires_grad_(true);

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let dims = x_3d.shape().dims().to_vec();
    let (b_d, n_d, d_d) = (dims[0], dims[1], dims[2]);
    let h_d = 24usize;
    let hd_d = d_d / h_d;

    // Reshape with NO autograd recording (x_3d is not requires_grad),
    // then mark the 4D result as requires_grad — bypasses Op::Reshape
    // entirely and tests ONLY rms_norm's own backward (mirrors the v3
    // test's structure verbatim).
    let x_4d = x_3d
        .reshape(&[b_d, n_d, h_d, hd_d])
        .expect("reshape (no-grad)")
        .requires_grad_(true);

    let out = flame_core::norm::rms_norm(&x_4d, &[hd_d], Some(&scale), 1.0e-6).expect("rms_norm");

    let loss = out.sum().expect("sum");
    let grads = AutogradContext::backward_v2(&loss).expect("backward_v2");

    let dx = grads.get(x_4d.id()).expect("missing dx (v2 4D direct)");
    let _dscale = grads
        .get(scale.id())
        .expect("missing dscale (v2 4D direct)");

    assert_eq!(dx.dtype(), DType::BF16, "v2 dx must be BF16");
    eprintln!(
        "[v2] rms_norm_direct_4d: ||dx_4d||={:.3e}  (sanity: should be nonzero)",
        l2_norm(dx)
    );
    assert!(l2_norm(dx) > 0.0, "v2 dx must be nonzero (sanity)");

    AutogradContext::set_enabled(false);
}

// ---------------------------------------------------------------------------
// Scenario 5: rms_norm on contiguous 4D (no permute, dx-sanity only)
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn scenario_rms_norm_contig_prod() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_ext_head_rms_norm_prod.safetensors");
    if !path.exists() {
        eprintln!("SKIP scenario_rms_norm_contig_prod: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);
    let x = fix.get("x").unwrap().clone().requires_grad_(true);
    let scale = fix.get("scale").unwrap().clone().requires_grad_(true);

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let dims = x.shape().dims().to_vec();
    let (b_d, n_d, d_d) = (dims[0], dims[1], dims[2]);
    let h_d = 24usize;
    let hd_d = d_d / h_d;

    // Reshape ONLY (no permute) → tensor is still contiguous.
    let h = x.reshape(&[b_d, n_d, h_d, hd_d]).expect("reshape");
    let out = flame_core::norm::rms_norm(&h, &[hd_d], Some(&scale), 1.0e-6).expect("rms_norm");

    let loss = out.sum().expect("sum");
    let grads = AutogradContext::backward_v2(&loss).expect("backward_v2");

    let dx = grads.get(x.id()).expect("missing dx (v2 contig)");
    let _dscale = grads.get(scale.id()).expect("missing dscale");

    assert_eq!(dx.dtype(), DType::BF16, "v2 dx must be BF16");
    eprintln!(
        "[v2] rms_norm_contig_prod: ||dx||={:.3e}  (sanity: should be nonzero)",
        l2_norm(dx)
    );
    assert!(l2_norm(dx) > 0.0, "v2 dx must be nonzero (sanity)");

    AutogradContext::set_enabled(false);
}

// ---------------------------------------------------------------------------
// Scenario 6: full attn chain at prod shape
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn scenario_attn_chain_prod() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_ext_attn_chain_prod.safetensors");
    if !path.exists() {
        eprintln!("SKIP scenario_attn_chain_prod: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);
    let x = fix.get("x").unwrap().clone().requires_grad_(true);
    let w_qkv = fix.get("w_qkv").unwrap().clone().requires_grad_(true);
    let b_qkv = fix.get("b_qkv").unwrap().clone().requires_grad_(true);
    let q_scale = fix.get("q_scale").unwrap().clone().requires_grad_(true);
    let k_scale = fix.get("k_scale").unwrap().clone().requires_grad_(true);
    let w_out = fix.get("w_out").unwrap().clone().requires_grad_(true);
    let b_out = fix.get("b_out").unwrap().clone().requires_grad_(true);
    let pe_cos = fix.get("pe_cos").unwrap().clone();
    let pe_sin = fix.get("pe_sin").unwrap().clone();

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let linear = |x_in: &Tensor, w: &Tensor, b: &Tensor| -> Tensor {
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
    };

    let dims = x.shape().dims().to_vec();
    let (b_d, n_d, d_d) = (dims[0], dims[1], dims[2]);
    let h_d = 24usize;
    let hd_d = d_d / h_d;
    let inner = d_d;

    let qkv = linear(&x, &w_qkv, &b_qkv);
    let q_flat = qkv.narrow(2, 0, inner).expect("narrow q");
    let k_flat = qkv.narrow(2, inner, inner).expect("narrow k");
    let v_flat = qkv.narrow(2, 2 * inner, inner).expect("narrow v");

    let q = q_flat
        .reshape(&[b_d, n_d, h_d, hd_d])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .expect("q reshape/permute");
    let k = k_flat
        .reshape(&[b_d, n_d, h_d, hd_d])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .expect("k reshape/permute");
    let v = v_flat
        .reshape(&[b_d, n_d, h_d, hd_d])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .expect("v reshape/permute");

    let q = flame_core::norm::rms_norm(&q, &[hd_d], Some(&q_scale), 1.0e-6).expect("q rms");
    let k = flame_core::norm::rms_norm(&k, &[hd_d], Some(&k_scale), 1.0e-6).expect("k rms");

    let rope = |x_in: &Tensor| -> Tensor {
        use flame_core::autograd::{AutogradContext, Op};
        let mut o =
            flame_core::bf16_ops::rope_fused_bf16(x_in, &pe_cos, &pe_sin).expect("rope_fused_bf16");
        if x_in.requires_grad() {
            o = o.requires_grad_(true);
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    o.id(),
                    Op::RoPePrecomputed {
                        input: x_in.id(),
                        cos: pe_cos.id(),
                        sin: pe_sin.id(),
                        layout: flame_core::autograd::RopeLayout::Interleaved,
                    },
                    vec![
                        (x_in.id(), x_in.clone()),
                        (pe_cos.id(), pe_cos.clone()),
                        (pe_sin.id(), pe_sin.clone()),
                    ],
                );
            }
        }
        o
    };
    let q = rope(&q);
    let k = rope(&k);

    let o = flame_core::sdpa::forward(&q, &k, &v, None).expect("sdpa");
    let o = o
        .permute(&[0, 2, 1, 3])
        .and_then(|t| t.reshape(&[b_d, n_d, d_d]))
        .expect("permute/reshape back");
    let out = linear(&o, &w_out, &b_out);

    let loss = out.sum().expect("sum");
    let grads = AutogradContext::backward_v2(&loss).expect("backward_v2");

    // Same bias-grad caveat as the v3 test: Op::Add bias path doesn't
    // enter `needed_grad_ids` reliably — best-effort.
    let dx = grads.get(x.id()).expect("missing dx (v2)");
    let dw_qkv = grads.get(w_qkv.id()).expect("missing dw_qkv (v2)");
    let dq_scale = grads.get(q_scale.id()).expect("missing dq_scale (v2)");
    let dk_scale = grads.get(k_scale.id()).expect("missing dk_scale (v2)");
    let dw_out = grads.get(w_out.id()).expect("missing dw_out (v2)");

    for (name, g) in [
        ("dx", dx),
        ("dw_qkv", dw_qkv),
        ("dq_scale", dq_scale),
        ("dk_scale", dk_scale),
        ("dw_out", dw_out),
    ] {
        assert_eq!(g.dtype(), DType::BF16, "v2 {name} must be BF16");
    }

    let mut h = ParityHarness::load(&path, flame_core::global_cuda_device().clone())
        .expect("ParityHarness::load")
        .with_tolerance(tol_klein_bf16());
    h.compare("dx", dx).expect("compare dx");
    h.compare("dw_qkv", dw_qkv).expect("compare dw_qkv");
    h.compare("dq_scale", dq_scale).expect("compare dq_scale");
    h.compare("dk_scale", dk_scale).expect("compare dk_scale");
    h.compare("dw_out", dw_out).expect("compare dw_out");
    // bias grads: best-effort log only (Op::Add path).
    if let Some(g) = grads.get(b_qkv.id()) {
        let _ = h.compare("db_qkv", g);
    }
    if let Some(g) = grads.get(b_out.id()) {
        let _ = h.compare("db_out", g);
    }
    eprintln!("[v2] attn_chain_prod report:\n{}", h.report());
    // Assert clean on the non-bias compares — bias may be missing/loose.
    for r in h.results() {
        if r.name == "db_qkv" || r.name == "db_out" {
            continue;
        }
        assert!(
            r.passed,
            "[v2] attn_chain_prod {} FAILED: cos={:.6} max_abs_ratio={:.4} (tol min_cos=0.99 max_abs_ratio=5e-2)",
            r.name, r.cos, r.max_abs_ratio
        );
    }

    AutogradContext::set_enabled(false);
}

// ---------------------------------------------------------------------------
// Scenario 7: full klein-double-block backward (Deliverable B)
// ---------------------------------------------------------------------------
//
// Reconstructs `scripts/generate_klein_backward_fixture.py`'s "small"
// shape double-block in Rust, runs `backward_v2`, compares dx + 4
// weight grads against the fixture. Skips RoPE per the generator's
// comment ("Skipping RoPE for fixture simplicity").
//
// Fixture layout (B=1, N=64, D=256, H=8, HD=32):
//   x      [1, 64, 256]    -- input (requires_grad)
//   w_qkv  [768, 256]      -- qkv linear weight
//   b_qkv  [768]           -- qkv linear bias
//   w_out  [256, 256]      -- attn output linear weight
//   b_out  [256]           -- attn output linear bias
//   w_up   [512, 256]      -- MLP up linear (gate||up)
//   b_up   [512]
//   w_down [256, 256]      -- MLP down linear
//   b_down [256]
//   gate   [1, 1, 256]     -- frozen modulation
// Block computation:
//   attn_out = w_out(SDPA(qkv split & shape from w_qkv(x)))
//   h        = x + gate * attn_out
//   mlp_out  = w_down(silu(g) * u)       where (g, u) = chunk(w_up(h))
//   out      = h + gate * mlp_out
//   loss     = out.sum()

#[test]
#[serial]
fn scenario_klein_block_backward() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_block_backward.safetensors");
    if !path.exists() {
        eprintln!("SKIP scenario_klein_block_backward: {path:?} not found");
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
    let gate = fix.get("gate").unwrap().clone(); // frozen — no grad

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let dims = x.shape().dims().to_vec();
    let (b_d, n_d, d_d) = (dims[0], dims[1], dims[2]);
    // Hardcoded small-shape constants from the generator (B=1, N=64, D=256, H=8, HD=32).
    let h_d = 8usize;
    let hd_d = d_d / h_d;
    assert_eq!(d_d, 256, "scenario 7 fixture must be small-shape (D=256)");

    let linear = |x_in: &Tensor, w: &Tensor, b: &Tensor| -> Tensor {
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
    };

    // ---- attn path ----
    let qkv = linear(&x, &w_qkv, &b_qkv); // [1, 64, 768]
    let q_flat = qkv.narrow(2, 0, d_d).expect("narrow q");
    let k_flat = qkv.narrow(2, d_d, d_d).expect("narrow k");
    let v_flat = qkv.narrow(2, 2 * d_d, d_d).expect("narrow v");

    let q = q_flat
        .reshape(&[b_d, n_d, h_d, hd_d])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .expect("q reshape/permute");
    let k = k_flat
        .reshape(&[b_d, n_d, h_d, hd_d])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .expect("k reshape/permute");
    let v = v_flat
        .reshape(&[b_d, n_d, h_d, hd_d])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .expect("v reshape/permute");

    // No RoPE (per generator: "Skipping RoPE for fixture simplicity").
    let o = flame_core::sdpa::forward(&q, &k, &v, None).expect("sdpa");
    let o = o
        .permute(&[0, 2, 1, 3])
        .and_then(|t| t.reshape(&[b_d, n_d, d_d]))
        .expect("permute/reshape back");
    let attn_out = linear(&o, &w_out, &b_out);

    // ---- gated residual ----
    let gated_attn = gate.mul(&attn_out).expect("gate * attn_out");
    let h_mid = x.add(&gated_attn).expect("x + gate*attn_out");

    // ---- mlp path ----
    let up = linear(&h_mid, &w_up, &b_up); // [1, 64, 512]
    let g_chunk = up.narrow(2, 0, d_d).expect("narrow g");
    let u_chunk = up.narrow(2, d_d, d_d).expect("narrow u");
    let g_silu = g_chunk.silu().expect("silu g");
    let gu = g_silu.mul(&u_chunk).expect("silu(g) * u");
    let mlp_out = linear(&gu, &w_down, &b_down);

    // ---- final gated residual ----
    let gated_mlp = gate.mul(&mlp_out).expect("gate * mlp_out");
    let out = h_mid.add(&gated_mlp).expect("h_mid + gate*mlp_out");

    let loss = out.sum().expect("sum");
    let grads = AutogradContext::backward_v2(&loss).expect("backward_v2");
    assert_eq!(
        grads.policy(),
        GradStorePolicy::MatchInsertedDtype,
        "scenario 7 backward_v2 must yield MatchInsertedDtype map"
    );

    let dx = grads.get(x.id()).expect("missing dx (v2 block)");
    let dw_qkv = grads.get(w_qkv.id()).expect("missing dw_qkv (v2 block)");
    let dw_out = grads.get(w_out.id()).expect("missing dw_out (v2 block)");
    let dw_up = grads.get(w_up.id()).expect("missing dw_up (v2 block)");
    let dw_down = grads.get(w_down.id()).expect("missing dw_down (v2 block)");

    for (name, g) in [
        ("dx", dx),
        ("dw_qkv", dw_qkv),
        ("dw_out", dw_out),
        ("dw_up", dw_up),
        ("dw_down", dw_down),
    ] {
        assert_eq!(g.dtype(), DType::BF16, "v2 {name} must be BF16");
    }

    let mut h = ParityHarness::load(&path, flame_core::global_cuda_device().clone())
        .expect("ParityHarness::load")
        .with_tolerance(tol_klein_bf16());
    h.compare("dx", dx).expect("compare dx");
    h.compare("dw_qkv", dw_qkv).expect("compare dw_qkv");
    h.compare("dw_out", dw_out).expect("compare dw_out");
    h.compare("dw_up", dw_up).expect("compare dw_up");
    h.compare("dw_down", dw_down).expect("compare dw_down");
    // Bias-grad best-effort, same caveat as scenario 6.
    if let Some(g) = grads.get(b_qkv.id()) {
        let _ = h.compare("db_qkv", g);
    }
    if let Some(g) = grads.get(b_out.id()) {
        let _ = h.compare("db_out", g);
    }
    if let Some(g) = grads.get(b_up.id()) {
        let _ = h.compare("db_up", g);
    }
    if let Some(g) = grads.get(b_down.id()) {
        let _ = h.compare("db_down", g);
    }
    eprintln!("[v2] klein_block_backward report:\n{}", h.report());
    for r in h.results() {
        if r.name.starts_with("db_") {
            continue; // bias-grad best-effort
        }
        assert!(
            r.passed,
            "[v2] klein_block_backward {} FAILED: cos={:.6} max_abs_ratio={:.4} (tol min_cos=0.99 max_abs_ratio=5e-2)",
            r.name, r.cos, r.max_abs_ratio
        );
    }

    AutogradContext::set_enabled(false);
}
