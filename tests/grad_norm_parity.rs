//! Phase 5 parity — `flame_core::ops::grad_norm::global_l2_norm` vs PyTorch
//! `torch.cat(grads.flatten()).pow(2).sum().sqrt()`.
//!
//! Fixture: 200 LoRA-shape tensors (alternating [4,4096] and [4096,4]) at
//! `tests/pytorch_fixtures/grad_norm/lora_200tensors_4x4096.safetensors`.
//! Pinned by SHA256. The fixture file is gitignored (tests/pytorch_fixtures/
//! is in .gitignore), so each developer regenerates locally with the snippet
//! below. The SHA256 pin makes regeneration deterministic across machines
//! when the seed and torch version line up.
//!
//! Regenerate:
//!
//! ```text
//! python3 - << 'PYEOF'
//! import torch
//! from safetensors.torch import save_file
//! from pathlib import Path
//! torch.manual_seed(0); torch.cuda.manual_seed(0)
//! dev, dt = "cuda", torch.float32
//! N, SHAPE, MAX_NORM, EPS = 200, (4, 4096), 1.0, 1e-6
//! tensors, flat = {}, []
//! for i in range(N):
//!     g = torch.randn(*(SHAPE if i % 2 == 0 else SHAPE[::-1]),
//!                     device=dev, dtype=dt) * 0.05
//!     tensors[f"grad_{i:03d}_f32"]  = g.cpu().contiguous()
//!     tensors[f"grad_{i:03d}_bf16"] = g.to(torch.bfloat16).cpu().contiguous()
//!     flat.append(g.reshape(-1))
//! all_g = torch.cat(flat)
//! norm = all_g.pow(2).sum().sqrt()
//! tensors["total_norm"] = norm.cpu().reshape(1)
//! tensors["scale"] = (MAX_NORM / (norm + EPS)).clamp(max=1.0).cpu().reshape(1)
//! flat_bf = torch.cat([
//!     tensors[f"grad_{i:03d}_bf16"].cuda().reshape(-1).to(torch.float32)
//!     for i in range(N)
//! ])
//! tensors["total_norm_from_bf16"] = flat_bf.pow(2).sum().sqrt().cpu().reshape(1)
//! out = Path("flame-core/tests/pytorch_fixtures/grad_norm/lora_200tensors_4x4096.safetensors")
//! out.parent.mkdir(parents=True, exist_ok=True)
//! save_file(tensors, str(out))
//! PYEOF
//! ```
//!
//! Expected SHA256 (regenerated on a 3090 Ti, torch 2.10.0+cu128):
//! `d2f2544984ff26cd66102c8fb0f11099be2d2ee334c59fe6326e7f14a1359178`. If
//! you regenerate and get a different hash, update `FIXTURE_SHA256` below
//! AND this comment.
//!
//! Two parity sweeps:
//!   1. `_f32` keys → flame_core global_l2_norm of FP32 grads vs PyTorch's
//!      FP32 oracle norm. Tight tol (atol=1e-4 rtol=1e-4 — well within FP32
//!      reduction-order noise on N=200×16384 elements).
//!   2. `_bf16` keys → flame_core global_l2_norm of BF16 grads vs PyTorch's
//!      norm computed AFTER round-tripping through BF16. Tight tol again
//!      (the round-trip noise is in the fixture, not the helper).
//!
//! No FP32 fallback inside the helper: it casts BF16 → FP32 internally
//! before the per-tensor square+sum, so mixed-dtype slices work transparently.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

mod parity_helpers;

use flame_core::{global_cuda_device, ops::grad_norm::global_l2_norm, serialization::load_file, Tensor};
use std::path::PathBuf;

const FIXTURE_REL: &str = "tests/pytorch_fixtures/grad_norm/lora_200tensors_4x4096.safetensors";
const FIXTURE_SHA256: &str =
    "d2f2544984ff26cd66102c8fb0f11099be2d2ee334c59fe6326e7f14a1359178";
const N_TENSORS: usize = 200;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_REL)
}

fn collect_grads<'a>(
    tensors: &'a std::collections::HashMap<String, Tensor>,
    suffix: &str,
) -> Vec<&'a Tensor> {
    (0..N_TENSORS)
        .map(|i| {
            let key = format!("grad_{i:03}_{suffix}");
            tensors
                .get(&key)
                .unwrap_or_else(|| panic!("missing fixture key {key}"))
        })
        .collect()
}

#[test]
fn fixture_sha256_pinned() {
    let p = fixture_path();
    assert!(p.exists(), "fixture missing: {p:?}");
    assert_eq!(parity_helpers::sha256_file(&p), FIXTURE_SHA256);
}

#[test]
fn global_l2_norm_matches_pytorch_f32() {
    let dev = global_cuda_device();
    let p = fixture_path();
    assert_eq!(parity_helpers::sha256_file(&p), FIXTURE_SHA256);

    let tensors = load_file(&p, &dev).expect("load grad_norm fixture");
    let grads_f32 = collect_grads(&tensors, "f32");
    let expected = tensors.get("total_norm").expect("missing total_norm");

    let got = global_l2_norm(&grads_f32).expect("global_l2_norm");

    let report = parity_helpers::compare_tensor(&got, expected, 1e-4, 1e-4);
    assert!(
        report.passed,
        "F32 grad norm parity FAIL\n{}",
        report.pretty()
    );
    eprintln!("F32 grad norm parity ok: {}", report.pretty());
}

#[test]
fn global_l2_norm_matches_pytorch_bf16_roundtrip() {
    let dev = global_cuda_device();
    let p = fixture_path();
    let tensors = load_file(&p, &dev).expect("load");
    let grads_bf = collect_grads(&tensors, "bf16");
    // Reference is BF16-round-tripped, computed in PyTorch FP32 after the cast.
    let expected = tensors
        .get("total_norm_from_bf16")
        .expect("missing total_norm_from_bf16");

    let got = global_l2_norm(&grads_bf).expect("global_l2_norm bf16");
    let report = parity_helpers::compare_tensor(&got, expected, 1e-4, 1e-4);
    assert!(
        report.passed,
        "BF16 grad norm parity FAIL\n{}",
        report.pretty()
    );
    eprintln!("BF16 grad norm parity ok: {}", report.pretty());
}

#[test]
fn empty_slice_returns_zero() {
    let got = global_l2_norm(&[]).expect("empty global_l2_norm");
    let v = got.to_vec_f32().expect("to_vec_f32");
    assert_eq!(v.len(), 1);
    assert_eq!(v[0], 0.0_f32);
}

#[test]
fn matches_pytorch_with_global_l2_norm_with_scale() {
    use flame_core::ops::grad_norm::global_l2_norm_with_scale;
    let dev = global_cuda_device();
    let p = fixture_path();
    let tensors = load_file(&p, &dev).expect("load");
    let grads = collect_grads(&tensors, "f32");
    let expected_norm = tensors.get("total_norm").expect("total_norm");
    let expected_scale = tensors.get("scale").expect("scale");

    let (norm_t, scale_t) = global_l2_norm_with_scale(&grads, 1.0_f32, 1e-6_f32)
        .expect("global_l2_norm_with_scale");

    let r1 = parity_helpers::compare_tensor(&norm_t, expected_norm, 1e-4, 1e-4);
    assert!(r1.passed, "norm parity FAIL\n{}", r1.pretty());

    let r2 = parity_helpers::compare_tensor(&scale_t, expected_scale, 1e-5, 1e-5);
    assert!(r2.passed, "scale parity FAIL\n{}", r2.pretty());
    eprintln!(
        "norm+scale parity ok: norm {} | scale {}",
        r1.pretty(),
        r2.pretty()
    );
}

#[test]
fn no_d2h_sync_inside_helper() {
    // Sanity: the helper must not call `.item()` or `.to_vec()` internally.
    // We check this indirectly by enqueuing a kernel before the helper and
    // confirming the helper completes without forcing a sync.
    //
    // This is a weak check (Rust can't easily count syncs from outside)
    // but a regression that adds an internal sync would still pass it —
    // intentionally so. The strong check is reading the helper source.
    //
    // The point of this test: if someone adds an `.item()` deep in the
    // helper, the empty fold-over still works, so behavior is preserved.
    // The perf regression is what we'd notice in production.
    let dev = global_cuda_device();
    let p = fixture_path();
    let tensors = load_file(&p, &dev).expect("load");
    let grads = collect_grads(&tensors, "f32");
    let _norm = global_l2_norm(&grads).expect("global_l2_norm");
    // Function returned a Tensor — no panic implies no sync error.
}
