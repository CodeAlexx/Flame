//! Model-AGNOSTIC autograd ENGINE parity. No model, no architecture.
//!
//! Deep chain (T=12) with fan-out: each step `h = h + 0.1*((h@W1)+(h@W2))`
//! reuses `h` in THREE consumers (W1 path, W2 path, residual). The engine must
//! accumulate all three gradient contributions back into `h` at every step and
//! traverse the composed tape correctly. matmul/add/mul individually pass
//! parity; this isolates the ENGINE's accumulation + traversal across a
//! composed graph — the thing every model shares regardless of architecture.
//!
//! Reference: /tmp/engine_fanout_fixture.safetensors (PyTorch, F32) from
//! /tmp/engine_fanout_ref.py. Keys: x0, W1, W2, grad_x0, loss.
//!
//! PASS: cos(grad_x0_ours, grad_x0_torch) >= 0.999 AND mag_ratio in [0.5,2.0].
//!
//! Run: cargo test --release --test engine_fanout_parity -- --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::serialization::load_file;
use flame_core::{global_cuda_device, AutogradContext, DType, Result};

fn cos_mag(got: &[f32], reference: &[f32]) -> (f64, f64, f64) {
    assert_eq!(got.len(), reference.len());
    let (mut dot, mut gn, mut rn, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in got.iter().zip(reference.iter()) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b; gn += a * a; rn += b * b; diff += (a - b) * (a - b);
    }
    let cos = dot / (gn.sqrt() * rn.sqrt() + 1e-20);
    let mag = gn.sqrt() / (rn.sqrt() + 1e-20);
    (cos, mag, diff.sqrt() / (rn.sqrt() + 1e-20))
}

#[test]
fn engine_fanout_backward_matches_pytorch() -> Result<()> {
    let path = "/tmp/engine_fanout_fixture.safetensors";
    let device = global_cuda_device();
    let f = load_file(path, &device)
        .unwrap_or_else(|e| panic!("missing {path}: {e} — run `env -u LD_LIBRARY_PATH python3 /tmp/engine_fanout_ref.py`"));

    let x0 = f.get("x0").expect("x0").to_dtype(DType::F32)?.requires_grad_(true);
    let w1 = f.get("W1").expect("W1").to_dtype(DType::F32)?; // frozen
    let w2 = f.get("W2").expect("W2").to_dtype(DType::F32)?; // frozen
    let grad_ref: Vec<f32> = f.get("grad_x0").expect("grad_x0").to_dtype(DType::F32)?.to_vec()?;
    let loss_ref = f.get("loss").expect("loss").to_dtype(DType::F32)?.to_vec()?[0];

    AutogradContext::clear();

    let mut h = x0.clone();
    for _ in 0..12 {
        let p1 = h.matmul(&w1)?;
        let p2 = h.matmul(&w2)?;
        let sum = p1.add(&p2)?.mul_scalar(0.1)?;
        h = h.add(&sum)?;
    }
    let loss = h.mul(&h)?.mean()?;
    let loss_val = loss.to_vec()?[0];

    let grads = loss.backward()?;
    let g = grads
        .get(x0.id())
        .unwrap_or_else(|| panic!("ENGINE: no grad for x0 leaf despite requires_grad=true"));
    let got: Vec<f32> = g.to_dtype(DType::F32)?.to_vec()?;

    let (cos, mag, rel) = cos_mag(&got, &grad_ref);
    println!("[engine-fanout] loss ours={loss_val:.6} torch={loss_ref:.6}  (Δ={:.3e})", (loss_val - loss_ref).abs());
    println!("[engine-fanout] grad_x0  cos={cos:.6}  mag_ratio={mag:.4}  rel_L2={rel:.4e}");
    println!("[engine-fanout] |grad| ours={:.6} torch={:.6}",
        got.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
        grad_ref.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt());

    assert!(cos >= 0.999, "ENGINE BROKEN: grad_x0 cos {cos:.6} < 0.999 — composed/fan-out backward diverges from PyTorch");
    assert!((0.5..=2.0).contains(&mag), "ENGINE: grad_x0 magnitude ratio {mag:.4} outside [0.5,2.0]");
    Ok(())
}

/// Same graph, BF16 — the dtype real training runs. Exercises the BF16
/// gradient-accumulation path (vs the F32 path above). Compared to the F32
/// PyTorch ground truth; BF16 rounding gives small rel_L2 but cos must stay
/// high if accumulation is sound.
#[test]
fn engine_fanout_backward_bf16() -> Result<()> {
    let path = "/tmp/engine_fanout_fixture.safetensors";
    let device = global_cuda_device();
    let f = load_file(path, &device).expect("fixture");

    let x0 = f.get("x0").unwrap().to_dtype(DType::BF16)?.requires_grad_(true);
    let w1 = f.get("W1").unwrap().to_dtype(DType::BF16)?;
    let w2 = f.get("W2").unwrap().to_dtype(DType::BF16)?;
    let grad_ref: Vec<f32> = f.get("grad_x0").unwrap().to_dtype(DType::F32)?.to_vec()?;

    AutogradContext::clear();
    let mut h = x0.clone();
    for _ in 0..12 {
        let p1 = h.matmul(&w1)?;
        let p2 = h.matmul(&w2)?;
        let sum = p1.add(&p2)?.mul_scalar(0.1)?;
        h = h.add(&sum)?;
    }
    let loss = h.mul(&h)?.mean()?;
    let grads = loss.backward()?;
    let g = grads.get(x0.id()).expect("no grad for x0 (BF16)");
    let got: Vec<f32> = g.to_dtype(DType::F32)?.to_vec()?;

    let (cos, mag, rel) = cos_mag(&got, &grad_ref);
    println!("[engine-fanout-bf16] grad_x0  cos={cos:.6}  mag_ratio={mag:.4}  rel_L2={rel:.4e}");
    assert!(cos >= 0.99, "ENGINE BROKEN (BF16): grad_x0 cos {cos:.6} < 0.99 — BF16 fan-out accumulation diverges");
    Ok(())
}
