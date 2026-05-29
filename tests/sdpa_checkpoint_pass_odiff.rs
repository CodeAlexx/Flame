//! Does the checkpoint's INITIAL forward (recording off → inference SDPA,
//! `forward_cudnn_sdpa_bf16`) produce a different O than the RECOMPUTE
//! (recording on → train SDPA, `forward_cudnn_sdpa_train_bf16`) on identical
//! Q/K/V? If O diverges, the loss (built from the initial fwd) and the gradient
//! (built from the recompute) rest on different activations — a candidate for
//! the multi-step training runaway. If O is identical, that lead is dead.
//!
//! This routes through the REAL `attention::sdpa` dispatch, which branches on
//! autograd recording state — exactly what differs between the two checkpoint
//! passes. klein 9B head config: H=32, D=128, plain MHA (not GQA).

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn seeded_bf16(b: usize, h: usize, n: usize, d: usize, seed: u64) -> Tensor {
    let dev = global_cuda_device();
    let scale = 1.0f32 / (d as f32).sqrt();
    Tensor::randn_seeded(Shape::from_dims(&[b, h, n, d]), 0.0, scale, seed, dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn compare(tag: &str, a: &Tensor, b: &Tensor) {
    let af = a.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let bf = b.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_eq!(af.len(), bf.len());
    let (mut dot, mut na, mut nb, mut max_abs, mut sum_d, mut sum_r) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut exact = 0usize;
    for (x, y) in af.iter().zip(bf.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        let dd = (x - y).abs();
        if dd == 0.0 {
            exact += 1;
        }
        if dd > max_abs {
            max_abs = dd;
        }
        sum_d += dd;
        sum_r += x.abs();
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-20);
    let mean_rel = sum_d / sum_r.max(1e-20);
    let pct_exact = 100.0 * exact as f64 / af.len() as f64;
    println!(
        "[{tag}] cos={:.8}  mean_rel={:.4e}  max_abs={:.4e}  bit_identical={:.1}%  n={}",
        cos, mean_rel, max_abs, pct_exact, af.len()
    );
}

#[test]
fn sdpa_initial_vs_recompute_o_diff() {
    let (b, h, n, d) = (1usize, 32usize, 256usize, 128usize);
    let q = seeded_bf16(b, h, n, d, 1);
    let k = seeded_bf16(b, h, n, d, 2);
    let v = seeded_bf16(b, h, n, d, 3);

    // Path A — checkpoint INITIAL forward: recording OFF → inference SDPA.
    let o_inf = {
        let _g = flame_core::autograd::AutogradContext::no_grad();
        flame_core::attention::sdpa(&q, &k, &v, None).unwrap()
    };

    // Path B — checkpoint RECOMPUTE: recording ON + an input requires grad → train SDPA.
    flame_core::autograd::AutogradContext::set_enabled(true);
    let qg = seeded_bf16(b, h, n, d, 1).requires_grad_(true); // identical values to q
    let o_train = flame_core::attention::sdpa(&qg, &k, &v, None).unwrap();

    global_cuda_device().synchronize().unwrap();

    // Reference: train vs train (same path, same inputs) — establishes the
    // noise floor (should be ~bit-identical / cos 1.0).
    flame_core::autograd::AutogradContext::set_enabled(true);
    let qg2 = seeded_bf16(b, h, n, d, 1).requires_grad_(true);
    let o_train2 = flame_core::attention::sdpa(&qg2, &k, &v, None).unwrap();
    global_cuda_device().synchronize().unwrap();

    println!("=== SDPA checkpoint-pass O diff (klein H=32 D=128) ===");
    compare("train_vs_train (noise floor)", &o_train, &o_train2);
    compare("INFERENCE_vs_TRAIN (the two checkpoint passes)", &o_inf, &o_train);
}
