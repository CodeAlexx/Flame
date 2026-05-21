//! Standalone test for `sdpa::forward_prefix_causal_full` autograd parity.
//!
//! Localizes the HiDream-O1 V LoRA-B grad collapse: the trap probes (in
//! `inference-flame/src/models/hidream_o1/`) show dV out of
//! `forward_prefix_causal_full` is at cos=0.012 vs PyTorch's two-pass-attn
//! equivalent. This test reproduces the structured path on synthetic Q/K/V
//! and compares its dV against a mathematically equivalent single-SDPA-with-
//! materialized-mask path. Both should produce identical forward outputs AND
//! identical gradients. If they disagree, the structured path's autograd
//! chain (narrow + contiguous + 2x Op::FlashAttention + narrow + cat) has
//! a bug that doesn't fire in the equivalent masked single-SDPA path.
//!
//! Test design (soul.md trap pattern):
//! - Tiny shape to keep cost low and avoid cuDNN special cases: B=1, H=2,
//!   S=16, D=8, prefix_len=8. head_dim=8 forces the fallback decomposed-
//!   recompute path (cuDNN flash needs head_dim ∈ {64, 96, 128}), so we
//!   exercise the same backward kernel both paths fall through to.
//! - Q/K/V generated deterministically from sin/cos so the test is
//!   reproducible across runs.
//! - Loss = sum(out²) → backward → compare dQ, dK, dV between the two paths.
//! - The mask for the equivalent single-SDPA mirrors the structured
//!   semantics exactly: prefix rows are causal AMONG THE PREFIX (no
//!   attention to suffix keys), suffix rows are fully unmasked.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{sdpa, set_default_dtype, DType, Device, Shape, Tensor};

// Match HiDream-O1 layer 35's exact attention shape so the bug can be
// reproduced in isolation. head_dim=128 routes through cuDNN flash; S=497
// triggers the cuDNN backward alignment bail (Nq % 64 != 0) and falls to
// `attention_backward_recompute`. prefix=263 matches the AR-prefix length.
const B: usize = 1;
const H: usize = 32;
const S: usize = 497;
const D: usize = 128;
const PREFIX: usize = 263;

/// Build deterministic Q/K/V at [B, H, S, D] from sin/cos waves. Returns
/// requires_grad=true tensors so backward flows.
fn make_qkv(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> (Tensor, Tensor, Tensor) {
    let n = B * H * S * D;
    let q_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() * 0.5).collect();
    let k_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos() * 0.5).collect();
    let v_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.19).sin() + 0.3) * 0.5)
        .collect();
    let shape = Shape::from_dims(&[B, H, S, D]);
    let q = Tensor::from_vec_dtype(q_data, shape.clone(), device.clone(), DType::BF16)
        .expect("from_vec_dtype q")
        .requires_grad_(true);
    let k = Tensor::from_vec_dtype(k_data, shape.clone(), device.clone(), DType::BF16)
        .expect("from_vec_dtype k")
        .requires_grad_(true);
    let v = Tensor::from_vec_dtype(v_data, shape, device.clone(), DType::BF16)
        .expect("from_vec_dtype v")
        .requires_grad_(true);
    (q, k, v)
}

/// Prefix-causal-full mask matching the semantics of
/// `forward_prefix_causal_full`:
///   - rows q ∈ [0, prefix): attend to k ∈ [0, q+1) AND k < prefix
///     (causal WITHIN the prefix — no attention to suffix keys, mirrors
///     the structured path's `forward_causal(q_prefix, k_prefix, v_prefix)`)
///   - rows q ∈ [prefix, S): attend to k ∈ [0, S) (unmasked, mirrors the
///     structured path's `forward(q, k, v, None)` whose suffix rows of out
///     are kept)
///
/// Returns a F32 [1, 1, S, S] keep-mask (1.0 = keep, 0.0 = mask out).
fn prefix_causal_full_mask(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Tensor {
    let mut data = vec![1.0f32; S * S];
    for q_idx in 0..S {
        for k_idx in 0..S {
            let cell = q_idx * S + k_idx;
            if q_idx < PREFIX {
                // Causal within prefix; no suffix keys.
                let keep = (k_idx <= q_idx) && (k_idx < PREFIX);
                data[cell] = if keep { 1.0 } else { 0.0 };
            } else {
                // Unmasked.
                data[cell] = 1.0;
            }
        }
    }
    Tensor::from_vec_dtype(data, Shape::from_dims(&[1, 1, S, S]), device.clone(), DType::F32)
        .expect("mask from_vec_dtype")
}

/// Compare two tensors and return (cos_sim, max_abs, mean_abs).
fn metrics(a: &Tensor, b: &Tensor) -> (f32, f32, f32) {
    let a_f32 = a.to_dtype(DType::F32).expect("a to f32");
    let b_f32 = b.to_dtype(DType::F32).expect("b to f32");
    let a_vec: Vec<f32> = a_f32.to_vec().expect("a to_vec");
    let b_vec: Vec<f32> = b_f32.to_vec().expect("b to_vec");
    assert_eq!(a_vec.len(), b_vec.len(), "tensor length mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (x, y) in a_vec.iter().zip(b_vec.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let cos = (dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32;
    let mean_abs = (sum_abs / a_vec.len() as f64) as f32;
    (cos, max_abs, mean_abs)
}

#[test]
fn sdpa_prefix_causal_full_grad_matches_masked_sdpa() {
    let device = Device::cuda(0).expect("CUDA required");
    set_default_dtype(DType::BF16);
    let cuda = device.cuda_device().clone();

    // ───────────────────────────── Path A: structured ────────────────────
    let (q_a, k_a, v_a) = make_qkv(&cuda);
    let out_a = sdpa::forward_prefix_causal_full(&q_a, &k_a, &v_a, PREFIX)
        .expect("forward_prefix_causal_full");
    // Loss = sum(out²). Square + sum keeps the test self-contained.
    let loss_a = out_a
        .to_dtype(DType::F32)
        .expect("out_a to f32")
        .square()
        .expect("square")
        .sum()
        .expect("sum");
    let grads_a = loss_a.backward().expect("backward A");
    let dq_a = grads_a
        .get(q_a.id())
        .expect("dq_a missing")
        .clone();
    let dk_a = grads_a
        .get(k_a.id())
        .expect("dk_a missing")
        .clone();
    let dv_a = grads_a
        .get(v_a.id())
        .expect("dv_a missing")
        .clone();

    // ───────────────────────────── Path B: masked single SDPA ────────────
    // Fresh tensors with identical data — autograd state from Path A's
    // backward is already consumed for those IDs.
    let (q_b, k_b, v_b) = make_qkv(&cuda);
    let mask = prefix_causal_full_mask(&cuda);
    let out_b = sdpa::forward(&q_b, &k_b, &v_b, Some(&mask)).expect("masked forward");
    let loss_b = out_b
        .to_dtype(DType::F32)
        .expect("out_b to f32")
        .square()
        .expect("square b")
        .sum()
        .expect("sum b");
    let grads_b = loss_b.backward().expect("backward B");
    let dq_b = grads_b.get(q_b.id()).expect("dq_b missing").clone();
    let dk_b = grads_b.get(k_b.id()).expect("dk_b missing").clone();
    let dv_b = grads_b.get(v_b.id()).expect("dv_b missing").clone();

    // ───────────────────────────── Forward parity check ──────────────────
    // Sanity: the two paths MUST produce the same forward output (modulo
    // BF16 rounding). If forward differs, the masks/decomposition aren't
    // equivalent and the gradient test is meaningless.
    let (fwd_cos, fwd_max, fwd_mean) = metrics(&out_a, &out_b);
    println!("forward parity: cos={fwd_cos:.6} max_abs={fwd_max:.3e} mean_abs={fwd_mean:.3e}");
    assert!(
        fwd_cos > 0.999,
        "forward outputs disagree (cos={fwd_cos:.6}) — mask construction or decomposition is wrong, gradient test is meaningless"
    );

    // ───────────────────────────── Gradient parity ───────────────────────
    let (dv_cos, dv_max, dv_mean) = metrics(&dv_a, &dv_b);
    let (dq_cos, dq_max, dq_mean) = metrics(&dq_a, &dq_b);
    let (dk_cos, dk_max, dk_mean) = metrics(&dk_a, &dk_b);
    println!(
        "dV cos={dv_cos:.6} max_abs={dv_max:.3e} mean_abs={dv_mean:.3e}"
    );
    println!(
        "dQ cos={dq_cos:.6} max_abs={dq_max:.3e} mean_abs={dq_mean:.3e}"
    );
    println!(
        "dK cos={dk_cos:.6} max_abs={dk_max:.3e} mean_abs={dk_mean:.3e}"
    );

    // Hard threshold: cos should be > 0.99 for mathematically equivalent
    // paths. The HiDream-O1 trap shows the actual divergence in the real
    // model is cos ≈ 0.012 — orders of magnitude worse than any BF16 noise
    // floor — so any gap here that's bigger than ~0.01 in cos is the bug.
    assert!(
        dv_cos > 0.99,
        "dV gradient diverges between structured and masked paths: cos={dv_cos:.6} \
         — this is the HiDream-O1 SDPA V LoRA-B grad collapse bug, localized here"
    );
    assert!(dq_cos > 0.99, "dQ diverges: cos={dq_cos:.6}");
    assert!(dk_cos > 0.99, "dK diverges: cos={dk_cos:.6}");
}

/// HiDream-O1 attention block: Q/K/V projections + RoPE + repeat_kv +
/// `sdpa_prefix_causal_full`. Designed to match the layer-35 chain that
/// shows V LoRA-B grad cos=0.05. If THIS test passes at HiDream-O1's
/// exact shapes, the bug isn't in the attention chain — it's in the
/// gradient checkpointing wrapper, multi-layer stacking, or somewhere
/// else upstream/downstream.
///
/// Shapes match HiDream-O1 8B:
///   B=1, H_q=32, H_kv=8, S=497, D=128, prefix=263.
///
/// We don't bother with the LoRA adapter here — just dense linears with
/// requires_grad on the weights. The bug would still surface in dW for the
/// projection weights if the attention backward is corrupt.
#[test]
fn hidream_o1_attention_block_grad() {
    let device = Device::cuda(0).expect("CUDA required");
    set_default_dtype(DType::BF16);
    let cuda = device.cuda_device().clone();

    const B: usize = 1;
    const H_Q: usize = 32;
    const H_KV: usize = 8;
    const S: usize = 497;
    const D: usize = 128;
    const PREFIX: usize = 263;
    const HIDDEN: usize = H_Q * D; // 4096

    // Synthetic hidden_states and projection weights.
    let n_h = B * S * HIDDEN;
    let h_data: Vec<f32> = (0..n_h).map(|i| (i as f32 * 0.013).sin() * 0.3).collect();
    let hidden = Tensor::from_vec_dtype(
        h_data,
        Shape::from_dims(&[B, S, HIDDEN]),
        cuda.clone(),
        DType::BF16,
    )
    .expect("hidden")
    .requires_grad_(true);

    let make_proj = |out: usize, salt: f32| -> Tensor {
        let n = out * HIDDEN;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.07 + salt).cos()) * 0.02)
            .collect();
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[out, HIDDEN]),
            cuda.clone(),
            DType::BF16,
        )
        .expect("proj w")
        .requires_grad_(true)
    };
    let w_q = make_proj(H_Q * D, 1.0);
    let w_k = make_proj(H_KV * D, 2.0);
    let w_v = make_proj(H_KV * D, 3.0);

    // LoRA adapters: A [rank, HIDDEN], B [out_dim, rank]. Standard PEFT
    // init: A Kaiming-ish, B zero (but zero-init B means LoRA contributes
    // nothing in the *first* forward — gradient still flows back to B).
    // Use small nonzero init for B so we exercise both sides of the LoRA
    // backward chain, not the degenerate B=0 case.
    const RANK: usize = 32;
    let make_lora_a = |salt: f32| -> Tensor {
        let n = RANK * HIDDEN;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.0017 + salt).sin()) * 0.01)
            .collect();
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[RANK, HIDDEN]),
            cuda.clone(),
            DType::BF16,
        )
        .expect("lora a")
        .requires_grad_(true)
    };
    let make_lora_b = |out: usize, salt: f32| -> Tensor {
        let n = out * RANK;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.0029 + salt).cos()) * 0.005)
            .collect();
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[out, RANK]),
            cuda.clone(),
            DType::BF16,
        )
        .expect("lora b")
        .requires_grad_(true)
    };
    let lora_a_q = make_lora_a(10.0);
    let lora_b_q = make_lora_b(H_Q * D, 11.0);
    let lora_a_k = make_lora_a(20.0);
    let lora_b_k = make_lora_b(H_KV * D, 21.0);
    let lora_a_v = make_lora_a(30.0);
    let lora_b_v = make_lora_b(H_KV * D, 31.0);
    const LORA_SCALE: f32 = 1.0; // alpha/rank when alpha=rank=32

    // Per-head RMSNorm weights (one [D] vector per norm). HiDream-O1 uses
    // eps=1e-6.
    let make_norm_w = |salt: f32| -> Tensor {
        let data: Vec<f32> = (0..D).map(|i| 1.0 + (i as f32 * 0.05 + salt).sin() * 0.1).collect();
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[D]),
            cuda.clone(),
            DType::BF16,
        )
        .expect("norm w")
        .requires_grad_(true)
    };
    let q_norm_w = make_norm_w(0.5);
    let k_norm_w = make_norm_w(0.7);
    const RMS_EPS: f32 = 1e-6;

    // RoPE cos/sin tables — broadcast shape [1, S, half] matches HiDream-O1
    // MRoPE output. Not requires_grad (precomputed positional embeddings).
    let half = D / 2;
    let cos_data: Vec<f32> = (0..S * half)
        .map(|i| ((i as f32 * 0.001).cos()).clamp(-0.999, 0.999))
        .collect();
    let sin_data: Vec<f32> = (0..S * half)
        .map(|i| ((i as f32 * 0.001).sin()).clamp(-0.999, 0.999))
        .collect();
    let pe_cos = Tensor::from_vec_dtype(
        cos_data.clone(),
        Shape::from_dims(&[1, S, half]),
        cuda.clone(),
        DType::BF16,
    )
    .expect("pe_cos");
    let pe_sin = Tensor::from_vec_dtype(
        sin_data.clone(),
        Shape::from_dims(&[1, S, half]),
        cuda.clone(),
        DType::BF16,
    )
    .expect("pe_sin");

    // Q/K/V projections via fused_linear3d_native_lora (records Op::Linear
    // for the base + Op::Add for the LoRA residual). This is the path
    // HiDream-O1's decoder uses when LoRA adapters are attached.
    let q = flame_core::ops::fused_inference::fused_linear3d_native_lora(
        &hidden,
        &w_q,
        None,
        Some(&lora_a_q),
        Some(&lora_b_q),
        LORA_SCALE,
    )
    .expect("q_proj lora");
    let k = flame_core::ops::fused_inference::fused_linear3d_native_lora(
        &hidden,
        &w_k,
        None,
        Some(&lora_a_k),
        Some(&lora_b_k),
        LORA_SCALE,
    )
    .expect("k_proj lora");
    let v = flame_core::ops::fused_inference::fused_linear3d_native_lora(
        &hidden,
        &w_v,
        None,
        Some(&lora_a_v),
        Some(&lora_b_v),
        LORA_SCALE,
    )
    .expect("v_proj lora");

    // Reshape + permute to [B, H, S, D].
    let q = q
        .reshape(&[B, S, H_Q, D])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .and_then(|t| t.contiguous())
        .expect("q reshape");
    let k = k
        .reshape(&[B, S, H_KV, D])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .and_then(|t| t.contiguous())
        .expect("k reshape");
    let v = v
        .reshape(&[B, S, H_KV, D])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .and_then(|t| t.contiguous())
        .expect("v reshape");

    // q_norm / k_norm: per-head RMSNorm on the last dim (D). HiDream-O1
    // reshapes to [B*H*N, D] before the call. V skips this step.
    let q_flat = q.reshape(&[B * H_Q * S, D]).expect("q_flat");
    let q_normed = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &q_flat,
        Some(&q_norm_w),
        RMS_EPS,
    )
    .expect("q_norm");
    let q = q_normed.reshape(&[B, H_Q, S, D]).expect("q reshape back");

    let k_flat = k.reshape(&[B * H_KV * S, D]).expect("k_flat");
    let k_normed = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &k_flat,
        Some(&k_norm_w),
        RMS_EPS,
    )
    .expect("k_norm");
    let k = k_normed.reshape(&[B, H_KV, S, D]).expect("k reshape back");

    // RoPE halfsplit on Q and K (V skips). pe_cos/pe_sin at [1, S, D/2] —
    // broadcast across batch and heads.
    let q = flame_core::bf16_ops::rope_halfsplit_bf16(&q, &pe_cos, &pe_sin)
        .expect("rope q");
    let k = flame_core::bf16_ops::rope_halfsplit_bf16(&k, &pe_cos, &pe_sin)
        .expect("rope k");

    let n_rep = H_Q / H_KV;
    let k = hidream_repeat_kv(&k, n_rep).expect("repeat_kv k");
    let v = hidream_repeat_kv(&v, n_rep).expect("repeat_kv v");
    assert_eq!(k.shape().dims(), &[B, H_Q, S, D]);
    assert_eq!(v.shape().dims(), &[B, H_Q, S, D]);

    // Structured path: forward_prefix_causal_full.
    let out_a = sdpa::forward_prefix_causal_full(&q, &k, &v, PREFIX)
        .expect("forward_prefix_causal_full");
    let loss_a = out_a
        .to_dtype(DType::F32)
        .expect("out_a to f32")
        .square()
        .expect("square")
        .sum()
        .expect("sum");
    let grads_a = loss_a.backward().expect("backward A");

    // Intermediate-tensor grads are dropped by the accumulator unless
    // explicitly retained — only leaf params survive. Compare on the
    // projection weights instead. dW_v is the right proxy for V LoRA-B
    // (both are leaf params consuming the same upstream backward signal).
    let dw_q_struct = grads_a
        .get(w_q.id())
        .expect("dw_q missing — q_proj weight grad not stored")
        .clone();
    let dw_k_struct = grads_a
        .get(w_k.id())
        .expect("dw_k missing — k_proj weight grad not stored")
        .clone();
    let dw_v_struct = grads_a
        .get(w_v.id())
        .expect("dw_v missing — v_proj weight grad not stored")
        .clone();
    let dh_struct = grads_a
        .get(hidden.id())
        .expect("dhidden missing")
        .clone();

    // ───────────────────────────── Path B: masked single SDPA ──────────
    let n_h2 = B * S * HIDDEN;
    let h_data2: Vec<f32> = (0..n_h2).map(|i| (i as f32 * 0.013).sin() * 0.3).collect();
    let hidden_b = Tensor::from_vec_dtype(
        h_data2,
        Shape::from_dims(&[B, S, HIDDEN]),
        cuda.clone(),
        DType::BF16,
    )
    .expect("hidden b")
    .requires_grad_(true);

    let make_proj_b = |out: usize, salt: f32| -> Tensor {
        let n = out * HIDDEN;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.07 + salt).cos()) * 0.02)
            .collect();
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[out, HIDDEN]),
            cuda.clone(),
            DType::BF16,
        )
        .expect("proj w b")
        .requires_grad_(true)
    };
    let w_q_b = make_proj_b(H_Q * D, 1.0);
    let w_k_b = make_proj_b(H_KV * D, 2.0);
    let w_v_b = make_proj_b(H_KV * D, 3.0);

    // Mirror LoRA adapters from Path A.
    let make_lora_a_b = |salt: f32| -> Tensor {
        let n = RANK * HIDDEN;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.0017 + salt).sin()) * 0.01)
            .collect();
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[RANK, HIDDEN]),
            cuda.clone(),
            DType::BF16,
        )
        .expect("lora a b")
        .requires_grad_(true)
    };
    let make_lora_b_b = |out: usize, salt: f32| -> Tensor {
        let n = out * RANK;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.0029 + salt).cos()) * 0.005)
            .collect();
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[out, RANK]),
            cuda.clone(),
            DType::BF16,
        )
        .expect("lora b b")
        .requires_grad_(true)
    };
    let lora_a_q_b = make_lora_a_b(10.0);
    let lora_b_q_b = make_lora_b_b(H_Q * D, 11.0);
    let lora_a_k_b = make_lora_a_b(20.0);
    let lora_b_k_b = make_lora_b_b(H_KV * D, 21.0);
    let lora_a_v_b = make_lora_a_b(30.0);
    let lora_b_v_b = make_lora_b_b(H_KV * D, 31.0);

    // Mirror q_norm/k_norm weights from Path A.
    let make_norm_w_b = |salt: f32| -> Tensor {
        let data: Vec<f32> = (0..D).map(|i| 1.0 + (i as f32 * 0.05 + salt).sin() * 0.1).collect();
        Tensor::from_vec_dtype(
            data,
            Shape::from_dims(&[D]),
            cuda.clone(),
            DType::BF16,
        )
        .expect("norm w b")
        .requires_grad_(true)
    };
    let q_norm_w_b = make_norm_w_b(0.5);
    let k_norm_w_b = make_norm_w_b(0.7);

    let q_b = flame_core::ops::fused_inference::fused_linear3d_native_lora(
        &hidden_b,
        &w_q_b,
        None,
        Some(&lora_a_q_b),
        Some(&lora_b_q_b),
        LORA_SCALE,
    )
    .expect("q_proj lora b");
    let k_b = flame_core::ops::fused_inference::fused_linear3d_native_lora(
        &hidden_b,
        &w_k_b,
        None,
        Some(&lora_a_k_b),
        Some(&lora_b_k_b),
        LORA_SCALE,
    )
    .expect("k_proj lora b");
    let v_b = flame_core::ops::fused_inference::fused_linear3d_native_lora(
        &hidden_b,
        &w_v_b,
        None,
        Some(&lora_a_v_b),
        Some(&lora_b_v_b),
        LORA_SCALE,
    )
    .expect("v_proj lora b");
    let q_b = q_b
        .reshape(&[B, S, H_Q, D])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .and_then(|t| t.contiguous())
        .expect("q_b reshape");
    let k_b = k_b
        .reshape(&[B, S, H_KV, D])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .and_then(|t| t.contiguous())
        .expect("k_b reshape");
    let v_b = v_b
        .reshape(&[B, S, H_KV, D])
        .and_then(|t| t.permute(&[0, 2, 1, 3]))
        .and_then(|t| t.contiguous())
        .expect("v_b reshape");

    let q_flat_b = q_b.reshape(&[B * H_Q * S, D]).expect("q_flat b");
    let q_normed_b = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &q_flat_b,
        Some(&q_norm_w_b),
        RMS_EPS,
    )
    .expect("q_norm b");
    let q_b = q_normed_b.reshape(&[B, H_Q, S, D]).expect("q reshape b");
    let k_flat_b = k_b.reshape(&[B * H_KV * S, D]).expect("k_flat b");
    let k_normed_b = flame_core::cuda_ops_bf16::rms_norm_bf16(
        &k_flat_b,
        Some(&k_norm_w_b),
        RMS_EPS,
    )
    .expect("k_norm b");
    let k_b = k_normed_b.reshape(&[B, H_KV, S, D]).expect("k reshape b");

    // Use the same pe_cos / pe_sin tables as Path A (deterministic data).
    let pe_cos_b = Tensor::from_vec_dtype(
        cos_data,
        Shape::from_dims(&[1, S, half]),
        cuda.clone(),
        DType::BF16,
    )
    .expect("pe_cos b");
    let pe_sin_b = Tensor::from_vec_dtype(
        sin_data,
        Shape::from_dims(&[1, S, half]),
        cuda.clone(),
        DType::BF16,
    )
    .expect("pe_sin b");
    let q_b = flame_core::bf16_ops::rope_halfsplit_bf16(&q_b, &pe_cos_b, &pe_sin_b)
        .expect("rope q b");
    let k_b = flame_core::bf16_ops::rope_halfsplit_bf16(&k_b, &pe_cos_b, &pe_sin_b)
        .expect("rope k b");

    let k_b = hidream_repeat_kv(&k_b, n_rep).expect("repeat_kv k b");
    let v_b = hidream_repeat_kv(&v_b, n_rep).expect("repeat_kv v b");

    let mask = prefix_causal_full_mask_for(S, PREFIX, &cuda);
    let out_b = sdpa::forward(&q_b, &k_b, &v_b, Some(&mask)).expect("masked forward");
    let loss_b = out_b
        .to_dtype(DType::F32)
        .expect("out_b to f32")
        .square()
        .expect("square b")
        .sum()
        .expect("sum b");
    let grads_b = loss_b.backward().expect("backward B");
    let dw_q_masked = grads_b
        .get(w_q_b.id())
        .expect("dw_q_b missing")
        .clone();
    let dw_k_masked = grads_b
        .get(w_k_b.id())
        .expect("dw_k_b missing")
        .clone();
    let dw_v_masked = grads_b
        .get(w_v_b.id())
        .expect("dw_v_b missing")
        .clone();
    let dh_masked = grads_b
        .get(hidden_b.id())
        .expect("dhidden_b missing")
        .clone();

    // Compare.
    let (fwd_cos, fwd_max, _) = metrics(&out_a, &out_b);
    println!("attention block fwd parity: cos={fwd_cos:.6} max_abs={fwd_max:.3e}");
    assert!(fwd_cos > 0.999, "fwd diverges, mask wrong");

    let dw_qnorm_struct = grads_a.get(q_norm_w.id()).expect("dq_norm_w missing").clone();
    let dw_knorm_struct = grads_a.get(k_norm_w.id()).expect("dk_norm_w missing").clone();
    let dw_qnorm_masked = grads_b.get(q_norm_w_b.id()).expect("dq_norm_w_b missing").clone();
    let dw_knorm_masked = grads_b.get(k_norm_w_b.id()).expect("dk_norm_w_b missing").clone();

    // THE ACTUAL SYMPTOM TENSORS: LoRA-B grads for Q/K/V. These are what
    // collapse to cos≈0.05 in the real model.
    let d_lora_b_q_struct = grads_a.get(lora_b_q.id()).expect("d_lora_b_q missing").clone();
    let d_lora_b_k_struct = grads_a.get(lora_b_k.id()).expect("d_lora_b_k missing").clone();
    let d_lora_b_v_struct = grads_a.get(lora_b_v.id()).expect("d_lora_b_v missing").clone();
    let d_lora_b_q_masked = grads_b.get(lora_b_q_b.id()).expect("d_lora_b_q_b missing").clone();
    let d_lora_b_k_masked = grads_b.get(lora_b_k_b.id()).expect("d_lora_b_k_b missing").clone();
    let d_lora_b_v_masked = grads_b.get(lora_b_v_b.id()).expect("d_lora_b_v_b missing").clone();

    let (dwq_cos, dwq_max, _) = metrics(&dw_q_struct, &dw_q_masked);
    let (dwk_cos, dwk_max, _) = metrics(&dw_k_struct, &dw_k_masked);
    let (dwv_cos, dwv_max, _) = metrics(&dw_v_struct, &dw_v_masked);
    let (dh_cos, dh_max, _) = metrics(&dh_struct, &dh_masked);
    let (dqn_cos, dqn_max, _) = metrics(&dw_qnorm_struct, &dw_qnorm_masked);
    let (dkn_cos, dkn_max, _) = metrics(&dw_knorm_struct, &dw_knorm_masked);
    println!(
        "block-grad cos: dW_Q={dwq_cos:.6} (max={dwq_max:.3e})  dW_K={dwk_cos:.6} (max={dwk_max:.3e})  dW_V={dwv_cos:.6} (max={dwv_max:.3e})"
    );
    println!(
        "                dQ_norm={dqn_cos:.6} (max={dqn_max:.3e})  dK_norm={dkn_cos:.6} (max={dkn_max:.3e})  dHidden={dh_cos:.6} (max={dh_max:.3e})"
    );
    assert!(dwv_cos > 0.99, "dW_v diverges (this is the V LoRA-B grad collapse bug, reproduced): cos={dwv_cos:.6}");
    assert!(dwq_cos > 0.99, "dW_q diverges: cos={dwq_cos:.6}");
    assert!(dwk_cos > 0.99, "dW_k diverges: cos={dwk_cos:.6}");
    assert!(dqn_cos > 0.99, "dQ_norm diverges: cos={dqn_cos:.6}");
    assert!(dkn_cos > 0.99, "dK_norm diverges: cos={dkn_cos:.6}");
    assert!(dh_cos > 0.99, "dHidden diverges: cos={dh_cos:.6}");

    // THE PRIMARY ASSERTION: LoRA-B grads. cos≈0.05 in the real model.
    let (dlbq_cos, dlbq_max, _) = metrics(&d_lora_b_q_struct, &d_lora_b_q_masked);
    let (dlbk_cos, dlbk_max, _) = metrics(&d_lora_b_k_struct, &d_lora_b_k_masked);
    let (dlbv_cos, dlbv_max, _) = metrics(&d_lora_b_v_struct, &d_lora_b_v_masked);
    println!(
        "LORA-B grad cos: dQ_lora_B={dlbq_cos:.6} (max={dlbq_max:.3e})  dK_lora_B={dlbk_cos:.6} (max={dlbk_max:.3e})  dV_lora_B={dlbv_cos:.6} (max={dlbv_max:.3e})"
    );
    assert!(
        dlbv_cos > 0.99,
        "V LoRA-B grad collapse reproduced in isolation: cos={dlbv_cos:.6}"
    );
    assert!(dlbq_cos > 0.99, "Q LoRA-B diverges: cos={dlbq_cos:.6}");
    assert!(dlbk_cos > 0.99, "K LoRA-B diverges: cos={dlbk_cos:.6}");
}

/// Same test as `hidream_o1_attention_block_grad` but with the forward
/// path wrapped in `checkpoint_offload_boundary` — matches what
/// HiDream-O1's model.rs:432 does. If THIS test fails while
/// `hidream_o1_attention_block_grad` passes, the bug is in the
/// checkpoint recompute mechanism (the only remaining suspect).
#[test]
fn hidream_o1_attention_block_grad_checkpointed() {
    let device = Device::cuda(0).expect("CUDA required");
    set_default_dtype(DType::BF16);
    let cuda = device.cuda_device().clone();

    // Install the grow activation cache so checkpoint_offload_boundary uses
    // its FULL path (Op::CheckpointOffloadBoundary) instead of falling back
    // to plain Op::Checkpoint. The HiDream-O1 trainer installs this; the
    // test path was previously falling back without it.
    {
        use std::sync::{Arc as A2, Mutex as M2};
        let cache = flame_core::activation_offload::GrowOnDemandActivationCache::new(
            cuda.clone(),
            64 * 1024 * 1024, // 64MB slab
        )
        .expect("grow cache new");
        flame_core::autograd::set_grow_activation_cache(A2::new(M2::new(cache)))
            .expect("install grow cache");
    }

    const B: usize = 1;
    const H_Q: usize = 32;
    const H_KV: usize = 8;
    const S: usize = 497;
    const D: usize = 128;
    const PREFIX: usize = 263;
    const HIDDEN: usize = H_Q * D;
    const RANK: usize = 32;
    const RMS_EPS: f32 = 1e-6;
    const LORA_SCALE: f32 = 1.0;
    let half = D / 2;
    let n_rep = H_Q / H_KV;

    // Inputs / weights — built once, used by both paths.
    let make_hidden = || -> Tensor {
        let n = B * S * HIDDEN;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.3).collect();
        Tensor::from_vec_dtype(data, Shape::from_dims(&[B, S, HIDDEN]), cuda.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true)
    };
    let make_proj_w = |out: usize, salt: f32| -> Tensor {
        let n = out * HIDDEN;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.07 + salt).cos()) * 0.02).collect();
        Tensor::from_vec_dtype(data, Shape::from_dims(&[out, HIDDEN]), cuda.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true)
    };
    let make_norm_w = |salt: f32| -> Tensor {
        let data: Vec<f32> = (0..D).map(|i| 1.0 + (i as f32 * 0.05 + salt).sin() * 0.1).collect();
        Tensor::from_vec_dtype(data, Shape::from_dims(&[D]), cuda.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true)
    };
    let make_lora_a = |salt: f32| -> Tensor {
        let n = RANK * HIDDEN;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.0017 + salt).sin()) * 0.01).collect();
        Tensor::from_vec_dtype(data, Shape::from_dims(&[RANK, HIDDEN]), cuda.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true)
    };
    let make_lora_b = |out: usize, salt: f32| -> Tensor {
        let n = out * RANK;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.0029 + salt).cos()) * 0.005).collect();
        Tensor::from_vec_dtype(data, Shape::from_dims(&[out, RANK]), cuda.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true)
    };

    let make_pe = |salt: f32| -> Tensor {
        let n = S * half;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.001 + salt).cos()).clamp(-0.999, 0.999)).collect();
        Tensor::from_vec_dtype(data, Shape::from_dims(&[1, S, half]), cuda.clone(), DType::BF16).unwrap()
    };

    // Closure: runs the full attention block forward. Used by BOTH paths,
    // so the two only differ in whether they're wrapped in checkpoint.
    let run_block = |hidden: &Tensor,
                     w_q: &Tensor, w_k: &Tensor, w_v: &Tensor,
                     lora_a_q: &Tensor, lora_b_q: &Tensor,
                     lora_a_k: &Tensor, lora_b_k: &Tensor,
                     lora_a_v: &Tensor, lora_b_v: &Tensor,
                     q_norm_w: &Tensor, k_norm_w: &Tensor,
                     pe_cos: &Tensor, pe_sin: &Tensor| -> flame_core::Result<Tensor> {
        use flame_core::ops::fused_inference::fused_linear3d_native_lora;
        let q = fused_linear3d_native_lora(hidden, w_q, None, Some(lora_a_q), Some(lora_b_q), LORA_SCALE)?;
        let k = fused_linear3d_native_lora(hidden, w_k, None, Some(lora_a_k), Some(lora_b_k), LORA_SCALE)?;
        let v = fused_linear3d_native_lora(hidden, w_v, None, Some(lora_a_v), Some(lora_b_v), LORA_SCALE)?;
        let q = q.reshape(&[B, S, H_Q, D])?.permute(&[0, 2, 1, 3])?.contiguous()?;
        let k = k.reshape(&[B, S, H_KV, D])?.permute(&[0, 2, 1, 3])?.contiguous()?;
        let v = v.reshape(&[B, S, H_KV, D])?.permute(&[0, 2, 1, 3])?.contiguous()?;
        let q_flat = q.reshape(&[B * H_Q * S, D])?;
        let q_normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&q_flat, Some(q_norm_w), RMS_EPS)?;
        let q = q_normed.reshape(&[B, H_Q, S, D])?;
        let k_flat = k.reshape(&[B * H_KV * S, D])?;
        let k_normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&k_flat, Some(k_norm_w), RMS_EPS)?;
        let k = k_normed.reshape(&[B, H_KV, S, D])?;
        let q = flame_core::bf16_ops::rope_halfsplit_bf16(&q, pe_cos, pe_sin)?;
        let k = flame_core::bf16_ops::rope_halfsplit_bf16(&k, pe_cos, pe_sin)?;
        let k = hidream_repeat_kv(&k, n_rep)?;
        let v = hidream_repeat_kv(&v, n_rep)?;
        sdpa::forward_prefix_causal_full(&q, &k, &v, PREFIX)
    };

    // ───────────────────────────── Path A: NON-checkpointed ─────────────
    let hidden_a = make_hidden();
    let w_q_a = make_proj_w(H_Q * D, 1.0);
    let w_k_a = make_proj_w(H_KV * D, 2.0);
    let w_v_a = make_proj_w(H_KV * D, 3.0);
    let la_q_a = make_lora_a(10.0);
    let lb_q_a = make_lora_b(H_Q * D, 11.0);
    let la_k_a = make_lora_a(20.0);
    let lb_k_a = make_lora_b(H_KV * D, 21.0);
    let la_v_a = make_lora_a(30.0);
    let lb_v_a = make_lora_b(H_KV * D, 31.0);
    let qn_a = make_norm_w(0.5);
    let kn_a = make_norm_w(0.7);
    let pe_cos_a = make_pe(0.0);
    let pe_sin_a = make_pe(1.5);

    // Inline path A so we can capture v_post_repeat_kv's id for retention.
    use flame_core::ops::fused_inference::fused_linear3d_native_lora;
    let q_a = fused_linear3d_native_lora(&hidden_a, &w_q_a, None, Some(&la_q_a), Some(&lb_q_a), LORA_SCALE).unwrap();
    let k_a = fused_linear3d_native_lora(&hidden_a, &w_k_a, None, Some(&la_k_a), Some(&lb_k_a), LORA_SCALE).unwrap();
    let v_a = fused_linear3d_native_lora(&hidden_a, &w_v_a, None, Some(&la_v_a), Some(&lb_v_a), LORA_SCALE).unwrap();
    // v_a is the V_proj OUTPUT, shape [B, S, Hkv*D]. This is the Python
    // probe point (forward_hook on v_proj). Capture its id for comparison.
    let v_proj_out_id_a = v_a.id();
    let q_a = q_a.reshape(&[B, S, H_Q, D]).unwrap().permute(&[0, 2, 1, 3]).unwrap().contiguous().unwrap();
    let k_a = k_a.reshape(&[B, S, H_KV, D]).unwrap().permute(&[0, 2, 1, 3]).unwrap().contiguous().unwrap();
    let v_a_pre = v_a.reshape(&[B, S, H_KV, D]).unwrap().permute(&[0, 2, 1, 3]).unwrap().contiguous().unwrap();
    let q_flat_a = q_a.reshape(&[B * H_Q * S, D]).unwrap();
    let q_normed_a = flame_core::cuda_ops_bf16::rms_norm_bf16(&q_flat_a, Some(&qn_a), RMS_EPS).unwrap();
    let q_a = q_normed_a.reshape(&[B, H_Q, S, D]).unwrap();
    let k_flat_a = k_a.reshape(&[B * H_KV * S, D]).unwrap();
    let k_normed_a = flame_core::cuda_ops_bf16::rms_norm_bf16(&k_flat_a, Some(&kn_a), RMS_EPS).unwrap();
    let k_a = k_normed_a.reshape(&[B, H_KV, S, D]).unwrap();
    let q_a = flame_core::bf16_ops::rope_halfsplit_bf16(&q_a, &pe_cos_a, &pe_sin_a).unwrap();
    let k_a = flame_core::bf16_ops::rope_halfsplit_bf16(&k_a, &pe_cos_a, &pe_sin_a).unwrap();
    let k_a = hidream_repeat_kv(&k_a, n_rep).unwrap();
    let v_a = hidream_repeat_kv(&v_a_pre, n_rep).unwrap();
    let v_a_probe_id = v_a.id();
    let out_a = sdpa::forward_prefix_causal_full(&q_a, &k_a, &v_a, PREFIX).unwrap();
    let loss_a = out_a.to_dtype(DType::F32).unwrap().square().unwrap().sum().unwrap();

    // Register v_a's id (post-repeat_kv) AND v_proj_out_id_a (pre-reshape)
    // for intermediate-grad retention BEFORE backward.
    let mut retain_ids = std::collections::HashSet::new();
    retain_ids.insert(v_a_probe_id);
    retain_ids.insert(v_proj_out_id_a);
    flame_core::autograd::AutogradContext::retain_intermediate_grads(retain_ids);
    let grads_a = loss_a.backward().expect("backward A");
    let retained_a = flame_core::autograd::AutogradContext::take_retained_intermediate_grads();
    let dv_intermediate_a = retained_a.get(&v_a_probe_id).cloned();
    let dv_proj_out_a = retained_a.get(&v_proj_out_id_a).cloned();
    println!(
        "non-checkpointed probes: v_post_repeat_kv={} v_proj_out={}",
        dv_intermediate_a.is_some(), dv_proj_out_a.is_some()
    );

    // Apply the parity binary's rearrangement to v_post_repeat_kv's grad and
    // compare against the DIRECT v_proj_out grad. If they match, the
    // rearrangement formula is correct. If not, the rearrangement is the bug.
    if let (Some(dv_post), Some(dv_proj)) = (dv_intermediate_a.as_ref(), dv_proj_out_a.as_ref()) {
        let dims = dv_post.shape().dims().to_vec();
        let (b, hq, s, d) = (dims[0], dims[1], dims[2], dims[3]);
        let rearranged = dv_post
            .reshape(&[b, H_KV, n_rep, s, d])
            .unwrap()
            .sum_dim_keepdim(2)
            .unwrap()
            .reshape(&[b, H_KV, s, d])
            .unwrap()
            .permute(&[0, 2, 1, 3])
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape(&[b, s, H_KV * d])
            .unwrap();
        let (rearr_cos, rearr_max, rearr_mean) = metrics(&rearranged, dv_proj);
        println!(
            "REARRANGEMENT validation: sum_reduce(v_post_repeat_kv).permute.reshape vs direct v_proj_out cos={rearr_cos:.6} max_abs={rearr_max:.3e} mean_abs={rearr_mean:.3e}"
        );
        assert!(
            rearr_cos > 0.99,
            "Parity-binary rearrangement formula is WRONG: cos={rearr_cos:.6} — this explains the HiDream-O1 trap reading cos=0.012!"
        );
    }

    // ───────────────────────────── Path B: checkpointed ─────────────────
    let hidden_b = make_hidden();
    let w_q_b = make_proj_w(H_Q * D, 1.0);
    let w_k_b = make_proj_w(H_KV * D, 2.0);
    let w_v_b = make_proj_w(H_KV * D, 3.0);
    let la_q_b = make_lora_a(10.0);
    let lb_q_b = make_lora_b(H_Q * D, 11.0);
    let la_k_b = make_lora_a(20.0);
    let lb_k_b = make_lora_b(H_KV * D, 21.0);
    let la_v_b = make_lora_a(30.0);
    let lb_v_b = make_lora_b(H_KV * D, 31.0);
    let qn_b = make_norm_w(0.5);
    let kn_b = make_norm_w(0.7);
    let pe_cos_b = make_pe(0.0);
    let pe_sin_b = make_pe(1.5);

    // Clone everything we need INTO the closure (Tensor::clone preserves id).
    let w_q_c = w_q_b.clone();
    let w_k_c = w_k_b.clone();
    let w_v_c = w_v_b.clone();
    let la_q_c = la_q_b.clone();
    let lb_q_c = lb_q_b.clone();
    let la_k_c = la_k_b.clone();
    let lb_k_c = lb_k_b.clone();
    let la_v_c = la_v_b.clone();
    let lb_v_c = lb_v_b.clone();
    let qn_c = qn_b.clone();
    let kn_c = kn_b.clone();
    let pe_cos_c = pe_cos_b.clone();
    let pe_sin_c = pe_sin_b.clone();

    // Capture v_post_repeat_kv's tensor id from inside the checkpoint
    // recompute, and register it for intermediate-grad retention — exactly
    // what the HiDream-O1 trap does.
    use std::sync::Arc;
    use std::sync::Mutex;
    let probe_id_slot: Arc<Mutex<Option<flame_core::TensorId>>> = Arc::new(Mutex::new(None));
    let slot_clone = probe_id_slot.clone();

    let out_b = flame_core::autograd::AutogradContext::checkpoint_offload_boundary(
        &[hidden_b.clone()],
        move |inputs: &[Tensor]| {
            let h = inputs[0].clone();
            use flame_core::ops::fused_inference::fused_linear3d_native_lora;
            let q = fused_linear3d_native_lora(&h, &w_q_c, None, Some(&la_q_c), Some(&lb_q_c), LORA_SCALE)?;
            let k = fused_linear3d_native_lora(&h, &w_k_c, None, Some(&la_k_c), Some(&lb_k_c), LORA_SCALE)?;
            let v = fused_linear3d_native_lora(&h, &w_v_c, None, Some(&la_v_c), Some(&lb_v_c), LORA_SCALE)?;
            let q = q.reshape(&[B, S, H_Q, D])?.permute(&[0, 2, 1, 3])?.contiguous()?;
            let k = k.reshape(&[B, S, H_KV, D])?.permute(&[0, 2, 1, 3])?.contiguous()?;
            let v = v.reshape(&[B, S, H_KV, D])?.permute(&[0, 2, 1, 3])?.contiguous()?;
            let q_flat = q.reshape(&[B * H_Q * S, D])?;
            let q_normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&q_flat, Some(&qn_c), RMS_EPS)?;
            let q = q_normed.reshape(&[B, H_Q, S, D])?;
            let k_flat = k.reshape(&[B * H_KV * S, D])?;
            let k_normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&k_flat, Some(&kn_c), RMS_EPS)?;
            let k = k_normed.reshape(&[B, H_KV, S, D])?;
            let q = flame_core::bf16_ops::rope_halfsplit_bf16(&q, &pe_cos_c, &pe_sin_c)?;
            let k = flame_core::bf16_ops::rope_halfsplit_bf16(&k, &pe_cos_c, &pe_sin_c)?;
            let k = hidream_repeat_kv(&k, n_rep)?;
            let v = hidream_repeat_kv(&v, n_rep)?;
            // Probe: record v_post_repeat_kv's id and register for retention.
            // Last-writer wins → the recompute pass overwrites with the
            // recompute-pass id.
            if flame_core::autograd::AutogradContext::is_checkpoint_recompute() {
                if let Ok(mut s) = slot_clone.lock() {
                    *s = Some(v.id());
                }
                let mut ids = std::collections::HashSet::new();
                ids.insert(v.id());
                flame_core::autograd::AutogradContext::retain_intermediate_grads_add(ids);
            }
            sdpa::forward_prefix_causal_full(&q, &k, &v, PREFIX)
        },
    ).expect("checkpoint forward");
    let loss_b = out_b.to_dtype(DType::F32).unwrap().square().unwrap().sum().unwrap();
    let grads_b = loss_b.backward().expect("backward B");
    let retained_b = flame_core::autograd::AutogradContext::take_retained_intermediate_grads();
    let v_probe_id_b = probe_id_slot.lock().unwrap().clone();
    let dv_intermediate_b = v_probe_id_b.and_then(|id| retained_b.get(&id).cloned());
    println!(
        "checkpointed v_post_repeat_kv probe: id_recorded={} grad_retained={}",
        v_probe_id_b.is_some(),
        dv_intermediate_b.is_some()
    );

    // ─────────────── Compare LoRA-B grads ──────────────────────────────
    let dlbq_a = grads_a.get(lb_q_a.id()).expect("lb_q_a grad").clone();
    let dlbk_a = grads_a.get(lb_k_a.id()).expect("lb_k_a grad").clone();
    let dlbv_a = grads_a.get(lb_v_a.id()).expect("lb_v_a grad").clone();
    let dlbq_b = grads_b.get(lb_q_b.id()).expect("lb_q_b grad").clone();
    let dlbk_b = grads_b.get(lb_k_b.id()).expect("lb_k_b grad").clone();
    let dlbv_b = grads_b.get(lb_v_b.id()).expect("lb_v_b grad").clone();

    let (q_cos, q_max, _) = metrics(&dlbq_a, &dlbq_b);
    let (k_cos, k_max, _) = metrics(&dlbk_a, &dlbk_b);
    let (v_cos, v_max, _) = metrics(&dlbv_a, &dlbv_b);
    println!(
        "checkpoint vs non-checkpoint LoRA-B grads: Q_lora_B={q_cos:.6} (max={q_max:.3e})  K_lora_B={k_cos:.6} (max={k_max:.3e})  V_lora_B={v_cos:.6} (max={v_max:.3e})"
    );

    assert!(
        v_cos > 0.99,
        "V LoRA-B grad collapses through checkpoint wrapper: cos={v_cos:.6} (THIS IS THE BUG)"
    );
    assert!(q_cos > 0.99, "Q LoRA-B diverges: cos={q_cos:.6}");
    assert!(k_cos > 0.99, "K LoRA-B diverges: cos={k_cos:.6}");

    // ─── Intermediate-grad comparison through the retain mechanism ─────
    // If LoRA-B grads match bit-exact but intermediate v_post_repeat_kv
    // grads diverge, the bug is in `retain_intermediate_grads_add` +
    // sub-tape retain handling (the trap path we added today).
    match (dv_intermediate_a.as_ref(), dv_intermediate_b.as_ref()) {
        (Some(a), Some(b)) => {
            let (vint_cos, vint_max, _) = metrics(a, b);
            println!(
                "v_post_repeat_kv intermediate-grad cos={vint_cos:.6} max_abs={vint_max:.3e} (non-checkpoint vs checkpoint via retain)"
            );
            assert!(
                vint_cos > 0.99,
                "Intermediate-grad capture diverges between non-checkpoint and checkpoint paths: cos={vint_cos:.6} — THIS IS THE TRAP MECHANISM BUG"
            );
        }
        (Some(_), None) => panic!("checkpoint retain failed to capture v_post_repeat_kv grad"),
        (None, _) => panic!("non-checkpoint retain failed to capture v_post_repeat_kv grad"),
    }
}

/// Helper: HiDream-O1 style repeat_kv. Stacks `n_rep` copies along a new
/// axis after H_kv, then reshapes to expand H_kv → H_kv*n_rep.
fn hidream_repeat_kv(x: &Tensor, n_rep: usize) -> flame_core::Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    assert_eq!(dims.len(), 4, "repeat_kv expects [B,H,S,D]");
    let b = dims[0];
    let h_kv = dims[1];
    let s = dims[2];
    let d = dims[3];
    let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
    let stacked = Tensor::stack(&copies, 2)?; // [B, H_kv, n_rep, S, D]
    stacked.reshape(&[b, h_kv * n_rep, s, d])
}

/// Helper: prefix-causal-full mask for arbitrary S and prefix.
fn prefix_causal_full_mask_for(
    s: usize,
    prefix: usize,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Tensor {
    let mut data = vec![1.0f32; s * s];
    for q_idx in 0..s {
        for k_idx in 0..s {
            let cell = q_idx * s + k_idx;
            if q_idx < prefix {
                let keep = (k_idx <= q_idx) && (k_idx < prefix);
                data[cell] = if keep { 1.0 } else { 0.0 };
            } else {
                data[cell] = 1.0;
            }
        }
    }
    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[1, 1, s, s]),
        device.clone(),
        DType::F32,
    )
    .expect("mask")
}
