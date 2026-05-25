//! Closed-loop LoRA-Linear training parity vs PyTorch.
//!
//! Source: `HANDOFF_2026-05-24_ZIMAGE_VAE_FIXED_ADAM_AMP_OPEN.md` — after
//! `tests/adam_torch_parity.rs` ruled out Suspect A (Adam math is bit-clean
//! vs torch.optim.AdamW), the open bug must be in what feeds Adam. This
//! test isolates the LoRA forward + backward in a minimal closed loop
//! (no model, no scheduler, no dataloader, no RNG variance) and asks:
//! does our LoRA-B grad L2 amplify vs PyTorch's over 50 steps?
//!
//! Mirrors `eridiffusion-core/src/lora.rs::LoRALinear::forward_delta`:
//!     out = (alpha / rank) * (input @ A^T @ B^T)
//! with F32 params, BF16 matmul (params cast inside forward), BF16 MSE
//! loss. Uses identical init / inputs / targets to the PyTorch reference
//! in `tests/parity/lora_closed_loop_ref.py`.
//!
//! If our per-step `||grad_B||` matches torch's, the LoRA forward/backward
//! is clean and the trainer-level 4× burst lives elsewhere (RNG, data
//! ordering, real-model bug, etc.). If ours diverges, this is the bug.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{
    adam::AdamW, global_cuda_device, loss::mse_loss_bf16, AutogradContext, DType, Parameter,
    Result, Shape, Tensor,
};
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Config {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    in_features: usize,
    out_features: usize,
    rank: usize,
    alpha: f32,
    batch: usize,
    seq: usize,
    steps: usize,
}

#[derive(serde::Deserialize)]
struct StepRecord {
    step: usize,
    loss: f32,
    grad_a_l2: f32,
    grad_b_l2: f32,
    grad_a_max_abs: f32,
    grad_b_max_abs: f32,
    a_l2_after: f32,
    b_l2_after: f32,
}

#[derive(serde::Deserialize)]
struct Fixture {
    config: Config,
    a_init: Vec<Vec<f32>>,           // [rank, in_features]
    b_init: Vec<Vec<f32>>,           // [out_features, rank]
    inputs_bf16_as_f32: Vec<Vec<Vec<Vec<f32>>>>,  // [steps][batch][seq][in]
    targets_bf16_as_f32: Vec<Vec<Vec<Vec<f32>>>>, // [steps][batch][seq][out]
    per_step: Vec<StepRecord>,
    a_final: Vec<Vec<f32>>,
    b_final: Vec<Vec<f32>>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("lora_closed_loop_ref.json")
}

fn load_fixture() -> Fixture {
    let path = fixture_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture {}: {} — regenerate via \
             `python3 tests/parity/lora_closed_loop_ref.py`",
            path.display(),
            e
        )
    });
    serde_json::from_slice(&bytes).expect("malformed lora_closed_loop_ref.json")
}

fn flatten4<T: Copy>(x: &[Vec<Vec<Vec<T>>>], k: usize) -> Vec<T> {
    x[k].iter().flatten().flatten().copied().collect()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0_f32, |a, &b| a.max(b.abs()))
}

/// Mirror of `LoRALinear::forward_delta` (F32 params, BF16 forward matmul,
/// alpha/rank scaling). Input is BF16 of shape [batch, seq, in].
fn lora_forward(
    input_bf16: &Tensor,
    a_f32: &Parameter,
    b_f32: &Parameter,
    alpha: f32,
    rank: usize,
    out_features: usize,
) -> Result<Tensor> {
    let in_features = input_bf16.shape().dims()[input_bf16.shape().dims().len() - 1];
    let orig_dims: Vec<usize> = input_bf16.shape().dims().to_vec();
    let leading: usize = orig_dims[..orig_dims.len() - 1].iter().product();
    let input_2d = input_bf16.reshape(&[leading, in_features])?;

    let a_bf16 = a_f32.tensor()?.to_dtype(DType::BF16)?;
    let b_bf16 = b_f32.tensor()?.to_dtype(DType::BF16)?;
    let a_t = a_bf16.transpose()?;
    let b_t = b_bf16.transpose()?;

    let intermediate = input_2d.matmul(&a_t)?;
    let out_2d = intermediate.matmul(&b_t)?;

    let scale = alpha / (rank as f32);
    let scaled = if (scale - 1.0).abs() > f32::EPSILON {
        out_2d.mul_scalar(scale)?
    } else {
        out_2d
    };

    let mut out_shape = orig_dims;
    *out_shape.last_mut().unwrap() = out_features;
    Ok(scaled.reshape(&out_shape)?)
}

#[test]
fn lora_closed_loop_matches_pytorch() -> Result<()> {
    let fx = load_fixture();
    let cfg = &fx.config;
    let device = global_cuda_device();

    // Build A [rank, in], B [out, rank] from fixture inits (row-major).
    let a_flat: Vec<f32> = fx.a_init.iter().flatten().copied().collect();
    let b_flat: Vec<f32> = fx.b_init.iter().flatten().copied().collect();
    assert_eq!(a_flat.len(), cfg.rank * cfg.in_features);
    assert_eq!(b_flat.len(), cfg.out_features * cfg.rank);

    let a_tensor = Tensor::from_vec(
        a_flat,
        Shape::from_dims(&[cfg.rank, cfg.in_features]),
        device.clone(),
    )?
    .requires_grad_(true);
    let b_tensor = Tensor::from_vec(
        b_flat,
        Shape::from_dims(&[cfg.out_features, cfg.rank]),
        device.clone(),
    )?
    .requires_grad_(true);

    let lora_a = Parameter::new(a_tensor);
    let lora_b = Parameter::new(b_tensor);
    let params = [lora_a.clone(), lora_b.clone()];

    let mut opt = AdamW::new(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);

    // Tolerances. BF16 matmul + F32 reduction is the dominant noise source.
    // Stricter than the 4× bug we're hunting; if the test passes, the bug
    // is NOT in this minimal LoRA path.
    let grad_l2_rel_tol: f32 = 0.05;       // 5% rel difference per step
    let loss_rel_tol: f32 = 0.05;
    let mut diverged = false;
    let mut report = String::new();

    for k in 0..cfg.steps {
        // Build BF16 input + target (round-trip through F32 → BF16 mirrors
        // PyTorch's saved float-list-then-cast).
        let in_flat = flatten4(&fx.inputs_bf16_as_f32, k);
        let tgt_flat = flatten4(&fx.targets_bf16_as_f32, k);
        assert_eq!(in_flat.len(), cfg.batch * cfg.seq * cfg.in_features);
        assert_eq!(tgt_flat.len(), cfg.batch * cfg.seq * cfg.out_features);

        let input_f32 = Tensor::from_vec(
            in_flat,
            Shape::from_dims(&[cfg.batch, cfg.seq, cfg.in_features]),
            device.clone(),
        )?;
        let input_bf16 = input_f32.to_dtype(DType::BF16)?;
        let target_f32 = Tensor::from_vec(
            tgt_flat,
            Shape::from_dims(&[cfg.batch, cfg.seq, cfg.out_features]),
            device.clone(),
        )?;
        let target_bf16 = target_f32.to_dtype(DType::BF16)?;

        // Forward + loss + backward.
        AutogradContext::clear();
        let out_bf16 = lora_forward(
            &input_bf16,
            &lora_a,
            &lora_b,
            cfg.alpha,
            cfg.rank,
            cfg.out_features,
        )?;
        let loss = mse_loss_bf16(&out_bf16, &target_bf16)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_vec()?[0];

        let grads = loss.backward()?;

        let g_a = grads
            .get(lora_a.id())
            .expect("no grad for lora_a — autograd path broken")
            .to_dtype(DType::F32)?
            .to_vec()?;
        let g_b = grads
            .get(lora_b.id())
            .expect("no grad for lora_b — autograd path broken")
            .to_dtype(DType::F32)?
            .to_vec()?;
        let our_gA_l2 = l2(&g_a);
        let our_gB_l2 = l2(&g_b);

        // Apply grads + step Adam (mirrors trainer plumbing).
        lora_a.set_grad(
            grads
                .get(lora_a.id())
                .unwrap()
                .clone()
                .to_dtype(DType::F32)?,
        )?;
        lora_b.set_grad(
            grads
                .get(lora_b.id())
                .unwrap()
                .clone()
                .to_dtype(DType::F32)?,
        )?;
        opt.step(&params)?;

        // Compare vs PyTorch reference.
        let r = &fx.per_step[k];
        let dl = (loss_val - r.loss).abs() / r.loss.max(1e-9);
        let dgA = (our_gA_l2 - r.grad_a_l2).abs() / r.grad_a_l2.max(1e-9);
        let dgB = (our_gB_l2 - r.grad_b_l2).abs() / r.grad_b_l2.max(1e-9);

        let bad_loss = dl > loss_rel_tol && r.loss.abs() > 1e-6;
        let bad_a = dgA > grad_l2_rel_tol && r.grad_a_l2 > 1e-9;
        let bad_b = dgB > grad_l2_rel_tol && r.grad_b_l2 > 1e-9;
        if bad_loss || bad_a || bad_b {
            diverged = true;
            report.push_str(&format!(
                "  step {:>2}: loss ours={:.4e} torch={:.4e} (Δ {:+.1}%)  \
                 ||gA|| ours={:.3e} torch={:.3e} (Δ {:+.1}%)  \
                 ||gB|| ours={:.3e} torch={:.3e} (Δ {:+.1}%){}\n",
                k + 1,
                loss_val,
                r.loss,
                dl * 100.0,
                our_gA_l2,
                r.grad_a_l2,
                dgA * 100.0,
                our_gB_l2,
                r.grad_b_l2,
                dgB * 100.0,
                if bad_b { "  <-- gB DIVERGED" } else { "" }
            ));
        } else if matches!(k + 1, 1 | 5 | 10 | 25 | 50) {
            eprintln!(
                "[step {:>2} OK] loss ours={:.4e} torch={:.4e}  ||gA|| ours={:.3e} torch={:.3e}  ||gB|| ours={:.3e} torch={:.3e}",
                k + 1, loss_val, r.loss, our_gA_l2, r.grad_a_l2, our_gB_l2, r.grad_b_l2
            );
        }
    }

    // End-state check.
    let a_after = lora_a.tensor()?.to_vec_f32()?;
    let b_after = lora_b.tensor()?.to_vec_f32()?;
    let ref_a: Vec<f32> = fx.a_final.iter().flatten().copied().collect();
    let ref_b: Vec<f32> = fx.b_final.iter().flatten().copied().collect();
    let da: Vec<f32> = a_after.iter().zip(ref_a.iter()).map(|(a, b)| a - b).collect();
    let db: Vec<f32> = b_after.iter().zip(ref_b.iter()).map(|(a, b)| a - b).collect();
    let a_diff_l2 = l2(&da);
    let b_diff_l2 = l2(&db);
    let a_diff_max = max_abs(&da);
    let b_diff_max = max_abs(&db);
    let a_l2 = l2(&a_after);
    let b_l2 = l2(&b_after);
    eprintln!(
        "[final] A: ours_L2={:.4e}  torch_L2={:.4e}  diff_L2={:.3e}  max_abs={:.3e}",
        a_l2,
        l2(&ref_a),
        a_diff_l2,
        a_diff_max
    );
    eprintln!(
        "[final] B: ours_L2={:.4e}  torch_L2={:.4e}  diff_L2={:.3e}  max_abs={:.3e}",
        b_l2,
        l2(&ref_b),
        b_diff_l2,
        b_diff_max
    );

    if diverged {
        panic!(
            "LoRA closed-loop diverges from torch.optim.AdamW reference\n\
             config: in={} out={} rank={} alpha={} batch={} seq={} steps={}\n\
             lr={} betas=({}, {}) eps={} wd={}\n\
             {}",
            cfg.in_features,
            cfg.out_features,
            cfg.rank,
            cfg.alpha,
            cfg.batch,
            cfg.seq,
            cfg.steps,
            cfg.lr,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            report
        );
    }
    Ok(())
}
