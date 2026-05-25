//! Stacked LoRA-chain training parity vs PyTorch.
//!
//! 30 layers of (frozen BF16 base Linear + trainable LoRA delta) stacked
//! in series, mirroring Z-Image's 30 transformer blocks. If our `lora_A`
//! grad amplifies vs PyTorch's in this chain (but not in the single-layer
//! `lora_closed_loop_parity` test that already passes), the bug is in
//! flame-core's handling of long autograd chains under LoRA pressure.
//!
//! Setup and config match `tests/parity/lora_chain_ref.py`.

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
    n_layers: usize,
    steps: usize,
}

#[derive(serde::Deserialize)]
struct PerStep {
    loss: f32,
    grad_a_l2_per_layer: Vec<f32>,
    grad_b_l2_per_layer: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    config: Config,
    base_weights: Vec<Vec<Vec<f32>>>,         // [n_layers][out][in]
    a_inits: Vec<Vec<Vec<f32>>>,              // [n_layers][rank][in]
    b_inits: Vec<Vec<Vec<f32>>>,              // [n_layers][out][rank]
    inputs_bf16_as_f32: Vec<Vec<Vec<Vec<f32>>>>,  // [steps][batch][seq][in]
    targets_bf16_as_f32: Vec<Vec<Vec<Vec<f32>>>>, // [steps][batch][seq][out]
    per_step: Vec<PerStep>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("lora_chain_ref.json")
}

fn flatten<T: Copy>(x: &[Vec<T>]) -> Vec<T> {
    x.iter().flatten().copied().collect()
}
fn flatten3<T: Copy>(x: &[Vec<Vec<T>>]) -> Vec<T> {
    x.iter().flatten().flatten().copied().collect()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

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
fn lora_chain_matches_pytorch() -> Result<()> {
    let path = fixture_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via \
             `python3 tests/parity/lora_chain_ref.py`",
            path.display(),
            e
        )
    });
    let fx: Fixture = serde_json::from_slice(&bytes).expect("malformed lora_chain_ref.json");
    let cfg = &fx.config;
    let device = global_cuda_device();

    // Build base weights as plain BF16 (no requires_grad).
    let mut base_w: Vec<Tensor> = Vec::with_capacity(cfg.n_layers);
    for li in 0..cfg.n_layers {
        let w_f32: Vec<f32> = flatten(&fx.base_weights[li]);
        let t = Tensor::from_vec(
            w_f32,
            Shape::from_dims(&[cfg.out_features, cfg.in_features]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        base_w.push(t);
    }

    // Build trainable LoRA params per layer.
    let mut a_params: Vec<Parameter> = Vec::with_capacity(cfg.n_layers);
    let mut b_params: Vec<Parameter> = Vec::with_capacity(cfg.n_layers);
    for li in 0..cfg.n_layers {
        let a_flat: Vec<f32> = flatten(&fx.a_inits[li]);
        let b_flat: Vec<f32> = flatten(&fx.b_inits[li]);
        let a = Tensor::from_vec(
            a_flat,
            Shape::from_dims(&[cfg.rank, cfg.in_features]),
            device.clone(),
        )?
        .requires_grad_(true);
        let b = Tensor::from_vec(
            b_flat,
            Shape::from_dims(&[cfg.out_features, cfg.rank]),
            device.clone(),
        )?
        .requires_grad_(true);
        a_params.push(Parameter::new(a));
        b_params.push(Parameter::new(b));
    }
    let mut all_params: Vec<Parameter> = a_params.iter().chain(b_params.iter()).cloned().collect();
    let _ = &mut all_params;

    let mut opt = AdamW::new(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);

    // Tolerance: 15% rel grad-L2 per layer (BF16 chain + 30 layers adds noise).
    let tol: f32 = 0.15;
    let mut report = String::new();
    let mut diverged = false;

    for k in 0..cfg.steps {
        let in_flat = flatten3(&fx.inputs_bf16_as_f32[k]);
        let tgt_flat = flatten3(&fx.targets_bf16_as_f32[k]);
        let input = Tensor::from_vec(
            in_flat,
            Shape::from_dims(&[cfg.batch, cfg.seq, cfg.in_features]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let target = Tensor::from_vec(
            tgt_flat,
            Shape::from_dims(&[cfg.batch, cfg.seq, cfg.out_features]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        AutogradContext::clear();

        // Chain forward: y = base(x) + lora(x), repeated n_layers times.
        let mut x = input;
        for li in 0..cfg.n_layers {
            // base(x) = x @ base_w^T — keep in BF16. matmul handles 3D x 2D.
            let leading = cfg.batch * cfg.seq;
            let x_2d = x.reshape(&[leading, cfg.in_features])?;
            let w_t = base_w[li].transpose()?;
            let base_out_2d = x_2d.matmul(&w_t)?;
            let base_out =
                base_out_2d.reshape(&[cfg.batch, cfg.seq, cfg.out_features])?;
            let lora_out = lora_forward(
                &x,
                &a_params[li],
                &b_params[li],
                cfg.alpha,
                cfg.rank,
                cfg.out_features,
            )?;
            x = base_out.add(&lora_out)?;
        }

        let loss = mse_loss_bf16(&x, &target)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_vec()?[0];
        let grads = loss.backward()?;

        let mut our_ga_per_layer: Vec<f32> = Vec::with_capacity(cfg.n_layers);
        let mut our_gb_per_layer: Vec<f32> = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let ga = grads
                .get(a_params[li].id())
                .unwrap_or_else(|| panic!("no grad for lora_a layer {}", li))
                .to_dtype(DType::F32)?
                .to_vec()?;
            let gb = grads
                .get(b_params[li].id())
                .unwrap_or_else(|| panic!("no grad for lora_b layer {}", li))
                .to_dtype(DType::F32)?
                .to_vec()?;
            our_ga_per_layer.push(l2(&ga));
            our_gb_per_layer.push(l2(&gb));

            // Push grad to param + step Adam later (set_grad).
            a_params[li].set_grad(grads.get(a_params[li].id()).unwrap().clone().to_dtype(DType::F32)?)?;
            b_params[li].set_grad(grads.get(b_params[li].id()).unwrap().clone().to_dtype(DType::F32)?)?;
        }
        opt.step(&all_params)?;

        // Compare per-layer.
        let r = &fx.per_step[k];
        let our_sum_a: f32 = our_ga_per_layer.iter().sum();
        let our_sum_b: f32 = our_gb_per_layer.iter().sum();
        let ref_sum_a: f32 = r.grad_a_l2_per_layer.iter().sum();
        let ref_sum_b: f32 = r.grad_b_l2_per_layer.iter().sum();

        // Highlight steps so trends are visible.
        if matches!(k + 1, 1 | 5 | 10 | 25 | 50) {
            eprintln!(
                "[step {:>2}] loss ours={:.4e} torch={:.4e}  \
                 sumA ours={:.3e} torch={:.3e} ({:+.0}%)  \
                 sumB ours={:.3e} torch={:.3e} ({:+.0}%)",
                k + 1,
                loss_val,
                r.loss,
                our_sum_a,
                ref_sum_a,
                if ref_sum_a > 1e-9 { 100.0 * (our_sum_a / ref_sum_a - 1.0) } else { 0.0 },
                our_sum_b,
                ref_sum_b,
                if ref_sum_b > 1e-9 { 100.0 * (our_sum_b / ref_sum_b - 1.0) } else { 0.0 },
            );
        }

        // Per-layer divergence check.
        for li in 0..cfg.n_layers {
            for (label, ours, theirs) in [
                ("A", our_ga_per_layer[li], r.grad_a_l2_per_layer[li]),
                ("B", our_gb_per_layer[li], r.grad_b_l2_per_layer[li]),
            ] {
                if theirs < 1e-9 {
                    continue; // both should be near-zero (e.g. A at step 1)
                }
                let rel = (ours - theirs).abs() / theirs;
                if rel > tol {
                    diverged = true;
                    if report.len() < 4000 {
                        report.push_str(&format!(
                            "  step {:>2} layer {:>2} {}: ours={:.3e} torch={:.3e} rel={:+.1}%\n",
                            k + 1,
                            li,
                            label,
                            ours,
                            theirs,
                            rel * 100.0
                        ));
                    }
                }
            }
        }
    }

    if diverged {
        panic!(
            "LoRA chain (30 layers, 50 steps) diverges from PyTorch reference:\n\
             config: in={} out={} rank={} alpha={} n_layers={} steps={}\n\
             worst divergences (first ~30 lines):\n{}",
            cfg.in_features,
            cfg.out_features,
            cfg.rank,
            cfg.alpha,
            cfg.n_layers,
            cfg.steps,
            report
        );
    }
    Ok(())
}
