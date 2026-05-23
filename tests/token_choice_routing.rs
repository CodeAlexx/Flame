//! Integration tests for `flame_core::ops::token_choice_routing`.
//!
//! These exercise the public API surface — making sure the routing plan
//! and the permute helper compose with `grouped_mm_bf16` and
//! `fused_gated_scatter_add_bf16` the way model code will use them.

use cudarc::driver::CudaDevice;
use flame_core::ops::token_choice_routing::{
    permute_tokens_for_token_choice, token_choice_route, ScoreMode,
};
use flame_core::{DType, Error, Result, Shape, Tensor};

#[test]
fn plan_offsets_match_pytorch_recipe() -> Result<()> {
    // Mirror transformers' GPT-OSS routing: topk -> softmax over K.
    // T=2, E=4, K=2.
    let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let logits = vec![
        0.0_f32, 1.0, 0.5, -1.0, //
        2.0, 0.0, -1.0, 1.0,
    ];
    let logits_t = Tensor::from_vec_dtype(
        logits,
        Shape::from_dims(&[2, 4]),
        device.clone(),
        DType::F32,
    )?;
    let plan = token_choice_route(&logits_t, 2, ScoreMode::TopKSoftmax)?;
    assert_eq!(plan.offsets, vec![0, 1, 2, 3, 4]);
    assert_eq!(plan.offsets_for_grouped_mm(), vec![1, 2, 3, 4]);
    assert_eq!(plan.permuted_token_indices, vec![1, 0, 0, 1]);
    Ok(())
}

#[test]
fn plan_top_k_softmax_weights_sum_to_one_per_token() -> Result<()> {
    let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let t = 7;
    let e = 32; // matches GPT-OSS num_local_experts
    let k = 4; // matches GPT-OSS num_experts_per_tok

    let mut state = 0x4242_4242_u64;
    let mut next = || -> f32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32) * 4.0 - 2.0
    };
    let logits: Vec<f32> = (0..t * e).map(|_| next()).collect();
    let logits_t = Tensor::from_vec_dtype(
        logits,
        Shape::from_dims(&[t, e]),
        device,
        DType::F32,
    )?;
    let plan = token_choice_route(&logits_t, k, ScoreMode::TopKSoftmax)?;

    let w = plan.expert_weights.to_vec()?;
    for ti in 0..t {
        let row_sum: f32 = w[ti * k..(ti + 1) * k].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "token {ti} row sum {row_sum} != 1.0"
        );
    }

    // Expert-major bookkeeping invariants.
    assert_eq!(plan.offsets.len(), e + 1);
    assert_eq!(plan.offsets[0], 0);
    assert_eq!(plan.offsets[e] as usize, t * k);
    assert_eq!(plan.permuted_token_indices.len(), t * k);
    assert_eq!(plan.expert_weights_flat.len(), t * k);

    // Every source token appears exactly K times.
    let mut hits = vec![0u32; t];
    for &v in &plan.permuted_token_indices {
        hits[v as usize] += 1;
    }
    for (ti, &h) in hits.iter().enumerate() {
        assert_eq!(h as usize, k, "token {ti} hits {h} != K");
    }
    Ok(())
}

#[test]
fn permute_helper_matches_routing_plan() -> Result<()> {
    let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let t = 4;
    let e = 4;
    let k = 2;
    let d = 3;

    let mut logits = vec![-1.0f32; t * e];
    for ti in 0..t {
        logits[ti * e + ti] = 2.0;
        logits[ti * e + (ti + 1) % e] = 1.0;
    }
    let logits_t = Tensor::from_vec_dtype(
        logits,
        Shape::from_dims(&[t, e]),
        device.clone(),
        DType::F32,
    )?;
    let plan = token_choice_route(&logits_t, k, ScoreMode::TopKSoftmax)?;

    let x_data: Vec<f32> = (0..t * d)
        .map(|i| (i / d) as f32 * 10.0 + (i % d) as f32)
        .collect();
    let x = Tensor::from_vec_dtype(
        x_data,
        Shape::from_dims(&[t, d]),
        device,
        DType::F32,
    )?;
    let permuted = permute_tokens_for_token_choice(&x, &plan)?;
    let got = permuted.to_vec()?;

    // Verify every output row matches its source token row.
    assert_eq!(got.len(), t * k * d);
    for (row, &src) in plan.permuted_token_indices.iter().enumerate() {
        for col in 0..d {
            let expected = src as f32 * 10.0 + col as f32;
            let v = got[row * d + col];
            assert!(
                (v - expected).abs() < 1e-5,
                "row {row} col {col} src {src}: got {v}, expected {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn score_mode_softmax_renorm_full_k_recovers_softmax() -> Result<()> {
    // K = E + SoftmaxRenorm reduces to the full softmax distribution.
    let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let t = 2;
    let e = 4;
    let logits = vec![0.0_f32, 1.0, 0.5, -1.0, 2.0, 0.0, -1.0, 1.0];
    let logits_t = Tensor::from_vec_dtype(
        logits.clone(),
        Shape::from_dims(&[t, e]),
        device,
        DType::F32,
    )?;
    let plan = token_choice_route(&logits_t, e, ScoreMode::SoftmaxRenorm)?;
    let w = plan.expert_weights.to_vec()?;
    let idx = plan.expert_indices.to_dtype(DType::F32)?.to_vec()?;
    for ti in 0..t {
        // Rebuild natural-order weights.
        let mut natural = vec![0.0f32; e];
        for k in 0..e {
            natural[idx[ti * e + k] as usize] = w[ti * e + k];
        }
        // Hand-computed softmax over E.
        let row = &logits[ti * e..(ti + 1) * e];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sumexp = 0.0;
        let mut probs = vec![0.0; e];
        for ei in 0..e {
            let v = (row[ei] - m).exp();
            probs[ei] = v;
            sumexp += v;
        }
        for v in probs.iter_mut() {
            *v /= sumexp;
        }
        for ei in 0..e {
            assert!(
                (natural[ei] - probs[ei]).abs() < 1e-5,
                "token {ti} expert {ei}: got {} expected {}",
                natural[ei],
                probs[ei]
            );
        }
    }
    Ok(())
}
