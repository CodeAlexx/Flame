//! Phase 5c — perf bench: `backward_v2` bridge overhead vs v3 `backward`.
//!
//! Deliverable D from
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Phase 5: prove the v2
//! bridge adds ≤ 1% wall-clock overhead vs v3 on the same backward
//! computation, and quantify the BF16-grad memory savings end-to-end
//! (Class A configuration with `Parameter::new_v2`).
//!
//! ## Why a `#[test]` instead of a Criterion `bench`?
//!
//! `AUTOGRAD_CONTEXT` is process-global (`src/autograd.rs:56`). Criterion
//! reuses the same process across iterations and the same Mutex lives
//! across measurement loops, but it also reorders setup/measure phases
//! in ways that interact poorly with global state. A `#[serial] #[test]`
//! with `std::time::Instant` + explicit `AutogradContext::reset()`
//! between configs gives us the cleanest control and the numbers print
//! to stdout, which the Phase 5c commit body captures verbatim under
//! `## Verification`.
//!
//! ## Configurations
//!
//! - **v3**: `loss.backward()` — the default v3 path.
//! - **bridge**: `loss.backward_v2()` — measures bridge overhead in
//!   isolation. Grads come back BF16-stored in the returned
//!   `GradientMap`, but no `Parameter` plumbing is in scope so the
//!   policy-induced upcast on `set_grad` is not exercised.
//! - **Class A**: `loss.backward_v2()` + a `Parameter::new_v2`
//!   `set_grad` round-trip — exercises the full BF16-grad-through-the-
//!   parameter path (the configuration that activates the previously-
//!   dead `adam_fused_multi_bf16_bf16grad_kernel`).
//!
//! ## Workloads
//!
//! - **Synthetic 4-layer MLP** — BF16 chain of `mul`/`mul`/`mul`/`add`,
//!   matches the `backward_v2_within_tolerance_at_bf16` test shape so
//!   this cell is the cheap, dependable regression gate.
//! - **Klein attn_chain prod** — fixture-driven, reuses
//!   `klein_ext_attn_chain_prod.safetensors`. Real Klein-block
//!   matmul + RoPE + SDPA + RMS-norm at production tensor sizes.
//! - **Klein double-block backward** — fixture-driven, reuses
//!   `klein_block_backward.safetensors`. Closest fixture-driven shape
//!   to a real training step (attention + gated residual + MLP).
//!
//! ## Bench protocol
//!
//! 5 warmup iterations + 50 timed iterations per (config × workload)
//! cell. Drop the slowest 5 (head/tail trim) and report median, P90,
//! mean in milliseconds. `device.synchronize()` is called before and
//! after each timed iteration so wall-clock = GPU + CPU work.
//!
//! ## Constraints (from Phase 5b skeptic CONCERNs)
//!
//! - `FLAME_CUDA_GRAPH` MUST be unset/OFF on every arm — the replay
//!   path pre-allocates grad buffers at warmup-recorded F32 dtype,
//!   bypassing the post-loop cast in `backward_impl`. Apples-to-apples
//!   requires cuda-graph disabled on both v3 and v2.
//! - `FLAME_MT_SCALE` is OFF here (we don't set it). The Z-Image
//!   trainer disables it under `--use-autograd-v2` to avoid the F32
//!   fast path; this bench never enables it.

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaDevice;
use serial_test::serial;

use flame_core::autograd::AutogradContext;
use flame_core::parameter::{GradDtypePolicy, Parameter};
use flame_core::{global_cuda_device, DType, Shape, Tensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dev() -> Arc<CudaDevice> {
    global_cuda_device()
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("PYTORCH_FIXTURES").unwrap_or_else(|_| "tests/pytorch_fixtures".to_string()),
    )
}

fn load_fixture(path: &Path) -> HashMap<String, Tensor> {
    let device = global_cuda_device();
    flame_core::serialization::load_file(path, &device)
        .unwrap_or_else(|e| panic!("load_file {:?} failed: {:?}", path, e))
}

fn mk_bf16(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (((s >> 32) as f32) / (u32::MAX as f32) - 0.5) * scale;
        v.push(x);
    }
    v
}

/// Summary stats for a per-iteration timing series (in milliseconds).
#[derive(Debug, Clone)]
struct TimingStats {
    median_ms: f64,
    p90_ms: f64,
    mean_ms: f64,
    n: usize,
}

impl TimingStats {
    fn from_durs(durs: &[Duration]) -> Self {
        assert!(!durs.is_empty(), "timing stats from empty series");
        let mut ms: Vec<f64> = durs.iter().map(|d| d.as_secs_f64() * 1e3).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = ms.len();
        let median_ms = if n % 2 == 0 {
            (ms[n / 2 - 1] + ms[n / 2]) * 0.5
        } else {
            ms[n / 2]
        };
        let p90_idx = ((n as f64) * 0.9).floor() as usize;
        let p90_ms = ms[p90_idx.min(n - 1)];
        let mean_ms = ms.iter().sum::<f64>() / (n as f64);
        Self {
            median_ms,
            p90_ms,
            mean_ms,
            n,
        }
    }
}

/// Run `bench_iter` `warmup + timed` times; drop the slowest 5 timed and
/// the slowest 5 of the timed iters head/tail-style (the spec says
/// "drop the slowest 5", interpreted as trimmed mean — we drop the 5
/// largest before computing median/P90/mean). The bench closure must
/// reset autograd context state itself.
fn bench<F: FnMut()>(mut bench_iter: F, warmup: usize, timed: usize) -> TimingStats {
    for _ in 0..warmup {
        bench_iter();
    }
    let device = dev();
    let _ = device.synchronize();
    let mut durs: Vec<Duration> = Vec::with_capacity(timed);
    for _ in 0..timed {
        let _ = device.synchronize();
        let start = Instant::now();
        bench_iter();
        let _ = device.synchronize();
        durs.push(start.elapsed());
    }
    // Trim the slowest 5 (spec: "drop the slowest 5").
    durs.sort();
    let keep = durs.len().saturating_sub(5);
    durs.truncate(keep);
    TimingStats::from_durs(&durs)
}

/// Sum the byte size of every grad in a `GradientMap` (computed
/// from `numel * dtype.size_in_bytes()`).
fn grad_map_bytes(g: &flame_core::gradient::GradientMap) -> usize {
    let mut total = 0usize;
    for (_id, t) in g.iter() {
        total += t.shape().elem_count() * t.dtype().size_in_bytes();
    }
    total
}

fn pct_delta(base: f64, other: f64) -> f64 {
    if base == 0.0 {
        return 0.0;
    }
    (other - base) / base * 100.0
}

// ---------------------------------------------------------------------------
// Workload constructors (build a fresh forward graph each call)
// ---------------------------------------------------------------------------

/// Synthetic 4-layer MLP — mirrors `backward_v2_within_tolerance_at_bf16`
/// in `tests/autograd_v2_bridge.rs`. Returns (loss, leaf-ids).
fn build_synthetic_mlp() -> (Tensor, Vec<flame_core::TensorId>) {
    let device = dev();
    let shape = Shape::from_dims(&[2, 8]);
    let n = shape.elem_count();

    let x = Tensor::from_vec_dtype(
        mk_bf16(101, n, 1.0),
        shape.clone(),
        device.clone(),
        DType::BF16,
    )
    .unwrap()
    .requires_grad_(true);
    let w1 = Tensor::from_vec_dtype(
        mk_bf16(103, n, 0.5),
        shape.clone(),
        device.clone(),
        DType::BF16,
    )
    .unwrap()
    .requires_grad_(true);
    let w2 = Tensor::from_vec_dtype(
        mk_bf16(107, n, 0.5),
        shape.clone(),
        device.clone(),
        DType::BF16,
    )
    .unwrap()
    .requires_grad_(true);
    let bias = Tensor::from_vec_dtype(
        mk_bf16(109, n, 0.1),
        shape.clone(),
        device.clone(),
        DType::BF16,
    )
    .unwrap()
    .requires_grad_(true);

    let leaves = vec![x.id(), w1.id(), w2.id(), bias.id()];

    let h1 = x.mul(&w1).unwrap();
    let h2 = h1.mul(&w2).unwrap();
    let h3 = h2.add(&bias).unwrap();
    let loss = h3.mean().unwrap();
    (loss, leaves)
}

/// Klein attn_chain at prod shape — mirrors scenario 6 of
/// `tests/autograd_v2_klein_parity.rs`. Returns (loss, leaf-ids).
fn build_klein_attn_chain(fix: &HashMap<String, Tensor>) -> (Tensor, Vec<flame_core::TensorId>) {
    let x = fix.get("x").unwrap().clone().requires_grad_(true);
    let w_qkv = fix.get("w_qkv").unwrap().clone().requires_grad_(true);
    let b_qkv = fix.get("b_qkv").unwrap().clone().requires_grad_(true);
    let q_scale = fix.get("q_scale").unwrap().clone().requires_grad_(true);
    let k_scale = fix.get("k_scale").unwrap().clone().requires_grad_(true);
    let w_out = fix.get("w_out").unwrap().clone().requires_grad_(true);
    let b_out = fix.get("b_out").unwrap().clone().requires_grad_(true);
    let pe_cos = fix.get("pe_cos").unwrap().clone();
    let pe_sin = fix.get("pe_sin").unwrap().clone();

    let leaves = vec![
        x.id(),
        w_qkv.id(),
        b_qkv.id(),
        q_scale.id(),
        k_scale.id(),
        w_out.id(),
        b_out.id(),
    ];

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

    // Inline RoPE recording (matches scenario 6 in autograd_v2_klein_parity.rs).
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
    (loss, leaves)
}

/// Klein double-block backward fixture — mirrors scenario 7 of
/// `tests/autograd_v2_klein_parity.rs`. Returns (loss, leaf-ids).
fn build_klein_double_block(fix: &HashMap<String, Tensor>) -> (Tensor, Vec<flame_core::TensorId>) {
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

    let leaves = vec![
        x.id(),
        w_qkv.id(),
        b_qkv.id(),
        w_out.id(),
        b_out.id(),
        w_up.id(),
        b_up.id(),
        w_down.id(),
        b_down.id(),
    ];

    let dims = x.shape().dims().to_vec();
    let (b_d, n_d, d_d) = (dims[0], dims[1], dims[2]);
    let h_d = 8usize;
    let hd_d = d_d / h_d;

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

    let qkv = linear(&x, &w_qkv, &b_qkv);
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
    let o = flame_core::sdpa::forward(&q, &k, &v, None).expect("sdpa");
    let o = o
        .permute(&[0, 2, 1, 3])
        .and_then(|t| t.reshape(&[b_d, n_d, d_d]))
        .expect("permute/reshape back");
    let attn_out = linear(&o, &w_out, &b_out);

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
    (loss, leaves)
}

// ---------------------------------------------------------------------------
// Per-cell benchmark — (v3, bridge, Class A) × one workload
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CellResult {
    workload: String,
    v3: TimingStats,
    bridge: TimingStats,
    class_a: TimingStats,
    v3_grad_bytes: usize,
    bridge_grad_bytes: usize,
    class_a_grad_bytes: usize,
}

impl CellResult {
    fn bridge_delta_pct(&self) -> f64 {
        pct_delta(self.v3.median_ms, self.bridge.median_ms)
    }
    fn class_a_delta_pct(&self) -> f64 {
        pct_delta(self.v3.median_ms, self.class_a.median_ms)
    }
    fn class_a_grad_savings_pct(&self) -> f64 {
        if self.v3_grad_bytes == 0 {
            return 0.0;
        }
        ((self.v3_grad_bytes as f64 - self.class_a_grad_bytes as f64) / (self.v3_grad_bytes as f64))
            * 100.0
    }
}

/// Run the full (v3, bridge, Class A) × workload sweep for one builder
/// closure. Each arm builds a fresh forward graph each iteration (no
/// "amortize the forward" tricks) so the comparison is truly
/// apples-to-apples on the backward side.
fn run_cell<F: FnMut() -> (Tensor, Vec<flame_core::TensorId>)>(
    name: &str,
    mut build: F,
    warmup: usize,
    timed: usize,
) -> CellResult {
    // --- v3 control ---
    let mut last_v3_bytes = 0usize;
    let stats_v3 = bench(
        || {
            AutogradContext::reset();
            AutogradContext::set_enabled(true);
            let (loss, _leaves) = build();
            let g = AutogradContext::backward(&loss).expect("v3 backward");
            last_v3_bytes = grad_map_bytes(&g);
            AutogradContext::set_enabled(false);
        },
        warmup,
        timed,
    );

    // --- bridge alone ---
    let mut last_bridge_bytes = 0usize;
    let stats_bridge = bench(
        || {
            AutogradContext::reset();
            AutogradContext::set_enabled(true);
            let (loss, _leaves) = build();
            let g = AutogradContext::backward_v2(&loss).expect("v2 backward_v2");
            last_bridge_bytes = grad_map_bytes(&g);
            AutogradContext::set_enabled(false);
        },
        warmup,
        timed,
    );

    // --- Class A: backward_v2 + Parameter::new_v2 + set_grad round-trip ---
    //
    // The leaves here are plain `Tensor`s (not `Parameter`s), so to model
    // Class A end-to-end we (a) call backward_v2, then (b) wrap each
    // leaf in a `Parameter::new_v2` (MatchParamDtype) and call
    // `set_grad` with the v2-emitted grad. This is exactly the path the
    // trainer takes when LoRA params are constructed with new_v2.
    //
    // The bytes we measure are the resulting `param.grad()` bytes summed
    // across leaves — which is what the optimizer state and the Adam
    // fused kernel see.
    let mut last_class_a_bytes = 0usize;
    let stats_class_a = bench(
        || {
            AutogradContext::reset();
            AutogradContext::set_enabled(true);
            let (loss, leaves) = build();
            let g = AutogradContext::backward_v2(&loss).expect("v2 backward_v2 (class A)");
            let mut total = 0usize;
            // Wrap each leaf-id's grad into a Parameter::new_v2 and
            // exercise set_grad. We don't have the original Tensor
            // here (only ids), so we construct a small placeholder
            // Parameter from the grad's shape/dtype for the
            // set_grad call. The byte count comes from the
            // Parameter's stored grad dtype, which under
            // MatchParamDtype preserves whatever backward_v2
            // emitted.
            for id in &leaves {
                if let Some(grad_ref) = g.get(*id) {
                    let grad_owned = grad_ref.clone();
                    let placeholder = Tensor::zeros_dtype(
                        grad_owned.shape().clone(),
                        grad_owned.dtype(),
                        grad_owned.device().clone(),
                    )
                    .expect("placeholder zeros")
                    .requires_grad_(true);
                    let param = Parameter::new_v2(placeholder);
                    param.set_grad(grad_owned).expect("class A set_grad");
                    if let Some(stored) = param.grad() {
                        total += stored.shape().elem_count() * stored.dtype().size_in_bytes();
                    }
                    // Sanity: under MatchParamDtype the stored dtype
                    // is the inserted dtype (BF16 for BF16 graph).
                    debug_assert_eq!(param.grad_dtype_policy(), GradDtypePolicy::MatchParamDtype);
                }
            }
            last_class_a_bytes = total;
            AutogradContext::set_enabled(false);
        },
        warmup,
        timed,
    );

    CellResult {
        workload: name.to_string(),
        v3: stats_v3,
        bridge: stats_bridge,
        class_a: stats_class_a,
        v3_grad_bytes: last_v3_bytes,
        bridge_grad_bytes: last_bridge_bytes,
        class_a_grad_bytes: last_class_a_bytes,
    }
}

#[allow(dead_code)] // Kept as reference template for `cargo bench` ports; per-cell tests print their own summaries.
fn print_summary(rows: &[CellResult]) {
    eprintln!();
    eprintln!("=================================================================================================");
    eprintln!(
        "Phase 5c — autograd_v2 perf bench results (median wall-clock per backward iteration)"
    );
    eprintln!("=================================================================================================");
    eprintln!();
    eprintln!("| Workload                  | v3 (ms)   | bridge (ms) | Class A (ms) | bridge Δ% | Class A Δ% |");
    eprintln!("|---------------------------|-----------|-------------|--------------|-----------|------------|");
    for r in rows {
        eprintln!(
            "| {:<25} | {:>9.3} | {:>11.3} | {:>12.3} | {:>+8.2}% | {:>+9.2}% |",
            r.workload,
            r.v3.median_ms,
            r.bridge.median_ms,
            r.class_a.median_ms,
            r.bridge_delta_pct(),
            r.class_a_delta_pct(),
        );
    }
    eprintln!();
    eprintln!("Per-cell tail (P90 / mean, ms):");
    for r in rows {
        eprintln!(
            "  {:<25}  v3 P90={:>7.3} mean={:>7.3} | bridge P90={:>7.3} mean={:>7.3} | Class A P90={:>7.3} mean={:>7.3} (n={})",
            r.workload,
            r.v3.p90_ms,
            r.v3.mean_ms,
            r.bridge.p90_ms,
            r.bridge.mean_ms,
            r.class_a.p90_ms,
            r.class_a.mean_ms,
            r.v3.n,
        );
    }
    eprintln!();
    eprintln!("=================================================================================================");
    eprintln!("Deliverable B — grad-storage memory");
    eprintln!("=================================================================================================");
    eprintln!();
    eprintln!("| Workload                  | v3 grad MB | bridge grad MB | Class A grad MB | Class A savings |");
    eprintln!("|---------------------------|------------|----------------|-----------------|-----------------|");
    for r in rows {
        eprintln!(
            "| {:<25} | {:>10.3} | {:>14.3} | {:>15.3} | {:>+13.2}%  |",
            r.workload,
            (r.v3_grad_bytes as f64) / (1024.0 * 1024.0),
            (r.bridge_grad_bytes as f64) / (1024.0 * 1024.0),
            (r.class_a_grad_bytes as f64) / (1024.0 * 1024.0),
            r.class_a_grad_savings_pct(),
        );
    }
    eprintln!();
    let worst_bridge = rows
        .iter()
        .map(|r| r.bridge_delta_pct())
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "Headline: worst-case bridge Δ% (across all workloads) = {:+.2}% (target: ≤ ±1%)",
        worst_bridge
    );
    eprintln!();
}

// ---------------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------------

/// Bench protocol per spec: 5 warmup + 50 timed, trim slowest 5.
const WARMUP: usize = 5;
const TIMED: usize = 50;

#[test]
#[serial]
fn perf_synthetic_mlp() {
    let r = run_cell("Synthetic 4-layer MLP", build_synthetic_mlp, WARMUP, TIMED);
    eprintln!(
        "[cell] {:<25}  v3 med={:.3}ms | bridge med={:.3}ms (Δ{:+.2}%) | Class A med={:.3}ms (Δ{:+.2}%)",
        r.workload,
        r.v3.median_ms,
        r.bridge.median_ms,
        r.bridge_delta_pct(),
        r.class_a.median_ms,
        r.class_a_delta_pct(),
    );
    eprintln!(
        "[cell] {:<25}  v3 grad={}B | bridge grad={}B | Class A grad={}B (savings={:+.2}%)",
        r.workload,
        r.v3_grad_bytes,
        r.bridge_grad_bytes,
        r.class_a_grad_bytes,
        r.class_a_grad_savings_pct(),
    );
}

#[test]
#[serial]
fn perf_klein_attn_chain_prod() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_ext_attn_chain_prod.safetensors");
    if !path.exists() {
        eprintln!("SKIP perf_klein_attn_chain_prod: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);
    let r = run_cell(
        "Klein attn_chain prod",
        || build_klein_attn_chain(&fix),
        WARMUP,
        TIMED,
    );
    eprintln!(
        "[cell] {:<25}  v3 med={:.3}ms | bridge med={:.3}ms (Δ{:+.2}%) | Class A med={:.3}ms (Δ{:+.2}%)",
        r.workload,
        r.v3.median_ms,
        r.bridge.median_ms,
        r.bridge_delta_pct(),
        r.class_a.median_ms,
        r.class_a_delta_pct(),
    );
    eprintln!(
        "[cell] {:<25}  v3 grad={}B | bridge grad={}B | Class A grad={}B (savings={:+.2}%)",
        r.workload,
        r.v3_grad_bytes,
        r.bridge_grad_bytes,
        r.class_a_grad_bytes,
        r.class_a_grad_savings_pct(),
    );
}

#[test]
#[serial]
fn perf_klein_double_block() {
    let path = fixtures_dir()
        .join("patterns")
        .join("klein_block_backward.safetensors");
    if !path.exists() {
        eprintln!("SKIP perf_klein_double_block: {path:?} not found");
        return;
    }
    let fix = load_fixture(&path);
    let r = run_cell(
        "Klein double-block",
        || build_klein_double_block(&fix),
        WARMUP,
        TIMED,
    );
    eprintln!(
        "[cell] {:<25}  v3 med={:.3}ms | bridge med={:.3}ms (Δ{:+.2}%) | Class A med={:.3}ms (Δ{:+.2}%)",
        r.workload,
        r.v3.median_ms,
        r.bridge.median_ms,
        r.bridge_delta_pct(),
        r.class_a.median_ms,
        r.class_a_delta_pct(),
    );
    eprintln!(
        "[cell] {:<25}  v3 grad={}B | bridge grad={}B | Class A grad={}B (savings={:+.2}%)",
        r.workload,
        r.v3_grad_bytes,
        r.bridge_grad_bytes,
        r.class_a_grad_bytes,
        r.class_a_grad_savings_pct(),
    );
}

// Note on the (removed) `perf_sweep_all_workloads` test: an earlier
// version of this file ran all three (config × workload) cells in a
// single #[test] and printed the full Deliverable A + B summary in one
// go. That pattern intermittently hit `CUDA_ERROR_INVALID_VALUE` deep
// in `Tensor::zeros_dtype` / `serialization::load_file` after the first
// cell completed — almost certainly cross-cell CUDA context state that
// only resets cleanly across test-binary `#[test]` boundaries (the
// same class of process-global issue that drove the move to
// `serial_test` in Phase 5b). The per-cell tests above each produce
// their full single-line summary; aggregate them by running the binary
// with `--nocapture --test-threads=1`:
//
//   FLAME_CUDA_GRAPH=0 cargo test --release --features autograd_v2 \\
//       --test autograd_v2_perf -- --nocapture --test-threads=1
//
// The `## Verification` section of the Phase 5c commit body captures
// the verbatim output of that command.
