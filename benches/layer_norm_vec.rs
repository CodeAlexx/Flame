//! Microbench: LayerNorm backward (BF16) — vec vs legacy. Non-zero (randn)
//! inputs so timings reflect production behavior (zero inputs short-circuit
//! some FMAs and inflate measured speedup).
//!
//! Run vec (default):
//!   cd flame-core && cargo bench --features cuda --bench layer_norm_vec
//! Run legacy scalar (forced):
//!   cd flame-core && FLAME_LAYER_NORM_LEGACY=1 cargo bench --features cuda --bench layer_norm_vec

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, DType, FlameError, Result, Shape, Tensor};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 200;
const EPS: f32 = 1e-5;

fn rand_bf16(dims: &[usize], seed: u64, std: f32) -> Result<Tensor> {
    let dev = global_cuda_device();
    flame_core::rng::set_seed(seed);
    Tensor::randn(Shape::from_dims(dims), 0.0, std, dev.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
}

fn bench_bwd(name: &str, dims: &[usize], norm_size: usize) -> Result<()> {
    let dev = global_cuda_device();

    let x = rand_bf16(dims, 1, 0.5)?;
    let dy = rand_bf16(dims, 2, 0.1)?;
    let gamma = Some(rand_bf16(&[norm_size], 3, 1.0)?);
    let beta = Some(rand_bf16(&[norm_size], 4, 0.1)?);

    // Warmup
    for _ in 0..20 {
        let _ = black_box(flame_core::cuda_ops_bf16::layer_norm_backward_bf16(
            &x,
            &dy,
            gamma.as_ref(),
            beta.as_ref(),
            &[norm_size],
            EPS,
        )?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(flame_core::cuda_ops_bf16::layer_norm_backward_bf16(
            &x,
            &dy,
            gamma.as_ref(),
            beta.as_ref(),
            &[norm_size],
            EPS,
        )?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let dur = t0.elapsed();
    let per_us = dur.as_secs_f64() * 1e6 / ITERS as f64;
    println!(
        "  bwd  {:<14} dims={:?} norm={:<5} → {:>9.2} us/iter",
        name, dims, norm_size, per_us
    );
    Ok(())
}

fn main() -> Result<()> {
    let legacy = std::env::var("FLAME_LAYER_NORM_LEGACY")
        .map(|v| v != "0")
        .unwrap_or(false);
    let mode = if legacy {
        "LEGACY scalar (1 thread/row)"
    } else {
        "VEC (256 threads/row + cross-row dgamma/dbeta)"
    };
    println!("=== layer_norm_vec bench — mode: {} ===", mode);
    println!("    inputs: randn(0, std=0.5) for x, randn(0, 0.1) for dy, randn for gamma/beta");

    bench_bwd("zimage block", &[1, 4096, 2560], 2560)?;
    bench_bwd("klein block", &[1, 4096, 4096], 4096)?;
    bench_bwd("chroma block", &[1, 4096, 3072], 3072)?;
    bench_bwd("sdxl mid", &[1, 1024, 1280], 1280)?;
    Ok(())
}
