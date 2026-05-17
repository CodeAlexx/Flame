//! Microbench: GroupNorm forward stats kernel (BF16) — vec=4 vs legacy smem-tree.
//! Run vec:    cargo bench --features cuda --bench group_norm_fwd_bench
//! Run legacy: FLAME_GROUP_NORM_STATS_LEGACY=1 cargo bench --features cuda --bench group_norm_fwd_bench

#![cfg(feature = "cuda")]

use flame_core::{cuda_ops_bf16, global_cuda_device, DType, FlameError, Result, Shape, Tensor};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 500;

fn rand_bf16(dims: &[usize], seed: u64, std: f32) -> Result<Tensor> {
    let dev = global_cuda_device();
    let _ = flame_core::rng::set_seed(seed);
    Tensor::randn(Shape::from_dims(dims), 0.0, std, dev.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
}

fn bench(name: &str, n: usize, c: usize, h: usize, w: usize, groups: i32) -> Result<()> {
    let dev = global_cuda_device();
    let x = rand_bf16(&[n, c, h, w], 1, 0.5)?;
    let g = rand_bf16(&[c], 2, 1.0)?;
    let b = rand_bf16(&[c], 3, 0.1)?;

    for _ in 0..30 {
        let _ = black_box(cuda_ops_bf16::group_norm_bf16(
            &x,
            Some(&g),
            Some(&b),
            groups,
            1e-5,
        )?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(cuda_ops_bf16::group_norm_bf16(
            &x,
            Some(&g),
            Some(&b),
            groups,
            1e-5,
        )?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    println!(
        "  {:<24} N={} C={} H×W={}×{} groups={}  → {:>7.2} us/iter",
        name, n, c, h, w, groups, per_us
    );
    Ok(())
}

fn main() -> Result<()> {
    let legacy = std::env::var("FLAME_GROUP_NORM_STATS_LEGACY")
        .map(|v| v != "0")
        .unwrap_or(false);
    println!(
        "=== group_norm fwd stats — mode: {} ===",
        if legacy {
            "LEGACY smem-tree"
        } else {
            "VEC=4 warp-shuffle"
        }
    );

    // Typical VAE block sizes
    bench("vae small (128ch 64²)", 1, 128, 64, 64, 32)?;
    bench("vae mid (256ch 64²)", 1, 256, 64, 64, 32)?;
    bench("vae deep (512ch 32²)", 1, 512, 32, 32, 32)?;
    bench("vae very deep (512 64²)", 1, 512, 64, 64, 32)?;
    bench("hi-res (256ch 128²)", 1, 256, 128, 128, 32)?;
    Ok(())
}
