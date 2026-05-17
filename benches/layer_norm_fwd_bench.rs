//! Microbench: LayerNorm forward (BF16) — vec=4 vs legacy smem-tree.
//! Run vec (default):    cargo bench --features cuda --bench layer_norm_fwd_bench
//! Run legacy:           FLAME_LAYER_NORM_FWD_LEGACY=1 cargo bench --features cuda --bench layer_norm_fwd_bench

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

fn bench(name: &str, dims: &[usize]) -> Result<()> {
    let dev = global_cuda_device();
    let last = *dims.last().unwrap();
    let x = rand_bf16(dims, 1, 0.5)?;
    let g = rand_bf16(&[last], 2, 1.0)?;
    let b = rand_bf16(&[last], 3, 0.1)?;

    for _ in 0..50 {
        let _ = black_box(cuda_ops_bf16::layer_norm_bf16(
            &x,
            Some(&g),
            Some(&b),
            1e-5,
        )?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(cuda_ops_bf16::layer_norm_bf16(
            &x,
            Some(&g),
            Some(&b),
            1e-5,
        )?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    println!(
        "  {:<20} dims={:?}  norm={:<5} → {:>7.2} us/iter",
        name, dims, last, per_us
    );
    Ok(())
}

fn main() -> Result<()> {
    let legacy = std::env::var("FLAME_LAYER_NORM_FWD_LEGACY")
        .map(|v| v != "0")
        .unwrap_or(false);
    println!(
        "=== layer_norm forward — mode: {} ===",
        if legacy {
            "LEGACY smem-tree"
        } else {
            "VEC=4 warp-shuffle"
        }
    );
    bench("flux 3072", &[1, 4608, 3072])?;
    bench("zimage 2560", &[1, 4096, 2560])?;
    bench("klein 4096", &[1, 4096, 4096])?;
    bench("small 1280", &[1, 1024, 1280])?;
    Ok(())
}
