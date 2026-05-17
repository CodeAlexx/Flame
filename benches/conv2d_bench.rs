//! Conv2d bench post algo-cache. Same shapes inference-flame bench used.

#![cfg(feature = "cuda")]

use flame_core::conv::Conv2d;
use flame_core::{global_cuda_device, FlameError, Result};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 200;

fn run_one(
    name: &str,
    shape: &[usize; 4],
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Result<()> {
    let dev = global_cuda_device();
    let conv = Conv2d::new_with_bias_zeroed(in_ch, out_ch, k, stride, pad, dev.clone(), true)?;
    let _ = flame_core::rng::set_seed(1);
    let x = flame_core::Tensor::randn(flame_core::Shape::from_dims(shape), 0.0, 0.5, dev.clone())?
        .to_dtype(flame_core::DType::BF16)?;

    for _ in 0..30 {
        let _ = black_box(conv.forward(&x)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(conv.forward(&x)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    println!(
        "  {:<20} dims={:?}  k={} → {:>8.2} us/iter",
        name, shape, k, per_us
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("=== conv2d bench post-algo-cache ===");
    // matches inference-flame bench/bench_flame.rs shapes
    run_one("vae_mid 256ch 256²", &[1, 256, 256, 256], 256, 256, 3, 1, 1)?;
    run_one(
        "vae_up 128ch 1024²",
        &[1, 128, 1024, 1024],
        128,
        128,
        3,
        1,
        1,
    )?;
    run_one("vae_1x1 512ch 128²", &[1, 512, 128, 128], 512, 512, 1, 1, 0)?;
    Ok(())
}
