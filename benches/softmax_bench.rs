//! Microbench: softmax_lastdim_bf16 at various cols, comparing against
//! morning bench numbers (softmax_small was 4.5x slower than PyTorch).

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, DType, FlameError, Result, Shape, Tensor};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 500;

fn rand_bf16(dims: &[usize], seed: u64) -> Result<Tensor> {
    let dev = global_cuda_device();
    let _ = flame_core::rng::set_seed(seed);
    Tensor::randn(Shape::from_dims(dims), 0.0, 0.5, dev.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
}

fn bench(name: &str, dims: &[usize]) -> Result<()> {
    let dev = global_cuda_device();
    let x = rand_bf16(dims, 1)?;

    for _ in 0..50 {
        let _ = black_box(x.softmax(-1)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(x.softmax(-1)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    println!("  {:<20} dims={:?}  → {:>7.2} us/iter", name, dims, per_us);
    Ok(())
}

fn main() -> Result<()> {
    bench("softmax_small (h=64)", &[30, 4096, 64])?;
    bench("softmax_med (h=128)", &[30, 4096, 128])?;
    bench("softmax_large (h=4096)", &[30, 4096, 4096])?;
    Ok(())
}
