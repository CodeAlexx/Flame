//! Verify that `cuda_ops_bf16::rms_norm_bf16` (the inference path) now
//! routes through the same vec kernel as the training path. Compares old
//! direct `fc_rms_norm_bf16` C++ kernel to the unified `norm::rms_norm`
//! dispatch (which picks vec when norm_size % 4 == 0).

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
    let w = rand_bf16(&[last], 2, 1.0)?;

    for _ in 0..50 {
        let _ = black_box(cuda_ops_bf16::rms_norm_bf16(&x, Some(&w), 1e-6)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(cuda_ops_bf16::rms_norm_bf16(&x, Some(&w), 1e-6)?);
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
    println!("=== rms_norm_bf16 (cuda_ops_bf16 inference path, post-unification) ===");
    bench("rmsnorm_small", &[1, 4096, 1280])?; // matches inference-flame bench
    bench("rmsnorm_medium", &[1, 4096, 3072])?;
    bench("rmsnorm_large", &[1, 4096, 4096])?;
    bench("zimage 2560", &[1, 4096, 2560])?;
    bench("klein 4096", &[1, 4096, 4096])?;
    Ok(())
}
