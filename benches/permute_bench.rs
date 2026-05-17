//! Microbench: current `permute_0213` / `permute_021` BF16 kernels at
//! production attention shapes. Goal: measure actual bandwidth before
//! deciding whether to write a tiled-transpose replacement.

#![cfg(feature = "cuda")]

use flame_core::{cuda_ops::GpuOps, global_cuda_device, DType, FlameError, Result, Shape, Tensor};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 500;

fn rand_bf16(dims: &[usize], seed: u64) -> Result<Tensor> {
    let dev = global_cuda_device();
    let _ = flame_core::rng::set_seed(seed);
    Tensor::randn(Shape::from_dims(dims), 0.0, 0.5, dev.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
}

fn bench_0213(name: &str, dims: &[usize]) -> Result<()> {
    let dev = global_cuda_device();
    let x = rand_bf16(dims, 1)?;
    let elems: usize = dims.iter().product();

    for _ in 0..50 {
        let _ = black_box(GpuOps::permute_0213(&x)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(GpuOps::permute_0213(&x)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let dur = t0.elapsed();
    let per_us = dur.as_secs_f64() * 1e6 / ITERS as f64;
    let mb_s = (elems as f64 * 2.0 * 2.0 / 1e6) / (per_us / 1e6); // BF16 r+w
    println!(
        "  0213  {:<24} dims={:?} elems={:<10} → {:>7.2} us  ({:>5.0} MB/s)",
        name, dims, elems, per_us, mb_s
    );
    Ok(())
}

fn bench_021(name: &str, dims: &[usize]) -> Result<()> {
    let dev = global_cuda_device();
    let x = rand_bf16(dims, 1)?;
    let elems: usize = dims.iter().product();

    for _ in 0..50 {
        let _ = black_box(GpuOps::permute_021(&x)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(GpuOps::permute_021(&x)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let dur = t0.elapsed();
    let per_us = dur.as_secs_f64() * 1e6 / ITERS as f64;
    let mb_s = (elems as f64 * 2.0 * 2.0 / 1e6) / (per_us / 1e6);
    println!(
        "  021   {:<24} dims={:?} elems={:<10} → {:>7.2} us  ({:>5.0} MB/s)",
        name, dims, elems, per_us, mb_s
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("=== permute current kernel bench ===");
    println!("    BF16, randn(0, 0.5). Goal: measure current bandwidth before optimizing.");

    // permute0213: [B, H, S, D] -> [B, S, H, D] (attention reshape back)
    bench_0213("zimage Q/K/V swap", &[1, 24, 4096, 128])?;
    bench_0213("klein 9b QKV reshape", &[1, 32, 4096, 128])?;
    bench_0213("chroma small", &[1, 16, 1024, 128])?;

    // permute021: [N, A, B] -> [N, B, A] (matrix transpose with batch)
    bench_021("typical attn matrix", &[24, 4096, 4096])?;
    bench_021("small batched", &[32, 256, 128])?;
    Ok(())
}
