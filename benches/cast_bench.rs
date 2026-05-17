//! Microbench: Tensor::to_dtype (BF16↔F32) — fast path vs legacy F32 staging.
//! No env gate (fast path is direct). Just measure absolute speed and compare
//! against the inference-flame bench's 15-30× PyTorch gap.

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, DType, FlameError, Result, Shape, Tensor};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 500;

fn rand(dims: &[usize], dt: DType, seed: u64) -> Result<Tensor> {
    let dev = global_cuda_device();
    let _ = flame_core::rng::set_seed(seed);
    let x = Tensor::randn(Shape::from_dims(dims), 0.0, 0.5, dev.clone())?;
    if x.dtype() != dt {
        x.to_dtype(dt)
    } else {
        Ok(x)
    }
}

fn bench_cast(name: &str, dims: &[usize], src: DType, dst: DType) -> Result<()> {
    let dev = global_cuda_device();
    let x = rand(dims, src, 1)?;
    let elems: usize = dims.iter().product();
    let src_bytes = if src == DType::BF16 { 2 } else { 4 };
    let dst_bytes = if dst == DType::BF16 { 2 } else { 4 };

    for _ in 0..50 {
        let _ = black_box(x.to_dtype(dst)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(x.to_dtype(dst)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    let bytes = elems * (src_bytes + dst_bytes);
    let gb_s = (bytes as f64 / 1e9) / (per_us / 1e6);
    println!(
        "  {:<20} {:?}->{:?}  dims={:?} → {:>7.2} us  ({:>5.0} GB/s rd+wr)",
        name,
        src,
        dst,
        dims,
        per_us,
        gb_s * 1000.0
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("=== Tensor::to_dtype fast-path bench (vs 15-30x PyTorch gap from morning bench) ===");
    bench_cast(
        "medium bf16->f32",
        &[1, 4096, 3840],
        DType::BF16,
        DType::F32,
    )?;
    bench_cast(
        "medium f32->bf16",
        &[1, 4096, 3840],
        DType::F32,
        DType::BF16,
    )?;
    bench_cast("klein bf16->f32", &[1, 4096, 4096], DType::BF16, DType::F32)?;
    bench_cast("klein f32->bf16", &[1, 4096, 4096], DType::F32, DType::BF16)?;
    Ok(())
}
