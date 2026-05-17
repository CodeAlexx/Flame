//! Microbench: current `swiglu_fused_bf16` kernel bandwidth.
//! Read-only — measure first, decide whether to write vec=2 path.

#![cfg(feature = "cuda")]

use flame_core::{
    bf16_ops::swiglu_fused_bf16, global_cuda_device, DType, FlameError, Result, Shape, Tensor,
};
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
    let g = rand_bf16(dims, 1)?;
    let u = rand_bf16(dims, 2)?;
    let elems: usize = dims.iter().product();

    for _ in 0..50 {
        let _ = black_box(swiglu_fused_bf16(&g, &u)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(swiglu_fused_bf16(&g, &u)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let dur = t0.elapsed();
    let per_us = dur.as_secs_f64() * 1e6 / ITERS as f64;
    // 2 reads + 1 write = 3× elems × 2 bytes
    let bytes = elems * 3 * 2;
    let mb_s = (bytes as f64 / 1e6) / (per_us / 1e6);
    println!(
        "  {:<28} dims={:?} elems={:<10} → {:>7.2} us  ({:>5.0} MB/s rd+wr+rd)",
        name, dims, elems, per_us, mb_s
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("=== swiglu_fused_bf16 current kernel bench ===");
    println!("    BF16, randn(0, 0.5). Goal: measure current bandwidth before optimizing.");

    bench("zimage FFN", &[1, 4096, 10240])?;
    bench("klein 9b FFN", &[1, 4096, 14336])?;
    bench("chroma FFN", &[1, 4096, 12288])?;
    bench("small", &[1, 1024, 4096])?;
    Ok(())
}
