//! Non-causal (diffusion) SDPA forward bench — flame-core cuDNN path.
//! Same harness as sdpa_causal_bench, calling attention::sdpa(q,k,v,None).
//! Shapes [B,H,S,D] = [1,16,1024,{64,128}] BF16 to match the Mojo sdpa bench.

use std::hint::black_box;
use std::time::Instant;

use flame_core::{attention, global_cuda_device, DType, FlameError, Result, Shape, Tensor};

const ITERS: usize = 100;

fn rand_bf16(dims: &[usize], seed: u64) -> Result<Tensor> {
    let dev = global_cuda_device();
    let _ = seed;
    Tensor::randn(Shape::from_dims(dims), 0.0, 0.5, dev.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
}

fn bench_sdpa_noncausal(name: &str, dims: &[usize]) -> Result<()> {
    let dev = global_cuda_device();
    let q = rand_bf16(dims, 1)?;
    let k = rand_bf16(dims, 2)?;
    let v = rand_bf16(dims, 3)?;

    for _ in 0..10 {
        let _ = black_box(attention::sdpa(&q, &k, &v, None)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(attention::sdpa(&q, &k, &v, None)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    println!(
        "  SDPA(cuDNN, non-causal)  {:<8} dims={:?} -> {:>9.2} us/iter",
        name, dims, per_us
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("=== flame-core SDPA non-causal (cuDNN), BF16, [B,H,S,D] ===");
    bench_sdpa_noncausal("Dh=64", &[1, 16, 1024, 64])?;
    bench_sdpa_noncausal("Dh=128", &[1, 16, 1024, 128])?;
    Ok(())
}
