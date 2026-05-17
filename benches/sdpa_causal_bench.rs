//! Microbench: compare two SDPA causal entry points at LTX/Wan video shapes.
//!
//! The production-incorrect bench in inference-flame uses
//! `cuda_ops_bf16::sdpa_stream_bf16(chunk=full_seq, causal=true)` — the
//! streaming kernel is for chunked attention on long sequences and has
//! per-chunk overhead that doesn't amortize for single-chunk use. The
//! production-correct entry is `attention::attention_impl(causal=true)`
//! which materializes a causal mask and uses the cuDNN fallback softmax
//! kernel path.
//!
//! This bench measures both at the same shapes the inference-flame bench
//! uses, so we can confirm whether the 930-1033x slowdown is a real perf
//! issue or a bench-API artifact.

#![cfg(feature = "cuda")]

use flame_core::{
    attention, cuda_ops_bf16, global_cuda_device, DType, FlameError, Result, Shape, Tensor,
};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 50;

fn rand_bf16(dims: &[usize], seed: u64) -> Result<Tensor> {
    let dev = global_cuda_device();
    let _ = flame_core::rng::set_seed(seed);
    Tensor::randn(Shape::from_dims(dims), 0.0, 0.5, dev.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
}

fn bench_stream_causal(name: &str, dims: &[usize], chunk: usize) -> Result<()> {
    let dev = global_cuda_device();
    let q = rand_bf16(dims, 1)?;
    let k = rand_bf16(dims, 2)?;
    let v = rand_bf16(dims, 3)?;

    for _ in 0..5 {
        let _ = black_box(cuda_ops_bf16::sdpa_stream_bf16(
            &q, &k, &v, None, chunk, true, None,
        )?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(cuda_ops_bf16::sdpa_stream_bf16(
            &q, &k, &v, None, chunk, true, None,
        )?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let per_ms = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
    println!(
        "  STREAM(chunk={}, causal=true)  {:<14} dims={:?} → {:>9.3} ms/iter",
        chunk, name, dims, per_ms
    );
    Ok(())
}

fn bench_attention_impl_causal(name: &str, dims: &[usize]) -> Result<()> {
    let dev = global_cuda_device();
    let q = rand_bf16(dims, 1)?;
    let k = rand_bf16(dims, 2)?;
    let v = rand_bf16(dims, 3)?;

    for _ in 0..5 {
        let _ = black_box(attention::attention_impl(&q, &k, &v, None, true, None)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(attention::attention_impl(&q, &k, &v, None, true, None)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let per_ms = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
    println!(
        "  ATTENTION_IMPL(causal=true)    {:<14} dims={:?} → {:>9.3} ms/iter",
        name, dims, per_ms
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("=== sdpa_causal: stream vs attention_impl ===");
    println!("    BF16, randn(0, 0.5). All shapes self-attn (q.dims == k.dims == v.dims).");

    let ltx = &[1usize, 32, 768, 128];
    let wan = &[1usize, 40, 1024, 128];

    println!("\nLTX self-attn shape:");
    bench_stream_causal("ltx", ltx, 768)?;
    bench_attention_impl_causal("ltx", ltx)?;

    println!("\nWan self-attn shape:");
    bench_stream_causal("wan", wan, 1024)?;
    bench_attention_impl_causal("wan", wan)?;

    Ok(())
}
