//! Microbench: BF16 slice along leading axis — fast-path (cudaMemcpyAsync) vs
//! generic kernel.
//!
//! Run fast-path:    cargo bench --features cuda --bench slice_copy_bench
//! Run legacy kernel: FLAME_SLICE_COPY_LEGACY=1 cargo bench --features cuda --bench slice_copy_bench
//!
//! Production hot path: batch/seq splitting in attention, modulation slicing
//! in DiT blocks. Auditor flagged 440-3206 calls per training step depending
//! on pipeline.

#![cfg(feature = "cuda")]

use flame_core::{
    cuda_ops_bf16::slice_axis_bf16, global_cuda_device, DType, FlameError, Result, Shape, Tensor,
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

fn bench_slice(name: &str, dims: &[usize], axis: usize, start: usize, len: usize) -> Result<()> {
    let dev = global_cuda_device();
    let x = rand_bf16(dims, 1)?;

    for _ in 0..50 {
        let _ = black_box(slice_axis_bf16(&x, axis, start, len)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(slice_axis_bf16(&x, axis, start, len)?);
    }
    dev.synchronize()
        .map_err(|e| FlameError::Cuda(format!("sync {e:?}")))?;
    let dur = t0.elapsed();
    let per_us = dur.as_secs_f64() * 1e6 / ITERS as f64;
    let copied: usize = len
        * dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .product::<usize>();
    let mb_s = (copied as f64 * 2.0 / 1e6) / (per_us / 1e6); // 1× read, 1× write of BF16... actually it's read+write so 2×
    let mb_total = mb_s * 2.0;
    println!(
        "  {:<28} dims={:?} axis={} [{}..{}] elems={} → {:>7.2} us  ({:>5.0} MB/s rd+wr)",
        name,
        dims,
        axis,
        start,
        start + len,
        copied,
        per_us,
        mb_total
    );
    Ok(())
}

fn main() -> Result<()> {
    let legacy = std::env::var("FLAME_SLICE_COPY_LEGACY")
        .map(|v| v != "0")
        .unwrap_or(false);
    let mode = if legacy {
        "LEGACY (slice_copy_kernel — div/mod per element)"
    } else {
        "FAST PATH (cudaMemcpyAsync)"
    };
    println!("=== slice_copy bench — mode: {} ===", mode);

    // Leading-axis slices (the fast path target). All inputs row-major contiguous.
    bench_slice("img/txt token split A", &[1, 4096, 2560], 0, 0, 1)?;
    bench_slice("batch slice big", &[4, 4096, 3072], 0, 0, 1)?;
    bench_slice("klein 9b head split", &[2, 4096, 4096], 0, 0, 1)?;
    bench_slice("zimage huge axis-0", &[8, 256, 2560], 0, 2, 4)?;
    Ok(())
}
