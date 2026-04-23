//! Microbench: contig-path overhead of the new `ops::silu_iter` dispatcher,
//! plus strided-path reference numbers.
//!
//! Gate 6 of HANDOFF_2026-04-22_TENSORITERATOR_PORT: the contig path must
//! be within 5% of the pre-migration `bf16_ops::silu_bf16` kernel. Since
//! `silu_bf16_iter(x)` on a contiguous `x` is literally a
//! `x.is_contiguous()` branch followed by `return bf16_ops::silu_bf16(x)`,
//! any regression here would come from the branch + function-call overhead
//! — which should be below noise on a ~300 µs kernel.
//!
//! Run:
//!   cd flame-core && cargo bench --features cuda --bench silu_iter_alloc

#![cfg(feature = "cuda")]

use flame_core::{bf16_ops, global_cuda_device, ops::silu_iter::silu_bf16_iter, DType, Result, Shape, Tensor};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 100;

fn bench_contig_reference(rows: usize, cols: usize) -> Result<()> {
    let dev = global_cuda_device();
    let dims = [rows, cols];
    let x = Tensor::zeros_dtype(Shape::from_dims(&dims), DType::BF16, dev.clone())?;
    assert!(x.is_contiguous());

    // Warmup
    for _ in 0..10 {
        let _ = black_box(bf16_ops::silu_bf16(&x)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(bf16_ops::silu_bf16(&x)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;
    let dt = t0.elapsed();
    let us = dt.as_nanos() as f64 / ITERS as f64 / 1_000.0;
    println!(
        "bf16_ops::silu_bf16        [{rows}, {cols}] contig  {us:.2} µs/iter  ({:.2} ms / {} iters)",
        dt.as_secs_f64() * 1000.0,
        ITERS
    );
    Ok(())
}

fn bench_contig_through_iter(rows: usize, cols: usize) -> Result<()> {
    let dev = global_cuda_device();
    let dims = [rows, cols];
    let x = Tensor::zeros_dtype(Shape::from_dims(&dims), DType::BF16, dev.clone())?;
    assert!(x.is_contiguous());

    // Warmup
    for _ in 0..10 {
        let _ = black_box(silu_bf16_iter(&x)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(silu_bf16_iter(&x)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;
    let dt = t0.elapsed();
    let us = dt.as_nanos() as f64 / ITERS as f64 / 1_000.0;
    println!(
        "silu_bf16_iter             [{rows}, {cols}] contig  {us:.2} µs/iter  ({:.2} ms / {} iters)",
        dt.as_secs_f64() * 1000.0,
        ITERS
    );
    Ok(())
}

fn bench_strided_via_iter(rows: usize, cols: usize) -> Result<()> {
    let dev = global_cuda_device();
    // Build a contig [rows, cols] tensor, then take its [cols, rows]
    // transpose view (strides [1, rows]). Every element is read via the
    // strided OffsetCalculator path.
    let x = Tensor::zeros_dtype(Shape::from_dims(&[rows, cols]), DType::BF16, dev.clone())?;
    let view = x.as_strided(&[cols, rows], &[1, rows], 0)?;
    assert!(!view.is_contiguous());

    // Warmup
    for _ in 0..10 {
        let _ = black_box(silu_bf16_iter(&view)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(silu_bf16_iter(&view)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;
    let dt = t0.elapsed();
    let us = dt.as_nanos() as f64 / ITERS as f64 / 1_000.0;
    println!(
        "silu_bf16_iter (strided)   [{cols}, {rows}] T-view  {us:.2} µs/iter  ({:.2} ms / {} iters)",
        dt.as_secs_f64() * 1000.0,
        ITERS
    );
    Ok(())
}

fn main() -> Result<()> {
    // Ensure GPU warm before timing.
    let dev = global_cuda_device();
    let _ = Tensor::zeros_dtype(Shape::from_dims(&[64, 64]), DType::BF16, dev.clone())?;

    println!("silu_iter_alloc — {} iterations per case\n", ITERS);

    // Primary gate: 4096×4096 contig. 16M BF16 elements = 32 MiB.
    println!("-- Primary gate (4096 × 4096) --");
    bench_contig_reference(4096, 4096)?;
    bench_contig_through_iter(4096, 4096)?;
    bench_strided_via_iter(4096, 4096)?;

    println!("\n-- Smaller shape (1024 × 1024) --");
    bench_contig_reference(1024, 1024)?;
    bench_contig_through_iter(1024, 1024)?;
    bench_strided_via_iter(1024, 1024)?;

    Ok(())
}
