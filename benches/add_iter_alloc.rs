//! Microbench: contig-path overhead of `ops::add_iter` dispatcher + strided
//! reference. Session 4 of the TensorIterator port.
//!
//! Phase 5b update (2026-04-23): the `bench_contig_reference` function
//! that compared against `bf16_elementwise::add_bf16` was removed —
//! that reference path was deleted. The bench now times only the new
//! iter-path (contig input + strided-view input).

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, ops::add_iter::add_bf16_iter, DType, Result, Shape, Tensor};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 100;

fn bench_contig_through_iter(rows: usize, cols: usize) -> Result<()> {
    let dev = global_cuda_device();
    let dims = [rows, cols];
    let a = Tensor::zeros_dtype(Shape::from_dims(&dims), DType::BF16, dev.clone())?;
    let b = Tensor::zeros_dtype(Shape::from_dims(&dims), DType::BF16, dev.clone())?;

    for _ in 0..10 {
        let _ = black_box(add_bf16_iter(&a, &b)?);
    }
    dev.synchronize().map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(add_bf16_iter(&a, &b)?);
    }
    dev.synchronize().map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;
    let dt = t0.elapsed();
    let us = dt.as_nanos() as f64 / ITERS as f64 / 1_000.0;
    println!(
        "add_bf16_iter              [{rows}, {cols}] contig  {us:.2} µs/iter  ({:.2} ms / {} iters)",
        dt.as_secs_f64() * 1000.0,
        ITERS
    );
    Ok(())
}

fn bench_strided_via_iter(rows: usize, cols: usize) -> Result<()> {
    let dev = global_cuda_device();
    let a_base = Tensor::zeros_dtype(Shape::from_dims(&[rows, cols]), DType::BF16, dev.clone())?;
    let b_base = Tensor::zeros_dtype(Shape::from_dims(&[rows, cols]), DType::BF16, dev.clone())?;
    let a_view = a_base.as_strided(&[cols, rows], &[1, rows], 0)?;
    let b_view = b_base.as_strided(&[cols, rows], &[1, rows], 0)?;

    for _ in 0..10 {
        let _ = black_box(add_bf16_iter(&a_view, &b_view)?);
    }
    dev.synchronize().map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(add_bf16_iter(&a_view, &b_view)?);
    }
    dev.synchronize().map_err(|e| flame_core::Error::Cuda(format!("sync {e:?}")))?;
    let dt = t0.elapsed();
    let us = dt.as_nanos() as f64 / ITERS as f64 / 1_000.0;
    println!(
        "add_bf16_iter (both strided) [{cols}, {rows}] T-view  {us:.2} µs/iter  ({:.2} ms / {} iters)",
        dt.as_secs_f64() * 1000.0,
        ITERS
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = global_cuda_device();
    let _ = Tensor::zeros_dtype(Shape::from_dims(&[64, 64]), DType::BF16, dev.clone())?;

    println!("add_iter_alloc — {} iterations per case\n", ITERS);

    println!("-- Primary gate (4096 × 4096) --");
    bench_contig_through_iter(4096, 4096)?;
    bench_strided_via_iter(4096, 4096)?;

    println!("\n-- Smaller shape (1024 × 1024) --");
    bench_contig_through_iter(1024, 1024)?;
    bench_strided_via_iter(1024, 1024)?;

    Ok(())
}
