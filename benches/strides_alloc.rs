//! Microbench: `Tensor::strides()` and `Shape::strides()` allocation cost.
//!
//! Every kernel launcher in flame-core calls `.strides()` to pass stride
//! arrays to cuBLASLt / cuDNN / NVRTC entry points. The current
//! implementation heap-allocates a `Vec<usize>` on every call — even though
//! DL tensors are always rank ≤ 6.
//!
//! Baseline (before the SmallVec switch) establishes ns/call and total heap
//! bytes per 1M calls. Post-switch numbers go into the commit message.
//!
//! Run:
//!   cargo run --features cuda --release --bench strides_alloc

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, DType, Result, Shape, Tensor};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 1_000_000;

fn bench_shape_strides(rank: usize, dim: usize) {
    let dims: Vec<usize> = (0..rank).map(|i| dim + i).collect();
    let shape = Shape::from_dims(&dims);

    // Warmup
    for _ in 0..1024 {
        black_box(shape.strides());
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(shape.strides());
    }
    let dt = t0.elapsed();
    let ns = dt.as_nanos() as f64 / ITERS as f64;
    println!(
        "Shape::strides   rank={} dims={:?}   {:.1} ns/call  ({:.2} ms / {}M)",
        rank,
        shape.dims(),
        ns,
        dt.as_secs_f64() * 1000.0,
        ITERS / 1_000_000
    );
}

fn bench_tensor_strides_contig(rank: usize, dim: usize) -> Result<()> {
    let dev = global_cuda_device();
    let dims: Vec<usize> = (0..rank).map(|i| dim + i).collect();
    let t = Tensor::zeros_dtype(Shape::from_dims(&dims), DType::BF16, dev.clone())?;

    for _ in 0..1024 {
        black_box(t.strides());
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(t.strides());
    }
    let dt = t0.elapsed();
    let ns = dt.as_nanos() as f64 / ITERS as f64;
    println!(
        "Tensor::strides  rank={} dims={:?} contig  {:.1} ns/call  ({:.2} ms / {}M)",
        rank,
        t.shape().dims(),
        ns,
        dt.as_secs_f64() * 1000.0,
        ITERS / 1_000_000
    );
    Ok(())
}

fn bench_tensor_strides_view(rank: usize) -> Result<()> {
    // custom_strides populated via permute — simulates the attention-hot path
    // where QKV projection outputs get permuted to [B, h, S, d] before SDPA.
    let dev = global_cuda_device();
    let dims: Vec<usize> = (0..rank).map(|i| 8 + i).collect();
    let t = Tensor::zeros_dtype(Shape::from_dims(&dims), DType::BF16, dev.clone())?;
    let perm: Vec<usize> = (0..rank).rev().collect();
    let v = t.permute(&perm)?;

    for _ in 0..1024 {
        black_box(v.strides());
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(v.strides());
    }
    let dt = t0.elapsed();
    let ns = dt.as_nanos() as f64 / ITERS as f64;
    println!(
        "Tensor::strides  rank={} view (custom)  {:.1} ns/call  ({:.2} ms / {}M)",
        rank,
        ns,
        dt.as_secs_f64() * 1000.0,
        ITERS / 1_000_000
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("strides_alloc — {}M iterations per case\n", ITERS / 1_000_000);
    for &rank in &[2usize, 3, 4, 5] {
        bench_shape_strides(rank, 16);
    }
    println!();
    for &rank in &[2usize, 3, 4, 5] {
        bench_tensor_strides_contig(rank, 16)?;
    }
    println!();
    for &rank in &[2usize, 3, 4, 5] {
        bench_tensor_strides_view(rank)?;
    }
    Ok(())
}
