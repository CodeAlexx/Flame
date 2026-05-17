//! Phase 0 dispatch-refactor verification bench v2 (2026-05-12, post-skeptic).
//!
//! v1 had three defects flagged by the skeptic:
//!   1. Path A and B called different kernels (Tensor::silu → silu_bf16_iter
//!      via TensorIterator, vs Path B direct bf16_ops::silu_bf16 NVRTC).
//!      Apples-to-oranges.
//!   2. Inputs had `requires_grad=false`, so Tensor::silu skipped record_op
//!      entirely. The "autograd ON" label was a lie.
//!   3. Forward-only, single shape — didn't represent training (no backward
//!      tape walk, no small-tensor case).
//!
//! v2 fixes:
//!   - **Same kernels both paths.** Path A and Path B both invoke
//!     `silu_bf16_iter` / `mul_scalar_bf16_iter` / `add_bf16_iter`. The only
//!     delta is whether the call site goes through `Tensor::silu` (which
//!     wraps with autograd machinery when `requires_grad=true`) vs direct
//!     iter call (which doesn't).
//!   - **Real autograd.** Input has `requires_grad_(true)`, so the Tensor
//!     method dispatch path actually records ops to the tape.
//!   - **Backward in the loop.** After the 3-op forward, call backward()
//!     on the final scalar to walk the tape — that's where the per-op
//!     saved-tensor clones matter for training step time.
//!   - **Two shapes**: hot training shape `[1, 4096, 2560]` and small
//!     `[1, 64, 2560]` (typical pre/post-pool intermediate).
//!
//! Run:
//!   cd flame-core && cargo bench --features cuda --bench dispatch_overhead

#![cfg(feature = "cuda")]

use flame_core::{
    global_cuda_device,
    tensor_iterator::ops::binary::{add_bf16_iter, mul_scalar_bf16_iter},
    tensor_iterator::ops::unary::silu_bf16_iter,
    AutogradContext, DType, Result, Shape, Tensor,
};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 100;
const WARMUP: usize = 20;

fn median_us(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n % 2 == 0 {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    } else {
        samples[n / 2]
    }
}

fn make_input(shape: &[usize], rg: bool) -> Result<Tensor> {
    let dev = global_cuda_device();
    let t = Tensor::randn(Shape::from_dims(shape), 0.0, 0.5, dev.clone())?.to_dtype(DType::BF16)?;
    Ok(t.requires_grad_(rg))
}

/// Path A — Tensor methods (autograd machinery exercised when requires_grad=true).
/// Forward 3-op chain followed by sum + backward to walk the tape.
fn bench_path_a(shape: &[usize]) -> Result<(f64, f64)> {
    let dev = global_cuda_device();
    AutogradContext::set_enabled(true);

    let x = make_input(shape, true)?;
    let z = make_input(shape, true)?;

    // Warmup
    for _ in 0..WARMUP {
        AutogradContext::clear();
        let y = x.silu()?;
        let y = y.mul_scalar(0.5)?;
        let y = y.add(&z)?;
        let loss = y.sum()?;
        let _ = AutogradContext::backward(&loss)?;
        black_box(loss);
    }
    dev.synchronize()
        .map_err(|e| flame_core::FlameError::Cuda(format!("sync {e:?}")))?;

    // Per-iter samples (each its own sync point).
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        AutogradContext::clear();
        let t = Instant::now();
        let y = x.silu()?;
        let y = y.mul_scalar(0.5)?;
        let y = y.add(&z)?;
        let loss = y.sum()?;
        let _ = AutogradContext::backward(&loss)?;
        black_box(loss);
        dev.synchronize()
            .map_err(|e| flame_core::FlameError::Cuda(format!("sync {e:?}")))?;
        samples.push(t.elapsed().as_nanos() as f64 / 1_000.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let med = median_us(&mut samples);
    Ok((mean, med))
}

/// Path B — same KERNELS as Path A, but direct *_iter calls and no autograd.
/// No backward (Path B has no tape).
fn bench_path_b(shape: &[usize]) -> Result<(f64, f64)> {
    let dev = global_cuda_device();
    AutogradContext::set_enabled(false);

    let x = make_input(shape, false)?;
    let z = make_input(shape, false)?;

    for _ in 0..WARMUP {
        let y = silu_bf16_iter(&x)?;
        let y = mul_scalar_bf16_iter(&y, 0.5)?;
        let _ = black_box(add_bf16_iter(&y, &z)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::FlameError::Cuda(format!("sync {e:?}")))?;

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let y = silu_bf16_iter(&x)?;
        let y = mul_scalar_bf16_iter(&y, 0.5)?;
        let out = add_bf16_iter(&y, &z)?;
        black_box(out);
        dev.synchronize()
            .map_err(|e| flame_core::FlameError::Cuda(format!("sync {e:?}")))?;
        samples.push(t.elapsed().as_nanos() as f64 / 1_000.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let med = median_us(&mut samples);

    AutogradContext::set_enabled(true);
    Ok((mean, med))
}

fn run_shape(label: &str, shape: &[usize]) -> Result<()> {
    println!("\n=== shape {} = {:?} ===", label, shape);
    let (a_mean, a_med) = bench_path_a(shape)?;
    let (b_mean, b_med) = bench_path_b(shape)?;
    println!(
        "  Path A (Tensor methods, autograd+backward)  -> mean {:>8.2} us/iter   median {:>8.2} us",
        a_mean, a_med
    );
    println!(
        "  Path B (direct iter calls, no autograd)     -> mean {:>8.2} us/iter   median {:>8.2} us",
        b_mean, b_med
    );
    let dm = (a_mean - b_mean) / a_mean * 100.0;
    let dM = (a_med - b_med) / a_med * 100.0;
    println!(
        "  Path A overhead vs Path B: mean={:>5.1}%   median={:>5.1}%",
        dm, dM
    );
    println!(
        "  (Path A includes: Tensor::method dispatch + autograd::record_op per op + tape walk + backward kernels;"
    );
    println!(
        "   Path B is forward-only on the same kernels, no recording. Delta = autograd+backward+dispatch cost.)"
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = global_cuda_device();
    // Warm the pool
    let _ = Tensor::zeros_dtype(Shape::from_dims(&[64, 64]), DType::BF16, dev.clone())?;

    println!("=== dispatch_overhead bench v2 — flame-core ===");
    println!("    3-op chain: silu -> mul_scalar(0.5) -> add(&z)");
    println!("    Path A: Tensor methods + requires_grad + backward (real training shape)");
    println!("    Path B: direct *_iter kernel calls + no autograd (forward only)");
    println!("    ITERS = {} (warmup = {})", ITERS, WARMUP);

    // Training hot shape
    run_shape("zimage block", &[1, 4096, 2560])?;
    // Small shape (more likely to expose per-op overhead)
    run_shape("small", &[1, 64, 2560])?;
    // Tiny shape (typical norm/bias)
    run_shape("tiny", &[1, 1, 2560])?;

    Ok(())
}
