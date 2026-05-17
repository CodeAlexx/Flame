//! Microbench: RMSNorm forward (BF16) — new vectorized vs legacy scalar.
//!
//! Run vec (default):
//!   cd flame-core && cargo bench --features cuda --bench rms_norm_vec
//! Run legacy scalar (forced):
//!   cd flame-core && FLAME_RMS_NORM_LEGACY=1 cargo bench --features cuda --bench rms_norm_vec

#![cfg(feature = "cuda")]

use flame_core::{
    autograd::AutogradContext, global_cuda_device, norm, DType, Result, Shape, Tensor,
};
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 200;

fn bench_fwd(name: &str, dims: &[usize], norm_size: usize) -> Result<()> {
    let dev = global_cuda_device();
    let x = Tensor::zeros_dtype(Shape::from_dims(dims), DType::BF16, dev.clone())?;
    let w = Tensor::zeros_dtype(Shape::from_dims(&[norm_size]), DType::BF16, dev.clone())?;

    for _ in 0..20 {
        let _ = black_box(norm::rms_norm(&x, &[norm_size], Some(&w), 1e-6)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(norm::rms_norm(&x, &[norm_size], Some(&w), 1e-6)?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::FlameError::Cuda(format!("sync {e:?}")))?;
    let dur = t0.elapsed();
    let per_us = dur.as_secs_f64() * 1e6 / ITERS as f64;
    println!(
        "  fwd  {:<14} dims={:?} norm={:<5} → {:>8.2} us/iter",
        name, dims, norm_size, per_us
    );
    Ok(())
}

fn bench_bwd(name: &str, dims: &[usize], norm_size: usize) -> Result<()> {
    let dev = global_cuda_device();
    // grad_out, input, weight all BF16 at production shapes
    let grad_out = Tensor::zeros_dtype(Shape::from_dims(dims), DType::BF16, dev.clone())?;
    let input = Tensor::zeros_dtype(Shape::from_dims(dims), DType::BF16, dev.clone())?;
    let weight = Tensor::zeros_dtype(Shape::from_dims(&[norm_size]), DType::BF16, dev.clone())?;

    // Pre-compute inv_rms via a forward call so backward has the artifact it expects.
    // We call the public path which both produces inv_rms and (via autograd) wires
    // up backward dispatch — so timing measures the BACKWARD kernel.
    let total = dims.iter().product::<usize>();
    let batch_size = total / norm_size;

    // Use the test/bench helper that exposes backward directly.
    use flame_core::norm::rms_norm_backward_for_bench as bwd;

    // Need a fresh inv_rms tensor matching what forward would produce. Pass zeros
    // — the kernel reads it as a per-row scalar; the math doesn't NaN with zeros
    // here because (0 * inv_cubed) = 0 and grad_x = 0 * scaled - x * 0 = 0.
    // For wall-time purposes the kernel runs the same loops regardless.
    let inv_rms = Tensor::zeros_dtype(Shape::from_dims(&[batch_size]), DType::F32, dev.clone())?;

    for _ in 0..20 {
        let _ = black_box(bwd(
            &grad_out,
            &input,
            Some(&weight),
            &inv_rms,
            batch_size,
            norm_size,
        )?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::FlameError::Cuda(format!("sync {e:?}")))?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(bwd(
            &grad_out,
            &input,
            Some(&weight),
            &inv_rms,
            batch_size,
            norm_size,
        )?);
    }
    dev.synchronize()
        .map_err(|e| flame_core::FlameError::Cuda(format!("sync {e:?}")))?;
    let dur = t0.elapsed();
    let per_us = dur.as_secs_f64() * 1e6 / ITERS as f64;
    println!(
        "  bwd  {:<14} dims={:?} norm={:<5} → {:>8.2} us/iter",
        name, dims, norm_size, per_us
    );
    let _ = AutogradContext::clear;
    Ok(())
}

fn main() -> Result<()> {
    let legacy = std::env::var("FLAME_RMS_NORM_LEGACY")
        .map(|v| v != "0")
        .unwrap_or(false);
    let mode = if legacy {
        "LEGACY scalar (1 thread/row)"
    } else {
        "VEC (256 threads/row)"
    };
    println!("=== rms_norm_vec bench — mode: {} ===", mode);

    bench_fwd("zimage block", &[1, 4096, 2560], 2560)?;
    bench_fwd("zimage qknorm", &[1, 24, 4096, 128], 128)?;
    bench_fwd("klein block", &[1, 4096, 4096], 4096)?;
    bench_fwd("chroma block", &[1, 4096, 3072], 3072)?;

    bench_bwd("zimage block", &[1, 4096, 2560], 2560)?;
    bench_bwd("zimage qknorm", &[1, 24, 4096, 128], 128)?;
    bench_bwd("klein block", &[1, 4096, 4096], 4096)?;
    bench_bwd("chroma block", &[1, 4096, 3072], 3072)?;
    Ok(())
}
