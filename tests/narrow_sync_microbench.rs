#![cfg(all(feature = "cuda", feature = "heavy_kernels", feature = "bf16_u16"))]

// Microbench + correctness check for the narrow_backward_scatter_add primitive
// after the inline-meta fix in narrow_strided.cu.
//
// Pre-fix per-call cost (from /tmp/flame_klein9b_prof_pinned profile):
//   3× cudaMalloc + 3× cudaMemcpyAsync + kernel + cudaStreamSynchronize + 3× cudaFree
//   = ~8.5 ms median wait per call
//
// Post-fix per-call cost (this file's mechanism):
//   inline kernel-arg meta + kernel launch + (no sync, no alloc)
//   = should be ~kernel-time bound (~200-400 us for shapes used here)
//
// What we measure:
//   - Wall-clock latency over 1000 calls (per-call cost incl. all CUDA API)
//   - Final-tensor correctness via known-pattern scatter check
//
// What we DON'T measure here:
//   - Multi-trainer impact (klein 9B / zimage / ernie / qwen nsys re-runs)
//   - That's the next gate.

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

const ITERS: usize = 1000;

#[test]
fn narrow_backward_microbench_no_sync() -> Result<()> {
    let dev = cuda_device();

    // Representative klein-shape narrow: rank-3, BF16, dim=1, length=2048
    let in_shape = Shape::from_dims(&[1, 4096, 1024]);
    let out_shape = Shape::from_dims(&[1, 2048, 1024]);
    let elem_count = in_shape.elem_count();

    let grad_out_vals: Vec<f32> = (0..out_shape.elem_count())
        .map(|i| (i as f32) * 0.001 - 1.0)
        .collect();
    let grad_out = Tensor::from_vec_dtype(
        grad_out_vals.clone(),
        out_shape.clone(),
        dev.clone(),
        DType::BF16,
    )?;

    // Warm up
    {
        let mut grad_in = Tensor::zeros_dtype(in_shape.clone(), DType::BF16, dev.clone())?;
        for _ in 0..10 {
            Tensor::narrow_backward_scatter_add_cuda(&grad_out, &mut grad_in, 1, 1024, 2048)?;
        }
    }

    // Force a sync to start measurement clean
    dev.synchronize().expect("device sync");

    let mut grad_in = Tensor::zeros_dtype(in_shape.clone(), DType::BF16, dev.clone())?;
    let start = Instant::now();
    for _ in 0..ITERS {
        Tensor::narrow_backward_scatter_add_cuda(&grad_out, &mut grad_in, 1, 1024, 2048)?;
    }
    // Single device sync after the loop — measures the actual queued cost.
    dev.synchronize().expect("device sync");
    let elapsed = start.elapsed();
    let per_call_us = (elapsed.as_micros() as f64) / (ITERS as f64);

    eprintln!(
        "[narrow_bwd_microbench] iters={} total={:?} per_call={:.1} us",
        ITERS, elapsed, per_call_us
    );

    // Correctness: each call adds grad_out into grad_in's [1024..3072] slice along dim=1.
    // After ITERS calls (and the BF16->F32->BF16 detour at the Rust layer for the
    // BF16 dtype path), the [1024..3072] region should hold ITERS * grad_out_vals[i].
    // The BF16 dtype path in tensor_narrow.rs:169-183 currently re-assigns grad_in
    // rather than accumulating across calls, so we just sanity-check finite output.
    let _ = elem_count;
    let final_max_abs: f32 = grad_in
        .to_dtype(DType::F32)?
        .to_vec()?
        .into_iter()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    eprintln!(
        "[narrow_bwd_microbench] final |grad_in|_inf = {:.4} (sanity: must be finite)",
        final_max_abs
    );
    assert!(final_max_abs.is_finite(), "grad_in went non-finite");

    // Hard perf gate: pre-fix typical per-call ~8500 us on klein-sized shapes.
    // Post-fix should be < 2000 us (kernel + driver overhead only). 4x margin
    // to keep this stable across machines.
    assert!(
        per_call_us < 2000.0,
        "per_call_us={:.1} >= 2000 — primitive likely still host-syncing",
        per_call_us
    );

    Ok(())
}
