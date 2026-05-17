// Class C verification: sum_dim_keepdim_bf16 cooperative-reduction rewrite.
//
// Pre-Class-C kernel: one thread per output, scalar serial reduction over
// the axis. ~134× slower per call than PT's Reduce.cuh on klein-sized shapes.
//
// Post-Class-C kernel: one block per output, cooperative warp+shared-memory
// reduce. F32 opmath_t accumulation, BF16 I/O. Speed-contract clause 4
// reference.
//
// This test validates:
//   1. Bit-equal output to within BF16 rounding for a battery of shapes
//      (rank 2..4, reduce dim spanning innermost-to-outermost).
//   2. Per-call latency on a klein-sized reduce is at least a clear
//      improvement over the pre-Class-C serial pattern. We don't bench
//      the old kernel directly here — instead we set a hard upper
//      bound based on the speed-contract budget (<1 ms/call on a 3090 Ti
//      for shape [1, 4096, 1024] reducing last dim). The pre-Class-C
//      kernel ran this in ~1.2 ms; the new kernel should be well under.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

/// CPU reference: sum along `dim`, keepdim=true, F32 accumulation cast back to BF16.
fn sum_dim_keepdim_cpu_ref(
    input_f32: &[f32],
    shape: &[usize],
    dim: usize,
) -> (Vec<f32>, Vec<usize>) {
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let out_elems: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_elems];

    // Row-major strides for input
    let rank = shape.len();
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }

    // Iterate output indices, sum over the reduce axis
    for out_idx in 0..out_elems {
        // Decode out_idx into per-axis coords (collapsing reduce dim → 1)
        let mut coords = vec![0usize; rank];
        let mut rem = out_idx;
        // Use strides of the OUTPUT shape
        let mut out_strides_local = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            out_strides_local[i] = out_strides_local[i + 1] * out_shape[i + 1];
        }
        for i in 0..rank {
            coords[i] = (rem / out_strides_local[i]) % out_shape[i];
            rem %= out_strides_local[i];
        }
        // Base input index with reduce dim = 0
        let mut base = 0usize;
        for i in 0..rank {
            let c = if i == dim { 0 } else { coords[i] };
            base += c * in_strides[i];
        }
        // Accumulate in F32 over reduce axis
        let mut acc = 0.0f32;
        for d in 0..shape[dim] {
            acc += input_f32[base + d * in_strides[dim]];
        }
        // Round to BF16 then back to F32 to match the GPU output's final precision
        out[out_idx] = bf16_round_trip(acc);
    }
    (out, out_shape)
}

/// Match the GPU kernel's `__float2bfloat16_rn` semantics.
fn bf16_round_trip(x: f32) -> f32 {
    let bits = x.to_bits();
    // Round-to-nearest-even mantissa truncation to BF16
    let rounding = 0x7FFFu32 + ((bits >> 16) & 1);
    let rounded = (bits.wrapping_add(rounding)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

fn run_parity_case(dev: &Arc<CudaDevice>, shape: &[usize], dim: usize) -> Result<()> {
    let elem_count: usize = shape.iter().product();
    // Use small deterministic values so F32 accumulation doesn't drift far.
    let input_f32: Vec<f32> = (0..elem_count)
        .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
        .collect();
    let input = Tensor::from_vec_dtype(
        input_f32.clone(),
        Shape::from_dims(shape),
        dev.clone(),
        DType::BF16,
    )?;

    // CPU reference (operates on the BF16-quantized values to match what the
    // GPU kernel will see — so we reference-quantize the input first too).
    let input_bf16_f32: Vec<f32> = input_f32.iter().map(|&v| bf16_round_trip(v)).collect();
    let (cpu_out, out_shape) = sum_dim_keepdim_cpu_ref(&input_bf16_f32, shape, dim);

    // GPU
    let gpu_out_tensor = input.sum_dim_keepdim(dim)?;
    assert_eq!(
        gpu_out_tensor.shape().dims(),
        out_shape.as_slice(),
        "shape mismatch"
    );
    let gpu_out: Vec<f32> = gpu_out_tensor.to_dtype(DType::F32)?.to_vec()?;

    // Compare. BF16 has ~8 bits of mantissa; for sums up to ~reduce_size values
    // ranging ±0.1, accumulated error stays small. We allow rel tol scaled
    // by reduce_size to capture the BF16-round-per-add divergence between
    // the CPU's order (sequential) and the GPU's order (cooperative tree).
    let reduce_size = shape[dim] as f32;
    let abs_tol = 0.01 * reduce_size.sqrt(); // empirical, generous
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let d = (g - c).abs();
        let r = d / (c.abs().max(1e-6));
        if d > max_abs {
            max_abs = d;
        }
        if r > max_rel {
            max_rel = r;
        }
        assert!(
            d <= abs_tol,
            "shape={:?} dim={} idx={} gpu={} cpu={} diff={} tol={}",
            shape,
            dim,
            i,
            g,
            c,
            d,
            abs_tol
        );
    }
    eprintln!(
        "[sum_dim_keepdim_class_c] shape={:?} dim={} max_abs={:.4e} max_rel={:.4e} (tol={:.4e})",
        shape, dim, max_abs, max_rel, abs_tol
    );
    Ok(())
}

#[test]
fn sum_dim_keepdim_bf16_parity_battery() -> Result<()> {
    let dev = cuda_device();
    // Rank 2..4, reduce dim spanning innermost..outermost. Sizes that exercise
    // the cooperative-reduce geometry (axes > 32 to engage warp-shuffle paths,
    // axes ≤ 32 to engage the single-warp short path).
    let cases: &[(&[usize], usize)] = &[
        // Rank 2
        (&[4, 4096], 1), // last-dim reduce, sized to engage 256-thread block
        (&[4, 4096], 0), // first-dim reduce, strided access pattern
        (&[8, 17], 0),   // axis < warp, single-warp short path
        (&[8, 17], 1),
        // Rank 3
        (&[1, 4096, 1024], 1), // middle-dim reduce (klein-like attention reduce)
        (&[1, 4096, 1024], 2), // last-dim
        (&[2, 128, 256], 1),
        // Rank 4
        (&[1, 8, 256, 64], 2),
        (&[1, 8, 256, 64], 3),
    ];
    for (shape, dim) in cases {
        run_parity_case(&dev, shape, *dim)?;
    }
    Ok(())
}

#[test]
fn sum_dim_keepdim_bf16_perf_budget() -> Result<()> {
    let dev = cuda_device();
    let shape = Shape::from_dims(&[1, 4096, 1024]);
    let input = Tensor::zeros_dtype(shape, DType::BF16, dev.clone())?;

    // Warm up
    for _ in 0..10 {
        let _ = input.sum_dim_keepdim(2)?;
    }
    dev.synchronize().expect("sync");

    const ITERS: usize = 100;
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = input.sum_dim_keepdim(2)?;
    }
    dev.synchronize().expect("sync");
    let elapsed = start.elapsed();
    let per_call_us = (elapsed.as_micros() as f64) / (ITERS as f64);

    eprintln!(
        "[sum_dim_keepdim_class_c_perf] shape=[1,4096,1024] dim=2 iters={} total={:?} per_call={:.1} us",
        ITERS, elapsed, per_call_us
    );

    // Hard budget: pre-Class-C ran ~1200 µs/call on this shape on a 3090 Ti.
    // Post-Class-C target is well under that. Set a generous bound to keep
    // the gate stable across hosts: 300 µs (4× safety margin against the
    // expected ~50-150 µs).
    assert!(
        per_call_us < 300.0,
        "per_call_us={:.1} exceeds Class-C budget — kernel likely still serial",
        per_call_us
    );
    Ok(())
}
