//! FA2 forward-kernel throughput bench.
//!
//! Times the in-tree FA2 forward path (`flame_flash_attention_bf16`) across
//! a set of sequence lengths and reports median ms + achieved GB/s.
//!
//! Harness: `harness = false`. Run with:
//!     cargo bench --bench fa2_forward
//!
//! Shapes:   seq_len ∈ {1024, 4096, 16384, 65536}, head_dim=128,
//!           num_heads=16, batch=1, BF16
//! Metric:   median step time over N trials, achieved GB/s, % of 3090's
//!           936 GB/s theoretical memory bandwidth.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use flame_core::{DType, Device, Shape, Tensor};
use std::time::Instant;

const RTX3090_PEAK_GBPS: f64 = 936.0;
const TRIALS: usize = 20;
const WARMUP: usize = 5;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Time a single kernel launch + synchronize, returning milliseconds.
fn time_one<F: FnMut() -> Result<()>>(mut f: F) -> Result<f64> {
    let t = Instant::now();
    f()?;
    Ok(t.elapsed().as_secs_f64() * 1000.0)
}

fn bench_fa2(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
) -> Result<(f64, f64)> {
    let dims = q.shape().dims();
    let bh = dims[0] as i32;
    let sq = dims[1] as i32;
    let sk = k.shape().dims()[1] as i32;
    let hd = dims[2] as i32;

    let q_ptr = q.as_device_ptr_bf16("fa2_bench:q")? as *const core::ffi::c_void;
    let k_ptr = k.as_device_ptr_bf16("fa2_bench:k")? as *const core::ffi::c_void;
    let v_ptr = v.as_device_ptr_bf16("fa2_bench:v")? as *const core::ffi::c_void;
    let o_ptr = out.as_device_ptr_bf16("fa2_bench:o")? as *mut core::ffi::c_void;
    let stream = flame_core::cuda::device_lt::stream_ptr(q.device())
        .map_err(|e| anyhow::anyhow!("stream_ptr: {e:?}"))?;

    let run_once = || -> Result<()> {
        let ret = unsafe {
            flame_core::cuda::ffi::flame_flash_attention_bf16(
                q_ptr, k_ptr, v_ptr, o_ptr, core::ptr::null_mut(),
                bh, sq, sk, hd, stream,
            )
        };
        if ret != 0 {
            anyhow::bail!("fa2 launch returned {ret}");
        }
        q.device()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync: {e:?}"))
    };

    for _ in 0..WARMUP {
        let mut f = run_once;
        time_one(&mut f)?;
    }
    let mut samples = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let mut f = run_once;
        samples.push(time_one(&mut f)?);
    }
    let med_ms = median(samples);

    let bh_ = dims[0] as f64;
    let n = dims[1] as f64;
    let d = dims[2] as f64;
    let bytes_touched = 4.0 * bh_ * n * d * 2.0;
    let gbps = (bytes_touched / (med_ms / 1000.0)) / 1.0e9;
    Ok((med_ms, gbps))
}

fn run_shape(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<()> {
    let bh = batch * num_heads;
    let q = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let k = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let v = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    let out = Tensor::empty_dtype(
        Shape::from_dims(&[bh, seq_len, head_dim]),
        DType::BF16,
        device.clone(),
    )?;

    println!(
        "\n=== B={batch} H={num_heads} N={seq_len} D={head_dim} ===  (BH={bh})"
    );
    println!(
        "{:<12}  {:>12}  {:>12}  {:>10}",
        "path", "median (ms)", "GB/s", "% peak"
    );

    let (ms, gbps) = bench_fa2(&q, &k, &v, &out)?;
    println!(
        "{:<12}  {:>12.3}  {:>12.1}  {:>9.1}%",
        "fa2",
        ms,
        gbps,
        100.0 * gbps / RTX3090_PEAK_GBPS
    );
    Ok(())
}

fn main() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skip] CUDA unavailable: {e}");
            return Ok(());
        }
    };
    let arc = device.cuda_device().clone();

    let seq_lens = [1024usize, 4096, 16384, 65536];
    for &n in &seq_lens {
        run_shape(&arc, 1, 16, n, 128)?;
    }
    Ok(())
}
