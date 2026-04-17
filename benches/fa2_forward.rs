//! FA2 forward-kernel throughput bench (Phase 1 of the FA2 port).
//!
//! Compares three forward paths, same input, same stream:
//!   (1) New FA2 kernel            — `flame_flash_attention_bf16`
//!   (2) Legacy BQ=32 kernel       — `flame_flash_attention_bf16_wmma_legacy`
//!   (3) PyTorch CUTLASS flash     — via `torch_sdpa.rs` (env-gated, skipped
//!                                   if libtorch is unavailable in this env)
//!
//! Harness: `harness = false`. Run with:
//!     cargo bench --bench fa2_forward
//!     FLAME_USE_TORCH_SDPA=1 cargo bench --bench fa2_forward   # enable torch col
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

fn bench_path(
    name: &str,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    use_legacy: bool,
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
            if use_legacy {
                flame_core::cuda::ffi::flame_flash_attention_bf16_wmma_legacy(
                    q_ptr, k_ptr, v_ptr, o_ptr, core::ptr::null_mut(),
                    bh, sq, sk, hd, stream,
                )
            } else {
                flame_core::cuda::ffi::flame_flash_attention_bf16(
                    q_ptr, k_ptr, v_ptr, o_ptr, core::ptr::null_mut(),
                    bh, sq, sk, hd, stream,
                )
            }
        };
        if ret != 0 {
            anyhow::bail!("{name} launch returned {ret}");
        }
        q.device()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync: {e:?}"))
    };

    // Warmup
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

    // Traffic model: attention reads Q+K+V BF16 (3*BH*N*HD*2 B) + writes O
    // (BH*N*HD*2 B) → total memory touched (lower bound, ignoring SMEM reuse
    // of K/V across Q tiles — FlashAttn has O(N²/BQ) effective BW).
    let bh_ = dims[0] as f64;
    let n = dims[1] as f64;
    let d = dims[2] as f64;
    let bytes_touched = 4.0 * bh_ * n * d * 2.0;
    let gbps = (bytes_touched / (med_ms / 1000.0)) / 1.0e9;
    Ok((med_ms, gbps))
}

/// Bench torch_sdpa path — returns None if libtorch isn't available
/// or FLAME_USE_TORCH_SDPA isn't set.
fn bench_torch(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Option<(f64, f64)>> {
    if std::env::var("FLAME_USE_TORCH_SDPA").ok().as_deref() != Some("1") {
        return Ok(None);
    }
    let dims = q.shape().dims();
    let bh_ = dims[0] as f64;
    let n = dims[1] as f64;
    let d = dims[2] as f64;
    let bytes_touched = 4.0 * bh_ * n * d * 2.0;

    // torch_sdpa takes [B, H, N, D] — we have [BH, N, D]; reshape is illegal
    // across BH so we take BH as H with B=1.
    let q4 = q.reshape(&[1, dims[0], dims[1], dims[2]])?;
    let k4 = k.reshape(&[1, dims[0], dims[1], dims[2]])?;
    let v4 = v.reshape(&[1, dims[0], dims[1], dims[2]])?;

    let run_once = || -> Result<Tensor> {
        let o = flame_core::torch_sdpa::torch_flash_sdpa(&q4, &k4, &v4)
            .map_err(|e| anyhow::anyhow!("torch_flash_sdpa: {e:?}"))?;
        q4.device()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync: {e:?}"))?;
        Ok(o)
    };

    // Warmup
    match run_once() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("  [torch] unavailable: {e}");
            return Ok(None);
        }
    }
    for _ in 1..WARMUP {
        let _ = run_once()?;
    }
    let mut samples = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let t = Instant::now();
        let _ = run_once()?;
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let med_ms = median(samples);
    let gbps = (bytes_touched / (med_ms / 1000.0)) / 1.0e9;
    Ok(Some((med_ms, gbps)))
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
        "{:<8}  {:>12}  {:>12}  {:>10}",
        "path", "median (ms)", "GB/s", "% peak"
    );

    let (new_ms, new_gbps) = bench_path("fa2", &q, &k, &v, &out, false)?;
    println!(
        "{:<8}  {:>12.3}  {:>12.1}  {:>9.1}%",
        "fa2 (new)",
        new_ms,
        new_gbps,
        100.0 * new_gbps / RTX3090_PEAK_GBPS
    );

    let (leg_ms, leg_gbps) = bench_path("legacy", &q, &k, &v, &out, true)?;
    println!(
        "{:<8}  {:>12.3}  {:>12.1}  {:>9.1}%",
        "legacy BQ=32",
        leg_ms,
        leg_gbps,
        100.0 * leg_gbps / RTX3090_PEAK_GBPS
    );
    println!("{:<8}  speedup new / legacy: {:.2}x", "", leg_ms / new_ms);

    match bench_torch(&q, &k, &v)? {
        Some((t_ms, t_gbps)) => {
            println!(
                "{:<8}  {:>12.3}  {:>12.1}  {:>9.1}%",
                "torch-sdpa",
                t_ms,
                t_gbps,
                100.0 * t_gbps / RTX3090_PEAK_GBPS
            );
            println!("{:<8}  speedup new / torch : {:.2}x", "", t_ms / new_ms);
        }
        None => {
            println!("{:<8}  (skipped — FLAME_USE_TORCH_SDPA not set or libtorch absent)", "torch-sdpa");
        }
    }
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

    // Spec: seq_len ∈ {1024, 4096, 16384, 65536}, H=16, D=128, B=1.
    // N=65536 × H=16 × D=128 × 4 tensors × 2 B = 1.0 GB for inputs alone.
    // 24 GB 3090 can fit this comfortably.
    let seq_lens = [1024usize, 4096, 16384, 65536];
    for &n in &seq_lens {
        run_shape(&arc, 1, 16, n, 128)?;
    }
    Ok(())
}
