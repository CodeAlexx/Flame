// flame-core/tests/permute_fastpath_bench.rs
//
// Micro-benchmark for the permute_generic fast-paths. Reports median µs/iter
// for fast-path vs generic for each top-frequency tuple in the zimage trace.
//
// Run with:
//   cargo test --release --features cuda --test permute_fastpath_bench \
//     -- --ignored --nocapture
//
// Marked #[ignore] so it doesn't run in normal `cargo test` — it's a
// benchmark, not a correctness check.

#![cfg(feature = "cuda")]

use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn make_device() -> Arc<cudarc::driver::CudaDevice> {
    cudarc::driver::CudaDevice::new(0).expect("CUDA device 0")
}

fn host_data(total: usize) -> Vec<f32> {
    (0..total).map(|i| (i % 7919) as f32 * 0.001).collect()
}

fn make_bf16(device: &Arc<cudarc::driver::CudaDevice>, dims: &[usize]) -> Tensor {
    let total: usize = dims.iter().product();
    Tensor::from_vec_dtype(host_data(total), Shape::from_dims(dims), device.clone(), DType::BF16)
        .expect("from_vec_dtype bf16")
}

fn bench_one(shape: &[usize], perm: &[usize], fastpath: bool, iters: usize, warmup: usize) -> f64 {
    let device = make_device();
    std::env::set_var(
        "FLAME_PERMUTE_FASTPATH",
        if fastpath { "1" } else { "0" },
    );
    let t = make_bf16(&device, shape);

    // Warmup
    for _ in 0..warmup {
        let p = t.permute(perm).expect("permute");
        let _ = p.contiguous().expect("contiguous");
    }
    device.synchronize().expect("sync");

    let t0 = Instant::now();
    for _ in 0..iters {
        let p = t.permute(perm).expect("permute");
        let _ = p.contiguous().expect("contiguous");
    }
    device.synchronize().expect("sync");
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1e6 / iters as f64
}

fn report(name: &str, shape: &[usize], perm: &[usize]) {
    let iters = 100;
    let warmup = 20;
    let fast_us = bench_one(shape, perm, true, iters, warmup);
    let slow_us = bench_one(shape, perm, false, iters, warmup);
    let speedup = slow_us / fast_us;
    println!(
        "{:<28}  shape={:<24}  perm={:?}  fast={:>8.2} µs  slow={:>8.2} µs  speedup={:>5.2}×",
        name,
        format!("{:?}", shape),
        perm,
        fast_us,
        slow_us,
        speedup,
    );
}

#[test]
#[ignore]
fn bench_permute_fastpaths() {
    println!("\n== permute_generic fast-path micro-bench ==\n");
    println!(
        "{:<28}  {:<31}  {:<14}  {:<13}  {:<13}  {:<10}",
        "name", "shape", "perm", "fast", "slow", "speedup"
    );

    // Top-frequency tuples from zimage trace.
    report("rank2_8x3840",     &[8, 3840], &[1, 0]);
    report("rank2_3840x8",     &[3840, 8], &[1, 0]);
    report("rank2_8x10240",    &[8, 10240], &[1, 0]);
    report("rank2_10240x8",    &[10240, 8], &[1, 0]);
    report("rank2_64x64",      &[64, 64], &[1, 0]);
    report("rank2_3840x3840",  &[3840, 3840], &[1, 0]);

    // Rank-4 inner-2 swap (QK^T).
    report("rank4_zimage_qk",  &[1, 30, 1536, 128], &[0, 1, 3, 2]);
    // Heavy case — 1.4 GB at BF16. Skip in CI.
    if std::env::var("FLAME_PERMUTE_TEST_BIG").is_ok() {
        report("rank4_zimage_attn", &[1, 30, 1536, 1536], &[0, 1, 3, 2]);
    }

    println!();
}
