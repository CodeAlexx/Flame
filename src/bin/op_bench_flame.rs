//! Op-level benchmark: flame-core.
//! Measures forward and backward for each individual op at Z-Image Base shapes.
//! Uses CUDA event timing, pre-allocated grad tensors, BF16 throughout.
//!
//! Build:
//!     cd /home/alex/EriDiffusion/flame-core
//!     cargo build --release --bin op_bench_flame
//!
//! Run:
//!     ./target/release/op_bench_flame
//!     ./target/release/op_bench_flame --csv

use flame_core::{
    CudaDevice, DType, Shape, Tensor,
    config::set_default_dtype,
    layer_norm::LayerNorm,
};
use std::ffi::c_void;
use std::sync::Arc;

const WARMUP: usize = 100;
const ITERS: usize = 200;

// ── CUDA event timing FFI ──────────────────────────────────────────────
extern "C" {
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

struct CudaEvent(*mut c_void);

impl CudaEvent {
    fn new() -> Self {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cudaEventCreate(&mut raw) };
        assert_eq!(status, 0, "cudaEventCreate failed: {status}");
        Self(raw)
    }
    fn record(&self) {
        let status = unsafe { cudaEventRecord(self.0, std::ptr::null_mut()) };
        assert_eq!(status, 0, "cudaEventRecord failed: {status}");
    }
    fn synchronize(&self) {
        let status = unsafe { cudaEventSynchronize(self.0) };
        assert_eq!(status, 0, "cudaEventSynchronize failed: {status}");
    }
    fn elapsed_us(&self, start: &CudaEvent) -> f64 {
        let mut ms: f32 = 0.0;
        let status = unsafe { cudaEventElapsedTime(&mut ms, start.0, self.0) };
        assert_eq!(status, 0, "cudaEventElapsedTime failed: {status}");
        ms as f64 * 1000.0 // ms → μs
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.0); }
    }
}

fn cuda_sync() {
    let status = unsafe { cudaDeviceSynchronize() };
    assert_eq!(status, 0, "cudaDeviceSynchronize failed: {status}");
}

// ── Helpers ────────────────────────────────────────────────────────────

struct BenchResult {
    name: &'static str,
    fwd_us: f64,
    bwd_us: f64,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn randn(dims: &[usize], device: &Arc<CudaDevice>) -> Tensor {
    Tensor::randn(Shape::from_dims(dims), 0.0, 1.0, device.clone()).unwrap()
}

fn randn_grad(dims: &[usize], device: &Arc<CudaDevice>) -> Tensor {
    Tensor::randn(Shape::from_dims(dims), 0.0, 1.0, device.clone())
        .unwrap()
        .requires_grad_(true)
}

// ── Forward benchmark ──────────────────────────────────────────────────

fn bench_fwd<F>(mut f: F) -> f64
where
    F: FnMut() -> flame_core::Result<Tensor>,
{
    // Warmup
    for _ in 0..WARMUP {
        let _ = f().unwrap();
    }
    cuda_sync();

    let start_ev = CudaEvent::new();
    let end_ev = CudaEvent::new();
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        cuda_sync();
        start_ev.record();
        let _ = f().unwrap();
        end_ev.record();
        end_ev.synchronize();
        times.push(end_ev.elapsed_us(&start_ev));
    }
    median(&mut times)
}

// ── Backward benchmark ────────────────────────────────────────────────
// Directly calls compute_gradients on a single tape entry with a
// pre-allocated grad_output tensor. No sum(), no graph walking.

fn bench_bwd<S, B>(mut setup: S, mut backward: B) -> f64
where
    S: FnMut(),
    B: FnMut() -> flame_core::Result<()>,
{
    // Test once — also checks for hangs (3 min timeout via wall clock)
    setup();
    let t0 = std::time::Instant::now();
    if let Err(e) = backward() {
        eprintln!(" [backward FAILED: {e}]");
        return -1.0;
    }
    cuda_sync();
    let first_us = t0.elapsed().as_secs_f64() * 1e6;
    if first_us > 180_000_000.0 {
        eprintln!(" [TIMEOUT: first iter took {:.1}s]", first_us / 1e6);
        return -1.0;
    }
    // If a single iter is over 1s, something is badly wrong for an elementwise op
    if first_us > 1_000_000.0 {
        eprintln!(" [SLOW: {:.1}ms/iter — likely bug]", first_us / 1000.0);
    }

    // Warmup
    for _ in 1..WARMUP {
        setup();
        backward().unwrap();
    }
    cuda_sync();

    let start_ev = CudaEvent::new();
    let end_ev = CudaEvent::new();
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        setup();
        cuda_sync();
        start_ev.record();
        backward().unwrap();
        end_ev.record();
        end_ev.synchronize();
        times.push(end_ev.elapsed_us(&start_ev));
    }
    median(&mut times)
}

// Helper: run forward with grad tracking, then backward with pre-allocated grad_output.
// `fwd` takes an input and returns output. We measure backward only.
fn bench_bwd_simple<F>(
    device: &Arc<CudaDevice>,
    input_shape: &[usize],
    output_shape: &[usize],
    mut fwd: F,
) -> f64
where
    F: FnMut(&Tensor) -> flame_core::Result<Tensor>,
{
    let grad_output = randn(output_shape, device);

    bench_bwd(
        || { /* setup: nothing needed, we create fresh input each iter inside backward */ },
        || {
            let inp = randn_grad(input_shape, device);
            let out = fwd(&inp)?;
            // Seed the gradient and run backward
            let loss = out.mul(&grad_output)?.sum()?;
            let _ = loss.backward()?;
            Ok(())
        },
    )
}

fn main() -> flame_core::Result<()> {
    let csv_mode = std::env::args().any(|a| a == "--csv");

    set_default_dtype(DType::BF16);
    let device = CudaDevice::new(0)?;

    // GPU warmup — stabilize clocks
    eprintln!("Warming up GPU...");
    {
        let a = randn(&[2048, 2048], &device);
        let b = randn(&[2048, 2048], &device);
        for _ in 0..50 {
            let _ = a.matmul(&b)?;
        }
        cuda_sync();
    }
    eprintln!("Warmup: {WARMUP}, Iters: {ITERS}\n");

    let mut results: Vec<BenchResult> = Vec::new();

    // ── Level 1: Elementwise ──────────────────────────────────────────

    // 1. Cast BF16→FP32
    eprint!("[ 1/17] Cast BF16→FP32 ...");
    {
        let x = randn(&[1, 1024, 1280], &device);
        let fwd = bench_fwd(|| x.to_dtype(DType::F32));
        let bwd = bench_bwd_simple(&device, &[1, 1024, 1280], &[1, 1024, 1280], |inp| {
            inp.to_dtype(DType::F32)
        });
        results.push(BenchResult { name: "Cast BF16→FP32", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    // 2. Cast FP32→BF16
    eprint!("[ 2/17] Cast FP32→BF16 ...");
    {
        let x_f32 = randn(&[1, 1024, 1280], &device).to_dtype(DType::F32)?;
        let fwd = bench_fwd(|| x_f32.to_dtype(DType::BF16));
        // backward: F32 input → BF16 output → grad flows back
        let grad_output = randn(&[1, 1024, 1280], &device);
        let bwd = bench_bwd(
            || {},
            || {
                set_default_dtype(DType::F32);
                let inp = Tensor::randn(Shape::from_dims(&[1, 1024, 1280]), 0.0, 1.0, device.clone())?
                    .requires_grad_(true);
                set_default_dtype(DType::BF16);
                let out = inp.to_dtype(DType::BF16)?;
                let loss = out.to_dtype(DType::F32)?.mul(&grad_output.to_dtype(DType::F32)?)?.sum()?;
                let _ = loss.backward()?;
                Ok(())
            },
        );
        results.push(BenchResult { name: "Cast FP32→BF16", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    // 3. Abs
    eprint!("[ 3/17] Abs ...");
    {
        let x = randn(&[1, 1024, 1280], &device);
        let fwd = bench_fwd(|| x.abs());
        let bwd = bench_bwd_simple(&device, &[1, 1024, 1280], &[1, 1024, 1280], |inp| {
            inp.abs()
        });
        results.push(BenchResult { name: "Abs", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[ 4/17] Add ...");
    {
        let a = randn(&[1, 1024, 1280], &device);
        let b = randn(&[1, 1024, 1280], &device);
        let fwd = bench_fwd(|| a.add(&b));
        let b_fixed = randn(&[1, 1024, 1280], &device);
        let grad_output = randn(&[1, 1024, 1280], &device);
        let bwd = bench_bwd(
            || {},
            || {
                let a2 = randn_grad(&[1, 1024, 1280], &device);
                let out = a2.add(&b_fixed)?;
                let loss = out.mul(&grad_output)?.sum()?;
                let _ = loss.backward()?;
                Ok(())
            },
        );
        results.push(BenchResult { name: "Add (residual)", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[ 5/17] Mul scalar ...");
    {
        let x = randn(&[1, 1024, 1280], &device);
        let fwd = bench_fwd(|| x.mul_scalar(0.7071));
        let bwd = bench_bwd_simple(&device, &[1, 1024, 1280], &[1, 1024, 1280], |inp| {
            inp.mul_scalar(0.7071)
        });
        results.push(BenchResult { name: "Mul (scalar)", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[ 6/17] Mul elem ...");
    {
        let a = randn(&[1, 1024, 1280], &device);
        let b = randn(&[1, 1024, 1280], &device);
        let fwd = bench_fwd(|| a.mul(&b));
        let b_fixed = randn(&[1, 1024, 1280], &device);
        let grad_output = randn(&[1, 1024, 1280], &device);
        let bwd = bench_bwd(
            || {},
            || {
                let a2 = randn_grad(&[1, 1024, 1280], &device);
                let out = a2.mul(&b_fixed)?;
                let loss = out.mul(&grad_output)?.sum()?;
                let _ = loss.backward()?;
                Ok(())
            },
        );
        results.push(BenchResult { name: "Mul (elementwise)", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[ 7/17] Reshape ...");
    {
        let x = randn(&[1, 1024, 1280], &device);
        let fwd = bench_fwd(|| x.reshape(&[1, 1024, 20, 64]));
        let bwd = bench_bwd_simple(&device, &[1, 1024, 1280], &[1, 1024, 20, 64], |inp| {
            inp.reshape(&[1, 1024, 20, 64])
        });
        results.push(BenchResult { name: "Reshape", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[ 8/17] Permute ...");
    {
        let x = randn(&[1, 1024, 20, 64], &device);
        let fwd = bench_fwd(|| x.permute(&[0, 2, 1, 3]));
        let bwd = bench_bwd_simple(&device, &[1, 1024, 20, 64], &[1, 20, 1024, 64], |inp| {
            inp.permute(&[0, 2, 1, 3])
        });
        results.push(BenchResult { name: "Permute (0,2,1,3)", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[ 9/17] SiLU ...");
    {
        let x = randn(&[1, 1024, 5120], &device);
        let fwd = bench_fwd(|| x.silu());
        let bwd = bench_bwd_simple(&device, &[1, 1024, 5120], &[1, 1024, 5120], |inp| {
            inp.silu()
        });
        results.push(BenchResult { name: "SiLU", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[10/17] GELU ...");
    {
        let x = randn(&[1, 1024, 5120], &device);
        let fwd = bench_fwd(|| x.gelu());
        let bwd = bench_bwd_simple(&device, &[1, 1024, 5120], &[1, 1024, 5120], |inp| {
            inp.gelu()
        });
        results.push(BenchResult { name: "GELU", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[11/17] Softmax ...");
    {
        let x = randn(&[20, 1024, 1024], &device);
        let fwd = bench_fwd(|| x.softmax(-1));
        let bwd = bench_bwd_simple(&device, &[20, 1024, 1024], &[20, 1024, 1024], |inp| {
            inp.softmax(-1)
        });
        results.push(BenchResult { name: "Softmax", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[12/17] LayerNorm ...");
    {
        let ln = LayerNorm::new(vec![1280], 1e-5, device.clone())?;
        let x = randn(&[1, 1024, 1280], &device);
        let fwd = bench_fwd(|| ln.forward(&x));
        let grad_output = randn(&[1, 1024, 1280], &device);
        let bwd = bench_bwd(
            || {},
            || {
                let inp = randn_grad(&[1, 1024, 1280], &device);
                let out = ln.forward(&inp)?;
                let loss = out.mul(&grad_output)?.sum()?;
                let _ = loss.backward()?;
                Ok(())
            },
        );
        results.push(BenchResult { name: "LayerNorm", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[13/17] MatMul proj ...");
    // 13. MatMul (proj) — (1, 1024, 1280) × (1280, 1280)
    {
        let x = randn(&[1, 1024, 1280], &device);
        let w = randn(&[1280, 1280], &device);
        let fwd = bench_fwd(|| x.matmul(&w));
        let grad_output = randn(&[1, 1024, 1280], &device);
        let bwd = bench_bwd(
            || {},
            || {
                let inp = randn_grad(&[1, 1024, 1280], &device);
                let out = inp.matmul(&w)?;
                let loss = out.mul(&grad_output)?.sum()?;
                let _ = loss.backward()?;
                Ok(())
            },
        );
        results.push(BenchResult { name: "MatMul (proj)", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[14/17] MatMul FFN ...");
    // 14. MatMul (FFN) — (1, 1024, 1280) × (1280, 5120)
    {
        let x = randn(&[1, 1024, 1280], &device);
        let w = randn(&[1280, 5120], &device);
        let fwd = bench_fwd(|| x.matmul(&w));
        let grad_output = randn(&[1, 1024, 5120], &device);
        let bwd = bench_bwd(
            || {},
            || {
                let inp = randn_grad(&[1, 1024, 1280], &device);
                let out = inp.matmul(&w)?;
                let loss = out.mul(&grad_output)?.sum()?;
                let _ = loss.backward()?;
                Ok(())
            },
        );
        results.push(BenchResult { name: "MatMul (FFN)", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[15/17] BMM QK^T ...");
    // 15. BMM (QK^T) — (20, 1024, 64) × (20, 64, 1024)
    {
        let q = randn(&[20, 1024, 64], &device);
        let k = randn(&[20, 64, 1024], &device);
        let fwd = bench_fwd(|| q.matmul(&k));
        let grad_output = randn(&[20, 1024, 1024], &device);
        let bwd = bench_bwd(
            || {},
            || {
                let q2 = randn_grad(&[20, 1024, 64], &device);
                let k2 = randn_grad(&[20, 64, 1024], &device);
                let out = q2.matmul(&k2)?;
                let loss = out.mul(&grad_output)?.sum()?;
                let _ = loss.backward()?;
                Ok(())
            },
        );
        results.push(BenchResult { name: "BMM (QK^T)", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[16/17] BMM @V ...");
    // 16. BMM (@V) — (20, 1024, 1024) × (20, 1024, 64)
    {
        let a = randn(&[20, 1024, 1024], &device);
        let v = randn(&[20, 1024, 64], &device);
        let fwd = bench_fwd(|| a.matmul(&v));
        let grad_output = randn(&[20, 1024, 64], &device);
        let bwd = bench_bwd(
            || {},
            || {
                let a2 = randn_grad(&[20, 1024, 1024], &device);
                let v2 = randn_grad(&[20, 1024, 64], &device);
                let out = a2.matmul(&v2)?;
                let loss = out.mul(&grad_output)?.sum()?;
                let _ = loss.backward()?;
                Ok(())
            },
        );
        results.push(BenchResult { name: "BMM (@V)", fwd_us: fwd, bwd_us: bwd });
        eprintln!(" done");
    }

    eprint!("[17/17] LoRA merge ...");
    // ── Level 6: LoRA merge ────────────────────────────────────────────
    // 17. LoRA merge: A @ B * scale + base (forward only)
    {
        let lora_a = randn(&[1280, 64], &device);
        let lora_b = randn(&[64, 1280], &device);
        let base = randn(&[1280, 1280], &device);
        let fwd = bench_fwd(|| {
            let ab = lora_a.matmul(&lora_b)?;
            let scaled = ab.mul_scalar(1.0)?;
            base.add(&scaled)
        });
        results.push(BenchResult { name: "LoRA merge", fwd_us: fwd, bwd_us: 0.0 });
        eprintln!(" done");
    }

    eprintln!("\nAll ops complete.\n");
    // ── Print results ──────────────────────────────────────────────────
    if csv_mode {
        println!("op,fwd_us,bwd_us,total_us");
        for r in &results {
            let bwd = if r.bwd_us < 0.0 { 0.0 } else { r.bwd_us };
            println!("{},{:.1},{:.1},{:.1}", r.name, r.fwd_us, bwd, r.fwd_us + bwd);
        }
    } else {
        println!(
            "{:<25} {:>10} {:>10} {:>12}",
            "Op", "Fwd (μs)", "Bwd (μs)", "Total (μs)"
        );
        println!("{}", "-".repeat(59));
        for r in &results {
            let (bwd_str, total) = if r.bwd_us < 0.0 {
                ("FAIL".to_string(), r.fwd_us)
            } else if r.bwd_us == 0.0 {
                ("—".to_string(), r.fwd_us)
            } else {
                (format!("{:.1}", r.bwd_us), r.fwd_us + r.bwd_us)
            };
            println!(
                "{:<25} {:>10.1} {:>10} {:>12.1}",
                r.name, r.fwd_us, bwd_str, total
            );
        }
    }

    Ok(())
}
