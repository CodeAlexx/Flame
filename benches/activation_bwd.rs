//! Microbench: fused activation backward kernels vs decomposed GpuOps.
//! Same binary, same tensors, CUDA events for timing. Reports median of
//! N iterations after warmup.
//!
//! Run:
//!   cargo run --features cuda --release --bench activation_bwd

#![cfg(feature = "cuda")]

use flame_core::cuda::ffi;
use flame_core::cuda_ops::GpuOps;
use flame_core::device::CudaStreamRawPtrExt;
use flame_core::{global_cuda_device, DType, Result, Shape, Tensor};
use std::ffi::c_void;

const N: usize = 1 << 20; // 1M elements
const WARMUP: usize = 50;
const ITERS: usize = 200;

extern "C" {
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

struct Ev(*mut c_void);
impl Ev {
    fn new() -> Self {
        let mut p: *mut c_void = std::ptr::null_mut();
        unsafe {
            assert_eq!(cudaEventCreate(&mut p), 0);
        }
        Self(p)
    }
    fn record(&self) {
        unsafe {
            assert_eq!(cudaEventRecord(self.0, std::ptr::null_mut()), 0);
        }
    }
    fn sync(&self) {
        unsafe {
            assert_eq!(cudaEventSynchronize(self.0), 0);
        }
    }
    fn elapsed_us(&self, start: &Ev) -> f64 {
        let mut ms: f32 = 0.0;
        unsafe {
            assert_eq!(cudaEventElapsedTime(&mut ms, start.0, self.0), 0);
        }
        ms as f64 * 1000.0
    }
}
impl Drop for Ev {
    fn drop(&mut self) {
        unsafe {
            cudaEventDestroy(self.0);
        }
    }
}

fn sync() {
    unsafe {
        assert_eq!(cudaDeviceSynchronize(), 0);
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    sync();
    let s = Ev::new();
    let e = Ev::new();
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        sync();
        s.record();
        f();
        e.record();
        e.sync();
        times.push(e.elapsed_us(&s));
    }
    median(&mut times)
}

fn bf16_ptr(t: &Tensor) -> *const c_void {
    t.as_device_ptr_bf16("bench").unwrap() as *const c_void
}

fn bf16_mut_ptr(t: &mut Tensor) -> *mut c_void {
    t.as_mut_device_ptr_bf16("bench").unwrap() as *mut c_void
}

fn run_one<FDec, FFused>(name: &str, mut decomposed: FDec, mut fused: FFused)
where
    FDec: FnMut(),
    FFused: FnMut(),
{
    let t_dec = time(&mut decomposed);
    let t_fused = time(&mut fused);
    let speedup = t_dec / t_fused;
    println!(
        "{name:>8}:  decomposed {:>8.2} µs   fused {:>8.2} µs   → {:>5.2}x",
        t_dec, t_fused, speedup
    );
}

fn main() -> Result<()> {
    let dev = global_cuda_device();
    let shape = Shape::from_dims(&[N]);

    let x_f32 = Tensor::randn(shape.clone(), 0.0, 1.0, dev.clone())?;
    let g_f32 = Tensor::randn(shape.clone(), 0.0, 1.0, dev.clone())?;
    let x = x_f32.to_dtype(DType::BF16)?;
    let g = g_f32.to_dtype(DType::BF16)?;
    let stream = dev.cuda_stream_raw_ptr();
    let n = N as i64;

    println!("\nActivation backward microbench  (BF16, N={}, warmup={}, iters={})", N, WARMUP, ITERS);
    println!("------------------------------------------------------------------------");

    // ReLU: decomposed = gt(0) + mul;  fused = 1 kernel
    {
        let zero = Tensor::zeros_dtype(shape.clone(), DType::BF16, dev.clone())?;
        let decomposed = || {
            let mask = x.gt(&zero).unwrap();
            let _ = GpuOps::mul(&g, &mask).unwrap();
        };
        let fused = || {
            let mut out = Tensor::empty_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let status = unsafe {
                ffi::flame_relu_backward_bf16(
                    bf16_ptr(&g),
                    bf16_ptr(&x),
                    bf16_mut_ptr(&mut out),
                    n,
                    stream,
                )
            };
            assert_eq!(status, 0);
        };
        run_one("ReLU", decomposed, fused);
    }

    // GELU tanh-approx: decomposed ≈ 12 GpuOps;  fused = 1 kernel
    {
        let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
        let decomposed = || {
            let x2 = GpuOps::mul(&x, &x).unwrap();
            let x3 = GpuOps::mul(&x2, &x).unwrap();
            let inner = GpuOps::add(&x, &GpuOps::mul_scalar(&x3, 0.044715).unwrap()).unwrap();
            let k = GpuOps::mul_scalar(&inner, c0).unwrap();
            let tanh_k = GpuOps::tanh(&k).unwrap();
            let one_plus_tanh = GpuOps::add_scalar(&tanh_k, 1.0).unwrap();
            let tanh_k_sq = GpuOps::mul(&tanh_k, &tanh_k).unwrap();
            let ones = Tensor::ones_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let sech2 = GpuOps::add(&ones, &GpuOps::mul_scalar(&tanh_k_sq, -1.0).unwrap()).unwrap();
            let dk_dx_inner = GpuOps::add_scalar(&GpuOps::mul_scalar(&x2, 3.0 * 0.044715).unwrap(), 1.0).unwrap();
            let dk_dx = GpuOps::mul_scalar(&dk_dx_inner, c0).unwrap();
            let term2 = GpuOps::mul(&GpuOps::mul(&x, &sech2).unwrap(), &dk_dx).unwrap();
            let derivative = GpuOps::mul_scalar(&GpuOps::add(&one_plus_tanh, &term2).unwrap(), 0.5).unwrap();
            let _ = GpuOps::mul(&g, &derivative).unwrap();
        };
        let fused = || {
            let mut out = Tensor::empty_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let status = unsafe {
                ffi::flame_gelu_backward_bf16(
                    bf16_ptr(&g),
                    bf16_ptr(&x),
                    bf16_mut_ptr(&mut out),
                    n,
                    stream,
                )
            };
            assert_eq!(status, 0);
        };
        run_one("GELU", decomposed, fused);
    }

    // Tanh: decomposed = tanh + mul + mul_scalar + add;  fused = 1 kernel
    {
        let decomposed = || {
            let tanh_x = GpuOps::tanh(&x).unwrap();
            let tanh_sq = GpuOps::mul(&tanh_x, &tanh_x).unwrap();
            let ones = Tensor::ones_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let deriv = GpuOps::add(&ones, &GpuOps::mul_scalar(&tanh_sq, -1.0).unwrap()).unwrap();
            let _ = GpuOps::mul(&g, &deriv).unwrap();
        };
        let fused = || {
            let mut out = Tensor::empty_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let status = unsafe {
                ffi::flame_tanh_backward_bf16(
                    bf16_ptr(&g),
                    bf16_ptr(&x),
                    bf16_mut_ptr(&mut out),
                    n,
                    stream,
                )
            };
            assert_eq!(status, 0);
        };
        run_one("Tanh", decomposed, fused);
    }

    // Sigmoid: decomposed = sigmoid + mul_scalar + add + mul + mul;  fused = 1
    {
        let decomposed = || {
            let sig = GpuOps::sigmoid(&x).unwrap();
            let ones = Tensor::ones_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let one_minus_sig = GpuOps::add(&ones, &GpuOps::mul_scalar(&sig, -1.0).unwrap()).unwrap();
            let deriv = GpuOps::mul(&sig, &one_minus_sig).unwrap();
            let _ = GpuOps::mul(&g, &deriv).unwrap();
        };
        let fused = || {
            let mut out = Tensor::empty_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let status = unsafe {
                ffi::flame_sigmoid_backward_bf16(
                    bf16_ptr(&g),
                    bf16_ptr(&x),
                    bf16_mut_ptr(&mut out),
                    n,
                    stream,
                )
            };
            assert_eq!(status, 0);
        };
        run_one("Sigmoid", decomposed, fused);
    }

    // SiLU: decomposed = sigmoid + mul_scalar + add + mul + mul + add + mul (reference);
    // fused = existing flame_silu_backward_bf16
    {
        let decomposed = || {
            let sig = GpuOps::sigmoid(&x).unwrap();
            let ones = Tensor::ones_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let one_minus_sig = GpuOps::add(&ones, &GpuOps::mul_scalar(&sig, -1.0).unwrap()).unwrap();
            let x_times_omsig = GpuOps::mul(&x, &one_minus_sig).unwrap();
            let inner = GpuOps::add(&ones, &x_times_omsig).unwrap();
            let deriv = GpuOps::mul(&sig, &inner).unwrap();
            let _ = GpuOps::mul(&g, &deriv).unwrap();
        };
        let fused = || {
            let mut out = Tensor::empty_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
            let status = unsafe {
                ffi::flame_silu_backward_bf16(
                    bf16_ptr(&g),
                    bf16_ptr(&x),
                    bf16_mut_ptr(&mut out),
                    n,
                    stream,
                )
            };
            assert_eq!(status, 0);
        };
        run_one("SiLU", decomposed, fused);
    }

    println!();
    Ok(())
}
