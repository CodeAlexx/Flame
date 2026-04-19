#![cfg(feature = "cuda")]

//! Direct-FFI parity tests for the fused activation backward kernels:
//! flame_{relu,gelu,tanh,sigmoid}_backward_{bf16,f32}.
//!
//! Each test:
//!  1. builds an input tensor on the device,
//!  2. calls the FFI entry point with raw device pointers + the default stream,
//!  3. compares the result against a CPU closed-form reference computed with
//!     the exact same formula the kernel implements.

use flame_core::device::CudaStreamRawPtrExt;
use flame_core::{global_cuda_device, CudaDevice, DType, Result, Shape, Tensor};
use half::bf16;
use std::sync::Arc;

fn device() -> Arc<CudaDevice> {
    global_cuda_device()
}

fn bf16_round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

// --- CPU reference implementations (must match kernel formulas) ---

fn relu_ref(x: f32, g: f32) -> f32 {
    if x > 0.0 {
        g
    } else {
        0.0
    }
}

fn gelu_tanh_ref(x: f32, g: f32) -> f32 {
    // mirrors gelu_backward.cu
    const C0: f32 = 0.7978845608028654; // sqrt(2/pi)
    const C1: f32 = 0.044715;
    let x2 = x * x;
    let x3 = x2 * x;
    let arg = C0 * (x + C1 * x3);
    let t = arg.tanh();
    let sech2 = 1.0 - t * t;
    let deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * C0 * (1.0 + 3.0 * C1 * x2);
    g * deriv
}

fn tanh_ref(x: f32, g: f32) -> f32 {
    let t = x.tanh();
    g * (1.0 - t * t)
}

fn sigmoid_ref(x: f32, g: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    g * s * (1.0 - s)
}

// --- Tolerance helpers matching tests/activation_backward.rs ---

fn tol(dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => 2e-2,
        DType::F32 => 1e-5,
        _ => 1e-5,
    }
}

// --- Raw pointer helpers ---

fn f32_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    use cudarc::driver::DevicePtr;
    let slice = t.as_slice_f32("fused_bwd_test:f32_ptr")?;
    Ok(*slice.device_ptr() as *const core::ffi::c_void)
}

fn f32_mut_ptr(t: &mut Tensor) -> Result<*mut core::ffi::c_void> {
    use cudarc::driver::DevicePtrMut;
    let slice = t.as_mut_slice_f32("fused_bwd_test:f32_mut_ptr")?;
    Ok(*slice.device_ptr_mut() as *mut core::ffi::c_void)
}

fn bf16_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    let p = t.as_device_ptr_bf16("fused_bwd_test:bf16_ptr")?;
    Ok(p as *const core::ffi::c_void)
}

fn bf16_mut_ptr(t: &mut Tensor) -> Result<*mut core::ffi::c_void> {
    let p = t.as_mut_device_ptr_bf16("fused_bwd_test:bf16_mut_ptr")?;
    Ok(p as *mut core::ffi::c_void)
}

// --- Generic parity runner ---

#[allow(clippy::too_many_arguments)]
fn run_parity_f32<F>(
    x_vals: &[f32],
    g_vals: &[f32],
    ref_fn: F,
    kernel: unsafe extern "C" fn(
        *const core::ffi::c_void,
        *const core::ffi::c_void,
        *mut core::ffi::c_void,
        i64,
        *mut core::ffi::c_void,
    ) -> i32,
    op_name: &str,
) -> Result<()>
where
    F: Fn(f32, f32) -> f32,
{
    let dev = device();
    let n = x_vals.len();
    assert_eq!(g_vals.len(), n);

    let x = Tensor::from_vec_dtype(x_vals.to_vec(), Shape::from_dims(&[n]), dev.clone(), DType::F32)?;
    let g = Tensor::from_vec_dtype(g_vals.to_vec(), Shape::from_dims(&[n]), dev.clone(), DType::F32)?;
    let mut out = Tensor::zeros_dtype(Shape::from_dims(&[n]), DType::F32, dev.clone())?;

    let stream = dev.cuda_stream_raw_ptr();
    let status = unsafe {
        kernel(
            f32_ptr(&g)?,
            f32_ptr(&x)?,
            f32_mut_ptr(&mut out)?,
            n as i64,
            stream,
        )
    };
    assert_eq!(status, 0, "{op_name} f32 kernel returned non-zero");

    let got = out.to_vec()?;
    let t = tol(DType::F32);
    let mut max_err = 0.0f32;
    for i in 0..n {
        let expected = ref_fn(x_vals[i], g_vals[i]);
        let diff = (got[i] - expected).abs();
        if diff > max_err {
            max_err = diff;
        }
        assert!(
            diff <= t,
            "{op_name} f32 idx={i} x={} g={} expected={} got={} diff={} tol={}",
            x_vals[i],
            g_vals[i],
            expected,
            got[i],
            diff,
            t
        );
    }
    println!("{op_name} f32 OK (n={n}, max_abs_err={max_err:.3e})");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_parity_bf16<F>(
    x_vals: &[f32],
    g_vals: &[f32],
    ref_fn: F,
    kernel: unsafe extern "C" fn(
        *const core::ffi::c_void,
        *const core::ffi::c_void,
        *mut core::ffi::c_void,
        i64,
        *mut core::ffi::c_void,
    ) -> i32,
    op_name: &str,
) -> Result<()>
where
    F: Fn(f32, f32) -> f32,
{
    let dev = device();
    let n = x_vals.len();
    assert_eq!(g_vals.len(), n);

    let x_f32 = Tensor::from_vec_dtype(x_vals.to_vec(), Shape::from_dims(&[n]), dev.clone(), DType::F32)?;
    let g_f32 = Tensor::from_vec_dtype(g_vals.to_vec(), Shape::from_dims(&[n]), dev.clone(), DType::F32)?;
    let x = x_f32.to_dtype(DType::BF16)?;
    let g = g_f32.to_dtype(DType::BF16)?;
    let mut out = Tensor::zeros_dtype(Shape::from_dims(&[n]), DType::BF16, dev.clone())?;

    let stream = dev.cuda_stream_raw_ptr();
    let status = unsafe {
        kernel(
            bf16_ptr(&g)?,
            bf16_ptr(&x)?,
            bf16_mut_ptr(&mut out)?,
            n as i64,
            stream,
        )
    };
    assert_eq!(status, 0, "{op_name} bf16 kernel returned non-zero");

    // Round-trip to f32 via to_dtype
    let got = out.to_dtype(DType::F32)?.to_vec()?;
    let t = tol(DType::BF16);
    let mut max_err = 0.0f32;
    for i in 0..n {
        // Reference: simulate kernel's BF16 rounding on inputs.
        let xr = bf16_round(x_vals[i]);
        let gr = bf16_round(g_vals[i]);
        let expected = bf16_round(ref_fn(xr, gr));
        let diff = (got[i] - expected).abs();
        if diff > max_err {
            max_err = diff;
        }
        let abs_tol = t.max(expected.abs() * 1e-2); // 1% relative safety
        assert!(
            diff <= abs_tol,
            "{op_name} bf16 idx={i} x={} g={} expected={} got={} diff={} tol={}",
            x_vals[i],
            g_vals[i],
            expected,
            got[i],
            diff,
            abs_tol
        );
    }
    println!("{op_name} bf16 OK (n={n}, max_abs_err={max_err:.3e})");
    Ok(())
}

fn make_inputs(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    // x spans [-6, 6]; if n==1 just 0.
    let mut x = Vec::with_capacity(n);
    if n == 1 {
        x.push(0.0);
    } else {
        let delta = 12.0 / ((n - 1) as f32);
        for i in 0..n {
            x.push(-6.0 + delta * i as f32);
        }
    }
    // g is a deterministic pseudo-random vector.
    let mut g = Vec::with_capacity(n);
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (state >> 32) as u32;
        g.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    (x, g)
}

// ---------------- ReLU ----------------

#[test]
fn relu_backward_f32_parity() -> Result<()> {
    let (x, g) = make_inputs(4096, 1);
    run_parity_f32(
        &x,
        &g,
        relu_ref,
        flame_core::cuda::ffi::flame_relu_backward_f32,
        "relu",
    )
}

#[test]
fn relu_backward_bf16_parity() -> Result<()> {
    let (x, g) = make_inputs(4096, 2);
    run_parity_bf16(
        &x,
        &g,
        relu_ref,
        flame_core::cuda::ffi::flame_relu_backward_bf16,
        "relu",
    )
}

// ---------------- GELU ----------------

#[test]
fn gelu_backward_f32_parity() -> Result<()> {
    let (x, g) = make_inputs(4096, 3);
    run_parity_f32(
        &x,
        &g,
        gelu_tanh_ref,
        flame_core::cuda::ffi::flame_gelu_backward_f32,
        "gelu",
    )
}

#[test]
fn gelu_backward_bf16_parity() -> Result<()> {
    let (x, g) = make_inputs(4096, 4);
    run_parity_bf16(
        &x,
        &g,
        gelu_tanh_ref,
        flame_core::cuda::ffi::flame_gelu_backward_bf16,
        "gelu",
    )
}

// ---------------- Tanh ----------------

#[test]
fn tanh_backward_f32_parity() -> Result<()> {
    let (x, g) = make_inputs(4096, 5);
    run_parity_f32(
        &x,
        &g,
        tanh_ref,
        flame_core::cuda::ffi::flame_tanh_backward_f32,
        "tanh",
    )
}

#[test]
fn tanh_backward_bf16_parity() -> Result<()> {
    let (x, g) = make_inputs(4096, 6);
    run_parity_bf16(
        &x,
        &g,
        tanh_ref,
        flame_core::cuda::ffi::flame_tanh_backward_bf16,
        "tanh",
    )
}

// ---------------- Sigmoid ----------------

#[test]
fn sigmoid_backward_f32_parity() -> Result<()> {
    let (x, g) = make_inputs(4096, 7);
    run_parity_f32(
        &x,
        &g,
        sigmoid_ref,
        flame_core::cuda::ffi::flame_sigmoid_backward_f32,
        "sigmoid",
    )
}

#[test]
fn sigmoid_backward_bf16_parity() -> Result<()> {
    let (x, g) = make_inputs(4096, 8);
    run_parity_bf16(
        &x,
        &g,
        sigmoid_ref,
        flame_core::cuda::ffi::flame_sigmoid_backward_bf16,
        "sigmoid",
    )
}

// ---------------- Tail / odd-n smoke ----------------

#[test]
fn tail_handling_f32() -> Result<()> {
    // n = 4097 tests the tail-guard `if (idx >= n) return` on the last warp.
    let (x, g) = make_inputs(4097, 42);
    // One kernel per op is enough; use tanh as representative.
    run_parity_f32(
        &x,
        &g,
        tanh_ref,
        flame_core::cuda::ffi::flame_tanh_backward_f32,
        "tanh_tail",
    )
}

// NOTE: deliberately no zero-n launch test.
// The current kernels compute grid = (n + 255)/256 with no guard at n==0,
// so a raw launch with n=0 yields cudaErrorInvalidConfiguration (grid=0).
// Callers (autograd backward ops) never pass n=0 because zero-element tensors
// never arrive at Op::{ReLU,GELU,Tanh,Sigmoid}. If a newer kernel drop adds
// a `if (n == 0) return 0;` guard, add a zero-n test then.
