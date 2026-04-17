#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Phase-1.5 torch-reference parity test: the new FA2 forward kernel vs.
//! PyTorch's CUTLASS `scaled_dot_product_flash_attention`.
//!
//! The existing `fa2_parity.rs` compares the new kernel against the preserved
//! legacy BQ=32 WMMA kernel. Both share BKV=64 K-accumulation order, so that
//! test can pass bit-exact without proving numerical correctness end-to-end.
//! This test brings in an independent reference (libtorch's flash SDPA) with a
//! different tile layout and accumulation order. If the outputs agree to within
//! the BF16 noise floor, the parity gate is real.
//!
//! Tolerance: max_abs ≤ 1e-2, max_rel ≤ 5e-3 (same as legacy parity test).
//!
//! If libtorch isn't available on this machine, the test skips gracefully —
//! libtorch is an optional runtime dependency of flame-core and we must not
//! take down CI on boxes where it isn't installed.

use anyhow::Result;
use flame_core::{torch_sdpa, DType, Device, Shape, Tensor};

/// Probe whether the full libtorch chain (c10 → c10_cuda → torch_cpu → torch_cuda)
/// can be loaded in this environment. Runs in a subprocess so a symbol-lookup
/// error aborting the probe does NOT take down the test binary.
///
/// Returns false on any load failure — including the known
/// `cudaGetDriverEntryPointByVersion` / `__nvJitLinkCreate_12_8` mismatches.
fn probe_libtorch_loadable() -> bool {
    // Use Python's dlopen (ctypes) with RTLD_GLOBAL, matching what torch_sdpa.rs
    // does internally. If any of the four libs fail, print "fail" on stdout.
    let script = r#"
import ctypes, sys
lib_dir = '/home/alex/.local/lib/python3.12/site-packages/torch/lib'
for dep in ['libc10.so','libc10_cuda.so','libtorch_cpu.so','libtorch_cuda.so']:
    try:
        ctypes.CDLL(f'{lib_dir}/{dep}', mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f'fail:{dep}:{e}')
        sys.exit(1)
print('ok')
"#;
    let out = match std::process::Command::new("python3")
        .args(["-c", script])
        .output()
    {
        Ok(o) => o,
        Err(_) => return false,
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() || !stdout.trim_end().ends_with("ok") {
        eprintln!(
            "[probe] libtorch dlopen chain failed — stdout: {}, stderr: {}",
            stdout.trim(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        return false;
    }
    true
}

/// Launch the new FA2 forward kernel directly via FFI. Mirrors the helper in
/// `fa2_parity.rs` but returns the 3D `[B*H, N, D]` output we need before
/// reshaping for the torch comparison.
fn launch_fa2(q_3d: &Tensor, k_3d: &Tensor, v_3d: &Tensor) -> Result<Tensor> {
    let dims = q_3d.shape().dims();
    let bh = dims[0] as i32;
    let sq = dims[1] as i32;
    let hd = dims[2] as i32;
    let sk = k_3d.shape().dims()[1] as i32;

    let out = Tensor::empty_dtype(
        Shape::from_dims(&[dims[0], dims[1], dims[2]]),
        DType::BF16,
        q_3d.device().clone(),
    )?;

    let q_ptr = q_3d.as_device_ptr_bf16("fa2_parity_torch:q")? as *const core::ffi::c_void;
    let k_ptr = k_3d.as_device_ptr_bf16("fa2_parity_torch:k")? as *const core::ffi::c_void;
    let v_ptr = v_3d.as_device_ptr_bf16("fa2_parity_torch:v")? as *const core::ffi::c_void;
    let o_ptr = out.as_device_ptr_bf16("fa2_parity_torch:o")? as *mut core::ffi::c_void;

    let stream = flame_core::cuda::device_lt::stream_ptr(q_3d.device())
        .map_err(|e| anyhow::anyhow!("stream_ptr: {e:?}"))?;
    let ret = unsafe {
        flame_core::cuda::ffi::flame_flash_attention_bf16(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            core::ptr::null_mut(),
            bh,
            sq,
            sk,
            hd,
            stream,
        )
    };
    if ret != 0 {
        anyhow::bail!("FA2 kernel returned nonzero: {ret}");
    }
    q_3d.device()
        .synchronize()
        .map_err(|e| anyhow::anyhow!("device synchronize: {e:?}"))?;
    Ok(out)
}

/// Per-config diff report. Never asserts — returns data so the driver test can
/// print the full 6-row table before asserting.
#[derive(Debug)]
struct DiffReport {
    seq_len: usize,
    head_dim: usize,
    max_abs: f32,
    max_rel: f32,
    passed: bool,
}

fn run_case(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    abs_tol: f32,
    rel_tol: f32,
) -> Result<DiffReport> {
    // FA2 kernel consumes 3D [B*H, N, D] with B=1, so B*H == num_heads.
    let bh = num_heads;
    let q_3d = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let k_3d = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let v_3d = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    // FA2 forward (new kernel).
    let out_fa2_3d = launch_fa2(&q_3d, &k_3d, &v_3d)?;

    // Torch SDPA requires 4D [B,H,S,D] — reshape to [1,H,S,D] so both kernels
    // see bit-identical inputs (same logical layout, and reshape is a metadata
    // op in flame-core so no data movement).
    let q_4d = q_3d.reshape(&[1, num_heads, seq_len, head_dim])?;
    let k_4d = k_3d.reshape(&[1, num_heads, seq_len, head_dim])?;
    let v_4d = v_3d.reshape(&[1, num_heads, seq_len, head_dim])?;

    let out_torch_4d = torch_sdpa::torch_flash_sdpa(&q_4d, &k_4d, &v_4d)?;

    // Flatten torch output to the FA2 layout for comparison.
    let fa2_f32 = out_fa2_3d.to_dtype(DType::F32)?.to_vec_f32()?;
    let torch_f32 = out_torch_4d.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(
        fa2_f32.len(),
        torch_f32.len(),
        "FA2 vs torch output length mismatch: {} vs {}",
        fa2_f32.len(),
        torch_f32.len()
    );

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut nan_inf = 0usize;
    for (a, b) in fa2_f32.iter().zip(torch_f32.iter()) {
        if !a.is_finite() || !b.is_finite() {
            nan_inf += 1;
            continue;
        }
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        let denom = b.abs().max(1e-6);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert_eq!(nan_inf, 0, "non-finite outputs: fa2 or torch produced NaN/Inf");

    let passed = max_abs <= abs_tol && max_rel <= rel_tol;
    Ok(DiffReport {
        seq_len,
        head_dim,
        max_abs,
        max_rel,
        passed,
    })
}

#[test]
fn fa2_matches_torch_flash_sdpa() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };
    let arc = device.cuda_device().clone();

    // Graceful skip: libtorch is an optional runtime dep. Some environments
    // have mismatched libtorch / CUDA driver / nvJitLink versions which cause
    // loading libtorch_cuda.so (or one of its transitively-loaded .so's) to
    // abort the entire process with "symbol lookup error" before `is_available`
    // can even return. Detect that case in a subprocess so we can skip cleanly
    // in-process.
    if !probe_libtorch_loadable() {
        eprintln!(
            "[skip] libtorch cannot be loaded in this env (symbol mismatch against system \
             CUDA/nvJitLink). Skipping torch parity — legacy parity (fa2_parity.rs) still ran."
        );
        return Ok(());
    }
    if !torch_sdpa::is_available() {
        eprintln!(
            "[skip] libtorch not linked, skipping torch parity — legacy parity (fa2_parity.rs) still ran"
        );
        return Ok(());
    }

    const ABS_TOL: f32 = 1e-2;
    const REL_TOL: f32 = 5e-3;

    // 6-config matrix from PHASE1_VERIFY_PARITY_PROMPT.md:
    //   seq_len ∈ {512, 4096, 16384}, head_dim ∈ {64, 128}, H=8, B=1.
    let seq_lens = [512usize, 4096, 16384];
    let head_dims = [64usize, 128];
    let num_heads = 8usize;

    let mut reports: Vec<DiffReport> = Vec::with_capacity(seq_lens.len() * head_dims.len());
    let mut torch_unavailable = false;

    'outer: for &n in &seq_lens {
        for &d in &head_dims {
            match run_case(&arc, num_heads, n, d, ABS_TOL, REL_TOL) {
                Ok(r) => reports.push(r),
                Err(e) => {
                    // Distinguish "libtorch call itself failed" from "real diff".
                    // If the FA2 launch or torch bridge errors (e.g. libtorch
                    // symbol mismatch hit at call time rather than dlopen time),
                    // downgrade to a skip rather than failing the suite.
                    let msg = format!("{e}");
                    if msg.contains("PyTorch flash SDPA")
                        || msg.contains("aoti_torch")
                        || msg.contains("libtorch")
                        || msg.contains("cudaGetDriverEntryPoint")
                    {
                        eprintln!(
                            "[skip] libtorch call failed at runtime (N={n}, D={d}): {msg} — \
                             skipping torch parity; legacy parity still ran"
                        );
                        torch_unavailable = true;
                        break 'outer;
                    }
                    return Err(e);
                }
            }
        }
    }

    if torch_unavailable {
        return Ok(());
    }

    // Print all 6 rows regardless of pass/fail — the spec requires visible
    // per-config results.
    eprintln!("FA2 vs torch parity:");
    for r in &reports {
        let status = if r.passed { "PASS" } else { "FAIL" };
        eprintln!(
            "  N={:<5} HD={:<3}  max_abs={:.3e}  max_rel={:.3e}  {status}",
            r.seq_len, r.head_dim, r.max_abs, r.max_rel
        );
    }

    let any_fail = reports.iter().any(|r| !r.passed);
    if any_fail {
        anyhow::bail!(
            "FA2 vs torch parity failed (tol: abs ≤ {ABS_TOL:.1e}, rel ≤ {REL_TOL:.1e}). \
             See table above for per-config results."
        );
    }
    Ok(())
}
