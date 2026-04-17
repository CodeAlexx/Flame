#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Phase-1 parity test: new FA2 forward kernel vs. the preserved legacy
//! BQ=32 kernel. The legacy FFI symbol `flame_flash_attention_bf16_wmma_legacy`
//! is exposed from `flash_attention_fwd_legacy.cu` specifically for this
//! comparison and will be deleted in Phase 3.
//!
//! Correctness gate: both kernels produce identical BF16 outputs within the
//! BF16 noise floor (max abs diff ≤ 1e-2, max rel diff ≤ 5e-3).

use anyhow::Result;
use flame_core::{DType, Device, Shape, Tensor};

/// Sample a tensor from N(0, 1). default_dtype may be BF16, so force F32
/// explicitly by constructing from a CPU F32 vector then uploading.
fn randn_f32(shape: &[usize], device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<Tensor> {
    Ok(Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, device.clone())?)
}

/// Raw launch of either forward variant. Returns the BF16 output tensor of
/// shape [bh, seq, hd].
fn launch_forward(
    q_3d: &Tensor,
    k_3d: &Tensor,
    v_3d: &Tensor,
    use_legacy: bool,
) -> Result<Tensor> {
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

    let q_ptr = q_3d.as_device_ptr_bf16("fa2_parity:q")? as *const core::ffi::c_void;
    let k_ptr = k_3d.as_device_ptr_bf16("fa2_parity:k")? as *const core::ffi::c_void;
    let v_ptr = v_3d.as_device_ptr_bf16("fa2_parity:v")? as *const core::ffi::c_void;
    let o_ptr = out.as_device_ptr_bf16("fa2_parity:o")? as *mut core::ffi::c_void;

    let stream = flame_core::cuda::device_lt::stream_ptr(q_3d.device())
        .map_err(|e| anyhow::anyhow!("stream_ptr: {e:?}"))?;
    let ret = unsafe {
        if use_legacy {
            flame_core::cuda::ffi::flame_flash_attention_bf16_wmma_legacy(
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
        } else {
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
        }
    };
    if ret != 0 {
        anyhow::bail!(
            "{} kernel returned nonzero: {ret}",
            if use_legacy { "legacy" } else { "FA2" }
        );
    }

    // Force completion before returning so any kernel-level fault surfaces
    // here rather than in the next tensor op.
    q_3d.device()
        .synchronize()
        .map_err(|e| anyhow::anyhow!("device synchronize: {e:?}"))?;
    Ok(out)
}

fn compare_outputs(
    new_out: &Tensor,
    legacy_out: &Tensor,
    label: &str,
    abs_tol: f32,
    rel_tol: f32,
) -> Result<()> {
    let new_f32 = new_out.to_dtype(DType::F32)?.to_vec_f32()?;
    let leg_f32 = legacy_out.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(new_f32.len(), leg_f32.len(), "[{label}] length mismatch");

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut nan_inf_count = 0usize;
    let mut worst_i: usize = 0;
    let mut worst_pair = (0f32, 0f32);

    let mut new_nonzero = 0usize;
    let mut new_min = f32::INFINITY;
    let mut new_max_val = f32::NEG_INFINITY;

    for (i, (a, b)) in new_f32.iter().zip(leg_f32.iter()).enumerate() {
        if a.abs() > 0.0 {
            new_nonzero += 1;
        }
        new_min = new_min.min(*a);
        new_max_val = new_max_val.max(*a);

        if !a.is_finite() || !b.is_finite() {
            nan_inf_count += 1;
            if nan_inf_count < 5 {
                eprintln!("[{label}] non-finite at idx {i}: new={a}, legacy={b}");
            }
            continue;
        }
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
            worst_i = i;
            worst_pair = (*a, *b);
        }
        let denom = b.abs().max(1e-6);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }

    assert_eq!(
        nan_inf_count, 0,
        "[{label}] FA2 output contains {nan_inf_count} non-finite values — kernel bug"
    );

    // Sanity: if every element is zero the test is meaningless.
    assert!(
        new_nonzero > 0,
        "[{label}] FA2 output is all zero — kernel wrote nothing"
    );

    eprintln!(
        "[{label}] max_abs={max_abs:.3e} max_rel={max_rel:.3e} worst_idx={worst_i} \
         new_stats: nonzero={new_nonzero}/{} range=[{new_min:.3e},{new_max_val:.3e}] \
         worst_pair=(new={}, leg={})",
        new_f32.len(),
        worst_pair.0,
        worst_pair.1
    );

    assert!(
        max_abs <= abs_tol,
        "[{label}] max abs diff {max_abs:.3e} exceeds tolerance {abs_tol:.3e}"
    );
    assert!(
        max_rel <= rel_tol,
        "[{label}] max rel diff {max_rel:.3e} exceeds tolerance {rel_tol:.3e}"
    );
    Ok(())
}

/// Run one shape configuration through both kernels and assert they match.
fn run_case(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<()> {
    let label = format!(
        "B={batch} H={num_heads} N={seq_len} D={head_dim}"
    );
    eprintln!("=== {label} ===");

    let bh = batch * num_heads;
    let q = randn_f32(&[bh, seq_len, head_dim], device)?.to_dtype(DType::BF16)?;
    let k = randn_f32(&[bh, seq_len, head_dim], device)?.to_dtype(DType::BF16)?;
    let v = randn_f32(&[bh, seq_len, head_dim], device)?.to_dtype(DType::BF16)?;

    let out_new = launch_forward(&q, &k, &v, false)?;
    let out_legacy = launch_forward(&q, &k, &v, true)?;

    // BF16 noise floor: 7-bit mantissa → relative accuracy ~8e-3. Absolute
    // diffs can be a bit larger on values with magnitude close to 1.
    compare_outputs(&out_new, &out_legacy, &label, 1e-2, 5e-3)?;
    Ok(())
}

#[test]
fn fa2_matches_legacy_wmma() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };
    let arc = device.cuda_device().clone();

    // Shape matrix from the Phase-1 spec.
    let head_dims = [64usize, 96, 128];
    let num_heads = [8usize, 16];
    let batches = [1usize, 2];
    let seq_lens = [128usize, 512, 1024, 4096, 16384];

    for &d in &head_dims {
        for &h in &num_heads {
            for &b in &batches {
                for &n in &seq_lens {
                    // Skip cases requiring too much GPU memory on 24 GB cards.
                    // Q+K+V+O BF16 alone is 4*(b*h*n*d*2) bytes; spare some
                    // headroom for the intermediate F32 comparison buffers.
                    let bytes_qkvo = 4 * (b * h) * n * d * 2;
                    if bytes_qkvo > 6 * 1024 * 1024 * 1024 {
                        eprintln!(
                            "[skip] B={b} H={h} N={n} D={d}: {} MiB > 6 GiB budget",
                            bytes_qkvo >> 20
                        );
                        continue;
                    }
                    run_case(&arc, b, h, n, d)?;
                }
            }
        }
    }
    Ok(())
}
