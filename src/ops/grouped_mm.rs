//! Grouped BF16 matmul (MoE expert dispatch).
//!
//! Wraps the `flame_grouped_mm_bf16` build-time CUDA kernel
//! (`src/cuda/grouped_mm.cu`) with a Tensor-API surface. Semantics mirror
//! `torch.nn.functional.grouped_mm(x, w, offs=offsets)`:
//!
//! ```text
//! x        : (T, K)        BF16, expert-major ordered tokens
//! w        : (E, K, N)     BF16, stacked per-expert weights
//! offsets  : (E,)          I32, EXCLUSIVE cumulative end indices into x
//!                          (expert e covers rows [offsets[e-1] .. offsets[e]),
//!                          with offsets[-1] := 0)
//! returns  : (T, N)        BF16
//! ```
//!
//! For each expert e, the kernel computes
//!   `y[offsets[e-1] .. offsets[e], :] = x[offsets[e-1] .. offsets[e], :] @ w[e]`
//! with FP32 WMMA accumulation. SM80+ required (3090 Ti is SM86 — fine).
//!
//! Used by:
//! - Nucleus-Image MoE expert FFN (gate-up + down projections)
//! - LLaDA2.0-Uni MoE backbone (when wired)
//!
//! See `docs/FLAME_KERNELS.md` for the kernel design notes.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, DevicePtr};

use crate::device::CudaStreamRawPtrExt;
use crate::{DType, Error, Result, Shape, Tensor};

/// Maximum E supported by the underlying kernel (one CUDA grid `z` slot per expert).
/// CUDA's `gridDim.z` upper bound is 65535 on all current SMs; we cap conservatively
/// here so error messages are clearer than a generic launch failure.
const MAX_EXPERTS: i32 = 1024;

/// Grouped BF16 matmul.
///
/// # Arguments
/// - `x`:        `(T, K)`     BF16 tensor of expert-major-ordered tokens.
/// - `w`:        `(E, K, N)`  BF16 tensor of stacked expert weights.
/// - `offsets`:  `&[i32]` of length `E`, EXCLUSIVE cumulative end indices.
///               Host slice — the wrapper does the HtoD copy. Offsets are
///               typically tiny (E ≤ 1024) so this is microseconds vs the
///               millisecond-class GEMM. For large/in-loop usage where
///               offsets are produced GPU-side, use the `_with_dev_offsets`
///               variant below (TODO when routing lands).
/// - `t_max`:    Maximum rows assigned to any single expert (used to size
///               grid.y). For Nucleus-Image expert-choice routing this is
///               constant `B*C` (capacity); pass `T / E` for uniform splits.
///
/// # Returns
/// `(T, N)` BF16 tensor of stacked per-expert outputs.
pub fn grouped_mm_bf16(x: &Tensor, w: &Tensor, offsets: &[i32], t_max: usize) -> Result<Tensor> {
    // ---- shape + dtype validation ------------------------------------
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "grouped_mm_bf16: x must be BF16, got {:?}",
            x.dtype()
        )));
    }
    if w.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "grouped_mm_bf16: w must be BF16, got {:?}",
            w.dtype()
        )));
    }

    let x_dims = x.shape().dims();
    let w_dims = w.shape().dims();

    if x_dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "grouped_mm_bf16: x must be 2-D (T, K), got shape {:?}",
            x_dims
        )));
    }
    if w_dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "grouped_mm_bf16: w must be 3-D (E, K, N), got shape {:?}",
            w_dims
        )));
    }

    let (t, k_x) = (x_dims[0], x_dims[1]);
    let (e, k_w, n) = (w_dims[0], w_dims[1], w_dims[2]);

    if k_x != k_w {
        return Err(Error::InvalidOperation(format!(
            "grouped_mm_bf16: K mismatch — x has K={k_x}, w has K={k_w}"
        )));
    }
    if e != offsets.len() {
        return Err(Error::InvalidOperation(format!(
            "grouped_mm_bf16: E mismatch — w has E={e}, offsets.len()={}",
            offsets.len()
        )));
    }
    if e as i32 > MAX_EXPERTS {
        return Err(Error::InvalidOperation(format!(
            "grouped_mm_bf16: E={e} exceeds MAX_EXPERTS={MAX_EXPERTS} (would overflow CUDA gridDim.z)"
        )));
    }

    // T_max is supplied by the caller because it depends on routing semantics.
    // Validate it: must be <= T (each expert's row range is a sub-slice of x).
    if t_max > t {
        return Err(Error::InvalidOperation(format!(
            "grouped_mm_bf16: t_max={t_max} > T={t}"
        )));
    }

    // Final offset must equal T (every row of x is covered by some expert).
    if let Some(&last) = offsets.last() {
        if last as usize != t {
            return Err(Error::InvalidOperation(format!(
                "grouped_mm_bf16: offsets[-1]={last} must equal T={t}"
            )));
        }
    }
    // Offsets must be non-decreasing.
    for w_pair in offsets.windows(2) {
        if w_pair[1] < w_pair[0] {
            return Err(Error::InvalidOperation(format!(
                "grouped_mm_bf16: offsets must be non-decreasing, got {w_pair:?}"
            )));
        }
    }

    // x and w must be contiguous (kernel reads with vectorised loads).
    if !x.is_contiguous() {
        return Err(Error::InvalidOperation(
            "grouped_mm_bf16: x must be contiguous".into(),
        ));
    }
    if !w.is_contiguous() {
        return Err(Error::InvalidOperation(
            "grouped_mm_bf16: w must be contiguous".into(),
        ));
    }

    // ---- allocate output ---------------------------------------------
    let device: Arc<CudaDevice> = x.device().clone();
    let mut y = Tensor::empty_dtype(Shape::from_dims(&[t, n]), DType::BF16, device.clone())?;

    // ---- HtoD copy of host offsets to a temp i32 device buffer -------
    // CudaSlice<i32> stores REAL i32 bytes (unlike flame-core's I32 Tensor
    // convention which is f32-bytes-relabeled). The kernel reads `int*`
    // off the buffer.
    let mut dev_offsets: cudarc::driver::CudaSlice<i32> = device
        .alloc_zeros(e)
        .map_err(|err| Error::Cuda(format!("alloc_zeros offsets: {err:?}")))?;
    device
        .htod_copy_into(offsets.to_vec(), &mut dev_offsets)
        .map_err(|err| Error::Cuda(format!("htod_copy offsets: {err:?}")))?;
    let off_ptr = *dev_offsets.device_ptr() as *const core::ffi::c_void;

    // ---- launch -------------------------------------------------------
    let stream = device.cuda_stream_raw_ptr();

    let x_ptr = x.as_device_ptr_bf16("grouped_mm_bf16.x")? as *const core::ffi::c_void;
    let w_ptr = w.as_device_ptr_bf16("grouped_mm_bf16.w")? as *const core::ffi::c_void;
    let y_ptr = y.as_mut_device_ptr_bf16("grouped_mm_bf16.y")? as *mut core::ffi::c_void;

    let status = unsafe {
        crate::cuda::ffi::flame_grouped_mm_bf16(
            x_ptr,
            w_ptr,
            off_ptr,
            y_ptr,
            t_max as i32,
            k_x as i32,
            n as i32,
            e as i32,
            stream,
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_grouped_mm_bf16 failed (cudaError {status}, T={t}, K={k_x}, N={n}, E={e}, T_max={t_max})"
        )));
    }
    // dev_offsets dropped after launch — the kernel runs synchronously on
    // the default stream when stream == null, so the buffer is no longer
    // referenced by the GPU at this point.
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    /// Naive scalar Rust reference for grouped_mm.
    /// Returns y as a flat Vec<f32> of length T*N.
    fn naive_grouped_mm_f32(
        x: &[f32],       // T*K row-major
        w: &[f32],       // E*K*N
        offsets: &[i32], // E
        t: usize,
        k: usize,
        n: usize,
        e: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; t * n];
        let mut prev = 0i32;
        for ei in 0..e {
            let lo = prev as usize;
            let hi = offsets[ei] as usize;
            prev = offsets[ei];
            for ti in lo..hi {
                for ni in 0..n {
                    let mut s = 0.0f32;
                    for ki in 0..k {
                        s += x[ti * k + ki] * w[ei * k * n + ki * n + ni];
                    }
                    y[ti * n + ni] = s;
                }
            }
        }
        y
    }

    fn bf16_round_trip_f32(v: f32) -> f32 {
        // Round to BF16 precision: keep upper 16 bits of the f32 representation.
        // This is what every BF16 cast does — used to produce a reference that
        // compares fairly against BF16-input math.
        let bits: u32 = v.to_bits();
        let truncated = bits & 0xFFFF_0000;
        f32::from_bits(truncated)
    }

    #[test]
    fn grouped_mm_matches_naive_small() -> Result<()> {
        // Toy: T=8, K=64, N=64, E=2, two equal-sized partitions.
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 8usize;
        let k = 64usize;
        let n = 64usize;
        let e = 2usize;
        let offsets = vec![4i32, 8i32];

        // Deterministic input/weight values.
        let x_f32: Vec<f32> = (0..t * k)
            .map(|i| {
                let v = ((i as f32) * 0.013).sin() * 0.3;
                bf16_round_trip_f32(v)
            })
            .collect();
        let w_f32: Vec<f32> = (0..e * k * n)
            .map(|i| {
                let v = ((i as f32) * 0.007).cos() * 0.2;
                bf16_round_trip_f32(v)
            })
            .collect();

        let x = Tensor::from_vec_dtype(
            x_f32.clone(),
            Shape::from_dims(&[t, k]),
            device.clone(),
            DType::BF16,
        )?;
        let w = Tensor::from_vec_dtype(
            w_f32.clone(),
            Shape::from_dims(&[e, k, n]),
            device.clone(),
            DType::BF16,
        )?;
        let t_max = (t / e) as usize; // both experts get T/E=4 rows
        let y = grouped_mm_bf16(&x, &w, &offsets, t_max)?;

        let y_host = y.to_vec()?;
        assert_eq!(y_host.len(), t * n);

        // Reference: BF16-rounded inputs, FP32 multiply-add.
        let ref_y = naive_grouped_mm_f32(&x_f32, &w_f32, &offsets, t, k, n, e);

        // BF16 GEMM accumulates in FP32 inside WMMA fragments, so the only
        // error source is the BF16 round of the output. Tolerance ~5e-3.
        let mut max_abs: f32 = 0.0;
        for i in 0..(t * n) {
            let diff = (y_host[i] - ref_y[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        assert!(
            max_abs < 1e-2,
            "grouped_mm output diverged from naive reference: max_abs={max_abs}"
        );
        Ok(())
    }
}
