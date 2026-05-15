//! Fused MoE unpermute: per-token gated scatter-add into an F32 accumulator.
//!
//! Wraps the `flame_fused_gated_scatter_add_bf16` build-time CUDA kernel
//! (`src/cuda/fused_gated_scatter_add.cu`). Semantics:
//!
//! ```text
//! accum[indices[t]] += expert_out[t] * gating[t]      (in-place, all t)
//!
//! expert_out : (T, D)  BF16, per-token expert outputs in expert-major order
//! gating     : (T,)    F32   per-token gating weight
//! indices    : (T,)    I32   global token indices (out-of-range rows are skipped)
//! accum      : (N, D)  F32   running accumulator, updated in-place
//! ```
//!
//! Used by Nucleus-Image and LLaDA MoE forward to combine per-expert outputs
//! back into per-token rows. The in-place F32 accumulator is needed because
//! multiple expert outputs can collide on the same target row when the same
//! token is picked by more than one expert (top-K with K>1, or — in
//! Nucleus's expert-choice scheme — when more than one expert's top-C
//! happens to include the same token).
//!
//! See `docs/FLAME_KERNELS.md` for the kernel design notes.

use cudarc::driver::DevicePtr;

use crate::device::CudaStreamRawPtrExt;
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Tensor};

#[allow(unused_imports)]
use cudarc::driver::DevicePtrMut;

/// Fused gated scatter-add. Updates `accum` in-place.
///
/// # Arguments
/// - `expert_out`:  `(T, D)`  BF16, per-token expert outputs
/// - `gating`:      `(T,)`    F32, per-token gating weight
/// - `indices`:     `&[i32]` of length `T`, target rows in `accum` (negative
///   or out-of-range rows are skipped by the kernel — no error, used to
///   model the "no expert picked this token" case). Host slice — wrapper
///   does the HtoD copy. For Phase 4 (MoeFFN) when indices are produced
///   GPU-side by routing, see the `_with_dev_indices` variant below (TODO).
/// - `accum`:       `(N, D)`  F32, in-place accumulator
pub fn fused_gated_scatter_add_bf16(
    expert_out: &Tensor,
    gating: &Tensor,
    indices: &[i32],
    accum: &mut Tensor,
) -> Result<()> {
    // ---- dtype + shape validation -----------------------------------
    if expert_out.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: expert_out must be BF16, got {:?}",
            expert_out.dtype()
        )));
    }
    if gating.dtype() != DType::F32 {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: gating must be F32, got {:?}",
            gating.dtype()
        )));
    }
    if accum.dtype() != DType::F32 {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: accum must be F32, got {:?}",
            accum.dtype()
        )));
    }

    let eo_dims = expert_out.shape().dims();
    let g_dims = gating.shape().dims();
    let acc_dims = accum.shape().dims();

    if eo_dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: expert_out must be 2-D (T, D), got {eo_dims:?}"
        )));
    }
    if g_dims.len() != 1 {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: gating must be 1-D (T,), got {g_dims:?}"
        )));
    }
    if acc_dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: accum must be 2-D (N, D), got {acc_dims:?}"
        )));
    }

    let (t, d) = (eo_dims[0], eo_dims[1]);
    let n = acc_dims[0];

    if g_dims[0] != t {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: gating length {} != T={}",
            g_dims[0], t
        )));
    }
    if indices.len() != t {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: indices.len()={} != T={}",
            indices.len(),
            t
        )));
    }
    if acc_dims[1] != d {
        return Err(Error::InvalidOperation(format!(
            "fused_gated_scatter_add: accum D={} != expert_out D={}",
            acc_dims[1], d
        )));
    }

    if !expert_out.is_contiguous() || !gating.is_contiguous() || !accum.is_contiguous() {
        return Err(Error::InvalidOperation(
            "fused_gated_scatter_add: tensor inputs must be contiguous".into(),
        ));
    }

    // ---- HtoD copy of host indices to a temp i32 device buffer ------
    // CudaSlice<i32> stores REAL i32 bytes (kernel reads `int*`).
    let device = accum.device().clone();
    let mut dev_indices: cudarc::driver::CudaSlice<i32> = device
        .alloc_zeros(t)
        .map_err(|err| Error::Cuda(format!("alloc_zeros indices: {err:?}")))?;
    device
        .htod_copy_into(indices.to_vec(), &mut dev_indices)
        .map_err(|err| Error::Cuda(format!("htod_copy indices: {err:?}")))?;
    let i_ptr = *dev_indices.device_ptr() as *const core::ffi::c_void;

    // ---- launch -----------------------------------------------------
    let stream = device.cuda_stream_raw_ptr();

    let eo_ptr = expert_out.as_device_ptr_bf16("fused_gated_scatter_add.expert_out")?
        as *const core::ffi::c_void;
    let g_ptr = f32_device_ptr(gating)? as *const core::ffi::c_void;
    let acc_ptr = f32_device_ptr_mut(accum)? as *mut core::ffi::c_void;

    let status = unsafe {
        crate::cuda::ffi::flame_fused_gated_scatter_add_bf16(
            eo_ptr, g_ptr, i_ptr, acc_ptr, t as i32, d as i32, n as i32, stream,
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_fused_gated_scatter_add_bf16 failed (cudaError {status}, T={t}, D={d}, N={n})"
        )));
    }
    Ok(())
}

fn f32_device_ptr(t: &Tensor) -> Result<*const f32> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidOperation(format!(
            "f32_device_ptr: expected F32, got {:?}",
            t.dtype()
        )));
    }
    match t.storage_ref() {
        TensorStorage::F32 { data, .. } => Ok(*data.device_ptr() as *const f32),
        _ => Err(Error::InvalidOperation(
            "f32_device_ptr: storage is not F32".into(),
        )),
    }
}

fn f32_device_ptr_mut(t: &mut Tensor) -> Result<*mut f32> {
    use cudarc::driver::DevicePtrMut;
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidOperation(format!(
            "f32_device_ptr_mut: expected F32, got {:?}",
            t.dtype()
        )));
    }
    match t.storage_mut() {
        TensorStorage::F32 { ref mut data, .. } => {
            let slice = crate::tensor_storage::ensure_unique_slice(data)?;
            Ok(*slice.device_ptr_mut() as *mut f32)
        }
        _ => Err(Error::InvalidOperation(
            "f32_device_ptr_mut: storage is not F32".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;
    use cudarc::driver::CudaDevice;

    /// Naive scalar Rust reference for fused_gated_scatter_add.
    fn naive_scatter_add(
        expert_out: &[f32], // T*D, BF16-rounded
        gating: &[f32],     // T
        indices: &[i32],    // T
        accum: &mut [f32],  // N*D
        t: usize,
        d: usize,
        n: usize,
    ) {
        for ti in 0..t {
            let idx = indices[ti];
            if idx < 0 || (idx as usize) >= n {
                continue; // out-of-range rows skipped by the kernel
            }
            let g = gating[ti];
            let row = (idx as usize) * d;
            for di in 0..d {
                accum[row + di] += expert_out[ti * d + di] * g;
            }
        }
    }

    fn bf16_round(v: f32) -> f32 {
        let bits = v.to_bits() & 0xFFFF_0000;
        f32::from_bits(bits)
    }

    #[test]
    fn fused_gated_scatter_add_matches_naive() -> Result<()> {
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 16usize;
        let d = 64usize;
        let n = 8usize;

        // expert_out: BF16-rounded values
        let eo_f32: Vec<f32> = (0..t * d)
            .map(|i| bf16_round(((i as f32) * 0.011).sin() * 0.5))
            .collect();
        // gating: stays F32
        let g_f32: Vec<f32> = (0..t)
            .map(|i| ((i as f32) * 0.13).cos() * 0.7 + 0.5)
            .collect();
        // indices: each token routes to a specific row; some collide so we
        // exercise the atomicAdd path.
        let idx_i32: Vec<i32> = (0..t).map(|i| ((i * 3) % (n + 2)) as i32 - 1).collect(); // some -1's get skipped
                                                                                          // accum: start non-zero so we verify it's add, not overwrite
        let acc_init: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.001).collect();

        let expert_out = Tensor::from_vec_dtype(
            eo_f32.clone(),
            Shape::from_dims(&[t, d]),
            device.clone(),
            DType::BF16,
        )?;
        let gating = Tensor::from_vec_dtype(
            g_f32.clone(),
            Shape::from_dims(&[t]),
            device.clone(),
            DType::F32,
        )?;
        let mut accum = Tensor::from_vec_dtype(
            acc_init.clone(),
            Shape::from_dims(&[n, d]),
            device.clone(),
            DType::F32,
        )?;

        fused_gated_scatter_add_bf16(&expert_out, &gating, &idx_i32, &mut accum)?;

        let got = accum.to_vec()?;

        // Reference
        let mut ref_acc = acc_init.clone();
        naive_scatter_add(&eo_f32, &g_f32, &idx_i32, &mut ref_acc, t, d, n);

        let mut max_abs: f32 = 0.0;
        for i in 0..(n * d) {
            let diff = (got[i] - ref_acc[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        assert!(
            max_abs < 5e-4,
            "fused_gated_scatter_add diverged from naive: max_abs={max_abs}"
        );
        Ok(())
    }
}
