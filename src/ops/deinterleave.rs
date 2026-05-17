//! Fast deinterleave for the last dim of contiguous tensors.
//!
//! Used by interleaved-SwiGLU MLPs: the up/gate linear emits an
//! `[..., 2K]` tensor where even-indexed columns belong to one half
//! (`x_glu`) and odd-indexed columns to the other (`x_linear`). The
//! generic strided-gather path (`materialize_strided_*_kernel`) takes
//! ~1.35 s for an 18 M-element F32 view because the stride-2 access
//! pattern coalesces poorly. This kernel uses `float2`/`__nv_bfloat162`
//! vectorized loads — one 8/4-byte load per thread, one 4/2-byte write
//! to each output buffer — which is memory-bound at full bandwidth
//! (~0.5 ms for the same workload on a 3090 Ti).

use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::dtype::DType;
use crate::{Error, Result, Shape, Tensor};

// Pass the input as plain `const float*` so the host-side slice type
// matches; the kernel reinterprets it as `const float2*` for an 8-byte
// vectorized load. CUDA allocations are at least 256-byte aligned, so the
// reinterpret is safe.
const DEINTERLEAVE_PAIR_F32_KERNEL: &str = r#"
extern "C" __global__
void deinterleave_pair_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ even,
    float* __restrict__ odd,
    long n_pairs
) {
    long idx = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (idx >= n_pairs) return;
    float2 v = reinterpret_cast<const float2*>(x)[idx];
    even[idx] = v.x;
    odd[idx] = v.y;
}
"#;

fn ensure_kernel(dev: &Arc<CudaDevice>) -> Result<cudarc::driver::CudaFunction> {
    let name = "deinterleave_pair_f32_kernel";
    if let Some(f) = dev.get_func(name, name) {
        return Ok(f);
    }
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(format!("{cuda_home}/include"));
    let ptx = compile_ptx_with_opts(DEINTERLEAVE_PAIR_F32_KERNEL, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc {name}: {e:?}")))?;
    if let Err(e) = dev.load_ptx(ptx, name, &[name]) {
        let msg = format!("{e:?}");
        if !msg.contains("already loaded") {
            return Err(Error::Cuda(format!("load_ptx {name}: {msg}")));
        }
    }
    dev.get_func(name, name)
        .ok_or_else(|| Error::Cuda(format!("kernel {name} not found after load")))
}

/// Deinterleave `x` along its last dim of size `2K`, returning two contiguous
/// tensors of shape `[..., K]`:
///
/// * `even[..., k] = x[..., 2 * k]`
/// * `odd [..., k] = x[..., 2 * k + 1]`
///
/// `x` must be contiguous F32 with an even last dim.
pub fn deinterleave_pair_f32(x: &Tensor) -> Result<(Tensor, Tensor)> {
    if x.dtype() != DType::F32 {
        return Err(Error::InvalidInput(format!(
            "deinterleave_pair_f32: input must be F32, got {:?}",
            x.dtype()
        )));
    }
    if !x.is_contiguous() {
        return Err(Error::InvalidInput(
            "deinterleave_pair_f32: input must be contiguous".into(),
        ));
    }
    let dims = x.shape().dims();
    if dims.is_empty() {
        return Err(Error::InvalidShape(
            "deinterleave_pair_f32: rank 0 tensor".into(),
        ));
    }
    let last = *dims.last().unwrap();
    if last % 2 != 0 {
        return Err(Error::InvalidShape(format!(
            "deinterleave_pair_f32: last dim {last} not even"
        )));
    }
    let half = last / 2;
    let leading: usize = dims[..dims.len() - 1].iter().product();
    let n_pairs = leading * half;

    let mut out_shape: Vec<usize> = dims.to_vec();
    *out_shape.last_mut().unwrap() = half;
    let out_shape = Shape::from_dims(&out_shape);

    let mut even = Tensor::empty_dtype(out_shape.clone(), DType::F32, x.device().clone())?;
    let mut odd = Tensor::empty_dtype(out_shape, DType::F32, x.device().clone())?;

    if n_pairs == 0 {
        return Ok((even, odd));
    }

    let dev = x.device();
    let func = ensure_kernel(dev)?;
    let cfg = LaunchConfig::for_num_elems(n_pairs as u32);

    let in_slice = x.storage_ref().try_as_slice_f32()?;
    let even_slice = even.storage_mut().try_as_mut_slice_f32()?;
    let odd_slice = odd.storage_mut().try_as_mut_slice_f32()?;
    crate::launch_kernel!(
        func,
        cfg,
        in_slice,
        &*even_slice,
        &*odd_slice,
        n_pairs as i64
    );

    Ok((even, odd))
}
