//! Fused CUDA kernels for inference: RMS norm, modulation, linear3d.
//! Each replaces multiple kernel launches with one.

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda::device_lt;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::DType;
use crate::{Error, Result, Shape, Tensor};
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::DevicePtr;

/// Persistent per-device cuBLASLt workspace for `fused_linear3d*`.
///
/// Both `fused_linear3d` and `fused_linear3d_native` used to
/// `device.alloc::<u8>(4 MiB)` on every call. Klein 9B issues hundreds of
/// linear calls per step, so that's hundreds of cudaMalloc/cudaFree
/// cycles per step. This cache allocates once per device (on first use)
/// and hands out the cached pointer on every call. The C-side shim
/// (fused_linear3d.cu) explicitly says workspace ownership is the
/// caller's concern — this is that cache on the caller side.
///
/// Thread-safety: the lock is held for the duration of the downstream
/// FFI call to cublasLtMatmul. That's microseconds, and flame-core is
/// single-device-per-process for inference; if multi-threaded training
/// ever needs to call these concurrently, we'd want per-stream
/// workspaces anyway (not per-device), so this design doesn't preclude
/// that future extension.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
mod linear_workspace {
    use super::*;
    use cudarc::driver::DeviceSlice;
    use std::sync::{Arc, Mutex, OnceLock};

    struct Entry {
        device: Arc<cudarc::driver::CudaDevice>,
        slice: cudarc::driver::CudaSlice<u8>,
    }

    static CACHE: OnceLock<Mutex<Option<Entry>>> = OnceLock::new();

    /// Returns a guard holding the workspace pointer + size. The mutex
    /// stays locked for the guard's lifetime; callers should drop the
    /// guard immediately after the FFI call completes.
    pub(super) fn acquire(
        device: &Arc<cudarc::driver::CudaDevice>,
        min_bytes: usize,
    ) -> Result<Guard> {
        let mutex = CACHE.get_or_init(|| Mutex::new(None));
        let mut guard = mutex
            .lock()
            .map_err(|_| Error::InvalidOperation("linear workspace mutex poisoned".into()))?;

        let needs_alloc = match guard.as_ref() {
            None => true,
            Some(entry) => !Arc::ptr_eq(&entry.device, device) || entry.slice.len() < min_bytes,
        };
        if needs_alloc {
            let new_slice: cudarc::driver::CudaSlice<u8> = unsafe { device.alloc(min_bytes) }
                .map_err(|e| Error::Cuda(format!("linear workspace alloc failed: {e:?}")))?;
            *guard = Some(Entry {
                device: device.clone(),
                slice: new_slice,
            });
        }
        // SAFETY: guard now holds a Some(Entry) that satisfies the size.
        // We store the raw pointer + size; the guard keeps the mutex held
        // so the CudaSlice isn't replaced while the caller uses the pointer.
        let entry = guard.as_ref().unwrap();
        let ptr = *entry.slice.device_ptr() as *mut u8;
        let size = entry.slice.len();
        Ok(Guard {
            _guard: guard,
            ptr,
            size,
        })
    }

    pub(super) struct Guard<'a> {
        _guard: std::sync::MutexGuard<'a, Option<Entry>>,
        ptr: *mut u8,
        size: usize,
    }

    impl<'a> Guard<'a> {
        pub(super) fn ptr(&self) -> *mut u8 {
            self.ptr
        }
        pub(super) fn size(&self) -> usize {
            self.size
        }
    }
}

/// GPU-side FP8 E4M3 → BF16 dequantization.
/// Input: raw FP8 bytes on GPU (CudaSlice<u8>), scale, shape.
/// Output: new BF16 Tensor.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn dequant_fp8_to_bf16(
    fp8_data: &cudarc::driver::CudaSlice<u8>,
    scale: f32,
    shape: Shape,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let numel = shape.elem_count();
    let bf16_out: cudarc::driver::CudaSlice<u16> = unsafe { device.alloc(numel)? };
    let stream = device_lt::stream_ptr(device)?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fp8_to_bf16(
            *fp8_data.device_ptr() as *const _,
            *bf16_out.device_ptr() as *mut _,
            scale,
            numel,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!("fp8_to_bf16 CUDA error: {ret}")));
    }

    Ok(Tensor::from_bf16_slice_gpu(
        bf16_out,
        shape,
        std::sync::Arc::clone(device),
    ))
}

/// GPU-side FP8 E4M3 → BF16 dequantization INTO an existing Tensor.
/// Zero allocation — writes directly into the output tensor's GPU memory.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn dequant_fp8_to_bf16_into(
    fp8_data: &cudarc::driver::CudaSlice<u8>,
    scale: f32,
    output: &Tensor,
) -> Result<()> {
    let numel = output.shape().elem_count();
    let stream = device_lt::stream_ptr(output.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fp8_to_bf16(
            *fp8_data.device_ptr() as *const _,
            output.as_device_ptr_bf16("dequant_into:output")? as *mut _,
            scale,
            numel,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!("fp8_to_bf16_into CUDA error: {ret}")));
    }
    Ok(())
}

/// Fused FP8 E4M3 dequant + transpose into a pre-allocated BF16 tensor.
/// Reads [M, N] row-major FP8 data, writes [N, M] row-major BF16.
/// One kernel launch, zero allocation.
///
/// - `fp8_data`: FP8 bytes on GPU, length = M * N
/// - `scale`: dequant scale factor
/// - `output`: pre-allocated BF16 tensor with shape [N, M]
/// - `m`: rows of the FP8 input (out_features)
/// - `n`: cols of the FP8 input (in_features)
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn dequant_fp8_transpose_into(
    fp8_data: &cudarc::driver::CudaSlice<u8>,
    scale: f32,
    output: &Tensor,
    m: usize,
    n: usize,
) -> Result<()> {
    let expected = m * n;
    let out_elems = output.shape().elem_count();
    if out_elems != expected {
        return Err(Error::InvalidShape(format!(
            "dequant_fp8_transpose_into: output has {out_elems} elements, expected {expected} (N={n} x M={m})"
        )));
    }

    let stream = device_lt::stream_ptr(output.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_dequant_transpose_bf16(
            *fp8_data.device_ptr() as *const _,
            output.as_device_ptr_bf16("dequant_transpose:output")? as *mut _,
            scale,
            m as i32,
            n as i32,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!(
            "flame_fused_dequant_transpose_bf16 CUDA error: {ret}"
        )));
    }
    Ok(())
}

/// GPU-side MXFP4 → BF16 dequantization.
///
/// MXFP4 packs 32 FP4 (E2M1) values per block, sharing one 8-bit E8M0
/// exponent scale. Used by GPT-OSS 20B (Lens text encoder) for MoE expert
/// weights. The 16 representable FP4 magnitudes are
/// `[0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]` (matches HuggingFace transformers'
/// `FP4_VALUES` exactly).
///
/// Inputs (both `CudaSlice<u8>` on GPU):
///   - `blocks`: 16 bytes per 32-element block (2 FP4 nibbles per byte).
///     Total length must be `rows_total * 16`.
///   - `scales`: one E8M0 exponent byte per 32-element block.
///     Total length must be `rows_total`.
///
/// Output: new BF16 tensor with shape `shape`. `shape.elem_count()` must
/// equal `rows_total * 32`.
///
/// Reference: `transformers/integrations/mxfp4.py::convert_moe_packed_tensors`.
/// The transpose at the end of that function is NOT done here — callers that
/// want PyTorch-layout `[E, H, 2*I]` should follow with `.transpose(1, 2).contiguous()`.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn dequant_mxfp4_to_bf16(
    blocks: &cudarc::driver::CudaSlice<u8>,
    scales: &cudarc::driver::CudaSlice<u8>,
    shape: Shape,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    use cudarc::driver::DeviceSlice;
    let numel = shape.elem_count();
    if numel % 32 != 0 {
        return Err(Error::InvalidShape(format!(
            "dequant_mxfp4_to_bf16: output element count {numel} must be a multiple of 32"
        )));
    }
    let rows_total = numel / 32;
    let blocks_bytes_expected = rows_total * 16;
    if blocks.len() < blocks_bytes_expected {
        return Err(Error::InvalidShape(format!(
            "dequant_mxfp4_to_bf16: blocks has {} bytes, expected {} (rows_total={})",
            blocks.len(),
            blocks_bytes_expected,
            rows_total
        )));
    }
    if scales.len() < rows_total {
        return Err(Error::InvalidShape(format!(
            "dequant_mxfp4_to_bf16: scales has {} bytes, expected {} (rows_total={})",
            scales.len(),
            rows_total,
            rows_total
        )));
    }

    let bf16_out: cudarc::driver::CudaSlice<u16> = unsafe { device.alloc(numel)? };
    let stream = device_lt::stream_ptr(device)?;

    let ret = unsafe {
        crate::cuda::ffi::flame_mxfp4_to_bf16(
            *blocks.device_ptr() as *const _,
            *scales.device_ptr() as *const _,
            *bf16_out.device_ptr() as *mut _,
            rows_total,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!("mxfp4_to_bf16 CUDA error: {ret}")));
    }

    Ok(Tensor::from_bf16_slice_gpu(
        bf16_out,
        shape,
        std::sync::Arc::clone(device),
    ))
}

/// GPU-side MXFP4 → BF16 dequantization INTO an existing BF16 tensor.
/// Zero allocation — writes directly into `output`'s GPU memory.
///
/// `output.shape().elem_count()` must equal `rows_total * 32` where
/// `rows_total = scales.len()`.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn dequant_mxfp4_to_bf16_into(
    blocks: &cudarc::driver::CudaSlice<u8>,
    scales: &cudarc::driver::CudaSlice<u8>,
    output: &Tensor,
) -> Result<()> {
    use cudarc::driver::DeviceSlice;
    if output.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "dequant_mxfp4_to_bf16_into: output must be BF16, got {:?}",
            output.dtype()
        )));
    }
    let numel = output.shape().elem_count();
    if numel % 32 != 0 {
        return Err(Error::InvalidShape(format!(
            "dequant_mxfp4_to_bf16_into: output element count {numel} must be a multiple of 32"
        )));
    }
    let rows_total = numel / 32;
    let blocks_bytes_expected = rows_total * 16;
    if blocks.len() < blocks_bytes_expected {
        return Err(Error::InvalidShape(format!(
            "dequant_mxfp4_to_bf16_into: blocks has {} bytes, expected {} (rows_total={})",
            blocks.len(),
            blocks_bytes_expected,
            rows_total
        )));
    }
    if scales.len() < rows_total {
        return Err(Error::InvalidShape(format!(
            "dequant_mxfp4_to_bf16_into: scales has {} bytes, expected {} (rows_total={})",
            scales.len(),
            rows_total,
            rows_total
        )));
    }

    let stream = device_lt::stream_ptr(output.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_mxfp4_to_bf16(
            *blocks.device_ptr() as *const _,
            *scales.device_ptr() as *const _,
            output.as_device_ptr_bf16("dequant_mxfp4_into:output")? as *mut _,
            rows_total,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!("mxfp4_to_bf16_into CUDA error: {ret}")));
    }
    Ok(())
}

/// Fused RMS normalization: BF16 → BF16 with weight multiply.
/// Replaces 6 kernel launches (cast + sq + mean + rsqrt + mul + mul_weight) with 1.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_rms_norm: input and weight must be BF16".into(),
        ));
    }

    let dims = input.shape().dims();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::InvalidInput("fused_rms_norm: empty shape".into()))?;
    let rows = input.shape().elem_count() / cols;

    let output = Tensor::empty_dtype(input.shape().clone(), DType::BF16, input.device().clone())?;

    let stream = device_lt::stream_ptr(input.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_rms_norm_bf16(
            input.as_device_ptr_bf16("fused_rms_norm:input")? as *const _,
            weight.as_device_ptr_bf16("fused_rms_norm:weight")? as *const _,
            output.as_device_ptr_bf16("fused_rms_norm:output")? as *mut _,
            rows as i32,
            cols as i32,
            eps,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_rms_norm CUDA error: {ret}")));
    }

    Ok(output)
}

/// Fused modulation: out = x * (1 + scale) + shift. All BF16.
/// Replaces 4 kernel launches (add_scalar + cast + mul + add) with 1.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::BF16 || scale.dtype() != DType::BF16 || shift.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_modulate: all inputs must be BF16".into(),
        ));
    }

    let n = x.shape().elem_count();
    let output = Tensor::empty_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;

    let stream = device_lt::stream_ptr(x.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_modulate_bf16(
            x.as_device_ptr_bf16("fused_modulate:x")? as *const _,
            scale.as_device_ptr_bf16("fused_modulate:scale")? as *const _,
            shift.as_device_ptr_bf16("fused_modulate:shift")? as *const _,
            output.as_device_ptr_bf16("fused_modulate:output")? as *mut _,
            n,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_modulate CUDA error: {ret}")));
    }

    Ok(output)
}

/// Fused 3D linear: [B, N, Cin] @ [Cin, Cout] + bias = [B, N, Cout].
/// No reshape kernels. Bias fused into cublasLt GEMM epilogue.
/// Weight must be PRE-TRANSPOSED to [Cin, Cout] (same as existing linear3d).
/// Replaces 4 launches (reshape + gemm + reshape + bias_add) with 1 cublasLt call.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_linear3d(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_linear3d: input and weight must be BF16".into(),
        ));
    }

    let in_shape = input.shape().dims();
    if in_shape.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d: input must be 3D [B,N,Cin], got {:?}",
            in_shape
        )));
    }
    let batch_size = in_shape[0];
    let seq_len = in_shape[1];
    let in_features = in_shape[2];

    let w_shape = weight.shape().dims();
    if w_shape.len() != 2 || w_shape[0] != in_features {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d: weight must be [Cin={in_features},Cout] (pre-transposed), got {:?}",
            w_shape
        )));
    }
    let out_features = w_shape[1];

    let out_shape = Shape::from_dims(&[batch_size, seq_len, out_features]);
    let output = Tensor::empty_dtype(out_shape, DType::BF16, input.device().clone())?;

    let device = input.device();
    let stream = device_lt::stream_ptr(device)?;
    let lt = device_lt::cublaslt_handle_ptr(device)?;

    let bias_ptr = if let Some(b) = bias {
        b.as_device_ptr_bf16("fused_linear3d:bias")? as *const _
    } else {
        std::ptr::null()
    };

    let workspace_size: usize = 4 * 1024 * 1024;
    let ws = linear_workspace::acquire(device, workspace_size)?;

    let ret = unsafe {
        crate::cuda::ffi::flame_linear3d_bf16(
            lt,
            input.as_device_ptr_bf16("fused_linear3d:input")? as *const _,
            weight.as_device_ptr_bf16("fused_linear3d:weight")? as *const _,
            bias_ptr,
            output.as_device_ptr_bf16("fused_linear3d:output")? as *mut _,
            batch_size as i32,
            seq_len as i32,
            in_features as i32,
            out_features as i32,
            ws.ptr() as *mut _,
            ws.size(),
            stream,
        )
    };
    drop(ws);

    if ret != 0 {
        return Err(Error::Cuda(format!("fused_linear3d cublasLt error: {ret}")));
    }

    Ok(output)
}

/// Fused 3D linear with bias epilogue, accepting the weight in **standard
/// PyTorch `[Cout, Cin]` row-major layout** (no pre-transpose required).
///
/// Internally uses cuBLASLt with TRANSA=T so the transpose happens inside the
/// GEMM, eliminating the per-call `transpose2d_bf16` pass that the FLUX
/// blocks were paying. For `single_blocks.linear1` (3072 → 21504) the
/// transpose alone was ~10–15 ms — this function gets it back.
///
/// `weight` shape: `[out_features, in_features]` BF16 (PyTorch nn.Linear default).
/// `bias` shape:   `[out_features]` BF16, optional.
/// `input` shape:  `[B, N, in_features]` BF16.
/// Returns `[B, N, out_features]`.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_linear3d_native(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_linear3d_native: input and weight must be BF16".into(),
        ));
    }

    let in_shape = input.shape().dims();
    if in_shape.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d_native: input must be 3D [B,N,Cin], got {:?}",
            in_shape
        )));
    }
    let batch_size = in_shape[0];
    let seq_len = in_shape[1];
    let in_features = in_shape[2];

    let w_shape = weight.shape().dims();
    if w_shape.len() != 2 || w_shape[1] != in_features {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d_native: weight must be [Cout, Cin={in_features}] (PyTorch layout), got {:?}",
            w_shape
        )));
    }
    let out_features = w_shape[0];

    let out_shape = Shape::from_dims(&[batch_size, seq_len, out_features]);
    let output = Tensor::empty_dtype(out_shape, DType::BF16, input.device().clone())?;

    let device = input.device();
    let stream = device_lt::stream_ptr(device)?;
    let lt = device_lt::cublaslt_handle_ptr(device)?;

    let bias_ptr = if let Some(b) = bias {
        b.as_device_ptr_bf16("fused_linear3d_native:bias")? as *const _
    } else {
        std::ptr::null()
    };

    let workspace_size: usize = 4 * 1024 * 1024;
    let ws = linear_workspace::acquire(device, workspace_size)?;

    let ret = unsafe {
        crate::cuda::ffi::flame_linear3d_bf16_native(
            lt,
            input.as_device_ptr_bf16("fused_linear3d_native:input")? as *const _,
            weight.as_device_ptr_bf16("fused_linear3d_native:weight")? as *const _,
            bias_ptr,
            output.as_device_ptr_bf16("fused_linear3d_native:output")? as *mut _,
            batch_size as i32,
            seq_len as i32,
            in_features as i32,
            out_features as i32,
            ws.ptr() as *mut _,
            ws.size(),
            stream,
        )
    };
    drop(ws);

    if ret != 0 {
        return Err(Error::Cuda(format!(
            "fused_linear3d_native cublasLt error: {ret}"
        )));
    }

    // CRITICAL: this used to be inference-only — `output` was created via
    // `Tensor::empty_dtype` with no `requires_grad`, no `Op::Linear`
    // recording. Calling it from a training-context tape silently broke
    // backward at every linear: gradient stopped at the linear's output
    // and never reached upstream/weights. Surfaced as ~random LoRA
    // gradient direction (cos_sim ≈ 0 against PyTorch) in the Z-Image
    // trainer's per-block grad parity. Fix: when input or weight requires
    // grad and an autograd tape is recording, set `output.requires_grad`
    // and record `Op::Linear` — the existing matmul-style backward at
    // `autograd.rs::Op::Linear` handles it.
    let need_grad = input.requires_grad
        || weight.requires_grad
        || bias.map(|b| b.requires_grad).unwrap_or(false);
    if need_grad && crate::autograd::AutogradContext::is_recording() {
        let mut output = output;
        output.requires_grad = true;
        let mut saved = vec![(input.id, input.clone()), (weight.id, weight.clone())];
        if let Some(b) = bias {
            saved.push((b.id, b.clone()));
        }
        crate::autograd::AutogradContext::record_op(
            output.id,
            crate::autograd::Op::Linear {
                input: input.id,
                weight: weight.id,
                bias: bias.map(|b| b.id),
            },
            saved,
        );
        return Ok(output);
    }

    Ok(output)
}

/// PyTorch-bit-exact parity variant of [`fused_linear3d_native`].
///
/// Same signature, same semantics, same fused-perf advantage. The only
/// difference is the underlying cuBLASLt configuration for the biased case:
/// this variant mirrors `at::cuda::blas::gemm_and_bias<at::BFloat16>`'s knobs
/// exactly, so the output is byte-equivalent to
/// `torch.nn.functional.linear(x_bf16, w_bf16, b_bf16)` under
/// `torch.autocast(bf16)`. For `bias=None`, PyTorch follows the no-bias
/// matmul path; the CUDA entry delegates to [`fused_linear3d_native`]'s native
/// no-bias kernel because that is the byte-exact and faster match for O1 Full's
/// `x_embedder.proj1`.
///
/// Use this in any inference/training path where strict per-op parity against
/// ai-toolkit / PyTorch matters (HiDream-O1 TimestepEmbedder, predecoder
/// linears, etc.). For models where the existing fused path is the
/// established baseline (Klein/Z-Image post-validation), keep using
/// `fused_linear3d_native` to preserve their existing parity.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_linear3d_native_pytorch_parity(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_linear3d_native_pytorch_parity: input and weight must be BF16".into(),
        ));
    }

    let in_shape = input.shape().dims();
    if in_shape.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d_native_pytorch_parity: input must be 3D [B,N,Cin], got {:?}",
            in_shape
        )));
    }
    let batch_size = in_shape[0];
    let seq_len = in_shape[1];
    let in_features = in_shape[2];

    let w_shape = weight.shape().dims();
    if w_shape.len() != 2 || w_shape[1] != in_features {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d_native_pytorch_parity: weight must be [Cout, Cin={in_features}] (PyTorch layout), got {:?}",
            w_shape
        )));
    }
    let out_features = w_shape[0];

    let out_shape = Shape::from_dims(&[batch_size, seq_len, out_features]);
    let output = Tensor::empty_dtype(out_shape, DType::BF16, input.device().clone())?;

    let device = input.device();
    let stream = device_lt::stream_ptr(device)?;
    let lt = device_lt::cublaslt_handle_ptr(device)?;

    let bias_ptr = if let Some(b) = bias {
        b.as_device_ptr_bf16("fused_linear3d_native_pytorch_parity:bias")? as *const _
    } else {
        std::ptr::null()
    };

    let workspace_size: usize = 4 * 1024 * 1024;
    let ws = linear_workspace::acquire(device, workspace_size)?;

    let ret = unsafe {
        crate::cuda::ffi::flame_linear3d_bf16_pytorch_parity(
            lt,
            input.as_device_ptr_bf16("fused_linear3d_native_pytorch_parity:input")? as *const _,
            weight.as_device_ptr_bf16("fused_linear3d_native_pytorch_parity:weight")? as *const _,
            bias_ptr,
            output.as_device_ptr_bf16("fused_linear3d_native_pytorch_parity:output")? as *mut _,
            batch_size as i32,
            seq_len as i32,
            in_features as i32,
            out_features as i32,
            ws.ptr() as *mut _,
            ws.size(),
            stream,
        )
    };
    drop(ws);

    if ret != 0 {
        return Err(Error::Cuda(format!(
            "fused_linear3d_native_pytorch_parity cublasLt error: {ret}"
        )));
    }

    // Autograd recording — same logic as fused_linear3d_native: when training-
    // context tape is recording and any input requires_grad, record Op::Linear
    // so backward flows through to weights/inputs/LoRA. Without this, the
    // gradient stops at the linear's output (silently invalid LoRA-B).
    let need_grad = input.requires_grad
        || weight.requires_grad
        || bias.map(|b| b.requires_grad).unwrap_or(false);
    if need_grad && crate::autograd::AutogradContext::is_recording() {
        let mut output = output;
        output.requires_grad = true;
        let mut saved = vec![(input.id, input.clone()), (weight.id, weight.clone())];
        if let Some(b) = bias {
            saved.push((b.id, b.clone()));
        }
        crate::autograd::AutogradContext::record_op(
            output.id,
            crate::autograd::Op::Linear {
                input: input.id,
                weight: weight.id,
                bias: bias.map(|b| b.id),
            },
            saved,
        );
        return Ok(output);
    }

    Ok(output)
}

/// Fused 3D linear with optional LoRA pair.
///
/// When `lora_a` and `lora_b` are both `None`, behavior is byte-identical to
/// `fused_linear3d_native(input, weight, bias)` — same kernel, same autograd
/// recording (`Op::Linear` when training-context).
///
/// When provided, computes:
///   `out = fused_native(x, weight, bias) + lora_scale * ((x @ A^T) @ B^T)`
/// where `A = lora_a [rank, Cin]` and `B = lora_b [Cout, rank]`.
///
/// Autograd: the base path records `Op::Linear` (already wired). The LoRA
/// residual uses `Tensor::matmul` + `Tensor::transpose` + `Tensor::mul_scalar`
/// + `Tensor::add`, each of which is already autograd-registered. As long as
/// `lora_a.requires_grad` / `lora_b.requires_grad` are set by the caller
/// and an `AutogradContext` is recording, gradients will flow into A and B
/// (and not into the frozen base `weight`).
///
/// `weight`/`bias` should be `requires_grad=false` (base frozen).
/// `lora_a`/`lora_b` should be `requires_grad=true` (trainable).
///
/// BF16 LoRA uses zero-copy transpose views; the downstream BF16 GEMM detects
/// them and uses cuBLASLt transpose flags. F32 LoRA intentionally casts the
/// input to F32, materializes transposed A/B views for the F32 GEMM path, then
/// casts the residual back to the base output dtype. That matches Python LoRA
/// training code that keeps adapter weights in F32 while the frozen base runs
/// mixed precision.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_linear3d_native_lora(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    lora_a: Option<&Tensor>,
    lora_b: Option<&Tensor>,
    lora_scale: f32,
) -> Result<Tensor> {
    // Base path — identical to non-LoRA call.
    let base = fused_linear3d_native(input, weight, bias)?;

    // No LoRA → byte-identical to base.
    let (a, b) = match (lora_a, lora_b) {
        (Some(a), Some(b)) => (a, b),
        (None, None) => return Ok(base),
        _ => {
            return Err(Error::InvalidInput(
                "fused_linear3d_native_lora: lora_a and lora_b must both be Some or both None"
                    .into(),
            ));
        }
    };

    // Shape checks: A is [rank, Cin], B is [Cout, rank].
    let in_dims = input.shape().dims();
    let cin = in_dims[in_dims.len() - 1];
    let cout = weight.shape().dims()[0];
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims.len() != 2 || a_dims[1] != cin {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d_native_lora: lora_a must be [rank, Cin={cin}], got {:?}",
            a_dims
        )));
    }
    if b_dims.len() != 2 || b_dims[0] != cout || b_dims[1] != a_dims[0] {
        return Err(Error::InvalidShape(format!(
            "fused_linear3d_native_lora: lora_b must be [Cout={cout}, rank={}], got {:?}",
            a_dims[0], b_dims
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(Error::InvalidInput(format!(
            "fused_linear3d_native_lora: lora_a and lora_b must have the same dtype, got {:?} and {:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    if a.dtype() != DType::BF16 && a.dtype() != DType::F32 {
        return Err(Error::InvalidInput(format!(
            "fused_linear3d_native_lora: unsupported LoRA dtype {:?}; expected BF16 or F32",
            a.dtype()
        )));
    }

    // LoRA residual: (x @ A^T) @ B^T * scale.
    // `matmul` handles 3D@2D natively (see tensor.rs:1694 — flattens to 2D
    // GEMM, reshapes back). Autograd is recorded by Op::MatMul + Op::Transpose
    // + Op::MulScalar.
    let residual = if a.dtype() == DType::F32 {
        // PyTorch runs ai-toolkit's F32 LoRA modules under CUDA autocast in the
        // O1 trainer. The module weights stay F32 leaves, but linear compute is
        // BF16. Cast through autograd so F32 params still receive F32 grads.
        let input_bf16 = input.to_dtype(DType::BF16)?;
        let a_bf16 = a.to_dtype(DType::BF16)?;
        let b_bf16 = b.to_dtype(DType::BF16)?;
        let lora_a_t = a_bf16.transpose()?.contiguous()?; // [Cin, rank]
        let lora_b_t = b_bf16.transpose()?.contiguous()?; // [rank, Cout]
        let xa = input_bf16.matmul(&lora_a_t)?; // [B, S, rank]
        let xab = xa.matmul(&lora_b_t)?; // [B, S, Cout]
        xab.mul_scalar(lora_scale)?
    } else {
        let lora_a_t = a.transpose()?.contiguous()?; // [Cin, rank]
        let lora_b_t = b.transpose()?.contiguous()?; // [rank, Cout]
        let xa = input.matmul(&lora_a_t)?; // [B, S, rank]
        let xab = xa.matmul(&lora_b_t)?; // [B, S, Cout]
        xab.mul_scalar(lora_scale)?
    };
    let residual = if residual.dtype() != base.dtype() {
        residual.to_dtype(base.dtype())?
    } else {
        residual
    };
    base.add(&residual)
}

/// Fused RMS norm + modulation in one kernel.
/// out = rms_norm(x, weight) * (1 + scale) + shift
/// Replaces fused_rms_norm + fused_modulate (2 launches → 1).
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_rms_norm_modulate(
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_rms_norm_modulate: inputs must be BF16".into(),
        ));
    }

    let dims = x.shape().dims();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::InvalidInput("empty shape".into()))?;
    let rows = x.shape().elem_count() / cols;

    let output = Tensor::empty_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    let stream = device_lt::stream_ptr(x.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_rms_norm_modulate_bf16(
            x.as_device_ptr_bf16("fused_norm_mod:x")? as *const _,
            weight.as_device_ptr_bf16("fused_norm_mod:w")? as *const _,
            scale.as_device_ptr_bf16("fused_norm_mod:scale")? as *const _,
            shift.as_device_ptr_bf16("fused_norm_mod:shift")? as *const _,
            output.as_device_ptr_bf16("fused_norm_mod:out")? as *mut _,
            rows as i32,
            cols as i32,
            eps,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!(
            "fused_rms_norm_modulate CUDA error: {ret}"
        )));
    }
    Ok(output)
}

/// Fused residual + gating: out = x + gate * attn_out.
/// Replaces mul + add (2 launches → 1).
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn fused_residual_gate(x: &Tensor, attn_out: &Tensor, gate: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "fused_residual_gate: inputs must be BF16".into(),
        ));
    }

    let n = x.shape().elem_count();
    let output = Tensor::empty_dtype(x.shape().clone(), DType::BF16, x.device().clone())?;
    let stream = device_lt::stream_ptr(x.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_fused_residual_gate_bf16(
            x.as_device_ptr_bf16("fused_res_gate:x")? as *const _,
            attn_out.as_device_ptr_bf16("fused_res_gate:attn")? as *const _,
            gate.as_device_ptr_bf16("fused_res_gate:gate")? as *const _,
            output.as_device_ptr_bf16("fused_res_gate:out")? as *mut _,
            n,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!(
            "fused_residual_gate CUDA error: {ret}"
        )));
    }
    Ok(output)
}
