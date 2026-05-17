//! torchao prototype `int8_weight_only_quantized_training` Rust port.
//!
//! Bit-exact-equivalent (modulo BF16 GEMM noise) to torchao 0.14.1
//! `int8_weight_only_quantized_training()` quantization + forward + grad-wrt-x.
//!
//! References (torchao 0.14.1):
//!   - `torchao/prototype/quantized_training/int8.py:23-52`
//!     (`quantize_int8_rowwise`)
//!   - `torchao/prototype/quantized_training/int8.py:55-115`
//!     (`Int8QuantizedTrainingLinearWeight`)
//!   - `torchao/prototype/quantized_training/int8.py:146-173`
//!     (`_Int8WeightOnlyLinear` forward + backward)
//!   - `torchao/prototype/quantized_training/int8.py:304-319`
//!     (`Int8WeightOnlyQuantizedTrainingConfig` + `_..._transform`)
//!
//! ## Scope
//!
//! Forward path + autograd-tracked backward through `x` only.
//!
//! The torchao training tensor's main differentiator vs the inference
//! `int8_weight_only` is that `F.linear` IS differentiable: the dispatch
//! routes through `_Int8WeightOnlyLinear.apply` which defines its own
//! `forward` and `backward`. SimpleTuner's LoRA path (`model_type=lora`)
//! freezes the base weight and trains only adapter deltas, so we need:
//!
//!   1. `grad_input` so the LoRA delta gradients flow upstream into earlier
//!      layers (this is the load-bearing path).
//!   2. `grad_bias` if bias is a Parameter (handled transparently by the
//!      standard `add` op recording).
//!
//! We do NOT implement:
//!
//!   - `grad_weight` (line 169-171 in int8.py). The int8 base weight is
//!     frozen under LoRA; never receives a gradient.
//!   - The `aten.copy_` writeback path (line 236-252 in int8.py) with
//!     stochastic rounding. Not exercised when the base weight has
//!     `requires_grad=False`. If someone ever wants full base-grad
//!     training, this is the missing piece — re-quantize the updated
//!     dequantized weight with `stochastic_rounding=True`.
//!
//! ## Numerical contract
//!
//! torchao's quant (int8.py:39-51, faithfully transcribed):
//!
//! ```text
//!   scale[r]     = absmax(W[r, :]) / 127                 // same dtype as W
//!   inv_scale[r] = 1.0 / max(scale[r].float(), eps)      // eps = 1e-12
//!   int_data[r]  = round(W[r, :].float() * inv_scale[r]).clip(-128, 127).int8()
//! ```
//!
//! Note: torchao stores `scale` in W's source dtype (whatever the original
//! float weight was). The eps is applied ONLY to `inv_scale` (the divisor
//! used during the multiply); the stored `scale` is unguarded. We host-
//! quantize from F32 so the stored scales are F32, matching torchao's
//! behavior when fed a Linear with F32 weight.
//!
//! Forward (int8.py:158):
//!
//! ```text
//!   out = (input @ int_data.T.to(input.dtype)) * scale + bias?
//! ```
//!
//! Backward — input only (int8.py:166-168):
//!
//! ```text
//!   grad_input = (grad_output * scale) @ int_data.to(grad_output.dtype)
//! ```
//!
//! Since the int8 weight is frozen (no requires_grad), and the BF16 cast
//! of the codes is performed inside a `no_grad` scope (or equivalently as
//! a tensor with `requires_grad=false`), the standard `matmul` autograd
//! op for `input @ cast.T` produces a `grad_input` numerically identical
//! to the torchao formula above — because `cast.T = int_data.T.to(bf16)`
//! and the `* scale` step records its own `Op::Mul` that backprops the
//! scale into the matmul gradient.
//!
//! ## Used by
//!
//! Trainers selecting `--base-quant int8-torchao` for SimpleTuner parity.
//! See `simpletuner/helpers/training/quantisation/__init__.py:465-490`
//! (the `int8-torchao` branch invokes
//! `quantize_(model, int8_weight_only_quantized_training())`).

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::CompileOptions;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CPU-side quantization (torchao parity)
// ---------------------------------------------------------------------------

/// Eps applied to `inv_scale` only — matches torchao default
/// (int8.py:25, `eps: float = 1e-12`).
pub const INT8_QT_EPS: f32 = 1e-12;

/// Round F32 to BF16 (truncation, NOT round-to-nearest) and back to F32.
///
/// This models exactly what happens when an F32 value is materialized into
/// a BF16 PyTorch tensor: PyTorch's BF16 dtype is the high 16 bits of F32
/// — values arriving via `tensor.to(torch.bfloat16)` use the default IEEE
/// round-to-nearest-even. We mirror that here. We use this helper to
/// faithfully reproduce torchao's path when the source weight is BF16:
/// `scale = bf16_W.abs().amax(1) / 127` is computed entirely in BF16,
/// then `.float()` upcasts the truncated bits losslessly.
#[inline]
fn bf16_round_trip(x: f32) -> f32 {
    let bits = x.to_bits();
    // Round to nearest even, ties to even (matches PyTorch BF16 cast).
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    let rounded = bits.wrapping_add(rounding_bias) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// torchao symmetric per-row absmax INT8 quantization.
///
/// Mirrors `quantize_int8_rowwise` from
/// `torchao/prototype/quantized_training/int8.py:23-52` with
/// `stochastic_rounding=False` (the path taken by
/// `Int8QuantizedTrainingLinearWeight.from_float` at
/// `int8.py:98-106`, which is what `quantize_(...)` invokes during
/// SimpleTuner's `int8-torchao` setup).
///
/// - `weight`: row-major buffer in F32 representation, shape `[out, in_]`.
/// - `source_is_bf16`: if `true`, simulate the BF16 round-trip that
///   torchao performs implicitly when the source `nn.Linear.weight` is
///   BF16. SimpleTuner always passes BF16 base weights, so this is the
///   realistic path (`true`).
/// - Returns `(codes, scales)` where `codes.len() == out * in_` and
///   `scales.len() == out`.
///
/// The math when `source_is_bf16=true` (torchao int8.py:40-51, with the
/// input cast trail made explicit):
///
/// ```text
///   absmax_bf16[r] = bf16( max_c |W_bf16[r, c]| )       // BF16 reduce
///   scale_bf16[r]  = bf16( absmax_bf16[r] / 127 )       // BF16 divide
///   scale_f32[r]   = scale_bf16[r].to(F32)              // lossless upcast
///   inv_scale[r]   = 1.0 / max(scale_f32[r], 1e-12)     // F32, eps on divisor
///   code[r, c]     = round(W_f32[r, c] * inv_scale[r]).clip(-128, 127)
/// ```
///
/// When `source_is_bf16=false` the same logic runs without the BF16
/// rounding steps (rare in diffusion but valid for F32 nn.Linear).
///
/// `eps` is applied ONLY to the `inv_scale` divisor; the stored `scale`
/// is left as `absmax / 127` even when absmax is 0 (in which case `scale`
/// is 0 too, the codes are all 0, and dequant returns 0 — which matches
/// torchao's behavior).
pub fn quantize_int8_qt(
    weight: &[f32],
    shape: [usize; 2],
    source_is_bf16: bool,
) -> (Vec<i8>, Vec<f32>) {
    let [out, in_] = shape;
    assert_eq!(
        weight.len(),
        out * in_,
        "quantize_int8_qt: weight len {} != out*in {}*{}",
        weight.len(),
        out,
        in_
    );

    let mut codes = vec![0i8; out * in_];
    let mut scales = vec![0.0f32; out];

    for r in 0..out {
        let row = &weight[r * in_..(r + 1) * in_];

        // absmax (torchao calls `.abs().amax(1)` -> per-row).
        // In BF16-source mode, each individual |x| is already representable
        // exactly in BF16 (sign+exponent only changes between original and
        // |x|), so the BF16 round-trip applies only at the final reduction
        // step. We do it conservatively after every max.
        let mut absmax = 0.0f32;
        for &x in row {
            let a = x.abs();
            if a > absmax {
                absmax = a;
            }
        }
        let absmax = if source_is_bf16 { bf16_round_trip(absmax) } else { absmax };

        // scale = absmax / 127 (NOT 127.5 — torchao's choice, see int8.py docstring).
        // In BF16-source mode, the result is rounded to BF16 (the storage dtype
        // of the actual stored `scale` tensor inside torchao).
        let scale_f32 = absmax / 127.0;
        let scale_stored = if source_is_bf16 { bf16_round_trip(scale_f32) } else { scale_f32 };
        scales[r] = scale_stored;

        // inv_scale = 1.0 / max(scale, eps)  (eps on divisor only)
        let inv_scale = 1.0 / scale_stored.max(INT8_QT_EPS);

        let out_row = &mut codes[r * in_..(r + 1) * in_];
        for c in 0..in_ {
            // torchao uses `tensor.round()` (int8.py:49), which is PyTorch's
            // round = IEEE 754 round-half-to-even (banker's rounding).
            // Rust's `f32::round()` uses round-half-away-from-zero. We must
            // mirror PyTorch's behavior — `f32::round_ties_even` (stabilized
            // in 1.77, available here) is the exact match.
            let v = row[c] * inv_scale;
            let q = v.round_ties_even().clamp(-128.0, 127.0);
            out_row[c] = q as i8;
        }
    }

    (codes, scales)
}

// ---------------------------------------------------------------------------
// On-device int8 weight + scale storage
// ---------------------------------------------------------------------------

/// Device-side container for one int8-quantized linear weight.
///
/// `codes` is a raw `CudaSlice<i8>` (TensorStorage has no I8 arm; see the
/// rationale at `adam8bit_kernel.rs:31-42` — same pattern). `scales` is a
/// normal F32 `Tensor` so it can be broadcast-multiplied via the standard
/// `mul` op (which records autograd correctly when downstream tensors
/// require grad).
///
/// The codes and scales are FROZEN: no autograd metadata, never receive
/// a gradient, never get updated by the optimizer. Only the LoRA delta
/// attached to the layer trains.
pub struct Int8QtWeight {
    /// Raw int8 codes, length `out * in_`, row-major.
    pub codes: CudaSlice<i8>,
    /// Per-row scale, shape `[out]`, F32, frozen (`requires_grad=false`).
    pub scales: Tensor,
    /// `[out, in_]` shape of the original weight.
    pub shape: [usize; 2],
    pub device: Arc<CudaDevice>,
}

impl Int8QtWeight {
    /// Upload host-quantized `(codes, scales)` to device.
    pub fn upload(
        codes: Vec<i8>,
        scales: Vec<f32>,
        shape: [usize; 2],
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let [out, in_] = shape;
        if codes.len() != out * in_ {
            return Err(Error::InvalidOperation(format!(
                "Int8QtWeight::upload codes len {} != out*in {}*{}",
                codes.len(),
                out,
                in_
            )));
        }
        if scales.len() != out {
            return Err(Error::InvalidOperation(format!(
                "Int8QtWeight::upload scales len {} != out {}",
                scales.len(),
                out
            )));
        }

        let codes_dev = device
            .htod_copy(codes)
            .map_err(|e| Error::Cuda(format!("Int8QtWeight::upload codes htod: {e:?}")))?;
        let scales_tensor =
            Tensor::from_vec(scales, Shape::from_dims(&[out]), device.clone())?;
        // F32 by default; explicitly mark not requiring grad — base is frozen.
        debug_assert!(!scales_tensor.requires_grad);

        Ok(Self {
            codes: codes_dev,
            scales: scales_tensor,
            shape,
            device,
        })
    }

    /// Cast the int8 codes to a BF16 `Tensor` of shape `[out, in_]`,
    /// FROZEN (no autograd metadata, `requires_grad=false`).
    ///
    /// This mirrors the `int_data.T.to(input.dtype)` step at
    /// `int8.py:158`, except we leave the transpose to the caller —
    /// the canonical `Tensor::transpose` is a zero-copy stride flip
    /// and feeds straight into `matmul`.
    ///
    /// Kernel: simple grid-stride loop, one i8→bf16 cast per thread.
    /// See `cast_i8_to_bf16_launch` below.
    pub fn codes_to_bf16(&self) -> Result<Tensor> {
        ensure_cast_kernel(&self.device)?;
        let [out, in_] = self.shape;
        let numel = out * in_;
        let mut out_tensor =
            Tensor::zeros_dtype(Shape::from_dims(&[out, in_]), DType::BF16, self.device.clone())?;

        let dst_ptr = out_tensor.as_mut_device_ptr_bf16("int8_qt:codes_to_bf16")? as u64;
        let src_ptr: u64 = *self.codes.device_ptr();
        let n_i64 = numel as i64;

        let f = self
            .device
            .get_func(MODULE_NAME, KERN_I8_TO_BF16)
            .ok_or_else(|| Error::Cuda(format!("missing kernel: {KERN_I8_TO_BF16}")))?;

        // 256 threads/block, ceil(numel/256) blocks.
        let block: u32 = 256;
        let grid: u32 = numel.div_ceil(block as usize) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(3);
        params.push(&dst_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&src_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&n_i64 as *const i64 as *mut std::ffi::c_void);

        unsafe {
            f.launch(cfg, &mut params)
                .map_err(|e| Error::Cuda(format!("i8_to_bf16 launch: {e:?}")))?;
        }
        Ok(out_tensor)
    }
}

// ---------------------------------------------------------------------------
// NVRTC kernel — int8 → BF16 cast
// ---------------------------------------------------------------------------

const MODULE_NAME: &str = "int8_qt_cast";
const KERN_I8_TO_BF16: &str = "i8_to_bf16_kernel";

const KERNEL_SRC: &str = r#"
#include <cuda_bf16.h>

// Trivial int8 -> bf16 cast (no scale applied; the scale multiply happens
// downstream in a separate BF16 elementwise op so autograd can record it).
//
// Launch geometry:
//   grid_dim  = ((n + 255) / 256, 1, 1)
//   block_dim = (256, 1, 1)
//   shared    = 0
extern "C" __global__ void i8_to_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const signed char* __restrict__ src,
    long long n
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (idx >= n) return;
    // sign-extend i8 -> int -> float -> bf16 (round-to-nearest)
    int v_i = (int)src[idx];
    float v_f = (float)v_i;
    dst[idx] = __float2bfloat16(v_f);
}
"#;

fn ensure_cast_kernel(device: &Arc<CudaDevice>) -> Result<()> {
    if device.get_func(MODULE_NAME, KERN_I8_TO_BF16).is_some() {
        return Ok(());
    }
    let include_dir = std::env::var("CUDA_INCLUDE_DIR")
        .or_else(|_| std::env::var("CUDA_HOME").map(|h| format!("{h}/include")))
        .unwrap_or_else(|_| "/usr/local/cuda/include".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(include_dir);
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SRC, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc int8_qt_cast: {e:?}")))?;
    device
        .load_ptx(ptx, MODULE_NAME, &[KERN_I8_TO_BF16])
        .map_err(|e| Error::Cuda(format!("load int8_qt_cast: {e:?}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Forward: y = (x @ codes_to_bf16(w).T) * scale + bias
// ---------------------------------------------------------------------------

/// torchao `_Int8WeightOnlyLinear.forward` parity, autograd-tracked through `x`.
///
/// Reference: `torchao/prototype/quantized_training/int8.py:146-160`. The
/// torchao formulation is:
///
/// ```text
///   out = (input @ int_data.T.to(input.dtype)) * scale + bias?
/// ```
///
/// Our composition produces the same scalar (modulo BF16 GEMM rounding),
/// with the standard flame-core autograd graph:
///
/// 1. `w_bf16 = codes_to_bf16(w)` — frozen, `requires_grad=false`.
/// 2. `tmp = x @ w_bf16.T` — `Op::Matmul` recorded if `x.requires_grad`.
/// 3. `scaled = tmp * scale.to(bf16)` — `Op::Mul` recorded.
/// 4. `out = scaled + bias` if bias is present — `Op::Add` recorded.
///
/// The frozen `w_bf16` carries `requires_grad=false`, so the matmul
/// backward only emits `grad_input` (no `grad_weight` work). The scale
/// multiply broadcasts `[out]` over the last dim of `[..., out]`, which
/// matches the torchao expression where `scale` (shape `[out]`) is
/// implicitly broadcast against the matmul output.
///
/// ## Shapes
///
/// - `x`:    BF16, `[..., in_]`
/// - `w`:    `[out, in_]`
/// - `bias`: optional BF16, `[out]`
/// - returns BF16, `[..., out]`
pub fn linear_int8_qt(
    x: &Tensor,
    w: &Int8QtWeight,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let [out, in_] = w.shape;
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "linear_int8_qt: input must be BF16, got {:?}",
            x.dtype()
        )));
    }
    let x_dims = x.shape().dims();
    if x_dims.is_empty() || x_dims[x_dims.len() - 1] != in_ {
        return Err(Error::ShapeMismatch {
            expected: Shape::from_dims(&[in_]),
            got: x.shape().clone(),
        });
    }
    if let Some(b) = bias {
        if b.dtype() != DType::BF16 {
            return Err(Error::InvalidOperation(format!(
                "linear_int8_qt: bias must be BF16, got {:?}",
                b.dtype()
            )));
        }
        if b.shape().dims() != [out] {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(&[out]),
                got: b.shape().clone(),
            });
        }
    }

    // 1. cast (frozen BF16 weight view)
    let w_bf16 = w.codes_to_bf16()?;
    debug_assert!(!w_bf16.requires_grad);

    // 2. matmul against w.T. transpose() is zero-copy stride flip.
    let w_t = w_bf16.transpose()?;
    let tmp = x.matmul(&w_t)?;

    // 3. broadcast multiply by per-row scale (cast to BF16 for the elt op).
    //    scale is F32 [out]; cast under no-grad since it's frozen.
    let scale_bf16 = w.scales.to_dtype_no_grad(DType::BF16)?;
    let scaled = tmp.mul(&scale_bf16)?;

    // 4. optional bias add.
    match bias {
        Some(b) => scaled.add(b),
        None => Ok(scaled),
    }
}
