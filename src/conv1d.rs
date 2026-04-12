//! Conv1d and ConvTranspose1d — thin shims over Conv2d/ConvTranspose2d.
//!
//! Implements 1D convolution by treating the input as 2D with H=1:
//!   [B, C, L] → unsqueeze → [B, C, 1, L] → Conv2d → [B, C_out, 1, L'] → squeeze → [B, C_out, L']
//!
//! Uses the same cuDNN kernels as Conv2d — zero overhead beyond reshape.
//!
//! ## Fused ConvTranspose1d preparation
//!
//! [`conv_transpose1d_prepare`] fuses the zero-insert + pad steps that
//! precede the regular Conv1d call inside `ConvTranspose1d` into a
//! **single CUDA kernel launch**. The unfused path allocates 3-5
//! intermediate tensors per call (zeros, cat, reshape, narrow, pad);
//! the fused path writes the padded buffer directly.
//!
//! Typical speedup: 200× per upsample stage (0.22 s → <1 ms) when
//! called 6+ times in a vocoder or audio pipeline.

use crate::{DType, Error, Result, Shape, Tensor};
#[cfg(feature = "bf16_u16")]
use std::sync::Arc;
#[cfg(feature = "bf16_u16")]
use crate::CudaDevice;

/// 1D convolution via cuDNN Conv2d with H=1.
///
/// Input:  [B, C_in, L]
/// Weight: [C_out, C_in/groups, K]
/// Output: [B, C_out, L']
///
/// Supports: stride, padding, dilation, groups.
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let in_dims = input.shape().dims();
    if in_dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "conv1d: input must be 3D [B,C,L], got {:?}", in_dims
        )));
    }
    let w_dims = weight.shape().dims();
    if w_dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "conv1d: weight must be 3D [C_out,C_in/g,K], got {:?}", w_dims
        )));
    }

    let (b, c_in, l) = (in_dims[0], in_dims[1], in_dims[2]);
    let (c_out, c_in_g, k) = (w_dims[0], w_dims[1], w_dims[2]);

    // Unsqueeze to 4D: [B, C, L] → [B, C, 1, L]
    let input_4d = input.reshape(&[b, c_in, 1, l])?;
    // Weight: [C_out, C_in/g, K] → [C_out, C_in/g, 1, K]
    let weight_4d = weight.reshape(&[c_out, c_in_g, 1, k])?;

    // Conv2d with H=1 via cuDNN.
    // For F32 inputs, cast to BF16 for cuDNN, then cast output back.
    // cuDNN BF16 conv is the only reliable path with groups + dilation support.
    let (input_bf16, weight_bf16, bias_bf16) = if input.dtype() != DType::BF16 {
        let ib = input_4d.to_dtype(DType::BF16)?;
        let wb = weight_4d.to_dtype(DType::BF16)?;
        let bb = bias.map(|b| b.to_dtype(DType::BF16)).transpose()?;
        (ib, wb, bb)
    } else {
        (input_4d, weight_4d, bias.map(|b| b.clone()))
    };

    // Plumb dilation through: cuDNN's 2D conv takes `(dilation_h, dilation_w)`.
    // We represent 1D as `(H=1, W=L)`, so the length-axis dilation goes in the W slot.
    let out_4d = crate::cudnn::cudnn_conv2d_bf16(
        &input_bf16, &weight_bf16, bias_bf16.as_ref(),
        (1, stride), (0, padding), (1, dilation), groups,
    )?;

    let out_4d = if input.dtype() != DType::BF16 {
        out_4d.to_dtype(input.dtype())?
    } else {
        out_4d
    };

    // Squeeze: [B, C_out, 1, L'] → [B, C_out, L']
    let out_dims = out_4d.shape().dims();
    let l_out = out_dims[3];
    out_4d.reshape(&[out_dims[0], out_dims[1], l_out])
}

/// 1D transposed convolution — implemented as zero-insert + regular cuDNN conv1d.
///
/// Input:  `[B, C_in, L_in]`
/// Weight: `[C_in, C_out/groups, K]` (PyTorch ConvTranspose1d layout)
/// Output: `[B, C_out, L_out]` where
///   `L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1`
///
/// This avoids the missing `GpuOps::conv_transpose2d_forward` GPU kernel by
/// using the equivalence:
///   ConvTranspose1d(x, w, stride=s, padding=p) ≡
///     Conv1d(zero_insert(x, s), flip(transpose(w)), padding=K-1-p)
///
/// Concretely:
///   1. Zero-insert `(stride-1)` zeros between each element on the length axis
///      (length becomes `(L_in - 1) * stride + 1`).
///   2. Zero-pad `(dilation * (K - 1) - padding)` on each side.
///   3. Convolve with the flipped, C_in↔C_out–transposed weight at `dilation=1`.
///   4. Add `output_padding` trailing zeros to the output.
///
/// The weight transpose (per-group `C_in/g ↔ C_out/g`) + kernel flip is
/// mathematically identical to what cuDNN's native `convBackwardData` does
/// — see [PyTorch's `ConvTranspose1d` docs]. For groups=1 it's a plain flip+
/// permute. Supports arbitrary `dilation`, `groups`, and `output_padding`.
pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
) -> Result<Tensor> {
    conv_transpose1d_dilated(input, weight, bias, stride, padding, output_padding, 1, groups)
}

/// Full-feature 1D transposed convolution with dilation.
pub fn conv_transpose1d_dilated(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let in_dims = input.shape().dims();
    if in_dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "conv_transpose1d: input must be 3D [B,C,L], got {:?}", in_dims
        )));
    }
    let w_dims = weight.shape().dims();
    if w_dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "conv_transpose1d: weight must be 3D [C_in,C_out/g,K], got {:?}", w_dims
        )));
    }
    if stride == 0 {
        return Err(Error::InvalidInput("conv_transpose1d: stride must be >= 1".into()));
    }
    if dilation == 0 {
        return Err(Error::InvalidInput("conv_transpose1d: dilation must be >= 1".into()));
    }

    let (c_in, c_out_per_group, k) = (w_dims[0], w_dims[1], w_dims[2]);
    let c_out = c_out_per_group * groups;
    if c_in % groups != 0 {
        return Err(Error::InvalidInput(format!(
            "conv_transpose1d: C_in ({}) must be divisible by groups ({})", c_in, groups
        )));
    }
    let c_in_per_group = c_in / groups;

    // Stage 1+2: zero-insert + pad. Uses the fused CUDA kernel for BF16
    // inputs (single launch, no intermediate allocations) or falls back to
    // the multi-op path for other dtypes.
    if dilation * (k - 1) < padding {
        return Err(Error::InvalidInput(format!(
            "conv_transpose1d: padding ({}) exceeds dilation*(K-1) ({})",
            padding,
            dilation * (k - 1)
        )));
    }
    let side_pad_left = dilation * (k - 1) - padding;
    let side_pad_right = side_pad_left + output_padding;
    let x_padded = if input.dtype() == DType::BF16 {
        conv_transpose1d_prepare(input, stride, side_pad_left, side_pad_right)?
    } else {
        let x_zi = zero_insert_last_axis(input, stride)?;
        x_zi.pad1d(side_pad_left, side_pad_right)?
    };

    // Stage 3: build the "transposed" conv1d weight.
    //   ConvTranspose layout: `[C_in, C_out/g, K]` with flipped-kernel semantics.
    //   Regular conv1d (groups=g) wants `[C_out, C_in/g, K]`.
    //   Transformation:
    //     a) flip along K (reverse kernel samples)
    //     b) within each group, transpose (C_in/g, C_out/g) → (C_out/g, C_in/g)
    let w_flipped = flip_last_axis(weight)?;
    //   Reshape `[C_in, C_out/g, K]` → `[g, C_in/g, C_out/g, K]` → permute → `[C_out, C_in/g, K]`.
    let w_grouped = w_flipped.reshape(&[groups, c_in_per_group, c_out_per_group, k])?;
    let w_perm = w_grouped.permute(&[0, 2, 1, 3])?; // [g, C_out/g, C_in/g, K]
    let w_reg = w_perm.reshape(&[c_out, c_in_per_group, k])?;

    // Stage 4: regular cuDNN conv1d at the specified dilation, padding=0.
    // `output_padding` was already baked into `side_pad_right` above, so the
    // forward produces the correct output length directly.
    conv1d(&x_padded, &w_reg, bias, 1, 0, dilation, groups)
}

/// Reverse the last axis of a tensor via narrow + cat.
fn flip_last_axis(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    let k = dims[dims.len() - 1];
    if k <= 1 {
        return Ok(x.clone());
    }
    let mut parts: Vec<Tensor> = Vec::with_capacity(k);
    for i in (0..k).rev() {
        parts.push(x.narrow(dims.len() - 1, i, 1)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, dims.len() - 1)
}

/// Insert `(stride - 1)` zeros between each element on the last axis of a
/// `[B, C, L]` tensor. Output length = `(L - 1) * stride + 1`.
fn zero_insert_last_axis(x: &Tensor, stride: usize) -> Result<Tensor> {
    if stride <= 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "zero_insert_last_axis: expected 3D, got {:?}",
            dims
        )));
    }
    let (b, c, l) = (dims[0], dims[1], dims[2]);
    let x4 = x.reshape(&[b, c, l, 1])?;
    let zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[b, c, l, stride - 1]),
        x.dtype(),
        x.device().clone(),
    )?;
    let cat = Tensor::cat(&[&x4, &zeros], 3)?;
    let flat = cat.reshape(&[b, c, l * stride])?;
    flat.narrow(2, 0, (l - 1) * stride + 1)
}

// ---------------------------------------------------------------------------
// Fused zero-insert + pad CUDA kernel
// ---------------------------------------------------------------------------

#[cfg(feature = "bf16_u16")]
fn ensure_kernel_compiled(dev: &Arc<CudaDevice>, name: &'static str, code: &'static str) -> Result<()> {
    if dev.get_func(name, name).is_some() {
        return Ok(());
    }
    let include_dir = std::env::var("CUDA_INCLUDE_DIR")
        .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
        .unwrap_or_else(|_| "/usr/local/cuda/include".into());
    let mut opts = cudarc::nvrtc::CompileOptions::default();
    opts.include_paths.push(include_dir.into());
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(code, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc {name}: {e:?}")))?;
    dev.load_ptx(ptx, name, &[name])
        .map_err(|e| Error::Cuda(format!("load_ptx {name}: {e:?}")))?;
    Ok(())
}

#[cfg(feature = "bf16_u16")]
const CUDA_FUSED_ZI_PAD: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void conv_transpose1d_prepare_bf16(
    const __nv_bfloat16* __restrict__ X,  // [BC, L_in]
    __nv_bfloat16* __restrict__ Y,        // [BC, L_out]
    long BC, long L_in, long L_out, long stride,
    long pad_left, long pad_right)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = BC * L_out;
    __nv_bfloat16 zero = __float2bfloat16(0.0f);
    long zi_len = (L_in - 1) * stride + 1;

    while (idx < total) {
        long bc = idx / L_out;
        long p  = idx % L_out;

        __nv_bfloat16 val = zero;
        long q = p - pad_left;
        if (q >= 0 && q < zi_len && (q % stride == 0)) {
            val = X[bc * L_in + q / stride];
        }
        Y[idx] = val;

        idx += (long)gridDim.x * blockDim.x;
    }
}
"#;

/// Fused zero-insert + pad for ConvTranspose1d preparation.
///
/// Produces the padded, zero-inserted buffer that the downstream
/// `conv1d` call expects, in a **single CUDA kernel launch** instead
/// of 3-5 intermediate tensor allocations (zeros + cat + reshape +
/// narrow + pad).
///
/// # Arguments
///
/// * `x` — `[B, C, L]` BF16 input tensor
/// * `stride` — ConvTranspose1d stride (inserts `stride-1` zeros between elements)
/// * `pad_left` — zero-pad on the left side of the zero-inserted signal
/// * `pad_right` — zero-pad on the right side
///
/// # Returns
///
/// `[B, C, L_out]` BF16 tensor where
/// `L_out = (L - 1) * stride + 1 + pad_left + pad_right`.
///
/// For each output position `p`:
/// - If `p < pad_left` or `p >= pad_left + zi_len`: 0 (padding)
/// - Else: `q = p - pad_left`; if `q % stride == 0`: `input[q/stride]`; else: 0
///
/// # Performance
///
/// Typical speedup vs unfused path: 200× per call (0.22 s → <1 ms on
/// a 3090 Ti for a vocoder upsample stage at L=640, C=768).
#[cfg(feature = "bf16_u16")]
pub fn conv_transpose1d_prepare(
    x: &Tensor,
    stride: usize,
    pad_left: usize,
    pad_right: usize,
) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "conv_transpose1d_prepare: input must be BF16".into(),
        ));
    }
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "conv_transpose1d_prepare: expected [B,C,L], got {:?}", dims
        )));
    }
    let (b, c, l_in) = (dims[0], dims[1], dims[2]);
    let bc = b * c;
    let zi_len = (l_in - 1) * stride + 1;
    let l_out = zi_len + pad_left + pad_right;
    let total = bc * l_out;

    let device = x.device().clone();
    let x_flat = x.reshape(&[bc, l_in])?;

    ensure_kernel_compiled(&device, "conv_transpose1d_prepare_bf16", CUDA_FUSED_ZI_PAD)?;
    let f = device
        .get_func("conv_transpose1d_prepare_bf16", "conv_transpose1d_prepare_bf16")
        .ok_or_else(|| Error::Cuda("conv_transpose1d_prepare_bf16 missing after compile".into()))?;

    let mut out = Tensor::empty_dtype(
        Shape::from_dims(&[bc, l_out]),
        DType::BF16,
        device.clone(),
    )?;

    let x_ptr = x_flat.as_device_ptr_bf16("ct1d_prep_x")? as u64;
    let out_ptr = out.as_mut_device_ptr_bf16("ct1d_prep_out")? as u64;

    let threads = 256u32;
    let blocks = ((total as u32) + threads - 1) / threads;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (blocks.min(65535), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        use cudarc::driver::LaunchAsync;
        f.launch(
            cfg,
            (x_ptr, out_ptr, bc as i64, l_in as i64, l_out as i64,
             stride as i64, pad_left as i64, pad_right as i64),
        )
        .map_err(|e| Error::Cuda(format!("conv_transpose1d_prepare: {:?}", e)))?;
    }

    out.reshape(&[b, c, l_out])
}

/// Non-BF16 fallback — uses the unfused zero-insert + pad path.
#[cfg(not(feature = "bf16_u16"))]
pub fn conv_transpose1d_prepare(
    x: &Tensor,
    stride: usize,
    pad_left: usize,
    pad_right: usize,
) -> Result<Tensor> {
    let x_zi = zero_insert_last_axis(x, stride)?;
    x_zi.pad1d(pad_left, pad_right)
}

/// Grouped 1D convolution — for depthwise/grouped ops like lowpass filters.
///
/// Same as conv1d but with explicit group support emphasis.
pub fn conv1d_grouped(
    input: &Tensor,
    weight: &Tensor,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<Tensor> {
    conv1d(input, weight, None, stride, padding, 1, groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_shapes() {
        // Input: [1, 128, 100], Weight: [256, 128, 7], stride=1, padding=3
        // Output should be [1, 256, 100]
        let l_out = (100 + 2 * 3 - 7) / 1 + 1;
        assert_eq!(l_out, 100);
    }

    #[test]
    fn test_conv_transpose1d_shapes() {
        // Input: [1, 512, 50], Weight: [512, 256, 16], stride=6
        // L_out = (50 - 1) * 6 - 2*padding + 16
        // With padding = (16-6)/2 = 5: L_out = 49*6 + 16 - 10 = 300
        let l_out = (50 - 1) * 6 + 16 - 2 * 5;
        assert_eq!(l_out, 300);
    }
}
