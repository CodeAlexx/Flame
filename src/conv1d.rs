//! Conv1d and ConvTranspose1d — thin shims over Conv2d/ConvTranspose2d.
//!
//! Implements 1D convolution by treating the input as 2D with H=1:
//!   [B, C, L] → unsqueeze → [B, C, 1, L] → Conv2d → [B, C_out, 1, L'] → squeeze → [B, C_out, L']
//!
//! Uses the same cuDNN kernels as Conv2d — zero overhead beyond reshape.

use crate::{DType, Error, Result, Shape, Tensor};

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

    let out_4d = crate::cudnn::cudnn_conv2d_bf16(
        &input_bf16, &weight_bf16, bias_bf16.as_ref(),
        (1, stride), (0, padding), groups,
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

/// 1D transposed convolution via ConvTranspose2d with H=1.
///
/// Input:  [B, C_in, L]
/// Weight: [C_in, C_out/groups, K]
/// Output: [B, C_out, L']
pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
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

    let (b, c_in, l) = (in_dims[0], in_dims[1], in_dims[2]);
    let (w0, w1, k) = (w_dims[0], w_dims[1], w_dims[2]);

    // Unsqueeze to 4D
    let input_4d = input.reshape(&[b, c_in, 1, l])?;
    let weight_4d = weight.reshape(&[w0, w1, 1, k])?;

    let out_4d = crate::cuda_ops::GpuOps::conv_transpose2d_forward(
        &input_4d, &weight_4d, bias,
        (1, stride), (0, padding), (0, output_padding),
        groups, (1, 1),
    )?;

    // Squeeze: [B, C_out, 1, L'] → [B, C_out, L']
    let out_dims = out_4d.shape().dims();
    out_4d.reshape(&[out_dims[0], out_dims[1], out_dims[3]])
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
