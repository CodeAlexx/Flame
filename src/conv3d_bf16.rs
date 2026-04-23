//! BF16 3D Convolution using im2vol + cuBLASLt GEMM
//!
//! Unfolds a 5D input `[N, C_in, D, H, W]` into a 2D column matrix via an
//! im2vol CUDA kernel (the 3D analogue of im2col), then performs a single
//! BF16 GEMM through `cuda_ops_bf16::gemm_bf16` to produce the convolution
//! output.  This avoids the per-element inner-loop of a naive conv kernel and
//! instead leverages tensor-core accelerated matrix multiplication.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::dtype::DType;
use crate::tensor_storage::TensorStorage;
use crate::{cuda_ops_bf16, Error, Result, Shape, Tensor, TensorId};

// ---------------------------------------------------------------------------
// CUDA kernel: im2vol
// ---------------------------------------------------------------------------
// Gathers elements from a BF16 5D input tensor into a 2D column matrix.
// Each thread handles one element of the output column matrix.
//
// Column matrix layout: [C_in * kD * kH * kW,  D_out * H_out * W_out]
//
// params is a packed i64 array:
//   [0]=C_in, [1]=D_in, [2]=H_in, [3]=W_in,
//   [4]=kD,   [5]=kH,   [6]=kW,
//   [7]=dD,   [8]=dH,   [9]=dW,
//   [10]=sD,  [11]=sH,  [12]=sW,
//   [13]=pD,  [14]=pH,  [15]=pW,
//   [16]=D_out, [17]=H_out, [18]=W_out,
//   [19]=col_rows, [20]=col_cols
const IM2VOL_BF16_KERNEL: &str = r#"
extern "C" __global__
void im2vol_bf16(
    const unsigned short* __restrict__ input,
    unsigned short*       __restrict__ columns,
    const long long*      __restrict__ params)
{
    int C_in  = (int)params[0];
    int D_in  = (int)params[1];
    int H_in  = (int)params[2];
    int W_in  = (int)params[3];
    int kD    = (int)params[4];
    int kH    = (int)params[5];
    int kW    = (int)params[6];
    int dD    = (int)params[7];
    int dH    = (int)params[8];
    int dW    = (int)params[9];
    int sD    = (int)params[10];
    int sH    = (int)params[11];
    int sW    = (int)params[12];
    int pD    = (int)params[13];
    int pH    = (int)params[14];
    int pW    = (int)params[15];
    int D_out = (int)params[16];
    int H_out = (int)params[17];
    int W_out = (int)params[18];
    int col_rows = (int)params[19];
    int col_cols = (int)params[20];

    long long total = (long long)col_rows * col_cols;

    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < total) {
        int col_c = (int)(tid / col_cols);
        int col_s = (int)(tid % col_cols);

        // Decode col_c -> (ic, kd, kh, kw)
        int kw_idx = col_c % kW;
        int tmp    = col_c / kW;
        int kh_idx = tmp % kH;
        tmp        = tmp / kH;
        int kd_idx = tmp % kD;
        int ic     = tmp / kD;

        // Decode col_s -> (od, oh, ow)
        int ow = col_s % W_out;
        int tmp2 = col_s / W_out;
        int oh = tmp2 % H_out;
        int od = tmp2 / H_out;

        // Input coordinates
        int id = od * sD - pD + kd_idx * dD;
        int ih = oh * sH - pH + kh_idx * dH;
        int iw = ow * sW - pW + kw_idx * dW;

        unsigned short val = 0;
        if (id >= 0 && id < D_in &&
            ih >= 0 && ih < H_in &&
            iw >= 0 && iw < W_in) {
            long long in_off = ((long long)ic * D_in + id) * (long long)(H_in * W_in)
                             + (long long)ih * W_in + iw;
            val = input[in_off];
        }
        columns[tid] = val;

        tid += (long long)gridDim.x * blockDim.x;
    }
}
"#;

// ---------------------------------------------------------------------------
// CUDA kernel: bias_add_bf16_conv3d
// ---------------------------------------------------------------------------
const BIAS_ADD_BF16_CONV3D: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void bias_add_bf16_conv3d(
    unsigned short* __restrict__ output,
    const unsigned short* __restrict__ bias,
    int C_out, long long spatial, long long total_elems)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < total_elems) {
        long long channel_spatial = (long long)C_out * spatial;
        long long within_batch = tid % channel_spatial;
        int c = (int)(within_batch / spatial);
        __nv_bfloat16 o = *((__nv_bfloat16*)&output[tid]);
        __nv_bfloat16 b = *((__nv_bfloat16*)&bias[c]);
        float val = __bfloat162float(o) + __bfloat162float(b);
        __nv_bfloat16 r = __float2bfloat16(val);
        output[tid] = *((unsigned short*)&r);
        tid += (long long)gridDim.x * blockDim.x;
    }
}
"#;

// ---------------------------------------------------------------------------
// CUDA kernel: copy_strided_bf16
// ---------------------------------------------------------------------------
// Copies `count` u16 elements from src to dst at given offsets.
const COPY_BF16_KERNEL: &str = r#"
extern "C" __global__
void copy_bf16(
    unsigned short* __restrict__ dst,
    const unsigned short* __restrict__ src,
    long long count)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < count) {
        dst[tid] = src[tid];
        tid += (long long)gridDim.x * blockDim.x;
    }
}
"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ensure(dev: &Arc<CudaDevice>, nm: &'static str, code: &'static str) -> Result<()> {
    if dev.get_func(nm, nm).is_some() {
        return Ok(());
    }
    let include_dir = std::env::var("CUDA_INCLUDE_DIR")
        .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
        .unwrap_or_else(|_| "/usr/local/cuda/include".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(include_dir);
    let ptx = compile_ptx_with_opts(code, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc {nm}: {e:?}")))?;
    dev.load_ptx(ptx, nm, &[nm])
        .map_err(|e| Error::Cuda(format!("load {nm}: {e:?}")))?;
    Ok(())
}

fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

/// Upload small i64 metadata array to device.
fn htod_params(dev: &Arc<CudaDevice>, data: &[i64]) -> Result<cudarc::driver::CudaSlice<i64>> {
    let mut gpu = unsafe { dev.alloc::<i64>(data.len()) }
        .map_err(|e| Error::Cuda(format!("alloc params: {e:?}")))?;
    dev.htod_copy_into(data.to_vec(), &mut gpu)
        .map_err(|e| Error::Cuda(format!("htod params: {e:?}")))?;
    Ok(gpu)
}

// ---------------------------------------------------------------------------
// Conv3dBF16
// ---------------------------------------------------------------------------

/// BF16 3D convolution layer backed by im2vol + cuBLASLt GEMM.
pub struct Conv3dBF16 {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub groups: usize,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl Conv3dBF16 {
    /// Build from pre-loaded weight and optional bias tensors.
    pub fn from_weights(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Self {
        Self::from_weights_with_config(weight, bias, stride, padding, (1, 1, 1), 1)
    }

    pub fn from_weights_with_config(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Self {
        let w = weight.shape().dims();
        assert!(w.len() == 5, "Conv3dBF16 weight must be 5D [C_out, C_in/groups, kD, kH, kW]");
        assert!(groups >= 1, "Conv3dBF16 groups must be >= 1");
        Conv3dBF16 {
            out_channels: w[0],
            in_channels: w[1] * groups,
            kernel_size: (w[2], w[3], w[4]),
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Forward pass.  `input` shape: `[N, C_in, D, H, W]`, dtype BF16.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // ---- validate ----
        if input.dtype() != DType::BF16 {
            return Err(Error::InvalidInput(format!(
                "Conv3dBF16: expected BF16 input, got {:?}",
                input.dtype()
            )));
        }
        let dims = input.shape().dims();
        if dims.len() != 5 {
            return Err(Error::InvalidInput(format!(
                "Conv3dBF16: expected 5D input [N,C,D,H,W], got {:?}",
                dims
            )));
        }
        let (batch, c_in, d_in, h_in, w_in) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        if c_in != self.in_channels {
            return Err(Error::InvalidInput(format!(
                "Conv3dBF16: expected {} input channels, got {}",
                self.in_channels, c_in
            )));
        }

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let (dd, dh, dw) = self.dilation;

        let d_out = (d_in + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
        let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

        #[cfg(feature = "cudnn")]
        {
            match crate::cudnn::cudnn_conv3d_bf16(
                input,
                &self.weight,
                self.bias.as_ref(),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            ) {
                Ok(out) => return Ok(out),
                Err(err) => {
                    if std::env::var_os("FLAME_CUDNN_CONV3D_STRICT").is_some() {
                        return Err(err);
                    }
                    if std::env::var_os("BF16_CONV_DEBUG").is_some() {
                        eprintln!("[conv3d_bf16] cuDNN path failed, falling back to im2vol+gemm: {err}");
                    }
                }
            }
        }

        let col_rows = c_in * kd * kh * kw;
        let col_cols = d_out * h_out * w_out;
        let col_numel = col_rows * col_cols;

        let dev = input.device().clone();

        if self.groups != 1 {
            return Err(Error::Unsupported(
                "Conv3dBF16 legacy fallback only supports groups=1; enable cuDNN for grouped Conv3d".into(),
            ));
        }

        // ---- weight reshaped to 2D [C_out, col_rows] (view, no copy) ----
        let weight_2d = self.weight.reshape(&[self.out_channels, col_rows])?;

        // ---- allocate column buffer (reused per batch element) ----
        let col_data = unsafe { dev.alloc::<u16>(col_numel) }
            .map_err(|e| Error::Cuda(format!("alloc im2vol columns: {e:?}")))?;
        let col_tensor = Tensor {
            storage: TensorStorage::BF16 {
                data: col_data.into(),
                numel: col_numel,
            },
            shape: Shape::from_dims(&[col_rows, col_cols]),
            device: dev.clone(),
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        };

        // ---- allocate output [N, C_out, D_out, H_out, W_out] ----
        let out_spatial = d_out * h_out * w_out;
        let out_numel = batch * self.out_channels * out_spatial;
        let out_data = unsafe { dev.alloc::<u16>(out_numel) }
            .map_err(|e| Error::Cuda(format!("alloc conv3d output: {e:?}")))?;
        let mut output = Tensor {
            storage: TensorStorage::BF16 {
                data: out_data.into(),
                numel: out_numel,
            },
            shape: Shape::from_dims(&[batch, self.out_channels, d_out, h_out, w_out]),
            device: dev.clone(),
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        };

        // ---- compile kernels ----
        ensure(&dev, "im2vol_bf16", IM2VOL_BF16_KERNEL)?;
        ensure(&dev, "copy_bf16", COPY_BF16_KERNEL)?;

        // ---- upload im2vol params (constant across batch) ----
        let params_host: Vec<i64> = vec![
            c_in as i64,
            d_in as i64,
            h_in as i64,
            w_in as i64,
            kd as i64,
            kh as i64,
            kw as i64,
            dd as i64,
            dh as i64,
            dw as i64,
            sd as i64,
            sh as i64,
            sw as i64,
            pd as i64,
            ph as i64,
            pw as i64,
            d_out as i64,
            h_out as i64,
            w_out as i64,
            col_rows as i64,
            col_cols as i64,
        ];
        let params_gpu = htod_params(&dev, &params_host)?;

        // ---- per-batch loop ----
        let input_batch_stride = c_in * d_in * h_in * w_in;
        let output_batch_stride = self.out_channels * out_spatial;

        let input_ptr = input.as_device_ptr_bf16("conv3d_bf16:input")? as u64;
        let col_ptr = col_tensor.as_device_ptr_bf16("conv3d_bf16:col")? as u64;
        let output_ptr = output.as_mut_device_ptr_bf16("conv3d_bf16:output")? as u64;

        for b in 0..batch {
            let in_offset = input_ptr + (b * input_batch_stride * 2) as u64;

            // --- im2vol: unfold input[b] into columns ---
            let im2vol_func = dev
                .get_func("im2vol_bf16", "im2vol_bf16")
                .ok_or_else(|| Error::Cuda("missing kernel: im2vol_bf16".into()))?;
            unsafe {
                im2vol_func
                    .launch(lc(col_numel), (in_offset, col_ptr, &params_gpu))
                    .map_err(|e| Error::Cuda(format!("im2vol launch: {e:?}")))?;
            }

            // --- GEMM: weight_2d [C_out, col_rows] @ col [col_rows, col_cols] -> [C_out, col_cols] ---
            let gemm_out = cuda_ops_bf16::gemm_bf16(&weight_2d, &col_tensor, None)?;

            // --- copy GEMM result into output[b] ---
            let gemm_ptr = gemm_out.as_device_ptr_bf16("conv3d_bf16:gemm")? as u64;
            let dst_ptr = output_ptr + (b * output_batch_stride * 2) as u64;

            let copy_func = dev
                .get_func("copy_bf16", "copy_bf16")
                .ok_or_else(|| Error::Cuda("missing kernel: copy_bf16".into()))?;
            unsafe {
                copy_func
                    .launch(
                        lc(output_batch_stride),
                        (dst_ptr, gemm_ptr, output_batch_stride as i64),
                    )
                    .map_err(|e| Error::Cuda(format!("copy launch: {e:?}")))?;
            }
        }

        // ---- add bias ----
        if let Some(ref bias) = self.bias {
            if bias.dtype() != DType::BF16 {
                return Err(Error::InvalidInput("Conv3dBF16: bias must be BF16".into()));
            }
            ensure(&dev, "bias_add_bf16_conv3d", BIAS_ADD_BF16_CONV3D)?;
            let bias_func = dev
                .get_func("bias_add_bf16_conv3d", "bias_add_bf16_conv3d")
                .ok_or_else(|| Error::Cuda("missing kernel: bias_add_bf16_conv3d".into()))?;

            let bias_ptr = bias.as_device_ptr_bf16("conv3d_bf16:bias")? as u64;
            let out_ptr = output.as_mut_device_ptr_bf16("conv3d_bf16:output_bias")? as u64;

            unsafe {
                bias_func
                    .launch(
                        lc(out_numel),
                        (
                            out_ptr,
                            bias_ptr,
                            self.out_channels as i32,
                            out_spatial as i64,
                            out_numel as i64,
                        ),
                    )
                    .map_err(|e| Error::Cuda(format!("bias_add launch: {e:?}")))?;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv3d_bf16_output_shape() -> Result<()> {
        let dev = crate::CudaDevice::new(0)?;

        let c_in = 4;
        let c_out = 8;
        let (kd, kh, kw) = (3, 3, 3);

        let weight = Tensor::zeros_dtype(
            Shape::from_dims(&[c_out, c_in, kd, kh, kw]),
            DType::BF16,
            dev.clone(),
        )?;

        let conv = Conv3dBF16::from_weights(weight, None, (1, 1, 1), (1, 1, 1));

        let input = Tensor::zeros_dtype(
            Shape::from_dims(&[1, c_in, 8, 16, 16]),
            DType::BF16,
            dev.clone(),
        )?;

        let output = conv.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, c_out, 8, 16, 16]);
        assert_eq!(output.dtype(), DType::BF16);
        Ok(())
    }

    #[test]
    fn test_conv3d_bf16_stride2() -> Result<()> {
        let dev = crate::CudaDevice::new(0)?;

        let c_in = 3;
        let c_out = 16;

        let weight = Tensor::zeros_dtype(
            Shape::from_dims(&[c_out, c_in, 3, 3, 3]),
            DType::BF16,
            dev.clone(),
        )?;

        let conv = Conv3dBF16::from_weights(weight, None, (2, 2, 2), (1, 1, 1));

        let input = Tensor::zeros_dtype(
            Shape::from_dims(&[2, c_in, 16, 32, 32]),
            DType::BF16,
            dev.clone(),
        )?;

        let output = conv.forward(&input)?;
        // d_out = (16+2-3)/2+1 = 8, h_out = (32+2-3)/2+1 = 16, w_out = 16
        assert_eq!(output.shape().dims(), &[2, c_out, 8, 16, 16]);
        Ok(())
    }
}
