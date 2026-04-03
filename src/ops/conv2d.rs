use crate::ops::gemm;
use crate::{config, DType, Error, Result, Shape, Tensor};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::{Arc, OnceLock};

const IM2COL_PTX: &str = r#"
#include <cuda_bf16.h>

extern "C" __device__ inline unsigned xorshift32(unsigned x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

extern "C" __global__ void im2col_f32(
    const float* __restrict__ x,
    float* __restrict__ col,
    const int* __restrict__ dims,
    const int* __restrict__ params)
{
    int N  = dims[0];
    int C  = dims[1];
    int H  = dims[2];
    int W  = dims[3];
    int KH = dims[4];
    int KW = dims[5];

    int SH = params[0];
    int SW = params[1];
    int PH = params[2];
    int PW = params[3];
    int DH = params[4];
    int DW = params[5];
    int OH = params[6];
    int OW = params[7];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_hw = N * OH * OW;
    if (idx >= n_hw * C) return;

    int hw = idx / C;
    int n = hw / (OH * OW);
    int rem = hw % (OH * OW);
    int oh = rem / OW;
    int ow = rem % OW;
    int c = idx % C;

    int base_col = ((n * OH + oh) * OW + ow);

    for (int kh = 0; kh < KH; ++kh) {
        int ih = oh * SH - PH + kh * DH;
        for (int kw = 0; kw < KW; ++kw) {
            int iw = ow * SW - PW + kw * DW;
            float val = 0.f;
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                int x_offset = ((n * C + c) * H + ih) * W + iw;
                val = x[x_offset];
            }
            int k = ((c * KH) + kh) * KW + kw;
            int col_offset = (base_col * (C * KH * KW)) + k;
            col[col_offset] = val;
        }
    }
}

extern "C" __global__ void im2col_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ col,
    const int* __restrict__ dims,
    const int* __restrict__ params)
{
    int N  = dims[0];
    int C  = dims[1];
    int H  = dims[2];
    int W  = dims[3];
    int KH = dims[4];
    int KW = dims[5];

    int SH = params[0];
    int SW = params[1];
    int PH = params[2];
    int PW = params[3];
    int DH = params[4];
    int DW = params[5];
    int OH = params[6];
    int OW = params[7];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_hw = N * OH * OW;
    if (idx >= n_hw * C) return;

    int hw = idx / C;
    int n = hw / (OH * OW);
    int rem = hw % (OH * OW);
    int oh = rem / OW;
    int ow = rem % OW;
    int c = idx % C;

    int base_col = ((n * OH + oh) * OW + ow);

    for (int kh = 0; kh < KH; ++kh) {
        int ih = oh * SH - PH + kh * DH;
        for (int kw = 0; kw < KW; ++kw) {
            int iw = ow * SW - PW + kw * DW;
            __nv_bfloat16 val = __float2bfloat16_rn(0.f);
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                int x_offset = ((n * C + c) * H + ih) * W + iw;
                val = x[x_offset];
            }
            int k = ((c * KH) + kh) * KW + kw;
            int col_offset = (base_col * (C * KH * KW)) + k;
            col[col_offset] = val;
        }
    }
}
"#;

static IM2COL_MODULE: OnceLock<()> = OnceLock::new();

const NHWC_TO_NCHW_PTX: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void nhwc_to_nchw_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n = idx / (C * W * H);

    int out_idx = ((n * C + c) * H + h) * W + w;
    y[out_idx] = x[idx];
}

extern "C" __global__ void nhwc_to_nchw_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n = idx / (C * W * H);

    int out_idx = ((n * C + c) * H + h) * W + w;
    y[out_idx] = x[idx];
}
"#;

static NHWC2NCHW_MODULE: OnceLock<()> = OnceLock::new();
const TRANSPOSE_PTX: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void transpose_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int r = idx / cols;
    int c = idx % cols;
    y[c * rows + r] = x[r * cols + c];
}

extern "C" __global__ void transpose_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int rows,
    int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int r = idx / cols;
    int c = idx % cols;
    y[c * rows + r] = x[r * cols + c];
}
"#;
static TRANSPOSE_MODULE: OnceLock<()> = OnceLock::new();

const BIAS_ADD_PTX: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void bias_add_nchw_f32(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N,
    int C,
    int H,
    int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int bias_idx = c; // bias shaped [1, C, 1, 1]
    y[idx] = x[idx] + bias[bias_idx];
}

extern "C" __global__ void bias_add_nchw_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    int N,
    int C,
    int H,
    int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int bias_idx = c;
    float xv = __bfloat162float(x[idx]);
    float bv = __bfloat162float(bias[bias_idx]);
    y[idx] = __float2bfloat16_rn(xv + bv);
}
"#;
static BIAS_ADD_MODULE: OnceLock<()> = OnceLock::new();

fn ensure_im2col_module(device: &Arc<CudaDevice>) -> Result<()> {
    if device.get_func("flame_conv_im2col", "im2col_f32").is_some() {
        return Ok(());
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(format!("{cuda_home}/include"));
    let ptx = compile_ptx_with_opts(IM2COL_PTX, opts)
        .map_err(|e| Error::KernelError(format!("im2col compile failed: {e}")))?;

    if let Err(e) = device.load_ptx(ptx, "flame_conv_im2col", &["im2col_f32", "im2col_bf16"]) {
        let err_msg = e.to_string();
        if !err_msg.contains("already loaded") {
            return Err(Error::KernelError(format!("load_ptx im2col: {err_msg}")));
        }
    }

    IM2COL_MODULE.get_or_init(|| ());
    Ok(())
}

fn ensure_nhwc2nchw_module(device: &Arc<CudaDevice>) -> Result<()> {
    if device
        .get_func("flame_conv_nhwc2nchw", "nhwc_to_nchw_f32")
        .is_some()
    {
        return Ok(());
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(format!("{cuda_home}/include"));
    let ptx = compile_ptx_with_opts(NHWC_TO_NCHW_PTX, opts)
        .map_err(|e| Error::KernelError(format!("nhwc->nchw compile failed: {e}")))?;

    if let Err(e) = device.load_ptx(
        ptx,
        "flame_conv_nhwc2nchw",
        &["nhwc_to_nchw_f32", "nhwc_to_nchw_bf16"],
    ) {
        let msg = e.to_string();
        if !msg.contains("already loaded") {
            return Err(Error::KernelError(format!("load_ptx nhwc->nchw: {msg}")));
        }
    }

    NHWC2NCHW_MODULE.get_or_init(|| ());
    Ok(())
}

fn ensure_transpose_module(device: &Arc<CudaDevice>) -> Result<()> {
    if device
        .get_func("flame_conv_transpose", "transpose_f32")
        .is_some()
    {
        return Ok(());
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(format!("{cuda_home}/include"));
    let ptx = compile_ptx_with_opts(TRANSPOSE_PTX, opts)
        .map_err(|e| Error::KernelError(format!("transpose compile failed: {e}")))?;

    if let Err(e) = device.load_ptx(
        ptx,
        "flame_conv_transpose",
        &["transpose_f32", "transpose_bf16"],
    ) {
        let msg = e.to_string();
        if !msg.contains("already loaded") {
            return Err(Error::KernelError(format!("load_ptx transpose: {msg}")));
        }
    }

    TRANSPOSE_MODULE.get_or_init(|| ());
    Ok(())
}

fn ensure_bias_add_module(device: &Arc<CudaDevice>) -> Result<()> {
    if device
        .get_func("flame_conv_bias", "bias_add_nchw_f32")
        .is_some()
    {
        return Ok(());
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(format!("{cuda_home}/include"));
    let ptx = compile_ptx_with_opts(BIAS_ADD_PTX, opts)
        .map_err(|e| Error::KernelError(format!("bias add compile failed: {e}")))?;

    if let Err(e) = device.load_ptx(
        ptx,
        "flame_conv_bias",
        &["bias_add_nchw_f32", "bias_add_nchw_bf16"],
    ) {
        let msg = e.to_string();
        if !msg.contains("already loaded") {
            return Err(Error::KernelError(format!("load_ptx bias add: {msg}")));
        }
    }

    BIAS_ADD_MODULE.get_or_init(|| ());
    Ok(())
}

fn launch_im2col(
    input: &Tensor,
    batch: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor> {
    let device = input.device();
    ensure_im2col_module(device)?;

    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    let out_height = (in_height + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let out_width = (in_width + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    let col_rows = batch * out_height * out_width;
    let col_cols = in_channels * kh * kw;

    let mut col = Tensor::zeros_dtype(
        Shape::from_dims(&[col_rows, col_cols]),
        input.dtype(),
        device.clone(),
    )?;

    #[cfg(feature = "dtype_trace")]
    crate::dtype_trace!(
        "im2col buffer allocated: dtype={:?} rows={} cols={}",
        col.dtype(),
        col_rows,
        col_cols
    );

    let cfg = LaunchConfig::for_num_elems((col_rows * in_channels) as u32);

    let dims = [
        batch as i32,
        in_channels as i32,
        in_height as i32,
        in_width as i32,
        kh as i32,
        kw as i32,
    ];
    let params = [
        sh as i32,
        sw as i32,
        ph as i32,
        pw as i32,
        dh as i32,
        dw as i32,
        out_height as i32,
        out_width as i32,
    ];
    let dims_gpu = copy_i32_to_device(device, &dims)?;
    let params_gpu = copy_i32_to_device(device, &params)?;

    match input.dtype() {
        DType::F32 => {
            let func = device
                .get_func("flame_conv_im2col", "im2col_f32")
                .ok_or_else(|| Error::KernelError("missing im2col_f32".into()))?;
            let x = match input.storage_ref().try_as_slice_f32() {
                Ok(slice) => slice,
                Err(err) => {
                    #[cfg(feature = "dtype_trace")]
                    crate::dtype_trace!(
                        "im2col f32 input slice failure: input dtype={:?}",
                        input.dtype()
                    );
                    return Err(err);
                }
            };
            let col_buf = match col.storage_mut().try_as_mut_slice_f32() {
                Ok(buf) => buf,
                Err(err) => {
                    #[cfg(feature = "dtype_trace")]
                    crate::dtype_trace!("im2col f32 col buffer failure: dtype={:?}", col.dtype());
                    return Err(err);
                }
            };
            crate::launch_kernel!(func, cfg, x, &*col_buf, &dims_gpu, &params_gpu);
        }
        DType::BF16 => {
            #[cfg(feature = "bf16_u16")]
            {
                let func = device
                    .get_func("flame_conv_im2col", "im2col_bf16")
                    .ok_or_else(|| Error::KernelError("missing im2col_bf16".into()))?;
                let x_ptr = input.as_device_ptr_bf16("im2col:input")? as u64;
                let col_ptr = col.as_mut_device_ptr_bf16("im2col:col")? as u64;
                crate::launch_kernel!(func, cfg, x_ptr, col_ptr, &dims_gpu, &params_gpu);
            }
            #[cfg(not(feature = "bf16_u16"))]
            {
                return Err(Error::Unsupported(
                    "BF16 requires the bf16_u16 feature".into(),
                ));
            }
        }
        other => {
            return Err(Error::InvalidInput(format!(
                "im2col unsupported dtype: {other:?}"
            )))
        }
    }

    Ok(col)
}

fn launch_nhwc_to_nchw(
    input: &Tensor,
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> Result<Tensor> {
    let device = input.device();
    ensure_nhwc2nchw_module(device)?;

    let total = batch * height * width * channels;
    let mut output = Tensor::zeros_dtype(
        Shape::from_dims(&[batch, channels, height, width]),
        input.dtype(),
        device.clone(),
    )?;

    #[cfg(feature = "dtype_trace")]
    crate::dtype_trace!(
        "nhwc->nchw buffer allocated: dtype={:?} shape=[{}, {}, {}, {}]",
        output.dtype(),
        batch,
        channels,
        height,
        width
    );

    let cfg = LaunchConfig::for_num_elems(total as u32);

    match input.dtype() {
        DType::F32 => {
            let func = device
                .get_func("flame_conv_nhwc2nchw", "nhwc_to_nchw_f32")
                .ok_or_else(|| Error::KernelError("missing nhwc_to_nchw_f32".into()))?;
            let x = match input.storage_ref().try_as_slice_f32() {
                Ok(slice) => slice,
                Err(err) => {
                    #[cfg(feature = "dtype_trace")]
                    crate::dtype_trace!(
                        "nhwc->nchw f32 input slice failure: dtype={:?}",
                        input.dtype()
                    );
                    return Err(err);
                }
            };
            let y = match output.storage_mut().try_as_mut_slice_f32() {
                Ok(slice) => slice,
                Err(err) => {
                    #[cfg(feature = "dtype_trace")]
                    crate::dtype_trace!(
                        "nhwc->nchw f32 output slice failure: dtype={:?}",
                        output.dtype()
                    );
                    return Err(err);
                }
            };
            crate::launch_kernel!(
                func,
                cfg,
                x,
                &*y,
                batch as i32,
                height as i32,
                width as i32,
                channels as i32
            );
        }
        DType::BF16 => {
            #[cfg(feature = "bf16_u16")]
            {
                let func = device
                    .get_func("flame_conv_nhwc2nchw", "nhwc_to_nchw_bf16")
                    .ok_or_else(|| Error::KernelError("missing nhwc_to_nchw_bf16".into()))?;
                let x_ptr = input.as_device_ptr_bf16("nhwc2nchw:input")? as u64;
                let y_ptr = output.as_mut_device_ptr_bf16("nhwc2nchw:output")? as u64;
                crate::launch_kernel!(
                    func,
                    cfg,
                    x_ptr,
                    y_ptr,
                    batch as i32,
                    height as i32,
                    width as i32,
                    channels as i32
                );
            }
            #[cfg(not(feature = "bf16_u16"))]
            {
                return Err(Error::Unsupported(
                    "BF16 requires the bf16_u16 feature".into(),
                ));
            }
        }
        other => {
            return Err(Error::InvalidInput(format!(
                "nhwc->nchw unsupported dtype: {other:?}"
            )))
        }
    }

    #[cfg(feature = "dtype_trace")]
    crate::dtype_trace!("nhwc->nchw complete: output dtype={:?}", output.dtype());

    Ok(output)
}

fn transpose_matrix(input: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let device = input.device();
    ensure_transpose_module(device)?;

    let mut output = Tensor::zeros_dtype(
        Shape::from_dims(&[cols, rows]),
        input.dtype(),
        device.clone(),
    )?;

    let cfg = LaunchConfig::for_num_elems((rows * cols) as u32);

    match input.dtype() {
        DType::F32 => {
            let func = device
                .get_func("flame_conv_transpose", "transpose_f32")
                .ok_or_else(|| Error::KernelError("missing transpose_f32".into()))?;
            let src = input.storage_ref().try_as_slice_f32()?;
            let dst = output.storage_mut().try_as_mut_slice_f32()?;
            crate::launch_kernel!(func, cfg, src, &*dst, rows as i32, cols as i32);
        }
        DType::BF16 => {
            #[cfg(feature = "bf16_u16")]
            {
                let func = device
                    .get_func("flame_conv_transpose", "transpose_bf16")
                    .ok_or_else(|| Error::KernelError("missing transpose_bf16".into()))?;
                let src_ptr = input.as_device_ptr_bf16("transpose:src")? as u64;
                let dst_ptr = output.as_mut_device_ptr_bf16("transpose:dst")? as u64;
                crate::launch_kernel!(func, cfg, src_ptr, dst_ptr, rows as i32, cols as i32);
            }
            #[cfg(not(feature = "bf16_u16"))]
            {
                return Err(Error::Unsupported(
                    "BF16 requires the bf16_u16 feature".into(),
                ));
            }
        }
        other => {
            return Err(Error::InvalidInput(format!(
                "transpose unsupported dtype: {other:?}"
            )))
        }
    }

    Ok(output)
}

fn bias_add_nchw_inplace(output: &mut Tensor, bias: &Tensor) -> Result<()> {
    if output.dtype() != bias.dtype() {
        return Err(Error::InvalidInput(format!(
            "dtype mismatch in bias add: output={:?} bias={:?}",
            output.dtype(),
            bias.dtype()
        )));
    }
    let device = output.device();
    ensure_bias_add_module(device)?;

    let shape = output.shape().dims();
    if shape.len() != 4 || bias.shape().dims() != [1, shape[1], 1, 1] {
        return Err(Error::InvalidInput(
            "bias_add_nchw expects output [N,C,H,W] and bias [1,C,1,1]".into(),
        ));
    }
    let n = shape[0] as i32;
    let c = shape[1] as i32;
    let h = shape[2] as i32;
    let w = shape[3] as i32;
    let total = (shape[0] * shape[1] * shape[2] * shape[3]) as u32;
    if total == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig::for_num_elems(total);

    let mut tmp = Tensor::zeros_dtype(output.shape().clone(), output.dtype(), device.clone())?;

    match output.dtype() {
        DType::F32 => {
            let func = device
                .get_func("flame_conv_bias", "bias_add_nchw_f32")
                .ok_or_else(|| Error::KernelError("missing bias_add_nchw_f32".into()))?;
            let x_slice = output.storage_ref().try_as_slice_f32()?;
            let bias_slice = bias.storage_ref().try_as_slice_f32()?;
            let out_slice = tmp.storage_mut().try_as_mut_slice_f32()?;
            unsafe {
                func.launch(cfg, (x_slice, bias_slice, out_slice, n, c, h, w))
                    .map_err(|e| Error::KernelError(e.to_string()))?;
            }
        }
        DType::BF16 => {
            #[cfg(feature = "bf16_u16")]
            {
                let func = device
                    .get_func("flame_conv_bias", "bias_add_nchw_bf16")
                    .ok_or_else(|| Error::KernelError("missing bias_add_nchw_bf16".into()))?;
                let x_ptr = output.as_device_ptr_bf16("bias_add:x")? as u64;
                let bias_ptr = bias.as_device_ptr_bf16("bias_add:bias")? as u64;
                let out_ptr = tmp.as_mut_device_ptr_bf16("bias_add:out")? as u64;
                unsafe {
                    func.launch(cfg, (x_ptr, bias_ptr, out_ptr, n, c, h, w))
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                }
            }
            #[cfg(not(feature = "bf16_u16"))]
            {
                return Err(Error::Unsupported("BF16 requires bf16_u16 feature".into()));
            }
        }
        other => {
            return Err(Error::InvalidInput(format!(
                "bias add unsupported dtype: {other:?}"
            )))
        }
    }

    *output = tmp;
    Ok(())
}

pub fn conv2d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> Result<Tensor> {
    if groups != 1 {
        return Err(Error::Unsupported(
            "conv2d groups != 1 not implemented".into(),
        ));
    }
    if input.dtype() != weight.dtype() {
        return Err(Error::InvalidInput(format!(
            "dtype mismatch in conv2d: input={:?} weight={:?}",
            input.dtype(),
            weight.dtype()
        )));
    }
    if let Some(b) = bias {
        if b.dtype() != input.dtype() {
            return Err(Error::InvalidInput(format!(
                "dtype mismatch in conv2d bias: input={:?} bias={:?}",
                input.dtype(),
                b.dtype()
            )));
        }
    }

    #[cfg(feature = "dtype_trace")]
    crate::dtype_trace!(
        "conv2d: dtype={:?}, stride={:?}, padding={:?}",
        input.dtype(),
        stride,
        padding
    );

    let input_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    if input_dims.len() != 4 || weight_dims.len() != 4 {
        return Err(Error::InvalidInput(
            "conv2d expects 4D input and weight".into(),
        ));
    }

    let batch = input_dims[0];
    let in_channels = input_dims[1];
    let in_height = input_dims[2];
    let in_width = input_dims[3];

    let out_channels = weight_dims[0];
    let kernel_h = weight_dims[2];
    let kernel_w = weight_dims[3];

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let dilation = (1usize, 1usize);

    let out_height = (in_height + 2 * pad_h - dilation.0 * (kernel_h - 1) - 1) / stride_h + 1;
    let out_width = (in_width + 2 * pad_w - dilation.1 * (kernel_w - 1) - 1) / stride_w + 1;

    let prev_dtype = config::default_dtype();
    config::set_default_dtype(input.dtype());

    let result = (|| {
        let col = launch_im2col(
            input,
            batch,
            in_channels,
            in_height,
            in_width,
            (kernel_h, kernel_w),
            stride,
            padding,
            dilation,
        )?;

        #[cfg(feature = "dtype_trace")]
        crate::dtype_trace!("conv2d->gemm: col.shape={:?}", col.shape().dims());

        let k = in_channels * kernel_h * kernel_w;
        let weight_reshaped = weight.reshape(&[out_channels, k])?;
        let weight_matrix = transpose_matrix(&weight_reshaped, out_channels, k)?;

        let y_2d = gemm::launch_gemm(&col, &weight_matrix)?;
        let y_nhwc = y_2d.reshape(&[batch, out_height, out_width, out_channels])?;
        let mut y = launch_nhwc_to_nchw(&y_nhwc, batch, out_height, out_width, out_channels)?;

        if let Some(b) = bias {
            let bias_view = b.reshape(&[1, out_channels, 1, 1])?;
            bias_add_nchw_inplace(&mut y, &bias_view)?;
        }

        Ok(y)
    })();

    config::set_default_dtype(prev_dtype);
    result
}
fn copy_i32_to_device(device: &Arc<CudaDevice>, data: &[i32]) -> Result<CudaSlice<i32>> {
    let mut buf = unsafe { device.alloc::<i32>(data.len()) }
        .map_err(|e| Error::KernelError(format!("alloc i32 buffer: {e}")))?;
    device
        .htod_copy_into(data.to_vec(), &mut buf)
        .map_err(|e| Error::KernelError(format!("copy i32 buffer: {e}")))?;
    Ok(buf)
}
