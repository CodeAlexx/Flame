//! BF16 convolution helper that keeps tensors in BF16 while accumulating in FP32.

#![allow(clippy::too_many_arguments)]

use crate::memory_pool::Workspace;
use crate::{
    device::Device,
    strict::{scope, GuardMode},
    DType, Error, Result, Shape, Tensor,
};

use super::conv2d_bf16_cudnn;

#[derive(Clone, Copy, Debug)]
pub struct Conv2dBF16Cfg {
    pub stride: (i32, i32),
    pub pad: (i32, i32),
    pub dil: (i32, i32),
    pub groups: i32,
}

impl Default for Conv2dBF16Cfg {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            pad: (0, 0),
            dil: (1, 1),
            groups: 1,
        }
    }
}

pub struct Conv2dBF16 {
    workspace: Workspace,
    cfg: Conv2dBF16Cfg,
}

impl Conv2dBF16 {
    pub fn new(device: &Device, cfg: Conv2dBF16Cfg) -> Result<Self> {
        if !device.is_cuda() {
            return Err(Error::InvalidOperation(
                "Conv2dBF16 requires a CUDA device".into(),
            ));
        }
        Ok(Self {
            workspace: Workspace::new(device.cuda_device_arc()),
            cfg,
        })
    }

    pub fn forward(&mut self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        scope("conv2d_bf16.forward", GuardMode::env_default(), || {
            if x.dtype() != DType::BF16 || x.storage_dtype() != DType::BF16 {
                return Err(Error::InvalidInput(
                    "Conv2dBF16 expects BF16 input tensor".into(),
                ));
            }
            if w.dtype() != DType::BF16 || w.storage_dtype() != DType::BF16 {
                return Err(Error::InvalidInput(
                    "Conv2dBF16 expects BF16 weights".into(),
                ));
            }

            let x_dims = x.shape().dims();
            if x_dims.len() != 4 {
                return Err(Error::InvalidInput(
                    "Conv2dBF16 expects NHWC rank-4 input".into(),
                ));
            }
            let (n, h, w_in, c_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);

            let w_dims = w.shape().dims();
            if w_dims.len() != 4 {
                return Err(Error::InvalidInput(
                    "Conv2dBF16 expects HWIO weight tensor".into(),
                ));
            }
            enum WeightLayout {
                Hwio,
                Oihw,
            }

            let (kh, kw, _ic, oc, layout) = if w_dims[2] == c_in {
                (
                    w_dims[0],
                    w_dims[1],
                    w_dims[2],
                    w_dims[3],
                    WeightLayout::Hwio,
                )
            } else if w_dims[1] == c_in {
                (
                    w_dims[2],
                    w_dims[3],
                    w_dims[1],
                    w_dims[0],
                    WeightLayout::Oihw,
                )
            } else {
                return Err(Error::InvalidInput(format!(
                    "Conv2dBF16 weight channels mismatch: expected input {}, got layout {:?}",
                    c_in, w_dims
                )));
            };

            if self.cfg.groups != 1 {
                return Err(Error::InvalidOperation(
                    "Conv2dBF16 currently supports groups=1".into(),
                ));
            }
            if self.cfg.pad.0 != self.cfg.pad.1 {
                return Err(Error::InvalidInput(
                    "Conv2dBF16 requires symmetric padding".into(),
                ));
            }
            if self.cfg.stride.0 != self.cfg.stride.1 {
                return Err(Error::InvalidInput(
                    "Conv2dBF16 requires symmetric stride".into(),
                ));
            }
            if self.cfg.dil.0 != self.cfg.dil.1 {
                return Err(Error::InvalidInput(
                    "Conv2dBF16 requires symmetric dilation".into(),
                ));
            }

            let ho =
                Conv2dBF16::calc_out_dim(h, kh, self.cfg.pad.0, self.cfg.stride.0, self.cfg.dil.0);
            let wo = Conv2dBF16::calc_out_dim(
                w_in,
                kw,
                self.cfg.pad.1,
                self.cfg.stride.1,
                self.cfg.dil.1,
            );

            let x_nchw = x.permute(&[0, 3, 1, 2])?;
            let mut w_oihw_tmp = None;
            let w_oihw = match layout {
                WeightLayout::Hwio => {
                    w_oihw_tmp = Some(w.permute(&[3, 2, 0, 1])?);
                    w_oihw_tmp.as_ref().unwrap()
                }
                WeightLayout::Oihw => w,
            };

            let mut y_nchw = Tensor::zeros_dtype(
                Shape::from_dims(&[n, oc, ho, wo]),
                DType::BF16,
                x.device().clone(),
            )?;

            conv2d_bf16_cudnn::run(&mut self.workspace, &x_nchw, w_oihw, &mut y_nchw, self.cfg)?;

            let mut y_owned = y_nchw.permute(&[0, 2, 3, 1])?;

            if let Some(b) = bias {
                if b.dtype() != DType::BF16 || b.storage_dtype() != DType::BF16 {
                    return Err(Error::InvalidInput("Conv2dBF16 expects BF16 bias".into()));
                }
                let bias_view = b.reshape(&[1, 1, 1, oc])?;
                y_owned = y_owned.add(&bias_view)?;
            }

            self.workspace.clear();
            Ok(y_owned)
        })
    }

    fn calc_out_dim(input: usize, kernel: usize, pad: i32, stride: i32, dil: i32) -> usize {
        let kernel_extent = (kernel as i32 - 1) * dil + 1;
        ((input as i32 + 2 * pad - kernel_extent) / stride + 1) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cudnn::conv2d::cudnn_conv2d;

    #[test]
    fn dtype_and_shape() -> Result<()> {
        let device = match Device::cuda(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        };
        let arc = device.cuda_device_arc();
        let n = 1usize;
        let h = 8usize;
        let w = 8usize;
        let ic = 16usize;
        let oc = 32usize;
        let kh = 3usize;
        let kw = 3usize;

        let x = Tensor::randn(Shape::from_dims(&[n, h, w, ic]), 0.0, 1.0, arc.clone())?
            .to_dtype(DType::BF16)?;
        let wts = Tensor::randn(Shape::from_dims(&[kh, kw, ic, oc]), 0.0, 1.0, arc.clone())?
            .to_dtype(DType::BF16)?;

        let mut conv = Conv2dBF16::new(
            &device,
            Conv2dBF16Cfg {
                stride: (1, 1),
                pad: (1, 1),
                dil: (1, 1),
                groups: 1,
            },
        )?;
        let y = conv.forward(&x, &wts, None)?;

        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.shape().dims(), &[n, h, w, oc]);
        Ok(())
    }

    #[test]
    fn matches_f32_reference() -> Result<()> {
        let device = match Device::cuda(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        };
        let arc = device.cuda_device_arc();
        let n = 1usize;
        let h = 16usize;
        let w = 16usize;
        let ic = 8usize;
        let oc = 8usize;
        let kh = 3usize;
        let kw = 3usize;

        let x_bf16 = Tensor::randn(Shape::from_dims(&[n, h, w, ic]), 0.0, 1.0, arc.clone())?
            .to_dtype(DType::BF16)?;
        let w_bf16 = Tensor::randn(Shape::from_dims(&[kh, kw, ic, oc]), 0.0, 1.0, arc.clone())?
            .to_dtype(DType::BF16)?;

        let mut conv = Conv2dBF16::new(
            &device,
            Conv2dBF16Cfg {
                stride: (1, 1),
                pad: (1, 1),
                dil: (1, 1),
                groups: 1,
            },
        )?;
        let y_bf16 = conv.forward(&x_bf16, &w_bf16, None)?;
        let y_bf32 = y_bf16.to_dtype(DType::F32)?;

        let x_ref = x_bf16.to_dtype(DType::F32)?.permute(&[0, 3, 1, 2])?;
        let w_ref = w_bf16.to_dtype(DType::F32)?.permute(&[3, 2, 0, 1])?;
        let y_ref = cudnn_conv2d(&x_ref, &w_ref, None, 1, 1)?.permute(&[0, 2, 3, 1])?;

        let diff = y_bf32.sub(&y_ref)?;
        let diff_vec = diff.to_vec()?;
        let max_abs = diff_vec.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
        assert!(max_abs < 2e-3, "max abs diff {}", max_abs);
        Ok(())
    }
}
