#![cfg(feature = "python")]

//! PyO3 bindings for Flame neural network layers.
//!
//! Provides Python-accessible wrappers around Flame's Linear, LayerNorm,
//! RMSNorm, and scaled dot-product attention for Klein transformer inference.

use pyo3::prelude::*;

use super::tensor::PyTensor;

fn flame_err(e: crate::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Fully-connected linear layer wrapping Flame's `crate::linear::Linear`.
#[pyclass(name = "Linear")]
pub struct PyLinear {
    inner: crate::linear::Linear,
}

#[pymethods]
impl PyLinear {
    /// Build a Linear layer from pre-existing weight (and optional bias) tensors.
    ///
    /// `weight` shape must be `[out_features, in_features]`.
    #[staticmethod]
    #[pyo3(signature = (weight, bias=None))]
    fn from_weight(weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<Self> {
        let w = &weight.inner;
        let dims = w.shape().dims();
        if dims.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Linear weight must be 2-D [out_features, in_features]",
            ));
        }
        let out_features = dims[0];
        let in_features = dims[1];

        // Create a zeroed Linear, then copy the provided parameters in.
        let device = w.device();
        let has_bias = bias.is_some();
        let mut lin =
            crate::linear::Linear::new_zeroed(in_features, out_features, has_bias, device)
                .map_err(flame_err)?;
        lin.copy_weight_from(w).map_err(flame_err)?;
        if let Some(b) = bias {
            lin.copy_bias_from(&b.inner).map_err(flame_err)?;
        }
        Ok(Self { inner: lin })
    }

    /// Forward pass: `output = input @ weight^T + bias`.
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward(&input.inner).map_err(flame_err)?;
        Ok(PyTensor { inner: out })
    }

    /// Return a reference to the weight tensor.
    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight.clone(),
        })
    }

    /// Return the bias tensor, or None.
    #[getter]
    fn bias(&self) -> Option<PyTensor> {
        self.inner
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    fn __repr__(&self) -> String {
        format!(
            "flame_core.Linear(in_features={}, out_features={}, bias={})",
            self.inner.in_features(),
            self.inner.out_features(),
            self.inner.has_bias(),
        )
    }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// Layer normalization: `(x - mean) / sqrt(var + eps) * weight + bias`.
#[pyclass(name = "LayerNorm")]
pub struct PyLayerNorm {
    inner: crate::layer_norm::LayerNorm,
}

#[pymethods]
impl PyLayerNorm {
    /// Create a LayerNorm layer.
    ///
    /// - `normalized_shape`: the shape of the last N dimensions to normalize over.
    /// - `eps`: epsilon for numerical stability (default 1e-5).
    /// - `weight` / `bias`: optional affine parameters. When omitted, a freshly
    ///   initialized affine (ones weight, zeros bias) is used.
    #[new]
    #[pyo3(signature = (normalized_shape, eps=None, weight=None, bias=None))]
    fn new(
        normalized_shape: Vec<usize>,
        eps: Option<f32>,
        weight: Option<&PyTensor>,
        bias: Option<&PyTensor>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let device = crate::global_cuda_device();
        let mut ln =
            crate::layer_norm::LayerNorm::new(normalized_shape, eps, device).map_err(flame_err)?;

        if let Some(w) = weight {
            ln.copy_weight_from(&w.inner).map_err(flame_err)?;
        }
        if let Some(b) = bias {
            ln.copy_bias_from(&b.inner).map_err(flame_err)?;
        }
        Ok(Self { inner: ln })
    }

    /// Forward pass.
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward(&input.inner).map_err(flame_err)?;
        Ok(PyTensor { inner: out })
    }

    fn __repr__(&self) -> String {
        format!(
            "flame_core.LayerNorm(normalized_shape={:?}, eps={})",
            self.inner.normalized_shape, self.inner.eps,
        )
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// RMS normalization: `x * rsqrt(mean(x^2) + eps) * weight`.
#[pyclass(name = "RMSNorm")]
pub struct PyRMSNorm {
    inner: crate::norm::RMSNorm,
}

#[pymethods]
impl PyRMSNorm {
    /// Build an RMSNorm from a pre-existing weight tensor.
    #[staticmethod]
    #[pyo3(signature = (weight, eps=None))]
    fn from_weight(weight: &PyTensor, eps: Option<f32>) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-6);
        let w = &weight.inner;
        let normalized_shape: Vec<usize> = w.shape().dims().to_vec();
        let device = w.device();
        let mut rms =
            crate::norm::RMSNorm::new(normalized_shape, eps, true, device.clone()).map_err(flame_err)?;
        rms.copy_weight_from(w).map_err(flame_err)?;
        Ok(Self { inner: rms })
    }

    /// Forward pass.
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward(&input.inner).map_err(flame_err)?;
        Ok(PyTensor { inner: out })
    }

    fn __repr__(&self) -> String {
        format!(
            "flame_core.RMSNorm(normalized_shape={:?}, eps={})",
            self.inner.normalized_shape, self.inner.eps,
        )
    }
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------

/// 2D Convolution layer wrapping Flame's `crate::conv::Conv2d`.
///
/// Flame's Conv2d internally handles NCHW<->NHWC conversion, so the Python
/// binding accepts standard NCHW tensors `[N, C, H, W]` and returns NCHW.
#[pyclass(name = "Conv2d")]
pub struct PyConv2d {
    inner: crate::conv::Conv2d,
}

#[pymethods]
impl PyConv2d {
    /// Build a Conv2d from a pre-existing weight tensor and optional bias.
    ///
    /// - `weight`: `[out_channels, in_channels/groups, kH, kW]` (BF16).
    /// - `bias`: optional `[out_channels]`.
    /// - `stride`: `[sH, sW]`.
    /// - `padding`: `[pH, pW]`.
    /// - `groups`: default 1.
    #[staticmethod]
    #[pyo3(signature = (weight, bias=None, stride=vec![1,1], padding=vec![0,0], groups=None))]
    fn from_weight(
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        groups: Option<usize>,
    ) -> PyResult<Self> {
        let w = &weight.inner;
        let dims = w.shape().dims();
        if dims.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Conv2d weight must be 4-D [out_channels, in_channels, kH, kW]",
            ));
        }
        let out_channels = dims[0];
        let in_channels_per_group = dims[1];
        let kh = dims[2];
        let kw = dims[3];
        let groups = groups.unwrap_or(1);
        let in_channels = in_channels_per_group * groups;

        let sh = stride.first().copied().unwrap_or(1);
        let sw = stride.get(1).copied().unwrap_or(sh);
        let ph = padding.first().copied().unwrap_or(0);
        let pw = padding.get(1).copied().unwrap_or(ph);

        let device = w.device();
        let has_bias = bias.is_some();

        let config = crate::conv::Conv2dConfig {
            in_channels,
            out_channels,
            kernel_size: (kh, kw),
            stride: (sh, sw),
            padding: (ph, pw),
            groups,
        };

        let mut conv =
            crate::conv::Conv2d::from_config_with_bias(config, device.clone(), has_bias)
                .map_err(flame_err)?;
        conv.copy_weight_from(w).map_err(flame_err)?;
        if let Some(b) = bias {
            conv.copy_bias_from(&b.inner).map_err(flame_err)?;
        }
        Ok(Self { inner: conv })
    }

    /// Forward pass. Input must be `[N, C, H, W]` BF16.
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward(&input.inner).map_err(flame_err)?;
        Ok(PyTensor { inner: out })
    }

    fn __repr__(&self) -> String {
        let c = &self.inner.config;
        format!(
            "flame_core.Conv2d(in={}, out={}, k={:?}, s={:?}, p={:?}, g={})",
            c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, c.groups,
        )
    }
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

/// Group normalization wrapping Flame's `crate::group_norm::GroupNorm`.
///
/// Accepts NCHW input `[N, C, H, W]` (uses `forward_nchw` internally).
#[pyclass(name = "GroupNorm")]
pub struct PyGroupNorm {
    inner: crate::group_norm::GroupNorm,
}

#[pymethods]
impl PyGroupNorm {
    /// Build a GroupNorm from pre-existing weight and bias tensors.
    ///
    /// - `num_groups`: number of groups.
    /// - `num_channels`: number of channels.
    /// - `weight`: optional `[num_channels]` tensor.
    /// - `bias`: optional `[num_channels]` tensor.
    /// - `eps`: epsilon (default 1e-5).
    #[staticmethod]
    #[pyo3(signature = (num_groups, num_channels, weight=None, bias=None, eps=None))]
    fn from_weight(
        num_groups: usize,
        num_channels: usize,
        weight: Option<&PyTensor>,
        bias: Option<&PyTensor>,
        eps: Option<f32>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let affine = weight.is_some() || bias.is_some();
        let device = crate::global_cuda_device();
        let mut gn = crate::group_norm::GroupNorm::new(
            num_groups,
            num_channels,
            eps,
            affine,
            crate::DType::BF16,
            device,
        )
        .map_err(flame_err)?;

        if let Some(w) = weight {
            gn.weight = Some(w.inner.clone());
        }
        if let Some(b) = bias {
            gn.bias = Some(b.inner.clone());
        }
        Ok(Self { inner: gn })
    }

    /// Forward pass. Input: `[N, C, H, W]` (NCHW layout).
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward_nchw(&input.inner).map_err(flame_err)?;
        Ok(PyTensor { inner: out })
    }

    fn __repr__(&self) -> String {
        format!(
            "flame_core.GroupNorm(groups={}, channels={}, eps={}, affine={})",
            self.inner.num_groups,
            self.inner.num_channels,
            self.inner.eps,
            self.inner.affine,
        )
    }
}

// ---------------------------------------------------------------------------
// Upsample2d
// ---------------------------------------------------------------------------

/// 2D nearest-neighbor upsampling wrapping Flame's `crate::upsampling::Upsample2d`.
///
/// Input: `[N, C, H, W]`. Output: `[N, C, H*scale, W*scale]`.
#[pyclass(name = "Upsample2d")]
pub struct PyUpsample2d {
    inner: crate::upsampling::Upsample2d,
}

#[pymethods]
impl PyUpsample2d {
    /// Create an Upsample2d layer with the given integer scale factor.
    #[new]
    #[pyo3(signature = (scale_factor))]
    fn new(scale_factor: usize) -> PyResult<Self> {
        let config = crate::upsampling::Upsample2dConfig::new(
            crate::upsampling::UpsampleMode::Nearest,
        )
        .with_scale_factor((scale_factor as f32, scale_factor as f32));
        Ok(Self {
            inner: crate::upsampling::Upsample2d::new(config),
        })
    }

    /// Forward pass. Input: `[N, C, H, W]`.
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward(&input.inner).map_err(flame_err)?;
        Ok(PyTensor { inner: out })
    }

    fn __repr__(&self) -> String {
        format!(
            "flame_core.Upsample2d(scale_factor={:?})",
            self.inner.config.scale_factor,
        )
    }
}

// ---------------------------------------------------------------------------
// Scaled Dot-Product Attention (module-level function)
// ---------------------------------------------------------------------------

/// Scaled dot-product attention.
///
/// Inputs are `[B, H, S, D]` BF16 tensors. Uses Flame's streaming SDPA kernel.
/// When `is_causal` is True a causal (lower-triangular) mask is applied natively
/// by the streaming kernel (no explicit mask tensor needed).
#[pyfunction]
#[pyo3(signature = (q, k, v, is_causal=None))]
pub fn scaled_dot_product_attention(
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    is_causal: Option<bool>,
) -> PyResult<PyTensor> {
    let causal = is_causal.unwrap_or(false);

    // Call the streaming BF16 SDPA kernel directly so we can pass the
    // native `causal` flag without needing to build a mask tensor.
    let q_dims = q.inner.shape().dims();
    if q_dims.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SDPA expects 4-D input [B, H, S, D]",
        ));
    }
    let q_len = q_dims[2];
    let d_q = q_dims[3];
    let scale = 1.0 / (d_q as f32).sqrt();
    let chunk = q_len.min(2048).max(1);

    let mut out = crate::cuda_ops_bf16::sdpa_stream_bf16(
        &q.inner,
        &k.inner,
        &v.inner,
        None,   // no explicit mask
        chunk,
        causal,
        Some(scale),
    )
    .map_err(flame_err)?;

    if out.dtype() != crate::DType::BF16 {
        out = out.to_dtype(crate::DType::BF16).map_err(flame_err)?;
    }
    Ok(PyTensor { inner: out })
}
