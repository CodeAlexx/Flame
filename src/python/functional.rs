#![cfg(feature = "python")]

//! Module-level functional operations for the `flame_core` Python module.
//! These mirror PyTorch's `torch.nn.functional` namespace for use in
//! transformer forward passes.

use pyo3::prelude::*;

use super::tensor::PyTensor;
use crate::{cuda_ops::GpuOps, DType, Shape, Tensor};

fn flame_err(e: crate::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// SiLU activation: `x * sigmoid(x)`.
#[pyfunction]
pub fn silu(t: &PyTensor) -> PyResult<PyTensor> {
    let out = GpuOps::silu(&t.inner).map_err(flame_err)?;
    Ok(PyTensor { inner: out })
}

/// GELU activation (tanh approximation).
#[pyfunction]
pub fn gelu(t: &PyTensor) -> PyResult<PyTensor> {
    let out = GpuOps::gelu(&t.inner).map_err(flame_err)?;
    Ok(PyTensor { inner: out })
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Functional layer normalization.
///
/// Converts to BF16 internally when the input is F32 (the CUDA kernels
/// require BF16 storage) and converts back afterwards.
#[pyfunction]
#[pyo3(signature = (t, normalized_shape, weight=None, bias=None, eps=None))]
pub fn layer_norm(
    t: &PyTensor,
    normalized_shape: Vec<usize>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: Option<f32>,
) -> PyResult<PyTensor> {
    let eps = eps.unwrap_or(1e-5);
    let orig_dtype = t.inner.dtype();

    // The internal layer_norm requires BF16 — promote if needed.
    let input_bf16 = if orig_dtype != DType::BF16 {
        t.inner.to_dtype(DType::BF16).map_err(flame_err)?
    } else {
        t.inner.clone()
    };

    let w_bf16;
    let w_ref = match weight {
        Some(w) => {
            if w.inner.dtype() != DType::BF16 {
                w_bf16 = w.inner.to_dtype(DType::BF16).map_err(flame_err)?;
                Some(&w_bf16)
            } else {
                Some(&w.inner)
            }
        }
        None => {
            // satisfy the borrow checker — w_bf16 is never read on this path
            w_bf16 = input_bf16.clone();
            None
        }
    };

    let b_bf16;
    let b_ref = match bias {
        Some(b) => {
            if b.inner.dtype() != DType::BF16 {
                b_bf16 = b.inner.to_dtype(DType::BF16).map_err(flame_err)?;
                Some(&b_bf16)
            } else {
                Some(&b.inner)
            }
        }
        None => {
            b_bf16 = input_bf16.clone();
            None
        }
    };

    let out = crate::layer_norm::layer_norm(&input_bf16, &normalized_shape, w_ref, b_ref, eps)
        .map_err(flame_err)?;

    // Convert back to the caller's dtype.
    let out = if orig_dtype != DType::BF16 {
        out.to_dtype(orig_dtype).map_err(flame_err)?
    } else {
        out
    };

    Ok(PyTensor { inner: out })
}

/// RMS normalization: `x * rsqrt(mean(x^2, dim=-1) + eps) * weight`.
///
/// Uses the native BF16 CUDA kernel when available, otherwise composes
/// from tensor primitives.
#[pyfunction]
#[pyo3(signature = (t, weight, eps=None))]
pub fn rms_norm(t: &PyTensor, weight: &PyTensor, eps: Option<f32>) -> PyResult<PyTensor> {
    let eps = eps.unwrap_or(1e-5);
    let orig_dtype = t.inner.dtype();

    // The BF16 kernel is gated behind feature flags — attempt it first.
    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        let x_bf16 = if orig_dtype != DType::BF16 {
            t.inner.to_dtype(DType::BF16).map_err(flame_err)?
        } else {
            t.inner.clone()
        };
        let w_bf16 = if weight.inner.dtype() != DType::BF16 {
            weight.inner.to_dtype(DType::BF16).map_err(flame_err)?
        } else {
            weight.inner.clone()
        };

        let out =
            crate::cuda_ops_bf16::rms_norm_bf16(&x_bf16, Some(&w_bf16), eps).map_err(flame_err)?;

        let out = if orig_dtype != DType::BF16 {
            out.to_dtype(orig_dtype).map_err(flame_err)?
        } else {
            out
        };

        return Ok(PyTensor { inner: out });
    }

    // Fallback: compose from primitives (F32 path).
    #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
    {
        // x^2
        let x2 = t.inner.mul(&t.inner).map_err(flame_err)?;
        // mean over last dim
        let last_dim = t.inner.shape().dims().len() as isize - 1;
        let mean = x2.mean_dim(last_dim, true).map_err(flame_err)?;
        // + eps
        let shifted = mean.add_scalar(eps).map_err(flame_err)?;
        // rsqrt
        let scale = shifted.rsqrt().map_err(flame_err)?;
        // x * scale
        let normed = t.inner.mul(&scale).map_err(flame_err)?;
        // * weight
        let out = normed.mul(&weight.inner).map_err(flame_err)?;
        Ok(PyTensor { inner: out })
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Softmax along a dimension (default: last dimension).
#[pyfunction]
#[pyo3(signature = (t, dim=None))]
pub fn softmax(t: &PyTensor, dim: Option<i64>) -> PyResult<PyTensor> {
    let dim = dim.unwrap_or(-1) as isize;
    let out = t.inner.softmax(dim).map_err(flame_err)?;
    Ok(PyTensor { inner: out })
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Scaled dot-product attention.
///
/// Inputs: q, k, v are `[batch, heads, seq_len, head_dim]`.
/// Optional additive attention mask and causal flag.
///
/// When `is_causal` is true, a lower-triangular causal mask is generated
/// internally (the `attn_mask` argument is ignored in that case).
#[pyfunction]
#[pyo3(name = "sdpa", signature = (q, k, v, attn_mask=None, is_causal=None))]
pub fn sdpa(
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    attn_mask: Option<&PyTensor>,
    is_causal: Option<bool>,
) -> PyResult<PyTensor> {
    let is_causal = is_causal.unwrap_or(false);
    let orig_dtype = q.inner.dtype();

    // Promote to BF16 for the SDPA kernel.
    let q_bf16 = promote_bf16(&q.inner)?;
    let k_bf16 = promote_bf16(&k.inner)?;
    let v_bf16 = promote_bf16(&v.inner)?;

    let mask_bf16_owned;
    let mask_ref = if is_causal {
        // Build a causal mask: lower-triangular of ones, shape [1, 1, q_len, k_len].
        let q_len = q_bf16.shape().dims()[2];
        let k_len = k_bf16.shape().dims()[2];
        let mut mask_data = vec![0.0f32; q_len * k_len];
        for i in 0..q_len {
            for j in 0..=i.min(k_len - 1) {
                mask_data[i * k_len + j] = 1.0;
            }
        }
        let device = q_bf16.device().clone();
        let mask_f32 = Tensor::from_vec(
            mask_data,
            Shape::from_dims(&[1, 1, q_len, k_len]),
            device,
        )
        .map_err(flame_err)?;
        mask_bf16_owned = mask_f32.to_dtype(DType::BF16).map_err(flame_err)?;
        Some(&mask_bf16_owned)
    } else if let Some(m) = attn_mask {
        mask_bf16_owned = promote_bf16(&m.inner)?;
        Some(&mask_bf16_owned)
    } else {
        mask_bf16_owned = q_bf16.clone(); // placeholder, never read
        None
    };

    let out = crate::sdpa::forward(&q_bf16, &k_bf16, &v_bf16, mask_ref).map_err(flame_err)?;

    let out = if orig_dtype != DType::BF16 {
        out.to_dtype(orig_dtype).map_err(flame_err)?
    } else {
        out
    };

    Ok(PyTensor { inner: out })
}

/// Promote a tensor to BF16 if it isn't already.
fn promote_bf16(t: &Tensor) -> PyResult<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t.clone())
    } else {
        t.to_dtype(DType::BF16).map_err(flame_err)
    }
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Functional linear transform: `input @ weight.T + bias`.
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None))]
pub fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    // weight is [out_features, in_features] — we need weight^T.
    let wt = weight.inner.transpose().map_err(flame_err)?;
    let out = input.inner.matmul(&wt).map_err(flame_err)?;

    let out = match bias {
        Some(b) => out.add(&b.inner).map_err(flame_err)?,
        None => out,
    };

    Ok(PyTensor { inner: out })
}

// ---------------------------------------------------------------------------
// Upsampling
// ---------------------------------------------------------------------------

/// Nearest-neighbor 2D upsampling by an integer scale factor.
///
/// Input: `[N, C, H, W]`. Output: `[N, C, H*scale, W*scale]`.
#[pyfunction]
pub fn upsample_nearest_2d(input: &PyTensor, scale_factor: usize) -> PyResult<PyTensor> {
    let dims = input.inner.shape().dims();
    if dims.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "upsample_nearest_2d expects 4-D input [N, C, H, W]",
        ));
    }
    let h_out = dims[2] * scale_factor;
    let w_out = dims[3] * scale_factor;
    let out = GpuOps::upsample2d_nearest(&input.inner, (h_out, w_out)).map_err(flame_err)?;
    Ok(PyTensor { inner: out })
}

/// Register all functional operations on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(silu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    // Note: scaled_dot_product_attention is already registered via nn.rs.
    // This version adds attn_mask support; register under a distinct name.
    m.add_function(wrap_pyfunction!(self::sdpa, m)?)?;
    m.add_function(wrap_pyfunction!(linear, m)?)?;
    m.add_function(wrap_pyfunction!(upsample_nearest_2d, m)?)?;
    Ok(())
}
