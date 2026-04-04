#![cfg(feature = "python")]

use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::{global_cuda_device, DType, Shape, Tensor};

pub(crate) fn parse_dtype(s: &str) -> PyResult<DType> {
    match s {
        "f32" | "float32" => Ok(DType::F32),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" => Ok(DType::F16),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown dtype: {}",
            s
        ))),
    }
}

fn dtype_to_str(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        _ => "unknown",
    }
}

fn flame_err(e: crate::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

#[pyclass(name = "Tensor")]
pub struct PyTensor {
    pub(crate) inner: Tensor,
}

/// Public helpers for cross-module use within the python feature.
impl PyTensor {
    pub(crate) fn to_numpy_inner<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        // Cast to F32 if needed, then read back
        let src = if self.inner.dtype() != DType::F32 {
            self.inner.to_dtype(DType::F32).map_err(flame_err)?
        } else {
            self.inner.clone()
        };
        let data = src.to_vec().map_err(flame_err)?;
        let shape: Vec<usize> = self.inner.dims().to_vec();
        // Create a flat 1-D array then reshape to the correct dimensions
        let flat = numpy::PyArray1::from_vec_bound(py, data);
        let arr = flat
            .reshape(numpy::ndarray::IxDyn(&shape))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr)
    }

    pub(crate) fn from_numpy_inner(array: &Bound<'_, numpy::PyArrayDyn<f32>>) -> PyResult<Self> {
        let readonly = array.readonly();
        let slice = readonly.as_slice().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "numpy array must be contiguous: {}",
                e
            ))
        })?;
        let shape_vec: Vec<usize> = array.shape().to_vec();
        let device = global_cuda_device();
        let t =
            Tensor::from_slice_dtype(slice, Shape::from_dims(&shape_vec), device, DType::F32)
                .map_err(flame_err)?;
        Ok(Self { inner: t })
    }
}

#[pymethods]
impl PyTensor {
    // ── Factory methods ──────────────────────────────────────────────

    #[staticmethod]
    fn zeros(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let device = global_cuda_device();
        let t = Tensor::zeros_dtype(Shape::from_dims(&shape), dt, device).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let device = global_cuda_device();
        let t = Tensor::ones_dtype(Shape::from_dims(&shape), dt, device).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    #[staticmethod]
    fn randn(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let device = global_cuda_device();
        let t = Tensor::randn(Shape::from_dims(&shape), 0.0, 1.0, device).map_err(flame_err)?;
        if t.dtype() != dt {
            let t = t.to_dtype(dt).map_err(flame_err)?;
            Ok(Self { inner: t })
        } else {
            Ok(Self { inner: t })
        }
    }

    /// Copy tensor data to CPU and return as a numpy f32 array.
    /// BF16/F16 tensors are automatically cast to f32.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        self.to_numpy_inner(py)
    }

    /// Create a Flame tensor from a numpy f32 array, copying to GPU.
    #[staticmethod]
    fn from_numpy(array: &Bound<'_, numpy::PyArrayDyn<f32>>) -> PyResult<Self> {
        Self::from_numpy_inner(array)
    }

    // ── Properties ───────────────────────────────────────────────────

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }

    #[getter]
    fn dtype(&self) -> String {
        dtype_to_str(self.inner.dtype()).to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "flame_core.Tensor(shape={:?}, dtype={})",
            self.inner.dims(),
            dtype_to_str(self.inner.dtype()),
        )
    }

    /// Create a BF16 tensor from raw bytes (2 bytes per element, little-endian u16).
    /// No f32 intermediate — direct BF16 upload to GPU.
    #[staticmethod]
    fn from_bytes_bf16(data: &[u8], shape: Vec<usize>) -> PyResult<Self> {
        if data.len() % 2 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("BF16 data must have even byte count"));
        }
        let numel: usize = shape.iter().product();
        if data.len() / 2 != numel {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {} elements ({} bytes) for shape {:?}, got {} bytes",
                    numel, numel * 2, shape, data.len())
            ));
        }
        // Reinterpret bytes as u16 slice
        let u16_data: Vec<u16> = data.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let device = global_cuda_device();
        let s = crate::Shape::from_dims(&shape);
        let mut tensor = Tensor::zeros_dtype(s, DType::BF16, device).map_err(flame_err)?;
        tensor.copy_from_bf16_slice(&u16_data).map_err(flame_err)?;
        Ok(Self { inner: tensor })
    }

    /// Create an F32 tensor from raw bytes (4 bytes per element, little-endian f32).
    #[staticmethod]
    fn from_bytes_f32(data: &[u8], shape: Vec<usize>) -> PyResult<Self> {
        if data.len() % 4 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("F32 data must have byte count divisible by 4"));
        }
        let numel: usize = shape.iter().product();
        if data.len() / 4 != numel {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {} elements ({} bytes) for shape {:?}, got {} bytes",
                    numel, numel * 4, shape, data.len())
            ));
        }
        let f32_data: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let device = global_cuda_device();
        let t = Tensor::from_vec(f32_data, crate::Shape::from_dims(&shape), device).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn clone_tensor(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    fn to_dtype(&self, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let t = self.inner.to_dtype(dt).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    // ── Operator overloads ───────────────────────────────────────────

    fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.add(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn __sub__(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.sub(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn __mul__(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.mul(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn __truediv__(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.div(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn __matmul__(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.matmul(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn __neg__(&self) -> PyResult<Self> {
        let t = self.inner.neg().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    // ── Element-wise binary ops (named methods) ──────────────────────

    fn add(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.add(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn sub(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.sub(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn mul(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.mul(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn div(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.div(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        let t = self.inner.matmul(&other.inner).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    // ── Scalar operations ────────────────────────────────────────────

    fn mul_scalar(&self, s: f32) -> PyResult<Self> {
        let t = self.inner.mul_scalar(s).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn add_scalar(&self, s: f32) -> PyResult<Self> {
        let t = self.inner.add_scalar(s).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn sub_scalar(&self, s: f32) -> PyResult<Self> {
        let t = self.inner.sub_scalar(s).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn div_scalar(&self, s: f32) -> PyResult<Self> {
        let t = self.inner.div_scalar(s).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    // ── Shape operations ─────────────────────────────────────────────

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let t = self.inner.reshape(&shape).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Alias for reshape (PyTorch compat)
    fn view(&self, shape: Vec<usize>) -> PyResult<Self> {
        let t = self.inner.reshape(&shape).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// General permute with arbitrary dimension ordering.
    fn permute(&self, dims: Vec<usize>) -> PyResult<Self> {
        let t = self.inner.permute(&dims).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// 2-D transpose (swap last two dims for matrices).
    fn transpose(&self) -> PyResult<Self> {
        let t = self.inner.transpose().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Swap two arbitrary dimensions.
    fn transpose_dims(&self, d0: usize, d1: usize) -> PyResult<Self> {
        let t = self.inner.transpose_dims(d0, d1).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Insert a dimension of size 1.
    fn unsqueeze(&self, dim: usize) -> PyResult<Self> {
        let t = self.inner.unsqueeze(dim).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Remove a dimension of size 1.
    fn squeeze(&self, dim: usize) -> PyResult<Self> {
        let t = self.inner.squeeze(Some(dim)).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Flatten dimensions [start_dim .. end_dim] inclusive.
    /// Negative end_dim is relative to the rank (e.g. -1 = last dim).
    fn flatten(&self, start_dim: usize, end_dim: i64) -> PyResult<Self> {
        let dims = self.inner.dims();
        let rank = dims.len();

        let end = if end_dim < 0 {
            (rank as i64 + end_dim) as usize
        } else {
            end_dim as usize
        };

        if start_dim > end || end >= rank {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid flatten range [{}, {}] for rank {}",
                start_dim, end_dim, rank
            )));
        }

        // Compute the flattened shape
        let mut new_shape: Vec<usize> = Vec::new();
        for &d in &dims[..start_dim] {
            new_shape.push(d);
        }
        let flat_size: usize = dims[start_dim..=end].iter().product();
        new_shape.push(flat_size);
        for &d in &dims[end + 1..] {
            new_shape.push(d);
        }

        let t = self.inner.reshape(&new_shape).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Broadcast / expand to a larger shape (PyTorch-style expand).
    fn expand(&self, shape: Vec<usize>) -> PyResult<Self> {
        let t = self.inner.expand(&shape).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Return a contiguous copy (Flame tensors are always contiguous, so this clones).
    fn contiguous(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    // ── Narrow / slice / split / chunk / cat ─────────────────────────

    /// Narrow along a dimension: extract [start, start+length) along `dim`.
    fn narrow(&self, dim: usize, start: usize, length: usize) -> PyResult<Self> {
        let t = self.inner.narrow(dim, start, length).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Split into equal-sized chunks of `size` along `dim`.
    /// Returns a list of tensors.
    fn split(&self, size: usize, dim: usize) -> PyResult<Vec<Self>> {
        let dim_size = self.inner.dims()[dim];
        let n_full = dim_size / size;
        let remainder = dim_size % size;

        let mut sizes = vec![size; n_full];
        if remainder > 0 {
            sizes.push(remainder);
        }

        let parts = self.inner.split(&sizes, dim).map_err(flame_err)?;
        Ok(parts
            .into_iter()
            .map(|t| Self { inner: t })
            .collect())
    }

    /// Split into `n` chunks along `dim`.
    fn chunk(&self, n: usize, dim: usize) -> PyResult<Vec<Self>> {
        let parts = self.inner.chunk(n, dim).map_err(flame_err)?;
        Ok(parts
            .into_iter()
            .map(|t| Self { inner: t })
            .collect())
    }

    /// Concatenate a list of tensors along `dim`. (Exposed as staticmethod.)
    #[staticmethod]
    fn cat(tensors: Vec<PyRef<'_, PyTensor>>, dim: usize) -> PyResult<Self> {
        let refs: Vec<&Tensor> = tensors.iter().map(|p| &p.inner).collect();
        let t = Tensor::cat(&refs, dim).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    // ── Reduction operations ─────────────────────────────────────────

    /// Sum of all elements.
    fn sum(&self) -> PyResult<Self> {
        let t = self.inner.sum().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Mean of all elements.
    fn mean(&self) -> PyResult<Self> {
        let t = self.inner.mean().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    /// Sum along a single dimension.
    fn sum_dim(&self, dim: usize, keepdim: bool) -> PyResult<Self> {
        if keepdim {
            let t = self.inner.sum_dim_keepdim(dim).map_err(flame_err)?;
            Ok(Self { inner: t })
        } else {
            let t = self.inner.sum_dim_keepdim(dim).map_err(flame_err)?;
            let t = t.squeeze(Some(dim)).map_err(flame_err)?;
            Ok(Self { inner: t })
        }
    }

    /// Mean along a single dimension.
    fn mean_dim(&self, dim: usize, keepdim: bool) -> PyResult<Self> {
        let t = self.inner.mean_dim(&[dim], keepdim).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    // ── Unary math operations ────────────────────────────────────────

    fn sqrt(&self) -> PyResult<Self> {
        let t = self.inner.sqrt().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn rsqrt(&self) -> PyResult<Self> {
        let t = self.inner.rsqrt().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn exp(&self) -> PyResult<Self> {
        let t = self.inner.exp().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn log(&self) -> PyResult<Self> {
        let t = self.inner.log().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn abs(&self) -> PyResult<Self> {
        let t = self.inner.abs().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn neg(&self) -> PyResult<Self> {
        let t = self.inner.neg().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn sin(&self) -> PyResult<Self> {
        let t = self.inner.sin().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn cos(&self) -> PyResult<Self> {
        let t = self.inner.cos().map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    fn pow_scalar(&self, exponent: f32) -> PyResult<Self> {
        let t = self.inner.pow(exponent).map_err(flame_err)?;
        Ok(Self { inner: t })
    }

    // ── Serialization helpers ────────────────────────────────────────

    /// Return tensor data as a flat Vec<f32> (casting from bf16/f16 if needed).
    fn to_list(&self) -> PyResult<Vec<f32>> {
        let src = if self.inner.dtype() != DType::F32 {
            self.inner.to_dtype(DType::F32).map_err(flame_err)?
        } else {
            self.inner.clone()
        };
        src.to_vec().map_err(flame_err)
    }

    /// Create a Flame tensor from a flat Vec<f32>, reshape, and optionally cast.
    #[staticmethod]
    fn from_list(data: Vec<f32>, shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let device = global_cuda_device();
        let t = Tensor::from_vec_dtype(data, Shape::from_dims(&shape), device, DType::F32)
            .map_err(flame_err)?;
        if dt != DType::F32 {
            let t = t.to_dtype(dt).map_err(flame_err)?;
            Ok(PyTensor { inner: t })
        } else {
            Ok(PyTensor { inner: t })
        }
    }

    fn tanh(&self) -> PyResult<Self> {
        Ok(Self { inner: self.inner.tanh().map_err(flame_err)? })
    }

    fn clamp(&self, min: f32, max: f32) -> PyResult<Self> {
        // Flame's clamp goes through maximum/minimum which may produce BF16.
        // Fall back to BF16 clamp kernel or manual clamp if needed.
        match self.inner.clamp(min, max) {
            Ok(t) => Ok(Self { inner: t }),
            Err(_) => {
                // Workaround: cast to BF16, clamp, cast back
                let orig_dt = self.inner.dtype();
                let bf16 = self.inner.to_dtype(DType::BF16).map_err(flame_err)?;
                let clamped = crate::bf16_clamp::clamp_bf16(&bf16, min, max).map_err(flame_err)?;
                let t = clamped.to_dtype(orig_dt).map_err(flame_err)?;
                Ok(Self { inner: t })
            }
        }
    }
}
