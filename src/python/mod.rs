#![cfg(feature = "python")]

mod bridge;
mod functional;
mod nn;
mod tensor;

use pyo3::prelude::*;
use tensor::PyTensor;

use crate::{global_cuda_device, Shape, Tensor};
use tensor::parse_dtype;

/// The `flame_core` Python module.
#[pymodule]
fn flame_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;

    // Module-level factory functions
    m.add_function(wrap_pyfunction!(py_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(py_ones, m)?)?;
    m.add_function(wrap_pyfunction!(py_randn, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(py_cat, m)?)?;
    m.add_function(wrap_pyfunction!(bridge::from_torch, m)?)?;
    m.add_function(wrap_pyfunction!(bridge::to_torch, m)?)?;

    // Neural network layers
    m.add_class::<nn::PyLinear>()?;
    m.add_class::<nn::PyLayerNorm>()?;
    m.add_class::<nn::PyRMSNorm>()?;
    m.add_class::<nn::PyConv2d>()?;
    m.add_class::<nn::PyGroupNorm>()?;
    m.add_class::<nn::PyUpsample2d>()?;
    m.add_function(wrap_pyfunction!(nn::scaled_dot_product_attention, m)?)?;

    // Functional operations (stateless, module-level)
    functional::register(m)?;

    // Convenience: module-level from_numpy
    m.add_function(wrap_pyfunction!(py_from_numpy, m)?)?;

    // New bindings
    m.add_function(wrap_pyfunction!(py_save_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(py_arange, m)?)?;
    m.add_function(wrap_pyfunction!(py_stack, m)?)?;
    m.add_function(wrap_pyfunction!(py_randn_seeded, m)?)?;

    Ok(())
}

fn flame_err(e: crate::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

#[pyfunction(name = "zeros")]
fn py_zeros(shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
    let dt = parse_dtype(dtype)?;
    let device = global_cuda_device();
    let t = Tensor::zeros_dtype(Shape::from_dims(&shape), dt, device).map_err(flame_err)?;
    Ok(PyTensor { inner: t })
}

#[pyfunction(name = "ones")]
fn py_ones(shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
    let dt = parse_dtype(dtype)?;
    let device = global_cuda_device();
    let t = Tensor::ones_dtype(Shape::from_dims(&shape), dt, device).map_err(flame_err)?;
    Ok(PyTensor { inner: t })
}

#[pyfunction(name = "randn")]
fn py_randn(shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
    let dt = parse_dtype(dtype)?;
    let device = global_cuda_device();
    let t = Tensor::randn(Shape::from_dims(&shape), 0.0, 1.0, device).map_err(flame_err)?;
    if t.dtype() != dt {
        let t = t.to_dtype(dt).map_err(flame_err)?;
        Ok(PyTensor { inner: t })
    } else {
        Ok(PyTensor { inner: t })
    }
}

#[pyfunction(name = "load_safetensors")]
fn py_load_safetensors(path: &str) -> PyResult<std::collections::HashMap<String, PyTensor>> {
    let device = global_cuda_device();
    let tensors = crate::serialization::load_file(path, &device).map_err(flame_err)?;
    Ok(tensors
        .into_iter()
        .map(|(k, v)| (k, PyTensor { inner: v }))
        .collect())
}

/// Module-level from_numpy: `flame_core.from_numpy(array)`
#[pyfunction(name = "from_numpy")]
fn py_from_numpy(array: &Bound<'_, numpy::PyArrayDyn<f32>>) -> PyResult<PyTensor> {
    PyTensor::from_numpy_inner(array)
}

/// Module-level cat: `flame_core.cat([t1, t2, ...], dim)`
#[pyfunction(name = "cat")]
fn py_cat(tensors: Vec<PyRef<'_, PyTensor>>, dim: usize) -> PyResult<PyTensor> {
    let refs: Vec<&Tensor> = tensors.iter().map(|p| &p.inner).collect();
    let t = Tensor::cat(&refs, dim).map_err(flame_err)?;
    Ok(PyTensor { inner: t })
}

/// Save a dict of named tensors to a safetensors file.
#[pyfunction(name = "save_safetensors")]
fn py_save_safetensors(
    tensors: std::collections::HashMap<String, PyRef<'_, PyTensor>>,
    path: &str,
) -> PyResult<()> {
    let map: std::collections::HashMap<String, Tensor> = tensors
        .iter()
        .map(|(k, v)| (k.clone(), v.inner.clone()))
        .collect();
    crate::serialization::save_file(&map, std::path::Path::new(path)).map_err(flame_err)
}

/// Create a 1-D tensor with values [start, start+step, ...) up to (not including) end.
#[pyfunction(name = "arange")]
#[pyo3(signature = (start, end, step=None, dtype=None))]
fn py_arange(start: f32, end: f32, step: Option<f32>, dtype: Option<&str>) -> PyResult<PyTensor> {
    let s = step.unwrap_or(1.0);
    let dt = dtype.map(parse_dtype).unwrap_or(Ok(crate::DType::F32))?;
    let mut vals = Vec::new();
    let mut v = start;
    while v < end {
        vals.push(v);
        v += s;
    }
    let n = vals.len();
    let device = global_cuda_device();
    let t = Tensor::from_vec_dtype(vals, Shape::from_dims(&[n]), device, crate::DType::F32)
        .map_err(flame_err)?;
    if dt != crate::DType::F32 {
        Ok(PyTensor {
            inner: t.to_dtype(dt).map_err(flame_err)?,
        })
    } else {
        Ok(PyTensor { inner: t })
    }
}

/// Stack tensors along a new dimension (unsqueeze each at dim, then cat).
#[pyfunction(name = "stack")]
fn py_stack(tensors: Vec<PyRef<'_, PyTensor>>, dim: usize) -> PyResult<PyTensor> {
    let unsqueezed: Vec<Tensor> = tensors
        .iter()
        .map(|t| {
            let mut new_shape = t.inner.dims().to_vec();
            new_shape.insert(dim, 1);
            t.inner.reshape(&new_shape)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(flame_err)?;
    let refs: Vec<&Tensor> = unsqueezed.iter().collect();
    let t = Tensor::cat(&refs, dim).map_err(flame_err)?;
    Ok(PyTensor { inner: t })
}

/// Generate a deterministically-seeded normal(0,1) tensor.
#[pyfunction(name = "randn_seeded")]
fn py_randn_seeded(shape: Vec<usize>, dtype: &str, seed: u64) -> PyResult<PyTensor> {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    let dt = parse_dtype(dtype)?;
    let device = global_cuda_device();
    let numel: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0f32, 1.0f32).unwrap();
    let vals: Vec<f32> = (0..numel).map(|_| normal.sample(&mut rng)).collect();
    let t = Tensor::from_vec_dtype(vals, Shape::from_dims(&shape), device, crate::DType::F32)
        .map_err(flame_err)?;
    if dt != crate::DType::F32 {
        Ok(PyTensor {
            inner: t.to_dtype(dt).map_err(flame_err)?,
        })
    } else {
        Ok(PyTensor { inner: t })
    }
}
