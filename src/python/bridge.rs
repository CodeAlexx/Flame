#![cfg(feature = "python")]

use pyo3::prelude::*;

use super::tensor::PyTensor;

/// Convert a PyTorch tensor to a Flame tensor via CPU round-trip.
/// Phase 1 MVP: torch.tensor.cpu().float().numpy() -> Flame GPU tensor.
/// Phase 2 will add direct device-to-device CUDA memcpy.
#[pyfunction]
pub fn from_torch(py: Python<'_>, torch_tensor: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    // Ensure contiguous, move to CPU, cast to float32
    let contig = torch_tensor.call_method0("contiguous")?;
    let cpu = contig.call_method0("cpu")?;
    let f32_tensor = cpu.call_method0("float")?; // .float() = .to(torch.float32)
    let np_array = f32_tensor.call_method0("numpy")?;
    let array: &Bound<'_, numpy::PyArrayDyn<f32>> = np_array.downcast()?;
    PyTensor::from_numpy_inner(array)
}

/// Convert a Flame tensor to a PyTorch CUDA tensor via CPU round-trip.
/// Phase 1 MVP: Flame GPU tensor -> numpy -> torch.from_numpy().cuda()
/// Phase 2 will add direct device-to-device CUDA memcpy.
#[pyfunction]
pub fn to_torch(py: Python<'_>, tensor: &PyTensor) -> PyResult<PyObject> {
    let np_array = tensor.to_numpy_inner(py)?;
    let torch = py.import_bound("torch")?;
    let torch_tensor = torch.call_method1("from_numpy", (np_array,))?;
    let cuda_tensor = torch_tensor.call_method1("to", ("cuda",))?;
    Ok(cuda_tensor.unbind().into())
}
