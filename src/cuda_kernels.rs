#![allow(dead_code, unused_imports, unused_variables, unused_mut)]
// Legacy CUDA kernels retained for reference.

use crate::cuda::ffi;
use crate::device::CudaStreamRawPtrExt;
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::DevicePtr;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx, compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;

// Import CUDA C kernel sources
use crate::cuda_kernel_sources::*;

#[inline]
fn ensure_f32_tensor(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 && t.storage_dtype() == DType::F32 {
        t.clone_result()
    } else {
        t.to_dtype(DType::F32)
    }
}

const CMP_GT_F32_KERNEL: &str = "cmp_gt_f32_kernel";
const CMP_GT_F32_CODE: &str = r#"
extern "C" __global__ void cmp_gt_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] > b[idx] ? 1.0f : 0.0f;
    }
}
"#;

const CMP_GE_F32_KERNEL: &str = "cmp_ge_f32_kernel";
const CMP_GE_F32_CODE: &str = r#"
extern "C" __global__ void cmp_ge_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] >= b[idx] ? 1.0f : 0.0f;
    }
}
"#;

const CMP_LT_F32_KERNEL: &str = "cmp_lt_f32_kernel";
const CMP_LT_F32_CODE: &str = r#"
extern "C" __global__ void cmp_lt_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] < b[idx] ? 1.0f : 0.0f;
    }
}
"#;

const CMP_LE_F32_KERNEL: &str = "cmp_le_f32_kernel";
const CMP_LE_F32_CODE: &str = r#"
extern "C" __global__ void cmp_le_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] <= b[idx] ? 1.0f : 0.0f;
    }
}
"#;

const CMP_EQ_F32_KERNEL: &str = "cmp_eq_f32_kernel";
const CMP_EQ_F32_CODE: &str = r#"
extern "C" __global__ void cmp_eq_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] == b[idx] ? 1.0f : 0.0f;
    }
}
"#;

const CMP_NE_F32_KERNEL: &str = "cmp_ne_f32_kernel";
const CMP_NE_F32_CODE: &str = r#"
extern "C" __global__ void cmp_ne_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] != b[idx] ? 1.0f : 0.0f;
    }
}
"#;

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

/// Helper to create output tensor from allocated data
pub fn create_output_tensor(data: CudaSlice<f32>, shape: Shape, device: Arc<CudaDevice>) -> Tensor {
    let numel = shape.elem_count();
    Tensor {
        storage: TensorStorage::F32 {
            data: data.into(),
            numel,
        },
        shape,
        device,
        id: crate::tensor::TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    }
}

/// GPU-only CUDA kernels using NVRTC runtime compilation
pub struct CudaKernels {
    pub device: Arc<CudaDevice>,
    pub kernels: HashMap<String, CudaFunction>,
}

impl CudaKernels {
    /// Create new GPU-only kernel instance with compiled kernels
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let mut kernels = HashMap::new();

        // Helper function to compile and load a kernel
        fn compile_and_load_kernel(
            device: &Arc<CudaDevice>,
            source: &str,
            kernel_name: &'static str,
        ) -> Result<CudaFunction> {
            let include_path = std::env::var("CUDA_HOME")
                .map(|p| format!("{}/include", p))
                .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
            let mut opts = CompileOptions::default();
            opts.include_paths.push(include_path);

            let ptx = compile_ptx_with_opts(source, opts)
                .map_err(|e| Error::Cuda(format!("Failed to compile {}: {:?}", kernel_name, e)))?;

            // Load the PTX module
            device.load_ptx(ptx, kernel_name, &[kernel_name])?;

            // Get the function
            device
                .get_func(kernel_name, kernel_name)
                .ok_or_else(|| Error::Cuda(format!("Failed to get function {}", kernel_name)))
        }

        // Compile all kernels
        kernels.insert(
            "add_kernel".to_string(),
            compile_and_load_kernel(&device, ADD_KERNEL, "add_kernel")?,
        );
        kernels.insert(
            "mul_kernel".to_string(),
            compile_and_load_kernel(&device, MUL_KERNEL, "mul_kernel")?,
        );
        kernels.insert(
            "mul_scalar_kernel".to_string(),
            compile_and_load_kernel(&device, MUL_SCALAR_KERNEL, "mul_scalar_kernel")?,
        );
        kernels.insert(
            "div_kernel".to_string(),
            compile_and_load_kernel(&device, DIV_KERNEL, "div_kernel")?,
        );
        kernels.insert(
            "add_scalar_kernel".to_string(),
            compile_and_load_kernel(&device, ADD_SCALAR_KERNEL, "add_scalar_kernel")?,
        );
        kernels.insert(
            "relu_kernel".to_string(),
            compile_and_load_kernel(&device, RELU_KERNEL, "relu_kernel")?,
        );
        kernels.insert(
            "sigmoid_kernel".to_string(),
            compile_and_load_kernel(&device, SIGMOID_KERNEL, "sigmoid_kernel")?,
        );
        kernels.insert(
            "gelu_kernel".to_string(),
            compile_and_load_kernel(&device, GELU_KERNEL, "gelu_kernel")?,
        );
        kernels.insert(
            "silu_kernel".to_string(),
            compile_and_load_kernel(&device, SILU_KERNEL, "silu_kernel")?,
        );
        kernels.insert(
            "tanh_kernel".to_string(),
            compile_and_load_kernel(&device, TANH_KERNEL, "tanh_kernel")?,
        );
        kernels.insert(
            "sum_kernel".to_string(),
            compile_and_load_kernel(&device, SUM_KERNEL, "sum_kernel")?,
        );
        kernels.insert(
            "transpose_kernel".to_string(),
            compile_and_load_kernel(&device, TRANSPOSE_KERNEL, "transpose_kernel")?,
        );
        kernels.insert(
            "update_weights_kernel".to_string(),
            compile_and_load_kernel(&device, UPDATE_WEIGHTS_KERNEL, "update_weights_kernel")?,
        );
        kernels.insert(
            "leaky_relu_kernel".to_string(),
            compile_and_load_kernel(&device, LEAKY_RELU_KERNEL, "leaky_relu_kernel")?,
        );
        kernels.insert(
            "elu_kernel".to_string(),
            compile_and_load_kernel(&device, ELU_KERNEL, "elu_kernel")?,
        );
        kernels.insert(
            "pow_kernel".to_string(),
            compile_and_load_kernel(&device, POW_KERNEL, "pow_kernel")?,
        );
        kernels.insert(
            "sin_kernel".to_string(),
            compile_and_load_kernel(&device, SIN_KERNEL, "sin_kernel")?,
        );
        kernels.insert(
            "cos_kernel".to_string(),
            compile_and_load_kernel(&device, COS_KERNEL, "cos_kernel")?,
        );
        kernels.insert(
            "floor_kernel".to_string(),
            compile_and_load_kernel(&device, FLOOR_KERNEL, "floor_kernel")?,
        );
        kernels.insert(
            "ceil_kernel".to_string(),
            compile_and_load_kernel(&device, CEIL_KERNEL, "ceil_kernel")?,
        );
        kernels.insert(
            "round_kernel".to_string(),
            compile_and_load_kernel(&device, ROUND_KERNEL, "round_kernel")?,
        );
        kernels.insert(
            "flip_last_dim_f32_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                FLIP_LAST_DIM_F32_KERNEL,
                "flip_last_dim_f32_kernel",
            )?,
        );
        kernels.insert(
            "flip_last_dim_bf16_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                FLIP_LAST_DIM_BF16_KERNEL,
                "flip_last_dim_bf16_kernel",
            )?,
        );
        kernels.insert(
            "sqrt_kernel".to_string(),
            compile_and_load_kernel(&device, SQRT_KERNEL, "sqrt_kernel")?,
        );
        kernels.insert(
            "narrow_kernel".to_string(),
            compile_and_load_kernel(&device, NARROW_KERNEL, "narrow_kernel")?,
        );
        kernels.insert(
            "permute_nhwc_to_nchw_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                PERMUTE_NHWC_TO_NCHW_KERNEL,
                "permute_nhwc_to_nchw_kernel",
            )?,
        );
        kernels.insert(
            "permute_nchw_to_nhwc_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                PERMUTE_NCHW_TO_NHWC_KERNEL,
                "permute_nchw_to_nhwc_kernel",
            )?,
        );
        kernels.insert(
            "index_select_kernel".to_string(),
            compile_and_load_kernel(&device, INDEX_SELECT_KERNEL, "index_select_kernel")?,
        );
        kernels.insert(
            "slice_kernel".to_string(),
            compile_and_load_kernel(&device, SLICE_KERNEL, "slice_kernel")?,
        );
        #[cfg(feature = "bf16_u16")]
        kernels.insert(
            "slice_kernel_bf16".to_string(),
            compile_and_load_kernel(&device, SLICE_KERNEL_BF16, "slice_kernel_bf16")?,
        );
        // Newly added elementwise math and reductions
        kernels.insert(
            "exp_kernel".to_string(),
            compile_and_load_kernel(&device, EXP_KERNEL, "exp_kernel")?,
        );
        kernels.insert(
            "log_kernel".to_string(),
            compile_and_load_kernel(&device, LOG_KERNEL, "log_kernel")?,
        );
        kernels.insert(
            "sum_dim_keepdim_kernel".to_string(),
            compile_and_load_kernel(&device, SUM_DIM_KEEPDIM_KERNEL, "sum_dim_keepdim_kernel")?,
        );
        kernels.insert(
            "max_elemwise_kernel".to_string(),
            compile_and_load_kernel(&device, MAX_ELEMWISE_KERNEL, "max_elemwise_kernel")?,
        );
        kernels.insert(
            "resize_bilinear_nhwc_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                RESIZE_BILINEAR_NHWC_KERNEL,
                "resize_bilinear_nhwc_kernel",
            )?,
        );
        kernels.insert(
            "center_crop_nhwc_kernel".to_string(),
            compile_and_load_kernel(&device, CENTER_CROP_NHWC_KERNEL, "center_crop_nhwc_kernel")?,
        );
        kernels.insert(
            "normalize_nhwc_kernel".to_string(),
            compile_and_load_kernel(&device, NORMALIZE_NHWC_KERNEL, "normalize_nhwc_kernel")?,
        );
        kernels.insert(
            "permute_generic_f32_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                PERMUTE_GENERIC_F32_KERNEL,
                "permute_generic_f32_kernel",
            )?,
        );
        #[cfg(feature = "bf16_u16")]
        kernels.insert(
            "permute_generic_bf16_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                PERMUTE_GENERIC_BF16_KERNEL,
                "permute_generic_bf16_kernel",
            )?,
        );
        kernels.insert(
            "materialize_strided_f32_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                MATERIALIZE_STRIDED_F32_KERNEL,
                "materialize_strided_f32_kernel",
            )?,
        );
        #[cfg(feature = "bf16_u16")]
        kernels.insert(
            "materialize_strided_bf16_kernel".to_string(),
            compile_and_load_kernel(
                &device,
                MATERIALIZE_STRIDED_BF16_KERNEL,
                "materialize_strided_bf16_kernel",
            )?,
        );
        kernels.insert(
            "permute_w_khwkicoc_to_ocickhkw".to_string(),
            compile_and_load_kernel(
                &device,
                PERMUTE_W_KH_KW_IC_OC_TO_OC_IC_KH_KW,
                "permute_w_khwkicoc_to_ocickhkw",
            )?,
        );
        kernels.insert(
            "permute_w_ocickhkw_to_khwkicoc".to_string(),
            compile_and_load_kernel(
                &device,
                PERMUTE_W_OC_IC_KH_KW_TO_KH_KW_IC_OC,
                "permute_w_ocickhkw_to_khwkicoc",
            )?,
        );

        Ok(Self { device, kernels })
    }

    /// Elementwise add with stride-based broadcasting (delegates to GPU helper)
    pub fn add_bc(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        crate::cuda_kernels_gpu::CudaKernels::add_bc(a, b)
    }

    /// Elementwise mul with stride-based broadcasting (delegates to GPU helper)
    pub fn mul_bc(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        crate::cuda_kernels_gpu::CudaKernels::mul_bc(a, b)
    }

    /// Element-wise addition kernel (rank-safe broadcasting)
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Normalize via broadcast if shapes differ (NumPy semantics)
        let (a_owned, b_owned);
        let (a_t, b_t) = if a.shape != b.shape {
            let target = a.shape.broadcast_shape_binary_op(&b.shape)?;
            let a_bc = if a.shape != target {
                a_owned = crate::cuda_kernels::CudaKernels::broadcast(a, &target)?;
                &a_owned
            } else {
                a
            };
            let b_bc = if b.shape != target {
                b_owned = crate::cuda_kernels::CudaKernels::broadcast(b, &target)?;
                &b_owned
            } else {
                b
            };
            (a_bc, b_bc)
        } else {
            (a, b)
        };

        let mut output =
            Tensor::empty_dtype(a_t.shape.clone(), crate::DType::F32, a_t.device.clone())?;
        let n = a_t.shape.elem_count();

        let kernel = self
            .kernels
            .get("add_kernel")
            .ok_or_else(|| Error::Cuda("add_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            a_t.storage.try_as_slice_f32()?,
            b_t.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    /// Element-wise multiplication kernel (rank-safe broadcasting)
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Normalize via broadcast if shapes differ (NumPy semantics)
        let (a_owned, b_owned);
        let (a_t, b_t) = if a.shape != b.shape {
            let target = a.shape.broadcast_shape_binary_op(&b.shape)?;
            let a_bc = if a.shape != target {
                a_owned = crate::cuda_kernels::CudaKernels::broadcast(a, &target)?;
                &a_owned
            } else {
                a
            };
            let b_bc = if b.shape != target {
                b_owned = crate::cuda_kernels::CudaKernels::broadcast(b, &target)?;
                &b_owned
            } else {
                b
            };
            (a_bc, b_bc)
        } else {
            (a, b)
        };

        let mut output =
            Tensor::empty_dtype(a_t.shape.clone(), crate::DType::F32, a_t.device.clone())?;
        let n = a_t.shape.elem_count();

        let kernel = self
            .kernels
            .get("mul_kernel")
            .ok_or_else(|| Error::Cuda("mul_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            a_t.storage.try_as_slice_f32()?,
            b_t.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    /// Scalar multiplication kernel
    pub fn mul_scalar(&self, tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let mut output = Tensor::empty_dtype(
            tensor.shape.clone(),
            crate::DType::F32,
            tensor.device.clone(),
        )?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("mul_scalar_kernel")
            .ok_or_else(|| Error::Cuda("mul_scalar_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            scalar,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    /// Add scalar kernel
    pub fn add_scalar(&self, tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let mut output = Tensor::empty_dtype(
            tensor.shape.clone(),
            crate::DType::F32,
            tensor.device.clone(),
        )?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("add_scalar_kernel")
            .ok_or_else(|| Error::Cuda("add_scalar_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            scalar,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    fn compare_f32(
        &self,
        a: &Tensor,
        b: &Tensor,
        kernel_name: &str,
        kernel_code: &str,
    ) -> Result<Tensor> {
        if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
            return Err(Error::InvalidOperation(
                "compare_f32: expected F32 tensors".into(),
            ));
        }
        if a.shape != b.shape {
            return Err(Error::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }

        Self::ensure_kernel(&self.device, kernel_name, kernel_code)?;
        let func = self
            .device
            .get_func(kernel_name, kernel_name)
            .ok_or_else(|| Error::Cuda(format!("Failed to get {}", kernel_name)))?;

        let mut output = Tensor::empty_dtype(a.shape.clone(), DType::F32, a.device.clone())?;
        let elems = a.shape.elem_count() as i32;
        let cfg = LaunchConfig::for_num_elems(elems as u32);

        launch_kernel!(
            func,
            cfg,
            a.storage.try_as_slice_f32()?,
            b.storage.try_as_slice_f32()?,
            output.storage_mut().try_as_mut_slice_f32()?,
            elems
        )?;

        Ok(output)
    }

    pub fn compare_gt(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.compare_f32(a, b, CMP_GT_F32_KERNEL, CMP_GT_F32_CODE)
    }

    pub fn compare_ge(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.compare_f32(a, b, CMP_GE_F32_KERNEL, CMP_GE_F32_CODE)
    }

    pub fn compare_lt(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.compare_f32(a, b, CMP_LT_F32_KERNEL, CMP_LT_F32_CODE)
    }

    pub fn compare_le(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.compare_f32(a, b, CMP_LE_F32_KERNEL, CMP_LE_F32_CODE)
    }

    pub fn compare_eq(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.compare_f32(a, b, CMP_EQ_F32_KERNEL, CMP_EQ_F32_CODE)
    }

    pub fn compare_ne(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.compare_f32(a, b, CMP_NE_F32_KERNEL, CMP_NE_F32_CODE)
    }

    /// ReLU activation kernel
    pub fn relu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("relu_kernel")
            .ok_or_else(|| Error::Cuda("relu_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    pub fn floor(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("floor_kernel")
            .ok_or_else(|| Error::Cuda("floor_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    pub fn ceil(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("ceil_kernel")
            .ok_or_else(|| Error::Cuda("ceil_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    pub fn round(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("round_kernel")
            .ok_or_else(|| Error::Cuda("round_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    pub fn flip_last_dim(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dims.is_empty() {
            return Err(Error::InvalidOperation(
                "flip_last_dim requires tensor with rank >= 1".into(),
            ));
        }

        let last_dim = dims[dims.len() - 1];
        let total = tensor.shape.elem_count();
        if last_dim == 0 || total == 0 {
            return Err(Error::InvalidInput("flip_last_dim on empty tensor".into()));
        }

        let rows = total / last_dim;
        let block_size = 256usize;
        let grid_size = (total + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        match tensor.dtype() {
            DType::BF16 => {
                let kernel_name = "flip_last_dim_bf16_kernel";
                let f = self
                    .device
                    .get_func(kernel_name, kernel_name)
                    .ok_or_else(|| Error::Cuda("flip_last_dim_bf16_kernel not found".into()))?;

                let mut out =
                    Tensor::empty_dtype(tensor.shape.clone(), DType::BF16, tensor.device.clone())?;

                launch_kernel!(
                    f,
                    cfg,
                    tensor.storage.try_as_slice_u16()?,
                    out.storage.try_as_slice_u16()?,
                    rows as i32,
                    last_dim as i32
                )?;

                Ok(out)
            }
            DType::F32 => {
                let kernel_name = "flip_last_dim_f32_kernel";
                let f = self
                    .device
                    .get_func(kernel_name, kernel_name)
                    .ok_or_else(|| Error::Cuda("flip_last_dim_f32_kernel not found".into()))?;

                let mut out =
                    Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;

                launch_kernel!(
                    f,
                    cfg,
                    tensor.storage.try_as_slice_f32()?,
                    out.storage.try_as_slice_f32()?,
                    rows as i32,
                    last_dim as i32
                )?;

                Ok(out)
            }
            other => Err(Error::Unsupported(format!(
                "flip_last_dim not implemented for dtype {:?}",
                other
            ))),
        }
    }

    /// Sigmoid activation kernel
    pub fn sigmoid(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("sigmoid_kernel")
            .ok_or_else(|| Error::Cuda("sigmoid_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    /// GELU activation kernel
    pub fn gelu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("gelu_kernel")
            .ok_or_else(|| Error::Cuda("gelu_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    /// SiLU (Swish) activation kernel
    pub fn silu(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("silu_kernel")
            .ok_or_else(|| Error::Cuda("silu_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    /// Tanh activation kernel
    pub fn tanh(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("tanh_kernel")
            .ok_or_else(|| Error::Cuda("tanh_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    /// Sum kernel - reduction operation
    pub fn sum(&self, tensor: &Tensor) -> Result<Tensor> {
        let n = tensor.shape.elem_count();

        // Allocate output directly as zeros
        let output_data = crate::tensor::alloc_zeros_from_pool(&self.device, 1)?;

        let kernel = self
            .kernels
            .get("sum_kernel")
            .ok_or_else(|| Error::Cuda("sum_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size).min(1024);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            &output_data,
            n as u32
        )?;

        Ok(Tensor {
            storage: TensorStorage::F32 {
                data: output_data.into(),
                numel: 1,
            },
            shape: Shape::from_dims(&[]),
            device: tensor.device.clone(),
            id: crate::tensor::TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        })
    }

    /// Sum reduction along specific dimensions
    pub fn sum_dims(&self, tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
        // Calculate output shape
        let mut output_shape = tensor.shape().dims().to_vec();
        for &dim in dims {
            output_shape[dim] = 1;
        }

        // For now, implement a simple version that handles the common case
        // of summing along batch dimensions for bias gradients
        if dims.is_empty() {
            return self.sum(tensor);
        }

        // Calculate strides and sizes
        let input_shape = tensor.shape().dims();
        let ndims = input_shape.len();
        let mut strides = vec![1; ndims];
        for i in (0..ndims - 1).rev() {
            strides[i] = strides[i + 1] * input_shape[i + 1];
        }

        // Total output elements
        let output_elems: usize = output_shape.iter().product();
        let mut output = Tensor::empty_dtype(
            Shape::from_dims(&output_shape),
            DType::F32,
            tensor.device.clone(),
        )?;

        // Implement GPU kernel for multi-dimensional reduction
        if true {
            // FLAME is GPU-only
            return crate::cuda_kernels_gpu::mean_reduce_dims(tensor, dims);
        }

        // For CPU tensors, use existing implementation
        let input_data = tensor.to_vec()?;
        let mut output_data = vec![0.0f32; output_elems];

        // Iterate over output positions
        for (out_pos_idx, out_val) in output_data.iter_mut().enumerate().take(output_elems) {
            let mut out_pos = vec![0; ndims];
            let mut idx = out_pos_idx;
            for (dim, pos) in output_shape.iter().rev().zip(out_pos.iter_mut().rev()) {
                *pos = idx % dim;
                idx /= dim;
            }

            // Sum over all positions that map to this output position
            let mut sum = 0.0f32;
            let mut count = 0;

            // Create iterator over dimensions to sum
            let mut pos = vec![0; ndims];
            loop {
                // Copy fixed dimensions from output position
                for (i, pos_val) in pos.iter_mut().enumerate().take(ndims) {
                    if !dims.contains(&i) {
                        *pos_val = out_pos[i];
                    }
                }

                // Calculate linear index
                let lin_idx: usize = pos.iter().zip(&strides).map(|(p, s)| p * s).sum();
                sum += input_data[lin_idx];
                count += 1;

                // Increment position over dimensions we're summing
                let mut carry = true;
                for &dim in dims {
                    if carry {
                        pos[dim] += 1;
                        if pos[dim] < input_shape[dim] {
                            carry = false;
                        } else {
                            pos[dim] = 0;
                        }
                    }
                }

                if carry {
                    break;
                }
            }
            *out_val = sum;
        }

        // Copy result back to GPU
        output.set_data(&output_data)?;
        Ok(output)
    }

    /// Transpose kernel for 2D matrices  
    pub fn transpose(&self, tensor: &Tensor) -> Result<Tensor> {
        let tensor_f32 = ensure_f32_tensor(tensor)?;
        let dims = tensor.shape.dims();
        if dims.len() != 2 {
            return Err(Error::InvalidOperation(format!(
                "Transpose requires 2D tensor, got {:?}",
                dims
            )));
        }

        let rows = dims[0];
        let cols = dims[1];
        let mut output = Tensor::empty_dtype(
            Shape::from_dims(&[cols, rows]),
            crate::DType::F32,
            tensor.device.clone(),
        )?;

        let kernel = self
            .kernels
            .get("transpose_kernel")
            .ok_or_else(|| Error::Cuda("transpose_kernel not found".into()))?
            .clone();

        let block_size = 16;
        let grid_x = cols.div_ceil(block_size);
        let grid_y = rows.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor_f32.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            rows as u32,
            cols as u32
        )?;

        Ok(output)
    }

    /// Weight update kernel (for SGD)
    pub fn update_weights(&self, weights: &Tensor, gradients: &Tensor, lr: f32) -> Result<Tensor> {
        if weights.shape != gradients.shape {
            return Err(Error::ShapeMismatch {
                expected: weights.shape.clone(),
                got: gradients.shape.clone(),
            });
        }

        let output = weights.clone_result()?;
        let n = weights.shape.elem_count();

        let kernel = self
            .kernels
            .get("update_weights_kernel")
            .ok_or_else(|| Error::Cuda("update_weights_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            output.storage.try_as_slice_f32()?,
            gradients.storage.try_as_slice_f32()?,
            lr,
            n as u32
        )?;

        Ok(output)
    }

    // Additional activation functions
    pub fn leaky_relu(&self, tensor: &Tensor, negative_slope: f32) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("leaky_relu_kernel")
            .ok_or_else(|| Error::Cuda("leaky_relu_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            negative_slope,
            n as u32
        )?;

        Ok(output)
    }

    pub fn elu(&self, tensor: &Tensor, alpha: f32) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("elu_kernel")
            .ok_or_else(|| Error::Cuda("elu_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            alpha,
            n as u32
        )?;

        Ok(output)
    }

    pub fn prelu(&self, _tensor: &Tensor, _weight: &Tensor) -> Result<Tensor> {
        // PReLU requires channel-wise parameters
        Err(Error::InvalidOperation(
            "PReLU GPU kernel not yet implemented".into(),
        ))
    }

    pub fn pow(&self, tensor: &Tensor, exponent: f32) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("pow_kernel")
            .ok_or_else(|| Error::Cuda("pow_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            exponent,
            n as u32
        )?;

        Ok(output)
    }

    pub fn sin(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("sin_kernel")
            .ok_or_else(|| Error::Cuda("sin_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    pub fn cos(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("cos_kernel")
            .ok_or_else(|| Error::Cuda("cos_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    pub fn sqrt(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output =
            Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();

        let kernel = self
            .kernels
            .get("sqrt_kernel")
            .ok_or_else(|| Error::Cuda("sqrt_kernel not found".into()))?
            .clone();

        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            config,
            tensor.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;

        Ok(output)
    }

    // Transposed convolution operations
    pub fn conv_transpose2d_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        groups: usize,
        dilation: (usize, usize),
    ) -> Result<Tensor> {
        fn zero_insert_width(x: &Tensor, stride_w: usize) -> Result<Tensor> {
            if stride_w <= 1 {
                return Ok(x.clone());
            }
            let dims = x.shape().dims();
            if dims.len() != 4 {
                return Err(Error::InvalidOperation(format!(
                    "conv_transpose2d zero_insert_width expects 4D input, got {:?}",
                    dims
                )));
            }
            let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
            let x5 = x.reshape(&[b, c, h, w, 1])?;
            let zeros = Tensor::zeros_dtype(
                Shape::from_dims(&[b, c, h, w, stride_w - 1]),
                x.dtype(),
                x.device().clone(),
            )?;
            let cat = Tensor::cat(&[&x5, &zeros], 4)?;
            let flat = cat.reshape(&[b, c, h, w * stride_w])?;
            flat.narrow(3, 0, (w - 1) * stride_w + 1)
        }

        fn zero_insert_height(x: &Tensor, stride_h: usize) -> Result<Tensor> {
            if stride_h <= 1 {
                return Ok(x.clone());
            }
            let dims = x.shape().dims();
            if dims.len() != 4 {
                return Err(Error::InvalidOperation(format!(
                    "conv_transpose2d zero_insert_height expects 4D input, got {:?}",
                    dims
                )));
            }
            let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
            let x5 = x.reshape(&[b, c, h, 1, w])?;
            let zeros = Tensor::zeros_dtype(
                Shape::from_dims(&[b, c, h, stride_h - 1, w]),
                x.dtype(),
                x.device().clone(),
            )?;
            let cat = Tensor::cat(&[&x5, &zeros], 3)?;
            let flat = cat.reshape(&[b, c, h * stride_h, w])?;
            flat.narrow(2, 0, (h - 1) * stride_h + 1)
        }

        fn zero_insert_hw(x: &Tensor, stride: (usize, usize)) -> Result<Tensor> {
            let x = zero_insert_height(x, stride.0)?;
            zero_insert_width(&x, stride.1)
        }

        fn pad_h(x: &Tensor, top: usize, bottom: usize) -> Result<Tensor> {
            if top == 0 && bottom == 0 {
                return Ok(x.clone());
            }
            let dims = x.shape().dims();
            if dims.len() != 4 {
                return Err(Error::InvalidOperation(format!(
                    "conv_transpose2d pad_h expects 4D input, got {:?}",
                    dims
                )));
            }
            let (b, c, _h, w) = (dims[0], dims[1], dims[2], dims[3]);
            let mut parts: Vec<Tensor> = Vec::new();
            if top > 0 {
                parts.push(Tensor::zeros_dtype(
                    Shape::from_dims(&[b, c, top, w]),
                    x.dtype(),
                    x.device().clone(),
                )?);
            }
            parts.push(x.clone());
            if bottom > 0 {
                parts.push(Tensor::zeros_dtype(
                    Shape::from_dims(&[b, c, bottom, w]),
                    x.dtype(),
                    x.device().clone(),
                )?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 2)
        }

        fn pad_w(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
            if left == 0 && right == 0 {
                return Ok(x.clone());
            }
            let dims = x.shape().dims();
            if dims.len() != 4 {
                return Err(Error::InvalidOperation(format!(
                    "conv_transpose2d pad_w expects 4D input, got {:?}",
                    dims
                )));
            }
            let (b, c, h, _w) = (dims[0], dims[1], dims[2], dims[3]);
            let mut parts: Vec<Tensor> = Vec::new();
            if left > 0 {
                parts.push(Tensor::zeros_dtype(
                    Shape::from_dims(&[b, c, h, left]),
                    x.dtype(),
                    x.device().clone(),
                )?);
            }
            parts.push(x.clone());
            if right > 0 {
                parts.push(Tensor::zeros_dtype(
                    Shape::from_dims(&[b, c, h, right]),
                    x.dtype(),
                    x.device().clone(),
                )?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 3)
        }

        fn pad_hw(x: &Tensor, top: usize, bottom: usize, left: usize, right: usize) -> Result<Tensor> {
            let x = pad_h(x, top, bottom)?;
            pad_w(&x, left, right)
        }

        fn flip_hw(weight: &Tensor) -> Result<Tensor> {
            let w = weight.flip(&[3])?;
            let w = w.permute(&[0, 1, 3, 2])?;
            let w = w.flip(&[3])?;
            w.permute(&[0, 1, 3, 2])
        }

        let in_dims = input.shape().dims();
        let w_dims = weight.shape().dims();
        if in_dims.len() != 4 {
            return Err(Error::InvalidOperation(format!(
                "ConvTranspose2d requires 4D input, got {:?}",
                in_dims
            )));
        }
        if w_dims.len() != 4 {
            return Err(Error::InvalidOperation(format!(
                "ConvTranspose2d requires 4D weight, got {:?}",
                w_dims
            )));
        }
        if input.dtype() != weight.dtype() {
            return Err(Error::InvalidOperation(format!(
                "ConvTranspose2d dtype mismatch: input={:?} weight={:?}",
                input.dtype(),
                weight.dtype()
            )));
        }
        if let Some(b) = bias {
            if b.dtype() != input.dtype() {
                return Err(Error::InvalidOperation(format!(
                    "ConvTranspose2d bias dtype mismatch: input={:?} bias={:?}",
                    input.dtype(),
                    b.dtype()
                )));
            }
        }
        if groups != 1 {
            return Err(Error::Unsupported(
                "ConvTranspose2d forward currently supports groups=1 only".into(),
            ));
        }
        if dilation != (1, 1) {
            return Err(Error::Unsupported(format!(
                "ConvTranspose2d forward currently supports dilation=(1,1) only, got {:?}",
                dilation
            )));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(Error::InvalidOperation(
                "ConvTranspose2d stride must be >= 1".into(),
            ));
        }
        if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
            return Err(Error::InvalidOperation(format!(
                "ConvTranspose2d output_padding {:?} must be smaller than stride {:?}",
                output_padding, stride
            )));
        }

        let in_channels = in_dims[1];
        let in_ch_w = w_dims[0];
        let out_channels = w_dims[1] * groups;
        let kh = w_dims[2];
        let kw = w_dims[3];
        if in_channels != in_ch_w {
            return Err(Error::InvalidOperation(format!(
                "ConvTranspose2d input channels {} does not match weight {}",
                in_channels, in_ch_w
            )));
        }
        if bias.is_some_and(|b| b.shape().dims() != [out_channels]) {
            return Err(Error::InvalidOperation(format!(
                "ConvTranspose2d bias must have shape [{}]",
                out_channels
            )));
        }
        if kh == 0 || kw == 0 {
            return Err(Error::InvalidOperation(
                "ConvTranspose2d kernel dimensions must be non-zero".into(),
            ));
        }
        if kh - 1 < padding.0 || kw - 1 < padding.1 {
            return Err(Error::InvalidOperation(format!(
                "ConvTranspose2d padding {:?} exceeds kernel-1 {:?}",
                padding,
                (kh - 1, kw - 1)
            )));
        }

        let pad_top = (kh - 1) - padding.0;
        let pad_bottom = pad_top + output_padding.0;
        let pad_left = (kw - 1) - padding.1;
        let pad_right = pad_left + output_padding.1;

        let x_zi = zero_insert_hw(input, stride)?;
        let x_padded = pad_hw(&x_zi, pad_top, pad_bottom, pad_left, pad_right)?;
        let weight_reg = flip_hw(weight)?.permute(&[1, 0, 2, 3])?;

        crate::ops::conv2d::conv2d_forward(&x_padded, &weight_reg, bias, (1, 1), (0, 0), 1)
    }

    pub fn conv_transpose2d_backward(
        &self,
        _grad_output: &Tensor,
        _input: &Tensor,
        _weight: &Tensor,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _output_padding: (usize, usize),
        _groups: usize,
        _dilation: (usize, usize),
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        Err(Error::InvalidOperation(
            "ConvTranspose2d backward GPU kernel not yet implemented".into(),
        ))
    }

    // Pooling operations - using CPU implementations for now
    pub fn maxpool2d_forward(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        let input_f32 = ensure_f32_tensor(input)?;
        crate::cuda_kernels_gpu::CudaKernels::maxpool2d_forward(
            &input_f32,
            kernel_size,
            stride,
            padding,
        )
    }

    pub fn maxpool2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        let grad_output_f32 = ensure_f32_tensor(grad_output)?;
        let input_f32 = ensure_f32_tensor(input)?;
        // First do forward pass to get indices
        let (_, indices) = crate::cuda_kernels_gpu::CudaKernels::maxpool2d_forward_with_indices(
            &input_f32,
            kernel_size,
            stride,
            padding,
        )?;
        crate::cuda_kernels_gpu::CudaKernels::maxpool2d_backward_with_indices(
            &grad_output_f32,
            &input_f32,
            &indices,
        )
    }

    pub fn avgpool2d_forward(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Result<Tensor> {
        let input_f32 = ensure_f32_tensor(input)?;
        crate::cuda_kernels_gpu::CudaKernels::avgpool2d_forward(
            &input_f32,
            kernel_size,
            stride,
            padding,
            count_include_pad,
        )
    }

    pub fn avgpool2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Result<Tensor> {
        let grad_output_f32 = ensure_f32_tensor(grad_output)?;
        let input_f32 = ensure_f32_tensor(input)?;
        crate::cuda_kernels_gpu::CudaKernels::avgpool2d_backward(
            &grad_output_f32,
            &input_f32,
            kernel_size,
            stride,
            padding,
            count_include_pad,
        )
    }

    pub fn adaptive_maxpool2d_forward(
        input: &Tensor,
        output_size: (usize, usize),
    ) -> Result<Tensor> {
        let input_f32 = ensure_f32_tensor(input)?;
        crate::cuda_kernels_gpu::CudaKernels::adaptive_maxpool2d_forward(&input_f32, output_size)
    }

    pub fn adaptive_avgpool2d_forward(
        input: &Tensor,
        output_size: (usize, usize),
    ) -> Result<Tensor> {
        let input_f32 = ensure_f32_tensor(input)?;
        crate::cuda_kernels_gpu::CudaKernels::adaptive_avgpool2d_forward(&input_f32, output_size)
    }

    // Upsampling operations
    pub fn upsample2d_nearest(
        &self,
        input: &Tensor,
        output_size: (usize, usize),
    ) -> Result<Tensor> {
        let (h_out, w_out) = output_size;
        let shape = input.shape().dims();
        let (batch, channels, _h_in, _w_in) = (shape[0], shape[1], shape[2], shape[3]);

        let mut output = Tensor::empty_dtype(
            Shape::from_dims(&[batch, channels, h_out, w_out]),
            input.dtype(),
            input.device.clone(),
        )?;

        let stream = self.device.cuda_stream_raw_ptr();

        unsafe {
            match input.dtype() {
                DType::F32 => {
                    let src = input.storage.try_as_slice_f32()?;
                    let dst = output.storage_mut().try_as_mut_slice_f32()?;
                    let src_ptr = *src.device_ptr() as *const core::ffi::c_void;
                    let dst_ptr = *dst.device_ptr() as *mut core::ffi::c_void;
                    let status = ffi::fc_upsample2d_nearest_f32(
                        src_ptr,
                        dst_ptr,
                        batch as i32,
                        channels as i32,
                        _h_in as i32,
                        _w_in as i32,
                        h_out as i32,
                        w_out as i32,
                        stream,
                    );
                    if status != 0 {
                        return Err(Error::Cuda(format!(
                            "upsample2d_nearest_f32 failed: {}",
                            status
                        )));
                    }
                }
                DType::BF16 => {
                    #[cfg(feature = "bf16_u16")]
                    {
                        let src_ptr = input.as_device_ptr_bf16("upsample2d_nearest:src")?
                            as *const core::ffi::c_void;
                        let dst_ptr = output.as_mut_device_ptr_bf16("upsample2d_nearest:dst")?
                            as *mut core::ffi::c_void;
                        // println!("DEBUG: upsample2d_nearest input shape={:?} strides={:?} output_size={:?}", shape, input.shape().strides(), output_size);
                        let status = ffi::fc_upsample2d_nearest_bf16(
                            src_ptr,
                            dst_ptr,
                            batch as i32,
                            channels as i32,
                            _h_in as i32,
                            _w_in as i32,
                            h_out as i32,
                            w_out as i32,
                            stream,
                        );
                        if status != 0 {
                            return Err(Error::Cuda(format!(
                                "upsample2d_nearest_bf16 failed: {}",
                                status
                            )));
                        }
                    }
                    #[cfg(not(feature = "bf16_u16"))]
                    {
                        return Err(Error::Unsupported(
                            "BF16 upsample requires bf16_u16 feature".into(),
                        ));
                    }
                }
                _ => {
                    return Err(Error::Unsupported(format!(
                        "upsample2d_nearest unsupported dtype {:?}",
                        input.dtype()
                    )))
                }
            }
        }
        Ok(output)
    }

    pub fn upsample2d_bilinear(
        &self,
        input: &Tensor,
        output_size: (usize, usize),
        align_corners: bool,
    ) -> Result<Tensor> {
        let (h_out, w_out) = output_size;
        let shape = input.shape().dims();
        let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);

        let mut output = Tensor::empty_dtype(
            Shape::from_dims(&[batch, channels, h_out, w_out]),
            input.dtype(),
            input.device.clone(),
        )?;

        let stream = self.device.cuda_stream_raw_ptr();

        unsafe {
            match input.dtype() {
                DType::F32 => {
                    let src = input.storage.try_as_slice_f32()?;
                    let dst = output.storage_mut().try_as_mut_slice_f32()?;
                    let src_ptr = *src.device_ptr() as *const core::ffi::c_void;
                    let dst_ptr = *dst.device_ptr() as *mut core::ffi::c_void;
                    let status = ffi::fc_upsample2d_bilinear_f32(
                        src_ptr,
                        dst_ptr,
                        batch as i32,
                        channels as i32,
                        h_in as i32,
                        w_in as i32,
                        h_out as i32,
                        w_out as i32,
                        align_corners as i32,
                        stream,
                    );
                    if status != 0 {
                        return Err(Error::Cuda(format!(
                            "upsample2d_bilinear_f32 failed: {}",
                            status
                        )));
                    }
                }
                DType::BF16 => {
                    #[cfg(feature = "bf16_u16")]
                    {
                        let src_ptr = input.as_device_ptr_bf16("upsample2d_bilinear:src")?
                            as *const core::ffi::c_void;
                        let dst_ptr = output.as_mut_device_ptr_bf16("upsample2d_bilinear:dst")?
                            as *mut core::ffi::c_void;
                        let status = ffi::fc_upsample2d_bilinear_bf16(
                            src_ptr,
                            dst_ptr,
                            batch as i32,
                            channels as i32,
                            h_in as i32,
                            w_in as i32,
                            h_out as i32,
                            w_out as i32,
                            align_corners as i32,
                            stream,
                        );
                        if status != 0 {
                            return Err(Error::Cuda(format!(
                                "upsample2d_bilinear_bf16 failed: {}",
                                status
                            )));
                        }
                    }
                    #[cfg(not(feature = "bf16_u16"))]
                    {
                        return Err(Error::Unsupported(
                            "BF16 bilinear upsample requires bf16_u16 feature".into(),
                        ));
                    }
                }
                _ => {
                    return Err(Error::Unsupported(format!(
                        "upsample2d_bilinear unsupported dtype {:?}",
                        input.dtype()
                    )))
                }
            }
        }
        Ok(output)
    }

    pub fn upsample2d_nearest_backward(
        &self,
        grad_output: &Tensor,
        input_size: (usize, usize),
        output_size: (usize, usize),
    ) -> Result<Tensor> {
        let _ = (grad_output, input_size, output_size);
        Err(Error::InvalidOperation(
            "Upsample2d nearest backward GPU kernel not yet implemented".into(),
        ))
    }

    pub fn upsample2d_bilinear_backward(
        &self,
        grad_output: &Tensor,
        input_size: (usize, usize),
        output_size: (usize, usize),
        align_corners: bool,
    ) -> Result<Tensor> {
        let _ = (grad_output, input_size, output_size, align_corners);
        Err(Error::InvalidOperation(
            "Upsample2d bilinear backward GPU kernel not yet implemented".into(),
        ))
    }

    /// Broadcast tensor to a new shape
    pub fn broadcast(input: &Tensor, target_shape: &Shape) -> Result<Tensor> {
        let target_i64: Vec<i64> = target_shape.dims().iter().map(|&d| d as i64).collect();
        crate::ops::broadcast::broadcast_to_impl(input, &target_i64)
    }

    // Ensure a kernel is compiled and loaded
    pub fn ensure_kernel(
        device: &Arc<CudaDevice>,
        kernel_name: &str,
        kernel_code: &str,
    ) -> Result<()> {
        // Check if kernel is already loaded
        if device.get_func(kernel_name, kernel_name).is_some() {
            return Ok(());
        }

        // Compile the kernel
        let mut opts = CompileOptions::default();
        if let Ok(extra) = std::env::var("CUDARC_NVRTC_EXTRA_INCLUDE_PATHS") {
            for path in extra.split(':').filter(|p| !p.is_empty()) {
                opts.include_paths.push(path.to_string());
            }
        }
        if opts.include_paths.is_empty() {
            if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
                opts.include_paths.push(format!("{cuda_home}/include"));
            } else if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                opts.include_paths.push(format!("{cuda_path}/include"));
            } else {
                opts.include_paths.push("/usr/local/cuda/include".into());
            }
        }
        if !opts.include_paths.iter().any(|p| p == "/usr/include") {
            opts.include_paths.push("/usr/include".into());
        }

        let ptx = compile_ptx_with_opts(kernel_code, opts)
            .map_err(|e| Error::Cuda(format!("Failed to compile {}: {:?}", kernel_name, e)))?;

        // Use Box::leak to get 'static lifetime for kernel names
        let kernel_name_static = Box::leak(kernel_name.to_string().into_boxed_str());

        // Load the PTX module
        device
            .load_ptx(ptx, kernel_name_static, &[kernel_name_static])
            .map_err(|e| Error::Cuda(format!("Failed to load kernel {}: {:?}", kernel_name, e)))?;

        Ok(())
    }

    /// Element-wise division
    pub fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(Error::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        let mut output = Tensor::empty_dtype(a.shape.clone(), DType::F32, a.device.clone())?;
        let n = a.shape.elem_count();
        let kernel = self
            .kernels
            .get("div_kernel")
            .ok_or_else(|| Error::Cuda("div_kernel not found".into()))?
            .clone();
        let block_size = 256usize;
        let grid_size = n.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(
            kernel,
            cfg,
            a.storage.try_as_slice_f32()?,
            b.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as u32
        )?;
        Ok(output)
    }

    /// Max reduction along dimension (GPU, with optional keepdim)
    pub fn max_dim(&self, tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dim >= dims.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        // Kernel computes keepdim=true; we reshape if keepdim=false
        let mut out_shape_keep = dims.to_vec();
        out_shape_keep[dim] = 1;
        let out_elems: usize = out_shape_keep.iter().product();

        let dims_f32: Vec<f32> = dims.iter().map(|&x| x as f32).collect();
        let mut dims_gpu =
            unsafe { self.device.alloc::<f32>(dims_f32.len()) }.map_err(|_| Error::CudaDriver)?;
        self.device
            .htod_copy_into(dims_f32, &mut dims_gpu)
            .map_err(|_| Error::CudaDriver)?;

        let dtype = tensor.dtype();
        let out = match dtype {
            DType::BF16 => {
                let kernel_code_bf16 = r#"
#include <cuda_bf16.h>

extern "C" __global__ void max_dim_keepdim_kernel_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const float* dims_f32,
    int ndim,
    int reduce_dim,
    int out_elems
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_elems) return;

    int dims[8];
    for (int i = 0; i < ndim && i < 8; ++i) dims[i] = (int)dims_f32[i];

    int strides[8];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    int rem = tid;
    int out_coords[8];
    for (int i = 0; i < ndim; ++i) {
        int size = (i == reduce_dim) ? 1 : dims[i];
        int stride = 1;
        for (int j = i + 1; j < ndim; ++j) {
            stride *= (j == reduce_dim) ? 1 : dims[j];
        }
        out_coords[i] = (size == 0) ? 0 : (rem / stride) % size;
    }

    int base_idx = 0;
    for (int i = 0; i < ndim; ++i) {
        int coord = (i == reduce_dim) ? 0 : out_coords[i];
        base_idx += coord * strides[i];
    }

    float maxv = -3.402823e38f;
    for (int d = 0; d < dims[reduce_dim]; ++d) {
        int idx = base_idx + d * strides[reduce_dim];
        float v = __bfloat162float(input[idx]);
        if (v > maxv) maxv = v;
    }
    output[tid] = __float2bfloat16_rn(maxv);
}
"#;

                Self::ensure_kernel(
                    &self.device,
                    "max_dim_keepdim_kernel_bf16",
                    kernel_code_bf16,
                )?;
                let f = self
                    .device
                    .get_func("max_dim_keepdim_kernel_bf16", "max_dim_keepdim_kernel_bf16")
                    .ok_or_else(|| {
                        Error::Cuda("Failed to get max_dim_keepdim_kernel_bf16".into())
                    })?;

                let mut out_bf16 = Tensor::empty_dtype(
                    Shape::from_dims(&out_shape_keep),
                    DType::BF16,
                    tensor.device.clone(),
                )?;

                let cfg = LaunchConfig::for_num_elems(out_elems as u32);
                launch_kernel!(
                    f,
                    cfg,
                    tensor.storage.try_as_slice_u16()?,
                    out_bf16.storage.try_as_slice_u16()?,
                    &dims_gpu,
                    dims.len() as i32,
                    dim as i32,
                    out_elems as i32
                )?;
                out_bf16
            }
            _ => {
                let kernel_code = r#"
extern "C" __global__ void max_dim_keepdim_kernel(
    const float* input,
    float* output,
    const float* dims_f32,
    int ndim,
    int reduce_dim,
    int out_elems
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_elems) return;

    int dims[8];
    for (int i = 0; i < ndim && i < 8; ++i) dims[i] = (int)dims_f32[i];

    int strides[8];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    int rem = tid;
    int out_coords[8];
    for (int i = 0; i < ndim; ++i) {
        int size = (i == reduce_dim) ? 1 : dims[i];
        int stride = 1;
        for (int j = i + 1; j < ndim; ++j) {
            stride *= (j == reduce_dim) ? 1 : dims[j];
        }
        out_coords[i] = (size == 0) ? 0 : (rem / stride) % size;
    }

    int base_idx = 0;
    for (int i = 0; i < ndim; ++i) {
        int coord = (i == reduce_dim) ? 0 : out_coords[i];
        base_idx += coord * strides[i];
    }

    float maxv = -3.402823e38f;
    for (int d = 0; d < dims[reduce_dim]; ++d) {
        int idx = base_idx + d * strides[reduce_dim];
        float v = input[idx];
        if (v > maxv) maxv = v;
    }
    output[tid] = maxv;
}
"#;

                Self::ensure_kernel(&self.device, "max_dim_keepdim_kernel", kernel_code)?;
                let f = self
                    .device
                    .get_func("max_dim_keepdim_kernel", "max_dim_keepdim_kernel")
                    .ok_or_else(|| Error::Cuda("Failed to get max_dim_keepdim_kernel".into()))?;

                let mut out_f32 = Tensor::empty_dtype(
                    Shape::from_dims(&out_shape_keep),
                    DType::F32,
                    tensor.device.clone(),
                )?;

                let cfg = LaunchConfig::for_num_elems(out_elems as u32);
                launch_kernel!(
                    f,
                    cfg,
                    tensor.storage.try_as_slice_f32()?,
                    out_f32.storage.try_as_slice_f32()?,
                    &dims_gpu,
                    dims.len() as i32,
                    dim as i32,
                    out_elems as i32
                )?;
                out_f32
            }
        };

        if keepdim {
            Ok(out)
        } else {
            let mut squeezed = Vec::with_capacity(dims.len() - 1);
            for (i, &d) in dims.iter().enumerate() {
                if i != dim {
                    squeezed.push(d);
                }
            }
            out.reshape(&squeezed)
        }
    }

    /// Sum along dimension with keepdim (GPU kernel)
    pub fn sum_dim_keepdim(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dim >= dims.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }
        // Build output shape keeping reduced dim = 1
        let mut out_shape = dims.to_vec();
        out_shape[dim] = 1;
        let out_elems: usize = out_shape.iter().product();

        // Upload dims to GPU as i32
        let dims_i32: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
        let mut dims_gpu =
            unsafe { self.device.alloc::<f32>(dims_i32.len()) }.map_err(|_| Error::CudaDriver)?;
        self.device
            .htod_copy_into(
                dims_i32.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                &mut dims_gpu,
            )
            .map_err(|_| Error::CudaDriver)?;

        let dtype = tensor.dtype();
        let out = match dtype {
            DType::BF16 => {
                Self::ensure_kernel(
                    &self.device,
                    "sum_dim_keepdim_kernel_bf16",
                    SUM_DIM_KEEPDIM_KERNEL_BF16,
                )?;
                let f = self
                    .device
                    .get_func("sum_dim_keepdim_kernel_bf16", "sum_dim_keepdim_kernel_bf16")
                    .ok_or_else(|| Error::Cuda("sum_dim_keepdim_kernel_bf16 not found".into()))?;

                let mut out_bf16 = Tensor::zeros_dtype(
                    Shape::from_dims(&out_shape),
                    DType::BF16,
                    tensor.device.clone(),
                )?;

                let cfg = LaunchConfig::for_num_elems(out_elems as u32);
                launch_kernel!(
                    f,
                    cfg,
                    tensor.storage.try_as_slice_u16()?,
                    out_bf16.storage.try_as_slice_u16()?,
                    &dims_gpu,
                    dims.len() as i32,
                    dim as i32,
                    out_elems as i32
                )?;
                out_bf16
            }
            _ => {
                let kernel = self
                    .kernels
                    .get("sum_dim_keepdim_kernel")
                    .ok_or_else(|| Error::Cuda("sum_dim_keepdim_kernel not found".into()))?
                    .clone();

                let mut out_f32 = Tensor::zeros_dtype(
                    Shape::from_dims(&out_shape),
                    DType::F32,
                    tensor.device.clone(),
                )?;

                let cfg = LaunchConfig::for_num_elems(out_elems as u32);
                launch_kernel!(
                    kernel,
                    cfg,
                    tensor.storage.try_as_slice_f32()?,
                    out_f32.storage.try_as_slice_f32()?,
                    &dims_gpu,
                    dims.len() as i32,
                    dim as i32,
                    out_elems as i32
                )?;
                out_f32
            }
        };

        Ok(out)
    }

    /// Sum all elements in a tensor
    pub fn sum_kernel(&self, tensor: &Tensor) -> Result<Tensor> {
        // Simple implementation - sum to scalar
        let data = tensor.to_vec()?;
        let sum: f32 = data.iter().sum();
        Tensor::from_vec(vec![sum], Shape::from_dims(&[1]), tensor.device.clone())
    }

    /// Elementwise max kernel
    pub fn max_elemwise(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape != b.shape {
            return Err(Error::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }
        let mut output = Tensor::empty_dtype(a.shape.clone(), DType::F32, a.device.clone())?;
        let n = a.shape.elem_count();
        let kernel = self
            .kernels
            .get("max_elemwise_kernel")
            .ok_or_else(|| Error::Cuda("max_elemwise_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(
            kernel,
            cfg,
            a.storage.try_as_slice_f32()?,
            b.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            n as i32
        )?;
        Ok(output)
    }

    /// Resize NHWC via bilinear interpolation
    pub fn resize_bilinear_nhwc(
        &self,
        input: &Tensor,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
    ) -> Result<Tensor> {
        let dims = input.shape.dims();
        let (n, h, w, c) = match dims {
            [n, h, w, c] => (*n, *h, *w, *c),
            _ => {
                return Err(Error::InvalidOperation(
                    "resize_bilinear_nhwc expects NHWC".into(),
                ))
            }
        };
        let output_shape = Shape::from_dims(&[n, out_h, out_w, c]);
        let total = n * out_h * out_w * c;
        let mut out = Tensor::empty_dtype(output_shape, DType::F32, input.device.clone())?;
        let k = self
            .kernels
            .get("resize_bilinear_nhwc_kernel")
            .ok_or_else(|| Error::Cuda("resize_bilinear_nhwc_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(total as u32);
        launch_kernel!(
            k,
            cfg,
            input.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            n as i32,
            h as i32,
            w as i32,
            c as i32,
            out_h as i32,
            out_w as i32,
            if align_corners { 1 } else { 0 }
        )?;
        Ok(out)
    }

    /// Center crop NHWC
    pub fn center_crop_nhwc(&self, input: &Tensor, tgt_h: usize, tgt_w: usize) -> Result<Tensor> {
        let dims = input.shape.dims();
        let (n, h, w, c) = match dims {
            [n, h, w, c] => (*n, *h, *w, *c),
            _ => {
                return Err(Error::InvalidOperation(
                    "center_crop_nhwc expects NHWC".into(),
                ))
            }
        };
        if tgt_h > h || tgt_w > w {
            return Err(Error::InvalidOperation(
                "center crop size exceeds input".into(),
            ));
        }
        let y0 = ((h - tgt_h) / 2) as i32;
        let x0 = ((w - tgt_w) / 2) as i32;
        let output_shape = Shape::from_dims(&[n, tgt_h, tgt_w, c]);
        let total = n * tgt_h * tgt_w * c;
        let mut out = Tensor::empty_dtype(output_shape, DType::F32, input.device.clone())?;
        let k = self
            .kernels
            .get("center_crop_nhwc_kernel")
            .ok_or_else(|| Error::Cuda("center_crop_nhwc_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(total as u32);
        launch_kernel!(
            k,
            cfg,
            input.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            n as i32,
            h as i32,
            w as i32,
            c as i32,
            y0,
            x0,
            tgt_h as i32,
            tgt_w as i32
        )?;
        Ok(out)
    }

    /// Normalize NHWC per channel
    pub fn normalize_nhwc(&self, input: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
        let dims = input.shape.dims();
        let (n, h, w, c) = match dims {
            [n, h, w, c] => (*n, *h, *w, *c),
            _ => {
                return Err(Error::InvalidOperation(
                    "normalize_nhwc expects NHWC".into(),
                ))
            }
        };
        if mean.len() != c || std.len() != c {
            return Err(Error::InvalidOperation(
                "mean/std size must match channels".into(),
            ));
        }
        let inv_std: Vec<f32> = std.iter().map(|&v| 1.0f32 / v).collect();
        let mut mean_gpu =
            crate::tensor::alloc_from_pool(&input.device, c).map_err(|_| Error::CudaDriver)?;
        let mut inv_gpu =
            crate::tensor::alloc_from_pool(&input.device, c).map_err(|_| Error::CudaDriver)?;
        input
            .device
            .htod_copy_into(mean.to_vec(), &mut mean_gpu)
            .map_err(|_| Error::CudaDriver)?;
        input
            .device
            .htod_copy_into(inv_std, &mut inv_gpu)
            .map_err(|_| Error::CudaDriver)?;
        let output_shape = input.shape.clone();
        let total = n * h * w * c;
        let mut out = Tensor::empty_dtype(output_shape, DType::F32, input.device.clone())?;
        let k = self
            .kernels
            .get("normalize_nhwc_kernel")
            .ok_or_else(|| Error::Cuda("normalize_nhwc_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(total as u32);
        launch_kernel!(
            k,
            cfg,
            input.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            &mean_gpu,
            &inv_gpu,
            n as i32,
            h as i32,
            w as i32,
            c as i32
        )?;
        Ok(out)
    }

    /// Permute tensor from NHWC to NCHW format on GPU
    pub fn permute_nhwc_to_nchw(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dims.len() != 4 {
            return Err(Error::InvalidOperation(format!(
                "Permute NHWC to NCHW requires 4D tensor, got {:?}",
                dims
            )));
        }

        let batch = dims[0];
        let height = dims[1];
        let width = dims[2];
        let channels = dims[3];

        let output_shape = Shape::from_dims(&[batch, channels, height, width]);
        let total_elements = tensor.shape.elem_count();
        let block_size = 256;
        let grid_size = total_elements.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if tensor.dtype() == DType::BF16 && tensor.storage_dtype() == DType::BF16 {
            CudaKernels::ensure_kernel(
                &self.device,
                "permute_nhwc_to_nchw_kernel_bf16",
                PERMUTE_NHWC_TO_NCHW_KERNEL_BF16,
            )?;
            let func = self
                .device
                .get_func(
                    "permute_nhwc_to_nchw_kernel_bf16",
                    "permute_nhwc_to_nchw_kernel_bf16",
                )
                .ok_or_else(|| Error::Cuda("permute_nhwc_to_nchw_kernel_bf16 not found".into()))?;

            let mut output = Tensor::empty_dtype(output_shape, DType::BF16, tensor.device.clone())?;

            let input_ptr = tensor.as_device_ptr_bf16("permute_nhwc_to_nchw:input")? as u64;
            let output_ptr = output.as_mut_device_ptr_bf16("permute_nhwc_to_nchw:output")? as u64;

            launch_kernel!(
                func,
                config,
                input_ptr,
                output_ptr,
                batch as u32,
                height as u32,
                width as u32,
                channels as u32
            )?;

            Ok(output)
        } else {
            let tensor_f32 = ensure_f32_tensor(tensor)?;
            let mut output = Tensor::empty_dtype(output_shape, DType::F32, tensor.device.clone())?;

            let kernel = self
                .kernels
                .get("permute_nhwc_to_nchw_kernel")
                .ok_or_else(|| Error::Cuda("permute_nhwc_to_nchw_kernel not found".into()))?
                .clone();

            launch_kernel!(
                kernel,
                config,
                tensor_f32.storage.try_as_slice_f32()?,
                output.storage.try_as_slice_f32()?,
                batch as u32,
                height as u32,
                width as u32,
                channels as u32
            )?;

            Ok(output)
        }
    }

    /// Permute tensor from NCHW to NHWC format on GPU
    pub fn permute_nchw_to_nhwc(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape.dims();
        if dims.len() != 4 {
            return Err(Error::InvalidOperation(format!(
                "Permute NCHW to NHWC requires 4D tensor, got {:?}",
                dims
            )));
        }
        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let output_shape = Shape::from_dims(&[batch, height, width, channels]);
        let total_elements = tensor.shape.elem_count();
        let block_size = 256;
        let grid_size = total_elements.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if tensor.dtype() == DType::BF16 && tensor.storage_dtype() == DType::BF16 {
            CudaKernels::ensure_kernel(
                &self.device,
                "permute_nchw_to_nhwc_kernel_bf16",
                PERMUTE_NCHW_TO_NHWC_KERNEL_BF16,
            )?;
            let func = self
                .device
                .get_func(
                    "permute_nchw_to_nhwc_kernel_bf16",
                    "permute_nchw_to_nhwc_kernel_bf16",
                )
                .ok_or_else(|| Error::Cuda("permute_nchw_to_nhwc_kernel_bf16 not found".into()))?;

            let mut output = Tensor::empty_dtype(output_shape, DType::BF16, tensor.device.clone())?;

            let input_ptr = tensor.as_device_ptr_bf16("permute_nchw_to_nhwc:input")? as u64;
            let output_ptr = output.as_mut_device_ptr_bf16("permute_nchw_to_nhwc:output")? as u64;

            launch_kernel!(
                func,
                config,
                input_ptr,
                output_ptr,
                batch as u32,
                channels as u32,
                height as u32,
                width as u32
            )?;

            Ok(output)
        } else {
            let tensor_f32 = ensure_f32_tensor(tensor)?;
            let mut output = Tensor::empty_dtype(output_shape, DType::F32, tensor.device.clone())?;

            let kernel = self
                .kernels
                .get("permute_nchw_to_nhwc_kernel")
                .ok_or_else(|| Error::Cuda("permute_nchw_to_nhwc_kernel not found".into()))?
                .clone();

            launch_kernel!(
                kernel,
                config,
                tensor_f32.storage.try_as_slice_f32()?,
                output.storage.try_as_slice_f32()?,
                batch as u32,
                channels as u32,
                height as u32,
                width as u32
            )?;

            Ok(output)
        }
    }

    pub fn permute_generic(&self, tensor: &Tensor, perm: &[usize]) -> Result<Tensor> {
        let shape = tensor.shape.dims();
        let rank = shape.len();
        if perm.len() != rank {
            return Err(Error::InvalidOperation(format!(
                "permute_generic dims {:?} do not match rank {}",
                perm, rank
            )));
        }
        if rank == 0 {
            return tensor.clone_result();
        }
        if rank > 8 {
            return Err(Error::Unsupported(format!(
                "permute_generic supports up to 8 dimensions, got {}",
                rank
            )));
        }

        let out_dims: Vec<usize> = perm.iter().map(|&idx| shape[idx]).collect();
        let total = tensor.shape.elem_count();
        let dtype = tensor.dtype();

        let mut in_strides = vec![1i64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            in_strides[i] = in_strides[i + 1] * shape[i + 1] as i64;
        }
        let mut out_strides = vec![1i64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1] as i64;
        }
        let perm_i32: Vec<i32> = perm
            .iter()
            .map(|&p| {
                i32::try_from(p).map_err(|_| Error::InvalidInput("perm index exceeds i32".into()))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut d_in_strides = unsafe { self.device.alloc::<i64>(rank) }
            .map_err(|_| Error::Cuda("alloc in_strides".into()))?;
        self.device
            .htod_copy_into(in_strides.clone(), &mut d_in_strides)?;
        let mut d_out_strides = unsafe { self.device.alloc::<i64>(rank) }
            .map_err(|_| Error::Cuda("alloc out_strides".into()))?;
        self.device
            .htod_copy_into(out_strides.clone(), &mut d_out_strides)?;
        let mut d_perm = unsafe { self.device.alloc::<i32>(rank) }
            .map_err(|_| Error::Cuda("alloc perm".into()))?;
        self.device.htod_copy_into(perm_i32.clone(), &mut d_perm)?;

        let block = 256u32;
        let grid = std::cmp::max(
            1,
            std::cmp::min(
                ((total + block as usize - 1) / block as usize) as u32,
                65_535,
            ),
        );
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        match dtype {
            DType::F32 => {
                let kernel = self
                    .device
                    .get_func("permute_generic_f32_kernel", "permute_generic_f32_kernel")
                    .ok_or_else(|| Error::Cuda("permute_generic_f32_kernel not found".into()))?;
                let mut output = Tensor::empty_dtype(
                    Shape::from_dims(&out_dims),
                    DType::F32,
                    tensor.device.clone(),
                )?;
                launch_kernel!(
                    kernel,
                    cfg,
                    tensor.storage.try_as_slice_f32()?,
                    output.storage.try_as_slice_f32()?,
                    rank as i32,
                    &d_in_strides,
                    &d_out_strides,
                    &d_perm,
                    total as i64
                )?;
                // No device-wide sync: the scatter kernel launches on the
                // default stream and every subsequent consumer uses the same
                // stream. The explicit sync was forcing ~128 device syncs per
                // Klein backward step (32 blocks × 4 transposes in
                // attention_backward_recompute) for no ordering benefit.
                Ok(output)
            }
            DType::BF16 => {
                #[cfg(feature = "bf16_u16")]
                {
                    let kernel = self
                        .device
                        .get_func("permute_generic_bf16_kernel", "permute_generic_bf16_kernel")
                        .ok_or_else(|| {
                            Error::Cuda("permute_generic_bf16_kernel not found".into())
                        })?;
                    let mut output = Tensor::empty_dtype(
                        Shape::from_dims(&out_dims),
                        DType::BF16,
                        tensor.device.clone(),
                    )?;

                    let input_ptr = tensor.as_device_ptr_bf16("permute_generic:input")? as u64;
                    let output_ptr =
                        output.as_mut_device_ptr_bf16("permute_generic:output")? as u64;

                    launch_kernel!(
                        kernel,
                        cfg,
                        input_ptr,
                        output_ptr,
                        rank as i32,
                        &d_in_strides,
                        &d_out_strides,
                        &d_perm,
                        total as i64
                    )?;
                    // No device-wide sync — see F32 branch comment.
                    // Klein attention backward hits this path 4 times per
                    // block (k/v/attn/d_logits transpose_dims), 32 blocks
                    // per step = 128 sync points eliminated.
                    Ok(output)
                }
                #[cfg(not(feature = "bf16_u16"))]
                {
                    Err(Error::Unsupported(
                        "BF16 permute requires the bf16_u16 feature".into(),
                    ))
                }
            }
            other => Err(Error::Unsupported(format!(
                "permute_generic unsupported dtype {:?}",
                other
            ))),
        }
    }

    /// Materialize a strided view (custom_strides and/or non-zero view_offset)
    /// into a contiguous row-major tensor of the same shape and dtype.
    ///
    /// Handles narrow/chunk views (strides from source, view_offset = start *
    /// stride[dim]) and any composition such as narrow-of-permute. Called by
    /// `Tensor::contiguous()` when the view cannot be represented as a pure
    /// permute of a contiguous buffer.
    pub fn materialize_view(&self, tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape.dims();
        let rank = shape.len();
        if rank == 0 {
            return tensor.clone_result();
        }
        if rank > 8 {
            return Err(Error::Unsupported(format!(
                "materialize_view supports up to 8 dimensions, got {}",
                rank
            )));
        }

        let total = tensor.shape.elem_count();
        if total == 0 {
            return Tensor::empty_dtype(
                Shape::from_dims(shape),
                tensor.dtype(),
                tensor.device.clone(),
            );
        }

        let in_strides_usize = tensor.strides();
        let in_strides: Vec<i64> = in_strides_usize.iter().map(|&s| s as i64).collect();
        let mut out_strides = vec![1i64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * shape[i + 1] as i64;
        }
        let in_offset_i64 = tensor.offset() as i64;

        let mut d_in_strides = unsafe { self.device.alloc::<i64>(rank) }
            .map_err(|_| Error::Cuda("alloc in_strides".into()))?;
        self.device
            .htod_copy_into(in_strides, &mut d_in_strides)?;
        let mut d_out_strides = unsafe { self.device.alloc::<i64>(rank) }
            .map_err(|_| Error::Cuda("alloc out_strides".into()))?;
        self.device
            .htod_copy_into(out_strides, &mut d_out_strides)?;

        let block = 256u32;
        let grid = std::cmp::max(
            1,
            std::cmp::min(
                ((total + block as usize - 1) / block as usize) as u32,
                65_535,
            ),
        );
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        match tensor.dtype() {
            DType::F32 => {
                let kernel = self
                    .device
                    .get_func(
                        "materialize_strided_f32_kernel",
                        "materialize_strided_f32_kernel",
                    )
                    .ok_or_else(|| {
                        Error::Cuda("materialize_strided_f32_kernel not found".into())
                    })?;
                let output = Tensor::empty_dtype(
                    Shape::from_dims(shape),
                    DType::F32,
                    tensor.device.clone(),
                )?;
                launch_kernel!(
                    kernel,
                    cfg,
                    tensor.storage.try_as_slice_f32()?,
                    output.storage.try_as_slice_f32()?,
                    rank as i32,
                    &d_in_strides,
                    &d_out_strides,
                    in_offset_i64,
                    total as i64
                )?;
                Ok(output)
            }
            DType::BF16 => {
                #[cfg(feature = "bf16_u16")]
                {
                    let kernel = self
                        .device
                        .get_func(
                            "materialize_strided_bf16_kernel",
                            "materialize_strided_bf16_kernel",
                        )
                        .ok_or_else(|| {
                            Error::Cuda("materialize_strided_bf16_kernel not found".into())
                        })?;
                    let mut output = Tensor::empty_dtype(
                        Shape::from_dims(shape),
                        DType::BF16,
                        tensor.device.clone(),
                    )?;
                    let input_ptr =
                        tensor.as_device_ptr_bf16("materialize_view:input")? as u64;
                    let output_ptr =
                        output.as_mut_device_ptr_bf16("materialize_view:output")? as u64;
                    launch_kernel!(
                        kernel,
                        cfg,
                        input_ptr,
                        output_ptr,
                        rank as i32,
                        &d_in_strides,
                        &d_out_strides,
                        in_offset_i64,
                        total as i64
                    )?;
                    Ok(output)
                }
                #[cfg(not(feature = "bf16_u16"))]
                {
                    Err(Error::Unsupported(
                        "BF16 materialize_view requires the bf16_u16 feature".into(),
                    ))
                }
            }
            other => Err(Error::Unsupported(format!(
                "materialize_view unsupported dtype {:?}",
                other
            ))),
        }
    }

    /// Permute weights [KH,KW,IC,OC] -> [OC,IC,KH,KW]
    pub fn weight_khwkicoc_to_ocickhkw(&self, w: &Tensor) -> Result<Tensor> {
        let d = w.shape.dims();
        if d.len() != 4 {
            return Err(Error::InvalidOperation("weight permute expects 4D".into()));
        }
        let (kh, kw, ic, oc) = (d[0], d[1], d[2], d[3]);
        let mut out = Tensor::empty_dtype(
            Shape::from_dims(&[oc, ic, kh, kw]),
            DType::F32,
            w.device.clone(),
        )?;
        let k = self
            .kernels
            .get("permute_w_khwkicoc_to_ocickhkw")
            .ok_or_else(|| Error::Cuda("permute_w_khwkicoc_to_ocickhkw not found".into()))?
            .clone();
        let total = (kh * kw * ic * oc) as u32;
        let cfg = LaunchConfig::for_num_elems(total);
        launch_kernel!(
            k,
            cfg,
            w.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            kh as i32,
            kw as i32,
            ic as i32,
            oc as i32
        )?;
        Ok(out)
    }

    /// Permute weights [OC,IC,KH,KW] -> [KH,KW,IC,OC]
    pub fn weight_ocickhkw_to_khwkicoc(&self, w: &Tensor) -> Result<Tensor> {
        let d = w.shape.dims();
        if d.len() != 4 {
            return Err(Error::InvalidOperation("weight permute expects 4D".into()));
        }
        let (oc, ic, kh, kw) = (d[0], d[1], d[2], d[3]);
        let mut out = Tensor::empty_dtype(
            Shape::from_dims(&[kh, kw, ic, oc]),
            DType::F32,
            w.device.clone(),
        )?;
        let k = self
            .kernels
            .get("permute_w_ocickhkw_to_khwkicoc")
            .ok_or_else(|| Error::Cuda("permute_w_ocickhkw_to_khwkicoc not found".into()))?
            .clone();
        let total = (kh * kw * ic * oc) as u32;
        let cfg = LaunchConfig::for_num_elems(total);
        launch_kernel!(
            k,
            cfg,
            w.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            oc as i32,
            ic as i32,
            kh as i32,
            kw as i32
        )?;
        Ok(out)
    }

    /// Elementwise exponential
    pub fn exp(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut out = Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        let kernel = self
            .kernels
            .get("exp_kernel")
            .ok_or_else(|| Error::Cuda("exp_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(
            kernel,
            cfg,
            tensor.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            n as i32
        )?;
        Ok(out)
    }

    /// Elementwise natural logarithm
    pub fn log(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut out = Tensor::empty_dtype(tensor.shape.clone(), DType::F32, tensor.device.clone())?;
        let n = tensor.shape.elem_count();
        let kernel = self
            .kernels
            .get("log_kernel")
            .ok_or_else(|| Error::Cuda("log_kernel not found".into()))?
            .clone();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        launch_kernel!(
            kernel,
            cfg,
            tensor.storage.try_as_slice_f32()?,
            out.storage.try_as_slice_f32()?,
            n as i32
        )?;
        Ok(out)
    }

    /// Index select along a dimension (GPU kernel)
    pub fn index_select(&self, tensor: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape().dims();
        let idx_shape = indices.shape().dims();
        if dim >= shape.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of bounds",
                dim
            )));
        }
        if idx_shape.len() != 1 {
            return Err(Error::InvalidOperation(format!(
                "Indices must be 1D, got {:?}",
                idx_shape
            )));
        }
        let num_indices = indices.shape().elem_count();
        let mut out_shape = shape.to_vec();
        out_shape[dim] = num_indices;
        let out_shape_s = Shape::from_dims(&out_shape);
        let mut output =
            Tensor::empty_dtype(out_shape_s.clone(), DType::F32, tensor.device.clone())?;

        // Prepare strides and dims arrays
        let ndim = shape.len() as i32;
        let in_dims: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let out_dims: Vec<i32> = out_shape.iter().map(|&x| x as i32).collect();
        let mut in_strides = vec![1i32; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            in_strides[i] = in_strides[i + 1] * shape[i + 1] as i32;
        }
        let mut out_strides = vec![1i32; out_shape.len()];
        for i in (0..out_shape.len() - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1] as i32;
        }

        // Copy arrays to device
        let mut d_in_dims = unsafe { self.device.alloc::<i32>(in_dims.len()) }?;
        self.device
            .htod_copy_into(in_dims.clone(), &mut d_in_dims)?;
        let mut d_out_dims = unsafe { self.device.alloc::<i32>(out_dims.len()) }?;
        self.device
            .htod_copy_into(out_dims.clone(), &mut d_out_dims)?;
        let mut d_in_strides = unsafe { self.device.alloc::<i32>(in_strides.len()) }?;
        self.device
            .htod_copy_into(in_strides.clone(), &mut d_in_strides)?;
        let mut d_out_strides = unsafe { self.device.alloc::<i32>(out_strides.len()) }?;
        self.device
            .htod_copy_into(out_strides.clone(), &mut d_out_strides)?;

        let kernel = self
            .kernels
            .get("index_select_kernel")
            .ok_or_else(|| Error::Cuda("index_select_kernel not found".into()))?
            .clone();

        let numel = out_shape_s.elem_count();
        let block = 256usize;
        let grid = numel.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(
            kernel,
            cfg,
            tensor.storage.try_as_slice_f32()?,
            indices.storage.try_as_slice_f32()?,
            output.storage.try_as_slice_f32()?,
            ndim,
            &d_in_dims,
            &d_out_dims,
            &d_in_strides,
            &d_out_strides,
            dim as i32,
            numel as i32
        )?;

        Ok(output)
    }

    /// General slice across multiple dimensions (GPU kernel)
    pub fn slice(&self, tensor: &Tensor, ranges: &[(usize, usize)]) -> Result<Tensor> {
        let shape = tensor.shape().dims();
        if ranges.len() != shape.len() {
            return Err(Error::InvalidOperation("ranges/dims mismatch".to_string()));
        }
        let mut out_shape = Vec::with_capacity(ranges.len());
        let mut starts: Vec<i32> = Vec::with_capacity(ranges.len());
        for (i, &(s, e)) in ranges.iter().enumerate() {
            if s >= e || e > shape[i] {
                return Err(Error::InvalidOperation(format!(
                    "invalid range {}-{} for dim {}",
                    s, e, i
                )));
            }
            out_shape.push(e - s);
            starts.push(s as i32);
        }
        let out_s = Shape::from_dims(&out_shape);
        let mut output = Tensor::empty_dtype(out_s.clone(), tensor.dtype(), tensor.device.clone())?;

        // Strides
        let mut in_strides = vec![1i32; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            in_strides[i] = in_strides[i + 1] * shape[i + 1] as i32;
        }
        let mut out_strides = vec![1i32; out_shape.len()];
        for i in (0..out_shape.len() - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1] as i32;
        }

        let mut d_in_strides = unsafe { self.device.alloc::<i32>(in_strides.len()) }?;
        self.device
            .htod_copy_into(in_strides.clone(), &mut d_in_strides)?;
        let mut d_out_strides = unsafe { self.device.alloc::<i32>(out_strides.len()) }?;
        self.device
            .htod_copy_into(out_strides.clone(), &mut d_out_strides)?;
        let mut d_starts = unsafe { self.device.alloc::<i32>(starts.len()) }?;
        self.device.htod_copy_into(starts.clone(), &mut d_starts)?;

        let numel = out_s.elem_count();
        let block = 256usize;
        let grid = numel.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        match tensor.dtype() {
            DType::F32 => {
                let kernel = self
                    .kernels
                    .get("slice_kernel")
                    .ok_or_else(|| Error::Cuda("slice_kernel not found".into()))?
                    .clone();
                launch_kernel!(
                    kernel,
                    cfg,
                    tensor.storage.try_as_slice_f32()?,
                    output.storage.try_as_slice_f32()?,
                    shape.len() as i32,
                    &d_in_strides,
                    &d_out_strides,
                    &d_starts,
                    numel as i32
                )?;
            }
            DType::BF16 => {
                #[cfg(feature = "bf16_u16")]
                {
                    let kernel = self
                        .kernels
                        .get("slice_kernel_bf16")
                        .ok_or_else(|| Error::Cuda("slice_kernel_bf16 not found".into()))?
                        .clone();
                    let input_ptr = tensor.as_device_ptr_bf16("slice:input")? as u64;
                    let output_ptr = output.as_mut_device_ptr_bf16("slice:output")? as u64;

                    launch_kernel!(
                        kernel,
                        cfg,
                        input_ptr,
                        output_ptr,
                        shape.len() as i32,
                        &d_in_strides,
                        &d_out_strides,
                        &d_starts,
                        numel as i32
                    )?;
                }
                #[cfg(not(feature = "bf16_u16"))]
                {
                    return Err(Error::Unsupported(
                        "BF16 slice requires the bf16_u16 feature".into(),
                    ));
                }
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "slice not implemented for dtype {:?}",
                    other
                )));
            }
        }

        Ok(output)
    }
}

/// Scatter add operation for efficient gradient accumulation
/// This is a generic tensor operation, not model-specific
pub fn scatter_add(
    input_shape: &[usize],
    grad_output: &Tensor,
    indices: &Tensor,
    dim: usize,
) -> Result<Tensor> {
    crate::cuda_kernels_gpu::CudaKernels::scatter_add(input_shape, grad_output, indices, dim)
}

pub fn gather_rows(input: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
    crate::cuda_kernels_gpu::CudaKernels::gather_rows(input, indices, dim)
}

/// Compile CUDA kernel from source to PTX
/// This is used by other modules that generating custom kernels
pub fn compile_kernel(kernel_name: &str, kernel_code: &str) -> Result<Vec<u8>> {
    let ptx = compile_ptx(kernel_code)
        .map_err(|e| Error::Cuda(format!("Failed to compile {}: {:?}", kernel_name, e)))?;
    // PTX is already a compiled binary, we can't extract bytes from it
    // For now, return an error as this function isn't used
    Err(Error::InvalidOperation(
        "Cannot extract bytes from compiled PTX".into(),
    ))
}

#[cfg(feature = "bf16_u16")]
pub fn gate_add_bf16_inplace(dst: &Tensor, gate: &Tensor) -> Result<()> {
    use crate::cuda_kernel_sources::GATE_ADD_BF16_INPLACE_KERNEL;

    if dst.dtype() != DType::BF16 || gate.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "gate_add_bf16_inplace: expected BF16".into(),
        ));
    }

    let dims = dst.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(
            "gate_add_bf16_inplace: dst must be 3D [B, T, H]".into(),
        ));
    }
    let batch = dims[0];
    let tokens = dims[1];
    let hidden = dims[2];

    let g_dims = gate.shape().dims();
    // gate can be [B, H] or [B, 1, H]
    if g_dims.len() == 2 {
        if g_dims[0] != batch || g_dims[1] != hidden {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(&[batch, hidden]),
                got: gate.shape().clone(),
            });
        }
    } else if g_dims.len() == 3 {
        if g_dims[0] != batch || g_dims[1] != 1 || g_dims[2] != hidden {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(&[batch, 1, hidden]),
                got: gate.shape().clone(),
            });
        }
    } else {
        return Err(Error::InvalidInput(
            "gate_add_bf16_inplace: gate must be [B, H] or [B, 1, H]".into(),
        ));
    }

    let device = dst.device();
    CudaKernels::ensure_kernel(
        device,
        "gate_add_bf16_inplace_kernel",
        GATE_ADD_BF16_INPLACE_KERNEL,
    )?;

    let func = device
        .get_func(
            "gate_add_bf16_inplace_kernel",
            "gate_add_bf16_inplace_kernel",
        )
        .ok_or_else(|| Error::Cuda("gate_add_bf16_inplace_kernel not found".into()))?;

    let total = batch * tokens * hidden;
    let cfg = LaunchConfig::for_num_elems(total as u32);

    unsafe {
        func.launch(
            cfg,
            (
                dst.storage.try_as_slice_u16()?,
                gate.storage.try_as_slice_u16()?,
                batch as i32,
                tokens as i32,
                hidden as i32,
            ),
        )
    }
    .map_err(|e| Error::Cuda(format!("gate_add_bf16_inplace launch failed: {:?}", e)))?;

    Ok(())
}
