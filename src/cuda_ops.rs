use crate::cuda::ffi;
use crate::cuda_kernels::CudaKernels;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda_ops_bf16;
use crate::device::CudaStreamRawPtrExt;
use crate::{
    strict::{allow_clone, allow_f32_in_kernel_scoped, scope, GuardMode},
    DType, Error, Result, Shape, Tensor,
};

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::DevicePtr;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::Mutex;

lazy_static! {
    // Keyed by device Arc pointer to ensure per-context cache, not just ordinal
    static ref KERNELS_CACHE: Mutex<HashMap<usize, Arc<CudaKernels>>> = Mutex::new(HashMap::new());
}

#[derive(Clone, Copy)]
enum CompareOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// GPU operations using CUDA kernels with NVRTC compilation
pub struct GpuOps;

impl GpuOps {
    fn cast_to_f32_tensor(t: &Tensor) -> Result<Tensor> {
        if t.dtype() == crate::DType::F32 {
            if t.storage_dtype() != crate::DType::F32 {
                if crate::trace::trace_on() {
                    eprintln!(
                        "[cast_to_f32_tensor] forcing owning FP32: storage={:?} shape={:?}",
                        t.storage_dtype(),
                        t.shape().dims()
                    );
                }
                return allow_f32_in_kernel_scoped(|| t.to_dtype(crate::DType::F32));
            }
            let _guard = allow_clone();
            return Ok(t.clone());
        }
        let cast = allow_f32_in_kernel_scoped(|| t.to_dtype(crate::DType::F32))?;
        if crate::trace::trace_on() {
            eprintln!(
                "[cast_to_f32_tensor] cast from {:?} storage {:?} -> {:?} storage {:?} shape {:?}",
                t.dtype(),
                t.storage_dtype(),
                cast.dtype(),
                cast.storage_dtype(),
                cast.shape().dims()
            );
        }
        Ok(cast)
    }

    fn restore_dtype(t: Tensor, dtype: crate::DType) -> Result<Tensor> {
        if dtype == crate::DType::F32 {
            Ok(t)
        } else {
            allow_f32_in_kernel_scoped(|| t.to_dtype(dtype))
        }
    }

    fn binary_target_dtype(a: crate::DType, b: crate::DType) -> crate::DType {
        use crate::DType::{BF16, F16, F32};
        if a == F32 || b == F32 {
            F32
        } else if a == F16 || b == F16 {
            F16
        } else if a == BF16 || b == BF16 {
            BF16
        } else {
            a
        }
    }

    /// Get or create CudaKernels instance for a device
    fn get_kernels(device: &Arc<cudarc::driver::CudaDevice>) -> Result<Arc<CudaKernels>> {
        let device_id = Arc::as_ptr(device) as usize;
        let mut cache = KERNELS_CACHE
            .lock()
            .map_err(|_| Error::Training("kernels cache mutex poisoned".into()))?;

        if let Some(kernels) = cache.get(&device_id) {
            Ok(kernels.clone())
        } else {
            let kernels = Arc::new(CudaKernels::new(device.clone())?);
            cache.insert(device_id, kernels.clone());
            Ok(kernels)
        }
    }

    /// Element-wise addition
    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            if a.dtype() == crate::DType::BF16 && b.dtype() == crate::DType::BF16 {
                return Self::add_bf16(a, b);
            }
        }
        let target_dtype = Self::binary_target_dtype(a.dtype(), b.dtype());
        let a_cast = Self::cast_to_f32_tensor(a)?;
        let b_cast = Self::cast_to_f32_tensor(b)?;
        if a_cast.device().ordinal() != b_cast.device().ordinal() {
            return Err(Error::InvalidInput(
                "tensors reside on different devices".into(),
            ));
        }
        let kernels = Self::get_kernels(&a_cast.device)?;
        let result_f32 = if a_cast.shape() != b_cast.shape() {
            static BC_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *BC_TRACE.get_or_init(|| {
                std::env::var("FLAME_BC_TRACE").ok().as_deref() == Some("1")
            })
            {
                eprintln!(
                    "[bc-trace] add lhs={:?} rhs={:?}",
                    a_cast.shape().dims().to_vec(),
                    b_cast.shape().dims().to_vec()
                );
            }
            kernels.add_bc(&a_cast, &b_cast)?
        } else {
            kernels.add(&a_cast, &b_cast)?
        };
        let out = Self::restore_dtype(result_f32, target_dtype)?;
        Ok(out)
    }

    fn cast_and_broadcast_f32(
        a: &Tensor,
        b: &Tensor,
    ) -> Result<(Tensor, Tensor, Arc<CudaKernels>, Shape)> {
        let target_shape = a.shape().broadcast_shape_binary_op(b.shape())?;

        let a_f32 = Self::cast_to_f32_tensor(a)?;
        let b_f32 = Self::cast_to_f32_tensor(b)?;

        if a_f32.device().ordinal() != b_f32.device().ordinal() {
            return Err(Error::InvalidInput(
                "tensors reside on different devices".into(),
            ));
        }

        let device = a_f32.device.clone();
        let kernels = Self::get_kernels(&device)?;

        let a_bc = if a_f32.shape() != &target_shape {
            CudaKernels::broadcast(&a_f32, &target_shape)?
        } else {
            let _guard = allow_clone();
            a_f32.clone()
        };

        let b_bc = if b_f32.shape() != &target_shape {
            CudaKernels::broadcast(&b_f32, &target_shape)?
        } else {
            let _guard = allow_clone();
            b_f32.clone()
        };

        Ok((a_bc, b_bc, kernels, target_shape))
    }

    fn compare_binary(a: &Tensor, b: &Tensor, op: CompareOp) -> Result<Tensor> {
        let (a_bc, b_bc, kernels, target_shape) = Self::cast_and_broadcast_f32(a, b)?;

        if matches!(op, CompareOp::Eq)
            && (a.shape().dims() != target_shape.dims() || b.shape().dims() != target_shape.dims())
        {
            return Err(Error::InvalidOperation(
                "Equality comparison requires matching shapes".into(),
            ));
        }

        let out = match op {
            CompareOp::Gt => kernels.compare_gt(&a_bc, &b_bc)?,
            CompareOp::Ge => kernels.compare_ge(&a_bc, &b_bc)?,
            CompareOp::Lt => kernels.compare_lt(&a_bc, &b_bc)?,
            CompareOp::Le => kernels.compare_le(&a_bc, &b_bc)?,
            CompareOp::Eq => kernels.compare_eq(&a_bc, &b_bc)?,
            CompareOp::Ne => kernels.compare_ne(&a_bc, &b_bc)?,
        };

        let target_dtype = if a.dtype() == crate::DType::BF16 || b.dtype() == crate::DType::BF16 {
            crate::DType::BF16
        } else {
            crate::DType::F32
        };
        if out.dtype() == target_dtype {
            Ok(out)
        } else {
            out.to_dtype(target_dtype)
        }
    }

    pub fn cmp_gt(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Self::compare_binary(a, b, CompareOp::Gt)
    }

    pub fn cmp_ge(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Self::compare_binary(a, b, CompareOp::Ge)
    }

    pub fn cmp_lt(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Self::compare_binary(a, b, CompareOp::Lt)
    }

    pub fn cmp_le(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Self::compare_binary(a, b, CompareOp::Le)
    }

    pub fn cmp_eq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Self::compare_binary(a, b, CompareOp::Eq)
    }

    pub fn cmp_ne(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Self::compare_binary(a, b, CompareOp::Ne)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    fn add_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.device().ordinal() != b.device().ordinal() {
            return Err(Error::InvalidInput(
                "tensors reside on different devices".into(),
            ));
        }
        let target = a.shape().broadcast_shape_binary_op(b.shape())?;
        if a.shape() == &target && b.shape() == &target {
            // Same shape — no broadcast, no copy needed
            return crate::ops::elt::add_same_dtype(a, b);
        }
        let a_bc_owned;
        let a_ref = if a.shape() != &target {
            a_bc_owned = a.broadcast_to(&target)?;
            &a_bc_owned
        } else { a };
        let b_bc_owned;
        let b_ref = if b.shape() != &target {
            b_bc_owned = b.broadcast_to(&target)?;
            &b_bc_owned
        } else { b };
        crate::ops::elt::add_same_dtype(a_ref, b_ref)
    }

    /// Element-wise multiplication
    pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            if a.dtype() == crate::DType::BF16 && b.dtype() == crate::DType::BF16 {
                return Self::mul_bf16(a, b);
            }
        }
        let target_dtype = Self::binary_target_dtype(a.dtype(), b.dtype());
        let a_cast = Self::cast_to_f32_tensor(a)?;
        let b_cast = Self::cast_to_f32_tensor(b)?;
        if a_cast.device().ordinal() != b_cast.device().ordinal() {
            return Err(Error::InvalidInput(
                "tensors reside on different devices".into(),
            ));
        }
        let kernels = Self::get_kernels(&a_cast.device)?;
        let result_f32 = if a_cast.shape() != b_cast.shape() {
            static BC_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *BC_TRACE.get_or_init(|| {
                std::env::var("FLAME_BC_TRACE").ok().as_deref() == Some("1")
            })
            {
                eprintln!(
                    "[bc-trace] mul lhs={:?} rhs={:?}",
                    a_cast.shape().dims().to_vec(),
                    b_cast.shape().dims().to_vec()
                );
            }
            kernels.mul_bc(&a_cast, &b_cast)?
        } else {
            kernels.mul(&a_cast, &b_cast)?
        };
        let out = Self::restore_dtype(result_f32, target_dtype)?;
        Ok(out)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    fn mul_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.device().ordinal() != b.device().ordinal() {
            return Err(Error::InvalidInput(
                "tensors reside on different devices".into(),
            ));
        }
        let target = a.shape().broadcast_shape_binary_op(b.shape())?;
        if a.shape() == &target && b.shape() == &target {
            return crate::ops::elt::mul_same_dtype(a, b);
        }
        let a_bc_owned;
        let a_ref = if a.shape() != &target {
            a_bc_owned = a.broadcast_to(&target)?;
            &a_bc_owned
        } else { a };
        let b_bc_owned;
        let b_ref = if b.shape() != &target {
            b_bc_owned = b.broadcast_to(&target)?;
            &b_bc_owned
        } else { b };
        crate::ops::elt::mul_same_dtype(a_ref, b_ref)
    }

    /// Scalar multiplication
    pub fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if tensor.dtype() == crate::DType::BF16 {
            return scope("cuda_ops.mul_scalar", GuardMode::env_default(), || {
                crate::ops::elt::mul_scalar_same_dtype(tensor, scalar)
            });
        }

        scope("cuda_ops.mul_scalar", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
            let kernels = Self::get_kernels(&tensor_f32.device)?;
            let result_f32 = kernels.mul_scalar(&tensor_f32, scalar)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// Scalar addition
    pub fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if tensor.dtype() == crate::DType::BF16 {
            return scope("cuda_ops.add_scalar", GuardMode::env_default(), || {
                crate::ops::elt::add_scalar_same_dtype(tensor, scalar)
            });
        }

        scope("cuda_ops.add_scalar", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
            let kernels = Self::get_kernels(&tensor_f32.device)?;
            let result_f32 = kernels.add_scalar(&tensor_f32, scalar)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// ReLU activation
    pub fn relu(tensor: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if tensor.dtype() == crate::DType::BF16 {
            // Phase 6: BF16 routes through the TensorIterator pipeline.
            // Replaced the legacy `cuda_ops_bf16::relu_bf16` → `fc_relu_bf16`
            // FFI. Both paths produce bit-identical output for finite inputs.
            return crate::ops::relu_iter::relu_bf16_iter(tensor);
        }

        scope("cuda_ops.relu", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
            let kernels = Self::get_kernels(&tensor_f32.device)?;
            let result_f32 = kernels.relu(&tensor_f32)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// Sigmoid activation
    pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if tensor.dtype() == crate::DType::BF16 {
            // Phase 6: BF16 routes through the TensorIterator pipeline. Same
            // f32-opmath math as the old F32-roundtrip path, now without the
            // intermediate materialization — enforces CLAUDE.md's "NEVER use
            // F32 fallbacks in inference code" for BF16 inputs.
            return crate::ops::sigmoid_iter::sigmoid_bf16_iter(tensor);
        }
        scope("cuda_ops.sigmoid", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
            let kernels = Self::get_kernels(&tensor_f32.device)?;
            let result_f32 = kernels.sigmoid(&tensor_f32)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// GELU activation
    pub fn gelu(tensor: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if tensor.dtype() == crate::DType::BF16 {
            return cuda_ops_bf16::gelu_bf16(tensor);
        }
        scope("cuda_ops.gelu", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
            let kernels = Self::get_kernels(&tensor_f32.device)?;
            let result_f32 = kernels.gelu(&tensor_f32)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// SiLU activation
    pub fn silu(tensor: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if tensor.dtype() == crate::DType::BF16 {
            return cuda_ops_bf16::silu_bf16(tensor);
        }
        scope("cuda_ops.silu", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
            let kernels = Self::get_kernels(&tensor_f32.device)?;
            let result_f32 = kernels.silu(&tensor_f32)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// Tanh activation
    pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if tensor.dtype() == crate::DType::BF16 {
            // Phase 6: BF16 routes through the TensorIterator pipeline.
            return crate::ops::tanh_iter::tanh_bf16_iter(tensor);
        }
        scope("cuda_ops.tanh", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
            let kernels = Self::get_kernels(&tensor_f32.device)?;
            let result_f32 = kernels.tanh(&tensor_f32)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// Sum reduction
    pub fn sum(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.sum(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Sum reduction along specific dimensions
    pub fn sum_dims(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.sum_dims(&tensor_f32, dims)?;
        let out = Self::restore_dtype(result_f32, target_dtype)?;
        Ok(out)
    }

    /// Transpose
    pub fn transpose(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.transpose(tensor)
    }

    /// Update weights
    pub fn update_weights(weights: &Tensor, gradients: &Tensor, lr: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&weights.device)?;
        kernels.update_weights(weights, gradients, lr)
    }

    /// Leaky ReLU
    pub fn leaky_relu(tensor: &Tensor, negative_slope: f32) -> Result<Tensor> {
        scope("cuda_ops.leaky_relu", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
            let kernels = Self::get_kernels(&tensor_f32.device)?;
            let result_f32 = kernels.leaky_relu(&tensor_f32, negative_slope)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// ELU
    pub fn elu(tensor: &Tensor, alpha: f32) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.elu(&tensor_f32, alpha)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// PReLU
    pub fn prelu(tensor: &Tensor, weight: &Tensor) -> Result<Tensor> {
        scope("cuda_ops.prelu", GuardMode::env_default(), || {
            let target_dtype = tensor.dtype();
            let input_f32 = Self::cast_to_f32_tensor(tensor)?;
            let weight_f32 = Self::cast_to_f32_tensor(weight)?;
            let kernels = Self::get_kernels(&input_f32.device)?;
            let result_f32 = kernels.prelu(&input_f32, &weight_f32)?;
            Self::restore_dtype(result_f32, target_dtype)
        })
    }

    /// Broadcast
    pub fn broadcast(tensor: &Tensor, target_shape: &crate::Shape) -> Result<Tensor> {
        CudaKernels::broadcast(tensor, target_shape)
    }

    /// Element-wise division
    pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            if a.dtype() == crate::DType::BF16 && b.dtype() == crate::DType::BF16 {
                // Phase 5b: the legacy `bf16_elementwise::div_bf16` was
                // deleted. Route through the TensorIterator pipeline
                // (handles broadcasting via stride=0 internally).
                return crate::ops::div_iter::div_bf16_iter(a, b);
            }
        }
        let target_dtype = Self::binary_target_dtype(a.dtype(), b.dtype());
        let a_cast = Self::cast_to_f32_tensor(a)?;
        let b_cast = Self::cast_to_f32_tensor(b)?;
        if a_cast.device().ordinal() != b_cast.device().ordinal() {
            return Err(Error::InvalidInput(
                "tensors reside on different devices".into(),
            ));
        }
        let kernels = Self::get_kernels(&a_cast.device)?;
        let result_f32 = kernels.div(&a_cast, &b_cast)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise maximum (assumes shapes already broadcasted to equal)
    pub fn max_elemwise(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            if a.dtype() == crate::DType::BF16 && b.dtype() == crate::DType::BF16 {
                // Phase 5b: route BF16 to the TensorIterator pipeline.
                return crate::ops::maximum_iter::maximum_bf16_iter(a, b);
            }
        }
        let kernels = Self::get_kernels(&a.device)?;
        kernels.max_elemwise(a, b)
    }

    /// NHWC image ops: resize bilinear
    pub fn resize_bilinear_nhwc(
        input: &Tensor,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
    ) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.resize_bilinear_nhwc(input, out_h, out_w, align_corners)
    }

    /// NHWC image ops: center crop
    pub fn center_crop_nhwc(input: &Tensor, tgt_h: usize, tgt_w: usize) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.center_crop_nhwc(input, tgt_h, tgt_w)
    }

    /// NHWC image ops: normalize per channel
    pub fn normalize_nhwc(input: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.normalize_nhwc(input, mean, std)
    }

    /// Max reduction along dimension
    pub fn max_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        let out = kernels.max_dim(tensor, dim, keepdim)?;
        if out.dtype() == tensor.dtype() {
            Ok(out)
        } else {
            out.to_dtype(tensor.dtype())
        }
    }

    /// Sum along dimension with keepdim
    pub fn sum_dim_keepdim(tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        let result = kernels.sum_dim_keepdim(tensor, dim)?;
        if result.dtype() == tensor.dtype() {
            Ok(result)
        } else {
            result.to_dtype(tensor.dtype())
        }
    }

    /// Variant of `sum_dim_keepdim` that returns a specific dtype (used to keep
    /// Flux tensors in BF16 while still accumulating in FP32 inside the kernel).
    pub fn sum_dim_keepdim_as(tensor: &Tensor, dim: usize, out_dtype: DType) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        let result = kernels.sum_dim_keepdim(tensor, dim)?;
        if result.dtype() == out_dtype {
            Ok(result)
        } else {
            result.to_dtype(out_dtype)
        }
    }

    /// Reduce to scalar by taking the maximum over all dimensions.
    pub fn reduce_max(tensor: &Tensor) -> Result<Tensor> {
        if tensor.shape().elem_count() == 0 {
            return Err(Error::InvalidInput("reduce_max on empty tensor".into()));
        }

        let mut current = tensor.clone_result()?;
        while current.shape().rank() > 0 {
            let dim = current.shape().rank() - 1;
            current = Self::max_dim(&current, dim, false)?;
        }
        Ok(current)
    }

    /// Reduce to scalar by taking the minimum over all dimensions.
    pub fn reduce_min(tensor: &Tensor) -> Result<Tensor> {
        if tensor.shape().elem_count() == 0 {
            return Err(Error::InvalidInput("reduce_min on empty tensor".into()));
        }

        let neg = tensor.neg()?;
        let max_neg = Self::reduce_max(&neg)?;
        max_neg.neg()
    }

    /// Flip tensor along the last dimension.
    pub fn flip_last_dim(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.flip_last_dim(tensor)
    }

    /// Permute (N, A, B, C) → (N, B, A, C) without leaving device memory.
    ///
    /// Flux attention hits this pattern constantly; the older CPU fallback
    /// copied tensors through host RAM and reallocated them as BF16, which is
    /// incompatible with our “GPU-only + no surprise dtype changes” mandate.
    pub fn permute_0213(tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "permute_0213 expects 4D tensor, got {:?}",
                dims
            )));
        }

        let n = i32::try_from(dims[0])
            .map_err(|_| Error::InvalidInput("permute_0213 dimension N exceeds i32".into()))?;
        let a = i32::try_from(dims[1])
            .map_err(|_| Error::InvalidInput("permute_0213 dimension A exceeds i32".into()))?;
        let b = i32::try_from(dims[2])
            .map_err(|_| Error::InvalidInput("permute_0213 dimension B exceeds i32".into()))?;
        let c = i32::try_from(dims[3])
            .map_err(|_| Error::InvalidInput("permute_0213 dimension C exceeds i32".into()))?;

        let device = tensor.device();
        let stream: *mut c_void = device.cuda_stream_raw_ptr();

        // Output stays in the source dtype so Flux never observes a surprise BF16
        // promotion/demotion.  The layout swap only changes axes.
        let mut output = Tensor::zeros_dtype(
            Shape::from_dims(&[dims[0], dims[2], dims[1], dims[3]]),
            tensor.dtype(),
            device.clone(),
        )?;

        unsafe {
            match tensor.dtype() {
                crate::DType::F32 => {
                    let src = tensor.storage.try_as_slice_f32()?;
                    let dst = output.storage_mut().try_as_mut_slice_f32()?;
                    let src_ptr = *src.device_ptr() as *const f32;
                    let dst_ptr = *dst.device_ptr() as *mut f32;
                    ffi::launch_permute0213_f32(src_ptr, dst_ptr, n, a, b, c, stream);
                }
                crate::DType::BF16 => {
                    #[cfg(feature = "bf16_u16")]
                    {
                        let src_ptr =
                            tensor.as_device_ptr_bf16("permute_0213:src")? as *const c_void;
                        let dst_ptr =
                            output.as_mut_device_ptr_bf16("permute_0213:dst")? as *mut c_void;
                        ffi::launch_permute0213_bf16(src_ptr, dst_ptr, n, a, b, c, stream);
                    }
                    #[cfg(not(feature = "bf16_u16"))]
                    {
                        return Err(Error::Unsupported(
                            "BF16 permute requires the bf16_u16 feature".into(),
                        ));
                    }
                }
                other => {
                    return Err(Error::Unsupported(format!(
                        "permute0213 unsupported dtype {other:?}"
                    )));
                }
            }
        }

        Ok(output)
    }

    /// Permute (N, A, B) → (N, B, A) without touching host memory.
    ///
    /// After reshaping attention tensors to `[batch * heads, seq_len, head_dim]`
    /// we immediately swap the last two axes.  The generic CPU fallback staged
    /// BF16 tensors as F32 and violated the BF16-only contract, so we keep this
    /// case on GPU for both F32 and BF16.
    pub fn permute_021(tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "permute_021 expects 3D tensor, got {:?}",
                dims
            )));
        }

        let n = i32::try_from(dims[0])
            .map_err(|_| Error::InvalidInput("permute_021 dimension N exceeds i32".into()))?;
        let a = i32::try_from(dims[1])
            .map_err(|_| Error::InvalidInput("permute_021 dimension A exceeds i32".into()))?;
        let b = i32::try_from(dims[2])
            .map_err(|_| Error::InvalidInput("permute_021 dimension B exceeds i32".into()))?;

        let device = tensor.device();
        let stream: *mut c_void = device.cuda_stream_raw_ptr();

        let mut output = Tensor::zeros_dtype(
            Shape::from_dims(&[dims[0], dims[2], dims[1]]),
            tensor.dtype(),
            device.clone(),
        )?;

        unsafe {
            match tensor.dtype() {
                crate::DType::F32 => {
                    let src = tensor.storage.try_as_slice_f32()?;
                    let dst = output.storage_mut().try_as_mut_slice_f32()?;
                    let src_ptr = *src.device_ptr() as *const f32;
                    let dst_ptr = *dst.device_ptr() as *mut f32;
                    ffi::launch_permute021_f32(src_ptr, dst_ptr, n, a, b, stream);
                }
                crate::DType::BF16 => {
                    #[cfg(feature = "bf16_u16")]
                    {
                        let src_ptr =
                            tensor.as_device_ptr_bf16("permute_021:src")? as *const c_void;
                        let dst_ptr =
                            output.as_mut_device_ptr_bf16("permute_021:dst")? as *mut c_void;
                        ffi::launch_permute021_bf16(src_ptr, dst_ptr, n, a, b, stream);
                    }
                    #[cfg(not(feature = "bf16_u16"))]
                    {
                        return Err(Error::Unsupported(
                            "BF16 permute requires the bf16_u16 feature".into(),
                        ));
                    }
                }
                other => {
                    return Err(Error::Unsupported(format!(
                        "permute_021 unsupported dtype {other:?}"
                    )));
                }
            }
        }

        Ok(output)
    }

    pub fn permute_generic(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.permute_generic(tensor, dims)
    }

    pub fn materialize_view(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.materialize_view(tensor)
    }

    /// NHWC -> NCHW
    pub fn permute_nhwc_to_nchw(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.permute_nhwc_to_nchw(tensor)
    }

    /// NCHW -> NHWC
    pub fn permute_nchw_to_nhwc(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.permute_nchw_to_nhwc(tensor)
    }

    /// Weights [KH,KW,IC,OC] -> [OC,IC,KH,KW]
    pub fn weight_khwkicoc_to_ocickhkw(w: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&w.device)?;
        kernels.weight_khwkicoc_to_ocickhkw(w)
    }

    /// Weights [OC,IC,KH,KW] -> [KH,KW,IC,OC]
    pub fn weight_ocickhkw_to_khwkicoc(w: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&w.device)?;
        kernels.weight_ocickhkw_to_khwkicoc(w)
    }

    /// Elementwise exponential
    pub fn exp(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.exp(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise natural logarithm
    pub fn log(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.log(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise power
    pub fn pow(tensor: &Tensor, exponent: f32) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.pow(&tensor_f32, exponent)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise sine
    pub fn sin(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.sin(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise cosine
    pub fn cos(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.cos(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise square root
    pub fn sqrt(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.sqrt(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise floor
    pub fn floor(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.floor(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise ceil
    pub fn ceil(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.ceil(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Elementwise round
    pub fn round(tensor: &Tensor) -> Result<Tensor> {
        let target_dtype = tensor.dtype();
        let tensor_f32 = Self::cast_to_f32_tensor(tensor)?;
        let kernels = Self::get_kernels(&tensor_f32.device)?;
        let result_f32 = kernels.round(&tensor_f32)?;
        Self::restore_dtype(result_f32, target_dtype)
    }

    /// Index select along a dimension
    pub fn index_select(tensor: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.index_select(tensor, dim, indices)
    }

    /// Slice along multiple dimensions
    pub fn slice(tensor: &Tensor, ranges: &[(usize, usize)]) -> Result<Tensor> {
        if tensor.dtype() == DType::BF16 {
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            {
                let mut current: Option<Tensor> = None;
                for (axis, &(start, end)) in ranges.iter().enumerate() {
                    let source = current.as_ref().unwrap_or(tensor);
                    let dim = source.shape().dims()[axis];
                    if start == 0 && end == dim {
                        continue;
                    }
                    let len = end.saturating_sub(start);
                    let sliced = cuda_ops_bf16::slice_axis_bf16(source, axis, start, len)?;
                    current = Some(sliced);
                }
                return Ok(match current {
                    Some(result) => result,
                    None => tensor.clone_result()?,
                });
            }

            #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
            {
                return Err(Error::Unsupported(
                    "BF16 slice requires cuda+bf16_u16 features".into(),
                ));
            }
        }

        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.slice(tensor, ranges)
    }

    /// Matrix multiplication using cuBLAS
    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        crate::ops::gemm::launch_gemm(a, b)
    }

    // Upsampling operations
    pub fn upsample2d_nearest(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.upsample2d_nearest(input, output_size)
    }

    pub fn upsample2d_bilinear(
        input: &Tensor,
        output_size: (usize, usize),
        align_corners: bool,
    ) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.upsample2d_bilinear(input, output_size, align_corners)
    }

    pub fn upsample2d_nearest_backward(
        grad_output: &Tensor,
        input_size: (usize, usize),
        output_size: (usize, usize),
    ) -> Result<Tensor> {
        let kernels = Self::get_kernels(&grad_output.device)?;
        kernels.upsample2d_nearest_backward(grad_output, input_size, output_size)
    }

    pub fn upsample2d_bilinear_backward(
        grad_output: &Tensor,
        input_size: (usize, usize),
        output_size: (usize, usize),
        align_corners: bool,
    ) -> Result<Tensor> {
        let kernels = Self::get_kernels(&grad_output.device)?;
        kernels.upsample2d_bilinear_backward(grad_output, input_size, output_size, align_corners)
    }

    // Transposed convolution operations
    pub fn conv_transpose2d_forward(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        groups: usize,
        dilation: (usize, usize),
    ) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.conv_transpose2d_forward(
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )
    }

    pub fn conv_transpose2d_backward(
        grad_output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        groups: usize,
        dilation: (usize, usize),
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let kernels = Self::get_kernels(&grad_output.device)?;
        kernels.conv_transpose2d_backward(
            grad_output,
            input,
            weight,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )
    }
}
