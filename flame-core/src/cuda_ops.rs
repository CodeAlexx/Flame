use crate::{Result, Tensor, FlameError, Shape};
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::cuda_kernels::CudaKernels;
use cudarc::cublas::CudaBlas;
use std::sync::Arc;
use lazy_static::lazy_static;
use std::sync::Mutex;
use std::collections::HashMap;

lazy_static! {
    // Keyed by device Arc pointer to ensure per-context cache, not just ordinal
    static ref KERNELS_CACHE: Mutex<HashMap<usize, Arc<CudaKernels>>> = Mutex::new(HashMap::new());
}

/// GPU operations using CUDA kernels with NVRTC compilation
pub struct GpuOps;

impl GpuOps {
    /// Get or create CudaKernels instance for a device
    fn get_kernels(device: &Arc<cudarc::driver::CudaDevice>) -> Result<Arc<CudaKernels>> {
        let device_id = Arc::as_ptr(device) as usize;
        let mut cache = KERNELS_CACHE
            .lock()
            .map_err(|_| FlameError::Training("kernels cache mutex poisoned".into()))?;
        
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
        let kernels = Self::get_kernels(&a.device)?;
        if a.shape != b.shape {
            if std::env::var("FLAME_BC_TRACE").ok().map(|v| v == "1").unwrap_or(false) {
                eprintln!(
                    "[bc-trace] add lhs={:?} rhs={:?}",
                    a.shape().dims().to_vec(),
                    b.shape().dims().to_vec()
                );
            }
            return kernels.add_bc(a, b);
        }
        kernels.add(a, b)
    }
    
    /// Element-wise multiplication  
    pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&a.device)?;
        if a.shape != b.shape {
            if std::env::var("FLAME_BC_TRACE").ok().map(|v| v == "1").unwrap_or(false) {
                eprintln!(
                    "[bc-trace] mul lhs={:?} rhs={:?}",
                    a.shape().dims().to_vec(),
                    b.shape().dims().to_vec()
                );
            }
            return kernels.mul_bc(a, b);
        }
        kernels.mul(a, b)
    }
    
    /// Scalar multiplication
    pub fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.mul_scalar(tensor, scalar)
    }
    
    /// Scalar addition
    pub fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.add_scalar(tensor, scalar)
    }
    
    /// ReLU activation
    pub fn relu(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.relu(tensor)
    }
    
    /// Sigmoid activation
    pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.sigmoid(tensor)
    }
    
    /// GELU activation
    pub fn gelu(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.gelu(tensor)
    }
    
    /// SiLU activation
    pub fn silu(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.silu(tensor)
    }
    
    /// Tanh activation
    pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.tanh(tensor)
    }
    
    /// Sum reduction
    pub fn sum(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.sum(tensor)
    }
    
    /// Sum reduction along specific dimensions
    pub fn sum_dims(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.sum_dims(tensor, dims)
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
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.leaky_relu(tensor, negative_slope)
    }
    
    /// ELU
    pub fn elu(tensor: &Tensor, alpha: f32) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.elu(tensor, alpha)
    }
    
    /// PReLU
    pub fn prelu(tensor: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.prelu(tensor, weight)
    }
    
    /// Broadcast
    pub fn broadcast(tensor: &Tensor, target_shape: &crate::Shape) -> Result<Tensor> {
        CudaKernels::broadcast(tensor, target_shape)
    }
    
    /// Element-wise division
    pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&a.device)?;
        kernels.div(a, b)
    }

    /// Elementwise maximum (assumes shapes already broadcasted to equal)
    pub fn max_elemwise(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&a.device)?;
        kernels.max_elemwise(a, b)
    }

    /// NHWC image ops: resize bilinear
    pub fn resize_bilinear_nhwc(input: &Tensor, out_h: usize, out_w: usize, align_corners: bool) -> Result<Tensor> {
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
        kernels.max_dim(tensor, dim, keepdim)
    }
    
    /// Sum along dimension with keepdim
    pub fn sum_dim_keepdim(tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.sum_dim_keepdim(tensor, dim)
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
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.exp(tensor)
    }

    /// Elementwise natural logarithm
    pub fn log(tensor: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.log(tensor)
    }

    /// Index select along a dimension
    pub fn index_select(tensor: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.index_select(tensor, dim, indices)
    }

    /// Slice along multiple dimensions
    pub fn slice(tensor: &Tensor, ranges: &[(usize, usize)]) -> Result<Tensor> {
        let kernels = Self::get_kernels(&tensor.device)?;
        kernels.slice(tensor, ranges)
    }
    
    /// Matrix multiplication using cuBLAS
    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Validate shapes
        let (m, k) = match a.shape.dims() {
            [m, k] => (*m, *k),
            _ => return Err(FlameError::InvalidOperation("matmul requires 2D tensors".into())),
        };
        
        let (k2, n) = match b.shape.dims() {
            [k2, n] => (*k2, *n),
            _ => return Err(FlameError::InvalidOperation("matmul requires 2D tensors".into())),
        };
        
        if k != k2 {
            return Err(FlameError::ShapeMismatch {
                expected: Shape::from_dims(&[k, n]),
                got: b.shape.clone(),
            });
        }
        
        let out_shape = Shape::from_dims(&[m, n]);
        
        // Use cuBLAS gemm
        let blas = CudaBlas::new(a.device.clone())
            .map_err(|_| FlameError::CuBlas)?;
            
        use cudarc::cublas::{GemmConfig, Gemm};
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };
        
        // Allocate output data
        let mut output_data = crate::tensor::alloc_from_pool(&a.device, m * n)
            .map_err(|_| FlameError::CudaDriver)?;
        
        unsafe {
            blas.gemm(cfg, b.storage.try_as_slice_f32()?, a.storage.try_as_slice_f32()?, &mut output_data)
                .map_err(|_| FlameError::CuBlas)?;
        }
        
        // Create output tensor without autograd recording
        Ok(Tensor {
            storage: TensorStorage::F32 { data: output_data, numel: out_shape.elem_count() },
            shape: out_shape,
            device: a.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
        })
    }
    
    // Upsampling operations
    pub fn upsample2d_nearest(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.upsample2d_nearest(input, output_size)
    }
    
    pub fn upsample2d_bilinear(input: &Tensor, output_size: (usize, usize), align_corners: bool) -> Result<Tensor> {
        let kernels = Self::get_kernels(&input.device)?;
        kernels.upsample2d_bilinear(input, output_size, align_corners)
    }
    
    pub fn upsample2d_nearest_backward(grad_output: &Tensor, input_size: (usize, usize), output_size: (usize, usize)) -> Result<Tensor> {
        let kernels = Self::get_kernels(&grad_output.device)?;
        kernels.upsample2d_nearest_backward(grad_output, input_size, output_size)
    }
    
    pub fn upsample2d_bilinear_backward(grad_output: &Tensor, input_size: (usize, usize), output_size: (usize, usize), align_corners: bool) -> Result<Tensor> {
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
        kernels.conv_transpose2d_forward(input, weight, bias, stride, padding, output_padding, groups, dilation)
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
        kernels.conv_transpose2d_backward(grad_output, input, weight, stride, padding, output_padding, groups, dilation)
    }
}
