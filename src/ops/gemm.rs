use crate::cuda_memory_alignment::alloc_aligned_f32;
use crate::shape::Shape;
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Tensor};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use std::convert::TryFrom;
use std::sync::OnceLock;

#[cfg(not(feature = "bf16_u16"))]
use super::cast;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ComputeType {
    F32,
    BF16AccF32,
}

pub(crate) fn compute_for_storage(dt: DType) -> Result<ComputeType, Error> {
    match dt {
        DType::F32 => Ok(ComputeType::F32),
        DType::BF16 => Ok(ComputeType::BF16AccF32),
        _ => Err(Error::InvalidInput("unsupported dtype for matmul".into())),
    }
}

pub(crate) fn launch_gemm(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::InvalidInput(format!(
            "dtype mismatch in matmul: lhs={:?} rhs={:?}",
            lhs.dtype(),
            rhs.dtype()
        )));
    }

    let mode = compute_for_storage(lhs.dtype())?;
    #[cfg(feature = "dtype_trace")]
    crate::dtype_trace!(
        "gemm: compute={:?} lhs_dtype={:?} rhs_dtype={:?}",
        mode,
        lhs.dtype(),
        rhs.dtype()
    );

    match mode {
        ComputeType::F32 => gemm_f32(lhs, rhs),
        ComputeType::BF16AccF32 => gemm_bf16_acc_f32(lhs, rhs),
    }
}

pub(crate) fn launch_bmm(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::InvalidInput(format!(
            "dtype mismatch in bmm: lhs={:?} rhs={:?}",
            lhs.dtype(),
            rhs.dtype()
        )));
    }

    let mode = compute_for_storage(lhs.dtype())?;
    match mode {
        ComputeType::F32 => bmm_f32(lhs, rhs),
        ComputeType::BF16AccF32 => bmm_bf16_acc_f32(lhs, rhs),
    }
}

fn gemm_trace_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("FLAME_TRACE_VERBOSE").ok().as_deref() == Some("1"))
}

fn gemm_f32(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
    let a_shape = lhs.shape().dims();
    let b_shape = rhs.shape().dims();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        eprintln!(
            "[gemm_f32 debug] rank mismatch: lhs shape {:?}, rhs shape {:?}",
            a_shape, b_shape
        );
        return Err(Error::InvalidInput("matmul expects 2D tensors".into()));
    }
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k_rhs, n) = (b_shape[0], b_shape[1]);
    if k != k_rhs {
        return Err(Error::InvalidInput(format!(
            "matmul dimension mismatch: lhs {:?}, rhs {:?}",
            a_shape, b_shape
        )));
    }
    if m == 0 || n == 0 {
        return Tensor::zeros_dtype(Shape::from_dims(&[m, n]), lhs.dtype(), lhs.device.clone());
    }

    let blas = CudaBlas::new(lhs.device.clone()).map_err(|_| Error::CuBlas)?;
    let (m_i32, n_i32, k_i32) = (
        i32::try_from(m).map_err(|_| Error::InvalidInput("matrix dimension too large".into()))?,
        i32::try_from(n).map_err(|_| Error::InvalidInput("matrix dimension too large".into()))?,
        i32::try_from(k).map_err(|_| Error::InvalidInput("matrix dimension too large".into()))?,
    );
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n_i32,
        n: m_i32,
        k: k_i32,
        alpha: 1.0f32,
        lda: n_i32,
        ldb: k_i32,
        beta: 0.0f32,
        ldc: n_i32,
    };

    if gemm_trace_enabled() {
        eprintln!(
            "[gemm_f32] lhs dtype {:?} storage {:?} shape {:?}; rhs dtype {:?} storage {:?} shape {:?}",
            lhs.dtype(),
            lhs.storage_dtype(),
            lhs.shape().dims(),
            rhs.dtype(),
            rhs.storage_dtype(),
            rhs.shape().dims()
        );
    }
    let mut out = alloc_aligned_f32(&lhs.device, m * n)?;
    let a = lhs.storage.try_as_slice_f32()?;
    let b = rhs.storage.try_as_slice_f32()?;
    unsafe {
        blas.gemm(cfg, b, a, &mut out).map_err(|_| Error::CuBlas)?;
    }

    Ok(Tensor {
        storage: TensorStorage::F32 {
            data: out.into(),
            numel: m * n,
        },
        shape: Shape::from_dims(&[m, n]),
        device: lhs.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    })
}

fn gemm_bf16_acc_f32(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
    let a_shape = lhs.shape().dims();
    let b_shape = rhs.shape().dims();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(Error::InvalidInput("matmul expects 2D tensors".into()));
    }
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k_rhs, n) = (b_shape[0], b_shape[1]);
    if k != k_rhs {
        return Err(Error::InvalidInput(format!(
            "matmul dimension mismatch: lhs {:?}, rhs {:?}",
            a_shape, b_shape
        )));
    }
    if m == 0 || n == 0 {
        return Tensor::zeros_dtype(Shape::from_dims(&[m, n]), lhs.dtype(), lhs.device.clone());
    }

    #[cfg(feature = "bf16_u16")]
    {
        crate::cuda_ops_bf16::gemm_bf16(lhs, rhs, None)
    }

    #[cfg(not(feature = "bf16_u16"))]
    {
        let lhs_f32 = cast::cast_bf16_to_f32(lhs)?;
        let rhs_f32 = cast::cast_bf16_to_f32(rhs)?;
        let y_f32 = gemm_f32(&lhs_f32, &rhs_f32)?;
        cast::cast_f32_to_bf16(&y_f32)
    }
}

fn bmm_f32(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    if lhs_dims.len() != 3 || rhs_dims.len() != 3 {
        return Err(Error::InvalidInput("bmm expects 3D tensors".into()));
    }
    let (batch, m, k) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, k_rhs, n) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);
    if batch != batch_rhs || k != k_rhs {
        return Err(Error::InvalidInput(format!(
            "bmm shape mismatch lhs {:?}, rhs {:?}",
            lhs_dims, rhs_dims
        )));
    }

    let mut out = Tensor::zeros_dtype(
        Shape::from_dims(&[batch, m, n]),
        lhs.dtype(),
        lhs.device.clone(),
    )?;
    launch_gemm_strided_batched(lhs, rhs, &mut out)?;
    Ok(out)
}

fn bmm_bf16_acc_f32(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    if lhs_dims.len() != 3 || rhs_dims.len() != 3 {
        return Err(Error::InvalidInput("bmm expects 3D tensors".into()));
    }
    let (batch, m, k) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, k_rhs, n) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);
    if batch != batch_rhs || k != k_rhs {
        return Err(Error::InvalidInput(format!(
            "bmm shape mismatch lhs {:?}, rhs {:?}",
            lhs_dims, rhs_dims
        )));
    }
    if m == 0 || n == 0 {
        return Tensor::zeros_dtype(Shape::from_dims(&[batch, m, n]), DType::BF16, lhs.device.clone());
    }

    let mut out = Tensor::zeros_dtype(
        Shape::from_dims(&[batch, m, n]),
        DType::BF16,
        lhs.device.clone(),
    )?;
    crate::ops::gemm_bf16::bmm_bf16_fp32acc_out(lhs, rhs, &mut out, false, false)?;
    Ok(out)
}

pub(crate) fn launch_gemm_strided_batched(
    lhs: &Tensor,
    rhs: &Tensor,
    out: &mut Tensor,
) -> Result<(), Error> {
    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 || out.dtype() != DType::F32 {
        return Err(Error::InvalidInput(
            "launch_gemm_strided_batched requires F32 tensors".into(),
        ));
    }

    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    let out_dims = out.shape().dims();

    if lhs_dims.len() != 3 || rhs_dims.len() != 3 || out_dims.len() != 3 {
        return Err(Error::InvalidInput(
            "launch_gemm_strided_batched expects 3D tensors".into(),
        ));
    }

    let (batch, m, k) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, k_rhs, n) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);
    if batch != batch_rhs || k != k_rhs {
        return Err(Error::InvalidInput(format!(
            "launch_gemm_strided_batched shape mismatch lhs {:?}, rhs {:?}",
            lhs_dims, rhs_dims
        )));
    }
    if out_dims != [batch, m, n] {
        return Err(Error::InvalidInput(format!(
            "launch_gemm_strided_batched output shape mismatch {:?} vs expected [{},{},{}]",
            out_dims, batch, m, n
        )));
    }

    if batch == 0 || m == 0 || n == 0 {
        return Ok(());
    }

    let blas = CudaBlas::new(lhs.device.clone()).map_err(|_| Error::CuBlas)?;
    let (m_i32, n_i32, k_i32) = (
        i32::try_from(m).map_err(|_| Error::InvalidInput("matrix dimension too large".into()))?,
        i32::try_from(n).map_err(|_| Error::InvalidInput("matrix dimension too large".into()))?,
        i32::try_from(k).map_err(|_| Error::InvalidInput("matrix dimension too large".into()))?,
    );
    let batch_i32 = i32::try_from(batch)
        .map_err(|_| Error::InvalidInput("batch dimension too large".into()))?;

    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n_i32,
            n: m_i32,
            k: k_i32,
            alpha: 1.0f32,
            lda: n_i32,
            ldb: k_i32,
            beta: 0.0f32,
            ldc: n_i32,
        },
        batch_size: batch_i32,
        stride_a: (k * n) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    let a = rhs.storage.try_as_slice_f32()?;
    let b = lhs.storage.try_as_slice_f32()?;
    let c = out.storage_mut().try_as_mut_slice_f32()?;

    unsafe {
        blas.gemm_strided_batched(cfg, a, b, c)
            .map_err(|_| Error::CuBlas)?;
    }

    Ok(())
}
