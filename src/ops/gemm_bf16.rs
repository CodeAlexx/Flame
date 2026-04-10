#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::DevicePtr;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda::device_lt;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::DType;
use crate::{Error, Result, Tensor};

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn bmm_bf16_fp32acc_out(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    trans_a: bool,
    trans_b: bool,
) -> Result<()> {
    if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 || out.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "bmm_bf16_fp32acc_out: tensors must be BF16".into(),
        ));
    }

    let ashp = a.shape().dims();
    let bshp = b.shape().dims();
    let oshp = out.shape().dims();
    if ashp.len() != 3 || bshp.len() != 3 || oshp.len() != 3 {
        return Err(Error::InvalidInput(
            "bmm_bf16_fp32acc_out: expect 3D tensors".into(),
        ));
    }
    if bshp[0] != ashp[0]
        || oshp[0] != ashp[0]
        || oshp[1] != ashp[1]
        || bshp[1] != ashp[2]
        || oshp[2] != bshp[2]
    {
        return Err(Error::InvalidInput(
            "bmm_bf16_fp32acc_out: shape mismatch".into(),
        ));
    }

    let batch = ashp[0] as i32;
    let m = ashp[1] as i32;
    let k = ashp[2] as i32;
    let n = bshp[2] as i32;

    let device = a.device();
    let stream = device_lt::stream_ptr(device)?;
    let lt = device_lt::cublaslt_handle_ptr(device)?;

    let lda: i64 = if trans_a { m } else { k } as i64;
    let ldb: i64 = if trans_b { k } else { n } as i64;
    let ldc: i64 = n as i64;
    let stride_a = (m * k) as i64;
    let stride_b = (k * n) as i64;
    let stride_c = (m * n) as i64;

    let op_a = if trans_a { 1 } else { 0 };
    let op_b = if trans_b { 1 } else { 0 };

    let a_ptr = a.as_device_ptr_bf16("bmm_bf16_fp32acc_out:a")?;
    let b_ptr = b.as_device_ptr_bf16("bmm_bf16_fp32acc_out:b")?;
    let out_ptr = out.as_mut_device_ptr_bf16("bmm_bf16_fp32acc_out:out")?;

    let status = unsafe {
        crate::cuda::ffi::gemm_bf16_fp32acc_stridedBatched(
            lt,
            op_a,
            op_b,
            m,
            n,
            k,
            a_ptr as *const core::ffi::c_void,
            lda,
            stride_a,
            b_ptr as *const core::ffi::c_void,
            ldb,
            stride_b,
            out_ptr as *mut core::ffi::c_void,
            ldc,
            stride_c,
            batch,
            1.0,
            0.0,
            stream,
        )
    };

    if status != 0 {
        return Err(Error::Cuda(format!("cuBLASLt gemm bf16 status {}", status)));
    }
    Ok(())
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn bmm_bf16_fp32acc_out(
    _a: &Tensor,
    _b: &Tensor,
    _out: &mut Tensor,
    _trans_a: bool,
    _trans_b: bool,
) -> Result<()> {
    Err(Error::Unsupported(
        "bmm_bf16_fp32acc_out requires the `cuda` and `bf16_u16` features".into(),
    ))
}

/// BF16 matmul with configurable trans_a/trans_b flags that executes in a
/// single cuBLASLt call without materializing any transposes.
/// Accepts 2D `[m_a, k_a]` or 3D `[batch, m_a, k_a]` inputs.
///
/// Logical output shape:
///   non-trans:             `[M_out, N_out] = [a.dim0, b.dim1]`, inner dim `a.dim1 == b.dim0`
///   trans_a only:          `[a.dim1, b.dim1]`, inner `a.dim0 == b.dim0`
///   trans_b only:          `[a.dim0, b.dim0]`, inner `a.dim1 == b.dim1`
///   both:                  `[a.dim1, b.dim0]`, inner `a.dim0 == b.dim1`
///
/// Primary fast path for MatMul / Linear backward, where we need either
/// `dC @ B^T` or `A^T @ dC`. Old path materialized those transposes via
/// `transpose2d_bf16` (full BF16 memcpy per call). Fused path is 2-3×
/// faster on Klein 9B backward MatMul.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn matmul_bf16_trans(
    a: &Tensor,
    b: &Tensor,
    trans_a: bool,
    trans_b: bool,
) -> Result<Tensor> {
    use crate::Shape;

    if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "matmul_bf16_trans: both inputs must be BF16".into(),
        ));
    }

    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    // Normalize to 3D [batch, d0, d1] by prepending a batch=1 dim.
    let (a3, b3, rank) = if a_dims.len() == 2 && b_dims.len() == 2 {
        (
            a.reshape(&[1, a_dims[0], a_dims[1]])?,
            b.reshape(&[1, b_dims[0], b_dims[1]])?,
            2usize,
        )
    } else if a_dims.len() == 3 && b_dims.len() == 3 {
        (a.clone(), b.clone(), 3usize)
    } else {
        return Err(Error::InvalidInput(format!(
            "matmul_bf16_trans: expect 2D×2D or 3D×3D, got {:?}×{:?}",
            a_dims, b_dims
        )));
    };

    let a3_dims = a3.shape().dims();
    let b3_dims = b3.shape().dims();
    if a3_dims[0] != b3_dims[0] {
        return Err(Error::InvalidInput(format!(
            "matmul_bf16_trans: batch mismatch {} vs {}",
            a3_dims[0], b3_dims[0]
        )));
    }
    let batch = a3_dims[0];

    // Logical op(A) shape = [m_eff, k_eff], op(B) shape = [k_eff, n_eff]
    let (m_eff, k_a) = if trans_a {
        (a3_dims[2], a3_dims[1])
    } else {
        (a3_dims[1], a3_dims[2])
    };
    let (k_b, n_eff) = if trans_b {
        (b3_dims[2], b3_dims[1])
    } else {
        (b3_dims[1], b3_dims[2])
    };
    if k_a != k_b {
        return Err(Error::InvalidInput(format!(
            "matmul_bf16_trans: inner dim mismatch {} vs {} (shapes {:?}×{:?}, trans_a={}, trans_b={})",
            k_a, k_b, a_dims, b_dims, trans_a, trans_b
        )));
    }

    // The C-side `gemm_bf16_fp32acc_stridedBatched` uses CUBLASLT_ORDER_ROW,
    // so we call it with native row-major conventions. The leading dimension
    // is always the physical row-major row stride (= cols) of the underlying
    // memory, regardless of the trans flag — the flag only changes how
    // cuBLASLt interprets the logical dims, not the memory stride.
    let mut out = Tensor::zeros_dtype(
        Shape::from_dims(&[batch, m_eff, n_eff]),
        DType::BF16,
        a.device().clone(),
    )?;

    let stream = crate::cuda::device_lt::stream_ptr(a.device())?;
    let lt = crate::cuda::device_lt::cublaslt_handle_ptr(a.device())?;

    // Row-major leading dims = physical innermost dim of each contiguous tensor
    let lda = a3_dims[2] as i64;
    let ldb = b3_dims[2] as i64;
    let ldc = n_eff as i64;

    let stride_a = (a3_dims[1] * a3_dims[2]) as i64;
    let stride_b = (b3_dims[1] * b3_dims[2]) as i64;
    let stride_c = (m_eff * n_eff) as i64;

    let op_a: i32 = if trans_a { 1 } else { 0 }; // cublasOperation_t: N=0, T=1
    let op_b: i32 = if trans_b { 1 } else { 0 };

    let a_ptr = a3.as_device_ptr_bf16("matmul_bf16_trans:a")?;
    let b_ptr = b3.as_device_ptr_bf16("matmul_bf16_trans:b")?;
    let out_ptr = out.as_mut_device_ptr_bf16("matmul_bf16_trans:out")?;

    let status = unsafe {
        crate::cuda::ffi::gemm_bf16_fp32acc_stridedBatched(
            lt,
            op_a,
            op_b,
            m_eff as i32, // m = rows of op(A) = rows of C
            n_eff as i32, // n = cols of op(B) = cols of C
            k_a as i32,   // k = cols of op(A) = rows of op(B)
            a_ptr as *const core::ffi::c_void,
            lda,
            stride_a,
            b_ptr as *const core::ffi::c_void,
            ldb,
            stride_b,
            out_ptr as *mut core::ffi::c_void,
            ldc,
            stride_c,
            batch as i32,
            1.0,
            0.0,
            stream,
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "matmul_bf16_trans: cuBLASLt status {} (m={}, n={}, k={}, trans_a={}, trans_b={}, a_shape={:?}, b_shape={:?})",
            status, m_eff, n_eff, k_a, trans_a, trans_b, a3_dims, b3_dims
        )));
    }

    if rank == 2 {
        out.reshape(&[m_eff, n_eff])
    } else {
        Ok(out)
    }
}

#[cfg(any(not(feature = "cuda"), not(feature = "bf16_u16")))]
pub fn matmul_bf16_trans(
    _a: &Tensor,
    _b: &Tensor,
    _trans_a: bool,
    _trans_b: bool,
) -> Result<Tensor> {
    Err(Error::Unsupported(
        "matmul_bf16_trans requires the `cuda` and `bf16_u16` features".into(),
    ))
}
