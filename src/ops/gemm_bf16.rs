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
