//! cuBLASLt BF16 matmul helpers (row-major, FP32 accumulate).

#[cfg(feature = "bf16_u16")]
use crate::DType;
use crate::{Error, Tensor};
#[cfg(feature = "bf16_u16")]
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
#[cfg(not(feature = "bf16_u16"))]
use cudarc::driver::CudaDevice;
#[cfg(feature = "bf16_u16")]
use cudarc::driver::{sys::CUdevice_attribute, CudaDevice, CudaView, CudaViewMut};
#[cfg(feature = "bf16_u16")]
use half::bf16;
#[cfg(feature = "bf16_u16")]
use std::convert::TryFrom;
use std::sync::Arc;

/// Perform BF16 matmul (row-major) with FP32 accumulation using cuBLASLt.
pub fn matmul_bf16_acc_f32_rowmajor(
    dev: &Arc<CudaDevice>,
    lhs: &Tensor,
    rhs: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, Error> {
    #[cfg(not(feature = "bf16_u16"))]
    {
        let _ = (dev, lhs, rhs, m, n, k);
        Err(Error::Unsupported(
            "bf16_u16 feature required for cuBLASLt BF16".into(),
        ))
    }

    #[cfg(feature = "bf16_u16")]
    {
        let major =
            dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        if major < 8 {
            return Err(Error::Unsupported("BF16 requires SM80+ (Ampere)".into()));
        }

        if lhs.dtype() != DType::BF16 || rhs.dtype() != DType::BF16 {
            return Err(Error::InvalidInput("Lt path expects BF16 inputs".into()));
        }

        let same_device = Arc::ptr_eq(lhs.device(), rhs.device())
            || lhs.device().ordinal() == rhs.device().ordinal();
        if !same_device {
            return Err(Error::InvalidInput(
                "matmul tensors must share device".into(),
            ));
        }

        // Prepare Lt handle (keeps device alive, launches on default stream)
        let lt = CudaBlasLT::new(dev.clone())
            .map_err(|err| Error::KernelError(format!("cublasLt handle: {err:?}")))?;

        // Allocate BF16 output buffer on device
        let mut output =
            Tensor::zeros_dtype(crate::Shape::from_dims(&[m, n]), DType::BF16, dev.clone())?;

        use cudarc::driver::DeviceSlice;

        // Obtain raw storage slices (u16) and reinterpret as bf16 views
        let a_slice = lhs
            .storage_ref()
            .try_as_slice_u16()
            .map_err(|_| Error::InvalidInput("Lt expects BF16(u16) storage".into()))?;
        let b_slice = rhs
            .storage_ref()
            .try_as_slice_u16()
            .map_err(|_| Error::InvalidInput("Lt expects BF16(u16) storage".into()))?;
        let d_slice = output
            .storage_mut()
            .try_as_mut_slice_u16()
            .map_err(|_| Error::InvalidOperation("output not BF16".into()))?;

        let a_view: CudaView<'_, bf16> = unsafe {
            a_slice
                .transmute::<bf16>(a_slice.len())
                .ok_or_else(|| Error::InvalidOperation("failed to transmute BF16 view".into()))?
        };
        let b_view: CudaView<'_, bf16> = unsafe {
            b_slice
                .transmute::<bf16>(b_slice.len())
                .ok_or_else(|| Error::InvalidOperation("failed to transmute BF16 view".into()))?
        };
        let mut d_view: CudaViewMut<'_, bf16> = unsafe {
            d_slice
                .transmute_mut::<bf16>(d_slice.len())
                .ok_or_else(|| Error::InvalidOperation("failed to transmute BF16 view".into()))?
        };

        // Configure matmul (row-major)
        let cfg = MatmulConfig {
            transa: false,
            transb: false,
            m: u64::try_from(n).map_err(|_| Error::InvalidInput("m too large".into()))?,
            n: u64::try_from(m).map_err(|_| Error::InvalidInput("n too large".into()))?,
            k: u64::try_from(k).map_err(|_| Error::InvalidInput("k too large".into()))?,
            alpha: 1.0,
            lda: i64::try_from(n).map_err(|_| Error::InvalidInput("lda overflow".into()))?,
            ldb: i64::try_from(k).map_err(|_| Error::InvalidInput("ldb overflow".into()))?,
            beta: 0.0,
            ldc: i64::try_from(n).map_err(|_| Error::InvalidInput("ldc overflow".into()))?,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };

        unsafe {
            lt.matmul(cfg, &b_view, &a_view, &mut d_view, None, None)
                .map_err(|err| Error::KernelError(format!("cublasLt matmul: {err:?}")))?;
        }

        Ok(output)
    }
}
