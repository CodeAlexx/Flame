use anyhow::{bail, Result};

use crate::{tensor_storage::TensorStorage, Tensor};

#[inline]
pub fn assert_cuda(tag: &str, t: &Tensor) -> Result<()> {
    let is_cuda = matches!(
        &t.storage,
        TensorStorage::F32 { .. }
            | TensorStorage::F16 { .. }
            | TensorStorage::BF16 { .. }
            | TensorStorage::BF16Arena { .. }
            | TensorStorage::I8 { .. }
            | TensorStorage::I32 { .. }
            | TensorStorage::Bool { .. }
    );

    if is_cuda {
        Ok(())
    } else {
        bail!("CPU tensor at {tag}: {:?}", t.shape())
    }
}

#[inline]
#[allow(dead_code)]
pub fn log_device(tag: &str, t: &Tensor) {
    let dev = if matches!(
        &t.storage,
        TensorStorage::F32 { .. }
            | TensorStorage::F16 { .. }
            | TensorStorage::BF16 { .. }
            | TensorStorage::BF16Arena { .. }
            | TensorStorage::I8 { .. }
            | TensorStorage::I32 { .. }
            | TensorStorage::Bool { .. }
    ) {
        "CUDA"
    } else {
        "CPU"
    };
    eprintln!(
        "[dev] {tag}: {dev} shape={:?} dtype={:?}",
        t.shape().dims(),
        t.dtype()
    );
}
