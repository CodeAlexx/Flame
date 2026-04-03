use crate::ops::cast::{cast_bf16_to_f32, cast_f32_to_bf16};
use crate::{DType, Error, Result, Tensor};

/// Helper to run compute in F32 and cast back to BF16 when required.
pub struct ComputeF32 {
    was_bf16: bool,
    buf: Tensor,
}

impl ComputeF32 {
    pub fn for_input(x: &Tensor) -> Result<Self> {
        match x.dtype() {
            DType::F32 => Ok(Self {
                was_bf16: false,
                buf: x.clone_result()?,
            }),
            DType::BF16 => Ok(Self {
                was_bf16: true,
                buf: cast_bf16_to_f32(x)?,
            }),
            other => Err(Error::InvalidInput(format!(
                "compute expects BF16 or F32, got {:?}",
                other
            ))),
        }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.buf
    }

    pub fn into_output(self, f32_tensor: Tensor) -> Result<Tensor> {
        if self.was_bf16 {
            cast_f32_to_bf16(&f32_tensor)
        } else {
            Ok(f32_tensor)
        }
    }
}

/// Convenience helper for unary ops.
pub fn unary<F>(x: &Tensor, f: F) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let cx = ComputeF32::for_input(x)?;
    let out = f(cx.tensor())?;
    cx.into_output(out)
}

/// Convenience helper for binary ops (same dtype on both inputs).
pub fn binary<F>(a: &Tensor, b: &Tensor, f: F) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
{
    if a.dtype() != b.dtype() {
        return Err(Error::InvalidInput(format!(
            "dtype mismatch: lhs={:?} rhs={:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    let ca = ComputeF32::for_input(a)?;
    let cb = ComputeF32::for_input(b)?;
    let out = f(ca.tensor(), cb.tensor())?;
    ca.into_output(out)
}
