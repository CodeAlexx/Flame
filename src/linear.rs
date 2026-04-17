use crate::memory_pool::MEMORY_POOL;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::staging::ArenaScratch;
use crate::strict::{scope, GuardMode};
use crate::tensor::contracts::{assert_nhwc_public, trap_is_bf16};
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Create a linear layer (convenience function)
pub fn linear(in_features: usize, out_features: usize, device: &Arc<CudaDevice>) -> Result<Linear> {
    Linear::new(in_features, out_features, true, device)
}

/// Linear (fully connected) layer
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

enum LinearInit {
    Random,
    Zeroed,
}

impl Linear {
    fn new_with_init(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Arc<CudaDevice>,
        init: LinearInit,
    ) -> Result<Self> {
        let weight_shape = Shape::from_dims(&[out_features, in_features]);
        let weight = match init {
            LinearInit::Random => {
                let bound = (6.0 / (in_features + out_features) as f32).sqrt();
                Tensor::randn(weight_shape, 0.0, bound, device.clone())?
                    .to_dtype(DType::BF16)?
                    .requires_grad_(true)
            }
            LinearInit::Zeroed => {
                Tensor::zeros_dtype(weight_shape, DType::BF16, device.clone())?.requires_grad_(true)
            }
        };
        let bias = if bias {
            Some(
                Tensor::zeros_dtype(
                    Shape::from_dims(&[out_features]),
                    DType::BF16,
                    device.clone(),
                )?
                .requires_grad_(true),
            )
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Create a new linear layer with Xavier/Glorot initialization.
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        Self::new_with_init(in_features, out_features, bias, device, LinearInit::Random)
    }

    /// Create a new linear layer with zeroed BF16 parameters (intended for checkpoint loading).
    pub fn new_zeroed(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        Self::new_with_init(in_features, out_features, bias, device, LinearInit::Zeroed)
    }

    /// Get input features
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn convert_param(reference: &Tensor, source: &Tensor, name: &str) -> Result<Tensor> {
        if reference.shape() != source.shape() {
            return Err(Error::ShapeMismatch {
                expected: reference.shape().clone(),
                got: source.shape().clone(),
            });
        }

        let mut tensor = if source.dtype() != reference.dtype() {
            source.to_dtype(reference.dtype())?
        } else if source.storage_dtype() != reference.storage_dtype() {
            source.to_dtype(reference.storage_dtype())?
        } else {
            source.clone()
        };

        if !Arc::ptr_eq(tensor.device(), reference.device()) {
            return Err(Error::InvalidInput(format!(
                "{name} expects tensor on the same device as the destination"
            )));
        }

        Ok(tensor)
    }

    /// Copy the weight tensor from an external source (shape/dtype checked).
    pub fn copy_weight_from(&mut self, source: &Tensor) -> Result<()> {
        let requires_grad = self.weight.requires_grad();
        let tensor = Self::convert_param(&self.weight, source, "Linear::copy_weight_from")?;
        self.weight = tensor.requires_grad_(requires_grad);
        Ok(())
    }

    /// Copy the bias tensor from an external source (shape/dtype checked).
    pub fn copy_bias_from(&mut self, source: &Tensor) -> Result<()> {
        let bias = self
            .bias
            .as_mut()
            .ok_or_else(|| Error::InvalidOperation("Linear has no bias parameter".into()))?;
        let requires_grad = bias.requires_grad();
        let tensor = Self::convert_param(bias, source, "Linear::copy_bias_from")?;
        *bias = tensor.requires_grad_(requires_grad);
        Ok(())
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        scope("linear.forward", GuardMode::env_default(), || {
            if input.rank() == 4 {
                assert_nhwc_public("Linear::forward in", input)?;
            }
            if input.dtype() != DType::BF16 || input.storage_dtype() != DType::BF16 {
                log::error!(
                    "Linear::forward received dtype {:?} storage {:?} (in_features={}, out_features={}) shape={:?}",
                    input.dtype(),
                    input.storage_dtype(),
                    self.in_features,
                    self.out_features,
                    input.shape().dims()
                );
                log::error!(
                    "Linear::forward weight dtype {:?} storage {:?}",
                    self.weight.dtype(),
                    self.weight.storage_dtype()
                );
            }
            trap_is_bf16("Linear::forward in", input)?;

            let input_shape = input.shape().dims();
            if input_shape[input_shape.len() - 1] != self.in_features {
                return Err(Error::ShapeMismatch {
                    expected: Shape::from_dims(&[self.in_features]),
                    got: Shape::from_dims(&[input_shape[input_shape.len() - 1]]),
                });
            }

            if let Some(arena_out) = self.try_forward_arena_fast_path(input, &input_shape)? {
                trap_is_bf16("Linear::forward out", &arena_out)?;
                if arena_out.rank() == 4 {
                    assert_nhwc_public("Linear::forward out", &arena_out)?;
                }
                return Ok(arena_out);
            }

            #[cfg(feature = "cudnn")]
            {
                if crate::cudnn::is_cudnn_linear_compatible(input, &self.weight, self.bias.as_ref())
                {
                    let mut output =
                        crate::cudnn::cudnn_linear(input, &self.weight, self.bias.as_ref())?;
                    if output.dtype() != DType::BF16 {
                        output = output.to_dtype(DType::BF16)?;
                    }

                    if (input.requires_grad() || self.weight.requires_grad()) && crate::autograd::AutogradContext::is_recording() {
                        use crate::autograd::{AutogradContext, Op};

                        let mut saved = vec![
                            (input.id(), input.clone()),
                            (self.weight.id(), self.weight.clone()),
                        ];

                        let bias_id = if let Some(bias) = &self.bias {
                            if bias.requires_grad() {
                                saved.push((bias.id(), bias.clone()));
                            }
                            Some(bias.id())
                        } else {
                            None
                        };

                        AutogradContext::record_op(
                            output.id(),
                            Op::Linear {
                                input: input.id(),
                                weight: self.weight.id(),
                                bias: bias_id,
                            },
                            saved,
                        );
                    }

                    trap_is_bf16("Linear::forward out", &output)?;
                    if output.rank() == 4 {
                        assert_nhwc_public("Linear::forward out", &output)?;
                    }

                    return Ok(output);
                }
            }

            let batch_size = input_shape[..input_shape.len() - 1]
                .iter()
                .product::<usize>();

            static CLEAR_POOL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *CLEAR_POOL.get_or_init(|| {
                std::env::var("FLAME_LINEAR_CLEAR_POOL").ok().as_deref() == Some("1")
            }) {
                MEMORY_POOL.clear_all_caches();
            }

            let input_2d = input.reshape(&[batch_size, self.in_features])?;
            trap_is_bf16("Linear::forward weight", &self.weight)?;
            let weight_2d = self
                .weight
                .reshape(&[self.out_features, self.in_features])?;

            // BF16 fast path: cuBLASLt GEMM with TRANSB=T (no materialized
            // transpose). Weight stays in row-major [out, in] and is read
            // transposed inside the GEMM. See ops::gemm_bf16::matmul_bf16_trans.
            let mut output = if input_2d.storage_dtype() == DType::BF16
                && weight_2d.storage_dtype() == DType::BF16
            {
                #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
                {
                    crate::ops::gemm_bf16::matmul_bf16_trans(
                        &input_2d, &weight_2d, false, true,
                    )?
                }
                #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
                {
                    // Without the fused path available, transpose once then matmul.
                    let weight_t = crate::bf16_elementwise::transpose2d_bf16(&weight_2d)?;
                    input_2d.matmul(&weight_t)?
                }
            } else {
                // Non-BF16 fallback: cast to F32 and materialize the transpose.
                let input_cast = if input_2d.dtype() == DType::F32 {
                    input_2d
                } else {
                    input_2d.to_dtype(DType::F32)?
                };
                let weight_cast = if weight_2d.dtype() == DType::F32 {
                    weight_2d
                } else {
                    weight_2d.to_dtype(DType::F32)?
                };
                let weight_t = weight_cast.transpose()?;
                input_cast.matmul(&weight_t)?
            };

            if let Some(bias) = &self.bias {
                if bias.dtype() != DType::BF16 || bias.storage_dtype() != DType::BF16 {
                    return Err(Error::InvalidInput(
                        "Linear::forward expects BF16 bias storage".into(),
                    ));
                }
                let bias_view = bias.reshape(&[1, self.out_features])?;
                output = output.add(&bias_view)?;
            }

            let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
            output_shape.push(self.out_features);
            let mut output = output.reshape(&output_shape)?;
            if output.dtype() != DType::BF16 {
                output = output.to_dtype(DType::BF16)?;
            }

            if (input.requires_grad() || self.weight.requires_grad()) && crate::autograd::AutogradContext::is_recording() {
                use crate::autograd::{AutogradContext, Op};

                let mut saved = vec![
                    (input.id(), input.clone()),
                    (self.weight.id(), self.weight.clone()),
                ];

                let bias_id = if let Some(bias) = &self.bias {
                    if bias.requires_grad() {
                        saved.push((bias.id(), bias.clone()));
                    }
                    Some(bias.id())
                } else {
                    None
                };

                AutogradContext::record_op(
                    output.id(),
                    Op::Linear {
                        input: input.id(),
                        weight: self.weight.id(),
                        bias: bias_id,
                    },
                    saved,
                );
            }

            trap_is_bf16("Linear::forward out", &output)?;
            if output.rank() == 4 {
                assert_nhwc_public("Linear::forward out", &output)?;
            }

            Ok(output)
        })

    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    pub fn forward_with_scratch(&self, input: &Tensor, scratch: &ArenaScratch) -> Result<Tensor> {
        let input_shape = input.shape().dims().to_vec();
        match self.forward_arena_with_scratch(input, &input_shape, scratch)? {
            Some(out) => Ok(out),
            None => Err(Error::InvalidOperation(
                "Linear::forward_with_scratch fast path unavailable for this input".into(),
            )),
        }
    }

    fn try_forward_arena_fast_path(
        &self,
        input: &Tensor,
        input_shape: &[usize],
    ) -> Result<Option<Tensor>> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            let scratch = ArenaScratch::from_tensor_with_align(input, ArenaScratch::DEFAULT_ALIGN);
            return self.forward_arena_with_scratch(input, input_shape, &scratch);
        }
        #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
        {
            let _ = (input, input_shape);
            Ok(None)
        }
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    fn forward_arena_with_scratch(
        &self,
        input: &Tensor,
        input_shape: &[usize],
        scratch: &ArenaScratch,
    ) -> Result<Option<Tensor>> {
        if input.dtype() != DType::BF16 || input.storage_dtype() != DType::BF16 {
            return Ok(None);
        }
        if input.requires_grad()
            || self.weight.requires_grad()
            || self
                .bias
                .as_ref()
                .map(|b| b.requires_grad())
                .unwrap_or(false)
        {
            return Ok(None);
        }
        if self.weight.dtype() != DType::BF16 || self.weight.storage_dtype() != DType::BF16 {
            return Ok(None);
        }
        if let Some(bias) = &self.bias {
            if bias.dtype() != DType::BF16 || bias.storage_dtype() != DType::BF16 {
                return Ok(None);
            }
        }
        let mut out_shape = input_shape[..input_shape.len() - 1].to_vec();
        out_shape.push(self.out_features);

        let mut output = match scratch.borrow_shape(Shape::from_dims(&out_shape)) {
            Ok(tensor) => tensor,
            Err(err) => {
                eprintln!(
                    "Linear::forward arena fast path borrow failed; falling back to matmul: {err}"
                );
                return Ok(None);
            }
        };
        self.forward_into_impl(input, &mut output)?;
        Ok(Some(output))
    }

    /// Forward pass that writes into a pre-allocated output tensor.
    pub fn forward_into(&self, input: &Tensor, output: &mut Tensor) -> Result<()> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            return self.forward_into_impl(input, output);
        }
        #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
        {
            let _ = (input, output);
            Err(Error::Unsupported(
                "Linear::forward_into requires cuda + bf16_u16 features".into(),
            ))
        }
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    fn forward_into_impl(&self, input: &Tensor, output: &mut Tensor) -> Result<()> {
        scope("linear.forward_into", GuardMode::env_default(), || {
            if input.rank() == 4 {
                assert_nhwc_public("Linear::forward_into in", input)?;
            } else {
                trap_is_bf16("Linear::forward_into in", input)?;
            }

            if output.rank() == 4 {
                assert_nhwc_public("Linear::forward_into out", output)?;
            } else {
                trap_is_bf16("Linear::forward_into out", output)?;
            }

            if !Arc::ptr_eq(input.device(), output.device()) {
                return Err(Error::InvalidInput(
                    "Linear::forward_into expects input/output tensors on the same device".into(),
                ));
            }

            if input.requires_grad()
                || self.weight.requires_grad()
                || self
                    .bias
                    .as_ref()
                    .map(|b| b.requires_grad())
                    .unwrap_or(false)
                || output.requires_grad()
            {
                return Err(Error::InvalidOperation(
                    "Linear::forward_into does not support autograd-enabled tensors".into(),
                ));
            }

            let input_shape = input.shape().dims();
            if input_shape[input_shape.len() - 1] != self.in_features {
                return Err(Error::ShapeMismatch {
                    expected: Shape::from_dims(&[self.in_features]),
                    got: Shape::from_dims(&[input_shape[input_shape.len() - 1]]),
                });
            }

            let mut expected = input_shape[..input_shape.len() - 1].to_vec();
            expected.push(self.out_features);
            if output.shape().dims() != expected {
                return Err(Error::ShapeMismatch {
                    expected: Shape::from_dims(&expected),
                    got: output.shape().clone(),
                });
            }

            let batch_size = input_shape[..input_shape.len() - 1]
                .iter()
                .product::<usize>();

            trap_is_bf16("Linear::forward_into weight", &self.weight)?;
            if let Some(bias) = &self.bias {
                trap_is_bf16("Linear::forward_into bias", bias)?;
            }

            // matmul_bf16_trans path: [batch, in] @ weight^T (fused TRANSB=T),
            // no materialized transpose. The cost is one GEMM-sized alloc
            // (previously amortized via `output` pre-allocation); the trans-flag
            // FFI variant of fc_gemm_bf16 is out of scope for this phase.
            let input_2d = input.reshape(&[batch_size, self.in_features])?;
            let weight_2d = self
                .weight
                .reshape(&[self.out_features, self.in_features])?;
            let mut result =
                crate::ops::gemm_bf16::matmul_bf16_trans(&input_2d, &weight_2d, false, true)?;

            if let Some(bias) = &self.bias {
                let bias_view = bias.reshape(&[1, self.out_features])?;
                result = result.add(&bias_view)?;
            }

            let result = result.reshape(&expected)?;
            output.copy_(&result)?;

            trap_is_bf16("Linear::forward_into out", output)?;
            if output.rank() == 4 {
                assert_nhwc_public("Linear::forward_into out", output)?;
            }

            Ok(())
        })
    }

    /// Get trainable parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(bias) = &self.bias {
            params.push(bias);
        }
        params
    }
}
