#![allow(unused_variables, dead_code)]
// Legacy attention module awaiting Phase 3 modernization.

use super::rope::apply_rope;
use crate::linear::Linear;
use crate::{DType, Error, Result, Shape, Tensor};
use std::sync::Arc;

fn build_causal_mask(q_seq: usize, kv_seq: usize, tensor: &Tensor) -> Result<Tensor> {
    let device = tensor.device().clone();

    let elements = q_seq * kv_seq;
    let mut data = vec![0.0f32; elements];
    for q in 0..q_seq {
        let row = &mut data[q * kv_seq..(q + 1) * kv_seq];
        if kv_seq == 0 {
            continue;
        }
        let cutoff = q.min(kv_seq.saturating_sub(1));
        for k in 0..=cutoff {
            row[k] = 1.0;
        }
    }

    let mask = Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[1, 1, q_seq, kv_seq]),
        device,
        DType::F32,
    )?;
    Ok(mask)
}

fn pack_heads(tensor: &Tensor, num_heads: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "pack_heads expects [B,S,D], got {:?}",
            dims
        )));
    }

    let (batch, seq, embed) = (dims[0], dims[1], dims[2]);
    if embed % num_heads != 0 {
        return Err(Error::InvalidInput(format!(
            "embed dim {} not divisible by heads {}",
            embed, num_heads
        )));
    }

    let head_dim = embed / num_heads;
    let view = tensor.reshape(&[batch, seq, num_heads, head_dim])?;
    let permuted = view.permute(&[0, 2, 1, 3])?;
    permuted.clone_result()
}

fn unpack_heads(tensor: &Tensor, num_heads: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "unpack_heads expects [B,H,S,Dh], got {:?}",
            dims
        )));
    }

    let (batch, heads, seq, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
    if heads != num_heads {
        return Err(Error::InvalidInput(format!(
            "expected {} heads, got {}",
            num_heads, heads
        )));
    }

    let view = tensor.permute(&[0, 2, 1, 3])?;
    let reshaped = view.reshape(&[batch, seq, heads * head_dim])?;
    reshaped.clone_result()
}

/// Multi-head attention configuration
#[derive(Clone)]
pub struct AttentionConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout: f32,
    pub bias: bool,
}

impl AttentionConfig {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim must be divisible by num_heads"
        );
        Self {
            embed_dim,
            num_heads,
            head_dim: embed_dim / num_heads,
            dropout: 0.0,
            bias: true,
        }
    }
}

/// Multi-head attention layer
pub struct MultiHeadAttention {
    pub config: AttentionConfig,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(config: AttentionConfig, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let q_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, &device)?;
        let k_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, &device)?;
        let v_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, &device)?;
        let out_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, &device)?;

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    /// Forward pass for multi-head attention
    /// query: [batch, seq_len, embed_dim]
    /// key: [batch, seq_len, embed_dim]
    /// value: [batch, seq_len, embed_dim]
    /// mask: Optional [batch, seq_len, seq_len] or [seq_len, seq_len]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if query.dtype() != DType::BF16
            || key.dtype() != DType::BF16
            || value.dtype() != DType::BF16
        {
            // 3D attention tensors: rank guard skipped; BF16 storage enforced on inputs.
            return Err(Error::InvalidInput(
                "MultiHeadAttention::forward expects BF16 tensors at the public boundary".into(),
            ));
        }

        let query_shape = query.shape().dims();
        if query_shape.len() != 3 {
            return Err(Error::InvalidOperation(format!(
                "Expected 3D query tensor, got {:?}",
                query_shape
            )));
        }

        let batch_size = query_shape[0];
        let seq_len = query_shape[1];
        let embed_dim = query_shape[2];

        if embed_dim != self.config.embed_dim {
            return Err(Error::InvalidOperation(format!(
                "Expected embed_dim {}, got {}",
                self.config.embed_dim, embed_dim
            )));
        }

        // Project Q, K, V
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        // Get key/value sequence length (may differ from query for cross-attention)
        let kv_seq_len = key.shape().dims()[1];

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape(&[
            batch_size,
            seq_len,
            self.config.num_heads,
            self.config.head_dim,
        ])?;
        let k = k.reshape(&[
            batch_size,
            kv_seq_len,
            self.config.num_heads,
            self.config.head_dim,
        ])?;
        let v = v.reshape(&[
            batch_size,
            kv_seq_len,
            self.config.num_heads,
            self.config.head_dim,
        ])?;

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let scale = (self.config.head_dim as f32).sqrt();
        let scores = self.scaled_dot_product_attention(&q, &k, &v, mask, scale)?;

        // Transpose back to [batch, seq_len, num_heads, head_dim]
        let scores = scores.permute(&[0, 2, 1, 3])?;

        // Reshape to [batch, seq_len, embed_dim]
        let scores = scores.reshape(&[batch_size, seq_len, self.config.embed_dim])?;

        // Output projection
        let mut output = self.out_proj.forward(&scores)?;
        if output.dtype() != DType::BF16 {
            output = output.to_dtype(DType::BF16)?;
        }
        Ok(output)
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        attention_impl(q, k, v, mask, false, Some(1.0 / scale))
    }

    /// Extract a single head from the attention tensor
    fn extract_head(
        &self,
        tensor: &Tensor,
        batch: usize,
        head: usize,
        dim1: usize,
        dim2: usize,
    ) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "extract_head expects [B,H,S,D], got {:?}",
                dims
            )));
        }
        if batch >= dims[0] || head >= dims[1] {
            return Err(Error::InvalidInput(format!(
                "extract_head index out of bounds: batch {} head {} dims {:?}",
                batch, head, dims
            )));
        }
        if dims[2] != dim1 || dims[3] != dim2 {
            return Err(Error::InvalidInput(format!(
                "extract_head dim mismatch: expected ({},{}), got ({},{})",
                dims[2], dims[3], dim1, dim2
            )));
        }

        let view = tensor
            .narrow(0, batch, 1)?
            .narrow(1, head, 1)?
            .reshape(&[dim1, dim2])?;
        view.clone_result()
    }

    /// Apply attention mask
    fn apply_attention_mask(&self, scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let target_shape = scores.shape().clone();
        let expanded = if mask.shape().dims() == target_shape.dims() {
            mask.clone_result()?
        } else {
            mask.broadcast_to(&target_shape)?
        };

        let mask_f32 = if expanded.dtype() == DType::F32 {
            expanded
        } else {
            expanded.to_dtype(DType::F32)?
        };

        let ones = Tensor::full(mask_f32.shape().clone(), 1.0f32, mask_f32.device().clone())?;
        let complement = ones.sub(&mask_f32)?;
        let penalty = complement.mul_scalar(-1.0e9)?;
        let penalty = if penalty.dtype() == scores.dtype() {
            penalty
        } else {
            penalty.to_dtype(scores.dtype())?
        };

        scores.add(&penalty)
    }

    /// Softmax along the last dimension
    fn softmax(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        tensor.softmax(dim as isize)
    }
}

pub fn attention_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: Option<f32>,
) -> Result<Tensor> {
    let effective_mask = match (mask, causal) {
        (Some(_), true) => {
            return Err(Error::Unsupported(
                "Combined mask + causal attention is not yet supported".into(),
            ));
        }
        (Some(m), false) => Some(m.clone_result()?),
        (None, true) => Some(build_causal_mask(
            q.shape().dims()[2],
            k.shape().dims()[2],
            q,
        )?),
        (None, false) => None,
    };

    let embed = q.shape().dims();
    if embed.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "Expected [B,H,T,D] inputs for SDPA attention, got {:?}",
            embed
        )));
    }
    let head_dim = embed[3] as f32;
    let default_scale = 1.0 / head_dim.sqrt();
    let target_scale = scale.unwrap_or(default_scale);

    let scale_factor = if default_scale == 0.0 {
        1.0
    } else {
        target_scale / default_scale
    };

    let q_scaled = if (scale_factor - 1.0).abs() < f32::EPSILON {
        q.clone_result()?
    } else {
        q.mul_scalar(scale_factor)?
    };

    // `crate::sdpa::forward` already performs the FP32 accumulate + BF16 convert
    // dance we rely on elsewhere, so lean on it for the actual computation.
    let output =
        crate::sdpa::forward(&q_scaled, k, v, effective_mask.as_ref()).map_err(|err| err)?;

    Ok(output)
}

// --- Stable SDPA/FlashAttention helpers (public) ---

/// Validate Q/K/V basic shape relationships. For brevity, only dimensionality is checked here.
fn validate_qkv(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<()> {
    let qd = q.shape().dims();
    let kd = k.shape().dims();
    let vd = v.shape().dims();
    if !(qd.len() == 4 && kd.len() == 4 && vd.len() == 4) {
        return Err(Error::InvalidInput("Q,K,V must be [B,H,SEQ,D]".into()));
    }
    let (b, h, q_len, dq) = (qd[0], qd[1], qd[2], qd[3]);
    let (bk, hk, k_len, dk) = (kd[0], kd[1], kd[2], kd[3]);
    let (bv, hv, kv, dv) = (vd[0], vd[1], vd[2], vd[3]);
    if !(b == bk && b == bv && h == hk && h == hv && dq == dk && dk == dv && k_len == kv) {
        return Err(Error::InvalidInput("Q,K,V dimension mismatch".into()));
    }
    Ok(())
}

fn validate_qkv_shapes(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<(usize, usize, usize, usize, usize)> {
    validate_qkv(q, k, v)?;
    let qd = q.shape().dims();
    Ok((qd[0], qd[1], qd[2], k.shape().dims()[2], qd[3]))
}

fn validate_mask_shape(
    mask: &Tensor,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
) -> Result<()> {
    let md = mask.shape().dims();
    if md.len() != 4 {
        return Err(Error::InvalidInput(
            "Mask must be 4D: [B,H,Q,K] / [B,1,Q,K] / [1,1,Q,K]".into(),
        ));
    }
    let (mb, mh, mq, mk) = (md[0], md[1], md[2], md[3]);
    let ok_b = mb == b || mb == 1;
    let ok_h = mh == h || mh == 1;
    let ok_q = mq == q_len;
    let ok_k = mk == k_len;
    if !(ok_b && ok_h && ok_q && ok_k) {
        return Err(Error::InvalidInput(
            "Mask dims must broadcast to [B,H,Q,K]".into(),
        ));
    }
    match mask.dtype() {
        DType::Bool | DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        _ => Err(Error::InvalidInput(
            "Mask dtype must be bool or float (additive)".into(),
        )),
    }
}

/// Scaled dot-product attention with FP32 softmax/reductions. Accepts [B,H,S,D] tensors.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        // Attention tensors use [B,H,S,D]; enforce BF16 storage at the boundary without NHWC layout.
        return Err(Error::InvalidInput(
            "sdpa expects BF16 tensors at the public boundary".into(),
        ));
    }

    let mut output = crate::sdpa::forward(q, k, v, mask)?;
    if output.dtype() != DType::BF16 {
        output = output.to_dtype(DType::BF16)?;
    }
    Ok(output)
}

/// Generic attend: prefer FlashAttention when feature enabled, else SDPA.
pub fn attend(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    sdpa(q, k, v, mask)
}

/// GeGLU activation for transformer FFN
pub struct GeGLU {
    pub proj: Linear,
}

impl GeGLU {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Project to 2x dim_out for gating
        let proj = Linear::new(dim_in, dim_out * 2, true, &device)?;
        Ok(Self { proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Project and split
        let projected = self.proj.forward(x)?;
        let dims = projected.shape().dims();
        let last_dim = dims[dims.len() - 1];

        // Split into two halves
        let half_dim = last_dim / 2;
        let value = projected
            .narrow(dims.len() - 1, 0, half_dim)?
            .clone_result()?;
        let gate = projected.narrow(dims.len() - 1, half_dim, half_dim)?;
        let gate = gate.gelu()?;
        value.mul(&gate)
    }
}

/// GELU activation function
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Feed-forward network with GeGLU activation
pub struct FeedForward {
    pub project_in: GeGLU,
    pub linear: Linear,
}

impl FeedForward {
    pub fn new(
        dim: usize,
        dim_out: Option<usize>,
        mult: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);

        let project_in = GeGLU::new(dim, inner_dim, device.clone())?;
        let linear = Linear::new(inner_dim, dim_out, true, &device)?;

        Ok(Self { project_in, linear })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.project_in.forward(x)?;
        self.linear.forward(&x)
    }
}

/// Rotary Position Embedding (RoPE)
pub struct RotaryEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
    pub theta: f32,
    pub freqs_cos: Option<Tensor>,
    pub freqs_sin: Option<Tensor>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self {
            dim,
            max_seq_len,
            theta,
            freqs_cos: None,
            freqs_sin: None,
        }
    }

    /// Initialize frequency tensors
    pub fn init_freqs(&mut self, device: Arc<cudarc::driver::CudaDevice>) -> Result<()> {
        if self.dim % 2 != 0 {
            return Err(Error::InvalidOperation(format!(
                "RoPE dim must be even, got {}",
                self.dim
            )));
        }

        let half_dim = self.dim / 2;
        let mut freqs = vec![0.0f32; half_dim];

        for (i, freq) in freqs.iter_mut().enumerate() {
            *freq = 1.0 / self.theta.powf(2.0 * i as f32 / self.dim as f32);
        }

        // Create position indices
        let mut cos_data = Vec::new();
        let mut sin_data = Vec::new();

        for pos in 0..self.max_seq_len {
            for &freq in freqs.iter() {
                let angle = pos as f32 * freq;
                let (sin_angle, cos_angle) = angle.sin_cos();
                cos_data.extend([cos_angle, cos_angle]);
                sin_data.extend([sin_angle, sin_angle]);
            }
        }

        self.freqs_cos = Some(Tensor::from_vec(
            cos_data,
            Shape::from_dims(&[self.max_seq_len, self.dim]),
            device.clone(),
        )?);

        self.freqs_sin = Some(Tensor::from_vec(
            sin_data,
            Shape::from_dims(&[self.max_seq_len, self.dim]),
            device,
        )?);

        Ok(())
    }

    /// Apply rotary embeddings to query or key tensors
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let _ = self
            .freqs_cos
            .as_ref()
            .ok_or(Error::InvalidOperation("RoPE not initialized".into()))?;
        let _ = self
            .freqs_sin
            .as_ref()
            .ok_or(Error::InvalidOperation("RoPE not initialized".into()))?;

        if seq_len > self.max_seq_len {
            return Err(Error::InvalidInput(format!(
                "RoPE seq_len {} exceeds max_seq_len {}",
                seq_len, self.max_seq_len
            )));
        }

        let dims = x.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "RoPE expects [B,H,S,Dh] tensor, got {:?}",
                dims
            )));
        }

        let (_, _, actual_seq_len, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
        if head_dim != self.dim {
            return Err(Error::InvalidOperation(format!(
                "Expected head_dim {}, got {}",
                self.dim, head_dim
            )));
        }

        if seq_len != actual_seq_len {
            return Err(Error::InvalidInput(format!(
                "RoPE seq_len {} must match tensor sequence {}",
                seq_len, actual_seq_len
            )));
        }

        let original_dtype = x.dtype();
        if original_dtype == DType::BF16 {
            apply_rope(x, self.dim, self.theta, 0)
        } else {
            let rope_input = x.to_dtype(DType::BF16)?;
            let rotated = apply_rope(&rope_input, self.dim, self.theta, 0)?;
            rotated.to_dtype(original_dtype)
        }
    }
}

/// Basic Transformer Block
pub struct TransformerBlock {
    pub self_attn: MultiHeadAttention,
    pub cross_attn: Option<MultiHeadAttention>,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: Option<LayerNorm>,
}

impl TransformerBlock {
    pub fn new(
        dim: usize,
        num_heads: usize,
        ff_mult: usize,
        use_cross_attn: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let attn_config = AttentionConfig::new(dim, num_heads);

        let self_attn = MultiHeadAttention::new(attn_config.clone(), device.clone())?;
        let cross_attn = if use_cross_attn {
            Some(MultiHeadAttention::new(attn_config, device.clone())?)
        } else {
            None
        };

        let ff = FeedForward::new(dim, None, ff_mult, device.clone())?;

        let norm1 = LayerNorm::new(dim, 1e-5, true, device.clone())?;
        let norm2 = LayerNorm::new(dim, 1e-5, true, device.clone())?;
        let norm3 = if use_cross_attn {
            Some(LayerNorm::new(dim, 1e-5, true, device)?)
        } else {
            None
        };

        Ok(Self {
            self_attn,
            cross_attn,
            ff,
            norm1,
            norm2,
            norm3,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        self_attn_mask: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self attention with residual
        let normed = self.norm1.forward(x)?;
        let self_attn_out = self
            .self_attn
            .forward(&normed, &normed, &normed, self_attn_mask)?;
        let x = x.add(&self_attn_out)?;

        // Cross attention if available
        let x = if let (Some(cross_attn), Some(norm3), Some(ctx)) =
            (&self.cross_attn, &self.norm3, context)
        {
            let normed = norm3.forward(&x)?;
            let cross_attn_out = cross_attn.forward(&normed, ctx, ctx, cross_attn_mask)?;
            x.add(&cross_attn_out)?
        } else {
            x
        };

        // Feed-forward with residual
        let normed = self.norm2.forward(&x)?;
        let ff_out = self.ff.forward(&normed)?;
        x.add(&ff_out)
    }
}

/// Layer normalization
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    pub fn new(
        normalized_shape: usize,
        eps: f32,
        elementwise_affine: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let (weight, bias) = if elementwise_affine {
            let weight = Tensor::from_vec_dtype(
                vec![1.0f32; normalized_shape],
                Shape::from_dims(&[normalized_shape]),
                device.clone(),
                DType::BF16,
            )?;
            let bias = Tensor::zeros_dtype(
                Shape::from_dims(&[normalized_shape]),
                DType::BF16,
                device,
            )?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        Ok(Self {
            normalized_shape: vec![normalized_shape],
            eps,
            elementwise_affine,
            weight,
            bias,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims();
        let normalized_dims = self.normalized_shape.len();

        if dims.len() < normalized_dims {
            return Err(Error::InvalidOperation(
                "Input dimensions less than normalized dimensions".into(),
            ));
        }

        // Check that last dimensions match
        for i in 0..normalized_dims {
            if dims[dims.len() - normalized_dims + i] != self.normalized_shape[i] {
                return Err(Error::InvalidOperation(
                    "Shape mismatch in layer norm".into(),
                ));
            }
        }

        crate::layer_norm::layer_norm(
            x,
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }
}
