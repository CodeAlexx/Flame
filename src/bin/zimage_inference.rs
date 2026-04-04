//! ZImage NextDiT inference — pure Rust, single-file binary.
//!
//! Architecture: Lumina2/NextDiT with joint attention, 3D RoPE, SwiGLU FFN.
//! - 30 main layers, 2 noise refiners, 2 context refiners
//! - dim=3840, 30 heads, head_dim=128
//! - Qwen3 4B text (cap_feat_dim=2560)
//! - z_image modulation: per-layer adaLN with tanh gates, min_mod=256
//! - Patchify 2x2 -> Linear(64, 3840), NOT Conv2d
//! - Model returns negated velocity: -img
//!
//! Usage:
//!   cargo run --bin zimage_inference -- \
//!       --model /path/to/zimage.safetensors \
//!       --embeddings /path/to/text_embeddings.safetensors \
//!       --output /path/to/output_latents.safetensors \
//!       --height 1024 --width 1024 --steps 30 --cfg 4.0

use cudarc::driver::CudaDevice;
use flame_core::attention::sdpa::sdpa;
use flame_core::layer_norm::layer_norm;
use flame_core::linear::Linear;
use flame_core::norm::RMSNorm;
use flame_core::serialization::{load_file, save_file};
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct NextDiTConfig {
    dim: usize,
    num_heads: usize,
    head_dim: usize,
    num_layers: usize,
    num_noise_refiner: usize,
    num_context_refiner: usize,
    cap_feat_dim: usize,
    mlp_hidden: usize,
    min_mod: usize,
    t_embedder_hidden: usize,
    patch_size: usize,
    in_channels: usize,
    axes_dims_rope: [usize; 3],
    rope_theta: f32,
    time_scale: f32,
    pad_tokens_multiple: usize,
}

impl Default for NextDiTConfig {
    fn default() -> Self {
        Self {
            dim: 3840,
            num_heads: 30,
            head_dim: 128,
            num_layers: 30,
            num_noise_refiner: 2,
            num_context_refiner: 2,
            cap_feat_dim: 2560,
            mlp_hidden: 10240,
            min_mod: 256,
            t_embedder_hidden: 1024,
            patch_size: 2,
            in_channels: 16,
            axes_dims_rope: [32, 48, 48],
            rope_theta: 256.0,
            time_scale: 1000.0,
            pad_tokens_multiple: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight-backed model — stores all weights in a HashMap, looks up by key
// ---------------------------------------------------------------------------

struct NextDiT {
    config: NextDiTConfig,
    /// Small weights that stay on GPU permanently (embedders, final layer, pad tokens)
    resident: HashMap<String, Tensor>,
    /// Path to safetensors file for on-demand block loading via mmap
    model_path: String,
    device: Arc<CudaDevice>,
    /// Temporarily loaded block weights (loaded on demand, dropped after use)
    block_cache: HashMap<String, Tensor>,
}

impl NextDiT {
    fn new(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<CudaDevice>,
    ) -> Self {
        Self {
            config: NextDiTConfig::default(),
            resident,
            model_path,
            device,
            block_cache: HashMap::new(),
        }
    }

    /// Load a block's weights from disk (mmap) into GPU, replacing any previous block.
    fn load_block(&mut self, prefix: &str) -> Result<()> {
        // Drop previous block weights to free VRAM
        self.block_cache.clear();

        let prefix_dot = format!("{prefix}.");
        let block_weights = load_file_filtered(&self.model_path, &self.device, |key| {
            key.starts_with(&prefix_dot)
        })?;

        println!("    [offload] Loaded {} tensors for {prefix}", block_weights.len());
        self.block_cache = block_weights;
        Ok(())
    }

    /// Drop current block weights to free VRAM.
    fn unload_block(&mut self) {
        self.block_cache.clear();
    }

    /// Get a weight tensor by key — checks block_cache first, then resident.
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.block_cache
            .get(key)
            .or_else(|| self.resident.get(key))
            .ok_or_else(|| Error::InvalidInput(format!("Missing weight key: {key}")))
    }

    // -- Linear helpers (matmul + optional bias) -----------------------------

    /// x @ weight.T  (weight shape: [out, in], no bias)
    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        // Linear::forward does x @ W^T + bias. We replicate manually for
        // weight-from-hashmap approach: reshape to 2D, matmul, reshape back.
        let x_dims = x.shape().dims().to_vec();
        let in_features = *x_dims.last().unwrap();
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
        let out_features = weight.shape().dims()[0];

        let x_2d = x.reshape(&[batch, in_features])?;
        let wt = transpose_2d(weight)?;
        let out_2d = x_2d.matmul(&wt)?;

        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(out_features);
        out_2d.reshape(&out_shape)
    }

    /// x @ weight.T + bias
    fn linear_with_bias(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let mut out = self.linear_no_bias(x, weight_key)?;
        let bias = self.w(bias_key)?;
        let out_dims = out.shape().dims().to_vec();
        // Broadcast bias [out_features] -> [1, ..., 1, out_features]
        let bias_1d = bias.reshape(&[1, *out_dims.last().unwrap()])?;
        let batch: usize = out_dims[..out_dims.len() - 1].iter().product();
        let out_feat = *out_dims.last().unwrap();
        let out_2d = out.reshape(&[batch, out_feat])?;
        let result_2d = out_2d.add(&bias_1d)?;
        result_2d.reshape(&out_dims)
    }

    // -- RMSNorm (functional, using weight from HashMap) ---------------------

    fn rms_norm(&self, x: &Tensor, weight_key: &str, eps: f32) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let norm_dim = weight.shape().dims()[0];
        // Use the RMSNorm struct from flame_core
        let mut norm = RMSNorm::new(vec![norm_dim], eps, true, self.device.clone())?;
        norm.copy_weight_from(weight)?;
        norm.forward(x)
    }

    // -- SwiGLU FFN ----------------------------------------------------------

    fn swiglu(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        // w2(silu(w1(x)) * w3(x))
        let w1_out = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w1.weight"))?;
        let w3_out = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w3.weight"))?;
        let gate = w1_out.silu()?;
        let hidden = gate.mul(&w3_out)?;
        self.linear_no_bias(&hidden, &format!("{prefix}.feed_forward.w2.weight"))
    }

    // -- Attention -----------------------------------------------------------

    fn joint_attention(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        prefix: &str,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let seq = dims[1];
        let dim = dims[2];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Fused QKV projection (no bias)
        let qkv = self.linear_no_bias(x, &format!("{prefix}.attention.qkv.weight"))?;
        let chunks = qkv.chunk(3, 2)?; // split on last dim
        let q = chunks[0].reshape(&[b, seq, num_heads, head_dim])?;
        let k = chunks[1].reshape(&[b, seq, num_heads, head_dim])?;
        let v = chunks[2].reshape(&[b, seq, num_heads, head_dim])?;

        // QK RMSNorm (per-head, applied on head_dim dimension)
        let q = self.rms_norm_per_head(&q, &format!("{prefix}.attention.q_norm.weight"))?;
        let k = self.rms_norm_per_head(&k, &format!("{prefix}.attention.k_norm.weight"))?;

        // Apply 3D RoPE
        let q = apply_rope_real(&q, rope_cos, rope_sin)?;
        let k = apply_rope_real(&k, rope_cos, rope_sin)?;

        // Transpose to [B, H, S, D] for SDPA
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let out = sdpa(&q, &k, &v, None)?;

        // Back to [B, S, H*D]
        let out = out.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(&[b, seq, num_heads * head_dim])?;

        // Output projection
        self.linear_no_bias(&out, &format!("{prefix}.attention.out.weight"))
    }

    /// Apply RMSNorm per head: input [B, S, H, D], norm weight [D]
    fn rms_norm_per_head(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let dims = x.shape().dims().to_vec();
        let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);

        // Flatten to [..., D] for RMSNorm
        let flat = x.reshape(&[b * s * h, d])?;
        let mut norm = RMSNorm::new(vec![d], 1e-6, true, self.device.clone())?;
        norm.copy_weight_from(weight)?;
        let normed = norm.forward(&flat)?;
        normed.reshape(&[b, s, h, d])
    }

    // -- Transformer block ---------------------------------------------------

    fn transformer_block(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        t_cond: Option<&Tensor>,
        prefix: &str,
    ) -> Result<Tensor> {
        let has_adaln = t_cond.is_some()
            && self
                .block_cache
                .contains_key(&format!("{prefix}.adaLN_modulation.0.weight"));

        // Compute modulation if conditioned
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if has_adaln {
            let t_cond = t_cond.unwrap();
            let mod_out = self.linear_with_bias(
                t_cond,
                &format!("{prefix}.adaLN_modulation.0.weight"),
                &format!("{prefix}.adaLN_modulation.0.bias"),
            )?;
            let chunks = mod_out.chunk(4, mod_out.shape().dims().len() - 1)?;
            (
                Some(chunks[0].clone()),
                Some(chunks[1].clone()),
                Some(chunks[2].clone()),
                Some(chunks[3].clone()),
            )
        } else {
            (None, None, None, None)
        };

        // Attention: pre-norm -> modulate -> attention -> post-norm -> gate + residual
        let mut x_norm =
            self.rms_norm(x, &format!("{prefix}.attention_norm1.weight"), 1e-6)?;

        if let Some(ref scale) = scale_msa {
            // x_norm = x_norm * (1 + scale.unsqueeze(1))
            let ones = Tensor::from_vec_dtype(
                vec![1.0f32],
                Shape::from_dims(&[1, 1, 1]),
                self.device.clone(),
                DType::BF16,
            )?;
            let scale_unsq = scale.unsqueeze(1)?;
            let factor = ones.add(&scale_unsq)?;
            x_norm = x_norm.mul(&factor)?;
        }

        let attn_out = self.joint_attention(&x_norm, rope_cos, rope_sin, prefix)?;
        let attn_out =
            self.rms_norm(&attn_out, &format!("{prefix}.attention_norm2.weight"), 1e-6)?;

        let mut x_out = if let Some(ref gate) = gate_msa {
            let g = gate.tanh()?.unsqueeze(1)?;
            let gated = g.mul(&attn_out)?;
            x.add(&gated)?
        } else {
            x.add(&attn_out)?
        };

        // FFN: pre-norm -> modulate -> FFN -> post-norm -> gate + residual
        let mut ff_norm =
            self.rms_norm(&x_out, &format!("{prefix}.ffn_norm1.weight"), 1e-6)?;

        if let Some(ref scale) = scale_mlp {
            let ones = Tensor::from_vec_dtype(
                vec![1.0f32],
                Shape::from_dims(&[1, 1, 1]),
                self.device.clone(),
                DType::BF16,
            )?;
            let scale_unsq = scale.unsqueeze(1)?;
            let factor = ones.add(&scale_unsq)?;
            ff_norm = ff_norm.mul(&factor)?;
        }

        let ff_out = self.swiglu(&ff_norm, prefix)?;
        let ff_out =
            self.rms_norm(&ff_out, &format!("{prefix}.ffn_norm2.weight"), 1e-6)?;

        x_out = if let Some(ref gate) = gate_mlp {
            let g = gate.tanh()?.unsqueeze(1)?;
            let gated = g.mul(&ff_out)?;
            x_out.add(&gated)?
        } else {
            x_out.add(&ff_out)?
        };

        Ok(x_out)
    }

    // -- Timestep embedder ---------------------------------------------------

    fn timestep_embed(&self, t: &Tensor) -> Result<Tensor> {
        // Sinusoidal embedding: t is (B,) scaled timestep
        let freq_dim = self.config.min_mod; // 256
        let half = freq_dim / 2;
        let max_period: f32 = 10000.0;

        let t_data = t.to_vec()?;
        let batch = t_data.len();

        // Build sinusoidal: [cos(t*f0), cos(t*f1), ..., sin(t*f0), sin(t*f1), ...]
        let mut emb_data = vec![0.0f32; batch * freq_dim];
        for b in 0..batch {
            let t_val = t_data[b];
            for i in 0..half {
                let freq = (-f32::ln(max_period) * (i as f32) / (half as f32)).exp();
                let angle = t_val * freq;
                emb_data[b * freq_dim + i] = angle.cos();
                emb_data[b * freq_dim + half + i] = angle.sin();
            }
        }

        let emb = Tensor::from_vec_dtype(
            emb_data,
            Shape::from_dims(&[batch, freq_dim]),
            self.device.clone(),
            DType::BF16,
        )?;

        // MLP: Linear(256, 1024) -> SiLU -> Linear(1024, 256)
        let h = self.linear_with_bias(&emb, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias")?;
        let h = h.silu()?;
        self.linear_with_bias(&h, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias")
    }

    // -- Caption embedder ----------------------------------------------------

    fn caption_embed(&self, cap_feats: &Tensor) -> Result<Tensor> {
        // RMSNorm(2560) -> Linear(2560, 3840)
        let normed = self.rms_norm(cap_feats, "cap_embedder.0.weight", 1e-6)?;
        // cap_embedder.1 is the Linear — check if it has bias
        if self.resident.contains_key("cap_embedder.1.bias") {
            self.linear_with_bias(&normed, "cap_embedder.1.weight", "cap_embedder.1.bias")
        } else {
            self.linear_no_bias(&normed, "cap_embedder.1.weight")
        }
    }

    // -- Patchify / Unpatchify -----------------------------------------------

    fn patchify(&self, x: &Tensor) -> Result<(Tensor, usize, usize)> {
        // (B, C, H, W) -> (B, N, patch_dim)
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let p = self.config.patch_size;
        let ph = h / p;
        let pw = w / p;

        // reshape to (B, C, ph, p, pw, p)
        let x = x.reshape(&[b, c, ph, p, pw, p])?;
        // permute to (B, ph, pw, p, p, C)
        let x = x.permute(&[0, 2, 4, 3, 5, 1])?;
        // reshape to (B, ph*pw, p*p*C)
        let x = x.reshape(&[b, ph * pw, p * p * c])?;
        Ok((x, ph, pw))
    }

    fn unpatchify(&self, x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
        // (B, N, patch_dim) -> (B, C, H, W)
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let p = self.config.patch_size;
        let c = self.config.in_channels;

        // reshape to (B, ph, pw, p, p, C)
        let x = x.reshape(&[b, ph, pw, p, p, c])?;
        // permute to (B, C, ph, p, pw, p)
        let x = x.permute(&[0, 5, 1, 3, 2, 4])?;
        // reshape to (B, C, H, W)
        x.reshape(&[b, c, ph * p, pw * p])
    }

    // -- Pad tokens ----------------------------------------------------------

    fn pad_to_multiple(
        &self,
        tokens: &Tensor,
        pad_token_key: &str,
        multiple: usize,
    ) -> Result<(Tensor, usize)> {
        let seq_len = tokens.shape().dims()[1];
        let pad_len = (multiple - (seq_len % multiple)) % multiple;
        if pad_len == 0 {
            return Ok((tokens.clone(), 0));
        }

        let b = tokens.shape().dims()[0];
        let dim = tokens.shape().dims()[2];
        let pad_token = self.w(pad_token_key)?; // shape [1, dim]

        // Expand pad token to [B, pad_len, dim]
        // Create pad by repeating the token
        let pad_single = pad_token.reshape(&[1, 1, dim])?;
        // Build pad tensor by repeating
        let mut pad_parts = Vec::new();
        for _ in 0..pad_len {
            pad_parts.push(&pad_single);
        }
        // Use cat along seq dimension for a single batch, then expand
        let pad_seq = if pad_len == 1 {
            pad_single.clone()
        } else {
            let refs: Vec<&Tensor> = pad_parts.into_iter().collect();
            Tensor::cat(&refs, 1)?
        };

        // Expand to batch size by repeating
        let mut batch_parts: Vec<Tensor> = Vec::new();
        for _ in 0..b {
            batch_parts.push(pad_seq.clone());
        }
        let batch_refs: Vec<&Tensor> = batch_parts.iter().collect();
        let pad_batch = if b == 1 {
            pad_seq
        } else {
            Tensor::cat(&batch_refs, 0)?
        };

        let result = Tensor::cat(&[tokens, &pad_batch], 1)?;
        Ok((result, pad_len))
    }

    // -- Full forward pass ---------------------------------------------------

    fn forward(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        cap_feats: &Tensor,
    ) -> Result<Tensor> {
        let pad_mult = self.config.pad_tokens_multiple;

        // Invert timestep and scale
        // t = 1.0 - timestep
        let t_data = timestep.to_vec()?;
        let inv_data: Vec<f32> = t_data
            .iter()
            .map(|v| (1.0 - v) * self.config.time_scale)
            .collect();
        let t_scaled = Tensor::from_vec_dtype(
            inv_data,
            timestep.shape().clone(),
            self.device.clone(),
            DType::BF16,
        )?;
        let t_cond = self.timestep_embed(&t_scaled)?;

        // Patchify and embed image
        let (x_patches, ph, pw) = self.patchify(x)?;
        // x_embedder is a linear with bias (resident weights)
        let x_emb = if self.resident.contains_key("x_embedder.bias") {
            self.linear_with_bias(&x_patches, "x_embedder.weight", "x_embedder.bias")?
        } else {
            self.linear_no_bias(&x_patches, "x_embedder.weight")?
        };
        let img_len = x_emb.shape().dims()[1];

        // Embed captions
        let c = self.caption_embed(cap_feats)?;

        // Pad caption to multiple of pad_tokens_multiple
        let (c, _cap_pad_len) = self.pad_to_multiple(&c, "cap_pad_token", pad_mult)?;
        let cap_len = c.shape().dims()[1];

        // Pad image to multiple of pad_tokens_multiple
        let (x_emb, img_pad_len) = self.pad_to_multiple(&x_emb, "x_pad_token", pad_mult)?;

        // Build position IDs and RoPE
        let (rope_cos_full, rope_sin_full) =
            self.build_3d_rope(cap_len, ph, pw, img_pad_len)?;

        // Split RoPE for caption and image portions
        let rope_cos_cap = rope_cos_full.narrow(0, 0, cap_len)?;
        let rope_sin_cap = rope_sin_full.narrow(0, 0, cap_len)?;
        let rope_cos_img = rope_cos_full.narrow(0, cap_len, x_emb.shape().dims()[1])?;
        let rope_sin_img = rope_sin_full.narrow(0, cap_len, x_emb.shape().dims()[1])?;

        // Context refiner: text self-attention (unconditioned)
        // Stream each block: load weights → forward → unload
        let mut c = c;
        for i in 0..self.config.num_context_refiner {
            let prefix = format!("context_refiner.{i}");
            self.load_block(&prefix)?;
            c = self.transformer_block(&c, &rope_cos_cap, &rope_sin_cap, None, &prefix)?;
            self.unload_block();
        }

        // Noise refiner: image-only self-attention (conditioned)
        let mut x_emb = x_emb;
        for i in 0..self.config.num_noise_refiner {
            let prefix = format!("noise_refiner.{i}");
            self.load_block(&prefix)?;
            x_emb = self.transformer_block(
                &x_emb,
                &rope_cos_img,
                &rope_sin_img,
                Some(&t_cond),
                &prefix,
            )?;
            self.unload_block();
        }

        // Concatenate text + image for main layers
        let mut xc = Tensor::cat(&[&c, &x_emb], 1)?;

        // Main transformer layers — stream each block from disk
        for i in 0..self.config.num_layers {
            let prefix = format!("layers.{i}");
            println!("  Layer {i}/{}", self.config.num_layers);
            self.load_block(&prefix)?;
            xc = self.transformer_block(
                &xc,
                &rope_cos_full,
                &rope_sin_full,
                Some(&t_cond),
                &prefix,
            )?;
            self.unload_block();
        }

        // Extract image tokens (skip text, remove padding)
        let x_out = xc.narrow(1, cap_len, img_len)?;

        // Final layer: LayerNorm(no affine) -> adaLN scale -> Linear
        let x_final = self.final_layer(&x_out, &t_cond)?;

        // Unpatchify
        let x_spatial = self.unpatchify(&x_final, ph, pw)?;

        // Negate (ZImage convention: return negated velocity)
        x_spatial.mul_scalar(-1.0)
    }

    // -- Final layer ---------------------------------------------------------

    fn final_layer(&self, x: &Tensor, t_cond: &Tensor) -> Result<Tensor> {
        let dim = self.config.dim;
        let patch_dim = self.config.patch_size * self.config.patch_size * self.config.in_channels;

        // LayerNorm without affine
        let x_norm = layer_norm(x, &[dim], None, None, 1e-6)?;

        // adaLN modulation: SiLU -> Linear
        let t_silu = t_cond.silu()?;
        let scale = self.linear_with_bias(
            &t_silu,
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias",
        )?;
        let scale_unsq = scale.unsqueeze(1)?;

        // x = x_norm * (1 + scale)
        let ones = Tensor::from_vec_dtype(
            vec![1.0f32],
            Shape::from_dims(&[1, 1, 1]),
            self.device.clone(),
            DType::BF16,
        )?;
        let factor = ones.add(&scale_unsq)?;
        let x_modulated = x_norm.mul(&factor)?;

        // Final linear projection
        self.linear_with_bias(
            &x_modulated,
            "final_layer.linear.weight",
            "final_layer.linear.bias",
        )
    }

    // -- 3D RoPE -------------------------------------------------------------

    fn build_3d_rope(
        &self,
        cap_len: usize,
        ph: usize,
        pw: usize,
        img_pad_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let axes_dims = self.config.axes_dims_rope;
        let theta = self.config.rope_theta;
        let img_seq = ph * pw + img_pad_len;
        let total_seq = cap_len + img_seq;
        let half_head_dim = self.config.head_dim / 2; // 64

        // Build position IDs: (total_seq, 3) for [t, h, w]
        // Caption: t=1..cap_len, h=0, w=0
        // Image: t=cap_len+1, h=0..ph-1, w=0..pw-1 (then zeros for padding)
        let mut pos_ids = vec![[0.0f32; 3]; total_seq];

        // Caption positions
        for i in 0..cap_len {
            pos_ids[i] = [(i + 1) as f32, 0.0, 0.0];
        }

        // Image positions
        for ih in 0..ph {
            for iw in 0..pw {
                let idx = cap_len + ih * pw + iw;
                pos_ids[idx] = [(cap_len + 1) as f32, ih as f32, iw as f32];
            }
        }
        // Padding positions remain [0, 0, 0]

        // Build cos/sin for each axis, concatenate
        let mut cos_data = vec![0.0f32; total_seq * half_head_dim];
        let mut sin_data = vec![0.0f32; total_seq * half_head_dim];

        let mut offset = 0;
        for (axis_idx, &axis_dim) in axes_dims.iter().enumerate() {
            let half_axis = axis_dim / 2;

            // Precompute frequencies for this axis
            let mut freqs = vec![0.0f32; half_axis];
            for i in 0..half_axis {
                freqs[i] = 1.0 / theta.powf(i as f32 / half_axis as f32);
            }

            for seq_idx in 0..total_seq {
                let pos = pos_ids[seq_idx][axis_idx];
                for (freq_idx, &freq) in freqs.iter().enumerate() {
                    let angle = pos * freq;
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();
                    cos_data[seq_idx * half_head_dim + offset + freq_idx] = cos_val;
                    sin_data[seq_idx * half_head_dim + offset + freq_idx] = sin_val;
                }
            }
            offset += half_axis;
        }

        let cos_tensor = Tensor::from_vec_dtype(
            cos_data,
            Shape::from_dims(&[total_seq, half_head_dim]),
            self.device.clone(),
            DType::BF16,
        )?;
        let sin_tensor = Tensor::from_vec_dtype(
            sin_data,
            Shape::from_dims(&[total_seq, half_head_dim]),
            self.device.clone(),
            DType::BF16,
        )?;

        Ok((cos_tensor, sin_tensor))
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

/// Transpose a 2D tensor [M, N] -> [N, M]
fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    t.permute(&[1, 0])
}

/// Apply RoPE using real-valued even/odd interleaving.
///
/// x shape: [B, S, H, D]  (D = head_dim = 128)
/// rope_cos, rope_sin shape: [S, D/2]  (D/2 = 64)
///
/// The complex-multiply RoPE maps to:
///   x_even' = x_even * cos - x_odd * sin
///   x_odd'  = x_even * sin + x_odd * cos
///
/// where even/odd indices are pairs along the head_dim dimension.
fn apply_rope_real(
    x: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half_d = d / 2;

    // Reshape x to [..., D/2, 2] to split even/odd
    let x_pairs = x.reshape(&[b, s, h, half_d, 2])?;
    let x_even = x_pairs.narrow(4, 0, 1)?.squeeze(4)?; // [B, S, H, D/2]
    let x_odd = x_pairs.narrow(4, 1, 1)?.squeeze(4)?; // [B, S, H, D/2]

    // Broadcast rope: [S, D/2] -> [1, S, 1, D/2]
    let cos = rope_cos.reshape(&[1, s, 1, half_d])?;
    let sin = rope_sin.reshape(&[1, s, 1, half_d])?;

    // x_even' = x_even * cos - x_odd * sin
    let new_even = x_even.mul(&cos)?.sub(&x_odd.mul(&sin)?)?;
    // x_odd' = x_even * sin + x_odd * cos
    let new_odd = x_even.mul(&sin)?.add(&x_odd.mul(&cos)?)?;

    // Interleave back: stack on last dim then flatten
    let new_even_exp = new_even.unsqueeze(4)?; // [B, S, H, D/2, 1]
    let new_odd_exp = new_odd.unsqueeze(4)?; // [B, S, H, D/2, 1]
    let stacked = Tensor::cat(&[&new_even_exp, &new_odd_exp], 4)?; // [B, S, H, D/2, 2]
    stacked.reshape(&[b, s, h, d])
}

// ---------------------------------------------------------------------------
// Sigma schedule
// ---------------------------------------------------------------------------

fn build_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            *v = shift * *v / (1.0 + (shift - 1.0) * *v);
        }
    }
    t
}

// ---------------------------------------------------------------------------
// Euler sampler
// ---------------------------------------------------------------------------

fn euler_step(
    model: &mut NextDiT,
    x: &Tensor,
    sigma: f32,
    sigma_next: f32,
    cap_feats: &Tensor,
    cap_feats_uncond: Option<&Tensor>,
    cfg_scale: f32,
) -> Result<Tensor> {
    let device = model.device.clone();
    let b = x.shape().dims()[0];

    // Build timestep tensor (sigma value)
    let sigma_tensor = Tensor::from_vec_dtype(
        vec![sigma; b],
        Shape::from_dims(&[b]),
        device.clone(),
        DType::BF16,
    )?;

    // Model prediction (conditional)
    let pred_cond = model.forward(x, &sigma_tensor, cap_feats)?;

    let pred = if let Some(uncond_feats) = cap_feats_uncond {
        if cfg_scale > 1.0 {
            // Unconditional prediction
            let pred_uncond = model.forward(x, &sigma_tensor, uncond_feats)?;
            // CFG: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            let diff = pred_cond.sub(&pred_uncond)?;
            let scaled = diff.mul_scalar(cfg_scale)?;
            pred_uncond.add(&scaled)?
        } else {
            pred_cond
        }
    } else {
        pred_cond
    };

    // Euler step: x_next = x + (pred - x) * (sigma_next - sigma) / sigma
    // For flow matching: the model predicts velocity v, and:
    // x_next = x + v * dt where dt = sigma_next - sigma
    let dt = sigma_next - sigma;
    let step = pred.mul_scalar(dt)?;
    x.add(&step)
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

struct Args {
    model_path: String,
    embeddings_path: String,
    output_path: String,
    height: usize,
    width: usize,
    steps: usize,
    cfg_scale: f32,
    shift: f32,
    seed: u64,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut embeddings_path = String::new();
    let mut output_path = String::from("output_latents.safetensors");
    let mut height: usize = 1024;
    let mut width: usize = 1024;
    let mut steps: usize = 30;
    let mut cfg_scale: f32 = 4.0;
    let mut shift: f32 = 1.0;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = args[i].clone();
            }
            "--embeddings" => {
                i += 1;
                embeddings_path = args[i].clone();
            }
            "--output" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--height" => {
                i += 1;
                height = args[i].parse().expect("Invalid height");
            }
            "--width" => {
                i += 1;
                width = args[i].parse().expect("Invalid width");
            }
            "--steps" => {
                i += 1;
                steps = args[i].parse().expect("Invalid steps");
            }
            "--cfg" => {
                i += 1;
                cfg_scale = args[i].parse().expect("Invalid cfg");
            }
            "--shift" => {
                i += 1;
                shift = args[i].parse().expect("Invalid shift");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid seed");
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!("Usage: zimage_inference --model <path> --embeddings <path> [options]");
        eprintln!("  --model       Path to ZImage safetensors weights");
        eprintln!("  --embeddings  Path to pre-computed text embeddings (safetensors)");
        eprintln!("  --output      Output latents path (default: output_latents.safetensors)");
        eprintln!("  --height      Image height in pixels (default: 1024)");
        eprintln!("  --width       Image width in pixels (default: 1024)");
        eprintln!("  --steps       Number of denoising steps (default: 30)");
        eprintln!("  --cfg         Classifier-free guidance scale (default: 4.0)");
        eprintln!("  --shift       Sigma schedule shift (default: 1.0)");
        eprintln!("  --seed        Random seed (default: 42)");
        std::process::exit(1);
    }

    Args {
        model_path,
        embeddings_path,
        output_path,
        height,
        width,
        steps,
        cfg_scale,
        shift,
        seed,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    println!("=== ZImage NextDiT Inference (Pure Rust) ===\n");

    let args = parse_args();

    // Create CUDA device
    let device = CudaDevice::new(0).map_err(|e| {
        Error::InvalidOperation(format!("Failed to create CUDA device: {e:?}"))
    })?;
    let device = Arc::new(device);
    println!("[+] CUDA device initialized");

    // Validate dimensions
    let latent_h = args.height / 8; // VAE downscale factor
    let latent_w = args.width / 8;
    println!(
        "[+] Image: {}x{} -> Latent: {}x{} -> Patches: {}x{}",
        args.height,
        args.width,
        latent_h,
        latent_w,
        latent_h / 2,
        latent_w / 2,
    );

    // Load only small resident weights (embedders, final layer, pad tokens)
    // Block weights (~440MB each) are streamed from disk via mmap on demand
    println!("[+] Loading resident weights from: {}", args.model_path);
    let resident_prefixes = [
        "x_embedder.", "cap_embedder.", "t_embedder.", "final_layer.",
        "x_pad_token", "cap_pad_token",
    ];
    let resident = load_file_filtered(&args.model_path, &device, |key| {
        resident_prefixes.iter().any(|p| key.starts_with(p))
    })?;
    println!("    Loaded {} resident tensors (embedders + final layer)", resident.len());

    // Print a few key shapes for sanity
    if let Some(t) = resident.get("x_embedder.weight") {
        println!("    x_embedder.weight: {:?}", t.shape().dims());
    }

    let mut model = NextDiT::new(args.model_path.clone(), resident, device.clone());

    // Load text embeddings
    println!(
        "[+] Loading text embeddings from: {}",
        args.embeddings_path
    );
    let emb_tensors = load_file(&args.embeddings_path, &device)?;

    // Expect keys: "cap_feats" (B, seq, 2560) and optionally "cap_feats_uncond"
    let cap_feats = emb_tensors.get("cap_feats").ok_or_else(|| {
        Error::InvalidInput(
            "Embeddings file must contain 'cap_feats' key (B, seq, 2560)".into(),
        )
    })?;
    let cap_feats = cap_feats.to_dtype(DType::BF16)?;
    println!("    cap_feats shape: {:?}", cap_feats.shape().dims());

    let cap_feats_uncond = emb_tensors.get("cap_feats_uncond").map(|t| {
        println!(
            "    cap_feats_uncond shape: {:?}",
            t.shape().dims()
        );
        t.clone()
    });
    let cap_feats_uncond_ref = cap_feats_uncond.as_ref();

    // Initialize random latents
    println!("[+] Generating initial noise (seed={})", args.seed);
    let x = Tensor::randn(
        Shape::from_dims(&[1, 16, latent_h, latent_w]),
        0.0,
        1.0,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    // Build sigma schedule
    let sigmas = build_sigma_schedule(args.steps, args.shift);
    println!(
        "[+] Sigma schedule: {} steps, shift={}, range [{:.4}, {:.4}]",
        args.steps,
        args.shift,
        sigmas.first().unwrap_or(&0.0),
        sigmas.last().unwrap_or(&0.0),
    );

    // Denoising loop
    println!("\n[+] Starting denoising loop ({} steps)...", args.steps);
    let mut x = x;
    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        println!(
            "  Step {}/{}: sigma={:.6} -> {:.6}",
            step + 1,
            args.steps,
            sigma,
            sigma_next,
        );

        x = euler_step(
            &mut model,
            &x,
            sigma,
            sigma_next,
            &cap_feats,
            cap_feats_uncond_ref,
            args.cfg_scale,
        )?;
    }

    println!("\n[+] Denoising complete!");
    println!("    Output latent shape: {:?}", x.shape().dims());

    // Save raw latents
    println!("[+] Saving latents to: {}", args.output_path);
    let mut output_map = HashMap::new();
    output_map.insert("latents".to_string(), x);
    save_file(&output_map, &args.output_path)?;

    println!("\n=== Done! ===");
    println!("    To decode: load latents and run through VAE decoder.");

    Ok(())
}
