#![cfg(feature = "autograd_v4")]

use super::attach_backward_node;
use crate::cuda_ops::GpuOps;
use crate::ops_ext::shape4;
use crate::{device::CudaStreamRawPtrExt, rng, DType, Error, Result, Shape, Tensor, TensorId};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use smallvec::SmallVec;
use std::sync::Arc;
use std::{mem, ptr};

use super::super::graph::{GradNode, Op};

#[cfg(feature = "sdpa_debug")]
use log::debug;
#[cfg(feature = "sdpa_debug")]
use std::time::Instant;

const NEG_INF: f32 = -1.0e30;

#[derive(Clone, Copy)]
pub struct SdpaConfig {
    pub causal: bool,
    pub scale: Option<f32>,
    pub dropout_p: f32,
    pub dropout_seed: Option<u64>,
    pub upcast_compute: bool,
    pub chunk_q: Option<usize>,
    pub chunk_kv: Option<usize>,
    pub save: SdpaSave,
}

impl Default for SdpaConfig {
    fn default() -> Self {
        Self {
            causal: false,
            scale: None,
            dropout_p: 0.0,
            dropout_seed: None,
            upcast_compute: true,
            chunk_q: None,
            chunk_kv: None,
            save: SdpaSave::SaveLSE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdpaSave {
    SaveProbs,
    SaveLSE,
    Recompute,
}

#[derive(Default, Clone)]
pub struct SdpaStats {
    pub fwd_ms: f32,
    pub bwd_ms: f32,
    pub bytes_alloc: u64,
    pub bytes_saved: u64,
    pub nan_inf_hits: u32,
    pub syncs: u32,
    pub recompute_blocks: u32,
}

pub struct SdpaHooks {
    pub on_fwd_start: fn(&SdpaConfig, (usize, usize, usize, usize)),
    pub on_fwd_end: fn(&SdpaStats),
    pub on_bwd_start: fn(),
    pub on_bwd_end: fn(&SdpaStats),
}

#[cfg(feature = "sdpa_debug")]
static SDPA_HOOKS: once_cell::sync::OnceCell<SdpaHooks> = once_cell::sync::OnceCell::new();

#[cfg(feature = "sdpa_debug")]
pub fn register_sdpa_hooks(hooks: SdpaHooks) -> bool {
    SDPA_HOOKS.set(hooks).is_ok()
}

#[derive(Clone)]
pub struct SdpaCtx {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub attn_mask: Option<Tensor>, // broadcasted [B,H,Q,K] additive mask (FP32)
    pub scale: f32,
    pub save: SdpaSave,
    pub causal: bool,
    pub lse: Option<Tensor>,                  // [B,H,Q,1] FP32 if SaveLSE
    pub probs: Option<Tensor>,                // [B,H,Q,K] BF16 if SaveProbs
    pub shapes: (usize, usize, usize, usize), // (B,H,Q,K)
    pub chunk_cfg: Option<(usize, usize)>,
    pub dropout_p: f32,
    pub dropout_seed: Option<u64>,
}

pub fn sdpa_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    cfg: &SdpaConfig,
) -> Result<(Tensor, SdpaCtx)> {
    if !cfg.upcast_compute {
        return Err(Error::Unsupported(
            "sdpa_forward: upcast_compute=false not supported".into(),
        ));
    }
    if cfg.dropout_p < 0.0 || cfg.dropout_p >= 1.0 {
        return Err(Error::InvalidInput(format!(
            "sdpa_forward: dropout_p={} is out of range [0, 1)",
            cfg.dropout_p
        )));
    }
    #[cfg(feature = "sdpa_debug")]
    let debug_start = Instant::now();
    #[cfg(feature = "sdpa_debug")]
    let debug_mask_kind = describe_mask_kind(mask);
    let dropout_seed = if cfg.dropout_p > 0.0 {
        Some(cfg.dropout_seed.unwrap_or_else(rng::next_u64))
    } else {
        None
    };

    let (batch_q, heads_q, q_len, head_dim) = shape4(q)?;
    let (batch_k, heads_k, k_len, head_dim_k) = shape4(k)?;
    let (batch_v, heads_v, v_len, head_dim_v) = shape4(v)?;

    if !(batch_q == batch_k && batch_q == batch_v) || !(heads_q == heads_k && heads_q == heads_v) {
        return Err(Error::InvalidInput(format!(
            "sdpa shape mismatch: q {:?}, k {:?}, v {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if !(head_dim == head_dim_k && head_dim == head_dim_v) {
        return Err(Error::InvalidInput(format!(
            "sdpa head dim mismatch: q {}, k {}, v {}",
            head_dim, head_dim_k, head_dim_v
        )));
    }
    if k_len != v_len {
        return Err(Error::InvalidInput(format!(
            "sdpa key/value length mismatch: k_len {} v_len {}",
            k_len, v_len
        )));
    }

    let shapes = (batch_q, heads_q, q_len, k_len);
    let default_scale = if head_dim > 0 {
        1.0 / (head_dim as f32).sqrt()
    } else {
        1.0
    };
    let scale = cfg.scale.unwrap_or(default_scale);
    let bh = batch_q * heads_q;

    #[cfg(feature = "sdpa_debug")]
    if let Some(hooks) = debug_hooks() {
        (hooks.on_fwd_start)(cfg, shapes);
    }

    let mask_tensor = prepare_mask(mask, shapes)?;
    let mask_flat = if let Some(ref m) = mask_tensor {
        Some(m.clone().reshape(&[bh, q_len, k_len])?)
    } else {
        None
    };

    let q_flat = q.reshape(&[bh, q_len, head_dim])?;
    let k_flat = k.reshape(&[bh, k_len, head_dim])?;
    let v_flat = v.reshape(&[bh, k_len, head_dim])?;

    let q_f32 = q_flat.to_dtype(DType::F32)?;
    let k_f32 = k_flat.to_dtype(DType::F32)?;
    let v_f32 = v_flat.to_dtype(DType::F32)?;

    let requested_chunk_q = cfg.chunk_q.unwrap_or(0);
    let requested_chunk_kv = cfg.chunk_kv.unwrap_or(0);
    let actual_chunk_q = if requested_chunk_q > 0 {
        requested_chunk_q.min(q_len).max(1)
    } else {
        q_len
    };
    let actual_chunk_kv = if requested_chunk_kv > 0 {
        requested_chunk_kv.min(k_len).max(1)
    } else {
        k_len
    };
    #[cfg(feature = "sdpa_debug")]
    let debug_tiles_q = (q_len + actual_chunk_q - 1) / actual_chunk_q;
    #[cfg(feature = "sdpa_debug")]
    let debug_tiles_k = (k_len + actual_chunk_kv - 1) / actual_chunk_kv;

    let chunk_enabled = matches!(cfg.save, SdpaSave::SaveLSE)
        && (actual_chunk_q < q_len || actual_chunk_kv < k_len);

    let (output_f32, probs_to_store, lse_to_store, chunk_cfg) = if chunk_enabled {
        let (out_f32, lse_flat) = forward_chunked_savelse(
            &q_f32,
            &k_f32,
            &v_f32,
            mask_flat.as_ref(),
            scale,
            actual_chunk_q,
            actual_chunk_kv,
            cfg.causal,
            cfg.dropout_p,
            dropout_seed,
        )?;
        (
            out_f32,
            None,
            Some(lse_flat.reshape(&[batch_q, heads_q, q_len, 1])?),
            Some((actual_chunk_q, actual_chunk_kv)),
        )
    } else {
        let mut logits = compute_logits(&q_f32, &k_f32, scale)?;
        if let Some(ref mask_bias) = mask_flat {
            logits = logits.add(mask_bias)?;
        }
        if cfg.causal {
            apply_causal_mask_inplace(&mut logits, batch_q, heads_q, q_len, k_len, 0, 0)?;
        }

        let (mut probs, lse) = compute_probabilities(&logits)?;
        if cfg.dropout_p > 0.0 {
            probs = apply_dropout_tile(&probs, cfg.dropout_p, dropout_seed, 0, 0)?;
        }
        let probs_store = if matches!(cfg.save, SdpaSave::SaveProbs) {
            Some(
                probs
                    .reshape(&[batch_q, heads_q, q_len, k_len])?
                    .to_dtype(DType::BF16)?,
            )
        } else {
            None
        };
        let lse_store = if matches!(cfg.save, SdpaSave::SaveLSE) {
            Some(lse.reshape(&[batch_q, heads_q, q_len, 1])?)
        } else {
            None
        };

        let output_f32 = probs.bmm(&v_f32)?;
        (output_f32, probs_store, lse_store, None)
    };
    let output_dtype = q.dtype();
    let mut output = if output_dtype == DType::F32 {
        output_f32
    } else {
        output_f32.to_dtype(output_dtype)?
    };
    output = output.reshape(&[batch_q, heads_q, q_len, head_dim])?;
    let requires_grad = q.requires_grad || k.requires_grad || v.requires_grad;
    output.requires_grad = requires_grad;

    let ctx = SdpaCtx {
        q: q.clone(),
        k: k.clone(),
        v: v.clone(),
        attn_mask: mask_tensor,
        scale,
        save: cfg.save,
        causal: cfg.causal,
        lse: lse_to_store,
        probs: probs_to_store,
        shapes,
        chunk_cfg,
        dropout_p: cfg.dropout_p,
        dropout_seed,
    };

    if requires_grad {
        let mut parent_ids = SmallVec::<[TensorId; 4]>::new();
        if q.requires_grad {
            parent_ids.push(q.id);
        }
        if k.requires_grad {
            parent_ids.push(k.id);
        }
        if v.requires_grad {
            parent_ids.push(v.id);
        }

        if !parent_ids.is_empty() {
            let node = GradNode::new(
                output.id,
                parent_ids,
                Op::Sdpa {
                    ctx: Arc::new(ctx.clone()),
                    q_id: q.id,
                    k_id: k.id,
                    v_id: v.id,
                },
            );
            attach_backward_node(node);
        }
    }

    #[cfg(feature = "sdpa_debug")]
    if let Some(hooks) = debug_hooks() {
        (hooks.on_fwd_end)(&SdpaStats::default());
    }

    #[cfg(feature = "sdpa_debug")]
    {
        let elapsed_ms = debug_start.elapsed().as_secs_f64() * 1_000.0;
        debug!(
            "sdpa_v4 forward: chunk_enabled={}, chunk_q={}, chunk_kv={}, tiles_q={}, tiles_k={}, dropout_p={:.3}, mask={}, causal={}, elapsed_ms={:.3}",
            chunk_enabled,
            actual_chunk_q,
            actual_chunk_kv,
            debug_tiles_q,
            debug_tiles_k,
            cfg.dropout_p,
            debug_mask_kind,
            cfg.causal,
            elapsed_ms
        );
    }

    Ok((output, ctx))
}

fn forward_chunked_savelse(
    q_f32: &Tensor,
    k_f32: &Tensor,
    v_f32: &Tensor,
    mask_flat: Option<&Tensor>,
    scale: f32,
    chunk_q: usize,
    chunk_kv: usize,
    causal: bool,
    dropout_p: f32,
    dropout_seed: Option<u64>,
) -> Result<(Tensor, Tensor)> {
    let dims = q_f32.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(
            "forward_chunked_savelse expects tensor shaped [BH,Q,D]".into(),
        ));
    }
    let bh = dims[0];
    let q_len = dims[1];
    let d = dims[2];
    let k_len = k_f32.shape().dims().get(1).copied().ok_or_else(|| {
        Error::InvalidInput("k tensor must be shaped [BH,K,D] for chunked SDPA".into())
    })?;

    if q_len == 0 || k_len == 0 {
        return Err(Error::InvalidInput(
            "forward_chunked_savelse requires non-empty Q/K lengths".into(),
        ));
    }

    let chunk_q = chunk_q.min(q_len).max(1);
    let chunk_kv = chunk_kv.min(k_len).max(1);
    let tiles_q = (q_len + chunk_q - 1) / chunk_q;
    let tiles_k = (k_len + chunk_kv - 1) / chunk_kv;

    let device = q_f32.device().clone();
    let mut output = Tensor::zeros_dtype(
        Shape::from_dims(&[bh, q_len, d]),
        DType::F32,
        device.clone(),
    )?;
    let mut lse_out = Tensor::zeros_dtype(
        Shape::from_dims(&[bh, q_len, 1]),
        DType::F32,
        device.clone(),
    )?;

    for iq in 0..tiles_q {
        let q_start = iq * chunk_q;
        let q_size = (q_len - q_start).min(chunk_q);
        let q_slice = q_f32.narrow(1, q_start, q_size)?;

        let mut lse_row: Option<Tensor> = None;
        for ik in 0..tiles_k {
            let k_start = ik * chunk_kv;
            let k_size = (k_len - k_start).min(chunk_kv);
            let k_slice = k_f32.narrow(1, k_start, k_size)?;
            let mut logits = q_slice
                .bmm(&k_slice.transpose_dims(1, 2)?)?
                .mul_scalar(scale)?;
            let mask_tile = match mask_flat {
                Some(mask) => Some(
                    mask.narrow(1, q_start, q_size)?
                        .narrow(2, k_start, k_size)?,
                ),
                None => None,
            };
            apply_mask_tile_inplace(
                &mut logits,
                mask_tile.as_ref(),
                bh,
                q_size,
                k_size,
                q_start,
                k_start,
                k_len,
                causal,
            )?;
            let tile_lse = row_logsumexp(&logits)?;
            lse_row = Some(match lse_row {
                None => tile_lse,
                Some(existing) => logaddexp(&existing, &tile_lse)?,
            });
        }

        let lse_row = lse_row.expect("SDPA chunked LSE missing");

        let mut acc: Option<Tensor> = None;
        for ik in 0..tiles_k {
            let k_start = ik * chunk_kv;
            let k_size = (k_len - k_start).min(chunk_kv);
            let k_slice = k_f32.narrow(1, k_start, k_size)?;
            let v_slice = v_f32.narrow(1, k_start, k_size)?;
            let mut logits = q_slice
                .bmm(&k_slice.transpose_dims(1, 2)?)?
                .mul_scalar(scale)?;
            let mask_tile = match mask_flat {
                Some(mask) => Some(
                    mask.narrow(1, q_start, q_size)?
                        .narrow(2, k_start, k_size)?,
                ),
                None => None,
            };
            apply_mask_tile_inplace(
                &mut logits,
                mask_tile.as_ref(),
                bh,
                q_size,
                k_size,
                q_start,
                k_start,
                k_len,
                causal,
            )?;
            let probs = logits.sub(&lse_row)?.exp()?;
            let probs = apply_dropout_tile(&probs, dropout_p, dropout_seed, q_start, k_start)?;
            let delta = probs.bmm(&v_slice)?;
            acc = Some(match acc {
                None => delta,
                Some(existing) => existing.add(&delta)?,
            });
        }

        let acc_tensor = acc.expect("SDPA chunk accumulation failed");
        copy_tile_into(&mut output, &acc_tensor, q_start)?;
        copy_tile_into(&mut lse_out, &lse_row, q_start)?;
    }

    Ok((output, lse_out))
}

fn copy_tile_into(dst: &mut Tensor, tile: &Tensor, q_start: usize) -> Result<()> {
    if dst.dtype() != DType::F32 || tile.dtype() != DType::F32 {
        return Err(Error::InvalidOperation(
            "copy_tile_into expects F32 tensors".into(),
        ));
    }
    let dst_dims = dst.shape().dims();
    let tile_dims = tile.shape().dims();
    if dst_dims.len() != 3 || tile_dims.len() != 3 {
        return Err(Error::InvalidInput(
            "copy_tile_into expects tensors shaped [BH,Q,D]".into(),
        ));
    }
    let bh = dst_dims[0];
    let q_len = dst_dims[1];
    let d = dst_dims[2];
    if tile_dims[0] != bh || tile_dims[2] != d {
        return Err(Error::InvalidInput(
            "copy_tile_into tile dimensions do not match destination".into(),
        ));
    }
    let tile_q = tile_dims[1];
    if q_start + tile_q > q_len {
        return Err(Error::InvalidInput(
            "copy_tile_into tile exceeds destination bounds".into(),
        ));
    }

    let dst_slice = dst
        .storage_mut()
        .try_as_mut_slice_f32()
        .map_err(|_| Error::InvalidOperation("destination must expose F32 storage".into()))?;
    let dst_ptr = *dst_slice.device_ptr_mut() as *mut f32;

    let src_slice = tile
        .storage_ref()
        .try_as_slice_f32()
        .map_err(|_| Error::InvalidOperation("tile must expose F32 storage".into()))?;
    let src_ptr = *src_slice.device_ptr() as *const f32;

    let row_elems = tile_q * d;
    let bytes = row_elems * mem::size_of::<f32>();
    let stream = dst.device().cuda_stream_raw_ptr();

    for b in 0..bh {
        let dst_offset = ((b * q_len) + q_start) * d;
        let src_offset = b * row_elems;
        let status = unsafe {
            crate::cuda::ffi::flame_cuda_memcpy_async(
                dst_ptr.add(dst_offset) as *mut core::ffi::c_void,
                src_ptr.add(src_offset) as *const core::ffi::c_void,
                bytes,
                3,
                stream,
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!(
                "flame_cuda_memcpy_async failed with status {status}"
            )));
        }
    }

    Ok(())
}

#[cfg(feature = "sdpa_debug")]
fn describe_mask_kind(mask: Option<&Tensor>) -> &'static str {
    match mask {
        None => "none",
        Some(t) => match t.dtype() {
            DType::Bool => "bool",
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            _ => "other",
        },
    }
}

fn apply_mask_tile_inplace(
    logits: &mut Tensor,
    additive_mask: Option<&Tensor>,
    bh: usize,
    q_size: usize,
    k_size: usize,
    q_offset: usize,
    k_offset: usize,
    k_total: usize,
    causal: bool,
) -> Result<()> {
    if additive_mask.is_none() && !causal {
        return Ok(());
    }

    let slice = logits
        .storage_mut()
        .try_as_mut_slice_f32()
        .map_err(|_| Error::InvalidOperation("mask fusion expects logits in F32 storage".into()))?;
    let logits_ptr = *slice.device_ptr_mut() as *mut f32;

    let (add_ptr, add_rank) = if let Some(mask) = additive_mask {
        if mask.dtype() != DType::F32 {
            return Err(Error::InvalidInput(
                "mask fusion expects additive mask in F32 dtype".into(),
            ));
        }
        let mask_slice = mask.storage_ref().try_as_slice_f32().map_err(|_| {
            Error::InvalidOperation("mask fusion expects contiguous F32 mask".into())
        })?;
        (*mask_slice.device_ptr() as *const f32, 3i32)
    } else {
        (ptr::null(), 0)
    };

    let status = unsafe {
        crate::cuda::ffi::flame_sdpa_add_mask_tile_fp32(
            logits_ptr,
            ptr::null(),
            add_ptr,
            bh as i32,
            q_size as i32,
            k_size as i32,
            q_offset as i32,
            k_offset as i32,
            0,
            add_rank,
            k_total as i32,
            1,
            if causal { 1 } else { 0 },
            logits.device().cuda_stream_raw_ptr(),
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_sdpa_add_mask_tile_fp32 failed with status {status}"
        )));
    }
    Ok(())
}

fn derive_tile_seed(base_seed: u64, q_offset: usize, k_offset: usize) -> u32 {
    base_seed.wrapping_add(((q_offset as u64) << 32) ^ (k_offset as u64)) as u32
}

fn apply_dropout_tile(
    probs: &Tensor,
    dropout_p: f32,
    dropout_seed: Option<u64>,
    q_offset: usize,
    k_offset: usize,
) -> Result<Tensor> {
    if dropout_p <= 0.0 {
        return Ok(probs.clone());
    }
    let seed = dropout_seed.ok_or_else(|| {
        Error::InvalidOperation("dropout seed required when dropout_p > 0".into())
    })?;
    if !(dropout_p < 1.0) {
        return Err(Error::InvalidInput(
            "dropout_p must be in range (0, 1) when enabled".into(),
        ));
    }
    let dims = probs.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(
            "apply_dropout_tile expects tensor shaped [BH,Q,K]".into(),
        ));
    }
    let tile_seed = derive_tile_seed(seed, q_offset, k_offset);
    let device = probs.device().clone();
    let random = rng::rand_on(&device, dims, DType::F32, tile_seed)?;
    let threshold = random.full_like(dropout_p)?;
    let keep_mask = random.ge(&threshold)?.to_dtype(DType::F32)?;
    let keep = 1.0 - dropout_p;
    let scaled_mask = keep_mask.mul_scalar(1.0 / keep)?;

    let probs_f32 = if probs.dtype() == DType::F32 {
        probs.clone()
    } else {
        probs.to_dtype(DType::F32)?
    };
    let dropped = probs_f32.mul(&scaled_mask)?;
    let mut result = if probs.dtype() == DType::F32 {
        dropped
    } else {
        dropped.to_dtype(probs.dtype())?
    };
    result.requires_grad = probs.requires_grad;
    Ok(result)
}

fn row_logsumexp(tensor: &Tensor) -> Result<Tensor> {
    let max = GpuOps::max_dim(tensor, 2, true)?;
    let shifted = tensor.sub(&max)?;
    let exp = shifted.exp()?;
    let sum = GpuOps::sum_dim_keepdim(&exp, 2)?;
    let log_sum = sum.log()?;
    max.add(&log_sum)
}

fn logaddexp(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let max = GpuOps::max_elemwise(a, b)?;
    let a_shift = a.sub(&max)?.exp()?;
    let b_shift = b.sub(&max)?.exp()?;
    let sum = a_shift.add(&b_shift)?;
    let log_sum = sum.log()?;
    max.add(&log_sum)
}

fn backward_chunked_savelse(
    q_f32: &Tensor,
    k_f32: &Tensor,
    v_f32: &Tensor,
    lse_flat: &Tensor,
    d_y: &Tensor,
    mask_flat: Option<&Tensor>,
    scale: f32,
    chunk_q: usize,
    chunk_kv: usize,
    causal: bool,
    dropout_p: f32,
    dropout_seed: Option<u64>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = q_f32.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(
            "backward_chunked_savelse expects tensor shaped [BH,Q,D]".into(),
        ));
    }
    let bh = dims[0];
    let q_len = dims[1];
    let d = dims[2];
    let k_len = k_f32.shape().dims().get(1).copied().ok_or_else(|| {
        Error::InvalidInput("k tensor must be shaped [BH,K,D] for chunked SDPA".into())
    })?;

    let chunk_q = chunk_q.min(q_len).max(1);
    let chunk_kv = chunk_kv.min(k_len).max(1);
    let tiles_q = (q_len + chunk_q - 1) / chunk_q;
    let tiles_k = (k_len + chunk_kv - 1) / chunk_kv;

    let mut d_q_chunks: Vec<Tensor> = Vec::with_capacity(tiles_q);
    let mut d_k_tiles: Vec<Tensor> = Vec::with_capacity(tiles_k);
    let mut d_v_tiles: Vec<Tensor> = Vec::with_capacity(tiles_k);
    for ik in 0..tiles_k {
        let k_start = ik * chunk_kv;
        let k_size = (k_len - k_start).min(chunk_kv);
        let zero_shape = Shape::from_dims(&[bh, k_size, d]);
        d_k_tiles.push(Tensor::zeros_dtype(
            zero_shape.clone(),
            DType::F32,
            k_f32.device().clone(),
        )?);
        d_v_tiles.push(Tensor::zeros_dtype(
            zero_shape,
            DType::F32,
            v_f32.device().clone(),
        )?);
    }

    for iq in 0..tiles_q {
        let q_start = iq * chunk_q;
        let q_size = (q_len - q_start).min(chunk_q);
        let q_slice = q_f32.narrow(1, q_start, q_size)?;
        let lse_slice = lse_flat.narrow(1, q_start, q_size)?;
        let d_y_slice = d_y.narrow(1, q_start, q_size)?;

        let mut acc_dq: Option<Tensor> = None;
        for ik in 0..tiles_k {
            let k_start = ik * chunk_kv;
            let k_size = (k_len - k_start).min(chunk_kv);
            let k_slice = k_f32.narrow(1, k_start, k_size)?;
            let v_slice = v_f32.narrow(1, k_start, k_size)?;

            let mut logits = q_slice
                .bmm(&k_slice.transpose_dims(1, 2)?)?
                .mul_scalar(scale)?;
            let mask_tile = match mask_flat {
                Some(mask) => Some(
                    mask.narrow(1, q_start, q_size)?
                        .narrow(2, k_start, k_size)?,
                ),
                None => None,
            };
            apply_mask_tile_inplace(
                &mut logits,
                mask_tile.as_ref(),
                bh,
                q_size,
                k_size,
                q_start,
                k_start,
                k_len,
                causal,
            )?;
            let probs = logits.sub(&lse_slice)?.exp()?;
            let probs = apply_dropout_tile(&probs, dropout_p, dropout_seed, q_start, k_start)?;

            let v_t = v_slice.transpose_dims(1, 2)?;
            let d_p = d_y_slice.bmm(&v_t)?;
            let rowdot = GpuOps::sum_dim_keepdim(&d_p.mul(&probs)?, 2)?;
            let d_s = d_p.sub(&probs.mul(&rowdot)?)?;

            let d_v_delta = probs.transpose_dims(1, 2)?.bmm(&d_y_slice)?;
            d_v_tiles[ik] = d_v_tiles[ik].add(&d_v_delta)?;

            let d_q_delta = d_s.bmm(&k_slice)?.mul_scalar(scale)?;
            acc_dq = Some(match acc_dq {
                None => d_q_delta,
                Some(existing) => existing.add(&d_q_delta)?,
            });

            let d_k_delta = d_s.transpose_dims(1, 2)?.bmm(&q_slice)?.mul_scalar(scale)?;
            d_k_tiles[ik] = d_k_tiles[ik].add(&d_k_delta)?;
        }

        d_q_chunks.push(acc_dq.expect("chunked dQ accumulation missing"));
    }

    let d_q_refs: Vec<&Tensor> = d_q_chunks.iter().collect();
    let d_k_refs: Vec<&Tensor> = d_k_tiles.iter().collect();
    let d_v_refs: Vec<&Tensor> = d_v_tiles.iter().collect();

    let d_q = Tensor::cat(&d_q_refs, 1)?.reshape(&[bh, q_len, d])?;
    let d_k = Tensor::cat(&d_k_refs, 1)?.reshape(&[bh, k_len, d])?;
    let d_v = Tensor::cat(&d_v_refs, 1)?.reshape(&[bh, k_len, d])?;

    Ok((d_q, d_k, d_v))
}

fn apply_causal_mask_inplace(
    logits: &mut Tensor,
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    q_offset: usize,
    k_offset: usize,
) -> Result<()> {
    let slice = logits
        .storage_mut()
        .try_as_mut_slice_f32()
        .map_err(|_| Error::InvalidOperation("SDPA v4 requires logits in FP32 storage".into()))?;
    let ptr = *slice.device_ptr_mut() as *mut f32;
    let stream = logits.device().cuda_stream_raw_ptr();
    let status = unsafe {
        crate::cuda::ffi::flame_apply_causal_mask_fp32(
            ptr,
            batch as i32,
            heads as i32,
            q_len as i32,
            k_len as i32,
            q_offset as i32,
            k_offset as i32,
            stream,
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "flame_apply_causal_mask_fp32 failed with status {}",
            status
        )));
    }
    Ok(())
}

pub fn sdpa_backward(ctx: &SdpaCtx, d_y: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "sdpa_debug")]
    if let Some(hooks) = debug_hooks() {
        (hooks.on_bwd_start)();
    }
    #[cfg(feature = "sdpa_debug")]
    let debug_start = Instant::now();

    let (batch, heads, q_len, k_len) = ctx.shapes;
    let head_dim = ctx.q.shape().dims()[3];
    let bh = batch * heads;
    #[cfg(feature = "sdpa_debug")]
    let (debug_chunk_q, debug_chunk_kv) = ctx.chunk_cfg.unwrap_or((q_len, k_len));
    #[cfg(feature = "sdpa_debug")]
    let debug_tiles_q = (q_len + debug_chunk_q - 1) / debug_chunk_q;
    #[cfg(feature = "sdpa_debug")]
    let debug_tiles_k = (k_len + debug_chunk_kv - 1) / debug_chunk_kv;
    #[cfg(feature = "sdpa_debug")]
    let debug_mask_kind = if ctx.attn_mask.is_some() {
        "f32_add"
    } else {
        "none"
    };

    let q_flat = ctx.q.reshape(&[bh, q_len, head_dim])?;
    let k_flat = ctx.k.reshape(&[bh, k_len, head_dim])?;
    let v_flat = ctx.v.reshape(&[bh, k_len, head_dim])?;

    let q_f32 = q_flat.to_dtype(DType::F32)?;
    let k_f32 = k_flat.to_dtype(DType::F32)?;
    let v_f32 = v_flat.to_dtype(DType::F32)?;

    let dy_flat = d_y.reshape(&[bh, q_len, head_dim])?;
    let dy_f32 = dy_flat.to_dtype(DType::F32)?;

    let mask_flat = if let Some(ref mask) = ctx.attn_mask {
        Some(mask.reshape(&[bh, q_len, k_len])?)
    } else {
        None
    };

    if let Some((chunk_q, chunk_kv)) = ctx.chunk_cfg {
        let lse_flat = ctx
            .lse
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("chunked SDPA missing stored LSE".into()))?
            .reshape(&[bh, q_len, 1])?
            .to_dtype(DType::F32)?;
        let (d_q_flat, d_k_flat, d_v_flat) = backward_chunked_savelse(
            &q_f32,
            &k_f32,
            &v_f32,
            &lse_flat,
            &dy_f32,
            mask_flat.as_ref(),
            ctx.scale,
            chunk_q,
            chunk_kv,
            ctx.causal,
            ctx.dropout_p,
            ctx.dropout_seed,
        )?;

        let mut d_q = d_q_flat.reshape(&[batch, heads, q_len, head_dim])?;
        let mut d_k = d_k_flat.reshape(&[batch, heads, k_len, head_dim])?;
        let mut d_v = d_v_flat.reshape(&[batch, heads, k_len, head_dim])?;

        d_q.requires_grad = false;
        d_k.requires_grad = false;
        d_v.requires_grad = false;

        #[cfg(feature = "sdpa_debug")]
        if let Some(hooks) = debug_hooks() {
            (hooks.on_bwd_end)(&SdpaStats::default());
        }

        #[cfg(feature = "sdpa_debug")]
        {
            let elapsed_ms = debug_start.elapsed().as_secs_f64() * 1_000.0;
            debug!(
                "sdpa_v4 backward: chunked=true, chunk_q={}, chunk_kv={}, tiles_q={}, tiles_k={}, dropout_p={:.3}, mask={}, causal={}, elapsed_ms={:.3}",
                chunk_q,
                chunk_kv,
                debug_tiles_q,
                debug_tiles_k,
                ctx.dropout_p,
                debug_mask_kind,
                ctx.causal,
                elapsed_ms
            );
        }

        return Ok((d_q, d_k, d_v));
    }

    let probs = match ctx.save {
        SdpaSave::SaveProbs => {
            let stored = ctx.probs.as_ref().ok_or_else(|| {
                Error::InvalidOperation("SDPA ctx missing stored probabilities".into())
            })?;
            let probs = stored.reshape(&[bh, q_len, k_len])?.to_dtype(DType::F32)?;
            probs
        }
        SdpaSave::SaveLSE => {
            let mut logits = compute_logits(&q_f32, &k_f32, ctx.scale)?;
            if let Some(ref mask_bias) = mask_flat {
                logits = logits.add(mask_bias)?;
            }
            if ctx.causal {
                apply_causal_mask_inplace(&mut logits, batch, heads, q_len, k_len, 0, 0)?;
            }
            let lse = ctx
                .lse
                .as_ref()
                .ok_or_else(|| Error::InvalidOperation("SDPA ctx missing stored LSE".into()))?
                .reshape(&[bh, q_len, 1])?
                .to_dtype(DType::F32)?;
            let probs = logits.sub(&lse)?.exp()?;
            probs
        }
        SdpaSave::Recompute => {
            let mut logits = compute_logits(&q_f32, &k_f32, ctx.scale)?;
            if let Some(ref mask_bias) = mask_flat {
                logits = logits.add(mask_bias)?;
            }
            if ctx.causal {
                apply_causal_mask_inplace(&mut logits, batch, heads, q_len, k_len, 0, 0)?;
            }
            let (probs, _) = compute_probabilities(&logits)?;
            probs
        }
    };

    let mut probs = probs;
    if ctx.dropout_p > 0.0 {
        probs = apply_dropout_tile(&probs, ctx.dropout_p, ctx.dropout_seed, 0, 0)?;
    }

    let mut d_v = probs.transpose_dims(1, 2)?.bmm(&dy_f32)?;

    let v_transpose = v_f32.transpose_dims(1, 2)?;
    let mut d_p = dy_f32.bmm(&v_transpose)?;
    let dp_mul_p = d_p.mul(&probs)?;
    let sum_dp = GpuOps::sum_dim_keepdim(&dp_mul_p, 2)?;
    let scaled_probs = probs.mul(&sum_dp)?;
    let mut d_s = d_p.sub(&scaled_probs)?;

    let mut d_q = d_s.bmm(&k_f32)?.mul_scalar(ctx.scale)?;
    let mut d_k = d_s
        .transpose_dims(1, 2)?
        .bmm(&q_f32)?
        .mul_scalar(ctx.scale)?;

    d_q = d_q.reshape(&[batch, heads, q_len, head_dim])?;
    d_k = d_k.reshape(&[batch, heads, k_len, head_dim])?;
    d_v = d_v.reshape(&[batch, heads, k_len, head_dim])?;

    d_q.requires_grad = false;
    d_k.requires_grad = false;
    d_v.requires_grad = false;

    #[cfg(feature = "sdpa_debug")]
    if let Some(hooks) = debug_hooks() {
        (hooks.on_bwd_end)(&SdpaStats::default());
    }

    #[cfg(feature = "sdpa_debug")]
    {
        let elapsed_ms = debug_start.elapsed().as_secs_f64() * 1_000.0;
        debug!(
            "sdpa_v4 backward: chunked=false, chunk_q={}, chunk_kv={}, tiles_q={}, tiles_k={}, dropout_p={:.3}, mask={}, causal={}, elapsed_ms={:.3}",
            debug_chunk_q,
            debug_chunk_kv,
            debug_tiles_q,
            debug_tiles_k,
            ctx.dropout_p,
            debug_mask_kind,
            ctx.causal,
            elapsed_ms
        );
    }

    Ok((d_q, d_k, d_v))
}

fn compute_logits(q: &Tensor, k: &Tensor, scale: f32) -> Result<Tensor> {
    let kt = k.transpose_dims(1, 2)?;
    let logits = q.bmm(&kt)?;
    logits.mul_scalar(scale)
}

fn compute_probabilities(logits: &Tensor) -> Result<(Tensor, Tensor)> {
    let max = GpuOps::max_dim(logits, 2, true)?;
    let shifted = logits.sub(&max)?;
    let exp_shifted = shifted.exp()?;
    let sum = GpuOps::sum_dim_keepdim(&exp_shifted, 2)?;
    let probs = exp_shifted.div(&sum)?;
    let log_sum = sum.log()?;
    let lse = max.add(&log_sum)?;
    Ok((probs, lse))
}

fn prepare_mask(
    mask: Option<&Tensor>,
    shapes: (usize, usize, usize, usize),
) -> Result<Option<Tensor>> {
    let (batch, heads, q_len, k_len) = shapes;
    if mask.is_none() {
        return Ok(None);
    }
    let mask = mask.unwrap();
    let target_shape = Shape::from_dims(&[batch, heads, q_len, k_len]);
    let broadcasted = if mask.shape().dims() == target_shape.dims() {
        mask.clone()
    } else {
        mask.broadcast_to(&target_shape)?
    };

    let additive = match mask.dtype() {
        DType::Bool => {
            let mask_f32 = broadcasted.to_dtype(DType::F32)?;
            let ones = mask_f32.full_like(1.0)?;
            let complement = ones.sub(&mask_f32)?;
            complement.mul_scalar(NEG_INF)?
        }
        DType::F32 => broadcasted.to_dtype(DType::F32)?,
        other => {
            return Err(Error::InvalidInput(format!(
                "unsupported mask dtype {:?}; expected Bool or F32",
                other
            )));
        }
    };
    Ok(Some(additive))
}

#[cfg(feature = "sdpa_debug")]
fn debug_hooks() -> Option<&'static SdpaHooks> {
    SDPA_HOOKS.get()
}
