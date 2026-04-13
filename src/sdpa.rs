#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda_ops_bf16;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda_ops_ffi::CudaStream;
use crate::device::CudaStreamRawPtrExt;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::ops::gemm_bf16::bmm_bf16_fp32acc_out;
use crate::staging::{borrow_bf16_arena_tensor, ArenaScratch};
use crate::tensor::contracts::trap_is_bf16;
use crate::{
    ops_ext::{full_like, shape4, transpose_last2},
    strict::{record_layout_fix, scope, GuardMode},
    DType, Error, Shape, Tensor,
};
type SdpaResult<T> = crate::Result<T>;

const NEG_INF: f32 = -1.0e9;

// Raw reader kept for the (rare) paths that need to check a flag that the
// cached helpers below don't cover. Prefer the cached accessors below.
fn parse_env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().and_then(|value| {
        let value = value.to_ascii_lowercase();
        match value.as_str() {
            "1" | "true" | "on" | "yes" => Some(true),
            "0" | "false" | "off" | "no" => Some(false),
            _ => None,
        }
    })
}

// Cached hot-path readers. `forward_bf16` is called once per
// SelfAttention / CrossAttention op, which is dozens to hundreds of calls
// per denoise step. Each of these used to be a fresh syscall.

/// Whether to run the in-tree flash attention kernel for head_dim ∈
/// {64, 96, 128} inputs without a mask.
///
/// **Default: true.** The kernel at `src/cuda/flash_attention_fwd.cu`
/// is a scalar FP32 dot-product implementation with 32×32 tiles — no
/// `wmma`, no tensor cores — and runs ~230 ms per Z-Image 1024² block
/// vs the theoretical ~3 ms. It's the worst flash implementation in
/// the tree, BUT it's the best option we have today because:
///
///   1. The cuBLASLt materialized fallback (`forward_bf16_fallback`)
///      is *faster per call* (87 ms on the same Z-Image shape on a
///      fresh allocator), but allocates a 2 GB F32 softmax staging
///      tensor that the cudarc mempool fragments across steps and
///      OOMs on a 24 GB card by step 2.
///   2. Q-tiling the fallback to 4×128M-elem chunks was measured at
///      410 ms/call — worse than the naive flash — because per-tile
///      allocator and kernel-launch overhead eats the memory savings.
///
/// The right fix is to rewrite this kernel with `wmma` / `mma.sync` on
/// SM_80+ so it hits tensor core throughput. That's ~1 day of careful
/// CUDA work. See `PERF_SDPA_FLASH_KERNEL.md` for the full diagnosis
/// and `PERF_SDPA_QTILE_ATTEMPT.md` for the Option-2 post-mortem.
///
/// Disable with `FLAME_NO_FLASH_ATTN=1` (e.g. when profiling the
/// fallback path on a bigger GPU where OOM isn't a concern).
#[inline]
fn use_flash_attn() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| !parse_env_flag("FLAME_NO_FLASH_ATTN").unwrap_or(false))
}

#[inline]
fn force_stream_sdpa() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| parse_env_flag("FLAME_SDPA_FORCE_STREAM").unwrap_or(false))
}

fn chunk_limit_from_env() -> Option<usize> {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("FLAME_SDPA_CHUNK_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
    })
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn allow_sdpa_f32_fallback() -> bool {
    if crate::strict::is_enabled() {
        return false;
    }
    match parse_env_flag("SDPA_NO_F32_FALLBACK") {
        Some(true) => false,
        Some(false) | None => true,
    }
}

pub fn forward(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> SdpaResult<Tensor> {
    // Fused wmma backward kernel: correct but ~1.4x slower than decomposed
    // path at small seq_len (1024). Needs profiling — likely the 7-stage
    // pipeline with per-stage __syncthreads is slower than 12 separate
    // fully-pipelined kernel launches at this size. May win at larger seq_len.
    // Enable with FLAME_FUSED_ATTN_BWD=1 for testing.
    if std::env::var("FLAME_FUSED_ATTN_BWD").ok().as_deref() == Some("1")
        && crate::autograd::AutogradContext::is_recording()
        && (q.requires_grad || k.requires_grad || v.requires_grad)
    {
        return forward_train(q, k, v, mask);
    }
    scope("sdpa.forward", GuardMode::env_default(), || {
        let output = forward_inner(q, k, v, mask)?;
        debug_assert_eq!(output.dtype(), DType::BF16);
        Ok(output)
    })
}

/// Fused SDPA entry point for training.
///
/// Forward uses the optimized BF16 SDPA implementation under `no_grad` and then
/// records a single `Op::FlashAttention` node so backward can recompute from
/// Q/K/V without taping the internal matmul/softmax ops.
pub fn forward_train(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> SdpaResult<Tensor> {
    scope("sdpa.forward_train", GuardMode::env_default(), || {
        let dims = q.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "sdpa.forward_train expects q shaped [B,H,Q,D], got {:?}",
                dims
            )));
        }
        let head_dim = dims[3];
        let k_dims = k.shape().dims();
        if k_dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "sdpa.forward_train expects k shaped [B,H,K,D], got {:?}",
                k_dims
            )));
        }
        let scale = if head_dim > 0 {
            1.0 / (head_dim as f32).sqrt()
        } else {
            1.0
        };

        // Run forward under no_grad using the wmma kernel, also capture LSE
        // for the fused backward kernel.
        let (output, lse_tensor) = {
            let _guard = crate::autograd::AutogradContext::no_grad();
            let (b, h, sq, hd) = (dims[0], dims[1], dims[2], head_dim);
            let sk = k_dims[2];
            let bh = (b * h) as i32;

            // Try wmma kernel path (needs HD=64/96/128, BF16)
            if (hd == 64 || hd == 96 || hd == 128)
                && q.dtype() == DType::BF16
                && k.dtype() == DType::BF16
                && v.dtype() == DType::BF16
            {
                use crate::cuda::device_lt;
                use cudarc::driver::DevicePtr;

                let out_tensor = Tensor::empty_dtype(q.shape().clone(), DType::BF16, q.device().clone())?;
                let lse = Tensor::zeros_dtype(
                    crate::Shape::from_dims(&[b * h, sq]),
                    DType::F32,
                    q.device().clone(),
                )?;

                // Reshape to [BH, N, HD]
                let q_3d = q.reshape(&[bh as usize, sq, hd])?;
                let k_3d = k.reshape(&[bh as usize, sk, hd])?;
                let v_3d = v.reshape(&[bh as usize, sk, hd])?;

                let q_ptr = q_3d.as_device_ptr_bf16("sdpa_train:q")? as *const core::ffi::c_void;
                let k_ptr = k_3d.as_device_ptr_bf16("sdpa_train:k")? as *const core::ffi::c_void;
                let v_ptr = v_3d.as_device_ptr_bf16("sdpa_train:v")? as *const core::ffi::c_void;
                let o_ptr = out_tensor.as_device_ptr_bf16("sdpa_train:o")? as *mut core::ffi::c_void;
                let lse_ptr = match &lse.storage {
                    crate::tensor_storage::TensorStorage::F32 { data, .. } =>
                        *crate::tensor_storage::slice_ref(data).device_ptr() as *mut f32,
                    _ => core::ptr::null_mut(),
                };

                let stream = device_lt::stream_ptr(q.device())?;
                let ret = unsafe {
                    crate::cuda::ffi::flame_flash_attention_bf16(
                        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
                        bh, sq as i32, sk as i32, hd as i32, stream,
                    )
                };
                if ret == 0 {
                    (out_tensor, Some(lse))
                } else {
                    // Fallback to decomposed forward
                    (forward_inner(q, k, v, mask)?, None)
                }
            } else {
                (forward_inner(q, k, v, mask)?, None)
            }
        };

        if q.requires_grad() || k.requires_grad() || v.requires_grad() {
            let mut out = output;
            out = out.requires_grad_(true);

            let mut saved = vec![
                (q.id(), q.clone()),
                (k.id(), k.clone()),
                (v.id(), v.clone()),
                // Save output for fused backward
                (out.id(), out.clone()),
            ];
            // Save LSE for fused backward kernel
            if let Some(ref lse) = lse_tensor {
                saved.push((lse.id(), lse.clone()));
            }
            let mask_id = if let Some(mask_tensor) = mask {
                saved.push((mask_tensor.id(), mask_tensor.clone()));
                Some(mask_tensor.id())
            } else {
                None
            };

            crate::autograd::AutogradContext::record_op(
                out.id(),
                crate::autograd::Op::FlashAttention {
                    query: q.id(),
                    key: k.id(),
                    value: v.id(),
                    mask: mask_id,
                    scale,
                    causal: false,
                },
                saved,
            );
            Ok(out)
        } else {
            Ok(output)
        }
    })
}

/// Scaled dot-product attention with an ADDITIVE float bias (not a binary mask).
///
/// This variant exists for T5-style attention where a learned relative position
/// bias is added to raw `Q @ K^T` logits before softmax. Unlike `forward()`,
/// which interprets `mask` as a binary keep-mask (multiplying `(1-mask) * -inf`),
/// this function adds `bias` DIRECTLY to the scores.
///
/// ## T5 reference (`modeling_t5.py`):
/// ```python
/// scores = torch.matmul(query_states, key_states.transpose(3, 2))  # raw, no scale
/// scores += position_bias_masked                                   # additive float bias
/// attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
/// attn_output = torch.matmul(attn_weights, value_states)
/// ```
///
/// Arguments:
/// - `q, k, v`: `[B, H, S, D]` BF16 or FP32
/// - `bias`: optional `[*, H|1, Q, K]` additive float tensor (broadcastable over B/H)
/// - `scale`: `None` → divide scores by `sqrt(d_q)` (standard SDPA);
///            `Some(s)` → multiply scores by `s` (pass `Some(1.0)` for T5 which
///            absorbs scaling into weight init and uses unscaled Q·K^T)
///
/// Softmax is always computed in FP32; output is cast back to `q.dtype()`.
pub fn forward_with_bias(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bias: Option<&Tensor>,
    scale: Option<f32>,
) -> SdpaResult<Tensor> {
    scope("sdpa.forward_with_bias", GuardMode::env_default(), || {
        let (bq, hq, q_len, d_q) = shape4(q)?;
        let (bk, hk, k_len, d_k) = shape4(k)?;
        let (bv, hv, v_len, d_v) = shape4(v)?;

        if !(bq == bk && bq == bv && hq == hk && hq == hv) {
            return Err(Error::InvalidInput(format!(
                "batch/head mismatch: q={:?}, k={:?}, v={:?}",
                q.shape(), k.shape(), v.shape()
            )));
        }
        if !(d_q == d_k && d_q == d_v) {
            return Err(Error::InvalidInput(format!(
                "embed mismatch: q_dim={} k_dim={} v_dim={}",
                d_q, d_k, d_v
            )));
        }
        if k_len != v_len {
            return Err(Error::InvalidInput(format!(
                "sequence mismatch: k_len={} v_len={}", k_len, v_len
            )));
        }

        if let Some(b) = bias {
            let bd = b.shape().dims();
            if bd.len() != 4 {
                return Err(Error::InvalidInput(format!(
                    "bias must be 4D [B|1, H|1, Q, K], got shape {:?}", bd
                )));
            }
            if !(bd[0] == bq || bd[0] == 1)
                || !(bd[1] == hq || bd[1] == 1)
                || bd[2] != q_len
                || bd[3] != k_len
            {
                return Err(Error::InvalidInput(format!(
                    "bias dims {:?} not broadcastable to [B,H,Q,K] = [{},{},{},{}]",
                    bd, bq, hq, q_len, k_len
                )));
            }
        }

        // Manual FP32 path: upcast → 3D reshape → raw GEMM → optional scale → add bias
        // → FP32 softmax → GEMM → downcast. We reshape to 3D explicitly because the 4D
        // `Tensor::bmm` path mis-dispatches by dtype after a `to_dtype` cast.
        let q32 = q.to_dtype(DType::F32)?;
        let k32 = k.to_dtype(DType::F32)?;
        let v32 = v.to_dtype(DType::F32)?;

        let bh = bq * hq;
        let q3 = q32.reshape(&[bh, q_len, d_q])?;
        let k3 = k32.reshape(&[bh, k_len, d_q])?;
        let v3 = v32.reshape(&[bh, v_len, d_v])?;

        // K^T (last two dims) → [bh, d_q, k_len]
        let k_t3 = transpose_last2(&k3)?;
        let mut scores3 = q3.bmm(&k_t3)?; // [bh, q_len, k_len]

        let scale_val = scale.unwrap_or_else(|| 1.0 / (d_q as f32).sqrt());
        if (scale_val - 1.0).abs() > f32::EPSILON {
            scores3 = scores3.mul_scalar(scale_val)?;
        }

        // Reshape scores back to 4D for bias broadcast, then to 3D for bmm.
        let mut scores4 = scores3.reshape(&[bq, hq, q_len, k_len])?;
        if let Some(bias_raw) = bias {
            let target_dims = scores4.shape().dims().to_vec();
            let expanded = if bias_raw.shape().dims() != target_dims.as_slice() {
                bias_raw.broadcast_to(&Shape::from_dims(&target_dims))?
            } else {
                bias_raw.reshape(&target_dims)?
            };
            let bias_f32 = expanded.to_dtype(DType::F32)?;
            scores4 = scores4.add(&bias_f32)?;
        }

        // softmax may downcast to BF16 internally; force back to F32 for the F32 BMM path.
        let attn4 = scores4.softmax(-1)?.to_dtype(DType::F32)?;
        let attn3 = attn4.reshape(&[bh, q_len, k_len])?;
        let out3 = attn3.bmm(&v3)?; // [bh, q_len, d_v]
        let output32 = out3.reshape(&[bq, hq, q_len, d_v])?;

        let out = if q.dtype() == DType::F32 {
            output32
        } else {
            output32.to_dtype(q.dtype())?
        };
        Ok(out)
    })
}

fn forward_inner(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> SdpaResult<Tensor> {
    let (bq, hq, q_len, d_q) = shape4(q)?;
    let (bk, hk, k_len, d_k) = shape4(k)?;
    let (bv, hv, v_len, d_v) = shape4(v)?;

    if !(bq == bk && bq == bv && hq == hk && hq == hv) {
        return Err(Error::InvalidInput(format!(
            "batch/head mismatch: q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if !(d_q == d_k && d_q == d_v) {
        return Err(Error::InvalidInput(format!(
            "embed mismatch: q_dim={} k_dim={} v_dim={}",
            d_q, d_k, d_v
        )));
    }
    if k_len != v_len {
        return Err(Error::InvalidInput(format!(
            "sequence mismatch: k_len={} v_len={}",
            k_len, v_len
        )));
    }

    if let Some(m) = mask {
        let dims = m.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "mask must be 4D [B,H,Q,K], got shape {:?}",
                dims
            )));
        }
        if !(dims[0] == bq || dims[0] == 1)
            || !(dims[1] == hq || dims[1] == 1)
            || dims[2] != q_len
            || dims[3] != k_len
        {
            return Err(Error::InvalidInput(format!(
                "mask dims {:?} not broadcastable to [B,H,Q,K] = [{},{},{},{}]",
                dims, bq, hq, q_len, k_len
            )));
        }
    }

    trap_is_bf16("sdpa.forward q", q)?;
    trap_is_bf16("sdpa.forward k", k)?;
    trap_is_bf16("sdpa.forward v", v)?;

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        if q.dtype() == DType::BF16 && k.dtype() == DType::BF16 && v.dtype() == DType::BF16 {
            return forward_bf16(q, k, v, mask, bq, hq, q_len, k_len, d_q);
        }
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    if !allow_sdpa_f32_fallback() {
        return Err(Error::Unsupported(
            "sdpa.forward: FP32 fallback disabled; inputs must remain BF16 on CUDA".into(),
        ));
    }

    forward_f32(q, k, v, mask, d_q)
}

#[cfg(feature = "autograd_v4")]
pub fn forward_v4(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: Option<f32>,
    chunk: Option<(usize, usize)>,
) -> SdpaResult<Tensor> {
    scope("sdpa.forward_v4", GuardMode::env_default(), || {
        let output = forward_v4_inner(q, k, v, mask, causal, scale, chunk)?;
        debug_assert_eq!(output.dtype(), DType::BF16);
        Ok(output)
    })
}

#[cfg(feature = "autograd_v4")]
fn forward_v4_inner(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: Option<f32>,
    chunk: Option<(usize, usize)>,
) -> SdpaResult<Tensor> {
    trap_is_bf16("sdpa.forward_v4 q", q)?;
    trap_is_bf16("sdpa.forward_v4 k", k)?;
    trap_is_bf16("sdpa.forward_v4 v", v)?;
    use crate::autograd_v4::{sdpa_forward, SdpaConfig, SdpaSave};

    let mut cfg = SdpaConfig::default();
    cfg.save = SdpaSave::SaveLSE;
    cfg.causal = causal;
    cfg.scale = scale;
    if let Some((cq, ck)) = chunk {
        cfg.chunk_q = Some(cq);
        cfg.chunk_kv = Some(ck);
    }

    let (output, _ctx) = sdpa_forward(q, k, v, mask, &cfg)?;
    Ok(output)
}

fn forward_f32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    d_q: usize,
) -> SdpaResult<Tensor> {
    let q32 = q.to_dtype(DType::F32)?;
    let k32 = k.to_dtype(DType::F32)?;
    let v32 = v.to_dtype(DType::F32)?;

    let k_t = transpose_last2(&k32)?;
    let mut scores = q32.bmm(&k_t)?;
    let scale = 1.0 / (d_q as f32).sqrt();
    scores = scores.mul_scalar(scale)?;

    if let Some(mask_raw) = mask {
        let target_dims = scores.shape().dims().to_vec();
        let expanded = if mask_raw.shape().dims() != target_dims.as_slice() {
            mask_raw.broadcast_to(&Shape::from_dims(&target_dims))?
        } else {
            mask_raw.reshape(&target_dims)?
        };
        let mask_f32 = expanded.to_dtype(DType::F32)?;
        let ones = full_like(&mask_f32, 1.0)?;
        let complement = ones.sub(&mask_f32)?;
        let penalty = complement.mul_scalar(NEG_INF)?;
        scores = scores.add(&penalty)?;
    }

    let attn = scores.softmax(-1)?;
    let output32 = attn.bmm(&v32)?;

    if q.dtype() == DType::F32 {
        Ok(output32)
    } else {
        output32.to_dtype(q.dtype())
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn forward_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d_q: usize,
) -> SdpaResult<Tensor> {
    let scale = 1.0 / (d_q as f32).sqrt();

    // Default path: Q-tiled cuBLASLt materialized fallback. Uses tensor
    // cores for the heavy math and the Q-tile keeps peak memory bounded
    // regardless of (Q, K) size. Beats the in-tree flash kernel by
    // ~10-30× because flash is a scalar FP32 kernel with no tensor cores.
    // Opt back into flash with `FLAME_USE_FLASH=1` once the kernel is
    // rewritten with `wmma`.
    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    if (d_q == 64 || d_q == 96 || d_q == 128) && mask.is_none() && use_flash_attn() {
        match forward_flash_bf16(q, k, v, b, h, q_len, k_len, d_q) {
            Ok(out) => return Ok(out),
            Err(e) => {
                log::warn!("flash_attention failed, falling back: {:?}", e);
            }
        }
    }

    let force_stream = force_stream_sdpa();
    if !force_stream {
        return forward_bf16_fallback(q, k, v, mask, b, h, q_len, k_len, d_q, scale);
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        let limit = chunk_limit_from_env().unwrap_or(2048);
        let chunk = std::cmp::max(1usize, q_len.min(limit));
        log::trace!(
            "sdpa_stream_bf16 launch: B={} H={} Q={} K={} Dh={} chunk={} (limit {})",
            b,
            h,
            q_len,
            k_len,
            d_q,
            chunk,
            limit
        );
        match cuda_ops_bf16::sdpa_stream_bf16(q, k, v, mask, chunk, false, Some(scale)) {
            Ok(out) => {
                let dims = out.shape().dims();
                if dims.len() != 4 || dims[0] != b || dims[1] != h || dims[2] != q_len {
                    return Err(Error::InvalidOperation(format!(
                        "sdpa_stream_bf16 produced unexpected shape {:?}, expected [{},{},{},{}]",
                        dims, b, h, q_len, d_q
                    )));
                }
                if dims[3] != d_q {
                    return Err(Error::InvalidOperation(format!(
                        "sdpa_stream_bf16 produced value dim {}, expected {}",
                        dims[3], d_q
                    )));
                }
                return Ok(out);
            }
            Err(Error::Unsupported(reason)) => {
                return Err(Error::Unsupported(format!(
                    "sdpa_stream_bf16 unsupported (no fallback): {}",
                    reason
                )))
            }
            Err(err) => return Err(err),
        }
    }

    Err(Error::Unsupported(
        "sdpa_stream_bf16 unavailable (no fallback)".into(),
    ))
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
/// Flash attention: single fused kernel for the entire attention computation.
/// Q, K, V must be [B, H, N, D] BF16, contiguous. D = 64, 96, or 128. No mask support.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn forward_flash_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d_q: usize,
) -> SdpaResult<Tensor> {
    use crate::cuda::device_lt;

    let bh = (b * h) as i32;
    let device = q.device();
    let stream = device_lt::stream_ptr(device)?;

    // Output tensor: same shape as Q. Flash attention fully writes every
    // element, so skip the zero init.
    let output = Tensor::empty_dtype(q.shape().clone(), DType::BF16, device.clone())?;

    let q_ptr = q.as_device_ptr_bf16("flash_attn:q")? as *const core::ffi::c_void;
    let k_ptr = k.as_device_ptr_bf16("flash_attn:k")? as *const core::ffi::c_void;
    let v_ptr = v.as_device_ptr_bf16("flash_attn:v")? as *const core::ffi::c_void;
    let o_ptr = output.as_device_ptr_bf16("flash_attn:o")? as *mut core::ffi::c_void;

    let ret = unsafe {
        crate::cuda::ffi::flame_flash_attention_bf16(
            q_ptr, k_ptr, v_ptr, o_ptr,
            core::ptr::null_mut(), // LSE: not needed for inference-only forward
            bh,
            q_len as i32,
            k_len as i32,
            d_q as i32,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("flash_attention CUDA error: {ret}")));
    }

    Ok(output)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn forward_bf16_fallback(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d_q: usize,
    scale: f32,
) -> SdpaResult<Tensor> {
    // Materialized SDPA: two batched BF16 GEMMs (FP32 acc) + FP32 softmax.
    //
    // **Q-TILING:** The scores tensor `[BH, Q, K]` can get large at
    // 1024² DiT shapes (Z-Image BH=30 Q=K=4096 → 500M F32 elements ≈ 2 GB
    // peak during softmax staging, OOM on 24 GB). We tile the Q dimension
    // so peak scores memory is bounded to `SCORES_TILE_ELEMS_MAX` F32
    // elements per tile. Tiling is correctness-preserving because softmax
    // rows are independent — each q row only attends to all K rows,
    // never to other q rows. Output tiles are concatenated at the end.
    //
    // Flash Attention keeps scores in FP32 SRAM; we emulate the precision
    // (upcast BF16 logits → FP32 for scale+mask+softmax → downcast back
    // to BF16 before the PV GEMM) while using cuBLASLt for the two
    // batched BMMs so the heavy math runs on tensor cores.
    //
    // VRAM per tile: `BH * Q_TILE * K * 4` bytes for the F32 softmax
    // staging (peak during softmax is ~8 bytes/elem because of one live
    // BF16 + one live F32 buffer simultaneously). Budget is set so that
    // every DiT shape we actually run fits in a SINGLE tile and takes
    // the fast non-tiled path; tiling only kicks in for genuinely huge
    // sequences (8k+ video transformers):
    //   Z-Image 1024²:     BH=30, Q=K=4096 → 503 M elems  → 1 tile ≈ 4 GB peak
    //   FLUX 1 DiT 1024²:  BH=24, Q=K=4608 → 510 M elems  → 1 tile
    //   Klein 9B 1024²:    BH=32, Q=K=4608 → 679 M elems  → 1 tile
    //   SDXL level1:       BH=10, Q=K=4096 → 168 M elems  → 1 tile
    //   CLIP 77:           BH=12, Q=K=77   → 0.07 M elems → 1 tile
    //   LTX-2.3:           BH=32, Q=K=768  → 19 M elems   → 1 tile
    //
    // Budget at 768 M elements (≈ 6 GB peak) matches the pre-existing
    // `use_materialized` threshold that this function replaced. An
    // earlier attempt with a 128 M budget caused Z-Image to slow down
    // from 230 ms/block (naive flash) to 410 ms/block (4 tiles of 126 M)
    // because per-tile allocator and kernel-launch overhead dominated
    // the memory savings — tile-at-any-size is measurably worse than
    // tile-only-when-necessary for the cuBLASLt materialized path.
    //
    // Mask path: masked attention (CLIP causal) is always small enough
    // to fit in one tile, so we keep the simple non-tiled path when a
    // mask is present.
    const SCORES_TILE_ELEMS_MAX: usize = 768_000_000;

    let device = q.device().clone();
    let bh = b * h;
    let t0 = std::time::Instant::now();

    // Flatten [B, H, seq, d] → [BH, seq, d] for batched GEMM
    let q_flat = q.reshape(&[bh, q_len, d_q])?;
    let k_flat = k.reshape(&[bh, k_len, d_q])?;
    let v_flat = v.reshape(&[bh, k_len, d_q])?;
    let k_t = k_flat.transpose_dims(1, 2)?; // [BH, d, K]
    let t_reshape = t0.elapsed().as_millis();

    // Decide tile size for Q. If a mask is present we bypass tiling and
    // take the single-shot path (masked shapes are small by construction).
    let full_elems = bh * q_len * k_len;
    let do_tile = mask.is_none() && full_elems > SCORES_TILE_ELEMS_MAX;

    let q_tile = if do_tile {
        // Target: bh * q_tile * k_len ≤ SCORES_TILE_ELEMS_MAX.
        let per_q = bh * k_len;
        (SCORES_TILE_ELEMS_MAX / per_q).max(1).min(q_len)
    } else {
        q_len
    };
    let num_tiles = q_len.div_ceil(q_tile);

    if do_tile {
        log::debug!(
            "sdpa_tiled: bh={} q_len={} k_len={} d_q={} q_tile={} num_tiles={}",
            bh, q_len, k_len, d_q, q_tile, num_tiles
        );
    }

    // Run one Q chunk through the materialized pipeline. Returns `[bh, len, d_q]` BF16.
    let run_one_tile = |q_chunk: &Tensor, mask_slice: Option<&Tensor>, len: usize|
        -> SdpaResult<Tensor>
    {
        // QK^T tile → [bh, len, K] via cuBLASLt (tensor cores)
        let logits_shape = Shape::from_dims(&[bh, len, k_len]);
        let mut logits_bf16 =
            Tensor::empty_dtype(logits_shape, DType::BF16, device.clone())?;
        bmm_bf16_fp32acc_out(q_chunk, &k_t, &mut logits_bf16, false, false)?;

        // Upcast to FP32 for scale / mask / softmax (Flash Attention precision).
        let logits_f32 = logits_bf16.to_dtype(DType::F32)?;
        drop(logits_bf16);
        let mut scores = logits_f32.mul_scalar(scale)?;

        if let Some(mask_tile) = mask_slice {
            // mask_tile comes in as `[bh, len, K]` F32 already prepared.
            let ones = full_like(mask_tile, 1.0)?;
            let complement = ones.sub(mask_tile)?;
            let penalty = complement.mul_scalar(NEG_INF)?;
            scores = scores.add(&penalty)?;
        }

        let attn = scores.softmax(-1)?;
        drop(scores);
        let attn_bf16 = if attn.dtype() != DType::BF16 {
            attn.to_dtype(DType::BF16)?
        } else {
            attn
        };

        // attn × V → [bh, len, d_q] via cuBLASLt
        let out_shape = Shape::from_dims(&[bh, len, d_q]);
        let mut projected = Tensor::empty_dtype(out_shape, DType::BF16, device.clone())?;
        bmm_bf16_fp32acc_out(&attn_bf16, &v_flat, &mut projected, false, false)?;
        Ok(projected)
    };

    let projected = if !do_tile {
        // Single-shot path: prepare the optional mask once, run one tile.
        let mask_prepared = if let Some(mask_raw) = mask {
            let target_dims = [b, h, q_len, k_len];
            let mask_f32 = if mask_raw.dtype() == DType::F32 {
                mask_raw.clone_result()?
            } else {
                mask_raw.to_dtype(DType::F32)?
            };
            let mask_bcast = if mask_f32.shape().dims() == target_dims {
                mask_f32
            } else {
                record_layout_fix("sdpa.mask_broadcast", mask_f32.shape());
                mask_f32.broadcast_to(&Shape::from_dims(&target_dims))?
            };
            Some(mask_bcast.reshape(&[bh, q_len, k_len])?)
        } else {
            None
        };
        run_one_tile(&q_flat, mask_prepared.as_ref(), q_len)?
    } else {
        // Tiled path: iterate Q chunks, concat outputs on dim=1.
        let mut tile_outputs: Vec<Tensor> = Vec::with_capacity(num_tiles);
        let mut start = 0;
        while start < q_len {
            let len = (q_len - start).min(q_tile);
            let q_chunk = q_flat.narrow(1, start, len)?;
            // No mask on the tiled path (gated above).
            let out_tile = run_one_tile(&q_chunk, None, len)?;
            tile_outputs.push(out_tile);
            start += len;
        }
        let refs: Vec<&Tensor> = tile_outputs.iter().collect();
        Tensor::cat(&refs, 1)?
    };

    let total = t0.elapsed().as_millis();
    if do_tile {
        log::info!(
            "[SDPA] tiled total={}ms (BH={} Q={} K={} d={} q_tile={} num_tiles={})",
            total, bh, q_len, k_len, d_q, q_tile, num_tiles
        );
    } else {
        log::info!(
            "[SDPA] reshape={}ms total={}ms (BH={} Q={} K={} d={})",
            t_reshape, total, bh, q_len, k_len, d_q
        );
    }

    projected.reshape(&[b, h, q_len, d_q])
}
