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
const CUDNN_SDPA_BWD_SEQ_ALIGN: usize = 128;

fn allow_cudnn_sdpa_bwd_causal() -> bool {
    std::env::var("FLAME_CUDNN_SDPA_BWD_CAUSAL")
        .ok()
        .as_deref()
        == Some("1")
}

fn trace_sdpa_dispatch_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| parse_env_flag("FLAME_SDPA_DISPATCH_TRACE").unwrap_or(false))
}

fn trace_sdpa_dispatch(args: std::fmt::Arguments<'_>) {
    if trace_sdpa_dispatch_enabled() {
        eprintln!("[sdpa-dispatch] {args}");
    }
}

#[inline]
fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        value
    } else {
        value.div_ceil(align) * align
    }
}

fn causal_keep_mask(
    q_len: usize,
    k_len: usize,
    q_start: usize,
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
    dtype: DType,
) -> SdpaResult<Tensor> {
    let mut data = vec![0.0f32; q_len * k_len];
    for q_idx in 0..q_len {
        let cutoff = (q_start + q_idx).min(k_len.saturating_sub(1));
        let row = q_idx * k_len;
        for k_idx in 0..=cutoff {
            data[row + k_idx] = 1.0;
        }
    }
    Tensor::from_vec_dtype(data, Shape::from_dims(&[1, 1, q_len, k_len]), device, dtype)
}

fn maybe_pad_for_cudnn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
) -> SdpaResult<(
    Tensor,
    Tensor,
    Tensor,
    Option<usize>,
    Option<(usize, usize)>,
)> {
    if causal && !allow_cudnn_sdpa_bwd_causal() {
        return Ok((q.clone(), k.clone(), v.clone(), None, None));
    }

    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        return Ok((q.clone(), k.clone(), v.clone(), None, None));
    }

    let q_len = q_dims[2];
    let k_len = k_dims[2];
    let v_len = v_dims[2];
    let head_dim = q_dims[3];
    if q_dims[1] != k_dims[1] || k_dims[1] != v_dims[1] {
        return Ok((q.clone(), k.clone(), v.clone(), None, None));
    }
    if k_len != v_len
        || q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || !(head_dim == 64 || head_dim == 96 || head_dim == 128)
    {
        return Ok((q.clone(), k.clone(), v.clone(), None, None));
    }
    if causal && q_len != k_len {
        return Ok((q.clone(), k.clone(), v.clone(), None, None));
    }
    if mask.is_some() {
        return Ok((q.clone(), k.clone(), v.clone(), None, None));
    }

    let target_q_len = align_up(q_len, CUDNN_SDPA_BWD_SEQ_ALIGN);
    let target_kv_len = align_up(k_len, CUDNN_SDPA_BWD_SEQ_ALIGN);
    if target_q_len == q_len && target_kv_len == k_len {
        return Ok((q.clone(), k.clone(), v.clone(), None, None));
    }

    let q_work = if target_q_len > q_len {
        let q_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[q_dims[0], q_dims[1], target_q_len - q_len, q_dims[3]]),
            q.dtype(),
            q.device().clone(),
        )?;
        Tensor::cat(&[q, &q_pad], 2)?
    } else {
        q.clone()
    };
    let k_work = if target_kv_len > k_len {
        let k_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[k_dims[0], k_dims[1], target_kv_len - k_len, k_dims[3]]),
            k.dtype(),
            k.device().clone(),
        )?;
        Tensor::cat(&[k, &k_pad], 2)?
    } else {
        k.clone()
    };
    let v_work = if target_kv_len > v_len {
        let v_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[v_dims[0], v_dims[1], target_kv_len - v_len, v_dims[3]]),
            v.dtype(),
            v.device().clone(),
        )?;
        Tensor::cat(&[v, &v_pad], 2)?
    } else {
        v.clone()
    };
    Ok((
        q_work,
        k_work,
        v_work,
        if target_q_len > q_len { Some(q_len) } else { None },
        Some((q_len, k_len)),
    ))
}

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

// cuDNN v9 Flash SDPA is the default and only flash path for unmasked
// head_dim ∈ {64, 96, 128} on both the inference and training paths.
// Inference goes through `forward_cudnn_sdpa_bf16` (fwd only); training
// goes through `forward_cudnn_sdpa_train_bf16` (fwd + Stats emit) and
// `flame_cudnn_sdpa_bwd_bf16` on backward. The in-tree WMMA forward
// (`flame_flash_attention_bf16` in `flash_attention_fwd.cu`) is retained
// solely as the reference kernel for `tests/cudnn_sdpa_parity.rs`. It
// is NOT on the Rust dispatch path — `forward_flash_bf16` was deleted
// in Phase 2c follow-up (2026-04-23) after Phase 2c made it unreachable.
// If cuDNN fails, the error surfaces; there is no silent fall-through
// to WMMA.

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

fn stream_threshold_from_env() -> Option<usize> {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("FLAME_SDPA_STREAM_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
    })
}

/// Per-Q-tile scores budget for the materialized SDPA fallback, expressed in
/// **F32 elements** (the dtype of the live softmax-staging buffer). The
/// materialized path peaks at roughly 8 bytes/score-elem (one live BF16
/// logits buffer + one live F32 upcast/softmax buffer), so a budget of N
/// F32 elements ≈ 8*N bytes peak per tile.
///
/// Default is 256 MiB worth of F32 (`256*1024*1024/4 = 67_108_864` elems),
/// which auto-tiles the head_dim-256 DiT shapes (Ideogram-4 1024²: BH=18,
/// Q=K=4114 → ~305 M score elems → ~1.22 GB single-shot F32 → OOM on a 24 GB
/// card once weights/latents are resident) into ~5 tiles of ~268 MB each.
/// Overridable via `FLAME_SDPA_MATERIALIZE_BUDGET_MB` (megabytes of F32
/// scores per tile). The default already auto-tiles — no env required.
fn materialize_budget_elems_from_env() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        const DEFAULT_BUDGET_MB: usize = 256;
        let mb = std::env::var("FLAME_SDPA_MATERIALIZE_BUDGET_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_BUDGET_MB);
        // MiB of F32 -> F32 element count. (mb << 20) / 4 == mb << 18.
        (mb << 20) / std::mem::size_of::<f32>()
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
    // When autograd is recording and any input requires grad, use the
    // training path which records a single Op::FlashAttention node.
    // Without this, the attention output has requires_grad=false and the
    // entire gradient chain through Q/K/V is broken — every trainer that
    // calls sdpa::forward (or attention::sdpa which delegates here) gets
    // zero gradients for attention LoRA params.
    if crate::autograd::AutogradContext::is_recording()
        && (q.requires_grad || k.requires_grad || v.requires_grad)
    {
        return forward_train_inner(q, k, v, mask, false);
    }
    scope("sdpa.forward", GuardMode::env_default(), || {
        let output = forward_inner(q, k, v, mask, false)?;
        debug_assert_eq!(output.dtype(), DType::BF16);
        Ok(output)
    })
}

/// Scaled dot-product attention with a top-left causal mask.
///
/// Use this for causal self-attention instead of materializing a binary
/// lower-triangular mask and passing it to [`forward`]. The causal flag can
/// route through cuDNN train forward/backward; a materialized mask cannot.
pub fn forward_causal(q: &Tensor, k: &Tensor, v: &Tensor) -> SdpaResult<Tensor> {
    if crate::autograd::AutogradContext::is_recording()
        && (q.requires_grad || k.requires_grad || v.requires_grad)
    {
        return forward_train_inner(q, k, v, None, true);
    }
    scope("sdpa.forward_causal", GuardMode::env_default(), || {
        let output = forward_inner(q, k, v, None, true)?;
        debug_assert_eq!(output.dtype(), DType::BF16);
        Ok(output)
    })
}

/// Scaled dot-product self-attention where the prefix rows are causal and the
/// remaining rows attend to the full sequence.
///
/// Prefix-causal-full self-attention:
/// rows `[0, prefix_len)` use a top-left causal mask over the prefix, while
/// rows `[prefix_len, S)` use unmasked full attention over all keys.
///
/// Current training implementation records this as one autograd op. Forward is
/// computed as causal prefix plus an FA2 full pass sliced to suffix rows, while
/// backward recomputes against the exact materialized prefix-causal/full mask.
/// This avoids the production O1 K/V gradient collapse from taping two
/// independent SDPA nodes over shared K/V, and avoids cuDNN plan failures on
/// O1's non-aligned full/suffix shapes.
pub fn forward_prefix_causal_full(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix_len: usize,
) -> SdpaResult<Tensor> {
    scope(
        "sdpa.forward_prefix_causal_full",
        GuardMode::env_default(),
        || {
            let q_dims = q.shape().dims();
            let k_dims = k.shape().dims();
            let v_dims = v.shape().dims();
            if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
                return Err(Error::InvalidOperation(format!(
                    "sdpa.forward_prefix_causal_full expects [B,H,S,D] tensors, got q={:?} k={:?} v={:?}",
                    q_dims, k_dims, v_dims
                )));
            }
            let seq_len = q_dims[2];
            if k_dims[2] != seq_len || v_dims[2] != seq_len {
                return Err(Error::InvalidInput(format!(
                    "sdpa.forward_prefix_causal_full requires self-attention lengths, got q={} k={} v={}",
                    seq_len, k_dims[2], v_dims[2]
                )));
            }
            if prefix_len > seq_len {
                return Err(Error::InvalidInput(format!(
                    "sdpa.forward_prefix_causal_full prefix_len {} exceeds seq_len {}",
                    prefix_len, seq_len
                )));
            }
            if prefix_len == 0 {
                return forward(q, k, v, None);
            }
            if prefix_len == seq_len {
                return forward_causal(q, k, v);
            }

            let full_mask = || -> SdpaResult<Tensor> {
                let mask = prefix_causal_full_mask(q, prefix_len)?;
                forward(q, k, v, Some(&mask))
            };

            if std::env::var("FLAME_PREFIX_CAUSAL_FULL_STRUCTURED")
                .ok()
                .as_deref()
                == Some("1")
            {
                return prefix_causal_full_structured_forward(q, k, v, prefix_len, true);
            }

            if std::env::var("FLAME_PREFIX_CAUSAL_FULL_SUFFIX_ONLY")
                .ok()
                .as_deref()
                == Some("1")
            {
                return prefix_causal_full_structured_forward(q, k, v, prefix_len, false);
            }

            if std::env::var("FLAME_PREFIX_CAUSAL_FULL_FULL_MASK")
                .ok()
                .as_deref()
                == Some("1")
            {
                return full_mask();
            }

            if std::env::var("FLAME_PREFIX_CAUSAL_FULL_SUFFIX_MASKED")
                .ok()
                .as_deref()
                == Some("1")
            {
                return prefix_causal_full_suffix_masked_forward(q, k, v, prefix_len);
            }

            if crate::autograd::AutogradContext::is_recording()
                && (q.requires_grad || k.requires_grad || v.requires_grad)
            {
                let output = {
                    let _guard = crate::autograd::AutogradContext::no_grad();
                    prefix_causal_full_flash_forward(q, k, v, prefix_len)?
                };
                let out = output.requires_grad_(true);
                let scale = 1.0 / (q_dims[3] as f32).sqrt();
                crate::autograd::AutogradContext::record_op(
                    out.id(),
                    crate::autograd::Op::PrefixCausalFullAttention {
                        query: q.id(),
                        key: k.id(),
                        value: v.id(),
                        prefix_len,
                        scale,
                    },
                    vec![
                        (q.id(), q.clone()),
                        (k.id(), k.clone()),
                        (v.id(), v.clone()),
                    ],
                );
                return Ok(out);
            }

            prefix_causal_full_flash_forward(q, k, v, prefix_len)
        },
    )
}

fn prefix_causal_full_mask(q: &Tensor, prefix_len: usize) -> SdpaResult<Tensor> {
    let seq_len = q.shape().dims()[2];
    let mut mask_data = vec![1.0f32; seq_len * seq_len];
    for q_idx in 0..seq_len {
        if q_idx < prefix_len {
            let row = q_idx * seq_len;
            for k_idx in 0..seq_len {
                mask_data[row + k_idx] = if k_idx <= q_idx && k_idx < prefix_len {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
    Tensor::from_vec_dtype(
        mask_data,
        Shape::from_dims(&[1, 1, seq_len, seq_len]),
        q.device().clone(),
        DType::F32,
    )
}

fn prefix_causal_full_structured_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix_len: usize,
    full_then_slice: bool,
) -> SdpaResult<Tensor> {
    let seq_len = q.shape().dims()[2];
    let q_prefix = q.narrow(2, 0, prefix_len)?.contiguous()?;
    let k_prefix = k.narrow(2, 0, prefix_len)?.contiguous()?;
    let v_prefix = v.narrow(2, 0, prefix_len)?.contiguous()?;
    let out_prefix = forward_causal(&q_prefix, &k_prefix, &v_prefix)?;

    let out_suffix = if full_then_slice {
        let out_full = forward_train_inner(q, k, v, None, false)?;
        out_full.narrow(2, prefix_len, seq_len - prefix_len)?
    } else {
        let q_suffix = q.narrow(2, prefix_len, seq_len - prefix_len)?.contiguous()?;
        forward_train_inner(&q_suffix, k, v, None, false)?
    };
    Tensor::cat(&[&out_prefix, &out_suffix], 2)
}

fn prefix_causal_full_suffix_masked_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix_len: usize,
) -> SdpaResult<Tensor> {
    let seq_len = q.shape().dims()[2];
    let q_prefix = q.narrow(2, 0, prefix_len)?.contiguous()?;
    let k_prefix = k.narrow(2, 0, prefix_len)?.contiguous()?;
    let v_prefix = v.narrow(2, 0, prefix_len)?.contiguous()?;
    let out_prefix = forward_causal(&q_prefix, &k_prefix, &v_prefix)?;

    let suffix_len = seq_len - prefix_len;
    let q_suffix = q.narrow(2, prefix_len, suffix_len)?.contiguous()?;
    let mask = Tensor::from_vec_dtype(
        vec![1.0f32; suffix_len * seq_len],
        Shape::from_dims(&[1, 1, suffix_len, seq_len]),
        q.device().clone(),
        DType::F32,
    )?;
    let out_suffix = forward(&q_suffix, k, v, Some(&mask))?;
    Tensor::cat(&[&out_prefix, &out_suffix], 2)
}

fn prefix_causal_full_flash_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix_len: usize,
) -> SdpaResult<Tensor> {
    let seq_len = q.shape().dims()[2];
    let q_dims = q.shape().dims();
    trace_sdpa_dispatch(format_args!(
        "prefix_causal_full enter B={} H={} S={} D={} prefix={}",
        q_dims.get(0).copied().unwrap_or(0),
        q_dims.get(1).copied().unwrap_or(0),
        q_dims.get(2).copied().unwrap_or(0),
        q_dims.get(3).copied().unwrap_or(0),
        prefix_len
    ));
    let q_prefix = q.narrow(2, 0, prefix_len)?.contiguous()?;
    let k_prefix = k.narrow(2, 0, prefix_len)?.contiguous()?;
    let v_prefix = v.narrow(2, 0, prefix_len)?.contiguous()?;
    trace_sdpa_dispatch(format_args!(
        "prefix_causal_full prefix -> causal Q={} K={} D={}",
        prefix_len,
        prefix_len,
        q_dims.get(3).copied().unwrap_or(0)
    ));
    let out_prefix = forward_causal(&q_prefix, &k_prefix, &v_prefix)?;

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        if std::env::var("FLAME_PREFIX_CAUSAL_FULL_TRY_CUDNN")
            .ok()
            .as_deref()
            == Some("1")
        {
            match prefix_causal_full_cudnn_full_forward(q, k, v) {
                Ok(out_full) => {
                    let out_suffix = out_full.narrow(2, prefix_len, seq_len - prefix_len)?;
                    return Tensor::cat(&[&out_prefix, &out_suffix], 2);
                }
                Err(err) => {
                    log::warn!(
                        "sdpa_prefix_causal_full: padded cuDNN full pass failed ({}); trying FA2 fallback",
                        err
                    );
                }
            }
        }
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        let q_dims = q.shape().dims();
        let k_dims = k.shape().dims();
        if q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && q_dims.len() == 4
            && k_dims.len() == 4
            && q_dims[3] == k_dims[3]
            && (q_dims[3] == 64 || q_dims[3] == 96 || q_dims[3] == 128)
        {
            trace_sdpa_dispatch(format_args!(
                "prefix_causal_full suffix source -> FA2 full-pass B={} H={} Q={} K={} D={}",
                q_dims[0],
                q_dims[1],
                q_dims[2],
                k_dims[2],
                q_dims[3]
            ));
            let out_full =
                forward_fa2_bf16(q, k, v, q_dims[0], q_dims[1], q_dims[2], k_dims[2], q_dims[3], false)?;
            let out_suffix = out_full.narrow(2, prefix_len, seq_len - prefix_len)?;
            return Tensor::cat(&[&out_prefix, &out_suffix], 2);
        }
    }

    let suffix_len = seq_len - prefix_len;
    let q_suffix = q.narrow(2, prefix_len, suffix_len)?.contiguous()?;
    trace_sdpa_dispatch(format_args!(
        "prefix_causal_full suffix source -> masked fallback Q={} K={} D={}",
        suffix_len,
        seq_len,
        q_dims.get(3).copied().unwrap_or(0)
    ));
    let mask = Tensor::from_vec_dtype(
        vec![1.0f32; suffix_len * seq_len],
        Shape::from_dims(&[1, 1, suffix_len, seq_len]),
        q.device().clone(),
        DType::F32,
    )?;
    let out_suffix = forward(&q_suffix, k, v, Some(&mask))?;
    Tensor::cat(&[&out_prefix, &out_suffix], 2)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn prefix_causal_full_cudnn_full_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> SdpaResult<Tensor> {
    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::InvalidInput(
            "sdpa_prefix_causal_full: cuDNN full pass expects 4D tensors".into(),
        ));
    }
    if q_dims[1] != k_dims[1] || k_dims[1] != v_dims[1] {
        return Err(Error::Unsupported(
            "sdpa_prefix_causal_full: cuDNN full pass does not support GQA heads".into(),
        ));
    }
    let (b, h, q_len, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || !(head_dim == 64 || head_dim == 96 || head_dim == 128)
    {
        return Err(Error::Unsupported(
            "sdpa_prefix_causal_full: cuDNN full pass requires BF16 head_dim in {64,96,128}"
                .into(),
        ));
    }

    let (q_work, k_work, v_work, slice_q_len, padding_lens) =
        maybe_pad_for_cudnn(q, k, v, None, false)?;
    let work_q_dims = q_work.shape().dims();
    let work_k_dims = k_work.shape().dims();
    let (out, _stats) = forward_cudnn_sdpa_train_bf16(
        &q_work,
        &k_work,
        &v_work,
        b,
        h,
        work_q_dims[2],
        work_k_dims[2],
        head_dim,
        false,
        padding_lens,
    )?;
    if let Some(orig_q_len) = slice_q_len {
        out.narrow(2, 0, orig_q_len)
    } else if work_q_dims[2] != q_len {
        out.narrow(2, 0, q_len)
    } else {
        Ok(out)
    }
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
    forward_train_inner(q, k, v, mask, false)
}

fn forward_train_inner(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
) -> SdpaResult<Tensor> {
    scope("sdpa.forward_train", GuardMode::env_default(), || {
        if mask.is_some() && causal {
            return Err(Error::Unsupported(
                "sdpa.forward_train: combined mask + causal attention is not supported".into(),
            ));
        }
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
        let v_dims = v.shape().dims();
        if v_dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "sdpa.forward_train expects v shaped [B,H,K,D], got {:?}",
                v_dims
            )));
        }
        let scale = if head_dim > 0 {
            1.0 / (head_dim as f32).sqrt()
        } else {
            1.0
        };

        let (q_work, k_work, v_work, slice_q_len, padding_lens) =
            maybe_pad_for_cudnn(q, k, v, mask, causal)?;
        let q_ref = &q_work;
        let k_ref = &k_work;
        let v_ref = &v_work;
        let work_dims = q_ref.shape().dims();
        let work_k_dims = k_ref.shape().dims();

        // Run forward under no_grad. Primary path: cuDNN SDPA forward-train,
        // which produces both O and Stats (per-row LSE) in one call. Backward
        // (autograd.rs::Op::FlashAttention) picks up the Stats tensor and
        // feeds it to `flame_cudnn_sdpa_bwd_bf16` — 30-50× faster than the
        // decomposed-recompute path this was doing pre-Phase-2c. Unsupported
        // shapes (head_dim ∉ {64,96,128}, masked, non-BF16) fall through to
        // `forward_inner`; backward recomputes from scratch in that case.
        let mut used_cudnn = false;
        let (output, stats_tensor) = {
            let _guard = crate::autograd::AutogradContext::no_grad();
            let (b, h, sq, hd) = (work_dims[0], work_dims[1], work_dims[2], head_dim);

            if mask.is_none()
                && (hd == 64 || hd == 96 || hd == 128)
                && work_dims[1] == work_k_dims[1]
                && work_k_dims[1] == v_ref.shape().dims()[1]
                && (!causal || allow_cudnn_sdpa_bwd_causal())
                && q_ref.dtype() == DType::BF16
                && k_ref.dtype() == DType::BF16
                && v_ref.dtype() == DType::BF16
            {
                match forward_cudnn_sdpa_train_bf16(
                    q_ref,
                    k_ref,
                    v_ref,
                    b,
                    h,
                    sq,
                    work_k_dims[2],
                    hd,
                    causal,
                    padding_lens,
                ) {
                    Ok((o, stats)) => {
                        used_cudnn = true;
                        (o, Some(stats))
                    }
                    Err(e) => {
                        log::error!("cudnn SDPA train-fwd failed: {:?}", e);
                        return Err(e);
                    }
                }
            } else {
                (forward_inner(q_ref, k_ref, v_ref, mask, causal)?, None)
            }
        };

        let saved_q = if used_cudnn { q_ref } else { q };
        let saved_k = if used_cudnn { k_ref } else { k };
        let saved_v = if used_cudnn { v_ref } else { v };
        let saved_slice_q_len = if used_cudnn { slice_q_len } else { None };
        let saved_padding_lens = if used_cudnn { padding_lens } else { None };

        if saved_q.requires_grad() || saved_k.requires_grad() || saved_v.requires_grad() {
            let mut out = output;
            out = out.requires_grad_(true);

            let mut saved = vec![
                (saved_q.id(), saved_q.clone()),
                (saved_k.id(), saved_k.clone()),
                (saved_v.id(), saved_v.clone()),
                // Save output for fused backward
                (out.id(), out.clone()),
            ];
            // Save Stats for cuDNN backward. Shape [B*H, Nq] FP32 —
            // autograd.rs::try_cudnn_sdpa_backward looks for this exact
            // shape+dtype to decide whether to route to cuDNN.
            if let Some(ref stats) = stats_tensor {
                saved.push((stats.id(), stats.clone()));
            }
            let mask_id = if let Some(mask_tensor) = mask {
                if used_cudnn {
                    None
                } else {
                    saved.push((mask_tensor.id(), mask_tensor.clone()));
                    Some(mask_tensor.id())
                }
            } else {
                None
            };

            crate::autograd::AutogradContext::record_op(
                out.id(),
                crate::autograd::Op::FlashAttention {
                    query: saved_q.id(),
                    key: saved_k.id(),
                    value: saved_v.id(),
                    mask: mask_id,
                    scale,
                    causal,
                    padding_lens: saved_padding_lens,
                    // Stage 2 fix (2026-05-12): record saved-O and saved-Stats
                    // ids directly so the backward dispatch can use
                    // `fetch_saved(id)` instead of a shape-find heuristic.
                    // The shape-finder was broken because `fetch_saved`
                    // materializes non-contig views into fresh `TensorId`s,
                    // so the id-exclusion fired false-positive and Q was
                    // picked up as O — destroyed cuDNN flash-bwd's dO·O^T
                    // identity and produced grad_norm=inf.
                    output: Some(out.id()),
                    stats: stats_tensor.as_ref().map(|s| s.id()),
                },
                saved,
            );
            if let Some(orig_q_len) = saved_slice_q_len {
                out.narrow(2, 0, orig_q_len)
            } else {
                Ok(out)
            }
        } else {
            if let Some(orig_q_len) = saved_slice_q_len {
                output.narrow(2, 0, orig_q_len)
            } else {
                Ok(output)
            }
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

        if let Some(b) = bias {
            let bd = b.shape().dims();
            if bd.len() != 4 {
                return Err(Error::InvalidInput(format!(
                    "bias must be 4D [B|1, H|1, Q, K], got shape {:?}",
                    bd
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

/// Scaled dot-product attention with **attention sinks** (per-head learned
/// scalar logits attached as an extra "virtual key" column before softmax).
///
/// This variant is used by GPT-OSS and other StreamingLLM-style models that
/// prevent attention from collapsing onto early tokens by giving every
/// query a learned no-op "sink" target. The math is:
///
/// ```text
/// logits      = Q @ K^T * scale + mask_penalty   # [B, H, Q, K]
/// sink_logits = sinks.broadcast_to([B, H, Q, 1]) # [B, H, Q, 1]
/// all_logits  = cat([logits, sink_logits], -1)   # [B, H, Q, K+1]
/// attn        = softmax(all_logits, -1)          # [B, H, Q, K+1]
/// attn_kv     = attn[..., :K]                    # drop the sink column
/// out         = attn_kv @ V                      # [B, H, Q, D]
/// ```
///
/// Because the sink column participates in softmax normalization but has
/// no corresponding value, attention weights over real keys can sum to
/// less than 1.0 — the sink absorbs the residual probability mass.
///
/// Arguments:
/// - `q, k, v`: `[B, H, S, D]` BF16 (GQA not supported here; replicate KV
///   heads at the call site).
/// - `mask`: optional **keep-mask** `[*, *, Q, K]` (1.0 = attend, 0.0 = masked).
///   Same semantics as [`forward`]. Applied as additive `-inf` on logits.
/// - `sinks`: `[H]` BF16 (or F32). One learned scalar per head.
///
/// Softmax is computed in FP32; output is cast back to `q.dtype()`.
///
/// This is a correctness-first materialized fallback (cuBLASLt-free FP32
/// `bmm` path mirroring [`forward_with_bias`]). GPT-OSS shapes
/// (24 layers, ≤512 tokens) make perf-tuning a non-issue at this stage.
pub fn forward_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sinks: &Tensor,
) -> SdpaResult<Tensor> {
    scope("sdpa.forward_with_sinks", GuardMode::env_default(), || {
        let (bq, hq, q_len, d_q) = shape4(q)?;
        let (bk, hk, k_len, d_k) = shape4(k)?;
        let (bv, hv, v_len, d_v) = shape4(v)?;

        if !(bq == bk && bq == bv && hq == hk && hq == hv) {
            return Err(Error::InvalidInput(format!(
                "sdpa.forward_with_sinks batch/head mismatch: q={:?}, k={:?}, v={:?}",
                q.shape(),
                k.shape(),
                v.shape()
            )));
        }
        if !(d_q == d_k && d_q == d_v) {
            return Err(Error::InvalidInput(format!(
                "sdpa.forward_with_sinks embed mismatch: q_dim={} k_dim={} v_dim={}",
                d_q, d_k, d_v
            )));
        }
        if k_len != v_len {
            return Err(Error::InvalidInput(format!(
                "sdpa.forward_with_sinks sequence mismatch: k_len={} v_len={}",
                k_len, v_len
            )));
        }

        // Sinks: [H] one scalar per head.
        let sinks_dims = sinks.shape().dims();
        if sinks_dims.len() != 1 || sinks_dims[0] != hq {
            return Err(Error::InvalidInput(format!(
                "sdpa.forward_with_sinks: sinks shape {:?} must be [H={}]",
                sinks_dims, hq
            )));
        }

        // Mask shape validation (matches `forward` rules).
        if let Some(m) = mask {
            let dims = m.shape().dims();
            if dims.len() != 4 {
                return Err(Error::InvalidInput(format!(
                    "sdpa.forward_with_sinks: mask must be 4D [B,H,Q,K], got {:?}",
                    dims
                )));
            }
            let bm = dims[0];
            let hm = dims[1];
            let qm = dims[2];
            let km = dims[3];
            if !(bm == bq || bm == 1)
                || !(hm == hq || hm == 1)
                || !(qm == q_len || qm == 1)
                || km != k_len
            {
                return Err(Error::InvalidInput(format!(
                    "sdpa.forward_with_sinks: mask dims {:?} not broadcastable to [B={},H={},Q={},K={}]",
                    dims, bq, hq, q_len, k_len
                )));
            }
        }

        // Manual FP32 path. Mirrors `forward_with_bias` structure so the
        // numerics line up with the existing materialized fallback.
        let q32 = q.to_dtype(DType::F32)?;
        let k32 = k.to_dtype(DType::F32)?;
        let v32 = v.to_dtype(DType::F32)?;
        let sinks32 = if sinks.dtype() == DType::F32 {
            sinks.clone_result()?
        } else {
            sinks.to_dtype(DType::F32)?
        };

        let bh = bq * hq;
        let q3 = q32.reshape(&[bh, q_len, d_q])?;
        let k3 = k32.reshape(&[bh, k_len, d_q])?;
        let v3 = v32.reshape(&[bh, v_len, d_v])?;

        // [bh, q_len, k_len] = Q @ K^T
        let k_t3 = transpose_last2(&k3)?;
        let mut scores3 = q3.bmm(&k_t3)?;

        let scale = 1.0f32 / (d_q as f32).sqrt();
        scores3 = scores3.mul_scalar(scale)?;

        // [B, H, Q, K] for mask add and sink concat.
        let mut scores4 = scores3.reshape(&[bq, hq, q_len, k_len])?;

        if let Some(mask_raw) = mask {
            // Keep-mask → additive -inf on masked positions.
            let target = [bq, hq, q_len, k_len];
            let mask_f32 = if mask_raw.dtype() == DType::F32 {
                mask_raw.clone_result()?
            } else {
                mask_raw.to_dtype(DType::F32)?
            };
            let mask_bcast = if mask_f32.shape().dims() == target {
                mask_f32
            } else {
                mask_f32.broadcast_to(&Shape::from_dims(&target))?
            };
            let ones = full_like(&mask_bcast, 1.0)?;
            let complement = ones.sub(&mask_bcast)?;
            let penalty = complement.mul_scalar(NEG_INF)?;
            scores4 = scores4.add(&penalty)?;
        }

        // Build sink-logit slab [B, H, Q, 1] by broadcasting sinks [H] →
        // [1, H, 1, 1] → [B, H, Q, 1]. broadcast_to is a stride-0 view;
        // contiguous-ify so the cat sees a real tensor.
        let sink_view = sinks32
            .reshape(&[1, hq, 1, 1])?
            .broadcast_to(&Shape::from_dims(&[bq, hq, q_len, 1]))?;
        let sink_col = sink_view.contiguous()?;

        // Concat logits || sink along K dim → [B, H, Q, K+1].
        let all_logits = Tensor::cat(&[&scores4, &sink_col], 3)?;

        // Softmax over the extended K+1 axis. force F32 in case softmax
        // downcasts internally.
        let attn_full = all_logits.softmax(-1)?.to_dtype(DType::F32)?;

        // Drop the sink column. Both shapes here are F32 contiguous from
        // softmax; narrow gives a zero-copy view, so make it contiguous
        // before the next BMM (the FP32 bmm path expects contiguous mem).
        let attn_kv = attn_full.narrow(3, 0, k_len)?.contiguous()?;

        let attn3 = attn_kv.reshape(&[bh, q_len, k_len])?;
        let out3 = attn3.bmm(&v3)?;
        let output32 = out3.reshape(&[bq, hq, q_len, d_v])?;

        let out = if q.dtype() == DType::F32 {
            output32
        } else {
            output32.to_dtype(q.dtype())?
        };
        Ok(out)
    })
}

fn forward_inner(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
) -> SdpaResult<Tensor> {
    let (bq, hq, q_len, d_q) = shape4(q)?;
    let (bk, hk, k_len, d_k) = shape4(k)?;
    let (bv, hv, v_len, d_v) = shape4(v)?;

    let gqa_heads = hk == hv && hk > 0 && hq % hk == 0;
    let heads_compatible = (hq == hk && hq == hv) || (mask.is_none() && gqa_heads);
    if !(bq == bk && bq == bv && heads_compatible) {
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

    if mask.is_some() && causal {
        return Err(Error::Unsupported(
            "sdpa.forward: combined mask + causal attention is not supported".into(),
        ));
    }
    if (hq != hk || hq != hv) && !(q.dtype() == DType::BF16 && d_q == 128 && mask.is_none()) {
        return Err(Error::Unsupported(format!(
            "sdpa.forward: GQA requires the mask-free BF16 head_dim=128 PyTorch FlashAttention path, got q={:?}, k={:?}, v={:?}, mask={}",
            q.shape(),
            k.shape(),
            v.shape(),
            mask.is_some()
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
        // Each dim must equal the matching target OR be 1 (PyTorch
        // broadcasting). The materialized BF16 fallback path expands a
        // singleton via `broadcast_to(target_dims)` before reshape (see
        // `forward_bf16_fallback` mask_prepared block), so a `[B,1,1,K]`
        // padding mask broadcasts cleanly through. The cuDNN-flash path
        // is gated on `mask.is_none()`, so no concern there. The
        // streaming path (`cuda_ops_bf16::sdpa_stream_bf16`) is only
        // reached for shapes > 2 G elements and the masked broadcast
        // case at that scale is not exercised by any current trainer;
        // if a future caller hits it we'll need a streaming-path
        // materialization too.
        if !(dims[0] == bq || dims[0] == 1)
            || !(dims[1] == hq || dims[1] == 1)
            || !(dims[2] == q_len || dims[2] == 1)
            || !(dims[3] == k_len || dims[3] == 1)
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
            return forward_bf16(q, k, v, mask, bq, hq, q_len, k_len, d_q, causal);
        }
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    if !allow_sdpa_f32_fallback() {
        return Err(Error::Unsupported(
            "sdpa.forward: FP32 fallback disabled; inputs must remain BF16 on CUDA".into(),
        ));
    }

    forward_f32(q, k, v, mask, d_q, causal)
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
    causal: bool,
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
    if causal {
        let target_dims = scores.shape().dims().to_vec();
        let q_len = q.shape().dims()[2];
        let k_len = k.shape().dims()[2];
        let causal_mask = causal_keep_mask(q_len, k_len, 0, q.device().clone(), DType::F32)?;
        let causal_bcast = if causal_mask.shape().dims() != target_dims.as_slice() {
            causal_mask.broadcast_to(&Shape::from_dims(&target_dims))?
        } else {
            causal_mask.reshape(&target_dims)?
        };
        let ones = full_like(&causal_bcast, 1.0)?;
        let complement = ones.sub(&causal_bcast)?;
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
    causal: bool,
) -> SdpaResult<Tensor> {
    let scale = 1.0 / (d_q as f32).sqrt();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    let hkv = k_dims.get(1).copied().unwrap_or(h);
    let v_heads = v_dims.get(1).copied().unwrap_or(hkv);
    let is_gqa = h != hkv || h != v_heads;

    if is_gqa {
        if mask.is_none() && hkv == v_heads && hkv > 0 && h % hkv == 0 && d_q == 128 {
            trace_sdpa_dispatch(format_args!(
                "forward_bf16 -> PyTorch FlashAttention GQA HD128 B={} Hq={} Hkv={} Q={} K={} causal={}",
                b, h, hkv, q_len, k_len, causal
            ));
            return forward_fa2_bf16(q, k, v, b, h, q_len, k_len, d_q, causal);
        }
        return Err(Error::Unsupported(format!(
            "sdpa.forward: GQA BF16 path requires mask-free head_dim=128 with Hq multiple of Hkv, got Hq={} Hk={} Hv={} D={} mask={}",
            h,
            hkv,
            v_heads,
            d_q,
            mask.is_some()
        )));
    }

    // Single SDPA path: the in-tree FA2 WMMA kernel (`flash_attention_fwd.cu`).
    // The former opt-in libtorch/AOTI bridge has been removed — FA2 Phase 1.6
    // is the canonical fast path and there is no libtorch linkage anywhere in
    // flame-core. See FA2 kernel docs for perf characteristics.

    // cuDNN v9 Flash SDPA — the only flash path for unmasked head_dim ∈
    // {64, 96, 128}. Measured 12.1× faster than the previous in-tree WMMA
    // kernel at Klein's shape. If cuDNN returns an error at this path we
    // surface it — no silent fall-through to WMMA. Parity vs WMMA was
    // validated on Klein+Chroma shapes at cos_sim = 1.000000 / mean_rel ~4e-6
    // (see `flame-core/tests/cudnn_sdpa_parity.rs`).
    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    if (d_q == 64 || d_q == 96 || d_q == 128)
        && mask.is_none()
        && (!causal || allow_cudnn_sdpa_bwd_causal())
    {
        trace_sdpa_dispatch(format_args!(
            "forward_bf16 -> cuDNN B={} H={} Q={} K={} D={} causal={}",
            b, h, q_len, k_len, d_q, causal
        ));
        return forward_cudnn_sdpa_bf16(q, k, v, b, h, q_len, k_len, d_q, causal).map_err(|e| {
            log::error!("cudnn SDPA failed (no WMMA fallback): {:?}", e);
            e
        });
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    if causal
        && mask.is_none()
        && q_len == k_len
        && (d_q == 64 || d_q == 96 || d_q == 128)
    {
        trace_sdpa_dispatch(format_args!(
            "forward_bf16 -> FA2 causal B={} H={} Q={} K={} D={}",
            b, h, q_len, k_len, d_q
        ));
        return forward_fa2_bf16(q, k, v, b, h, q_len, k_len, d_q, true);
    }

    // Auto-stream policy: the materialized fallback allocates an FP32 scores
    // tensor of `B * H * Q * K` elements plus a BF16 copy, peaking at
    // ~8 B/elem. For LTX-2.3 second-pass self-attn (B=1, H=30, Q=K=11088)
    // that's 3.68 G elements ≈ 29 GB. Anything past ~2 G elements must go
    // through the tiled/streaming kernel or we OOM on a 24 GB card. The
    // streaming path handles `mask` (full self-attn masks included) via
    // `cuda_ops_bf16::sdpa_stream_bf16`, so we can route masked + unmasked
    // large shapes to it uniformly.
    //
    // Threshold is overridable via FLAME_SDPA_STREAM_THRESHOLD (elements,
    // default 2_000_000_000). The legacy FLAME_SDPA_FORCE_STREAM=1 still
    // forces the stream path for any shape.
    let force_stream = force_stream_sdpa();
    let auto_stream = {
        let threshold = stream_threshold_from_env().unwrap_or(2_000_000_000usize);
        b.saturating_mul(h)
            .saturating_mul(q_len)
            .saturating_mul(k_len)
            > threshold
    };
    if !force_stream && !auto_stream {
        trace_sdpa_dispatch(format_args!(
            "forward_bf16 -> materialized fallback B={} H={} Q={} K={} D={} mask={} causal={}",
            b,
            h,
            q_len,
            k_len,
            d_q,
            mask.is_some(),
            causal
        ));
        return forward_bf16_fallback(q, k, v, mask, b, h, q_len, k_len, d_q, scale, causal);
    }
    if auto_stream && !force_stream {
        log::debug!(
            "sdpa: auto-routing to stream path (B={} H={} Q={} K={} elems={})",
            b,
            h,
            q_len,
            k_len,
            b * h * q_len * k_len,
        );
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
        match cuda_ops_bf16::sdpa_stream_bf16(q, k, v, mask, chunk, causal, Some(scale)) {
            Ok(out) => {
                trace_sdpa_dispatch(format_args!(
                    "forward_bf16 -> stream B={} H={} Q={} K={} D={} chunk={} mask={} causal={}",
                    b,
                    h,
                    q_len,
                    k_len,
                    d_q,
                    chunk,
                    mask.is_some(),
                    causal
                ));
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
/// cuDNN v9 Flash SDPA path. Same input layout as `forward_flash_bf16`:
/// Q, K, V are [B, H, N, D] BF16 contiguous. The cuDNN shim accepts a
/// 4D layout, so we pass `B*H` as the batch and `1` as the head dim —
/// that's layout-equivalent because with H=1 the head stride collapses
/// and memory walk order matches the [B*H, N, D] WMMA path.
fn forward_cudnn_sdpa_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d_q: usize,
    causal: bool,
) -> SdpaResult<Tensor> {
    use crate::cuda::device_lt;

    // Stride refactor Phase 2b: thread per-tensor strides + offsets into the
    // cuDNN graph so the hot-path permute-before-attention doesn't have to
    // materialize. The cuDNN frontend SDPA op takes a 4D stride vector per
    // tensor; we use each tensor's logical strides interpreted as [B,H,N,D]
    // after collapsing B*H into the batch dim (see collapse note below).
    //
    // Collapse: we still pass B*H as batch / 1 as head to match the single
    // graph key we've validated. Strides are in [B*H, 1, N, D] layout — we
    // compute them by taking the tensor's real [B,H,N,D] strides and
    // combining B-stride + H-stride into the collapsed dimension. When Q is
    // a contiguous [B,H,N,D] BF16 tensor, H*stride_B + stride_H*1 == N*D,
    // matching the hardcoded path. For a `permute([0,2,1,3])` view
    // ([B,H,N,D] from an incoming [B,N,H,D] contiguous layout) the real
    // strides are (N*H*D, D, H*D, 1), and cuDNN handles the non-contiguous
    // walk directly.
    let q_strides_4d = tensor_strides_as_4d_bhnd(q)?;
    let k_strides_4d = tensor_strides_as_4d_bhnd(k)?;
    let v_strides_4d = tensor_strides_as_4d_bhnd(v)?;

    let bh = (b * h) as i32;
    let device = q.device();
    let stream = device_lt::stream_ptr(device)?;
    let scale = 1.0f32 / (d_q as f32).sqrt();

    let output = Tensor::empty_dtype(q.shape().clone(), DType::BF16, device.clone())?;
    let o_strides_4d = tensor_strides_as_4d_bhnd(&output)?;

    let q_ptr = q.as_device_ptr_bf16("cudnn_sdpa:q")? as *const core::ffi::c_void;
    let k_ptr = k.as_device_ptr_bf16("cudnn_sdpa:k")? as *const core::ffi::c_void;
    let v_ptr = v.as_device_ptr_bf16("cudnn_sdpa:v")? as *const core::ffi::c_void;
    let o_ptr = output.as_device_ptr_bf16("cudnn_sdpa:o")? as *mut core::ffi::c_void;

    // view_offset is in elements; the shim advances the pointer in elements
    // (BF16 = 2 bytes).
    let q_off = q.offset() as i64;
    let k_off = k.offset() as i64;
    let v_off = v.offset() as i64;
    let o_off = output.offset() as i64;

    // Pass native 4D [B, H, N, D] to cuDNN with the tensor's real strides.
    // cudnn_frontend's SDPA supports arbitrary 4D strides natively, so there
    // is no need to collapse B*H into the batch dim. This supports permute
    // views ([B,N,H,D] contiguous → permute(0,2,1,3) = strided [B,H,N,D])
    // without any materialization — the single biggest win of the stride
    // refactor for attention.
    let [s_b_q, s_h_q, s_n_q, s_d_q] = q_strides_4d;
    let [s_b_k, s_h_k, s_n_k, s_d_k] = k_strides_4d;
    let [s_b_v, s_h_v, s_n_v, s_d_v] = v_strides_4d;
    let [s_b_o, s_h_o, s_n_o, s_d_o] = o_strides_4d;
    let q_strides: [i64; 4] = [s_b_q, s_h_q, s_n_q, s_d_q];
    let k_strides: [i64; 4] = [s_b_k, s_h_k, s_n_k, s_d_k];
    let v_strides: [i64; 4] = [s_b_v, s_h_v, s_n_v, s_d_v];
    let o_strides: [i64; 4] = [s_b_o, s_h_o, s_n_o, s_d_o];
    let _ = bh; // kept for potential future diagnostic logging

    let ret = unsafe {
        crate::cuda::ffi::flame_cudnn_sdpa_bf16(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            b as i32,
            h as i32,
            q_len as i32,
            k_len as i32,
            d_q as i32,
            scale,
            q_strides.as_ptr(),
            k_strides.as_ptr(),
            v_strides.as_ptr(),
            o_strides.as_ptr(),
            q_off,
            k_off,
            v_off,
            o_off,
            if causal { 1 } else { 0 },
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("cudnn_sdpa CUDA error: {ret}")));
    }
    Ok(output)
}

/// cuDNN v9 Flash SDPA training-forward. Identical output to
/// `forward_cudnn_sdpa_bf16` but also emits Stats (per-row log-sum-exp,
/// contiguous FP32 `[B*H, Nq]`) which the autograd backward reads to skip
/// recomputing the forward in `flame_cudnn_sdpa_bwd_bf16`.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn forward_cudnn_sdpa_train_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d_q: usize,
    causal: bool,
    padding_lens: Option<(usize, usize)>,
) -> SdpaResult<(Tensor, Tensor)> {
    use crate::cuda::device_lt;
    use cudarc::driver::DevicePtr;

    let q_strides_4d = tensor_strides_as_4d_bhnd(q)?;
    let k_strides_4d = tensor_strides_as_4d_bhnd(k)?;
    let v_strides_4d = tensor_strides_as_4d_bhnd(v)?;

    let device = q.device();
    let stream = device_lt::stream_ptr(device)?;
    let scale = 1.0f32 / (d_q as f32).sqrt();

    let output = Tensor::empty_dtype(q.shape().clone(), DType::BF16, device.clone())?;
    let o_strides_4d = tensor_strides_as_4d_bhnd(&output)?;

    // Stats: contiguous FP32 [B*H, N_q]. The cuDNN shim writes this with a
    // 4D view [B, H, N_q, 1] stride [H*N_q, N_q, 1, 1], which is
    // bytewise-identical to a row-major 2D [B*H, N_q].
    let stats = Tensor::zeros_dtype(
        crate::Shape::from_dims(&[b * h, q_len]),
        DType::F32,
        device.clone(),
    )?;

    let q_ptr = q.as_device_ptr_bf16("cudnn_sdpa_train:q")? as *const core::ffi::c_void;
    let k_ptr = k.as_device_ptr_bf16("cudnn_sdpa_train:k")? as *const core::ffi::c_void;
    let v_ptr = v.as_device_ptr_bf16("cudnn_sdpa_train:v")? as *const core::ffi::c_void;
    let o_ptr = output.as_device_ptr_bf16("cudnn_sdpa_train:o")? as *mut core::ffi::c_void;
    let stats_ptr = match &stats.storage {
        crate::tensor_storage::TensorStorage::F32 { data, .. } => {
            *crate::tensor_storage::slice_ref(data).device_ptr() as *mut core::ffi::c_void
        }
        _ => {
            return Err(Error::InvalidOperation(
                "cudnn_sdpa_train: expected F32 storage for stats".into(),
            ))
        }
    };
    let q_off = q.offset() as i64;
    let k_off = k.offset() as i64;
    let v_off = v.offset() as i64;
    let o_off = output.offset() as i64;
    let stats_off = stats.offset() as i64;
    let (real_q_len, real_k_len) = padding_lens.unwrap_or((q_len, k_len));

    let q_strides: [i64; 4] = q_strides_4d;
    let k_strides: [i64; 4] = k_strides_4d;
    let v_strides: [i64; 4] = v_strides_4d;
    let o_strides: [i64; 4] = o_strides_4d;

    let ret = unsafe {
        crate::cuda::ffi::flame_cudnn_sdpa_bf16_train_fwd(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            stats_ptr,
            b as i32,
            h as i32,
            q_len as i32,
            k_len as i32,
            d_q as i32,
            scale,
            q_strides.as_ptr(),
            k_strides.as_ptr(),
            v_strides.as_ptr(),
            o_strides.as_ptr(),
            q_off,
            k_off,
            v_off,
            o_off,
            stats_off,
            if causal { 1 } else { 0 },
            real_q_len as i32,
            real_k_len as i32,
            stream,
        )
    };

    if ret != 0 {
        return Err(Error::Cuda(format!("cudnn_sdpa_train CUDA error: {ret}")));
    }
    Ok((output, stats))
}

/// Convert a tensor's strides to a fixed-length `[i64; 4]` in [B,H,N,D] order.
/// Accepts a 4-D tensor only.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn tensor_strides_as_4d_bhnd(t: &Tensor) -> SdpaResult<[i64; 4]> {
    let mut s = [0usize; 8];
    let rank = t.fill_strides_into(&mut s);
    if rank != 4 {
        return Err(Error::InvalidInput(format!(
            "cudnn_sdpa: expected 4D tensor, got rank {}",
            rank
        )));
    }
    Ok([s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64])
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn forward_pytorch_flash_hd128_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b: usize,
    hq: usize,
    hkv: usize,
    q_len: usize,
    k_len: usize,
    causal: bool,
) -> SdpaResult<Tensor> {
    use crate::cuda::device_lt;
    use cudarc::driver::DevicePtr;

    if hkv == 0 || hq % hkv != 0 {
        return Err(Error::InvalidInput(format!(
            "pytorch_flash_hd128: invalid head counts Hq={} Hkv={}",
            hq, hkv
        )));
    }

    let q_strides = tensor_strides_as_4d_bhnd(q)?;
    let k_strides = tensor_strides_as_4d_bhnd(k)?;
    let v_strides = tensor_strides_as_4d_bhnd(v)?;
    if q_strides[3] != 1 || k_strides[3] != 1 || v_strides[3] != 1 {
        return Err(Error::Unsupported(format!(
            "pytorch_flash_hd128 requires contiguous head dimension, got q={:?} k={:?} v={:?}",
            q_strides, k_strides, v_strides
        )));
    }

    let device = q.device();
    let stream = device_lt::stream_ptr(device)?;
    let output = Tensor::empty_dtype(q.shape().clone(), DType::BF16, device.clone())?;
    let o_strides = tensor_strides_as_4d_bhnd(&output)?;
    if o_strides[3] != 1 {
        return Err(Error::Unsupported(format!(
            "pytorch_flash_hd128 requires contiguous output head dimension, got {:?}",
            o_strides
        )));
    }

    let q_ptr = q.as_device_ptr_bf16("pytorch_flash_hd128:q")? as *const core::ffi::c_void;
    let k_ptr = k.as_device_ptr_bf16("pytorch_flash_hd128:k")? as *const core::ffi::c_void;
    let v_ptr = v.as_device_ptr_bf16("pytorch_flash_hd128:v")? as *const core::ffi::c_void;
    let o_ptr = output.as_device_ptr_bf16("pytorch_flash_hd128:o")? as *mut core::ffi::c_void;

    trace_sdpa_dispatch(format_args!(
        "forward_fa2_bf16 -> PyTorch FlashAttention HD128 B={} Hq={} Hkv={} Q={} K={} causal={}",
        b, hq, hkv, q_len, k_len, causal
    ));

    let ret = unsafe {
        crate::cuda::ffi::flame_pytorch_flash_attn_bf16_hd128(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            core::ptr::null_mut(),
            b as i32,
            hq as i32,
            hkv as i32,
            q_len as i32,
            k_len as i32,
            q_strides.as_ptr(),
            k_strides.as_ptr(),
            v_strides.as_ptr(),
            o_strides.as_ptr(),
            q.offset() as i64,
            k.offset() as i64,
            v.offset() as i64,
            output.offset() as i64,
            1.0 / 128.0f32.sqrt(),
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!(
            "pytorch_flash_hd128 CUDA error: {ret}"
        )));
    }
    Ok(output)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn forward_fa2_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    d_q: usize,
    causal: bool,
) -> SdpaResult<Tensor> {
    use crate::cuda::device_lt;

    if !(d_q == 64 || d_q == 96 || d_q == 128) {
        return Err(Error::Unsupported(format!(
            "FA2 BF16 forward supports head_dim 64/96/128, got {d_q}"
        )));
    }

    if d_q == 128 {
        let k_dims = k.shape().dims();
        let hkv = k_dims.get(1).copied().unwrap_or(h);
        return forward_pytorch_flash_hd128_bf16(q, k, v, b, h, hkv, q_len, k_len, causal);
    }

    let bh = b * h;
    trace_sdpa_dispatch(format_args!(
        "forward_fa2_bf16 ffi B={} H={} BH={} Q={} K={} D={} causal={}",
        b, h, bh, q_len, k_len, d_q, causal
    ));
    let q3 = q.reshape(&[bh, q_len, d_q])?;
    let k3 = k.reshape(&[bh, k_len, d_q])?;
    let v3 = v.reshape(&[bh, k_len, d_q])?;
    let out3 = Tensor::empty_dtype(
        Shape::from_dims(&[bh, q_len, d_q]),
        DType::BF16,
        q.device().clone(),
    )?;

    let q_ptr = q3.as_device_ptr_bf16("fa2_sdpa:q")? as *const core::ffi::c_void;
    let k_ptr = k3.as_device_ptr_bf16("fa2_sdpa:k")? as *const core::ffi::c_void;
    let v_ptr = v3.as_device_ptr_bf16("fa2_sdpa:v")? as *const core::ffi::c_void;
    let o_ptr = out3.as_device_ptr_bf16("fa2_sdpa:o")? as *mut core::ffi::c_void;
    let stream = device_lt::stream_ptr(q.device())?;

    let ret = unsafe {
        crate::cuda::ffi::flame_flash_attention_bf16(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            core::ptr::null_mut(),
            bh as i32,
            q_len as i32,
            k_len as i32,
            d_q as i32,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!("flame_flash_attention_bf16 error: {ret}")));
    }

    out3.reshape(&[b, h, q_len, d_q])
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
    causal: bool,
) -> SdpaResult<Tensor> {
    // Materialized SDPA: two batched BF16 GEMMs (FP32 acc) + FP32 softmax.
    //
    // **Q-TILING:** The scores tensor `[BH, Q, K]` can get large at
    // 1024² DiT shapes (Z-Image BH=30 Q=K=4096 → 500M F32 elements ≈ 2 GB
    // peak during softmax staging, OOM on 24 GB). We tile the Q dimension
    // so peak scores memory is bounded to `scores_tile_elems_max` F32
    // elements per tile (see `materialize_budget_elems_from_env`). Tiling
    // is correctness-preserving because softmax rows are independent —
    // each q row only attends to all K rows, never to other q rows. Output
    // tiles are concatenated at the end.
    //
    // Flash Attention keeps scores in FP32 SRAM; we emulate the precision
    // (upcast BF16 logits → FP32 for scale+mask+softmax → downcast back
    // to BF16 before the PV GEMM) while using cuBLASLt for the two
    // batched BMMs so the heavy math runs on tensor cores.
    //
    // VRAM per tile: `BH * Q_TILE * K * 4` bytes for the F32 softmax
    // staging (peak during softmax is ~8 bytes/elem because of one live
    // BF16 + one live F32 buffer simultaneously). The budget bounds the
    // per-tile F32 scores element count and is read from
    // `materialize_budget_elems_from_env()` — default 256 MiB of F32
    // (67 M elems ≈ 0.5 GB peak/tile), overridable via
    // `FLAME_SDPA_MATERIALIZE_BUDGET_MB`. The DEFAULT auto-tiles; no env
    // is required. Representative shapes and their tile counts at the
    // 67 M-elem default:
    //   Ideogram-4 1024² (HD256): BH=18, Q=K=4114 → 305 M elems → ~5 tiles
    //   Z-Image 1024²:     BH=30, Q=K=4096 → 503 M elems → ~8 tiles
    //   FLUX 1 DiT 1024²:  BH=24, Q=K=4608 → 510 M elems → ~8 tiles
    //   Klein 9B 1024²:    BH=32, Q=K=4608 → 679 M elems → ~11 tiles
    //   SDXL level1:       BH=10, Q=K=4096 → 168 M elems → ~3 tiles
    //   CLIP 77:           BH=12, Q=K=77   → 0.07 M elems → 1 tile
    //   LTX-2.3:           BH=32, Q=K=768  → 19 M elems   → 1 tile
    //
    // History: an earlier 768 M-elem budget kept every DiT shape in a
    // SINGLE tile, but at HD=256 (Ideogram-4) a 305 M-elem single shot
    // materializes a 1.22 GB F32 scores tensor (+0.6 GB BF16 logits +
    // softmax temp) and OOMs on a 24 GB card once weights/latents are
    // resident. The 256 MiB default bounds peak scores memory regardless
    // of head_dim while leaving the budget tunable up (e.g.
    // `FLAME_SDPA_MATERIALIZE_BUDGET_MB=3072`) for callers that prefer the
    // single-shot path and have the VRAM. Tiling is correctness-preserving:
    // each Q-tile attends to ALL K with a full per-row softmax, so tiles
    // are independent and the concatenated output is bit-identical to the
    // single-shot result (same GEMMs, same scale, same softmax — tiling
    // only partitions rows).
    //
    // Mask path: masked attention (CLIP causal) is always small enough
    // to fit in one tile, so the masked path divides the budget for the
    // extra F32 mask-tile materialization but otherwise behaves the same.
    let scores_tile_elems_max = materialize_budget_elems_from_env();

    let device = q.device().clone();
    let bh = b * h;
    let t0 = std::time::Instant::now();

    // Flatten [B, H, seq, d] → [BH, seq, d] for batched GEMM
    let q_flat = q.reshape(&[bh, q_len, d_q])?;
    let k_flat = k.reshape(&[bh, k_len, d_q])?;
    let v_flat = v.reshape(&[bh, k_len, d_q])?;
    let k_t = k_flat.transpose_dims(1, 2)?; // [BH, d, K]
    let t_reshape = t0.elapsed().as_millis();

    // Decide tile size for Q. Tiling applies to masked paths too: we
    // slice the mask along Q per tile (2026-05-23 fix to unblock Lens
    // 1024² self-attn, where mask=[1,1,1,K] and a non-tiled scores
    // materialization OOMs under tight memory pressure even though the
    // mask itself is small).
    let full_elems = bh * q_len * k_len;
    // For masked shapes, reduce the per-tile budget because the F32 mask
    // materialization adds ~bh*Q_tile*K_len*4 bytes on top of scores.
    // For masked paths we materialize a F32 mask tile in addition to the
    // BF16 scores tile + its F32 upcast + softmax temp. Aggregate peak is
    // ~4× the bare scores tile (BF16 + F32_cast + F32_softmax + F32_mask),
    // so divide by 8 to leave headroom for fragmentation on a 24 GB card
    // under tight memory.
    let tile_budget = if mask.is_some() {
        scores_tile_elems_max / 8
    } else {
        scores_tile_elems_max
    };
    let do_tile = full_elems > tile_budget;

    let q_tile = if do_tile {
        // Target: bh * q_tile * k_len ≤ tile_budget.
        let per_q = bh * k_len;
        (tile_budget / per_q).max(1).min(q_len)
    } else {
        q_len
    };
    let num_tiles = q_len.div_ceil(q_tile);

    if do_tile {
        log::debug!(
            "sdpa_tiled: bh={} q_len={} k_len={} d_q={} q_tile={} num_tiles={}",
            bh,
            q_len,
            k_len,
            d_q,
            q_tile,
            num_tiles
        );
    }

    // Run one Q chunk through the materialized pipeline. Returns `[bh, len, d_q]` BF16.
    let run_one_tile =
        |q_chunk: &Tensor,
         mask_slice: Option<&Tensor>,
         len: usize,
         q_start: usize|
         -> SdpaResult<Tensor> {
            // QK^T tile → [bh, len, K] via cuBLASLt (tensor cores)
            let logits_shape = Shape::from_dims(&[bh, len, k_len]);
            let mut logits_bf16 = Tensor::empty_dtype(logits_shape, DType::BF16, device.clone())?;
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
            if causal {
                let causal_mask =
                    causal_keep_mask(len, k_len, q_start, device.clone(), DType::F32)?;
                let causal_bcast =
                    causal_mask.broadcast_to(&Shape::from_dims(&[b, h, len, k_len]))?;
                let causal_tile = causal_bcast.reshape(&[bh, len, k_len])?;
                let ones = full_like(&causal_tile, 1.0)?;
                let complement = ones.sub(&causal_tile)?;
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
        run_one_tile(&q_flat, mask_prepared.as_ref(), q_len, 0)?
    } else {
        // Tiled path: iterate Q chunks, concat outputs on dim=1.
        // Per-tile mask prep: keep the source mask in its compact form
        // (may be Q-broadcast as `[*, *, 1, K]`) and only broadcast +
        // materialize at tile size. For full-Q masks (`[*, *, Q, K]`),
        // slice along the Q dim per tile.
        let mask_source_f32: Option<Tensor> = if let Some(mask_raw) = mask {
            let mf32 = if mask_raw.dtype() == DType::F32 {
                mask_raw.clone_result()?
            } else {
                mask_raw.to_dtype(DType::F32)?
            };
            Some(mf32)
        } else {
            None
        };
        let mut tile_outputs: Vec<Tensor> = Vec::with_capacity(num_tiles);
        let mut start = 0;
        while start < q_len {
            let len = (q_len - start).min(q_tile);
            let q_chunk = q_flat.narrow(1, start, len)?;
            let mask_tile: Option<Tensor> = if let Some(ref mf32) = mask_source_f32 {
                let dims = mf32.shape().dims();
                if dims.len() != 4 {
                    return Err(Error::InvalidInput(format!(
                        "sdpa_fallback tiled: mask must be 4D, got {:?}",
                        dims
                    )));
                }
                // Slice along Q if the mask has a full Q dim; otherwise
                // keep Q-broadcast as-is (will broadcast in the run_one_tile
                // prep below).
                let m_q = dims[2];
                let mq_slice = if m_q == q_len {
                    mf32.narrow(2, start, len)?
                } else if m_q == 1 {
                    mf32.clone_result()?
                } else {
                    return Err(Error::InvalidInput(format!(
                        "sdpa_fallback tiled: mask Q dim {} not broadcastable to {}",
                        m_q, q_len
                    )));
                };
                let target = [b, h, len, k_len];
                let bcast = if mq_slice.shape().dims() == target {
                    mq_slice
                } else {
                    mq_slice.broadcast_to(&Shape::from_dims(&target))?
                };
                Some(bcast.reshape(&[bh, len, k_len])?)
            } else {
                None
            };
            let out_tile = run_one_tile(&q_chunk, mask_tile.as_ref(), len, start)?;
            tile_outputs.push(out_tile);
            start += len;
        }
        let refs: Vec<&Tensor> = tile_outputs.iter().collect();
        Tensor::cat(&refs, 1)?
    };

    let total = t0.elapsed().as_millis();
    if do_tile {
        log::debug!(
            "[SDPA] tiled total={}ms (BH={} Q={} K={} d={} q_tile={} num_tiles={})",
            total,
            bh,
            q_len,
            k_len,
            d_q,
            q_tile,
            num_tiles
        );
    } else {
        log::debug!(
            "[SDPA] reshape={}ms total={}ms (BH={} Q={} K={} d={})",
            t_reshape,
            total,
            bh,
            q_len,
            k_len,
            d_q
        );
    }

    projected.reshape(&[b, h, q_len, d_q])
}
