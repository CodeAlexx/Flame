//! Sliding-window causal keep-mask builder for SDPA.
//!
//! Used by GPT-OSS-style alternating-attention layers, where every other
//! transformer block restricts each query position to a fixed window of
//! recent keys (including itself).
//!
//! The mask is constructed on the host and uploaded once per sequence
//! length / window size pair. Same pattern as
//! [`crate::sdpa::causal_keep_mask`] (private). The output is a *keep-mask*
//! shaped `[1, 1, seq_len, seq_len]` where `1.0` means "attend" and `0.0`
//! means "masked out", suitable for [`crate::sdpa::forward`] (which
//! converts keep-masks to additive `-inf` penalties internally) and the
//! materialized BF16 fallback path.
//!
//! Semantics: position `q` attends to position `k` iff
//! `q.saturating_sub(window_size - 1) <= k <= q`.
//! In particular every diagonal entry (`k == q`) is always `1.0`, and the
//! pattern degenerates to a pure causal mask when `window_size >= seq_len`.

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::{DType, Error, Shape, Tensor};

/// Build a sliding-window causal keep-mask of shape `[1, 1, seq_len, seq_len]`.
///
/// `window_size` is the total window length *including* the query itself
/// (so `window_size = 1` means strictly diagonal / self-only attention,
/// and `window_size = 128` matches GPT-OSS, which lets each query look at
/// itself plus the previous 127 tokens).
///
/// Returns a tensor with the requested `dtype`. The caller is responsible
/// for broadcasting batch / head dims; the SDPA mask path accepts
/// `[*, *, Q, K]` with leading dims either matching or broadcastable.
///
/// `window_size = 0` is rejected; callers that want a pure causal mask
/// should instead pass `window_size >= seq_len`, which produces the
/// equivalent shape.
pub fn sliding_window_causal_keep_mask(
    seq_len: usize,
    window_size: usize,
    device: &Arc<CudaDevice>,
    dtype: DType,
) -> Result<Tensor, Error> {
    if seq_len == 0 {
        return Err(Error::InvalidInput(
            "sliding_window_causal_keep_mask: seq_len must be > 0".into(),
        ));
    }
    if window_size == 0 {
        return Err(Error::InvalidInput(
            "sliding_window_causal_keep_mask: window_size must be > 0".into(),
        ));
    }

    // Host-build: for each query row q, the keep window is
    //   [max(0, q - window_size + 1), q]   inclusive.
    // The remainder of the row (including all positions k > q) stays 0.
    let mut data = vec![0.0f32; seq_len * seq_len];
    for q in 0..seq_len {
        let lo = q.saturating_sub(window_size.saturating_sub(1));
        let row = q * seq_len;
        for k in lo..=q {
            data[row + k] = 1.0;
        }
    }

    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[1, 1, seq_len, seq_len]),
        device.clone(),
        dtype,
    )
}
