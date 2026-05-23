//! Unit tests for `flame_core::attention::sliding_window_causal_keep_mask`.
//!
//! These verify three properties:
//! 1. With `window_size >= seq_len`, the result is identical to a pure
//!    lower-triangular causal mask.
//! 2. For `window_size < seq_len`, distant past positions are zeroed out.
//! 3. The diagonal is always 1.0 (every query attends to itself).

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::attention::sliding_window_causal_keep_mask;
use flame_core::{DType, Device, Result};

fn to_host_f32(t: &flame_core::Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.to_vec()
}

#[test]
fn sliding_window_pure_causal_at_short_seq() -> Result<()> {
    // window_size = 128, seq_len = 64 → every causal pair is in-window,
    // so the result matches a pure lower-triangular mask.
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device().clone();
    let seq_len = 64;
    let window_size = 128;

    let mask = sliding_window_causal_keep_mask(seq_len, window_size, &cuda, DType::F32)?;
    let host = to_host_f32(&mask)?;
    assert_eq!(host.len(), seq_len * seq_len);

    for q in 0..seq_len {
        for k in 0..seq_len {
            let got = host[q * seq_len + k];
            let want = if k <= q { 1.0 } else { 0.0 };
            assert!(
                (got - want).abs() < 1e-6,
                "pure causal mismatch at q={} k={}: got={} want={}",
                q,
                k,
                got,
                want
            );
        }
    }
    Ok(())
}

#[test]
fn sliding_window_blocks_distant_past() -> Result<()> {
    // window_size = 128, seq_len = 256, inspect row 200:
    //   valid columns are [200 - 127, 200] = [73, 200].
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device().clone();
    let seq_len = 256;
    let window_size = 128;

    let mask = sliding_window_causal_keep_mask(seq_len, window_size, &cuda, DType::F32)?;
    let host = to_host_f32(&mask)?;

    let q = 200usize;
    for k in 0..seq_len {
        let got = host[q * seq_len + k];
        let in_window = k >= q + 1 - window_size && k <= q;
        let want = if in_window { 1.0 } else { 0.0 };
        assert!(
            (got - want).abs() < 1e-6,
            "row {} col {}: got={} want={} (window=[{}, {}])",
            q,
            k,
            got,
            want,
            q + 1 - window_size,
            q
        );
    }

    // Also spot-check row 5 (early, window not yet saturated).
    let q = 5usize;
    for k in 0..seq_len {
        let got = host[q * seq_len + k];
        let want = if k <= q { 1.0 } else { 0.0 };
        assert!(
            (got - want).abs() < 1e-6,
            "row {} col {}: got={} want={}",
            q,
            k,
            got,
            want
        );
    }
    Ok(())
}

#[test]
fn sliding_window_diagonal_always_valid() -> Result<()> {
    // For every (seq_len, window_size) combo, mask[q, q] must be 1.0.
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device().clone();

    for &(seq_len, window_size) in &[
        (1usize, 1usize),
        (16, 1),
        (16, 8),
        (32, 16),
        (128, 128),
        (256, 64),
    ] {
        let mask = sliding_window_causal_keep_mask(seq_len, window_size, &cuda, DType::F32)?;
        let host = to_host_f32(&mask)?;
        for q in 0..seq_len {
            let got = host[q * seq_len + q];
            assert!(
                (got - 1.0).abs() < 1e-6,
                "diag mismatch at (seq={}, win={}, q={}): got={}",
                seq_len,
                window_size,
                q,
                got
            );
        }
    }
    Ok(())
}

#[test]
fn sliding_window_dtype_bf16_roundtrips() -> Result<()> {
    // BF16 mask should match F32 mask after cast (values are 0 / 1 so
    // BF16 is exact).
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device().clone();
    let seq_len = 32;
    let window_size = 8;

    let m_f32 = sliding_window_causal_keep_mask(seq_len, window_size, &cuda, DType::F32)?;
    let m_bf16 = sliding_window_causal_keep_mask(seq_len, window_size, &cuda, DType::BF16)?;

    let a = to_host_f32(&m_f32)?;
    let b = to_host_f32(&m_bf16)?;
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "bf16/f32 divergence at {}: f32={} bf16={}",
            i,
            x,
            y
        );
    }
    Ok(())
}
