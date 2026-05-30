#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

// Reproduction microtest for the L2P step-2 backward crash
// (CUDA_ERROR_MISALIGNED_ADDRESS), handoff 2026-05-30.
//
// The l2p trainer aborts in the step-2 *backward* recompute. The reported
// backtrace ends at:
//   add_at_col_range -> Tensor::contiguous -> materialize_view -> MISALIGNED
//
// `add_at_col_range` is the LoRA `Slot::RowRange` accumulator in
// inference-flame/src/lora.rs:589. It patches a `delta [rows, len]` into a
// `base [rows, total]` at columns `start..start+len` by:
//   parts = [ base.narrow(1,0,start).contiguous(),       // head (if start>0)
//             base.narrow(1,start,len).contiguous().add(delta),  // mid
//             base.narrow(1,start+len,tail).contiguous() ]       // tail (if any)
//   cat(parts, dim=1)
//
// The real fused-qkv shapes (l2p NextDiT body: dim=3840):
//   total = 3*3840 = 11520, len = 3840, start in {0, 3840, 7680}.
//
// This test replicates that composition at the real shapes and runs FORWARD +
// BACKWARD through autograd (the backward exercises cat-backward -> narrow
// backward scatter, and the recompute re-runs the forward materialize_view).
//
// MEASUREMENT, not assertion (TENETS.md tenet 4): CUDA errors are async, so the
// handoff backtrace names where the abort *surfaced*, not provably where it was
// *caused*. If this isolated composition aborts, the materialize/narrow path is
// genuinely the cause. If it completes cleanly, the abort is an async error from
// a neighbor kernel and the attribution in the handoff is a red herring.

use cudarc::driver::CudaDevice;
use flame_core::{AutogradContext, DType, Error, Result, Shape, Tensor};
use std::sync::Arc;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

fn assert_finite(t: &Tensor, tag: &str) {
    let host = t.to_dtype(DType::F32).unwrap();
    let values = host.to_vec_f32().unwrap();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "[{tag}] tensor contained non-finite values"
    );
}

/// Byte-identical replica of inference-flame `add_at_col_range`.
fn add_at_col_range(base: &Tensor, delta: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let dims = base.shape().dims();
    if dims.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "add_at_col_range needs 2D base, got {:?}",
            dims
        )));
    }
    let total = dims[1];
    let head_len = start;
    let tail_len = total - start - len;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if head_len > 0 {
        parts.push(base.narrow(1, 0, head_len)?.contiguous()?);
    }
    let mid = base.narrow(1, start, len)?.contiguous()?;
    parts.push(mid.add(delta)?);
    if tail_len > 0 {
        parts.push(base.narrow(1, start + len, tail_len)?.contiguous()?);
    }
    let part_refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&part_refs, 1)
}

/// Run one forward+backward through add_at_col_range at a given column slot.
fn run_slot(rows: usize, total: usize, start: usize, len: usize) -> Result<()> {
    let dev = cuda_device();
    AutogradContext::reset();

    let base = Tensor::randn(Shape::from_dims(&[rows, total]), 0.0, 1.0, dev.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let delta = Tensor::randn(Shape::from_dims(&[rows, len]), 0.0, 1.0, dev.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let out = add_at_col_range(&base, &delta, start, len)?;
    assert_eq!(out.shape().dims(), &[rows, total]);
    assert_finite(&out, "forward_out");

    // Scalar loss -> backward. Exercises cat-backward + narrow backward scatter.
    let loss = out.mul(&out)?.mean()?;
    let gradients = AutogradContext::backward(&loss)?;
    let public = gradients.take_public_grads()?;
    assert!(
        !public.is_empty(),
        "no public grads produced for start={start} len={len}"
    );
    for grad in public.values() {
        assert_finite(grad, "grad");
    }
    Ok(())
}

// The three fused-qkv slots l2p actually uses (dim=3840).
const L2P_DIM: usize = 3840;
const L2P_QKV_TOTAL: usize = 3 * L2P_DIM; // 11520

#[test]
fn l2p_add_at_col_range_qkv_q_slot() -> Result<()> {
    // Q: start=0 (head skipped, mid + tail).
    run_slot(256, L2P_QKV_TOTAL, 0, L2P_DIM)
}

#[test]
fn l2p_add_at_col_range_qkv_k_slot() -> Result<()> {
    // K: start=3840 (head + mid + tail).
    run_slot(256, L2P_QKV_TOTAL, L2P_DIM, L2P_DIM)
}

#[test]
fn l2p_add_at_col_range_qkv_v_slot() -> Result<()> {
    // V: start=7680 (head + mid, tail skipped).
    run_slot(256, L2P_QKV_TOTAL, 2 * L2P_DIM, L2P_DIM)
}

// Odd row count + odd start, to probe whether a non-2-aligned element offset
// (would be byte-misaligned for any wider-than-u16 vectorized access) is the
// trigger, independent of the qkv-specific (all-even) offsets above.
#[test]
fn l2p_add_at_col_range_odd_offset() -> Result<()> {
    run_slot(257, 1536, 513, 511)
}
