// flame-core/tests/permute_fastpath_parity.rs
//
// 2026-05-12 perf: parity test for the permute_generic fast-paths added in
// cuda/permute0213.cu (launch_permute10_*, launch_permute0132_*).
//
// Strategy:
//   1. Build a deterministic source tensor.
//   2. Run permute with FLAME_PERMUTE_FASTPATH=1 → tuned kernel.
//   3. Run permute with FLAME_PERMUTE_FASTPATH=0 → generic kernel.
//   4. Assert outputs are bit-identical (it's a copy — must be exact).
//
// Cases mirror the production zimage trace (top-frequency shapes/perms).

#![cfg(feature = "cuda")]

use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

fn make_device() -> Arc<cudarc::driver::CudaDevice> {
    cudarc::driver::CudaDevice::new(0).expect("CUDA device 0")
}

/// Build a deterministic F32 host vector — element i = (i % 7919) * 0.001.
/// 7919 is prime → no accidental aliasing across reshapes.
fn host_data(total: usize) -> Vec<f32> {
    (0..total).map(|i| (i % 7919) as f32 * 0.001).collect()
}

fn make_bf16(device: &Arc<cudarc::driver::CudaDevice>, dims: &[usize]) -> Tensor {
    let total: usize = dims.iter().product();
    Tensor::from_vec_dtype(
        host_data(total),
        Shape::from_dims(dims),
        device.clone(),
        DType::BF16,
    )
    .expect("from_vec_dtype bf16")
}

fn make_f32(device: &Arc<cudarc::driver::CudaDevice>, dims: &[usize]) -> Tensor {
    let total: usize = dims.iter().product();
    Tensor::from_vec(host_data(total), Shape::from_dims(dims), device.clone())
        .expect("from_vec f32")
}

fn read_bf16(t: &Tensor) -> Vec<u16> {
    t.to_vec_bf16().expect("to_vec_bf16")
}

fn read_f32(t: &Tensor) -> Vec<f32> {
    t.to_vec_f32().expect("to_vec_f32")
}

/// Materialize permute(shape, perm) under a specific fast-path setting and
/// return the raw bytes for comparison.
fn run_permute_bf16(shape: &[usize], perm: &[usize], fastpath: bool) -> Vec<u16> {
    let device = make_device();
    std::env::set_var("FLAME_PERMUTE_FASTPATH", if fastpath { "1" } else { "0" });
    let t = make_bf16(&device, shape);
    let p = t.permute(perm).expect("permute view");
    let c = p.contiguous().expect("contiguous");
    read_bf16(&c)
}

fn run_permute_f32(shape: &[usize], perm: &[usize], fastpath: bool) -> Vec<f32> {
    let device = make_device();
    std::env::set_var("FLAME_PERMUTE_FASTPATH", if fastpath { "1" } else { "0" });
    let t = make_f32(&device, shape);
    let p = t.permute(perm).expect("permute view");
    let c = p.contiguous().expect("contiguous");
    read_f32(&c)
}

fn check_bf16(shape: &[usize], perm: &[usize]) {
    let fast = run_permute_bf16(shape, perm, true);
    let slow = run_permute_bf16(shape, perm, false);
    assert_eq!(
        fast.len(),
        slow.len(),
        "len mismatch for shape={:?} perm={:?}: fast={} slow={}",
        shape,
        perm,
        fast.len(),
        slow.len()
    );
    for (i, (a, b)) in fast.iter().zip(slow.iter()).enumerate() {
        if a != b {
            panic!(
                "bf16 bit mismatch at idx {} for shape={:?} perm={:?}: fast=0x{:04x} slow=0x{:04x}",
                i, shape, perm, a, b
            );
        }
    }
}

fn check_f32(shape: &[usize], perm: &[usize]) {
    let fast = run_permute_f32(shape, perm, true);
    let slow = run_permute_f32(shape, perm, false);
    assert_eq!(fast.len(), slow.len());
    for (i, (a, b)) in fast.iter().zip(slow.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            panic!(
                "f32 bit mismatch at idx {} for shape={:?} perm={:?}: fast={} slow={}",
                i, shape, perm, a, b
            );
        }
    }
}

#[test]
fn rank2_transpose_small_inner() {
    // Top trace shape — [M=8, K=3840] → [3840, 8] (matmul backward weight grad).
    check_bf16(&[8, 3840], &[1, 0]);
    check_bf16(&[3840, 8], &[1, 0]);
    check_bf16(&[10240, 8], &[1, 0]);
    check_bf16(&[8, 10240], &[1, 0]);
}

#[test]
fn rank2_transpose_aligned_32() {
    check_bf16(&[64, 3840], &[1, 0]);
    check_bf16(&[3840, 64], &[1, 0]);
    check_bf16(&[128, 128], &[1, 0]);
}

#[test]
fn rank2_transpose_odd_sizes() {
    check_bf16(&[3, 17], &[1, 0]);
    check_bf16(&[17, 3], &[1, 0]);
    check_bf16(&[1, 256], &[1, 0]);
    check_bf16(&[256, 1], &[1, 0]);
}

#[test]
fn rank2_transpose_f32() {
    check_f32(&[8, 3840], &[1, 0]);
    check_f32(&[64, 64], &[1, 0]);
}

#[test]
fn rank4_inner_swap_zimage_qk() {
    check_bf16(&[1, 30, 1536, 128], &[0, 1, 3, 2]);
    // Big one — 1.4 GB at BF16 if not skipped. Skip in CI but run locally.
    if std::env::var("FLAME_PERMUTE_TEST_BIG").is_ok() {
        check_bf16(&[1, 30, 1536, 1536], &[0, 1, 3, 2]);
    }
}

#[test]
fn rank4_inner_swap_misc() {
    check_bf16(&[2, 4, 64, 128], &[0, 1, 3, 2]);
    check_bf16(&[1, 1, 8, 16], &[0, 1, 3, 2]);
    check_bf16(&[3, 5, 33, 17], &[0, 1, 3, 2]);
}

#[test]
fn rank4_inner_swap_f32() {
    check_f32(&[1, 4, 256, 128], &[0, 1, 3, 2]);
}

#[test]
fn fallthrough_uncovered_perm() {
    // perm=[2,0,1,3] is not a covered fast-path tuple — fastpath dispatcher
    // returns None and both modes call the generic kernel. Output should
    // still match (sanity check that fastpath=true doesn't corrupt).
    check_bf16(&[2, 3, 4, 5], &[2, 0, 1, 3]);
}
