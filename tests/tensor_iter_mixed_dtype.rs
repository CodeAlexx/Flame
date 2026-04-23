// Phase 8 (dtype promotion) tests for `flame_core::tensor_iterator::promote`
// and the common-dtype wiring on `TensorIteratorConfig::build`.
//
// Scope per Phase 8 brief:
//   * Table-driven promotion check against PyTorch's `c10::promoteTypes`.
//   * BF16+BF16 goes through the new TensorIterator pipeline (smoke — the
//     existing `bf16_tensor_ops::bf16_add_matches_cpu` covers value
//     correctness; here we only assert dtype preservation).
//   * BF16+F32 routes through `GpuOps` (fallback) and produces an F32
//     output — Phase 8 does not move BF16+F32 into the new pipeline;
//     Phase 9/10 may.
//   * F16+BF16 → F32 per PyTorch's table (value-check via CPU reference).
//   * BF16+integer → BF16 per PyTorch table (floating wins).
//
// All tests run under `cuda` + `bf16_u16` features; the promotion table
// test is pure-Rust and would run without CUDA too, but we gate the
// whole file so it aligns with the other iter_* tests.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use cudarc::driver::CudaDevice;
use flame_core::device::Device;
use flame_core::tensor_iterator::{promote_dtypes, promote_many};
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

/// Promotion-table pins — every pair transcribed from
/// `pytorch/c10/core/ScalarType.cpp:111–126` for the dtypes flame-core
/// supports. If this test fails, either the table was transcribed wrong
/// or `promote_dtypes` has a bug.
#[test]
fn promote_dtypes_matches_pytorch() {
    use DType::*;

    // Diagonal: same dtype returns same.
    for d in [U8, I8, I32, I64, F16, F32, F64, Bool, BF16] {
        assert_eq!(promote_dtypes(d, d), d, "diagonal {:?}", d);
    }

    // Symmetry.
    for a in [U8, I8, I32, I64, F16, F32, F64, Bool, BF16] {
        for b in [U8, I8, I32, I64, F16, F32, F64, Bool, BF16] {
            assert_eq!(
                promote_dtypes(a, b),
                promote_dtypes(b, a),
                "asymmetry {:?}+{:?}",
                a,
                b
            );
        }
    }

    // --- u8 row (PyTorch: u1 = uint8) --------------------------------
    assert_eq!(promote_dtypes(U8, I32), I32); // u1+i4 = i4
    assert_eq!(promote_dtypes(U8, I64), I64); // u1+i8 = i8
    assert_eq!(promote_dtypes(U8, F16), F16); // u1+f2 = f2
    assert_eq!(promote_dtypes(U8, F32), F32); // u1+f4 = f4
    assert_eq!(promote_dtypes(U8, F64), F64); // u1+f8 = f8
    assert_eq!(promote_dtypes(U8, Bool), U8); // u1+b1 = u1
    assert_eq!(promote_dtypes(U8, BF16), BF16); // u1+bf = bf

    // --- i1 (I8) row -------------------------------------------------
    assert_eq!(promote_dtypes(I8, I32), I32); // i1+i4 = i4
    assert_eq!(promote_dtypes(I8, I64), I64); // i1+i8 = i8
    assert_eq!(promote_dtypes(I8, F16), F16); // i1+f2 = f2
    assert_eq!(promote_dtypes(I8, F32), F32); // i1+f4 = f4
    assert_eq!(promote_dtypes(I8, F64), F64); // i1+f8 = f8
    assert_eq!(promote_dtypes(I8, Bool), I8); // i1+b1 = i1
    assert_eq!(promote_dtypes(I8, BF16), BF16); // i1+bf = bf

    // --- i4 (I32) row ------------------------------------------------
    assert_eq!(promote_dtypes(I32, I64), I64); // i4+i8 = i8
    assert_eq!(promote_dtypes(I32, F16), F16); // i4+f2 = f2
    assert_eq!(promote_dtypes(I32, F32), F32); // i4+f4 = f4
    assert_eq!(promote_dtypes(I32, F64), F64); // i4+f8 = f8
    assert_eq!(promote_dtypes(I32, Bool), I32); // i4+b1 = i4
    assert_eq!(promote_dtypes(I32, BF16), BF16); // i4+bf = bf

    // --- i8 (I64) row ------------------------------------------------
    assert_eq!(promote_dtypes(I64, F16), F16); // i8+f2 = f2
    assert_eq!(promote_dtypes(I64, F32), F32); // i8+f4 = f4
    assert_eq!(promote_dtypes(I64, F64), F64); // i8+f8 = f8
    assert_eq!(promote_dtypes(I64, Bool), I64); // i8+b1 = i8
    assert_eq!(promote_dtypes(I64, BF16), BF16); // i8+bf = bf

    // --- f2 (F16) row ------------------------------------------------
    assert_eq!(promote_dtypes(F16, F32), F32); // f2+f4 = f4
    assert_eq!(promote_dtypes(F16, F64), F64); // f2+f8 = f8
    assert_eq!(promote_dtypes(F16, Bool), F16); // f2+b1 = f2
    // f2+bf = f4 — the famous PyTorch "smaller types don't promote to
    // each other" rule. Must NOT return F16 or BF16.
    assert_eq!(promote_dtypes(F16, BF16), F32);

    // --- f4 (F32) row ------------------------------------------------
    assert_eq!(promote_dtypes(F32, F64), F64); // f4+f8 = f8
    assert_eq!(promote_dtypes(F32, Bool), F32); // f4+b1 = f4
    assert_eq!(promote_dtypes(F32, BF16), F32); // f4+bf = f4

    // --- f8 (F64) row ------------------------------------------------
    assert_eq!(promote_dtypes(F64, Bool), F64); // f8+b1 = f8
    assert_eq!(promote_dtypes(F64, BF16), F64); // f8+bf = f8

    // --- b1 row ------------------------------------------------------
    assert_eq!(promote_dtypes(Bool, BF16), BF16); // b1+bf = bf
}

/// Left-fold promotion across >2 operands. Matches PyTorch
/// `TensorIterator::compute_common_dtype_only_for_inputs` behaviour.
#[test]
fn promote_many_folds_left() {
    use DType::*;
    assert_eq!(promote_many([BF16, F32, BF16]), Some(F32));
    assert_eq!(promote_many([F16, BF16, F32]), Some(F32));
    assert_eq!(promote_many([BF16, BF16, BF16]), Some(BF16));
    assert_eq!(promote_many([I32, BF16]), Some(BF16));
    // Three mixed ints + float: float wins.
    assert_eq!(promote_many([U8, I32, I64, F32]), Some(F32));
}

/// BF16+BF16 Tensor::add goes through the new TensorIterator pipeline
/// and produces a BF16 output. Value correctness is covered elsewhere
/// (bf16_tensor_ops). This test is dtype-only.
#[test]
fn bf16_bf16_add_preserves_bf16() -> Result<()> {
    let dev = cuda_device();
    let shape = Shape::from_dims(&[4, 8]);
    let a = Tensor::from_vec_dtype(
        vec![0.5f32; 32],
        shape.clone(),
        dev.clone(),
        DType::BF16,
    )?;
    let b = Tensor::from_vec_dtype(
        vec![0.25f32; 32],
        shape.clone(),
        dev.clone(),
        DType::BF16,
    )?;
    let out = a.add(&b)?;
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.storage_dtype(), DType::BF16);
    assert_eq!(out.shape().dims(), &[4, 8]);

    // Spot-check one element.
    let host = out.to_dtype(DType::F32)?.to_vec_f32()?;
    assert!(
        (host[0] - 0.75).abs() < 1e-2,
        "BF16+BF16 add value wrong: got {}",
        host[0]
    );
    Ok(())
}

/// BF16+F32 Tensor::add: Phase 8 routes this through the legacy GpuOps
/// path (not the new TensorIterator pipeline). Output dtype per
/// `promote_dtypes` is F32 (= `binary_target_dtype`, which Phase 8
/// consolidated onto the promotion ladder).
#[test]
fn bf16_f32_add_promotes_to_f32_via_gpuops() -> Result<()> {
    let _dev = Device::cuda(0)?;
    let cuda = cuda_device();
    let shape = Shape::from_dims(&[4, 8]);

    // a = BF16 filled with 1.0, b = F32 filled with 2.5 → expect 3.5 F32.
    let a = Tensor::from_vec_dtype(
        vec![1.0f32; 32],
        shape.clone(),
        cuda.clone(),
        DType::BF16,
    )?;
    let b = Tensor::from_vec_dtype(
        vec![2.5f32; 32],
        shape.clone(),
        cuda.clone(),
        DType::F32,
    )?;

    // Confirm the promoted common dtype matches PyTorch's table.
    assert_eq!(promote_dtypes(DType::BF16, DType::F32), DType::F32);

    let out = a.add(&b)?;
    assert_eq!(out.dtype(), DType::F32);
    let host = out.to_vec_f32()?;
    for v in host.iter() {
        assert!(
            (v - 3.5).abs() < 1e-4,
            "BF16+F32 add value wrong: got {}",
            v
        );
    }
    Ok(())
}

/// F16+BF16 promotes to F32 per PyTorch — the smaller-types-don't-
/// promote-to-each-other rule. Phase 8: verify the target dtype the
/// iterator would pick matches the table.
///
/// Phase-8 note: flame-core's `Tensor::add(F16, BF16)` does not have a
/// dedicated code path (neither tensor.rs's BF16+BF16 branch nor the
/// GpuOps::add mixed path go through F16 directly — `cast_to_f32_tensor`
/// converts each independently). The output is therefore F32 by
/// construction. We assert this end-to-end below.
#[test]
fn f16_bf16_add_promotes_to_f32() -> Result<()> {
    let cuda = cuda_device();
    let shape = Shape::from_dims(&[4, 8]);
    let a = Tensor::from_vec_dtype(
        vec![1.0f32; 32],
        shape.clone(),
        cuda.clone(),
        DType::F16,
    )?;
    let b = Tensor::from_vec_dtype(
        vec![2.0f32; 32],
        shape.clone(),
        cuda.clone(),
        DType::BF16,
    )?;

    // Pin the target dtype picked by promote_dtypes.
    assert_eq!(promote_dtypes(DType::F16, DType::BF16), DType::F32);

    let out = a.add(&b)?;
    assert_eq!(out.dtype(), DType::F32);
    let host = out.to_vec_f32()?;
    for v in host.iter() {
        assert!(
            (v - 3.0).abs() < 1e-3,
            "F16+BF16 add value wrong: got {}",
            v
        );
    }
    Ok(())
}

/// BF16+integer promotes to BF16 (floating wins per PyTorch's table).
/// flame-core's `Tensor::add` does not have an integer path, so we
/// only pin the promote-table result here.
#[test]
fn bf16_i32_promotion_table() {
    assert_eq!(promote_dtypes(DType::BF16, DType::I32), DType::BF16);
    assert_eq!(promote_dtypes(DType::BF16, DType::I64), DType::BF16);
    assert_eq!(promote_dtypes(DType::BF16, DType::I8), DType::BF16);
    assert_eq!(promote_dtypes(DType::BF16, DType::U8), DType::BF16);
    assert_eq!(promote_dtypes(DType::BF16, DType::Bool), DType::BF16);
}
