// MXFP4 dequant kernel unit tests.
//
// Validates `flame_core::ops::fused_inference::dequant_mxfp4_to_bf16` against
// hand-computed BF16 reference values. Bit-exact LUT match required.
//
// Reference: transformers/integrations/mxfp4.py::convert_moe_packed_tensors
// FP4_VALUES = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
//               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use cudarc::driver::CudaDevice;
use flame_core::ops::fused_inference::dequant_mxfp4_to_bf16;
use flame_core::{DType, Shape, Tensor};
use half::bf16;
use std::sync::Arc;

/// FP4 LUT — must match the kernel's __constant__ table and transformers'
/// FP4_VALUES list exactly.
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// CPU reference dequant: mirror of the kernel logic, used to generate
/// expected values for the GPU output.
fn cpu_dequant_mxfp4(blocks: &[u8], scales: &[u8]) -> Vec<bf16> {
    let rows_total = scales.len();
    assert_eq!(blocks.len(), rows_total * 16);
    let mut out = Vec::with_capacity(rows_total * 32);
    for r in 0..rows_total {
        let scale_exp = scales[r] as i32 - 127;
        let scale_mul = 2.0_f32.powi(scale_exp);
        let blk = &blocks[r * 16..(r + 1) * 16];
        for i in 0..16 {
            let byte = blk[i];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            // Order matches kernel: low nibble → even index, high nibble → odd
            out.push(bf16::from_f32(FP4_LUT[lo] * scale_mul));
            out.push(bf16::from_f32(FP4_LUT[hi] * scale_mul));
        }
    }
    out
}

fn try_get_device() -> Option<Arc<CudaDevice>> {
    match CudaDevice::new(0) {
        Ok(dev) => Some(dev),
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err:?}");
            None
        }
    }
}

fn htod_u8(dev: &Arc<CudaDevice>, data: &[u8]) -> Result<cudarc::driver::CudaSlice<u8>> {
    dev.htod_sync_copy(data)
        .map_err(|e| anyhow::anyhow!("htod_sync_copy failed: {e:?}"))
}

fn bf16_vec_from_tensor(t: &Tensor) -> Result<Vec<bf16>> {
    let raw: Vec<u16> = t.to_vec_bf16()?;
    Ok(raw.into_iter().map(bf16::from_bits).collect())
}

/// Test 1: All 16 LUT entries exercised in one block, scale = 127 (2^0 = 1.0).
/// The 8 bytes [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] decode to
/// the full LUT in order. Remaining 8 bytes are zero (→ all zeros).
#[test]
fn mxfp4_dequant_lut_identity() -> Result<()> {
    let Some(device) = try_get_device() else { return Ok(()); };

    let mut blocks = vec![0u8; 16];
    // low nibble = idx 0,2,4,6,8,a,c,e (even LUT indices)
    // high nibble = idx 1,3,5,7,9,b,d,f (odd LUT indices)
    blocks[0] = 0x10; // lo=0, hi=1
    blocks[1] = 0x32; // lo=2, hi=3
    blocks[2] = 0x54; // lo=4, hi=5
    blocks[3] = 0x76; // lo=6, hi=7
    blocks[4] = 0x98; // lo=8, hi=9
    blocks[5] = 0xBA; // lo=a, hi=b
    blocks[6] = 0xDC; // lo=c, hi=d
    blocks[7] = 0xFE; // lo=e, hi=f
    // bytes 8..15 stay 0 (low=0, high=0 → 0.0, 0.0)

    let scales = vec![127u8]; // scale = 2^(127-127) = 1.0

    let blocks_gpu = htod_u8(&device, &blocks)?;
    let scales_gpu = htod_u8(&device, &scales)?;

    let out = dequant_mxfp4_to_bf16(
        &blocks_gpu,
        &scales_gpu,
        Shape::from_dims(&[32]),
        &device,
    )?;
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape().elem_count(), 32);

    let out_vec: Vec<bf16> = bf16_vec_from_tensor(&out)?;

    // Expected first 16 outputs interleave (lo, hi) per byte:
    //  byte 0 (0x10): 0.0, 0.5
    //  byte 1 (0x32): 1.0, 1.5
    //  byte 2 (0x54): 2.0, 3.0
    //  byte 3 (0x76): 4.0, 6.0
    //  byte 4 (0x98): -0.0, -0.5
    //  byte 5 (0xBA): -1.0, -1.5
    //  byte 6 (0xDC): -2.0, -3.0
    //  byte 7 (0xFE): -4.0, -6.0
    let expected_first_16: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for i in 0..16 {
        let got = out_vec[i].to_f32();
        let exp = expected_first_16[i];
        assert_eq!(
            got, exp,
            "LUT entry {i}: got {got}, expected {exp}"
        );
    }
    // Remaining 16 outputs (from zero bytes) must be 0.0.
    for i in 16..32 {
        assert_eq!(out_vec[i].to_f32(), 0.0, "tail entry {i} must be 0.0");
    }
    Ok(())
}

/// Test 2: scale = 128 → multiplier 2^1 = 2.0. Confirms scale math.
/// Test 3: scale = 126 → multiplier 2^-1 = 0.5. Confirms negative exp.
#[test]
fn mxfp4_dequant_scale_application() -> Result<()> {
    let Some(device) = try_get_device() else { return Ok(()); };

    // One block: bytes 0..3 = 0x10, 0x32, 0x54, 0x76 (covers magnitudes
    // 0,0.5,1,1.5,2,3,4,6); rest zero.
    let mut blocks = vec![0u8; 16];
    blocks[0] = 0x10;
    blocks[1] = 0x32;
    blocks[2] = 0x54;
    blocks[3] = 0x76;

    for &scale_byte in &[128u8, 126u8, 130u8, 120u8] {
        let scales = vec![scale_byte];
        let blocks_gpu = htod_u8(&device, &blocks)?;
        let scales_gpu = htod_u8(&device, &scales)?;

        let out = dequant_mxfp4_to_bf16(
            &blocks_gpu,
            &scales_gpu,
            Shape::from_dims(&[32]),
            &device,
        )?;
        let out_vec: Vec<bf16> = bf16_vec_from_tensor(&out)?;

        let mul = 2.0_f32.powi(scale_byte as i32 - 127);
        let expected: [f32; 8] = [
            0.0 * mul, 0.5 * mul, 1.0 * mul, 1.5 * mul,
            2.0 * mul, 3.0 * mul, 4.0 * mul, 6.0 * mul,
        ];
        for i in 0..8 {
            let got = out_vec[i].to_f32();
            let exp_bf16 = bf16::from_f32(expected[i]).to_f32();
            assert_eq!(
                got, exp_bf16,
                "scale={scale_byte} idx={i}: got {got}, exp {exp_bf16} (raw {})",
                expected[i]
            );
        }
    }
    Ok(())
}

/// Test 4: Multi-block random data; CPU vs GPU must match bit-exactly.
/// Covers grid-stride loop & multi-block correctness.
#[test]
fn mxfp4_dequant_multi_block_random() -> Result<()> {
    let Some(device) = try_get_device() else { return Ok(()); };

    // 1024 blocks → 16 KB of FP4 + 1 KB of scales → 32 K BF16 outputs.
    let rows_total = 1024usize;
    let blocks: Vec<u8> = (0..rows_total * 16).map(|i| (i * 73 + 5) as u8).collect();
    // Scale bytes uniformly across a useful range: 100..150 (2^-27 .. 2^23).
    let scales: Vec<u8> = (0..rows_total).map(|i| (100 + (i % 51)) as u8).collect();

    let blocks_gpu = htod_u8(&device, &blocks)?;
    let scales_gpu = htod_u8(&device, &scales)?;

    let out = dequant_mxfp4_to_bf16(
        &blocks_gpu,
        &scales_gpu,
        Shape::from_dims(&[rows_total, 32]),
        &device,
    )?;
    let out_vec: Vec<bf16> = bf16_vec_from_tensor(&out)?;
    assert_eq!(out_vec.len(), rows_total * 32);

    let expected = cpu_dequant_mxfp4(&blocks, &scales);

    // Bit-exact: both kernel and reference do `f32 * 2^exp → bf16`, so the
    // BF16 round-to-nearest result is identical.
    let mut mismatches = 0usize;
    for i in 0..expected.len() {
        if out_vec[i].to_bits() != expected[i].to_bits() {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "mismatch at {i}: gpu={} ({:#06x}) cpu={} ({:#06x})",
                    out_vec[i].to_f32(),
                    out_vec[i].to_bits(),
                    expected[i].to_f32(),
                    expected[i].to_bits()
                );
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} mismatches out of {}", expected.len());
    Ok(())
}

/// Test 5: GPT-OSS realistic shape — `[E=2, rows=64, G=4]` blocks, scales,
/// producing `[E=2, rows=64, G*32=128]` BF16. Validates the kernel doesn't
/// care about logical shape — only flat numel.
#[test]
fn mxfp4_dequant_gpt_oss_like_shape() -> Result<()> {
    let Some(device) = try_get_device() else { return Ok(()); };

    let e = 2usize;
    let rows = 64usize;
    let g = 4usize;
    let rows_total = e * rows * g; // = total 32-element blocks
    let n_bytes = rows_total * 16;

    let blocks: Vec<u8> = (0..n_bytes).map(|i| ((i * 13) ^ 0xA5) as u8).collect();
    let scales: Vec<u8> = (0..rows_total).map(|i| (110 + (i % 30)) as u8).collect();

    let blocks_gpu = htod_u8(&device, &blocks)?;
    let scales_gpu = htod_u8(&device, &scales)?;

    // Output shape: [E, rows, G*32]. Numel = rows_total * 32.
    let out = dequant_mxfp4_to_bf16(
        &blocks_gpu,
        &scales_gpu,
        Shape::from_dims(&[e, rows, g * 32]),
        &device,
    )?;
    let out_vec: Vec<bf16> = bf16_vec_from_tensor(&out)?;
    let expected = cpu_dequant_mxfp4(&blocks, &scales);

    let mut mismatches = 0usize;
    for i in 0..expected.len() {
        if out_vec[i].to_bits() != expected[i].to_bits() {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{} mismatches", expected.len());
    Ok(())
}

/// Test 6: numel-not-multiple-of-32 must error cleanly.
#[test]
fn mxfp4_dequant_bad_shape_errors() -> Result<()> {
    let Some(device) = try_get_device() else { return Ok(()); };

    let blocks_gpu = htod_u8(&device, &vec![0u8; 16])?;
    let scales_gpu = htod_u8(&device, &vec![127u8])?;

    // 31 is not a multiple of 32 → should be an InvalidShape error.
    let bad = dequant_mxfp4_to_bf16(
        &blocks_gpu,
        &scales_gpu,
        Shape::from_dims(&[31]),
        &device,
    );
    assert!(bad.is_err(), "expected error on numel=31, got Ok");
    Ok(())
}
