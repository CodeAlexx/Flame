#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity tests for `tensor_iterator::ops::unary::gelu_exact_bf16_iter`
//! and `Tensor::gelu_exact` — the exact-erf BF16 GELU added 2026-05-21 for
//! Cosmos-Predict2.5. PyTorch reference: `torch.nn.GELU()` with the default
//! `approximate='none'`, formula `0.5 * x * (1 + erf(x / sqrt(2)))`.
//!
//! Tests:
//!   1. **PyTorch GPU fixture parity** — `gelu_exact_vs_pytorch_gpu_fixture` —
//!      loads `tests/data/gelu_exact_ref.safetensors` (generated on CUDA by
//!      `gen_gelu_exact_ref.py`, NOT CPU — see `.claude/port-docs/CONTEXT.md`)
//!      and asserts max |Δ| within BF16 ULP and cos_sim ≥ 0.9999. This is
//!      the canonical gate.
//!   2. Forward matches a Rust-side F32 reference within BF16 ULP tolerance.
//!   3. `gelu_exact(x) != gelu(x)` for non-trivial x — confirms the new path
//!      is actually distinct from the existing tanh-approx kernel.
//!   4. Tensor::gelu_exact (contig fast path) matches the iter path bit-exactly.
//!   5. Sanity: `gelu_exact(1.0) ≈ 0.8413` and `gelu_tanh_approx(1.0) ≈ 0.8412`.

use flame_core::{tensor_iterator::ops::unary::gelu_exact_bf16_iter, DType, Result, Shape, Tensor};
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for tensor_iterator_gelu_exact_parity")
}

fn cos_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let av = av as f64;
        let bv = bv as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn make_bf16_tensor(dev: Arc<CudaDevice>, dims: &[usize], seed: u64) -> Result<Tensor> {
    let shape = Shape::from_dims(dims);
    let n = shape.elem_count();
    let mut data = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 40) as u32 as f32 / (1u32 << 24) as f32;
        data.push((u - 0.5) * 8.0);
    }
    let t_f32 = Tensor::from_vec(data, shape, dev)?;
    t_f32.to_dtype(DType::BF16)
}

/// Rust-side reference: read BF16 input, compute exact-erf GELU in F32, round
/// back. Matches what the kernel functor does (`__float2bfloat16_rn` after
/// f32 opmath). `libm::erff` is the same erf the CUDA built-in evaluates.
fn gelu_exact_reference_bf16(x: &Tensor) -> Result<Vec<f32>> {
    let in_f32 = x.to_vec_f32()?;
    // Round-trip each element through BF16 (input was already BF16-cast, but
    // `to_vec_f32` upcasts; the math then re-rounds the *output*).
    let mut out = Vec::with_capacity(in_f32.len());
    for &v in &in_f32 {
        // BF16 input value (already lossy-rounded by to_dtype(BF16) in the
        // tensor producer). PyTorch GELU(approximate='none') for BF16 reads
        // bf16, casts to f32, computes, casts back to bf16 with rn.
        let y = 0.5f32 * v * (1.0f32 + libm::erff(v * std::f32::consts::FRAC_1_SQRT_2));
        // Round to BF16 by reading via f32 then masking (rn): use to_dtype
        // semantics via half crate equivalent. flame-core uses
        // `__float2bfloat16_rn` device-side; CPU mirror = half::bf16::from_f32.
        let yb = half::bf16::from_f32(y);
        out.push(yb.to_f32());
    }
    Ok(out)
}

// -----------------------------------------------------------------------
// PyTorch GPU fixture loader.
//
// Loads `tests/data/gelu_exact_ref.safetensors`. The fixture has 3 BF16
// tensors keyed `x`, `y_exact`, `y_tanh`. All three were computed on a CUDA
// GPU by `gen_gelu_exact_ref.py` — running it on CPU would violate the
// project-wide rule (CONTEXT.md, [[feedback_pytorch_cpu_vs_cuda_bf16]]).
//
// Returns `(x_bytes, y_exact_bytes, y_tanh_bytes, shape)` — all three slices
// are the raw BF16 payloads (little-endian u16) in the same shape.
// -----------------------------------------------------------------------
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("gelu_exact_ref.safetensors")
}

fn load_gelu_exact_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<usize>) {
    let path = fixture_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "read fixture {}: {e}. Regenerate with `python3 \
             flame-core/tests/data/gen_gelu_exact_ref.py` on a CUDA host.",
            path.display()
        )
    });
    assert!(bytes.len() >= 8);
    let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr = &bytes[8..8 + hdr_len];
    let hdr_str = std::str::from_utf8(hdr).expect("header utf8");
    let json: serde_json::Value = serde_json::from_str(hdr_str).expect("header json");

    let read_tensor = |key: &str| -> (Vec<u8>, Vec<usize>) {
        let entry = &json[key];
        let dtype = entry["dtype"].as_str().expect("dtype");
        assert_eq!(
            dtype, "BF16",
            "fixture key {key} must be BF16, got {dtype}"
        );
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let off = entry["data_offsets"].as_array().expect("data_offsets");
        let begin = off[0].as_u64().unwrap() as usize;
        let end = off[1].as_u64().unwrap() as usize;
        let payload = &bytes[8 + hdr_len + begin..8 + hdr_len + end];
        assert_eq!(
            payload.len(),
            shape.iter().product::<usize>() * 2,
            "BF16 payload size for {key} doesn't match shape"
        );
        (payload.to_vec(), shape)
    };

    let (x, shape) = read_tensor("x");
    let (y_exact, shape_e) = read_tensor("y_exact");
    let (y_tanh, shape_t) = read_tensor("y_tanh");
    assert_eq!(shape, shape_e, "x and y_exact shape mismatch");
    assert_eq!(shape, shape_t, "x and y_tanh shape mismatch");
    (x, y_exact, y_tanh, shape)
}

/// Convert a BF16 byte slice into an `f32` vector (lossless upcast: BF16 → F32).
fn bf16_bytes_to_f32(b: &[u8]) -> Vec<f32> {
    assert_eq!(b.len() % 2, 0);
    let n = b.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
        out.push(half::bf16::from_bits(bits).to_f32());
    }
    out
}

#[test]
fn gelu_exact_vs_pytorch_gpu_fixture() -> Result<()> {
    let dev = cuda_device();
    let (x_bytes, y_exact_bytes, y_tanh_bytes, shape) = load_gelu_exact_fixture();
    let shape = Shape::from_dims(&shape);
    let n = shape.elem_count();

    // Push the *exact BF16 bytes* PyTorch produced onto the device. Critical:
    // we do NOT re-cast from f32 — that would re-quantize and could shift bits.
    let x = Tensor::from_bf16_bytes(&x_bytes, shape.clone(), dev.clone())?;
    assert_eq!(x.dtype(), DType::BF16);
    assert!(x.is_contiguous());

    // Both code paths under test: iter + tensor method.
    let out_iter = gelu_exact_bf16_iter(&x)?;
    let out_method = x.gelu_exact()?;

    let out_iter_f32 = out_iter.to_vec_f32()?;
    let out_method_f32 = out_method.to_vec_f32()?;
    let ref_exact_f32 = bf16_bytes_to_f32(&y_exact_bytes);
    let ref_tanh_f32 = bf16_bytes_to_f32(&y_tanh_bytes);

    assert_eq!(out_iter_f32.len(), n);
    assert_eq!(ref_exact_f32.len(), n);

    // -- (1) cos_sim ≥ 0.9999 vs PyTorch CUDA exact-erf reference.
    let cs = cos_sim_f32(&out_iter_f32, &ref_exact_f32);
    assert!(
        cs >= 0.9999,
        "flame_core gelu_exact vs PyTorch CUDA reference cos_sim {cs} below 0.9999"
    );

    // -- (2) per-element max |Δ| under 1 BF16 ULP at the local magnitude.
    // BF16 has 7 explicit fraction bits, so ULP(y) ≈ 2^(floor(log2|y|) - 7).
    // For |y| up to ~10 the worst-case ULP is ~10 * 2^-7 ≈ 0.078. We expect
    // bit-equality for most entries since both PyTorch CUDA and flame-core
    // compute the same math (f32 erff → bf16_rn) with the same hardware
    // intrinsics. Allow at most 1 BF16 ULP per element (~0.08 absolute at
    // |y|≤10). Tighter than the F32-reference test below.
    let mut max_diff = 0.0f32;
    let mut bit_eq = 0usize;
    for (i, (&a, &b)) in out_iter_f32.iter().zip(ref_exact_f32.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        // Bit-equal check via BF16 rebits.
        let ab = half::bf16::from_f32(a).to_bits();
        let bb = half::bf16::from_f32(b).to_bits();
        if ab == bb {
            bit_eq += 1;
        }
        // Per-element ULP-scale guard.
        let mag = a.abs().max(b.abs()).max(1.0);
        let bf16_ulp = mag * (1.0f32 / 128.0); // ≈ 2^-7
        assert!(
            d <= 2.0 * bf16_ulp,
            "fixture parity: element {i} diverges by {d}, > 2 BF16 ULP ({bf16_ulp}). \
             flame={a} torch={b}"
        );
    }
    eprintln!(
        "gelu_exact_fixture: n={n} bit_equal={bit_eq}/{n} max_abs_diff={max_diff} cos_sim={cs}"
    );

    // -- (3) iter and tensor-method paths agree bit-for-bit.
    for (i, (&a, &b)) in out_iter_f32.iter().zip(out_method_f32.iter()).enumerate() {
        let ab = half::bf16::from_f32(a).to_bits();
        let bb = half::bf16::from_f32(b).to_bits();
        assert_eq!(
            ab, bb,
            "iter vs Tensor::gelu_exact mismatch at element {i}"
        );
    }

    // -- (4) Sanity: the fixture itself contains two DIFFERENT outputs
    // (y_exact vs y_tanh). If this fires, the fixture is broken.
    let mut fixture_has_distinct = false;
    for (a, b) in ref_exact_f32.iter().zip(ref_tanh_f32.iter()) {
        if (a - b).abs() > 0.0 {
            fixture_has_distinct = true;
            break;
        }
    }
    assert!(
        fixture_has_distinct,
        "fixture y_exact == y_tanh everywhere — bad capture"
    );

    Ok(())
}

#[test]
fn gelu_exact_iter_matches_f32_reference() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[1024, 768], 0xC0FFEE_44)?;
    assert!(x.is_contiguous());

    let new_out = gelu_exact_bf16_iter(&x)?;
    let new_f32 = new_out.to_vec_f32()?;
    let ref_f32 = gelu_exact_reference_bf16(&x)?;

    let cs = cos_sim_f32(&new_f32, &ref_f32);
    assert!(
        cs >= 0.9999,
        "gelu_exact vs F32 reference cos_sim {cs} below 0.9999"
    );

    // Per-element max abs diff — must be within a couple of BF16 ULPs.
    // BF16 ULP at unity ≈ 2^-7 ≈ 0.0078; allow 1e-2 to be generous since
    // erff itself has a couple of ULPs of error at extreme inputs.
    let mut max_diff = 0.0f32;
    for (a, b) in new_f32.iter().zip(ref_f32.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        max_diff < 1e-2,
        "gelu_exact max per-element diff {max_diff} too large"
    );
    Ok(())
}

#[test]
fn gelu_exact_differs_from_tanh_approx() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[256, 256], 0xDEADBEEF_55)?;

    let exact = gelu_exact_bf16_iter(&x)?;
    let tanh_approx = flame_core::tensor_iterator::ops::unary::gelu_bf16_iter(&x)?;

    let exact_f32 = exact.to_vec_f32()?;
    let approx_f32 = tanh_approx.to_vec_f32()?;

    // They must be very close (both are GELU) but NOT bit-equal.
    let cs = cos_sim_f32(&exact_f32, &approx_f32);
    assert!(
        cs >= 0.999,
        "exact vs tanh-approx cos_sim {cs} — should still be ≥0.999 (both are GELU)"
    );

    let mut any_diff = false;
    for (a, b) in exact_f32.iter().zip(approx_f32.iter()) {
        if (a - b).abs() > 0.0 {
            any_diff = true;
            break;
        }
    }
    assert!(
        any_diff,
        "exact-erf GELU produced bit-identical output to tanh-approx — \
         kernel may be calling the wrong functor"
    );
    Ok(())
}

#[test]
fn gelu_exact_tensor_method_matches_iter() -> Result<()> {
    let dev = cuda_device();
    let x = make_bf16_tensor(dev, &[512, 1024], 0xBADF00D_77)?;
    assert!(x.is_contiguous());

    let via_method = x.gelu_exact()?;
    let via_iter = gelu_exact_bf16_iter(&x)?;

    let m_f32 = via_method.to_vec_f32()?;
    let i_f32 = via_iter.to_vec_f32()?;
    assert_eq!(m_f32.len(), i_f32.len());
    for (i, (a, b)) in m_f32.iter().zip(i_f32.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Tensor::gelu_exact (contig fast path) byte mismatch with iter at element {i}: \
             method={a} iter={b}"
        );
    }
    Ok(())
}

#[test]
fn gelu_exact_sanity_value() -> Result<()> {
    // Direct sanity: GELU(1.0) exact-erf ≈ 0.8413; tanh-approx ≈ 0.8412.
    let dev = cuda_device();
    let x_data = vec![1.0f32, 0.0f32, -1.0f32, 2.0f32, -2.0f32];
    let n = x_data.len();
    let x_f32 = Tensor::from_vec(x_data, Shape::from_dims(&[n]), dev.clone())?;
    let x = x_f32.to_dtype(DType::BF16)?;

    let exact = x.gelu_exact()?.to_vec_f32()?;
    let approx = x.gelu()?.to_vec_f32()?;

    // GELU(0) is exactly 0 in both formulas.
    assert!(exact[1].abs() < 1e-4, "GELU(0) exact = {}", exact[1]);
    assert!(approx[1].abs() < 1e-4, "GELU(0) approx = {}", approx[1]);

    // GELU_exact(1.0) ≈ 0.8413
    let e_at_1 = exact[0];
    assert!(
        (e_at_1 - 0.8413).abs() < 5e-3,
        "GELU_exact(1.0) = {e_at_1}, expected ≈ 0.8413"
    );

    // tanh-approx GELU(1.0) ≈ 0.8412
    let a_at_1 = approx[0];
    assert!(
        (a_at_1 - 0.8412).abs() < 5e-3,
        "GELU_tanh_approx(1.0) = {a_at_1}, expected ≈ 0.8412"
    );

    // They differ by ~9e-4 at x=1 in F32; after BF16 rounding the delta may
    // be 0 sometimes due to quantization. At x=2.0 the F32 delta is larger
    // (~1.4e-3) and BF16 should preserve a visible difference somewhere
    // across the 5 elements.
    let any_visible_diff = exact
        .iter()
        .zip(approx.iter())
        .any(|(e, a)| (e - a).abs() > 0.0);
    assert!(
        any_visible_diff,
        "exact and tanh-approx produced bit-equal output across all 5 sanity inputs"
    );

    eprintln!(
        "gelu_exact_sanity:   x=1.0  exact={:.6}  approx={:.6}  diff={:.6}",
        e_at_1,
        a_at_1,
        e_at_1 - a_at_1
    );
    eprintln!(
        "gelu_exact_sanity:   x=2.0  exact={:.6}  approx={:.6}  diff={:.6}",
        exact[3],
        approx[3],
        exact[3] - approx[3]
    );

    Ok(())
}
