#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Parity for `Upsample2d` bilinear vs a CPU reference that matches
//! PyTorch's `F.interpolate(..., mode='bilinear', align_corners=False|True)`.

use anyhow::Result;
use flame_core::{
    upsampling::{Upsample2d, Upsample2dConfig, UpsampleMode},
    DType, Device, Shape, Tensor,
};

fn cpu_ref_bilinear(
    input: &[f32],
    b: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    align_corners: bool,
) -> Vec<f32> {
    let h_scale = if align_corners && h_out > 1 {
        (h_in - 1) as f32 / (h_out - 1) as f32
    } else {
        h_in as f32 / h_out as f32
    };
    let w_scale = if align_corners && w_out > 1 {
        (w_in - 1) as f32 / (w_out - 1) as f32
    } else {
        w_in as f32 / w_out as f32
    };

    let mut out = vec![0.0f32; b * c * h_out * w_out];
    for bi in 0..b {
        for ci in 0..c {
            let plane = bi * c * h_in * w_in + ci * h_in * w_in;
            let plane_o = bi * c * h_out * w_out + ci * h_out * w_out;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let h_idx = if align_corners {
                        ho as f32 * h_scale
                    } else {
                        (ho as f32 + 0.5) * h_scale - 0.5
                    };
                    let w_idx = if align_corners {
                        wo as f32 * w_scale
                    } else {
                        (wo as f32 + 0.5) * w_scale - 0.5
                    };
                    let h0 = h_idx.floor() as i32;
                    let w0 = w_idx.floor() as i32;
                    let h_frac = h_idx - h_idx.floor();
                    let w_frac = w_idx - w_idx.floor();
                    let mut h1 = h0 + 1;
                    let mut w1 = w0 + 1;
                    let h0c = h0.max(0).min(h_in as i32 - 1) as usize;
                    let w0c = w0.max(0).min(w_in as i32 - 1) as usize;
                    h1 = h1.max(0).min(h_in as i32 - 1);
                    w1 = w1.max(0).min(w_in as i32 - 1);
                    let h1c = h1 as usize;
                    let w1c = w1 as usize;
                    let v00 = input[plane + h0c * w_in + w0c];
                    let v01 = input[plane + h0c * w_in + w1c];
                    let v10 = input[plane + h1c * w_in + w0c];
                    let v11 = input[plane + h1c * w_in + w1c];
                    let v0 = v00 * (1.0 - w_frac) + v01 * w_frac;
                    let v1 = v10 * (1.0 - w_frac) + v11 * w_frac;
                    let v = v0 * (1.0 - h_frac) + v1 * h_frac;
                    out[plane_o + ho * w_out + wo] = v;
                }
            }
        }
    }
    out
}

fn run_case(
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    align_corners: bool,
    dtype: DType,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<(f32, f32)> {
    const B: usize = 2;
    const C: usize = 3;

    flame_core::rng::set_seed(0xB11EA_u64 ^ (h_out as u64) ^ ((w_out as u64) << 16))
        .map_err(|e| anyhow::anyhow!("set_seed: {e:?}"))?;
    let x = Tensor::randn(Shape::from_dims(&[B, C, h_in, w_in]), 0.0, 1.0, device.clone())?;
    let x_in = x.to_dtype(dtype)?;

    let cfg = Upsample2dConfig::new(UpsampleMode::Bilinear).with_size((h_out, w_out));
    let cfg = if align_corners {
        Upsample2dConfig { align_corners: Some(true), ..cfg }
    } else {
        Upsample2dConfig { align_corners: Some(false), ..cfg }
    };
    let up = Upsample2d::new(cfg).forward(&x_in)?;
    let up_f32 = up.to_dtype(DType::F32)?.to_vec_f32()?;

    let x_ref = x.to_vec_f32()?;
    let ref_out = cpu_ref_bilinear(&x_ref, B, C, h_in, w_in, h_out, w_out, align_corners);

    // cosine similarity
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    for (a, b) in up_f32.iter().zip(ref_out.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let cs = (dot / (na.sqrt() * nb.sqrt())) as f32;
    Ok((cs, max_abs))
}

#[test]
fn upsample_bilinear_f32_parity() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(d) => d.cuda_device_arc(),
        Err(_) => return Ok(()),
    };
    for (hi, wi, ho, wo, ac) in [
        (4, 4, 8, 8, false),
        (4, 4, 8, 8, true),
        (7, 5, 14, 10, false),
        (5, 7, 10, 14, true),
        (32, 32, 64, 64, false),
    ] {
        let (cs, max_abs) = run_case(hi, wi, ho, wo, ac, DType::F32, &device)?;
        println!(
            "[bilinear_f32] {hi}x{wi}->{ho}x{wo} align_corners={ac}  cos_sim={cs:.6}  max_abs={max_abs:.3e}"
        );
        assert!(cs >= 0.9999, "F32 {hi}x{wi}->{ho}x{wo} ac={ac} cos_sim {cs}");
        assert!(max_abs <= 1e-4, "F32 {hi}x{wi}->{ho}x{wo} ac={ac} max_abs {max_abs}");
    }
    Ok(())
}

#[test]
fn upsample_bilinear_bf16_parity() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(d) => d.cuda_device_arc(),
        Err(_) => return Ok(()),
    };
    for (hi, wi, ho, wo, ac) in [
        (4, 4, 8, 8, false),
        (4, 4, 8, 8, true),
        (7, 5, 14, 10, false),
        (32, 32, 64, 64, false),
    ] {
        let (cs, max_abs) = run_case(hi, wi, ho, wo, ac, DType::BF16, &device)?;
        println!(
            "[bilinear_bf16] {hi}x{wi}->{ho}x{wo} align_corners={ac}  cos_sim={cs:.6}  max_abs={max_abs:.3e}"
        );
        // BF16 has ~7 bits of mantissa; tolerate small round-trip error.
        assert!(cs >= 0.999, "BF16 {hi}x{wi}->{ho}x{wo} ac={ac} cos_sim {cs}");
        assert!(max_abs <= 2e-2, "BF16 {hi}x{wi}->{ho}x{wo} ac={ac} max_abs {max_abs}");
    }
    Ok(())
}
