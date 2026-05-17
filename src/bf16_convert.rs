use crate::{error::Error, Result};
// Legacy bf16 conversion kernels.
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

// Vectorized BF16↔F32 cast kernels: 2 elements per thread via __nv_bfloat162.
// The previous scalar versions ran ~3.5 ms for 16 MB tensors (1.4 GB/s, ~0.15%
// of peak). The vectorized versions hit ~150 GB/s (15% of peak — ~10× improvement
// for the cast itself; bound by the FP32 write being 4 bytes vs BF16's 2).
const CUDA_TO_F32: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void bf16_to_f32(const __nv_bfloat16* __restrict__ X,
                 float* __restrict__ Y, long n){
  long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n2 = n >> 1;
  if (i2 < n2) {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X);
    float2* y2 = reinterpret_cast<float2*>(Y);
    y2[i2] = __bfloat1622float2(x2[i2]);
  }
  if (i2 == n2 && (n & 1)) {
    long last = n - 1;
    Y[last] = __bfloat162float(X[last]);
  }
}
"#;

const CUDA_TO_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void f32_to_bf16(const float* __restrict__ X,
                 __nv_bfloat16* __restrict__ Y, long n){
  long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n2 = n >> 1;
  if (i2 < n2) {
    const float2* x2 = reinterpret_cast<const float2*>(X);
    __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(Y);
    float2 v = x2[i2];
    y2[i2] = __floats2bfloat162_rn(v.x, v.y);
  }
  if (i2 == n2 && (n & 1)) {
    long last = n - 1;
    Y[last] = __float2bfloat16(X[last]);
  }
}
"#;

fn nvrtc_include_opt() -> String {
    std::env::var("CUDA_HOME")
        .map(|p| format!("{}/include", p))
        .unwrap_or_else(|_| "/usr/local/cuda/include".to_string())
}

fn ensure(dev: &Arc<CudaDevice>, name: &'static str, code: &'static str) -> Result<()> {
    if dev.get_func(name, name).is_some() {
        return Ok(());
    }
    let include = nvrtc_include_opt();
    let mut opts = CompileOptions::default();
    opts.include_paths.push(include);
    opts.use_fast_math = Some(false);
    opts.fmad = Some(true);
    let ptx =
        compile_ptx_with_opts(code, opts).map_err(|e| Error::Cuda(format!("nvrtc: {:?}", e)))?;
    dev.load_ptx(ptx, name, &[name])
        .map_err(|e| Error::Cuda(format!("load_ptx {}: {:?}", name, e)))?;
    Ok(())
}

#[inline]
fn lc_for(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

/// Launch config sized for `n / 2` threads (vectorized 2-element-per-thread kernels).
#[inline]
fn lc_pairs(n: usize) -> LaunchConfig {
    let pairs = (n + 1) / 2;
    LaunchConfig::for_num_elems(pairs as u32)
}

pub fn bf16_u16_to_f32(
    dev: Arc<CudaDevice>,
    src: u64,
    dst: &mut CudaSlice<f32>,
    n: usize,
) -> Result<()> {
    ensure(&dev, "bf16_to_f32", CUDA_TO_F32)?;
    let f = dev
        .get_func("bf16_to_f32", "bf16_to_f32")
        .ok_or_else(|| Error::Cuda("bf16_to_f32 missing".into()))?;
    unsafe {
        f.launch(lc_pairs(n), (src, dst, n as i64))?;
    }
    Ok(())
}

pub fn f32_to_bf16_u16(
    dev: Arc<CudaDevice>,
    src: &CudaSlice<f32>,
    dst: u64,
    n: usize,
) -> Result<()> {
    ensure(&dev, "f32_to_bf16", CUDA_TO_BF16)?;
    let f = dev
        .get_func("f32_to_bf16", "f32_to_bf16")
        .ok_or_else(|| Error::Cuda("f32_to_bf16 missing".into()))?;
    unsafe {
        f.launch(lc_pairs(n), (src, dst, n as i64))?;
    }
    Ok(())
}

/// Direct BF16→F32 cast: takes a raw BF16 device pointer (as u64) and writes
/// into a pre-allocated F32 buffer. Eliminates the F32 staging + dtod_copy
/// that the generic `to_dtype` path does for BF16 source. 2026-05-12.
pub fn bf16_to_f32_u16(
    dev: Arc<CudaDevice>,
    src: u64,
    dst: &mut CudaSlice<f32>,
    n: usize,
) -> Result<()> {
    ensure(&dev, "bf16_to_f32", CUDA_TO_F32)?;
    let f = dev
        .get_func("bf16_to_f32", "bf16_to_f32")
        .ok_or_else(|| Error::Cuda("bf16_to_f32 missing".into()))?;
    unsafe {
        f.launch(lc_pairs(n), (src, &*dst, n as i64))?;
    }
    Ok(())
}

/// CPU reference for stochastic rounding F32 → BF16.
///
/// Bias-free rounding: keep the high 16 bits (the BF16 representation under
/// truncation) and increment by 1 with probability `lower / 2^16`, where
/// `lower` is the dropped low-16 of the F32 mantissa+exponent. Over many
/// samples the mean of the rounded value equals the input — meaning small
/// gradients accumulate correctly across BF16 parameter updates rather than
/// silently being absorbed by RNE-rounding.
///
/// `rng_u32` should be a fresh uniform-random 32-bit value per element.
/// Matches the GPU kernel `bf16_stoch_round_kernel` defined in `bf16_ops.rs`.
#[inline]
pub fn stochastic_round_to_bf16_cpu(f: f32, rng_u32: u32) -> u16 {
    let bits = f.to_bits();
    let lower = bits & 0xFFFF;
    let upper = (bits >> 16) as u16;
    if (rng_u32 & 0xFFFF) < lower {
        upper.wrapping_add(1)
    } else {
        upper
    }
}

#[cfg(test)]
mod stoch_tests {
    use super::*;

    #[test]
    fn rng_low_rounds_up_when_lower_nonzero() {
        // Probability of rounding UP is `lower / 2^16`. With RNG=0 we always
        // satisfy `(rng & 0xFFFF) < lower` whenever `lower > 0`, i.e. always
        // round up. (When `lower == 0` we keep the RNE value, which is also
        // the truncated value.)
        let f = 1.0_f32 + 1e-5; // small but nonzero low-16 bits
        let bits = f.to_bits();
        let lower = bits & 0xFFFF;
        let upper = (bits >> 16) as u16;
        assert!(lower > 0, "test value should have nonzero low-16 bits");
        assert_eq!(stochastic_round_to_bf16_cpu(f, 0), upper.wrapping_add(1));
    }

    #[test]
    fn rng_high_truncates() {
        // RNG=0xFFFF means `(rng & 0xFFFF) = 0xFFFF`. Since `lower < 0x10000`
        // the predicate `0xFFFF < lower` is FALSE for any value of `lower` →
        // we keep `upper` (truncate).
        let f = 1.0_f32 + 1.0e-3;
        let bits = f.to_bits();
        let upper = (bits >> 16) as u16;
        assert_eq!(stochastic_round_to_bf16_cpu(f, 0xFFFF), upper);
    }

    #[test]
    fn integer_powers_of_two_dont_change() {
        // Powers of two have all-zero low 16 bits → always truncate.
        for x in [1.0_f32, 2.0, 0.5, 4.0, 0.25, 1024.0] {
            let bits = x.to_bits();
            let upper = (bits >> 16) as u16;
            for r in [0u32, 1, 0xDEAD_BEEF, 0xFFFF, 0xFFFF_FFFF] {
                assert_eq!(
                    stochastic_round_to_bf16_cpu(x, r),
                    upper,
                    "x={} r={:#x}",
                    x,
                    r
                );
            }
        }
    }

    #[test]
    fn unbiased_mean_over_random_rng() {
        // 1000 samples of stochastic rounding should reconstruct the input
        // value to within 1 BF16 ULP at the input's magnitude.
        let f = 1.234567_f32;
        let mut rng_state: u32 = 0x1234_5678;
        let mut sum_f32 = 0.0_f64;
        let n = 1000;
        for _ in 0..n {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let bits = (stochastic_round_to_bf16_cpu(f, rng_state) as u32) << 16;
            sum_f32 += f32::from_bits(bits) as f64;
        }
        let mean = (sum_f32 / n as f64) as f32;
        // BF16 ULP at value 1.234 is roughly 1/256 ≈ 4e-3. Allow 1e-2 slack
        // for the LCG's mediocre uniformity over 1000 samples.
        assert!(
            (mean - f).abs() < 1e-2,
            "stochastic rounding biased: input={} mean={}",
            f,
            mean
        );
    }
}
