//! Bit-exact parity tests for the new PyTorch-compatible RNG primitives:
//!   * `flame_core::rng::rand_torch`            ↔ `torch.rand`
//!   * `flame_core::rng::bernoulli_torch`       ↔ `torch.empty(..).bernoulli_(p)`
//!   * `flame_core::rng::randint_torch`         ↔ `torch.randint(low, high, ..)`
//!   * `flame_core::rng::kaiming_uniform_torch` ↔ `nn.init.kaiming_uniform_`
//!   * `flame_core::rng::xavier_uniform_torch`  ↔ `nn.init.xavier_uniform_`
//!
//! All fixtures were generated on an RTX 3090 Ti (84 SMs); since CUDA
//! distribution kernels depend on SM count (grid size), the tests skip
//! cleanly if the current GPU's SM count doesn't match the fixture's
//! recorded `sm_count` metadata.

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, Shape};
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("torch_compat_fixtures")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DType {
    F32,
    I32,
}

fn read_fixture(
    path: &PathBuf,
    expect: DType,
) -> (Vec<u8>, std::collections::HashMap<String, String>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!("read fixture {}: {e}", path.display());
    });
    assert!(bytes.len() >= 8);
    let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr = &bytes[8..8 + hdr_len];
    let hdr_str = std::str::from_utf8(hdr).expect("header utf8");
    let json: serde_json::Value = serde_json::from_str(hdr_str).expect("header json");
    let data_off = json["data"]["data_offsets"]
        .as_array()
        .expect("data_offsets");
    let begin = data_off[0].as_u64().unwrap() as usize;
    let end = data_off[1].as_u64().unwrap() as usize;
    let dtype = json["data"]["dtype"].as_str().unwrap();
    let shape = json["data"]["shape"]
        .as_array()
        .expect("shape array")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let numel: usize = shape.iter().product();
    let expected_dtype = match expect {
        DType::F32 => "F32",
        DType::I32 => "I32",
    };
    assert_eq!(
        dtype,
        expected_dtype,
        "fixture {} expected {}, got {}",
        path.display(),
        expected_dtype,
        dtype
    );
    let payload = bytes[8 + hdr_len + begin..8 + hdr_len + end].to_vec();
    let mut meta = std::collections::HashMap::new();
    if let Some(m) = json.get("__metadata__").and_then(|v| v.as_object()) {
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                meta.insert(k.clone(), s.to_string());
            }
        }
    }
    (payload, meta, numel)
}

fn load_fixture_f32(name: &str) -> (Vec<f32>, std::collections::HashMap<String, String>) {
    let path = fixture_dir().join(name);
    let (payload, meta, numel) = read_fixture(&path, DType::F32);
    assert_eq!(payload.len(), numel * 4, "fixture {name} f32 byte count");
    let mut out = vec![0.0f32; numel];
    for (i, c) in payload.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    (out, meta)
}

fn load_fixture_i32(name: &str) -> (Vec<i32>, std::collections::HashMap<String, String>) {
    let path = fixture_dir().join(name);
    let (payload, meta, numel) = read_fixture(&path, DType::I32);
    assert_eq!(payload.len(), numel * 4, "fixture {name} i32 byte count");
    let mut out = vec![0i32; numel];
    for (i, c) in payload.chunks_exact(4).enumerate() {
        out[i] = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    (out, meta)
}

fn current_sm_count() -> u32 {
    use cudarc::driver::sys::CUdevice_attribute as A;
    let dev = global_cuda_device();
    dev.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map(|x| x as u32)
        .unwrap_or(0)
}

fn sm_skip(meta: &std::collections::HashMap<String, String>, label: &str) -> bool {
    if let Some(ref_sm) = meta.get("sm_count").and_then(|s| s.parse::<u32>().ok()) {
        let cur = current_sm_count();
        if cur != ref_sm {
            eprintln!(
                "skipping {label}: fixture SM count {} != current {} (fixture was \
                 generated on {})",
                ref_sm,
                cur,
                meta.get("gpu_name").map(String::as_str).unwrap_or("?")
            );
            return true;
        }
    }
    false
}

fn read_back_i32(t: &flame_core::Tensor) -> Vec<i32> {
    t.to_vec_i32().expect("to_vec_i32")
}

// ---------- rand_torch ----------

fn check_rand(seed: u64, n: usize) {
    let name = format!("rand_torch_seed{seed}_n{n}.safetensors");
    let (expected, meta) = load_fixture_f32(&name);
    if sm_skip(&meta, &format!("rand seed={seed} n={n}")) {
        return;
    }
    let dev = global_cuda_device();
    let actual: Vec<f32> = flame_core::rng::rand_torch(seed, Shape::from_dims(&[n]), dev.clone())
        .expect("rand_torch")
        .to_vec()
        .expect("copy to host");
    let mut mismatches: Vec<(usize, f32, f32)> = Vec::new();
    for i in 0..n {
        if actual[i].to_bits() != expected[i].to_bits() {
            mismatches.push((i, actual[i], expected[i]));
        }
    }
    if !mismatches.is_empty() {
        eprintln!(
            "rand_torch seed={seed} n={n}: {}/{} bit-mismatches. First few:",
            mismatches.len(),
            n
        );
        for (i, a, e) in mismatches.iter().take(8) {
            eprintln!(
                "  [{i}] flame={a} (0x{:08x})  torch={e} (0x{:08x})  Δ={}",
                a.to_bits(),
                e.to_bits(),
                a - e
            );
        }
        panic!("rand_torch parity failure for seed={seed} n={n}");
    }
}

#[test]
#[ignore]
fn rand_torch_seed1234_n8() {
    check_rand(1234, 8);
}
#[test]
#[ignore]
fn rand_torch_seed1234_n64() {
    check_rand(1234, 64);
}
#[test]
#[ignore]
fn rand_torch_seed1234_n1024() {
    check_rand(1234, 1024);
}
#[test]
#[ignore]
fn rand_torch_seed42_n8() {
    check_rand(42, 8);
}
#[test]
#[ignore]
fn rand_torch_seed42_n64() {
    check_rand(42, 64);
}
#[test]
#[ignore]
fn rand_torch_seed42_n1024() {
    check_rand(42, 1024);
}
#[test]
#[ignore]
fn rand_torch_seed999_n8() {
    check_rand(999, 8);
}
#[test]
#[ignore]
fn rand_torch_seed999_n64() {
    check_rand(999, 64);
}
#[test]
#[ignore]
fn rand_torch_seed999_n1024() {
    check_rand(999, 1024);
}

// ---------- bernoulli_torch ----------

fn check_bernoulli(seed: u64, p: f32, n: usize) {
    let p_tag = format!("{p:.2}").replace('.', "p");
    let name = format!("bernoulli_torch_seed{seed}_p{p_tag}_n{n}.safetensors");
    let (expected, meta) = load_fixture_f32(&name);
    if sm_skip(&meta, &format!("bernoulli seed={seed} p={p} n={n}")) {
        return;
    }
    let dev = global_cuda_device();
    let actual: Vec<f32> =
        flame_core::rng::bernoulli_torch(seed, Shape::from_dims(&[n]), p, dev.clone())
            .expect("bernoulli_torch")
            .to_vec()
            .expect("copy to host");
    let mut mismatches = 0usize;
    let mut first_diff: Vec<(usize, f32, f32)> = Vec::new();
    for i in 0..n {
        if actual[i].to_bits() != expected[i].to_bits() {
            mismatches += 1;
            if first_diff.len() < 8 {
                first_diff.push((i, actual[i], expected[i]));
            }
        }
    }
    if mismatches > 0 {
        eprintln!(
            "bernoulli_torch seed={seed} p={p} n={n}: {}/{} mismatches",
            mismatches, n
        );
        for (i, a, e) in &first_diff {
            eprintln!("  [{i}] flame={a}  torch={e}");
        }
        panic!("bernoulli_torch parity failure");
    }
}

#[test]
#[ignore]
fn bernoulli_seed42_p050_n1024() {
    check_bernoulli(42, 0.5, 1024);
}
#[test]
#[ignore]
fn bernoulli_seed42_p050_n4096() {
    check_bernoulli(42, 0.5, 4096);
}
#[test]
#[ignore]
fn bernoulli_seed42_p010_n1024() {
    check_bernoulli(42, 0.1, 1024);
}
#[test]
#[ignore]
fn bernoulli_seed42_p010_n4096() {
    check_bernoulli(42, 0.1, 4096);
}
#[test]
#[ignore]
fn bernoulli_seed99_p050_n1024() {
    check_bernoulli(99, 0.5, 1024);
}
#[test]
#[ignore]
fn bernoulli_seed99_p050_n4096() {
    check_bernoulli(99, 0.5, 4096);
}
#[test]
#[ignore]
fn bernoulli_seed99_p010_n1024() {
    check_bernoulli(99, 0.1, 1024);
}
#[test]
#[ignore]
fn bernoulli_seed99_p010_n4096() {
    check_bernoulli(99, 0.1, 4096);
}

// ---------- randint_torch ----------

fn check_randint(seed: u64, low: i64, high: i64, n: usize) {
    let low_tag = if low < 0 {
        format!("m{}", -low)
    } else {
        format!("{low}")
    };
    let name = format!("randint_torch_seed{seed}_low{low_tag}_high{high}_n{n}.safetensors");
    let (expected, meta) = load_fixture_i32(&name);
    if sm_skip(&meta, &format!("randint seed={seed} [{low},{high}) n={n}")) {
        return;
    }
    let dev = global_cuda_device();
    let tensor = flame_core::rng::randint_torch(seed, low, high, Shape::from_dims(&[n]), dev.clone())
        .expect("randint_torch");
    let actual = read_back_i32(&tensor);
    let mut mismatches = 0usize;
    let mut first_diff: Vec<(usize, i32, i32)> = Vec::new();
    for i in 0..n {
        if actual[i] != expected[i] {
            mismatches += 1;
            if first_diff.len() < 8 {
                first_diff.push((i, actual[i], expected[i]));
            }
        }
    }
    if mismatches > 0 {
        eprintln!(
            "randint_torch seed={seed} [{low},{high}) n={n}: {}/{} mismatches",
            mismatches, n
        );
        for (i, a, e) in &first_diff {
            eprintln!("  [{i}] flame={a}  torch={e}");
        }
        panic!("randint_torch parity failure");
    }
}

#[test]
#[ignore]
fn randint_seed42_low0_high100_n1024() {
    check_randint(42, 0, 100, 1024);
}
#[test]
#[ignore]
fn randint_seed42_lowm5_high100_n1024() {
    check_randint(42, -5, 100, 1024);
}
#[test]
#[ignore]
fn randint_seed42_low0_high1000_n1024() {
    check_randint(42, 0, 1000, 1024);
}
#[test]
#[ignore]
fn randint_seed99_low0_high100_n1024() {
    check_randint(99, 0, 100, 1024);
}
#[test]
#[ignore]
fn randint_seed99_lowm5_high1000_n1024() {
    check_randint(99, -5, 1000, 1024);
}

// ---------- kaiming_uniform_torch ----------

fn check_kaiming(seed: u64, shape: &[usize]) {
    let shape_tag = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");
    let name = format!("kaiming_uniform_seed{seed}_shape{shape_tag}.safetensors");
    let (expected, meta) = load_fixture_f32(&name);
    if sm_skip(&meta, &format!("kaiming seed={seed} shape={shape:?}")) {
        return;
    }
    // Default PyTorch kaiming_uniform_ is mode='fan_in', a=0,
    // nonlinearity='leaky_relu'. For a 2-D weight of shape [out, in], fan_in
    // = in (the last dim's product, see _calculate_fan_in_and_fan_out).
    let fan_in = shape[1];
    let dev = global_cuda_device();
    let actual: Vec<f32> = flame_core::rng::kaiming_uniform_torch(
        Shape::from_dims(shape),
        0.0_f64,
        fan_in,
        "leaky_relu",
        seed,
        dev.clone(),
    )
    .expect("kaiming_uniform_torch")
    .to_vec()
    .expect("copy to host");
    let n = shape.iter().product::<usize>();
    let mut mismatches = 0usize;
    let mut first_diff: Vec<(usize, f32, f32)> = Vec::new();
    for i in 0..n {
        if actual[i].to_bits() != expected[i].to_bits() {
            mismatches += 1;
            if first_diff.len() < 8 {
                first_diff.push((i, actual[i], expected[i]));
            }
        }
    }
    if mismatches > 0 {
        eprintln!(
            "kaiming_uniform_torch seed={seed} shape={shape:?}: {}/{} bit-mismatches",
            mismatches, n
        );
        for (i, a, e) in &first_diff {
            eprintln!(
                "  [{i}] flame={a} (0x{:08x})  torch={e} (0x{:08x})  Δ={}",
                a.to_bits(),
                e.to_bits(),
                a - e
            );
        }
        panic!("kaiming_uniform_torch parity failure");
    }
}

#[test]
#[ignore]
fn kaiming_uniform_seed42_shape256x128() {
    check_kaiming(42, &[256, 128]);
}
#[test]
#[ignore]
fn kaiming_uniform_seed42_shape1024x512() {
    check_kaiming(42, &[1024, 512]);
}

// Regression fixtures for the f64-precision fix.
//
// Pre-fix, `kaiming_gain` returned f32 and the subsequent `gain as f64 /
// sqrt(fan)` could not recover the 29-bit mantissa loss. For
// `fan_in ∈ {137, 768, 1280, 3072}` (non-power-of-2, common Linear/Conv
// shapes) the f32 narrowing of `gain = sqrt(2.0)` perturbs the final
// `bound` by 1 ulp, causing *every* output sample to differ by 1 ulp.
// The existing power-of-2 fixtures (fan=128, 512) silently passed because
// `2/fan = 2^-N` is exactly representable in f32.
fn check_kaiming_fan(seed: u64, fan: usize) {
    let name = format!("kaiming_uniform_seed{seed}_fan{fan}.safetensors");
    let (expected, meta) = load_fixture_f32(&name);
    if sm_skip(&meta, &format!("kaiming seed={seed} fan={fan}")) {
        return;
    }
    // Fixture is shape=[1, fan], so fan_in = fan and total numel = fan.
    let shape = [1usize, fan];
    let dev = global_cuda_device();
    let actual: Vec<f32> = flame_core::rng::kaiming_uniform_torch(
        Shape::from_dims(&shape),
        0.0_f64,
        fan,
        "leaky_relu",
        seed,
        dev.clone(),
    )
    .expect("kaiming_uniform_torch")
    .to_vec()
    .expect("copy to host");
    let n = shape.iter().product::<usize>();
    let mut mismatches = 0usize;
    let mut first_diff: Vec<(usize, f32, f32)> = Vec::new();
    for i in 0..n {
        if actual[i].to_bits() != expected[i].to_bits() {
            mismatches += 1;
            if first_diff.len() < 8 {
                first_diff.push((i, actual[i], expected[i]));
            }
        }
    }
    if mismatches > 0 {
        eprintln!(
            "kaiming_uniform_torch seed={seed} fan={fan}: {}/{} bit-mismatches",
            mismatches, n
        );
        for (i, a, e) in &first_diff {
            eprintln!(
                "  [{i}] flame={a} (0x{:08x})  torch={e} (0x{:08x})  Δ={}",
                a.to_bits(),
                e.to_bits(),
                a - e
            );
        }
        panic!("kaiming_uniform_torch parity failure for non-power-of-2 fan");
    }
}

#[test]
#[ignore]
fn kaiming_uniform_seed42_fan137() {
    check_kaiming_fan(42, 137);
}
#[test]
#[ignore]
fn kaiming_uniform_seed42_fan768() {
    check_kaiming_fan(42, 768);
}
#[test]
#[ignore]
fn kaiming_uniform_seed42_fan1280() {
    check_kaiming_fan(42, 1280);
}
#[test]
#[ignore]
fn kaiming_uniform_seed42_fan3072() {
    check_kaiming_fan(42, 3072);
}

// ---------- xavier_uniform_torch ----------

fn check_xavier(seed: u64, shape: &[usize]) {
    let shape_tag = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");
    let name = format!("xavier_uniform_seed{seed}_shape{shape_tag}.safetensors");
    let (expected, meta) = load_fixture_f32(&name);
    if sm_skip(&meta, &format!("xavier seed={seed} shape={shape:?}")) {
        return;
    }
    // For a 2-D weight of shape [out, in], _calculate_fan_in_and_fan_out
    // returns (in, out).
    let fan_in = shape[1];
    let fan_out = shape[0];
    let dev = global_cuda_device();
    let actual: Vec<f32> = flame_core::rng::xavier_uniform_torch(
        Shape::from_dims(shape),
        1.0_f64,
        fan_in,
        fan_out,
        seed,
        dev.clone(),
    )
    .expect("xavier_uniform_torch")
    .to_vec()
    .expect("copy to host");
    let n = shape.iter().product::<usize>();
    let mut mismatches = 0usize;
    let mut first_diff: Vec<(usize, f32, f32)> = Vec::new();
    for i in 0..n {
        if actual[i].to_bits() != expected[i].to_bits() {
            mismatches += 1;
            if first_diff.len() < 8 {
                first_diff.push((i, actual[i], expected[i]));
            }
        }
    }
    if mismatches > 0 {
        eprintln!(
            "xavier_uniform_torch seed={seed} shape={shape:?}: {}/{} bit-mismatches",
            mismatches, n
        );
        for (i, a, e) in &first_diff {
            eprintln!(
                "  [{i}] flame={a} (0x{:08x})  torch={e} (0x{:08x})  Δ={}",
                a.to_bits(),
                e.to_bits(),
                a - e
            );
        }
        panic!("xavier_uniform_torch parity failure");
    }
}

#[test]
#[ignore]
fn xavier_uniform_seed42_shape256x128() {
    check_xavier(42, &[256, 128]);
}
#[test]
#[ignore]
fn xavier_uniform_seed42_shape1024x512() {
    check_xavier(42, &[1024, 512]);
}

// Regression fixtures for the f64-precision fix on `xavier_uniform_torch`.
//
// Pre-fix the public API took `gain: f32`, forcing callers passing
// `math.sqrt(2.0)` (the standard relu gain) to round to f32 before the
// call. The internal `(gain as f64) * sqrt(2/(fan_in+fan_out))` cannot
// recover that 1-ulp loss, so `bound` is off by 1 ulp and every sample
// diverges. The existing `gain=1.0` fixtures hid the bug because 1.0 is
// f32-exact.
fn check_xavier_gain(seed: u64, gain: f64, gain_tag: &str, shape: &[usize]) {
    let shape_tag = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");
    let name = format!(
        "xavier_uniform_seed{seed}_gain{gain_tag}_shape{shape_tag}.safetensors"
    );
    let (expected, meta) = load_fixture_f32(&name);
    if sm_skip(
        &meta,
        &format!("xavier seed={seed} gain={gain_tag} shape={shape:?}"),
    ) {
        return;
    }
    let fan_in = shape[1];
    let fan_out = shape[0];
    let dev = global_cuda_device();
    let actual: Vec<f32> = flame_core::rng::xavier_uniform_torch(
        Shape::from_dims(shape),
        gain,
        fan_in,
        fan_out,
        seed,
        dev.clone(),
    )
    .expect("xavier_uniform_torch")
    .to_vec()
    .expect("copy to host");
    let n = shape.iter().product::<usize>();
    let mut mismatches = 0usize;
    let mut first_diff: Vec<(usize, f32, f32)> = Vec::new();
    for i in 0..n {
        if actual[i].to_bits() != expected[i].to_bits() {
            mismatches += 1;
            if first_diff.len() < 8 {
                first_diff.push((i, actual[i], expected[i]));
            }
        }
    }
    if mismatches > 0 {
        eprintln!(
            "xavier_uniform_torch seed={seed} gain={gain_tag} shape={shape:?}: {}/{} bit-mismatches",
            mismatches, n
        );
        for (i, a, e) in &first_diff {
            eprintln!(
                "  [{i}] flame={a} (0x{:08x})  torch={e} (0x{:08x})  Δ={}",
                a.to_bits(),
                e.to_bits(),
                a - e
            );
        }
        panic!("xavier_uniform_torch parity failure for non-f32-exact gain");
    }
}

#[test]
#[ignore]
fn xavier_uniform_seed42_gainsqrt2_shape256x128() {
    check_xavier_gain(42, 2.0_f64.sqrt(), "sqrt2", &[256, 128]);
}
#[test]
#[ignore]
fn xavier_uniform_seed42_gain5div3_shape256x128() {
    check_xavier_gain(42, 5.0_f64 / 3.0_f64, "5div3", &[256, 128]);
}
