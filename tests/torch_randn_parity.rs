//! Bit-exact parity test for `flame_core::rng::randn_torch` vs `torch.randn`.
//!
//! The reference fixtures were generated on an RTX 3090 Ti (84 SMs). Since
//! `torch.randn` on CUDA depends on SM count, running this test on a GPU
//! with a different SM count is expected to mismatch. The test checks the
//! GPU's SM count against the fixture metadata and skips with a printed
//! message if they don't match — we DON'T want a false-positive failure.

#![cfg(feature = "cuda")]

use flame_core::{global_cuda_device, Shape};
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("torch_randn_fixtures")
}

fn load_fixture(seed: u64, n: usize) -> (Vec<f32>, std::collections::HashMap<String, String>) {
    let path = fixture_dir().join(format!("seed{seed}_n{n}.safetensors"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("read fixture {}: {e}", path.display());
    });
    // Parse safetensors header manually — minimal local impl avoids pulling
    // a new dep into [dev-dependencies] just for this test.
    assert!(bytes.len() >= 8);
    let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr = &bytes[8..8 + hdr_len];
    let hdr_str = std::str::from_utf8(hdr).expect("header utf8");
    // Tiny JSON probe — we only need the offsets and dtype for "data" plus
    // the optional __metadata__ k/v map.
    let json: serde_json::Value = serde_json::from_str(hdr_str).expect("header json");
    let data_off = json["data"]["data_offsets"]
        .as_array()
        .expect("data_offsets");
    let begin = data_off[0].as_u64().unwrap() as usize;
    let end = data_off[1].as_u64().unwrap() as usize;
    let dtype = json["data"]["dtype"].as_str().unwrap();
    assert_eq!(dtype, "F32", "fixture must be F32");
    let payload = &bytes[8 + hdr_len + begin..8 + hdr_len + end];
    assert_eq!(payload.len(), n * 4);
    let mut out = vec![0.0f32; n];
    for (i, c) in payload.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    let mut meta = std::collections::HashMap::new();
    if let Some(m) = json.get("__metadata__").and_then(|v| v.as_object()) {
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                meta.insert(k.clone(), s.to_string());
            }
        }
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

fn check_one(seed: u64, n: usize) {
    let (expected, meta) = load_fixture(seed, n);
    // Skip if SM count mismatches the fixture's reference GPU.
    if let Some(ref_sm) = meta.get("sm_count").and_then(|s| s.parse::<u32>().ok()) {
        let cur = current_sm_count();
        if cur != ref_sm {
            eprintln!(
                "skipping seed={seed} n={n}: fixture SM count {} != current {} (fixture was \
                 generated on {})",
                ref_sm,
                cur,
                meta.get("gpu_name").map(String::as_str).unwrap_or("?")
            );
            return;
        }
    }

    let dev = global_cuda_device();
    let actual_tensor = flame_core::rng::randn_torch(seed, Shape::from_dims(&[n]), dev.clone())
        .expect("randn_torch");
    let actual: Vec<f32> = actual_tensor
        .to_vec()
        .expect("copy to host");
    assert_eq!(actual.len(), expected.len());

    // Bit-exact comparison on the u32 bit-pattern.
    let mut mismatches: Vec<(usize, f32, f32)> = Vec::new();
    for i in 0..n {
        let ab = actual[i].to_bits();
        let eb = expected[i].to_bits();
        if ab != eb {
            mismatches.push((i, actual[i], expected[i]));
        }
    }
    if !mismatches.is_empty() {
        eprintln!(
            "seed={seed} n={n}: {}/{} mismatches (bit-exact). First few:",
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
        // Also show expected & actual first few unconditionally for context.
        eprintln!("first 8 expected: {:?}", &expected[..n.min(8)]);
        eprintln!("first 8 actual:   {:?}", &actual[..n.min(8)]);
        panic!("randn_torch parity failure for seed={seed} n={n}");
    }
}

#[test]
#[ignore]
fn randn_torch_seed1234_n8() {
    check_one(1234, 8);
}

#[test]
#[ignore]
fn randn_torch_seed1234_n64() {
    check_one(1234, 64);
}

#[test]
#[ignore]
fn randn_torch_seed1234_n1024() {
    check_one(1234, 1024);
}

#[test]
#[ignore]
fn randn_torch_seed42_n8() {
    check_one(42, 8);
}

#[test]
#[ignore]
fn randn_torch_seed42_n64() {
    check_one(42, 64);
}

#[test]
#[ignore]
fn randn_torch_seed999_n1024() {
    check_one(999, 1024);
}
