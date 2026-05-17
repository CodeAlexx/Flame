//! Shared parity helpers for cross-test use.
//!
//! Lives at `tests/parity_helpers/mod.rs` per Rust convention for
//! integration-test helper modules — files in subdirs of `tests/` are NOT
//! compiled as standalone test crates and act as helper modules consumed
//! by sibling test files via `mod parity_helpers;`.
//!
//! ## Tolerance defaults (Fusion Sprint Phase 0)
//!
//! - **BF16**: `atol = 1e-2`, `rtol = 1e-2`
//!   - BF16 has ~3 decimal digits of mantissa precision; element-wise
//!     ops stay within ULP-of-output, but reductions accumulate.
//! - **FP32 reductions**: `atol = 1e-5`, `rtol = 1e-5`
//!   - Used for norm stats, optimizer moments, grad clip, loss reduction.
//!
//! ## Top-K delta printing
//!
//! On mismatch, [`compare_tensor`] prints the top-K (default 8) largest
//! absolute deltas with their flat index, the actual value, and the
//! expected value. This is essential for diagnosing fused-kernel regressions
//! where a stride bug shows up at one corner of the output.
//!
//! ## Reference policy
//!
//! Comparison is always promoted to FP32 host-side. The "expected" tensor
//! comes from a PyTorch-generated `.safetensors` fixture; flame-core is
//! the system under test. NEVER compare flame-core against itself.

#![allow(dead_code)]

use flame_core::Tensor;

/// Default BF16 tolerance for element-wise ops with no reduction.
pub const ATOL_BF16: f64 = 1e-2;
/// Default BF16 relative tolerance.
pub const RTOL_BF16: f64 = 1e-2;
/// Default FP32 tolerance for reductions (norm stats, optimizer moments,
/// grad clip, loss reduction).
pub const ATOL_FP32_REDUCTION: f64 = 1e-5;
/// Default FP32 relative tolerance for reductions.
pub const RTOL_FP32_REDUCTION: f64 = 1e-5;

/// Default number of largest-delta entries to print on mismatch.
pub const TOPK_PRINT: usize = 8;

/// Comparison report. Always populated, even on success — callers can
/// pretty-print it for visibility in passing tests.
#[derive(Debug, Clone)]
pub struct CompareReport {
    pub n: usize,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub cos_sim: f64,
    /// `(flat_index, actual, expected, abs_delta)`
    pub top_k: Vec<(usize, f32, f32, f32)>,
    pub atol: f64,
    pub rtol: f64,
    pub passed: bool,
}

impl CompareReport {
    pub fn pretty(&self) -> String {
        let mut s = format!(
            "[parity] n={} max_abs={:.6e} mean_abs={:.6e} cos={:.8} (atol={:.0e} rtol={:.0e}) {}",
            self.n,
            self.max_abs_diff,
            self.mean_abs_diff,
            self.cos_sim,
            self.atol,
            self.rtol,
            if self.passed { "PASS" } else { "FAIL" }
        );
        if !self.passed && !self.top_k.is_empty() {
            s.push_str("\n  top deltas:");
            for (i, a, e, d) in &self.top_k {
                s.push_str(&format!(
                    "\n    [{i:>10}] got={a:>14.6e} exp={e:>14.6e} |Δ|={d:.4e}"
                ));
            }
        }
        s
    }
}

/// Pull both tensors to host as FP32. Panics on shape mismatch — that is
/// always a test author error and never something we want to "tolerate".
fn to_fp32_pair(got: &Tensor, expected: &Tensor) -> (Vec<f32>, Vec<f32>) {
    let g = got.to_vec_f32().expect("to_vec_f32 on got");
    let e = expected.to_vec_f32().expect("to_vec_f32 on expected");
    assert_eq!(
        g.len(),
        e.len(),
        "parity_helpers::compare_tensor: length mismatch (got={} expected={})",
        g.len(),
        e.len()
    );
    (g, e)
}

/// Compare two tensors with absolute + relative tolerance.
///
/// Pass criterion (per element): `|got - expected| <= atol + rtol * |expected|`.
///
/// Returns a `CompareReport`. Caller decides whether to assert; this lets
/// you log on success and panic with `report.pretty()` on failure.
pub fn compare_tensor(got: &Tensor, expected: &Tensor, atol: f64, rtol: f64) -> CompareReport {
    let (g, e) = to_fp32_pair(got, expected);
    let n = g.len();

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut all_pass = true;

    // (idx, |Δ|) heap-of-K — small K so a linear scan is fine.
    let mut topk: Vec<(usize, f32, f32, f32)> = Vec::with_capacity(TOPK_PRINT + 1);

    for i in 0..n {
        let a = g[i] as f64;
        let b = e[i] as f64;
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d;
        dot += a * b;
        na += a * a;
        nb += b * b;

        let tol = atol + rtol * b.abs();
        if d > tol {
            all_pass = false;
        }

        // Insert into top-K by |Δ|.
        let entry = (i, g[i], e[i], d as f32);
        if topk.len() < TOPK_PRINT {
            topk.push(entry);
            topk.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap_or(std::cmp::Ordering::Equal));
        } else if (d as f32) > topk.last().unwrap().3 {
            topk.pop();
            topk.push(entry);
            topk.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    let mean_abs = if n == 0 { 0.0 } else { sum_abs / n as f64 };
    let cos_sim = if na == 0.0 || nb == 0.0 {
        if na == 0.0 && nb == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        dot / (na.sqrt() * nb.sqrt())
    };

    CompareReport {
        n,
        max_abs_diff: max_abs,
        mean_abs_diff: mean_abs,
        cos_sim,
        top_k: topk,
        atol,
        rtol,
        passed: all_pass,
    }
}

/// BF16 element-wise op default: `atol=1e-2 rtol=1e-2`.
pub fn compare_bf16(got: &Tensor, expected: &Tensor) -> CompareReport {
    compare_tensor(got, expected, ATOL_BF16, RTOL_BF16)
}

/// FP32 reduction default: `atol=1e-5 rtol=1e-5`.
pub fn compare_fp32_reduction(got: &Tensor, expected: &Tensor) -> CompareReport {
    compare_tensor(got, expected, ATOL_FP32_REDUCTION, RTOL_FP32_REDUCTION)
}

/// Assert helper — panics with `report.pretty()` on failure.
#[track_caller]
pub fn assert_parity_bf16(name: &str, got: &Tensor, expected: &Tensor) {
    let r = compare_bf16(got, expected);
    if !r.passed {
        panic!("parity FAIL [{name}]\n{}", r.pretty());
    } else {
        eprintln!(
            "parity ok  [{name}]: max_abs={:.4e} cos={:.8}",
            r.max_abs_diff, r.cos_sim
        );
    }
}

/// SHA-256 of the raw fixture file bytes. Use to pin a fixture and detect
/// silent regeneration / corruption. Hex-encoded lowercase.
pub fn sha256_file(path: &std::path::Path) -> String {
    use std::io::Read;
    let mut f =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("sha256_file: open {path:?}: {e}"));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .unwrap_or_else(|e| panic!("sha256_file: read {path:?}: {e}"));
    sha256_bytes(&buf)
}

/// SHA-256 of a byte slice. Tiny inline implementation so we don't need to
/// add a new dependency just for parity checksums.
pub fn sha256_bytes(data: &[u8]) -> String {
    let h = sha256_inline(data);
    let mut s = String::with_capacity(64);
    for b in h {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// Tiny inline SHA-256 (RFC 6234). ~80 lines. Used ONLY for fixture
// fingerprinting — not a security primitive.
// ---------------------------------------------------------------------------

fn sha256_inline(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let mj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(mj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for i in 0..8 {
        out[i * 4..(i + 1) * 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}
