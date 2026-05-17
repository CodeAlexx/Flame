//! Reusable per-layer parity harness.
//!
//! The recurring failure mode this catches: someone ships a
//! "looks-equivalent" rewrite of a kernel or layer, runs an end-to-end
//! sample, the output looks plausible, and the regression is only
//! discovered hours later when a different prompt produces garbage
//! (MagiHuman deinterleave incident, soul.md 2026-05-01).
//!
//! The fix is to check parity at every reasonable layer boundary —
//! intermediate activations — against a reference (typically a PyTorch
//! forward dump) before declaring a port equivalent.  Every model port
//! used to re-invent this; this module is the shared harness so we do
//! it the same way every time.
//!
//! ## Workflow
//!
//! 1. Run a reference forward in PyTorch with `register_forward_hook`
//!    on each layer of interest.  Collect `{layer_name → output_tensor}`.
//! 2. Save with `safetensors.torch.save_file(dict, path)`.  See
//!    `flame-core/scripts/dump_pytorch_layers.py` for a one-page recipe.
//! 3. In a Rust test or smoke binary, load the dump, run our forward,
//!    and call `harness.compare(name, &our_tensor)` at each boundary.
//! 4. `harness.report()` produces a table; `harness.assert_clean()`
//!    panics with the table if any layer fails tolerance.
//!
//! ## Naming convention
//!
//! Reference dump keys are arbitrary strings; the convention is the
//! PyTorch module path (`transformer_blocks.0.attn.to_q.output`).  Keep
//! the keys stable across reference and Rust callers — if you rename a
//! layer in PyTorch, redump and update the Rust call sites in the same
//! commit.

use crate::serialization::{load_tensors, SerializationFormat};
use crate::{DType, Error, Result, Tensor};
use cudarc::driver::CudaDevice;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Per-call comparison thresholds.
///
/// Defaults are calibrated for BF16 forward-pass parity: a clean port
/// typically scores `cos > 0.9999` and `max_abs_ratio < 0.05`.  Loosen
/// for very-low-precision ops (FP8 quant, e.g. `min_cos: 0.99`).
#[derive(Debug, Clone, Copy)]
pub struct ParityTolerance {
    /// Minimum cosine similarity to pass.
    pub min_cos: f32,
    /// Maximum allowed `max_abs(diff) / max_abs(reference)` ratio to pass.
    /// 0.05 means "we may be wrong by up to 5% of the reference's largest
    /// absolute value".
    pub max_abs_ratio: f32,
}

impl Default for ParityTolerance {
    fn default() -> Self {
        Self {
            min_cos: 0.9999,
            max_abs_ratio: 0.05,
        }
    }
}

/// Result of comparing one tensor against the reference dump.
#[derive(Debug, Clone)]
pub struct ParityResult {
    pub name: String,
    /// Cosine similarity in F32; ranges (-1, 1].  NaN when one input is
    /// all-zero (zero norm).
    pub cos: f32,
    /// Max absolute difference (F32).
    pub max_abs: f32,
    /// Mean absolute difference (F32).
    pub mean_abs: f32,
    /// `max_abs / max_abs(reference)`; F32.  0 when reference is zero.
    pub max_abs_ratio: f32,
    /// `true` when the result satisfied [`ParityTolerance`] at compare
    /// time.
    pub passed: bool,
    /// `Some(reason)` when the comparison could not run (shape mismatch,
    /// missing key, etc.); `passed` is then `false` and metric fields
    /// are NaN.
    pub note: Option<String>,
}

/// Loads a reference dump and compares Rust-side tensors against it.
///
/// `ParityHarness` is stateful: every `compare()` call appends to an
/// internal results buffer that [`report`](Self::report) renders as a
/// table at the end.  Construct one per forward sweep.
pub struct ParityHarness {
    refs: HashMap<String, Tensor>,
    tol: ParityTolerance,
    results: Vec<ParityResult>,
}

impl ParityHarness {
    /// Load a `.safetensors` reference dump.  Uses flame-core's existing
    /// loader so dtype + device handling matches the rest of the stack.
    pub fn load(path: impl AsRef<Path>, device: Arc<CudaDevice>) -> Result<Self> {
        let refs = load_tensors(path.as_ref(), device, SerializationFormat::SafeTensors)?;
        Ok(Self {
            refs,
            tol: ParityTolerance::default(),
            results: Vec::new(),
        })
    }

    /// Override the default tolerance.  Chainable on construction.
    pub fn with_tolerance(mut self, tol: ParityTolerance) -> Self {
        self.tol = tol;
        self
    }

    /// Iterate the names available in the reference dump.  Useful for
    /// asserting completeness of your call sites.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.refs.keys()
    }

    /// Returns true when the dump contains the named tensor.
    pub fn has(&self, name: &str) -> bool {
        self.refs.contains_key(name)
    }

    /// Compare `ours` against the reference at `name`, append the
    /// outcome to the internal results buffer, and return a copy.
    ///
    /// Returns a `ParityResult` with `note: Some(_)` and `passed: false`
    /// when the comparison cannot run (missing key, shape mismatch, dtype
    /// conversion failure).  Real numeric divergence shows up via
    /// `passed = false` with `note: None`.
    pub fn compare(&mut self, name: &str, ours: &Tensor) -> Result<ParityResult> {
        let result = self.compute(name, ours)?;
        self.results.push(result.clone());
        Ok(result)
    }

    fn compute(&self, name: &str, ours: &Tensor) -> Result<ParityResult> {
        let Some(reference) = self.refs.get(name) else {
            return Ok(ParityResult {
                name: name.to_string(),
                cos: f32::NAN,
                max_abs: f32::NAN,
                mean_abs: f32::NAN,
                max_abs_ratio: f32::NAN,
                passed: false,
                note: Some(format!("no key '{name}' in reference dump")),
            });
        };
        if ours.shape().dims() != reference.shape().dims() {
            return Ok(ParityResult {
                name: name.to_string(),
                cos: f32::NAN,
                max_abs: f32::NAN,
                mean_abs: f32::NAN,
                max_abs_ratio: f32::NAN,
                passed: false,
                note: Some(format!(
                    "shape mismatch: ours {:?} vs ref {:?}",
                    ours.shape().dims(),
                    reference.shape().dims()
                )),
            });
        }

        let ours_v = to_f32_vec(ours)?;
        let ref_v = to_f32_vec(reference)?;
        if ours_v.len() != ref_v.len() {
            return Err(Error::InvalidOperation(format!(
                "parity: F32-flat element count mismatch on '{name}': {} vs {}",
                ours_v.len(),
                ref_v.len()
            )));
        }
        let n = ours_v.len();

        let (mut dot, mut sum_a2, mut sum_b2) = (0f64, 0f64, 0f64);
        let (mut max_abs, mut sum_abs) = (0f32, 0f64);
        let mut max_abs_ref = 0f32;
        for i in 0..n {
            let a = ours_v[i] as f64;
            let b = ref_v[i] as f64;
            dot += a * b;
            sum_a2 += a * a;
            sum_b2 += b * b;
            let d = (ours_v[i] - ref_v[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d as f64;
            let ar = ref_v[i].abs();
            if ar > max_abs_ref {
                max_abs_ref = ar;
            }
        }

        let denom = (sum_a2 * sum_b2).sqrt();
        let cos = if denom > 0.0 {
            (dot / denom) as f32
        } else {
            f32::NAN
        };
        let mean_abs = if n > 0 {
            (sum_abs / n as f64) as f32
        } else {
            0.0
        };
        let max_abs_ratio = if max_abs_ref > 0.0 {
            max_abs / max_abs_ref
        } else {
            0.0
        };

        let passed =
            cos.is_finite() && cos >= self.tol.min_cos && max_abs_ratio <= self.tol.max_abs_ratio;

        Ok(ParityResult {
            name: name.to_string(),
            cos,
            max_abs,
            mean_abs,
            max_abs_ratio,
            passed,
            note: None,
        })
    }

    /// All compare()'d results in order.
    pub fn results(&self) -> &[ParityResult] {
        &self.results
    }

    /// True when every recorded result passed.
    pub fn is_clean(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Multi-line table summary.  Pass-rows shown compactly, fail-rows
    /// detailed.  Columns: `name | cos | max_abs | mean_abs | ratio | status`.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "[parity] tol min_cos={:.6} max_abs_ratio={:.4}\n",
            self.tol.min_cos, self.tol.max_abs_ratio
        ));
        out.push_str(&format!(
            "{:<48}  {:>10}  {:>12}  {:>12}  {:>10}  {}\n",
            "name", "cos", "max_abs", "mean_abs", "ratio", "status"
        ));
        for r in &self.results {
            let status = if r.passed { "OK" } else { "FAIL" };
            out.push_str(&format!(
                "{:<48}  {:>10.6}  {:>12.6e}  {:>12.6e}  {:>10.4}  {}",
                truncate(&r.name, 48),
                r.cos,
                r.max_abs,
                r.mean_abs,
                r.max_abs_ratio,
                status,
            ));
            if let Some(note) = &r.note {
                out.push_str(&format!("  ({note})"));
            }
            out.push('\n');
        }
        let n_total = self.results.len();
        let n_pass = self.results.iter().filter(|r| r.passed).count();
        out.push_str(&format!("[parity] {}/{} passed\n", n_pass, n_total));
        out
    }

    /// Panic with [`report`](Self::report) when any recorded result
    /// failed.  No-op when clean.
    pub fn assert_clean(&self) {
        if !self.is_clean() {
            panic!("{}", self.report());
        }
    }
}

fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    if t.dtype() == DType::F32 {
        t.to_vec()
    } else {
        t.to_dtype(DType::F32)?.to_vec()
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}
