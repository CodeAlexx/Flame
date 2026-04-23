//! TensorIterator op wrappers — Rust entry points for BF16 elementwise ops.
//!
//! This module is the Phase 11 destination for the per-op files that
//! previously lived under `src/ops/<op>_iter.rs`. Per the reference doc
//! §10 target layout, ops are grouped by category:
//!
//! - `unary` — silu, gelu, square, abs, relu, sigmoid, tanh, neg
//! - `transcendentals` — exp, log, sqrt, rsqrt, recip (f32 opmath inside)
//! - `binary` — add, sub, mul, div, maximum, minimum, mul_scalar, add_scalar
//! - `comparison` — ge, gt, le, lt, eq, ne (BF16 0.0/1.0 output)
//!
//! Each op keeps its `declare_stub!`, Rust kernel wrapper, `_bf16_iter`
//! public entry, and `extern "C"` FFI decl. The content is copy-paste
//! verbatim from the pre-Phase-11 `ops::*_iter` files — Phase 11 is a
//! pure relocation.

pub mod binary;
pub mod comparison;
pub mod transcendentals;
pub mod unary;
