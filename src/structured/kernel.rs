//! `StructuredKernel` trait — flame-core's port of PyTorch's
//! `structured_*_native::meta` + `::impl_` split.
//!
//! See `aten/src/ATen/native/Activation.cpp` for the silu pair as the
//! canonical reference shape.

use crate::{Result, Tensor};

/// Trait modeled on PyTorch's `structured_*_native::meta` + `::impl_` split.
///
/// Implementors split a kernel into three pieces:
///
/// - [`meta`](Self::meta): validate inputs and pre-allocate the output
///   tensor. No GPU compute happens here. Equivalent to PyTorch's
///   `structured_*_native::meta`.
/// - [`impl_`](Self::impl_): write the kernel result into the
///   pre-allocated output. Returns the output (moved through) for
///   ergonomics. Equivalent to PyTorch's `structured_*_native::impl_`.
/// - [`dispatch`](Self::dispatch): the combined entry point that wires
///   `meta` → autograd recording (when applicable) → `impl_`. Tensor
///   methods like `Tensor::silu_structured` call this.
///
/// Why a trait and not just three free functions? The trait gives each op
/// a single named type (`SiluStructured`, future `GeluStructured`, …) that
/// callers can refer to. It also makes the contract explicit: meta is
/// allocation, impl_ is GPU compute, and they must not be swapped.
pub trait StructuredKernel {
    /// The borrowed input shape for this op. For unary ops this is
    /// `&'a Tensor`; binary ops would use a 2-tuple, etc. Using a GAT-
    /// like associated type lets each op pick its natural argument shape
    /// without forcing every op through `&[&Tensor]`.
    type Input<'a>;

    /// Validate inputs and allocate the output tensor. No GPU compute.
    ///
    /// Returns `Err` for shape/dtype/device mismatches.
    fn meta(input: Self::Input<'_>) -> Result<Tensor>;

    /// Run the kernel, writing the result into the pre-allocated `output`
    /// tensor. Returns `output` (moved through) so callers can chain
    /// without re-binding.
    fn impl_(input: Self::Input<'_>, output: Tensor) -> Result<Tensor>;

    /// Combined dispatch: `meta` → autograd record (if `requires_grad`) →
    /// `impl_`. This is what `Tensor::*_structured` methods invoke.
    fn dispatch(input: Self::Input<'_>) -> Result<Tensor>;
}
