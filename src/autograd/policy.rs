/// Gradient storage policy for [`crate::gradient::GradientMap`].
///
/// Two paths live side-by-side during the autograd v2 migration:
///
/// - [`GradStorePolicy::InternalFP32_PublicBF16`] (v1 / v3 default):
///   every gradient reaching the GradientMap is upcast to F32. Loss
///   seed (`set_ones`) is F32. Internal accumulators are F32. The
///   public-facing read paths (`get_public_grad`, `take_public_grads`)
///   convert F32 → BF16 on read. Matches the behavior every existing
///   trainer (Klein, Z-Image, Chroma, Qwen, ERNIE) was built against.
///
/// - [`GradStorePolicy::MatchInsertedDtype`] (autograd v2 / Phase 4b):
///   the gradient's native dtype is preserved end-to-end. A BF16 grad
///   stays BF16 in the map; an F32 grad stays F32. `set_ones_dtype`
///   seeds the loss with the requested dtype. Accumulation happens at
///   the stored dtype (F32 is only permitted as `opmath_t` inside the
///   `add_inplace_same_dtype` kernel; the result is written back at the
///   storage dtype). Mirrors `Parameter::new_v2` +
///   `GradDtypePolicy::MatchParamDtype`. See `docs/BF16_GRAD_DECISION.md`
///   Option A.
///
/// Default is [`GradStorePolicy::InternalFP32_PublicBF16`] so that
/// [`crate::gradient::GradientMap::new`] / `with_index` continue to
/// produce v1/v3-shaped maps with zero behavior change. Autograd v2
/// callers construct via `GradientMap::new_v2` / `with_index_v2`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradStorePolicy {
    /// v1 / v3 behavior. F32 storage, BF16 read-out under the public
    /// `get_public_grad` API.
    InternalFP32_PublicBF16,
    /// Autograd v2 / Phase 4b. Preserve the inserted gradient's native
    /// dtype on `set`, `insert`, `accumulate`, and `get_public_grad`.
    /// Loss seed dtype is supplied by the caller via `set_ones_dtype`
    /// (defaults to F32 via `set_ones` for legacy compatibility).
    MatchInsertedDtype,
}

impl Default for GradStorePolicy {
    fn default() -> Self {
        Self::InternalFP32_PublicBF16
    }
}
