// Origin: flame-core TensorIterator port, Phase 1.
// Reference: pytorch/aten/src/ATen/TensorIterator.h L783–L993
//            (TensorIteratorConfig).
// Status: Phase 1 — fluent builder + field storage. `build` is a stub
//         that constructs the base with populated operands but does NOT
//         run the compute pipeline; Phase 3 replaces it with the full
//         flow (`compute_shape` → `compute_strides` → `reorder` →
//         `coalesce` → `allocate_or_resize_outputs` → dispatch).
//
// Builder semantics match PyTorch closely enough that a code reviewer
// holding TensorIterator.h side-by-side can tick each method off. The
// one place we diverge intentionally is around ownership: PyTorch has
// `add_borrowed_*` and `add_owned_*` variants; flame-core's Rust API
// encodes this at the type level via the `'a` lifetime, and
// `add_output(None)` is the signal for "iterator should allocate".

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::DType;

use super::base::{OperandInfo, OperandSrc, TensorIteratorBase};
use super::broadcast::{
    can_use_32bit_indexing, coalesce_dimensions, compute_shape, compute_strides,
    reorder_dimensions, OperandView,
};
use super::cache::{flag_bits, CachedIterGeometry, IterCacheKey, KeyShape, KeyStrides};
use super::dim_vec::{DimVec, I64StrideVec};
use smallvec::SmallVec;

/// Port of `at::TensorIteratorConfig`. Constructed via `new()`, chained
/// with the fluent methods, terminated with one of the `build_*` calls
/// (Phase 3 adds `build_unary_op` / `build_binary_op` on this side to
/// match PyTorch's naming; Phase 1 exposes a generic `build()` stub).
pub struct TensorIteratorConfig<'a> {
    /// Pending operands in PyTorch order: outputs first, then inputs.
    /// `None` in an output slot means "allocate at build time".
    pub(crate) tensors: smallvec::SmallVec<[Option<OperandSrc<'a>>; 4]>,
    pub(crate) num_outputs: usize,
    pub(crate) num_inputs: usize,

    /// Bypass shape computation with a fixed shape. Port of
    /// `TensorIteratorConfig::declare_static_shape` (h:957). Phase 1
    /// honours this in the geometry pipeline, but the full shape-squash
    /// variant (`squash_dims`) is not yet supported.
    pub(crate) static_shape: Option<DimVec>,

    /// Port of `declare_static_dtype` (h:955). When set, all operands
    /// must match this dtype; outputs with no provided tensor get
    /// allocated at this dtype.
    pub(crate) static_dtype: Option<DType>,

    /// Field exists; no device enum in flame-core beyond CUDA, so this
    /// just records that the user asked for static device. Phase 1
    /// treats the flag as informational; Phase 3 cross-checks it when
    /// allocating outputs.
    pub(crate) static_device_declared: bool,

    /// Port of `check_mem_overlap` flag (h:844). Default true.
    pub(crate) check_mem_overlap: bool,

    /// Port of `check_all_same_dtype` flag (h:854). Default true.
    pub(crate) check_all_same_dtype: bool,

    /// Port of `check_all_same_device` flag (h:863). Default true.
    pub(crate) check_all_same_device: bool,

    /// Port of `enforce_safe_casting_to_output` flag (h:873). Default
    /// false. Used by Phase 8 promotion.
    pub(crate) enforce_safe_casting_to_output: bool,

    // NOTE (enforce_linear_iteration deferred): PyTorch TensorIterator.cpp:
    // 248-251 short-circuits reorder_dimensions when this flag is true.
    // Not wired in flame-core — no Phase 1-11 op needs forced linear
    // iteration (no affine-strided custom kernels that bypass reorder).
    // If a future phase adds one, thread the flag through config → base
    // → broadcast::reorder_dimensions.
    /// Port of `promote_inputs_to_common_dtype` (h:897). Phase 1 stores
    /// the flag but does not act on it (no promotion until Phase 8).
    pub(crate) promote_inputs_to_common_dtype: bool,

    /// Port of `promote_integer_inputs_to_float` (h:911). Phase 1
    /// stores the flag; Phase 8 enforces it. Setting this without
    /// `promote_inputs_to_common_dtype` is a user error in PyTorch via
    /// `TORCH_INTERNAL_ASSERT`; we validate lazily at `build` time
    /// rather than in the setter (so the fluent chain stays infallible).
    pub(crate) promote_integer_inputs_to_float: bool,

    /// Port of `is_reduction` flag (h:919).
    pub(crate) is_reduction: bool,

    /// Port of `allow_cpu_scalars` flag (h:924).
    pub(crate) allow_cpu_scalars: bool,

    /// Port of `cast_common_dtype_to_outputs` (h:936).
    pub(crate) cast_common_dtype_to_outputs: bool,

    /// Port of `resize_outputs` (h:945). Default true.
    pub(crate) resize_outputs: bool,
}

impl<'a> Default for TensorIteratorConfig<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> TensorIteratorConfig<'a> {
    /// Construct a config with PyTorch's default flag values. Port of
    /// `TensorIteratorConfig()` (h:788).
    pub fn new() -> Self {
        Self {
            tensors: smallvec::SmallVec::new(),
            num_outputs: 0,
            num_inputs: 0,
            static_shape: None,
            static_dtype: None,
            static_device_declared: false,
            check_mem_overlap: true,
            check_all_same_dtype: true,
            check_all_same_device: true,
            enforce_safe_casting_to_output: false,
            promote_inputs_to_common_dtype: false,
            promote_integer_inputs_to_float: false,
            is_reduction: false,
            allow_cpu_scalars: false,
            cast_common_dtype_to_outputs: false,
            resize_outputs: true,
        }
    }

    /// Add a pre-allocated output tensor, or `None` to request
    /// allocation at build time. Port of `add_output(const TensorBase&)`
    /// (h:798). Outputs MUST be added before any input, matching
    /// PyTorch's contract (h:797).
    pub fn add_output(mut self, out: Option<&'a Tensor>) -> Self {
        debug_assert!(
            self.num_inputs == 0,
            "TensorIteratorConfig: outputs must be added before inputs (PyTorch's contract)"
        );
        self.tensors.push(out.map(OperandSrc::Borrowed));
        self.num_outputs += 1;
        self
    }

    /// Add an input tensor. Port of `add_input(const TensorBase&)`
    /// (h:801). Borrowed for the life of the config + resulting iterator.
    pub fn add_input(mut self, input: &'a Tensor) -> Self {
        self.tensors.push(Some(OperandSrc::Borrowed(input)));
        self.num_inputs += 1;
        self
    }

    /// Port of `declare_static_dtype_and_device` (h:952).
    pub fn declare_static_dtype_and_device(mut self, dtype: DType) -> Self {
        self.static_dtype = Some(dtype);
        self.static_device_declared = true;
        self
    }

    /// Port of `declare_static_dtype` (h:955).
    pub fn declare_static_dtype(mut self, dtype: DType) -> Self {
        self.static_dtype = Some(dtype);
        self
    }

    /// Port of `declare_static_shape(IntArrayRef)` (h:957).
    pub fn declare_static_shape(mut self, shape: &[usize]) -> Self {
        self.static_shape = Some(DimVec::from_slice(shape));
        self
    }

    /// Port of `check_all_same_dtype(bool)` (h:854).
    pub fn check_all_same_dtype(mut self, v: bool) -> Self {
        self.check_all_same_dtype = v;
        self
    }

    /// Port of `check_all_same_device(bool)` (h:863).
    pub fn check_all_same_device(mut self, v: bool) -> Self {
        self.check_all_same_device = v;
        self
    }

    /// Port of `allow_cpu_scalars(bool)` (h:924).
    pub fn allow_cpu_scalars(mut self, v: bool) -> Self {
        self.allow_cpu_scalars = v;
        self
    }

    /// Port of `promote_inputs_to_common_dtype(bool)` (h:897).
    /// Mirrors PyTorch's side-effect: setting this to true clears
    /// `check_all_same_dtype`.
    pub fn promote_inputs_to_common_dtype(mut self, v: bool) -> Self {
        self.promote_inputs_to_common_dtype = v;
        if v {
            self.check_all_same_dtype = false;
        }
        self
    }

    /// Port of `promote_integer_inputs_to_float(bool)` (h:911).
    pub fn promote_integer_inputs_to_float(mut self, v: bool) -> Self {
        self.promote_integer_inputs_to_float = v;
        self
    }

    /// Port of `cast_common_dtype_to_outputs(bool)` (h:936).
    pub fn cast_common_dtype_to_outputs(mut self, v: bool) -> Self {
        self.cast_common_dtype_to_outputs = v;
        if v {
            self.check_all_same_dtype = false;
        }
        self
    }

    /// Port of `enforce_safe_casting_to_output(bool)` (h:873).
    pub fn enforce_safe_casting_to_output(mut self, v: bool) -> Self {
        self.enforce_safe_casting_to_output = v;
        self
    }

    /// Port of `is_reduction(bool)` (h:919).
    pub fn is_reduction(mut self, v: bool) -> Self {
        self.is_reduction = v;
        self
    }

    /// Port of `resize_outputs(bool)` (h:945).
    pub fn resize_outputs(mut self, v: bool) -> Self {
        self.resize_outputs = v;
        self
    }

    /// Port of `set_check_mem_overlap(bool)` (h:844).
    pub fn set_check_mem_overlap(mut self, v: bool) -> Self {
        self.check_mem_overlap = v;
        self
    }

    /// Build the cache key for this config's current state. Cheap — just
    /// clones the per-operand shape/strides SmallVecs we'd touch anyway.
    /// See `cache::IterCacheKey` for the contract.
    fn build_cache_key(&self) -> IterCacheKey {
        let mut operand_shapes: SmallVec<[KeyShape; 4]> = SmallVec::new();
        let mut operand_strides: SmallVec<[KeyStrides; 4]> = SmallVec::new();
        let mut operand_dtypes: SmallVec<[u8; 4]> = SmallVec::new();
        let mut pending_output_mask: u8 = 0;
        for (i, src) in self.tensors.iter().enumerate() {
            match src {
                Some(s) => {
                    let t = s.tensor();
                    let dims = t.shape().dims();
                    let strides = t.strides();
                    operand_shapes.push(KeyShape::from_slice(dims));
                    operand_strides.push(strides);
                    operand_dtypes.push(t.dtype() as u8);
                }
                None => {
                    // Pending output slot. Mask the bit; carry empty
                    // shape/stride/dtype placeholders so vector lengths
                    // match `num_outputs + num_inputs`.
                    if i < 8 {
                        pending_output_mask |= 1u8 << i;
                    }
                    operand_shapes.push(KeyShape::new());
                    operand_strides.push(KeyStrides::new());
                    operand_dtypes.push(0u8);
                }
            }
        }

        let mut flags: u32 = 0;
        if self.resize_outputs {
            flags |= flag_bits::RESIZE_OUTPUTS;
        }
        if self.is_reduction {
            flags |= flag_bits::IS_REDUCTION;
        }
        if self.promote_inputs_to_common_dtype {
            flags |= flag_bits::PROMOTE_INPUTS;
        }
        if self.static_device_declared {
            flags |= flag_bits::STATIC_DEVICE_DECLARED;
        }
        if self.cast_common_dtype_to_outputs {
            flags |= flag_bits::CAST_COMMON_DTYPE_TO_OUTPUTS;
        }
        if self.enforce_safe_casting_to_output {
            flags |= flag_bits::ENFORCE_SAFE_CASTING;
        }

        let static_shape = self
            .static_shape
            .as_ref()
            .map(|s| KeyShape::from_slice(s.as_slice()));

        IterCacheKey {
            operand_shapes,
            operand_strides,
            operand_dtypes,
            pending_output_mask,
            num_outputs: self.num_outputs.min(255) as u8,
            static_dtype: self.static_dtype.map(|d| d as u8),
            static_shape,
            flags,
        }
    }

    /// Build into a base iterator.
    ///
    /// Phase 3 wires the full pipeline. The PyTorch reference is
    /// `TensorIteratorBase::build` at TensorIterator.cpp:1493:
    ///
    ///   1. `populate_operands`           → push OperandInfo array.
    ///   2. `compute_shape`               → broadcast inputs to one shape.
    ///   3. `compute_strides`             → broadcast byte-strides per operand.
    ///   4. `reorder_dimensions`          → stride-ascending (innermost-first).
    ///   5. `allocate_or_resize_outputs`  → allocate any `None` outputs.
    ///   6. `coalesce_dimensions`         → merge adjacent compatible dims.
    ///   7. set `requires_32bit_indexing_` and `all_ops_same_shape_`.
    ///
    /// What is *not* implemented in Phase 3:
    ///
    ///   * `mark_outputs` / `compute_mem_overlaps` — flame-core has no
    ///     equivalent of PyTorch's `is_read_write` flag, and mem-overlap
    ///     detection is a separate Phase-3-deferred item.
    ///   * `compute_names` / `compute_types` beyond the Phase-1 static-dtype
    ///     fallback. dtype promotion lands at Phase 8.
    ///   * `fast_set_up` — the fast-path short-circuit is still off; every
    ///     iterator runs the full geometry pipeline. Adding fast-setup is
    ///     deferred because it buys no correctness, only throughput.
    ///   * `will_resize` — flame-core tensors are shape-fixed; a provided
    ///     output whose shape disagrees with the iteration shape is
    ///     rejected with `ShapeMismatch`, never silently resized.
    pub fn build(self) -> Result<TensorIteratorBase<'a>> {
        // --- Phase 1 dispatch cache: try fast path -------------------
        // Build the cache key from the config + operand metadata. Cheap:
        // SmallVec clones of shape/stride vectors that we'd touch anyway
        // in compute_shape/compute_strides. On hit, we skip steps 2,3,4,6,7
        // (broadcast/strides/reorder/coalesce/32bit) and replay the cached
        // geometry. Step 1 (populate_operands) and Step 5
        // (allocate_or_resize_outputs) still run per call because they
        // reference live tensor pointers / per-call allocation.
        let cache_key = if super::cache::cache_disabled() {
            None
        } else {
            Some(self.build_cache_key())
        };
        let cached_hit: Option<CachedIterGeometry> =
            cache_key.as_ref().and_then(super::cache::lookup);
        if cached_hit.is_some() {
            super::cache::note_hit();
        }

        let mut base = TensorIteratorBase::new();
        base.num_outputs_ = self.num_outputs;
        base.num_inputs_ = self.num_inputs;
        base.is_reduction_ = self.is_reduction;
        base.static_dtype_ = self.static_dtype;

        // --- Step 1: populate_operands --------------------------------
        // target_dtype resolution logic:
        //   1. If `static_dtype` was declared, every operand gets that.
        //   2. Otherwise each operand's target_dtype is inherited from
        //      its source tensor (a pending output has no tensor yet, so
        //      we fall back to the first input's dtype — matching what
        //      PyTorch does when `add_output(undefined)` is used without
        //      a static dtype).
        //   3. If BOTH are unset (pending output + no inputs yet), we
        //      leave target_dtype as BF16 (flame-core's Phase-3 default)
        //      and the full-promotion Phase 8 `compute_types` will
        //      correct it.
        let fallback_dtype = self
            .static_dtype
            .or_else(|| {
                for (i, src) in self.tensors.iter().enumerate() {
                    if i < self.num_outputs {
                        continue;
                    }
                    if let Some(s) = src {
                        return Some(s.tensor().dtype());
                    }
                }
                None
            })
            .unwrap_or(DType::BF16);

        let static_dtype = self.static_dtype;
        let num_outputs = self.num_outputs;
        let static_shape = self.static_shape.clone();
        let check_same_dtype = self.check_all_same_dtype;
        let is_reduction = self.is_reduction;

        for (i, src) in self.tensors.into_iter().enumerate() {
            let is_output = i < num_outputs;
            let target_dtype = match &src {
                Some(s) => static_dtype.unwrap_or_else(|| s.tensor().dtype()),
                None => fallback_dtype,
            };
            base.operands.push(OperandInfo {
                src,
                target_dtype,
                is_output,
                stride_bytes: smallvec::SmallVec::new(),
            });
        }

        // Phase-3 dtype check: if `check_all_same_dtype` is on (default),
        // every defined operand must match. Mirrors the invariant PyTorch
        // enforces during `compute_types` for the common-case builders
        // (`build_unary_op` / `build_binary_op`) that pass `check_all_same_dtype(true)`.
        // Skipped when promotion was requested (the promote-setters clear
        // the flag as a side effect, matching PyTorch).
        if check_same_dtype {
            let mut common: Option<DType> = static_dtype;
            for op in &base.operands {
                if let Some(src) = op.src.as_ref() {
                    let t_dtype = src.tensor().dtype();
                    match common {
                        None => common = Some(t_dtype),
                        Some(c) if c != t_dtype => {
                            return Err(Error::InvalidOperation(format!(
                                "TensorIteratorConfig::build: check_all_same_dtype=true but \
                                 operands disagree: first={:?}, later={:?}",
                                c, t_dtype
                            )));
                        }
                        _ => {}
                    }
                }
            }
        }

        // common_dtype: when `promote_inputs_to_common_dtype` is set
        // (Phase 8 — matches PyTorch TensorIterator.cpp ~L285
        // `compute_common_dtype_only_for_inputs`), fold the input dtypes
        // through the promotion ladder. Otherwise keep Phase 3's
        // behaviour: static_dtype, else first input's dtype.
        base.common_dtype_ = if let Some(sd) = static_dtype {
            Some(sd)
        } else if self.promote_inputs_to_common_dtype {
            let input_dtypes: Vec<DType> = base
                .operands
                .iter()
                .filter(|op| !op.is_output && op.src.is_some())
                .map(|op| op.src.as_ref().unwrap().tensor().dtype())
                .collect();
            super::promote::promote_many(input_dtypes).or_else(|| {
                // No inputs at all (e.g. reduction with only outputs) —
                // match Phase 3 fallback, inheriting from the output
                // slot if any.
                base.operands
                    .iter()
                    .find(|op| op.src.is_some())
                    .map(|op| op.target_dtype)
            })
        } else {
            base.operands
                .iter()
                .find(|op| !op.is_output && op.src.is_some())
                .map(|op| op.target_dtype)
        };

        // --- Phase 1 dispatch cache: hot path on hit ------------------
        // If we have a cached geometry for this key, apply it and skip
        // steps 2-4 + 6-7. We still need to allocate / verify outputs
        // (step 5) per-call, but we use the cached `logical_output_shape`
        // and `stride_bytes` directly rather than re-deriving them via
        // `allocate_or_resize_outputs`.
        if let Some(cached) = cached_hit {
            // Apply cached state.
            base.shape_ = cached.shape.clone();
            base.perm_ = cached.perm.clone();
            base.has_coalesced_dimensions_ = cached.has_coalesced_dimensions;
            base.all_ops_same_shape_ = cached.all_ops_same_shape;
            base.requires_32bit_indexing_ = cached.requires_32bit_indexing;
            base.common_dtype_ = cached.common_dtype;
            base.fast_setup_ = cached.fast_setup;
            // Replay per-operand stride_bytes from cache (post-coalesce).
            debug_assert_eq!(cached.stride_bytes.len(), base.operands.len());
            for (op, sb) in base.operands.iter_mut().zip(cached.stride_bytes.iter()) {
                op.stride_bytes = sb.clone();
            }

            // Step 5 equivalent: allocate / verify outputs against the
            // cached logical output shape. Pending outputs get a fresh
            // contig tensor of that shape on the device inferred from
            // the first defined input.
            let logical = &cached.logical_output_shape;
            // Locate device from the first defined input (post-step-1
            // base.operands already wired).
            let mut device = None;
            for j in num_outputs..base.operands.len() {
                if let Some(src) = base.operands[j].src.as_ref() {
                    device = Some(src.tensor().device().clone());
                    break;
                }
            }
            for i in 0..num_outputs {
                let op = &mut base.operands[i];
                match &op.src {
                    None => {
                        let dtype = op.target_dtype;
                        let dev = device.clone().ok_or_else(|| {
                            Error::InvalidOperation(
                                "tensor_iterator cache hit: no defined operand to infer \
                                 output device from"
                                    .into(),
                            )
                        })?;
                        let shape_obj = crate::shape::Shape::from_dims(logical.as_slice());
                        let allocated = Tensor::empty_dtype(shape_obj, dtype, dev)?;
                        op.src = Some(OperandSrc::Owned(allocated));
                    }
                    Some(src) => {
                        let provided = src.tensor().shape().dims();
                        if provided != logical.as_slice() {
                            return Err(Error::ShapeMismatch {
                                expected: crate::shape::Shape::from_dims(logical.as_slice()),
                                got: src.tensor().shape().clone(),
                            });
                        }
                    }
                }
            }

            return Ok(base);
        }

        // If a static shape was declared, record it and skip broadcast
        // shape inference. PyTorch honours `declare_static_shape` by
        // setting `shape_` directly and marking resize-outputs off.
        if let Some(s) = static_shape {
            base.shape_ = s;
            base.all_ops_same_shape_ = true;
        } else {
            // --- Step 2: compute_shape --------------------------------
            // Only INPUTS contribute to broadcast inference. Output
            // slots — whether pending (`None`) or pre-provided — are
            // excluded when `resize_outputs=true` (default). PyTorch
            // `TensorIterator.cpp:1237` skips outputs-to-be-resized.
            // flame-core doesn't resize; it verifies provided outputs
            // match the iteration shape in `allocate_or_resize_outputs`
            // and errors with `ShapeMismatch` on disagreement. Skipping
            // outputs here keeps the contract "shape inference from
            // inputs only" regardless of whether the output exists yet.
            let skip_outputs_in_shape_inference = self.resize_outputs;
            let mut views_owned: Vec<(Vec<usize>, Vec<usize>, usize)> = Vec::new();
            for op in &base.operands {
                if op.is_output && skip_outputs_in_shape_inference {
                    continue;
                }
                let t = match op.src.as_ref() {
                    Some(s) => s.tensor(),
                    None => continue,
                };
                let strides = t.strides();
                views_owned.push((
                    t.shape().dims().to_vec(),
                    strides.to_vec(),
                    op.target_dtype.size_in_bytes(),
                ));
            }
            let views: Vec<OperandView<'_>> = views_owned
                .iter()
                .map(|(s, es, sz)| OperandView {
                    shape: s.as_slice(),
                    element_strides: es.as_slice(),
                    elem_size: *sz,
                })
                .collect();
            if views.is_empty() && !is_reduction {
                return Err(Error::InvalidOperation(
                    "TensorIteratorConfig::build: no defined operand to infer shape from".into(),
                ));
            }
            let (shape, all_same) = compute_shape(&views)?;
            base.shape_ = shape;
            base.all_ops_same_shape_ = all_same;
        }

        // --- Step 3: compute_strides ----------------------------------
        // Build the per-operand broadcast byte-stride array. Pending
        // outputs get a zero-filled array of the right length here; the
        // allocator in step 5 overwrites it with the real contig stride.
        {
            let ndim = base.shape_.len();
            let mut per_operand: Vec<I64StrideVec> = Vec::with_capacity(base.operands.len());
            for op in &base.operands {
                if let Some(src) = op.src.as_ref() {
                    let t = src.tensor();
                    let es = t.strides();
                    let view = OperandView {
                        shape: t.shape().dims(),
                        element_strides: es.as_slice(),
                        elem_size: op.target_dtype.size_in_bytes(),
                    };
                    // Reuse the shared compute_strides via a 1-operand call;
                    // saves duplicating the padding logic here.
                    let mut strides = compute_strides(base.shape_.as_slice(), &[view]);
                    per_operand.push(strides.remove(0));
                } else {
                    // Pending output: placeholder zero-filled stride array.
                    per_operand.push(smallvec::smallvec![0i64; ndim]);
                }
            }
            for (op, s) in base.operands.iter_mut().zip(per_operand.into_iter()) {
                op.stride_bytes = s;
            }
        }

        // --- Step 4: reorder_dimensions -------------------------------
        // Drives the innermost-stride-first ordering. Invalid for rank<=1
        // (the helper returns identity in that case).
        {
            let mut all_strides: Vec<I64StrideVec> = base
                .operands
                .iter()
                .map(|op| op.stride_bytes.clone())
                .collect();
            let perm = reorder_dimensions(&mut base.shape_, &mut all_strides);
            base.perm_ = perm;
            for (op, s) in base.operands.iter_mut().zip(all_strides.into_iter()) {
                op.stride_bytes = s;
            }
        }

        // Capture logical_output_shape for the Phase 1 cache. This is
        // invert_perm(base.shape_) at the post-reorder, pre-coalesce point —
        // exactly the shape `allocate_or_resize_outputs` allocates against.
        // For rank <= 1 perm is identity, so the inverse equals shape_.
        let logical_output_shape_for_cache: DimVec = {
            let ndim = base.shape_.len();
            let perm_slice = base.perm_.as_slice();
            if ndim == 0 {
                DimVec::new()
            } else if perm_slice.len() == ndim {
                let mut out: DimVec = smallvec::smallvec![0usize; ndim];
                for new_dim in 0..ndim {
                    let old_dim = perm_slice[new_dim];
                    out[old_dim] = base.shape_[new_dim];
                }
                out
            } else {
                base.shape_.clone()
            }
        };

        // --- Step 5: allocate_or_resize_outputs -----------------------
        base.allocate_or_resize_outputs()?;

        // --- Step 6: coalesce_dimensions ------------------------------
        {
            let mut all_strides: Vec<I64StrideVec> = base
                .operands
                .iter()
                .map(|op| op.stride_bytes.clone())
                .collect();
            let changed = coalesce_dimensions(&mut base.shape_, &mut all_strides);
            base.has_coalesced_dimensions_ = changed;
            for (op, s) in base.operands.iter_mut().zip(all_strides.into_iter()) {
                op.stride_bytes = s;
            }
        }

        // --- Step 7: finalize flags -----------------------------------
        {
            let all_strides: Vec<I64StrideVec> = base
                .operands
                .iter()
                .map(|op| op.stride_bytes.clone())
                .collect();
            base.requires_32bit_indexing_ =
                can_use_32bit_indexing(base.shape_.as_slice(), &all_strides);
        }

        // --- Phase 1 dispatch cache: populate on miss -----------------
        // Skipped if the cache is disabled (FLAME_TI_CACHE_DISABLE=1) or
        // if we never built a key (e.g. cache_disabled() was true at the
        // top of the function — kept in sync via the cache_key Option).
        if let Some(key) = cache_key {
            let stride_bytes: smallvec::SmallVec<[I64StrideVec; 4]> = base
                .operands
                .iter()
                .map(|op| op.stride_bytes.clone())
                .collect();
            let target_dtypes: smallvec::SmallVec<[DType; 4]> =
                base.operands.iter().map(|op| op.target_dtype).collect();
            let geom = CachedIterGeometry {
                shape: base.shape_.clone(),
                perm: base.perm_.clone(),
                logical_output_shape: logical_output_shape_for_cache,
                stride_bytes,
                has_coalesced_dimensions: base.has_coalesced_dimensions_,
                all_ops_same_shape: base.all_ops_same_shape_,
                requires_32bit_indexing: base.requires_32bit_indexing_,
                common_dtype: base.common_dtype_,
                static_dtype: base.static_dtype_,
                fast_setup: base.fast_setup_,
                target_dtypes,
            };
            super::cache::insert(key, geom);
        }

        Ok(base)
    }
}
