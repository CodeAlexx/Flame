// Origin: flame-core TensorIterator port, Phase 1.
// Reference: pytorch/aten/src/ATen/TensorIterator.h L117–L206 (OperandInfo),
//            L248–L734 (TensorIteratorBase), L238–L243 (FastSetupType),
//            L998–L1032 (SplitUntil32Bit stub).
// Status: Phase 1 — fields, accessors, and geometry. `build_unary_op` and
//         `build_binary_op` are present as stubs returning
//         `Err(Error::NotImplemented)` per plan §2 Phase 1. Phase 3 fills
//         them in after DispatchStub lands.

use crate::error::{Error, Result};
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::DType;

use super::dim_vec::{contiguous_element_strides, element_strides_to_bytes, DimVec, I64StrideVec};

/// Port of `at::FastSetupType` (TensorIterator.h:238). Phase 1 only
/// meaningfully uses `Contiguous`; the other variants are present so
/// future phases have a stable enum to match on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastSetupType {
    None,
    Contiguous,
    ChannelsLast,
    NonOverlappingDense,
}

/// Source tag for an operand: either a borrowed reference to an input
/// tensor (caller owns it for the life of the iterator) or an owned output
/// that the iterator allocated itself via `allocate_or_resize_outputs`.
///
/// Borrowed is the common case. Owned is produced in Phase 3+ when an
/// output slot was requested as `add_output(None)`. Phase 1 never
/// constructs `Owned` itself, but reserves the variant so Phase 3's
/// Builder doesn't have to refactor the enum.
#[derive(Debug)]
pub enum OperandSrc<'a> {
    Borrowed(&'a Tensor),
    Owned(Tensor),
}

impl<'a> OperandSrc<'a> {
    #[inline]
    pub fn tensor(&self) -> &Tensor {
        match self {
            OperandSrc::Borrowed(t) => t,
            OperandSrc::Owned(t) => t,
        }
    }
}

/// Port of `OperandInfo` (TensorIterator.h:117). flame-core-side
/// simplifications versus PyTorch:
///
///   - `element_strides` (usize) is the primary stride representation;
///     `stride_bytes` (i64) is computed from it via
///     `dim_vec::element_strides_to_bytes` after broadcasting.
///   - `target_dtype` / `current_dtype` / `device` are rolled into a
///     single `target_dtype` field; Phase 1 is BF16-only (no promotion)
///     so they never diverge. Phase 8 adds a `current_dtype` when
///     promotion lands.
///   - `is_read_write` / `is_const` / `original_tensor` are omitted —
///     flame-core's autograd tape records read/write explicitly at the
///     `Tensor::*` method level, so those flags would carry no
///     information in Phase 1.
pub struct OperandInfo<'a> {
    /// Source tensor (borrowed input or iterator-allocated output).
    /// `None` only while an output slot is pending allocation (used in
    /// `TensorIteratorConfig::add_output(None)` → allocation at build
    /// time). Phase 1 builders do not allocate, so this is always
    /// `Some` after a successful `add_input` and `None` for
    /// `add_output(None)` until Phase 3 fills it in.
    pub src: Option<OperandSrc<'a>>,

    /// Desired dtype for this operand. For an input, equals the source
    /// tensor's dtype (phase 1: no promotion). For a pending output,
    /// taken from `TensorIteratorConfig::declare_static_dtype_and_device`
    /// or inherited from the first input.
    pub target_dtype: DType,

    /// Whether this operand is an output (vs. an input). Outputs come
    /// first in the operands array, mirroring PyTorch.
    pub is_output: bool,

    /// Broadcast byte-strides (populated post-`compute_strides_pass`).
    /// `len() == base.shape.len()` when populated; empty before the
    /// iterator is built.
    pub stride_bytes: I64StrideVec,
}

impl<'a> std::fmt::Debug for OperandInfo<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperandInfo")
            .field("is_output", &self.is_output)
            .field("target_dtype", &self.target_dtype)
            .field("stride_bytes", &self.stride_bytes)
            .field("has_src", &self.src.is_some())
            .finish()
    }
}

/// Port of `at::TensorIteratorBase` (TensorIterator.h:248).
///
/// Phase 1 scope: fields + accessors + the geometry helpers (shape,
/// byte-stride, contiguity, 32-bit indexing flag). The `build_*_op`
/// methods are stubbed — see top-of-file comment.
///
/// Lifetime `'a` tracks borrowed input tensors. Outputs created via
/// `add_output(None)` become `Owned` at build time and carry no
/// external lifetime; the iterator itself stays `'a`-generic to
/// accommodate the borrowed inputs that dominate the common case.
pub struct TensorIteratorBase<'a> {
    /// Ordered operands. Outputs first, then inputs.
    /// `SmallVec<_; 4>` matches PyTorch's `SmallVector<OperandInfo, 4>`.
    pub(crate) operands: smallvec::SmallVec<[OperandInfo<'a>; 4]>,

    /// Broadcast + coalesced iteration shape. Same role as PyTorch's
    /// `shape_`. Populated by `compute_shape` during build.
    pub(crate) shape_: DimVec,

    /// Permutation from `reorder_dimensions`, same semantics as
    /// PyTorch's `perm_` (`perm_[new_dim] = old_dim`). Cleared / invalid
    /// after `coalesce_dimensions` runs.
    pub(crate) perm_: DimVec,

    /// Number of output operands (prefix of `operands`).
    pub(crate) num_outputs_: usize,

    /// Number of input operands.
    pub(crate) num_inputs_: usize,

    /// Set once `coalesce_dimensions` has run. Invalidates `perm_`.
    pub(crate) has_coalesced_dimensions_: bool,

    /// Whether this iterator is for a reduction. Phase 1 never sets this
    /// to `true` but the field exists so Phase 4+ can configure it.
    pub(crate) is_reduction_: bool,

    /// Set by `compute_shape` when every defined operand had an equal
    /// logical shape (no broadcasting). Phase 3 uses this for fast-set-up
    /// dispatch.
    pub(crate) all_ops_same_shape_: bool,

    /// Whether the iterator represents the final-output stage of a
    /// multi-step kernel (e.g. a fused reduction). Always `true` in
    /// Phase 1 — we don't split reductions yet.
    pub(crate) final_output_: bool,

    /// Common dtype across all operands. Phase 1: always equals the
    /// static dtype when set, or the first input's dtype. Phase 8 adds
    /// the full promotion table.
    pub(crate) common_dtype_: Option<DType>,

    /// Bypassed static dtype (from `declare_static_dtype_and_device`).
    pub(crate) static_dtype_: Option<DType>,

    /// Whether iteration can use 32-bit indexing. Populated by
    /// `compute_32bit_indexing_flag` post-geometry.
    pub(crate) requires_32bit_indexing_: bool,

    /// Fast-setup classification. `None` until compute-types runs. Phase
    /// 1 only classifies plain `Contiguous` / `None`.
    pub(crate) fast_setup_: FastSetupType,
}

impl<'a> TensorIteratorBase<'a> {
    /// Construct an empty iterator. Not public: built via
    /// `TensorIteratorConfig::build*`.
    pub(crate) fn new() -> Self {
        Self {
            operands: smallvec::SmallVec::new(),
            shape_: smallvec::SmallVec::new(),
            perm_: smallvec::SmallVec::new(),
            num_outputs_: 0,
            num_inputs_: 0,
            has_coalesced_dimensions_: false,
            is_reduction_: false,
            all_ops_same_shape_: false,
            final_output_: true,
            common_dtype_: None,
            static_dtype_: None,
            requires_32bit_indexing_: true,
            fast_setup_: FastSetupType::None,
        }
    }

    // ---------- Accessors (port of TensorIterator.h:272–360 subset) ----------

    /// Number of iteration dims after reorder/coalesce. Port of
    /// `TensorIteratorBase::ndim()` (h:272).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape_.len()
    }

    /// Iteration shape. Port of `TensorIteratorBase::shape()` (h:275).
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape_
    }

    /// Number of iteration elements (product of shape). Port of
    /// `TensorIteratorBase::numel()` (cpp:691).
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape_.iter().product()
    }

    /// Total number of operands (inputs + outputs). Port of
    /// `TensorIteratorBase::ntensors()` (h:279).
    #[inline]
    pub fn ntensors(&self) -> usize {
        self.operands.len()
    }

    /// Number of output operands. Port of `noutputs()` (h:282).
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.num_outputs_
    }

    /// Number of input operands. Port of `ninputs()` (h:285).
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs_
    }

    /// Is this iterator for a reduction? Port of `is_reduction()` (h: field).
    #[inline]
    pub fn is_reduction(&self) -> bool {
        self.is_reduction_
    }

    /// Common dtype. Panics if never set — same contract as PyTorch's
    /// `common_dtype()` (h:313) which `TORCH_INTERNAL_ASSERT`s.
    #[inline]
    pub fn common_dtype(&self) -> DType {
        self.common_dtype_
            .expect("common_dtype queried before it was set")
    }

    /// Byte strides for operand `arg`. Port of
    /// `TensorIteratorBase::strides(arg)` (h:306), which in PyTorch
    /// returns `stride_bytes`. Flame-core keeps the same contract so
    /// Phase 2 kernels can pass the returned slice directly to a
    /// device-side `int64_t*`.
    #[inline]
    pub fn byte_strides(&self, arg: usize) -> &[i64] {
        &self.operands[arg].stride_bytes
    }

    /// Element strides for operand `arg`. flame-core-native accessor (no
    /// exact PyTorch equivalent — PyTorch exposes byte-strides). Useful
    /// when the caller wants to feed the stride array to flame-core's
    /// own kernels that expect element strides (the common case).
    ///
    /// Returns `None` if byte-strides were never populated, or if the
    /// dtype size is zero (impossible with real DTypes).
    pub fn element_strides(&self, arg: usize) -> Option<crate::shape::Strides> {
        let op = &self.operands[arg];
        let elem_size = op.target_dtype.size_in_bytes();
        if elem_size == 0 {
            return None;
        }
        let mut out: crate::shape::Strides = smallvec::smallvec![0usize; op.stride_bytes.len()];
        let es = elem_size as i64;
        for (i, &b) in op.stride_bytes.iter().enumerate() {
            // Broadcast dims have stride_bytes == 0; preserve that in
            // element units.
            if b == 0 {
                out[i] = 0;
            } else if b % es != 0 {
                // Non-divisible byte-stride shouldn't happen for a
                // single-dtype iterator — surface it rather than silently
                // truncate.
                return None;
            } else {
                out[i] = (b / es) as usize;
            }
        }
        Some(out)
    }

    /// Operand dtype. Port of `dtype(arg)` (h:310).
    #[inline]
    pub fn dtype(&self, arg: usize) -> DType {
        self.operands[arg].target_dtype
    }

    /// True if the iterator is a trivial 1-D contiguous walk. Port of
    /// `TensorIteratorBase::is_contiguous` (cpp:806).
    ///
    /// Matches PyTorch: true when `numel == 1`, or `ndim == 1` and every
    /// operand's inner byte-stride equals its element size. Coalescing
    /// reduces "row-major contig across all ranks" down to ndim==1, so
    /// this acts as "fully coalesce-able" in practice.
    pub fn is_contiguous(&self) -> bool {
        if self.numel() == 1 {
            return true;
        }
        if self.ndim() != 1 {
            return false;
        }
        for op in &self.operands {
            if op.stride_bytes.is_empty() {
                continue;
            }
            let es = op.target_dtype.size_in_bytes() as i64;
            if op.stride_bytes[0] != es {
                return false;
            }
        }
        true
    }

    /// True if stride arithmetic fits in 32 bits. Port of
    /// `TensorIteratorBase::can_use_32bit_indexing` (cpp:1300). Phase 1
    /// computes this during build; the accessor here reports the cached
    /// value so callers don't re-walk the strides on every kernel call.
    #[inline]
    pub fn can_use_32bit_indexing(&self) -> bool {
        self.requires_32bit_indexing_
    }

    /// Base data pointer of operand `arg`. Port of `data_ptr(arg)`
    /// (h:309).  Returns `None` for a pending (unallocated) output.
    ///
    /// Phase 1 note: flame-core's BF16 storage exposes its device
    /// pointer via `Tensor::as_device_ptr_bf16`. To stay dtype-agnostic
    /// in the iterator, this accessor returns `*const u8` (the
    /// byte-level base). Phase 2 kernels cast to the appropriate typed
    /// pointer from the functor.
    pub fn data_ptr_u8(&self, arg: usize) -> Option<*const u8> {
        let src = self.operands[arg].src.as_ref()?;
        let tensor = src.tensor();
        match tensor.dtype() {
            DType::BF16 => tensor
                .as_device_ptr_bf16("tensor_iterator::data_ptr_u8")
                .ok()
                .map(|p| p as *const u8),
            other => panic!(
                "TensorIteratorBase::data_ptr_u8: unsupported dtype {:?} in Phase 1 (BF16 only)",
                other
            ),
        }
    }

    // ---------- Build entries (Phase 3) ----------

    /// Port of `TensorIteratorBase::build_unary_op` (TensorIterator.h:585).
    ///
    /// The PyTorch version is a method that mutates `*this`; flame-core
    /// keeps Phase 1's free-function signature returning a fresh
    /// `TensorIteratorBase<'a>` so the iterator's lifetime matches the
    /// borrowed input. The behaviour matches PyTorch's:
    ///
    ///   * `set_check_mem_overlap(true)` — noted but no mem-overlap check
    ///     yet (Phase-3-deferred per config.rs doc).
    ///   * `cast_common_dtype_to_outputs(false)`
    ///   * `enforce_safe_casting_to_output(false)`
    ///   * `check_all_same_dtype(true)` — Phase 3 enforces this in
    ///     `TensorIteratorConfig::build`.
    ///
    /// When `out` is `None`, the iterator allocates a fresh contiguous
    /// output via `allocate_or_resize_outputs`. When `out` is `Some`, its
    /// shape must match the broadcast iteration shape or `build` returns
    /// `ShapeMismatch` (flame-core does not implement implicit resize —
    /// see config.rs `will_resize` discussion).
    pub fn build_unary_op(
        out: Option<&'a Tensor>,
        a: &'a Tensor,
    ) -> Result<TensorIteratorBase<'a>> {
        super::config::TensorIteratorConfig::new()
            .set_check_mem_overlap(true)
            .cast_common_dtype_to_outputs(false)
            .enforce_safe_casting_to_output(false)
            .check_all_same_dtype(true)
            .add_output(out)
            .add_input(a)
            .build()
    }

    /// Port of `TensorIteratorBase::build_binary_op` (TensorIterator.h:571).
    ///
    /// PyTorch's `BINARY_OP_CONFIG()` defaults:
    ///   * `set_check_mem_overlap(true)`
    ///   * `allow_cpu_scalars(true)` — irrelevant on flame-core (no CPU
    ///     tensors), but we set the flag for fidelity.
    ///   * `promote_inputs_to_common_dtype(true)` — stored but not acted on
    ///     in Phase 3 (promotion lands at Phase 8). The `compute_types`
    ///     flow still picks the first defined operand's dtype, matching
    ///     the current BF16-only contract.
    ///   * `cast_common_dtype_to_outputs(true)`
    ///   * `enforce_safe_casting_to_output(true)`
    pub fn build_binary_op(
        out: Option<&'a Tensor>,
        a: &'a Tensor,
        b: &'a Tensor,
    ) -> Result<TensorIteratorBase<'a>> {
        super::config::TensorIteratorConfig::new()
            .set_check_mem_overlap(true)
            .allow_cpu_scalars(true)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(true)
            .add_output(out)
            .add_input(a)
            .add_input(b)
            .build()
    }

    // ---------- allocate_or_resize_outputs (Phase 3) ----------

    /// Port of `TensorIteratorBase::allocate_or_resize_outputs`
    /// (TensorIterator.cpp:574).
    ///
    /// Runs after `reorder_dimensions` has populated `shape_` and `perm_`,
    /// and before `coalesce_dimensions`. For each output:
    ///
    ///   * If the slot is `None` (caller wrote `add_output(None)`):
    ///     allocate a fresh **contiguous** tensor of shape `invert_perm(shape_)`
    ///     and element-strides `invert_perm(reordered contig strides)`. In
    ///     flame-core's Phase 3 this reduces to a row-major contig output
    ///     — the iterator-side byte-strides for the new output are then
    ///     the reordered contig strides.
    ///
    ///   * If the slot is `Some` and the output is already defined:
    ///     verify its shape matches the iteration shape (post-reorder).
    ///     Mismatch → `ShapeMismatch`. flame-core tensors are shape-fixed;
    ///     PyTorch's implicit resize path is intentionally omitted.
    ///
    /// PyTorch's `inverted` fast-path (contig row-major output) is the
    /// only path taken in Phase 3 — flame-core's freshly-allocated output
    /// is always row-major contig in its invert-perm'd shape.
    pub fn allocate_or_resize_outputs(&mut self) -> Result<()> {
        // Compute invert-perm of the current (reordered) shape: the
        // "logical" output shape the user sees. For rank ≤ 1 this is a
        // no-op identity.
        let ndim = self.shape_.len();
        let perm_slice = self.perm_.as_slice();
        // invert_perm[old_dim] = new_dim
        let inverted_shape: DimVec = if ndim == 0 {
            DimVec::new()
        } else if perm_slice.len() == ndim {
            let mut out: DimVec = smallvec::smallvec![0usize; ndim];
            for new_dim in 0..ndim {
                let old_dim = perm_slice[new_dim];
                debug_assert!(old_dim < ndim, "perm out of range");
                out[old_dim] = self.shape_[new_dim];
            }
            out
        } else {
            // Defensive: no perm recorded (e.g. rank-0/1 fast path) —
            // treat as identity.
            self.shape_.clone()
        };

        for i in 0..self.num_outputs_ {
            let op = &mut self.operands[i];
            match &op.src {
                None => {
                    // Need device + dtype to allocate. Device comes from
                    // the first defined operand (every operand must be on
                    // the same device per `check_all_same_device`). Dtype
                    // is the operand's target_dtype (already resolved in
                    // populate_operands).
                    let dtype = op.target_dtype;
                    if ndim == 0 && inverted_shape.is_empty() {
                        return Err(Error::InvalidOperation(
                            "allocate_or_resize_outputs: cannot allocate rank-0 output in Phase 3"
                                .into(),
                        ));
                    }
                    // Hunt for the device. Phase 3 looks at inputs first.
                    let device = {
                        let mut found = None;
                        for j in self.num_outputs_..self.operands.len() {
                            if let Some(src) = self.operands[j].src.as_ref() {
                                found = Some(src.tensor().device().clone());
                                break;
                            }
                        }
                        match found {
                            Some(d) => d,
                            None => {
                                return Err(Error::InvalidOperation(
                                    "allocate_or_resize_outputs: no defined operand to infer \
                                     output device from"
                                        .into(),
                                ));
                            }
                        }
                    };
                    let shape_obj = Shape::from_dims(inverted_shape.as_slice());
                    let allocated = Tensor::empty_dtype(shape_obj, dtype, device)?;
                    // Recompute byte-strides for the newly-allocated
                    // contig output in the iterator's reordered frame.
                    // The new tensor is row-major contig → its
                    // element-strides in the LOGICAL (invert-perm) shape
                    // are contiguous_element_strides(inverted_shape).
                    // Broadcast those back into the iterator frame by
                    // applying `perm_` to produce stride_bytes of length
                    // `ndim` in post-reorder order.
                    let logical_es =
                        contiguous_element_strides(inverted_shape.as_slice());
                    let elem_size = dtype.size_in_bytes();
                    let mut reordered: I64StrideVec =
                        smallvec::smallvec![0i64; ndim];
                    for new_dim in 0..ndim {
                        let old_dim = perm_slice[new_dim];
                        reordered[new_dim] =
                            (logical_es[old_dim] as i64) * (elem_size as i64);
                    }
                    // Also patch the just-allocated output's
                    // element-stride field in the operand info.
                    let op = &mut self.operands[i];
                    op.src = Some(super::base::OperandSrc::Owned(allocated));
                    op.stride_bytes = reordered;
                }
                Some(src) => {
                    // Provided output: verify the shape matches the
                    // iteration shape (in logical/invert-perm frame).
                    let t = src.tensor();
                    let provided = t.shape().dims();
                    if provided != inverted_shape.as_slice() {
                        return Err(Error::ShapeMismatch {
                            expected: Shape::from_dims(inverted_shape.as_slice()),
                            got: t.shape().clone(),
                        });
                    }
                    // Byte-strides were already populated in
                    // compute_strides + reorder. Nothing more to do.
                }
            }
        }
        Ok(())
    }
}

/// Convenience: produce a stride_bytes vector matching a tensor's real
/// byte-stride layout. Used to populate `OperandInfo::stride_bytes`
/// before broadcast inference in some call paths.
#[allow(dead_code)]
pub(crate) fn tensor_byte_strides(t: &Tensor) -> I64StrideVec {
    let elem_size = t.dtype().size_in_bytes();
    element_strides_to_bytes(&t.strides(), elem_size)
}

// ---------- Phase 4 plumbing: owned-output extraction ----------

impl<'a> TensorIteratorBase<'a> {
    /// Extract an iterator-allocated output by index, consuming the iterator.
    /// Returns `Err(InvalidOperation)` if the output slot is out of range,
    /// or if the slot holds a borrowed tensor (caller-provided output) —
    /// borrowed outputs stay with the caller, the iterator owns nothing.
    ///
    /// Phase 4 calls this after `build_unary_op(None, &input)` /
    /// `build_binary_op(None, &a, &b)` to retrieve the freshly-allocated
    /// output tensor for returning from `Tensor::silu()` / etc.
    pub fn take_output(mut self, idx: usize) -> Result<Tensor> {
        if idx >= self.num_outputs_ {
            return Err(Error::InvalidOperation(format!(
                "take_output({idx}): only {} outputs exist",
                self.num_outputs_
            )));
        }
        let slot = self.operands[idx].src.take();
        match slot {
            Some(OperandSrc::Owned(t)) => Ok(t),
            Some(OperandSrc::Borrowed(_)) => Err(Error::InvalidOperation(format!(
                "take_output({idx}): output was caller-provided; \
                 iterator cannot take ownership of a borrowed tensor"
            ))),
            None => Err(Error::InvalidOperation(format!(
                "take_output({idx}): output slot is empty (was it \
                 already taken?)"
            ))),
        }
    }
}

// ---------- Phase 4 plumbing: IterMetadata marshalling + stream ----------

#[cfg(feature = "cuda")]
impl<'a> TensorIteratorBase<'a> {
    /// Build a `IterMetadata` POD from the current iterator state. Callable
    /// after `TensorIteratorConfig::build()` finished (i.e. the full shape /
    /// stride / alloc / reorder / coalesce pipeline has run).
    ///
    /// Narrow views: the iterator's per-operand `stride_bytes` does NOT
    /// include the source tensor's `view_offset`; that's passed through
    /// `offsets_elems[arg]` as element offsets so the device side can add
    /// it after casting `data_ptrs[arg]` to the typed pointer.
    ///
    /// Iterator-allocated outputs always have offset 0 (fresh contig row-
    /// major), so `offsets_elems[0] = 0` whenever the caller used
    /// `add_output(None)`.
    ///
    /// Operand layout in the POD matches PyTorch's convention: outputs
    /// first (indices `[0..num_outputs)`) then inputs
    /// (`[num_outputs..num_args)`).
    pub fn build_iter_metadata(&self) -> Result<super::iter_metadata::IterMetadata> {
        use super::iter_metadata::{IterMetadata, FLAME_MAX_DIMS, MAX_NARGS};

        let ndim = self.shape_.len();
        if ndim > FLAME_MAX_DIMS {
            return Err(Error::InvalidOperation(format!(
                "build_iter_metadata: ndim {ndim} exceeds FLAME_MAX_DIMS ({FLAME_MAX_DIMS})"
            )));
        }
        let num_args = self.operands.len();
        if num_args == 0 || num_args > MAX_NARGS {
            return Err(Error::InvalidOperation(format!(
                "build_iter_metadata: num_args {num_args} out of [1, {MAX_NARGS}]"
            )));
        }

        let mut meta = IterMetadata::zeroed();
        meta.ndim = ndim as i32;
        meta.num_args = num_args as i32;
        meta.num_outputs = self.num_outputs_ as i32;
        meta._pad = 0;
        meta.numel = self.numel() as i64;
        meta.is_contiguous = self.is_contiguous();
        meta.requires_32bit_indexing = self.requires_32bit_indexing_;

        // Sizes (post-coalesce iteration shape).
        for dim in 0..ndim {
            meta.sizes[dim] = self.shape_[dim] as i64;
        }

        // Per-operand element strides + offsets + base pointers.
        for (arg, op) in self.operands.iter().enumerate() {
            if op.stride_bytes.len() != ndim {
                return Err(Error::InvalidOperation(format!(
                    "build_iter_metadata: operand {arg} has stride len {} != ndim {ndim}",
                    op.stride_bytes.len()
                )));
            }
            let elem_size = op.target_dtype.size_in_bytes() as i64;
            if elem_size == 0 {
                return Err(Error::InvalidOperation(format!(
                    "build_iter_metadata: operand {arg} has zero-byte dtype"
                )));
            }
            for dim in 0..ndim {
                let b = op.stride_bytes[dim];
                let es = if b == 0 {
                    // Broadcast dim — stride stays 0.
                    0
                } else {
                    if b % elem_size != 0 {
                        return Err(Error::InvalidOperation(format!(
                            "build_iter_metadata: operand {arg} dim {dim} byte-stride {b} \
                             not divisible by elem_size {elem_size}"
                        )));
                    }
                    b / elem_size
                };
                meta.strides[arg][dim] = es;
            }

            // Offset + data_ptr. Iterator-allocated (Owned) outputs have
            // offset 0. Borrowed inputs/outputs carry their source
            // tensor's view_offset.
            let (ptr, offset_elems) = match op.src.as_ref() {
                Some(OperandSrc::Owned(t)) => {
                    let raw = match t.dtype() {
                        DType::BF16 => t.as_device_ptr_bf16(
                            "build_iter_metadata: owned output ptr",
                        )?,
                        other => {
                            return Err(Error::InvalidOperation(format!(
                                "build_iter_metadata: operand {arg} dtype {other:?} \
                                 not supported in Phase 4 (BF16-only)"
                            )))
                        }
                    };
                    (raw as *mut std::os::raw::c_void, t.offset() as i64)
                }
                Some(OperandSrc::Borrowed(t)) => {
                    let raw = match t.dtype() {
                        DType::BF16 => t.as_device_ptr_bf16(
                            "build_iter_metadata: borrowed operand ptr",
                        )?,
                        other => {
                            return Err(Error::InvalidOperation(format!(
                                "build_iter_metadata: operand {arg} dtype {other:?} \
                                 not supported in Phase 4 (BF16-only)"
                            )))
                        }
                    };
                    (raw as *mut std::os::raw::c_void, t.offset() as i64)
                }
                None => {
                    return Err(Error::InvalidOperation(format!(
                        "build_iter_metadata: operand {arg} slot is empty — \
                         allocate_or_resize_outputs must run before this call"
                    )))
                }
            };
            meta.data_ptrs[arg] = ptr;
            meta.offsets_elems[arg] = offset_elems;
        }

        Ok(meta)
    }

    /// Device stream for the iterator's operands. All defined operands are
    /// on the same device (enforced by `check_all_same_device` in the
    /// Phase-3 config), so any of them works; we take the first defined
    /// operand.
    ///
    /// Returns `Err` if no operand is defined. Flame-core's stream handle
    /// is a raw `*mut c_void` because the cudarc stream lifetime is managed
    /// by the device, not the caller.
    pub fn stream(&self) -> Result<*mut std::os::raw::c_void> {
        use crate::device::CudaStreamRawPtrExt;

        for op in &self.operands {
            if let Some(src) = op.src.as_ref() {
                return Ok(src.tensor().device().cuda_stream_raw_ptr());
            }
        }
        Err(Error::InvalidOperation(
            "TensorIteratorBase::stream: no defined operand".into(),
        ))
    }
}
