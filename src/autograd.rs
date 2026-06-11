#![allow(
    unused_variables,
    unused_mut,
    unused_imports,
    dead_code,
    unreachable_patterns
)]
// TODO(Phase 3): tighten autograd implementation; temp allow for unused scaffolding.

//! Automatic differentiation engine for FLAME
//! This module provides a clean, integrated autograd system that works
//! seamlessly with the Tensor API.
pub mod policy;

use crate::activation_offload::{ActivationOffloadPool, OffloadHandle};
use crate::autograd::policy::GradStorePolicy;
use crate::cuda::ffi;
use crate::cuda_kernels_gpu::CudaKernels;
use crate::cuda_ops::GpuOps;
use crate::device::CudaStreamRawPtrExt;
use crate::gradient::GradientMap;
use crate::gradient_checkpointing::{CHECKPOINT_HAS_ENTRIES, CHECKPOINT_MANAGER};
use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Inline-storage buffer for tensors saved on the tape. Most ops save 0-3
/// tensors (input, weight, bias), so storing them inline eliminates a Vec
/// heap allocation per recorded op — and with ~2660 tape entries per Klein
/// 4B step that adds up.
pub type SavedTensors = SmallVec<[(TensorId, Tensor); 3]>;

/// Phase 2 — `SavedRef`-based saved tensors. Inline-sized to 4 (the spec).
/// Carries the storage version snapshot so backward errors loudly if any
/// op modified the saved tensor in-place between save and unpack.
pub type SavedRefs = SmallVec<[crate::saved_ref::SavedRef; 4]>;

/// Return type of `compute_gradients` per tape entry. Most ops produce 1-3
/// grads, so inlining up to 3 avoids per-op heap allocation during backward.
pub type GradVec = SmallVec<[(TensorId, Tensor); 3]>;

/// Atomic mirror of `AutogradContextInner::enabled`. Checked by tensor ops
/// BEFORE constructing Op enums or cloning saved_tensors, avoiding GPU memcpys
/// and mutex locks when autograd is disabled (e.g. during checkpoint forward).
/// Updated alongside `ctx.enabled` in `record_op`, `checkpoint`, `backward`, etc.
static AUTOGRAD_ENABLED: AtomicBool = AtomicBool::new(true);

thread_local! {
    /// Non-zero only while a checkpoint closure is being replayed by backward.
    /// Model code can use this to issue backward-order prefetches without
    /// changing normal forward checkpoint behavior.
    static CHECKPOINT_RECOMPUTE_DEPTH: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

struct CheckpointRecomputeGuard;

impl CheckpointRecomputeGuard {
    #[inline]
    fn enter() -> Self {
        CHECKPOINT_RECOMPUTE_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for CheckpointRecomputeGuard {
    fn drop(&mut self) {
        CHECKPOINT_RECOMPUTE_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "checkpoint recompute depth underflow");
            depth.set(current.saturating_sub(1));
        });
    }
}

lazy_static::lazy_static! {
    /// Global autograd context - thread-safe
    static ref AUTOGRAD_CONTEXT: Mutex<AutogradContextInner> = Mutex::new(AutogradContextInner::new());

    /// Test-only: when `Some`, the listed `TensorId`s have their gradients
    /// cloned into `RETAINED_INTERMEDIATE_GRADS` during backward so callers
    /// can probe intermediate gradients despite autograd's drain-on-take
    /// semantics. Set via `AutogradContext::retain_intermediate_grads`,
    /// drained via `take_retained_intermediate_grads`. None outside tests.
    static ref RETAINED_INTERMEDIATE_GRAD_IDS:
        Mutex<Option<std::collections::HashSet<TensorId>>> = Mutex::new(None);
    static ref RETAINED_INTERMEDIATE_GRADS:
        Mutex<HashMap<TensorId, Tensor>> = Mutex::new(HashMap::new());
}

/// Global activation offload pool. Set via `set_activation_offload_pool`
/// at training setup; can be torn down via `clear_activation_offload_pool`
/// when the trainer wants to free the pool's GPU staging buffers (e.g.
/// to free memory for an inline VAE decode at high resolution before
/// resuming training).
static ACTIVATION_POOL: std::sync::RwLock<Option<Arc<Mutex<ActivationOffloadPool>>>> =
    std::sync::RwLock::new(None);

/// Install the activation offload pool for checkpoint_offload to use.
/// Call at training setup. Replaces any previously-installed pool (the
/// previous pool's Arc is dropped, releasing its pinned host memory + GPU
/// staging buffers when the last reference goes away).
pub fn set_activation_offload_pool(pool: Arc<Mutex<ActivationOffloadPool>>) -> Result<()> {
    let mut g = ACTIVATION_POOL
        .write()
        .map_err(|_| Error::InvalidOperation("activation pool RwLock poisoned".into()))?;
    *g = Some(pool);
    Ok(())
}

/// Drop the global activation offload pool, releasing its pinned host
/// memory and GPU staging buffers. Subsequent `checkpoint_offload` calls
/// fall back to plain recompute checkpoint until a new pool is installed
/// via `set_activation_offload_pool`.
///
/// Used by trainers that need to free the pool's GPU memory for an
/// inline high-resolution sample/decode that would otherwise OOM.
pub fn clear_activation_offload_pool() {
    if let Ok(mut g) = ACTIVATION_POOL.write() {
        *g = None;
    }
}

/// Global grow-on-demand activation cache (Phase 1 of OFFLOAD_NEXT_GEN_DESIGN).
/// Parallel to `ACTIVATION_POOL` but uses the new dynamic-slab cache and
/// the `checkpoint_offload_boundary` API shape (closure takes &[Tensor]).
/// None by default; trainers that want narrow-scope boundary offload call
/// `set_grow_activation_cache` at setup.
static GROW_CACHE: std::sync::RwLock<
    Option<Arc<Mutex<crate::activation_offload::GrowOnDemandActivationCache>>>,
> = std::sync::RwLock::new(None);

/// Install the grow-on-demand activation cache for `checkpoint_offload_boundary`.
pub fn set_grow_activation_cache(
    cache: Arc<Mutex<crate::activation_offload::GrowOnDemandActivationCache>>,
) -> Result<()> {
    let mut g = GROW_CACHE
        .write()
        .map_err(|_| Error::InvalidOperation("grow cache RwLock poisoned".into()))?;
    *g = Some(cache);
    Ok(())
}

/// Drop the global grow-on-demand activation cache.
pub fn clear_grow_activation_cache() {
    if let Ok(mut g) = GROW_CACHE.write() {
        *g = None;
    }
}

/// Rotation layout for fused-RoPE autograd ops.
///
/// RoPE is an orthogonal rotation of `head_dim` channels; two layouts are in
/// use in this codebase:
/// - `Interleaved`: pairs `(2d, 2d+1)` rotate together. Used by FLUX, Klein,
///   Chroma, Wan, Z-Image, HiDream-O1 MRoPE.
/// - `Halfsplit`: pairs `(d, d+half)` rotate together (HuggingFace
///   `rotate_half` convention). Used by Qwen3, LLaMA, Mistral.
/// - `HalfsplitPytorch`: same channel pairing as `Halfsplit`, but preserves
///   PyTorch/HF BF16 expression round points in both forward and backward.
///
/// The two are different rotations of `head_dim`. Backward MUST apply the
/// same layout as forward — otherwise the gradient direction is essentially
/// random while its magnitude looks fine. Symptom: dx cos ≈ 0.01-0.05
/// against the autograd reference, dx norm matches within BF16 noise.
///
/// Previously the backward dispatcher shape-sniffed the saved `cos` tensor
/// (rank-3 → assumed Interleaved). HiDream-O1's MRoPE emits cos of shape
/// `[1, S, half]` (rank-3) but the forward was Halfsplit → backward applied
/// the wrong rotation, collapsing Q/K LoRA-B gradients. The fix is this
/// explicit tag carried alongside `Op::RoPePrecomputed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// (2d, 2d+1) pairs. `bf16_ops::rope_fused_bf16`.
    Interleaved,
    /// (d, d+half) pairs. `bf16_ops::rope_halfsplit_bf16`.
    Halfsplit,
    /// (d, d+half) pairs with PyTorch BF16 expression round points.
    /// `bf16_ops::rope_halfsplit_bf16_pytorch`.
    HalfsplitPytorch,
}

/// Operation types for autograd
#[derive(Debug, Clone)]
pub enum Op {
    Add {
        lhs: TensorId,
        rhs: TensorId,
        lhs_shape: Shape,
        rhs_shape: Shape,
    },
    Sub {
        lhs: TensorId,
        rhs: TensorId,
    },
    Mul {
        lhs: TensorId,
        rhs: TensorId,
    },
    Div {
        lhs: TensorId,
        rhs: TensorId,
        lhs_shape: Shape,
        rhs_shape: Shape,
    },
    MulScalar {
        input: TensorId,
        scalar: f32,
    },
    AddScalar {
        input: TensorId,
        scalar: f32,
    },
    MatMul {
        lhs: TensorId,
        rhs: TensorId,
    },
    ReLU {
        input: TensorId,
    },
    GELU {
        input: TensorId,
    },
    SiLU {
        input: TensorId,
    },
    Tanh {
        input: TensorId,
    },
    Sigmoid {
        input: TensorId,
    },
    Square {
        input: TensorId,
    },
    Sqrt {
        input: TensorId,
    },
    Sum {
        input: TensorId,
        input_shape: Shape,
    },
    Mean {
        input: TensorId,
        input_shape: Shape,
    },
    Transpose {
        input: TensorId,
    },
    Conv2d {
        input: TensorId,
        weight: TensorId,
        stride: usize,
        padding: usize,
    },
    Linear {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
    },
    LayerNorm {
        input: TensorId,
        normalized_shape: Vec<usize>,
    },
    RMSNorm {
        input: TensorId,
        weight: Option<TensorId>,
        eps: f32,
        inv_rms: TensorId,
        normalized_shape: Vec<usize>,
    },
    BatchMatMul {
        lhs: TensorId,
        rhs: TensorId,
    },
    Reshape {
        input: TensorId,
        new_shape: Vec<usize>,
    },
    Permute {
        input: TensorId,
        dims: Vec<usize>,
    },
    /// Broadcast `input` (shape `src_shape`) to `dst_shape`. Backward sums
    /// `output_grad` along every broadcast axis (size mismatch or left-padded
    /// axes), then reshapes back to `src_shape`. Without this op, autograd
    /// silently detached any grad-bearing tensor passed through
    /// `Tensor::broadcast_to` whenever the shapes actually differed (the
    /// fast-path reshape branch was fine). Surfaced 2026-05-21 as the
    /// HiDream-O1 `t_embedder1.mlp.{0,2}` LoRA dead-grad — the LoRA output
    /// reached `scatter_tms_token` via `reshape().broadcast_to(...)` and
    /// the broadcast detached it.
    Broadcast {
        input: TensorId,
        src_shape: Vec<usize>,
        dst_shape: Vec<usize>,
    },
    /// Nearest-neighbor 2D upsample. SDXL/SD3/Cascade up-blocks use this.
    UpsampleNearest2D {
        input: TensorId,
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
    },
    /// 2D max pooling. Backward is the scatter of upstream grad to the
    /// max-arg positions (recomputed from the saved input, since the
    /// forward kernel does not currently emit indices — see
    /// `pooling::MaxPool2d::backward` which takes `_indices: Option<&Tensor>`
    /// and ignores it).
    MaxPool2D {
        input: TensorId,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    },
    AddBias {
        input: TensorId,
        bias: TensorId,
    },
    SumDim {
        input: TensorId,
        dim: usize,
    },
    SumDimKeepdim {
        input: TensorId,
        dim: usize,
    },
    SumDims {
        input: TensorId,
        dims: Vec<usize>,
    },
    Repeat {
        input: TensorId,
        repeats: Vec<usize>,
    },
    MaxDim {
        input: TensorId,
        dim: usize,
        keepdim: bool,
    },
    Clamp {
        input: TensorId,
        min: f32,
        max: f32,
    },
    Embedding {
        weight: TensorId,
        indices: TensorId,
    },
    IndexSelect {
        input: TensorId,
        indices: TensorId,
        dim: usize,
    },
    /// Replace slices at `indices` along `dim` of `input` with the
    /// corresponding slices from `values`. Backward: grad w.r.t. input is
    /// upstream with indexed rows zeroed; grad w.r.t. values is index_select
    /// of upstream at `indices`.
    IndexAssign {
        input: TensorId,
        indices: TensorId,
        values: TensorId,
        dim: usize,
    },
    Cat {
        inputs: Vec<TensorId>,
        dim: usize,
    },
    Split {
        input: TensorId,
        sizes: Vec<usize>,
        dim: usize,
    },
    Slice {
        input: TensorId,
        ranges: Vec<(usize, usize)>,
        input_shape: Shape,
    },
    Abs {
        input: TensorId,
    },
    Log {
        input: TensorId,
    },
    Softmax {
        input: TensorId,
        dim: isize,
    },
    LogSoftmax {
        input: TensorId,
        dim: isize,
    },
    Maximum {
        a: TensorId,
        b: TensorId,
    },
    Minimum {
        a: TensorId,
        b: TensorId,
    },
    Where {
        cond: TensorId,
        t: TensorId,
        f: TensorId,
    },
    MSELoss {
        predictions: TensorId,
        targets: TensorId,
        num_elements: usize,
    },
    L1Loss {
        predictions: TensorId,
        targets: TensorId,
        num_elements: usize,
    },
    HuberLoss {
        predictions: TensorId,
        targets: TensorId,
        delta: f32,
        num_elements: usize,
    },
    BCELoss {
        predictions: TensorId,
        targets: TensorId,
        num_elements: usize,
    },
    NLLLoss {
        log_probs: TensorId,
        targets: TensorId,
        batch_size: usize,
    },
    GroupNorm {
        input: TensorId,
        num_groups: usize,
        weight: Option<TensorId>,
        bias: Option<TensorId>,
    },
    FlashAttention {
        query: TensorId,
        key: TensorId,
        value: TensorId,
        mask: Option<TensorId>,
        scale: f32,
        causal: bool,
        padding_lens: Option<(usize, usize)>,
        /// Saved output tensor id (recorded by `sdpa::forward_train`).
        /// `None` keeps backward-compat with any older record sites.
        /// Stage 2 fix (2026-05-12): replaces the broken shape-find heuristic
        /// at the backward dispatch — `fetch_saved` materializes non-contig
        /// views with fresh `TensorId`s, so the id-exclusion never fired and
        /// Q was being picked up as O. Direct id lookup fixes it.
        output: Option<TensorId>,
        /// Saved Stats (LSE) tensor id for cuDNN backward.
        stats: Option<TensorId>,
    },
    /// HiDream-O1 mixed self-attention: prefix rows are causal within the
    /// prefix, suffix rows attend to the full sequence. Forward is allowed to
    /// use a faster structured decomposition, but backward recomputes as one
    /// masked SDPA so shared K/V gradients match the PyTorch single-op oracle.
    PrefixCausalFullAttention {
        query: TensorId,
        key: TensorId,
        value: TensorId,
        prefix_len: usize,
        scale: f32,
    },
    SageAttention {
        query_id: TensorId,
        key_id: TensorId,
        value_id: TensorId,
        scale: f32,
        causal: bool,
        quantized: bool,
    },
    /// NHWC conv2d op wrapper using NCHW kernels under the hood
    Conv2dNHWC {
        input: TensorId,
        weight: TensorId,
        stride: usize,
        padding: usize,
    },
    Cast {
        input: TensorId,
        from: DType,
        to: DType,
    },
    /// Activation checkpoint: stores input + recompute function.
    /// During backward, re-runs forward from saved input to rebuild the
    /// sub-tape, backward through it, then drops everything.
    /// The `recompute_fn` is the forward closure captured as a trait object.
    Checkpoint {
        input: TensorId,
        /// Number of tape entries the original forward produced (for validation).
        original_tape_len: usize,
    },
    /// Phase 2b boundary checkpoint: inputs pushed to grow cache; tape entry
    /// holds only the input IDs and cache handles, NOT strong tensor refs.
    /// Between forward and backward, the original input GPU storage can be
    /// freed once all clones drop. At backward time, the recompute closure
    /// (stored in `ctx.checkpoint_fns` keyed by output_id) pulls inputs
    /// from cache, calls the user closure with pulled tensors, and the
    /// resulting sub-tape is walked in reverse.
    ///
    /// Semantically equivalent to `Op::Checkpoint` but with the strong-ref
    /// path replaced by cache replay. Single-shot (one push, one pull).
    CheckpointOffloadBoundary {
        /// IDs of the original input tensors (for grad routing back to the
        /// outer autograd graph). Storage-less — we don't hold tensor refs.
        input_ids: Vec<TensorId>,
        /// Side-channel populated by the recompute closure with the IDs
        /// the cache assigned to the pulled tensors. Backward reads this
        /// after `recompute_fn()` returns to map pulled-ID grads back to
        /// `input_ids`. Same length and order as `input_ids`.
        pulled_ids_slot: Arc<Mutex<Vec<TensorId>>>,
    },
    /// Fused SwiGLU: silu(gate) * up in one kernel.
    /// Backward: d_gate = dsilu(gate) * up * dout, d_up = silu(gate) * dout
    FusedSwiGLU {
        gate: TensorId,
        up: TensorId,
    },
    /// Packed SwiGLU over a tensor whose last dim is [gate | up].
    /// Backward returns one packed gradient and avoids dual narrow scatters.
    FusedSwiGLUSplit {
        input: TensorId,
    },
    /// Fused QKV split+permute output. Each recorded output contributes its
    /// Q, K, or V slice back into one full QKV gradient tensor.
    QkvSplitPermute {
        input: TensorId,
        part: u8,
        heads: usize,
        head_dim: usize,
    },
    /// Fused RoPE with precomputed cos/sin.
    /// Backward: apply_rope(grad, cos, -sin) via the SAME fused kernel that the
    /// forward used — selected by the `layout` field, NOT shape-sniffing the
    /// cos tensor. See `RopeLayout` doc for why the explicit tag matters.
    RoPePrecomputed {
        input: TensorId,
        cos: TensorId,
        sin: TensorId,
        layout: RopeLayout,
    },
    /// Fused gate-residual: out = residual + gate.unsqueeze(1) * x
    /// where residual,x are [B,N,dim] and gate is [B,dim].
    /// Backward:
    ///   grad_residual = grad_out
    ///   grad_x        = grad_out * gate.unsqueeze(1)
    ///   grad_gate     = (grad_out * x).sum(dim=1)
    GateResidual {
        residual: TensorId,
        gate: TensorId,
        x: TensorId,
    },
    /// Level 2 activation offload: runs forward ONCE with autograd enabled,
    /// captures the sub-tape, offloads all saved tensors to CPU via the
    /// ActivationOffloadPool. During backward, pulls saved tensors from CPU
    /// and walks the stored sub-tape — NO recompute needed.
    /// Fallback: if pool is full during forward, falls back to standard
    /// Op::Checkpoint (recompute, no offload).
    CheckpointOffload {
        input: TensorId,
        /// The sub-tape captured during the single forward pass. Each entry's
        /// saved_tensors have been replaced with empty vecs — the actual data
        /// lives in the offload pool, keyed by offload_map.
        sub_tape: Vec<OffloadedTapeEntry>,
    },
}

/// A tape entry where saved tensors have been offloaded to CPU. The
/// tensors themselves are replaced with OffloadHandles; during backward
/// the CheckpointOffload handler pulls them from the pool before passing
/// to compute_gradients.
#[derive(Clone, Debug)]
pub struct OffloadedTapeEntry {
    pub output_id: TensorId,
    pub op: Op,
    /// Original saved tensor IDs + shapes (needed to rebuild Tensors after pull).
    pub saved_ids: SmallVec<[TensorId; 3]>,
    /// Offload handles corresponding 1:1 with saved_ids. If a tensor could
    /// not be offloaded (e.g. non-BF16), its handle is None and the tensor
    /// is stored inline in `resident_fallback`.
    pub offload_handles: SmallVec<[Option<OffloadHandle>; 3]>,
    /// Tensors that couldn't be offloaded (non-BF16, too large, etc).
    pub resident_fallback: SmallVec<[(TensorId, Tensor); 3]>,
}

/// Entry in the computation tape.
///
/// Phase 2: two parallel saved-state paths. New ops should populate
/// `saved_refs` via `SavedRef::capture`. Legacy ops continue to populate
/// `saved_tensors`. The `FLAME_AUTOGRAD_SAVED_LEGACY=1` env knob forces
/// every call site through the legacy path (the `Into<SavedTensors>`
/// builder selects between them).
///
/// Backward consumers use `get_saved(id)` which transparently checks both
/// paths, so the migration is non-disruptive.
#[derive(Clone)]
struct TapeEntry {
    /// Output tensor ID
    output_id: TensorId,

    /// Operation that produced the output
    op: Op,

    /// Legacy saved-tensor path. Inline-sized for 0-3 tensors. Populated
    /// when call sites pass `vec![(id, tensor)]` (unchanged callers).
    saved_tensors: SavedTensors,

    /// Phase 2 saved-ref path. Populated when call sites use
    /// `record_op_refs` or pass a `SavedRefs` to `record_op`.
    saved_refs: SavedRefs,
}

impl TapeEntry {
    /// Look up a saved tensor by ID (linear scan — fast for typical 1-3 entries).
    /// Checks SavedRef path first (with version-counter validation), then
    /// falls back to the legacy `saved_tensors` path. Returns a borrow into
    /// whichever path holds the saved tensor.
    #[inline]
    fn get_saved(&self, id: &TensorId) -> Option<&Tensor> {
        if let Some(r) = self.saved_refs.iter().find(|r| r.id == *id) {
            // Version check via unpack_ref — on mismatch returns None so the
            // backward consumer surfaces its standard "missing saved tensor"
            // error (loud-fail). This is the empirical safety net the spec
            // requires.
            return r.unpack_ref().ok();
        }
        self.saved_tensors
            .iter()
            .find(|(k, _)| k == id)
            .map(|(_, v)| v)
    }

    /// Iterate over saved tensor IDs across both paths.
    #[inline]
    fn saved_keys(&self) -> impl Iterator<Item = &TensorId> {
        self.saved_refs
            .iter()
            .map(|r| &r.id)
            .chain(self.saved_tensors.iter().map(|(k, _)| k))
    }

    /// Total saved count across both paths.
    #[inline]
    fn saved_count(&self) -> usize {
        self.saved_refs.len() + self.saved_tensors.len()
    }

    /// Positional accessor — returns the (id, &tensor) at logical index `i`
    /// across the combined view of `saved_tensors` then `saved_refs`. Backward
    /// code that does `saved_tensors[1].1` should use this instead, so the
    /// SavedRef migration is positionally compatible. Returns `None` if out
    /// of range.
    #[inline]
    fn saved_at(&self, i: usize) -> Option<(TensorId, &Tensor)> {
        if i < self.saved_tensors.len() {
            let (id, t) = &self.saved_tensors[i];
            Some((*id, t))
        } else {
            let j = i - self.saved_tensors.len();
            if j < self.saved_refs.len() {
                let r = &self.saved_refs[j];
                // Skip version check here — positional access is best-effort
                // for backward consumers that expected a `Tensor` value.
                Some((r.id, &r.tensor))
            } else {
                None
            }
        }
    }
}

impl std::fmt::Debug for TapeEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapeEntry")
            .field("output_id", &self.output_id)
            .field("op", &self.op)
            .field("saved_count", &self.saved_count())
            .finish()
    }
}

/// Internal autograd context
struct AutogradContextInner {
    /// Computation tape
    tape: Vec<TapeEntry>,

    /// Whether we're currently recording operations
    enabled: bool,

    /// Recompute closures for activation checkpointing.
    /// Keyed by the checkpoint output tensor ID.
    checkpoint_fns: HashMap<TensorId, Arc<dyn Fn() -> Result<Tensor> + Send + Sync>>,
}

impl AutogradContextInner {
    fn new() -> Self {
        Self {
            tape: Vec::new(),
            enabled: true,
            checkpoint_fns: HashMap::new(),
        }
    }

    fn record(&mut self, entry: TapeEntry) {
        if self.enabled {
            self.tape.push(entry);
        }
    }

    fn clear(&mut self) {
        self.tape.clear();
        self.checkpoint_fns.clear();
    }
}

/// Determine if a multi-axis slice can be handled by chained narrow scatters on GPU.
/// Returns true when all ranges are contiguous narrows (no strides) and at least one axis is sliced.
fn can_gpu_multi_axis(ranges: &[(usize, usize)], input_dims: &[usize]) -> bool {
    let mut sliced = 0usize;
    for (i, &(s, e)) in ranges.iter().enumerate() {
        if !(s == 0 && e == input_dims[i]) {
            if s > e || e > input_dims[i] {
                return false;
            }
            sliced += 1;
        }
    }
    sliced > 1
}

/// Get raw device pointer for any dtype (cuda_ptr/cuda_ptr_mut only work for F32).
fn tensor_raw_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    use cudarc::driver::DevicePtr;
    match &t.storage {
        TensorStorage::F32 { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        TensorStorage::F16 { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16 { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16Arena { ptr, .. } => Ok(ptr.as_ptr() as *const core::ffi::c_void),
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16View { ptr, .. } => Ok(ptr.as_ptr() as *const core::ffi::c_void),
        TensorStorage::I32 { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        _ => Err(Error::InvalidOperation(
            "unsupported dtype for raw ptr".into(),
        )),
    }
}

fn tensor_raw_ptr_mut(t: &mut Tensor) -> Result<*mut core::ffi::c_void> {
    use cudarc::driver::DevicePtrMut;
    match &mut t.storage {
        TensorStorage::F32 { data, .. } => {
            let slice = crate::tensor_storage::ensure_unique_slice(data)?;
            Ok(*slice.device_ptr_mut() as *mut core::ffi::c_void)
        }
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16 { data, .. } => {
            let slice = crate::tensor_storage::ensure_unique_slice(data)?;
            Ok(*slice.device_ptr_mut() as *mut core::ffi::c_void)
        }
        _ => Err(Error::InvalidOperation(
            "unsupported dtype for raw ptr mut".into(),
        )),
    }
}

/// Function-pointer signature of the fused unary-activation backward kernels.
/// All four (relu/gelu/tanh/sigmoid) + silu share this ABI:
/// `(grad_out, input, grad_in, n, stream) -> i32`.
type UnaryBwdKernelFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    i64,
    *mut core::ffi::c_void,
) -> i32;

/// Launch a fused unary-activation backward kernel and return the new gradient tensor.
///
/// BF16→BF16 is the fast path. Anything else is cast to F32 and served by the
/// F32 kernel (output remains F32; parameter-grad accumulation handles the dtype
/// match downstream).
fn fused_unary_backward(
    op_name: &str,
    output_grad: &Tensor,
    input: &Tensor,
    device: &Arc<CudaDevice>,
    bf16_kernel: UnaryBwdKernelFn,
    f32_kernel: UnaryBwdKernelFn,
) -> Result<Tensor> {
    let n = input.shape().elem_count() as i64;
    if n == 0 {
        return Tensor::empty_dtype(input.shape().clone(), input.dtype(), device.clone());
    }
    let stream = device.cuda_stream_raw_ptr();

    if output_grad.dtype() == DType::BF16 && input.dtype() == DType::BF16 {
        let mut out = Tensor::empty_dtype(input.shape().clone(), DType::BF16, device.clone())?;
        let status = unsafe {
            bf16_kernel(
                tensor_raw_ptr(output_grad)?,
                tensor_raw_ptr(input)?,
                tensor_raw_ptr_mut(&mut out)?,
                n,
                stream,
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!("{op_name} bf16 kernel failed")));
        }
        return Ok(out);
    }

    if output_grad.dtype() == DType::F32 && input.dtype() == DType::F32 {
        let mut out = Tensor::empty_dtype(input.shape().clone(), DType::F32, device.clone())?;
        let status = unsafe {
            f32_kernel(
                tensor_raw_ptr(output_grad)?,
                tensor_raw_ptr(input)?,
                tensor_raw_ptr_mut(&mut out)?,
                n,
                stream,
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!("{op_name} f32 kernel failed")));
        }
        return Ok(out);
    }

    // Mixed dtypes: cast to F32, run F32 kernel.
    let og_f32 = if output_grad.dtype() != DType::F32 {
        output_grad.to_dtype_no_grad(DType::F32)?
    } else {
        output_grad.clone()
    };
    let x_f32 = if input.dtype() != DType::F32 {
        input.to_dtype_no_grad(DType::F32)?
    } else {
        input.clone()
    };
    let mut out = Tensor::empty_dtype(x_f32.shape().clone(), DType::F32, device.clone())?;
    let status = unsafe {
        f32_kernel(
            tensor_raw_ptr(&og_f32)?,
            tensor_raw_ptr(&x_f32)?,
            tensor_raw_ptr_mut(&mut out)?,
            n,
            stream,
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!(
            "{op_name} f32 kernel failed (mixed dtype)"
        )));
    }
    Ok(out)
}

// Local GPU narrow scatter-add (single-axis). No cross-crate deps.
fn gpu_scatter_add_narrow(
    grad_out: &Tensor,
    grad_in: &mut Tensor,
    dim: usize,
    start: usize,
) -> Result<()> {
    use core::ffi::c_void;

    let rank = grad_in.shape().dims().len();
    debug_assert_eq!(grad_out.shape().dims().len(), rank);
    debug_assert_eq!(grad_out.dtype(), grad_in.dtype(), "scatter dtype mismatch");

    // out_shape (row-major strides)
    let out_dims = grad_out.shape().dims().to_vec();
    let out_shape: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let mut out_strides: Vec<i64> = vec![0; rank];
    if rank > 0 {
        out_strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }
    }

    // in_strides: assume contiguous (row-major)
    let in_dims = grad_in.shape().dims();
    let mut in_strides: Vec<i64> = vec![0; rank];
    if rank > 0 {
        in_strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * (in_dims[i + 1] as i64);
        }
    }

    let n_elements: i64 = out_shape.iter().product();
    let elem_size: i64 = grad_in.dtype().size_in_bytes() as i64;
    let stream: *mut c_void = grad_in.device().cuda_stream_raw_ptr();

    // Use dtype-aware raw pointers (cuda_ptr/cuda_ptr_mut return null for BF16)
    let src_ptr = tensor_raw_ptr(grad_out)?;
    let dst_ptr = tensor_raw_ptr_mut(grad_in)?;

    let code = unsafe {
        ffi::narrow_backward_scatter_add_launch(
            src_ptr,
            dst_ptr,
            rank as i32,
            out_shape.as_ptr(),
            in_strides.as_ptr(),
            out_strides.as_ptr(),
            dim as i32,
            start as i64,
            elem_size,
            n_elements,
            stream,
        )
    };
    if code != 0 {
        return Err(crate::Error::KernelError(format!(
            "narrow_backward_scatter_add_launch failed: {}",
            code
        )));
    }
    Ok(())
}

/// Backward compatibility helper: free backward function under `flame_core::autograd::backward`.
/// Ignores `retain_graph` for now and returns the gradient map from the current autograd engine.
pub fn backward(loss: &Tensor, _retain_graph: bool) -> Result<crate::GradientMap> {
    if loss.rank() == 4 {
        assert_nhwc_bf16_public("autograd::backward in", loss)?;
    }
    AutogradContext::backward(loss)
}
/// cuDNN SDPA backward. Returns `Some((dQ, dK, dV))` when cuDNN handled the
/// shape, `None` when the caller should fall back to decomposed recompute.
///
/// Requirements for the cuDNN path:
///   - 4D BF16 Q/K/V/O/dO, head_dim ∈ {64, 96, 128}
///   - no arbitrary binary mask; causal is supported through cuDNN's flag
///   - Stats tensor saved from forward — contiguous FP32 `[B*H, N_q]`
///
/// Uses the tensors' native strides so permute-views work without
/// materialization. The output dQ/dK/dV are freshly-allocated contiguous.

/// Diagnostic: log every SDPA-backward dispatch decision when
/// `FLAME_LOG_SDPA_BWD=1`. Helps pin down whether cuDNN fires or each
/// block falls back to `attention_backward_recompute` (which is ~12 launches
/// vs 1 for cuDNN).
#[inline]
fn sdpa_bwd_log(msg: &str) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("FLAME_LOG_SDPA_BWD").ok().as_deref() == Some("1")) {
        return;
    }
    eprintln!("[sdpa-bwd] {msg}");
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttentionBwdRecomputeMode {
    Bf16,
    Bf16ScoreF32Softmax,
    F32,
}

const CUDNN_SDPA_BWD_SEQ_ALIGN: usize = 128;

#[inline]
fn attention_bwd_recompute_mode() -> AttentionBwdRecomputeMode {
    match std::env::var("FLAME_ATTENTION_BWD_RECOMPUTE").ok().as_deref() {
        Some("fp32") | Some("f32") => AttentionBwdRecomputeMode::F32,
        Some("bf16-score-fp32-softmax") | Some("f32-softmax") => {
            AttentionBwdRecomputeMode::Bf16ScoreF32Softmax
        }
        Some("bf16") | None => {
            if std::env::var("FLAME_ATTENTION_BWD_FP32_RECOMPUTE")
                .ok()
                .as_deref()
                == Some("1")
            {
                AttentionBwdRecomputeMode::F32
            } else {
                AttentionBwdRecomputeMode::Bf16
            }
        }
        Some(other) => {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[sdpa-bwd] unknown FLAME_ATTENTION_BWD_RECOMPUTE={other:?}; using bf16"
                );
            }
            AttentionBwdRecomputeMode::Bf16
        }
    }
}

fn softmax_lastdim_no_default_cast(logits: &Tensor) -> Result<Tensor> {
    let dim = logits.shape().rank().saturating_sub(1);
    let mut max_vals = GpuOps::max_dim(logits, dim, true)?;
    if max_vals.dtype() != logits.dtype() {
        max_vals = max_vals.to_dtype_no_grad(logits.dtype())?;
    }
    let shifted = logits.sub(&max_vals)?;
    let exp_vals = shifted.exp()?;
    let mut sum_exp = GpuOps::sum_dim_keepdim(&exp_vals, dim)?;
    if sum_exp.dtype() != exp_vals.dtype() {
        sum_exp = sum_exp.to_dtype_no_grad(exp_vals.dtype())?;
    }
    exp_vals.div(&sum_exp)
}

fn add_unsaved_view_inputs_to_needed(
    op: &Op,
    ids: &mut std::collections::HashSet<TensorId>,
) {
    match op {
        Op::Split { input, .. } | Op::Slice { input, .. } => {
            ids.insert(*input);
        }
        _ => {}
    }
}

fn causal_keep_mask_for_logits(
    q_len: usize,
    k_len: usize,
    device: Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; q_len * k_len];
    for q_idx in 0..q_len {
        let cutoff = q_idx.min(k_len.saturating_sub(1));
        let row = q_idx * k_len;
        for k_idx in 0..=cutoff {
            data[row + k_idx] = 1.0;
        }
    }
    Tensor::from_vec_dtype(data, Shape::from_dims(&[1, 1, q_len, k_len]), device, DType::F32)
}

fn prefix_causal_full_keep_mask_for_logits(
    seq_len: usize,
    prefix_len: usize,
    device: Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let mut data = vec![1.0f32; seq_len * seq_len];
    for q_idx in 0..seq_len {
        if q_idx < prefix_len {
            let row = q_idx * seq_len;
            for k_idx in 0..seq_len {
                data[row + k_idx] = if k_idx <= q_idx && k_idx < prefix_len {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[1, 1, seq_len, seq_len]),
        device,
        DType::F32,
    )
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn try_cudnn_sdpa_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o: Option<&Tensor>,
    d_o: &Tensor,
    stats: Option<&Tensor>,
    mask: Option<&Tensor>,
    causal: bool,
    padding_lens: Option<(usize, usize)>,
    scale: f32,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    // FLAME_NO_CUDNN_SDPA_BWD=1 forces the decomposed-recompute backward.
    // Diagnostic escape hatch: kept available for parity-harness comparisons
    // and emergency rollback if a future cuDNN regression appears. Read on
    // every call (not cached via OnceLock) so test harnesses can toggle the
    // env mid-process. SDPA-bwd is called ~30/step in zimage and ~32/step
    // in klein — a per-call `std::env::var` read (~50 ns) is negligible
    // against the cuDNN kernel cost (~10-15 ms).
    if std::env::var("FLAME_NO_CUDNN_SDPA_BWD").ok().as_deref() == Some("1") {
        sdpa_bwd_log("disabled-by-env");
        return Ok(None);
    }
    if causal
        && std::env::var("FLAME_CUDNN_SDPA_BWD_CAUSAL")
            .ok()
            .as_deref()
            != Some("1")
    {
        sdpa_bwd_log("bail:causal-disabled");
        return Ok(None);
    }
    // All the reasons to bail out.
    if mask.is_some() {
        sdpa_bwd_log("bail:mask-present");
        return Ok(None);
    }
    let (Some(o), Some(stats)) = (o, stats) else {
        sdpa_bwd_log("bail:missing-o-or-stats");
        return Ok(None);
    };
    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        sdpa_bwd_log(&format!("bail:rank q={} k={}", q_dims.len(), k_dims.len()));
        return Ok(None);
    }
    let (b, h, n_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let n_kv = k_dims[2];
    if q_dims[1] != k_dims[1] || k_dims[1] != v_dims[1] {
        sdpa_bwd_log(&format!(
            "bail:gqa-not-cudnn Hq={} Hk={} Hv={}",
            q_dims[1], k_dims[1], v_dims[1]
        ));
        return Ok(None);
    }
    if !(d == 64 || d == 96 || d == 128) {
        sdpa_bwd_log(&format!("bail:head_dim={d}"));
        return Ok(None);
    }
    // cuDNN flash bwd requires Nq and Nkv to be 128-token aligned on the
    // train-bwd path. 64-token alignment was not sufficient: L2P at
    // Nq=Nkv=1216 (64*19, but not 128-aligned) reports success from the
    // frontend and then leaves CUDA in CUDA_ERROR_MISALIGNED_ADDRESS during
    // cleanup. Fall back to decomposed for non-aligned saved shapes.
    if n_q % CUDNN_SDPA_BWD_SEQ_ALIGN != 0 || n_kv % CUDNN_SDPA_BWD_SEQ_ALIGN != 0 {
        sdpa_bwd_log(&format!("bail:nq-nkv-align Nq={n_q} Nkv={n_kv}"));
        return Ok(None);
    }
    let (real_n_q, real_n_kv) = padding_lens.unwrap_or((n_q, n_kv));
    if real_n_q == 0 || real_n_q > n_q || real_n_kv == 0 || real_n_kv > n_kv {
        sdpa_bwd_log(&format!(
            "bail:padding-lens real_Nq={real_n_q} real_Nkv={real_n_kv} Nq={n_q} Nkv={n_kv}"
        ));
        return Ok(None);
    }
    // Q/K/V/O must be BF16 (saved from the BF16 forward). If they aren't,
    // we're in a non-flash codepath and must bail to the decomposed fallback.
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || o.dtype() != DType::BF16
    {
        sdpa_bwd_log(&format!(
            "bail:dtype q={:?} k={:?} v={:?} o={:?}",
            q.dtype(),
            k.dtype(),
            v.dtype(),
            o.dtype()
        ));
        return Ok(None);
    }
    // Stage 2 fix (2026-05-12): `d_o` is provided by GradientMap which stores
    // grads as F32 (gradient.rs `InternalFP32_PublicBF16`). The earlier
    // `grad_norm=inf` regression was caused by the saved-O shape-find bug at
    // `autograd.rs:4231-4240` picking up Q as O (fresh `TensorId`s after
    // `fetch_saved`'s `.contiguous()` defeated the id-exclusion). With that
    // bug fixed (direct id lookup from `Op::FlashAttention { output, .. }`),
    // casting `d_o` F32→BF16 here is mathematically safe — BF16 has the same
    // 8-bit exponent as F32 so gradient direction and magnitude class are
    // preserved. Precision loss is ~1e-3 relative which is well inside the
    // BF16 training tolerance band already accepted on every forward op.
    let d_o_bf16_owned: Tensor;
    let d_o = if d_o.dtype() != DType::BF16 {
        d_o_bf16_owned = d_o.to_dtype_no_grad(DType::BF16)?;
        &d_o_bf16_owned
    } else {
        d_o
    };
    if stats.shape().dims() != [b * h, n_q] {
        sdpa_bwd_log(&format!(
            "bail:stats-shape got={:?} want=[{},{n_q}]",
            stats.shape().dims(),
            b * h
        ));
        return Ok(None);
    }

    use cudarc::driver::DevicePtr;
    let q_base = q.as_device_ptr_bf16("cudnn_sdpa_bwd:q")? as usize;
    let k_base = k.as_device_ptr_bf16("cudnn_sdpa_bwd:k")? as usize;
    let v_base = v.as_device_ptr_bf16("cudnn_sdpa_bwd:v")? as usize;
    let o_base = o.as_device_ptr_bf16("cudnn_sdpa_bwd:o")? as usize;
    let do_base = d_o.as_device_ptr_bf16("cudnn_sdpa_bwd:do")? as usize;
    let stats_base = match &stats.storage {
        crate::tensor_storage::TensorStorage::F32 { data, .. } => {
            *crate::tensor_storage::slice_ref(data).device_ptr() as usize
        }
        _ => return Ok(None),
    };
    let aligned = [
        ("q", q_base + q.offset() * 2),
        ("k", k_base + k.offset() * 2),
        ("v", v_base + v.offset() * 2),
        ("o", o_base + o.offset() * 2),
        ("do", do_base + d_o.offset() * 2),
        ("stats", stats_base + stats.offset() * 4),
    ];
    if let Some((name, addr)) = aligned.iter().find(|(_, addr)| *addr % 16 != 0) {
        sdpa_bwd_log(&format!("bail:ptr-align {name}={addr:#x}"));
        return Ok(None);
    }
    // Offset/contiguity guard (BACKLOG_qwen_cudnn_sdpa_bwd_misalign.md §6.1,
    // L2P real-cause handoff §6.1 — the PRIME suspect, 2026-06-11): the
    // crash class survived the 64->128 seq-align bump at 64-aligned-not-128
    // shapes, and the suspected mechanism is OFFSET VIEWS / non-dense strides
    // (l2p feeds chunk(3,2) views of fused qkv; zimage feeds contiguous
    // tensors and never crashed). cuDNN flash bwd's alignment assumptions on
    // interior rows are not satisfiable for arbitrary strides even when the
    // base pointer is 16B-aligned. ADDITIVE guard: the cuDNN path now demands
    // zero storage offset AND dense row-major (contiguous) layout on every
    // operand; anything else routes to the always-correct decomposed
    // backward. Contiguous-input models (zimage/klein) are untouched by
    // construction; over-bail is observable via these logs. GPU verification
    // owed: l2p/qwen repro at seq 1216/1280 + zimage cuDNN-bwd re-parity.
    let offset_views = [
        ("q", q.offset(), q.is_contiguous()),
        ("k", k.offset(), k.is_contiguous()),
        ("v", v.offset(), v.is_contiguous()),
        ("o", o.offset(), o.is_contiguous()),
        ("do", d_o.offset(), d_o.is_contiguous()),
    ];
    if let Some((name, off, contig)) = offset_views
        .iter()
        .find(|(_, off, contig)| *off != 0 || !*contig)
    {
        sdpa_bwd_log(&format!(
            "bail:offset-or-noncontig {name} offset={off} contiguous={contig}"
        ));
        return Ok(None);
    }

    sdpa_bwd_log(&format!(
        "fired cuDNN bf16 [B={b},H={h},Nq={n_q},Nkv={n_kv},d={d},causal={causal}]"
    ));

    // Read each tensor's native 4D strides.
    let q_strides = strides_4d(q)?;
    let k_strides = strides_4d(k)?;
    let v_strides = strides_4d(v)?;
    let o_strides = strides_4d(o)?;
    let do_strides = strides_4d(d_o)?;

    // Allocate contiguous dQ/dK/dV. Strides computed from shape
    // (row-major [B, H, N, D] layout).
    let dq = Tensor::empty_dtype(q.shape().clone(), DType::BF16, device.clone())?;
    let dk = Tensor::empty_dtype(k.shape().clone(), DType::BF16, device.clone())?;
    let dv = Tensor::empty_dtype(v.shape().clone(), DType::BF16, device.clone())?;
    let dq_strides = strides_4d(&dq)?;
    let dk_strides = strides_4d(&dk)?;
    let dv_strides = strides_4d(&dv)?;

    let stream = crate::cuda::device_lt::stream_ptr(device)?;

    let q_ptr = q_base as *const core::ffi::c_void;
    let k_ptr = k_base as *const core::ffi::c_void;
    let v_ptr = v_base as *const core::ffi::c_void;
    let o_ptr = o_base as *const core::ffi::c_void;
    let do_ptr = do_base as *const core::ffi::c_void;
    let dq_ptr = dq.as_device_ptr_bf16("cudnn_sdpa_bwd:dq")? as *mut core::ffi::c_void;
    let dk_ptr = dk.as_device_ptr_bf16("cudnn_sdpa_bwd:dk")? as *mut core::ffi::c_void;
    let dv_ptr = dv.as_device_ptr_bf16("cudnn_sdpa_bwd:dv")? as *mut core::ffi::c_void;

    let stats_ptr = stats_base as *const core::ffi::c_void;

    let q_off = q.offset() as i64;
    let k_off = k.offset() as i64;
    let v_off = v.offset() as i64;
    let o_off = o.offset() as i64;
    let do_off = d_o.offset() as i64;
    let dq_off = dq.offset() as i64;
    let dk_off = dk.offset() as i64;
    let dv_off = dv.offset() as i64;
    let stats_off = stats.offset() as i64;

    let ret = unsafe {
        crate::cuda::ffi::flame_cudnn_sdpa_bwd_bf16(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            do_ptr,
            stats_ptr,
            dq_ptr,
            dk_ptr,
            dv_ptr,
            b as i32,
            h as i32,
            n_q as i32,
            n_kv as i32,
            d as i32,
            scale,
            q_strides.as_ptr(),
            k_strides.as_ptr(),
            v_strides.as_ptr(),
            o_strides.as_ptr(),
            do_strides.as_ptr(),
            dq_strides.as_ptr(),
            dk_strides.as_ptr(),
            dv_strides.as_ptr(),
            q_off,
            k_off,
            v_off,
            o_off,
            do_off,
            stats_off,
            dq_off,
            dk_off,
            dv_off,
            if causal { 1 } else { 0 },
            real_n_q as i32,
            real_n_kv as i32,
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!("cudnn_sdpa_bwd CUDA error: {ret}")));
    }
    Ok(Some((dq, dk, dv)))
}

#[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
fn try_cudnn_sdpa_backward(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _o: Option<&Tensor>,
    _d_o: &Tensor,
    _stats: Option<&Tensor>,
    _mask: Option<&Tensor>,
    _causal: bool,
    _padding_lens: Option<(usize, usize)>,
    _scale: f32,
    _device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    Ok(None)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn strides_4d(t: &Tensor) -> Result<[i64; 4]> {
    let mut s = [0usize; 8];
    let rank = t.fill_strides_into(&mut s);
    if rank != 4 {
        return Err(Error::InvalidInput(format!(
            "cudnn_sdpa_bwd: expected 4D tensor, got rank {}",
            rank
        )));
    }
    Ok([s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64])
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn try_pytorch_flash_hd128_forward_with_lse(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        sdpa_bwd_log(&format!(
            "pt-flash-hd128-fwd bail:rank q={} k={} v={}",
            q_dims.len(),
            k_dims.len(),
            v_dims.len()
        ));
        return Ok(None);
    }
    let (b, hq, n_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (hkv, n_kv) = (k_dims[1], k_dims[2]);
    if d != 128 || k_dims[3] != 128 || v_dims[3] != 128 {
        sdpa_bwd_log(&format!("pt-flash-hd128-fwd bail:head_dim={d}"));
        return Ok(None);
    }
    if q_dims[0] != k_dims[0]
        || q_dims[0] != v_dims[0]
        || k_dims[1] != v_dims[1]
        || k_dims[2] != v_dims[2]
    {
        sdpa_bwd_log(&format!(
            "pt-flash-hd128-fwd bail:shape q={:?} k={:?} v={:?}",
            q_dims, k_dims, v_dims
        ));
        return Ok(None);
    }
    if hkv == 0 || hq % hkv != 0 {
        sdpa_bwd_log(&format!("pt-flash-hd128-fwd bail:gqa Hq={hq} Hkv={hkv}"));
        return Ok(None);
    }
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        sdpa_bwd_log(&format!(
            "pt-flash-hd128-fwd bail:dtype q={:?} k={:?} v={:?}",
            q.dtype(),
            k.dtype(),
            v.dtype()
        ));
        return Ok(None);
    }
    if scale <= 0.0 {
        sdpa_bwd_log(&format!("pt-flash-hd128-fwd bail:scale={scale}"));
        return Ok(None);
    }

    let q_strides = strides_4d(q)?;
    let k_strides = strides_4d(k)?;
    let v_strides = strides_4d(v)?;
    if q_strides[3] != 1 || k_strides[3] != 1 || v_strides[3] != 1 {
        sdpa_bwd_log(&format!(
            "pt-flash-hd128-fwd bail:last-stride q={:?} k={:?} v={:?}",
            q_strides, k_strides, v_strides
        ));
        return Ok(None);
    }

    let device = q.device();
    let output = Tensor::empty_dtype(q.shape().clone(), DType::BF16, device.clone())?;
    let mut lse = Tensor::empty_dtype(Shape::from_dims(&[b, hq, n_q]), DType::F32, device.clone())?;
    let o_strides = strides_4d(&output)?;
    let stream = crate::cuda::device_lt::stream_ptr(device)?;

    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let q_ptr = q.as_device_ptr_bf16("pt_flash_hd128_fwd:q")? as *const core::ffi::c_void;
    let k_ptr = k.as_device_ptr_bf16("pt_flash_hd128_fwd:k")? as *const core::ffi::c_void;
    let v_ptr = v.as_device_ptr_bf16("pt_flash_hd128_fwd:v")? as *const core::ffi::c_void;
    let o_ptr = output.as_device_ptr_bf16("pt_flash_hd128_fwd:o")? as *mut core::ffi::c_void;
    let lse_ptr = {
        let lse_slice = lse.as_mut_slice_f32("pt_flash_hd128_fwd:lse")?;
        *lse_slice.device_ptr_mut() as *mut f32
    };

    let ret = unsafe {
        crate::cuda::ffi::flame_pytorch_flash_attn_bf16_hd128(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            lse_ptr,
            b as i32,
            hq as i32,
            hkv as i32,
            n_q as i32,
            n_kv as i32,
            q_strides.as_ptr(),
            k_strides.as_ptr(),
            v_strides.as_ptr(),
            o_strides.as_ptr(),
            q.offset() as i64,
            k.offset() as i64,
            v.offset() as i64,
            output.offset() as i64,
            scale,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!(
            "pytorch_flash_hd128 forward CUDA error: {ret}"
        )));
    }
    Ok(Some((output, lse)))
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn try_pytorch_flash_hd128_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dout: &Tensor,
    causal: bool,
    scale: f32,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    if std::env::var("FLAME_NO_PYTORCH_FLASH_HD128_BWD")
        .ok()
        .as_deref()
        == Some("1")
    {
        sdpa_bwd_log("pt-flash-hd128-bwd disabled-by-env");
        return Ok(None);
    }

    let Some((o, lse)) = try_pytorch_flash_hd128_forward_with_lse(q, k, v, causal, scale)? else {
        return Ok(None);
    };
    let d_o_bf16_owned;
    let d_o = if dout.dtype() != DType::BF16 {
        d_o_bf16_owned = dout.to_dtype_no_grad(DType::BF16)?;
        &d_o_bf16_owned
    } else {
        dout
    };
    let d_o_contig_owned;
    let d_o = if strides_4d(d_o)?[3] != 1 {
        d_o_contig_owned = d_o.contiguous()?;
        &d_o_contig_owned
    } else {
        d_o
    };

    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let (b, hq, n_q) = (q_dims[0], q_dims[1], q_dims[2]);
    let (hkv, n_kv) = (k_dims[1], k_dims[2]);
    let do_strides = strides_4d(d_o)?;
    let q_strides = strides_4d(q)?;
    let k_strides = strides_4d(k)?;
    let v_strides = strides_4d(v)?;
    let o_strides = strides_4d(&o)?;

    let device = q.device();
    let dq = Tensor::empty_dtype(q.shape().clone(), DType::BF16, device.clone())?;
    let dk = Tensor::empty_dtype(k.shape().clone(), DType::BF16, device.clone())?;
    let dv = Tensor::empty_dtype(v.shape().clone(), DType::BF16, device.clone())?;
    let dq_strides = strides_4d(&dq)?;
    let dk_strides = strides_4d(&dk)?;
    let dv_strides = strides_4d(&dv)?;

    use cudarc::driver::DevicePtr;
    let stream = crate::cuda::device_lt::stream_ptr(device)?;
    let do_ptr = d_o.as_device_ptr_bf16("pt_flash_hd128_bwd:do")? as *const core::ffi::c_void;
    let q_ptr = q.as_device_ptr_bf16("pt_flash_hd128_bwd:q")? as *const core::ffi::c_void;
    let k_ptr = k.as_device_ptr_bf16("pt_flash_hd128_bwd:k")? as *const core::ffi::c_void;
    let v_ptr = v.as_device_ptr_bf16("pt_flash_hd128_bwd:v")? as *const core::ffi::c_void;
    let o_ptr = o.as_device_ptr_bf16("pt_flash_hd128_bwd:o")? as *const core::ffi::c_void;
    let dq_ptr = dq.as_device_ptr_bf16("pt_flash_hd128_bwd:dq")? as *mut core::ffi::c_void;
    let dk_ptr = dk.as_device_ptr_bf16("pt_flash_hd128_bwd:dk")? as *mut core::ffi::c_void;
    let dv_ptr = dv.as_device_ptr_bf16("pt_flash_hd128_bwd:dv")? as *mut core::ffi::c_void;
    let lse_ptr = *lse.as_slice_f32("pt_flash_hd128_bwd:lse")?.device_ptr() as *const f32;

    let ret = unsafe {
        crate::cuda::ffi::flame_pytorch_flash_attn_bf16_hd128_bwd(
            do_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            lse_ptr,
            dq_ptr,
            dk_ptr,
            dv_ptr,
            b as i32,
            hq as i32,
            hkv as i32,
            n_q as i32,
            n_kv as i32,
            do_strides.as_ptr(),
            q_strides.as_ptr(),
            k_strides.as_ptr(),
            v_strides.as_ptr(),
            o_strides.as_ptr(),
            dq_strides.as_ptr(),
            dk_strides.as_ptr(),
            dv_strides.as_ptr(),
            d_o.offset() as i64,
            q.offset() as i64,
            k.offset() as i64,
            v.offset() as i64,
            o.offset() as i64,
            dq.offset() as i64,
            dk.offset() as i64,
            dv.offset() as i64,
            scale,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if ret != 0 {
        return Err(Error::Cuda(format!(
            "pytorch_flash_hd128 backward CUDA error: {ret}"
        )));
    }

    sdpa_bwd_log(&format!(
        "fired PyTorch flash hd128 [B={b},H={hq},Nq={n_q},Nkv={n_kv},causal={causal}]"
    ));
    Ok(Some((dq, dk, dv)))
}

#[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
fn try_pytorch_flash_hd128_backward(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _dout: &Tensor,
    _causal: bool,
    _scale: f32,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    Ok(None)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn try_prefix_causal_full_pytorch_flash_hd128_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dout: &Tensor,
    prefix_len: usize,
    scale: f32,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    if std::env::var("FLAME_PREFIX_CAUSAL_FULL_NO_PT_FLASH_BWD")
        .ok()
        .as_deref()
        == Some("1")
    {
        sdpa_bwd_log("prefix-pt-flash-hd128 disabled-by-env");
        return Ok(None);
    }

    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    if q_dims.len() != 4
        || k_dims.len() != 4
        || v_dims.len() != 4
        || q_dims[3] != 128
        || k_dims[3] != 128
        || v_dims[3] != 128
        || q.dtype() != DType::BF16
    {
        return Ok(None);
    }
    let seq_len = q_dims[2];
    if prefix_len == 0 {
        return try_pytorch_flash_hd128_backward(q, k, v, dout, false, scale);
    }
    if prefix_len == seq_len {
        return try_pytorch_flash_hd128_backward(q, k, v, dout, true, scale);
    }
    if prefix_len > seq_len {
        return Ok(None);
    }

    let dout_bf16_owned;
    let dout_bf16 = if dout.dtype() != DType::BF16 {
        dout_bf16_owned = dout.to_dtype_no_grad(DType::BF16)?;
        &dout_bf16_owned
    } else {
        dout
    };

    let suffix_len = seq_len - prefix_len;
    let q_prefix = q.narrow(2, 0, prefix_len)?.contiguous()?;
    let k_prefix = k.narrow(2, 0, prefix_len)?.contiguous()?;
    let v_prefix = v.narrow(2, 0, prefix_len)?.contiguous()?;
    let dout_prefix = dout_bf16.narrow(2, 0, prefix_len)?.contiguous()?;
    let Some((dq_prefix, dk_prefix, dv_prefix)) =
        try_pytorch_flash_hd128_backward(&q_prefix, &k_prefix, &v_prefix, &dout_prefix, true, scale)?
    else {
        return Ok(None);
    };

    let zero_prefix_dout = Tensor::zeros_dtype(
        Shape::from_dims(&[q_dims[0], q_dims[1], prefix_len, q_dims[3]]),
        DType::BF16,
        q.device().clone(),
    )?;
    let dout_suffix = dout_bf16.narrow(2, prefix_len, suffix_len)?.contiguous()?;
    let dout_full = Tensor::cat(&[&zero_prefix_dout, &dout_suffix], 2)?;
    let Some((dq_full, dk_full, dv_full)) =
        try_pytorch_flash_hd128_backward(q, k, v, &dout_full, false, scale)?
    else {
        return Ok(None);
    };

    let dq_suffix = dq_full.narrow(2, prefix_len, suffix_len)?.contiguous()?;
    let grad_q = Tensor::cat(&[&dq_prefix, &dq_suffix], 2)?;

    let zero_k_tail = Tensor::zeros_dtype(
        Shape::from_dims(&[k_dims[0], k_dims[1], suffix_len, k_dims[3]]),
        dk_prefix.dtype(),
        q.device().clone(),
    )?;
    let zero_v_tail = Tensor::zeros_dtype(
        Shape::from_dims(&[v_dims[0], v_dims[1], suffix_len, v_dims[3]]),
        dv_prefix.dtype(),
        q.device().clone(),
    )?;
    let dk_prefix_padded = Tensor::cat(&[&dk_prefix, &zero_k_tail], 2)?;
    let dv_prefix_padded = Tensor::cat(&[&dv_prefix, &zero_v_tail], 2)?;
    let grad_k = dk_prefix_padded.add(&dk_full)?;
    let grad_v = dv_prefix_padded.add(&dv_full)?;

    Ok(Some((grad_q, grad_k, grad_v)))
}

// Local, dependency-free SDPA backward (recompute path)
// SDPA backward (recompute path)
/// SDPA backward via recompute. If `cached_attn` is Some, uses it directly
/// as the softmax output — skipping the Q@K^T + softmax recompute entirely.
fn attention_backward_recompute(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dout: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: f32,
    cached_attn: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let recompute_mode = attention_bwd_recompute_mode();
    if recompute_mode != AttentionBwdRecomputeMode::Bf16 {
        return attention_backward_recompute_precise(
            q,
            k,
            v,
            dout,
            mask,
            causal,
            scale,
            cached_attn,
            recompute_mode,
        );
    }

    // All ops in BF16 (bmm requires matching dtypes)
    let q = if q.dtype() != DType::BF16 {
        &q.to_dtype_no_grad(DType::BF16)?
    } else {
        q
    };
    let k = if k.dtype() != DType::BF16 {
        &k.to_dtype_no_grad(DType::BF16)?
    } else {
        k
    };
    let v = if v.dtype() != DType::BF16 {
        &v.to_dtype_no_grad(DType::BF16)?
    } else {
        v
    };
    let dout = if dout.dtype() != DType::BF16 {
        &dout.to_dtype_no_grad(DType::BF16)?
    } else {
        dout
    };

    let attn_owned;
    let attn: &Tensor = if let Some(cached) = cached_attn {
        let cached = if cached.dtype() != DType::BF16 {
            attn_owned = cached.to_dtype_no_grad(DType::BF16)?;
            &attn_owned
        } else {
            cached
        };
        cached
    } else {
        let kt = k.transpose_dims(2, 3)?;
        let mut logits = q.bmm(&kt)?;
        logits = logits.mul_scalar(scale)?;
        if let Some(mask_raw) = mask {
            let target_dims = logits.shape().dims().to_vec();
            let mask_f32 = if mask_raw.dtype() == DType::F32 {
                mask_raw.clone()
            } else {
                mask_raw.to_dtype_no_grad(DType::F32)?
            };
            let mask_bcast = if mask_f32.shape().dims() == target_dims.as_slice() {
                mask_f32
            } else {
                mask_f32.broadcast_to(&Shape::from_dims(&target_dims))?
            };
            let ones = mask_bcast.full_like(1.0)?;
            let complement = ones.sub(&mask_bcast)?;
            let penalty = complement.mul_scalar(-1.0e9)?;
            logits = logits.add(&penalty)?;
        }
        if causal {
            let target_dims = logits.shape().dims().to_vec();
            let q_len = q.shape().dims()[2];
            let k_len = k.shape().dims()[2];
            let causal_mask = causal_keep_mask_for_logits(q_len, k_len, q.device().clone())?;
            let causal_bcast = if causal_mask.shape().dims() == target_dims.as_slice() {
                causal_mask
            } else {
                causal_mask.broadcast_to(&Shape::from_dims(&target_dims))?
            };
            let ones = causal_bcast.full_like(1.0)?;
            let complement = ones.sub(&causal_bcast)?;
            let penalty = complement.mul_scalar(-1.0e9)?;
            logits = logits.add(&penalty)?;
        }
        attn_owned = logits.softmax(-1)?;
        &attn_owned
    };

    let d_v = attn.transpose_dims(2, 3)?.bmm(dout)?;
    let d_attn = dout.bmm(&v.transpose_dims(2, 3)?)?;

    let dattn_times_attn = d_attn.mul(attn)?;
    let sum_term = dattn_times_attn.sum_dim_keepdim(3)?;
    let d_logits = d_attn.sub(&sum_term)?.mul(attn)?;
    let d_logits = d_logits.mul_scalar(scale)?;

    let d_q = d_logits.bmm(k)?;
    let d_k = d_logits.transpose_dims(2, 3)?.bmm(q)?;

    Ok((d_q, d_k, d_v))
}

fn attention_backward_recompute_precise(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dout: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: f32,
    cached_attn: Option<&Tensor>,
    mode: AttentionBwdRecomputeMode,
) -> Result<(Tensor, Tensor, Tensor)> {
    sdpa_bwd_log(&format!("recompute-mode:{mode:?}"));

    let score_dtype = match mode {
        AttentionBwdRecomputeMode::Bf16ScoreF32Softmax => DType::BF16,
        AttentionBwdRecomputeMode::F32 => DType::F32,
        AttentionBwdRecomputeMode::Bf16 => DType::BF16,
    };

    let q_score_owned;
    let q_score = if q.dtype() != score_dtype {
        q_score_owned = q.to_dtype_no_grad(score_dtype)?;
        &q_score_owned
    } else {
        q
    };
    let k_score_owned;
    let k_score = if k.dtype() != score_dtype {
        k_score_owned = k.to_dtype_no_grad(score_dtype)?;
        &k_score_owned
    } else {
        k
    };

    let q_f32_owned;
    let q_f32 = if q.dtype() != DType::F32 {
        q_f32_owned = q.to_dtype_no_grad(DType::F32)?;
        &q_f32_owned
    } else {
        q
    };
    let k_f32_owned;
    let k_f32 = if k.dtype() != DType::F32 {
        k_f32_owned = k.to_dtype_no_grad(DType::F32)?;
        &k_f32_owned
    } else {
        k
    };
    let v_f32_owned;
    let v_f32 = if v.dtype() != DType::F32 {
        v_f32_owned = v.to_dtype_no_grad(DType::F32)?;
        &v_f32_owned
    } else {
        v
    };
    let dout_f32_owned;
    let dout_f32 = if dout.dtype() != DType::F32 {
        dout_f32_owned = dout.to_dtype_no_grad(DType::F32)?;
        &dout_f32_owned
    } else {
        dout
    };

    let attn_owned;
    let attn: &Tensor = if let Some(cached) = cached_attn {
        let cached = if cached.dtype() != DType::F32 {
            attn_owned = cached.to_dtype_no_grad(DType::F32)?;
            &attn_owned
        } else {
            cached
        };
        cached
    } else {
        let kt = k_score.transpose_dims(2, 3)?;
        let mut logits = q_score.bmm(&kt)?;
        logits = logits.mul_scalar(scale)?;
        if logits.dtype() != DType::F32 {
            logits = logits.to_dtype_no_grad(DType::F32)?;
        }
        if let Some(mask_raw) = mask {
            let target_dims = logits.shape().dims().to_vec();
            let mask_f32 = if mask_raw.dtype() == DType::F32 {
                mask_raw.clone()
            } else {
                mask_raw.to_dtype_no_grad(DType::F32)?
            };
            let mask_bcast = if mask_f32.shape().dims() == target_dims.as_slice() {
                mask_f32
            } else {
                mask_f32.broadcast_to(&Shape::from_dims(&target_dims))?
            };
            let ones = mask_bcast.full_like(1.0)?;
            let complement = ones.sub(&mask_bcast)?;
            let penalty = complement.mul_scalar(-1.0e9)?;
            logits = logits.add(&penalty)?;
        }
        if causal {
            let target_dims = logits.shape().dims().to_vec();
            let q_len = q_score.shape().dims()[2];
            let k_len = k_score.shape().dims()[2];
            let causal_mask = causal_keep_mask_for_logits(q_len, k_len, q_score.device().clone())?;
            let causal_bcast = if causal_mask.shape().dims() == target_dims.as_slice() {
                causal_mask
            } else {
                causal_mask.broadcast_to(&Shape::from_dims(&target_dims))?
            };
            let ones = causal_bcast.full_like(1.0)?;
            let complement = ones.sub(&causal_bcast)?;
            let penalty = complement.mul_scalar(-1.0e9)?;
            logits = logits.add(&penalty)?;
        }
        attn_owned = softmax_lastdim_no_default_cast(&logits)?;
        &attn_owned
    };

    let d_v = attn.transpose_dims(2, 3)?.bmm(dout_f32)?;
    let d_attn = dout_f32.bmm(&v_f32.transpose_dims(2, 3)?)?;

    let dattn_times_attn = d_attn.mul(attn)?;
    let sum_term = dattn_times_attn.sum_dim_keepdim(3)?;
    let d_logits = d_attn.sub(&sum_term)?.mul(attn)?;
    let d_logits = d_logits.mul_scalar(scale)?;

    let d_q = d_logits.bmm(k_f32)?;
    let d_k = d_logits.transpose_dims(2, 3)?.bmm(q_f32)?;

    if std::env::var("FLAME_ATTENTION_BWD_RETURN_INPUT_DTYPE")
        .ok()
        .as_deref()
        == Some("1")
    {
        let d_q = if d_q.dtype() != q.dtype() {
            d_q.to_dtype_no_grad(q.dtype())?
        } else {
            d_q
        };
        let d_k = if d_k.dtype() != k.dtype() {
            d_k.to_dtype_no_grad(k.dtype())?
        } else {
            d_k
        };
        let d_v = if d_v.dtype() != v.dtype() {
            d_v.to_dtype_no_grad(v.dtype())?
        } else {
            d_v
        };
        return Ok((d_q, d_k, d_v));
    }

    Ok((d_q, d_k, d_v))
}

/// Public API for autograd
pub struct AutogradContext;

impl AutogradContext {
    /// Fast lock-free check: is autograd currently recording?
    /// Tensor ops should call this BEFORE constructing Op enums or cloning
    /// saved_tensors to avoid wasted GPU memcpys when autograd is disabled.
    #[inline(always)]
    pub fn is_recording() -> bool {
        AUTOGRAD_ENABLED.load(Ordering::Relaxed)
    }

    /// True only while backward is replaying a checkpoint closure.
    ///
    /// This is intentionally narrower than `is_recording()`: recompute records
    /// a local sub-tape, but ordinary forward paths may also record and should
    /// not receive backward-order offload hints.
    #[inline(always)]
    pub fn is_checkpoint_recompute() -> bool {
        CHECKPOINT_RECOMPUTE_DEPTH.with(|depth| depth.get() != 0)
    }
}

impl AutogradContext {
    /// Record an operation in the computation graph.
    ///
    /// `saved_tensors` accepts anything convertible into `SavedTensors`
    /// (the inline-sized `SmallVec<[(TensorId, Tensor); 3]>`). This keeps
    /// the existing `vec![...]` call sites compiling unchanged while
    /// avoiding a heap allocation for the common 0-3 tensor case.
    pub fn record_op(output_id: TensorId, op: Op, saved_tensors: impl Into<SavedTensors>) {
        // Fast path: skip lock entirely when autograd is disabled (e.g., during backward).
        // This prevents deadlock when backward ops call high-level Tensor methods
        // that would otherwise try to re-acquire the already-held context lock.
        if !AUTOGRAD_ENABLED.load(Ordering::Relaxed) {
            return;
        }

        let saved_tensors: SavedTensors = saved_tensors.into();

        // Phase 2: optionally convert to SavedRef path. Default behavior
        // populates `saved_refs` (lighter — single Tensor::clone per saved
        // tensor plus an Arc<AtomicU32> snapshot, vs. legacy's Tensor::clone
        // bundled into a Vec-converted SmallVec). Legacy mode keeps the old
        // `saved_tensors` path for A/B testing via
        // `FLAME_AUTOGRAD_SAVED_LEGACY=1`.
        let (legacy_tensors, refs) = if crate::saved_ref::legacy_saved_mode() {
            (saved_tensors, SavedRefs::new())
        } else {
            let mut refs = SavedRefs::with_capacity(saved_tensors.len());
            for (id, t) in &saved_tensors {
                refs.push(crate::saved_ref::SavedRef {
                    id: *id,
                    tensor: t.clone(),
                    version_counter: t.storage_ref().version_handle(),
                    version_at_save: t.storage_ref().version(),
                });
            }
            (SavedTensors::new(), refs)
        };

        let mut ctx = match AUTOGRAD_CONTEXT.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };

        // Double-check under lock (race between atomic check and lock acquisition)
        if !ctx.enabled {
            return;
        }

        // Apply checkpointing policy to saved tensors (CPU offload registration).
        // Fast path: skip the CHECKPOINT_MANAGER mutex when no entries exist.
        // CHECKPOINT_HAS_ENTRIES is false by default and only set when CPUOffload
        // policy is active, so for pure Recompute checkpointing this is a ~1ns
        // atomic load instead of a ~25ns uncontended mutex lock per recorded op.
        if (!legacy_tensors.is_empty() || !refs.is_empty())
            && CHECKPOINT_HAS_ENTRIES.load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Ok(mut mgr) = CHECKPOINT_MANAGER.lock() {
                for (id, tensor) in &legacy_tensors {
                    let _ = mgr.checkpoint_saved_tensor(*id, tensor);
                }
                for r in &refs {
                    let _ = mgr.checkpoint_saved_tensor(r.id, &r.tensor);
                }
            }
        }

        ctx.record(TapeEntry {
            output_id,
            op,
            saved_tensors: legacy_tensors,
            saved_refs: refs,
        });
    }

    /// Clear the computation graph
    pub fn clear() {
        if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
            ctx.clear();
        }
        // Phase 2: flush the version-counter side table at step boundaries.
        // Any in-flight SavedRef keeps a private Arc<AtomicU32> clone so
        // its check stays valid; the flush only prevents unbounded growth
        // across long training runs.
        crate::tensor_storage::clear_version_table();
    }

    /// Test-only: register a set of intermediate tensor IDs whose gradients
    /// should be cloned out during backward (before the standard `take`-drain
    /// frees them). After backward, call `take_retained_intermediate_grads`
    /// to recover them. Used by parity diagnostics that need to inspect
    /// intermediate gradients without modifying autograd's drain semantics.
    pub fn retain_intermediate_grads(ids: std::collections::HashSet<TensorId>) {
        if let Ok(mut slot) = RETAINED_INTERMEDIATE_GRAD_IDS.lock() {
            *slot = Some(ids);
        }
        if let Ok(mut grads) = RETAINED_INTERMEDIATE_GRADS.lock() {
            grads.clear();
        }
    }

    /// Additive variant of [`retain_intermediate_grads`]: extend the existing
    /// retain set (creating it if not present) with `ids`. Used by diagnostic
    /// trap code that needs to register intermediate `TensorId`s mid-backward
    /// (specifically, during `Op::CheckpointOffloadBoundary` recompute — the
    /// outer-tape retain snapshot has already been taken at that point, but
    /// the sub-tape backward in `Op::CheckpointOffloadBoundary` re-reads the
    /// static set when draining sub_grads, so additions there are honored).
    /// Does NOT clear `RETAINED_INTERMEDIATE_GRADS`.
    pub fn retain_intermediate_grads_add(ids: std::collections::HashSet<TensorId>) {
        if let Ok(mut slot) = RETAINED_INTERMEDIATE_GRAD_IDS.lock() {
            slot.get_or_insert_with(std::collections::HashSet::new).extend(ids);
        }
    }

    /// Test-only: take the captured intermediate gradients and reset the
    /// retain-set. Returns a HashMap keyed by `TensorId`.
    pub fn take_retained_intermediate_grads() -> HashMap<TensorId, Tensor> {
        if let Ok(mut slot) = RETAINED_INTERMEDIATE_GRAD_IDS.lock() {
            *slot = None;
        }
        match RETAINED_INTERMEDIATE_GRADS.lock() {
            Ok(mut g) => std::mem::take(&mut *g),
            Err(_) => HashMap::new(),
        }
    }

    /// Number of entries currently on the autograd tape.
    pub fn tape_len() -> usize {
        AUTOGRAD_CONTEXT
            .lock()
            .map(|ctx| ctx.tape.len())
            .unwrap_or(0)
    }

    /// Reset the entire autograd context (for testing)
    pub fn reset() {
        if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
            *ctx = AutogradContextInner::new();
        }
    }

    /// Disable autograd (e.g., for inference)
    pub fn set_enabled(enabled: bool) {
        if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
            ctx.enabled = enabled;
            AUTOGRAD_ENABLED.store(enabled, Ordering::Relaxed);
        }
    }

    /// Context manager for no_grad mode
    pub fn no_grad() -> NoGradGuard {
        NoGradGuard::new()
    }

    /// Compute gradients via backpropagation with debug logging
    pub fn backward_debug(loss: &Tensor) -> Result<GradientMap> {
        println!("=== AUTOGRAD DEBUG START ===");
        println!("Loss tensor shape: {:?}", loss.shape);
        println!("Loss requires_grad: {}", loss.requires_grad);

        if !loss.requires_grad {
            return Err(Error::InvalidInput(
                "backward() called on tensor that doesn't require grad".into(),
            ));
        }

        if loss.shape.elem_count() != 1 {
            return Err(Error::InvalidInput(
                "backward() requires scalar loss tensor".into(),
            ));
        }

        let device = loss.device.clone();

        // Initialize gradient storage
        let mut gradients = GradientMap::new(device.clone());
        gradients.set_ones(loss.id, loss.shape.clone())?;
        println!("Root gradient initialized");

        // Process tape in reverse under lock
        {
            let mut ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd context mutex poisoned".into()))?;
            println!("Tape length: {}", ctx.tape.len());

            // Print all operations in tape
            for (i, entry) in ctx.tape.iter().enumerate() {
                println!(
                    "Op {}: {:?} -> tensor_id {:?}",
                    i, entry.op, entry.output_id
                );
            }

            // Disable autograd during backward pass
            let prev_enabled = ctx.enabled;
            ctx.enabled = false;
            AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);

            // Process tape in reverse with timing
            for (i, entry) in ctx.tape.iter().enumerate().rev() {
                let tape_idx = ctx.tape.len() - 1 - i;
                println!(
                    "\nProcessing op {} (reverse index {}): {:?}",
                    tape_idx, i, entry.op
                );
                let start = std::time::Instant::now();

                if let Some(output_grad) = gradients.get(entry.output_id) {
                    println!("  Output grad shape: {:?}", output_grad.shape());
                    let output_grad = output_grad.clone();

                    // Process gradients based on operation type
                    match compute_gradients(entry, &output_grad, &device) {
                        Ok(input_grads) => {
                            println!("  Computed {} input gradients", input_grads.len());

                            // Accumulate gradients
                            for (tensor_id, grad) in input_grads {
                                println!(
                                    "    Accumulating grad for tensor {:?}, shape: {:?}",
                                    tensor_id,
                                    grad.shape()
                                );
                                gradients.accumulate(tensor_id, grad)?;
                            }
                        }
                        Err(e) => {
                            println!("  ERROR computing gradients: {:?}", e);
                            ctx.enabled = prev_enabled;
                            AUTOGRAD_ENABLED.store(prev_enabled, Ordering::Relaxed);
                            return Err(e);
                        }
                    }
                } else {
                    println!("  No output gradient found, skipping");
                }

                let elapsed = start.elapsed();
                println!("  Op {} completed in {:?}", tape_idx, elapsed);

                if elapsed > std::time::Duration::from_secs(2) {
                    println!("  !!! SLOW OPERATION DETECTED !!!");
                    ctx.enabled = prev_enabled;
                    AUTOGRAD_ENABLED.store(prev_enabled, Ordering::Relaxed);
                    return Err(Error::InvalidOperation(format!(
                        "Op {} took too long: {:?}",
                        tape_idx, elapsed
                    )));
                }
            }

            // Clear tape and restore state
            ctx.tape.clear();
            ctx.enabled = prev_enabled;
            AUTOGRAD_ENABLED.store(prev_enabled, Ordering::Relaxed);
        }

        println!("\n=== AUTOGRAD DEBUG COMPLETE ===");
        println!("Total gradients computed: {}", gradients.len());
        Ok(gradients)
    }

    /// Compute gradients via backpropagation (v3 default — F32 grad
    /// storage policy). Public API; preserves all v3 behavior bit-equal
    /// to pre-Phase-5b semantics.
    pub fn backward(loss: &Tensor) -> Result<GradientMap> {
        Self::backward_impl(loss, GradStorePolicy::InternalFP32_PublicBF16)
    }

    /// Compute gradients via backpropagation, returning a v2-policy
    /// [`GradientMap`] whose stored gradients honor the loss tensor's
    /// dtype end-to-end (Option A of `docs/BF16_GRAD_DECISION.md`).
    ///
    /// Bridges the v3 op-dispatch backward into a v2-policy gradient
    /// map. The v3 op kernels still compute their input gradients in
    /// their native dtype (typically F32 today); the bridge unifies
    /// dtypes by casting each emitted input-grad to the loss tensor's
    /// dtype before accumulation. This is option (b) from the Phase 5b
    /// design note — the grad-map-write-time cast — and is required
    /// because the v2 `accumulate` contract errors on dtype mismatch.
    ///
    /// The forward graph does NOT need to be authored in v2 ops to use
    /// this entry: it works as a drop-in replacement for `backward`
    /// when callers want a v2-policy `GradientMap` (e.g. for
    /// `Parameter::new_v2` + `AdamWV2`).
    ///
    /// # End-to-end BF16-grad caveat
    /// This bridge emits BF16 grads only *into the returned
    /// `GradientMap`*. If the caller then assigns those grads to a
    /// parameter built via the default [`crate::Parameter::new`]
    /// (which uses
    /// [`crate::parameter::GradDtypePolicy::CastToF32`]), `set_grad`
    /// will upcast BF16 → F32 — silently nullifying the memory
    /// savings. To realize the full Option-A path
    /// (BF16-grad-end-to-end through the optimizer step) construct
    /// parameters via [`crate::Parameter::new_v2`] (Phase 4a). See
    /// `docs/BF16_GRAD_DECISION.md` §Phase 5b caveat.
    ///
    /// # CUDA Graph backward limitation
    /// Under `FLAME_CUDA_GRAPH=1` the replay path pre-allocates grad
    /// buffers at the warmup-recorded dtype (v3: F32) and bypasses the
    /// per-op cast in `backward_impl`. v2 + cuda-graph backward is
    /// therefore not yet supported; callers needing BF16 grads should
    /// leave `FLAME_CUDA_GRAPH` unset (the default).
    ///
    /// See `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Phase 5 Deliverable C
    /// Route (ii).
    #[cfg(feature = "autograd_v2")]
    pub fn backward_v2(loss: &Tensor) -> Result<GradientMap> {
        Self::backward_impl(loss, GradStorePolicy::MatchInsertedDtype)
    }

    /// Shared backward body used by both [`AutogradContext::backward`]
    /// (v3 / `InternalFP32_PublicBF16` policy) and the v2 bridge
    /// [`AutogradContext::backward_v2`] (`MatchInsertedDtype` policy).
    ///
    /// When `policy == MatchInsertedDtype`, the loss seed and every
    /// accumulated input gradient are cast to `loss.dtype()` before
    /// being inserted into the grad map. This unifies the dtype across
    /// the whole graph so `GradientMap::accumulate_v2`'s "no dtype
    /// mismatch" contract is satisfied.
    fn backward_impl(loss: &Tensor, policy: GradStorePolicy) -> Result<GradientMap> {
        // Cache profiling flag once (avoid syscall per-op)
        static PROFILE_CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let profile = *PROFILE_CACHED.get_or_init(|| {
            std::env::var("FLAME_PROFILE")
                .ok()
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        if !loss.requires_grad {
            return Err(Error::InvalidOperation(
                "backward() called on tensor that doesn't require grad".into(),
            ));
        }

        if loss.shape.elem_count() != 1 {
            return Err(Error::InvalidOperation(
                "backward() requires scalar loss tensor".into(),
            ));
        }

        if loss.rank() == 4 {
            assert_nhwc_bf16_public("AutogradContext::backward loss", loss)?;
        }

        let device = loss.device.clone();

        // Drain the tape and build index structures under the lock, then
        // release it before the backward loop. This is critical: Op::Checkpoint
        // backward re-acquires the lock to re-enable autograd and record
        // recomputed ops. Holding the lock across the whole loop deadlocks.
        let gradients = {
            let (
                tape_entries,
                prev_enabled,
                compact_index,
                needed_grad_ids,
                use_cuda_graph,
                tape_len,
                graph_phase,
            ) = {
                let mut ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd context mutex poisoned".into()))?;

                // Disable autograd during backward pass
                let prev_enabled = ctx.enabled;
                ctx.enabled = false;
                AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);

                // Build compact index from the tape for Vec-based gradient storage.
                let compact_index = {
                    use crate::gradient::CompactIndex;
                    let id_iter = std::iter::once(loss.id).chain(ctx.tape.iter().flat_map(|e| {
                        let mut ids = vec![e.output_id];
                        for (tid, _) in &e.saved_tensors {
                            ids.push(*tid);
                        }
                        for r in &e.saved_refs {
                            ids.push(r.id);
                        }
                        match &e.op {
                            Op::Add { lhs, rhs, .. }
                            | Op::Sub { lhs, rhs }
                            | Op::Mul { lhs, rhs }
                            | Op::Div { lhs, rhs, .. }
                            | Op::MatMul { lhs, rhs }
                            | Op::BatchMatMul { lhs, rhs }
                            | Op::Maximum { a: lhs, b: rhs }
                            | Op::Minimum { a: lhs, b: rhs } => {
                                ids.push(*lhs);
                                ids.push(*rhs);
                            }
                            Op::MulScalar { input, .. }
                            | Op::AddScalar { input, .. }
                            | Op::ReLU { input }
                            | Op::GELU { input }
                            | Op::SiLU { input }
                            | Op::Tanh { input }
                            | Op::Sigmoid { input }
                            | Op::Square { input }
                            | Op::Sqrt { input }
                            | Op::Sum { input, .. }
                            | Op::Mean { input, .. }
                            | Op::Transpose { input }
                            | Op::Reshape { input, .. }
                            | Op::Permute { input, .. }
                            | Op::Broadcast { input, .. }
                            | Op::UpsampleNearest2D { input, .. }
                            | Op::MaxPool2D { input, .. }
                            | Op::SumDim { input, .. }
                            | Op::SumDimKeepdim { input, .. }
                            | Op::SumDims { input, .. }
                            | Op::Repeat { input, .. }
                            | Op::MaxDim { input, .. }
                            | Op::Clamp { input, .. }
                            | Op::Abs { input }
                            | Op::Log { input }
                            | Op::Softmax { input, .. }
                            | Op::LogSoftmax { input, .. }
                            | Op::Checkpoint { input, .. }
                            | Op::CheckpointOffload { input, .. }
                            | Op::Cast { input, .. } => {
                                ids.push(*input);
                            }
                            Op::CheckpointOffloadBoundary { input_ids, .. } => {
                                for tid in input_ids {
                                    ids.push(*tid);
                                }
                            }
                            Op::Conv2d { input, weight, .. }
                            | Op::Conv2dNHWC { input, weight, .. }
                            | Op::AddBias {
                                input,
                                bias: weight,
                            } => {
                                ids.push(*input);
                                ids.push(*weight);
                            }
                            Op::Linear {
                                input,
                                weight,
                                bias,
                            } => {
                                ids.push(*input);
                                ids.push(*weight);
                                if let Some(b) = bias {
                                    ids.push(*b);
                                }
                            }
                            Op::LayerNorm { input, .. } => {
                                ids.push(*input);
                            }
                            Op::RMSNorm {
                                input,
                                weight,
                                inv_rms,
                                ..
                            } => {
                                ids.push(*input);
                                ids.push(*inv_rms);
                                if let Some(w) = weight {
                                    ids.push(*w);
                                }
                            }
                            Op::GroupNorm {
                                input,
                                weight,
                                bias,
                                ..
                            } => {
                                ids.push(*input);
                                if let Some(w) = weight {
                                    ids.push(*w);
                                }
                                if let Some(b) = bias {
                                    ids.push(*b);
                                }
                            }
                            Op::Embedding { weight, indices }
                            | Op::IndexSelect {
                                input: weight,
                                indices,
                                ..
                            } => {
                                ids.push(*weight);
                                ids.push(*indices);
                            }
                            Op::IndexAssign {
                                input,
                                indices,
                                values,
                                ..
                            } => {
                                ids.push(*input);
                                ids.push(*indices);
                                ids.push(*values);
                            }
                            Op::Cat { inputs, .. } => {
                                ids.extend(inputs.iter());
                            }
                            Op::Split { input, .. } | Op::Slice { input, .. } => {
                                ids.push(*input);
                            }
                            Op::Where { cond, t, f } => {
                                ids.push(*cond);
                                ids.push(*t);
                                ids.push(*f);
                            }
                            Op::MSELoss {
                                predictions,
                                targets,
                                ..
                            }
                            | Op::L1Loss {
                                predictions,
                                targets,
                                ..
                            }
                            | Op::HuberLoss {
                                predictions,
                                targets,
                                ..
                            }
                            | Op::BCELoss {
                                predictions,
                                targets,
                                ..
                            } => {
                                ids.push(*predictions);
                                ids.push(*targets);
                            }
                            Op::NLLLoss {
                                log_probs, targets, ..
                            } => {
                                ids.push(*log_probs);
                                ids.push(*targets);
                            }
                            Op::FlashAttention {
                                query,
                                key,
                                value,
                                mask,
                                ..
                            } => {
                                ids.push(*query);
                                ids.push(*key);
                                ids.push(*value);
                                if let Some(m) = mask {
                                    ids.push(*m);
                                }
                            }
                            Op::PrefixCausalFullAttention {
                                query, key, value, ..
                            } => {
                                ids.push(*query);
                                ids.push(*key);
                                ids.push(*value);
                            }
                            Op::SageAttention {
                                query_id,
                                key_id,
                                value_id,
                                ..
                            } => {
                                ids.push(*query_id);
                                ids.push(*key_id);
                                ids.push(*value_id);
                            }
                            Op::FusedSwiGLU { gate, up } => {
                                ids.push(*gate);
                                ids.push(*up);
                            }
                            Op::FusedSwiGLUSplit { input } => {
                                ids.push(*input);
                            }
                            Op::QkvSplitPermute { input, .. } => {
                                ids.push(*input);
                            }
                            Op::RoPePrecomputed { input, cos, sin, layout: _ } => {
                                ids.push(*input);
                                ids.push(*cos);
                                ids.push(*sin);
                            }
                            Op::GateResidual { residual, gate, x } => {
                                ids.push(*residual);
                                ids.push(*gate);
                                ids.push(*x);
                            }
                        }
                        ids
                    }));
                    CompactIndex::from_tensor_ids(id_iter)
                };

                // Build set of tensor IDs that actually need gradients
                let needed_grad_ids: std::collections::HashSet<TensorId> = {
                    let mut ids = std::collections::HashSet::new();
                    ids.insert(loss.id);
                    for e in ctx.tape.iter() {
                        ids.insert(e.output_id);
                        add_unsaved_view_inputs_to_needed(&e.op, &mut ids);
                        for (tid, tensor) in &e.saved_tensors {
                            if tensor.requires_grad() {
                                ids.insert(*tid);
                            }
                        }
                        for r in &e.saved_refs {
                            if r.tensor.requires_grad() {
                                ids.insert(r.id);
                            }
                        }
                    }
                    ids
                };

                let use_cuda_graph = crate::cuda_graph::cuda_graph_enabled();
                let tape_len = ctx.tape.len();

                let graph_phase = if use_cuda_graph {
                    let cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                        .lock()
                        .map_err(|_| {
                            Error::Training("backward graph cache mutex poisoned".into())
                        })?;
                    cache.phase(tape_len)
                } else {
                    crate::cuda_graph::BackwardPhase::Warmup
                };

                // Drain the tape — we now own all entries, lock can be released.
                let tape_entries: Vec<TapeEntry> = ctx.tape.drain(..).collect();

                (
                    tape_entries,
                    prev_enabled,
                    compact_index,
                    needed_grad_ids,
                    use_cuda_graph,
                    tape_len,
                    graph_phase,
                )
            }; // ← lock released here

            // Initialize gradient storage with compact index for O(1) Vec-based access.
            // Phase 5b bridge: under MatchInsertedDtype policy construct a v2 grad map
            // and seed the loss at its native dtype (BF16-preserving end-to-end).
            let mut gradients = match policy {
                GradStorePolicy::InternalFP32_PublicBF16 => {
                    GradientMap::with_index(device.clone(), compact_index)
                }
                GradStorePolicy::MatchInsertedDtype => {
                    GradientMap::with_index_v2(device.clone(), compact_index)
                }
            };
            let grad_target_dtype = match policy {
                GradStorePolicy::InternalFP32_PublicBF16 => DType::F32,
                GradStorePolicy::MatchInsertedDtype => loss.dtype(),
            };
            // Seed the loss grad at F32 under BOTH policies. v3 op-dispatch
            // kernels (compute_gradients) assume F32 output_grad — e.g.
            // Op::Mean's backward calls `GpuOps::broadcast` over the
            // 1.0/n-scaled seed; with a BF16 scalar seed several op
            // kernels return CUDA_ERROR_INVALID_VALUE because they were
            // never authored against BF16 scalars.
            //
            // The seed entry is `take`n on the first reverse step and
            // never persists; what does land in the v2 GradientMap is
            // the per-input-grad cast below (loop in the op-dispatch
            // body), which downcasts each emitted F32 input-grad to
            // `grad_target_dtype` (loss.dtype()) before
            // `accumulate_v2`. So the BF16-grad-end-to-end story holds
            // for the *stored* grads (the value the bridge promises);
            // only the transient seed stays F32.
            gradients.set_ones(loss.id, loss.shape.clone())?;

            // ── CUDA Graph replay path ──────────────────────────────
            if use_cuda_graph && graph_phase == crate::cuda_graph::BackwardPhase::Replay {
                let t_replay = std::time::Instant::now();
                let stream: *mut core::ffi::c_void = core::ptr::null_mut();

                {
                    let cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                        .lock()
                        .map_err(|_| {
                            Error::Training("backward graph cache mutex poisoned".into())
                        })?;
                    for entry in cache.grad_recipe() {
                        let grad =
                            Tensor::zeros_dtype(entry.shape.clone(), entry.dtype, device.clone())?;
                        gradients.set(entry.tensor_id, grad);
                    }

                    if let Some(exec) = cache.exec() {
                        crate::cuda_graph::launch(exec, stream)?;
                        crate::cuda_graph::stream_synchronize(stream)?;
                    }
                }

                {
                    let mut cache =
                        crate::cuda_graph::BACKWARD_GRAPH_CACHE
                            .lock()
                            .map_err(|_| {
                                Error::Training("backward graph cache mutex poisoned".into())
                            })?;
                    cache.advance();
                }

                if profile {
                    let dt = t_replay.elapsed();
                    eprintln!("\n╔══════════════════════════════════════════════════════════");
                    eprintln!("║ FLAME BACKWARD — CUDA GRAPH REPLAY");
                    eprintln!("╠══════════════════════════════════════════════════════════");
                    eprintln!("║ Tape entries:     {} (cached)", tape_len);
                    eprintln!("║ Replay time:      {:.3}ms", dt.as_secs_f64() * 1000.0);
                    eprintln!("╚══════════════════════════════════════════════════════════\n");
                }

                // Re-enable autograd
                {
                    let mut ctx = AUTOGRAD_CONTEXT
                        .lock()
                        .map_err(|_| Error::Training("autograd context mutex poisoned".into()))?;
                    ctx.enabled = prev_enabled;
                    AUTOGRAD_ENABLED.store(prev_enabled, Ordering::Relaxed);
                }

                return Ok(gradients);
            }

            // ── Capture or warmup path ──────────────────────────────
            let capturing =
                use_cuda_graph && graph_phase == crate::cuda_graph::BackwardPhase::Capture;
            let stream: *mut core::ffi::c_void = core::ptr::null_mut();

            if capturing {
                crate::cuda_graph::begin_capture(stream, crate::cuda_graph::CAPTURE_MODE_GLOBAL)?;
                if profile {
                    eprintln!(
                        "[cuda_graph] began capture on default stream (tape_len={})",
                        tape_len
                    );
                }
            }

            // Process tape in reverse (now from our owned Vec, lock is NOT held).
            // Checkpoint ops can safely re-acquire the lock for recompute.
            let t_backward_start = std::time::Instant::now();
            let mut nodes_executed = 0usize;
            let mut total_kernel_time = std::time::Duration::ZERO;
            let mut total_accum_time = std::time::Duration::ZERO;
            let mut slowest_nodes: Vec<(std::time::Duration, String)> = Vec::new();
            // Per-op-kind time aggregation for profiling the backward pass.
            // Keyed by the numeric discriminant of Op so we can decode in logs.
            let mut per_op_time: std::collections::HashMap<u64, (std::time::Duration, usize)> =
                std::collections::HashMap::new();

            let backward_result: Result<()> = (|| {
                let finite_check = crate::debug_finite::is_enabled();
                // Snapshot the test-only retain set once (cheap clone of a
                // small HashSet, only Some during diagnostic runs).
                let retain_ids: Option<std::collections::HashSet<TensorId>> =
                    RETAINED_INTERMEDIATE_GRAD_IDS
                        .lock()
                        .ok()
                        .and_then(|s| s.clone());
                for entry in tape_entries.iter().rev() {
                    if let Some(output_grad) = gradients.take(entry.output_id) {
                        if let Some(ref ids) = retain_ids {
                            if ids.contains(&entry.output_id) {
                                if let Ok(mut g) = RETAINED_INTERMEDIATE_GRADS.lock() {
                                    g.insert(entry.output_id, output_grad.clone());
                                }
                            }
                        }
                        let t_node = std::time::Instant::now();

                        // FLAME_DEBUG_FINITE: check incoming grad before the
                        // op computes anything. If NaN/Inf arrived here it
                        // was produced by the next-later op in the tape.
                        if finite_check {
                            let site = format!(
                                "bwd:{}@{}:output_grad",
                                op_tag(&entry.op),
                                entry.output_id.0
                            );
                            crate::debug_finite::check(&site, &output_grad)?;
                        }

                        // Compute input gradients
                        let input_grads = match compute_gradients(entry, &output_grad, &device) {
                            Ok(g) => g,
                            Err(e) => {
                                eprintln!(
                                    "[bwd:ERROR] #{} {:?} shape={:?}: {e:?}",
                                    nodes_executed,
                                    std::mem::discriminant(&entry.op),
                                    output_grad.shape().dims()
                                );
                                return Err(e);
                            }
                        };
                        let kernel_dt = t_node.elapsed();
                        total_kernel_time += kernel_dt;
                        // Aggregate by op kind.
                        let disc_u64: u64 = unsafe {
                            std::mem::transmute::<_, u64>(std::mem::discriminant(&entry.op))
                        };
                        let e = per_op_time
                            .entry(disc_u64)
                            .or_insert((std::time::Duration::ZERO, 0));
                        e.0 += kernel_dt;
                        e.1 += 1;

                        // FLAME_DEBUG_FINITE: check the produced input_grads.
                        // A non-finite result here means THIS op's backward
                        // formula produced garbage from a finite output_grad.
                        if finite_check {
                            for (tid, g) in &input_grads {
                                let site = format!(
                                    "bwd:{}@{}:grad_for:{}",
                                    op_tag(&entry.op),
                                    entry.output_id.0,
                                    tid.0
                                );
                                crate::debug_finite::check(&site, g)?;
                            }
                        }

                        // Accumulate gradients (skip frozen weight IDs to save memory).
                        // Checkpoint backward returns ALL internal gradients (including
                        // LoRA params) that aren't in needed_grad_ids — always accept those.
                        //
                        // Phase 5b bridge: both policies accumulate at F32 internally
                        // (v3 op kernels produce F32 grads). Under MatchInsertedDtype the
                        // *persistent* leaf grads are downcast to `grad_target_dtype`
                        // (loss.dtype()) in a post-loop pass below. Doing the cast at
                        // insertion-time would propagate BF16 into the next reverse-tape
                        // step's `output_grad`, and several v3 backward kernels are not
                        // authored against BF16 scalar/non-leaf grads (observed:
                        // `CUDA_ERROR_INVALID_VALUE` from Mean/Mul backwards with BF16
                        // output_grad).
                        let is_checkpoint = matches!(
                            &entry.op,
                            Op::Checkpoint { .. }
                                | Op::CheckpointOffload { .. }
                                | Op::CheckpointOffloadBoundary { .. }
                        );
                        let t_accum = std::time::Instant::now();
                        for (tensor_id, grad) in input_grads {
                            if is_checkpoint || needed_grad_ids.contains(&tensor_id) {
                                // Both v3 and the bridge accumulate at F32. v2 will be
                                // downcast in the post-loop pass.
                                let grad = if grad.dtype() == DType::F32 {
                                    grad
                                } else {
                                    grad.to_dtype(DType::F32)?
                                };
                                gradients.accumulate(tensor_id, grad)?;
                            }
                            // else: gradient for frozen weight — drop it
                        }
                        total_accum_time += t_accum.elapsed();

                        nodes_executed += 1;

                        if profile && !capturing {
                            slowest_nodes.push((kernel_dt, format!("{:?}", entry.op)));
                        }
                    }
                }
                log::debug!(
                    "[backward] processed {} of {} entries, gradients.len()={}",
                    nodes_executed,
                    tape_entries.len(),
                    gradients.len()
                );
                Ok(())
            })();

            // ── End capture (even on error, to leave stream in valid state) ──
            if capturing {
                match backward_result {
                    Ok(()) => {
                        // End capture and instantiate the graph
                        let graph = crate::cuda_graph::end_capture(stream)?;
                        let exec = crate::cuda_graph::instantiate(&graph)?;

                        // Build the gradient allocation recipe from the gradient map.
                        // This records (tensor_id, shape, dtype) for every gradient
                        // so replay can reproduce the same allocations.
                        let grad_recipe: Vec<crate::cuda_graph::GradAllocEntry> = {
                            let mut recipe = Vec::new();
                            if let Ok(iter) = gradients.iter_fp32() {
                                for (tid, tensor) in iter {
                                    recipe.push(crate::cuda_graph::GradAllocEntry {
                                        tensor_id: tid,
                                        shape: tensor.shape.clone(),
                                        dtype: tensor.dtype(),
                                    });
                                }
                            }
                            recipe
                        };

                        // Store in cache
                        let mut cache =
                            crate::cuda_graph::BACKWARD_GRAPH_CACHE
                                .lock()
                                .map_err(|_| {
                                    Error::Training("backward graph cache mutex poisoned".into())
                                })?;
                        cache.store(exec, tape_len, grad_recipe);
                        cache.advance();

                        if profile {
                            eprintln!("[cuda_graph] captured and instantiated graph ({} nodes, {} grad entries)",
                                nodes_executed, cache.grad_recipe().len());
                        }

                        // The captured graph was NOT executed during capture.
                        // We need to launch it once now to actually compute
                        // the gradients for this step.
                        if let Some(exec) = cache.exec() {
                            crate::cuda_graph::launch(exec, stream)?;
                            crate::cuda_graph::stream_synchronize(stream)?;
                        }
                    }
                    Err(e) => {
                        // Capture failed — cancel by ending capture (graph will be empty/null)
                        // and fall back to non-graph mode on next step.
                        let _ = crate::cuda_graph::end_capture(stream);
                        let mut cache =
                            crate::cuda_graph::BACKWARD_GRAPH_CACHE
                                .lock()
                                .map_err(|_| {
                                    Error::Training("backward graph cache mutex poisoned".into())
                                })?;
                        cache.invalidate();

                        eprintln!(
                            "[cuda_graph] capture failed, falling back to normal backward: {:?}",
                            e
                        );

                        // Re-enable autograd
                        {
                            let mut ctx = AUTOGRAD_CONTEXT.lock().map_err(|_| {
                                Error::Training("autograd context mutex poisoned".into())
                            })?;
                            ctx.enabled = prev_enabled;
                            AUTOGRAD_ENABLED.store(prev_enabled, Ordering::Relaxed);
                        }
                        return Err(e);
                    }
                }
            } else {
                // Normal (non-capture) path — propagate any error
                backward_result?;

                // Advance the graph cache step counter (warmup → capture next time)
                if use_cuda_graph {
                    let mut cache =
                        crate::cuda_graph::BACKWARD_GRAPH_CACHE
                            .lock()
                            .map_err(|_| {
                                Error::Training("backward graph cache mutex poisoned".into())
                            })?;
                    cache.advance();
                    if profile {
                        let phase_name = if graph_phase == crate::cuda_graph::BackwardPhase::Warmup
                        {
                            "warmup"
                        } else {
                            "normal"
                        };
                        eprintln!(
                            "[cuda_graph] {} step complete, next step will capture",
                            phase_name
                        );
                    }
                }
            }

            // Per-op-kind summary — only emitted when explicit profiling is on.
            if profile && !capturing && !per_op_time.is_empty() {
                let mut sorted: Vec<_> = per_op_time.iter().collect();
                sorted.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
                eprintln!(
                    "[bwd:per-op] top 8 by total time across {} nodes:",
                    nodes_executed
                );
                for (disc, (dt, n)) in sorted.iter().take(8) {
                    eprintln!(
                        "  disc={:<4} total={:6.1}ms  count={:4}  avg={:5.2}ms",
                        disc,
                        dt.as_secs_f64() * 1000.0,
                        n,
                        dt.as_secs_f64() * 1000.0 / (*n as f64).max(1.0)
                    );
                }
            }

            if profile && !capturing {
                let total_dt = t_backward_start.elapsed();
                let overhead = total_dt.saturating_sub(total_kernel_time + total_accum_time);
                slowest_nodes.sort_by(|a, b| b.0.cmp(&a.0));

                eprintln!("\n╔══════════════════════════════════════════════════════════");
                eprintln!("║ FLAME BACKWARD PROFILE");
                if use_cuda_graph {
                    eprintln!("║ CUDA Graph:       ENABLED (phase: {:?})", graph_phase);
                }
                eprintln!("╠══════════════════════════════════════════════════════════");
                eprintln!("║ Tape entries:     {}", tape_len);
                eprintln!("║ Nodes executed:   {}", nodes_executed);
                eprintln!("║ Total backward:   {:.3}s", total_dt.as_secs_f64());
                eprintln!(
                    "║ Kernel time:      {:.3}s ({:.1}%)",
                    total_kernel_time.as_secs_f64(),
                    100.0 * total_kernel_time.as_secs_f64() / total_dt.as_secs_f64().max(1e-9)
                );
                eprintln!(
                    "║ Accum time:       {:.3}s ({:.1}%)",
                    total_accum_time.as_secs_f64(),
                    100.0 * total_accum_time.as_secs_f64() / total_dt.as_secs_f64().max(1e-9)
                );
                eprintln!(
                    "║ Overhead:         {:.3}s ({:.1}%)",
                    overhead.as_secs_f64(),
                    100.0 * overhead.as_secs_f64() / total_dt.as_secs_f64().max(1e-9)
                );
                eprintln!(
                    "║ Per-node avg:     {:.3}ms",
                    total_dt.as_secs_f64() * 1000.0 / nodes_executed.max(1) as f64
                );
                eprintln!("╠══════════════════════════════════════════════════════════");
                eprintln!("║ Top 10 slowest ops:");
                for (i, (dt, op)) in slowest_nodes.iter().take(10).enumerate() {
                    eprintln!(
                        "║  {:2}. {:8.3}ms  {}",
                        i + 1,
                        dt.as_secs_f64() * 1000.0,
                        op
                    );
                }
                eprintln!("╚══════════════════════════════════════════════════════════\n");
            }

            // Re-enable autograd (tape was already drained at the top).
            // Also drop the checkpoint closure cache here: each forward
            // inserts one Arc<dyn Fn> per checkpoint() call, capturing per-
            // block state (weights, RoPE tables, KV pair, LoRA adapters,
            // saved tensors). Backward consumes them via `checkpoint_fns
            // .get(...)` but doesn't remove them — so without this clear,
            // `checkpoint_fns` grows by ~N-layers per training step and
            // pins megabytes of GPU storage per entry. Trainers using
            // gradient checkpointing on streamed weights (sensenova_u1,
            // anima, etc.) OOM after ~30-100 steps from this alone.
            {
                let mut ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd context mutex poisoned".into()))?;
                ctx.enabled = prev_enabled;
                AUTOGRAD_ENABLED.store(prev_enabled, Ordering::Relaxed);
                ctx.checkpoint_fns.clear();
            }

            // Phase 5b bridge post-loop pass: under MatchInsertedDtype,
            // downcast every persistent leaf grad to `grad_target_dtype`
            // (the loss tensor's dtype). The backward loop above ran at
            // F32 internally (v3 op kernels expect F32 output_grad);
            // intermediate grads were drained by `take()` and only leaf
            // grads remain. Walking those entries here realizes the
            // BF16-end-to-end storage promised by the v2 policy without
            // perturbing the v3 backward-kernel contract.
            if matches!(policy, GradStorePolicy::MatchInsertedDtype)
                && grad_target_dtype != DType::F32
            {
                gradients.cast_all_to_dtype(grad_target_dtype)?;
            }

            gradients
        };

        Ok(gradients)
    }

    /// Run a closure under activation checkpointing.
    ///
    /// During forward, the closure's intermediate tape entries are captured
    /// and removed from the main tape. Only the input tensor is saved.
    /// During backward, the closure is re-executed from the saved input
    /// to recompute intermediates, then backward runs through them.
    ///
    /// This trades ~2x compute for O(1) memory per checkpointed block
    /// instead of O(intermediates).
    ///
    /// Usage in klein-trainer:
    /// ```rust
    /// // Before (OOM):
    /// // img = double_block_forward(&weights, ...)?;
    /// // After (checkpointed):
    /// let (img_new, txt_new) = AutogradContext::checkpoint(
    ///     &[img.clone(), txt.clone()],
    ///     || double_block_forward(&weights, ...),
    /// )?;
    /// ```
    pub fn checkpoint<F>(inputs: &[Tensor], f: F) -> Result<Tensor>
    where
        F: Fn() -> Result<Tensor> + Send + Sync + 'static,
    {
        // Check if autograd is even enabled — if not, just run the closure directly.
        let was_enabled = {
            let ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.enabled
        };

        if !was_enabled {
            // No autograd → just run closure, no checkpoint overhead.
            return f();
        }

        // CRITICAL FIX: Disable autograd during the checkpoint forward pass.
        // The old code left autograd enabled, which meant every op inside the
        // closure called record_op → clone_result (GPU d2d copy) → push to tape,
        // only to immediately truncate the entire sub-tape afterward. This caused
        // hundreds of wasted GPU memcpys per block (e.g. ~300 for a double block).
        //
        // With autograd disabled during forward, the closure runs at inference
        // speed — no tape recording, no saved tensor copies. The recompute
        // closure stored below will re-run the forward WITH autograd enabled
        // during backward, which is when we actually need the tape.
        //
        // 2026-05-14 re-entry fix: save the PRIOR enabled state and restore
        // it after the forward closure, instead of unconditionally toggling
        // to true at the end. Critical when this `checkpoint` is called
        // from inside another `checkpoint`'s recompute closure (which has
        // already set enabled=false): the unconditional `store(true)` at
        // function exit clobbered the outer state and let subsequent ops
        // in the outer closure record into the tape during what was
        // supposed to be a no_grad scope. RAII via the explicit
        // save/restore below; the early `?` from `f()?` cannot leak the
        // wrong state because we use a guard struct that restores on drop.
        struct EnabledGuard {
            prior: bool,
        }
        impl Drop for EnabledGuard {
            fn drop(&mut self) {
                if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
                    ctx.enabled = self.prior;
                }
                AUTOGRAD_ENABLED.store(self.prior, Ordering::Relaxed);
            }
        }
        let _guard = {
            let mut ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            let prior = ctx.enabled;
            ctx.enabled = false;
            AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);
            EnabledGuard { prior }
        };

        let output = f()?;

        // Record the Checkpoint tape entry WHILE autograd is still off (we
        // touch ctx state but do NOT call record_op-on-tensor-ops). Then
        // the guard drops and restores the prior state at function exit —
        // re-enabling tape recording for the outer scope only if the
        // outer scope had it on.
        let input_id = inputs
            .first()
            .ok_or_else(|| Error::InvalidInput("checkpoint requires at least one input".into()))?
            .id;

        let saved: SavedTensors = inputs.iter().map(|inp| (inp.id, inp.clone())).collect();

        let mut out_with_grad = output;
        out_with_grad.requires_grad = true;

        // Only register the tape entry if the OUTER scope wants autograd
        // (prior == true). When prior == false we're inside a nested
        // no_grad / inner checkpoint and the outer scope will not run
        // backward on this Checkpoint entry anyway.
        //
        // IMPORTANT: drop the guard FIRST so ctx.enabled is restored to
        // `prior` BEFORE we call `ctx.record()` — `record()` silently
        // drops entries when `ctx.enabled == false` (see line ~602).
        let should_record = _guard.prior;
        drop(_guard);

        if should_record {
            let mut ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.checkpoint_fns.insert(out_with_grad.id, Arc::new(f));
            ctx.record(TapeEntry {
                output_id: out_with_grad.id,
                op: Op::Checkpoint {
                    input: input_id,
                    original_tape_len: 0,
                },
                saved_tensors: saved,
                saved_refs: SavedRefs::new(),
            });
        }

        Ok(out_with_grad)
    }

    /// Phase 2b (OFFLOAD_NEXT_GEN_DESIGN): narrow-scope checkpoint with
    /// closure-input-passing semantics + grow-cache backed memory offload.
    /// The closure takes `&[Tensor]` so it does NOT capture inputs by
    /// reference — between forward and backward the original input GPU
    /// storage can be freed once all clones drop. At backward time, the
    /// recompute closure pulls inputs from the global grow cache, calls
    /// the user closure with the pulled tensors, and the resulting
    /// sub-tape is backwards-traversed (same pattern as `Op::Checkpoint`).
    ///
    /// Behavior:
    ///   - Autograd off OR no grow cache installed: degrades to plain
    ///     `checkpoint(inputs, || f(inputs))` — closure-capture path.
    ///   - With cache installed: forward runs `f(inputs)` once with
    ///     autograd disabled; inputs pushed to grow cache; tape entry
    ///     holds input_ids + handles (no strong tensor refs);
    ///     `checkpoint_fns` stores a recompute closure that pulls inputs
    ///     from cache and calls `f(pulled)`.
    ///
    /// Install the cache once at training setup via
    /// `set_grow_activation_cache(...)`.
    pub fn checkpoint_offload_boundary<F>(inputs: &[Tensor], f: F) -> Result<Tensor>
    where
        F: Fn(&[Tensor]) -> Result<Tensor> + Send + Sync + 'static,
    {
        // Fast path: autograd off → just run f and return.
        let was_enabled = {
            let ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.enabled
        };
        if !was_enabled {
            return f(inputs);
        }

        // Check if the grow cache is installed.
        let cache_opt: Option<Arc<Mutex<crate::activation_offload::GrowOnDemandActivationCache>>> =
            GROW_CACHE.read().ok().and_then(|g| g.as_ref().cloned());

        if cache_opt.is_none() {
            // Degraded path: wrap as zero-arg closure with input clones and
            // delegate to plain `checkpoint`. Same memory profile as plain
            // checkpoint (strong refs hold inputs alive).
            let inputs_owned: Vec<Tensor> = inputs.to_vec();
            let f_arc = Arc::new(f);
            return Self::checkpoint(inputs, move || {
                let f = Arc::clone(&f_arc);
                f(&inputs_owned)
            });
        }

        let cache = cache_opt.unwrap();
        let input_ids: Vec<TensorId> = inputs.iter().map(|t| t.id).collect();
        let pulled_ids_slot: Arc<Mutex<Vec<TensorId>>> = Arc::new(Mutex::new(Vec::new()));

        // 2026-05-14 re-entry fix: save prior enabled state and restore it
        // on scope exit via a guard. Same rationale as in `checkpoint`.
        struct EnabledGuard {
            prior: bool,
        }
        impl Drop for EnabledGuard {
            fn drop(&mut self) {
                if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
                    ctx.enabled = self.prior;
                }
                AUTOGRAD_ENABLED.store(self.prior, Ordering::Relaxed);
            }
        }
        let _guard = {
            let mut ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            let prior = ctx.enabled;
            ctx.enabled = false;
            AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);
            EnabledGuard { prior }
        };

        let output = f(inputs)?;

        // Push inputs to grow cache. Get handles.
        let handles: Vec<crate::activation_offload::GrowHandle> = {
            let mut cache_locked = cache
                .lock()
                .map_err(|_| Error::Training("grow cache mutex poisoned".into()))?;
            inputs
                .iter()
                .map(|t| cache_locked.push(t))
                .collect::<Result<Vec<_>>>()?
        };

        // Re-enable autograd and build the recompute closure.
        let cache_for_closure = cache.clone();
        let slot_for_closure = pulled_ids_slot.clone();
        let f_arc = Arc::new(f);
        let recompute_fn = move || -> Result<Tensor> {
            // Pull the pushed tensors back to device. Each pull enqueues
            // an HtoD on the transfer stream and makes the default stream
            // wait on the pull event, so subsequent default-stream kernels
            // see fully-landed data.
            let mut cache_locked = cache_for_closure
                .lock()
                .map_err(|_| Error::Training("grow cache mutex poisoned".into()))?;
            let pulled: Vec<Tensor> = handles
                .iter()
                .map(|h| cache_locked.pull(*h))
                .collect::<Result<Vec<_>>>()?;
            drop(cache_locked);

            // Mark pulled tensors as requiring grad so the sub-tape
            // built inside `f` propagates grads back through them. The
            // pulled IDs are fresh (independent of the original input
            // IDs); the backward arm remaps via `pulled_ids_slot`.
            let pulled: Vec<Tensor> = pulled
                .into_iter()
                .map(|mut t| {
                    t.requires_grad = true;
                    t
                })
                .collect();

            // Publish pulled IDs into the side-channel slot so the
            // backward arm can remap grads from pulled-ID → input-ID.
            {
                let mut slot = slot_for_closure
                    .lock()
                    .map_err(|_| Error::Training("pulled_ids_slot mutex poisoned".into()))?;
                *slot = pulled.iter().map(|t| t.id).collect();
            }

            let f = Arc::clone(&f_arc);
            f(&pulled)
        };

        // Mark the returned output as requiring grad. Store the recompute
        // closure under the output id, then record the new Op variant on
        // the tape with EMPTY saved_tensors (no strong refs to inputs).
        // Only record if the OUTER scope had autograd enabled.
        //
        // IMPORTANT: drop the guard FIRST so ctx.enabled is restored to
        // `prior` BEFORE `ctx.record()` — record() drops entries when
        // ctx.enabled == false.
        let mut out_with_grad = output;
        out_with_grad.requires_grad = true;

        let should_record = _guard.prior;
        drop(_guard);

        if should_record {
            let mut ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.checkpoint_fns
                .insert(out_with_grad.id, Arc::new(recompute_fn));
            ctx.record(TapeEntry {
                output_id: out_with_grad.id,
                op: Op::CheckpointOffloadBoundary {
                    input_ids,
                    pulled_ids_slot,
                },
                saved_tensors: SavedTensors::new(), // intentionally empty
                saved_refs: SavedRefs::new(),
            });
        }

        Ok(out_with_grad)
    }

    /// Level 2 activation offload checkpoint. Runs `f()` ONCE with autograd
    /// enabled, captures the sub-tape, then offloads every saved tensor to
    /// CPU via the global ActivationOffloadPool. During backward, saved
    /// tensors are pulled from CPU and the stored sub-tape is walked —
    /// NO recompute needed. This eliminates ~1.5s/step of recompute overhead.
    ///
    /// If no pool is set or the pool runs out of slots mid-forward, falls
    /// back to standard `checkpoint()` (recompute, no offload).
    ///
    /// Requires `set_activation_offload_pool()` called at training setup.
    pub fn checkpoint_offload<F>(inputs: &[Tensor], f: F) -> Result<Tensor>
    where
        F: Fn() -> Result<Tensor> + Send + Sync + 'static,
    {
        // No pool installed (or torn down) → fall back to standard checkpoint.
        let pool_arc = match ACTIVATION_POOL
            .read()
            .ok()
            .and_then(|g| g.as_ref().cloned())
        {
            Some(p) => p,
            None => return Self::checkpoint(inputs, f),
        };

        // No autograd → just run closure (sampling path).
        let was_enabled = {
            let ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.enabled
        };
        if !was_enabled {
            return f();
        }

        // Record tape position before the forward pass.
        let tape_start = {
            let ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.tape.len()
        };

        // Run forward ONCE with autograd ENABLED — ops record to the tape.
        // This is the key difference from checkpoint(): we keep the sub-tape
        // instead of discarding it and storing a recompute closure.
        let output = f()?;

        let input_id = inputs
            .first()
            .ok_or_else(|| {
                Error::InvalidInput("checkpoint_offload requires at least one input".into())
            })?
            .id;

        // Extract the sub-tape entries produced by the forward pass.
        let sub_tape: Vec<TapeEntry> = {
            let mut ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            if ctx.tape.len() > tape_start {
                ctx.tape.drain(tape_start..).collect()
            } else {
                Vec::new()
            }
        };

        // Offload every saved tensor from the sub-tape to CPU. If any push
        // fails (pool full), fall back to standard checkpoint for this block.
        let mut offloaded_entries: Vec<OffloadedTapeEntry> = Vec::with_capacity(sub_tape.len());
        let mut offload_failed = false;
        {
            let mut pool = pool_arc
                .lock()
                .map_err(|_| Error::Training("offload pool mutex poisoned".into()))?;

            for entry in &sub_tape {
                let mut saved_ids = SmallVec::new();
                let mut handles = SmallVec::new();
                let mut resident = SmallVec::new();

                // Walk both legacy saved_tensors AND new saved_refs. The
                // SavedRef path stores a Tensor inside, so we can adapt it
                // to the same (tid, tensor) shape for offload registration.
                let combined: Vec<(TensorId, Tensor)> = entry
                    .saved_tensors
                    .iter()
                    .map(|(tid, t)| (*tid, t.clone()))
                    .chain(entry.saved_refs.iter().map(|r| (r.id, r.tensor.clone())))
                    .collect();
                for (tid, tensor) in &combined {
                    saved_ids.push(*tid);
                    // Only offload BF16 activations that require grad (intermediates).
                    // Skip: frozen weights (no grad, already GPU-resident, unchanged
                    // between forward and backward), non-BF16 tensors (F32 grads etc).
                    if tensor.dtype() == DType::BF16 && tensor.requires_grad {
                        match pool.push(tensor) {
                            Ok(h) => handles.push(Some(h)),
                            Err(e) => {
                                // Pool exhausted. Abort offload for this block.
                                log::warn!("checkpoint_offload: pool push failed ({e}), falling back to recompute. \
                                    tensor shape={:?} dtype={:?}, {} handles already pushed",
                                    tensor.shape().dims(), tensor.dtype(), handles.len());
                                offload_failed = true;
                                break;
                            }
                        }
                    } else {
                        handles.push(None);
                        resident.push((*tid, tensor.clone()));
                    }
                }

                if offload_failed {
                    // Pull back partial handles from the current entry
                    // (tensors 0..K-1 that succeeded before K failed).
                    for h in &handles {
                        if let Some(handle) = h {
                            let _ = pool.pull(*handle);
                        }
                    }
                    // Pull back all handles from previously completed entries.
                    for oe in &offloaded_entries {
                        for h in &oe.offload_handles {
                            if let Some(handle) = h {
                                let _ = pool.pull(*handle);
                            }
                        }
                    }
                    break;
                }

                offloaded_entries.push(OffloadedTapeEntry {
                    output_id: entry.output_id,
                    op: entry.op.clone(),
                    saved_ids,
                    offload_handles: handles,
                    resident_fallback: resident,
                });
            }
        }

        if offload_failed {
            log::warn!("checkpoint_offload: falling back to recompute checkpoint (pool exhausted after {} entries)",
                offloaded_entries.len());
            drop(sub_tape);
            return Self::checkpoint(inputs, f);
        }

        log::debug!(
            "checkpoint_offload: offloaded {} sub-tape entries successfully",
            offloaded_entries.len()
        );
        // Success: record a single CheckpointOffload op on the outer tape
        // with the offloaded sub-tape. No recompute closure stored.
        let mut out_with_grad = output;
        out_with_grad.requires_grad = true;

        {
            let mut ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.record(TapeEntry {
                output_id: out_with_grad.id,
                op: Op::CheckpointOffload {
                    input: input_id,
                    sub_tape: offloaded_entries,
                },
                saved_tensors: SmallVec::new(), // All saved tensors are in the sub-tape
                saved_refs: SavedRefs::new(),
            });
        }

        Ok(out_with_grad)
    }
}

/// RAII guard for no_grad mode
pub struct NoGradGuard {
    prev_state: bool,
}

impl NoGradGuard {
    fn new() -> Self {
        if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
            let prev = ctx.enabled;
            ctx.enabled = false;
            AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);
            Self { prev_state: prev }
        } else {
            Self { prev_state: true }
        }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
            ctx.enabled = self.prev_state;
            AUTOGRAD_ENABLED.store(self.prev_state, Ordering::Relaxed);
        }
    }
}

/// Compute gradients for a single operation
fn compute_gradients(
    entry: &TapeEntry,
    output_grad_raw: &Tensor,
    device: &Arc<CudaDevice>,
) -> Result<GradVec> {
    // Keep gradients in their incoming dtype (typically F32 from GradientMap).
    // GpuOps handles mixed F32×BF16 operations internally by casting.
    // Forcing BF16 here caused overflow (Inf) in deep checkpoint backward chains.
    let output_grad = output_grad_raw;

    // Optional backtrace: print op and shapes when FLAME_BACKWARD_TRACE=1
    // Cached to avoid syscall per-op (was ~0.5ms × 800 ops = 400ms overhead)
    static TRACE_CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let trace = *TRACE_CACHED.get_or_init(|| {
        std::env::var("FLAME_BACKWARD_TRACE")
            .ok()
            .map(|v| v == "1")
            .unwrap_or(false)
    });
    if trace {
        let og = output_grad.shape().dims().to_vec();
        let saved: Vec<Vec<usize>> = entry
            .saved_tensors
            .iter()
            .map(|(_, t)| t.shape().dims().to_vec())
            .chain(
                entry
                    .saved_refs
                    .iter()
                    .map(|r| r.tensor.shape().dims().to_vec()),
            )
            .collect();
        println!("[backtrace] op={:?}", entry.op);
        println!("[backtrace] out_grad={:?} saved={:?}", og, saved);
    }
    // Helper to fetch saved tensors, restoring or recomputing if checkpointed.
    // Fast path: skip the CHECKPOINT_MANAGER mutex when no checkpoints exist
    // (Relaxed atomic load ~1ns vs ~25ns for uncontended mutex lock per op).
    //
    // IMPORTANT: materialize non-contiguous saves before returning. Backward
    // kernels read saved tensors via raw device pointers (tensor_raw_ptr,
    // storage.try_as_slice_*) that are the parent storage base — they
    // ignore custom_strides and view_offset. A saved narrow/permute view
    // would silently alias the wrong storage region. Bug #4 (Klein 4B
    // 2026-04-25): saved `up_proj`/`gate_proj` (narrow views of `gate_up`,
    // itself a narrow of `qkv_mlp`) caused Op::Mul/Op::SiLU backward to
    // read from offset 0 of `qkv_mlp` instead of the view's start.
    let mut fetch_saved = |tid: &TensorId| -> Result<Tensor> {
        let raw = if CHECKPOINT_HAS_ENTRIES.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(t) = CHECKPOINT_MANAGER
                .lock()
                .map_err(|_| Error::Training("checkpoint manager mutex poisoned".into()))?
                .fetch_saved(*tid, device)?
            {
                t
            } else {
                entry
                    .get_saved(tid)
                    .cloned()
                    .ok_or_else(|| Error::InvalidOperation("Missing saved tensor".into()))?
            }
        } else {
            entry
                .get_saved(tid)
                .cloned()
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor".into()))?
        };
        if raw.is_contiguous() {
            Ok(raw)
        } else {
            raw.contiguous()
        }
    };

    let grads = match &entry.op {
        Op::Add {
            lhs,
            rhs,
            lhs_shape,
            rhs_shape,
        } => {
            // Gradient flows unchanged to both inputs, but handle broadcasting.
            let grad_lhs = if lhs_shape != output_grad.shape() {
                reduce_grad_for_broadcast(output_grad, lhs_shape)?
            } else {
                output_grad.clone()
            };

            let grad_rhs = if rhs_shape != output_grad.shape() {
                reduce_grad_for_broadcast(output_grad, rhs_shape)?
            } else {
                output_grad.clone()
            };

            Ok(smallvec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }

        Op::Sub { lhs, rhs } => {
            // d/dx(x-y) = 1, d/dy(x-y) = -1
            let neg_grad = GpuOps::mul_scalar(output_grad, -1.0)?;
            Ok(smallvec![(*lhs, output_grad.clone()), (*rhs, neg_grad)])
        }

        Op::Mul { lhs, rhs } => {
            // d/dx(x*y) = y, d/dy(x*y) = x
            static DEBUG_CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let _verbose = *DEBUG_CACHED
                .get_or_init(|| std::env::var("DEBUG_AUTOGRAD").ok().as_deref() == Some("1"));
            if _verbose {
                println!("  Computing Mul gradients...");
            }
            if _verbose {
                println!("  Getting saved tensors for lhs={:?}, rhs={:?}", lhs, rhs);
            }

            let lhs_tensor = &fetch_saved(lhs)?;
            let rhs_tensor = &fetch_saved(rhs)?;

            if _verbose {
                println!("  Got saved tensors, computing grad_lhs...");
            }
            // Use GPU ops directly to avoid autograd recording
            let mut grad_lhs = GpuOps::mul(output_grad, rhs_tensor)?;
            if _verbose {
                println!("  grad_lhs computed, computing grad_rhs...");
            }
            let mut grad_rhs = GpuOps::mul(output_grad, lhs_tensor)?;

            // Reduce for broadcasting if shapes differ
            if grad_lhs.shape() != lhs_tensor.shape() {
                grad_lhs = reduce_grad_for_broadcast(&grad_lhs, lhs_tensor.shape())?;
            }
            if grad_rhs.shape() != rhs_tensor.shape() {
                grad_rhs = reduce_grad_for_broadcast(&grad_rhs, rhs_tensor.shape())?;
            }
            if _verbose {
                println!("  Both gradients computed");
            }

            Ok(smallvec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }

        Op::MulScalar { input, scalar } => {
            // d/dx(s*x) = s
            let grad = output_grad.mul_scalar(*scalar)?;
            Ok(smallvec![(*input, grad)])
        }

        Op::AddScalar { input, scalar: _ } => {
            // d/dx(x+s) = 1
            Ok(smallvec![(*input, output_grad.clone())])
        }

        Op::MatMul { lhs, rhs } => {
            let lhs_tensor = &fetch_saved(lhs)?;
            let rhs_tensor = &fetch_saved(rhs)?;
            let og_dtype = output_grad.dtype();
            let need_lhs_grad = lhs_tensor.requires_grad();
            let need_rhs_grad = rhs_tensor.requires_grad();
            if !need_lhs_grad && !need_rhs_grad {
                return Ok(SmallVec::new());
            }

            // ---- Systemic F32 backward (precision fix) ----
            // The BF16 paths below round both the incoming grad and the
            // produced grad to BF16 at the cuBLASLt boundary; across ~2
            // matmuls/block × 32 blocks this cumulative rounding rotates
            // klein's composed backward (PROVEN: per-block bwd cos 0.51-0.68
            // while fwd cos 0.99). Under FLAME_BWD_F32, compute the matmul
            // backward entirely in F32. Handles 2D and 3D×2D (the Linear-layer
            // cases in Klein/Flux blocks); other ranks fall through to BF16.
            if std::env::var("FLAME_BWD_F32").as_deref() == Ok("1") {
                let (lr, rr, ogr) =
                    (lhs_tensor.rank(), rhs_tensor.rank(), output_grad.rank());
                let two_d = lr == 2 && rr == 2 && ogr == 2;
                let three_by_two = lr == 3 && rr == 2 && ogr == 3;
                if two_d || three_by_two {
                    let f32c = |t: &Tensor| -> Result<Tensor> {
                        let u = t.to_dtype_no_grad(DType::F32)?;
                        if u.is_contiguous() { Ok(u) } else { u.contiguous() }
                    };
                    let og_f32 = f32c(output_grad)?;
                    let rhs_f32 = f32c(rhs_tensor)?;
                    let lhs_f32 = f32c(lhs_tensor)?;
                    let rd = rhs_f32.shape().dims().to_vec();
                    let (k, n) = (rd[0], rd[1]);
                    let (og2d, lhs2d, lhs_outshape) = if two_d {
                        let m = og_f32.shape().dims()[0];
                        (og_f32.clone(), lhs_f32.clone(), vec![m, k])
                    } else {
                        let d = lhs_f32.shape().dims().to_vec();
                        let (bb, m) = (d[0], d[1]);
                        (
                            og_f32.reshape(&[bb * m, n])?,
                            lhs_f32.reshape(&[bb * m, k])?,
                            vec![bb, m, k],
                        )
                    };
                    let mut grads: GradVec = SmallVec::new();
                    if need_lhs_grad {
                        // grad_lhs = og @ rhs^T
                        let rhs_t = rhs_f32.transpose()?.contiguous()?;
                        let gl = og2d.matmul(&rhs_t)?;
                        let gl = if two_d { gl } else { gl.reshape(&lhs_outshape)? };
                        grads.push((*lhs, gl));
                    }
                    if need_rhs_grad {
                        // grad_rhs = lhs^T @ og
                        let lhs_t = lhs2d.transpose()?.contiguous()?;
                        grads.push((*rhs, lhs_t.matmul(&og2d)?));
                    }
                    return Ok(grads);
                }
            }

            // Fast path: BF16 2D. Use cuBLASLt with trans flags so neither
            // operand materializes a transpose. This is the dominant backward
            // for Linear layers in Klein/Flux DiT blocks — old path used
            // `transpose2d_bf16` which was a full BF16 memcpy per call, 2× per
            // MatMul backward. New path: 0 transposes.
            //
            // The incoming grad is F32 (gradient::accumulate upcasts everything
            // to F32 for stability). Cast it down to BF16 for the fused kernel.
            // cuBLASLt gemm internally accumulates in F32, so this cast only
            // affects the INPUT grad precision, not the math.
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            if lhs_tensor.dtype() == DType::BF16
                && rhs_tensor.dtype() == DType::BF16
                && lhs_tensor.rank() == 2
                && rhs_tensor.rank() == 2
                && output_grad.rank() == 2
            {
                let grad_bf16_owned;
                let grad_bf16: &Tensor = if output_grad.dtype() == DType::BF16 {
                    output_grad
                } else {
                    grad_bf16_owned = output_grad.to_dtype_no_grad(DType::BF16)?;
                    &grad_bf16_owned
                };
                let mut grads: GradVec = SmallVec::new();
                if need_lhs_grad {
                    // grad_lhs = output_grad @ rhs^T
                    let grad_lhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                        grad_bf16, rhs_tensor, false, true,
                    )?;
                    grads.push((*lhs, grad_lhs));
                }
                if need_rhs_grad {
                    // grad_rhs = lhs^T @ output_grad
                    let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                        lhs_tensor, grad_bf16, true, false,
                    )?;
                    grads.push((*rhs, grad_rhs));
                }
                return Ok(grads);
            }

            // 3D×2D path: lhs=[B,M,K], rhs=[K,N], out=[B,M,N]
            // Flatten to 2D, use matmul_bf16_trans with trans flags to avoid
            // materializing transposes (each was a full GPU memcpy).
            if lhs_tensor.rank() == 3 && rhs_tensor.rank() == 2 {
                let ld = lhs_tensor.shape().dims();
                let (batch, m, k) = (ld[0], ld[1], ld[2]);
                let n = rhs_tensor.shape().dims()[1];
                let og_2d = output_grad.reshape(&[batch * m, n])?;
                let lhs_2d = lhs_tensor.reshape(&[batch * m, k])?;

                // Use matmul_bf16_trans when possible — 0 transposes vs 2
                #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
                if rhs_tensor.dtype() == DType::BF16 && lhs_2d.dtype() == DType::BF16 {
                    let og_bf16 = if og_2d.dtype() == DType::BF16 {
                        og_2d
                    } else {
                        og_2d.to_dtype_no_grad(DType::BF16)?
                    };
                    let mut grads: GradVec = SmallVec::new();
                    if need_lhs_grad {
                        // grad_lhs = og @ rhs^T
                        let grad_lhs_2d = crate::ops::gemm_bf16::matmul_bf16_trans(
                            &og_bf16, rhs_tensor, false, true,
                        )?;
                        let grad_lhs = grad_lhs_2d.reshape(&[batch, m, k])?;
                        grads.push((*lhs, grad_lhs));
                    }
                    if need_rhs_grad {
                        // grad_rhs = lhs^T @ og
                        let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                            &lhs_2d, &og_bf16, true, false,
                        )?;
                        grads.push((*rhs, grad_rhs));
                    }
                    return Ok(grads);
                }

                // Fallback for non-BF16
                let og_cast = if og_2d.dtype() != rhs_tensor.dtype() {
                    og_2d.to_dtype_no_grad(rhs_tensor.dtype())?
                } else {
                    og_2d.clone()
                };
                let lhs_cast = if lhs_2d.dtype() != og_cast.dtype() {
                    lhs_2d.to_dtype_no_grad(og_cast.dtype())?
                } else {
                    lhs_2d
                };

                let mut grads: GradVec = SmallVec::new();
                if need_lhs_grad {
                    let rhs_t = GpuOps::transpose(rhs_tensor)?;
                    let grad_lhs_2d = GpuOps::matmul(&og_cast, &rhs_t)?;
                    let grad_lhs = grad_lhs_2d.reshape(&[batch, m, k])?;
                    grads.push((*lhs, grad_lhs));
                }
                if need_rhs_grad {
                    let lhs_t = GpuOps::transpose(&lhs_cast)?;
                    let grad_rhs = GpuOps::matmul(&lhs_t, &og_cast)?;
                    grads.push((*rhs, grad_rhs));
                }

                return Ok(grads);
            }

            // 3D×3D path: lhs=[B,M,K], rhs=[B,K,N], out=[B,M,N]
            // Use matmul_bf16_trans to avoid materializing transposes.
            // grad_lhs = output_grad @ rhs^T   (trans_b=true)
            // grad_rhs = lhs^T @ output_grad   (trans_a=true)
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            if lhs_tensor.rank() == 3
                && rhs_tensor.rank() == 3
                && lhs_tensor.dtype() == DType::BF16
                && rhs_tensor.dtype() == DType::BF16
            {
                let grad_bf16_owned;
                let grad_bf16: &Tensor = if output_grad.dtype() == DType::BF16 {
                    output_grad
                } else {
                    grad_bf16_owned = output_grad.to_dtype_no_grad(DType::BF16)?;
                    &grad_bf16_owned
                };
                let mut grads: GradVec = SmallVec::new();
                if need_lhs_grad {
                    let grad_lhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                        grad_bf16, rhs_tensor, false, true,
                    )?;
                    grads.push((*lhs, grad_lhs));
                }
                if need_rhs_grad {
                    let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                        lhs_tensor, grad_bf16, true, false,
                    )?;
                    grads.push((*rhs, grad_rhs));
                }
                return Ok(grads);
            }

            // Slow / fallback path (non-BF16, non-2D, mixed dtype): materialize
            // transposes and call GpuOps::matmul as before.
            let mut grads: GradVec = SmallVec::new();
            if need_lhs_grad {
                let rhs_for_grad = if rhs_tensor.dtype() != og_dtype {
                    let cast = rhs_tensor.to_dtype_no_grad(og_dtype)?;
                    GpuOps::transpose(&cast)?
                } else if og_dtype == DType::BF16 && rhs_tensor.rank() == 2 {
                    crate::bf16_elementwise::transpose2d_bf16(rhs_tensor)?
                } else {
                    GpuOps::transpose(rhs_tensor)?
                };
                let grad_lhs = GpuOps::matmul(output_grad, &rhs_for_grad)?;
                grads.push((*lhs, grad_lhs));
            }

            if need_rhs_grad {
                let lhs_for_grad = if lhs_tensor.dtype() != og_dtype {
                    let cast = lhs_tensor.to_dtype_no_grad(og_dtype)?;
                    GpuOps::transpose(&cast)?
                } else if og_dtype == DType::BF16 && lhs_tensor.rank() == 2 {
                    crate::bf16_elementwise::transpose2d_bf16(lhs_tensor)?
                } else {
                    GpuOps::transpose(lhs_tensor)?
                };
                let grad_rhs = GpuOps::matmul(&lhs_for_grad, output_grad)?;
                grads.push((*rhs, grad_rhs));
            }

            Ok(grads)
        }

        Op::ReLU { input } => {
            // Fused ReLU backward: single CUDA kernel (grad * 1[x > 0]).
            let x = fetch_saved(input)?;
            let grad = fused_unary_backward(
                "relu_backward",
                output_grad,
                &x,
                &device,
                crate::cuda::ffi::flame_relu_backward_bf16,
                crate::cuda::ffi::flame_relu_backward_f32,
            )?;
            Ok(smallvec![(*input, grad)])
        }

        Op::GELU { input } => {
            // Fused GELU backward (tanh-approx): single CUDA kernel.
            let x = fetch_saved(input)?;
            let grad = fused_unary_backward(
                "gelu_backward",
                output_grad,
                &x,
                &device,
                crate::cuda::ffi::flame_gelu_backward_bf16,
                crate::cuda::ffi::flame_gelu_backward_f32,
            )?;
            Ok(smallvec![(*input, grad)])
        }

        Op::SiLU { input } => {
            // Fused SiLU backward: single CUDA kernel instead of 7 GpuOps calls.
            // SiLU'(x) = sig(x) + x*sig(x)*(1-sig(x))
            let x = fetch_saved(input)?;
            let n = x.shape().elem_count() as i64;
            let stream = device.cuda_stream_raw_ptr();

            // Try fused kernel path (BF16 or F32)
            let grad = if output_grad.dtype() == DType::BF16 && x.dtype() == DType::BF16 {
                let mut out = Tensor::empty_dtype(x.shape().clone(), DType::BF16, device.clone())?;
                let status = unsafe {
                    crate::cuda::ffi::flame_silu_backward_bf16(
                        tensor_raw_ptr(output_grad)?,
                        tensor_raw_ptr(&x)?,
                        tensor_raw_ptr_mut(&mut out)?,
                        n,
                        stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda("flame_silu_backward_bf16 failed".into()));
                }
                out
            } else if output_grad.dtype() == DType::F32 && x.dtype() == DType::F32 {
                let mut out = Tensor::empty_dtype(x.shape().clone(), DType::F32, device.clone())?;
                let status = unsafe {
                    crate::cuda::ffi::flame_silu_backward_f32(
                        tensor_raw_ptr(output_grad)?,
                        tensor_raw_ptr(&x)?,
                        tensor_raw_ptr_mut(&mut out)?,
                        n,
                        stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda("flame_silu_backward_f32 failed".into()));
                }
                out
            } else {
                // Fallback: mixed dtypes — cast and use f32 kernel
                let og_f32 = if output_grad.dtype() != DType::F32 {
                    output_grad.to_dtype_no_grad(DType::F32)?
                } else {
                    output_grad.clone()
                };
                let x_f32 = if x.dtype() != DType::F32 {
                    x.to_dtype_no_grad(DType::F32)?
                } else {
                    x
                };
                let mut out =
                    Tensor::empty_dtype(x_f32.shape().clone(), DType::F32, device.clone())?;
                let status = unsafe {
                    crate::cuda::ffi::flame_silu_backward_f32(
                        tensor_raw_ptr(&og_f32)?,
                        tensor_raw_ptr(&x_f32)?,
                        tensor_raw_ptr_mut(&mut out)?,
                        n,
                        stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda("flame_silu_backward_f32 failed".into()));
                }
                out
            };
            Ok(smallvec![(*input, grad)])
        }

        Op::Tanh { input } => {
            // Fused Tanh backward: single CUDA kernel (grad * (1 - tanh(x)^2)).
            let x = fetch_saved(input)?;
            let grad = fused_unary_backward(
                "tanh_backward",
                output_grad,
                &x,
                &device,
                crate::cuda::ffi::flame_tanh_backward_bf16,
                crate::cuda::ffi::flame_tanh_backward_f32,
            )?;
            Ok(smallvec![(*input, grad)])
        }

        Op::Sigmoid { input } => {
            // Fused Sigmoid backward: single CUDA kernel (grad * sig * (1 - sig)).
            let x = fetch_saved(input)?;
            let grad = fused_unary_backward(
                "sigmoid_backward",
                output_grad,
                &x,
                &device,
                crate::cuda::ffi::flame_sigmoid_backward_bf16,
                crate::cuda::ffi::flame_sigmoid_backward_f32,
            )?;
            Ok(smallvec![(*input, grad)])
        }

        Op::Square { input } => {
            // d/dx(x^2) = 2x
            let input_tensor = &fetch_saved(input)?;
            let two_x = GpuOps::mul_scalar(input_tensor, 2.0)?;
            let grad = GpuOps::mul(output_grad, &two_x)?;
            Ok(smallvec![(*input, grad)])
        }

        Op::Sqrt { input } => {
            // d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 * x^(-0.5)
            // Use the output (sqrt(x)) from saved tensor to compute: grad * 0.5 / sqrt(x)
            let input_tensor = fetch_saved(input)?;
            let sqrt_x = GpuOps::sqrt(&input_tensor)?;
            let half_inv_sqrt = GpuOps::div(&GpuOps::mul_scalar(output_grad, 0.5)?, &sqrt_x)?;
            Ok(smallvec![(*input, half_inv_sqrt)])
        }

        Op::Sum { input, input_shape } => {
            // Gradient of sum: broadcast scalar grad to input shape.
            // Sum forward accumulates in F32 for precision, but backward is
            // just a broadcast — no accumulation, so match the input dtype.
            let up_ranked = expand_to_rank(output_grad, input_shape.dims().len())?;
            let expanded = GpuOps::broadcast(&up_ranked, input_shape)?;
            let input_tensor = entry.get_saved(input);
            let result = if let Some(inp) = input_tensor {
                if expanded.dtype() != inp.dtype() {
                    expanded.to_dtype_no_grad(inp.dtype())?
                } else {
                    expanded
                }
            } else {
                expanded
            };
            Ok(smallvec![(*input, result)])
        }

        Op::Cast { input, from, to: _ } => {
            // Gradient of cast passes through; cast grad back to input dtype
            let g = output_grad.to_dtype_no_grad(*from)?;
            Ok(smallvec![(*input, g)])
        }

        Op::Checkpoint {
            input,
            original_tape_len: _,
        } => {
            // ────────────────────────────────────────────────────────────────
            // PyTorch-style checkpoint backward (detach-recompute pattern)
            //
            // 1. Detach saved inputs → new leaf tensors (fresh IDs, same GPU
            //    storage, requires_grad=true). This disconnects from the outer
            //    autograd graph.
            // 2. Re-run the closure with autograd enabled → builds an
            //    ISOLATED local sub-tape rooted at the detached leaves.
            // 3. Walk the local sub-tape in reverse (eager-free), computing
            //    VJPs. Intermediates freed as each entry is consumed.
            // 4. Collect gradients from the detached leaf IDs.
            // 5. Map them back to the original input IDs for the outer backward.
            //
            // Memory: peak = ONE block's intermediates, not all blocks.
            // ────────────────────────────────────────────────────────────────
            let _ckpt_t0 = std::time::Instant::now();

            let recompute_fn = {
                let ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.checkpoint_fns
                    .get(&entry.output_id)
                    .cloned()
                    .ok_or_else(|| {
                        Error::Training("Checkpoint backward: no recompute closure found".into())
                    })?
            };

            // ── Step 1: Detach inputs ──
            // Create leaf tensors with fresh IDs. The closure will operate on
            // the SAME GPU data (Arc bump, zero copy) but the new IDs are not
            // connected to any tape entry in the outer graph.
            let original_input_ids: Vec<TensorId> = entry
                .saved_tensors
                .iter()
                .map(|(tid, _)| *tid)
                .chain(entry.saved_refs.iter().map(|r| r.id))
                .collect();
            // We don't actually pass detached inputs to the closure — the
            // closure captures its own input clones. The detach concept here
            // means the sub-tape we build is isolated because we record to
            // a fresh section of the global tape and drain it.

            // ── Step 2: Recompute with autograd → local sub-tape ──
            let tape_start = {
                let ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.tape.len()
            };

            {
                let mut ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.enabled = true;
                AUTOGRAD_ENABLED.store(true, Ordering::Relaxed);
            }

            let _recomp_t0 = std::time::Instant::now();
            let recomputed_output = {
                let _recompute_guard = CheckpointRecomputeGuard::enter();
                (recompute_fn)()?
            };
            let _recomp_dt = _recomp_t0.elapsed();

            // Drain the local sub-tape from the global tape
            let mut sub_tape: Vec<TapeEntry> = {
                let mut ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.enabled = false;
                AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);
                if ctx.tape.len() > tape_start {
                    ctx.tape.drain(tape_start..).collect()
                } else {
                    Vec::new()
                }
            };

            // Drop the recomputed output — its storage is shared via Arc
            // with the sub-tape's saved tensors.
            drop(recomputed_output);

            // ── Step 3: Eager-free local backward ──
            // Collect IDs we need gradients for: checkpoint inputs + trainable
            // params (requires_grad=true). Intermediate output_ids are included
            // for gradient FLOW but dropped from the final result.
            let n_sub_entries = sub_tape.len();
            let mut trainable_ids = std::collections::HashSet::new();
            trainable_ids.insert(*input);
            let mut all_needed = std::collections::HashSet::new();
            all_needed.insert(*input);
            for e in &sub_tape {
                all_needed.insert(e.output_id);
                add_unsaved_view_inputs_to_needed(&e.op, &mut all_needed);
                for (sid, st) in &e.saved_tensors {
                    if st.requires_grad() {
                        trainable_ids.insert(*sid);
                        all_needed.insert(*sid);
                    }
                }
                for r in &e.saved_refs {
                    if r.tensor.requires_grad() {
                        trainable_ids.insert(r.id);
                        all_needed.insert(r.id);
                    }
                }
            }

            // Build compact gradient map for the sub-backward
            let sub_compact = {
                use crate::gradient::CompactIndex;
                let ids = sub_tape
                    .iter()
                    .flat_map(|e| {
                        let mut ids = vec![e.output_id];
                        for (sid, _) in &e.saved_tensors {
                            ids.push(*sid);
                        }
                        for r in &e.saved_refs {
                            ids.push(r.id);
                        }
                        ids
                    })
                    .chain(std::iter::once(*input));
                CompactIndex::from_tensor_ids(ids)
            };
            let mut sub_grads =
                crate::gradient::GradientMap::with_index(device.clone(), sub_compact);

            // Seed with upstream gradient at the recomputed output
            if let Some(last_entry) = sub_tape.last() {
                sub_grads.set(last_entry.output_id, output_grad.clone());
            }

            // Walk sub-tape in reverse, CONSUMING each entry (eager-free).
            // Each entry's saved tensors are freed immediately after its
            // gradient computation completes → peak VRAM = ONE op's saved
            // tensors, not the entire block's worth.
            static CKPT_SUBPROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let ckpt_subprofile = *CKPT_SUBPROFILE.get_or_init(|| {
                std::env::var("FLAME_CKPT_SUBPROFILE")
                    .ok()
                    .map(|v| v == "1")
                    .unwrap_or(false)
            });
            let mut sub_op_time: Option<
                std::collections::HashMap<&'static str, (std::time::Duration, usize)>,
            > = if ckpt_subprofile {
                Some(std::collections::HashMap::new())
            } else {
                None
            };
            let mut sub_accum_time = std::time::Duration::ZERO;
            let mut n_bf16_casts = 0u32;
            sub_tape.reverse();
            // Soul.md trap support: snapshot the retain set so the inner walk
            // can capture intermediate grads from recompute-pass IDs. Mirrors
            // the same logic in `Op::CheckpointOffloadBoundary` backward.
            let sub_retain_ids: Option<std::collections::HashSet<TensorId>> =
                RETAINED_INTERMEDIATE_GRAD_IDS
                    .lock()
                    .ok()
                    .and_then(|s| s.clone());
            for sub_entry in sub_tape.drain(..) {
                if let Some(sg) = sub_grads.take(sub_entry.output_id) {
                    if let Some(ref ids) = sub_retain_ids {
                        if ids.contains(&sub_entry.output_id) {
                            if let Ok(mut store) = RETAINED_INTERMEDIATE_GRADS.lock() {
                                store.insert(sub_entry.output_id, sg.clone());
                            }
                        }
                    }
                    let sub_tag = op_tag(&sub_entry.op);
                    let sub_t0 = std::time::Instant::now();
                    let input_grads = compute_gradients(&sub_entry, &sg, device)?;
                    let sub_dt = sub_t0.elapsed();
                    if let Some(ref mut times) = sub_op_time {
                        let e = times
                            .entry(sub_tag)
                            .or_insert((std::time::Duration::ZERO, 0));
                        e.0 += sub_dt;
                        e.1 += 1;
                    }
                    let accum_t0 = std::time::Instant::now();
                    for (tid, g) in input_grads {
                        if all_needed.contains(&tid) {
                            if g.dtype() != DType::F32 {
                                n_bf16_casts += 1;
                            }
                            sub_grads.accumulate(tid, g)?;
                        }
                    }
                    if sub_op_time.is_some() {
                        sub_accum_time += accum_t0.elapsed();
                    }
                }
                // sub_entry dropped → saved tensors freed → GPU memory reclaimed
            }
            drop(sub_tape);

            let _sub_bwd_dt = std::time::Instant::now() - _recomp_t0 - _recomp_dt;

            // ── Step 4: Flush allocation pool ──
            // The eager-free above dropped CudaSlices back into flame-core's
            // pool cache. Flush now so subsequent checkpoints can reuse VRAM.
            // 2026-05-15 diagnostic: `FLAME_SKIP_POOL_CLEAR=1` skips the
            // flush — tests whether the clear is the cuMemcpy2D INVALID_VALUE
            // trigger (per audit hypothesis H_CLEARCACHE).
            if std::env::var("FLAME_SKIP_POOL_CLEAR").as_deref() != Ok("1") {
                crate::cuda_alloc_pool::clear_pool_cache();
            }

            // Profiling
            static CKPT_PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let ckpt_profile = *CKPT_PROFILE.get_or_init(|| {
                std::env::var("FLAME_PROFILE")
                    .ok()
                    .map(|v| v == "1")
                    .unwrap_or(false)
            });
            if ckpt_profile {
                let total_dt = _ckpt_t0.elapsed();
                eprintln!(
                    "[checkpoint:{}] recomp={:.1}ms total={:.1}ms ({} entries, {} bf16_casts)",
                    entry.output_id.0,
                    _recomp_dt.as_secs_f64() * 1000.0,
                    total_dt.as_secs_f64() * 1000.0,
                    n_sub_entries,
                    n_bf16_casts
                );
                if let Some(times) = sub_op_time {
                    let mut rows: Vec<_> = times.into_iter().collect();
                    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
                    let mut msg = String::new();
                    for (idx, (tag, (dt, count))) in rows.into_iter().take(6).enumerate() {
                        if idx > 0 {
                            msg.push_str("; ");
                        }
                        let avg = dt.as_secs_f64() * 1000.0 / count as f64;
                        msg.push_str(&format!(
                            "{} {:.1}ms/{} avg {:.2}ms",
                            tag,
                            dt.as_secs_f64() * 1000.0,
                            count,
                            avg
                        ));
                    }
                    eprintln!(
                        "[checkpoint:{}:sub] accum={:.1}ms top={}",
                        entry.output_id.0,
                        sub_accum_time.as_secs_f64() * 1000.0,
                        msg
                    );
                }
            }

            // ── Step 5: Return only chain + trainable gradients ──
            // Intermediate chain gradients served their purpose during the
            // local backward and are NOT propagated to the outer graph.
            let mut result: GradVec = SmallVec::new();
            for (tid, g) in sub_grads.drain_all()? {
                if trainable_ids.contains(&tid) {
                    result.push((tid, g));
                }
            }
            Ok(result)
        }

        Op::CheckpointOffloadBoundary {
            input_ids,
            pulled_ids_slot,
        } => {
            // Same detach-recompute pattern as Op::Checkpoint above, but:
            //   - No strong tensor refs in saved_tensors (entry.saved_tensors
            //     is empty by construction).
            //   - Input IDs come from the Op variant, not saved_tensors.
            //   - The recompute closure (stored in checkpoint_fns) pulls
            //     inputs from the GrowOnDemandActivationCache internally
            //     before calling the user closure.
            let recompute_fn = {
                let ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.checkpoint_fns
                    .get(&entry.output_id)
                    .cloned()
                    .ok_or_else(|| {
                        Error::Training(
                            "CheckpointOffloadBoundary backward: no recompute closure".into(),
                        )
                    })?
            };

            // Recompute with autograd → local sub-tape.
            let tape_start = {
                let ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.tape.len()
            };
            {
                let mut ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.enabled = true;
                AUTOGRAD_ENABLED.store(true, Ordering::Relaxed);
            }
            let recomputed_output = {
                let _recompute_guard = CheckpointRecomputeGuard::enter();
                (recompute_fn)()?
            };
            let mut sub_tape: Vec<TapeEntry> = {
                let mut ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.enabled = false;
                AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);
                if ctx.tape.len() > tape_start {
                    ctx.tape.drain(tape_start..).collect()
                } else {
                    Vec::new()
                }
            };
            drop(recomputed_output);

            // Read the pulled-tensor IDs the recompute closure wrote into
            // the side-channel. These are the IDs the sub-tape uses for
            // its leaves; we need to remap their grads to `input_ids`.
            let pulled_ids: Vec<TensorId> = pulled_ids_slot
                .lock()
                .map_err(|_| Error::Training("pulled_ids_slot mutex poisoned".into()))?
                .clone();
            if pulled_ids.len() != input_ids.len() {
                return Err(Error::Training(format!(
                    "CheckpointOffloadBoundary: pulled_ids len {} != input_ids len {}",
                    pulled_ids.len(),
                    input_ids.len()
                )));
            }
            let pulled_to_original: std::collections::HashMap<TensorId, TensorId> = pulled_ids
                .iter()
                .zip(input_ids.iter())
                .map(|(p, o)| (*p, *o))
                .collect();

            // Build trainable/all-needed sets. Include BOTH the original
            // input IDs (so the caller's final lookup hits) AND the pulled
            // IDs (since the sub-tape's ops will accumulate grads under
            // those). Remapping happens at the end.
            let mut trainable_ids = std::collections::HashSet::new();
            let mut all_needed = std::collections::HashSet::new();
            for tid in input_ids {
                trainable_ids.insert(*tid);
                all_needed.insert(*tid);
            }
            for tid in &pulled_ids {
                trainable_ids.insert(*tid);
                all_needed.insert(*tid);
            }
            for e in &sub_tape {
                all_needed.insert(e.output_id);
                add_unsaved_view_inputs_to_needed(&e.op, &mut all_needed);
                for (sid, st) in &e.saved_tensors {
                    if st.requires_grad() {
                        trainable_ids.insert(*sid);
                        all_needed.insert(*sid);
                    }
                }
                for r in &e.saved_refs {
                    if r.tensor.requires_grad() {
                        trainable_ids.insert(r.id);
                        all_needed.insert(r.id);
                    }
                }
            }

            let sub_compact = {
                use crate::gradient::CompactIndex;
                let ids = sub_tape
                    .iter()
                    .flat_map(|e| {
                        let mut ids = vec![e.output_id];
                        for (sid, _) in &e.saved_tensors {
                            ids.push(*sid);
                        }
                        for r in &e.saved_refs {
                            ids.push(r.id);
                        }
                        ids
                    })
                    .chain(input_ids.iter().copied())
                    .chain(pulled_ids.iter().copied());
                CompactIndex::from_tensor_ids(ids)
            };
            let mut sub_grads =
                crate::gradient::GradientMap::with_index(device.clone(), sub_compact);

            if let Some(last_entry) = sub_tape.last() {
                sub_grads.set(last_entry.output_id, output_grad.clone());
            }

            sub_tape.reverse();
            // Soul.md trap support: snapshot the retain set NOW so the inner
            // walk can capture intermediate grads. The outer `backward()`
            // snapshot at line ~2044 only sees IDs registered BEFORE backward
            // started; checkpoint recompute creates fresh IDs that the
            // `retain_intermediate_grads_add` API registers DURING recompute,
            // so we re-read here.
            let sub_retain_ids: Option<std::collections::HashSet<TensorId>> =
                RETAINED_INTERMEDIATE_GRAD_IDS
                    .lock()
                    .ok()
                    .and_then(|s| s.clone());
            for sub_entry in sub_tape.drain(..) {
                if let Some(sg) = sub_grads.take(sub_entry.output_id) {
                    if let Some(ref ids) = sub_retain_ids {
                        if ids.contains(&sub_entry.output_id) {
                            if let Ok(mut store) = RETAINED_INTERMEDIATE_GRADS.lock() {
                                store.insert(sub_entry.output_id, sg.clone());
                            }
                        }
                    }
                    let input_grads = compute_gradients(&sub_entry, &sg, device)?;
                    for (tid, g) in input_grads {
                        if all_needed.contains(&tid) {
                            sub_grads.accumulate(tid, g)?;
                        }
                    }
                }
            }
            drop(sub_tape);
            crate::cuda_alloc_pool::clear_pool_cache();

            let mut result: GradVec = SmallVec::new();
            for (tid, g) in sub_grads.drain_all()? {
                // If this is a pulled-input grad, remap to the original
                // outer-graph input ID. Otherwise pass through unchanged
                // (param IDs, frozen weight IDs, etc.).
                let out_tid = pulled_to_original.get(&tid).copied().unwrap_or(tid);
                if trainable_ids.contains(&tid) || trainable_ids.contains(&out_tid) {
                    result.push((out_tid, g));
                }
            }
            Ok(result)
        }

        Op::CheckpointOffload { input, sub_tape } => {
            // Level 2: NO recompute. Walk the stored sub-tape, pulling
            // offloaded saved tensors from CPU as needed.
            let pool_arc = ACTIVATION_POOL
                .read()
                .ok()
                .and_then(|g| g.as_ref().cloned())
                .ok_or_else(|| {
                    Error::Training("CheckpointOffload backward: no offload pool set".into())
                })?;

            // Build the set of tensor IDs we need gradients for.
            let sub_needed: std::collections::HashSet<TensorId> = {
                let mut s = std::collections::HashSet::new();
                s.insert(*input);
                for oe in sub_tape {
                    s.insert(oe.output_id);
                    for sid in &oe.saved_ids {
                        s.insert(*sid);
                    }
                }
                s
            };

            // Seed the sub-gradient map with the incoming gradient.
            let mut sub_grads = crate::gradient::GradientMap::new(device.clone());
            if let Some(last_oe) = sub_tape.last() {
                sub_grads.set(last_oe.output_id, output_grad.clone());
            }

            // Walk the sub-tape in reverse (backward order). For each entry,
            // pull offloaded saved tensors from CPU, rebuild a TapeEntry,
            // and call compute_gradients.
            let mut pool = pool_arc
                .lock()
                .map_err(|_| Error::Training("offload pool mutex poisoned".into()))?;

            for oe in sub_tape.iter().rev() {
                if let Some(sg) = sub_grads.take(oe.output_id) {
                    // Rebuild saved_tensors by pulling from pool or using
                    // resident fallback.
                    let mut saved: SavedTensors = SmallVec::new();
                    for (i, sid) in oe.saved_ids.iter().enumerate() {
                        if let Some(Some(handle)) = oe.offload_handles.get(i) {
                            // Pull from CPU → fresh GPU tensor.
                            let pulled = pool.pull(*handle)?;
                            saved.push((*sid, pulled));
                        } else {
                            // Resident fallback (non-BF16 tensor).
                            if let Some((_, t)) =
                                oe.resident_fallback.iter().find(|(id, _)| id == sid)
                            {
                                saved.push((*sid, t.clone()));
                            }
                        }
                    }

                    // Build a temporary TapeEntry for compute_gradients.
                    let tmp_entry = TapeEntry {
                        output_id: oe.output_id,
                        op: oe.op.clone(),
                        saved_tensors: saved,
                        saved_refs: SavedRefs::new(),
                    };

                    let input_grads = compute_gradients(&tmp_entry, &sg, device)?;
                    for (tid, g) in input_grads {
                        if sub_needed.contains(&tid) {
                            sub_grads.accumulate(tid, g)?;
                        }
                    }
                }
            }
            drop(pool);

            let mut result: GradVec = SmallVec::new();
            for (tid, g) in sub_grads.drain_all()? {
                result.push((tid, g));
            }
            Ok(result)
        }

        Op::FusedSwiGLU { gate, up } => {
            // SwiGLU forward: out = silu(gate) * up
            // Fused backward kernel: 1 kernel computes both d_gate and d_up.
            let gate_tensor = fetch_saved(gate)?;
            let up_tensor = fetch_saved(up)?;
            let n = gate_tensor.shape().elem_count() as i64;
            let stream = device.cuda_stream_raw_ptr();

            if output_grad.dtype() == DType::BF16
                && gate_tensor.dtype() == DType::BF16
                && up_tensor.dtype() == DType::BF16
            {
                let mut d_gate_t =
                    Tensor::empty_dtype(gate_tensor.shape().clone(), DType::BF16, device.clone())?;
                let mut d_up_t =
                    Tensor::empty_dtype(up_tensor.shape().clone(), DType::BF16, device.clone())?;
                let status = unsafe {
                    crate::cuda::ffi::flame_swiglu_backward_bf16(
                        tensor_raw_ptr(output_grad)?,
                        tensor_raw_ptr(&gate_tensor)?,
                        tensor_raw_ptr(&up_tensor)?,
                        tensor_raw_ptr_mut(&mut d_gate_t)?,
                        tensor_raw_ptr_mut(&mut d_up_t)?,
                        n,
                        stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda("flame_swiglu_backward_bf16 failed".into()));
                }
                Ok(smallvec![(*gate, d_gate_t), (*up, d_up_t)])
            } else {
                // Fallback: decompose into individual ops
                let og = if output_grad.dtype() != DType::BF16 {
                    output_grad.to_dtype_no_grad(DType::BF16)?
                } else {
                    output_grad.clone()
                };
                let gt = if gate_tensor.dtype() != DType::BF16 {
                    gate_tensor.to_dtype_no_grad(DType::BF16)?
                } else {
                    gate_tensor
                };
                let ut = if up_tensor.dtype() != DType::BF16 {
                    up_tensor.to_dtype_no_grad(DType::BF16)?
                } else {
                    up_tensor
                };
                let mut d_gate_t =
                    Tensor::empty_dtype(gt.shape().clone(), DType::BF16, device.clone())?;
                let mut d_up_t =
                    Tensor::empty_dtype(ut.shape().clone(), DType::BF16, device.clone())?;
                let status = unsafe {
                    crate::cuda::ffi::flame_swiglu_backward_bf16(
                        tensor_raw_ptr(&og)?,
                        tensor_raw_ptr(&gt)?,
                        tensor_raw_ptr(&ut)?,
                        tensor_raw_ptr_mut(&mut d_gate_t)?,
                        tensor_raw_ptr_mut(&mut d_up_t)?,
                        gt.shape().elem_count() as i64,
                        stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda(
                        "flame_swiglu_backward_bf16 (fallback) failed".into(),
                    ));
                }
                Ok(smallvec![(*gate, d_gate_t), (*up, d_up_t)])
            }
        }

        Op::FusedSwiGLUSplit { input } => {
            let input_tensor = fetch_saved(input)?;
            if input_tensor.dtype() != DType::BF16 {
                return Err(Error::InvalidOperation(
                    "FusedSwiGLUSplit backward expects BF16 input".into(),
                ));
            }

            let dims = input_tensor.shape().dims();
            let ndim = dims.len();
            if ndim == 0 {
                return Err(Error::InvalidOperation(
                    "FusedSwiGLUSplit backward expects rank >= 1".into(),
                ));
            }
            let last = dims[ndim - 1];
            if last % 2 != 0 {
                return Err(Error::InvalidOperation(format!(
                    "FusedSwiGLUSplit backward last dim {last} is not even"
                )));
            }
            let half = last / 2;
            let out_n = output_grad.shape().elem_count();
            if out_n * 2 != input_tensor.shape().elem_count() {
                return Err(Error::ShapeMismatch {
                    expected: Shape::from_dims(&{
                        let mut out_dims = dims.to_vec();
                        out_dims[ndim - 1] = half;
                        out_dims
                    }),
                    got: output_grad.shape().clone(),
                });
            }

            let output_grad_bf16;
            let grad_src = if output_grad.dtype() == DType::BF16 {
                output_grad
            } else {
                output_grad_bf16 = output_grad.to_dtype_no_grad(DType::BF16)?;
                &output_grad_bf16
            };

            let mut meta: Vec<i64> = input_tensor.strides().iter().map(|&s| s as i64).collect();
            meta.extend(dims.iter().map(|&d| d as i64));
            let mut d_meta: cudarc::driver::CudaSlice<i64> = unsafe { device.alloc(meta.len()) }
                .map_err(|e| Error::Cuda(format!("alloc swiglu split meta: {:?}", e)))?;
            device
                .htod_copy_into(meta, &mut d_meta)
                .map_err(|e| Error::Cuda(format!("htod swiglu split meta: {:?}", e)))?;
            use cudarc::driver::DevicePtr;

            let mut grad_input =
                Tensor::empty_dtype(input_tensor.shape().clone(), DType::BF16, device.clone())?;
            let status = unsafe {
                crate::cuda::ffi::flame_swiglu_split_backward_bf16(
                    tensor_raw_ptr(grad_src)?,
                    tensor_raw_ptr(&input_tensor)?,
                    tensor_raw_ptr_mut(&mut grad_input)?,
                    *d_meta.device_ptr() as *const i64,
                    input_tensor.offset() as i64,
                    ndim as i32,
                    out_n as i64,
                    half as i64,
                    device.cuda_stream_raw_ptr(),
                )
            };
            if status != 0 {
                return Err(Error::Cuda(
                    "flame_swiglu_split_backward_bf16 failed".into(),
                ));
            }
            Ok(smallvec![(*input, grad_input)])
        }

        Op::QkvSplitPermute {
            input,
            part,
            heads,
            head_dim,
        } => {
            let input_tensor = fetch_saved(input)?;
            let dims = input_tensor.shape().dims();
            if dims.len() != 3 {
                return Err(Error::InvalidOperation(format!(
                    "QkvSplitPermute backward expects input [B,N,3*H*D], got {:?}",
                    dims
                )));
            }
            let (b, n, c) = (dims[0], dims[1], dims[2]);
            let hd = *heads * *head_dim;
            if c != 3 * hd {
                return Err(Error::InvalidOperation(format!(
                    "QkvSplitPermute backward input last dim {c} != 3*{heads}*{head_dim}"
                )));
            }
            if *part > 2 {
                return Err(Error::InvalidOperation(format!(
                    "QkvSplitPermute backward invalid part {part}"
                )));
            }
            let og_dims = output_grad.shape().dims();
            if og_dims != &[b, *heads, n, *head_dim] {
                return Err(Error::ShapeMismatch {
                    expected: Shape::from_dims(&[b, *heads, n, *head_dim]),
                    got: output_grad.shape().clone(),
                });
            }

            let output_grad_f32;
            let (grad_src, grad_dtype) = match output_grad.dtype() {
                DType::F32 => (output_grad, DType::F32),
                DType::BF16 => (output_grad, DType::BF16),
                _ => {
                    output_grad_f32 = output_grad.to_dtype_no_grad(DType::F32)?;
                    (&output_grad_f32, DType::F32)
                }
            };

            let mut grad_strides: Vec<i64> =
                grad_src.strides().iter().map(|&s| s as i64).collect();
            if grad_strides.len() != 4 {
                return Err(Error::InvalidOperation(format!(
                    "QkvSplitPermute backward expected rank-4 grad strides, got {:?}",
                    grad_strides
                )));
            }
            let mut d_strides: cudarc::driver::CudaSlice<i64> =
                unsafe { device.alloc(grad_strides.len()) }
                    .map_err(|e| Error::Cuda(format!("alloc qkv split grad strides: {:?}", e)))?;
            device
                .htod_copy_into(std::mem::take(&mut grad_strides), &mut d_strides)
                .map_err(|e| Error::Cuda(format!("htod qkv split grad strides: {:?}", e)))?;
            use cudarc::driver::DevicePtr;

            let mut grad_input =
                Tensor::empty_dtype(input_tensor.shape().clone(), grad_dtype, device.clone())?;
            let status = unsafe {
                match grad_dtype {
                    DType::BF16 => crate::cuda::ffi::flame_qkv_split_permute_backward_bf16(
                        tensor_raw_ptr(grad_src)?,
                        tensor_raw_ptr_mut(&mut grad_input)?,
                        *d_strides.device_ptr() as *const i64,
                        grad_src.offset() as i64,
                        *part as i32,
                        b as i64,
                        *heads as i64,
                        n as i64,
                        *head_dim as i64,
                        device.cuda_stream_raw_ptr(),
                    ),
                    DType::F32 => crate::cuda::ffi::flame_qkv_split_permute_backward_f32(
                        tensor_raw_ptr(grad_src)?,
                        tensor_raw_ptr_mut(&mut grad_input)?,
                        *d_strides.device_ptr() as *const i64,
                        grad_src.offset() as i64,
                        *part as i32,
                        b as i64,
                        *heads as i64,
                        n as i64,
                        *head_dim as i64,
                        device.cuda_stream_raw_ptr(),
                    ),
                    _ => unreachable!(),
                }
            };
            if status != 0 {
                return Err(Error::Cuda(
                    "flame_qkv_split_permute_backward failed".into(),
                ));
            }
            Ok(smallvec![(*input, grad_input)])
        }

        Op::GateResidual { residual, gate, x } => {
            if std::env::var("DEBUG_GATE_RESIDUAL").ok().as_deref() == Some("1") {
                eprintln!("[gate_residual::BACKWARD] hit");
            }
            // forward: out = residual + gate.unsqueeze(1) * x
            //   residual,x ∈ [B,N,dim], gate ∈ [B,dim], out ∈ [B,N,dim]
            // backward:
            //   grad_residual = grad_out
            //   grad_x        = grad_out * gate.unsqueeze(1)
            //   grad_gate     = (grad_out * x).sum(dim=1)
            let x_tensor = fetch_saved(x)?;
            let gate_tensor = fetch_saved(gate)?;
            let dims = x_tensor.shape().dims().to_vec();
            let (b, _n, dim) = (dims[0], dims[1], dims[2]);

            // grad_residual = grad_out (identity)
            let grad_residual = output_grad.clone();

            // grad_x = grad_out * gate.unsqueeze(1) — relies on GpuOps::mul's
            // broadcast then reduces if shapes drift.
            let gate_3d = gate_tensor.reshape(&[b, 1, dim])?;
            let grad_x_full = GpuOps::mul(output_grad, &gate_3d)?;
            let grad_x = if grad_x_full.shape() != x_tensor.shape() {
                reduce_grad_for_broadcast(&grad_x_full, x_tensor.shape())?
            } else {
                grad_x_full
            };

            // grad_gate = sum over N of (grad_out * x) → [B, dim]
            // reduce_grad_for_broadcast doesn't handle this layout
            // ([B,N,dim] → [B,dim] needs sum on axis 1, not axis 0); do it
            // explicitly by sum_dim_keepdim then reshape.
            let grad_gate_full = GpuOps::mul(output_grad, &x_tensor)?;
            let grad_gate_3d = grad_gate_full.sum_dim_keepdim(1)?; // [B, 1, dim]
            let grad_gate = grad_gate_3d.reshape(&[b, dim])?;

            Ok(smallvec![
                (*residual, ensure_bf16(grad_residual)?),
                (*gate, ensure_bf16(grad_gate)?),
                (*x, ensure_bf16(grad_x)?),
            ])
        }

        Op::RoPePrecomputed { input, cos, sin, layout } => {
            // RoPE is an orthogonal rotation; backward = forward with -sin
            // applied via the SAME layout the forward used. The layout tag
            // is carried alongside the op (set by the forward call site),
            // not shape-sniffed from cos — the previous shape-sniff
            // incorrectly classified HiDream-O1's MRoPE cos `[1, S, half]`
            // as Interleaved while the forward was Halfsplit, collapsing
            // Q/K LoRA-B gradients to a near-random direction.
            // See [`RopeLayout`] doc.
            let grad_bf16 = if output_grad.dtype() != DType::BF16 {
                output_grad.to_dtype_no_grad(DType::BF16)?
            } else {
                output_grad.clone()
            };
            let cos_tensor = fetch_saved(cos)?;
            let sin_tensor = fetch_saved(sin)?;
            let neg_sin = GpuOps::mul_scalar(&sin_tensor, -1.0)?;
            let grad_input = match layout {
                RopeLayout::Interleaved => {
                    // F32-pe forward (rope_fused_bf16_f32pe) saves F32 cos/sin; the
                    // BF16-only kernel would reject them. Use the matching f32pe
                    // backward when cos is F32 (also keeps F32 RoPE precision in bwd).
                    if cos_tensor.dtype() == DType::F32 {
                        crate::bf16_ops::rope_fused_bf16_f32pe(&grad_bf16, &cos_tensor, &neg_sin)?
                    } else {
                        crate::bf16_ops::rope_fused_bf16(&grad_bf16, &cos_tensor, &neg_sin)?
                    }
                }
                RopeLayout::Halfsplit => {
                    crate::bf16_ops::rope_halfsplit_bf16(&grad_bf16, &cos_tensor, &neg_sin)?
                }
                RopeLayout::HalfsplitPytorch => {
                    crate::bf16_ops::rope_halfsplit_bf16_pytorch(&grad_bf16, &cos_tensor, &neg_sin)?
                }
            };
            Ok(smallvec![(*input, grad_input)])
        }

        Op::Mean { input, input_shape } => {
            // d/dx mean(x) = 1/n for each element
            let n = input_shape.elem_count() as f32;
            let grad_scaled = GpuOps::mul_scalar(output_grad, 1.0 / n)?;
            // Normalize upstream rank before GPU broadcast
            let up_ranked = expand_to_rank(&grad_scaled, input_shape.dims().len())?;
            let expanded = GpuOps::broadcast(&up_ranked, input_shape)?;
            Ok(smallvec![(*input, expanded)])
        }

        Op::Transpose { input } => {
            let grad = if output_grad.dtype() == DType::BF16 {
                crate::bf16_elementwise::transpose2d_bf16(output_grad)?
            } else {
                GpuOps::transpose(output_grad)?
            };
            Ok(smallvec![(*input, grad)])
        }

        Op::Conv2d {
            input,
            weight,
            stride,
            padding,
        } => {
            // Use CUDA Conv2D backward
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let weight_tensor = entry
                .get_saved(weight)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for weight".into()))?;

            // cuda_conv2d::conv2d_backward is F32-only (training-path kernel).
            // Cast BF16 inputs/grads to F32 here; the resulting F32 grads are
            // accepted by accumulate_parameter_grads and the per-tensor
            // gradient store.
            let go_f32 = if output_grad.dtype() == DType::F32 {
                output_grad.clone()
            } else {
                output_grad.to_dtype(DType::F32)?
            };
            let in_f32 = if input_tensor.dtype() == DType::F32 {
                input_tensor.clone()
            } else {
                input_tensor.to_dtype(DType::F32)?
            };
            let w_f32 = if weight_tensor.dtype() == DType::F32 {
                weight_tensor.clone()
            } else {
                weight_tensor.to_dtype(DType::F32)?
            };

            let (grad_input, grad_weight, grad_bias) =
                crate::cuda_conv2d::CudaConv2d::conv2d_backward(
                    &go_f32,
                    &in_f32,
                    &w_f32,
                    (*stride, *stride),
                    (*padding, *padding),
                )?;

            let mut grads: GradVec = smallvec![(*input, grad_input), (*weight, grad_weight)];

            // Handle bias gradient if present
            if let Some(grad_bias) = grad_bias {
                // Check if bias was saved in the tape entry
                // The bias would be the third saved tensor if it exists
                if entry.saved_count() > 2 {
                    // Get the bias tensor ID from the saved tensors
                    let bias_id = entry
                        .saved_keys()
                        .find(|&id| id != input && id != weight)
                        .copied();

                    if let Some(bias_id) = bias_id {
                        grads.push((bias_id, grad_bias));
                    }
                }
            }

            Ok(grads)
        }

        Op::Conv2dNHWC {
            input,
            weight,
            stride,
            padding,
        } => {
            // Saved tensors for NHWC path should include NCHW input and OC,IC,KH,KW weight
            let input_nchw = entry.get_saved(input).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for input(NCHW)".into())
            })?;
            let weight_ocic = entry.get_saved(weight).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for weight(OC,IC,KH,KW)".into())
            })?;

            assert_nhwc_bf16_public("AutogradContext::conv2d_backward grad_out", output_grad)?;
            let grad_out_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(output_grad)?;

            let (grad_in_nchw, grad_w_ocic, grad_b) =
                crate::cuda_conv2d::CudaConv2d::conv2d_backward(
                    &grad_out_nchw,
                    input_nchw,
                    weight_ocic,
                    (*stride, *stride),
                    (*padding, *padding),
                )?;

            // Convert grads back to NHWC / [KH,KW,IC,OC]
            let grad_input = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&grad_in_nchw)?;
            let grad_input = ensure_bf16(grad_input)?;
            let grad_weight = crate::cuda_ops::GpuOps::weight_ocickhkw_to_khwkicoc(&grad_w_ocic)?;
            let grad_weight = ensure_bf16(grad_weight)?;

            let mut grads: GradVec = smallvec![(*input, grad_input), (*weight, grad_weight)];
            if let Some(gb) = grad_b {
                let gb = ensure_bf16(gb)?;
                grads.push((
                    entry
                        .saved_keys()
                        .copied()
                        .find(|&k| k != *input && k != *weight)
                        .unwrap_or(*weight),
                    gb,
                ));
            }
            Ok(grads)
        }

        Op::LayerNorm {
            input,
            normalized_shape,
        } => {
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            guard_tensor(
                "AutogradContext::layer_norm_backward saved input",
                input_tensor,
            )?;

            // Always take the BF16 fast kernel. The incoming grad arrives as
            // F32 (gradient::accumulate upcasts), so cast it back to BF16 —
            // the fused kernel does F32 math internally for stability.
            let grad_bf16_owned;
            let grad_bf16: &Tensor = if output_grad.dtype() == DType::BF16 {
                output_grad
            } else {
                grad_bf16_owned = output_grad.to_dtype_no_grad(DType::BF16)?;
                &grad_bf16_owned
            };
            let input_bf16_owned;
            let input_bf16: &Tensor = if input_tensor.dtype() == DType::BF16 {
                input_tensor
            } else {
                input_bf16_owned = input_tensor.to_dtype_no_grad(DType::BF16)?;
                &input_bf16_owned
            };
            // Same contiguity fix as the layer_norm forward helper: the bwd
            // kernel reads the device pointer as a contiguous `[batch *
            // norm_size]` buffer. Saved tensors from strided callers (narrow,
            // permute) must be contiguified or the gradient is scrambled.
            let input_bf16_contig;
            let input_bf16 = if input_bf16.is_contiguous() {
                input_bf16
            } else {
                input_bf16_contig = input_bf16.contiguous()?;
                &input_bf16_contig
            };
            let grad_bf16_contig;
            let grad_bf16 = if grad_bf16.is_contiguous() {
                grad_bf16
            } else {
                grad_bf16_contig = grad_bf16.contiguous()?;
                &grad_bf16_contig
            };
            // Extract weight and bias from saved_tensors if present.
            // Forward saves: [input, weight?, bias?, mean, rstd]
            // Weight is at index 1 if it was saved (affine LayerNorm).
            // Bias is at index 2 if both weight and bias were saved.
            let weight_tensor = if entry.saved_count() > 3 {
                // 5 entries: input, weight, bias, mean, rstd
                // OR 4 entries: input, weight, mean, rstd (weight-only, no bias)
                entry.saved_at(1).map(|(_, t)| t)
            } else {
                None
            };
            let bias_tensor = if entry.saved_count() > 4 {
                // 5 entries: input, weight, bias, mean, rstd
                entry.saved_at(2).map(|(_, t)| t)
            } else {
                None
            };

            let w_bf16_owned;
            let w_bf16: Option<&Tensor> = match weight_tensor {
                Some(w) if w.dtype() == DType::BF16 => Some(w),
                Some(w) => {
                    w_bf16_owned = w.to_dtype_no_grad(DType::BF16)?;
                    Some(&w_bf16_owned)
                }
                None => None,
            };
            let b_bf16_owned;
            let b_bf16: Option<&Tensor> = match bias_tensor {
                Some(b) if b.dtype() == DType::BF16 => Some(b),
                Some(b) => {
                    b_bf16_owned = b.to_dtype_no_grad(DType::BF16)?;
                    Some(&b_bf16_owned)
                }
                None => None,
            };

            // ---- F32 non-affine LayerNorm backward (precision fix) ----
            // The BF16 kernel below receives the grad already cast down to
            // BF16 (grad_bf16), then computes the cancellation
            //   dL/dx = rstd * (g - mean(g) - xhat * mean(g*xhat))
            // whose result is only the ~1/sigma survivor. On BF16-rounded
            // input that cancellation amplifies rounding ~sigma x and drops
            // the transform-branch gradient — PROVEN to rotate klein's
            // per-block backward to cos 0.51-0.68 while the forward is cos
            // 0.99. Recompute in F32 from the saved F32 mean/rstd. Non-affine
            // only (modulate_pre uses weight=None); affine LN keeps the kernel.
            if w_bf16.is_none()
                && (std::env::var("FLAME_LN_BWD_F32").as_deref() == Ok("1")
                    || std::env::var("FLAME_BWD_F32").as_deref() == Ok("1"))
            {
                let sc = entry.saved_count();
                if let (Some((_, mean_t)), Some((_, rstd_t))) =
                    (entry.saved_at(sc - 2), entry.saved_at(sc - 1))
                {
                    let dims = input_bf16.shape().dims().to_vec();
                    let last = dims.len() - 1;
                    let d = dims[last] as f32;
                    let mut bshape = dims.clone();
                    bshape[last] = 1;
                    let x = input_bf16.to_dtype_no_grad(DType::F32)?;
                    let g_owned;
                    let g: &Tensor = if output_grad.dtype() == DType::F32 {
                        output_grad
                    } else {
                        g_owned = output_grad.to_dtype_no_grad(DType::F32)?;
                        &g_owned
                    };
                    let g = if g.is_contiguous() { g.clone() } else { g.contiguous()? };
                    let mean = mean_t.to_dtype_no_grad(DType::F32)?.reshape(&bshape)?;
                    let rstd = rstd_t.to_dtype_no_grad(DType::F32)?.reshape(&bshape)?;
                    let xhat = x.sub(&mean)?.mul(&rstd)?;
                    let g_mean = GpuOps::sum_dim_keepdim(&g, last)?.mul_scalar(1.0 / d)?;
                    let g_xhat = GpuOps::sum_dim_keepdim(&g.mul(&xhat)?, last)?
                        .mul_scalar(1.0 / d)?;
                    let dx = g
                        .sub(&g_mean)?
                        .sub(&xhat.mul(&g_xhat)?)?
                        .mul(&rstd)?;
                    return Ok(smallvec![(*input, dx)]);
                }
            }

            let (grad_input, grad_weight, grad_bias) =
                crate::cuda_ops_bf16::layer_norm_backward_bf16(
                    input_bf16,
                    grad_bf16,
                    w_bf16,
                    b_bf16,
                    normalized_shape,
                    1e-5, // eps (match forward)
                )?;

            let grad_input = ensure_bf16(grad_input)?;
            let mut gradients: GradVec = smallvec![(*input, grad_input)];

            // Add weight and bias gradients if they exist
            if let Some(grad_w) = grad_weight {
                // For LayerNorm with affine parameters, weight and bias are separate tensors
                // Find them in saved_tensors (they would be saved after the input tensor)
                if entry.saved_count() > 1 {
                    // Second tensor is weight
                    if let Some((wid, _)) = entry.saved_at(1) {
                        gradients.push((wid, ensure_bf16(grad_w)?));
                    }
                }
            }
            if let Some(grad_b) = grad_bias {
                // For LayerNorm with affine parameters, bias is the third tensor
                if entry.saved_count() > 2 {
                    // Third tensor is bias
                    if let Some((bid, _)) = entry.saved_at(2) {
                        gradients.push((bid, ensure_bf16(grad_b)?));
                    }
                }
            }

            Ok(gradients)
        }

        Op::RMSNorm {
            input,
            weight,
            eps: _,
            inv_rms,
            normalized_shape,
        } => {
            // Route through fetch_saved so saved strided views are
            // materialized — rms_norm_backward reads inputs via raw
            // pointers that ignore custom_strides/view_offset (Bug #4).
            let input_tensor = fetch_saved(input)?;
            guard_tensor("AutogradContext::rmsnorm_backward input", &input_tensor)?;

            let inv_rms_tensor = fetch_saved(inv_rms)?;

            let weight_tensor = match weight.as_ref() {
                Some(w_id) => Some(fetch_saved(w_id)?),
                None => None,
            };
            guard_optional_tensor(
                "AutogradContext::rmsnorm_backward weight",
                weight_tensor.as_ref(),
            )?;
            let need_weight_grad = weight_tensor
                .as_ref()
                .map(|w| w.requires_grad())
                .unwrap_or(false);

            let (grad_input, grad_weight) = crate::norm::rms_norm_backward(
                output_grad,
                &input_tensor,
                weight_tensor.as_ref(),
                &inv_rms_tensor,
                normalized_shape,
                need_weight_grad,
            )?;

            let mut grads: GradVec = smallvec![(*input, ensure_bf16(grad_input)?)];

            if let Some(&w_id) = weight.as_ref() {
                if need_weight_grad {
                    let gw = grad_weight.ok_or_else(|| {
                        Error::InvalidOperation("RMSNorm backward missing weight grad".into())
                    })?;
                    grads.push((w_id, ensure_bf16(gw)?));
                }
            }

            Ok(grads)
        }

        Op::Linear {
            input,
            weight,
            bias,
        } => {
            // Forward: output = input @ weight^T + bias
            //   input:  [..., in_features]
            //   weight: [out_features, in_features]
            //   output: [..., out_features]
            // Backward:
            //   grad_input  = grad_output @ weight                  (no transposes)
            //   grad_weight = grad_output^T @ input                 (already [out, in])
            //   grad_bias   = sum over all dims except last
            //
            // Matches Op::BatchMatMul path: one fused cuBLASLt call per grad
            // via `matmul_bf16_trans` — no materialized transposes.
            let input_tensor = &fetch_saved(input)?;
            let weight_tensor = &fetch_saved(weight)?;
            guard_tensor("AutogradContext::linear_backward input", input_tensor)?;
            guard_tensor("AutogradContext::linear_backward weight", weight_tensor)?;

            let input_shape = input_tensor.shape().dims().to_vec();
            let grad_shape = output_grad.shape().dims().to_vec();
            if input_shape.is_empty() || grad_shape.is_empty() {
                return Err(Error::InvalidOperation(
                    "Op::Linear backward: scalar tensors not supported".into(),
                ));
            }
            let in_features = input_shape[input_shape.len() - 1];
            let out_features = grad_shape[grad_shape.len() - 1];
            let batch: usize = input_shape[..input_shape.len() - 1].iter().product();
            // Sanity: leading-dim product must match between input and grad.
            let grad_batch: usize = grad_shape[..grad_shape.len() - 1].iter().product();
            if batch != grad_batch {
                return Err(Error::InvalidOperation(format!(
                    "Op::Linear backward: leading-dim mismatch input {:?} vs grad {:?}",
                    input_shape, grad_shape
                )));
            }

            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            let (grad_input, grad_weight) = if input_tensor.dtype() == DType::BF16
                && weight_tensor.dtype() == DType::BF16
                && output_grad.dtype() == DType::BF16
            {
                let input_2d = input_tensor.reshape(&[batch, in_features])?;
                let grad_2d = output_grad.reshape(&[batch, out_features])?;
                let weight_2d = weight_tensor.reshape(&[out_features, in_features])?;

                // grad_input = grad @ weight (both non-transposed) → [batch, in]
                let grad_input_2d =
                    crate::ops::gemm_bf16::matmul_bf16_trans(&grad_2d, &weight_2d, false, false)?;
                // grad_weight = grad^T @ input → [out, in] directly (no final transpose)
                let grad_weight_2d =
                    crate::ops::gemm_bf16::matmul_bf16_trans(&grad_2d, &input_2d, true, false)?;
                let grad_input = grad_input_2d.reshape(&input_shape)?;
                (grad_input, grad_weight_2d)
            } else {
                // Fallback for non-BF16 saved tensors: cast to BF16 and reuse the
                // fast path, matching Op::BatchMatMul's behavior at line ~2591.
                let input_bf16 = input_tensor.to_dtype_no_grad(DType::BF16)?;
                let weight_bf16 = weight_tensor.to_dtype_no_grad(DType::BF16)?;
                let grad_bf16 = output_grad.to_dtype_no_grad(DType::BF16)?;
                let input_2d = input_bf16.reshape(&[batch, in_features])?;
                let grad_2d = grad_bf16.reshape(&[batch, out_features])?;
                let weight_2d = weight_bf16.reshape(&[out_features, in_features])?;
                let grad_input_2d =
                    crate::ops::gemm_bf16::matmul_bf16_trans(&grad_2d, &weight_2d, false, false)?;
                let grad_weight_2d =
                    crate::ops::gemm_bf16::matmul_bf16_trans(&grad_2d, &input_2d, true, false)?;
                let grad_input = grad_input_2d.reshape(&input_shape)?;
                (grad_input, grad_weight_2d)
            };

            #[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
            let (grad_input, grad_weight) = {
                // No fused BF16 path available — fall back to materialized transposes.
                let weight_t = weight_tensor.transpose()?;
                let grad_input = output_grad.matmul(&weight_t)?;
                let input_t = input_tensor.transpose()?;
                let grad_weight = output_grad.transpose()?.matmul(&input_t)?.transpose()?;
                (grad_input, grad_weight)
            };

            let grad_input = ensure_bf16(grad_input)?;
            let grad_weight = ensure_bf16(grad_weight)?;

            let mut grads: GradVec = smallvec![(*input, grad_input), (*weight, grad_weight)];

            // Gradient w.r.t. bias (if present)
            if let Some(bias_id) = bias {
                // Sum over all dimensions except the last (features)
                let grad_bias = output_grad
                    .sum_dims(&(0..output_grad.shape().dims().len() - 1).collect::<Vec<_>>())?;
                let grad_bias = ensure_bf16(grad_bias)?;
                grads.push((*bias_id, grad_bias));
            }

            Ok(grads)
        }

        Op::BatchMatMul { lhs, rhs } => {
            let lhs_tensor = &fetch_saved(lhs)?;
            let rhs_tensor = &fetch_saved(rhs)?;
            let need_lhs_grad = lhs_tensor.requires_grad();
            let need_rhs_grad = rhs_tensor.requires_grad();
            if !need_lhs_grad && !need_rhs_grad {
                return Ok(SmallVec::new());
            }

            // Fast path: BF16 3D. Single cuBLASLt strided-batched call per
            // grad, no materialized transposes. Used when Op::BatchMatMul
            // appears on the outer tape (most paths now use bmm inline in
            // attention_backward_recompute, which has its own fast path).
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            if lhs_tensor.dtype() == DType::BF16
                && rhs_tensor.dtype() == DType::BF16
                && lhs_tensor.rank() == 3
                && rhs_tensor.rank() == 3
                && output_grad.rank() == 3
            {
                let grad_bf16_owned;
                let grad_bf16: &Tensor = if output_grad.dtype() == DType::BF16 {
                    output_grad
                } else {
                    grad_bf16_owned = output_grad.to_dtype_no_grad(DType::BF16)?;
                    &grad_bf16_owned
                };
                let mut grads: GradVec = SmallVec::new();
                if need_lhs_grad {
                    let grad_lhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                        grad_bf16, rhs_tensor, false, true,
                    )?;
                    grads.push((*lhs, grad_lhs));
                }
                if need_rhs_grad {
                    let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                        lhs_tensor, grad_bf16, true, false,
                    )?;
                    grads.push((*rhs, grad_rhs));
                }
                return Ok(grads);
            }

            // Fallback for non-BF16 3D×3D: cast to BF16, use the fast path above.
            // This is rare — only hits when operands are F32/F16.
            let lhs_bf16 = lhs_tensor.to_dtype_no_grad(DType::BF16)?;
            let rhs_bf16 = rhs_tensor.to_dtype_no_grad(DType::BF16)?;
            let grad_bf16 = output_grad.to_dtype_no_grad(DType::BF16)?;
            let mut grads: GradVec = SmallVec::new();
            if need_lhs_grad {
                let grad_lhs =
                    crate::ops::gemm_bf16::matmul_bf16_trans(&grad_bf16, &rhs_bf16, false, true)?;
                grads.push((*lhs, grad_lhs));
            }
            if need_rhs_grad {
                let grad_rhs =
                    crate::ops::gemm_bf16::matmul_bf16_trans(&lhs_bf16, &grad_bf16, true, false)?;
                grads.push((*rhs, grad_rhs));
            }
            Ok(grads)
        }

        Op::Reshape { input, .. } => {
            // Gradient of reshape is reshape of gradient back to original shape
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let grad = output_grad.reshape(input_tensor.shape().dims())?;
            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::Broadcast { input, src_shape, dst_shape } => {
            // dInput/dOutput = sum(output_grad) along every broadcast axis.
            // Two kinds of broadcast axes:
            //   1. Left-padded axes: dst has rank > src. Sum (collapse) every
            //      left-padded dim out.
            //   2. Same-rank broadcasts: dst[d] > 1 where src[d] == 1.
            //      sum_dim_keepdim along d so the dim becomes 1.
            // Then reshape to src_shape.
            //
            // Implementation uses GpuOps::sum_dim_keepdim directly (no
            // autograd recording inside the backward handler — we're already
            // in the tape walk).
            let pad = dst_shape.len().saturating_sub(src_shape.len());
            let mut current = output_grad.clone();
            // Walk dst axes in DESCENDING order so each sum-keepdim doesn't
            // shift the indices of the remaining axes.
            for d in (0..dst_shape.len()).rev() {
                let must_reduce = if d < pad {
                    true
                } else {
                    let src_d = d - pad;
                    src_shape[src_d] == 1 && dst_shape[d] > 1
                };
                if must_reduce {
                    current = crate::cuda_ops::GpuOps::sum_dim_keepdim(&current, d)?;
                }
            }
            // current.shape is now src_shape left-padded to dst rank with 1s.
            // Reshape back to src_shape (drops the padded 1s).
            let grad = if current.shape().dims() != &src_shape[..] {
                current.reshape(src_shape)?
            } else {
                current
            };
            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::Permute { input, dims } => {
            // Gradient of permute is inverse permute.
            // FIX 2026-05-29: `.permute()` returns a STRIDED VIEW. Handing that
            // strided grad to a downstream view-op backward (e.g. a mid-dim
            // narrow's scatter) makes it misread against contiguous strides —
            // proven by the adjoint test: `narrow(mid) -> permute` was rel 1.97
            // (sign-flipped), while `narrow -> contiguous` is rel 3e-7. Backward
            // grads must be dense; materialize before propagating.
            let inverse_dims = inverse_permutation(dims);
            let grad = output_grad.permute(&inverse_dims)?.contiguous()?;
            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::UpsampleNearest2D {
            input,
            input_h,
            input_w,
            output_h,
            output_w,
        } => {
            // Backward kernel sums grad_output over the source block of each
            // input pixel (the inverse of nearest-neighbor "duplicate" forward).
            let grad = crate::cuda_ops::GpuOps::upsample2d_nearest_backward(
                output_grad,
                (*input_h, *input_w),
                (*output_h, *output_w),
            )?;
            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::MaxPool2D {
            input,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        } => {
            // The forward kernel doesn't emit indices, so the backward
            // re-derives the argmax from the saved input. The MaxPool2d
            // backward method ignores its `indices` arg today
            // (`pooling.rs:_indices: Option<&Tensor>`). Both the saved
            // input and the upstream grad must be BF16-NHWC because
            // `MaxPool2d::backward` enforces `assert_nhwc_bf16_public`
            // on its inputs — backward grads from the autograd engine
            // arrive in F32 by default, so we cast first.
            let input_tensor = fetch_saved(input)?;
            debug_assert_eq!(
                input_tensor.dtype(),
                DType::BF16,
                "Op::MaxPool2D saved input must be BF16 — forward asserted this contract"
            );
            let input_bf16 = input_tensor;
            let grad_bf16 = if output_grad.dtype() == DType::BF16 {
                output_grad.clone()
            } else {
                output_grad.to_dtype(DType::BF16)?
            };
            let mut cfg = crate::pooling::MaxPool2dConfig::new((*kernel_h, *kernel_w));
            cfg.stride = Some((*stride_h, *stride_w));
            cfg.padding = (*padding_h, *padding_w);
            let pool = crate::pooling::MaxPool2d::new(cfg);
            let grad = pool.backward(&grad_bf16, &input_bf16, None)?;
            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::AddBias { input, bias } => {
            // d/dx(x + b) = grad
            // d/db(x + b) = sum(grad) over batch and spatial dims
            let grad_input = output_grad.clone();

            // Sum over all dimensions except the bias dimension (usually channels)
            let ndims = output_grad.shape().dims().len();
            let mut sum_dims = vec![0]; // batch dimension
            if ndims > 2 {
                // Add spatial dimensions
                sum_dims.extend(2..ndims);
            }
            let grad_bias = ensure_bf16(output_grad.sum_dims(&sum_dims)?)?;

            Ok(smallvec![
                (*input, ensure_bf16(grad_input)?),
                (*bias, grad_bias)
            ])
        }

        Op::SumDim { input, dim } => {
            // Gradient of sum is broadcast back to original shape
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let mut grad_shape = input_tensor.shape().dims().to_vec();
            grad_shape[*dim] = 1;
            let grad_reshaped = output_grad.reshape(&grad_shape)?;
            let grad = grad_reshaped.broadcast_to(input_tensor.shape())?;
            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::Clamp { input, min, max } => {
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let grad_bf16 = if output_grad.dtype() != DType::BF16 {
                &output_grad.to_dtype_no_grad(DType::BF16)?
            } else {
                output_grad
            };
            let grad =
                crate::autograd_ops_complete::clamp_backward(grad_bf16, input_tensor, *min, *max)?;
            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::Div {
            lhs,
            rhs,
            lhs_shape,
            rhs_shape,
        } => {
            // d/dx (x/y) = 1/y
            // d/dy (x/y) = -x/y^2
            let lhs_tensor = fetch_saved(lhs)?;
            let rhs_tensor = fetch_saved(rhs)?;
            // No dtype guard — saved tensors may be F32 (e.g., from head_rms_norm).
            // GpuOps handles mixed BF16/F32 internally.

            // Broadcast saved tensors to output shape for correct gradient computation
            let lhs_bc = if lhs_tensor.shape() != output_grad.shape() {
                GpuOps::broadcast(&lhs_tensor, output_grad.shape())?
            } else {
                lhs_tensor
            };
            let rhs_bc = if rhs_tensor.shape() != output_grad.shape() {
                GpuOps::broadcast(&rhs_tensor, output_grad.shape())?
            } else {
                rhs_tensor
            };

            // Gradient w.r.t. lhs: grad * (1/rhs)
            let mut grad_lhs = GpuOps::div(output_grad, &rhs_bc)?;

            // Gradient w.r.t. rhs: grad * (-lhs/rhs^2)
            let rhs_squared = GpuOps::mul(&rhs_bc, &rhs_bc)?;
            let neg_lhs = GpuOps::mul_scalar(&lhs_bc, -1.0)?;
            let grad_rhs_term = GpuOps::div(&neg_lhs, &rhs_squared)?;
            let mut grad_rhs = GpuOps::mul(output_grad, &grad_rhs_term)?;

            // Reduce for broadcasting — use original shapes from Op
            if grad_lhs.shape() != lhs_shape {
                grad_lhs = reduce_grad_for_broadcast(&grad_lhs, lhs_shape)?;
            }
            if grad_rhs.shape() != rhs_shape {
                grad_rhs = reduce_grad_for_broadcast(&grad_rhs, rhs_shape)?;
            }

            Ok(smallvec![
                (*lhs, ensure_bf16(grad_lhs)?),
                (*rhs, ensure_bf16(grad_rhs)?),
            ])
        }

        Op::MaxDim {
            input,
            dim,
            keepdim,
        } => {
            // For max reduction, gradient flows only through the max elements
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;

            // Get the max values and indices
            let max_vals = input_tensor.max_dim(*dim, *keepdim)?;

            // Create a mask where input equals max (handling broadcasting)
            let max_broadcast = if *keepdim {
                max_vals.clone()
            } else {
                max_vals.unsqueeze(*dim)?
            };

            // Create mask where input == max_broadcast
            let mask = input_tensor.eq(&max_broadcast)?;

            // Broadcast gradient if needed
            let grad_broadcast = if *keepdim {
                output_grad.clone()
            } else {
                output_grad.unsqueeze(*dim)?
            };

            // Apply mask
            let grad = grad_broadcast.mul(&mask)?;

            Ok(smallvec![(*input, grad)])
        }

        Op::SumDimKeepdim { input, dim } => {
            // For sum with keepdim, gradient is broadcast back
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let input_shape = input_tensor.shape();

            // Broadcast gradient back to input shape
            let grad = output_grad.broadcast_to(input_shape)?;
            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::SumDims { input, dims } => {
            // Gradient broadcast back over all reduced dims
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let input_shape = input_tensor.shape();

            // Starting from output_grad, expand reduced dims as size-1 where needed then broadcast
            let mut grad = output_grad.clone();
            let mut target = input_shape.dims().to_vec();
            // Create a shape with 1s at reduced dims
            let mut reshape_dims = input_shape.dims().to_vec();
            for &d in dims {
                if d < reshape_dims.len() {
                    reshape_dims[d] = 1;
                }
            }
            grad = if grad.shape().dims() != &reshape_dims[..] {
                // If output_grad is already squeezed, reshape up to insert 1s
                grad.reshape(&reshape_dims)?
            } else {
                grad
            };
            let grad = grad.broadcast_to(input_shape)?;
            Ok(smallvec![(*input, grad)])
        }

        Op::Repeat { input, repeats } => {
            let input_tensor = entry.get_saved(input).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for repeat input".into())
            })?;
            let input_shape = input_tensor.shape().dims().to_vec();

            let mut grad = output_grad.clone();
            let mut current_shape = grad.shape().dims().to_vec();

            if current_shape.len() != repeats.len() {
                return Err(Error::InvalidOperation(format!(
                    "repeat backward rank mismatch: grad shape {:?}, repeats {:?}",
                    current_shape, repeats
                )));
            }

            for axis in (0..repeats.len()).rev() {
                let rep = repeats[axis];
                if rep == 1 {
                    continue;
                }

                if axis >= current_shape.len()
                    || axis >= input_shape.len()
                    || current_shape[axis] != input_shape[axis] * rep
                {
                    return Err(Error::InvalidOperation(format!(
                            "repeat backward dimension mismatch on axis {}: grad dim {}, input dim {}, rep {}",
                            axis,
                            current_shape
                                .get(axis)
                                .copied()
                                .unwrap_or_default(),
                            input_shape
                                .get(axis)
                                .copied()
                                .unwrap_or_default(),
                            rep
                        )));
                }

                let mut reshape_dims = Vec::with_capacity(current_shape.len() + 1);
                for (i, &dim) in current_shape.iter().enumerate() {
                    if i == axis {
                        reshape_dims.push(input_shape[i]);
                        reshape_dims.push(rep);
                    } else {
                        reshape_dims.push(dim);
                    }
                }

                grad = grad.reshape(&reshape_dims)?;
                grad = grad.sum_dim_keepdim(axis + 1)?;

                let mut squeezed = grad.shape().dims().to_vec();
                if axis + 1 < squeezed.len() {
                    squeezed.remove(axis + 1);
                }
                grad = grad.reshape(&squeezed)?;
                current_shape = squeezed;
            }

            if current_shape != input_shape {
                grad = grad.reshape(&input_shape)?;
                current_shape = input_shape.clone();
            }

            debug_assert_eq!(current_shape, input_shape);

            Ok(smallvec![(*input, ensure_bf16(grad)?)])
        }

        Op::Embedding { weight, indices } => {
            // For embedding, gradient flows back to weight matrix
            // Gradient w.r.t weight: scatter_add gradients to corresponding rows
            let indices_tensor = entry.get_saved(indices).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for indices".into())
            })?;
            let weight_tensor = entry
                .get_saved(weight)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for weight".into()))?;

            // Create zero gradient for weight
            let mut weight_grad = Tensor::zeros(
                weight_tensor.shape().clone(),
                weight_tensor.device().clone(),
            )?;

            // Scatter add gradients using GPU kernel
            let weight_grad_f32 = CudaKernels::scatter_add(
                weight_tensor.shape().dims(),
                output_grad,
                indices_tensor,
                0,
            )?;

            let weight_grad = if weight_tensor.dtype() == DType::F32 {
                weight_grad_f32
            } else {
                weight_grad_f32.to_dtype_no_grad(weight_tensor.dtype())?
            };

            Ok(smallvec![(*weight, weight_grad)])
        }

        Op::IndexSelect {
            input,
            indices,
            dim,
        } => {
            // Gradient flows back to selected indices
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let indices_tensor = entry.get_saved(indices).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for indices".into())
            })?;

            // Create zero gradient for input
            // FLAME is GPU-only, always use CUDA scatter_add kernel
            let grad_input_f32 = crate::cuda_kernels::scatter_add(
                input_tensor.shape().dims(),
                output_grad,
                indices_tensor,
                *dim,
            )?;

            let grad_input = if input_tensor.dtype() == DType::F32 {
                grad_input_f32
            } else {
                grad_input_f32.to_dtype_no_grad(input_tensor.dtype())?
            };

            Ok(smallvec![(*input, grad_input)])
        }

        Op::IndexAssign {
            input,
            indices,
            values,
            dim,
        } => {
            // Backward:
            //   grad_input  = output_grad with slices at `indices` zeroed
            //                 (the assignment overwrites those rows, so they
            //                 don't propagate back to `input`).
            //   grad_values = index_select(output_grad, dim, indices) — only
            //                 the indexed rows of upstream affect `values`.
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let indices_tensor = entry.get_saved(indices).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for indices".into())
            })?;
            let values_tensor = entry
                .get_saved(values)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for values".into()))?;

            // grad_values: gather upstream rows at `indices` along `dim`
            let grad_values_raw = output_grad.index_select(*dim, indices_tensor)?;
            let grad_values = if values_tensor.dtype() == grad_values_raw.dtype() {
                grad_values_raw
            } else {
                grad_values_raw.to_dtype_no_grad(values_tensor.dtype())?
            };

            // grad_input: zero the indexed rows in upstream. Build a
            // values_zero tensor matching the gathered shape and re-use
            // index_assign to overwrite. Use no_grad path to avoid recording.
            let zeros_for_assign = Tensor::zeros_dtype(
                grad_values.shape().clone(),
                output_grad.dtype(),
                output_grad.device().clone(),
            )?;
            let grad_input_full =
                output_grad.index_assign_no_grad(*dim, indices_tensor, &zeros_for_assign)?;
            let grad_input = if input_tensor.dtype() == grad_input_full.dtype() {
                grad_input_full
            } else {
                grad_input_full.to_dtype_no_grad(input_tensor.dtype())?
            };

            Ok(smallvec![(*input, grad_input), (*values, grad_values)])
        }

        Op::Cat { inputs, dim } => {
            // Split gradient back to original tensors
            let mut grads: GradVec = SmallVec::new();
            let mut offset = 0;

            for &input_id in inputs {
                let input_tensor = entry.get_saved(&input_id).ok_or_else(|| {
                    Error::InvalidOperation("Missing saved tensor for input in Cat".into())
                })?;
                let size = input_tensor.shape().dims()[*dim];

                // Slice gradient for this input
                let mut ranges = Vec::new();
                for (i, &dim_size) in output_grad.shape().dims().iter().enumerate() {
                    if i == *dim {
                        ranges.push((offset, offset + size));
                    } else {
                        ranges.push((0, dim_size));
                    }
                }

                let grad_slice = output_grad.slice(&ranges)?;
                grads.push((input_id, grad_slice));
                offset += size;
            }

            Ok(grads)
        }

        Op::Slice {
            input,
            ranges,
            input_shape,
        } => {
            // Backward for slice: scatter output_grad into a zeros tensor at the sliced position.
            let device = device.clone();
            let grad_dtype = output_grad.dtype();
            let mut grad_in = Tensor::zeros_dtype(input_shape.clone(), grad_dtype, device.clone())?;
            let in_dims = input_shape.dims();

            // Detect single-axis narrow
            let mut narrow_dim: Option<(usize, usize, usize)> = None;
            for (i, &(s, e)) in ranges.iter().enumerate() {
                if !(s == 0 && e == in_dims[i]) {
                    if narrow_dim.is_some() {
                        narrow_dim = None;
                        break;
                    }
                    narrow_dim = Some((i, s, e - s));
                }
            }

            if let Some((dim, start, _length)) = narrow_dim {
                gpu_scatter_add_narrow(output_grad, &mut grad_in, dim, start)?;
                Ok(smallvec![(*input, grad_in)])
            } else if can_gpu_multi_axis(ranges, in_dims) {
                let mut tmp = output_grad.clone();
                let mut axes: Vec<(usize, usize, usize)> = Vec::new();
                for (i, &(s, e)) in ranges.iter().enumerate() {
                    if !(s == 0 && e == in_dims[i]) {
                        axes.push((i, s, e - s));
                    }
                }
                for (axis, s, _len) in axes.into_iter().rev() {
                    let mut expanded_dims = tmp.shape().dims().to_vec();
                    expanded_dims[axis] = in_dims[axis];
                    let expanded_shape = crate::Shape::from_dims(&expanded_dims);
                    let mut expanded =
                        Tensor::zeros_dtype(expanded_shape, grad_dtype, device.clone())?;
                    gpu_scatter_add_narrow(&tmp, &mut expanded, axis, s)?;
                    tmp = expanded;
                }
                Ok(smallvec![(*input, tmp)])
            } else {
                // Fallback: same as multi-axis
                let mut tmp = output_grad.clone();
                let mut axes: Vec<(usize, usize, usize)> = Vec::new();
                for (i, &(s, e)) in ranges.iter().enumerate() {
                    if !(s == 0 && e == in_dims[i]) {
                        axes.push((i, s, e - s));
                    }
                }
                for (axis, s, _len) in axes.into_iter().rev() {
                    let mut expanded_dims = tmp.shape().dims().to_vec();
                    expanded_dims[axis] = in_dims[axis];
                    let expanded_shape = crate::Shape::from_dims(&expanded_dims);
                    let mut expanded =
                        Tensor::zeros_dtype(expanded_shape, grad_dtype, device.clone())?;
                    gpu_scatter_add_narrow(&tmp, &mut expanded, axis, s)?;
                    tmp = expanded;
                }
                Ok(smallvec![(*input, tmp)])
            }
        }

        Op::Split { input, sizes, dim } => {
            // Concatenate gradients back to original tensor
            // We need to collect gradients for all split outputs
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let input_size = input_tensor.shape().dims()[*dim];

            // Create gradient tensor filled with zeros
            let mut combined_grad =
                Tensor::zeros(input_tensor.shape().clone(), input_tensor.device().clone())?;

            // The output_grad corresponds to one of the split outputs
            // We need to place it at the correct position
            // Since we don't track which split output this is, we'll accumulate all available gradients

            // For proper implementation, we'd need to track split output indices
            // For now, we'll assume the gradient applies to the entire input
            // This is correct when all splits have gradients flowing back
            combined_grad = combined_grad.add(output_grad)?;

            Ok(smallvec![(*input, combined_grad)])
        }

        Op::Abs { input } => {
            // d/dx |x| = sign(x)
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let sign = input_tensor.sign()?;
            let grad = output_grad.mul(&sign)?;
            Ok(smallvec![(*input, grad)])
        }

        Op::Log { input } => {
            // d/dx log(x) = 1/x
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let reciprocal =
                Tensor::ones(input_tensor.shape().clone(), input_tensor.device().clone())?
                    .div(input_tensor)?;
            let grad = output_grad.mul(&reciprocal)?;
            Ok(smallvec![(*input, grad)])
        }

        Op::Softmax { input, dim } => {
            // Use the complete softmax backward implementation.
            // autograd_ops_complete functions have BF16 guards, but the gradient
            // chain is F32 by design. Cast to BF16 at the boundary.
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let output = input_tensor.softmax(*dim)?;
            let grad_bf16 = if output_grad.dtype() != DType::BF16 {
                &output_grad.to_dtype_no_grad(DType::BF16)?
            } else {
                output_grad
            };
            let grad = crate::autograd_ops_complete::softmax_backward(&output, grad_bf16, *dim)?;
            Ok(smallvec![(*input, grad)])
        }

        Op::LogSoftmax { input, dim } => {
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let output = input_tensor.log_softmax(*dim)?;
            let grad_bf16 = if output_grad.dtype() != DType::BF16 {
                &output_grad.to_dtype_no_grad(DType::BF16)?
            } else {
                output_grad
            };
            let grad =
                crate::autograd_ops_complete::log_softmax_backward(grad_bf16, &output, *dim)?;
            Ok(smallvec![(*input, grad)])
        }

        Op::Maximum { a, b } => {
            let a_tensor = entry.get_saved(a).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for a in Maximum".into())
            })?;
            let b_tensor = entry.get_saved(b).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for b in Maximum".into())
            })?;
            let mask_a = a_tensor.ge(b_tensor)?; // 1 where a>=b
            let mask_b = mask_a.neg()?.add_scalar(1.0)?; // 1 - mask
            let mut grad_a = output_grad.mul(&mask_a)?;
            let mut grad_b = output_grad.mul(&mask_b)?;
            if grad_a.shape() != a_tensor.shape() {
                grad_a = reduce_grad_for_broadcast(&grad_a, a_tensor.shape())?;
            }
            if grad_b.shape() != b_tensor.shape() {
                grad_b = reduce_grad_for_broadcast(&grad_b, b_tensor.shape())?;
            }
            Ok(smallvec![(*a, grad_a), (*b, grad_b)])
        }

        Op::Minimum { a, b } => {
            let a_tensor = entry.get_saved(a).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for a in Minimum".into())
            })?;
            let b_tensor = entry.get_saved(b).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for b in Minimum".into())
            })?;
            let mask_a = a_tensor.le(b_tensor)?; // 1 where a<=b
            let mask_b = mask_a.neg()?.add_scalar(1.0)?;
            let mut grad_a = output_grad.mul(&mask_a)?;
            let mut grad_b = output_grad.mul(&mask_b)?;
            if grad_a.shape() != a_tensor.shape() {
                grad_a = reduce_grad_for_broadcast(&grad_a, a_tensor.shape())?;
            }
            if grad_b.shape() != b_tensor.shape() {
                grad_b = reduce_grad_for_broadcast(&grad_b, b_tensor.shape())?;
            }
            Ok(smallvec![(*a, grad_a), (*b, grad_b)])
        }

        Op::Where { cond, t, f } => {
            let cond_tensor = entry.get_saved(cond).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for cond in Where".into())
            })?;
            let t_tensor = entry.get_saved(t).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for true tensor in Where".into())
            })?;
            let f_tensor = entry.get_saved(f).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for false tensor in Where".into())
            })?;
            let mask_t = cond_tensor.clone(); // 1 where true
            let mask_f = mask_t.neg()?.add_scalar(1.0)?; // 1 - mask
            let mut grad_t = output_grad.mul(&mask_t)?;
            let mut grad_f = output_grad.mul(&mask_f)?;
            if grad_t.shape() != t_tensor.shape() {
                grad_t = reduce_grad_for_broadcast(&grad_t, t_tensor.shape())?;
            }
            if grad_f.shape() != f_tensor.shape() {
                grad_f = reduce_grad_for_broadcast(&grad_f, f_tensor.shape())?;
            }
            Ok(smallvec![(*t, grad_t), (*f, grad_f)])
        }

        Op::MSELoss {
            predictions,
            targets,
            num_elements,
        } => {
            // For MSE: d/dx[(x-y)^2] = 2(x-y)/n
            let predictions_tensor = entry.get_saved(predictions).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for predictions".into())
            })?;
            let targets_tensor = entry.get_saved(targets).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for targets".into())
            })?;

            // Gradient is 2 * (predictions - targets) / num_elements
            // Note: output_grad may be a scalar (shape=[]). Some GPU broadcast helpers
            // do not support 0-D inputs. Expand the scalar explicitly to the diff shape
            // before the elementwise multiply to avoid shape-mismatch issues.
            let diff = predictions_tensor.sub(targets_tensor)?; // same shape as predictions/targets
            let scale = 2.0 / (*num_elements as f32);

            // Scale upstream grad first
            let scaled = output_grad.mul_scalar(scale)?;

            // Rank-normalize upstream grad, then broadcast on GPU to diff shape
            let up_ranked = expand_to_rank(&scaled, diff.shape().dims().len())?;
            let up_broadcast = GpuOps::broadcast(&up_ranked, diff.shape())?;

            // Now shapes match for elementwise multiply
            let grad_predictions = GpuOps::mul(&up_broadcast, &diff)?;
            let grad_targets = grad_predictions.mul_scalar(-1.0)?;

            Ok(smallvec![
                (*predictions, grad_predictions),
                (*targets, grad_targets),
            ])
        }

        Op::L1Loss {
            predictions,
            targets,
            num_elements,
        } => {
            // For L1: d/dx|x-y| = sign(x-y)/n
            let predictions_tensor = entry.get_saved(predictions).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for predictions".into())
            })?;
            let targets_tensor = entry.get_saved(targets).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for targets".into())
            })?;

            let diff = predictions_tensor.sub(targets_tensor)?;
            let sign = diff.sign()?;
            let scale = 1.0 / (*num_elements as f32);
            let grad_predictions = output_grad.mul_scalar(scale)?.mul(&sign)?;
            let grad_targets = grad_predictions.mul_scalar(-1.0)?;

            Ok(smallvec![
                (*predictions, grad_predictions),
                (*targets, grad_targets),
            ])
        }

        Op::HuberLoss {
            predictions,
            targets,
            delta,
            num_elements,
        } => {
            // Huber gradient:
            // if |x-y| <= delta: (x-y)/n
            // if |x-y| > delta: delta*sign(x-y)/n
            let predictions_tensor = entry.get_saved(predictions).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for predictions".into())
            })?;
            let targets_tensor = entry.get_saved(targets).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for targets".into())
            })?;

            let diff = predictions_tensor.sub(targets_tensor)?;
            let abs_diff = diff.abs()?;
            let delta_vec = vec![*delta; diff.shape().elem_count()];
            let delta_tensor =
                Tensor::from_vec(delta_vec, diff.shape().clone(), diff.device().clone())?;

            // Create mask for |diff| <= delta
            let mask = abs_diff.le(&delta_tensor)?;

            // Quadratic gradient: diff
            let quad_grad = diff.clone();

            // Linear gradient: delta * sign(diff)
            let linear_grad = diff.sign()?.mul_scalar(*delta)?;

            // Combine using mask
            let combined_grad = mask.where_tensor(&quad_grad, &linear_grad)?;

            let scale = 1.0 / (*num_elements as f32);
            let grad_predictions = output_grad.mul_scalar(scale)?.mul(&combined_grad)?;
            let grad_targets = grad_predictions.mul_scalar(-1.0)?;

            Ok(smallvec![
                (*predictions, grad_predictions),
                (*targets, grad_targets),
            ])
        }

        Op::BCELoss {
            predictions,
            targets,
            num_elements,
        } => {
            // BCE gradient: d/dp[-y*log(p) - (1-y)*log(1-p)] = (p-y)/(p(1-p))/n
            let predictions_tensor = entry.get_saved(predictions).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for predictions".into())
            })?;
            let targets_tensor = entry.get_saved(targets).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for targets".into())
            })?;

            // Clamp predictions to avoid division by zero
            let eps = 1e-7;
            let pred_clamped = predictions_tensor.clamp(eps, 1.0 - eps)?;

            // Compute (predictions - targets) / (predictions * (1 - predictions))
            let numerator = pred_clamped.sub(targets_tensor)?;
            let one_minus_pred = pred_clamped.neg()?.add_scalar(1.0)?;
            let denominator = pred_clamped.mul(&one_minus_pred)?;

            let grad_base = numerator.div(&denominator)?;
            let scale = 1.0 / (*num_elements as f32);
            let grad_predictions = output_grad.mul_scalar(scale)?.mul(&grad_base)?;

            // No gradient w.r.t targets for BCE
            Ok(smallvec![(*predictions, grad_predictions)])
        }

        Op::NLLLoss {
            log_probs,
            targets,
            batch_size,
        } => {
            // NLL gradient: sparse gradient, -1/batch_size at target indices
            let log_probs_tensor = entry.get_saved(log_probs).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for log_probs".into())
            })?;
            let targets_tensor = entry.get_saved(targets).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for targets".into())
            })?;

            // Create zero gradient tensor
            let mut grad_log_probs = Tensor::zeros(
                log_probs_tensor.shape().clone(),
                log_probs_tensor.device().clone(),
            )?;

            // Set gradients at target indices using GPU scatter
            let scale = -1.0 / (*batch_size as f32);

            // Create a tensor with the gradient values to scatter
            let grad_values = Tensor::ones(
                Shape::from_dims(&[*batch_size]),
                log_probs_tensor.device().clone(),
            )?
            .mul_scalar(scale)?;

            // Use scatter_add to place gradients at target indices
            let grad_log_probs = CudaKernels::scatter_add(
                grad_log_probs.shape().dims(),
                &grad_values,
                targets_tensor,
                1,
            )?;

            let final_grad = output_grad.mul(&grad_log_probs)?;

            Ok(smallvec![(*log_probs, final_grad)])
        }

        Op::GroupNorm {
            input,
            num_groups,
            weight,
            bias,
        } => {
            // GroupNorm backward pass
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let shape = input_tensor.shape().dims();
            let num_channels = shape[1];

            // Saved mean and variance may live in either the legacy
            // saved_tensors vector or the saved-ref path. Use the combined
            // positional accessor so GroupNorm backward works under both
            // storage modes.
            let stat_shape = [shape[0], *num_groups];
            let mut stats = (0..entry.saved_count()).filter_map(|i| {
                let (_, t) = entry.saved_at(i)?;
                (t.shape().dims() == stat_shape).then_some(t)
            });
            let mean = stats
                .next()
                .ok_or_else(|| Error::InvalidOperation("Missing saved mean".into()))?;
            let var = stats
                .next()
                .ok_or_else(|| Error::InvalidOperation("Missing saved variance".into()))?;

            // Compute gradients
            let weight_tensor = weight.and_then(|w| entry.get_saved(&w));
            let bias_tensor = bias.and_then(|b| entry.get_saved(&b));

            let grad_bf16 = if output_grad.dtype() != DType::BF16 {
                &output_grad.to_dtype_no_grad(DType::BF16)?
            } else {
                output_grad
            };
            let (grad_input, grad_weight, grad_bias) =
                crate::autograd_ops_complete::group_norm_backward(
                    grad_bf16,
                    input_tensor,
                    mean,
                    var,
                    weight_tensor,
                    *num_groups,
                    1e-5,
                )?;

            let mut grads: GradVec = smallvec![(*input, grad_input)];
            if let (Some(w_id), Some(gw)) = (*weight, grad_weight) {
                grads.push((w_id, gw));
            }
            if let (Some(b_id), Some(gb)) = (*bias, grad_bias) {
                grads.push((b_id, gb));
            }

            Ok(grads)
        }

        Op::FlashAttention {
            query,
            key,
            value,
            mask,
            scale,
            causal,
            padding_lens,
            output,
            stats,
        } => {
            // Phase 2c (2026-04-23): cuDNN SDPA backward is the primary path
            // for shapes the frontend supports (head_dim ∈ {64, 96, 128},
            // unmasked, 4D BF16). Decomposed recompute is the fallback for
            // everything else — masks, odd head_dims, non-BF16 dtypes,
            // non-4D tensors. See HANDOFF_2026-04-23.md §4.
            //
            // Saved Q/K/V in Klein/Chroma/Wan/FLUX are permute views of the
            // materialized reshape output (no `.contiguous()` after permute
            // in the trainer). Route through fetch_saved so their custom
            // strides are materialized before flame_cudnn_sdpa_bwd_bf16 /
            // attention_backward_recompute reads them via raw pointers.
            // (Bug #4 follow-up; without this the SDPA backward at every
            // block produces ~5%-direction-wrong dq/dk/dv that compounds
            // multiplicatively across all 25 Klein blocks.)
            let query_tensor_owned = fetch_saved(query)?;
            let key_tensor_owned = fetch_saved(key)?;
            let value_tensor_owned = fetch_saved(value)?;
            let query_tensor = &query_tensor_owned;
            let key_tensor = &key_tensor_owned;
            let value_tensor = &value_tensor_owned;
            let mask_tensor_owned = if let Some(m_id) = mask {
                Some(fetch_saved(m_id)?)
            } else {
                None
            };
            let mask_tensor = mask_tensor_owned.as_ref();

            let q_dims = query_tensor.shape().dims().to_vec();
            let k_dims = key_tensor.shape().dims().to_vec();

            // Stage 2 fix (2026-05-12): direct id lookup for saved O and
            // saved Stats. The previous shape-find heuristic was broken —
            // `fetch_saved` materializes non-contig views via `.contiguous()`
            // which produces FRESH `TensorId`s, so `t.id != query_tensor.id`
            // always fired false-positive and the first shape match (saved Q)
            // was picked up as O. This destroyed cuDNN flash-bwd's
            // `dO·O^T` identity → `grad_norm=inf`. Direct lookup keyed on
            // the ids recorded at forward time fixes it.
            let output_tensor_owned = match output {
                Some(id) => Some(fetch_saved(id)?),
                None => None,
            };
            let stats_tensor_owned = match stats {
                Some(id) => Some(fetch_saved(id)?),
                None => None,
            };
            let output_tensor = output_tensor_owned.as_ref();
            let stats_tensor = stats_tensor_owned.as_ref();

            // Try cuDNN backward for supported shapes.
            let cudnn_result = try_cudnn_sdpa_backward(
                query_tensor,
                key_tensor,
                value_tensor,
                output_tensor,
                output_grad,
                stats_tensor,
                mask_tensor,
                *causal,
                *padding_lens,
                *scale,
                device,
            )?;

            let (grad_q, grad_k, grad_v) = if let Some(triple) = cudnn_result {
                triple
            } else if let Some(triple) = try_pytorch_flash_hd128_backward(
                query_tensor,
                key_tensor,
                value_tensor,
                output_grad,
                *causal,
                *scale,
            )? {
                triple
            } else {
                // Decomposed recompute fallback.
                //
                // 2026-05-21 fix: when cuDNN forward used `maybe_pad_for_cudnn`
                // to zero-pad Q/K/V to a 64-aligned sequence (e.g. HiDream-O1
                // seq=497 → padded 512), the saved Q/K/V are PADDED but the
                // recompute backward treated them as the real attention
                // sequence. Softmax over `seq+pad` keys yields different
                // weights than softmax over `seq` keys (the pad keys at
                // K=zeros have logit=0 → exp(0)=1 → absorb probability mass),
                // and dQ/dK/dV from that contaminated softmax are wrong
                // direction. Symptom: HiDream-O1 attention LoRA-B grads cos
                // ~0.01 (essentially random) vs Python ref, while MLP/o_proj
                // LoRA-B grads matched at cos>0.999.
                //
                // Fix: slice padded Q/K/V/output_grad to the real lengths
                // recorded in `padding_lens`, run the recompute on the real
                // shape (correct softmax), then zero-pad dQ/dK/dV back to the
                // saved padded shape so the upstream cat-pad backward routes
                // grad to the original (unpadded) parameter ids correctly.
                let (q_for_bwd, k_for_bwd, v_for_bwd, dout_for_bwd, pad_info) =
                    if let Some((real_q_len, real_kv_len)) = *padding_lens {
                        if real_q_len < q_dims[2] || real_kv_len < k_dims[2] {
                            let q_s = query_tensor.narrow(2, 0, real_q_len)?.contiguous()?;
                            let k_s = key_tensor.narrow(2, 0, real_kv_len)?.contiguous()?;
                            let v_s = value_tensor.narrow(2, 0, real_kv_len)?.contiguous()?;
                            let dout_s = output_grad.narrow(2, 0, real_q_len)?.contiguous()?;
                            (q_s, k_s, v_s, dout_s, Some((q_dims[2], k_dims[2], real_q_len, real_kv_len)))
                        } else {
                            (
                                query_tensor.clone(),
                                key_tensor.clone(),
                                value_tensor.clone(),
                                output_grad.clone(),
                                None,
                            )
                        }
                    } else {
                        (
                            query_tensor.clone(),
                            key_tensor.clone(),
                            value_tensor.clone(),
                            output_grad.clone(),
                            None,
                        )
                    };

                let q_bwd_dims = q_for_bwd.shape().dims();
                let k_bwd_dims = k_for_bwd.shape().dims();
                let expected_attn_shape = if q_bwd_dims.len() == 4 && k_bwd_dims.len() == 4 {
                    Some([q_bwd_dims[0], q_bwd_dims[1], q_bwd_dims[2], k_bwd_dims[2]])
                } else {
                    None
                };
                let cached_attn: Option<&Tensor> = expected_attn_shape.and_then(|s| {
                    entry
                        .saved_tensors
                        .iter()
                        .map(|(_, t)| t)
                        .find(|t| t.shape().dims() == s)
                });
                let (dq, dk, dv) = attention_backward_recompute(
                    &q_for_bwd,
                    &k_for_bwd,
                    &v_for_bwd,
                    &dout_for_bwd,
                    mask_tensor,
                    *causal,
                    *scale,
                    cached_attn,
                )?;

                if let Some((padded_q_len, padded_kv_len, real_q_len, real_kv_len)) = pad_info {
                    let device = dq.device().clone();
                    let dtype = dq.dtype();
                    let pad_q_rows = padded_q_len - real_q_len;
                    let pad_kv_rows = padded_kv_len - real_kv_len;
                    let dq_padded = if pad_q_rows > 0 {
                        let dq_dims = dq.shape().dims().to_vec();
                        let zero = Tensor::zeros_dtype(
                            Shape::from_dims(&[dq_dims[0], dq_dims[1], pad_q_rows, dq_dims[3]]),
                            dtype,
                            device.clone(),
                        )?;
                        Tensor::cat(&[&dq, &zero], 2)?
                    } else {
                        dq
                    };
                    let dk_padded = if pad_kv_rows > 0 {
                        let dk_dims = dk.shape().dims().to_vec();
                        let zero = Tensor::zeros_dtype(
                            Shape::from_dims(&[dk_dims[0], dk_dims[1], pad_kv_rows, dk_dims[3]]),
                            dtype,
                            device.clone(),
                        )?;
                        Tensor::cat(&[&dk, &zero], 2)?
                    } else {
                        dk
                    };
                    let dv_padded = if pad_kv_rows > 0 {
                        let dv_dims = dv.shape().dims().to_vec();
                        let zero = Tensor::zeros_dtype(
                            Shape::from_dims(&[dv_dims[0], dv_dims[1], pad_kv_rows, dv_dims[3]]),
                            dtype,
                            device.clone(),
                        )?;
                        Tensor::cat(&[&dv, &zero], 2)?
                    } else {
                        dv
                    };
                    (dq_padded, dk_padded, dv_padded)
                } else {
                    (dq, dk, dv)
                }
            };
            Ok(smallvec![
                (*query, grad_q),
                (*key, grad_k),
                (*value, grad_v)
            ])
        }

        Op::PrefixCausalFullAttention {
            query,
            key,
            value,
            prefix_len,
            scale,
        } => {
            let query_tensor_owned = fetch_saved(query)?;
            let key_tensor_owned = fetch_saved(key)?;
            let value_tensor_owned = fetch_saved(value)?;
            let q_dims = query_tensor_owned.shape().dims().to_vec();
            let k_dims = key_tensor_owned.shape().dims().to_vec();
            let v_dims = value_tensor_owned.shape().dims().to_vec();
            if q_dims.len() != 4
                || k_dims.len() != 4
                || v_dims.len() != 4
                || q_dims[2] != k_dims[2]
                || k_dims[2] != v_dims[2]
            {
                return Err(Error::InvalidOperation(format!(
                    "PrefixCausalFullAttention backward expects self-attention [B,H,S,D], got q={:?} k={:?} v={:?}",
                    q_dims, k_dims, v_dims
                )));
            }
            if *prefix_len > q_dims[2] {
                return Err(Error::InvalidOperation(format!(
                    "PrefixCausalFullAttention backward prefix_len {} exceeds seq_len {}",
                    prefix_len, q_dims[2]
                )));
            }
            if std::env::var("FLAME_PREFIX_CAUSAL_FULL_BWD_STRUCTURED")
                .ok()
                .as_deref()
                == Some("1")
            {
                let seq_len = q_dims[2];
                let suffix_len = seq_len - *prefix_len;
                let q_prefix = query_tensor_owned.narrow(2, 0, *prefix_len)?.contiguous()?;
                let k_prefix = key_tensor_owned.narrow(2, 0, *prefix_len)?.contiguous()?;
                let v_prefix = value_tensor_owned.narrow(2, 0, *prefix_len)?.contiguous()?;
                let dout_prefix = output_grad.narrow(2, 0, *prefix_len)?.contiguous()?;
                let (dq_prefix, dk_prefix, dv_prefix) = attention_backward_recompute(
                    &q_prefix,
                    &k_prefix,
                    &v_prefix,
                    &dout_prefix,
                    None,
                    true,
                    *scale,
                    None,
                )?;

                let q_suffix = query_tensor_owned
                    .narrow(2, *prefix_len, suffix_len)?
                    .contiguous()?;
                let dout_suffix = output_grad.narrow(2, *prefix_len, suffix_len)?.contiguous()?;
                let (dq_suffix, dk_suffix, dv_suffix) = attention_backward_recompute(
                    &q_suffix,
                    &key_tensor_owned,
                    &value_tensor_owned,
                    &dout_suffix,
                    None,
                    false,
                    *scale,
                    None,
                )?;

                let grad_q = Tensor::cat(&[&dq_prefix, &dq_suffix], 2)?;
                let zero_k_tail = Tensor::zeros_dtype(
                    Shape::from_dims(&[k_dims[0], k_dims[1], suffix_len, k_dims[3]]),
                    dk_prefix.dtype(),
                    query_tensor_owned.device().clone(),
                )?;
                let zero_v_tail = Tensor::zeros_dtype(
                    Shape::from_dims(&[v_dims[0], v_dims[1], suffix_len, v_dims[3]]),
                    dv_prefix.dtype(),
                    query_tensor_owned.device().clone(),
                )?;
                let dk_prefix_padded = Tensor::cat(&[&dk_prefix, &zero_k_tail], 2)?;
                let dv_prefix_padded = Tensor::cat(&[&dv_prefix, &zero_v_tail], 2)?;
                let grad_k = dk_prefix_padded.add(&dk_suffix)?;
                let grad_v = dv_prefix_padded.add(&dv_suffix)?;
                return Ok(smallvec![
                    (*query, grad_q),
                    (*key, grad_k),
                    (*value, grad_v)
                ]);
            }
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            if let Some((grad_q, grad_k, grad_v)) =
                try_prefix_causal_full_pytorch_flash_hd128_backward(
                    &query_tensor_owned,
                    &key_tensor_owned,
                    &value_tensor_owned,
                    output_grad,
                    *prefix_len,
                    *scale,
                )?
            {
                return Ok(smallvec![
                    (*query, grad_q),
                    (*key, grad_k),
                    (*value, grad_v)
                ]);
            }
            let mask = prefix_causal_full_keep_mask_for_logits(
                q_dims[2],
                *prefix_len,
                query_tensor_owned.device().clone(),
            )?;
            let (grad_q, grad_k, grad_v) = attention_backward_recompute(
                &query_tensor_owned,
                &key_tensor_owned,
                &value_tensor_owned,
                output_grad,
                Some(&mask),
                false,
                *scale,
                None,
            )?;
            Ok(smallvec![
                (*query, grad_q),
                (*key, grad_k),
                (*value, grad_v)
            ])
        }

        Op::SageAttention {
            query_id,
            key_id,
            value_id,
            scale,
            causal,
            quantized,
        } => {
            // SageAttention backward pass
            let query_tensor = entry
                .get_saved(query_id)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for query".into()))?;
            let key_tensor = entry
                .get_saved(key_id)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for key".into()))?;
            let value_tensor = entry
                .get_saved(value_id)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for value".into()))?;

            // Get attention weights (should be saved with a known ID)
            // In SageAttention forward, attention_weights.id is saved in saved_tensors
            let attention_weights = entry
                .saved_tensors
                .iter()
                .map(|(_, t)| t)
                .find(|t| {
                    t.shape().dims().len() == 4
                        && t.shape().dims()[0] == query_tensor.shape().dims()[0]
                        && t.shape().dims()[1] == query_tensor.shape().dims()[1]
                        && t.shape().dims()[2] == query_tensor.shape().dims()[2]
                        && t.shape().dims()[3] == key_tensor.shape().dims()[2]
                })
                .ok_or_else(|| Error::InvalidOperation("Missing saved attention weights".into()))?;

            // Call sage attention backward
            let (grad_q, grad_k, grad_v) = crate::sage_attention::sage_attention_backward(
                output_grad,
                query_tensor,
                key_tensor,
                value_tensor,
                attention_weights,
                *scale,
                *causal,
                *quantized,
            )?;

            Ok(smallvec![
                (*query_id, grad_q),
                (*key_id, grad_k),
                (*value_id, grad_v),
            ])
        }

        _ => {
            // This should not happen if all operations are implemented
            Err(Error::InvalidOperation(format!(
                "Gradient not implemented for operation: {:?}",
                entry.op
            )))
        }
    }?;

    // Return gradients in their native dtype (typically F32 from GpuOps).
    // Previously every gradient was cast to BF16 here, then immediately cast
    // back to F32 in GradientMap::accumulate — a wasteful round-trip that also
    // caused dtype mismatches when the next backward op received F32 output_grad
    // but BF16 saved_tensors, triggering F32 casts in GpuOps::mul etc.
    Ok(grads)
}

/// Reduce gradient for broadcast operations
/// When a tensor was broadcast during forward pass, we need to sum gradients
/// along the broadcast dimensions during backward pass
fn reduce_grad_for_broadcast(grad: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    let gd = grad.shape().dims().to_vec();
    let td = target_shape.dims().to_vec();

    // Fast path
    if gd == td {
        return Ok(grad.clone());
    }

    // Left-pad target dims with 1s to match grad rank (NumPy semantics)
    let g_rank = gd.len();
    let t_rank = td.len();
    let mut padded_td = vec![1usize; g_rank];
    for i in 0..t_rank {
        padded_td[g_rank - t_rank + i] = td[i];
    }

    // Upcast to F32 before summing — BF16 overflows when reducing large dims
    // (e.g., summing 24×768=18432 elements exceeds BF16 max ~65504).
    let orig_dtype = grad.dtype();
    let mut result = if orig_dtype != DType::F32 {
        grad.to_dtype_no_grad(DType::F32)?
    } else {
        grad.clone()
    };
    for axis in 0..g_rank {
        if padded_td[axis] == 1 && gd[axis] != 1 {
            result = result.sum_dim_keepdim(axis)?;
        }
    }

    let mut result = result.reshape(&td)?;
    if result.dtype() != orig_dtype {
        result = result.to_dtype_no_grad(orig_dtype)?;
    }
    Ok(result)
}

/// Broadcast tensor to target shape (GPU-only, no CPU sync).
/// NOTE: currently unused — backward path uses Tensor::broadcast_to() which
/// delegates to the GPU broadcast kernel. Kept as utility.
fn broadcast_to(tensor: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    if tensor.shape == *target_shape {
        return Ok(tensor.clone());
    }
    // Delegate to the Tensor method which uses GPU broadcast_to_impl
    tensor.broadcast_to(target_shape)
}

/// Expand a tensor by appending size-1 dimensions until it reaches target rank
/// Useful to make scalar or low-rank upstream gradients rank-compatible before GPU broadcast
fn expand_to_rank(tensor: &Tensor, target_rank: usize) -> Result<Tensor> {
    let mut dims = tensor.shape().dims().to_vec();
    if dims.len() == target_rank {
        return Ok(tensor.clone());
    }
    while dims.len() < target_rank {
        dims.push(1);
    }
    tensor.reshape(&dims)
}

/// Helper function to compute inverse permutation
fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inverse[p] = i;
    }
    inverse
}

// Comparison operations are implemented in tensor_ops_extended.rs
/// Identity function — previously cast every gradient to BF16, but accumulate()
/// immediately casts back to FP32. Removing the round-trip saves ~40 CUDA kernels
/// + allocations per backward pass.
#[inline]
fn ensure_bf16(tensor: Tensor) -> Result<Tensor> {
    Ok(tensor)
}

#[inline]
fn guard_tensor(op: &str, tensor: &Tensor) -> Result<()> {
    if tensor.rank() == 4 {
        assert_nhwc_bf16_public(op, tensor)
    } else {
        // Accept both BF16 and F32 — mixed-precision training produces F32
        // intermediates (e.g. RMSNorm, modulate_pre) that legitimately appear
        // as saved tensors in backward.
        Ok(())
    }
}

#[inline]
fn guard_optional_tensor(op: &str, tensor: Option<&Tensor>) -> Result<()> {
    if let Some(t) = tensor {
        guard_tensor(op, t)?;
    }
    Ok(())
}

/// Short stable tag for an `Op` variant — used by FLAME_DEBUG_FINITE so
/// backward traces show which op emitted a non-finite grad.
fn op_tag(op: &Op) -> &'static str {
    match op {
        Op::Add { .. } => "Add",
        Op::Sub { .. } => "Sub",
        Op::Mul { .. } => "Mul",
        Op::Div { .. } => "Div",
        Op::MulScalar { .. } => "MulScalar",
        Op::AddScalar { .. } => "AddScalar",
        Op::MatMul { .. } => "MatMul",
        Op::BatchMatMul { .. } => "BatchMatMul",
        Op::Linear { .. } => "Linear",
        Op::ReLU { .. } => "ReLU",
        Op::GELU { .. } => "GELU",
        Op::SiLU { .. } => "SiLU",
        Op::Tanh { .. } => "Tanh",
        Op::Sigmoid { .. } => "Sigmoid",
        Op::Square { .. } => "Square",
        Op::Sqrt { .. } => "Sqrt",
        Op::Abs { .. } => "Abs",
        Op::Log { .. } => "Log",
        Op::Sum { .. } => "Sum",
        Op::Mean { .. } => "Mean",
        Op::SumDim { .. } => "SumDim",
        Op::SumDimKeepdim { .. } => "SumDimKeepdim",
        Op::SumDims { .. } => "SumDims",
        Op::MaxDim { .. } => "MaxDim",
        Op::Maximum { .. } => "Maximum",
        Op::Minimum { .. } => "Minimum",
        Op::Clamp { .. } => "Clamp",
        Op::Softmax { .. } => "Softmax",
        Op::LogSoftmax { .. } => "LogSoftmax",
        Op::Transpose { .. } => "Transpose",
        Op::Reshape { .. } => "Reshape",
        Op::Permute { .. } => "Permute",
        Op::Broadcast { .. } => "Broadcast",
        Op::UpsampleNearest2D { .. } => "UpsampleNearest2D",
        Op::MaxPool2D { .. } => "MaxPool2D",
        Op::Repeat { .. } => "Repeat",
        Op::Cast { .. } => "Cast",
        Op::Cat { .. } => "Cat",
        Op::Split { .. } => "Split",
        Op::Slice { .. } => "Slice",
        Op::Where { .. } => "Where",
        Op::Conv2d { .. } => "Conv2d",
        Op::Conv2dNHWC { .. } => "Conv2dNHWC",
        Op::AddBias { .. } => "AddBias",
        Op::LayerNorm { .. } => "LayerNorm",
        Op::RMSNorm { .. } => "RMSNorm",
        Op::GroupNorm { .. } => "GroupNorm",
        Op::Embedding { .. } => "Embedding",
        Op::IndexSelect { .. } => "IndexSelect",
        Op::IndexAssign { .. } => "IndexAssign",
        Op::MSELoss { .. } => "MSELoss",
        Op::L1Loss { .. } => "L1Loss",
        Op::HuberLoss { .. } => "HuberLoss",
        Op::BCELoss { .. } => "BCELoss",
        Op::NLLLoss { .. } => "NLLLoss",
        Op::FlashAttention { .. } => "FlashAttention",
        Op::PrefixCausalFullAttention { .. } => "PrefixCausalFullAttention",
        Op::SageAttention { .. } => "SageAttention",
        Op::FusedSwiGLU { .. } => "FusedSwiGLU",
        Op::FusedSwiGLUSplit { .. } => "FusedSwiGLUSplit",
        Op::QkvSplitPermute { .. } => "QkvSplitPermute",
        Op::RoPePrecomputed { .. } => "RoPePrecomputed",
        Op::GateResidual { .. } => "GateResidual",
        Op::Checkpoint { .. } => "Checkpoint",
        Op::CheckpointOffloadBoundary { .. } => "CheckpointOffloadBoundary",
        Op::CheckpointOffload { .. } => "CheckpointOffload",
    }
}
