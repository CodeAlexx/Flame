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

use crate::cuda::ffi;
use crate::cuda_kernels_gpu::CudaKernels;
use crate::cuda_ops::GpuOps;
use crate::device::CudaStreamRawPtrExt;
use crate::gradient::GradientMap;
use crate::activation_offload::{ActivationOffloadPool, OffloadHandle};
use crate::gradient_checkpointing::{CHECKPOINT_HAS_ENTRIES, CHECKPOINT_MANAGER};
use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Inline-storage buffer for tensors saved on the tape. Most ops save 0-3
/// tensors (input, weight, bias), so storing them inline eliminates a Vec
/// heap allocation per recorded op — and with ~2660 tape entries per Klein
/// 4B step that adds up.
pub type SavedTensors = SmallVec<[(TensorId, Tensor); 3]>;

/// Atomic mirror of `AutogradContextInner::enabled`. Checked by tensor ops
/// BEFORE constructing Op enums or cloning saved_tensors, avoiding GPU memcpys
/// and mutex locks when autograd is disabled (e.g. during checkpoint forward).
/// Updated alongside `ctx.enabled` in `record_op`, `checkpoint`, `backward`, etc.
static AUTOGRAD_ENABLED: AtomicBool = AtomicBool::new(true);

lazy_static::lazy_static! {
    /// Global autograd context - thread-safe
    static ref AUTOGRAD_CONTEXT: Mutex<AutogradContextInner> = Mutex::new(AutogradContextInner::new());
}

/// Global activation offload pool. Set once via `set_activation_offload_pool`
/// at training setup, then used by `checkpoint_offload` during forward and
/// by `Op::CheckpointOffload` backward to push/pull activations.
static ACTIVATION_POOL: std::sync::OnceLock<Arc<Mutex<ActivationOffloadPool>>> = std::sync::OnceLock::new();

/// Install the activation offload pool for checkpoint_offload to use.
/// Call once at training setup. Returns Err if already set.
pub fn set_activation_offload_pool(pool: Arc<Mutex<ActivationOffloadPool>>) -> Result<()> {
    ACTIVATION_POOL.set(pool).map_err(|_| {
        Error::InvalidOperation("Activation offload pool already set".into())
    })
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
    /// Fused SwiGLU: silu(gate) * up in one kernel.
    /// Backward: d_gate = dsilu(gate) * up * dout, d_up = silu(gate) * dout
    FusedSwiGLU {
        gate: TensorId,
        up: TensorId,
    },
    /// Fused RoPE with precomputed cos/sin.
    /// Backward: apply_rope(grad, cos, -sin) via same fused kernel.
    RoPePrecomputed {
        input: TensorId,
        cos: TensorId,
        sin: TensorId,
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

/// Entry in the computation tape
#[derive(Clone)]
struct TapeEntry {
    /// Output tensor ID
    output_id: TensorId,

    /// Operation that produced the output
    op: Op,

    /// Saved tensors needed for backward pass. Inline-sized for the common
    /// 0-3 tensor case (input, weight, bias) to avoid a Vec heap allocation
    /// per recorded op.
    saved_tensors: SavedTensors,
}

impl TapeEntry {
    /// Look up a saved tensor by ID (linear scan — fast for typical 1-3 entries)
    #[inline]
    fn get_saved(&self, id: &TensorId) -> Option<&Tensor> {
        self.saved_tensors
            .iter()
            .find(|(k, _)| k == id)
            .map(|(_, v)| v)
    }

    /// Iterate over saved tensor IDs
    #[inline]
    fn saved_keys(&self) -> impl Iterator<Item = &TensorId> {
        self.saved_tensors.iter().map(|(k, _)| k)
    }
}

impl std::fmt::Debug for TapeEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapeEntry")
            .field("output_id", &self.output_id)
            .field("op", &self.op)
            .field("saved_count", &self.saved_tensors.len())
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
        _ => Err(Error::InvalidOperation("unsupported dtype for raw ptr".into())),
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
        _ => Err(Error::InvalidOperation("unsupported dtype for raw ptr mut".into())),
    }
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
    scale: f32,
    cached_attn: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    // All ops in BF16 (bmm requires matching dtypes)
    let q = if q.dtype() != DType::BF16 { &q.to_dtype_no_grad(DType::BF16)? } else { q };
    let k = if k.dtype() != DType::BF16 { &k.to_dtype_no_grad(DType::BF16)? } else { k };
    let v = if v.dtype() != DType::BF16 { &v.to_dtype_no_grad(DType::BF16)? } else { v };
    let dout = if dout.dtype() != DType::BF16 { &dout.to_dtype_no_grad(DType::BF16)? } else { dout };

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
        if let Some(m) = mask {
            logits = logits.add(m)?;
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
}

impl AutogradContext {
    /// Record an operation in the computation graph.
    ///
    /// `saved_tensors` accepts anything convertible into `SavedTensors`
    /// (the inline-sized `SmallVec<[(TensorId, Tensor); 3]>`). This keeps
    /// the existing `vec![...]` call sites compiling unchanged while
    /// avoiding a heap allocation for the common 0-3 tensor case.
    pub fn record_op(
        output_id: TensorId,
        op: Op,
        saved_tensors: impl Into<SavedTensors>,
    ) {
        // Fast path: skip lock entirely when autograd is disabled (e.g., during backward).
        // This prevents deadlock when backward ops call high-level Tensor methods
        // that would otherwise try to re-acquire the already-held context lock.
        if !AUTOGRAD_ENABLED.load(Ordering::Relaxed) {
            return;
        }

        let saved_tensors: SavedTensors = saved_tensors.into();

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
        if !saved_tensors.is_empty()
            && CHECKPOINT_HAS_ENTRIES.load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Ok(mut mgr) = CHECKPOINT_MANAGER.lock() {
                for (id, tensor) in &saved_tensors {
                    let _ = mgr.checkpoint_saved_tensor(*id, tensor);
                }
            }
        }

        ctx.record(TapeEntry {
            output_id,
            op,
            saved_tensors,
        });
    }

    /// Clear the computation graph
    pub fn clear() {
        if let Ok(mut ctx) = AUTOGRAD_CONTEXT.lock() {
            ctx.clear();
        }
    }

    /// Number of entries currently on the autograd tape.
    pub fn tape_len() -> usize {
        AUTOGRAD_CONTEXT.lock().map(|ctx| ctx.tape.len()).unwrap_or(0)
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

    /// Compute gradients via backpropagation
    pub fn backward(loss: &Tensor) -> Result<GradientMap> {
        // Cache profiling flag once (avoid syscall per-op)
        static PROFILE_CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let profile = *PROFILE_CACHED.get_or_init(|| {
            std::env::var("FLAME_PROFILE").ok().map(|v| v == "1").unwrap_or(false)
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
            let (tape_entries, prev_enabled, compact_index, needed_grad_ids, use_cuda_graph, tape_len, graph_phase) = {
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
                    let id_iter = std::iter::once(loss.id).chain(
                        ctx.tape.iter().flat_map(|e| {
                            let mut ids = vec![e.output_id];
                            for (tid, _) in &e.saved_tensors {
                                ids.push(*tid);
                            }
                            match &e.op {
                                Op::Add { lhs, rhs, .. } | Op::Sub { lhs, rhs }
                                | Op::Mul { lhs, rhs } | Op::Div { lhs, rhs, .. }
                                | Op::MatMul { lhs, rhs } | Op::BatchMatMul { lhs, rhs }
                                | Op::Maximum { a: lhs, b: rhs } | Op::Minimum { a: lhs, b: rhs } => {
                                    ids.push(*lhs); ids.push(*rhs);
                                }
                                Op::MulScalar { input, .. } | Op::AddScalar { input, .. }
                                | Op::ReLU { input } | Op::GELU { input } | Op::SiLU { input }
                                | Op::Tanh { input } | Op::Sigmoid { input } | Op::Square { input }
                                | Op::Sqrt { input }
                                | Op::Sum { input, .. } | Op::Mean { input, .. }
                                | Op::Transpose { input } | Op::Reshape { input, .. }
                                | Op::Permute { input, .. } | Op::SumDim { input, .. }
                                | Op::SumDimKeepdim { input, .. } | Op::SumDims { input, .. }
                                | Op::Repeat { input, .. } | Op::MaxDim { input, .. }
                                | Op::Clamp { input, .. } | Op::Abs { input }
                                | Op::Log { input } | Op::Softmax { input, .. }
                                | Op::LogSoftmax { input, .. } | Op::Checkpoint { input, .. }
                                | Op::CheckpointOffload { input, .. }
                                | Op::Cast { input, .. } => {
                                    ids.push(*input);
                                }
                                Op::Conv2d { input, weight, .. } | Op::Conv2dNHWC { input, weight, .. }
                                | Op::AddBias { input, bias: weight } => {
                                    ids.push(*input); ids.push(*weight);
                                }
                                Op::Linear { input, weight, bias } => {
                                    ids.push(*input); ids.push(*weight);
                                    if let Some(b) = bias { ids.push(*b); }
                                }
                                Op::LayerNorm { input, .. } => { ids.push(*input); }
                                Op::RMSNorm { input, weight, inv_rms, .. } => {
                                    ids.push(*input); ids.push(*inv_rms);
                                    if let Some(w) = weight { ids.push(*w); }
                                }
                                Op::GroupNorm { input, weight, bias, .. } => {
                                    ids.push(*input);
                                    if let Some(w) = weight { ids.push(*w); }
                                    if let Some(b) = bias { ids.push(*b); }
                                }
                                Op::Embedding { weight, indices } | Op::IndexSelect { input: weight, indices, .. } => {
                                    ids.push(*weight); ids.push(*indices);
                                }
                                Op::Cat { inputs, .. } => { ids.extend(inputs.iter()); }
                                Op::Split { input, .. } | Op::Slice { input, .. } => { ids.push(*input); }
                                Op::Where { cond, t, f } => { ids.push(*cond); ids.push(*t); ids.push(*f); }
                                Op::MSELoss { predictions, targets, .. }
                                | Op::L1Loss { predictions, targets, .. }
                                | Op::HuberLoss { predictions, targets, .. }
                                | Op::BCELoss { predictions, targets, .. } => {
                                    ids.push(*predictions); ids.push(*targets);
                                }
                                Op::NLLLoss { log_probs, targets, .. } => {
                                    ids.push(*log_probs); ids.push(*targets);
                                }
                                Op::FlashAttention { query, key, value, mask, .. } => {
                                    ids.push(*query); ids.push(*key); ids.push(*value);
                                    if let Some(m) = mask { ids.push(*m); }
                                }
                                Op::SageAttention { query_id, key_id, value_id, .. } => {
                                    ids.push(*query_id); ids.push(*key_id); ids.push(*value_id);
                                }
                                Op::FusedSwiGLU { gate, up } => {
                                    ids.push(*gate); ids.push(*up);
                                }
                                Op::RoPePrecomputed { input, cos, sin } => {
                                    ids.push(*input); ids.push(*cos); ids.push(*sin);
                                }
                            }
                            ids
                        })
                    );
                    CompactIndex::from_tensor_ids(id_iter)
                };

                // Build set of tensor IDs that actually need gradients
                let needed_grad_ids: std::collections::HashSet<TensorId> = {
                    let mut ids = std::collections::HashSet::new();
                    ids.insert(loss.id);
                    for e in ctx.tape.iter() {
                        ids.insert(e.output_id);
                        for (tid, tensor) in &e.saved_tensors {
                            if tensor.requires_grad() {
                                ids.insert(*tid);
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
                        .map_err(|_| Error::Training("backward graph cache mutex poisoned".into()))?;
                    cache.phase(tape_len)
                } else {
                    crate::cuda_graph::BackwardPhase::Warmup
                };

                // Drain the tape — we now own all entries, lock can be released.
                let tape_entries: Vec<TapeEntry> = ctx.tape.drain(..).collect();

                (tape_entries, prev_enabled, compact_index, needed_grad_ids, use_cuda_graph, tape_len, graph_phase)
            }; // ← lock released here

            // Initialize gradient storage with compact index for O(1) Vec-based access
            let mut gradients = GradientMap::with_index(device.clone(), compact_index);
            gradients.set_ones(loss.id, loss.shape.clone())?;

            // ── CUDA Graph replay path ──────────────────────────────
            if use_cuda_graph && graph_phase == crate::cuda_graph::BackwardPhase::Replay {
                let t_replay = std::time::Instant::now();
                let stream: *mut core::ffi::c_void = core::ptr::null_mut();

                {
                    let cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                        .lock()
                        .map_err(|_| Error::Training("backward graph cache mutex poisoned".into()))?;
                    for entry in cache.grad_recipe() {
                        let grad = Tensor::zeros_dtype(
                            entry.shape.clone(),
                            entry.dtype,
                            device.clone(),
                        )?;
                        gradients.set(entry.tensor_id, grad);
                    }

                    if let Some(exec) = cache.exec() {
                        crate::cuda_graph::launch(exec, stream)?;
                        crate::cuda_graph::stream_synchronize(stream)?;
                    }
                }

                {
                    let mut cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                        .lock()
                        .map_err(|_| Error::Training("backward graph cache mutex poisoned".into()))?;
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
            let capturing = use_cuda_graph
                && graph_phase == crate::cuda_graph::BackwardPhase::Capture;
            let stream: *mut core::ffi::c_void = core::ptr::null_mut();

            if capturing {
                crate::cuda_graph::begin_capture(
                    stream,
                    crate::cuda_graph::CAPTURE_MODE_GLOBAL,
                )?;
                if profile {
                    eprintln!("[cuda_graph] began capture on default stream (tape_len={})", tape_len);
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
                for entry in tape_entries.iter().rev() {
                    if let Some(output_grad) = gradients.take(entry.output_id) {
                        let t_node = std::time::Instant::now();

                        // Compute input gradients
                        let input_grads = match compute_gradients(entry, &output_grad, &device) {
                            Ok(g) => g,
                            Err(e) => {
                                eprintln!("[bwd:ERROR] #{} {:?} shape={:?}: {e:?}",
                                    nodes_executed, std::mem::discriminant(&entry.op),
                                    output_grad.shape().dims());
                                return Err(e);
                            }
                        };
                        let kernel_dt = t_node.elapsed();
                        total_kernel_time += kernel_dt;
                        // Aggregate by op kind.
                        let disc_u64: u64 =
                            unsafe { std::mem::transmute::<_, u64>(std::mem::discriminant(&entry.op)) };
                        let e = per_op_time.entry(disc_u64).or_insert((std::time::Duration::ZERO, 0));
                        e.0 += kernel_dt;
                        e.1 += 1;

                        // Accumulate gradients (skip frozen weight IDs to save memory).
                        // Checkpoint backward returns ALL internal gradients (including
                        // LoRA params) that aren't in needed_grad_ids — always accept those.
                        let is_checkpoint = matches!(&entry.op, Op::Checkpoint { .. } | Op::CheckpointOffload { .. });
                        let t_accum = std::time::Instant::now();
                        for (tensor_id, grad) in input_grads {
                            if is_checkpoint || needed_grad_ids.contains(&tensor_id) {
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
                log::debug!("[backward] processed {} of {} entries, gradients.len()={}", nodes_executed, tape_entries.len(), gradients.len());
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
                        let mut cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                            .lock()
                            .map_err(|_| Error::Training("backward graph cache mutex poisoned".into()))?;
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
                        let mut cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                            .lock()
                            .map_err(|_| Error::Training("backward graph cache mutex poisoned".into()))?;
                        cache.invalidate();

                        eprintln!("[cuda_graph] capture failed, falling back to normal backward: {:?}", e);

                        // Re-enable autograd
                        {
                            let mut ctx = AUTOGRAD_CONTEXT
                                .lock()
                                .map_err(|_| Error::Training("autograd context mutex poisoned".into()))?;
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
                    let mut cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                        .lock()
                        .map_err(|_| Error::Training("backward graph cache mutex poisoned".into()))?;
                    cache.advance();
                    if profile {
                        let phase_name = if graph_phase == crate::cuda_graph::BackwardPhase::Warmup {
                            "warmup"
                        } else {
                            "normal"
                        };
                        eprintln!("[cuda_graph] {} step complete, next step will capture", phase_name);
                    }
                }
            }

            // Per-op-kind summary — only emitted when explicit profiling is on.
            if profile && !capturing && !per_op_time.is_empty() {
                let mut sorted: Vec<_> = per_op_time.iter().collect();
                sorted.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
                eprintln!("[bwd:per-op] top 8 by total time across {} nodes:", nodes_executed);
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
                eprintln!("║ Kernel time:      {:.3}s ({:.1}%)",
                    total_kernel_time.as_secs_f64(),
                    100.0 * total_kernel_time.as_secs_f64() / total_dt.as_secs_f64().max(1e-9));
                eprintln!("║ Accum time:       {:.3}s ({:.1}%)",
                    total_accum_time.as_secs_f64(),
                    100.0 * total_accum_time.as_secs_f64() / total_dt.as_secs_f64().max(1e-9));
                eprintln!("║ Overhead:         {:.3}s ({:.1}%)",
                    overhead.as_secs_f64(),
                    100.0 * overhead.as_secs_f64() / total_dt.as_secs_f64().max(1e-9));
                eprintln!("║ Per-node avg:     {:.3}ms",
                    total_dt.as_secs_f64() * 1000.0 / nodes_executed.max(1) as f64);
                eprintln!("╠══════════════════════════════════════════════════════════");
                eprintln!("║ Top 10 slowest ops:");
                for (i, (dt, op)) in slowest_nodes.iter().take(10).enumerate() {
                    eprintln!("║  {:2}. {:8.3}ms  {}", i + 1, dt.as_secs_f64() * 1000.0, op);
                }
                eprintln!("╚══════════════════════════════════════════════════════════\n");
            }

            // Re-enable autograd (tape was already drained at the top)
            {
                let mut ctx = AUTOGRAD_CONTEXT
                    .lock()
                    .map_err(|_| Error::Training("autograd context mutex poisoned".into()))?;
                ctx.enabled = prev_enabled;
                AUTOGRAD_ENABLED.store(prev_enabled, Ordering::Relaxed);
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
            let ctx = AUTOGRAD_CONTEXT.lock()
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
        // Disable autograd + atomic flag so tensor ops skip clone/record entirely
        {
            let mut ctx = AUTOGRAD_CONTEXT.lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.enabled = false;
        }
        AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);

        let output = f()?;

        // Re-enable autograd and record a single Checkpoint entry.
        let input_id = inputs.first()
            .ok_or_else(|| Error::InvalidInput("checkpoint requires at least one input".into()))?
            .id;

        let saved: SavedTensors = inputs.iter().map(|inp| (inp.id, inp.clone())).collect();

        let mut out_with_grad = output;
        out_with_grad.requires_grad = true;

        {
            let mut ctx = AUTOGRAD_CONTEXT.lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.enabled = true;
            AUTOGRAD_ENABLED.store(true, Ordering::Relaxed);
            ctx.checkpoint_fns.insert(out_with_grad.id, Arc::new(f));
            ctx.record(TapeEntry {
                output_id: out_with_grad.id,
                op: Op::Checkpoint {
                    input: input_id,
                    original_tape_len: 0,
                },
                saved_tensors: saved,
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
        // No pool → fall back to standard checkpoint.
        let pool_arc = match ACTIVATION_POOL.get() {
            Some(p) => Arc::clone(p),
            None => return Self::checkpoint(inputs, f),
        };

        // No autograd → just run closure (sampling path).
        let was_enabled = {
            let ctx = AUTOGRAD_CONTEXT.lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.enabled
        };
        if !was_enabled {
            return f();
        }

        // Record tape position before the forward pass.
        let tape_start = {
            let ctx = AUTOGRAD_CONTEXT.lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.tape.len()
        };

        // Run forward ONCE with autograd ENABLED — ops record to the tape.
        // This is the key difference from checkpoint(): we keep the sub-tape
        // instead of discarding it and storing a recompute closure.
        let output = f()?;

        let input_id = inputs.first()
            .ok_or_else(|| Error::InvalidInput("checkpoint_offload requires at least one input".into()))?
            .id;

        // Extract the sub-tape entries produced by the forward pass.
        let sub_tape: Vec<TapeEntry> = {
            let mut ctx = AUTOGRAD_CONTEXT.lock()
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
            let mut pool = pool_arc.lock()
                .map_err(|_| Error::Training("offload pool mutex poisoned".into()))?;

            for entry in &sub_tape {
                let mut saved_ids = SmallVec::new();
                let mut handles = SmallVec::new();
                let mut resident = SmallVec::new();

                for (tid, tensor) in &entry.saved_tensors {
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

        log::debug!("checkpoint_offload: offloaded {} sub-tape entries successfully", offloaded_entries.len());
        // Success: record a single CheckpointOffload op on the outer tape
        // with the offloaded sub-tape. No recompute closure stored.
        let mut out_with_grad = output;
        out_with_grad.requires_grad = true;

        {
            let mut ctx = AUTOGRAD_CONTEXT.lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.record(TapeEntry {
                output_id: out_with_grad.id,
                op: Op::CheckpointOffload {
                    input: input_id,
                    sub_tape: offloaded_entries,
                },
                saved_tensors: SmallVec::new(), // All saved tensors are in the sub-tape
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
) -> Result<Vec<(TensorId, Tensor)>> {
    // Keep gradients in their incoming dtype (typically F32 from GradientMap).
    // GpuOps handles mixed F32×BF16 operations internally by casting.
    // Forcing BF16 here caused overflow (Inf) in deep checkpoint backward chains.
    let output_grad = output_grad_raw;

    // Optional backtrace: print op and shapes when FLAME_BACKWARD_TRACE=1
    // Cached to avoid syscall per-op (was ~0.5ms × 800 ops = 400ms overhead)
    static TRACE_CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let trace = *TRACE_CACHED.get_or_init(|| {
        std::env::var("FLAME_BACKWARD_TRACE").ok().map(|v| v == "1").unwrap_or(false)
    });
    if trace {
        let og = output_grad.shape().dims().to_vec();
        let saved: Vec<Vec<usize>> = entry
            .saved_tensors
            .iter()
            .map(|(_, t)| t.shape().dims().to_vec())
            .collect();
        println!("[backtrace] op={:?}", entry.op);
        println!("[backtrace] out_grad={:?} saved={:?}", og, saved);
    }
    // Helper to fetch saved tensors, restoring or recomputing if checkpointed.
    // Fast path: skip the CHECKPOINT_MANAGER mutex when no checkpoints exist
    // (Relaxed atomic load ~1ns vs ~25ns for uncontended mutex lock per op).
    let mut fetch_saved = |tid: &TensorId| -> Result<Tensor> {
        if CHECKPOINT_HAS_ENTRIES.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(t) = CHECKPOINT_MANAGER
                .lock()
                .map_err(|_| Error::Training("checkpoint manager mutex poisoned".into()))?
                .fetch_saved(*tid, device)?
            {
                return Ok(t);
            }
        }
        entry
            .get_saved(tid)
            .cloned()
            .ok_or_else(|| Error::InvalidOperation("Missing saved tensor".into()))
    };

    let grads = match &entry.op {
        Op::Add { lhs, rhs, lhs_shape, rhs_shape } => {
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

            Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }

        Op::Sub { lhs, rhs } => {
            // d/dx(x-y) = 1, d/dy(x-y) = -1
            let neg_grad = GpuOps::mul_scalar(output_grad, -1.0)?;
            Ok(vec![(*lhs, output_grad.clone()), (*rhs, neg_grad)])
        }

        Op::Mul { lhs, rhs } => {
            // d/dx(x*y) = y, d/dy(x*y) = x
            static DEBUG_CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let _verbose = *DEBUG_CACHED.get_or_init(|| {
                std::env::var("DEBUG_AUTOGRAD").ok().as_deref() == Some("1")
            });
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

            Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }

        Op::MulScalar { input, scalar } => {
            // d/dx(s*x) = s
            let grad = output_grad.mul_scalar(*scalar)?;
            Ok(vec![(*input, grad)])
        }

        Op::AddScalar { input, scalar: _ } => {
            // d/dx(x+s) = 1
            Ok(vec![(*input, output_grad.clone())])
        }

        Op::MatMul { lhs, rhs } => {
            let lhs_tensor = &fetch_saved(lhs)?;
            let rhs_tensor = &fetch_saved(rhs)?;
            let og_dtype = output_grad.dtype();

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
                // grad_lhs = output_grad @ rhs^T
                let grad_lhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                    grad_bf16,
                    rhs_tensor,
                    false,
                    true,
                )?;
                // grad_rhs = lhs^T @ output_grad
                let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                    lhs_tensor,
                    grad_bf16,
                    true,
                    false,
                )?;
                return Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)]);
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
                    // grad_lhs = og @ rhs^T
                    let grad_lhs_2d = crate::ops::gemm_bf16::matmul_bf16_trans(
                        &og_bf16, rhs_tensor, false, true,
                    )?;
                    let grad_lhs = grad_lhs_2d.reshape(&[batch, m, k])?;
                    // grad_rhs = lhs^T @ og
                    let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                        &lhs_2d, &og_bf16, true, false,
                    )?;
                    return Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)]);
                }

                // Fallback for non-BF16
                let og_cast = if og_2d.dtype() != rhs_tensor.dtype() {
                    og_2d.to_dtype_no_grad(rhs_tensor.dtype())?
                } else { og_2d.clone() };
                let lhs_cast = if lhs_2d.dtype() != og_cast.dtype() {
                    lhs_2d.to_dtype_no_grad(og_cast.dtype())?
                } else { lhs_2d };

                let rhs_t = GpuOps::transpose(rhs_tensor)?;
                let grad_lhs_2d = GpuOps::matmul(&og_cast, &rhs_t)?;
                let grad_lhs = grad_lhs_2d.reshape(&[batch, m, k])?;

                let lhs_t = GpuOps::transpose(&lhs_cast)?;
                let grad_rhs = GpuOps::matmul(&lhs_t, &og_cast)?;

                return Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)]);
            }

            // 3D×3D path: lhs=[B,M,K], rhs=[B,K,N], out=[B,M,N]
            // Use matmul_bf16_trans to avoid materializing transposes.
            // grad_lhs = output_grad @ rhs^T   (trans_b=true)
            // grad_rhs = lhs^T @ output_grad   (trans_a=true)
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            if lhs_tensor.rank() == 3 && rhs_tensor.rank() == 3
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
                let grad_lhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                    grad_bf16, rhs_tensor, false, true,
                )?;
                let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                    lhs_tensor, grad_bf16, true, false,
                )?;
                return Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)]);
            }

            // Slow / fallback path (non-BF16, non-2D, mixed dtype): materialize
            // transposes and call GpuOps::matmul as before.
            let rhs_for_grad = if rhs_tensor.dtype() != og_dtype {
                let cast = rhs_tensor.to_dtype_no_grad(og_dtype)?;
                GpuOps::transpose(&cast)?
            } else if og_dtype == DType::BF16 && rhs_tensor.rank() == 2 {
                crate::bf16_elementwise::transpose2d_bf16(rhs_tensor)?
            } else {
                GpuOps::transpose(rhs_tensor)?
            };
            let grad_lhs = GpuOps::matmul(output_grad, &rhs_for_grad)?;

            let lhs_for_grad = if lhs_tensor.dtype() != og_dtype {
                let cast = lhs_tensor.to_dtype_no_grad(og_dtype)?;
                GpuOps::transpose(&cast)?
            } else if og_dtype == DType::BF16 && lhs_tensor.rank() == 2 {
                crate::bf16_elementwise::transpose2d_bf16(lhs_tensor)?
            } else {
                GpuOps::transpose(lhs_tensor)?
            };
            let grad_rhs = GpuOps::matmul(&lhs_for_grad, output_grad)?;

            Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }

        Op::ReLU { input } => {
            // d/dx ReLU(x) = 1 if x > 0, else 0
            let input_tensor = &fetch_saved(input)?;
            let grad = relu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }

        Op::GELU { input } => {
            // GELU'(x) ≈ 0.5*(1+tanh(k)) + 0.5*x*(1-tanh²(k)) * dk/dx
            // where k = sqrt(2/π) * (x + 0.044715*x³), dk/dx = sqrt(2/π)*(1+3*0.044715*x²)
            // Use GpuOps only — no high-level Tensor methods (deadlock in backward).
            let x = fetch_saved(input)?;
            let x2 = GpuOps::mul(&x, &x)?;
            let x3 = GpuOps::mul(&x2, &x)?;
            let inner = GpuOps::add(
                &x,
                &GpuOps::mul_scalar(&x3, 0.044715)?,
            )?;
            let k = GpuOps::mul_scalar(&inner, (2.0f32 / std::f32::consts::PI).sqrt())?;
            let tanh_k = GpuOps::tanh(&k)?;
            let one_plus_tanh = GpuOps::add_scalar(&tanh_k, 1.0)?;
            let tanh_k_sq = GpuOps::mul(&tanh_k, &tanh_k)?;
            let sech2 = GpuOps::add(
                &Tensor::ones_dtype(tanh_k.shape().clone(), tanh_k.dtype(), device.clone())?,
                &GpuOps::mul_scalar(&tanh_k_sq, -1.0)?,
            )?;
            let dk_dx_inner = GpuOps::add_scalar(
                &GpuOps::mul_scalar(&x2, 3.0 * 0.044715)?,
                1.0,
            )?;
            let dk_dx = GpuOps::mul_scalar(&dk_dx_inner, (2.0f32 / std::f32::consts::PI).sqrt())?;
            let term2 = GpuOps::mul(&GpuOps::mul(&x, &sech2)?, &dk_dx)?;
            let derivative = GpuOps::mul_scalar(&GpuOps::add(&one_plus_tanh, &term2)?, 0.5)?;
            let grad = GpuOps::mul(output_grad, &derivative)?;
            Ok(vec![(*input, grad)])
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
                        n, stream,
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
                        n, stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda("flame_silu_backward_f32 failed".into()));
                }
                out
            } else {
                // Fallback: mixed dtypes — cast and use f32 kernel
                let og_f32 = if output_grad.dtype() != DType::F32 { output_grad.to_dtype_no_grad(DType::F32)? } else { output_grad.clone() };
                let x_f32 = if x.dtype() != DType::F32 { x.to_dtype_no_grad(DType::F32)? } else { x };
                let mut out = Tensor::empty_dtype(x_f32.shape().clone(), DType::F32, device.clone())?;
                let status = unsafe {
                    crate::cuda::ffi::flame_silu_backward_f32(
                        tensor_raw_ptr(&og_f32)?,
                        tensor_raw_ptr(&x_f32)?,
                        tensor_raw_ptr_mut(&mut out)?,
                        n, stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda("flame_silu_backward_f32 failed".into()));
                }
                out
            };
            Ok(vec![(*input, grad)])
        }

        Op::Tanh { input } => {
            // tanh'(x) = 1 - tanh²(x)
            // Use GpuOps only — no high-level Tensor methods (deadlock in backward).
            let x = fetch_saved(input)?;
            let tanh_x = GpuOps::tanh(&x)?;
            let tanh_sq = GpuOps::mul(&tanh_x, &tanh_x)?;
            let ones = Tensor::ones_dtype(tanh_sq.shape().clone(), tanh_sq.dtype(), device.clone())?;
            let derivative = GpuOps::add(&ones, &GpuOps::mul_scalar(&tanh_sq, -1.0)?)?;
            let grad = GpuOps::mul(output_grad, &derivative)?;
            Ok(vec![(*input, grad)])
        }

        Op::Sigmoid { input } => {
            // sigmoid'(x) = sig(x) * (1 - sig(x))
            // Use GpuOps only — no high-level Tensor methods (deadlock in backward).
            let x = fetch_saved(input)?;
            let sig = GpuOps::sigmoid(&x)?;
            let ones = Tensor::ones_dtype(sig.shape().clone(), sig.dtype(), device.clone())?;
            let one_minus_sig = GpuOps::add(&ones, &GpuOps::mul_scalar(&sig, -1.0)?)?;
            let derivative = GpuOps::mul(&sig, &one_minus_sig)?;
            let grad = GpuOps::mul(output_grad, &derivative)?;
            Ok(vec![(*input, grad)])
        }

        Op::Square { input } => {
            // d/dx(x^2) = 2x
            let input_tensor = &fetch_saved(input)?;
            let two_x = GpuOps::mul_scalar(input_tensor, 2.0)?;
            let grad = GpuOps::mul(output_grad, &two_x)?;
            Ok(vec![(*input, grad)])
        }

        Op::Sqrt { input } => {
            // d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 * x^(-0.5)
            // Use the output (sqrt(x)) from saved tensor to compute: grad * 0.5 / sqrt(x)
            let input_tensor = fetch_saved(input)?;
            let sqrt_x = GpuOps::sqrt(&input_tensor)?;
            let half_inv_sqrt = GpuOps::div(
                &GpuOps::mul_scalar(output_grad, 0.5)?,
                &sqrt_x,
            )?;
            Ok(vec![(*input, half_inv_sqrt)])
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
            Ok(vec![(*input, result)])
        }

        Op::Cast { input, from, to: _ } => {
            // Gradient of cast passes through; cast grad back to input dtype
            let g = output_grad.to_dtype_no_grad(*from)?;
            Ok(vec![(*input, g)])
        }

        Op::Checkpoint { input, original_tape_len } => {
            let _ckpt_t0 = std::time::Instant::now();

            let recompute_fn = {
                let ctx = AUTOGRAD_CONTEXT.lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.checkpoint_fns.get(&entry.output_id).cloned()
                    .ok_or_else(|| Error::Training(
                        "Checkpoint backward: no recompute closure found".into()
                    ))?
            };

            let tape_start = {
                let ctx = AUTOGRAD_CONTEXT.lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.tape.len()
            };

            {
                let mut ctx = AUTOGRAD_CONTEXT.lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.enabled = true;
                AUTOGRAD_ENABLED.store(true, Ordering::Relaxed);
            }

            let _recomp_t0 = std::time::Instant::now();
            let recomputed_output = (recompute_fn)()?;
            let _recomp_dt = _recomp_t0.elapsed();

            let recomputed_tape: Vec<TapeEntry> = {
                let mut ctx = AUTOGRAD_CONTEXT.lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.enabled = false;
                AUTOGRAD_ENABLED.store(false, Ordering::Relaxed);
                if ctx.tape.len() > tape_start {
                    ctx.tape.drain(tape_start..).collect()
                } else {
                    Vec::new()
                }
            };

            // Drop the recomputed output tensor. Its GPU data is shared via
            // Arc with the tape's saved tensors; this just drops one Arc ref.
            drop(recomputed_output);

            // Build the gradient propagation filter for the sub-backward.
            //
            // sub_needed controls which tensor IDs get gradients accumulated.
            // We need:
            //   1. checkpoint input (chain gradient for upstream propagation)
            //   2. all intermediate output_ids (needed for gradient flow)
            //   3. trainable params with requires_grad (LoRA weights)
            //
            // BUT we must NOT return intermediates to the outer gradient map,
            // only the input gradient + trainable params. The sub_needed set
            // is broad (all intermediates) to allow gradient FLOW, but the
            // result filter at the end is strict.
            let sub_needed: std::collections::HashSet<TensorId> = {
                let mut s = std::collections::HashSet::new();
                s.insert(*input);
                for e in &recomputed_tape {
                    // Include intermediates for gradient flow within sub-backward
                    s.insert(e.output_id);
                    // Include trainable params
                    for (sid, st) in &e.saved_tensors {
                        if st.requires_grad() {
                            s.insert(*sid);
                        }
                    }
                }
                s
            };

            // Collect IDs of trainable parameters (requires_grad=true in saved tensors).
            // These + the checkpoint input are the ONLY gradients we return to the outer backward.
            let trainable_ids: std::collections::HashSet<TensorId> = {
                let mut s = std::collections::HashSet::new();
                s.insert(*input);
                for e in &recomputed_tape {
                    for (sid, st) in &e.saved_tensors {
                        if st.requires_grad() {
                            s.insert(*sid);
                        }
                    }
                }
                s
            };

            let _sub_bwd_t0 = std::time::Instant::now();
            // Build compact index for the sub-tape (O(1) gradient lookups
            // instead of HashMap — same optimization as the main backward).
            let sub_compact = {
                use crate::gradient::CompactIndex;
                let ids = recomputed_tape.iter().flat_map(|e| {
                    let mut ids = vec![e.output_id];
                    for (sid, _) in &e.saved_tensors { ids.push(*sid); }
                    ids
                }).chain(std::iter::once(*input));
                CompactIndex::from_tensor_ids(ids)
            };
            let mut sub_grads = crate::gradient::GradientMap::with_index(
                device.clone(), sub_compact);
            if let Some(last_entry) = recomputed_tape.last() {
                sub_grads.set(last_entry.output_id, output_grad.clone());
            }

            // ── Eager-free sub-backward ──
            // Process the sub-tape in reverse, CONSUMING each entry so its
            // saved tensors (GPU clones) are freed immediately after use.
            // This bounds peak VRAM to one entry's saved tensors + gradients,
            // not the entire sub-tape's worth (~500MB for a transformer block).
            let n_sub_entries = recomputed_tape.len();
            let mut n_bf16_casts = 0u32;
            // Reverse the tape so we can pop from the end (O(1) per entry).
            let mut reversed_tape: Vec<TapeEntry> = recomputed_tape;
            reversed_tape.reverse();
            for sub_entry in reversed_tape.drain(..) {
                if let Some(sg) = sub_grads.take(sub_entry.output_id) {
                    let input_grads = compute_gradients(&sub_entry, &sg, device)?;
                    for (tid, g) in input_grads {
                        if sub_needed.contains(&tid) {
                            if g.dtype() != DType::F32 { n_bf16_casts += 1; }
                            sub_grads.accumulate(tid, g)?;
                        }
                    }
                }
                // sub_entry is dropped here — its saved_tensors free GPU memory
            }
            drop(reversed_tape); // ensure the Vec itself is freed
            let _sub_bwd_dt = _sub_bwd_t0.elapsed();
            static CKPT_PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let ckpt_profile = *CKPT_PROFILE.get_or_init(|| {
                std::env::var("FLAME_PROFILE").ok().map(|v| v == "1").unwrap_or(false)
            });
            if ckpt_profile {
                eprintln!("[checkpoint:{}] recomp={:.1}ms sub_bwd={:.1}ms ({} entries, {} bf16_casts)",
                    entry.output_id.0,
                    _recomp_dt.as_secs_f64() * 1000.0,
                    _sub_bwd_dt.as_secs_f64() * 1000.0,
                    n_sub_entries,
                    n_bf16_casts);
            }

            // Flush flame-core's GPU allocation pool to reclaim cached
            // memory from the eager-freed saved tensors above.
            crate::cuda_alloc_pool::clear_pool_cache();

            // Return ONLY the chain gradient + trainable param gradients.
            // Intermediate chain gradients are NOT returned — they were needed
            // for gradient flow within the sub-backward but have no consumers
            // in the outer backward. Returning them would cause O(blocks) VRAM
            // growth in the outer gradient map.
            let mut result = Vec::new();
            for (tid, g) in sub_grads.drain_all()? {
                if trainable_ids.contains(&tid) {
                    result.push((tid, g));
                }
            }
            Ok(result)
        }

        Op::CheckpointOffload { input, sub_tape } => {
            // Level 2: NO recompute. Walk the stored sub-tape, pulling
            // offloaded saved tensors from CPU as needed.
            let pool_arc = ACTIVATION_POOL.get()
                .ok_or_else(|| Error::Training(
                    "CheckpointOffload backward: no offload pool set".into()
                ))?;

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
            let mut pool = pool_arc.lock()
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
                            if let Some((_, t)) = oe.resident_fallback.iter().find(|(id, _)| id == sid) {
                                saved.push((*sid, t.clone()));
                            }
                        }
                    }

                    // Build a temporary TapeEntry for compute_gradients.
                    let tmp_entry = TapeEntry {
                        output_id: oe.output_id,
                        op: oe.op.clone(),
                        saved_tensors: saved,
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

            let mut result = Vec::new();
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
                let mut d_gate_t = Tensor::empty_dtype(
                    gate_tensor.shape().clone(), DType::BF16, device.clone(),
                )?;
                let mut d_up_t = Tensor::empty_dtype(
                    up_tensor.shape().clone(), DType::BF16, device.clone(),
                )?;
                let status = unsafe {
                    crate::cuda::ffi::flame_swiglu_backward_bf16(
                        tensor_raw_ptr(output_grad)?,
                        tensor_raw_ptr(&gate_tensor)?,
                        tensor_raw_ptr(&up_tensor)?,
                        tensor_raw_ptr_mut(&mut d_gate_t)?,
                        tensor_raw_ptr_mut(&mut d_up_t)?,
                        n, stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda("flame_swiglu_backward_bf16 failed".into()));
                }
                Ok(vec![(*gate, d_gate_t), (*up, d_up_t)])
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
                let mut d_gate_t = Tensor::empty_dtype(
                    gt.shape().clone(), DType::BF16, device.clone(),
                )?;
                let mut d_up_t = Tensor::empty_dtype(
                    ut.shape().clone(), DType::BF16, device.clone(),
                )?;
                let status = unsafe {
                    crate::cuda::ffi::flame_swiglu_backward_bf16(
                        tensor_raw_ptr(&og)?,
                        tensor_raw_ptr(&gt)?,
                        tensor_raw_ptr(&ut)?,
                        tensor_raw_ptr_mut(&mut d_gate_t)?,
                        tensor_raw_ptr_mut(&mut d_up_t)?,
                        gt.shape().elem_count() as i64, stream,
                    )
                };
                if status != 0 {
                    return Err(Error::Cuda("flame_swiglu_backward_bf16 (fallback) failed".into()));
                }
                Ok(vec![(*gate, d_gate_t), (*up, d_up_t)])
            }
        }

        Op::RoPePrecomputed { input, cos, sin } => {
            // RoPE is an orthogonal rotation. Backward = apply_rope(grad, cos, -sin).
            // Use rope_halfsplit_bf16 which handles both broadcast [1, N, half]
            // and per-head [BH, N, half] cos/sin layouts (unlike rope_fused_bf16
            // which only handles broadcast).
            let grad_bf16 = if output_grad.dtype() != DType::BF16 {
                output_grad.to_dtype_no_grad(DType::BF16)?
            } else {
                output_grad.clone()
            };
            let cos_tensor = fetch_saved(cos)?;
            let sin_tensor = fetch_saved(sin)?;
            let neg_sin = GpuOps::mul_scalar(&sin_tensor, -1.0)?;
            let grad_input =
                crate::bf16_ops::rope_halfsplit_bf16(&grad_bf16, &cos_tensor, &neg_sin)?;
            Ok(vec![(*input, grad_input)])
        }

        Op::Mean { input, input_shape } => {
            // d/dx mean(x) = 1/n for each element
            let n = input_shape.elem_count() as f32;
            let grad_scaled = GpuOps::mul_scalar(output_grad, 1.0 / n)?;
            // Normalize upstream rank before GPU broadcast
            let up_ranked = expand_to_rank(&grad_scaled, input_shape.dims().len())?;
            let expanded = GpuOps::broadcast(&up_ranked, input_shape)?;
            Ok(vec![(*input, expanded)])
        }

        Op::Transpose { input } => {
            let grad = if output_grad.dtype() == DType::BF16 {
                crate::bf16_elementwise::transpose2d_bf16(output_grad)?
            } else {
                GpuOps::transpose(output_grad)?
            };
            Ok(vec![(*input, grad)])
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

            let (grad_input, grad_weight, grad_bias) =
                crate::cuda_conv2d::CudaConv2d::conv2d_backward(
                    output_grad,
                    input_tensor,
                    weight_tensor,
                    (*stride, *stride),
                    (*padding, *padding),
                )?;

            let mut grads = vec![(*input, grad_input), (*weight, grad_weight)];

            // Handle bias gradient if present
            if let Some(grad_bias) = grad_bias {
                // Check if bias was saved in the tape entry
                // The bias would be the third saved tensor if it exists
                if entry.saved_tensors.len() > 2 {
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

            let mut grads = vec![(*input, grad_input), (*weight, grad_weight)];
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
            // Extract weight and bias from saved_tensors if present.
            // Forward saves: [input, weight?, bias?, mean, rstd]
            // Weight is at index 1 if it was saved (affine LayerNorm).
            // Bias is at index 2 if both weight and bias were saved.
            let weight_tensor = if entry.saved_tensors.len() > 3 {
                // 5 entries: input, weight, bias, mean, rstd
                // OR 4 entries: input, weight, mean, rstd (weight-only, no bias)
                Some(&entry.saved_tensors[1].1)
            } else {
                None
            };
            let bias_tensor = if entry.saved_tensors.len() > 4 {
                // 5 entries: input, weight, bias, mean, rstd
                Some(&entry.saved_tensors[2].1)
            } else {
                None
            };

            let w_bf16_owned;
            let w_bf16: Option<&Tensor> = match weight_tensor {
                Some(w) if w.dtype() == DType::BF16 => Some(w),
                Some(w) => { w_bf16_owned = w.to_dtype_no_grad(DType::BF16)?; Some(&w_bf16_owned) }
                None => None,
            };
            let b_bf16_owned;
            let b_bf16: Option<&Tensor> = match bias_tensor {
                Some(b) if b.dtype() == DType::BF16 => Some(b),
                Some(b) => { b_bf16_owned = b.to_dtype_no_grad(DType::BF16)?; Some(&b_bf16_owned) }
                None => None,
            };

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
            let mut gradients = vec![(*input, grad_input)];

            // Add weight and bias gradients if they exist
            if let Some(grad_w) = grad_weight {
                // For LayerNorm with affine parameters, weight and bias are separate tensors
                // Find them in saved_tensors (they would be saved after the input tensor)
                if entry.saved_tensors.len() > 1 {
                    // Second tensor is weight
                    gradients.push((entry.saved_tensors[1].0, ensure_bf16(grad_w)?));
                }
            }
            if let Some(grad_b) = grad_bias {
                // For LayerNorm with affine parameters, bias is the third tensor
                if entry.saved_tensors.len() > 2 {
                    // Third tensor is bias
                    gradients.push((entry.saved_tensors[2].0, ensure_bf16(grad_b)?));
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
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            guard_tensor("AutogradContext::rmsnorm_backward input", input_tensor)?;

            let inv_rms_tensor = entry.get_saved(inv_rms).ok_or_else(|| {
                Error::InvalidOperation("Missing saved tensor for inv_rms".into())
            })?;

            let weight_tensor = weight.as_ref().and_then(|w| entry.get_saved(w));
            guard_optional_tensor("AutogradContext::rmsnorm_backward weight", weight_tensor)?;

            let (grad_input, grad_weight) = crate::norm::rms_norm_backward(
                output_grad,
                input_tensor,
                weight_tensor,
                inv_rms_tensor,
                normalized_shape,
            )?;

            let mut grads = vec![(*input, ensure_bf16(grad_input)?)];

            if let Some(&w_id) = weight.as_ref() {
                if let Some(gw) = grad_weight {
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
            // d/dx(Wx + b) = W^T @ grad
            // d/dW(Wx + b) = grad @ x^T
            // d/db(Wx + b) = grad
            let input_tensor = &fetch_saved(input)?;
            let weight_tensor = &fetch_saved(weight)?;
            guard_tensor("AutogradContext::linear_backward input", input_tensor)?;
            guard_tensor("AutogradContext::linear_backward weight", weight_tensor)?;

            // Gradient w.r.t. input: W^T @ grad
            let weight_t = weight_tensor.transpose()?;
            let grad_input = output_grad.matmul(&weight_t)?;

            // Gradient w.r.t. weight: grad @ input^T
            let input_t = input_tensor.transpose()?;
            let grad_weight = output_grad.transpose()?.matmul(&input_t)?.transpose()?;
            let grad_input = ensure_bf16(grad_input)?;
            let grad_weight = ensure_bf16(grad_weight)?;

            let mut grads = vec![(*input, grad_input), (*weight, grad_weight)];

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
                let grad_lhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                    grad_bf16, rhs_tensor, false, true,
                )?;
                let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                    lhs_tensor, grad_bf16, true, false,
                )?;
                return Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)]);
            }

            // Fallback for non-BF16 3D×3D: cast to BF16, use the fast path above.
            // This is rare — only hits when operands are F32/F16.
            let lhs_bf16 = lhs_tensor.to_dtype_no_grad(DType::BF16)?;
            let rhs_bf16 = rhs_tensor.to_dtype_no_grad(DType::BF16)?;
            let grad_bf16 = output_grad.to_dtype_no_grad(DType::BF16)?;
            let grad_lhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                &grad_bf16, &rhs_bf16, false, true,
            )?;
            let grad_rhs = crate::ops::gemm_bf16::matmul_bf16_trans(
                &lhs_bf16, &grad_bf16, true, false,
            )?;
            Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }

        Op::Reshape { input, .. } => {
            // Gradient of reshape is reshape of gradient back to original shape
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let grad = output_grad.reshape(input_tensor.shape().dims())?;
            Ok(vec![(*input, ensure_bf16(grad)?)])
        }

        Op::Permute { input, dims } => {
            // Gradient of permute is inverse permute
            let inverse_dims = inverse_permutation(dims);
            let grad = output_grad.permute(&inverse_dims)?;
            Ok(vec![(*input, ensure_bf16(grad)?)])
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

            Ok(vec![(*input, ensure_bf16(grad_input)?), (*bias, grad_bias)])
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
            Ok(vec![(*input, ensure_bf16(grad)?)])
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
            let grad = crate::autograd_ops_complete::clamp_backward(
                grad_bf16,
                input_tensor,
                *min,
                *max,
            )?;
            Ok(vec![(*input, ensure_bf16(grad)?)])
        }

        Op::Div { lhs, rhs, lhs_shape, rhs_shape } => {
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

            Ok(vec![
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

            Ok(vec![(*input, grad)])
        }

        Op::SumDimKeepdim { input, dim } => {
            // For sum with keepdim, gradient is broadcast back
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let input_shape = input_tensor.shape();

            // Broadcast gradient back to input shape
            let grad = output_grad.broadcast_to(input_shape)?;
            Ok(vec![(*input, ensure_bf16(grad)?)])
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
            Ok(vec![(*input, grad)])
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

            Ok(vec![(*input, ensure_bf16(grad)?)])
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

            Ok(vec![(*weight, weight_grad)])
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

            Ok(vec![(*input, grad_input)])
        }

        Op::Cat { inputs, dim } => {
            // Split gradient back to original tensors
            let mut grads = Vec::new();
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
                Ok(vec![(*input, grad_in)])
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
                    let mut expanded = Tensor::zeros_dtype(expanded_shape, grad_dtype, device.clone())?;
                    gpu_scatter_add_narrow(&tmp, &mut expanded, axis, s)?;
                    tmp = expanded;
                }
                Ok(vec![(*input, tmp)])
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
                    let mut expanded = Tensor::zeros_dtype(expanded_shape, grad_dtype, device.clone())?;
                    gpu_scatter_add_narrow(&tmp, &mut expanded, axis, s)?;
                    tmp = expanded;
                }
                Ok(vec![(*input, tmp)])
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

            Ok(vec![(*input, combined_grad)])
        }

        Op::Abs { input } => {
            // d/dx |x| = sign(x)
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let sign = input_tensor.sign()?;
            let grad = output_grad.mul(&sign)?;
            Ok(vec![(*input, grad)])
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
            Ok(vec![(*input, grad)])
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
            Ok(vec![(*input, grad)])
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
            Ok(vec![(*input, grad)])
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
            Ok(vec![(*a, grad_a), (*b, grad_b)])
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
            Ok(vec![(*a, grad_a), (*b, grad_b)])
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
            Ok(vec![(*t, grad_t), (*f, grad_f)])
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

            Ok(vec![
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

            Ok(vec![
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

            Ok(vec![
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
            Ok(vec![(*predictions, grad_predictions)])
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

            Ok(vec![(*log_probs, final_grad)])
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

            // Saved mean and variance should be in saved_tensors
            let mean = entry
                .saved_tensors
                .iter()
                .map(|(_, t)| t)
                .find(|t| t.shape().dims() == [shape[0], *num_groups])
                .ok_or_else(|| Error::InvalidOperation("Missing saved mean".into()))?;
            let var = entry
                .saved_tensors
                .iter()
                .map(|(_, t)| t)
                .skip(1)
                .find(|t| t.shape().dims() == [shape[0], *num_groups])
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

            let mut grads = vec![(*input, grad_input)];
            if let (Some(w_id), Some(gw)) = (*weight, grad_weight) {
                grads.push((w_id, gw));
            }
            if let (Some(b_id), Some(gb)) = (*bias, grad_bias) {
                grads.push((b_id, gb));
            }

            Ok(grads)
        }

        #[cfg(feature = "flash_attn")]
        Op::FlashAttention {
            query,
            key,
            value,
            mask,
            scale,
            causal: _,
        } => {
            let query_tensor = entry
                .get_saved(query)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for query".into()))?;
            let key_tensor = entry
                .get_saved(key)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for key".into()))?;
            let value_tensor = entry
                .get_saved(value)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for value".into()))?;

            // Find saved output and LSE tensors for fused backward
            let output_tensor = entry
                .saved_tensors
                .iter()
                .map(|(_, t)| t)
                .find(|t| {
                    t.shape().dims() == output_grad.shape().dims()
                        && t.id != query_tensor.id
                        && t.id != key_tensor.id
                        && t.id != value_tensor.id
                });

            // Find LSE tensor: shape [batch_heads, seq_len_q], dtype F32
            let q_dims = query_tensor.shape().dims();
            let lse_tensor = if q_dims.len() == 4 {
                let bh = q_dims[0] * q_dims[1];
                let sq = q_dims[2];
                entry.saved_tensors.iter().map(|(_, t)| t).find(|t| {
                    t.dtype() == DType::F32 && t.shape().dims() == [bh, sq]
                })
            } else {
                None
            };

            // Try fused wmma backward kernel (requires output + LSE saved from forward)
            let fused_result = if let (Some(out_t), Some(lse_t)) = (output_tensor, lse_tensor) {
                if q_dims.len() == 4
                    && query_tensor.dtype() == DType::BF16
                    && output_grad.dtype() == DType::BF16
                {
                    let (b, h, sq, hd) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
                    let sk = key_tensor.shape().dims()[2];
                    let bh = (b * h) as i32;

                    // Reshape to [BH, N, HD] for kernel
                    let q_3d = query_tensor.reshape(&[bh as usize, sq, hd])?;
                    let k_3d = key_tensor.reshape(&[bh as usize, sk, hd])?;
                    let v_3d = value_tensor.reshape(&[bh as usize, sk, hd])?;
                    let o_3d = out_t.reshape(&[bh as usize, sq, hd])?;
                    let do_3d = output_grad.reshape(&[bh as usize, sq, hd])?;

                    // Allocate BF16 output grad tensors
                    let dq = Tensor::empty_dtype(q_3d.shape().clone(), DType::BF16, device.clone())?;
                    let dk = Tensor::empty_dtype(k_3d.shape().clone(), DType::BF16, device.clone())?;
                    let dv = Tensor::empty_dtype(v_3d.shape().clone(), DType::BF16, device.clone())?;

                    // Only dQ needs FP32 staging (atomicAdd across KV-tile blocks).
                    // dK, dV are written directly as BF16 by the kernel (register accum).
                    let dq_f32_n = bh as usize * sq * hd;
                    let mut dq_f32 = crate::cuda_memory_alignment::alloc_aligned_f32(device, dq_f32_n)?;
                    device.memset_zeros(&mut dq_f32).map_err(|e| Error::Cuda(format!("{e:?}")))?;

                    use cudarc::driver::DevicePtr;
                    let stream = crate::cuda::device_lt::stream_ptr(device)?;
                    let ret = unsafe {
                        crate::cuda::ffi::flame_flash_attention_backward_bf16(
                            q_3d.as_device_ptr_bf16("fa_bwd:q")? as *const core::ffi::c_void,
                            k_3d.as_device_ptr_bf16("fa_bwd:k")? as *const core::ffi::c_void,
                            v_3d.as_device_ptr_bf16("fa_bwd:v")? as *const core::ffi::c_void,
                            o_3d.as_device_ptr_bf16("fa_bwd:o")? as *const core::ffi::c_void,
                            do_3d.as_device_ptr_bf16("fa_bwd:do")? as *const core::ffi::c_void,
                            {
                                match &lse_t.storage {
                                    crate::tensor_storage::TensorStorage::F32 { data, .. } =>
                                        *crate::tensor_storage::slice_ref(data).device_ptr()
                                            as *const core::ffi::c_void,
                                    _ => core::ptr::null(),
                                }
                            },
                            dq.as_device_ptr_bf16("fa_bwd:dq")? as *mut core::ffi::c_void,
                            dk.as_device_ptr_bf16("fa_bwd:dk")? as *mut core::ffi::c_void,
                            dv.as_device_ptr_bf16("fa_bwd:dv")? as *mut core::ffi::c_void,
                            bh,
                            sq as i32,
                            sk as i32,
                            hd as i32,
                            stream,
                            *dq_f32.device_ptr() as *mut f32,
                        )
                    };
                    if ret == 0 {
                        // Reshape back to [B, H, N, D]
                        let dq = dq.reshape(&[b, h, sq, hd])?;
                        let dk = dk.reshape(&[b, h, sk, hd])?;
                        let dv = dv.reshape(&[b, h, sk, hd])?;
                        Some((dq, dk, dv))
                    } else {
                        eprintln!("[flash_attn_bwd] fused kernel failed (ret={ret}), falling back");
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let (grad_q, grad_k, grad_v) = if let Some(triple) = fused_result {
                triple
            } else {
                // Fallback to decomposed recompute path
                let mask_tensor = if let Some(m_id) = mask {
                    entry.get_saved(m_id)
                } else {
                    None
                };
                attention_backward_recompute(
                    query_tensor, key_tensor, value_tensor,
                    output_grad, mask_tensor, *scale, None,
                )?
            };

            Ok(vec![(*query, grad_q), (*key, grad_k), (*value, grad_v)])
        }
        #[cfg(not(feature = "flash_attn"))]
        Op::FlashAttention {
            query,
            key,
            value,
            mask,
            scale,
            causal: _,
        } => {
            // Recompute SDPA backward path
            let query_tensor = entry
                .get_saved(query)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for query".into()))?;
            let key_tensor = entry
                .get_saved(key)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for key".into()))?;
            let value_tensor = entry
                .get_saved(value)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for value".into()))?;
            let mask_tensor = if let Some(m_id) = mask {
                entry.get_saved(m_id)
            } else {
                None
            };

            // Look for a cached attn tensor in saved_tensors (shape
            // [batch, heads, q_len, k_len]). Callers that pre-compute
            // softmax in forward append it to saved_tensors; the rest
            // fall back to recomputing logits+softmax in backward.
            let expected_attn_shape = {
                let q_dims = query_tensor.shape().dims();
                let k_dims = key_tensor.shape().dims();
                if q_dims.len() == 4 && k_dims.len() == 4 {
                    Some([q_dims[0], q_dims[1], q_dims[2], k_dims[2]])
                } else {
                    None
                }
            };
            let cached_attn: Option<&Tensor> = expected_attn_shape.and_then(|s| {
                entry
                    .saved_tensors
                    .iter()
                    .map(|(_, t)| t)
                    .find(|t| t.shape().dims() == s)
            });

            let (grad_q, grad_k, grad_v) = attention_backward_recompute(
                query_tensor,
                key_tensor,
                value_tensor,
                output_grad,
                mask_tensor,
                *scale,
                cached_attn,
            )?;
            Ok(vec![(*query, grad_q), (*key, grad_k), (*value, grad_v)])
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

            Ok(vec![
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

/// ReLU backward pass
fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
    // ReLU backward: gradient passes through where input > 0
    // We need to create a mask without using tensor operations that record to autograd

    // Create zero tensor for comparison
    let zero_data = crate::tensor::alloc_zeros_from_pool(&input.device, input.shape.elem_count())?;
    let zero = Tensor {
        storage: TensorStorage::F32 {
            data: zero_data.into(),
            numel: input.shape.elem_count(),
        },
        shape: input.shape.clone(),
        device: input.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    // Use comparison to create mask
    let mask = input.gt(&zero)?;

    // Apply mask to gradient using GPU ops
    GpuOps::mul(grad_output, &mask)
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
#[inline]
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
