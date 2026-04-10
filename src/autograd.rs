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
use crate::gradient_checkpointing::{CHECKPOINT_HAS_ENTRIES, CHECKPOINT_MANAGER};
use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

lazy_static::lazy_static! {
    /// Global autograd context - thread-safe
    static ref AUTOGRAD_CONTEXT: Mutex<AutogradContextInner> = Mutex::new(AutogradContextInner::new());
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
}

/// Entry in the computation tape
#[derive(Clone)]
struct TapeEntry {
    /// Output tensor ID
    output_id: TensorId,

    /// Operation that produced the output
    op: Op,

    /// Saved tensors needed for backward pass (Vec for cache locality; most ops save 1-3 tensors)
    saved_tensors: Vec<(TensorId, Tensor)>,
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

    let code = unsafe {
        ffi::narrow_backward_scatter_add_launch(
            grad_out.cuda_ptr() as *const c_void,
            grad_in.cuda_ptr_mut() as *mut c_void,
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
fn attention_backward_recompute(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dout: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    // logits = (Q K^T) * scale [+ mask]
    let kt = k.transpose_dims(2, 3)?; // [B,H,D,Sk]
    let mut logits = q.bmm(&kt)?; // [B,H,Sq,Sk]
    logits = logits.mul_scalar(scale)?;
    if let Some(m) = mask {
        logits = logits.add(m)?;
    }

    // attn = softmax(logits)
    let attn = logits.softmax(-1)?; // [B,H,Sq,Sk]

    // dV = attn^T @ dO
    let attn_t = attn.transpose_dims(2, 3)?; // [B,H,Sk,Sq]
    let d_v = attn_t.bmm(dout)?; // [B,H,Sk,D]

    // dAttn = dO @ V^T
    let vt = v.transpose_dims(2, 3)?; // [B,H,D,Sk]
    let d_attn = dout.bmm(&vt)?; // [B,H,Sq,Sk]

    // softmax backward: (dA - sum(dA*A, -1, keepdim)) * A
    let dattn_times_attn = d_attn.mul(&attn)?;
    let sum_term = dattn_times_attn.sum_dim_keepdim(3)?; // [B,H,Sq,1]
    let d_logits = d_attn.sub(&sum_term)?.mul(&attn)?; // [B,H,Sq,Sk]

    // dQ = dLogits @ K
    let d_q = d_logits.bmm(k)?; // [B,H,Sq,D]
                                // dK = dLogits^T @ Q
    let d_k = d_logits.transpose_dims(2, 3)?.bmm(q)?; // [B,H,Sk,D]

    Ok((d_q, d_k, d_v))
}

/// Public API for autograd
pub struct AutogradContext;

impl AutogradContext {
    /// Record an operation in the computation graph
    pub fn record_op(output_id: TensorId, op: Op, saved_tensors: Vec<(TensorId, Tensor)>) {
        let mut ctx = match AUTOGRAD_CONTEXT.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };

        // Only record if autograd is enabled
        if !ctx.enabled {
            return;
        }

        // Apply checkpointing policy to saved tensors (CPU offload registration)
        {
            if let Ok(mut mgr) = CHECKPOINT_MANAGER.lock() {
                for (id, tensor) in &saved_tensors {
                    let _ = mgr.checkpoint_saved_tensor(*id, tensor);
                }
            } else {
                return;
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
                    let output_grad = output_grad.clone_result()?;

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
                    return Err(Error::InvalidOperation(format!(
                        "Op {} took too long: {:?}",
                        tape_idx, elapsed
                    )));
                }
            }

            // Clear tape and restore state
            ctx.tape.clear();
            ctx.enabled = prev_enabled;
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

        // Process tape in reverse under lock
        let gradients = {
            let mut ctx = AUTOGRAD_CONTEXT
                .lock()
                .map_err(|_| Error::Training("autograd context mutex poisoned".into()))?;

            // Disable autograd during backward pass
            let prev_enabled = ctx.enabled;
            ctx.enabled = false;

            // Build compact index from the tape for Vec-based gradient storage.
            // Collect all tensor IDs that will be accessed during backward:
            // output_ids (for gradient lookup) and input_ids from each op (for accumulation).
            let compact_index = {
                use crate::gradient::CompactIndex;
                let id_iter = std::iter::once(loss.id).chain(
                    ctx.tape.iter().flat_map(|e| {
                        let mut ids = vec![e.output_id];
                        // Also include saved tensor IDs (they receive accumulated gradients)
                        for (tid, _) in &e.saved_tensors {
                            ids.push(*tid);
                        }
                        // Extract input IDs from the op variant
                        match &e.op {
                            Op::Add { lhs, rhs, .. } | Op::Sub { lhs, rhs }
                            | Op::Mul { lhs, rhs } | Op::Div { lhs, rhs }
                            | Op::MatMul { lhs, rhs } | Op::BatchMatMul { lhs, rhs }
                            | Op::Maximum { a: lhs, b: rhs } | Op::Minimum { a: lhs, b: rhs } => {
                                ids.push(*lhs); ids.push(*rhs);
                            }
                            Op::MulScalar { input, .. } | Op::AddScalar { input, .. }
                            | Op::ReLU { input } | Op::GELU { input } | Op::SiLU { input }
                            | Op::Tanh { input } | Op::Sigmoid { input } | Op::Square { input }
                            | Op::Sum { input, .. } | Op::Mean { input, .. }
                            | Op::Transpose { input } | Op::Reshape { input, .. }
                            | Op::Permute { input, .. } | Op::SumDim { input, .. }
                            | Op::SumDimKeepdim { input, .. } | Op::SumDims { input, .. }
                            | Op::Repeat { input, .. } | Op::MaxDim { input, .. }
                            | Op::Clamp { input, .. } | Op::Abs { input }
                            | Op::Log { input } | Op::Softmax { input, .. }
                            | Op::LogSoftmax { input, .. } | Op::Checkpoint { input, .. }
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
                        }
                        ids
                    })
                );
                CompactIndex::from_tensor_ids(id_iter)
            };

            // Initialize gradient storage with compact index for O(1) Vec-based access
            let mut gradients = GradientMap::with_index(device.clone(), compact_index);
            gradients.set_ones(loss.id, loss.shape.clone())?;

            // ── CUDA Graph integration ──────────────────────────────
            //
            // When FLAME_CUDA_GRAPH=1, the backward pass follows a
            // warmup-capture-replay protocol:
            //
            //   Step 0 (warmup): normal backward. Fills the allocator
            //     pool so no cudaMalloc happens on subsequent steps.
            //   Step 1 (capture): wrap the backward kernel loop in
            //     cudaStreamBeginCapture / cudaStreamEndCapture. All
            //     kernel launches are recorded, not executed.
            //   Step 2+ (replay): launch the cached graph. All kernels
            //     replay using the same memory addresses (guaranteed by
            //     the caching allocator's exact-size matching).
            //
            // If the tape length changes between steps, the graph is
            // invalidated and re-captured on the next step.

            let use_cuda_graph = crate::cuda_graph::cuda_graph_enabled();
            let tape_len = ctx.tape.len();

            // Determine CUDA graph phase (if enabled)
            let graph_phase = if use_cuda_graph {
                let cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                    .lock()
                    .map_err(|_| Error::Training("backward graph cache mutex poisoned".into()))?;
                cache.phase(tape_len)
            } else {
                crate::cuda_graph::BackwardPhase::Warmup // unused, but needs a value
            };

            // ── Replay path: skip the entire backward loop ──────────
            //
            // On replay we need to:
            // 1. Allocate gradient tensors in the same order/sizes as capture
            //    (the caching allocator returns the same device pointers).
            // 2. Populate the GradientMap with those tensors.
            // 3. Launch the graph (which writes correct values to those buffers).
            if use_cuda_graph && graph_phase == crate::cuda_graph::BackwardPhase::Replay {
                let t_replay = std::time::Instant::now();
                let stream: *mut core::ffi::c_void = core::ptr::null_mut();

                // Reproduce the gradient allocations from capture
                {
                    let cache = crate::cuda_graph::BACKWARD_GRAPH_CACHE
                        .lock()
                        .map_err(|_| Error::Training("backward graph cache mutex poisoned".into()))?;
                    for entry in cache.grad_recipe() {
                        // Allocate a tensor with the same shape/dtype — the pool
                        // returns the same device pointer as during capture.
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

                // Advance step counter
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

                // Re-enable autograd and clear tape
                ctx.enabled = prev_enabled;
                ctx.tape.clear();

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

            // Process tape in reverse. Use `take` instead of `get + clone_result`
            // to avoid an extra GPU memcpy per entry. The gradient is consumed
            // (moved out of the map) — if a later entry re-accumulates into the
            // same tensor_id, `accumulate` handles the addition.
            let t_backward_start = std::time::Instant::now();
            let mut nodes_executed = 0usize;
            let mut total_kernel_time = std::time::Duration::ZERO;
            let mut total_accum_time = std::time::Duration::ZERO;
            let mut slowest_nodes: Vec<(std::time::Duration, String)> = Vec::new();

            let backward_result: Result<()> = (|| {
                for entry in ctx.tape.iter().rev() {
                    if let Some(output_grad) = gradients.take(entry.output_id) {
                        let t_node = std::time::Instant::now();

                        // Compute input gradients
                        let input_grads = compute_gradients(entry, &output_grad, &device)?;
                        let kernel_dt = t_node.elapsed();
                        total_kernel_time += kernel_dt;

                        // Accumulate gradients
                        let t_accum = std::time::Instant::now();
                        for (tensor_id, grad) in input_grads {
                            gradients.accumulate(tensor_id, grad)?;
                        }
                        total_accum_time += t_accum.elapsed();

                        nodes_executed += 1;

                        if profile && !capturing {
                            slowest_nodes.push((kernel_dt, format!("{:?}", entry.op)));
                        }
                    }
                }
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

                        // Re-enable autograd and clear tape
                        ctx.enabled = prev_enabled;
                        ctx.tape.clear();
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

            // Re-enable autograd
            ctx.enabled = prev_enabled;

            // Clear tape after backward pass
            ctx.tape.clear();

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
        // 1. Record the tape position before the closure
        let tape_start = {
            let ctx = AUTOGRAD_CONTEXT.lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            ctx.tape.len()
        };

        // 2. Run the closure — it appends entries to the main tape
        let output = f()?;

        // 3. Count and DROP the sub-tape entries (this frees intermediate activations!)
        let sub_tape_len = {
            let mut ctx = AUTOGRAD_CONTEXT.lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            let len = ctx.tape.len() - tape_start;
            // Drop all intermediate entries — this is the memory saving!
            ctx.tape.truncate(tape_start);
            len
        };

        // 4. Record a single Checkpoint entry on the main tape.
        let input_id = inputs.first()
            .ok_or_else(|| Error::InvalidInput("checkpoint requires at least one input".into()))?
            .id;

        let saved: Vec<(TensorId, Tensor)> = inputs.iter().map(|inp| (inp.id, inp.clone())).collect();

        let output_id = output.id;

        {
            let mut ctx = AUTOGRAD_CONTEXT.lock()
                .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
            if ctx.enabled {
                // Store the recompute closure
                ctx.checkpoint_fns.insert(output_id, Arc::new(f));
                ctx.record(TapeEntry {
                    output_id,
                    op: Op::Checkpoint {
                        input: input_id,
                        original_tape_len: sub_tape_len,
                    },
                    saved_tensors: saved,
                });
            }
        }

        Ok(output)
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
        }
    }
}

/// Compute gradients for a single operation
fn compute_gradients(
    entry: &TapeEntry,
    output_grad: &Tensor,
    device: &Arc<CudaDevice>,
) -> Result<Vec<(TensorId, Tensor)>> {
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
                output_grad.clone_result()?
            };

            let grad_rhs = if rhs_shape != output_grad.shape() {
                reduce_grad_for_broadcast(output_grad, rhs_shape)?
            } else {
                output_grad.clone_result()?
            };

            Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }

        Op::Sub { lhs, rhs } => {
            // d/dx(x-y) = 1, d/dy(x-y) = -1
            let neg_grad = GpuOps::mul_scalar(output_grad, -1.0)?;
            Ok(vec![(*lhs, output_grad.clone_result()?), (*rhs, neg_grad)])
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
            Ok(vec![(*input, output_grad.clone_result()?)])
        }

        Op::MatMul { lhs, rhs } => {
            // d/dA(A @ B) = grad @ B^T
            // d/dB(A @ B) = A^T @ grad
            let lhs_tensor = &fetch_saved(lhs)?;
            let rhs_tensor = &fetch_saved(rhs)?;

            // Gradient w.r.t. lhs: grad @ B^T
            // Use GPU ops directly to avoid autograd recording
            let rhs_t = GpuOps::transpose(rhs_tensor)?;
            let grad_lhs = GpuOps::matmul(output_grad, &rhs_t)?;

            // Gradient w.r.t. rhs: A^T @ grad
            let lhs_t = GpuOps::transpose(lhs_tensor)?;
            let grad_rhs = GpuOps::matmul(&lhs_t, output_grad)?;

            Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }

        Op::ReLU { input } => {
            // d/dx ReLU(x) = 1 if x > 0, else 0
            let input_tensor = &fetch_saved(input)?;
            let grad = relu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }

        Op::GELU { input } => {
            // Use the complete GELU backward implementation
            let input_tensor = &fetch_saved(input)?;
            let grad = crate::autograd_ops_complete::gelu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }

        Op::SiLU { input } => {
            // Use the complete SiLU backward implementation
            let input_tensor = &fetch_saved(input)?;
            let grad = crate::autograd_ops_complete::silu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }

        Op::Tanh { input } => {
            // Use the complete Tanh backward implementation
            let input_tensor = &fetch_saved(input)?;
            let output_tensor = input_tensor.tanh()?;
            let grad = crate::autograd_ops_complete::tanh_backward(output_grad, &output_tensor)?;
            Ok(vec![(*input, grad)])
        }

        Op::Sigmoid { input } => {
            // Use the complete Sigmoid backward implementation
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let output_tensor = input_tensor.sigmoid()?;
            let grad = crate::autograd_ops_complete::sigmoid_backward(output_grad, &output_tensor)?;
            Ok(vec![(*input, grad)])
        }

        Op::Square { input } => {
            // d/dx(x^2) = 2x
            let input_tensor = &fetch_saved(input)?;
            let two_x = GpuOps::mul_scalar(input_tensor, 2.0)?;
            let grad = GpuOps::mul(output_grad, &two_x)?;
            Ok(vec![(*input, grad)])
        }

        Op::Sum { input, input_shape } => {
            // Gradient of sum: broadcast grad to input shape
            // Normalize upstream rank to avoid 0-D edge cases in GPU kernels
            let up_ranked = expand_to_rank(output_grad, input_shape.dims().len())?;
            let expanded = GpuOps::broadcast(&up_ranked, input_shape)?;
            Ok(vec![(*input, expanded)])
        }

        Op::Cast { input, from, to: _ } => {
            // Gradient of cast passes through; cast grad back to input dtype
            let g = output_grad.to_dtype(*from)?;
            Ok(vec![(*input, g)])
        }

        Op::Checkpoint { input, original_tape_len } => {
            // Activation checkpointing backward:
            // 1. Re-run forward closure to rebuild the sub-tape
            // 2. Backward through the recomputed sub-tape
            // 3. Drop everything (frees recomputed intermediates)

            // Get the recompute closure from the side-channel
            let recompute_fn = {
                let ctx = AUTOGRAD_CONTEXT.lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.checkpoint_fns.get(&entry.output_id).cloned()
                    .ok_or_else(|| Error::Training(
                        "Checkpoint backward: no recompute closure found".into()
                    ))?
            };

            // Record tape position, re-run forward
            let tape_start = {
                let ctx = AUTOGRAD_CONTEXT.lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.tape.len()
            };

            // Enable autograd temporarily for recomputation
            {
                let mut ctx = AUTOGRAD_CONTEXT.lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.enabled = true;
            }

            let _recomputed_output = (recompute_fn)()?;

            // Extract the recomputed sub-tape
            let recomputed_tape: Vec<TapeEntry> = {
                let mut ctx = AUTOGRAD_CONTEXT.lock()
                    .map_err(|_| Error::Training("autograd mutex poisoned".into()))?;
                ctx.enabled = false; // re-disable
                if ctx.tape.len() > tape_start {
                    ctx.tape.drain(tape_start..).collect()
                } else {
                    Vec::new()
                }
            };

            // Backward through the recomputed sub-tape
            let mut sub_grads = crate::gradient::GradientMap::new(device.clone());
            // The recomputed output has a DIFFERENT TensorId than the original.
            // We need to seed the gradient at the recomputed output's ID.
            if let Some(last_entry) = recomputed_tape.last() {
                sub_grads.set(last_entry.output_id, output_grad.clone());
            }

            for sub_entry in recomputed_tape.iter().rev() {
                if let Some(sg) = sub_grads.take(sub_entry.output_id) {
                    let input_grads = compute_gradients(sub_entry, &sg, device)?;
                    for (tid, g) in input_grads {
                        sub_grads.accumulate(tid, g)?;
                    }
                }
            }

            // The recomputed forward used the SAME input tensors (from saved_tensors),
            // so gradients for those tensor IDs are in sub_grads.
            let mut result = Vec::new();
            for (tid, _) in &entry.saved_tensors {
                if let Some(g) = sub_grads.take(*tid) {
                    result.push((*tid, g));
                }
            }
            Ok(result)
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
            // Gradient of transpose is transpose of gradient
            let grad = GpuOps::transpose(output_grad)?;
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
            // Use the complete LayerNorm backward implementation
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            guard_tensor(
                "AutogradContext::layer_norm_backward saved input",
                input_tensor,
            )?;

            // Support for affine LayerNorm with weight and bias
            let mean = input_tensor.mean_dims(normalized_shape, true)?;
            let var = input_tensor.var_dims(normalized_shape, true, true)?;
            let normalized = input_tensor
                .sub(&mean)?
                .div(&var.add_scalar(1e-5)?.sqrt()?)?;

            // Check if weight and bias tensors were saved (affine=true)
            // For LayerNorm with affine parameters, we just need to know if they exist
            let has_affine = entry.saved_tensors.len() > 1;

            let (grad_input, grad_weight, grad_bias) =
                crate::autograd_ops_complete::layer_norm_backward(
                    output_grad,
                    input_tensor,
                    &normalized,
                    None, // weight not available in this context
                    None, // bias not available in this context
                    &mean,
                    &var,
                    normalized_shape,
                    1e-5, // eps
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
            // Similar to MatMul but preserves batch dimension
            let lhs_tensor = &fetch_saved(lhs)?;
            let rhs_tensor = &fetch_saved(rhs)?;
            guard_tensor("AutogradContext::batch_matmul lhs", lhs_tensor)?;
            guard_tensor("AutogradContext::batch_matmul rhs", rhs_tensor)?;

            // Gradient w.r.t. lhs: grad @ rhs^T (batched)
            let rhs_t = rhs_tensor.transpose_batch()?;
            let grad_lhs = ensure_bf16(output_grad.batch_matmul(&rhs_t)?)?;

            // Gradient w.r.t. rhs: lhs^T @ grad (batched)
            let lhs_t = lhs_tensor.transpose_batch()?;
            let grad_rhs = ensure_bf16(lhs_t.batch_matmul(output_grad)?)?;

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
            let grad_input = output_grad.clone_result()?;

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
            // Use the clamp backward implementation
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            guard_tensor("AutogradContext::clamp_backward input", input_tensor)?;
            let grad = crate::autograd_ops_complete::clamp_backward(
                output_grad,
                input_tensor,
                *min,
                *max,
            )?;
            Ok(vec![(*input, ensure_bf16(grad)?)])
        }

        Op::Div { lhs, rhs } => {
            // d/dx (x/y) = 1/y
            // d/dy (x/y) = -x/y^2
            let lhs_tensor = fetch_saved(lhs)?;
            let rhs_tensor = fetch_saved(rhs)?;
            guard_tensor("AutogradContext::div_backward lhs", &lhs_tensor)?;
            guard_tensor("AutogradContext::div_backward rhs", &rhs_tensor)?;

            // Gradient w.r.t. lhs: grad * (1/rhs)
            let mut grad_lhs = GpuOps::div(output_grad, &rhs_tensor)?;

            // Gradient w.r.t. rhs: grad * (-lhs/rhs^2)
            let rhs_squared = GpuOps::mul(&rhs_tensor, &rhs_tensor)?; // x^2 = x * x
            let neg_lhs = GpuOps::mul_scalar(&lhs_tensor, -1.0)?;
            let grad_rhs_term = GpuOps::div(&neg_lhs, &rhs_squared)?;
            let mut grad_rhs = GpuOps::mul(output_grad, &grad_rhs_term)?;

            // Reduce for broadcasting if shapes differ
            if grad_lhs.shape() != lhs_tensor.shape() {
                grad_lhs = reduce_grad_for_broadcast(&grad_lhs, lhs_tensor.shape())?;
            }
            if grad_rhs.shape() != rhs_tensor.shape() {
                grad_rhs = reduce_grad_for_broadcast(&grad_rhs, rhs_tensor.shape())?;
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
                max_vals.clone_result()?
            } else {
                max_vals.unsqueeze(*dim)?
            };

            // Create mask where input == max_broadcast
            let mask = input_tensor.eq(&max_broadcast)?;

            // Broadcast gradient if needed
            let grad_broadcast = if *keepdim {
                output_grad.clone_result()?
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
            let mut grad = output_grad.clone_result()?;
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

            let mut grad = output_grad.clone_result()?;
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
                weight_grad_f32.to_dtype(weight_tensor.dtype())?
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
                grad_input_f32.to_dtype(input_tensor.dtype())?
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
            // Backward for slice: prefer CUDA scatter-add when slice is a narrow along a single dim
            let device = device.clone();
            let mut grad_in = Tensor::zeros(input_shape.clone(), device.clone())?;

            // Detect single-axis narrow: exactly one dim has (start,end) not equal to (0, size)
            let in_dims = input_shape.dims();
            let mut narrow_dim: Option<(usize, usize, usize)> = None; // (dim, start, length)
            for (i, &(s, e)) in ranges.iter().enumerate() {
                let full = s == 0 && e == in_dims[i];
                if !full {
                    if narrow_dim.is_some() {
                        narrow_dim = None; // more than one axis sliced
                        break;
                    }
                    let len = e - s;
                    narrow_dim = Some((i, s, len));
                }
            }

            if let Some((dim, start, length)) = narrow_dim {
                // Use local CUDA scatter-add helper for single-axis narrow
                gpu_scatter_add_narrow(output_grad, &mut grad_in, dim, start)?;
                Ok(vec![(*input, grad_in)])
            } else if can_gpu_multi_axis(ranges, input_shape.dims()) {
                // Multi-axis: perform chained CUDA scatters in reverse axis order
                let mut tmp = output_grad.clone_result()?;
                // Collect sliced axes (dim,start,len)
                let mut axes: Vec<(usize, usize, usize)> = Vec::new();
                for (i, &(s, e)) in ranges.iter().enumerate() {
                    if !(s == 0 && e == input_shape.dims()[i]) {
                        axes.push((i, s, e - s));
                    }
                }
                // Apply expansions in reverse
                for (axis, s, len) in axes.into_iter().rev() {
                    let mut expanded_dims = tmp.shape().dims().to_vec();
                    expanded_dims[axis] = input_shape.dims()[axis];
                    let expanded_shape = crate::Shape::from_dims(&expanded_dims);
                    let mut expanded =
                        Tensor::zeros_dtype(expanded_shape, grad_in.dtype(), device.clone())?;
                    // Scatter tmp into expanded along this axis
                    gpu_scatter_add_narrow(&tmp, &mut expanded, axis, s)?;
                    tmp = expanded;
                }
                Ok(vec![(*input, tmp)])
            } else {
                // GPU fallback for multi-axis slices that failed can_gpu_multi_axis
                // (only possible with invalid ranges — forward pass should prevent this)
                let mut tmp = output_grad.clone_result()?;
                let mut axes: Vec<(usize, usize, usize)> = Vec::new();
                for (i, &(s, e)) in ranges.iter().enumerate() {
                    if !(s == 0 && e == input_shape.dims()[i]) {
                        if s > e || e > input_shape.dims()[i] {
                            return Err(Error::InvalidOperation(format!(
                                "Slice backward: invalid range dim={} start={} end={} size={}",
                                i, s, e, input_shape.dims()[i]
                            )));
                        }
                        axes.push((i, s, e - s));
                    }
                }
                for (axis, s, _len) in axes.into_iter().rev() {
                    let mut expanded_dims = tmp.shape().dims().to_vec();
                    expanded_dims[axis] = input_shape.dims()[axis];
                    let expanded_shape = crate::Shape::from_dims(&expanded_dims);
                    let mut expanded =
                        Tensor::zeros_dtype(expanded_shape, grad_in.dtype(), device.clone())?;
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
            // Use the complete softmax backward implementation
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let output = input_tensor.softmax(*dim)?;
            let grad = crate::autograd_ops_complete::softmax_backward(output_grad, &output, *dim)?;
            Ok(vec![(*input, grad)])
        }

        Op::LogSoftmax { input, dim } => {
            // Use the complete log_softmax backward implementation
            let input_tensor = entry
                .get_saved(input)
                .ok_or_else(|| Error::InvalidOperation("Missing saved tensor for input".into()))?;
            let output = input_tensor.log_softmax(*dim)?;
            let grad =
                crate::autograd_ops_complete::log_softmax_backward(output_grad, &output, *dim)?;
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
            let mask_t = cond_tensor.clone_result()?; // 1 where true
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
            let quad_grad = diff.clone_result()?;

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

            let (grad_input, grad_weight, grad_bias) =
                crate::autograd_ops_complete::group_norm_backward(
                    output_grad,
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
            // Prefer FlashAttention backward; fall back to recompute path on failure.
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

            // Try fused backward first
            let fused = (|| {
                // Find saved output (if required by fused path)
                let output_tensor = entry
                    .saved_tensors
                    .iter()
                    .map(|(_, t)| t)
                    .find(|t| {
                        t.shape().dims() == output_grad.shape().dims()
                            && t.id != query_tensor.id
                            && t.id != key_tensor.id
                            && t.id != value_tensor.id
                    })
                    .ok_or_else(|| Error::InvalidOperation("Missing saved output tensor".into()))?;
                crate::flash_attention::flash_attention_backward(
                    output_grad,
                    query_tensor,
                    key_tensor,
                    value_tensor,
                    mask_tensor,
                    output_tensor,
                    *scale,
                    false,
                )
            })();

            let (grad_q, grad_k, grad_v) = match fused {
                Ok(triple) => triple,
                Err(_) => {
                    // Fallback to recompute SDPA backward (local helper)
                    let (dq, dk, dv) = attention_backward_recompute(
                        query_tensor,
                        key_tensor,
                        value_tensor,
                        output_grad,
                        mask_tensor,
                        *scale,
                    )?;
                    (dq, dk, dv)
                }
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
            let (grad_q, grad_k, grad_v) = attention_backward_recompute(
                query_tensor,
                key_tensor,
                value_tensor,
                output_grad,
                mask_tensor,
                *scale,
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

    grads
        .into_iter()
        .map(|(id, tensor)| ensure_bf16(tensor).map(|t| (id, t)))
        .collect()
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

    // Reduce over axes where target had size 1 but grad had >1
    let mut result = grad.clone_result()?;
    let mut cur_rank = g_rank;
    let mut axis = 0usize;
    while axis < cur_rank {
        let cur_dims = result.shape().dims().to_vec();
        let tdim = padded_td[axis];
        let gdim = cur_dims[axis];
        if tdim == 1 && gdim != 1 {
            result = result.sum_dim(axis)?; // removes this axis
                                            // After removal, ranks shift left; keep axis the same
            cur_rank -= 1;
            // No axis += 1 here
        } else {
            axis += 1;
        }
    }

    // Now result has same rank as grad minus reduced axes. Reshape to padded_td (with 1s)
    // Build shape after reductions: for each axis in g_rank, keep dim if padded_td[axis] != 1 else 1
    // But since we removed axes where padded_td==1 and gdim>1, the remaining axes map to those with either padded_td!=1 or gdim==1.
    // Construct the final padded shape by iterating padded_td and taking either target dim or 1; then drop leading ones to match td rank.
    let mut final_padded: Vec<usize> = padded_td.clone();
    // Ensure total elements match before reshape to td
    result = if final_padded.len() != result.shape().dims().len() {
        // Rebuild by inserting size-1 dims where needed
        // Current result rank equals number of axes where not reduced.
        // We'll expand result by adding size-1 dims on the left until ranks match.
        let mut need = final_padded.len() - result.shape().dims().len();
        let mut shape = result.shape().dims().to_vec();
        while need > 0 {
            shape.insert(0, 1);
            need -= 1;
        }
        result.reshape(&shape)?
    } else {
        result
    };

    // Finally, reshape to exact target dims
    result.reshape(&td)
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
        return tensor.clone_result();
    }
    // Delegate to the Tensor method which uses GPU broadcast_to_impl
    tensor.broadcast_to(target_shape)
}

/// Expand a tensor by appending size-1 dimensions until it reaches target rank
/// Useful to make scalar or low-rank upstream gradients rank-compatible before GPU broadcast
fn expand_to_rank(tensor: &Tensor, target_rank: usize) -> Result<Tensor> {
    let mut dims = tensor.shape().dims().to_vec();
    if dims.len() == target_rank {
        return tensor.clone_result();
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
fn ensure_bf16(mut tensor: Tensor) -> Result<Tensor> {
    if tensor.dtype() != DType::BF16 {
        tensor = tensor.to_dtype(DType::BF16)?;
    }
    Ok(tensor)
}

#[inline]
fn guard_tensor(op: &str, tensor: &Tensor) -> Result<()> {
    if tensor.rank() == 4 {
        assert_nhwc_bf16_public(op, tensor)
    } else if tensor.dtype() != DType::BF16 {
        Err(Error::InvalidInput(format!("{op}: expected BF16 storage")))
    } else {
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
