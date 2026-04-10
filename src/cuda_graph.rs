//! CUDA Graph capture and replay for the autograd backward pass.
//!
//! CUDA Graphs record a sequence of kernel launches once, then replay
//! the entire sequence with a single `cudaGraphLaunch` call. This
//! eliminates per-kernel launch overhead (~5-10us x 2000 ops = 10-20ms)
//! which dominates wall time when individual kernels are fast.
//!
//! # Usage
//!
//! Gated behind `FLAME_CUDA_GRAPH=1` environment variable.
//!
//! The protocol is warmup-then-capture:
//!   1. Step 1: normal backward (fills the allocator pool so no cudaMalloc
//!      happens on subsequent steps).
//!   2. Step 2: capture the backward pass into a CUDA graph.
//!   3. Step 3+: replay the captured graph.
//!
//! If the tape structure changes (different number of entries), the
//! captured graph is invalidated and re-captured on the next step.

use crate::cuda::ffi;
use core::ffi::c_void;
use std::sync::OnceLock;

// ── Environment flag ────────────────────────────────────────────────

/// Returns true if `FLAME_CUDA_GRAPH=1` is set.
#[inline]
pub fn cuda_graph_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_CUDA_GRAPH").ok().as_deref() == Some("1"))
}

// ── RAII wrappers ───────────────────────────────────────────────────

/// Owns a `cudaGraph_t` handle. Calls `cudaGraphDestroy` on drop.
pub struct CudaGraph {
    raw: *mut c_void,
}

// SAFETY: cudaGraph_t is not inherently thread-bound; the CUDA runtime
// serialises access internally. We only use it from a single thread in
// practice (the backward pass), but Send/Sync are required by Mutex.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::cudaGraphDestroy(self.raw);
            }
        }
    }
}

/// Owns a `cudaGraphExec_t` handle. Calls `cudaGraphExecDestroy` on drop.
pub struct CudaGraphExec {
    raw: *mut c_void,
}

unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::cudaGraphExecDestroy(self.raw);
            }
        }
    }
}

// ── Capture mode constants ──────────────────────────────────────────

/// `cudaStreamCaptureModeGlobal` — any kernel launch on any thread
/// during capture that is not on a captured stream will error.
pub const CAPTURE_MODE_GLOBAL: i32 = 0;

// ── High-level API ──────────────────────────────────────────────────

/// Begin capturing kernel launches on the given stream.
///
/// All kernel launches on `stream` after this call are recorded into
/// a graph instead of being executed. Call [`end_capture`] to finish
/// recording and obtain the graph.
///
/// # Safety
/// - `stream` must be a valid CUDA stream pointer (null = default stream).
/// - No `cudaMalloc` / `cudaFree` may occur during capture (use the
///   caching allocator, which must be warmed up beforehand).
pub fn begin_capture(stream: *mut c_void, mode: i32) -> crate::Result<()> {
    let status = unsafe { ffi::cudaStreamBeginCapture(stream, mode) };
    if status != 0 {
        return Err(crate::Error::KernelError(format!(
            "cudaStreamBeginCapture failed with status {}",
            status
        )));
    }
    Ok(())
}

/// End capture on `stream` and return the recorded graph.
pub fn end_capture(stream: *mut c_void) -> crate::Result<CudaGraph> {
    let mut graph_ptr: *mut c_void = std::ptr::null_mut();
    let status = unsafe { ffi::cudaStreamEndCapture(stream, &mut graph_ptr) };
    if status != 0 {
        return Err(crate::Error::KernelError(format!(
            "cudaStreamEndCapture failed with status {}",
            status
        )));
    }
    if graph_ptr.is_null() {
        return Err(crate::Error::KernelError(
            "cudaStreamEndCapture returned null graph".into(),
        ));
    }
    Ok(CudaGraph { raw: graph_ptr })
}

/// Instantiate a captured graph into an executable form.
pub fn instantiate(graph: &CudaGraph) -> crate::Result<CudaGraphExec> {
    let mut exec_ptr: *mut c_void = std::ptr::null_mut();
    let status = unsafe {
        ffi::cudaGraphInstantiate(
            &mut exec_ptr,
            graph.raw,
            std::ptr::null_mut(), // error_node (debug)
            std::ptr::null_mut(), // log buffer
            0,                    // log size
        )
    };
    if status != 0 {
        return Err(crate::Error::KernelError(format!(
            "cudaGraphInstantiate failed with status {}",
            status
        )));
    }
    if exec_ptr.is_null() {
        return Err(crate::Error::KernelError(
            "cudaGraphInstantiate returned null exec".into(),
        ));
    }
    Ok(CudaGraphExec { raw: exec_ptr })
}

/// Launch a previously instantiated graph on `stream`.
pub fn launch(exec: &CudaGraphExec, stream: *mut c_void) -> crate::Result<()> {
    let status = unsafe { ffi::cudaGraphLaunch(exec.raw, stream) };
    if status != 0 {
        return Err(crate::Error::KernelError(format!(
            "cudaGraphLaunch failed with status {}",
            status
        )));
    }
    Ok(())
}

/// Synchronize a CUDA stream (wait for all pending work to complete).
pub fn stream_synchronize(stream: *mut c_void) -> crate::Result<()> {
    let status = unsafe { ffi::cudaStreamSynchronize(stream) };
    if status != 0 {
        return Err(crate::Error::KernelError(format!(
            "cudaStreamSynchronize failed with status {}",
            status
        )));
    }
    Ok(())
}

// ── Backward graph cache ────────────────────────────────────────────

use crate::tensor::TensorId;
use crate::{DType, Shape};

/// Records the allocation pattern for a single gradient tensor.
/// Used to reproduce the exact same allocations on replay so the
/// caching allocator returns the same device pointers.
#[derive(Clone, Debug)]
pub struct GradAllocEntry {
    pub tensor_id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
}

/// Cached graph state for the backward pass, stored globally.
///
/// The backward graph is valid as long as the tape structure is
/// identical (same number of entries = same ops, same shapes).
/// If the tape length changes, we invalidate and re-capture.
pub struct BackwardGraphCache {
    /// The instantiated executable graph, if captured.
    exec: Option<CudaGraphExec>,
    /// Tape length at capture time — used for invalidation.
    tape_len: usize,
    /// Step counter: 0 = warmup, 1 = capture, 2+ = replay.
    step: usize,
    /// Gradient allocation recipe from capture step.
    /// On replay, we allocate these in order to reproduce the same
    /// device pointers, then populate the GradientMap before launching.
    grad_recipe: Vec<GradAllocEntry>,
}

impl BackwardGraphCache {
    pub const fn new() -> Self {
        Self {
            exec: None,
            tape_len: 0,
            step: 0,
            grad_recipe: Vec::new(),
        }
    }

    /// Returns the current phase for the backward pass.
    ///
    /// - `Warmup`: run backward normally (step 0, fills alloc pool).
    /// - `Capture`: begin graph capture, run backward, end capture (step 1).
    /// - `Replay`: launch the cached graph (step 2+, if tape matches).
    pub fn phase(&self, current_tape_len: usize) -> BackwardPhase {
        if self.step == 0 {
            return BackwardPhase::Warmup;
        }
        if self.step == 1 {
            return BackwardPhase::Capture;
        }
        // step >= 2: replay if tape structure matches
        if let Some(_exec) = &self.exec {
            if self.tape_len == current_tape_len {
                BackwardPhase::Replay
            } else {
                // Tape changed — need to re-capture
                BackwardPhase::Capture
            }
        } else {
            BackwardPhase::Capture
        }
    }

    /// Advance the step counter after a backward pass.
    pub fn advance(&mut self) {
        self.step += 1;
    }

    /// Store the captured graph executable, tape length, and gradient recipe.
    pub fn store(
        &mut self,
        exec: CudaGraphExec,
        tape_len: usize,
        grad_recipe: Vec<GradAllocEntry>,
    ) {
        self.exec = Some(exec);
        self.tape_len = tape_len;
        self.grad_recipe = grad_recipe;
    }

    /// Invalidate the cached graph (e.g., on tape structure change).
    pub fn invalidate(&mut self) {
        self.exec = None;
        self.tape_len = 0;
        self.grad_recipe.clear();
        // Reset step to 1 so next call captures instead of warming up again
        // (the pool is already warm from previous steps).
        self.step = 1;
    }

    /// Get the cached executable for replay.
    pub fn exec(&self) -> Option<&CudaGraphExec> {
        self.exec.as_ref()
    }

    /// Get the gradient allocation recipe.
    pub fn grad_recipe(&self) -> &[GradAllocEntry] {
        &self.grad_recipe
    }
}

/// Phase of the backward graph lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardPhase {
    /// Step 0: run backward normally to warm up the allocator pool.
    Warmup,
    /// Step 1 (or re-capture): record kernels into a CUDA graph.
    Capture,
    /// Step 2+: replay the cached graph.
    Replay,
}

/// Global backward graph cache, protected by a mutex.
///
/// This is safe because `AutogradContext::backward` already holds the
/// `AUTOGRAD_CONTEXT` mutex for the duration of the backward pass,
/// ensuring single-threaded access.
use std::sync::Mutex;

lazy_static::lazy_static! {
    pub static ref BACKWARD_GRAPH_CACHE: Mutex<BackwardGraphCache> =
        Mutex::new(BackwardGraphCache::new());
}
