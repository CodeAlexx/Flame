//! Multi-device dispatch context.
//!
//! Per `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §8 (charter decision
//! 2026-05-13) and §recommended-change 16: the v0 engine **must not**
//! hardcode single-stream / single-device assumptions. Stream and device
//! are explicit parameters at every dispatch point. Phase 1 only needs
//! to ship the **surface** — feature-complete NCCL/DDP plumbing lands
//! in a separate ~3-week workstream after v2.
//!
//! The shape chosen here is an explicit `DispatchCtx` parameter passed
//! to `GradFn::apply` and `InputBuffer::add`, **not** a thread-local
//! stream context. PyTorch uses thread-local in `at::cuda::getCurrentCUDAStream`
//! because Python is multi-threaded with the GIL. flame-core is
//! single-threaded inside the engine (per Phase 2 design — `GraphRoot`
//! drives a ready queue serially), so an explicit ctx is cheaper and
//! makes the surface visible at the trait level. A future Phase 2+
//! revisit may layer a thread-local convenience on top if multi-engine
//! becomes a real need, but the trait shape stays explicit.
//!
//! Phase 1 stores only `(Device, raw_stream_ptr)`. The raw pointer is
//! today always `null` (default stream) because flame-core's existing
//! `Device::cuda_stream_raw_ptr()` is hard-coded null until non-default
//! streams are plumbed; that does NOT make the parameter useless — it
//! locks the trait shape so v2 won't need a breaking ABI change when
//! the first non-default stream lands.

use crate::device::Device;
use std::os::raw::c_void;

/// (device, stream) pair. Cheap to clone — `Device` is an `Arc` wrapper.
///
/// `stream` is a raw `cudaStream_t` pointer (null = default stream). It
/// is `*mut c_void` rather than a `cudarc::driver::CudaStream` because
/// flame-core's existing FFI launch wrappers already take the raw
/// pointer form (see `Device::cuda_stream_raw_ptr`). Wrapping in a
/// typed stream object can land later without changing the trait
/// surface — the raw pointer is the storage shape.
#[derive(Clone, Debug)]
pub struct DeviceStream {
    pub device: Device,
    /// Raw `cudaStream_t`. `null_mut` = default stream. Never
    /// dereferenced here; passed through to FFI launchers.
    pub stream: *mut c_void,
}

// Safety: the raw stream pointer is owned by the CUDA driver, not by
// `DeviceStream`. We treat it as a handle (like a file descriptor),
// not as a Rust pointer. The engine is single-threaded; the trait
// requires `Send + Sync` for `GradFn`, and `DispatchCtx` is passed by
// `&` to `apply()`, so cross-thread access is not a concern at this
// surface — but we mark Send/Sync explicitly so the trait bounds compose.
unsafe impl Send for DeviceStream {}
unsafe impl Sync for DeviceStream {}

impl DeviceStream {
    /// Build a (device, default-stream) pair. The most common shape in
    /// Phase 1 because non-default streams aren't plumbed yet.
    pub fn default_stream(device: Device) -> Self {
        Self {
            device,
            stream: std::ptr::null_mut(),
        }
    }
}

/// Dispatch context passed to every `GradFn::apply` and every
/// `InputBuffer::add`. Phase 1 carried only the active `DeviceStream`;
/// Phase 3a adds `create_graph` so per-node accumulation paths
/// (`AccumulateGrad::apply`, `InputBuffer::add`) can pick recording vs
/// in-place behavior off a single ctx-threaded flag. Engine populates
/// `create_graph` from `GraphRoot::create_graph` at execute start.
///
/// Future fields may include a workspace allocator handle, a graph
/// capture handle, or an NCCL communicator handle for DDP. Adding a
/// field is non-breaking because the type is passed by `&` everywhere
/// and is constructed at the engine boundary.
#[derive(Clone, Debug)]
pub struct DispatchCtx {
    pub stream: DeviceStream,
    /// True iff backward should record its own ops onto the v2 tape
    /// (i.e. higher-order gradients are wanted). When true,
    /// `AccumulateGrad::apply` takes its recording out-of-place path
    /// (so the new add itself becomes a v2-recorded op) instead of the
    /// inference-fast in-place path; `InputBuffer::add` similarly.
    ///
    /// Phase 3a default: `false`. Phase 3a's `AccumulateGrad`
    /// out-of-place branch routes through `ops::add::add_v2` so the
    /// resulting compound graph is differentiable a second time.
    pub create_graph: bool,
}

impl DispatchCtx {
    pub fn new(stream: DeviceStream) -> Self {
        Self {
            stream,
            create_graph: false,
        }
    }

    /// Convenience: build a ctx for `(device, default-stream)`.
    pub fn default_for(device: Device) -> Self {
        Self::new(DeviceStream::default_stream(device))
    }

    pub fn with_create_graph(mut self, b: bool) -> Self {
        self.create_graph = b;
        self
    }

    pub fn device(&self) -> &Device {
        &self.stream.device
    }
}
