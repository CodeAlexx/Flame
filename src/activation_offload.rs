//! Activation offload pool — push GPU activations to pinned host RAM during
//! forward, pull them back during backward.
//!
//! This is the foundation of the "offload instead of recompute" path. It
//! mirrors the CUDA event + stream pattern used by `flame-swap::FlameSwap`
//! for weights: a dedicated non-blocking transfer stream, per-slot
//! `done`/`ready` events, and zero host-side `cudaStreamSynchronize` calls
//! in the hot path.
//!
//! ## Concurrency model
//!
//! Two transfer pipelines must run concurrently with compute:
//!   1. Weight pipeline (existing FlameSwap)
//!   2. Activation pipeline (this module)
//!
//! Both own their own CUDA streams (separate from each other AND from the
//! default stream) so DtoH of activations can overlap HtoD of weights and
//! with default-stream kernels.
//!
//! ## Correctness (stream ordering, event-only)
//!
//! `push(tensor)`:
//!   1. Find a free slot (state `Idle`).
//!   2. Record `done_event[slot]` on the default stream — captures the
//!      producer kernel's progress.
//!   3. Transfer stream waits on `done_event[slot]` — the DtoH copy cannot
//!      start until the producer has finished writing the tensor.
//!   4. Enqueue `cudaMemcpyAsync(host_buf, gpu_ptr, bytes, DtoH, transfer)`.
//!   5. Mark slot `Pushed`. Return an opaque handle.
//!
//! `pull(handle)`:
//!   1. Validate handle (matches current epoch, slot is `Pushed`).
//!   2. Allocate a fresh BF16 tensor via `Tensor::empty_dtype`.
//!   3. The transfer stream already ordered the DtoH before whatever it does
//!      next, so the HtoD enqueued on the SAME transfer stream is naturally
//!      ordered after the DtoH (same-stream ordering is implicit).
//!   4. Enqueue `cudaMemcpyAsync(gpu_dst, host_buf, bytes, HtoD, transfer)`.
//!   5. Record `ready_event[slot]` on the transfer stream.
//!   6. `default_stream_wait_event(ready_event[slot])` — any subsequent
//!      default-stream consumer of the returned tensor will automatically
//!      wait on the HtoD.
//!   7. Mark slot `Idle` (frees it for reuse).
//!
//! ## Slot reuse safety
//!
//! A slot cycles `Idle → Pushed → Idle`. `push` scans for an `Idle` slot
//! starting at `next_slot` (round-robin for rough locality). If every slot
//! is `Pushed` (ring full), `push` returns `Error::InvalidOperation` instead
//! of corrupting host memory. The caller must size `num_slots` ≥ the maximum
//! number of in-flight activations between push and pull.
//!
//! `clear()` bumps the epoch, marking all outstanding handles stale, and
//! resets every slot to `Idle`. It is the caller's responsibility to ensure
//! that no in-flight HtoD is still pending; typically `clear` is called
//! between training phases (e.g. between forward+backward and sampling).
//! If you need a hard barrier, call `synchronize()` first.
//!
//! ## Dtype support
//!
//! BF16 is mandatory (Klein only uses BF16 for activations). F32 is also
//! supported because it falls out for free. Every other dtype returns
//! `Error::Unsupported`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};

use crate::pinned::{PinnedAllocFlags, PinnedHostBuffer};
use crate::tensor::Tensor;
use crate::tensor_storage::TensorStorage;
use crate::{DType, Error, Result, Shape};

// ---------------------------------------------------------------------------
// Local CUDA stream + event FFI
//
// flame-core's `cuda/ffi.rs` exposes `flame_cuda_memcpy_async` but not stream
// creation or event primitives. We declare the minimal set needed here so the
// pool is self-contained and does not require changes to the global FFI layer.
// These mirror the implementations in `flame-swap/src/ffi.rs` exactly.
// ---------------------------------------------------------------------------

extern "C" {
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
}

// Re-declare the memcpy helper so we do not have to route through
// `pinned::memcpy_async_device_to_host` (which is fine, but the explicit raw
// FFI keeps the hot path inspectable in one place).
extern "C" {
    fn flame_cuda_memcpy_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: i32, // 1=H2D, 2=D2H, 3=D2D
        stream: *mut c_void,
    ) -> i32;
}

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;
const CUDA_STREAM_NON_BLOCKING: u32 = 0x01;
const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;

fn check_cuda(code: i32, ctx: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Cuda(format!("{ctx} (cuda error {code})")))
    }
}

/// RAII CUDA stream wrapper. Non-blocking (won't serialise with the null
/// stream implicitly — we use explicit events for ordering instead).
struct TransferStream {
    raw: *mut c_void,
}

// SAFETY: CUDA stream handles are thread-safe per the CUDA runtime.
unsafe impl Send for TransferStream {}
unsafe impl Sync for TransferStream {}

impl TransferStream {
    fn new() -> Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        check_cuda(
            unsafe { cudaStreamCreateWithFlags(&mut raw, CUDA_STREAM_NON_BLOCKING) },
            "cudaStreamCreateWithFlags (activation offload)",
        )?;
        Ok(Self { raw })
    }

    #[inline]
    fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        check_cuda(
            unsafe { cudaStreamWaitEvent(self.raw, event.raw, 0) },
            "cudaStreamWaitEvent on transfer stream",
        )
    }
}

impl Drop for TransferStream {
    fn drop(&mut self) {
        unsafe {
            cudaStreamDestroy(self.raw);
        }
    }
}

/// RAII CUDA event wrapper with timing disabled (faster than the default).
struct CudaEvent {
    raw: *mut c_void,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    fn new() -> Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        check_cuda(
            unsafe { cudaEventCreateWithFlags(&mut raw, CUDA_EVENT_DISABLE_TIMING) },
            "cudaEventCreateWithFlags (activation offload)",
        )?;
        Ok(Self { raw })
    }

    fn record_default(&self) -> Result<()> {
        check_cuda(
            unsafe { cudaEventRecord(self.raw, std::ptr::null_mut()) },
            "cudaEventRecord (default stream)",
        )
    }

    fn record_on(&self, stream: &TransferStream) -> Result<()> {
        check_cuda(
            unsafe { cudaEventRecord(self.raw, stream.as_raw()) },
            "cudaEventRecord (transfer stream)",
        )
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            cudaEventDestroy(self.raw);
        }
    }
}

/// Make the default (null) stream wait on `event`. Any subsequent
/// default-stream kernel launch will block until the work that recorded the
/// event has finished.
fn default_stream_wait_event(event: &CudaEvent) -> Result<()> {
    check_cuda(
        unsafe { cudaStreamWaitEvent(std::ptr::null_mut(), event.raw, 0) },
        "cudaStreamWaitEvent (default stream)",
    )
}

// ---------------------------------------------------------------------------
// Slot state machine
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    /// Slot is free; push may reuse it.
    Idle,
    /// DtoH copy has been enqueued on the transfer stream. Pull may pick it up.
    /// Pull issues the reverse HtoD, waits with an event, then returns the slot
    /// to `Idle` — there is no separate "pulled / draining" state because the
    /// transfer stream's ordering already covers the DtoH→HtoD sequence.
    Pushed,
}

/// Per-slot metadata captured at push time, consumed at pull time.
#[derive(Clone)]
struct SlotMeta {
    shape: Shape,
    dtype: DType,
    bytes: usize,
}

// ---------------------------------------------------------------------------
// Public handle
// ---------------------------------------------------------------------------

/// Opaque handle returned by `push`, consumed by `pull`.
///
/// Handles are `Copy` so callers can stash them anywhere (tape entries,
/// Vec, closures). Each handle carries the pool's epoch at push time;
/// `clear()` bumps the epoch and makes every previously issued handle stale.
#[derive(Clone, Copy, Debug)]
pub struct OffloadHandle {
    slot: usize,
    epoch: u64,
}

impl OffloadHandle {
    #[inline]
    pub fn slot(&self) -> usize {
        self.slot
    }
}

// ---------------------------------------------------------------------------
// The pool
// ---------------------------------------------------------------------------

/// A pool of pinned host buffers for asynchronous activation offload.
///
/// Construct once at training setup (e.g. when the swap pipeline is built),
/// size `num_slots` ≥ max in-flight activations between push and pull, size
/// `max_bytes` ≥ the largest activation you plan to offload.
///
/// The pool owns one non-blocking CUDA stream. All DtoH and HtoD copies
/// issued by this pool land on that stream; ordering between push and pull
/// of the same slot is therefore implicit (same-stream FIFO). Ordering
/// against the default compute stream is explicit via per-slot events.
pub struct ActivationOffloadPool {
    device: Arc<CudaDevice>,
    /// Pinned host backing — one buffer per slot, sized to `slot_bytes`.
    host_buffers: Vec<PinnedHostBuffer<u8>>,
    /// Per-slot metadata (Some when `state == Pushed`).
    meta: Vec<Option<SlotMeta>>,
    state: Vec<SlotState>,
    /// Per-slot "producer finished" event, recorded on the default stream
    /// and waited on by the transfer stream before the DtoH fires.
    done_event: Vec<CudaEvent>,
    /// Per-slot "HtoD finished" event, recorded on the transfer stream and
    /// waited on by the default stream before any consumer touches the
    /// pulled tensor.
    ready_event: Vec<CudaEvent>,
    /// Dedicated transfer stream.
    transfer: TransferStream,
    /// Where the round-robin free-slot search starts.
    next_slot: usize,
    /// Handle epoch. Bumped by `clear()` to invalidate stale handles.
    epoch: u64,
    /// Per-slot capacity in bytes.
    slot_bytes: usize,
}

impl ActivationOffloadPool {
    /// Build a pool with `num_slots` pinned host buffers of `max_bytes` each.
    ///
    /// Memory cost: `num_slots * max_bytes` bytes of pinned host RAM. No GPU
    /// staging buffers are needed — we copy directly between the source
    /// tensor and the pinned host buffer (DtoH) or between the pinned host
    /// buffer and a freshly allocated destination tensor (HtoD).
    pub fn new(device: &Arc<CudaDevice>, num_slots: usize, max_bytes: usize) -> Result<Self> {
        if num_slots == 0 {
            return Err(Error::InvalidInput(
                "ActivationOffloadPool::new: num_slots must be > 0".into(),
            ));
        }
        if max_bytes == 0 {
            return Err(Error::InvalidInput(
                "ActivationOffloadPool::new: max_bytes must be > 0".into(),
            ));
        }

        let mut host_buffers = Vec::with_capacity(num_slots);
        let mut done_event = Vec::with_capacity(num_slots);
        let mut ready_event = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            host_buffers.push(PinnedHostBuffer::<u8>::with_capacity_elems(
                max_bytes,
                PinnedAllocFlags::DEFAULT,
            )?);
            done_event.push(CudaEvent::new()?);
            ready_event.push(CudaEvent::new()?);
        }

        let transfer = TransferStream::new()?;

        Ok(Self {
            device: Arc::clone(device),
            host_buffers,
            meta: vec![None; num_slots],
            state: vec![SlotState::Idle; num_slots],
            done_event,
            ready_event,
            transfer,
            next_slot: 0,
            epoch: 1,
            slot_bytes: max_bytes,
        })
    }

    /// Number of slots (fixed at construction).
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.host_buffers.len()
    }

    /// Per-slot capacity in bytes (fixed at construction).
    #[inline]
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Total pinned host RAM held by this pool.
    #[inline]
    pub fn host_bytes(&self) -> usize {
        self.slot_bytes * self.host_buffers.len()
    }

    /// Number of slots currently holding a pushed activation.
    pub fn in_flight(&self) -> usize {
        self.state.iter().filter(|s| **s == SlotState::Pushed).count()
    }

    /// Asynchronously copy `tensor` to a free pinned host slot on the
    /// transfer stream and return a handle. Returns immediately — no host
    /// synchronisation is performed.
    ///
    /// The caller must have guaranteed that by the time the ambient default
    /// stream reaches "now", the producer kernel for `tensor` has been
    /// submitted. We record an event on the default stream here to capture
    /// that point, then gate the transfer-stream DtoH on it.
    pub fn push(&mut self, tensor: &Tensor) -> Result<OffloadHandle> {
        // Dtype gate.
        let dtype = tensor.dtype();
        match dtype {
            DType::BF16 | DType::F32 => {}
            other => {
                return Err(Error::Unsupported(format!(
                    "ActivationOffloadPool::push: dtype {:?} not supported (BF16/F32 only)",
                    other
                )));
            }
        }

        // Byte size.
        let numel = tensor.shape().elem_count();
        let bytes = numel * dtype.size_in_bytes();
        if bytes == 0 {
            return Err(Error::InvalidInput(
                "ActivationOffloadPool::push: tensor is empty".into(),
            ));
        }
        if bytes > self.slot_bytes {
            return Err(Error::InvalidInput(format!(
                "ActivationOffloadPool::push: tensor {} bytes exceeds slot capacity {} bytes",
                bytes, self.slot_bytes
            )));
        }

        // Find a free slot. Round-robin from next_slot.
        let n = self.host_buffers.len();
        let mut chosen: Option<usize> = None;
        for step in 0..n {
            let idx = (self.next_slot + step) % n;
            if self.state[idx] == SlotState::Idle {
                chosen = Some(idx);
                break;
            }
        }
        let slot = chosen.ok_or_else(|| {
            Error::InvalidOperation(format!(
                "ActivationOffloadPool full: {} slots all in-flight. Increase num_slots.",
                n
            ))
        })?;

        // Extract source device pointer (u64) from the tensor storage.
        let src_ptr: u64 = src_device_ptr(tensor.storage_ref())?;

        // Destination host pointer.
        let dst_ptr = self.host_buffers[slot].as_ptr() as *mut c_void;

        // 1) Gate the transfer stream on the producer kernel's progress.
        //    Record a default-stream event AFTER the producer's kernel has
        //    been submitted (which is "now" from Rust's perspective) and
        //    have the transfer stream wait on it before touching the GPU.
        self.done_event[slot].record_default()?;
        self.transfer.wait_event(&self.done_event[slot])?;

        // 2) Enqueue the async DtoH copy on the transfer stream.
        let ret = unsafe {
            flame_cuda_memcpy_async(
                dst_ptr,
                src_ptr as *const c_void,
                bytes,
                CUDA_MEMCPY_D2H,
                self.transfer.as_raw(),
            )
        };
        if ret != 0 {
            return Err(Error::Cuda(format!(
                "activation offload push: cudaMemcpyAsync DtoH failed ({})",
                ret
            )));
        }

        // 3) Record slot state + metadata.
        self.meta[slot] = Some(SlotMeta {
            shape: tensor.shape().clone(),
            dtype,
            bytes,
        });
        self.state[slot] = SlotState::Pushed;

        // Advance the round-robin cursor past the chosen slot.
        self.next_slot = (slot + 1) % n;

        Ok(OffloadHandle {
            slot,
            epoch: self.epoch,
        })
    }

    /// Asynchronously copy a previously pushed activation back from pinned
    /// host RAM into a freshly allocated GPU tensor. Returns the tensor
    /// immediately — any subsequent default-stream consumer will
    /// automatically wait on the HtoD via the per-slot ready event.
    ///
    /// The slot is freed (state → `Idle`) so the next `push` may reuse it.
    pub fn pull(&mut self, handle: OffloadHandle) -> Result<Tensor> {
        if handle.epoch != self.epoch {
            return Err(Error::InvalidOperation(format!(
                "ActivationOffloadPool::pull: handle epoch {} is stale (current {}). \
                 Was clear() called between push and pull?",
                handle.epoch, self.epoch
            )));
        }
        let slot = handle.slot;
        if slot >= self.host_buffers.len() {
            return Err(Error::InvalidOperation(format!(
                "ActivationOffloadPool::pull: slot {} out of range ({} slots)",
                slot,
                self.host_buffers.len()
            )));
        }
        if self.state[slot] != SlotState::Pushed {
            return Err(Error::InvalidOperation(format!(
                "ActivationOffloadPool::pull: slot {} state is {:?}, expected Pushed. \
                 Double pull, or handle reuse after clear?",
                slot, self.state[slot]
            )));
        }

        let meta = self.meta[slot]
            .take()
            .expect("meta present when state == Pushed");

        // Allocate a fresh destination tensor. This fully owns its GPU
        // storage (unlike FlameSwap's view_from_buffer path, which returns
        // non-owning views into a shared buffer). We do this because:
        //   a) The pool does not own per-slot GPU staging buffers (we save
        //      GPU RAM relative to the alternative).
        //   b) The returned tensor needs to outlive the caller's backward
        //      pass through autograd, long after the pool may have reused
        //      the slot.
        let mut out = Tensor::empty_dtype(meta.shape.clone(), meta.dtype, Arc::clone(&self.device))?;

        // Get the destination device pointer.
        let dst_ptr: u64 = dst_device_ptr(out.storage_mut())?;

        // Source host pointer (already written by the DtoH on the transfer
        // stream; same-stream ordering guarantees the HtoD below sees it).
        let src_ptr = self.host_buffers[slot].as_ptr() as *const c_void;

        // Enqueue the async HtoD copy on the SAME transfer stream. Because
        // the DtoH (recorded at push time) and this HtoD land on the same
        // stream, CUDA guarantees the HtoD will not start until the DtoH
        // has finished writing the host buffer. No extra event needed for
        // the DtoH→HtoD ordering.
        let ret = unsafe {
            flame_cuda_memcpy_async(
                dst_ptr as *mut c_void,
                src_ptr,
                meta.bytes,
                CUDA_MEMCPY_H2D,
                self.transfer.as_raw(),
            )
        };
        if ret != 0 {
            // Restore meta so the slot is not leaked in a weird half-state.
            self.meta[slot] = Some(meta);
            return Err(Error::Cuda(format!(
                "activation offload pull: cudaMemcpyAsync HtoD failed ({})",
                ret
            )));
        }

        // Publish "data ready on GPU" for default-stream consumers.
        self.ready_event[slot].record_on(&self.transfer)?;
        default_stream_wait_event(&self.ready_event[slot])?;

        // Slot is now free. The host buffer is no longer read (the HtoD is
        // the last op that touches it on the transfer stream, and any
        // subsequent `push` that chooses this slot will re-gate on a fresh
        // `done_event` record — so host memory is safe to overwrite on the
        // NEXT push because the transfer stream will natively order past
        // its own HtoD before firing the next DtoH).
        self.state[slot] = SlotState::Idle;

        Ok(out)
    }

    /// Drop every slot back to `Idle` and invalidate all outstanding handles.
    ///
    /// This does NOT host-synchronise. If there is a risk that an in-flight
    /// HtoD is still reading a pinned buffer when a subsequent caller
    /// reuses the slot, you must issue a `cudaDeviceSynchronize` or similar
    /// barrier yourself before calling `clear`.
    ///
    /// Typical use: between training phases (forward+backward → sampling)
    /// where the backward has already pulled every push.
    pub fn clear(&mut self) {
        for s in self.state.iter_mut() {
            *s = SlotState::Idle;
        }
        for m in self.meta.iter_mut() {
            *m = None;
        }
        self.next_slot = 0;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Skip 0 so we can use it as a sentinel if needed later.
            self.epoch = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Storage pointer extraction
// ---------------------------------------------------------------------------

/// Extract a raw device pointer (u64) from a source tensor's storage for
/// asynchronous copying. Supports BF16 (in all three backing variants:
/// owning u16 slice, arena view, non-owning view) and F32. Any other dtype
/// returns `Error::Unsupported`.
fn src_device_ptr(storage: &TensorStorage) -> Result<u64> {
    match storage {
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16 { data, .. } => {
            use crate::tensor_storage::slice_ref;
            Ok(*slice_ref(data).device_ptr())
        }
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16Arena { ptr, .. } => Ok(ptr.as_ptr() as u64),
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16View { ptr, .. } => Ok(ptr.as_ptr() as u64),
        #[cfg(not(feature = "bf16_u16"))]
        TensorStorage::BF16 { data, .. } => {
            use crate::tensor_storage::slice_ref;
            Ok(*slice_ref(data).device_ptr())
        }
        TensorStorage::F32 { data, .. } => {
            use crate::tensor_storage::slice_ref;
            Ok(*slice_ref(data).device_ptr())
        }
        TensorStorage::F16 { .. } => Err(Error::Unsupported(
            "ActivationOffloadPool: F16 source not supported".into(),
        )),
        TensorStorage::I8 { .. }
        | TensorStorage::I32 { .. }
        | TensorStorage::Bool { .. } => Err(Error::Unsupported(
            "ActivationOffloadPool: integer/bool dtypes not supported".into(),
        )),
    }
}

/// Extract a raw mutable device pointer from a freshly allocated
/// destination tensor. The destination is always produced by
/// `Tensor::empty_dtype`, which yields a fully owning BF16 or F32 storage
/// (never a view/arena), so the slice accessors apply directly.
fn dst_device_ptr(storage: &mut TensorStorage) -> Result<u64> {
    match storage {
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16 { data, .. } => {
            use crate::tensor_storage::ensure_unique_slice;
            let slice = ensure_unique_slice(data)?;
            Ok(*slice.device_ptr_mut())
        }
        #[cfg(not(feature = "bf16_u16"))]
        TensorStorage::BF16 { data, .. } => {
            use crate::tensor_storage::ensure_unique_slice;
            let slice = ensure_unique_slice(data)?;
            Ok(*slice.device_ptr_mut())
        }
        TensorStorage::F32 { data, .. } => {
            use crate::tensor_storage::ensure_unique_slice;
            let slice = ensure_unique_slice(data)?;
            Ok(*slice.device_ptr_mut())
        }
        _ => Err(Error::Unsupported(
            "ActivationOffloadPool::dst_device_ptr: unexpected destination storage".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    /// Round-trip a small BF16 tensor through the pool and verify the
    /// contents match bit-for-bit. Requires a real CUDA device (matches
    /// the pattern used by other flame-core tests under `src/tests/`).
    #[test]
    fn push_pull_round_trip_bf16() -> Result<()> {
        let dev = Device::cuda(0)?;
        let device = dev.cuda_device().clone();

        // Pool sized for two in-flight BF16 tensors of up to 1 KiB each.
        let mut pool = ActivationOffloadPool::new(&device, 2, 1024)?;
        assert_eq!(pool.num_slots(), 2);
        assert_eq!(pool.slot_bytes(), 1024);
        assert_eq!(pool.host_bytes(), 2048);

        // Source BF16 tensor with distinctive values.
        let n = 32usize;
        let src_f32: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125 - 1.0).collect();
        let src = Tensor::from_vec_dtype(
            src_f32.clone(),
            Shape::from_dims(&[4, 8]),
            device.clone(),
            DType::BF16,
        )?;
        assert_eq!(src.dtype(), DType::BF16);

        // Read what the BF16-quantised source actually holds (ground truth
        // for the round-trip comparison — BF16 is lossy wrt the original f32
        // inputs but should be preserved through the memcpy cycle).
        let ground_truth = src.to_vec()?;

        // Push → pull.
        let h = pool.push(&src)?;
        assert_eq!(pool.in_flight(), 1);
        let pulled = pool.pull(h)?;
        assert_eq!(pool.in_flight(), 0);

        assert_eq!(pulled.dtype(), DType::BF16);
        assert_eq!(pulled.shape().dims(), &[4, 8]);

        let after = pulled.to_vec()?;
        assert_eq!(after.len(), ground_truth.len());
        for (i, (a, b)) in ground_truth.iter().zip(after.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "mismatch at index {i}: before={a}, after={b}",
            );
        }
        Ok(())
    }

    /// Two concurrent pushes should use distinct slots, and LIFO pulls
    /// should work (simulates autograd's reverse-order consumption).
    #[test]
    fn push_pull_lifo_two_slots_bf16() -> Result<()> {
        let dev = Device::cuda(0)?;
        let device = dev.cuda_device().clone();
        let mut pool = ActivationOffloadPool::new(&device, 2, 2048)?;

        let a = Tensor::from_vec_dtype(
            vec![1.0f32, 2.0, 3.0, 4.0],
            Shape::from_dims(&[4]),
            device.clone(),
            DType::BF16,
        )?;
        let b = Tensor::from_vec_dtype(
            vec![-1.0f32, -2.0, -3.0, -4.0],
            Shape::from_dims(&[4]),
            device.clone(),
            DType::BF16,
        )?;
        let a_gt = a.to_vec()?;
        let b_gt = b.to_vec()?;

        let ha = pool.push(&a)?;
        let hb = pool.push(&b)?;
        assert_ne!(ha.slot(), hb.slot());
        assert_eq!(pool.in_flight(), 2);

        // LIFO: pull the most recently pushed first.
        let pb = pool.pull(hb)?;
        let pa = pool.pull(ha)?;
        assert_eq!(pool.in_flight(), 0);

        assert_eq!(pb.to_vec()?, b_gt);
        assert_eq!(pa.to_vec()?, a_gt);
        Ok(())
    }

    /// Filling every slot and then trying another push must return an error,
    /// not silently overwrite.
    #[test]
    fn push_when_full_errors() -> Result<()> {
        let dev = Device::cuda(0)?;
        let device = dev.cuda_device().clone();
        let mut pool = ActivationOffloadPool::new(&device, 1, 512)?;

        let a = Tensor::from_vec_dtype(
            vec![1.0f32; 8],
            Shape::from_dims(&[8]),
            device.clone(),
            DType::BF16,
        )?;
        let _ha = pool.push(&a)?;

        let b = Tensor::from_vec_dtype(
            vec![2.0f32; 8],
            Shape::from_dims(&[8]),
            device.clone(),
            DType::BF16,
        )?;
        let err = pool.push(&b);
        assert!(err.is_err(), "second push into 1-slot pool should fail");
        Ok(())
    }

    /// `clear` should free every slot and invalidate existing handles.
    #[test]
    fn clear_invalidates_handles() -> Result<()> {
        let dev = Device::cuda(0)?;
        let device = dev.cuda_device().clone();
        let mut pool = ActivationOffloadPool::new(&device, 2, 256)?;

        let a = Tensor::from_vec_dtype(
            vec![5.0f32; 4],
            Shape::from_dims(&[4]),
            device.clone(),
            DType::BF16,
        )?;
        let h = pool.push(&a)?;
        assert_eq!(pool.in_flight(), 1);

        pool.clear();
        assert_eq!(pool.in_flight(), 0);

        let err = pool.pull(h);
        assert!(err.is_err(), "pull after clear must fail");
        Ok(())
    }
}
