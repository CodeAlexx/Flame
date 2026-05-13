//! Double-buffered block weight offloader for training and inference.
//!
//! Loads all block weights into CUDA-pinned CPU memory at init. Two GPU-side
//! buffer slots enable prefetch overlap: while compute runs on block N,
//! block N+1 is being H2D-copied on a dedicated transfer stream.
//!
//! Replaces FlameSwap entirely for training. No file I/O on the hot path.
//!
//! ## FP8-pinned mode
//!
//! Set `BLOCKOFF_FP8_PINNED=1` to keep `F8_E4M3` tensors as raw FP8 bytes in
//! pinned memory instead of dequantizing them to BF16 at load. This halves
//! pinned RAM for any FP8 checkpoint (the Wan 2.2 T2V-A14B experts drop from
//! ~28 GB each to ~14 GB each, enough to fit both experts in 62 GB system
//! RAM). GPU-side dequant happens inside `prepare_weights`, so the returned
//! tensors are BF16 exactly as before — callers don't change.
//!
//! ## Submodules (Phase 1 FlexTensor port, 2026-05-12)
//!
//! * [`telemetry`] — per-prefetch / per-step counters and an opt-in event
//!   ring buffer. Disabled-mode hooks are a single relaxed atomic load.
//!   Lifted from the measurement-and-observation half of FlexTensor's
//!   `instrumentation/` package. Activated via
//!   `FLAME_OFFLOAD_TELEMETRY={on,trace}` env var, or by calling
//!   [`telemetry::global()`].`set_enabled(true)`.
//! * [`transfer_benchmark`] — one-time PCIe H2D/D2H bandwidth sweep used
//!   to build a [`transfer_benchmark::TransferBandwidthProfile`] for
//!   future strategy work (Phase 2). Not on the per-step path.

pub mod planner;
pub mod strategy;
pub mod telemetry;
pub mod transfer_benchmark;

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::offload::strategy::{AccessHints, OffloaderState, Strategy, ACCESS_HISTORY_CAP};

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr};
use crate::{
    memcpy_async_host_to_device, DType, PinnedAllocFlags, PinnedHostBuffer, Shape, Tensor,
};

// ---------------------------------------------------------------------------
// Phase 0 (block-offloader event safety): CUDA event helpers
//
// Two events per slot:
//   * `compute_done` — recorded on the default stream when a `BlockHandle` is
//     dropped. Any subsequent prefetch into the same slot must wait on this
//     event (via `cudaStreamWaitEvent` on the transfer stream) before the
//     H2D begins, so kernels still queued for the previous block can finish
//     reading the slot's storage before it is overwritten.
//   * `h2d_done` — recorded on the transfer stream after the per-tensor
//     `memcpy_async`es for a block complete. The default stream waits on
//     this event before `prepare_weights` runs, replacing the old
//     CPU-side `cudaStreamSynchronize` so compute and H2D can overlap on
//     the GPU instead of round-tripping through the host.
//
// FFI is duplicated locally instead of pulled from `flame_core::cuda::ffi`
// because the existing event-FFI helpers in flame-core's `activation_offload`
// module are private. Keeping a small private FFI block here avoids an
// out-of-scope flame-core public-API change for Phase 0.
// ---------------------------------------------------------------------------

extern "C" {
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

/// `cudaEventCreateWithFlags(cudaEventDisableTiming)` — events used purely
/// for ordering, not measurement, are cheaper without the timing buffer.
const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;

struct CudaEvent {
    raw: *mut c_void,
}

// SAFETY: CUDA event handles are thread-safe per the CUDA runtime. The raw
// pointer is opaque and never dereferenced on the host.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    fn new() -> anyhow::Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let s = unsafe { cudaEventCreateWithFlags(&mut raw, CUDA_EVENT_DISABLE_TIMING) };
        if s != 0 {
            anyhow::bail!("cudaEventCreateWithFlags failed: {s}");
        }
        Ok(Self { raw })
    }

    /// Record this event on the default (null) stream. Captures all kernels
    /// queued on the default stream up to the call site.
    fn record_default(&self) -> anyhow::Result<()> {
        let s = unsafe { cudaEventRecord(self.raw, std::ptr::null_mut()) };
        if s != 0 {
            anyhow::bail!("cudaEventRecord (default stream) failed: {s}");
        }
        Ok(())
    }

    /// Record this event on the given non-default stream.
    fn record_on(&self, stream: *mut c_void) -> anyhow::Result<()> {
        let s = unsafe { cudaEventRecord(self.raw, stream) };
        if s != 0 {
            anyhow::bail!("cudaEventRecord (stream) failed: {s}");
        }
        Ok(())
    }

    /// Block the calling host thread until this event fires. Used by the
    /// streaming-mode prefetch path before reusing a shared pinned staging
    /// buffer: the staging is the H2D source, so the CPU must wait until the
    /// previous H2D has finished reading it before overwriting.
    fn synchronize(&self) -> anyhow::Result<()> {
        let s = unsafe { cudaEventSynchronize(self.raw) };
        if s != 0 {
            anyhow::bail!("cudaEventSynchronize failed: {s}");
        }
        Ok(())
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        // Best-effort: ignore failure (e.g. device already shut down).
        unsafe {
            cudaEventDestroy(self.raw);
        }
    }
}

/// Make `stream` wait until `event` fires. Pure GPU-side dependency — does
/// NOT block the host.
fn stream_wait_event(stream: *mut c_void, event: &CudaEvent) -> anyhow::Result<()> {
    let s = unsafe { cudaStreamWaitEvent(stream, event.raw, 0) };
    if s != 0 {
        anyhow::bail!("cudaStreamWaitEvent failed: {s}");
    }
    Ok(())
}

/// Make the default (null) stream wait on `event`. Subsequent default-stream
/// kernel launches will not start until the work that recorded the event has
/// finished.
fn default_stream_wait_event(event: &CudaEvent) -> anyhow::Result<()> {
    stream_wait_event(std::ptr::null_mut(), event)
}

/// Per-slot event tracker, shared between the offloader and any outstanding
/// `BlockHandle` for the slot. Wrapped in `Arc` so handles can outlive a
/// `&mut BlockOffloader` borrow without aliasing the offloader's mutable
/// state.
struct SlotEvents {
    /// Recorded on the default stream when the slot's `BlockHandle` is
    /// dropped. Subsequent transfer-stream H2D into this slot waits on it.
    compute_done: CudaEvent,
    /// Recorded on the transfer stream after H2D into the slot completes.
    /// The default stream waits on it before `prepare_weights`.
    h2d_done: CudaEvent,
    /// True once a `BlockHandle` for the slot has been dropped — i.e.
    /// `compute_done` reflects all default-stream work that the handle's
    /// holder may have queued. False means the slot was either never used
    /// for compute, or is still in use by a live handle (or a caller of the
    /// legacy `await_block` API that did not take a handle).
    compute_recorded: AtomicBool,
    /// True once `h2d_done` has been recorded on the transfer stream for the
    /// current contents of the slot. False means H2D never ran (Empty slot)
    /// or the slot was just cleared.
    h2d_recorded: AtomicBool,
}

impl SlotEvents {
    fn new() -> anyhow::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            compute_done: CudaEvent::new()?,
            h2d_done: CudaEvent::new()?,
            compute_recorded: AtomicBool::new(false),
            h2d_recorded: AtomicBool::new(false),
        }))
    }
}

// ---------------------------------------------------------------------------
// BlockFacilitator trait
// ---------------------------------------------------------------------------

/// Model-specific geometry provider. Each trainer implements this to describe
/// its block structure so BlockOffloader can classify safetensors keys.
pub trait BlockFacilitator {
    /// How many blocks this model has.
    fn block_count(&self) -> usize;

    /// Given a safetensors key name, returns `Some(block_idx)` if it belongs
    /// to a block, or `None` if it's a shared (non-block) weight.
    fn classify_key(&self, key: &str) -> Option<usize>;
}

// ---------------------------------------------------------------------------
// PinnedTensor — one tensor stored in pinned host memory
// ---------------------------------------------------------------------------

/// What dtype the pinned bytes actually hold. Most tensors are BF16 after
/// load-time conversion. When `BLOCKOFF_FP8_PINNED=1` is set, `F8_E4M3`
/// tensors keep their raw FP8 bytes and carry their dequant scale — GPU
/// dequant to BF16 then happens inside `prepare_weights`.
#[derive(Clone, Copy)]
enum PinnedDtype {
    Bf16,
    Fp8 { scale: f32 },
}

struct PinnedTensor {
    buffer: PinnedHostBuffer<u8>,
    shape: Vec<usize>,
    /// Number of logical elements. For BF16 the pinned buffer is 2*num_elems
    /// bytes; for FP8 it is num_elems bytes.
    num_elems: usize,
    dtype: PinnedDtype,
}

// ---------------------------------------------------------------------------
// Streaming-mode state
//
// In `Pinned` mode (default), every block's BF16-converted bytes live in
// pinned host RAM for the offloader's lifetime — total ≈ full model size.
// For Qwen-Image-2512 that is ≈39 GB, which OOMs a 62 GB box once libtorch
// and any leftover pinned pages are accounted for.
//
// In `Streaming` mode, the safetensors files stay mmap'd and the offloader
// records only file offsets per block. At each `prefetch_block` we copy the
// block's bytes from mmap into one of two pinned staging buffers (sized to
// the largest block), then issue async H2D from the staging into fresh GPU
// tensors — exactly as the pinned path does, just with the staging filled
// on demand. Total pinned RAM = 2 × max_block_bytes ≈ 1.3 GB for the same
// model.
//
// The two staging buffers are paired with the two GPU slots: staging[i]
// always feeds slots[i]. Before reusing staging[i], the CPU waits on the
// previous tenant's `h2d_done` event so the in-flight H2D has actually
// finished reading the buffer. This is a host-side wait and means the CPU
// memcpy phase of prefetch cannot overlap with that prior H2D — for
// inference (compute >> transfer) the wait is usually zero, and for
// training the streaming path is opt-in (pinned remains the default for
// training where overlap matters).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum StreamSrcDtype {
    Bf16,
    F16,
    F32,
}

struct StreamingTensorEntry {
    name: String,
    file_idx: usize,
    file_offset: usize,
    /// Bytes the source occupies in the file (depends on src_dtype).
    src_byte_len: usize,
    src_dtype: StreamSrcDtype,
    shape: Vec<usize>,
    num_elems: usize,
}

struct StreamingState {
    /// Live mmaps of every input safetensors file. Indexed by `file_idx`.
    files: Vec<memmap2::Mmap>,
    /// Per-block ordered list of tensors. Order must be deterministic — the
    /// streaming prefetch packs them sequentially into the staging buffer.
    blocks: Vec<Vec<StreamingTensorEntry>>,
    /// Two pinned staging buffers, sized to the largest block's BF16 byte
    /// length. `staging[i]` feeds `slots[i]`.
    staging: [PinnedHostBuffer<u8>; 2],
    /// Capacity of each staging buffer in bytes (`max_block_bf16_bytes`).
    staging_capacity: usize,
}

// ---------------------------------------------------------------------------
// SlotState — one GPU-side buffer slot
// ---------------------------------------------------------------------------

/// One FP8 tensor still living as raw u8 bytes on the GPU, awaiting dequant
/// inside `prepare_weights`. The `CudaSlice<u8>` is kept alive in the slot so
/// that the async H2D has definitely finished before we launch the dequant
/// kernel (transfer-stream sync happens before prepare_weights runs).
struct Fp8Pending {
    data: CudaSlice<u8>,
    shape: Vec<usize>,
    scale: f32,
}

enum SlotState {
    Empty,
    /// Raw GPU tensors — H2D done but prepare_weights not yet applied.
    Raw {
        block_idx: usize,
        /// BF16 tensors already in final form (copied directly from pinned BF16).
        tensors: HashMap<String, Tensor>,
        /// FP8 pending entries: dequant-to-BF16 happens in prepare_weights.
        fp8_pending: HashMap<String, Fp8Pending>,
        /// Phase 0 event tracker — see `SlotEvents`.
        events: Arc<SlotEvents>,
    },
    /// Ready for compute — prepare_weights applied.
    Prepared {
        block_idx: usize,
        tensors: Arc<HashMap<String, Tensor>>,
        /// Phase 0 event tracker — see `SlotEvents`.
        events: Arc<SlotEvents>,
    },
}

impl SlotState {
    fn block_idx(&self) -> Option<usize> {
        match self {
            SlotState::Empty => None,
            SlotState::Raw { block_idx, .. } => Some(*block_idx),
            SlotState::Prepared { block_idx, .. } => Some(*block_idx),
        }
    }

    /// Borrow the event tracker for a non-Empty slot. Returns None for Empty.
    fn events(&self) -> Option<&Arc<SlotEvents>> {
        match self {
            SlotState::Empty => None,
            SlotState::Raw { events, .. } | SlotState::Prepared { events, .. } => Some(events),
        }
    }

    fn take(&mut self) -> SlotState {
        std::mem::replace(self, SlotState::Empty)
    }
}

// ---------------------------------------------------------------------------
// BlockOffloader
// ---------------------------------------------------------------------------

/// Double-buffered block offloader. Holds all block weights in pinned CPU
/// memory. Two GPU slots for prefetch/compute overlap.
pub struct BlockOffloader {
    /// Per-block weights in pinned CPU memory. Index = block_idx.
    /// Empty `Vec` when `streaming` is `Some` — the streaming path stores
    /// per-block tensor entries inside `StreamingState` instead.
    cpu_blocks: Vec<HashMap<String, PinnedTensor>>,

    /// Streaming-mode state. `Some` when constructed via
    /// [`Self::load_streaming`]; `None` for the default pinned path.
    streaming: Option<StreamingState>,

    /// The CUDA device for GPU allocations.
    device: Arc<CudaDevice>,

    /// Dedicated CUDA stream for async H2D transfers.
    transfer_stream: CudaStream,

    /// Two GPU-side buffer slots for ping-pong.
    slots: [SlotState; 2],

    /// Which slot index (0 or 1) holds the current compute block.
    active: usize,

    /// Block index currently being prefetched (None = idle).
    prefetch_in_flight: Option<usize>,

    /// Total pinned CPU bytes allocated.
    total_pinned_bytes: usize,

    /// Whether to keep weights in PyTorch-native `[Cout, Cin]` layout instead
    /// of pre-transposing every 2D `.weight` to `[Cin, Cout]` in
    /// `prepare_weights`. Default `false` (legacy behavior — pre-transpose
    /// for callers that use `Tensor::matmul` against `[Cin, Cout]`).
    ///
    /// Set to `true` via [`Self::with_native_layout`] when the caller's
    /// forward path uses `flame_core::ops::fused_inference::fused_linear3d_native`,
    /// which does the transpose inside cuBLASLt via TRANSA=T and therefore
    /// expects native `[Cout, Cin]`. Pre-transposing in that case would put
    /// the weight in the wrong layout for the GEMM and silently produce
    /// garbage.
    native_layout: bool,

    /// Phase 2 (FlexTensor port): opt-in resident-set strategy.
    ///
    /// When `None` (default), every code path runs exactly as it did
    /// before Phase 2 — bit-identical klein 9B behavior, every existing
    /// trainer inherits the 2-slot pipeline with no config change.
    ///
    /// When `Some`, [`prefetch_block`](Self::prefetch_block) consults
    /// the strategy at the top of each call to update telemetry on the
    /// strategy's resident-set decisions. The strategy's plan is
    /// advisory only — the offloader still drives the 2-slot mechanic.
    /// Phase 3 will widen the slot ring to honor a strategy's
    /// `desired_resident` exactly.
    strategy: Option<Box<dyn Strategy>>,

    /// Cached per-block byte sizes used by the strategy. Computed once
    /// at construction; stays constant for the offloader's lifetime.
    block_size_cache: Vec<usize>,

    /// Most-recent-first access history (bounded ring). Updated on
    /// every `prefetch_block` call when a strategy is attached.
    access_history: VecDeque<u32>,

    /// Counter of prefetches issued since the last strategy `plan()`
    /// call. Threaded through `AccessHints::prefetches_since_last_plan`.
    prefetches_since_plan: u32,
}

// Safety: BlockOffloader is always accessed behind a Mutex (serialized).
// CudaStream contains a raw pointer that isn't Send, but CUDA streams are
// thread-safe when access is serialized — which the Mutex guarantees.
unsafe impl Send for BlockOffloader {}
unsafe impl Sync for BlockOffloader {}

impl BlockOffloader {
    /// Load all block weights from safetensors file(s) into pinned CPU memory.
    ///
    /// Opens each file, reads block tensors into `cudaMallocHost` pinned buffers
    /// (converted to BF16), then closes all files. Never touches disk again.
    pub fn load(
        paths: &[&str],
        facilitator: &dyn BlockFacilitator,
        device: Arc<CudaDevice>,
    ) -> anyhow::Result<Self> {
        Self::load_inner(paths, facilitator, device, false)
    }

    /// Like [`Self::load`], but always treats `F8_E4M3` tensors as "keep raw
    /// FP8 bytes on host, GPU-dequant-to-BF16 via `dequant_fp8_to_bf16`".
    ///
    /// Use this for checkpoints that are already FP8-cast on disk (e.g. the
    /// LTX-2.3 22B distilled-fp8 safetensors) so we never pay the 2× pinned-RAM
    /// cost of CPU-dequanting to BF16. Per-tensor `weight_scale` sidecar F32
    /// scalars are looked up in metadata and threaded through the GPU kernel —
    /// this is the same math that `fp8_scaled_mm` / Lightricks `FP8Linear`
    /// applies, modulo the activation-quantization that we skip (activations
    /// stay BF16). If no scale sidecar is found, the kernel runs with
    /// `scale = 1.0`, which is equivalent to PyTorch's native
    /// `float8_e4m3fn.to(bfloat16)` IEEE-direct cast (Lightricks `Fp8CastLinear`).
    pub fn load_fp8_stream(
        paths: &[&str],
        facilitator: &dyn BlockFacilitator,
        device: Arc<CudaDevice>,
    ) -> anyhow::Result<Self> {
        Self::load_inner(paths, facilitator, device, true)
    }

    fn load_inner(
        paths: &[&str],
        facilitator: &dyn BlockFacilitator,
        device: Arc<CudaDevice>,
        force_fp8_pinned: bool,
    ) -> anyhow::Result<Self> {
        let block_count = facilitator.block_count();
        let mut cpu_blocks: Vec<HashMap<String, PinnedTensor>> =
            (0..block_count).map(|_| HashMap::new()).collect();
        let mut total_pinned_bytes: usize = 0;
        let fp8_pinned_mode =
            force_fp8_pinned || std::env::var("BLOCKOFF_FP8_PINNED").is_ok();

        for &path in paths {
            let (header, data_start, mmap) = Self::mmap_safetensors(path)?;

            let metadata: serde_json::Value = serde_json::from_str(&header)?;
            let metadata_obj = metadata
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("invalid safetensors metadata"))?;

            for (name, info) in metadata_obj {
                if name == "__metadata__" {
                    continue;
                }

                let block_idx = match facilitator.classify_key(name) {
                    Some(idx) => idx,
                    None => continue,
                };
                if block_idx >= block_count {
                    anyhow::bail!(
                        "classify_key returned {block_idx} >= block_count {block_count} for {name:?}"
                    );
                }

                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing shape for {name}"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                let num_elems: usize = shape.iter().product();
                if num_elems == 0 {
                    continue;
                }

                let offsets = info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing data_offsets for {name}"))?;
                let start = data_start
                    + offsets.first().and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow::anyhow!("bad start offset for {name}"))? as usize;
                let end = data_start
                    + offsets.get(1).and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow::anyhow!("bad end offset for {name}"))? as usize;

                let dtype_str = info["dtype"].as_str().unwrap_or("F32");
                if !matches!(dtype_str, "F32" | "BF16" | "F16" | "F8_E4M3") {
                    continue;
                }

                let raw = &mmap[start..end];

                // Skip sidecar scalar scale tensors in FP8-pinned mode —
                // they're used via the metadata lookup below and don't carry
                // block weights themselves. Two naming conventions are in the
                // wild:
                //   - LTX-2:   `foo.weight_scale`  (we add "_scale" to the key)
                //   - Comfy-scaled (Wan2.2, SD3, etc.): `foo.scale_weight` and
                //     `foo.scale_input`  (separate sibling keys, NOT suffixes)
                // Only skip when fp8_pinned_mode is on so BF16 checkpoints that
                // happen to have scalar tensors with `_scale` in the name are
                // not regressed.
                if fp8_pinned_mode
                    && dtype_str == "F32"
                    && num_elems == 1
                    && (name.ends_with("_scale")
                        || name.ends_with(".scale_weight")
                        || name.ends_with(".scale_input"))
                {
                    continue;
                }

                // FP8-pinned branch: store raw FP8 bytes and the scale; dequant
                // happens on GPU inside `prepare_weights`. Halves pinned RAM.
                if dtype_str == "F8_E4M3" && fp8_pinned_mode {
                    // Try both naming conventions: LTX-2 appends `_scale`,
                    // Comfy-scaled replaces `.weight` with `.scale_weight`.
                    let lookup_scale = |key: &str| -> Option<f32> {
                        metadata_obj.get(key).and_then(|si| {
                            let so = si["data_offsets"].as_array()?;
                            let ss = data_start + so[0].as_u64()? as usize;
                            Some(f32::from_le_bytes([
                                mmap[ss], mmap[ss + 1], mmap[ss + 2], mmap[ss + 3],
                            ]))
                        })
                    };
                    let scale = lookup_scale(&format!("{name}_scale"))
                        .or_else(|| {
                            name.strip_suffix(".weight")
                                .and_then(|base| lookup_scale(&format!("{base}.scale_weight")))
                        })
                        .unwrap_or(1.0);


                    let byte_len = raw.len(); // == num_elems for FP8
                    let mut pinned = PinnedHostBuffer::<u8>::with_capacity_elems(
                        byte_len, PinnedAllocFlags::DEFAULT,
                    ).map_err(|e| anyhow::anyhow!("pinned alloc for {name}: {e}"))?;
                    pinned.as_mut_bytes()[..byte_len].copy_from_slice(raw);
                    unsafe { pinned.set_len(byte_len); }

                    total_pinned_bytes += byte_len;
                    cpu_blocks[block_idx].insert(
                        name.clone(),
                        PinnedTensor {
                            buffer: pinned,
                            shape,
                            num_elems,
                            dtype: PinnedDtype::Fp8 { scale },
                        },
                    );
                    continue;
                }

                let bf16_u16: Vec<u16> = match dtype_str {
                    "BF16" => {
                        // BF16 bytes on disk are already LE u16 — a single
                        // memcpy is bandwidth-bound, while the scalar
                        // chunk-by-chunk reconstruction the original code did
                        // tops out around 100-150 MB/s on this box. For
                        // Qwen-Image-2512 (~39 GB), that's ~285s of wall time
                        // for what should be ~5-10s of memcpy.
                        //
                        // NOTE: a "skip the Vec<u16> intermediate" variant
                        // (writing directly into pinned memory in a single
                        // memcpy) was tried and measured ~50-65s SLOWER on
                        // Qwen-Image-2512 (187-195s vs 131s for this Vec-then-
                        // memcpy pattern). Pinned host memory writes are
                        // bandwidth-limited (~3 GB/s), so the indirect Vec
                        // intermediate exploits the CPU cache for the first
                        // memcpy and the second mmcopy hits a faster optimized
                        // bulk-memcpy path into the pinned buffer. Counter-
                        // intuitive but measured. Keep the Vec<u16> intermediate.
                        let mut out = vec![0u16; num_elems];
                        let byte_len = num_elems * 2;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                raw.as_ptr(),
                                out.as_mut_ptr() as *mut u8,
                                byte_len,
                            );
                        }
                        out
                    }
                    "F16" => {
                        let mut out = vec![0u16; num_elems];
                        for (v, chunk) in out.iter_mut().zip(raw.chunks_exact(2)) {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            *v = f32_to_bf16(f16_to_f32(bits));
                        }
                        out
                    }
                    "F32" => {
                        let mut out = vec![0u16; num_elems];
                        for (v, chunk) in out.iter_mut().zip(raw.chunks_exact(4)) {
                            let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            *v = f32_to_bf16(f);
                        }
                        out
                    }
                    "F8_E4M3" => {
                        // Two naming conventions — see fp8_pinned branch above
                        // for details. LTX-2: `foo.weight_scale`; Comfy-scaled:
                        // `foo.scale_weight`.
                        let lookup = |key: &str| -> Option<f32> {
                            metadata_obj.get(key).and_then(|si| {
                                let so = si["data_offsets"].as_array()?;
                                let ss = data_start + so[0].as_u64()? as usize;
                                Some(f32::from_le_bytes([mmap[ss], mmap[ss+1], mmap[ss+2], mmap[ss+3]]))
                            })
                        };
                        let scale = lookup(&format!("{name}_scale"))
                            .or_else(|| {
                                name.strip_suffix(".weight")
                                    .and_then(|b| lookup(&format!("{b}.scale_weight")))
                            })
                            .unwrap_or(1.0);
                        let mut out = vec![0u16; num_elems];
                        for (v, &byte) in out.iter_mut().zip(raw.iter()) {
                            *v = f32_to_bf16(fp8_e4m3_to_f32(byte) * scale);
                        }
                        out
                    }
                    _ => unreachable!(),
                };

                let byte_len = bf16_u16.len() * 2;
                let mut pinned =
                    PinnedHostBuffer::<u8>::with_capacity_elems(byte_len, PinnedAllocFlags::DEFAULT)
                        .map_err(|e| anyhow::anyhow!("pinned alloc for {name}: {e}"))?;

                let src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(bf16_u16.as_ptr() as *const u8, byte_len)
                };
                pinned.as_mut_bytes()[..byte_len].copy_from_slice(src_bytes);
                unsafe { pinned.set_len(byte_len); }

                total_pinned_bytes += byte_len;

                cpu_blocks[block_idx].insert(
                    name.clone(),
                    PinnedTensor {
                        buffer: pinned,
                        shape,
                        num_elems,
                        dtype: PinnedDtype::Bf16,
                    },
                );
            }
        }

        let transfer_stream = device
            .fork_default_stream()
            .map_err(|e| anyhow::anyhow!("failed to create transfer stream: {e:?}"))?;

        log::info!(
            "BlockOffloader: loaded {} blocks, {:.1} MB pinned CPU memory",
            block_count,
            total_pinned_bytes as f64 / (1024.0 * 1024.0),
        );

        let block_size_cache: Vec<usize> = cpu_blocks
            .iter()
            .map(|m| m.values().map(|pt| pt.buffer.len_bytes()).sum())
            .collect();

        Ok(Self {
            cpu_blocks,
            streaming: None,
            device,
            transfer_stream,
            slots: [SlotState::Empty, SlotState::Empty],
            active: 0,
            prefetch_in_flight: None,
            total_pinned_bytes,
            native_layout: false,
            strategy: None,
            block_size_cache,
            access_history: VecDeque::with_capacity(ACCESS_HISTORY_CAP),
            prefetches_since_plan: 0,
        })
    }

    /// Streaming-mode constructor: keeps the safetensors files mmap'd and
    /// records per-block tensor offsets, instead of pinning every block's
    /// BF16 bytes upfront. Two pinned staging buffers (sized to the largest
    /// block) are filled on demand at each `prefetch_block`.
    ///
    /// Use this when the model's pinned-RAM footprint would not fit:
    /// Qwen-Image-2512 (≈39 GB pinned) on a 62 GB box already pushes the
    /// limit and OOMs once libtorch and any leftover state are loaded.
    /// Streaming brings pinned RAM down to `2 × max_block_bytes` (≈1.3 GB
    /// for Qwen-Image-2512).
    ///
    /// Limitations:
    /// - Source dtypes supported: `BF16`, `F16`, `F32`. `F8_E4M3` is rejected
    ///   (the pinned path's CPU-dequant or `BLOCKOFF_FP8_PINNED` GPU-dequant
    ///   would have to move into the prefetch hot path; not needed for any
    ///   current streaming caller).
    /// - The per-block H2D source is a shared staging buffer reused across
    ///   prefetches into the same slot. Before reuse, the CPU waits on the
    ///   prior tenant's `h2d_done` event — host-side, so the CPU memcpy phase
    ///   of prefetch cannot overlap with that prior H2D. For inference this
    ///   wait is usually zero (compute >> transfer); for training keep using
    ///   the pinned path.
    pub fn load_streaming(
        paths: &[&str],
        facilitator: &dyn BlockFacilitator,
        device: Arc<CudaDevice>,
    ) -> anyhow::Result<Self> {
        let block_count = facilitator.block_count();
        let mut blocks: Vec<Vec<StreamingTensorEntry>> =
            (0..block_count).map(|_| Vec::new()).collect();
        let mut files: Vec<memmap2::Mmap> = Vec::with_capacity(paths.len());

        for &path in paths {
            let (header, data_start, mmap) = Self::mmap_safetensors(path)?;
            let file_idx = files.len();
            files.push(mmap);

            let metadata: serde_json::Value = serde_json::from_str(&header)?;
            let metadata_obj = metadata
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("invalid safetensors metadata"))?;

            for (name, info) in metadata_obj {
                if name == "__metadata__" {
                    continue;
                }
                let block_idx = match facilitator.classify_key(name) {
                    Some(idx) => idx,
                    None => continue,
                };
                if block_idx >= block_count {
                    anyhow::bail!(
                        "classify_key returned {block_idx} >= block_count {block_count} for {name:?}"
                    );
                }

                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing shape for {name}"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                let num_elems: usize = shape.iter().product();
                if num_elems == 0 {
                    continue;
                }

                let offsets = info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing data_offsets for {name}"))?;
                let start = data_start
                    + offsets.first().and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow::anyhow!("bad start offset for {name}"))? as usize;
                let end = data_start
                    + offsets.get(1).and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow::anyhow!("bad end offset for {name}"))? as usize;
                let src_byte_len = end.saturating_sub(start);

                let dtype_str = info["dtype"].as_str().unwrap_or("F32");
                let src_dtype = match dtype_str {
                    "BF16" => StreamSrcDtype::Bf16,
                    "F16" => StreamSrcDtype::F16,
                    "F32" => StreamSrcDtype::F32,
                    "F8_E4M3" => anyhow::bail!(
                        "BlockOffloader::load_streaming: F8_E4M3 not supported in streaming mode (key {name:?}); use the pinned `load_fp8_stream` constructor instead"
                    ),
                    _ => continue,
                };

                blocks[block_idx].push(StreamingTensorEntry {
                    name: name.clone(),
                    file_idx,
                    file_offset: start,
                    src_byte_len,
                    src_dtype,
                    shape,
                    num_elems,
                });
            }
        }

        // Largest per-block BF16 footprint determines staging size.
        let max_block_bf16_bytes = blocks
            .iter()
            .map(|entries| entries.iter().map(|e| e.num_elems * 2).sum::<usize>())
            .max()
            .unwrap_or(0);
        if max_block_bf16_bytes == 0 {
            anyhow::bail!(
                "BlockOffloader::load_streaming: no block tensors classified across {} files",
                paths.len()
            );
        }

        let staging0 = PinnedHostBuffer::<u8>::with_capacity_elems(
            max_block_bf16_bytes,
            PinnedAllocFlags::DEFAULT,
        )
        .map_err(|e| anyhow::anyhow!("staging buffer 0 alloc ({} bytes): {e}", max_block_bf16_bytes))?;
        let staging1 = PinnedHostBuffer::<u8>::with_capacity_elems(
            max_block_bf16_bytes,
            PinnedAllocFlags::DEFAULT,
        )
        .map_err(|e| anyhow::anyhow!("staging buffer 1 alloc ({} bytes): {e}", max_block_bf16_bytes))?;

        let total_pinned_bytes = max_block_bf16_bytes * 2;

        let transfer_stream = device
            .fork_default_stream()
            .map_err(|e| anyhow::anyhow!("failed to create transfer stream: {e:?}"))?;

        log::info!(
            "BlockOffloader (streaming): {} blocks, max block {:.1} MB, staging {:.1} MB pinned ({} files mmap'd)",
            block_count,
            max_block_bf16_bytes as f64 / (1024.0 * 1024.0),
            total_pinned_bytes as f64 / (1024.0 * 1024.0),
            files.len(),
        );

        let block_size_cache: Vec<usize> = streaming_blocks_byte_sizes(&blocks);

        let streaming = StreamingState {
            files,
            blocks,
            staging: [staging0, staging1],
            staging_capacity: max_block_bf16_bytes,
        };

        Ok(Self {
            cpu_blocks: Vec::new(),
            streaming: Some(streaming),
            device,
            transfer_stream,
            slots: [SlotState::Empty, SlotState::Empty],
            active: 0,
            prefetch_in_flight: None,
            total_pinned_bytes,
            native_layout: false,
            strategy: None,
            block_size_cache,
            access_history: VecDeque::with_capacity(ACCESS_HISTORY_CAP),
            prefetches_since_plan: 0,
        })
    }

    /// Opt into native `[Cout, Cin]` weight layout — disables the
    /// `prepare_weights` pre-transpose and leaves 2D `.weight` tensors as
    /// stored in the safetensors file. Required by callers using
    /// `flame_core::ops::fused_inference::fused_linear3d_native`. See the
    /// `native_layout` field doc for details.
    pub fn with_native_layout(mut self, native: bool) -> Self {
        self.native_layout = native;
        self
    }

    /// Attach a resident-set [`Strategy`] (Phase 2 FlexTensor port).
    ///
    /// Default behavior — no strategy attached — is bit-identical to
    /// the pre-Phase-2 code paths. Attaching a strategy adds advisory
    /// planning hooks inside [`Self::prefetch_block`] and emits the
    /// resulting `target_resident_bytes` / decision counts into the
    /// global telemetry sink (see [`telemetry::strategy_counters`]).
    ///
    /// Strategies do not commandeer the slot mechanic; they advise on
    /// eviction order and prefetch priority. The 2-slot pipeline still
    /// runs underneath.
    pub fn set_strategy(&mut self, strategy: Box<dyn Strategy>) {
        self.strategy = Some(strategy);
    }

    /// Builder-style variant of [`Self::set_strategy`].
    pub fn with_strategy(mut self, strategy: Box<dyn Strategy>) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Detach the current strategy. The offloader reverts to the
    /// default 2-slot path.
    pub fn clear_strategy(&mut self) {
        self.strategy = None;
    }

    /// Currently-attached strategy's name, or `"none"` when no strategy
    /// is set. Used by tests / diagnostics.
    pub fn strategy_name(&self) -> &'static str {
        match self.strategy.as_deref() {
            Some(s) => s.name(),
            None => "none",
        }
    }

    /// Per-block byte sizes the offloader sees. Length =
    /// `block_count()`. Exposed for callers that want to size their
    /// own [`Strategy`] budgets against the offloader's view of the
    /// model.
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_size_cache
    }

    /// Currently-resident block IDs across both slots. May be empty
    /// (cold start), 1 element (only one slot used), or 2 elements.
    pub fn resident_blocks(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(2);
        for slot in &self.slots {
            if let Some(b) = slot.block_idx() {
                out.push(b);
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Public API: prefetch / await / ensure
    // -----------------------------------------------------------------------

    /// Start async H2D of `block_idx` into the prefetch slot (non-blocking).
    ///
    /// If block_idx is already on either slot, this is a no-op.
    /// If a different prefetch is in flight, syncs it first.
    pub fn prefetch_block(&mut self, block_idx: usize) -> anyhow::Result<()> {
        // Phase 1 FlexTensor port: telemetry. Disabled-mode is a relaxed
        // atomic load and an Option<Instant>::None constructor — measured
        // at <5 ns under perf, safe on the hot path.
        let tele = telemetry::global();
        let timer = tele.record_prefetch_begin();
        let already_resident = self.slots[0].block_idx() == Some(block_idx)
            || self.slots[1].block_idx() == Some(block_idx);

        // Already on a slot?
        if already_resident {
            tele.record_prefetch_already_resident(timer, block_idx);
            return Ok(());
        }

        // Phase 2 FlexTensor port: consult attached strategy.
        //
        // Critical: this block runs only when a strategy has been
        // explicitly attached via `set_strategy` / `with_strategy`.
        // The default (no strategy) path falls straight through to the
        // pre-Phase-2 `prefetch_block_inner` below — bit-identical to
        // the old behavior. klein 9B regression gate is anchored on
        // this fact.
        self.consult_strategy(block_idx);

        // Best-effort bytes count for telemetry. Pinned path: sum pinned
        // buffer lengths. Streaming path: sum BF16 footprint from entries.
        // Computed before the inner call so we don't have to plumb a
        // return value through `prefetch_block_streaming_inner`.
        let bytes = if let Some(stream_state) = self.streaming.as_ref() {
            stream_state
                .blocks
                .get(block_idx)
                .map(|entries| entries.iter().map(|e| e.num_elems * 2).sum::<usize>())
                .unwrap_or(0) as u64
        } else {
            self.cpu_blocks
                .get(block_idx)
                .map(|m| m.values().map(|pt| pt.buffer.len_bytes()).sum::<usize>())
                .unwrap_or(0) as u64
        };

        let result = self.prefetch_block_inner(block_idx);
        if result.is_ok() {
            tele.record_prefetch_end(timer, block_idx, bytes);
        }
        result
    }

    /// Phase 2: invoke the attached resident-set strategy (if any) and
    /// emit its decision into telemetry. No state mutation outside of
    /// `access_history` and `prefetches_since_plan` — the plan itself
    /// is advisory.
    ///
    /// Cheap when no strategy is attached: a single `Option::is_none`
    /// check and a `VecDeque` push (≤ 100 ns on commodity hardware).
    /// In the no-strategy case the access history is still maintained
    /// so attaching a strategy later sees real recent-access signal —
    /// the cost is a `u32` push into a 64-cap ring, which is bounded
    /// to a single memmove of at most 64 × 4 bytes.
    fn consult_strategy(&mut self, block_idx: usize) {
        // Update access history regardless of strategy attachment.
        // Most-recent-first (push_front) so strategies can read
        // `history[0]` as "the block being asked for".
        if self.access_history.len() == ACCESS_HISTORY_CAP {
            self.access_history.pop_back();
        }
        self.access_history.push_front(block_idx as u32);

        // No strategy? Done — pre-Phase-2 path runs unchanged.
        let Some(strat) = self.strategy.as_mut() else {
            return;
        };

        // Build the state snapshot. Cheap — borrows into existing
        // fields, no allocations.
        let resident = self.slots.iter().filter_map(|s| s.block_idx()).collect::<Vec<_>>();
        let hints = AccessHints {
            vram_authoritative: false,
            prefetches_since_last_plan: self.prefetches_since_plan,
        };
        let state = OffloaderState {
            block_count: self.block_size_cache.len(),
            block_sizes: &self.block_size_cache,
            resident: &resident,
            requested: block_idx,
            access_history: &self.access_history,
            free_vram_bytes: 0,
            total_vram_bytes: 0,
            hints,
        };

        let plan = strat.plan(&state);
        self.prefetches_since_plan = 0;

        // Emit decision into telemetry. The hook is a single relaxed
        // atomic load + early return when telemetry is disabled.
        telemetry::global().record_strategy_decision(
            strat.name(),
            plan.evict.len() as u64,
            plan.keep.len() as u64,
            plan.target_resident_bytes,
        );
    }

    fn prefetch_block_inner(&mut self, block_idx: usize) -> anyhow::Result<()> {

        // Different prefetch in flight? Sync it first (wasteful but safe).
        if let Some(inflight) = self.prefetch_in_flight {
            if inflight != block_idx {
                self.sync_transfer_stream()?;
                // Promote the raw slot to... just leave it as Raw, await will handle it.
                self.prefetch_in_flight = None;
            }
        }

        // Pick the non-active slot.
        let target = 1 - self.active;

        // Phase 0 safety: before reusing a slot, ensure any default-stream
        // compute that may still be reading the slot's storage has finished.
        //
        //   * Prepared + handle drop: a `BlockHandle` was dropped, which
        //     recorded `compute_done` on the default stream. Make the
        //     transfer stream wait on that event — pure GPU-side dependency,
        //     no host stall.
        //   * Prepared + legacy: the slot was returned via `await_block`
        //     (no handle), so we do not know when the caller's compute
        //     finished. Fall back to host-side `cudaDeviceSynchronize` to be
        //     safe. Callers wanting overlap should migrate to
        //     `await_block_handle`.
        //   * Raw: a prefetched-but-never-awaited block. No compute kernels
        //     have been queued against it yet (callers always go through
        //     `await_block*` before issuing work). The legacy
        //     `prefetch_in_flight` check above already drained the H2D, so
        //     it is safe to drop the slot's GPU tensors with no further
        //     synchronization.
        if matches!(self.slots[target], SlotState::Prepared { .. }) {
            if let Some(prior) = self.slots[target].events() {
                if prior.compute_recorded.load(Ordering::Acquire) {
                    stream_wait_event(
                        self.transfer_stream.stream as *mut c_void,
                        &prior.compute_done,
                    )?;
                } else {
                    let s = unsafe { cudaDeviceSynchronize() };
                    if s != 0 {
                        anyhow::bail!("cudaDeviceSynchronize before slot reuse: {s}");
                    }
                }
            }
        }

        // Streaming-mode extra: the per-slot pinned staging buffer is the H2D
        // source and is reused across prefetches into this slot. Wait on the
        // prior tenant's `h2d_done` event host-side before overwriting it.
        // No-op when the event was never recorded (slot was Empty) or when
        // the H2D has already drained.
        if self.streaming.is_some() {
            if let Some(prior) = self.slots[target].events() {
                if prior.h2d_recorded.load(Ordering::Acquire) {
                    prior.h2d_done.synchronize()?;
                }
            }
        }

        self.slots[target] = SlotState::Empty; // drop old GPU tensors (now safe)

        if self.streaming.is_some() {
            return self.prefetch_block_streaming_inner(block_idx, target);
        }

        let block = &self.cpu_blocks[block_idx];
        if block.is_empty() {
            // Empty block — synthesize a Prepared slot with fresh events so
            // BlockHandle Drop has a target. Mark `h2d_recorded` so an
            // `await_block` after this won't try to wait on an unrecorded
            // event.
            let events = SlotEvents::new()?;
            events.h2d_recorded.store(true, Ordering::Release);
            self.slots[target] = SlotState::Prepared {
                block_idx,
                tensors: Arc::new(HashMap::new()),
                events,
            };
            return Ok(());
        }

        // Allocate fresh per-slot events. Old events Arc is dropped here
        // (after the wait above), which destroys its CudaEvent and frees
        // the runtime-owned event handle.
        let events = SlotEvents::new()?;

        // Ensure transfer stream sees prior default-stream work.
        // (Redundant with the per-slot event wait above for the slot we are
        // overwriting, but cheap and protects against unrelated work the
        // caller may have queued on the default stream between the slot wait
        // and the H2D issue below.)
        self.transfer_stream
            .wait_for_default()
            .map_err(|e| anyhow::anyhow!("wait_for_default: {e:?}"))?;

        let stream_ptr = self.transfer_stream.stream as *mut c_void;
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(block.len());
        let mut fp8_pending: HashMap<String, Fp8Pending> = HashMap::new();

        if std::env::var("BLOCKOFF_MEM_DEBUG").is_ok() {
            let (free, total) = crate::cuda::utils::cuda_mem_get_info().unwrap_or((0, 0));
            eprintln!(
                "[blockoff] prefetch block {} starting: GPU free={} MiB / total={} MiB",
                block_idx, free / (1024 * 1024), total / (1024 * 1024)
            );
        }
        // Allocate GPU buffers with `unsafe alloc` (no zero-fill). A prior
        // `alloc_zeros` here was racing: `alloc_zeros` issues a memset on the
        // *default* stream, while our `memcpy_async` runs on `transfer_stream`.
        // The two are unordered — on NVIDIA the memset can complete AFTER the
        // memcpy, zeroing the just-copied bytes. The symptom was
        // non-deterministic all-zero tensors in `await_block` results
        // (reproduced by `ltx2_fp8_stream_parity` one-block-at-a-time load).
        // Since every allocated byte is immediately overwritten by the memcpy
        // on the same stream as the kernel that will read it, no initial
        // value is ever observed — the unsafe alloc is safe.
        for (key, pt) in block {
            match pt.dtype {
                PinnedDtype::Bf16 => {
                    let gpu_buf = unsafe { self.device.alloc::<u16>(pt.num_elems) }.map_err(|e| {
                        let (free, total) =
                            crate::cuda::utils::cuda_mem_get_info().unwrap_or((0, 0));
                        anyhow::anyhow!(
                            "GPU alloc for {key} ({} elems, need={} MiB) failed; free={} MiB total={} MiB: {e:?}",
                            pt.num_elems, (pt.num_elems * 2) / (1024 * 1024),
                            free / (1024 * 1024), total / (1024 * 1024)
                        )
                    })?;

                    let dst = (*gpu_buf.device_ptr() as u64) as *mut c_void;
                    let src = pt.buffer.as_ptr() as *const c_void;
                    let bytes = pt.buffer.len_bytes();
                    memcpy_async_host_to_device(dst, src, bytes, stream_ptr)
                        .map_err(|e| anyhow::anyhow!("H2D for {key}: {e}"))?;

                    let tensor = Tensor::from_bf16_slice_gpu(
                        gpu_buf, Shape::from_dims(&pt.shape), self.device.clone(),
                    );
                    tensors.insert(key.clone(), tensor);
                }
                PinnedDtype::Fp8 { scale } => {
                    let gpu_buf = unsafe { self.device.alloc::<u8>(pt.num_elems) }.map_err(|e| {
                        let (free, total) =
                            crate::cuda::utils::cuda_mem_get_info().unwrap_or((0, 0));
                        anyhow::anyhow!(
                            "GPU alloc (FP8) for {key} ({} elems, need={} MiB) failed; free={} MiB total={} MiB: {e:?}",
                            pt.num_elems, pt.num_elems / (1024 * 1024),
                            free / (1024 * 1024), total / (1024 * 1024)
                        )
                    })?;

                    let dst = (*gpu_buf.device_ptr() as u64) as *mut c_void;
                    let src = pt.buffer.as_ptr() as *const c_void;
                    let bytes = pt.buffer.len_bytes();
                    memcpy_async_host_to_device(dst, src, bytes, stream_ptr)
                        .map_err(|e| anyhow::anyhow!("H2D (FP8) for {key}: {e}"))?;

                    fp8_pending.insert(
                        key.clone(),
                        Fp8Pending {
                            data: gpu_buf,
                            shape: pt.shape.clone(),
                            scale,
                        },
                    );
                }
            }
        }

        // All H2D copies are queued on the transfer stream. Record the
        // h2d-done event so the default stream can wait on it without a
        // host-side `cudaStreamSynchronize`.
        events.h2d_done.record_on(stream_ptr)?;
        events.h2d_recorded.store(true, Ordering::Release);

        self.slots[target] = SlotState::Raw {
            block_idx,
            tensors,
            fp8_pending,
            events,
        };
        self.prefetch_in_flight = Some(block_idx);
        Ok(())
    }

    /// Streaming-mode prefetch: copies the block's bytes from the mmap'd
    /// safetensors files into `staging[target]`, then issues async H2D from
    /// the staging buffer into fresh GPU tensors. Caller is responsible for:
    ///   * already having waited on `slots[target]`'s prior `compute_done`
    ///     (transfer-stream wait) and `h2d_done` (host-side sync) events;
    ///   * having reset `slots[target]` to `Empty`.
    fn prefetch_block_streaming_inner(
        &mut self,
        block_idx: usize,
        target: usize,
    ) -> anyhow::Result<()> {
        // Empty block — synthesize a Prepared slot with fresh events and
        // return, matching the pinned path's empty-block branch.
        {
            let stream_state = self
                .streaming
                .as_ref()
                .expect("streaming mode dispatch invariant");
            if stream_state.blocks[block_idx].is_empty() {
                let events = SlotEvents::new()?;
                events.h2d_recorded.store(true, Ordering::Release);
                self.slots[target] = SlotState::Prepared {
                    block_idx,
                    tensors: Arc::new(HashMap::new()),
                    events,
                };
                return Ok(());
            }
        }

        let events = SlotEvents::new()?;

        // Match pinned path: ensure transfer stream sees prior default-stream
        // work before we issue H2D on it.
        self.transfer_stream
            .wait_for_default()
            .map_err(|e| anyhow::anyhow!("wait_for_default: {e:?}"))?;

        if std::env::var("BLOCKOFF_MEM_DEBUG").is_ok() {
            let (free, total) = crate::cuda::utils::cuda_mem_get_info().unwrap_or((0, 0));
            eprintln!(
                "[blockoff streaming] prefetch block {} starting: GPU free={} MiB / total={} MiB",
                block_idx,
                free / (1024 * 1024),
                total / (1024 * 1024)
            );
        }

        let device = self.device.clone();
        let stream_ptr = self.transfer_stream.stream as *mut c_void;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        // Single mutable borrow scope for streaming state. The staging buffer
        // is mutated via raw pointer so we can both write CPU bytes into it
        // (Phase A) and read them out as the H2D source (Phase B) within the
        // same iteration without violating Rust aliasing — the only outstanding
        // reference to the staging memory inside this block is the raw
        // pointer.
        {
            let stream_state = self
                .streaming
                .as_mut()
                .expect("streaming mode dispatch invariant");
            let StreamingState {
                files,
                blocks,
                staging,
                staging_capacity,
            } = stream_state;
            let staging_capacity_local = *staging_capacity;
            let block_entries = &blocks[block_idx];
            let staging_buf = &mut staging[target];
            let staging_ptr: *mut u8 = staging_buf.as_mut_ptr();
            // Mark the staging len so debug prints / future readers see the
            // real fill amount; not load-bearing for correctness because every
            // H2D uses an explicit byte count.
            unsafe {
                staging_buf.set_len(staging_capacity_local.min(staging_buf.capacity_bytes()));
            }

            tensors.reserve(block_entries.len());
            let mut cursor: usize = 0;
            for entry in block_entries {
                let bf16_bytes = entry.num_elems * 2;
                if cursor + bf16_bytes > staging_capacity_local {
                    anyhow::bail!(
                        "BlockOffloader streaming: staging overflow at block {} tensor {} \
                         (cursor={} need={} cap={})",
                        block_idx,
                        entry.name,
                        cursor,
                        bf16_bytes,
                        staging_capacity_local
                    );
                }

                // Phase A — CPU memcpy/convert from mmap → staging at offset.
                let raw_end = entry.file_offset + entry.src_byte_len;
                let mmap = &files[entry.file_idx];
                if raw_end > mmap.len() {
                    anyhow::bail!(
                        "BlockOffloader streaming: out-of-range slice for {} ({}..{} > {})",
                        entry.name,
                        entry.file_offset,
                        raw_end,
                        mmap.len()
                    );
                }
                let raw = &mmap[entry.file_offset..raw_end];
                unsafe {
                    let dst = staging_ptr.add(cursor);
                    match entry.src_dtype {
                        StreamSrcDtype::Bf16 => {
                            std::ptr::copy_nonoverlapping(raw.as_ptr(), dst, bf16_bytes);
                        }
                        StreamSrcDtype::F16 => {
                            let dst_u16 = dst as *mut u16;
                            for i in 0..entry.num_elems {
                                let bits =
                                    u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                                *dst_u16.add(i) = f32_to_bf16(f16_to_f32(bits));
                            }
                        }
                        StreamSrcDtype::F32 => {
                            let dst_u16 = dst as *mut u16;
                            for i in 0..entry.num_elems {
                                let f = f32::from_le_bytes([
                                    raw[i * 4],
                                    raw[i * 4 + 1],
                                    raw[i * 4 + 2],
                                    raw[i * 4 + 3],
                                ]);
                                *dst_u16.add(i) = f32_to_bf16(f);
                            }
                        }
                    }
                }

                // Phase B — alloc GPU and async H2D from staging[target] at
                // `cursor` into the fresh GPU buffer. Same `unsafe alloc`
                // rationale as the pinned path: every byte is overwritten by
                // the immediately-following memcpy_async on the same stream.
                let gpu_buf = unsafe { device.alloc::<u16>(entry.num_elems) }.map_err(|e| {
                    let (free, total) =
                        crate::cuda::utils::cuda_mem_get_info().unwrap_or((0, 0));
                    anyhow::anyhow!(
                        "GPU alloc (streaming) for {} ({} elems, need={} MiB) failed; \
                         free={} MiB total={} MiB: {e:?}",
                        entry.name,
                        entry.num_elems,
                        bf16_bytes / (1024 * 1024),
                        free / (1024 * 1024),
                        total / (1024 * 1024)
                    )
                })?;
                let gpu_dst = (*gpu_buf.device_ptr() as u64) as *mut c_void;
                let host_src = unsafe { staging_ptr.add(cursor) } as *const c_void;
                memcpy_async_host_to_device(gpu_dst, host_src, bf16_bytes, stream_ptr)
                    .map_err(|e| {
                        anyhow::anyhow!("H2D (streaming) for {}: {e}", entry.name)
                    })?;

                let tensor = Tensor::from_bf16_slice_gpu(
                    gpu_buf,
                    Shape::from_dims(&entry.shape),
                    device.clone(),
                );
                tensors.insert(entry.name.clone(), tensor);

                cursor += bf16_bytes;
            }
        }

        // All H2D copies are on the transfer stream. Record h2d_done so the
        // default stream can gate its compute kernels on the GPU side.
        events.h2d_done.record_on(stream_ptr)?;
        events.h2d_recorded.store(true, Ordering::Release);

        self.slots[target] = SlotState::Raw {
            block_idx,
            tensors,
            fp8_pending: HashMap::new(),
            events,
        };
        self.prefetch_in_flight = Some(block_idx);
        Ok(())
    }

    /// Wait for prefetched block, prepare weights, return ready tensors.
    ///
    /// If `block_idx` is already prepared on the active slot, returns instantly.
    /// If it is in the prefetch slot, gates the default stream on the
    /// transfer-stream h2d-done event and prepares.
    ///
    /// Legacy API: returns a bare `Arc<HashMap>`. Subsequent `prefetch_block`
    /// calls that reuse this slot will fall back to a host-side
    /// `cudaDeviceSynchronize` because there is no scoped handle to record a
    /// compute-done event. Migrate hot paths to [`Self::await_block_handle`]
    /// for event-driven slot reuse with no host stall.
    pub fn await_block(
        &mut self,
        block_idx: usize,
    ) -> anyhow::Result<Arc<HashMap<String, Tensor>>> {
        // Phase 1 FlexTensor port: telemetry. "Hit" = slot already had
        // the block resident (Raw or Prepared) — prefetch landed in time.
        // "Miss" = await had to fall back to issuing its own H2D.
        let tele = telemetry::global();
        let timer = tele.record_await_begin();
        let is_hit = self.slots[0].block_idx() == Some(block_idx)
            || self.slots[1].block_idx() == Some(block_idx);

        let result = self.await_block_inner(block_idx);
        if result.is_ok() {
            if is_hit {
                tele.record_await_end_hit(timer, block_idx);
            } else {
                tele.record_await_end_miss(timer, block_idx);
            }
        }
        result
    }

    fn await_block_inner(
        &mut self,
        block_idx: usize,
    ) -> anyhow::Result<Arc<HashMap<String, Tensor>>> {
        // Check active slot — already prepared?
        if let SlotState::Prepared { block_idx: idx, ref tensors, .. } = self.slots[self.active] {
            if idx == block_idx {
                return Ok(tensors.clone());
            }
        }

        // Check both slots for a Raw or Prepared match
        for slot_idx in 0..2 {
            let matches = self.slots[slot_idx].block_idx() == Some(block_idx);
            if !matches {
                continue;
            }

            // If it's already prepared, just swap active and return.
            if let SlotState::Prepared { ref tensors, .. } = self.slots[slot_idx] {
                self.active = slot_idx;
                return Ok(tensors.clone());
            }

            // It's Raw — gate default stream on the slot's h2d-done event
            // (no host wait), then prepare and promote.
            if let Some(events) = self.slots[slot_idx].events() {
                if events.h2d_recorded.load(Ordering::Acquire) {
                    default_stream_wait_event(&events.h2d_done)?;
                } else {
                    // No event recorded — fall back to host-side sync.
                    self.sync_transfer_stream()?;
                }
            } else {
                self.sync_transfer_stream()?;
            }
            self.prefetch_in_flight = None;

            let raw = self.slots[slot_idx].take();
            if let SlotState::Raw { block_idx: idx, mut tensors, fp8_pending, events } = raw {
                Self::prepare_weights(&mut tensors, fp8_pending, self.native_layout)?;
                let arc = Arc::new(tensors);
                // Reset the per-handle compute_recorded flag — a new tenant
                // is taking the slot. h2d_recorded stays true (the H2D
                // already fired and the data is now on the GPU).
                events.compute_recorded.store(false, Ordering::Release);
                self.slots[slot_idx] = SlotState::Prepared {
                    block_idx: idx,
                    tensors: arc.clone(),
                    events,
                };
                self.active = slot_idx;
                return Ok(arc);
            }
        }

        // Miss — sync any in-flight, do full sync load into non-active slot.
        if self.prefetch_in_flight.is_some() {
            self.sync_transfer_stream()?;
            self.prefetch_in_flight = None;
        }
        // Use the inner prefetch so the wrapping `await_block` (which is
        // currently counting *this* call) doesn't also charge a second
        // prefetch_issued event. Telemetry attributes the work to await's
        // miss path; that is more useful for tuning prefetch-lead-time
        // than double counting both sides.
        self.prefetch_block_inner(block_idx)?;

        let target = 1 - self.active;
        // Gate default stream on the just-issued H2D via the slot's event,
        // not a host-side cudaStreamSynchronize.
        if let Some(events) = self.slots[target].events() {
            if events.h2d_recorded.load(Ordering::Acquire) {
                default_stream_wait_event(&events.h2d_done)?;
            } else {
                self.sync_transfer_stream()?;
            }
        } else {
            self.sync_transfer_stream()?;
        }
        self.prefetch_in_flight = None;

        let raw = self.slots[target].take();
        if let SlotState::Raw { block_idx: idx, mut tensors, fp8_pending, events } = raw {
            Self::prepare_weights(&mut tensors, fp8_pending, self.native_layout)?;
            let arc = Arc::new(tensors);
            events.compute_recorded.store(false, Ordering::Release);
            self.slots[target] = SlotState::Prepared {
                block_idx: idx,
                tensors: arc.clone(),
                events,
            };
            self.active = target;
            return Ok(arc);
        }

        // Empty block fallback — `prefetch_block` already promoted to
        // Prepared with empty tensors. Just locate it and return.
        if let SlotState::Prepared { ref tensors, .. } = self.slots[target] {
            self.active = target;
            return Ok(tensors.clone());
        }

        anyhow::bail!("await_block: slot in unexpected state after prefetch")
    }

    /// Like [`Self::await_block`] but returns a scoped [`BlockHandle`].
    ///
    /// The handle marks the slot as in-use; the next `prefetch_block` that
    /// would reuse the slot waits on the handle's `compute_done` event
    /// before issuing H2D, instead of a host-side `cudaDeviceSynchronize`.
    ///
    /// Hold the handle for the entire duration of the block's compute (i.e.
    /// past every kernel launch that reads the block's weights), then drop
    /// it. Drop is the signal that compute on this block is complete.
    pub fn await_block_handle(
        &mut self,
        block_idx: usize,
    ) -> anyhow::Result<BlockHandle> {
        // Reuse await_block to get the prepared Arc; then attach the slot's
        // event tracker into a fresh handle.
        let tensors = self.await_block(block_idx)?;
        let events = self.slots[self.active]
            .events()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("await_block_handle: active slot has no events"))?;
        // A fresh handle takes ownership of the slot's compute lifecycle.
        // Reset the flag so a new compute_done event will be recorded on
        // drop, and the next prefetch will wait on it.
        events.compute_recorded.store(false, Ordering::Release);
        Ok(BlockHandle { tensors, events })
    }

    /// Backward-compatible sync API. Same as prefetch + await.
    /// Existing callers keep working without changes.
    pub fn ensure_block(
        &mut self,
        block_idx: usize,
    ) -> anyhow::Result<Arc<HashMap<String, Tensor>>> {
        self.prefetch_block(block_idx)?;
        self.await_block(block_idx)
    }

    /// Drop GPU-side block tensors from both slots.
    ///
    /// Drains in-flight transfers and waits for any prior default-stream
    /// compute that may still be reading slot storage before freeing it.
    pub fn evict_block(&mut self) {
        // Drain any prefetch in flight so we don't drop tensors out from
        // under the transfer stream.
        if self.prefetch_in_flight.is_some() {
            let _ = self.sync_transfer_stream();
            self.prefetch_in_flight = None;
        }
        // Wait for any pending default-stream compute that may still be
        // reading slot storage.
        let s = unsafe { cudaDeviceSynchronize() };
        if s != 0 {
            log::warn!("evict_block: cudaDeviceSynchronize returned {s}");
        }
        self.slots[0] = SlotState::Empty;
        self.slots[1] = SlotState::Empty;
    }

    /// How many blocks are loaded.
    pub fn block_count(&self) -> usize {
        self.cpu_blocks.len()
    }

    /// Total pinned CPU memory used, in bytes.
    pub fn pinned_bytes(&self) -> usize {
        self.total_pinned_bytes
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn sync_transfer_stream(&self) -> anyhow::Result<()> {
        unsafe {
            cudarc::driver::result::stream::synchronize(self.transfer_stream.stream)
                .map_err(|e| anyhow::anyhow!("stream sync: {e:?}"))
        }
    }

    fn prepare_weights(
        bw: &mut HashMap<String, Tensor>,
        fp8_pending: HashMap<String, Fp8Pending>,
        native_layout: bool,
    ) -> anyhow::Result<()> {
        // Dequant any FP8-pending entries first. The transfer-stream sync
        // before this call guarantees the H2D is done; the dequant kernel
        // runs on the default stream (same as the transpose below). After
        // the kernel is enqueued the Fp8Pending `data` drops are safe
        // because the kernel launch already captured the pointer.
        for (key, pending) in fp8_pending {
            let tensor = crate::ops::fused_inference::dequant_fp8_to_bf16(
                &pending.data,
                pending.scale,
                Shape::from_dims(&pending.shape),
                &pending.data.device().clone(),
            )
            .map_err(|e| anyhow::anyhow!("FP8 dequant for {key}: {e:?}"))?;
            bw.insert(key, tensor);
        }

        let keys: Vec<String> = bw.keys().cloned().collect();
        for key in keys {
            let t = bw.remove(&key).unwrap();
            let t = if t.dtype() != DType::BF16 { t.to_dtype(DType::BF16)? } else { t };
            let t = t.requires_grad_(false);
            // Default (legacy) layout: pre-transpose every 2D `.weight` to
            // `[Cin, Cout]` for callers using `Tensor::matmul`. When
            // `native_layout` is set (callers using `fused_linear3d_native`),
            // skip the transpose so the GEMM gets PyTorch-native
            // `[Cout, Cin]` and uses cuBLASLt TRANSA=T internally.
            let t = if !native_layout
                && key.ends_with(".weight")
                && t.rank() == 2
                && !key.ends_with(".scale")
            {
                t.transpose()?.requires_grad_(false)
            } else {
                t
            };
            bw.insert(key, t);
        }
        Ok(())
    }

    fn mmap_safetensors(path: &str) -> anyhow::Result<(String, usize, memmap2::Mmap)> {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("failed to open {path}: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| anyhow::anyhow!("failed to mmap {path}: {e}"))?;
        if mmap.len() < 8 {
            anyhow::bail!("file too small for safetensors: {path}");
        }
        let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_size;
        if header_end > mmap.len() {
            anyhow::bail!("header extends past EOF in {path}");
        }
        let header = std::str::from_utf8(&mmap[8..header_end])
            .map_err(|e| anyhow::anyhow!("invalid UTF-8 in header of {path}: {e}"))?
            .to_string();
        Ok((header, header_end, mmap))
    }
}

// ---------------------------------------------------------------------------
// BlockHandle — Phase 0 scoped slot lifetime
// ---------------------------------------------------------------------------

/// RAII handle to a prepared block in a `BlockOffloader` slot.
///
/// Returned by [`BlockOffloader::await_block_handle`]. Holding the handle
/// signals "compute on this block is still in flight"; dropping it records a
/// default-stream `compute_done` event so a subsequent `prefetch_block` that
/// wants the same slot can wait on the GPU instead of stalling the host.
///
/// Deref to `&HashMap<String, Tensor>` for ergonomic tensor lookup. If the
/// caller needs to clone the underlying `Arc` (e.g. to thread tensors into
/// downstream APIs that expect `Arc<HashMap<...>>`), use [`Self::arc`].
///
/// **Lifetime contract**: drop the handle AFTER every kernel that reads the
/// block's weights has been queued on the default stream. Dropping mid-block
/// would record `compute_done` too early and let the next prefetch overwrite
/// the slot before the in-flight kernels read it.
pub struct BlockHandle {
    tensors: Arc<HashMap<String, Tensor>>,
    events: Arc<SlotEvents>,
}

impl BlockHandle {
    /// Borrow the block's weight map.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Look up a single weight by name.
    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }

    /// Clone the underlying `Arc<HashMap>`. The slot's compute lifetime is
    /// still tied to the handle's drop, not the cloned Arc — callers that
    /// need slot-safe lifetime semantics must not retain the cloned Arc past
    /// the handle's drop.
    pub fn arc(&self) -> Arc<HashMap<String, Tensor>> {
        Arc::clone(&self.tensors)
    }
}

impl std::ops::Deref for BlockHandle {
    type Target = HashMap<String, Tensor>;
    fn deref(&self) -> &Self::Target {
        &self.tensors
    }
}

impl Drop for BlockHandle {
    fn drop(&mut self) {
        // Record compute_done on the default stream. Any later
        // `prefetch_block` that wants this slot will wait on this event
        // (via `cudaStreamWaitEvent` on the transfer stream) before H2D.
        if let Err(e) = self.events.compute_done.record_default() {
            log::error!("BlockHandle drop: failed to record compute_done event: {e}");
        }
        self.events.compute_recorded.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Dtype conversion helpers
// ---------------------------------------------------------------------------

fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mantissa = bits & 0x7;
    if exp == 0xF && mantissa == 0x7 { return f32::NAN; }
    let f = if exp == 0 {
        (mantissa as f32) / 8.0 * (2.0f32).powi(-6)
    } else {
        (1.0 + mantissa as f32 / 8.0) * (2.0f32).powi(exp as i32 - 7)
    };
    if sign == 1 { -f } else { f }
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13));
    }
    if exp == 0x1F {
        if frac == 0 { return f32::from_bits((sign << 31) | (0xFF << 23)); }
        return f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13));
    }
    let f32_exp = exp + (127 - 15);
    f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
}

/// Per-block byte size (BF16 footprint) for the streaming path. Used
/// by the strategy plug-point to populate `OffloaderState::block_sizes`.
fn streaming_blocks_byte_sizes(blocks: &[Vec<StreamingTensorEntry>]) -> Vec<usize> {
    blocks
        .iter()
        .map(|entries| entries.iter().map(|e| e.num_elems * 2).sum::<usize>())
        .collect()
}

#[inline]
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}
