#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Arc, OnceLock};

use crate::cuda_ops_ffi::{
    flame_arena_alloc, flame_arena_create, flame_arena_destroy, flame_arena_record_and_release,
    flame_arena_reset, flame_bf16_copy_async, flame_bf16_zero_async,
    flame_conv2d_autotune_get_stats, flame_conv2d_autotune_reset_stats, flame_d2d_async,
    flame_d2h_async, flame_h2d_async, flame_sdpa_autotune_flush_cache,
    flame_sdpa_autotune_get_stats, flame_sdpa_autotune_reset_stats, flame_status_to_result,
    CudaStream, FlameConv2dAutotuneStats, FlameSdpaAutotuneStats, FlameStreamArenaHandle,
    FlameStreamArenaOpaque, FLAME_CUDA_OK,
};
use crate::device::CudaStreamRawPtrExt;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;

#[derive(Debug)]
pub struct ArenaLeaseInner {
    device_idx: i32,
    stream: CudaStream,
}

unsafe impl Send for ArenaLeaseInner {}
unsafe impl Sync for ArenaLeaseInner {}

impl Drop for ArenaLeaseInner {
    fn drop(&mut self) {
        if let Err(err) = arena_record_and_release(self.device_idx, &self.stream) {
            eprintln!(
                "arena_record_and_release failed for device {} stream {:?}: {}",
                self.device_idx,
                self.stream.as_raw(),
                err
            );
        }
    }
}

pub type ArenaLease = Arc<ArenaLeaseInner>;

#[derive(Debug)]
pub struct Bf16ArenaBuffer {
    ptr: NonNull<u16>,
    elems: usize,
    device: Arc<CudaDevice>,
    lease: ArenaLease,
}

impl Bf16ArenaBuffer {
    pub fn len(&self) -> usize {
        self.elems
    }

    pub fn as_ptr(&self) -> *const u16 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut u16 {
        self.ptr.as_ptr()
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn into_tensor(self, shape: Shape) -> Result<Tensor> {
        if shape.elem_count() != self.elems {
            return Err(Error::ShapeMismatch {
                expected: shape,
                got: Shape::from_dims(&[self.elems]),
            });
        }
        let tensor =
            Tensor::from_bf16_arena(shape, self.device.clone(), self.ptr, self.lease.clone())?;
        Ok(tensor)
    }
}

#[derive(Hash, Eq, PartialEq)]
struct ArenaKey {
    device: i32,
    stream: *mut c_void,
}

struct ArenaEntry {
    handle: Option<NonNull<FlameStreamArenaOpaque>>,
    device: i32,
    stream: *mut c_void,
    capacity: u64,
    stats: ArenaStats,
}

impl ArenaEntry {
    fn new(device: i32, stream: *mut c_void) -> Self {
        Self {
            handle: None,
            device,
            stream,
            capacity: arena_capacity_bytes(),
            stats: ArenaStats::default(),
        }
    }

    fn ensure_handle(&mut self) -> Result<NonNull<FlameStreamArenaOpaque>> {
        if let Some(handle) = self.handle {
            return Ok(handle);
        }
        let mut raw: FlameStreamArenaHandle = ptr::null_mut();
        flame_status_to_result(
            unsafe { flame_arena_create(self.device, self.stream, self.capacity, &mut raw) },
            "flame_arena_create",
        )?;
        let handle = NonNull::new(raw)
            .ok_or_else(|| Error::Cuda("flame_arena_create returned null handle".into()))?;
        self.handle = Some(handle);
        Ok(handle)
    }
}

impl Drop for ArenaEntry {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let status = unsafe { flame_arena_destroy(handle.as_ptr()) };
            if status != FLAME_CUDA_OK {
                // We cannot propagate errors from Drop; log best-effort.
                eprintln!(
                    "flame_arena_destroy failed with status={} for device {} stream {:?}",
                    status, self.device, self.stream
                );
            }
        }
    }
}

struct ArenaCache {
    entries: HashMap<ArenaKey, ArenaEntry>,
}

thread_local! {
    static ARENAS: RefCell<ArenaCache> = RefCell::new(ArenaCache {
        entries: HashMap::new(),
    });
}

#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    pub allocations: u64,
    pub releases: u64,
    pub bytes_requested: u64,
    pub bytes_active: u64,
    pub bytes_peak: u64,
}

impl ArenaStats {
    fn record_alloc(&mut self, bytes: usize) {
        self.allocations = self.allocations.saturating_add(1);
        let bytes_u64 = bytes as u64;
        self.bytes_requested = self.bytes_requested.saturating_add(bytes_u64);
        self.bytes_active = self.bytes_active.saturating_add(bytes_u64);
        if self.bytes_active > self.bytes_peak {
            self.bytes_peak = self.bytes_active;
        }
    }

    fn record_release(&mut self) {
        self.releases = self.releases.saturating_add(1);
        self.bytes_active = 0;
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn arena_capacity_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        if let Some(bytes) = std::env::var("FLAME_ARENA_CAP")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            return bytes;
        }
        if let Some(mb) = std::env::var("FLAME_ARENA_CAP_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            return mb.saturating_mul(1024 * 1024);
        }
        1024 * 1024 * 1024 // default to 1 GiB arena
    })
}

fn with_entry<F, R>(device: i32, stream: &CudaStream, f: F) -> Result<R>
where
    F: FnOnce(&mut ArenaEntry) -> Result<R>,
{
    ARENAS
        .try_with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = ArenaKey {
                device,
                stream: stream.as_raw(),
            };
            let entry = cache
                .entries
                .entry(key)
                .or_insert_with(|| ArenaEntry::new(device, stream.as_raw()));
            f(entry)
        })
        .map_err(|_| Error::Cuda("flame arena TLS poisoned".into()))?
}

pub fn arena_alloc(
    device: i32,
    stream: &CudaStream,
    bytes: usize,
    align: usize,
) -> Result<*mut u8> {
    if bytes == 0 {
        return Err(Error::InvalidInput(
            "arena_alloc: zero-byte allocation requested".into(),
        ));
    }
    with_entry(device, stream, |entry| {
        let handle = entry.ensure_handle()?;
        let mut out: *mut c_void = ptr::null_mut();
        flame_status_to_result(
            unsafe {
                flame_arena_alloc(
                    handle.as_ptr(),
                    bytes as u64,
                    align.max(16) as u64,
                    &mut out,
                )
            },
            "flame_arena_alloc",
        )
        .map_err(|e| {
            eprintln!(
                "arena_alloc failed: bytes={} align={} error={:?}",
                bytes, align, e
            );
            e
        })?;
        entry.stats.record_alloc(bytes);
        Ok(out as *mut u8)
    })
}

pub fn arena_record_and_release(device: i32, stream: &CudaStream) -> Result<()> {
    // Callers must invoke this after enqueuing work that uses arena buffers so the
    // CUDA event fence can release the bump pointer before the next launch.
    with_entry(device, stream, |entry| {
        let handle = entry.ensure_handle()?;
        flame_status_to_result(
            unsafe { flame_arena_record_and_release(handle.as_ptr()) },
            "flame_arena_record_and_release",
        )?;
        entry.stats.record_release();
        Ok(())
    })
}

pub fn arena_reset(device: i32, stream: &CudaStream) -> Result<()> {
    with_entry(device, stream, |entry| {
        let handle = entry.ensure_handle()?;
        flame_status_to_result(
            unsafe { flame_arena_reset(handle.as_ptr()) },
            "flame_arena_reset",
        )
    })
}

pub fn arena_stats(device: i32, stream: &CudaStream) -> Result<ArenaStats> {
    ARENAS
        .try_with(|cache| {
            let cache = cache.borrow();
            let key = ArenaKey {
                device,
                stream: stream.as_raw(),
            };
            Ok(cache
                .entries
                .get(&key)
                .map(|entry| entry.stats.clone())
                .unwrap_or_default())
        })
        .map_err(|_| Error::Cuda("flame arena TLS poisoned".into()))?
}

pub fn arena_reset_stats(device: i32, stream: &CudaStream) -> Result<()> {
    let result = ARENAS
        .try_with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = ArenaKey {
                device,
                stream: stream.as_raw(),
            };
            if let Some(entry) = cache.entries.get_mut(&key) {
                entry.stats.reset();
            }
            Ok(())
        })
        .map_err(|_| Error::Cuda("flame arena TLS poisoned".into()))?;

    result
}

pub fn h2d_async(
    dst_device: *mut c_void,
    src_host: *const c_void,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    flame_status_to_result(
        unsafe { flame_h2d_async(dst_device, src_host, bytes as u64, stream.as_raw()) },
        "flame_h2d_async",
    )
}

pub fn d2h_async(
    dst_host: *mut c_void,
    src_device: *const c_void,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    flame_status_to_result(
        unsafe { flame_d2h_async(dst_host, src_device, bytes as u64, stream.as_raw()) },
        "flame_d2h_async",
    )
}

pub fn d2d_async(
    dst_device: *mut c_void,
    src_device: *const c_void,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    flame_status_to_result(
        unsafe { flame_d2d_async(dst_device, src_device, bytes as u64, stream.as_raw()) },
        "flame_d2d_async",
    )
}

pub fn bf16_zero_async(ptr: *mut c_void, elems: usize, stream: &CudaStream) -> Result<()> {
    if elems == 0 {
        return Ok(());
    }
    flame_status_to_result(
        unsafe { flame_bf16_zero_async(ptr, elems as u64, stream.as_raw()) },
        "flame_bf16_zero_async",
    )
}

pub fn bf16_copy_async(
    dst: *mut c_void,
    src: *const c_void,
    elems: usize,
    stream: &CudaStream,
) -> Result<()> {
    if elems == 0 {
        return Ok(());
    }
    flame_status_to_result(
        unsafe { flame_bf16_copy_async(dst, src, elems as u64, stream.as_raw()) },
        "flame_bf16_copy_async",
    )
}

pub fn borrow_bf16_arena_buffer(
    device: Arc<CudaDevice>,
    stream: &CudaStream,
    elems: usize,
    align: usize,
) -> Result<Bf16ArenaBuffer> {
    if elems == 0 {
        return Err(Error::InvalidInput(
            "borrow_bf16_arena_buffer: zero elements requested".into(),
        ));
    }
    let bytes = elems
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| Error::InvalidInput("bf16 arena borrow byte size overflow".into()))?;
    let device_idx = device.ordinal() as i32;
    let raw = arena_alloc(device_idx, stream, bytes, align.max(16))?;
    let ptr = NonNull::new(raw as *mut u16)
        .ok_or_else(|| Error::Cuda("arena_alloc returned null pointer for BF16 scratch".into()))?;
    let lease = Arc::new(ArenaLeaseInner {
        device_idx,
        stream: *stream,
    });
    Ok(Bf16ArenaBuffer {
        ptr,
        elems,
        device,
        lease,
    })
}

pub fn borrow_bf16_arena_tensor(
    device: Arc<CudaDevice>,
    stream: &CudaStream,
    shape: Shape,
    align: usize,
) -> Result<Tensor> {
    let elems = shape.elem_count();
    if elems == 0 {
        return Err(Error::InvalidInput(
            "borrow_bf16_arena_tensor: zero elements requested".into(),
        ));
    }
    let buffer = borrow_bf16_arena_buffer(device, stream, elems, align)?;
    buffer.into_tensor(shape)
}

#[derive(Debug)]
pub struct ArenaScratch {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    align: usize,
}

impl ArenaScratch {
    pub const DEFAULT_ALIGN: usize = 128;

    pub fn new(device: Arc<CudaDevice>, stream: CudaStream) -> Self {
        Self::with_align(device, stream, Self::DEFAULT_ALIGN)
    }

    pub fn with_align(device: Arc<CudaDevice>, stream: CudaStream, align: usize) -> Self {
        Self {
            device,
            stream,
            align: align.max(16),
        }
    }

    pub fn from_tensor(tensor: &Tensor) -> Self {
        let stream = CudaStream::from_raw(tensor.device().cuda_stream_raw_ptr());
        Self::new(tensor.device().clone(), stream)
    }

    pub fn from_tensor_with_align(tensor: &Tensor, align: usize) -> Self {
        let stream = CudaStream::from_raw(tensor.device().cuda_stream_raw_ptr());
        Self::with_align(tensor.device().clone(), stream, align)
    }

    pub fn borrow_shape(&self, shape: Shape) -> Result<Tensor> {
        borrow_bf16_arena_tensor(self.device.clone(), &self.stream, shape, self.align)
    }

    pub fn borrow_like(&self, tensor: &Tensor) -> Result<Tensor> {
        self.borrow_shape(tensor.shape().clone())
    }

    pub fn copy_from(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.dtype() != DType::BF16 || tensor.storage_dtype() != DType::BF16 {
            return tensor.clone_result();
        }
        let mut out = self.borrow_like(tensor)?;
        out.copy_bf16_region_from(0, tensor, 0, tensor.shape().elem_count())?;
        Ok(out)
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn align(&self) -> usize {
        self.align
    }
}

impl Clone for ArenaScratch {
    fn clone(&self) -> Self {
        Self::with_align(
            self.device.clone(),
            CudaStream::from_raw(self.stream.as_raw()),
            self.align,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dAutotuneStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tuned: u64,
    pub weak: u64,
    pub fallbacks: u64,
    pub workspace_skips: u64,
    pub errors: u64,
    pub reprobes: u64,
}

impl From<FlameConv2dAutotuneStats> for Conv2dAutotuneStats {
    fn from(raw: FlameConv2dAutotuneStats) -> Self {
        Self {
            cache_hits: raw.cache_hits,
            cache_misses: raw.cache_misses,
            tuned: raw.tuned,
            weak: raw.weak,
            fallbacks: raw.fallbacks,
            workspace_skips: raw.workspace_skips,
            errors: raw.errors,
            reprobes: raw.reprobes,
        }
    }
}

pub fn conv2d_autotune_stats() -> Result<Conv2dAutotuneStats> {
    let mut raw = FlameConv2dAutotuneStats::default();
    flame_status_to_result(
        unsafe { flame_conv2d_autotune_get_stats(&mut raw as *mut _) },
        "flame_conv2d_autotune_get_stats",
    )?;
    Ok(raw.into())
}

pub fn reset_conv2d_autotune_stats() -> Result<()> {
    flame_status_to_result(
        unsafe { flame_conv2d_autotune_reset_stats() },
        "flame_conv2d_autotune_reset_stats",
    )
}

#[derive(Debug, Clone, Copy)]
pub struct SdpaAutotuneStats {
    pub env_forced: u64,
    pub clamped: u64,
    pub skipped: u64,
    pub fallback: u64,
    pub errors: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tuned: u64,
    pub last_q_chunk: u64,
    pub last_k_chunk: u64,
    pub cache_saved: u64,
    pub cache_loads: u64,
    pub cache_load_errors: u64,
    pub cache_entries: u64,
    pub last_candidate_count: u64,
    pub last_best_time_ns: u64,
    pub last_plan_source: u64,
    pub last_shape_b: u64,
    pub last_shape_h: u64,
    pub last_shape_q: u64,
    pub last_shape_k: u64,
    pub last_shape_dh: u64,
    pub last_shape_dv: u64,
    pub last_shape_mask_heads: u64,
    pub last_shape_causal: u64,
}

impl From<FlameSdpaAutotuneStats> for SdpaAutotuneStats {
    fn from(raw: FlameSdpaAutotuneStats) -> Self {
        Self {
            env_forced: raw.env_forced,
            clamped: raw.clamped,
            skipped: raw.skipped,
            fallback: raw.fallback,
            errors: raw.errors,
            cache_hits: raw.cache_hits,
            cache_misses: raw.cache_misses,
            tuned: raw.tuned,
            last_q_chunk: raw.last_q_chunk,
            last_k_chunk: raw.last_k_chunk,
            cache_saved: raw.cache_saved,
            cache_loads: raw.cache_loads,
            cache_load_errors: raw.cache_load_errors,
            cache_entries: raw.cache_entries,
            last_candidate_count: raw.last_candidate_count,
            last_best_time_ns: raw.last_best_time_ns,
            last_plan_source: raw.last_plan_source,
            last_shape_b: raw.last_shape_b,
            last_shape_h: raw.last_shape_h,
            last_shape_q: raw.last_shape_q,
            last_shape_k: raw.last_shape_k,
            last_shape_dh: raw.last_shape_dh,
            last_shape_dv: raw.last_shape_dv,
            last_shape_mask_heads: raw.last_shape_mask_heads,
            last_shape_causal: raw.last_shape_causal,
        }
    }
}

pub fn sdpa_autotune_stats() -> Result<SdpaAutotuneStats> {
    let mut raw = FlameSdpaAutotuneStats::default();
    flame_status_to_result(
        unsafe { flame_sdpa_autotune_get_stats(&mut raw as *mut _) },
        "flame_sdpa_autotune_get_stats",
    )?;
    Ok(raw.into())
}

pub fn reset_sdpa_autotune_stats() -> Result<()> {
    flame_status_to_result(
        unsafe { flame_sdpa_autotune_reset_stats() },
        "flame_sdpa_autotune_reset_stats",
    )
}

pub fn flush_sdpa_autotune_cache() -> Result<()> {
    flame_status_to_result(
        unsafe { flame_sdpa_autotune_flush_cache() },
        "flame_sdpa_autotune_flush_cache",
    )
}
