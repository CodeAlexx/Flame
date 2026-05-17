//! Process-lifetime cuBLASLt handle and stream pointer, fronted by a
//! thread-local cache.
//!
//! Background: every `bmm_bf16_fp32acc_out` / `matmul_bf16_trans` /
//! `fused_linear3d*` / `fused_modulate` / `fused_rms_norm` call goes through
//! `stream_ptr(device)` + `cublaslt_handle_ptr(device)` to obtain the raw
//! pointers the .cu shims expect. Pre-TLS, every lookup grabbed a global
//! mutex on a HashMap<ordinal, ctx>; PyTorch's `CublasHandlePool.cpp`
//! reference shows the same pattern can be made lock-free on the hot path
//! by stashing the pointer in thread-local storage (TLS) keyed on the
//! device ordinal, and only consulting the global map on a TLS miss
//! (first call on a new thread for a given device).
//!
//! Construction is still routed through the global mutex so cublasLtCreate
//! runs exactly once per device, process-wide. Multiple training threads
//! sharing one logical device share the handle (cuBLASLt handles are
//! Send+Sync per NVIDIA docs — concurrent calls are serialized by the
//! library when needed).
//!
//! Rollback: `FLAME_HANDLE_TLS_DISABLE=1` falls back to the
//! global-mutex-on-every-call behavior. Set this if a multi-device
//! workflow needs to see live changes after `cudaSetDevice`-equivalent
//! migrations, or to bisect a suspected TLS-coherency issue.

use crate::{Error, Result};
use core::ffi::c_void;
use cudarc::driver::CudaDevice;
use once_cell::sync::Lazy;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

struct LtContext {
    stream: *mut c_void,
    handle: *mut c_void,
}

unsafe impl Send for LtContext {}
unsafe impl Sync for LtContext {}

// Global, mutex-protected source of truth. Construction happens here
// (cublasLtCreate is a heavy call we want to make exactly once).
static CONTEXTS: Lazy<Mutex<HashMap<usize, LtContext>>> = Lazy::new(|| Mutex::new(HashMap::new()));

// Cheap TLS cache. `(ordinal, stream, handle)` — when the requested
// device ordinal matches the cached ordinal, we return the cached
// pointers without touching the global mutex.
thread_local! {
    static TLS_CTX: Cell<Option<(usize, *mut c_void, *mut c_void)>> = const { Cell::new(None) };
}

fn tls_disabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("FLAME_HANDLE_TLS_DISABLE").ok().as_deref() == Some("1"))
}

fn init_context(_device: &Arc<CudaDevice>) -> Result<LtContext> {
    // Use the default CUDA stream (null) so cublasLt GEMMs pipeline with
    // elementwise kernels without implicit sync barriers between streams.
    let stream: *mut c_void = core::ptr::null_mut();

    let mut handle: *mut c_void = core::ptr::null_mut();
    let handle_status = unsafe { crate::cuda::ffi::cublasLtCreate(&mut handle as *mut _) };
    if handle_status != 0 {
        return Err(Error::Cuda(format!(
            "cublasLtCreate failed: {}",
            handle_status
        )));
    }

    Ok(LtContext { stream, handle })
}

/// Look the context up in the global map, constructing it on first use.
/// Returns `(stream, handle)` for the requested device ordinal.
fn lookup_or_init(device: &Arc<CudaDevice>) -> Result<(*mut c_void, *mut c_void)> {
    let key = device.ordinal();
    {
        let map = CONTEXTS.lock().unwrap();
        if let Some(ctx) = map.get(&key) {
            return Ok((ctx.stream, ctx.handle));
        }
    }

    let ctx = init_context(device)?;
    let pair = (ctx.stream, ctx.handle);
    let mut map = CONTEXTS.lock().unwrap();
    // Another thread may have inserted first; that's fine — both have the
    // same stream (null) and valid handles, and cublasLtHandle leaks at
    // process exit anyway (singleton-per-device by design).
    map.entry(key).or_insert(ctx);
    Ok(pair)
}

/// Return `(stream, handle)` for `device`, hitting the TLS cache when the
/// device ordinal matches the last call on this thread.
#[inline]
fn cached(device: &Arc<CudaDevice>) -> Result<(*mut c_void, *mut c_void)> {
    if tls_disabled() {
        return lookup_or_init(device);
    }

    let ordinal = device.ordinal();
    let hit = TLS_CTX.with(|c| match c.get() {
        Some((o, s, h)) if o == ordinal => Some((s, h)),
        _ => None,
    });
    if let Some(pair) = hit {
        return Ok(pair);
    }

    let pair = lookup_or_init(device)?;
    TLS_CTX.with(|c| c.set(Some((ordinal, pair.0, pair.1))));
    Ok(pair)
}

pub fn stream_ptr(device: &Arc<CudaDevice>) -> Result<*mut c_void> {
    cached(device).map(|(s, _)| s)
}

pub fn cublaslt_handle_ptr(device: &Arc<CudaDevice>) -> Result<*mut c_void> {
    cached(device).map(|(_, h)| h)
}
