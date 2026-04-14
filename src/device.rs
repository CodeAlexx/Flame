use crate::{Error, Result};
use core::ffi::c_void;
use cudarc::driver::CudaDevice as CudarcDevice;
use std::sync::{Arc, OnceLock};

static GLOBAL_DEV: OnceLock<Arc<CudarcDevice>> = OnceLock::new();

// Raw FFI for CUDA mempool APIs (not in cuda_runtime_sys 0.3)
extern "C" {
    fn cudaDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: i32) -> i32;
    fn cudaMemPoolSetAttribute(pool: *mut c_void, attr: i32, value: *mut c_void) -> i32;
    fn cudaMemPoolTrimTo(pool: *mut c_void, min_bytes_to_keep: usize) -> i32;
}

/// Trim the CUDA mempool: release cached (freed) GPU memory back to the driver.
///
/// `min_bytes_to_keep`: minimum bytes of cached memory to retain. Pass 0 to
/// release everything that's not currently in-use.
///
/// Safe to call during training between steps (e.g. after backward + optimizer step).
/// Particularly important for models with gradient checkpointing, where the
/// checkpoint recompute creates temporary allocations that the mempool caches
/// indefinitely with MAX threshold.
pub fn trim_cuda_mempool(min_bytes_to_keep: usize) {
    unsafe {
        let mut pool: *mut c_void = std::ptr::null_mut();
        let status = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        if status != 0 || pool.is_null() {
            return;
        }
        let status = cudaMemPoolTrimTo(pool, min_bytes_to_keep);
        if status != 0 {
            log::warn!("cudaMemPoolTrimTo failed (status={})", status);
        }
    }
}

/// Configure CUDA memory pool to cache aggressively (never release to OS).
/// This makes cudaMallocAsync/cudaFreeAsync near-zero cost after warmup.
fn configure_cuda_mempool(device_ordinal: i32) {
    // cudaMemPoolAttrReleaseThreshold = 4
    const ATTR_RELEASE_THRESHOLD: i32 = 4;
    unsafe {
        let mut pool: *mut c_void = std::ptr::null_mut();
        let status = cudaDeviceGetDefaultMemPool(&mut pool, device_ordinal);
        if status != 0 || pool.is_null() {
            log::warn!("Failed to get default CUDA mempool (status={})", status);
            return;
        }
        // Set release threshold to max — never release cached memory
        let mut threshold: u64 = u64::MAX;
        let status = cudaMemPoolSetAttribute(
            pool,
            ATTR_RELEASE_THRESHOLD,
            &mut threshold as *mut u64 as *mut c_void,
        );
        if status != 0 {
            log::warn!("Failed to set mempool release threshold (status={})", status);
        } else {
            log::info!("CUDA mempool: release threshold set to MAX (infinite caching)");
        }
    }
}

/// Get a global CUDA device (device 0). GPU-only; panics if CUDA init fails.
pub fn global_cuda_device() -> Arc<CudarcDevice> {
    GLOBAL_DEV
        .get_or_init(|| {
            let dev = CudarcDevice::new(0).expect(
                "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
            );
            configure_cuda_mempool(0);
            dev
        })
        .clone()
}

/// Device management for FLAME
#[derive(Clone)]
pub struct Device {
    inner: Arc<CudarcDevice>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("ordinal", &self.ordinal())
            .finish()
    }
}

impl Device {
    /// Wrap an existing Arc<CudaDevice> in a Device.
    pub fn from_arc(device: Arc<CudarcDevice>) -> Self {
        Self { inner: device }
    }

    /// Create a new device for the given GPU ordinal
    pub fn cuda(ordinal: usize) -> Result<Self> {
        let device = CudarcDevice::new(ordinal)
            .map_err(|e| Error::Cuda(format!("Failed to create CUDA device: {:?}", e)))?;
        Ok(Self { inner: device })
    }

    /// Get the underlying CUDA device
    pub fn cuda_device(&self) -> &Arc<CudarcDevice> {
        &self.inner
    }

    /// Get a clone of the underlying CUDA device Arc
    pub fn cuda_device_arc(&self) -> Arc<CudarcDevice> {
        self.inner.clone()
    }

    /// Get device ordinal
    pub fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    /// Synchronize the device
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize().map_err(|_| Error::CudaDriver)
    }

    /// Set random seed for the device
    pub fn set_seed(&self, _seed: u64) -> Result<()> {
        // For now, this is a no-op as FLAME doesn't have a global RNG state
        // In the future, we might want to integrate with cuRAND
        Ok(())
    }

    /// Create a CPU device (not supported in FLAME)
    pub fn cpu() -> Result<Self> {
        Err(Error::InvalidOperation(
            "FLAME only supports CUDA devices".into(),
        ))
    }

    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        false // FLAME only supports CUDA
    }

    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        true // FLAME only supports CUDA
    }

    /// Returns the CUDA stream as a raw pointer usable by FFI launchers.
    /// Currently returns the default stream (null) unless a custom stream is plumbed.
    pub fn cuda_stream_raw_ptr(&self) -> *mut core::ffi::c_void {
        core::ptr::null_mut()
    }
}

/// Extension trait to expose a raw CUDA stream pointer from cudarc's `CudaDevice`.
/// Currently returns the default (null) stream, which is valid for kernel launches.
pub trait CudaStreamRawPtrExt {
    fn cuda_stream_raw_ptr(&self) -> *mut c_void;
}

impl CudaStreamRawPtrExt for cudarc::driver::CudaDevice {
    fn cuda_stream_raw_ptr(&self) -> *mut c_void {
        core::ptr::null_mut()
    }
}

impl From<Arc<CudarcDevice>> for Device {
    fn from(device: Arc<CudarcDevice>) -> Self {
        Self { inner: device }
    }
}

/// Device enum for matching expected external APIs
#[derive(Clone, Debug)]
pub enum DeviceEnum {
    Cuda(Device),
}

impl DeviceEnum {
    /// Create a CUDA device
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Ok(DeviceEnum::Cuda(Device::cuda(ordinal)?))
    }

    /// Check if CPU device
    pub fn is_cpu(&self) -> bool {
        false
    }

    /// Check if CUDA device  
    pub fn is_cuda(&self) -> bool {
        true
    }
}
