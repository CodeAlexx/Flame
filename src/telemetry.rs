use crate::{DType, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};

static BF16_BYTES: AtomicU64 = AtomicU64::new(0);
static F32_BYTES: AtomicU64 = AtomicU64::new(0);
static DTYPE_TRAPS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct TelemetrySnapshot {
    pub bf16_bytes: u64,
    pub f32_bytes: u64,
    pub dtype_traps: u64,
}

#[inline]
pub fn reset_counters() {
    BF16_BYTES.store(0, Ordering::Relaxed);
    F32_BYTES.store(0, Ordering::Relaxed);
    DTYPE_TRAPS.store(0, Ordering::Relaxed);
}

#[inline]
pub fn record_tensor_bytes(_tag: &str, tensor: &Tensor) {
    let bytes = tensor.dtype().size_in_bytes() as u64 * tensor.shape().elem_count() as u64;
    match tensor.dtype() {
        DType::BF16 => {
            BF16_BYTES.fetch_add(bytes, Ordering::Relaxed);
        }
        DType::F32 => {
            F32_BYTES.fetch_add(bytes, Ordering::Relaxed);
        }
        _ => {}
    }
}

#[inline]
pub fn record_dtype_trap(_op: &str, _logical: DType, _storage: DType) {
    DTYPE_TRAPS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn snapshot() -> TelemetrySnapshot {
    TelemetrySnapshot {
        bf16_bytes: BF16_BYTES.load(Ordering::Relaxed),
        f32_bytes: F32_BYTES.load(Ordering::Relaxed),
        dtype_traps: DTYPE_TRAPS.load(Ordering::Relaxed),
    }
}
