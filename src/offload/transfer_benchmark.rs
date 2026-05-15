//! On-device PCIe bandwidth benchmark for `BlockOffloader` planning.
//!
//! Rust port of `flextensor/memory_transfer_benchmark.py` +
//! `memory_transfer_interpolator.py`. Measures **actual** H2D and D2H
//! bandwidth at the current hardware/CUDA-driver combo and builds a
//! log-log interpolator that future strategy code can query as
//! `bytes → predicted seconds` (and inverse).
//!
//! ## Why this lives in flame-core
//!
//! The bandwidth profile is hardware-specific, not model-specific. Every
//! `BlockOffloader` caller (klein 9B, qwen, ltx2, wan22, nucleus, …) on a
//! given box sees the same PCIe bus. A single, cached, one-time benchmark
//! at process init is shared across all callers — per tenet §1 (fix the
//! primitive, ship every model).
//!
//! ## What it measures (and what it deliberately does NOT measure)
//!
//! * **Measures:** wall time on the GPU for `cudaMemcpyAsync` from
//!   pinned host memory to GPU and back, across a geometric sweep of
//!   transfer sizes. Timing uses `cudaEvent` start/stop bracketed around
//!   one `memcpy_async` followed by a single `cudaEventSynchronize` on
//!   the stop event — this gives true device-observed wall time, not a
//!   host-side `Instant::now()` window.
//! * **Does NOT measure:** the cost of any *kernel* launch on the same
//!   stream, prefetch overlap quality, or per-step memory churn from the
//!   trainer. Those belong to the
//!   [`telemetry`](super::telemetry) module (host-observed wall time)
//!   and to the future Phase 2/3 work, respectively.
//!
//! ## Sync contract
//!
//! Per clauses 1 and 5 of `SPEED_CONTRACT.md`:
//!
//! * The bench is invoked **at init time only** — never on the per-step
//!   path. `cudaEventSynchronize` here is the standard CUDA timing
//!   pattern and is exempt under clause 1's "init time" allowance.
//! * On the live offloader hot path nothing in this module is touched.
//!   The bench result is consumed via [`TransferBandwidthProfile`], an
//!   already-computed struct.

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::{CudaDevice, CudaStream, DevicePtr};

use crate::{
    memcpy_async_device_to_host, memcpy_async_host_to_device, PinnedAllocFlags, PinnedHostBuffer,
};

// --- CUDA event FFI -------------------------------------------------------
//
// Mirrors the same FFI block in `offload/mod.rs::CudaEvent`. Local
// duplication keeps the public flame-core surface from gaining a new
// event helper just for this one consumer; tenet §2 ("APIs make the right
// thing easy") is preserved because no caller has to learn a new event
// API to use [`run_benchmark`].

extern "C" {
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
}

struct BenchEvent {
    raw: *mut c_void,
}

impl BenchEvent {
    fn new() -> anyhow::Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let s = unsafe { cudaEventCreate(&mut raw) };
        if s != 0 {
            anyhow::bail!("transfer_benchmark: cudaEventCreate failed: {s}");
        }
        Ok(Self { raw })
    }
    fn record(&self, stream: *mut c_void) -> anyhow::Result<()> {
        let s = unsafe { cudaEventRecord(self.raw, stream) };
        if s != 0 {
            anyhow::bail!("transfer_benchmark: cudaEventRecord failed: {s}");
        }
        Ok(())
    }
    fn synchronize(&self) -> anyhow::Result<()> {
        let s = unsafe { cudaEventSynchronize(self.raw) };
        if s != 0 {
            anyhow::bail!("transfer_benchmark: cudaEventSynchronize failed: {s}");
        }
        Ok(())
    }
}

impl Drop for BenchEvent {
    fn drop(&mut self) {
        unsafe {
            cudaEventDestroy(self.raw);
        }
    }
}

fn elapsed_ms(start: &BenchEvent, end: &BenchEvent) -> anyhow::Result<f32> {
    let mut ms: f32 = 0.0;
    let s = unsafe { cudaEventElapsedTime(&mut ms, start.raw, end.raw) };
    if s != 0 {
        anyhow::bail!("transfer_benchmark: cudaEventElapsedTime failed: {s}");
    }
    Ok(ms)
}

// --- Result types ---------------------------------------------------------

/// Direction tag for a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
}

/// One measured point on the bandwidth curve.
#[derive(Debug, Clone, Copy)]
pub struct TransferMeasurement {
    pub bytes: usize,
    pub direction: TransferDirection,
    /// Median over the trial repeats, in nanoseconds.
    pub duration_ns: u64,
    /// Per-call effective bandwidth at this size, in bytes per second.
    /// Computed as `bytes / duration_seconds`.
    pub bandwidth_bps: f64,
}

impl TransferMeasurement {
    /// Bandwidth in GB/s (1e9 bytes per second). Convenience accessor for
    /// human-readable summaries.
    pub fn bandwidth_gbps(&self) -> f64 {
        self.bandwidth_bps / 1e9
    }
}

/// Knobs for the benchmark sweep.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Smallest transfer size to measure, in bytes. Must be ≥ 1.
    pub min_bytes: usize,
    /// Largest transfer size to measure, in bytes.
    pub max_bytes: usize,
    /// Number of geometric sample points between `min_bytes` and
    /// `max_bytes`, inclusive. ≥ 2 (otherwise the interpolator falls
    /// back to flat extrapolation).
    pub samples: usize,
    /// Repeat count per (size, direction). The recorded `duration_ns`
    /// is the median across these.
    pub trials: usize,
    /// Warmup transfers before timed trials begin (each size, each
    /// direction). One warmup is enough to fault in the pinned pages and
    /// pull the kernel-launch cost out of the first measurement.
    pub warmup_trials: usize,
    /// If `true`, also measure D2H. If `false`, only measure H2D
    /// (the path BlockOffloader cares about most). Defaults to
    /// `true` because clause 5 of the speed contract spans both
    /// directions.
    pub measure_d2h: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            min_bytes: 1024,              // 1 KiB
            max_bytes: 256 * 1024 * 1024, // 256 MiB
            samples: 9,                   // 1K, 4K, 16K, 64K, 256K, 1M, 4M, 16M, 64M (approx)
            trials: 5,
            warmup_trials: 1,
            measure_d2h: true,
        }
    }
}

/// Bandwidth profile for one device: a set of measurements + the
/// log-log interpolator built from them.
///
/// The profile is conceptually immutable after construction. Hold it in
/// an `Arc` if multiple subsystems need to consult it.
#[derive(Debug, Clone)]
pub struct TransferBandwidthProfile {
    h2d: Vec<TransferMeasurement>,
    d2h: Vec<TransferMeasurement>,
    /// Effective peak (size = max_bytes) H2D bandwidth in bytes/sec.
    pub peak_h2d_bps: f64,
    /// Effective peak (size = max_bytes) D2H bandwidth in bytes/sec.
    /// Zero when D2H was disabled.
    pub peak_d2h_bps: f64,
}

impl TransferBandwidthProfile {
    /// Reconstruct a profile from previously-persisted parts. Intended only
    /// for [`crate::offload::state::load_profile`] — the field layout is a
    /// private contract between the two modules. Callers that want a fresh
    /// profile call [`run_benchmark`].
    #[doc(hidden)]
    pub fn __from_persisted_parts(
        h2d: Vec<TransferMeasurement>,
        d2h: Vec<TransferMeasurement>,
        peak_h2d_bps: f64,
        peak_d2h_bps: f64,
    ) -> Self {
        Self {
            h2d,
            d2h,
            peak_h2d_bps,
            peak_d2h_bps,
        }
    }

    /// All H2D measurements, ordered by increasing transfer size.
    pub fn h2d(&self) -> &[TransferMeasurement] {
        &self.h2d
    }

    /// All D2H measurements, ordered by increasing transfer size.
    pub fn d2h(&self) -> &[TransferMeasurement] {
        &self.d2h
    }

    /// Predict H2D wall time for an arbitrary `bytes` count using log-log
    /// linear interpolation across the measured points. Returns
    /// `Duration::ZERO` if `bytes == 0`.
    pub fn predict_h2d(&self, bytes: usize) -> Duration {
        predict_duration(&self.h2d, bytes)
    }

    /// Predict D2H wall time for an arbitrary `bytes` count. Returns
    /// `Duration::ZERO` if `bytes == 0` or D2H wasn't measured.
    pub fn predict_d2h(&self, bytes: usize) -> Duration {
        if self.d2h.is_empty() {
            return Duration::ZERO;
        }
        predict_duration(&self.d2h, bytes)
    }

    /// Format the profile as a stable diagnostic table — intended for
    /// `eprintln!` / log output. Matches the layout of FlexTensor's
    /// `format_memory_transfer_table`.
    pub fn format_table(&self) -> String {
        let width = 96;
        let bar = "=".repeat(width);
        let dash = "-".repeat(width);
        let mut lines: Vec<String> = Vec::new();
        lines.push(bar.clone());
        lines.push("Memory Transfer Bandwidth Profile".to_string());
        lines.push(bar.clone());
        lines.push(format!(
            "{:>14} {:>10} {:>10} {:>14} {:>14}",
            "Size (bytes)", "Size", "Dir", "Duration (ms)", "Bandwidth (GB/s)"
        ));
        lines.push(dash);
        for m in self.h2d.iter().chain(self.d2h.iter()) {
            let dir = match m.direction {
                TransferDirection::HostToDevice => "H2D",
                TransferDirection::DeviceToHost => "D2H",
            };
            lines.push(format!(
                "{:>14} {:>10} {:>10} {:>14.3} {:>14.3}",
                m.bytes,
                format_size(m.bytes),
                dir,
                m.duration_ns as f64 * 1e-6,
                m.bandwidth_gbps(),
            ));
        }
        lines.push(bar);
        lines.join("\n")
    }
}

fn format_size(b: usize) -> String {
    const K: f64 = 1024.0;
    const M: f64 = K * K;
    const G: f64 = K * K * K;
    let f = b as f64;
    if f >= G {
        format!("{:.1} GiB", f / G)
    } else if f >= M {
        format!("{:.1} MiB", f / M)
    } else if f >= K {
        format!("{:.1} KiB", f / K)
    } else {
        format!("{} B", b)
    }
}

/// Log-log linear interpolation in the style of FlexTensor's
/// `MemoryTransferInterpolator.bytes_to_duration`. Extrapolates beyond
/// the measured range using the slope between the two endpoint points
/// (clamping to nonnegative).
fn predict_duration(points: &[TransferMeasurement], bytes: usize) -> Duration {
    if bytes == 0 || points.is_empty() {
        return Duration::ZERO;
    }
    if points.len() == 1 {
        return Duration::from_nanos(points[0].duration_ns);
    }
    let log_x: Vec<f64> = points.iter().map(|m| (m.bytes as f64).ln()).collect();
    let log_y: Vec<f64> = points
        .iter()
        .map(|m| (m.duration_ns as f64).max(1.0).ln())
        .collect();
    let target = (bytes as f64).ln();

    // Find the bracket [i, i+1] s.t. log_x[i] ≤ target ≤ log_x[i+1].
    let n = log_x.len();
    let (i, i1) = if target <= log_x[0] {
        (0, 1)
    } else if target >= log_x[n - 1] {
        (n - 2, n - 1)
    } else {
        // Linear scan is fine for the small sample counts we measure.
        let mut idx = 0;
        for k in 0..(n - 1) {
            if target >= log_x[k] && target <= log_x[k + 1] {
                idx = k;
                break;
            }
        }
        (idx, idx + 1)
    };

    let slope = (log_y[i1] - log_y[i]) / (log_x[i1] - log_x[i]);
    let log_dur = log_y[i] + slope * (target - log_x[i]);
    let dur_ns = log_dur.exp().max(0.0);
    Duration::from_nanos(dur_ns as u64)
}

// --- Bench driver ---------------------------------------------------------

/// Run the bandwidth sweep on the given device with the given config.
/// Allocates one pinned host buffer + one GPU buffer of `cfg.max_bytes`
/// at start; reuses them across all samples to avoid measuring
/// allocator cost.
///
/// Cancels early with an error if any transfer fails.
pub fn run_benchmark(
    device: &Arc<CudaDevice>,
    cfg: &BenchmarkConfig,
) -> anyhow::Result<TransferBandwidthProfile> {
    anyhow::ensure!(cfg.min_bytes > 0, "min_bytes must be > 0");
    anyhow::ensure!(
        cfg.max_bytes >= cfg.min_bytes,
        "max_bytes ({}) must be >= min_bytes ({})",
        cfg.max_bytes,
        cfg.min_bytes
    );
    anyhow::ensure!(cfg.samples >= 1, "samples must be >= 1");
    anyhow::ensure!(cfg.trials >= 1, "trials must be >= 1");

    // Pinned host buffer + device buffer of max_bytes, reused across
    // every (size, direction, trial). Allocating once and slicing keeps
    // measurements free of allocator/free cost.
    let mut pinned: PinnedHostBuffer<u8> =
        PinnedHostBuffer::with_capacity_elems(cfg.max_bytes, PinnedAllocFlags::DEFAULT).map_err(
            |e| {
                anyhow::anyhow!(
                    "transfer_benchmark: pinned alloc of {} bytes failed: {e}",
                    cfg.max_bytes
                )
            },
        )?;
    // Touch each page so the OS materializes them and we don't time
    // first-touch faults inside the measured window. Pinned host pages
    // from `cudaHostAlloc` are technically already locked but writing
    // them keeps the access pattern realistic for the staging-buffer
    // refill in `offload/mod.rs`.
    {
        let buf = pinned.as_mut_bytes();
        for i in (0..cfg.max_bytes).step_by(4096) {
            buf[i] = (i & 0xFF) as u8;
        }
    }
    unsafe {
        pinned.set_len(cfg.max_bytes);
    }
    let pinned_ptr: *mut u8 = pinned.as_ptr() as *mut u8;

    let gpu_buf = unsafe { device.alloc::<u8>(cfg.max_bytes) }
        .map_err(|e| anyhow::anyhow!("transfer_benchmark: gpu alloc failed: {e:?}"))?;
    let gpu_ptr: *mut c_void = (*gpu_buf.device_ptr() as u64) as *mut c_void;

    let stream: CudaStream = device
        .fork_default_stream()
        .map_err(|e| anyhow::anyhow!("transfer_benchmark: stream create: {e:?}"))?;
    let stream_ptr = stream.stream as *mut c_void;

    let sizes = geometric_sweep(cfg.min_bytes, cfg.max_bytes, cfg.samples);

    let mut h2d: Vec<TransferMeasurement> = Vec::with_capacity(sizes.len());
    let mut d2h: Vec<TransferMeasurement> = Vec::with_capacity(sizes.len());

    for size in &sizes {
        let size = *size;
        if size == 0 || size > cfg.max_bytes {
            continue;
        }

        // ----- H2D -----
        for _ in 0..cfg.warmup_trials {
            memcpy_async_host_to_device(gpu_ptr, pinned_ptr as *const c_void, size, stream_ptr)
                .map_err(|e| anyhow::anyhow!("transfer_benchmark: H2D warmup: {e}"))?;
        }
        let mut samples_ns: Vec<u64> = Vec::with_capacity(cfg.trials);
        for _ in 0..cfg.trials {
            let start = BenchEvent::new()?;
            let end = BenchEvent::new()?;
            start.record(stream_ptr)?;
            memcpy_async_host_to_device(gpu_ptr, pinned_ptr as *const c_void, size, stream_ptr)
                .map_err(|e| anyhow::anyhow!("transfer_benchmark: H2D: {e}"))?;
            end.record(stream_ptr)?;
            end.synchronize()?;
            let ms = elapsed_ms(&start, &end)?;
            samples_ns.push((ms as f64 * 1e6) as u64);
        }
        let dur_ns = median_u64(&mut samples_ns);
        let bps = if dur_ns == 0 {
            0.0
        } else {
            (size as f64) / (dur_ns as f64 * 1e-9)
        };
        h2d.push(TransferMeasurement {
            bytes: size,
            direction: TransferDirection::HostToDevice,
            duration_ns: dur_ns,
            bandwidth_bps: bps,
        });

        // ----- D2H -----
        if cfg.measure_d2h {
            for _ in 0..cfg.warmup_trials {
                memcpy_async_device_to_host(pinned_ptr as *mut c_void, gpu_ptr, size, stream_ptr)
                    .map_err(|e| anyhow::anyhow!("transfer_benchmark: D2H warmup: {e}"))?;
            }
            let mut samples_ns: Vec<u64> = Vec::with_capacity(cfg.trials);
            for _ in 0..cfg.trials {
                let start = BenchEvent::new()?;
                let end = BenchEvent::new()?;
                start.record(stream_ptr)?;
                memcpy_async_device_to_host(pinned_ptr as *mut c_void, gpu_ptr, size, stream_ptr)
                    .map_err(|e| anyhow::anyhow!("transfer_benchmark: D2H: {e}"))?;
                end.record(stream_ptr)?;
                end.synchronize()?;
                let ms = elapsed_ms(&start, &end)?;
                samples_ns.push((ms as f64 * 1e6) as u64);
            }
            let dur_ns = median_u64(&mut samples_ns);
            let bps = if dur_ns == 0 {
                0.0
            } else {
                (size as f64) / (dur_ns as f64 * 1e-9)
            };
            d2h.push(TransferMeasurement {
                bytes: size,
                direction: TransferDirection::DeviceToHost,
                duration_ns: dur_ns,
                bandwidth_bps: bps,
            });
        }
    }

    let peak_h2d_bps = h2d.last().map(|m| m.bandwidth_bps).unwrap_or(0.0);
    let peak_d2h_bps = d2h.last().map(|m| m.bandwidth_bps).unwrap_or(0.0);

    Ok(TransferBandwidthProfile {
        h2d,
        d2h,
        peak_h2d_bps,
        peak_d2h_bps,
    })
}

fn geometric_sweep(min_bytes: usize, max_bytes: usize, samples: usize) -> Vec<usize> {
    if samples == 1 {
        return vec![max_bytes];
    }
    if min_bytes == max_bytes {
        return vec![min_bytes];
    }
    let lo = (min_bytes as f64).ln();
    let hi = (max_bytes as f64).ln();
    let mut out: Vec<usize> = (0..samples)
        .map(|i| {
            let t = i as f64 / (samples - 1) as f64;
            let v = (lo + t * (hi - lo)).exp().round() as usize;
            v.max(1).min(max_bytes)
        })
        .collect();
    // De-duplicate (round() can collapse very-close points at low end).
    out.sort_unstable();
    out.dedup();
    out
}

fn median_u64(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    let n = samples.len();
    if n % 2 == 0 {
        (samples[n / 2 - 1] + samples[n / 2]) / 2
    } else {
        samples[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_device() -> anyhow::Result<Arc<CudaDevice>> {
        CudaDevice::new(0).map_err(|e| anyhow::anyhow!("CUDA device 0 unavailable: {e:?}"))
    }

    /// Quick end-to-end check: small sweep, must produce positive
    /// bandwidth at every measured size and the curve must be
    /// (weakly) monotone non-decreasing in *bandwidth* across the
    /// upper half of the size range — small transfers are launch-cost
    /// bound and may be non-monotone, but the upper sizes asymptote to
    /// the PCIe bus.
    #[test]
    fn transfer_benchmark_smoke() -> anyhow::Result<()> {
        let device = cuda_device()?;
        let cfg = BenchmarkConfig {
            min_bytes: 64 * 1024,
            max_bytes: 16 * 1024 * 1024,
            samples: 5,
            trials: 3,
            warmup_trials: 1,
            measure_d2h: true,
        };
        let profile = run_benchmark(&device, &cfg)?;
        eprintln!("{}", profile.format_table());

        // Sanity: at least 2 H2D points, every point has positive
        // throughput and positive duration.
        assert!(
            profile.h2d().len() >= 2,
            "expected ≥2 H2D points, got {}",
            profile.h2d().len()
        );
        for m in profile.h2d() {
            assert!(m.bytes > 0);
            assert!(
                m.duration_ns > 0,
                "h2d duration must be > 0 for {} bytes",
                m.bytes
            );
            assert!(
                m.bandwidth_bps > 0.0,
                "h2d bandwidth must be > 0 for {} bytes",
                m.bytes
            );
        }
        for m in profile.d2h() {
            assert!(m.duration_ns > 0);
            assert!(m.bandwidth_bps > 0.0);
        }

        // Upper-half monotonicity: the *largest* H2D measurement must
        // hit at least 60% of the median bandwidth across the upper
        // half of the sweep. This catches gross regressions (e.g. an
        // accidental cudaMemcpy instead of cudaMemcpyAsync) without
        // being fragile to host noise.
        let n = profile.h2d().len();
        let upper: Vec<f64> = profile.h2d()[n / 2..]
            .iter()
            .map(|m| m.bandwidth_bps)
            .collect();
        let mut sorted = upper.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_bps = sorted[sorted.len() / 2];
        let peak_bps = profile.peak_h2d_bps;
        assert!(
            peak_bps >= 0.6 * median_bps,
            "peak H2D bandwidth ({:.3} GB/s) is < 0.6× median of upper half ({:.3} GB/s) — likely launch-cost \
             dominated and not a real bandwidth measurement",
            peak_bps / 1e9,
            median_bps / 1e9,
        );

        Ok(())
    }

    /// Interpolator returns finite, positive durations for in-range,
    /// below-range and above-range queries.
    #[test]
    fn transfer_benchmark_interpolation() -> anyhow::Result<()> {
        let device = cuda_device()?;
        let cfg = BenchmarkConfig {
            min_bytes: 64 * 1024,
            max_bytes: 4 * 1024 * 1024,
            samples: 4,
            trials: 2,
            warmup_trials: 1,
            measure_d2h: false,
        };
        let profile = run_benchmark(&device, &cfg)?;

        let inside = profile.predict_h2d(512 * 1024);
        assert!(inside.as_nanos() > 0);

        let below = profile.predict_h2d(8 * 1024);
        assert!(below.as_nanos() > 0);

        let above = profile.predict_h2d(64 * 1024 * 1024);
        assert!(above.as_nanos() > 0);

        // Above-range prediction must be ≥ measured at max_bytes — log-log
        // extrapolation with positive slope (which PCIe is) cannot drop.
        let measured_at_max =
            Duration::from_nanos(profile.h2d().last().expect("h2d empty").duration_ns);
        assert!(
            above >= measured_at_max,
            "extrapolation above max_bytes ({:?}) must be >= measured at max_bytes ({:?})",
            above,
            measured_at_max,
        );

        Ok(())
    }
}
