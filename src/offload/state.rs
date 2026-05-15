//! On-disk persistence for the offload subsystem.
//!
//! Phase 3 of the FlexTensor port. Persists what survives a process restart:
//! the [`transfer_benchmark::TransferBandwidthProfile`](super::transfer_benchmark::TransferBandwidthProfile)
//! measured during the [`OffloadManager`](super::manager::OffloadManager)
//! [`Profiling`](super::manager::OffloadPhase::Profiling) phase.
//!
//! ## What this is (and is NOT)
//!
//! This is the **profile cache** half of FlexTensor's `state_handler.py`:
//! save / load the bandwidth profile so subsequent processes can skip the
//! ~1 s PCIe sweep. It does **not** persist:
//!
//! * `__torch_function__` tensor-mode state — flame-core has no equivalent
//!   surface; block geometry is declared via
//!   [`BlockFacilitator`](super::BlockFacilitator).
//! * Strategy state — strategies are stateless (`TwoSlot`) or carry only a
//!   couple of `u64`s (`Adaptive::last_target_bytes`). Persisting them
//!   buys nothing.
//! * Per-step telemetry counters — those reset every process, by design.
//!
//! ## Default path
//!
//! `${XDG_CACHE_HOME:-$HOME/.cache}/flame-core/offload_profile.json` unless
//! `FLAME_OFFLOAD_PROFILE_PATH` is set. The directory is created on save if
//! it does not exist.
//!
//! ## Sync contract
//!
//! Save / load are init-time only — they touch the disk, not the GPU, and
//! never run on the per-step path. No `cudaStreamSynchronize`. Clauses 1
//! and 5 of `SPEED_CONTRACT.md` are satisfied trivially.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::transfer_benchmark::{TransferBandwidthProfile, TransferDirection, TransferMeasurement};

/// Filename the manager writes the profile cache under by default.
pub const DEFAULT_PROFILE_FILENAME: &str = "offload_profile.json";

/// Environment variable override for the profile cache path.
pub const PROFILE_PATH_ENV: &str = "FLAME_OFFLOAD_PROFILE_PATH";

/// Serializable mirror of [`TransferDirection`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum SerdeDirection {
    H2d,
    D2h,
}

impl From<TransferDirection> for SerdeDirection {
    fn from(d: TransferDirection) -> Self {
        match d {
            TransferDirection::HostToDevice => SerdeDirection::H2d,
            TransferDirection::DeviceToHost => SerdeDirection::D2h,
        }
    }
}

impl From<SerdeDirection> for TransferDirection {
    fn from(d: SerdeDirection) -> Self {
        match d {
            SerdeDirection::H2d => TransferDirection::HostToDevice,
            SerdeDirection::D2h => TransferDirection::DeviceToHost,
        }
    }
}

/// Serializable mirror of [`TransferMeasurement`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct SerdeMeasurement {
    bytes: usize,
    direction: SerdeDirection,
    duration_ns: u64,
    bandwidth_bps: f64,
}

impl From<&TransferMeasurement> for SerdeMeasurement {
    fn from(m: &TransferMeasurement) -> Self {
        Self {
            bytes: m.bytes,
            direction: m.direction.into(),
            duration_ns: m.duration_ns,
            bandwidth_bps: m.bandwidth_bps,
        }
    }
}

impl From<SerdeMeasurement> for TransferMeasurement {
    fn from(m: SerdeMeasurement) -> Self {
        TransferMeasurement {
            bytes: m.bytes,
            direction: m.direction.into(),
            duration_ns: m.duration_ns,
            bandwidth_bps: m.bandwidth_bps,
        }
    }
}

/// Serialized profile envelope. A small schema version is included so a
/// future schema bump can reject silently-wrong cached profiles instead of
/// loading garbage. The version is checked on load and a mismatch is a
/// hard error (forces the manager to re-run the bench).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileEnvelope {
    /// Schema version. Bump when the on-disk format changes incompatibly.
    version: u32,
    /// Optional free-form label (e.g. device name, driver version). Currently
    /// unused by the loader; reserved for future cache invalidation.
    #[serde(default)]
    label: String,
    h2d: Vec<SerdeMeasurement>,
    d2h: Vec<SerdeMeasurement>,
    peak_h2d_bps: f64,
    peak_d2h_bps: f64,
}

const SCHEMA_VERSION: u32 = 1;

/// Resolve the path the manager should read/write the profile under.
///
/// Precedence:
/// 1. `FLAME_OFFLOAD_PROFILE_PATH` env var (full file path).
/// 2. `$XDG_CACHE_HOME/flame-core/offload_profile.json` if `XDG_CACHE_HOME` is set.
/// 3. `$HOME/.cache/flame-core/offload_profile.json`.
/// 4. Falls back to `./.flame-core-offload-profile.json` if neither env
///    variable is available (e.g. CI without `$HOME`).
pub fn default_profile_path() -> PathBuf {
    if let Ok(explicit) = std::env::var(PROFILE_PATH_ENV) {
        return PathBuf::from(explicit);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| {
                let mut p = PathBuf::from(h);
                p.push(".cache");
                p
            })
        });
    match base {
        Some(mut p) => {
            p.push("flame-core");
            p.push(DEFAULT_PROFILE_FILENAME);
            p
        }
        None => PathBuf::from("./.flame-core-offload-profile.json"),
    }
}

/// Persist a bandwidth profile to disk as JSON.
///
/// Creates parent directories if missing. Overwrites an existing file.
/// Returns the path actually written for caller logging.
pub fn save_profile(
    path: &std::path::Path,
    profile: &TransferBandwidthProfile,
    label: impl Into<String>,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                anyhow::anyhow!("offload state: create_dir_all({:?}) failed: {e}", parent)
            })?;
        }
    }

    let envelope = ProfileEnvelope {
        version: SCHEMA_VERSION,
        label: label.into(),
        h2d: profile.h2d().iter().map(SerdeMeasurement::from).collect(),
        d2h: profile.d2h().iter().map(SerdeMeasurement::from).collect(),
        peak_h2d_bps: profile.peak_h2d_bps,
        peak_d2h_bps: profile.peak_d2h_bps,
    };
    let json = serde_json::to_vec_pretty(&envelope)
        .map_err(|e| anyhow::anyhow!("offload state: serialize: {e}"))?;
    std::fs::write(path, json)
        .map_err(|e| anyhow::anyhow!("offload state: write({:?}) failed: {e}", path))?;
    Ok(())
}

/// Reconstruct a bandwidth profile from disk.
///
/// Hard-errors on missing file, IO error, JSON parse error, or schema
/// version mismatch. Callers (typically [`OffloadManager`](super::manager::OffloadManager))
/// fall back to running `transfer_benchmark::run_benchmark` when this fails.
pub fn load_profile(path: &std::path::Path) -> anyhow::Result<TransferBandwidthProfile> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("offload state: read({:?}) failed: {e}", path))?;
    let env: ProfileEnvelope = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("offload state: parse({:?}) failed: {e}", path))?;
    if env.version != SCHEMA_VERSION {
        anyhow::bail!(
            "offload state: schema version mismatch (file={}, expected={SCHEMA_VERSION})",
            env.version
        );
    }
    let h2d: Vec<TransferMeasurement> = env.h2d.into_iter().map(Into::into).collect();
    let d2h: Vec<TransferMeasurement> = env.d2h.into_iter().map(Into::into).collect();

    // Reconstruct using the same constructor the bench produces. The profile
    // has no public new(), so we expose a builder here that mirrors its
    // field layout. Keep this in sync with the struct.
    Ok(TransferBandwidthProfile::__from_persisted_parts(
        h2d,
        d2h,
        env.peak_h2d_bps,
        env.peak_d2h_bps,
    ))
}

/// Convenience: compare two predicted H2D durations and return their
/// relative error in `[0, +inf)`. Returns `0.0` when both are zero;
/// returns `f64::INFINITY` if `a` is zero but `b` is not.
///
/// Used by tests / diagnostics to verify a loaded profile predicts within
/// tolerance of a freshly-measured one.
pub fn relative_error(a: Duration, b: Duration) -> f64 {
    let an = a.as_nanos() as f64;
    let bn = b.as_nanos() as f64;
    if an == 0.0 && bn == 0.0 {
        return 0.0;
    }
    if an == 0.0 {
        return f64::INFINITY;
    }
    (an - bn).abs() / an
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_profile() -> TransferBandwidthProfile {
        // Use the same persisted-parts constructor as load_profile so we
        // exercise that path. Mirror the measurement vector shape a real
        // bench would produce.
        let h2d = vec![
            TransferMeasurement {
                bytes: 64 * 1024,
                direction: TransferDirection::HostToDevice,
                duration_ns: 4_000,
                bandwidth_bps: 64.0 * 1024.0 / 4_000e-9,
            },
            TransferMeasurement {
                bytes: 4 * 1024 * 1024,
                direction: TransferDirection::HostToDevice,
                duration_ns: 200_000,
                bandwidth_bps: 4.0 * 1024.0 * 1024.0 / 200_000e-9,
            },
        ];
        let d2h = vec![TransferMeasurement {
            bytes: 4 * 1024 * 1024,
            direction: TransferDirection::DeviceToHost,
            duration_ns: 220_000,
            bandwidth_bps: 4.0 * 1024.0 * 1024.0 / 220_000e-9,
        }];
        TransferBandwidthProfile::__from_persisted_parts(
            h2d,
            d2h,
            4.0 * 1024.0 * 1024.0 / 200_000e-9,
            4.0 * 1024.0 * 1024.0 / 220_000e-9,
        )
    }

    #[test]
    fn save_load_round_trip_matches_predictions() {
        let dir = std::env::temp_dir().join("flame_core_offload_state_tests");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("profile.json");
        let profile = fake_profile();

        save_profile(&path, &profile, "unit-test").expect("save");
        let loaded = load_profile(&path).expect("load");

        // Predictions across a sweep must match within 1% (we measure
        // them through the same log-log interpolator, so they should be
        // bit-identical, modulo float round-trip).
        for &bytes in &[
            1024usize,
            64 * 1024,
            1 << 20,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
        ] {
            let a = profile.predict_h2d(bytes);
            let b = loaded.predict_h2d(bytes);
            let rel = relative_error(a, b);
            assert!(
                rel < 0.01,
                "predict_h2d({bytes}) round-trip diverged: orig={:?} loaded={:?} rel={rel}",
                a,
                b
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn schema_version_mismatch_is_hard_error() {
        let dir = std::env::temp_dir().join("flame_core_offload_state_tests");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad_schema.json");
        std::fs::write(
            &path,
            br#"{"version":999,"label":"","h2d":[],"d2h":[],"peak_h2d_bps":0.0,"peak_d2h_bps":0.0}"#,
        )
        .unwrap();
        let err = load_profile(&path).unwrap_err();
        assert!(
            format!("{err}").contains("schema version mismatch"),
            "expected schema-mismatch error, got: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn default_profile_path_respects_env_override() {
        std::env::set_var(PROFILE_PATH_ENV, "/tmp/flame_test_explicit.json");
        let p = default_profile_path();
        assert_eq!(p, PathBuf::from("/tmp/flame_test_explicit.json"));
        std::env::remove_var(PROFILE_PATH_ENV);
    }
}
