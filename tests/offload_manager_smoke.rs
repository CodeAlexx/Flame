//! Smoke tests for `flame_core::offload::manager` and
//! `flame_core::offload::state` (Phase 3 FlexTensor port).
//!
//! Exercises:
//!   * Construction in `NotInitialized` state.
//!   * `discover` records block sizes and emits a telemetry event.
//!   * `profile` populates a `TransferBandwidthProfile` (here using the
//!     state cache to avoid the multi-second PCIe sweep on every run).
//!   * `activate` picks the right strategy under two synthetic scenarios:
//!       - `force_strategy = TwoSlot`  → strategy_name() == "TwoSlot"
//!       - `force_strategy = Adaptive` → strategy_name() == "Adaptive"
//!   * `state::save_profile` / `state::load_profile` round-trip.
//!   * `load_profile`'s `predict_h2d` matches the live profile within
//!     1% on a sweep of byte sizes.
//!
//! These tests need a CUDA device because `BlockOffloader` requires one
//! for its transfer-stream. They construct a tiny synthetic 2-block
//! "model" via an inline `BlockFacilitator` so no real safetensors file
//! is needed — the offloader is happy with an empty residency, and that
//! is enough to drive the manager through every phase.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::offload::manager::{ForcedStrategy, ManagerConfig, OffloadManager, OffloadPhase};
use flame_core::offload::state;
use flame_core::offload::strategy::TwoSlot;
use flame_core::offload::telemetry;
use flame_core::offload::transfer_benchmark::{
    BenchmarkConfig, TransferBandwidthProfile, TransferDirection, TransferMeasurement,
};
use flame_core::offload::{BlockFacilitator, BlockOffloader};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cuda_device() -> anyhow::Result<Arc<CudaDevice>> {
    CudaDevice::new(0).map_err(|e| anyhow::anyhow!("CUDA device 0 unavailable: {e:?}"))
}

/// Trivial 2-block facilitator with no real safetensors keys. Used to
/// construct an empty `BlockOffloader` for state-machine tests — the
/// manager only cares about geometry (block_sizes()), not actual
/// weights.
struct EmptyFacilitator;
impl BlockFacilitator for EmptyFacilitator {
    fn block_count(&self) -> usize {
        2
    }
    fn classify_key(&self, _key: &str) -> Option<usize> {
        None
    }
}

/// Build a tiny safetensors file with zero block keys (so the
/// offloader's load loop walks one file, finds nothing it cares about,
/// and constructs the offloader with `block_count=2` empty blocks).
fn write_empty_safetensors(path: &std::path::Path) -> anyhow::Result<()> {
    use std::io::Write;
    // Empty header: zero data offset, no tensors. Safetensors layout =
    // [u64 header_len][header_json_bytes][data]. Empty JSON object
    // satisfies parser.
    let header = b"{}";
    let header_len = (header.len() as u64).to_le_bytes();
    let mut f = std::fs::File::create(path)?;
    f.write_all(&header_len)?;
    f.write_all(header)?;
    Ok(())
}

/// Construct a `BlockOffloader` over an empty 2-block synthetic model.
/// Each call writes a uniquely-named temp file so parallel tests can't
/// race on the same path.
fn make_empty_offloader(device: Arc<CudaDevice>) -> anyhow::Result<BlockOffloader> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!(
        "flame_core_offload_manager_empty_{}_{}.safetensors",
        std::process::id(),
        n,
    ));
    write_empty_safetensors(&tmp)?;
    let path_str = tmp.to_string_lossy().to_string();
    let paths = [path_str.as_str()];
    let off = BlockOffloader::load(&paths, &EmptyFacilitator, device)?;
    let _ = std::fs::remove_file(&tmp);
    Ok(off)
}

/// Build a hand-crafted bandwidth profile so tests don't need to run
/// the real PCIe sweep (which adds 100s of ms per test).
fn fake_profile() -> TransferBandwidthProfile {
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

// ---------------------------------------------------------------------------
// Manager state machine
// ---------------------------------------------------------------------------

/// Construction places the manager in `NotInitialized`. Discovery moves
/// to `Discovery` and records block sizes; profiling (with a primed
/// cache) moves to `Profiling`; activate picks the forced strategy and
/// wires it through to the underlying offloader.
#[test]
fn manager_state_machine_walks_all_phases() -> anyhow::Result<()> {
    let device = cuda_device()?;
    let offloader = make_empty_offloader(device.clone())?;

    // Seed the profile cache so `profile()` doesn't have to run the
    // full PCIe sweep. Use a per-test path so we don't fight any other
    // test (or a real cache) on disk.
    let cache_path =
        std::env::temp_dir().join("flame_core_offload_manager_smoke_profile.json");
    let _ = std::fs::remove_file(&cache_path);
    state::save_profile(&cache_path, &fake_profile(), "manager_state_machine_smoke")?;

    let cfg = ManagerConfig {
        profile_cache_path: Some(cache_path.clone()),
        force_strategy: Some(ForcedStrategy::TwoSlot),
        ..ManagerConfig::default()
    };
    let mut mgr = OffloadManager::with_config(device.clone(), offloader, cfg);

    // Construction → NotInitialized.
    assert_eq!(
        mgr.phase(),
        OffloadPhase::NotInitialized,
        "manager must start in NotInitialized phase"
    );
    assert_eq!(mgr.active_strategy_name(), "none");
    assert!(mgr.block_sizes().is_empty());

    // Enable telemetry so we can observe the transition events.
    let t = telemetry::global();
    t.set_enabled(true);
    let snap_before = t.snapshot();

    // Discovery.
    mgr.discover()?;
    assert_eq!(mgr.phase(), OffloadPhase::Discovery);
    assert_eq!(
        mgr.block_sizes().len(),
        2,
        "expected 2 blocks from EmptyFacilitator"
    );

    // Profiling — uses cached profile.
    mgr.run_profile()?;
    assert_eq!(mgr.phase(), OffloadPhase::Profiling);
    let p = mgr
        .bandwidth_profile()
        .expect("profile populated after run_profile()");
    assert!(p.peak_h2d_bps > 0.0, "loaded profile must have peak h2d");
    assert!(!p.h2d().is_empty(), "loaded profile must have h2d samples");

    // Activate — forced to TwoSlot.
    mgr.activate()?;
    assert_eq!(mgr.phase(), OffloadPhase::Active);
    assert_eq!(mgr.active_strategy_name(), "TwoSlot");
    assert_eq!(
        mgr.offloader().strategy_name(),
        "TwoSlot",
        "underlying BlockOffloader should hold the same strategy"
    );

    let snap_after = t.snapshot();
    // Three transitions (discover, profile_cached, activate) → +3
    // strategy_plans on the telemetry sink.
    assert!(
        snap_after.strategy_plans - snap_before.strategy_plans >= 3,
        "expected >=3 transition events recorded in telemetry; \
         before={} after={}",
        snap_before.strategy_plans,
        snap_after.strategy_plans,
    );

    // Cleanup.
    let _ = std::fs::remove_file(&cache_path);
    Ok(())
}

/// Forced Adaptive activates Adaptive; forced TwoSlot activates TwoSlot.
/// Verifies the strategy plumb-through is symmetric for both branches.
#[test]
fn manager_forced_adaptive_installs_adaptive() -> anyhow::Result<()> {
    let device = cuda_device()?;
    let offloader = make_empty_offloader(device.clone())?;

    let cache_path = std::env::temp_dir()
        .join("flame_core_offload_manager_smoke_adaptive_profile.json");
    let _ = std::fs::remove_file(&cache_path);
    state::save_profile(&cache_path, &fake_profile(), "manager_forced_adaptive_smoke")?;

    let cfg = ManagerConfig {
        profile_cache_path: Some(cache_path.clone()),
        force_strategy: Some(ForcedStrategy::Adaptive),
        ..ManagerConfig::default()
    };
    let mut mgr = OffloadManager::with_config(device.clone(), offloader, cfg);
    mgr.discover_profile_activate()?;
    assert_eq!(mgr.active_strategy_name(), "Adaptive");
    assert_eq!(mgr.offloader().strategy_name(), "Adaptive");

    let _ = std::fs::remove_file(&cache_path);
    Ok(())
}

/// `auto_select_strategy` plumbing — when neither forced choice is
/// given the manager queries cuda_mem_get_info and picks based on the
/// `2*max < 0.3*free` rule. On this host (3090 Ti with ~24 GB), an
/// empty model (max_block_bytes=0) hits the short-circuit and selects
/// `TwoSlot`. This test asserts that property to ensure the
/// auto-select code path runs without panic / strategy-name regression.
#[test]
fn manager_auto_selects_two_slot_when_empty_model() -> anyhow::Result<()> {
    let device = cuda_device()?;
    let offloader = make_empty_offloader(device.clone())?;

    let cache_path = std::env::temp_dir()
        .join("flame_core_offload_manager_smoke_auto_profile.json");
    let _ = std::fs::remove_file(&cache_path);
    state::save_profile(&cache_path, &fake_profile(), "manager_auto_select_smoke")?;

    let cfg = ManagerConfig {
        profile_cache_path: Some(cache_path.clone()),
        force_strategy: None,
        ..ManagerConfig::default()
    };
    let mut mgr = OffloadManager::with_config(device.clone(), offloader, cfg);
    mgr.discover_profile_activate()?;
    // Empty model → max_block_bytes == 0 → TwoSlot short-circuit.
    assert_eq!(
        mgr.active_strategy_name(),
        "TwoSlot",
        "empty-model auto-select should land on TwoSlot"
    );

    let _ = std::fs::remove_file(&cache_path);
    Ok(())
}

/// Calling `profile()` before `discover()` is a hard error.
#[test]
fn manager_profile_before_discover_errors() -> anyhow::Result<()> {
    let device = cuda_device()?;
    let offloader = make_empty_offloader(device.clone())?;
    let mut mgr = OffloadManager::new(device, offloader);

    let err = mgr.run_profile().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("requires discover"),
        "expected discover-required error, got: {msg}"
    );
    Ok(())
}

/// Calling `activate()` before `run_profile()` is a hard error.
#[test]
fn manager_activate_before_profile_errors() -> anyhow::Result<()> {
    let device = cuda_device()?;
    let offloader = make_empty_offloader(device.clone())?;
    let mut mgr = OffloadManager::new(device, offloader);

    mgr.discover()?;
    let err = mgr.activate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("requires run_profile"),
        "expected run_profile-required error, got: {msg}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// state.rs round-trip
// ---------------------------------------------------------------------------

/// Save + load preserves the profile's prediction surface (≤ 1% error
/// across a sweep of byte sizes).
#[test]
fn state_save_load_round_trip_predictions_match() -> anyhow::Result<()> {
    let path = std::env::temp_dir().join("flame_core_offload_state_round_trip.json");
    let _ = std::fs::remove_file(&path);

    let p = fake_profile();
    state::save_profile(&path, &p, "round_trip_smoke")?;
    let loaded = state::load_profile(&path)?;

    for &bytes in &[1024usize, 64 * 1024, 1 << 20, 4 * 1024 * 1024, 16 * 1024 * 1024] {
        let a = p.predict_h2d(bytes);
        let b = loaded.predict_h2d(bytes);
        let rel = state::relative_error(a, b);
        assert!(
            rel < 0.01,
            "round-trip diverged at {bytes} bytes: orig={:?} loaded={:?} rel={rel}",
            a,
            b
        );
    }

    // Peak bandwidths survive.
    assert!((loaded.peak_h2d_bps - p.peak_h2d_bps).abs() < 1.0);
    assert!((loaded.peak_d2h_bps - p.peak_d2h_bps).abs() < 1.0);

    let _ = std::fs::remove_file(&path);
    Ok(())
}

/// Loading a corrupt file is a hard error (does not silently produce a
/// default profile).
#[test]
fn state_load_corrupt_file_errors() {
    let path = std::env::temp_dir().join("flame_core_offload_state_corrupt.json");
    std::fs::write(&path, b"this is not json").unwrap();
    let err = state::load_profile(&path).unwrap_err();
    assert!(
        format!("{err}").contains("parse"),
        "expected parse error, got: {err}"
    );
    let _ = std::fs::remove_file(&path);
}

/// `default_profile_path` honors the `FLAME_OFFLOAD_PROFILE_PATH` env
/// override.
#[test]
fn state_default_profile_path_env_override() {
    std::env::set_var(state::PROFILE_PATH_ENV, "/tmp/flame_test_explicit_smoke.json");
    let p = state::default_profile_path();
    assert_eq!(
        p,
        std::path::PathBuf::from("/tmp/flame_test_explicit_smoke.json")
    );
    std::env::remove_var(state::PROFILE_PATH_ENV);
}

// ---------------------------------------------------------------------------
// Combined: manager + state interplay
// ---------------------------------------------------------------------------

/// If the cache file is absent on `profile()`, the manager falls back to
/// running `transfer_benchmark::run_benchmark` and writes the result to
/// the cache. The next `OffloadManager::profile()` call should pick the
/// fresh cache up.
///
/// We use a very small benchmark config so this test stays under 1 s.
#[test]
fn manager_falls_back_to_bench_and_writes_cache() -> anyhow::Result<()> {
    let device = cuda_device()?;
    let offloader = make_empty_offloader(device.clone())?;

    let cache_path = std::env::temp_dir()
        .join("flame_core_offload_manager_smoke_writeback.json");
    let _ = std::fs::remove_file(&cache_path);

    let cfg = ManagerConfig {
        profile_cache_path: Some(cache_path.clone()),
        benchmark: BenchmarkConfig {
            min_bytes: 64 * 1024,
            max_bytes: 1024 * 1024,
            samples: 3,
            trials: 2,
            warmup_trials: 1,
            measure_d2h: false,
        },
        force_strategy: Some(ForcedStrategy::TwoSlot),
        ..ManagerConfig::default()
    };
    let mut mgr = OffloadManager::with_config(device.clone(), offloader, cfg.clone());
    mgr.discover()?;
    mgr.run_profile()?;
    assert!(
        cache_path.exists(),
        "run_profile() must persist the bench result back to the cache"
    );
    let bench_p = mgr
        .bandwidth_profile()
        .expect("profile present")
        .clone();

    // Drop the manager so we have a clean slate, then construct a new
    // one against the same cache. The new manager should pick up the
    // cache and skip the bench.
    drop(mgr);

    let offloader2 = make_empty_offloader(device.clone())?;
    let mut mgr2 = OffloadManager::with_config(device, offloader2, cfg);
    mgr2.discover()?;
    mgr2.run_profile()?;
    let cached_p = mgr2.bandwidth_profile().expect("profile present");
    // Match within 1% on a representative sweep.
    for &bytes in &[64 * 1024usize, 256 * 1024, 1024 * 1024] {
        let a = bench_p.predict_h2d(bytes);
        let b = cached_p.predict_h2d(bytes);
        let rel = state::relative_error(a, b);
        assert!(
            rel < 0.01,
            "cached profile prediction diverged at {bytes} bytes: \
             bench={:?} cached={:?} rel={rel}",
            a,
            b
        );
    }
    let _ = std::fs::remove_file(&cache_path);
    Ok(())
}

/// TwoSlot strategy survives being attached via the manager.
/// Regression gate: pre-existing `set_strategy` semantics must keep
/// working (we don't subtly break the offloader's strategy plumbing).
#[test]
fn manager_does_not_break_offloader_set_strategy() -> anyhow::Result<()> {
    let device = cuda_device()?;
    let mut offloader = make_empty_offloader(device.clone())?;
    offloader.set_strategy(Box::new(TwoSlot::new()));
    assert_eq!(offloader.strategy_name(), "TwoSlot");
    Ok(())
}
