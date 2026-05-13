// Telemetry export smoke (Phase 4, 2026-05-12).
//
// Exercises `snapshot_to_file`, `ring_buffer_to_file`, `dump_all`, and
// the periodic-dump hook. All four are host-only paths (no CUDA), but
// the test crate is feature-gated `cuda + bf16_u16` to match the
// existing offload smoke tests' shape — that's how the rest of the
// flame-core test harness is invoked.
//
// Tests are wrapped in a single function so they run serially against
// the process-global Telemetry singleton (a `static OnceCell<Telemetry>`).
// Parallel test threads would race on snapshot/ringbuffer state.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use std::path::PathBuf;

use flame_core::offload::telemetry::{
    self, TelemetryCounters, TelemetryEvent, TelemetryEventKind, DUMP_EVENTS_FILENAME,
    DUMP_SNAPSHOT_FILENAME,
};

#[test]
fn telemetry_export_smoke() {
    let t = telemetry::global();
    t.reset();
    t.set_enabled(true);
    t.set_event_log_capacity(64);
    t.set_periodic_dump_interval(0);

    // Tempdir under target/ to avoid polluting /tmp on CI shares.
    let base: PathBuf = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!(
        "flame_telemetry_export_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Drive synthetic events: 3 prefetch + 2 await hit + 1 miss.
    for i in 0..3 {
        let timer = t.record_prefetch_begin();
        t.record_prefetch_end(timer, i, 1024 * 1024);
    }
    for i in 0..2 {
        let timer = t.record_await_begin();
        t.record_await_end_hit(timer, i);
    }
    {
        let timer = t.record_await_begin();
        t.record_await_end_miss(timer, 99);
    }

    // ── Part 1: snapshot_to_file ──────────────────────────────────
    let snap_path = dir.join("snap.json");
    telemetry::snapshot_to_file(&snap_path).expect("snapshot_to_file write");
    let bytes = std::fs::read(&snap_path).unwrap();
    let parsed: TelemetryCounters = serde_json::from_slice(&bytes).unwrap();
    let live = t.snapshot();
    assert_eq!(parsed.prefetch_issued, live.prefetch_issued);
    assert_eq!(parsed.h2d_bytes_total, live.h2d_bytes_total);
    assert_eq!(parsed.await_hits, live.await_hits);
    assert_eq!(parsed.await_misses, live.await_misses);
    assert!(parsed.prefetch_issued >= 3);

    // ── Part 2: ring_buffer_to_file produces valid JSON-lines ────
    let events_path = dir.join("events.jsonl");
    let n = telemetry::ring_buffer_to_file(&events_path).expect("ring_buffer_to_file write");
    assert_eq!(n, t.event_log().len(), "returned count == buffer length");
    assert!(n >= 6, "should have at least 6 events recorded, got {n}");
    let text = std::fs::read_to_string(&events_path).unwrap();
    let line_count = text.lines().count();
    assert_eq!(line_count, n, "JSON-lines line count matches event count");
    for (idx, line) in text.lines().enumerate() {
        let ev: TelemetryEvent = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line {idx} not valid JSON: {e}: {line:?}"));
        // Sanity: kinds must be one of the four enum values.
        match ev.kind {
            TelemetryEventKind::PrefetchIssued
            | TelemetryEventKind::PrefetchAlreadyResident
            | TelemetryEventKind::AwaitHit
            | TelemetryEventKind::AwaitMiss => {}
        }
    }

    // ── Part 3: dump_all writes both files into the dir ──────────
    let dump_dir = dir.join("dump");
    let chosen = telemetry::dump_all(Some(&dump_dir)).expect("dump_all write");
    assert_eq!(chosen, dump_dir);
    let snap_in_dump = dump_dir.join(DUMP_SNAPSHOT_FILENAME);
    let events_in_dump = dump_dir.join(DUMP_EVENTS_FILENAME);
    assert!(
        snap_in_dump.exists(),
        "dump_all should create {DUMP_SNAPSHOT_FILENAME}"
    );
    assert!(
        events_in_dump.exists(),
        "dump_all should create {DUMP_EVENTS_FILENAME}"
    );
    // Re-parse to confirm both files are valid.
    let snap2: TelemetryCounters =
        serde_json::from_slice(&std::fs::read(&snap_in_dump).unwrap()).unwrap();
    assert_eq!(snap2.prefetch_issued, live.prefetch_issued);
    let txt2 = std::fs::read_to_string(&events_in_dump).unwrap();
    let parsed_events: Vec<TelemetryEvent> = txt2
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(parsed_events.len(), n);

    // ── Part 4: periodic dump fires at the configured interval ───
    //
    // Reset to a fresh state, configure interval=3, drive 7 events,
    // and verify the snapshot file in the env-configured dir gets
    // rewritten on the 3rd and 6th events. We use a fresh sub-dir
    // and FLAME_OFFLOAD_TELEMETRY_DUMP_DIR so the previous test's
    // files don't confuse the read.
    let periodic_dir = dir.join("periodic");
    std::fs::create_dir_all(&periodic_dir).unwrap();
    let snap_in_periodic = periodic_dir.join(DUMP_SNAPSHOT_FILENAME);

    // SAFETY: tests are run serially so setting env is local-scope safe.
    std::env::set_var(
        "FLAME_OFFLOAD_TELEMETRY_DUMP_DIR",
        periodic_dir.as_os_str(),
    );

    t.reset();
    t.set_periodic_dump_interval(3);
    assert_eq!(t.periodic_dump_interval(), 3);

    // Drive 7 events. Dump should fire after event #3 and #6.
    for i in 0..7 {
        let timer = t.record_prefetch_begin();
        t.record_prefetch_end(timer, i, 1024);
    }

    assert!(
        snap_in_periodic.exists(),
        "periodic dump should have created the snapshot file by event 3"
    );
    let pbytes = std::fs::read(&snap_in_periodic).unwrap();
    let parsed_periodic: TelemetryCounters = serde_json::from_slice(&pbytes).unwrap();
    // The last periodic dump fires on event #6 (writes the snapshot at
    // that point: 6 prefetches counted). Event #7 increments but does
    // not trigger another dump.
    assert!(
        parsed_periodic.prefetch_issued >= 6,
        "snapshot from last periodic dump should reflect >= 6 prefetches \
         (got {})",
        parsed_periodic.prefetch_issued
    );

    // ── Cleanup ──────────────────────────────────────────────────
    std::env::remove_var("FLAME_OFFLOAD_TELEMETRY_DUMP_DIR");
    t.set_periodic_dump_interval(0);
    t.reset();
    t.set_enabled(false);
    t.set_event_log_capacity(0);
    let _ = std::fs::remove_dir_all(&dir);
}
