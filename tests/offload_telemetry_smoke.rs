// Smoke test for flame_core::offload::telemetry counters and event ring.
//
// Per tenets §4 (measurement beats assertion) and the SPEED_CONTRACT
// requirement to leave a regression gate behind every primitive fix:
// the telemetry layer that Phase 1 of the FlexTensor port wired into
// BlockOffloader is observe-only, but it needs to be observably non-zero
// when exercised. This test exercises the public record_* API directly
// (no real model required) to prove counters increment correctly.
//
// The trainer-level smoke (Builder agent verdict on klein 9B) already
// confirms the hooks don't disrupt training. This test closes the loop
// on whether the counters actually count.
//
// Tests are combined into a single function so they run serially without
// racing the process-global Telemetry singleton.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::offload::telemetry;

#[test]
fn telemetry_counters_and_event_ring_smoke() {
    let t = telemetry::global();

    // ── Part 1: counter deltas ────────────────────────────────────────
    //
    // Snapshot baseline. We measure deltas across our recorded events
    // rather than absolute values so this is robust against state left
    // over from any prior test in the same binary.

    t.set_enabled(true);
    assert!(t.is_enabled(), "telemetry must be enabled to record");

    let before = t.snapshot();

    // Drive a few synthetic prefetch events.
    for i in 0..4 {
        let timer = t.record_prefetch_begin();
        // Sleep so wall_ns delta is non-zero. A no-op begin/end pair could
        // round to zero on some clocks.
        std::thread::sleep(std::time::Duration::from_micros(50));
        t.record_prefetch_end(timer, i, 1024 * 1024); // 1 MiB each
    }

    // Drive a couple of already-resident events — should bump
    // `prefetch_already_resident`, not `h2d_bytes_total`.
    for i in 0..2 {
        let timer = t.record_prefetch_begin();
        t.record_prefetch_already_resident(timer, 100 + i);
    }

    // Drive await hits and misses.
    for i in 0..3 {
        let timer = t.record_await_begin();
        t.record_await_end_hit(timer, i);
    }
    {
        let timer = t.record_await_begin();
        t.record_await_end_miss(timer, 99);
    }

    let after = t.snapshot();

    assert_eq!(
        after.prefetch_issued - before.prefetch_issued,
        4,
        "4 prefetch_end calls should increment prefetch_issued by 4"
    );
    assert_eq!(
        after.prefetch_already_resident - before.prefetch_already_resident,
        2,
        "2 already_resident calls should increment that counter by 2"
    );
    assert_eq!(
        after.h2d_bytes_total - before.h2d_bytes_total,
        4 * 1024 * 1024,
        "4 prefetch_end at 1 MiB each should add 4 MiB to h2d_bytes_total"
    );
    assert!(
        after.prefetch_wall_ns > before.prefetch_wall_ns,
        "prefetch wall ns should advance: before={} after={}",
        before.prefetch_wall_ns,
        after.prefetch_wall_ns
    );
    assert_eq!(
        after.await_hits - before.await_hits,
        3,
        "3 await hits expected"
    );
    assert_eq!(
        after.await_misses - before.await_misses,
        1,
        "1 await miss expected"
    );

    let bw = after.effective_h2d_bps();
    assert!(
        bw.is_finite() && bw > 0.0,
        "effective_h2d_bps should be a finite positive bandwidth: {bw}"
    );
    let hr = after.await_hit_ratio();
    assert!(
        (0.0..=1.0).contains(&hr),
        "await_hit_ratio must be in [0,1], got {hr}"
    );

    eprintln!(
        "[offload_telemetry_smoke] counters delta: h2d={} MiB, prefetch_wall_ns={}, await_hits={}, await_misses={}, eff_h2d={:.2} GB/s, hit_ratio={:.2}",
        (after.h2d_bytes_total - before.h2d_bytes_total) / (1024 * 1024),
        after.prefetch_wall_ns - before.prefetch_wall_ns,
        after.await_hits - before.await_hits,
        after.await_misses - before.await_misses,
        bw / 1e9,
        hr
    );

    // ── Part 2: event ring ────────────────────────────────────────────
    //
    // Switch on the per-event trace, drive a fresh batch, drain, and
    // confirm we recorded exactly what we asked for. (Counter mode stays
    // on; capacity=0 in Part 1 means the prior events did not enter the
    // ring buffer, so we are starting clean here.)

    t.set_event_log_capacity(64);
    assert!(
        t.is_trace_enabled(),
        "trace mode should be on after set_event_log_capacity > 0"
    );

    let before_events = t.event_log().len();

    {
        let timer = t.record_prefetch_begin();
        std::thread::sleep(std::time::Duration::from_micros(20));
        t.record_prefetch_end(timer, 0, 2048);
    }
    {
        let timer = t.record_await_begin();
        t.record_await_end_hit(timer, 0);
    }
    {
        let timer = t.record_await_begin();
        t.record_await_end_miss(timer, 1);
    }

    let events = t.event_log();
    let after_events = events.len();
    assert_eq!(
        after_events - before_events,
        3,
        "event ring should grow by exactly the 3 events we recorded; before={}, after={}",
        before_events,
        after_events
    );

    // Tear down so we don't leak ring buffer state to a hypothetical
    // sibling test binary that may share this Telemetry singleton.
    t.set_event_log_capacity(0);

    eprintln!(
        "[offload_telemetry_smoke] event ring delta: {} events",
        after_events - before_events
    );
}
