# FlameCore Runtime – Agent Guardrails

## Non-Negotiables
- **No FP32 staging on device.** Banned APIs: `Tensor::to_vec_f32()`, `tensor.to_host()`, `copy_to_f32_device()`, or any helper that deep-clones BF16 tensors to FP32. If a test needs host data, cast in chunks inside the test harness.
- **STRICT_BF16 is non-optional.** Guards must panic loudly whenever a caller would fall back to FP32 or CPU. Debug bypasses and silent conversions are forbidden.
- **Device→host copies are chunked.** Parity/logging must use the shared dumper: lease ≤8 MB BF16 slices, launch a conversion kernel, copy to pinned host, write, release, repeat. Never grab the entire tensor in one shot.
- **Mirror every change.** After a FlameCore edit lands, port it to `FlameCore_clean_repo2` with a `[mirror]` commit before merging. The repos must stay byte-identical.
- **Keep pressure valves wired.** Honor `ARENA_TEMP_SOFTCAP_MB` and `TILE_VRAM_FRACTION`; any path that ignores them is a blocker.

## Safe Practices
- Allocate with `staging::lease` / `arena_reset`; raw `cudaMalloc` is for last-resort prototypes only.
- Extend the existing kernel suite (`repeat_bf16`, `broadcast_bf16`, `narrow_strided`, etc.) instead of cloning CPU logic. Keep accumulators FP32, storage BF16, and document any epsilon tweaks.
- Tests must run under `STRICT_BF16=1` and assert `storage_dtype().is_bf16()` for outputs.

## Required Checks
- `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, and `cargo test --features cuda,bf16_u16,heavy_kernels -- --nocapture`.
- BF16 parity suites: `cargo test --test cuda_bf16_parity --features cuda,bf16_u16,heavy_kernels`.
- STRICT harness: `cargo test --test strict_bf16_harness --features cuda,bf16_u16,strict_bf16 -- --exact`.
- Capture allocator counters (`device_fp32_alloc_bytes_during_infer`, arena high watermark) in test logs so the mirror repo can validate identical behavior.

## Escalation Path
If you discover legacy code that still clones to FP32 or hits OOM, document it in `docs/FLAME_SD35_SDPA_NOTES.md` and tag it as a red blocker. Stop all work that would reintroduce the issue and coordinate with the inference repo before landing any fixes.
