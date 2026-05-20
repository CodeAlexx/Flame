# `flame_core::offload` — Getting Started

Consumer-facing tutorial for the block-offload module. Skim this first,
then drill into [`FLAME_MODULES.md`](./FLAME_MODULES.md) and
[`FLAME_INDEX.md`](./FLAME_INDEX.md) for the full reference.

## 1. What is the offload module

`flame_core::offload` is a framework primitive for **swapping model
blocks between pinned host RAM and GPU VRAM** during training and
inference. It exists because frontier diffusion DiTs (Wan2.2 14B+14B,
HiDream-O1 32B, sensenova_u1 8B-MoT, etc.) do not fit in 24 GB or 48 GB
of VRAM. The offloader holds all block weights in pinned host memory,
keeps **two GPU-side staging slots**, and uses a dedicated CUDA transfer
stream to overlap H2D copies with compute. Block N+1's weights arrive
while block N is still doing its forward.

The primitive replaces the older `FlameSwap` mechanism for training and
the older Python-style "stream from disk" pattern for inference. No file
I/O on the hot path.

## 2. Basic usage — `BlockOffloader::load`

The canonical pattern (used by `EriDiffusion-v2/klein-trainer`,
`wan22-trainer`, every block-shaped model in `inference-flame`):

```rust
use flame_core::offload::{BlockOffloader, BlockFacilitator};

let mut offloader = BlockOffloader::load(
    safetensors_paths,    // &[PathBuf]
    Arc::new(MyFacilitator { /* … */ }),
    Arc::clone(&device),
)?;

// Per-step block loop:
for blk in 0..offloader.block_count() {
    offloader.prefetch_block(blk + 1).ok();          // start next H2D
    let handle = offloader.await_block_handle(blk)?; // wait for current
    let weights = handle.weights();
    // … run block forward using weights["..."] …
    drop(handle); // records `compute_done` for slot re-use ordering
}
```

`BlockFacilitator` is a one-trait-per-model description of the block
geometry — see `EriDiffusion-v2/crates/eridiffusion-core/src/models/klein.rs`
around line 539 for a complete reference implementation.

## 3. Opt into the Adaptive strategy

By default the offloader runs a stateless 2-slot ping-pong (the
pre-Phase-2 behavior). For heavy memory-pressure workloads — anything
where VRAM headroom is tight and unexpected OOMs are a known risk —
attach a [`Strategy`](../src/offload/strategy.rs) that grows / shrinks
its resident set based on observed VRAM:

```rust
use flame_core::offload::strategy::Adaptive;

if std::env::var("FLAME_OFFLOAD_ADAPTIVE").as_deref() == Ok("1") {
    offloader.set_strategy(Box::new(Adaptive::new()));
}
```

This is the pattern in current trainer call sites
(`klein-trainer`, `chroma-trainer`, `wan22-trainer`,
`sensenova_u1-trainer`, `ernie-trainer`, `qwenimage-trainer`).
Default off; `FLAME_OFFLOAD_ADAPTIVE=1` flips it on. Measured ~3 %
step-time win on Klein 9B (handoff `HANDOFF_2026-05-12_LORA_BASELINE.md`).

Other strategies if you want manual control:

- `strategy::TwoSlot::new()` — the default, but explicit.
- `strategy::Knapsack::with_budget(byte_budget)` — value-per-byte selection
  inside a fixed byte budget. For when you've already done the VRAM math
  and just need the offloader to respect a cap.
- `strategy::Adaptive::new().with_watermarks(0.55, 0.80)` — same Adaptive
  but tuned for a different VRAM-pressure profile.

## 4. Training checkpoint rule

For block-checkpointed trainers that use `checkpoint_offload_boundary`,
the offloader policy and weight layout are part of the performance
contract:

- Use the flame-core `BlockOffloader` directly. The old two-slot-only
  offloader works functionally, but backward recompute will refetch the
  entire layer stack and can be several times slower.
- Set a resident-set policy before loading, usually
  `FLAME_LAYER_OFFLOAD_FRACTION=0.77` on 24 GB cards. This widens the GPU
  slot window so backward replay can reuse a layer window instead of doing
  a full second H2D pass.
- If the model's matmul kernels use native `[Cout, Cin]` weights
  (`fused_linear3d_native` / cuBLASLt transpose path), call
  `.with_native_layout(true)` on the offloader. Otherwise the default
  pre-transpose path may be immediately undone by model code, doubling
  transfer/recompute work for no numerical change.
- In the model loop, call `plan_layer_access(i, true, false)` before the
  forward block and `plan_layer_access(i, false, false)` inside checkpoint
  recompute before `await_block_handle(i)`.

Klein 9B and HiDream-O1 both rely on this shape. If a future trainer
uses boundary checkpointing but regresses to 2-slot ping-pong or
pre-transpose/untranspose, expect an avoidable step-time cliff.

Important distinction: `checkpoint_offload_boundary` is a boundary-input
checkpoint, not a no-recompute activation-offload path. The grow cache
stores the block input boundary tensors; the checkpoint op carries the
output ID, and backward replays the block closure to build a local
sub-tape. This is the right shape when the full model cannot keep every
block's tape resident, but it still pays recompute.

Measured HiDream-O1 note from 2026-05-20: after moving to flame-core
`BlockOffloader`, `.with_native_layout(true)`, and the
`FLAME_LAYER_OFFLOAD_FRACTION` resident policy, 512 LoRA training stayed
around 3.5 s/step. Widening the resident target from `0.77` to `0.50`
(about 3.0 GiB to 6.6 GiB of block weights) did not materially improve
speed. That points at decoder recompute/model kernels, not the old
offloader or the resident-window size. For that class of model, further
speed work should look at partial checkpoint coverage or a true
no-recompute sub-tape activation offload path rather than retuning the
block loader.

## 5. Auto-strategy via `OffloadManager` (Phase 3)

Even better: don't pick a strategy by hand. `OffloadManager` runs a
state machine (`NotInitialized → Discovery → Profiling → Active`) that
benchmarks PCIe bandwidth once, caches the result, and selects a
`Strategy` based on the actual VRAM headroom your model leaves:

```rust
use flame_core::offload::{OffloadManager, BlockOffloader};

let offloader = BlockOffloader::load(paths, facilitator, device.clone())?;
let mut manager = OffloadManager::new(device.clone(), offloader);
manager.discover_profile_activate()?;
let offloader = manager.offloader_mut();
// … per-step loop unchanged …
```

`discover_profile_activate()` is the one-shot. If you want fine control,
call the three transitions individually: `discover()`, `run_profile()`,
`activate()`. The decision rule in `activate()` is roughly:

```
if 2 * max_block_bytes < 0.3 * (free_VRAM - headroom)  → TwoSlot
else                                                    → Adaptive
```

Force a specific strategy via `ManagerConfig::force_strategy` (see
[`offload::ForcedStrategy`](../src/offload/manager.rs:157)).

## 6. Telemetry: turning it on

The offload module already emits counters at every `prefetch_block` /
`await_block` call. By default counters are inert (one relaxed atomic
load per call). Activation:

```
FLAME_OFFLOAD_TELEMETRY=on      # counters only
FLAME_OFFLOAD_TELEMETRY=trace   # counters + per-event ring buffer
FLAME_OFFLOAD_TELEMETRY_RING=8192  # override ring capacity (default 4096)
```

Or programmatically:

```rust
use flame_core::offload::telemetry;

telemetry::global().set_enabled(true);
telemetry::global().set_event_log_capacity(8192); // trace mode
```

What's counted: H2D bytes, prefetch wall ns, await wall ns, await hit
ratio, prefetch issued vs already-resident, plus Phase 2 strategy
decisions (plans / evictions / target resident bytes).

## 7. Telemetry: reading the data

**Two ways.**

**Programmatic** (cheap, in-process):

```rust
let snap = telemetry::global().snapshot();
eprintln!("{}", telemetry::format_counters(&snap));
```

**Exported to disk** (Phase 4, new — works from any monitoring script,
no source edits in the trainer):

```rust
// One-shot dump at end-of-run:
telemetry::dump_all(Some(Path::new("/tmp/offload_log")))?;

// Or env-driven:
//   FLAME_OFFLOAD_TELEMETRY_DUMP_DIR=/tmp/offload_log
telemetry::dump_all(None)?;
```

Files produced:

- `flame_offload_telemetry_snapshot.json` — single JSON document.
  Schema = `TelemetryCounters` (serde derive on the public type).
- `flame_offload_telemetry_events.jsonl` — JSON-lines, one
  `TelemetryEvent` per line.

**Continuous periodic dump** while training:

```
FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_EVENTS=1000
FLAME_OFFLOAD_TELEMETRY_DUMP_DIR=/tmp/offload_log
```

The interval counts **events**, not training steps. Each
`record_prefetch_end` / `record_await_end_hit` / `record_await_end_miss`
call ticks the counter. On klein 9B that's roughly ~64 events per
training step in `trace` mode, so `=1000` fires the dump every ~16
training steps. Set the value with that ratio in mind for your model.

Every N recorded events, the global telemetry sink writes both files
atomically (tmp file + rename). A monitor script can `cat
/tmp/offload_log/flame_offload_telemetry_snapshot.json` at any time;
either it gets the previous good snapshot or the freshly-written one,
never a partial write.

(The legacy env var name `FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_STEPS`
is still recognized for back-compat but is deprecated — its `STEPS`
suffix was a misnomer.)

## 8. Transfer benchmark cache

Phase 3's profile sweep takes ~1 second. Cache it across runs:

```
FLAME_OFFLOAD_PROFILE_PATH=/var/cache/flame-core/offload_profile.json
```

`OffloadManager::run_profile()` will load this file if it exists (schema-
version checked) and skip the sweep. If the file is missing or has the
wrong schema, the sweep runs and the result is written back. Default
location: `$XDG_CACHE_HOME/flame-core/offload_profile.json`.

## 9. When to use which strategy

| Strategy | When to use | Cost |
|---|---|---|
| `TwoSlot` (default) | Block size << free VRAM. The PCIe-bandwidth case where prefetch lands in time. | Zero. Pre-Phase-2 path; bit-identical. |
| `Adaptive` | Heavy VRAM pressure (24 GB box on a 14B+ model, sensenova_u1 @ 2048², HiDream-O1 32B). OOM-prone configs. | One `cudaMemGetInfo` per plan + the Knapsack scoring underneath. Both microsecond-scale. |
| `Knapsack` | You already have a measured byte budget and want a strict cap. Less useful for unattended runs than Adaptive. | Sort over block IDs once per plan; ~1 µs at 40 blocks. |

When in doubt: let `OffloadManager` decide. The decision is one
non-blocking driver query (`cudaMemGetInfo`) at activate time.

## 10. References

- [`SPEED_CONTRACT.md`](./SPEED_CONTRACT.md) clause 1 — no implicit
  `cudaStreamSynchronize` on the hot path. The offload module satisfies
  this: per-slot CUDA events replace host stalls; telemetry exports are
  pure host I/O.
- [`FLAME_MODULES.md`](./FLAME_MODULES.md) — paragraph reference for each
  submodule (`offload::telemetry`, `offload::strategy`,
  `offload::manager`, `offload::state`, `offload::transfer_benchmark`).
- [`FLAME_INDEX.md`](./FLAME_INDEX.md) — flat symbol → `file:line` lookup
  for every public name.

## 10. Quick smoke

After turning telemetry on, the simplest sanity check is:

```
FLAME_OFFLOAD_TELEMETRY=on cargo test --release \
  --features "cuda,heavy_kernels,bf16_u16" \
  --test offload_telemetry_smoke -- --nocapture
```

This drives synthetic prefetch / await events and asserts the counters
update correctly. For the export path:

```
cargo test --release --features "cuda,heavy_kernels,bf16_u16" \
  --test offload_telemetry_export -- --nocapture
```

Both must pass before you call your trainer wiring done.
