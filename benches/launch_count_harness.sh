#!/usr/bin/env bash
# launch_count_harness.sh — capture an nsys profile of a bounded trainer
# run and emit a per-step launch/driver-api JSON via parse_nsys_to_json.py.
#
# Usage:
#   launch_count_harness.sh <TRAINER_BIN> <STEPS> <WARMUP> <LABEL> -- [trainer args...]
#
# Example (zimage, 20 steps total, 5 warmup):
#   launch_count_harness.sh train_zimage 20 5 zimage_phase0_baseline -- \
#     --config /path/to/zimage.toml --steps 20 --steps-override
#
# Output:
#   flame-core/benches/baselines/<LABEL>.json
#   /tmp/<LABEL>.nsys-rep  (kept for later analysis)
#
# Skeptic notes:
#   - The trainer's `--steps N` should equal STEPS (the second positional).
#     If they disagree the per-step divisor is wrong and the JSON is bogus.
#   - WARMUP is subtracted from the divisor only — the raw `--steps N` MUST
#     be the full count, including warmup, or per-step numbers under-count.
#   - Allocator pool state can carry between runs of the same binary if the
#     pool is process-local; this harness exits per-run so that's not an
#     issue, but `FLAME_ALLOC_POOL=0` is recommended for reproducibility on
#     prepare_*.rs binaries (irrelevant for training binaries).
#   - The harness sets `CUDA_DEVICE_MAX_CONNECTIONS=1` to keep the launch
#     queue single-stream — multi-stream profiles fragment kernel counts.

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 <TRAINER_BIN> <STEPS> <WARMUP> <LABEL> -- [trainer args...]" >&2
    echo "" >&2
    echo "  TRAINER_BIN  cargo bin name, e.g. train_zimage" >&2
    echo "  STEPS        total trainer steps including warmup" >&2
    echo "  WARMUP       warmup steps to subtract from divisor" >&2
    echo "  LABEL        identifier for the output JSON" >&2
    exit 2
fi

TRAINER="$1"
STEPS="$2"
WARMUP="$3"
LABEL="$4"
shift 4

# Eat the literal `--` between harness args and trainer args, if present.
if [[ "${1:-}" == "--" ]]; then
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAME_CORE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EDV2_DIR="$(cd "$FLAME_CORE_DIR/../EriDiffusion-v2" && pwd)"
BASELINES_DIR="$SCRIPT_DIR/baselines"
mkdir -p "$BASELINES_DIR"

NSYS="${NSYS:-/usr/local/cuda-12.4/bin/nsys}"
if [[ ! -x "$NSYS" ]]; then
    echo "nsys not found at $NSYS (set NSYS=/path/to/nsys to override)" >&2
    exit 3
fi

PROFILE_OUT="/tmp/${LABEL}.nsys-rep"
JSON_OUT="$BASELINES_DIR/${LABEL}.json"

# Allocator + launch hygiene. Keep these in one place so all runs match.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RUST_LOG="${RUST_LOG:-info}"

echo "[harness] trainer=$TRAINER steps=$STEPS warmup=$WARMUP label=$LABEL"
echo "[harness] writing profile to $PROFILE_OUT"
echo "[harness] trainer args: $*"

# Run under nsys. --trace=cuda keeps the trace size manageable; add nvtx if
# the trainer ever emits NVTX ranges to identify per-step boundaries.
"$NSYS" profile \
    --trace=cuda \
    --cuda-flush-interval=1000 \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --output="${PROFILE_OUT%.nsys-rep}" \
    cargo run --release --manifest-path "$EDV2_DIR/Cargo.toml" \
        --bin "$TRAINER" -- "$@"

if [[ ! -f "$PROFILE_OUT" ]]; then
    echo "[harness] expected $PROFILE_OUT but nsys did not produce it" >&2
    exit 4
fi

echo "[harness] parsing → $JSON_OUT"
python3 "$SCRIPT_DIR/parse_nsys_to_json.py" \
    "$PROFILE_OUT" \
    --steps "$STEPS" \
    --warmup "$WARMUP" \
    --label "$LABEL" \
    --out "$JSON_OUT"

echo "[harness] done. baseline: $JSON_OUT"
echo ""
python3 - <<EOF
import json
with open("$JSON_OUT") as f:
    d = json.load(f)
print(f"  flame_core: {d['metadata']['flame_core_commit']}")
print(f"  edv2:       {d['metadata']['edv2_commit']}")
print(f"  GPU busy:   {d['gpu_activity']['gpu_busy_pct']}%")
print(f"  launches/step: {d['totals']['kernel_launches_per_step']}")
print(f"  driver_api/step: {d['totals']['driver_api_calls_per_step']}")
print("  Top 5 kernels per step:")
for k in d['kernels'][:5]:
    print(f"    {k['count_per_step']:>8.1f}  {k['name'][:70]}")
EOF
