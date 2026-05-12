#!/usr/bin/env python3
"""
parse_nsys_to_json.py

Reads an nsys SQLite export (or .nsys-rep, which it auto-exports) and emits
a JSON summary used by the launch-count harness:
  - per-kernel launch count + cumulative time
  - per-driver-API call count + cumulative time
  - GPU busy / idle fraction (merged interval sweep)
  - metadata (git commit, gpu model, n_steps, etc.) supplied via CLI flags

Usage:
  parse_nsys_to_json.py <path.nsys-rep|.sqlite> --steps N
        [--warmup K] [--label NAME] [--out OUT.json]

Per-step counts are reported as `count / max(steps - warmup, 1)` — averaging
over the steady-state window assumes step boundaries are not extracted from
the trace. For profiles where init/JIT dominates the first few steps, pass
--warmup so the per-step numbers reflect steady state.

Skeptic notes (Phase 0):
  - Kernel names are pinned by exact-match (StringIds.value), not regex.
    Renaming a kernel in source will register as a different entry; that's
    deliberate — silent rename masking would defeat the harness.
  - We do NOT subtract warmup-window counts from totals; the harness expects
    the caller to either pass a steady-state-only profile, or pass --warmup
    so per-step numbers are computed over the steady window. Raw totals stay
    untouched for traceability.
"""

import argparse
import json
import os
import subprocess
import sys
import sqlite3
import shutil
import datetime


def ensure_sqlite(path: str) -> str:
    """Return path to a usable .sqlite file. Exports .nsys-rep if needed."""
    if path.endswith(".sqlite"):
        if not os.path.exists(path):
            sys.exit(f"sqlite not found: {path}")
        return path
    if not path.endswith(".nsys-rep"):
        sys.exit(f"unrecognized extension (expected .nsys-rep or .sqlite): {path}")
    sqlite_path = path[: -len(".nsys-rep")] + ".sqlite"
    if not os.path.exists(sqlite_path):
        nsys = shutil.which("nsys") or "/usr/local/cuda-12.4/bin/nsys"
        if not os.path.exists(nsys):
            sys.exit(f"nsys not found at {nsys}; cannot export sqlite from .nsys-rep")
        rc = subprocess.run(
            [nsys, "export", "--type", "sqlite", path, "-o", sqlite_path],
            capture_output=True,
        )
        if rc.returncode != 0:
            sys.exit(
                f"nsys export failed (rc={rc.returncode}):\n"
                f"stderr: {rc.stderr.decode(errors='replace')}"
            )
    return sqlite_path


def collect_kernels(cur, divisor: int):
    rows = cur.execute(
        """
        SELECT
            s.value AS name,
            COUNT(*) AS n,
            SUM(k.end - k.start) AS total_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        GROUP BY s.value
        ORDER BY n DESC
        """
    ).fetchall()
    return [
        {
            "name": name,
            "total_count": int(n),
            "count_per_step": round(n / divisor, 3),
            "total_time_ns": int(total),
            "time_per_step_ns": int(total / divisor),
        }
        for (name, n, total) in rows
    ]


def collect_driver_api(cur, divisor: int):
    """Driver API calls (cuCtxSetCurrent, cuMemAllocAsync, etc.)."""
    try:
        rows = cur.execute(
            """
            SELECT s.value AS name, COUNT(*) AS n, SUM(r.end - r.start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            GROUP BY s.value
            ORDER BY n DESC
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "name": name,
            "total_count": int(n),
            "count_per_step": round(n / divisor, 3),
            "total_time_ns": int(total or 0),
            "time_per_step_ns": int((total or 0) / divisor),
        }
        for (name, n, total) in rows
    ]


def gpu_busy_idle(cur):
    """Sweep kernel+memcpy+memset intervals and report merged busy/idle."""
    try:
        cur.execute(
            """
            SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
            UNION ALL SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
            UNION ALL SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET
            ORDER BY start
            """
        )
    except sqlite3.OperationalError:
        return None
    intervals = cur.fetchall()
    if not intervals:
        return None
    span_start = intervals[0][0]
    span_end = max(e for _, e in intervals)
    span = span_end - span_start
    busy = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            if e > ce:
                ce = e
        else:
            busy += ce - cs
            cs, ce = s, e
    busy += ce - cs
    return {
        "kernel_span_ns": int(span),
        "gpu_busy_ns": int(busy),
        "gpu_idle_ns": int(span - busy),
        "gpu_busy_pct": round(100.0 * busy / span, 2) if span > 0 else 0.0,
    }


def gpu_info():
    info = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        if out:
            first = out.splitlines()[0].split(",")
            if len(first) >= 2:
                info["gpu_model"] = first[0].strip()
                info["driver_version"] = first[1].strip()
    except Exception:
        pass
    return info


def git_commit(repo_root: str):
    if not os.path.isdir(os.path.join(repo_root, ".git")):
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", repo_root, "diff", "--quiet"],
            capture_output=True,
        ).returncode
        return out + ("-dirty" if dirty != 0 else "")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help=".nsys-rep or .sqlite")
    ap.add_argument(
        "--steps",
        type=int,
        required=True,
        help="total trainer steps in this profile (divisor for per-step values)",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="warmup steps included in profile; subtracted from divisor only",
    )
    ap.add_argument("--label", default=None, help="label embedded in JSON metadata")
    ap.add_argument("--out", default=None, help="write JSON here; default stdout")
    ap.add_argument(
        "--top-kernels",
        type=int,
        default=40,
        help="cap on per-kernel entries (kept top-N by count)",
    )
    ap.add_argument(
        "--top-driver-api",
        type=int,
        default=20,
        help="cap on per-driver-API entries (kept top-N by count)",
    )
    args = ap.parse_args()

    sqlite_path = ensure_sqlite(args.profile)
    divisor = max(args.steps - args.warmup, 1)

    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    kernels = collect_kernels(cur, divisor)
    driver_api = collect_driver_api(cur, divisor)
    busy = gpu_busy_idle(cur)

    # Two commits to pin: flame-core (where kernels live) and EriDiffusion-v2
    # (where trainers live). Either may be missing in non-git checkouts.
    benches_dir = os.path.dirname(os.path.abspath(__file__))
    flame_core_root = os.path.abspath(os.path.join(benches_dir, os.pardir))
    edv2_root = os.path.abspath(
        os.path.join(flame_core_root, os.pardir, "EriDiffusion-v2")
    )

    summary = {
        "metadata": {
            "label": args.label,
            "profile_path": os.path.abspath(args.profile),
            "sqlite_path": sqlite_path,
            "steps": args.steps,
            "warmup": args.warmup,
            "divisor": divisor,
            "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "flame_core_commit": git_commit(flame_core_root),
            "edv2_commit": git_commit(edv2_root),
            **gpu_info(),
        },
        "gpu_activity": busy,
        "kernels": kernels[: args.top_kernels],
        "driver_api": driver_api[: args.top_driver_api],
        "totals": {
            "n_distinct_kernels": len(kernels),
            "total_kernel_launches": sum(k["total_count"] for k in kernels),
            "kernel_launches_per_step": round(
                sum(k["total_count"] for k in kernels) / divisor, 1
            ),
            "n_distinct_driver_apis": len(driver_api),
            "total_driver_api_calls": sum(d["total_count"] for d in driver_api),
            "driver_api_calls_per_step": round(
                sum(d["total_count"] for d in driver_api) / divisor, 1
            ),
        },
    }

    out = json.dumps(summary, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
            f.write("\n")
    else:
        print(out)


if __name__ == "__main__":
    main()
