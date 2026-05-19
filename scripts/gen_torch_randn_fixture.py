"""Generate torch.randn parity fixtures for `flame_core::rng::randn_torch`.

Run with the project's torch venv, e.g.:

    /home/alex/Lance/.venv/bin/python flame-core/scripts/gen_torch_randn_fixture.py

This writes one safetensors per (seed, n) pair into
`flame-core/tests/torch_randn_fixtures/seed{seed}_n{n}.safetensors` with a
single tensor named "data".

Important: torch.randn on CUDA is NOT portable across GPUs (grid size
depends on SM count). The fixture is only valid for the exact GPU it was
generated on. The GPU name is stored as safetensors metadata to allow the
Rust test to detect a mismatch.
"""
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "torch_randn_fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "CUDA-only fixture generator"
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
gpu_name = props.name
sm_count = props.multi_processor_count
max_threads_per_sm = props.max_threads_per_multi_processor
print(f"GPU: {gpu_name}  SMs={sm_count}  maxThreadsPerSM={max_threads_per_sm}")

CASES = [
    (1234, 8),
    (1234, 64),
    (1234, 1024),
    (42, 8),
    (42, 64),
    (999, 1024),
]

metadata = {
    "gpu_name": gpu_name,
    "sm_count": str(sm_count),
    "max_threads_per_sm": str(max_threads_per_sm),
    "torch_version": torch.__version__,
}

for seed, n in CASES:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, dtype=torch.float32, device=device, generator=g)
    cpu = x.detach().cpu().contiguous()
    out_path = OUT_DIR / f"seed{seed}_n{n}.safetensors"
    save_file({"data": cpu}, str(out_path), metadata=metadata)
    print(f"wrote {out_path}  first8={cpu[:min(8, n)].tolist()}")

print("done")
