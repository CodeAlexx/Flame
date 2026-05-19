"""Generate PyTorch parity fixtures for `flame_core::rng::torch_compat`.

Covers:
  * `torch.rand`   -> `flame_core::rng::rand_torch`
  * `torch.empty(...).bernoulli_(p, generator=g)` -> `bernoulli_torch`
  * `torch.randint(low, high, shape, dtype=torch.int32, generator=g)` -> `randint_torch`
  * `torch.nn.init.kaiming_uniform_` -> `kaiming_uniform_torch`
  * `torch.nn.init.xavier_uniform_`  -> `xavier_uniform_torch`

Run with the project's torch venv, e.g.:

    /home/alex/Lance/.venv/bin/python flame-core/scripts/gen_torch_compat_fixtures.py

Fixtures are written under `flame-core/tests/torch_compat_fixtures/`.

torch.rand / torch.randint / bernoulli on CUDA depend on the GPU's SM
count (grid size) — the GPU name + SM count are stored as safetensors
metadata so the Rust test can skip cleanly if it doesn't match.
"""
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "torch_compat_fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "CUDA-only fixture generator"
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
gpu_name = props.name
sm_count = props.multi_processor_count
max_threads_per_sm = props.max_threads_per_multi_processor
print(f"GPU: {gpu_name}  SMs={sm_count}  maxThreadsPerSM={max_threads_per_sm}")

metadata = {
    "gpu_name": gpu_name,
    "sm_count": str(sm_count),
    "max_threads_per_sm": str(max_threads_per_sm),
    "torch_version": torch.__version__,
}


def save(name: str, t: torch.Tensor):
    out_path = OUT_DIR / name
    save_file({"data": t.detach().cpu().contiguous()}, str(out_path), metadata=metadata)
    print(f"wrote {out_path}  shape={tuple(t.shape)} dtype={t.dtype}")


# -----------------------------------------------------------------------
# torch.rand
# -----------------------------------------------------------------------
RAND_CASES = [
    (1234, 8),
    (1234, 64),
    (1234, 1024),
    (42, 8),
    (42, 64),
    (42, 1024),
    (999, 8),
    (999, 64),
    (999, 1024),
]
for seed, n in RAND_CASES:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(n, dtype=torch.float32, device=device, generator=g)
    save(f"rand_torch_seed{seed}_n{n}.safetensors", x)


# -----------------------------------------------------------------------
# torch.bernoulli (scalar p path via `empty().bernoulli_(p, generator=g)`)
# -----------------------------------------------------------------------
BERNOULLI_CASES = [
    (42, 0.5, 1024),
    (42, 0.5, 4096),
    (42, 0.1, 1024),
    (42, 0.1, 4096),
    (99, 0.5, 1024),
    (99, 0.5, 4096),
    (99, 0.1, 1024),
    (99, 0.1, 4096),
]
for seed, p, n in BERNOULLI_CASES:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty(n, dtype=torch.float32, device=device).bernoulli_(p, generator=g)
    # Sanity check — never anything other than 0/1
    assert set(x.unique().tolist()).issubset({0.0, 1.0}), x.unique().tolist()
    p_tag = f"{p:.2f}".replace(".", "p")
    save(f"bernoulli_torch_seed{seed}_p{p_tag}_n{n}.safetensors", x)


# -----------------------------------------------------------------------
# torch.randint  (output cast to int32 for direct storage parity)
# -----------------------------------------------------------------------
RANDINT_CASES = [
    (42, 0, 100, 1024),
    (42, -5, 100, 1024),
    (42, 0, 1000, 1024),
    (99, 0, 100, 1024),
    (99, -5, 1000, 1024),
]
for seed, low, high, n in RANDINT_CASES:
    g = torch.Generator(device=device).manual_seed(seed)
    # torch.randint default dtype is int64; the kernel for range < 2^32 fills
    # values that fit in int32, so we cast for parity-storage purposes.
    x_i64 = torch.randint(low, high, (n,), device=device, generator=g, dtype=torch.int64)
    assert x_i64.min().item() >= low and x_i64.max().item() < high
    x = x_i64.to(torch.int32)
    low_tag = str(low).replace("-", "m")
    save(
        f"randint_torch_seed{seed}_low{low_tag}_high{high}_n{n}.safetensors",
        x,
    )


# -----------------------------------------------------------------------
# kaiming_uniform_
# -----------------------------------------------------------------------
KAIMING_CASES = [
    # (seed, shape, a, mode, nonlinearity)
    (42, (256, 128), 0.0, "fan_in", "leaky_relu"),
    (42, (1024, 512), 0.0, "fan_in", "leaky_relu"),
]
for seed, shape, a, mode, nl in KAIMING_CASES:
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.empty(*shape, dtype=torch.float32, device=device)
    torch.nn.init.kaiming_uniform_(w, a=a, mode=mode, nonlinearity=nl, generator=g)
    shape_tag = "x".join(str(d) for d in shape)
    save(f"kaiming_uniform_seed{seed}_shape{shape_tag}.safetensors", w)

# Regression fixtures for non-power-of-2 fan_in values. Pre-fix the
# `kaiming_gain` helper returned f32, dropping 29 bits of mantissa from
# `sqrt(2.0)`; for these fans `2/fan` is not exactly representable in f32
# so the final `bound` was 1 ulp off PyTorch. Power-of-2 fans (128, 512)
# accidentally passed because `2/fan = 2^-N` is exactly representable.
# Shape is (1, fan) so fan_in = fan exactly.
KAIMING_FAN_CASES = [
    (42, 137),
    (42, 768),
    (42, 1280),
    (42, 3072),
]
for seed, fan in KAIMING_FAN_CASES:
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.empty(1, fan, dtype=torch.float32, device=device)
    torch.nn.init.kaiming_uniform_(w, a=0.0, mode="fan_in", nonlinearity="leaky_relu", generator=g)
    save(f"kaiming_uniform_seed{seed}_fan{fan}.safetensors", w)


# -----------------------------------------------------------------------
# xavier_uniform_
# -----------------------------------------------------------------------
XAVIER_CASES = [
    # (seed, shape, gain)
    (42, (256, 128), 1.0),
    (42, (1024, 512), 1.0),
]
for seed, shape, gain in XAVIER_CASES:
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.empty(*shape, dtype=torch.float32, device=device)
    torch.nn.init.xavier_uniform_(w, gain=gain, generator=g)
    shape_tag = "x".join(str(d) for d in shape)
    save(f"xavier_uniform_seed{seed}_shape{shape_tag}.safetensors", w)

# Regression fixtures for non-f32-exact `gain` values. Pre-fix the public
# API took `gain: f32`, forcing callers passing `math.sqrt(2.0)` (relu's
# `calculate_gain` output) to lose 29 bits of mantissa before the f64
# bound math. The existing gain=1.0 fixtures hid the bug because 1.0 is
# f32-exact.
XAVIER_GAIN_CASES = [
    # (seed, shape, gain, gain_tag)
    (42, (256, 128), math.sqrt(2.0), "sqrt2"),
    (42, (256, 128), 5.0 / 3.0, "5div3"),
]
for seed, shape, gain, gain_tag in XAVIER_GAIN_CASES:
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.empty(*shape, dtype=torch.float32, device=device)
    torch.nn.init.xavier_uniform_(w, gain=gain, generator=g)
    shape_tag = "x".join(str(d) for d in shape)
    save(
        f"xavier_uniform_seed{seed}_gain{gain_tag}_shape{shape_tag}.safetensors",
        w,
    )

print("done")
