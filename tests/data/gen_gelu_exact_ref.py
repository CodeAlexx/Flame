#!/usr/bin/env python3
"""Generate the PyTorch CUDA reference fixture for `Tensor::gelu_exact`.

Output: `flame-core/tests/data/gelu_exact_ref.safetensors` with keys:
    x        — BF16 [1, 16, 8], CUDA-generated, deterministic from seed 0.
    y_exact  — BF16, F.gelu(x, approximate='none') computed on CUDA.
    y_tanh   — BF16, F.gelu(x, approximate='tanh')  computed on CUDA. Sanity
               key so the Rust test can assert the fixture itself contains two
               distinct outputs (proves it's not all zeros / all NaN).

Why CUDA-only: PyTorch CPU BF16 GELU differs from CUDA BF16 GELU by enough to
matter for sub-ULP parity checks (the project-wide rule: never generate parity
references on CPU; even though this is a single elementwise op the rule stands).

Input range: `torch.randn([1, 16, 8]) * 4.0` after `torch.manual_seed(0)` —
spans roughly ±10 BF16, hitting both saturating tails (where erf ≈ ±1, so
exact and tanh-approx agree) and the linear-ish middle (where they differ).

Run once; the .safetensors output is checked in.
"""

import torch
from safetensors.torch import save_file
from pathlib import Path

assert torch.cuda.is_available(), "CUDA GPU required to generate gelu_exact_ref"

torch.manual_seed(0)
device = "cuda"

# 128 elements — small enough to commit as a fixture, big enough to span the
# distribution. BF16 directly on CUDA so the *input bits* are what flame-core
# will see at test time.
x = (torch.randn(1, 16, 8, dtype=torch.bfloat16, device=device) * 4.0).contiguous()

y_exact = torch.nn.functional.gelu(x, approximate="none")
y_tanh = torch.nn.functional.gelu(x, approximate="tanh")

# Sanity: both computations actually ran on CUDA.
assert x.is_cuda and y_exact.is_cuda and y_tanh.is_cuda, "fixture must be CUDA-computed"

# `.cpu()` only for saving — math already happened on GPU.
save_file(
    {"x": x.cpu(), "y_exact": y_exact.cpu(), "y_tanh": y_tanh.cpu()},
    str(Path(__file__).with_name("gelu_exact_ref.safetensors")),
)

print(f"wrote {Path(__file__).with_name('gelu_exact_ref.safetensors')}")
print(f"  x.shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
print(f"  x[0,0]={x[0,0].tolist()}")
print(f"  y_exact[0,0]={y_exact[0,0].tolist()}")
print(f"  y_tanh[0,0]={y_tanh[0,0].tolist()}")
print(f"  max|y_exact - y_tanh| = {(y_exact.float() - y_tanh.float()).abs().max().item():.6f}")
