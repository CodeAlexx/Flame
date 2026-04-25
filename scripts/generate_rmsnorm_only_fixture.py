#!/usr/bin/env python3
"""Just rms_norm on a contiguous input — no view/permute."""
import torch
from pathlib import Path
from safetensors.torch import save_file

torch.manual_seed(42)
B, H, N, HD = 1, 8, 64, 32
device = "cuda"
dtype = torch.bfloat16

x = (torch.randn(B, H, N, HD, device=device, dtype=dtype) * 0.05).requires_grad_(True)
scale = (torch.randn(HD, device=device, dtype=dtype) * 0.05).requires_grad_(True)

eps = 1e-6
var = x.float().pow(2).mean(dim=-1, keepdim=True)
rstd = torch.rsqrt(var + eps)
out = (x.float() * rstd).to(dtype) * scale

loss = out.sum()
loss.backward()

Path("tests/pytorch_fixtures/patterns").mkdir(parents=True, exist_ok=True)
save_file({
    "x": x.detach().cpu().contiguous(),
    "scale": scale.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dscale": scale.grad.cpu().contiguous(),
}, "tests/pytorch_fixtures/patterns/klein_ext_rms_norm_contig.safetensors")
print(f"loss={loss.item():.4f} ||dx||={x.grad.float().norm().item():.3e} "
      f"||dscale||={scale.grad.float().norm().item():.3e}")
