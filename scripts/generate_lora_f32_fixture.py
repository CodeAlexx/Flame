#!/usr/bin/env python3
"""LoRA with F32 params + cast-to-BF16 + matmul chain.
Mirrors flame-diffusion::lora.rs::forward_delta exactly:
    a_f32 → to_dtype(BF16) → transpose → contiguous → matmul → ...
Klein uses F32 LoRA params; previous parity tests used BF16 LoRA."""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

torch.manual_seed(42)
B, N, D = 1, 64, 256
RANK = 16
ALPHA = float(RANK)
device = "cuda"

x = (torch.randn(B, N, D, device=device, dtype=torch.bfloat16) * 0.05).requires_grad_(True)
# F32 params (like Klein's Parameter)
lora_a = (torch.randn(RANK, D, device=device, dtype=torch.float32) * 0.05).requires_grad_(True)
lora_b = (torch.randn(D * 2, RANK, device=device, dtype=torch.float32) * 0.05).requires_grad_(True)


def forward():
    a_bf16 = lora_a.to(torch.bfloat16)
    b_bf16 = lora_b.to(torch.bfloat16)
    delta = F.linear(F.linear(x, a_bf16), b_bf16) * (ALPHA / RANK)
    return delta


out = forward()
loss = out.sum()
loss.backward()

Path("tests/pytorch_fixtures/patterns").mkdir(parents=True, exist_ok=True)
save_file({
    "x": x.detach().cpu().contiguous(),
    "lora_a": lora_a.detach().cpu().contiguous(),
    "lora_b": lora_b.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dlora_a": lora_a.grad.cpu().contiguous(),
    "dlora_b": lora_b.grad.cpu().contiguous(),
}, "tests/pytorch_fixtures/patterns/klein_ext_lora_f32.safetensors")
print(f"loss={loss.item():.4f}  ||dlora_a||={lora_a.grad.float().norm().item():.3e}  "
      f"||dlora_b||={lora_b.grad.float().norm().item():.3e}")
print(f"  lora_a dtype: {lora_a.dtype}, lora_a.grad dtype: {lora_a.grad.dtype}")
