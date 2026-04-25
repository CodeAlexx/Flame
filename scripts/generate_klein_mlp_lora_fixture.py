#!/usr/bin/env python3
"""Klein's MLP with LoRA on the gate-up linear — the EXACT structure
of img_mlp_0.lora_a/b probes that finite-diff says is wrong."""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

torch.manual_seed(42)
B, N, D = 1, 64, 256
RANK = 16
ALPHA = float(RANK)
device = "cuda"
dtype = torch.bfloat16


def m(shape, seed, scale=0.05):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=dtype) * scale)


# Klein's structure:
#   img_mlp_in = modulate_pre(img, shift, scale)
#   gate_up = linear3d(img_mlp_in, gate_up_w) + LoRA_0.forward_delta(img_mlp_in)
#   ... swiglu ...
#   linear_down(act, down_w)
#   out = img + gate * mlp_out
x = m((B, N, D), 1)
mod_shift = m((B, 1, D), 2)
mod_scale = m((B, 1, D), 3)
gate_up_w = m((D * 2, D), 4)
gate_up_b = m((D * 2,), 5)
lora_a = m((RANK, D), 6).requires_grad_(True)  # the probe params
lora_b = m((D * 2, RANK), 7).requires_grad_(True)
down_w = m((D, D), 8)
down_b = m((D,), 9)
gate2 = m((B, 1, D), 10)


def forward():
    # modulate_pre
    normed = F.layer_norm(x, [D], eps=1e-6)
    img_mlp_in = normed * (1.0 + mod_scale) + mod_shift
    # gate_up = linear3d + LoRA
    gate_up = F.linear(img_mlp_in, gate_up_w, gate_up_b)
    # LoRA delta
    delta = F.linear(F.linear(img_mlp_in, lora_a), lora_b) * (ALPHA / RANK)
    gate_up = gate_up + delta
    # SwiGLU
    g, u = gate_up.chunk(2, dim=-1)
    act = F.silu(g) * u
    # down
    mlp_out = F.linear(act, down_w, down_b)
    # residual + gate
    out = x + gate2 * mlp_out
    return out


out = forward()
loss = out.sum()
loss.backward()

Path("tests/pytorch_fixtures/patterns").mkdir(parents=True, exist_ok=True)
save_file({
    "x": x.detach().cpu().contiguous(),
    "mod_shift": mod_shift.detach().cpu().contiguous(),
    "mod_scale": mod_scale.detach().cpu().contiguous(),
    "gate_up_w": gate_up_w.detach().cpu().contiguous(),
    "gate_up_b": gate_up_b.detach().cpu().contiguous(),
    "lora_a": lora_a.detach().cpu().contiguous(),
    "lora_b": lora_b.detach().cpu().contiguous(),
    "down_w": down_w.detach().cpu().contiguous(),
    "down_b": down_b.detach().cpu().contiguous(),
    "gate2": gate2.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dlora_a": lora_a.grad.cpu().contiguous(),
    "dlora_b": lora_b.grad.cpu().contiguous(),
}, "tests/pytorch_fixtures/patterns/klein_ext_mlp_lora.safetensors")
print(f"loss={loss.item():.4f}  ||dlora_a||={lora_a.grad.float().norm().item():.3e}  "
      f"||dlora_b||={lora_b.grad.float().norm().item():.3e}")
