#!/usr/bin/env python3
"""Multi-block Klein fixture: stack 3 attention+MLP double-blocks with
residuals, run backward, save grads. Tests whether per-block parity
composes correctly across blocks (the per-block error compounding
hypothesis from session 2)."""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

NUM_BLOCKS = 3
B, N, D = 1, 64, 256
H, HD = 8, 32


def m(shape, seed, scale=0.05):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=dtype) * scale)


x = m((B, N, D), 1).requires_grad_(True)
gate = m((1, 1, D), 99)  # frozen modulation gate


def make_block_weights(block_idx):
    """Each block has its own weights. Block_idx seeds them deterministically."""
    base = 100 + block_idx * 20
    return {
        "w_qkv": m((D * 3, D), base + 1).requires_grad_(True),
        "b_qkv": m((D * 3,), base + 2).requires_grad_(True),
        "w_out": m((D, D), base + 3).requires_grad_(True),
        "b_out": m((D,), base + 4).requires_grad_(True),
        "w_up": m((D * 2, D), base + 5).requires_grad_(True),
        "b_up": m((D * 2,), base + 6).requires_grad_(True),
        "w_down": m((D, D), base + 7).requires_grad_(True),
        "b_down": m((D,), base + 8).requires_grad_(True),
    }


blocks = [make_block_weights(i) for i in range(NUM_BLOCKS)]


def block(h, w):
    """One Klein-like double block: attention + MLP, both with residual."""
    qkv = F.linear(h, w["w_qkv"], w["b_qkv"])
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
    k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
    v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    attn_out = F.linear(o, w["w_out"], w["b_out"])
    h = h + gate * attn_out

    gate_up = F.linear(h, w["w_up"], w["b_up"])
    g, u = gate_up.chunk(2, dim=-1)
    act = F.silu(g) * u
    mlp_out = F.linear(act, w["w_down"], w["b_down"])
    return h + gate * mlp_out


h = x
for w in blocks:
    h = block(h, w)
out = h

loss = out.sum()
loss.backward()

save_dict = {
    "x": x.detach().cpu().contiguous(),
    "gate": gate.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
}
for i, w in enumerate(blocks):
    for name, t in w.items():
        save_dict[f"b{i}_{name}"] = t.detach().cpu().contiguous()
        save_dict[f"b{i}_d{name}"] = t.grad.cpu().contiguous()

Path("tests/pytorch_fixtures/patterns").mkdir(parents=True, exist_ok=True)
save_file(save_dict, "tests/pytorch_fixtures/patterns/klein_ext_multiblock.safetensors")

print(f"loss={loss.item():.4f}")
print(f"  ||dx||={x.grad.float().norm().item():.3e}")
for i, w in enumerate(blocks):
    g_qkv = w["w_qkv"].grad.float().norm().item()
    g_down = w["w_down"].grad.float().norm().item()
    print(f"  block{i}: ||dw_qkv||={g_qkv:.3e}  ||dw_down||={g_down:.3e}")
