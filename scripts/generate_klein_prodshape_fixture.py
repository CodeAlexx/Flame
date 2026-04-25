#!/usr/bin/env python3
"""Klein-scale single-block fixture: matches Klein's actual production
tensor sizes (H=24, HD=128, D=3072, seq=1536). cuDNN SDPA may take a
different code path at these shapes than at the small ones used by all
prior parity fixtures."""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

# Klein's actual shapes
B, N, D = 1, 1536, 3072
H, HD = 24, 128
assert H * HD == D


def m(shape, seed, scale=0.05):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=dtype) * scale)


x = m((B, N, D), 1).requires_grad_(True)
gate = m((1, 1, D), 99)

w_qkv = m((D * 3, D), 2).requires_grad_(True)
b_qkv = m((D * 3,), 3).requires_grad_(True)
w_out = m((D, D), 4).requires_grad_(True)
b_out = m((D,), 5).requires_grad_(True)
w_up = m((D * 2, D), 6).requires_grad_(True)
b_up = m((D * 2,), 7).requires_grad_(True)
w_down = m((D, D), 8).requires_grad_(True)
b_down = m((D,), 9).requires_grad_(True)


def block(h):
    qkv = F.linear(h, w_qkv, b_qkv)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
    k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
    v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    attn_out = F.linear(o, w_out, b_out)
    h = h + gate * attn_out

    gate_up = F.linear(h, w_up, b_up)
    g, u = gate_up.chunk(2, dim=-1)
    act = F.silu(g) * u
    mlp_out = F.linear(act, w_down, b_down)
    return h + gate * mlp_out


out = block(x)
loss = out.sum()
loss.backward()

Path("tests/pytorch_fixtures/patterns").mkdir(parents=True, exist_ok=True)
save_file({
    "x": x.detach().cpu().contiguous(),
    "gate": gate.detach().cpu().contiguous(),
    "w_qkv": w_qkv.detach().cpu().contiguous(),
    "b_qkv": b_qkv.detach().cpu().contiguous(),
    "w_out": w_out.detach().cpu().contiguous(),
    "b_out": b_out.detach().cpu().contiguous(),
    "w_up": w_up.detach().cpu().contiguous(),
    "b_up": b_up.detach().cpu().contiguous(),
    "w_down": w_down.detach().cpu().contiguous(),
    "b_down": b_down.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dw_qkv": w_qkv.grad.cpu().contiguous(),
    "db_qkv": b_qkv.grad.cpu().contiguous(),
    "dw_out": w_out.grad.cpu().contiguous(),
    "db_out": b_out.grad.cpu().contiguous(),
    "dw_up": w_up.grad.cpu().contiguous(),
    "db_up": b_up.grad.cpu().contiguous(),
    "dw_down": w_down.grad.cpu().contiguous(),
    "db_down": b_down.grad.cpu().contiguous(),
}, "tests/pytorch_fixtures/patterns/klein_ext_prodshape.safetensors")
print(f"loss={loss.item():.4f}")
print(f"  ||dx||={x.grad.float().norm().item():.3e}")
print(f"  ||dw_qkv||={w_qkv.grad.float().norm().item():.3e}")
print(f"  ||dw_down||={w_down.grad.float().norm().item():.3e}")
