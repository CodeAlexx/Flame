#!/usr/bin/env python3
"""Klein's full attention-chain backward fixture.

Tests the chain that Klein's attention does:
    qkv = linear(x, w_qkv, b_qkv)             # [B, N, 3*D]
    q, k, v = chunk(qkv, 3, dim=-1)            # 3 × [B, N, D]
    q = q.view(B, N, H, HD).permute(0,2,1,3)   # [B, H, N, HD] STRIDED view
    q = head_rms_norm(q, q_scale)              # rms_norm * scale on last dim
    q = rope_interleaved(q, cos, sin)          # interleaved-pair RoPE
    (k same as q)
    o = sdpa(q, k, v)                          # [B, H, N, HD]
    o = o.permute(0,2,1,3).reshape(B, N, D)
    out = linear(o, w_out, b_out)

This is exactly the chain Klein's `forward_double_block_attn` does
(except modulate_pre is upstream and joint cat is mixed in). Tests
that the WHOLE attention path's backward matches PyTorch end-to-end.
If THIS fails but each sub-test passes individually, the bug is in
the COMPOSITION of these ops — i.e., some saved-tensor or autograd-
tape interaction across the chain.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file


def rope_interleaved(x, cos, sin):
    """Klein's RoPE: pairs (x[2d], x[2d+1]) rotated by angle θ_d.
    cos/sin are [1, 1, N, half] tables. x is [B, H, N, D]."""
    B, H, N, D = x.shape
    half = D // 2
    # Reshape to expose pairs: [B, H, N, half, 2]
    x_pairs = x.reshape(B, H, N, half, 2)
    x_even = x_pairs[..., 0]  # [B, H, N, half]
    x_odd = x_pairs[..., 1]
    # cos/sin: [1, 1, N, half]
    c = cos.reshape(1, 1, N, half)
    s = sin.reshape(1, 1, N, half)
    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c
    return torch.stack([out_even, out_odd], dim=-1).reshape(B, H, N, D)


def head_rms_norm(x, scale, eps=1e-6):
    """RMSNorm on last dim with learnable scale. x is [..., HD]."""
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    return (x.float() * rstd).to(x.dtype) * scale


torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

B, N, D = 1, 64, 256
H, HD = 8, 32  # 8 heads × 32 dim = 256
half = HD // 2


def m(shape, seed, scale=0.05):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=dtype) * scale)


x = m((B, N, D), 1).requires_grad_(True)
w_qkv = m((D * 3, D), 2).requires_grad_(True)
b_qkv = m((D * 3,), 3).requires_grad_(True)
q_scale = (m((HD,), 4) + 1.0).detach().requires_grad_(True)  # ~1.0 init like Klein
k_scale = (m((HD,), 5) + 1.0).detach().requires_grad_(True)
w_out = m((D, D), 6).requires_grad_(True)
b_out = m((D,), 7).requires_grad_(True)

# RoPE tables (frozen, like Klein's pe_cos/pe_sin built from img_ids).
pe_cos = m((1, 1, N, half), 8) + 0.5
pe_sin = m((1, 1, N, half), 9) + 0.5


def attn_chain():
    qkv = F.linear(x, w_qkv, b_qkv)  # [B, N, 3D]
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, N, H, HD).permute(0, 2, 1, 3)  # [B, H, N, HD] strided
    k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
    v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
    # head_rms_norm on Q and K (not V)
    q = head_rms_norm(q, q_scale)
    k = head_rms_norm(k, k_scale)
    # RoPE on Q and K
    q = rope_interleaved(q, pe_cos, pe_sin)
    k = rope_interleaved(k, pe_cos, pe_sin)
    # SDPA
    o = F.scaled_dot_product_attention(q, k, v)
    # back to [B, N, D]
    o = o.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    # output projection
    return F.linear(o, w_out, b_out)


out = attn_chain()
loss = out.sum()
loss.backward()

Path("tests/pytorch_fixtures/patterns").mkdir(parents=True, exist_ok=True)
save_file({
    "x": x.detach().cpu().contiguous(),
    "w_qkv": w_qkv.detach().cpu().contiguous(),
    "b_qkv": b_qkv.detach().cpu().contiguous(),
    "q_scale": q_scale.detach().cpu().contiguous(),
    "k_scale": k_scale.detach().cpu().contiguous(),
    "w_out": w_out.detach().cpu().contiguous(),
    "b_out": b_out.detach().cpu().contiguous(),
    "pe_cos": pe_cos.detach().cpu().contiguous(),
    "pe_sin": pe_sin.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dw_qkv": w_qkv.grad.cpu().contiguous(),
    "db_qkv": b_qkv.grad.cpu().contiguous(),
    "dq_scale": q_scale.grad.cpu().contiguous(),
    "dk_scale": k_scale.grad.cpu().contiguous(),
    "dw_out": w_out.grad.cpu().contiguous(),
    "db_out": b_out.grad.cpu().contiguous(),
}, "tests/pytorch_fixtures/patterns/klein_ext_attn_chain.safetensors")
print(f"loss={loss.item():.4f}")
print(f"  ||dx||={x.grad.float().norm().item():.3e}")
print(f"  ||dw_qkv||={w_qkv.grad.float().norm().item():.3e}")
print(f"  ||dq_scale||={q_scale.grad.float().norm().item():.3e}")
print(f"  ||dk_scale||={k_scale.grad.float().norm().item():.3e}")
print(f"  ||dw_out||={w_out.grad.float().norm().item():.3e}")
