#!/usr/bin/env python3
"""Klein's FULL double-block at production shape, including:
- modulate_pre on the input (layer_norm + (1+scale)*x + shift)
- linear+LoRA on qkv projection
- split_qkv + head_rms_norm + RoPE + SDPA + out_proj+LoRA + residual+gate
- modulate_pre on intermediate (second modulation)
- linear+LoRA on gate_up + SwiGLU + linear+LoRA on down + residual+gate

This exercises the EXACT structure Klein's `forward_double_block_attn`
runs, end to end, at production shapes.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file


def rope_interleaved(x, cos, sin):
    B, H, N, D = x.shape
    half = D // 2
    x_pairs = x.reshape(B, H, N, half, 2)
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]
    c = cos.reshape(1, 1, N, half)
    s = sin.reshape(1, 1, N, half)
    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c
    return torch.stack([out_even, out_odd], dim=-1).reshape(B, H, N, D)


def head_rms_norm(x, scale, eps=1e-6):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    return (x.float() * rstd).to(x.dtype) * scale


def modulate_pre(x, shift, scale):
    normed = F.layer_norm(x, [x.shape[-1]], eps=1e-6)
    return normed * (1.0 + scale) + shift


torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

# Klein production shape
B, N, D = 1, 256, 768  # smaller than klein 4B's 1536/3072 — keeps test fast
H, HD = 12, 64
half = HD // 2
RANK = 16
ALPHA = float(RANK)


def m(shape, seed, scale=0.05):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=dtype) * scale)


x = m((B, N, D), 1).requires_grad_(True)

# Modulation params (frozen — they come from time embedding upstream)
shift1 = m((B, 1, D), 100)
scale1 = m((B, 1, D), 101)
gate1 = m((1, 1, D), 102)
shift2 = m((B, 1, D), 103)
scale2 = m((B, 1, D), 104)
gate2 = m((1, 1, D), 105)

# Base weights (frozen — like Klein's base model)
w_qkv = m((D * 3, D), 200)
b_qkv = m((D * 3,), 201)
w_out = m((D, D), 202)
b_out = m((D,), 203)
w_up = m((D * 2, D), 204)
b_up = m((D * 2,), 205)
w_down = m((D, D), 206)
b_down = m((D,), 207)

# RMSNorm scales (frozen)
q_scale = (m((HD,), 300) + 1.0).detach()
k_scale = (m((HD,), 301) + 1.0).detach()

# RoPE table (frozen)
pe_cos = m((1, 1, N, half), 400) + 0.5
pe_sin = m((1, 1, N, half), 401) + 0.5

# LoRA params (LEARNABLE — these are the probes)
lora_qkv_a = m((RANK, D), 500).requires_grad_(True)
lora_qkv_b = m((D * 3, RANK), 501).requires_grad_(True)
lora_out_a = m((RANK, D), 502).requires_grad_(True)
lora_out_b = m((D, RANK), 503).requires_grad_(True)
lora_up_a = m((RANK, D), 504).requires_grad_(True)
lora_up_b = m((D * 2, RANK), 505).requires_grad_(True)
lora_down_a = m((RANK, D), 506).requires_grad_(True)
lora_down_b = m((D, RANK), 507).requires_grad_(True)


def lora_delta(x_in, a, b):
    return F.linear(F.linear(x_in, a), b) * (ALPHA / RANK)


def linear_lora(x_in, w, bias, lora_a, lora_b):
    base = F.linear(x_in, w, bias)
    return base + lora_delta(x_in, lora_a, lora_b)


def block(x_in):
    # === Attention path with modulate_pre ===
    x_normed = modulate_pre(x_in, shift1, scale1)

    qkv = linear_lora(x_normed, w_qkv, b_qkv, lora_qkv_a, lora_qkv_b)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
    k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
    v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

    q = head_rms_norm(q, q_scale)
    k = head_rms_norm(k, k_scale)
    q = rope_interleaved(q, pe_cos, pe_sin)
    k = rope_interleaved(k, pe_cos, pe_sin)

    o = F.scaled_dot_product_attention(q, k, v)
    o = o.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    attn_out = linear_lora(o, w_out, b_out, lora_out_a, lora_out_b)

    h = x_in + gate1 * attn_out  # residual + gate

    # === MLP path with modulate_pre ===
    h_normed = modulate_pre(h, shift2, scale2)

    gate_up = linear_lora(h_normed, w_up, b_up, lora_up_a, lora_up_b)
    g, u = gate_up.chunk(2, dim=-1)
    act = F.silu(g) * u
    mlp_out = linear_lora(act, w_down, b_down, lora_down_a, lora_down_b)

    return h + gate2 * mlp_out


out = block(x)
loss = out.sum()
loss.backward()

save_dict = {
    "x": x.detach().cpu().contiguous(),
    "shift1": shift1.detach().cpu().contiguous(),
    "scale1": scale1.detach().cpu().contiguous(),
    "gate1": gate1.detach().cpu().contiguous(),
    "shift2": shift2.detach().cpu().contiguous(),
    "scale2": scale2.detach().cpu().contiguous(),
    "gate2": gate2.detach().cpu().contiguous(),
    "w_qkv": w_qkv.detach().cpu().contiguous(),
    "b_qkv": b_qkv.detach().cpu().contiguous(),
    "w_out": w_out.detach().cpu().contiguous(),
    "b_out": b_out.detach().cpu().contiguous(),
    "w_up": w_up.detach().cpu().contiguous(),
    "b_up": b_up.detach().cpu().contiguous(),
    "w_down": w_down.detach().cpu().contiguous(),
    "b_down": b_down.detach().cpu().contiguous(),
    "q_scale": q_scale.detach().cpu().contiguous(),
    "k_scale": k_scale.detach().cpu().contiguous(),
    "pe_cos": pe_cos.detach().cpu().contiguous(),
    "pe_sin": pe_sin.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
}
for name in ["lora_qkv_a", "lora_qkv_b", "lora_out_a", "lora_out_b",
             "lora_up_a", "lora_up_b", "lora_down_a", "lora_down_b"]:
    t = locals()[name]
    save_dict[name] = t.detach().cpu().contiguous()
    save_dict[f"d{name}"] = t.grad.cpu().contiguous()

Path("tests/pytorch_fixtures/patterns").mkdir(parents=True, exist_ok=True)
save_file(save_dict, "tests/pytorch_fixtures/patterns/klein_ext_full_block.safetensors")

print(f"loss={loss.item():.4f}")
for name in ["lora_qkv_a", "lora_qkv_b", "lora_up_a", "lora_up_b",
             "lora_down_a", "lora_down_b"]:
    g = locals()[name].grad
    print(f"  ||d{name}|| = {g.float().norm().item():.3e}")
