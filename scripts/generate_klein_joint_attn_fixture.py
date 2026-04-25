#!/usr/bin/env python3
"""Klein's full JOINT attention (text + image streams).

Klein's actual joint attention has TWO streams (text and image), each
with its own qkv projection and per-head RMSNorm + RoPE. The streams
are then CONCATENATED along the seq dim before SDPA, then split back
and projected with separate output linears. This fixture covers the
full chain end-to-end:

    img_qkv = linear(img, w_img_qkv, b_img_qkv)
    txt_qkv = linear(txt, w_txt_qkv, b_txt_qkv)
    (img_q, img_k, img_v) = split + permute(img_qkv)
    (txt_q, txt_k, txt_v) = split + permute(txt_qkv)
    img_q = head_rms_norm(img_q, img_q_scale)  # per-head RMS on img Q
    img_k = head_rms_norm(img_k, img_k_scale)
    txt_q = head_rms_norm(txt_q, txt_q_scale)
    txt_k = head_rms_norm(txt_k, txt_k_scale)
    q = cat([txt_q, img_q], dim=2)             # joint along seq
    k = cat([txt_k, img_k], dim=2)
    v = cat([txt_v, img_v], dim=2)
    q = rope_interleaved(q, cos, sin)
    k = rope_interleaved(k, cos, sin)
    o = sdpa(q, k, v)
    txt_o, img_o = o.split([N_txt, N_img], dim=2)
    img_out = linear(img_o.permute_back, w_img_out, b_img_out)
    txt_out = linear(txt_o.permute_back, w_txt_out, b_txt_out)
    final = (img_out + txt_out.sum())   # both contribute to scalar loss
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


torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

B, N_img, N_txt, D = 1, 48, 16, 256
H, HD = 8, 32
half = HD // 2
N_total = N_img + N_txt


def m(shape, seed, scale=0.05):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=dtype) * scale)


img = m((B, N_img, D), 1).requires_grad_(True)
txt = m((B, N_txt, D), 2).requires_grad_(True)

w_img_qkv = m((D * 3, D), 3).requires_grad_(True)
b_img_qkv = m((D * 3,), 4).requires_grad_(True)
w_txt_qkv = m((D * 3, D), 5).requires_grad_(True)
b_txt_qkv = m((D * 3,), 6).requires_grad_(True)

img_q_scale = (m((HD,), 7) + 1.0).detach().requires_grad_(True)
img_k_scale = (m((HD,), 8) + 1.0).detach().requires_grad_(True)
txt_q_scale = (m((HD,), 9) + 1.0).detach().requires_grad_(True)
txt_k_scale = (m((HD,), 10) + 1.0).detach().requires_grad_(True)

w_img_out = m((D, D), 11).requires_grad_(True)
b_img_out = m((D,), 12).requires_grad_(True)
w_txt_out = m((D, D), 13).requires_grad_(True)
b_txt_out = m((D,), 14).requires_grad_(True)

# RoPE table for the JOINT seq length (text + image)
pe_cos = m((1, 1, N_total, half), 15) + 0.5
pe_sin = m((1, 1, N_total, half), 16) + 0.5


def split_qkv(qkv, n):
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, n, H, HD).permute(0, 2, 1, 3)
    k = k.view(B, n, H, HD).permute(0, 2, 1, 3)
    v = v.view(B, n, H, HD).permute(0, 2, 1, 3)
    return q, k, v


def joint_attn():
    img_qkv = F.linear(img, w_img_qkv, b_img_qkv)
    txt_qkv = F.linear(txt, w_txt_qkv, b_txt_qkv)

    img_q, img_k, img_v = split_qkv(img_qkv, N_img)
    txt_q, txt_k, txt_v = split_qkv(txt_qkv, N_txt)

    img_q = head_rms_norm(img_q, img_q_scale)
    img_k = head_rms_norm(img_k, img_k_scale)
    txt_q = head_rms_norm(txt_q, txt_q_scale)
    txt_k = head_rms_norm(txt_k, txt_k_scale)

    # Klein's order: cat(txt, img) along seq (dim=2)
    q = torch.cat([txt_q, img_q], dim=2)  # [B, H, N_total, HD]
    k = torch.cat([txt_k, img_k], dim=2)
    v = torch.cat([txt_v, img_v], dim=2)

    q = rope_interleaved(q, pe_cos, pe_sin)
    k = rope_interleaved(k, pe_cos, pe_sin)

    o = F.scaled_dot_product_attention(q, k, v)  # [B, H, N_total, HD]

    # Split back into txt and img halves (Klein's pattern: txt first)
    txt_o = o[:, :, :N_txt, :]
    img_o = o[:, :, N_txt:, :]

    img_o = img_o.permute(0, 2, 1, 3).contiguous().view(B, N_img, D)
    txt_o = txt_o.permute(0, 2, 1, 3).contiguous().view(B, N_txt, D)

    img_out = F.linear(img_o, w_img_out, b_img_out)
    txt_out = F.linear(txt_o, w_txt_out, b_txt_out)
    return img_out, txt_out


img_out, txt_out = joint_attn()
loss = img_out.sum() + txt_out.sum()
loss.backward()

Path("tests/pytorch_fixtures/patterns").mkdir(parents=True, exist_ok=True)
save_file({
    "img": img.detach().cpu().contiguous(),
    "txt": txt.detach().cpu().contiguous(),
    "w_img_qkv": w_img_qkv.detach().cpu().contiguous(),
    "b_img_qkv": b_img_qkv.detach().cpu().contiguous(),
    "w_txt_qkv": w_txt_qkv.detach().cpu().contiguous(),
    "b_txt_qkv": b_txt_qkv.detach().cpu().contiguous(),
    "img_q_scale": img_q_scale.detach().cpu().contiguous(),
    "img_k_scale": img_k_scale.detach().cpu().contiguous(),
    "txt_q_scale": txt_q_scale.detach().cpu().contiguous(),
    "txt_k_scale": txt_k_scale.detach().cpu().contiguous(),
    "w_img_out": w_img_out.detach().cpu().contiguous(),
    "b_img_out": b_img_out.detach().cpu().contiguous(),
    "w_txt_out": w_txt_out.detach().cpu().contiguous(),
    "b_txt_out": b_txt_out.detach().cpu().contiguous(),
    "pe_cos": pe_cos.detach().cpu().contiguous(),
    "pe_sin": pe_sin.detach().cpu().contiguous(),
    "img_out": img_out.detach().cpu().contiguous(),
    "txt_out": txt_out.detach().cpu().contiguous(),
    "dimg": img.grad.cpu().contiguous(),
    "dtxt": txt.grad.cpu().contiguous(),
    "dw_img_qkv": w_img_qkv.grad.cpu().contiguous(),
    "dw_txt_qkv": w_txt_qkv.grad.cpu().contiguous(),
    "dimg_q_scale": img_q_scale.grad.cpu().contiguous(),
    "dimg_k_scale": img_k_scale.grad.cpu().contiguous(),
    "dtxt_q_scale": txt_q_scale.grad.cpu().contiguous(),
    "dtxt_k_scale": txt_k_scale.grad.cpu().contiguous(),
    "dw_img_out": w_img_out.grad.cpu().contiguous(),
    "dw_txt_out": w_txt_out.grad.cpu().contiguous(),
}, "tests/pytorch_fixtures/patterns/klein_ext_joint_attn.safetensors")
print(f"loss={loss.item():.4f}")
print(f"  ||dimg||={img.grad.float().norm().item():.3e}")
print(f"  ||dtxt||={txt.grad.float().norm().item():.3e}")
print(f"  ||dw_img_qkv||={w_img_qkv.grad.float().norm().item():.3e}")
print(f"  ||dimg_q_scale||={img_q_scale.grad.float().norm().item():.3e}")
print(f"  ||dtxt_q_scale||={txt_q_scale.grad.float().norm().item():.3e}")
