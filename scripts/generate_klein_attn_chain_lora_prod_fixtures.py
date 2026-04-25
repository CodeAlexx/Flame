#!/usr/bin/env python3
"""
Two more production-shape parity fixtures for the Klein gradient hunt.

After ruling out cuDNN SDPA bwd, recent norm-backward stride fixes,
and standalone head_rms_norm + apply_rope at HD=128, the chaos in the
klein_finite_diff_test bisect must come from either:

  1. Composition at production scale — `linear → split_qkv → rms_norm
     → rope → sdpa → permute → reshape → linear` chain at H=24/HD=128.
     The existing `klein_ext_attn_chain` runs this at H=8/HD=32.

  2. LoRA `forward_delta` at production scale — rank=16, D=3072,
     F32 lora_a/b cast to BF16, `transpose().contiguous()` chain.
     The existing `klein_ext_lora_delta` runs at rank=4/D=256
     and `klein_ext_lora_f32` at toy shape.

This script writes:

  klein_ext_attn_chain_prod.safetensors
    Full attention chain at B=1, N=1024, D=3072, H=24, HD=128.
    Chain mirrors flame-core's parity_klein_ext_attn_chain layout.

  klein_ext_lora_delta_prod.safetensors
    LoRA forward_delta at B=1, N=1024, in=3072, out=9216, rank=16.
    F32 params, BF16 input, alpha=rank (scale=1).

Output: flame-core/tests/pytorch_fixtures/patterns/

Usage:
    cd flame-core
    python3 scripts/generate_klein_attn_chain_lora_prod_fixtures.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

OUT_DIR = Path("tests/pytorch_fixtures/patterns")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make(shape, seed, scale=0.05, dtype=DTYPE):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(shape, generator=g, device=DEVICE, dtype=dtype) * scale


# ---------------------------------------------------------------------------
# Fixture A: full attn_chain at production shape
#
# Chain (matches flame-core parity test):
#   qkv  = Linear(x, w_qkv, b_qkv)
#   q,k,v = split(qkv) -> reshape [B,N,H,HD] -> permute [0,2,1,3]
#   q = head_rms_norm(q, q_scale)
#   k = head_rms_norm(k, k_scale)
#   q,k = apply_rope_interleaved(q, k, pe_cos, pe_sin)
#   o = SDPA(q, k, v)
#   o = permute back + reshape -> [B,N,D]
#   out = Linear(o, w_out, b_out)
# ---------------------------------------------------------------------------
print("[A] attn_chain_prod (B=1, N=1024, D=3072, H=24, HD=128)...")

B, N, H, HD = 1, 1024, 24, 128
D = H * HD  # 3072

x       = make((B, N, D),    SEED + 0).requires_grad_(True)
w_qkv   = make((3 * D, D),   SEED + 1).requires_grad_(True)
b_qkv   = make((3 * D,),     SEED + 2).requires_grad_(True)
q_scale = make((HD,),        SEED + 3).requires_grad_(True)
k_scale = make((HD,),        SEED + 4).requires_grad_(True)
w_out   = make((D, D),       SEED + 5).requires_grad_(True)
b_out   = make((D,),         SEED + 6).requires_grad_(True)

# Build cos/sin tables (Interleaved RoPE), shape [1, 1, N, HD/2].
freqs = torch.arange(0.0, HD, 2.0, device=DEVICE, dtype=torch.float32) \
    .mul_(-torch.tensor(10000.0).log() / HD).exp_()
positions = torch.arange(0, N, device=DEVICE, dtype=torch.float32)
angles = positions[:, None] * freqs[None, :]
pe_cos = angles.cos().to(DTYPE).view(1, 1, N, HD // 2).contiguous()
pe_sin = angles.sin().to(DTYPE).view(1, 1, N, HD // 2).contiguous()


def head_rms_norm_torch(t, scale, eps=1e-6):
    # t: [B, H, N, HD]; scale: [HD]
    var = t.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    return (t.float() * rstd).to(DTYPE) * scale


def rope_interleaved_torch(t, cos, sin):
    # t: [B, H, N, HD]; cos, sin: [1, 1, N, HD/2]
    Bd, Hd, Nd, Dd = t.shape
    half = Dd // 2
    pairs = t.float().view(Bd, Hd, Nd, half, 2)
    x_even = pairs[..., 0]
    x_odd = pairs[..., 1]
    c = cos.float().view(1, 1, Nd, half)
    s = sin.float().view(1, 1, Nd, half)
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    out = torch.stack([y_even, y_odd], dim=-1).view(Bd, Hd, Nd, Dd).to(DTYPE)
    return out


def attn_chain_forward():
    qkv = F.linear(x, w_qkv, b_qkv)              # [B, N, 3D]
    q_flat = qkv[..., 0:D]
    k_flat = qkv[..., D:2 * D]
    v_flat = qkv[..., 2 * D:3 * D]
    q = q_flat.contiguous().view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
    k = k_flat.contiguous().view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
    v = v_flat.contiguous().view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
    q = head_rms_norm_torch(q, q_scale)
    k = head_rms_norm_torch(k, k_scale)
    q = rope_interleaved_torch(q, pe_cos, pe_sin)
    k = rope_interleaved_torch(k, pe_cos, pe_sin)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    return F.linear(o, w_out, b_out)


out = attn_chain_forward()
loss = out.sum()
loss.backward()

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
}, str(OUT_DIR / "klein_ext_attn_chain_prod.safetensors"))
print(f"  out.shape={tuple(out.shape)}  ||dx||={x.grad.float().norm().item():.3e}  "
      f"||dw_qkv||={w_qkv.grad.float().norm().item():.3e}  "
      f"||dq_scale||={q_scale.grad.float().norm().item():.3e}")


# ---------------------------------------------------------------------------
# Fixture B: LoRA forward_delta at production scale
#
# Mirrors flame-diffusion/src/lora.rs::forward_delta exactly:
#   a_t = a.transpose().contiguous()  # [in, rank]
#   b_t = b.transpose().contiguous()  # [rank, out]
#   delta_2d = input2d @ a_t @ b_t * (alpha/rank)
#
# Production: in=3072, out=9216 (3D for qkv on a single block),
# rank=16, alpha=16 (so scale=1.0). lora_a/b are F32 (Parameter
# stability), input is BF16 (matches Klein training).
# ---------------------------------------------------------------------------
print("\n[B] lora_delta_prod (B=1, N=1024, in=3072, out=9216, rank=16)...")

IN_F = D                    # 3072
OUT_F = 3 * D               # 9216
RANK = 16
ALPHA = float(RANK)         # scale = alpha/rank = 1.0

x_l    = make((B, N, IN_F), SEED + 100).requires_grad_(True)
lora_a = make((RANK, IN_F), SEED + 101, dtype=torch.float32).requires_grad_(True)
lora_b = make((OUT_F, RANK), SEED + 102, dtype=torch.float32).requires_grad_(True)


def lora_forward():
    # Match flame-diffusion/src/lora.rs::forward_delta:
    #  a is F32 [rank, in], b is F32 [out, rank]; cast both to BF16,
    #  transpose+contiguify, double matmul.
    a_bf16 = lora_a.to(DTYPE)
    b_bf16 = lora_b.to(DTYPE)
    a_t = a_bf16.transpose(0, 1).contiguous()  # [in, rank] BF16
    b_t = b_bf16.transpose(0, 1).contiguous()  # [rank, out] BF16
    leading = B * N
    x2d = x_l.reshape(leading, IN_F)
    delta_2d = x2d @ a_t @ b_t                  # [leading, out]
    delta_2d = delta_2d * (ALPHA / RANK)
    return delta_2d.view(B, N, OUT_F)


out_l = lora_forward()
loss_l = out_l.sum()
loss_l.backward()

save_file({
    "x":      x_l.detach().cpu().contiguous(),
    "lora_a": lora_a.detach().cpu().contiguous(),
    "lora_b": lora_b.detach().cpu().contiguous(),
    "output": out_l.detach().cpu().contiguous(),
    "dx":      x_l.grad.cpu().contiguous(),
    "dlora_a": lora_a.grad.cpu().contiguous(),
    "dlora_b": lora_b.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_lora_delta_prod.safetensors"))
print(f"  out.shape={tuple(out_l.shape)}  ||dx||={x_l.grad.float().norm().item():.3e}  "
      f"||dlora_a||={lora_a.grad.float().norm().item():.3e}  "
      f"||dlora_b||={lora_b.grad.float().norm().item():.3e}")


# ---------------------------------------------------------------------------
# Fixture C: attn_chain_prod minus SDPA — substitutes a deterministic
# non-attention reduction (q + k + v summed across N) so SDPA isn't on
# the path. If `parity_klein_attn_chain_prod_diag` (with SDPA) shows
# wrong dq_scale/dk_scale but `parity_klein_attn_chain_no_sdpa_prod_diag`
# shows correct, SDPA backward is the residual bug source. Same
# everything else as Fixture A.
# ---------------------------------------------------------------------------
print("\n[C] attn_chain_no_sdpa_prod (B=1, N=1024, D=3072, H=24, HD=128)...")

x_c       = make((B, N, D),    SEED + 200).requires_grad_(True)
w_qkv_c   = make((3 * D, D),   SEED + 201).requires_grad_(True)
b_qkv_c   = make((3 * D,),     SEED + 202).requires_grad_(True)
q_scale_c = make((HD,),        SEED + 203).requires_grad_(True)
k_scale_c = make((HD,),        SEED + 204).requires_grad_(True)
w_out_c   = make((D, D),       SEED + 205).requires_grad_(True)
b_out_c   = make((D,),         SEED + 206).requires_grad_(True)


def attn_chain_no_sdpa_forward():
    qkv = F.linear(x_c, w_qkv_c, b_qkv_c)
    q_flat = qkv[..., 0:D]
    k_flat = qkv[..., D:2 * D]
    v_flat = qkv[..., 2 * D:3 * D]
    q = q_flat.contiguous().view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
    k = k_flat.contiguous().view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
    v = v_flat.contiguous().view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
    q = head_rms_norm_torch(q, q_scale_c)
    k = head_rms_norm_torch(k, k_scale_c)
    q = rope_interleaved_torch(q, pe_cos, pe_sin)
    k = rope_interleaved_torch(k, pe_cos, pe_sin)
    # NO SDPA. Straight pointwise q+k+v then permute back.
    o = (q + k + v)
    o = o.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    return F.linear(o, w_out_c, b_out_c)


out_c = attn_chain_no_sdpa_forward()
loss_c = out_c.sum()
loss_c.backward()

save_file({
    "x": x_c.detach().cpu().contiguous(),
    "w_qkv": w_qkv_c.detach().cpu().contiguous(),
    "b_qkv": b_qkv_c.detach().cpu().contiguous(),
    "q_scale": q_scale_c.detach().cpu().contiguous(),
    "k_scale": k_scale_c.detach().cpu().contiguous(),
    "w_out": w_out_c.detach().cpu().contiguous(),
    "b_out": b_out_c.detach().cpu().contiguous(),
    "pe_cos": pe_cos.detach().cpu().contiguous(),
    "pe_sin": pe_sin.detach().cpu().contiguous(),
    "output": out_c.detach().cpu().contiguous(),
    "dx": x_c.grad.cpu().contiguous(),
    "dw_qkv": w_qkv_c.grad.cpu().contiguous(),
    "db_qkv": b_qkv_c.grad.cpu().contiguous(),
    "dq_scale": q_scale_c.grad.cpu().contiguous(),
    "dk_scale": k_scale_c.grad.cpu().contiguous(),
    "dw_out": w_out_c.grad.cpu().contiguous(),
    "db_out": b_out_c.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_attn_chain_no_sdpa_prod.safetensors"))
print(f"  out.shape={tuple(out_c.shape)}  ||dx||={x_c.grad.float().norm().item():.3e}  "
      f"||dq_scale||={q_scale_c.grad.float().norm().item():.3e}")


print("\nattn_chain_prod + lora_delta_prod + attn_chain_no_sdpa_prod fixtures written.")
