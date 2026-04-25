#!/usr/bin/env python3
"""
Klein-extension backward fixtures.

flame-core's basic backward chain (linear+bias+SDPA+SwiGLU+residual)
already matches PyTorch on parity tests. The remaining Klein finite-diff
gap (50-1100×) is in one of Klein's *specific extensions* — LoRA,
modulate_pre (layer_norm + (1+scale)*x + shift), head_rms_norm with
learnable scale, RoPE, or joint cat. This script generates one isolated
backward fixture per extension; the matching Rust parity tests reveal
which one diverges.

Outputs to flame-core/tests/pytorch_fixtures/patterns/klein_ext_*.safetensors.

Usage:
    cd flame-core
    python scripts/generate_klein_extensions_fixtures.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Single canonical small shape for all extension tests.
B, N, D = 1, 64, 256
H, HD = 8, 32  # 8 × 32 = 256


def make(shape, seed, scale=0.05):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=DEVICE, dtype=DTYPE) * scale)


OUT_DIR = Path("tests/pytorch_fixtures/patterns")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Extension 1: modulate_pre = layer_norm(x, dim) * (1 + scale) + shift
# ---------------------------------------------------------------------------
print("[1/4] modulate_pre + linear...")

x = make((B, N, D), SEED).requires_grad_(True)
scale = make((B, 1, D), SEED + 1).requires_grad_(True)
shift = make((B, 1, D), SEED + 2).requires_grad_(True)
w = make((D, D), SEED + 3).requires_grad_(True)
b = make((D,), SEED + 4).requires_grad_(True)


def modulate_pre_forward():
    normed = F.layer_norm(x, [D], eps=1e-6)
    modulated = normed * (1.0 + scale) + shift
    return F.linear(modulated, w, b)


out = modulate_pre_forward()
loss = out.sum()
loss.backward()

save_file({
    "x": x.detach().cpu().contiguous(),
    "scale": scale.detach().cpu().contiguous(),
    "shift": shift.detach().cpu().contiguous(),
    "w": w.detach().cpu().contiguous(),
    "b": b.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dscale": scale.grad.cpu().contiguous(),
    "dshift": shift.grad.cpu().contiguous(),
    "dw": w.grad.cpu().contiguous(),
    "db": b.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_modulate_pre.safetensors"))
print(f"  loss={loss.item():.4f}  ||dx||={x.grad.float().norm().item():.3e}  "
      f"||dscale||={scale.grad.float().norm().item():.3e}  "
      f"||dshift||={shift.grad.float().norm().item():.3e}")


# ---------------------------------------------------------------------------
# Extension 2: head_rms_norm — RMSNorm on last dim with learnable scale,
#              applied to a permuted view of [B, N, H, HD] -> [B, H, N, HD]
# ---------------------------------------------------------------------------
print("[2/4] head_rms_norm on permuted view...")

x = make((B, N, D), SEED + 10).requires_grad_(True)
norm_scale = make((HD,), SEED + 11).requires_grad_(True)


def head_rms_norm_forward():
    # Klein's pattern: reshape to [B, N, H, HD], permute to [B, H, N, HD],
    # rms_norm along last dim with learnable scale.
    h = x.view(B, N, H, HD).permute(0, 2, 1, 3)  # [B, H, N, HD] strided
    # Simple RMSNorm: x / sqrt(mean(x^2) + eps) * scale
    eps = 1e-6
    var = h.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    normed = (h.float() * rstd).to(DTYPE) * norm_scale
    return normed


out = head_rms_norm_forward()
loss = out.sum()
loss.backward()

save_file({
    "x": x.detach().cpu().contiguous(),
    "scale": norm_scale.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dscale": norm_scale.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_head_rms_norm.safetensors"))
print(f"  loss={loss.item():.4f}  ||dx||={x.grad.float().norm().item():.3e}  "
      f"||dscale||={norm_scale.grad.float().norm().item():.3e}")


# ---------------------------------------------------------------------------
# Extension 3: LoRA forward_delta = scale * (input @ A^T) @ B^T
# ---------------------------------------------------------------------------
print("[3/4] LoRA forward_delta...")

RANK = 4
ALPHA = float(RANK)

x = make((B, N, D), SEED + 20).requires_grad_(True)
lora_a = make((RANK, D), SEED + 21).requires_grad_(True)
lora_b = make((D, RANK), SEED + 22).requires_grad_(True)


def lora_forward():
    # input @ A^T → [B, N, rank]
    mid = F.linear(x, lora_a)  # F.linear computes x @ A^T
    out = F.linear(mid, lora_b) * (ALPHA / RANK)
    return out


out = lora_forward()
loss = out.sum()
loss.backward()

save_file({
    "x": x.detach().cpu().contiguous(),
    "lora_a": lora_a.detach().cpu().contiguous(),
    "lora_b": lora_b.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dlora_a": lora_a.grad.cpu().contiguous(),
    "dlora_b": lora_b.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_lora_delta.safetensors"))
print(f"  loss={loss.item():.4f}  ||dx||={x.grad.float().norm().item():.3e}  "
      f"||dlora_a||={lora_a.grad.float().norm().item():.3e}  "
      f"||dlora_b||={lora_b.grad.float().norm().item():.3e}")


# ---------------------------------------------------------------------------
# Extension 4: Joint cat of two streams along seq dim, then SDPA
# ---------------------------------------------------------------------------
print("[4/4] joint cat + SDPA backward...")

N_txt, N_img = 16, 48  # text + image tokens
q_txt = make((B, H, N_txt, HD), SEED + 30).requires_grad_(True)
q_img = make((B, H, N_img, HD), SEED + 31).requires_grad_(True)
k_txt = make((B, H, N_txt, HD), SEED + 32).requires_grad_(True)
k_img = make((B, H, N_img, HD), SEED + 33).requires_grad_(True)
v_txt = make((B, H, N_txt, HD), SEED + 34).requires_grad_(True)
v_img = make((B, H, N_img, HD), SEED + 35).requires_grad_(True)


def joint_cat_sdpa():
    # Klein's joint attention: cat along seq dim.
    q = torch.cat([q_txt, q_img], dim=2)
    k = torch.cat([k_txt, k_img], dim=2)
    v = torch.cat([v_txt, v_img], dim=2)
    return F.scaled_dot_product_attention(q, k, v)


out = joint_cat_sdpa()
loss = out.sum()
loss.backward()

save_file({
    "q_txt": q_txt.detach().cpu().contiguous(),
    "q_img": q_img.detach().cpu().contiguous(),
    "k_txt": k_txt.detach().cpu().contiguous(),
    "k_img": k_img.detach().cpu().contiguous(),
    "v_txt": v_txt.detach().cpu().contiguous(),
    "v_img": v_img.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dq_txt": q_txt.grad.cpu().contiguous(),
    "dq_img": q_img.grad.cpu().contiguous(),
    "dk_txt": k_txt.grad.cpu().contiguous(),
    "dk_img": k_img.grad.cpu().contiguous(),
    "dv_txt": v_txt.grad.cpu().contiguous(),
    "dv_img": v_img.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_joint_cat_sdpa.safetensors"))
print(f"  loss={loss.item():.4f}  ||dq_txt||={q_txt.grad.float().norm().item():.3e}  "
      f"||dq_img||={q_img.grad.float().norm().item():.3e}")

print("\nall extension fixtures saved.")
