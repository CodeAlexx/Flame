#!/usr/bin/env python3
"""
Production-shape fixtures for `head_rms_norm` and `apply_rope` (Interleaved).

Existing parity coverage runs head_rms_norm at H=8/HD=32/N=64 and has no
standalone `apply_rope` backward test at all. Klein 4B's training shape is
H=24/HD=128/N=1024-1536. The bisect at the production shape produces
chaotic per-block α (random sign, random magnitude) that *only* shows up
at production scale, so any HD=128-specific backward bug is invisible to
the existing fixtures.

This script writes two new fixtures:

  klein_ext_head_rms_norm_prod.safetensors
    Klein's reshape→permute→rms_norm pattern at [B=1, H=24, N=1024, HD=128]
    via the strided permute view that matches `split_qkv` in
    klein-trainer/src/model.rs.

  klein_ext_apply_rope_prod.safetensors
    Standalone Interleaved RoPE forward + backward at
    [B=1, H=24, N=1536, HD=128]. cos/sin are frozen (no grad); only x
    receives gradient.

Output: flame-core/tests/pytorch_fixtures/patterns/

Usage:
    cd flame-core
    python3 scripts/generate_klein_prodshape_norm_rope_fixtures.py
"""

import torch
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
# Fixture 1: head_rms_norm at Klein single-block production shape
#
# Klein single block applies q-norm/k-norm AFTER split_qkv. split_qkv goes
# `[B, N, D] -> reshape([B, N, H, HD]) -> permute([0, 2, 1, 3])`. The
# output is a strided view; the existing `bcc37a7` forward fix and
# `bdc84c8` backward fix both call `.contiguous()` to work around the
# physical-vs-logical-stride bug. Production shape = H=24, HD=128, N=1024.
# ---------------------------------------------------------------------------
print("[1/2] head_rms_norm at production shape (B=1, H=24, N=1024, HD=128)...")

B, N, H, HD = 1, 1024, 24, 128
D = H * HD  # 3072

x = make((B, N, D), SEED).requires_grad_(True)
norm_scale = make((HD,), SEED + 1).requires_grad_(True)


def head_rms_norm_forward():
    # Mirror Klein-trainer's split_qkv pattern: reshape then permute, so
    # the input to rms_norm carries the permute strides.
    h = x.view(B, N, H, HD).permute(0, 2, 1, 3)  # [B, H, N, HD] strided
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
}, str(OUT_DIR / "klein_ext_head_rms_norm_prod.safetensors"))
print(f"  loss={loss.item():.4f}  ||dx||={x.grad.float().norm().item():.3e}  "
      f"||dscale||={norm_scale.grad.float().norm().item():.3e}")
print(f"  out.shape={tuple(out.shape)}  dx.shape={tuple(x.grad.shape)}")


# ---------------------------------------------------------------------------
# Fixture 2: apply_rope (Interleaved) at production post-cat shape
#
# Klein applies RoPE AFTER cat([txt_q, img_q]) so N = N_txt + N_img.
# In our bisect: N_txt=512, N_img=1024 -> N=1536. cos/sin are
# precomputed [1, 1, N, HD/2] tables.
#
# Interleaved RoPE math (matching flame-core's rope_fused_bf16_kernel):
#   for each pair (2d, 2d+1) along HD:
#     y_even = x_even * cos - x_odd * sin
#     y_odd  = x_even * sin + x_odd * cos
# ---------------------------------------------------------------------------
print("\n[2/2] apply_rope (Interleaved) at production shape "
      "(B=1, H=24, N=1536, HD=128)...")

B2, N2, H2, HD2 = 1, 1536, 24, 128
HALF2 = HD2 // 2

# Reset autograd state from prior fixture (we re-use names below).
x2 = make((B2, H2, N2, HD2), SEED + 100).requires_grad_(True)
# cos/sin should be in [-1, 1]. Build from a meaningful angle pattern so
# the kernel exercises a realistic distribution rather than a near-zero
# random one.
torch.manual_seed(SEED + 200)
freqs = torch.arange(0.0, HD2, 2.0, device=DEVICE, dtype=torch.float32) \
    .mul_(-torch.tensor(10000.0).log() / HD2).exp_()  # [HALF2]
positions = torch.arange(0, N2, device=DEVICE, dtype=torch.float32)  # [N2]
angles = positions[:, None] * freqs[None, :]  # [N2, HALF2]
pe_cos = angles.cos().to(DTYPE).view(1, 1, N2, HALF2).contiguous()
pe_sin = angles.sin().to(DTYPE).view(1, 1, N2, HALF2).contiguous()


def apply_rope_interleaved_forward():
    # Match flame-core's rope_fused_bf16_kernel exactly:
    # y_even = x_even * cos - x_odd * sin
    # y_odd  = x_even * sin + x_odd * cos
    # cos/sin are [1, 1, N, half]; broadcast across BH.
    BH = B2 * H2
    x_flat = x2.reshape(BH, N2, HD2)
    x_pairs = x_flat.float().view(BH, N2, HALF2, 2)
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]
    c = pe_cos.float().view(N2, HALF2)
    s = pe_sin.float().view(N2, HALF2)
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    y = torch.stack([y_even, y_odd], dim=-1).view(BH, N2, HD2).to(DTYPE)
    return y.view(B2, H2, N2, HD2)


out2 = apply_rope_interleaved_forward()
loss2 = out2.sum()
loss2.backward()

save_file({
    "x": x2.detach().cpu().contiguous(),
    "pe_cos": pe_cos.detach().cpu().contiguous(),
    "pe_sin": pe_sin.detach().cpu().contiguous(),
    "output": out2.detach().cpu().contiguous(),
    "dx": x2.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_apply_rope_prod.safetensors"))
print(f"  loss={loss2.item():.4f}  ||dx||={x2.grad.float().norm().item():.3e}")
print(f"  out.shape={tuple(out2.shape)}  dx.shape={tuple(x2.grad.shape)}")
print(f"  pe_cos.shape={tuple(pe_cos.shape)}  pe_sin.shape={tuple(pe_sin.shape)}")

print("\nProduction-shape RMSNorm + RoPE fixtures written.")
