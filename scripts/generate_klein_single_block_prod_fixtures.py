#!/usr/bin/env python3
"""
Production-shape fixtures for Klein single-block ops not covered by
`klein_ext_attn_chain_prod`. Targets the post-Bug-#1/#2-fix bisect chaos
that's still present in the live trainer; the chaos must live in one of:
modulate_pre, swiglu activation, gate_residual, or the full single-block
composition that wires them together.

Each fixture is a separate file so a Rust parity test can fail on one
without blocking the others.

  klein_ext_modulate_pre_prod.safetensors
    LayerNorm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    at x=[1, 1536, 3072] BF16, scale/shift=[1, 3072] BF16, eps=1e-6.
    Single-block N is 1536 (cat'd doubles output).

  klein_ext_swiglu_act_prod.safetensors
    silu(gate) * up at gate=up=[1, 1536, 12288] BF16.
    (Just the activation — linear3d coverage already exists.)

  klein_ext_gate_residual_prod.safetensors
    x + (update * gate.unsqueeze(1).broadcast) at
    x=update=[1, 1536, 3072] BF16, gate=[1, 3072] BF16.

Output: flame-core/tests/pytorch_fixtures/patterns/

Usage:
    cd flame-core
    python3 scripts/generate_klein_single_block_prod_fixtures.py
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
# Fixture 1: modulate_pre at single-block production shape
#
# Mirrors klein-trainer/src/model.rs::modulate_pre exactly:
#   normed = layer_norm(x, [D], weight=None, bias=None, eps=1e-6)  # elementwise only
#   out    = normed * (1 + scale.unsqueeze(1).broadcast) + shift.unsqueeze(1).broadcast
#
# x: [B, N, D] BF16
# scale, shift: [B, D] BF16  (from linear projection of timestep embedding;
#                              shape mirrors what shared_modulation_from_silu
#                              produces — `[B, D]` after narrow on the chunk dim)
#
# Bisect chaos with random LoRA seeds drives a non-trivial upstream `go` into
# this op, so we use a random upstream rather than ones() — same idea as the
# attn_chain fixture.
# ---------------------------------------------------------------------------
print("[1/3] modulate_pre at single-block prod shape "
      "(x=[1, 1536, 3072], scale/shift=[1, 3072])...")

B, N, D = 1, 1536, 3072

x = make((B, N, D), SEED).requires_grad_(True)
scale = make((B, D), SEED + 1, scale=0.1).requires_grad_(True)
shift = make((B, D), SEED + 2, scale=0.1).requires_grad_(True)
go = make((B, N, D), SEED + 3)  # upstream gradient

# Forward exactly mirrors modulate_pre. layer_norm with weight=None/bias=None
# is elementwise-only (per-row mean+var normalization, no affine).
normed = F.layer_norm(x, [D], weight=None, bias=None, eps=1e-6)
out = normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# Backward against random go.
out.backward(go)

save_file({
    "x": x.detach().cpu().contiguous(),
    "scale": scale.detach().cpu().contiguous(),
    "shift": shift.detach().cpu().contiguous(),
    "go": go.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dscale": scale.grad.cpu().contiguous(),
    "dshift": shift.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_modulate_pre_prod.safetensors"))
print(f"  ||dx||={x.grad.float().norm().item():.3e}  "
      f"||dscale||={scale.grad.float().norm().item():.3e}  "
      f"||dshift||={shift.grad.float().norm().item():.3e}")

# ---------------------------------------------------------------------------
# Fixture 2: swiglu activation only (silu(gate) * up)
#
# `gate` and `up` are halves of the linear1 fused output. We test the
# pure activation here; linear3d backward parity is covered elsewhere.
# ---------------------------------------------------------------------------
print("\n[2/3] swiglu activation at single-block prod shape "
      "(gate=up=[1, 1536, 12288])...")

MLP = 12288
gate = make((B, N, MLP), SEED + 10).requires_grad_(True)
up = make((B, N, MLP), SEED + 11).requires_grad_(True)
go2 = make((B, N, MLP), SEED + 12)

act = F.silu(gate) * up
act.backward(go2)

save_file({
    "gate": gate.detach().cpu().contiguous(),
    "up": up.detach().cpu().contiguous(),
    "go": go2.detach().cpu().contiguous(),
    "output": act.detach().cpu().contiguous(),
    "dgate": gate.grad.cpu().contiguous(),
    "dup": up.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_swiglu_act_prod.safetensors"))
print(f"  ||dgate||={gate.grad.float().norm().item():.3e}  "
      f"||dup||={up.grad.float().norm().item():.3e}")

# ---------------------------------------------------------------------------
# Fixture 3: gate_residual (x + update * gate.unsqueeze(1).broadcast)
#
# Mirrors klein-trainer/src/model.rs::gate_residual:
#   gate_b = gate.to_bf16().unsqueeze(1).broadcast_to(update.shape())
#   x.add(update.mul(gate_b))
#
# Both x and update enter as BF16; gate is also BF16 (from a slice of
# the per-block modulation `[B, D]`).
# ---------------------------------------------------------------------------
print("\n[3/3] gate_residual at single-block prod shape "
      "(x=update=[1, 1536, 3072], gate=[1, 3072])...")

x2 = make((B, N, D), SEED + 20).requires_grad_(True)
update = make((B, N, D), SEED + 21).requires_grad_(True)
gate_v = make((B, D), SEED + 22, scale=0.1).requires_grad_(True)
go3 = make((B, N, D), SEED + 23)

# x + update * gate.unsqueeze(1) (broadcast across N)
out3 = x2 + update * gate_v.unsqueeze(1)
out3.backward(go3)

save_file({
    "x": x2.detach().cpu().contiguous(),
    "update": update.detach().cpu().contiguous(),
    "gate": gate_v.detach().cpu().contiguous(),
    "go": go3.detach().cpu().contiguous(),
    "output": out3.detach().cpu().contiguous(),
    "dx": x2.grad.cpu().contiguous(),
    "dupdate": update.grad.cpu().contiguous(),
    "dgate": gate_v.grad.cpu().contiguous(),
}, str(OUT_DIR / "klein_ext_gate_residual_prod.safetensors"))
print(f"  ||dx||={x2.grad.float().norm().item():.3e}  "
      f"||dupdate||={update.grad.float().norm().item():.3e}  "
      f"||dgate||={gate_v.grad.float().norm().item():.3e}")

print("\nSingle-block ingredient fixtures written.")
