#!/usr/bin/env python3
"""RoPE backward parity reference — interleaved and halfsplit layouts.

For each layout, runs:
    y = rope(x, cos, sin)
    loss = y.float().sum()
    dx = grad(loss, x)

Since RoPE is an isometry, backward = forward with -sin.
Tests both layouts at Z-Image's actual shapes.

Z-Image uses Halfsplit (or HalfsplitPytorch) layout based on the rope_fused
call sites in the model. Klein uses HalfsplitPytorch.

Layout semantics:
  - Interleaved: pairs adjacent elements [x0,x1,x2,x3,...] → rotates (x0,x1), (x2,x3), ...
  - Halfsplit: splits first half and second half [x[:D//2], x[D//2:]] → rotates pairs
"""

import json, os, sys, math
import torch

def rope_interleaved_torch(x, cos, sin):
    """RoPE with interleaved pairing. x [..., D], cos/sin [..., D//2]."""
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    # rotate: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

def rope_halfsplit_torch(x, cos, sin):
    """RoPE with half-split layout. x [..., D], cos/sin [..., D//2]."""
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    return out

def compute_case(layout, b, h, n, d_head, seed):
    """layout: 'interleaved' or 'halfsplit'"""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    half = d_head // 2

    # x: [b, h, n, d_head] BF16
    x = (torch.randn(b, h, n, d_head, device=device) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
    # cos/sin: [1, 1, n, half] BF16  (broadcast over b, h)
    # Use plausible RoPE frequencies
    theta = torch.arange(half, device=device, dtype=torch.float32)
    theta = 1.0 / (10000.0 ** (theta / half))
    t = torch.arange(n, device=device, dtype=torch.float32)
    freqs = torch.outer(t, theta)  # [n, half]
    cos_vals = freqs.cos().to(torch.bfloat16)[None, None]  # [1, 1, n, half]
    sin_vals = freqs.sin().to(torch.bfloat16)[None, None]  # [1, 1, n, half]

    if layout == 'interleaved':
        # broadcast cos/sin to [b, h, n, half] via expand
        cos_b = cos_vals.expand(b, h, n, half)
        sin_b = sin_vals.expand(b, h, n, half)
        y = rope_interleaved_torch(x, cos_b, sin_b)
    else:
        cos_b = cos_vals.expand(b, h, n, half)
        sin_b = sin_vals.expand(b, h, n, half)
        y = rope_halfsplit_torch(x, cos_b, sin_b)

    loss = y.float().sum()
    loss.backward()

    return {
        "layout": layout,
        "b": b, "h": h, "n": n, "d_head": d_head,
        "seed": seed,
        # Flat arrays — easier to deserialize in Rust without serde nested vec magic.
        "x_bf16_as_f32": x.detach().float().cpu().flatten().tolist(),
        "cos_bf16_as_f32": cos_vals.detach().float().cpu().flatten().tolist(),
        "sin_bf16_as_f32": sin_vals.detach().float().cpu().flatten().tolist(),
        "dx_f32": x.grad.detach().float().cpu().flatten().tolist(),
    }

def main():
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "rope_bwd_ref.json",
    )

    configs = [
        # (layout, b, h, n, d_head, seed)
        # Z-Image head_dim=128, num_heads=30, typical seq~256..4096
        ("halfsplit", 2, 30, 256, 128, 3001),
        ("halfsplit", 2, 30, 1024, 128, 3002),
        # Klein uses HalfsplitPytorch which is same formula but different element grouping
        ("halfsplit", 2, 32, 256, 128, 3003),
        # Interleaved layout (HiDream-O1, known buggy before fix)
        ("interleaved", 2, 16, 256, 128, 3004),
        # Small sanity
        ("halfsplit", 1, 4, 16, 8, 3005),
        ("interleaved", 1, 4, 16, 8, 3006),
    ]

    cases = []
    for layout, b, h, n, d, seed in configs:
        print(f"  {layout} ({b},{h},{n},{d}) seed={seed} ...")
        cases.append(compute_case(layout, b, h, n, d, seed))

    with open(fixture_path, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"wrote {fixture_path} ({len(cases)} cases)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
