#!/usr/bin/env python3
"""RMSNorm backward parity reference — trainer shapes.

For each (batch, norm_size) shape, runs:
    y = x / rms(x)            (no affine weight — simplest form)
    loss = y.float().sum()
    dx = grad(loss, x)

Also tests with affine weight (elementwise multiply after norm).

Saves JSON fixture for the Rust test.
"""

import json, os, sys
import torch
import torch.nn.functional as F

def rms_norm_torch(x, weight=None, eps=1e-5):
    """PyTorch reference RMSNorm (matches nn.RMSNorm semantics)."""
    # x: [..., norm_size]
    x_f32 = x.float()
    rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = (x_f32 * rms).to(x.dtype)
    if weight is not None:
        normed = normed * weight.to(x.dtype)
    return normed

def compute_case(batch, norm_size, with_weight, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    x = (torch.randn(batch, norm_size, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
    if with_weight:
        w = (torch.ones(norm_size, device=device, dtype=torch.float32) + torch.randn(norm_size, device=device) * 0.05).to(torch.bfloat16).detach().requires_grad_(True)
    else:
        w = None

    y = rms_norm_torch(x, w, eps=1e-5)
    loss = y.float().sum()
    loss.backward()

    case = {
        "batch": batch,
        "norm_size": norm_size,
        "with_weight": with_weight,
        "seed": seed,
        "x_bf16_as_f32": x.detach().float().cpu().tolist(),
        "dx_f32": x.grad.detach().float().cpu().tolist(),
    }
    if with_weight:
        case["w_bf16_as_f32"] = w.detach().float().cpu().tolist()
        case["dw_f32"] = w.grad.detach().float().cpu().tolist()
    return case

def main():
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "rms_norm_bwd_ref.json",
    )

    configs = [
        # (batch, norm_size, with_weight, seed)
        # Z-Image hidden = 3840
        (32, 3840, False, 2001),
        (32, 3840, True,  2002),
        # Z-Image head_dim = 128 (QK RMSNorm path)
        (64, 128, False, 2003),
        (64, 128, True,  2004),
        # Klein hidden = 4096
        (32, 4096, False, 2005),
        (32, 4096, True,  2006),
        # Small sanity
        (4, 64, False, 2007),
    ]

    cases = []
    for batch, norm_size, with_w, seed in configs:
        tag = "w+" if with_w else "no-w"
        print(f"  ({batch}, {norm_size}) {tag} seed={seed} ...")
        cases.append(compute_case(batch, norm_size, with_w, seed))

    with open(fixture_path, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"wrote {fixture_path} ({len(cases)} cases)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
