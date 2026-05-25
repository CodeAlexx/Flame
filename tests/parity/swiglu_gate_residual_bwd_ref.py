#!/usr/bin/env python3
"""SwiGLU and GateResidual backward parity reference.

SwiGLU: out = silu(gate) * up
  - gate, up: [B, N, D] BF16
  - out: [B, N, D] BF16

GateResidual: out = residual + tanh(gate).unsqueeze(1) * x
  - residual, x: [B, N, D] BF16
  - gate: [B, D] BF16
  - out: [B, N, D] BF16

Both are fused Z-Image ops in the main training loop.
"""

import json, os, sys
import torch
import torch.nn.functional as F

def swiglu_torch(gate, up):
    """PyTorch reference for SwiGLU: silu(gate) * up."""
    return F.silu(gate) * up

def gate_residual_torch(residual, gate, x):
    """PyTorch reference for gate_residual: residual + gate[:,None,:] * x.

    Matches the CUDA kernel: Y[idx] = r + g * x (no tanh — direct multiply).
    gate: [B, D] broadcast over seq dim N.
    """
    # gate: [B, D] → [B, 1, D] for broadcast
    gate_expanded = gate.unsqueeze(1)  # No tanh — matches the kernel
    return residual + gate_expanded * x

def compute_swiglu(b, n, d, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    gate = (torch.randn(b, n, d, device=device) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
    up   = (torch.randn(b, n, d, device=device) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)

    out = swiglu_torch(gate, up)
    loss = out.float().sum()
    loss.backward()

    return {
        "op": "swiglu",
        "b": b, "n": n, "d": d, "seed": seed,
        "gate_f32": gate.detach().float().cpu().flatten().tolist(),
        "up_f32": up.detach().float().cpu().flatten().tolist(),
        "dgate_f32": gate.grad.detach().float().cpu().flatten().tolist(),
        "dup_f32": up.grad.detach().float().cpu().flatten().tolist(),
    }

def compute_gate_residual(b, n, d, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    residual = (torch.randn(b, n, d, device=device) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
    gate     = (torch.randn(b, d, device=device) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
    x        = (torch.randn(b, n, d, device=device) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)

    out = gate_residual_torch(residual, gate, x)
    loss = out.float().sum()
    loss.backward()

    return {
        "op": "gate_residual",
        "b": b, "n": n, "d": d, "seed": seed,
        "residual_f32": residual.detach().float().cpu().flatten().tolist(),
        "gate_f32": gate.detach().float().cpu().flatten().tolist(),
        "x_f32": x.detach().float().cpu().flatten().tolist(),
        "dresidual_f32": residual.grad.detach().float().cpu().flatten().tolist(),
        "dgate_f32": gate.grad.detach().float().cpu().flatten().tolist(),
        "dx_f32": x.grad.detach().float().cpu().flatten().tolist(),
    }

def main():
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "swiglu_gate_residual_bwd_ref.json",
    )

    cases = []
    # SwiGLU cases at Z-Image MLP hidden=10240 (truncated to 256 for fixture size)
    for b, n, d, seed in [
        (2, 16, 256, 4001),   # Z-Image block style (compact)
        (2, 64, 512, 4002),   # Klein block style (compact)
        (1, 4, 32, 4003),     # Small sanity
    ]:
        print(f"  swiglu ({b},{n},{d}) seed={seed} ...")
        cases.append(compute_swiglu(b, n, d, seed))

    # GateResidual cases
    for b, n, d, seed in [
        (2, 16, 256, 5001),
        (2, 64, 512, 5002),
        (1, 4, 32, 5003),
    ]:
        print(f"  gate_residual ({b},{n},{d}) seed={seed} ...")
        cases.append(compute_gate_residual(b, n, d, seed))

    with open(fixture_path, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"wrote {fixture_path} ({len(cases)} cases)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
