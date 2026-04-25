#!/usr/bin/env python3
"""
Generate a Klein-double-block backward fixture for parity testing.

Saves: input + all weights + sum().backward() grads on each weight + grad_input.
The flame-core test reconstructs the same chain, runs backward, and compares.

This is the ground-truth backward oracle for Klein's autograd bug — if
flame-core's backward grads diverge from PyTorch's by more than the
BF16-noise-floor, the bug is in one of the ops in the chain.

Usage:
    python generate_klein_backward_fixture.py \
        --output flame-core/tests/pytorch_fixtures/patterns/klein_block_backward.safetensors

Requires: torch, safetensors. Same SEED + SHAPES as the main generator.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16


def make_input(shape, seed, scale=0.05):
    """Scaled-down inputs/weights so the multi-matmul chain doesn't
    blow grads into BF16-noise-floor territory. 0.05 keeps each linear's
    output ~O(1) regardless of fan-in."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=DEVICE, dtype=DTYPE) * scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="flame-core/tests/pytorch_fixtures/patterns/klein_block_backward.safetensors",
    )
    parser.add_argument(
        "--shape",
        choices=["small", "klein"],
        default="small",
        help="small = (1, 64, 256) for fast tests; klein = (4, 1024, 4608) for realistic",
    )
    args = parser.parse_args()

    if args.shape == "small":
        B, N, D = 1, 64, 256
        H, HD = 8, 32  # 8 heads × 32 head_dim = 256
    else:
        B, N, D = 4, 1024, 4608
        H, HD = 16, 288

    print(f"shape: B={B}, N={N}, D={D}, H={H}, HD={HD}")

    # Inputs (frozen weights — only x is the gradient sink for the chain;
    # weights collect their own grads).
    x = make_input((B, N, D), seed=SEED)
    w_qkv = make_input((D * 3, D), seed=SEED + 1).requires_grad_(True)
    b_qkv = make_input((D * 3,), seed=SEED + 3).requires_grad_(True)
    w_out = make_input((D, D), seed=SEED + 2).requires_grad_(True)
    b_out = make_input((D,), seed=SEED + 4).requires_grad_(True)
    w_up = make_input((D * 2, D), seed=SEED + 6).requires_grad_(True)
    b_up = make_input((D * 2,), seed=SEED + 7).requires_grad_(True)
    w_down = make_input((D, D), seed=SEED + 8).requires_grad_(True)
    b_down = make_input((D,), seed=SEED + 9).requires_grad_(True)
    gate = make_input((1, 1, D), seed=SEED + 10)  # frozen modulation

    # Required for grad_input
    x.requires_grad_(True)

    def klein_attn_path(h):
        qkv = F.linear(h, w_qkv, b_qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        # Skipping RoPE for fixture simplicity; same on flame-core side
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return F.linear(out, w_out, b_out)

    def klein_mlp_path(h):
        gate_up = F.linear(h, w_up, b_up)
        g, u = gate_up.chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, w_down, b_down)

    # Klein double block (same structure as the main generator — with
    # RoPE removed because flame-core's RoPE table is model-specific).
    attn_out = klein_attn_path(x)
    h = x + gate * attn_out  # residual + gate
    mlp_out = klein_mlp_path(h)
    out = h + gate * mlp_out

    # Sum-loss → backward
    loss = out.sum()
    loss.backward()

    # Collect grads
    grads = {
        "x": x.detach().cpu().contiguous(),
        "w_qkv": w_qkv.detach().cpu().contiguous(),
        "b_qkv": b_qkv.detach().cpu().contiguous(),
        "w_out": w_out.detach().cpu().contiguous(),
        "b_out": b_out.detach().cpu().contiguous(),
        "w_up": w_up.detach().cpu().contiguous(),
        "b_up": b_up.detach().cpu().contiguous(),
        "w_down": w_down.detach().cpu().contiguous(),
        "b_down": b_down.detach().cpu().contiguous(),
        "gate": gate.detach().cpu().contiguous(),
        "output": out.detach().cpu().contiguous(),
        # Grads from PyTorch — the oracle
        "dx": x.grad.cpu().contiguous(),
        "dw_qkv": w_qkv.grad.cpu().contiguous(),
        "db_qkv": b_qkv.grad.cpu().contiguous(),
        "dw_out": w_out.grad.cpu().contiguous(),
        "db_out": b_out.grad.cpu().contiguous(),
        "dw_up": w_up.grad.cpu().contiguous(),
        "db_up": b_up.grad.cpu().contiguous(),
        "dw_down": w_down.grad.cpu().contiguous(),
        "db_down": b_down.grad.cpu().contiguous(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(grads, str(out_path))
    print(f"saved {out_path}")
    print(f"  loss = {loss.item():.6f}")
    print(f"  ||dx|| = {x.grad.float().norm().item():.4e}")
    print(f"  ||dw_qkv|| = {w_qkv.grad.float().norm().item():.4e}")
    print(f"  ||dw_down|| = {w_down.grad.float().norm().item():.4e}")


if __name__ == "__main__":
    main()
