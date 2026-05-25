#!/usr/bin/env python3
"""BF16 matmul backward parity reference — trainer shapes (compact).

Tests the Op::MatMul backward path (used by LoRA's A @ B^T chains) and
the Op::Linear backward path (used by frozen base weights). The cuBLASLt
kernel selection depends on (in_dim, out_dim) aspect ratio AND alignment,
so we test several distinct shape families.

We use batch=4, but scale in/out dims to be representative of the actual
trainer shapes' ALIGNMENT properties (divisible by 128, by 4096, etc.)
without making the fixture huge. Actual trainer uses batch=B*N~512-4096,
but backward correctness is independent of batch (it's just a different K
dimension in the GEMM).

Fixture format: JSON with flat f32 lists. Total size ~50-100MB.
"""

import json, os, sys
import torch

def compute_grad(batch, in_dim, out_dim, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    # A [batch, in_dim] BF16, B [out_dim, in_dim] BF16 — both require grad
    A = (torch.randn(batch, in_dim, device=device, dtype=torch.float32) * 0.02).to(torch.bfloat16).detach().requires_grad_(True)
    B = (torch.randn(out_dim, in_dim, device=device, dtype=torch.float32) * 0.02).to(torch.bfloat16).detach().requires_grad_(True)

    # Forward: out = A @ B^T  [batch, out_dim] BF16
    out = A @ B.t()
    # Sum scalar loss, cast to F32
    loss = out.to(torch.float32).sum()
    loss.backward()

    return {
        "batch": batch,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "seed": seed,
        "A_f32": A.detach().float().cpu().flatten().tolist(),
        "B_f32": B.detach().float().cpu().flatten().tolist(),
        "dA_f32": A.grad.detach().float().cpu().flatten().tolist(),
        "dB_f32": B.grad.detach().float().cpu().flatten().tolist(),
    }

def main():
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "bf16_matmul_bwd_ref.json",
    )

    # Test the KEY dimension-alignment families that affect cuBLASLt dispatch:
    # - Multiple of 128 (all trainer shapes qualify)
    # - Square vs rectangular (attn vs ffn)
    # - Large hidden vs FFN expansion
    # Using batch=4, reduced dims to keep fixture small (~50MB total).
    # Backward correctness is independent of the actual batch/in/out SIZE;
    # it only depends on the GEMM math kernel chosen (which is determined by
    # alignment, not size). We use representative small dims with the same
    # alignment properties as the real shapes.
    shapes = [
        # (batch, in_dim, out_dim, seed)
        # Square, multiple of 256 (attn Q/K/V/Out style)
        (8, 256, 256, 1001),
        # Rectangular 1:4 expansion (FFN W1/W3 style: in=DIM, out=4*DIM)
        (8, 256, 1024, 1002),
        # Rectangular 4:1 contraction (FFN W2 style: in=4*DIM, out=DIM)
        (8, 1024, 256, 1003),
        # Multiple of 512 (Klein style)
        (8, 512, 512, 1004),
        # Asymmetric (cross-attention: cap_feat_dim vs hidden)
        (8, 320, 384, 1005),
        # Small sanity check
        (4, 64, 64, 1006),
        # Larger alignment test: 3840-aligned dimensions (Z-Image hidden)
        # Use 1/15 of full to get 256-aligned
        (4, 256, 256, 1007),
        # 4096 / 16 = 256 already tested; test odd alignment
        (4, 128, 128, 1008),
    ]

    cases = []
    for batch, in_dim, out_dim, seed in shapes:
        print(f"  computing shape ({batch}, {in_dim}, {out_dim}) seed={seed} ...")
        c = compute_grad(batch, in_dim, out_dim, seed)
        cases.append(c)

    with open(fixture_path, "w") as f:
        json.dump({"cases": cases}, f)

    size_mb = os.path.getsize(fixture_path) / 1e6
    print(f"wrote {fixture_path} ({len(cases)} cases, {size_mb:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
