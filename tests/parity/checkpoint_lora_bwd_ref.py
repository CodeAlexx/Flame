#!/usr/bin/env python3
"""
Checkpoint LoRA backward parity fixture generator.

Experiment C from LOCALIZE_FINDINGS_2026-05-24.

Simulates the Op::Checkpoint backward path used in ALL 5 trainers:

  For each case:
    1. lhs_activation (batch, in_dim), requires_grad=True
    2. lora_a (rank, in_dim), requires_grad=True  (LoRA-A, the up-projection)
    3. lora_b (out_dim, rank), requires_grad=True  (LoRA-B, the down-projection)
    4. frozen_w (out_dim, in_dim), requires_grad=False  (base model frozen weight)

  Forward (INSIDE checkpoint):
    - x_lora_a = lhs @ lora_a^T          [batch, rank]
    - x_lora   = x_lora_a @ lora_b^T     [batch, out_dim]
    - x_base   = lhs @ frozen_w^T         [batch, out_dim]
    - out       = x_lora + x_base         [batch, out_dim]

  Forward (OUTSIDE checkpoint — for plain version):
    Same computation but NOT wrapped in torch.utils.checkpoint.checkpoint.

  References:
    - grad_lhs:   grad w.r.t. lhs_activation
    - grad_lora_a: grad w.r.t. lora_a
    - grad_lora_b: grad w.r.t. lora_b
    - grad_frozen_w: None (frozen, no grad)

  PASS criterion: cos_sim >= 0.999 AND rel_L2 <= 1%
    for grad_lhs, grad_lora_a, grad_lora_b
    AND checkpointed version MATCHES plain version.

Run:
  /home/alex/OneTrainer/venv/bin/python \
    /home/alex/EriDiffusion/flame-core/tests/parity/checkpoint_lora_bwd_ref.py
"""

import json
import struct
import numpy as np
import torch
import torch.utils.checkpoint

def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)

def make_case(batch, in_dim, rank, out_dim, seed):
    set_seed(seed)
    lhs = torch.randn(batch, in_dim, dtype=torch.float32) * 0.1
    lora_a = torch.randn(rank, in_dim, dtype=torch.float32) * 0.02
    lora_b = torch.randn(out_dim, rank, dtype=torch.float32) * 0.02
    frozen_w = torch.randn(out_dim, in_dim, dtype=torch.float32) * 0.1

    # Cast to BF16 for realism (all trainers use BF16)
    lhs_bf = lhs.to(torch.bfloat16)
    lora_a_bf = lora_a.to(torch.bfloat16)
    lora_b_bf = lora_b.to(torch.bfloat16)
    frozen_w_bf = frozen_w.to(torch.bfloat16)

    # ── Plain forward (no checkpoint) ──
    lhs_p    = lhs_bf.detach().requires_grad_(True)
    lora_a_p = lora_a_bf.detach().requires_grad_(True)
    lora_b_p = lora_b_bf.detach().requires_grad_(True)
    frozen_p = frozen_w_bf.detach()  # NO requires_grad

    def lora_block_forward(lhs, lora_a, lora_b, frozen_w):
        x_lora_a = lhs @ lora_a.t()
        x_lora   = x_lora_a @ lora_b.t()
        x_base   = lhs @ frozen_w.t()
        return x_lora + x_base

    out_plain = lora_block_forward(lhs_p, lora_a_p, lora_b_p, frozen_p)
    loss_plain = out_plain.to(torch.float32).sum()
    loss_plain.backward()

    grad_lhs_plain   = lhs_p.grad.float().numpy().flatten().tolist()
    grad_lora_a_plain = lora_a_p.grad.float().numpy().flatten().tolist()
    grad_lora_b_plain = lora_b_p.grad.float().numpy().flatten().tolist()

    # ── Checkpointed forward ──
    lhs_c    = lhs_bf.detach().requires_grad_(True)
    lora_a_c = lora_a_bf.detach().requires_grad_(True)
    lora_b_c = lora_b_bf.detach().requires_grad_(True)
    frozen_c = frozen_w_bf.detach()  # NO requires_grad

    # PyTorch checkpoint wraps the function — inputs must be passed explicitly
    # to get grad flow. We use use_reentrant=False (newer PyTorch default).
    def ckpt_fn(lhs, lora_a, lora_b, frozen_w):
        return lora_block_forward(lhs, lora_a, lora_b, frozen_w)

    out_ckpt = torch.utils.checkpoint.checkpoint(
        ckpt_fn, lhs_c, lora_a_c, lora_b_c, frozen_c,
        use_reentrant=False
    )
    loss_ckpt = out_ckpt.to(torch.float32).sum()
    loss_ckpt.backward()

    grad_lhs_ckpt    = lhs_c.grad.float().numpy().flatten().tolist()
    grad_lora_a_ckpt = lora_a_c.grad.float().numpy().flatten().tolist()
    grad_lora_b_ckpt = lora_b_c.grad.float().numpy().flatten().tolist()

    def cos_rel(a, b):
        a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20)
        rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-20)
        return float(cos), float(rel)

    cp_a = cos_rel(grad_lora_a_ckpt, grad_lora_a_plain)
    cp_b = cos_rel(grad_lora_b_ckpt, grad_lora_b_plain)
    cp_x = cos_rel(grad_lhs_ckpt,    grad_lhs_plain)
    print(f"  seed={seed} ({batch},{in_dim},{rank},{out_dim}): "
          f"plain vs ckpt — lhs cos={cp_x[0]:.6f} rel={cp_x[1]:.4e} | "
          f"loraA cos={cp_a[0]:.6f} rel={cp_a[1]:.4e} | "
          f"loraB cos={cp_b[0]:.6f} rel={cp_b[1]:.4e}")

    return {
        "batch": batch,
        "in_dim": in_dim,
        "rank": rank,
        "out_dim": out_dim,
        "seed": seed,
        # Input tensors (BF16 values stored as F32 for lossless round-trip)
        "lhs_f32": lhs_bf.float().numpy().flatten().tolist(),
        "lora_a_f32": lora_a_bf.float().numpy().flatten().tolist(),
        "lora_b_f32": lora_b_bf.float().numpy().flatten().tolist(),
        "frozen_w_f32": frozen_w_bf.float().numpy().flatten().tolist(),
        # Plain (non-checkpointed) reference gradients
        "grad_lhs_plain_f32":    grad_lhs_plain,
        "grad_lora_a_plain_f32": grad_lora_a_plain,
        "grad_lora_b_plain_f32": grad_lora_b_plain,
        # Checkpointed reference gradients (should match plain within BF16 noise)
        "grad_lhs_ckpt_f32":    grad_lhs_ckpt,
        "grad_lora_a_ckpt_f32": grad_lora_a_ckpt,
        "grad_lora_b_ckpt_f32": grad_lora_b_ckpt,
    }

if __name__ == "__main__":
    import os
    print(f"PyTorch {torch.__version__}")
    print("Generating checkpoint_lora_bwd_ref.json ...")

    cases = [
        # (batch, in_dim, rank, out_dim, seed)
        # Match typical LoRA trainer shapes: rank=8 or rank=16, full linear dims
        (8,  256,  8, 256, 7001),   # Small: RMSNorm-sized
        (8,  256,  8, 1024, 7002),  # Wide: MLP expansion
        (4,  1024, 16, 1024, 7003), # Larger: deeper model hidden dim
    ]

    fixture_cases = []
    for args in cases:
        case = make_case(*args)
        fixture_cases.append(case)

    out_path = os.path.join(os.path.dirname(__file__), "checkpoint_lora_bwd_ref.json")
    with open(out_path, "w") as f:
        json.dump({"cases": fixture_cases}, f)
    print(f"Written: {out_path}")
    print("Done.")
