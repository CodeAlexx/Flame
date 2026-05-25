#!/usr/bin/env python3
"""Stacked LoRA-Linear chain training parity reference.

Builds 30 layers of (frozen base Linear + trainable LoRA delta) stacked
in series, mirroring Z-Image's 30 transformer blocks. Each layer:

    y = base(x) + (alpha/rank) * (x @ A^T @ B^T)

Runs 50 Adam steps on the chain with the zimage/klein OT-preset config
and dumps per-layer per-step (grad_A_l2, grad_B_l2). If amplification of
lora_A grads appears in this chain (but not in the single-layer test
`lora_closed_loop_ref`), the bug is in flame-core's long-autograd-chain
handling under LoRA pressure.

Setup mirrors the LoRALinear / OT LoRAModule contract: F32 LoRA params,
BF16 forward matmul, BF16 base linear, BF16 final MSE.
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def lora_forward_bf16(input_bf16, a_f32, b_f32, alpha, rank):
    a_bf16 = a_f32.to(torch.bfloat16)
    b_bf16 = b_f32.to(torch.bfloat16)
    intermediate = input_bf16 @ a_bf16.t()
    out = intermediate @ b_bf16.t()
    scale = alpha / rank
    if abs(scale - 1.0) > 1e-9:
        out = out * scale
    return out


def main() -> int:
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lora_chain_ref.json",
    )

    LR = 3e-4
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    WD = 0.01
    IN_FEATURES = 64
    OUT_FEATURES = 64
    RANK = 8
    ALPHA = 1.0
    BATCH = 2
    SEQ = 4
    N_LAYERS = 30
    STEPS = 50

    device = torch.device("cuda")
    torch.manual_seed(0xC0DEFACE)

    # Frozen base linears (BF16) — mirror the real model's frozen W per layer.
    base_weights = []
    for _ in range(N_LAYERS):
        w = (torch.randn(OUT_FEATURES, IN_FEATURES, device=device, dtype=torch.float32) * (1.0 / IN_FEATURES ** 0.5))
        # Store as BF16 (matches inference dtype; doesn't need autograd).
        base_weights.append(w.to(torch.bfloat16))

    # LoRA inits per layer.
    bound = 1.0 / (IN_FEATURES ** 0.5)
    a_inits = [
        ((torch.rand(RANK, IN_FEATURES, device=device, dtype=torch.float32) * 2.0 - 1.0) * bound)
        for _ in range(N_LAYERS)
    ]
    b_inits = [
        torch.zeros(OUT_FEATURES, RANK, device=device, dtype=torch.float32)
        for _ in range(N_LAYERS)
    ]

    # Deterministic inputs / targets.
    inputs = [
        (torch.randn(BATCH, SEQ, IN_FEATURES, device=device, dtype=torch.bfloat16) * 0.5)
        for _ in range(STEPS)
    ]
    targets = [
        (torch.randn(BATCH, SEQ, OUT_FEATURES, device=device, dtype=torch.bfloat16) * 0.1)
        for _ in range(STEPS)
    ]

    a_params = [x.clone().detach().requires_grad_(True) for x in a_inits]
    b_params = [x.clone().detach().requires_grad_(True) for x in b_inits]
    params = a_params + b_params
    opt = torch.optim.AdamW(params, lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WD)

    # per_step[step_index] = {
    #   "loss": ..., "grad_a_l2_per_layer": [...], "grad_b_l2_per_layer": [...]
    # }
    per_step = []
    for k in range(STEPS):
        opt.zero_grad(set_to_none=True)
        x = inputs[k]
        for li in range(N_LAYERS):
            base_out = x @ base_weights[li].t()
            lora_out = lora_forward_bf16(x, a_params[li], b_params[li], ALPHA, RANK)
            x = base_out + lora_out
        loss = F.mse_loss(x, targets[k])
        loss.backward()

        ga_l2 = [
            float(torch.linalg.vector_norm(a.grad.detach().float()).item())
            for a in a_params
        ]
        gb_l2 = [
            float(torch.linalg.vector_norm(b.grad.detach().float()).item())
            for b in b_params
        ]
        per_step.append({
            "loss": float(loss.detach().float().item()),
            "grad_a_l2_per_layer": ga_l2,
            "grad_b_l2_per_layer": gb_l2,
        })
        opt.step()

    fixture = {
        "config": {
            "lr": LR, "beta1": BETA1, "beta2": BETA2, "eps": EPS, "weight_decay": WD,
            "in_features": IN_FEATURES, "out_features": OUT_FEATURES,
            "rank": RANK, "alpha": ALPHA, "batch": BATCH, "seq": SEQ,
            "n_layers": N_LAYERS, "steps": STEPS,
        },
        "base_weights": [w.detach().float().cpu().tolist() for w in base_weights],
        "a_inits": [a.detach().cpu().tolist() for a in a_inits],
        "b_inits": [b.detach().cpu().tolist() for b in b_inits],
        "inputs_bf16_as_f32": [x.detach().float().cpu().tolist() for x in inputs],
        "targets_bf16_as_f32": [t.detach().float().cpu().tolist() for t in targets],
        "per_step": per_step,
        "a_final": [a.detach().cpu().tolist() for a in a_params],
        "b_final": [b.detach().cpu().tolist() for b in b_params],
    }
    with open(fixture_path, "w") as f:
        json.dump(fixture, f)

    print(f"wrote {fixture_path}")
    print(f"  n_layers={N_LAYERS} in/out={IN_FEATURES}/{OUT_FEATURES} rank={RANK} steps={STEPS}")
    for show in (0, 4, 9, 24, 49):
        gaL = per_step[show]["grad_a_l2_per_layer"]
        gbL = per_step[show]["grad_b_l2_per_layer"]
        # show layer 0 and layer N-1 grads
        print(
            f"  step {show+1:>2}: loss={per_step[show]['loss']:.4e}  "
            f"L0(A={gaL[0]:.2e},B={gbL[0]:.2e})  "
            f"L{N_LAYERS-1}(A={gaL[-1]:.2e},B={gbL[-1]:.2e})  "
            f"sumA={sum(gaL):.3e} sumB={sum(gbL):.3e}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
