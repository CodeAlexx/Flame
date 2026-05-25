#!/usr/bin/env python3
"""Closed-loop LoRA-Linear training parity reference.

Mirrors `eridiffusion-core/src/lora.rs::LoRALinear::forward_delta`:

    out = (alpha / rank) * (input @ A^T @ B^T)

with F32 params, BF16 matmul (params and input cast to BF16 inside the
forward), then the BF16 output flows into BF16 MSE loss. Backward
produces grads on F32 params (the .grad-storage cast policy).

Runs 50 Adam steps on the same deterministic data and dumps per-step
LoRA-A grad L2, LoRA-B grad L2, and the post-step F32 params. The Rust
test feeds the SAME inputs through `LoRALinear` + `flame_core::AdamW`
and asserts per-step parity.

Goal: localize the 4× LoRA-B grad amplification the handoff describes
(zimage trainer vs OneTrainer). If the closed-loop test ALSO amplifies,
the bug lives in this minimal LoRA forward/backward — the trainer
overhead is irrelevant. If it doesn't, the bug is in something the
trainer adds (real model, RNG, dataloader, etc.) and we need to climb
the stack.

Config matches zimage/klein OT-preset: lr=3e-4, betas=(0.9, 0.999),
eps=1e-8, weight_decay=0.01.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F


def lora_forward_bf16(input_bf16, a_f32, b_f32, alpha, rank):
    """Mirror of LoRALinear::forward_delta: F32 params cast to BF16 in forward.

    The cast happens via torch.Tensor.to(bfloat16) which preserves autograd —
    same as flame-core's Op::Cast. Grad flows back through the cast as a
    `to_dtype(F32)` on the upstream BF16 grad.
    """
    a_bf16 = a_f32.to(torch.bfloat16)
    b_bf16 = b_f32.to(torch.bfloat16)
    # input @ A^T @ B^T  (LoRALinear's matmul ordering)
    intermediate = input_bf16 @ a_bf16.t()
    out = intermediate @ b_bf16.t()
    scale = alpha / rank
    if abs(scale - 1.0) > 1e-9:
        out = out * scale
    return out


def main() -> int:
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lora_closed_loop_ref.json",
    )

    # ── Config (zimage/klein OT-preset) ─────────────────────────────────
    LR = 3e-4
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    WD = 0.01

    IN_FEATURES = 64
    OUT_FEATURES = 64
    RANK = 8
    ALPHA = 1.0           # matches handoff lora_alpha=1.0
    BATCH = 2
    SEQ = 4
    STEPS = 50

    device = torch.device("cuda")
    torch.manual_seed(0xCAFEBABE)

    # ── Init: Kaiming uniform on A (bound = 1/sqrt(in)), zeros on B ──
    # Match LoRALinear::new exactly: uniform[-bound, +bound] in row-major
    # [rank, in_features] for A; zeros in [out_features, rank] for B.
    bound = 1.0 / (IN_FEATURES ** 0.5)
    a_init = (torch.rand(RANK, IN_FEATURES, device=device, dtype=torch.float32) * 2.0 - 1.0) * bound
    b_init = torch.zeros(OUT_FEATURES, RANK, device=device, dtype=torch.float32)

    # ── Deterministic input + target batch ──
    # input is BF16, matching trainer reality. Target is BF16 zeros (MSE
    # against zero target = sum of squares of forward output / N).
    inputs = [
        torch.randn(BATCH, SEQ, IN_FEATURES, device=device, dtype=torch.bfloat16) * 0.5
        for _ in range(STEPS)
    ]
    # Non-zero targets so the loss is real. With B=0 init the LoRA forward
    # outputs 0; if targets are also 0 the loss is identically 0 every step
    # and we measure nothing. Real LoRA training has a base-network output
    # the LoRA learns to correct against; here the target stands in for
    # that residual signal so per-step gradients are nonzero from step 1.
    targets = [
        torch.randn(BATCH, SEQ, OUT_FEATURES, device=device, dtype=torch.bfloat16) * 0.1
        for _ in range(STEPS)
    ]

    a = a_init.clone().detach().requires_grad_(True)
    b = b_init.clone().detach().requires_grad_(True)
    opt = torch.optim.AdamW([a, b], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WD)

    per_step = []
    for k in range(STEPS):
        opt.zero_grad(set_to_none=True)
        out_bf16 = lora_forward_bf16(inputs[k], a, b, ALPHA, RANK)
        # MSE loss, mean reduction (PyTorch default).
        loss = F.mse_loss(out_bf16, targets[k])
        loss.backward()

        # Capture pre-step grad L2s (these are what the trainer reports).
        ga = a.grad.detach().float()
        gb = b.grad.detach().float()
        per_step.append({
            "step": k + 1,
            "loss": float(loss.detach().float().item()),
            "grad_a_l2": float(torch.linalg.vector_norm(ga).item()),
            "grad_b_l2": float(torch.linalg.vector_norm(gb).item()),
            "grad_a_max_abs": float(ga.abs().max().item()),
            "grad_b_max_abs": float(gb.abs().max().item()),
        })

        opt.step()

        # After the step, also capture param L2 norms.
        per_step[-1]["a_l2_after"] = float(torch.linalg.vector_norm(a.detach().float()).item())
        per_step[-1]["b_l2_after"] = float(torch.linalg.vector_norm(b.detach().float()).item())

    fixture = {
        "config": {
            "lr": LR,
            "beta1": BETA1,
            "beta2": BETA2,
            "eps": EPS,
            "weight_decay": WD,
            "in_features": IN_FEATURES,
            "out_features": OUT_FEATURES,
            "rank": RANK,
            "alpha": ALPHA,
            "batch": BATCH,
            "seq": SEQ,
            "steps": STEPS,
        },
        "a_init": a_init.detach().cpu().tolist(),
        "b_init": b_init.detach().cpu().tolist(),
        # bf16 inputs serialized as float32 lists — Rust will cast back to BF16.
        "inputs_bf16_as_f32": [x.detach().float().cpu().tolist() for x in inputs],
        "targets_bf16_as_f32": [t.detach().float().cpu().tolist() for t in targets],
        "per_step": per_step,
        # Final params for end-state check.
        "a_final": a.detach().cpu().tolist(),
        "b_final": b.detach().cpu().tolist(),
    }

    with open(fixture_path, "w") as f:
        json.dump(fixture, f)

    print(f"wrote {fixture_path}")
    print(f"  in={IN_FEATURES} out={OUT_FEATURES} rank={RANK} alpha={ALPHA} steps={STEPS}")
    print(f"  step  1: loss={per_step[0]['loss']:.4e}  ||gA||={per_step[0]['grad_a_l2']:.3e}  ||gB||={per_step[0]['grad_b_l2']:.3e}")
    print(f"  step  5: loss={per_step[4]['loss']:.4e}  ||gA||={per_step[4]['grad_a_l2']:.3e}  ||gB||={per_step[4]['grad_b_l2']:.3e}")
    print(f"  step 10: loss={per_step[9]['loss']:.4e}  ||gA||={per_step[9]['grad_a_l2']:.3e}  ||gB||={per_step[9]['grad_b_l2']:.3e}")
    print(f"  step 25: loss={per_step[24]['loss']:.4e}  ||gA||={per_step[24]['grad_a_l2']:.3e}  ||gB||={per_step[24]['grad_b_l2']:.3e}")
    print(f"  step 50: loss={per_step[49]['loss']:.4e}  ||gA||={per_step[49]['grad_a_l2']:.3e}  ||gB||={per_step[49]['grad_b_l2']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
